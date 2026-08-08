"""数据管线：加载 → 清洗 → TFRecord → tf.data.Dataset。

数据来源两种：
- `hf`：从 HuggingFace（默认 hf-mirror.com 镜像）下载 parquet 分片。
- `local`：读本地 `data/parquet` 目录（即 download_pdb.py 的产物）。

流程：
1. 对每个 shard：entries（结构元数据）+ atoms（Cα 原子）→ 三元组 (Seq, Env, Coords)。
2. 清洗过滤（has_env、长度、shard/数量上限）。
3. 写 TFRecord（一次预处理，训练时快速读取）。
4. 读 TFRecord → tf.data.Dataset（padded_batch，支持变长）。

用法：
    python -m spice_pre.data.dataset --config configs/pretrain.yaml build   # 生成 TFRecord
    python -m spice_pre.data.dataset --config configs/pretrain.yaml stats   # 查看规模
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import tensorflow as tf

from spice_pre.config import Config, load_config
from spice_pre.data.preprocessing import (
    normalize_env,
    res_names_to_seq,
    seq_to_tokens,
)


# ---------------------------------------------------------------------------
# 数据来源解析
# ---------------------------------------------------------------------------
def _is_colab() -> bool:
    """是否运行在 Google Colab（代码跑在 Google 云上，官方端点直连即可）。"""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _set_hf_endpoint(cfg: Config) -> None:
    """设置 HF 下载端点。

    - 本地开发（国内）：用配置里的 hf-mirror.com 镜像。
    - Colab：虚拟机在 Google 云上，官方 huggingface.co 直连即可；
      强设国内镜像反而会 LocalEntryNotFoundError。
      若用户显式设置了 HF_ENDPOINT 环境变量则尊重它。

    huggingface_hub 只在首次 import 时把 HF_ENDPOINT 读进 constants.ENDPOINT，
    所以除了设置环境变量外，还要直接改写 constants.ENDPOINT，保证即使库已
    被 import（如 Jupyter kernel 复用）也能立即生效。
    """
    if _is_colab():
        endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    else:
        endpoint = cfg.data.hf_endpoint
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        import huggingface_hub.constants as _constants

        _constants.ENDPOINT = endpoint
    except Exception:  # pragma: no cover
        pass


def list_shards(cfg: Config) -> List[str]:
    """返回 entries_shard_*.parquet 文件名列表（已排序）。"""
    if cfg.data.source == "local":
        pat = os.path.join(cfg.data.local_dir, "entries_shard_*.parquet")
        files = sorted(os.path.basename(p) for p in glob.glob(pat))
        # 仅保留 entries/atoms 成对存在的 shard
        return [
            f
            for f in files
            if os.path.exists(
                os.path.join(cfg.data.local_dir, f.replace("entries_", "atoms_"))
            )
        ]
    # hf
    _set_hf_endpoint(cfg)
    import huggingface_hub.constants as hf_constants
    from huggingface_hub import list_repo_files

    files = list_repo_files(cfg.data.hf_repo, repo_type="dataset")
    print(f"[HF] endpoint: {hf_constants.ENDPOINT} | repo: {cfg.data.hf_repo}")
    return sorted(f for f in files if f.startswith("entries_shard_"))


def resolve_path(cfg: Config, shard_fname: str) -> str:
    """把 shard 文件名解析成本地路径（local 直接拼，hf 下载到缓存）。"""
    if cfg.data.source == "local":
        return os.path.join(cfg.data.local_dir, shard_fname)
    _set_hf_endpoint(cfg)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=cfg.data.hf_repo,
        filename=shard_fname,
        repo_type="dataset",
        cache_dir=cfg.data.cache_dir,
    )


# ---------------------------------------------------------------------------
# 单 shard 清洗 → 三元组
# ---------------------------------------------------------------------------
def _records_from_shard(
    entries_path: str, atoms_path: str, cfg: Config
) -> List[dict]:
    """读一个 shard 的 entries + atoms，返回 [(tokens, mask, coords, env)]。"""
    en = pl.read_parquet(entries_path)
    if cfg.data.use_env_filtered:
        en = en.filter(pl.col("has_env"))
    en = en.select(
        ["pdb_id", "ph", "temperature", "ionic_strength_m", "n_residues"]
    )
    if cfg.data.structures_per_shard:
        en = en.head(cfg.data.structures_per_shard)
    if en.height == 0:
        return []

    # 只取 Cα 原子，限定在目标 pdb 集合内
    at = pl.read_parquet(atoms_path).filter(pl.col("is_ca"))
    at = at.filter(pl.col("pdb_id").is_in(en["pdb_id"].to_list()))
    at = at.sort(["pdb_id", "chain_id", "res_seq"])
    # 多构象（NMR）/ 交替构象去重：同一 (chain, res_seq) 取第一条。
    # ⚠️ polars unique(keep="first") 会按唯一键分组重排，破坏行序！
    # 必须在其后重新按骨架顺序 (pdb_id, chain_id, res_seq) 排序，
    # 否则序列↔坐标不对齐（相邻残基 Cα 距离 ~20Å 而非 ~3.8Å），
    # 模型永远学不出拓扑（曾导致 pre-train 出 blob、CE 卡在均匀基线）。
    at = at.unique(subset=["pdb_id", "chain_id", "res_seq"], keep="first")
    at = at.sort(["pdb_id", "chain_id", "res_seq"])

    env_map = {r["pdb_id"]: r for r in en.to_dicts()}
    records: List[dict] = []
    for grp in at.partition_by("pdb_id", maintain_order=True):
        pid = grp["pdb_id"][0]
        row = env_map.get(pid)
        if row is None:
            continue
        res_names = grp["res_name"].to_list()
        seq = res_names_to_seq(res_names)
        n = len(res_names)
        if not (cfg.data.min_seq_len <= n <= cfg.data.max_seq_len):
            continue
        coords = grp.select(["x", "y", "z"]).to_numpy().astype(np.float32)  # [L,3]
        tokens = seq_to_tokens(seq)
        env = normalize_env(
            row["ph"], row["temperature"], row["ionic_strength_m"],
            cfg.data.default_env,
        )
        records.append(
            {
                "tokens": tokens,
                "mask": np.ones(n, dtype=np.float32),
                "coords": coords,
                "env": env,
            }
        )
    return records


# ---------------------------------------------------------------------------
# TFRecord 序列化
# ---------------------------------------------------------------------------
def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: np.ndarray):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


def _make_example(rec: dict) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "tokens": _bytes_feature(rec["tokens"].astype(np.int32).tobytes()),
                "mask": _bytes_feature(rec["mask"].astype(np.float32).tobytes()),
                "coords": _bytes_feature(rec["coords"].astype(np.float32).tobytes()),
                "env": _float_feature(rec["env"].astype(np.float32)),
            }
        )
    )


def build_tfrecords(cfg: Config, verbose: bool = True) -> int:
    """把全部（或部分）shard 清洗后写入 TFRecord。返回记录总数。"""
    os.makedirs(cfg.data.tfrecord_dir, exist_ok=True)
    shards = list_shards(cfg)
    if cfg.data.max_shards:
        shards = shards[: cfg.data.max_shards]
    if not shards:
        raise FileNotFoundError(
            f"没有找到 entries shard（source={cfg.data.source}, "
            f"local_dir={cfg.data.local_dir}）"
        )

    total = 0
    for idx, shard_fname in enumerate(shards):
        entries_path = resolve_path(cfg, shard_fname)
        atoms_path = resolve_path(cfg, shard_fname.replace("entries_", "atoms_"))
        recs = _records_from_shard(entries_path, atoms_path, cfg)
        if not recs:
            if verbose:
                print(f"[{idx}] {shard_fname}: 0 records, skipped")
            continue
        out_path = os.path.join(cfg.data.tfrecord_dir, f"shard_{idx:04d}.tfrecord")
        with tf.io.TFRecordWriter(out_path) as writer:
            for r in recs:
                writer.write(_make_example(r).SerializeToString())
        total += len(recs)
        if verbose:
            print(f"[{idx}] {shard_fname}: {len(recs)} records -> {out_path}")
    if verbose:
        print(f"TOTAL records written: {total}")
    return total


# ---------------------------------------------------------------------------
# 读 TFRecord → tf.data.Dataset
# ---------------------------------------------------------------------------
def _parse_example(proto: tf.Tensor, max_seq_len: int):
    feats = {
        "tokens": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "coords": tf.io.FixedLenFeature([], tf.string),
        "env": tf.io.FixedLenFeature([3], tf.float32),
    }
    ex = tf.io.parse_single_example(proto, feats)
    tokens = tf.io.decode_raw(ex["tokens"], tf.int32)[:max_seq_len]
    mask = tf.io.decode_raw(ex["mask"], tf.float32)[:max_seq_len]
    coords = tf.reshape(tf.io.decode_raw(ex["coords"], tf.float32), (-1, 3))
    coords = coords[:max_seq_len]
    env = ex["env"]

    x = {"tokens": tokens, "mask": mask, "env": env, "coords": coords}
    return x, coords  # (features, label)


def load_tfrecord_dataset(
    cfg: Config,
    split: str = "train",
    shuffle_buffer: int = 4096,
) -> tf.data.Dataset:
    """读取 TFRecord，返回 tf.data.Dataset（未 batch，未 padded）。

    split="train" / "val"：按 TFRecord 文件名排序后按 val_split 比例切分文件。
    """
    files = sorted(glob.glob(os.path.join(cfg.data.tfrecord_dir, "shard_*.tfrecord")))
    if not files:
        raise FileNotFoundError(
            "没有 TFRecord，请先运行: python -m spice_pre.data.dataset build"
        )

    n_files = len(files)
    # 按 val_split 比例计算验证集文件数（至少 1 个文件），train 取其余文件
    n_val = max(1, int(round(n_files * cfg.train.val_split)))
    # 文件太少时保护：保证 train 至少 1 个文件
    if n_files <= 1:
        n_val = 0
    elif n_val >= n_files:
        n_val = n_files - 1
    if split == "train":
        files = files[: n_files - n_val]
    else:
        files = files[-n_val:] if n_val else files

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=os.cpu_count())
    ds = ds.map(
        lambda p: _parse_example(p, cfg.data.max_seq_len),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if split == "train":
        ds = ds.shuffle(shuffle_buffer)
    return ds


def count_records(cfg: Config) -> int:
    """统计 TFRecord 总记录数（用于打印 epoch 步数）。"""
    files = glob.glob(os.path.join(cfg.data.tfrecord_dir, "shard_*.tfrecord"))
    if not files:
        return 0
    ds = tf.data.TFRecordDataset(files)
    n = sum(1 for _ in ds)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="YAML 配置文件路径")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="覆盖 data.max_shards（调试用）")
    ap.add_argument("--structures", type=int, default=None,
                    help="覆盖 data.structures_per_shard（调试用）")
    ap.add_argument("--keep-env-all", action="store_true",
                    help="覆盖 use_env_filtered=False（用全部结构+默认环境）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="清洗并生成 TFRecord")
    sub.add_parser("stats", help="统计 TFRecord 规模")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_shards is not None:
        cfg.data.max_shards = args.max_shards
    if args.structures is not None:
        cfg.data.structures_per_shard = args.structures
    if args.keep_env_all:
        cfg.data.use_env_filtered = False
    if args.cmd == "build":
        build_tfrecords(cfg)
    elif args.cmd == "stats":
        n = count_records(cfg)
        print(f"TFRecord records: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
