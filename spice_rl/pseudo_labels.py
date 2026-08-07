"""环节四：双路回流 —— 伪标签收集与回流。

路径 B 存活突变体的时间平均坐标（engine.pseudo_labels）构成伪标签
(突变后Seq, Env_fail, 稳定坐标)，以置信度权重（存活步数占比）回流到
Pre-train 数据集，触发微调。

流程：
    pseudo_*.npz（train_post 落盘）→ load_pseudo_labels → write_pseudo_tfrecord
    → 与 Pre-train 原 TFRecord 合并（make_finetune_dataset）→ finetune_pretrain.py
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import tensorflow as tf

# 复用 spice_pre 的 TFRecord 序列化/解析（保证格式一致，避免二次实现出错）
from spice_pre.data.dataset import _make_example, _parse_example
from spice_pre.data.preprocessing import normalize_env, seq_to_tokens


def load_pseudo_labels(pseudo_dir: str) -> List[dict]:
    """读取伪标签 npz。每条：{seq, env[3](raw ph/temp/ionic), coords[L,3], weight, file}。

    weight = 存活步数（从文件名 pseudo_{idx}_{steps}.npz 解析）。
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pseudo_dir, "pseudo_*.npz"))):
        try:
            d = np.load(f, allow_pickle=True)
            seq = str(d["seq"])
            env = np.asarray(d["env"], np.float32).reshape(-1)
            coords = np.asarray(d["coords"], np.float32)
            steps = int(os.path.basename(f).rsplit(".", 1)[0].split("_")[-1])
        except Exception:  # noqa: BLE001
            continue
        if not seq or coords.ndim != 2 or coords.shape[1] != 3:
            continue
        out.append({"seq": seq, "env": env, "coords": coords, "weight": steps, "file": f})
    return out


def write_pseudo_tfrecord(
    pseudo_dir: str,
    out_path: str,
    max_seq_len: int,
    weight_repeat: int = 8,
    survive_steps: int = 200,
) -> int:
    """把伪标签转成 Pre-train 格式 TFRecord（与 spice_pre 一致）。

    按置信度权重（存活步数占比）重复写入，实现"以置信度权重加入数据集"。
    Returns: 写入条数。
    """
    recs = load_pseudo_labels(pseudo_dir)
    if not recs:
        return 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = 0
    with tf.io.TFRecordWriter(out_path) as w:
        for r in recs:
            seq = r["seq"][:max_seq_len]
            tokens = seq_to_tokens(seq)
            L = tokens.shape[0]
            if L == 0:
                continue
            env = normalize_env(r["env"][0], r["env"][1], r["env"][2])
            coords = r["coords"][:L].astype(np.float32)
            weight_norm = float(np.clip(r["weight"] / max(1, survive_steps), 0.1, 1.0))
            reps = max(1, int(round(weight_repeat * weight_norm)))
            ex = _make_example(
                {
                    "tokens": tokens,
                    "mask": np.ones(L, np.float32),
                    "coords": coords,
                    "env": env,
                }
            )
            for _ in range(reps):
                w.write(ex.SerializeToString())
                count += 1
    print(f"伪标签回流: {len(recs)} 条 -> {out_path}（加权后 {count} 条）")
    return count


def make_finetune_dataset(
    cfg,
    pseudo_tfrecord: Optional[str],
    max_seq_len: int,
    split: str = "train",
    shuffle_buffer: int = 4096,
):
    """合并 Pre-train 原 TFRecord + 伪标签 TFRecord → tf.data.Dataset。

    返回 (features, coords)，与 spice_pre 的 load_tfrecord_dataset 同构，
    可直接 padded_batch 后训练。
    """
    files = sorted(
        glob.glob(os.path.join(cfg.post.pretrain_tfrecord_dir, "shard_*.tfrecord"))
    )
    if pseudo_tfrecord and os.path.exists(pseudo_tfrecord):
        files.append(pseudo_tfrecord)
    if not files:
        raise FileNotFoundError("没有可用的 TFRecord（原 Pre-train 或伪标签）")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=os.cpu_count())
    ds = ds.map(
        lambda p: _parse_example(p, max_seq_len),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if split == "train":
        ds = ds.shuffle(shuffle_buffer)
    return ds
