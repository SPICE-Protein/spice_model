"""Data Pipeline: Load -> Clean -> TFRecord -> tf.data.Dataset.

Supports two data sources:
- `hf`: download Parquet shards from HuggingFace (defaults to hf-mirror.com).
- `local`: read from local `data/parquet` directory (produced by download_pdb.py).

Pipeline steps:
1. For each shard: parse entries (structural metadata) + atoms (Cα coordinates) into tuples of (Seq, Env, Coords).
2. Clean and filter (by has_env, sequence length, max shards, structures per shard).
3. Serialize to TFRecords (one-time preprocessing for fast streaming during training).
4. Load TFRecord -> tf.data.Dataset (using padded_batch to support variable sequence lengths).

Usage:
    python -m spice_pre.data.dataset --config configs/pretrain.yaml build   # Clean and generate TFRecords
    python -m spice_pre.data.dataset --config configs/pretrain.yaml stats   # Compute dataset statistics
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
# Data Source Parsing
# ---------------------------------------------------------------------------
def _is_colab() -> bool:
    """Checks if running in Google Colab (environments have direct access to official huggingface endpoints)."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _set_hf_endpoint(cfg: Config) -> None:
    """Sets the HuggingFace download endpoint.

    - Local Development (Mainland China): Fallback to the configured hf-mirror.com.
    - Colab: Executes on Google Cloud, directly connecting to the official huggingface.co;
      forcing a domestic mirror would instead cause LocalEntryNotFoundError.
      If the user has explicitly set HF_ENDPOINT as an environment variable, it takes precedence.

    huggingface_hub reads HF_ENDPOINT into constants.ENDPOINT only on its first import;
    therefore, we override constants.ENDPOINT directly in addition to setting the env variable 
    to guarantee immediate effect even if the library was already imported (e.g. in reused Jupyter kernels).
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
    """Returns a sorted list of entries_shard_*.parquet filenames."""
    if cfg.data.source == "local":
        pat = os.path.join(cfg.data.local_dir, "entries_shard_*.parquet")
        files = sorted(os.path.basename(p) for p in glob.glob(pat))
        # Keep only shards where both entries and atoms exist in pairs
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
    """Resolves a shard filename to its local file path (concatenated directly for local, downloaded to cache for hf)."""
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
# Shard-level Cleansing -> Record Tuples
# ---------------------------------------------------------------------------
def _records_from_shard(
    entries_path: str, atoms_path: str, cfg: Config
) -> List[dict]:
    """Loads a single shard's entries and atoms, returning a list of dicts: [(tokens, mask, coords, env)]."""
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

    # Select only Cα atoms and filter by the target pdb set
    at = pl.read_parquet(atoms_path).filter(pl.col("is_ca"))
    at = at.filter(pl.col("pdb_id").is_in(en["pdb_id"].to_list()))
    at = at.sort(["pdb_id", "chain_id", "res_seq"])
    # Deduplicate multiple conformations (NMR) or alternative locations (altloc): keep the first entry per (chain, res_seq).
    # ⚠️ Polars' `unique(keep="first")` groups and rearranges rows, which disrupts the original residue order!
    # We must explicitly re-sort by backbone order (pdb_id, chain_id, res_seq) afterward; otherwise, 
    # sequence tokens and coordinate arrays become misaligned (consecutive Cα distance spikes to ~20Å instead of the physical ~3.8Å), 
    # preventing the model from learning protein topologies (which previously led pre-training to predict amorphous blobs with CE stuck at uniform baselines).
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
# TFRecord Serialization
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
    """Cleanses all (or a subset of) shards and serializes them into TFRecords. Returns the total record count."""
    os.makedirs(cfg.data.tfrecord_dir, exist_ok=True)
    shards = list_shards(cfg)
    if cfg.data.max_shards:
        shards = shards[: cfg.data.max_shards]
    if not shards:
        raise FileNotFoundError(
            f"No entries shards found (source={cfg.data.source}, "
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
        # ⚠️ Data efficiency ablation: global sequence chain truncation (max_chains, 0 = unlimited) — strictly limits the dataset to N chains across all shards
        if cfg.data.max_chains:
            remain = cfg.data.max_chains - total
            if remain <= 0:
                if verbose:
                    print(f"[{idx}] Reached max_chains={cfg.data.max_chains}, halting serialization")
                break
            recs = recs[:remain]
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
# Deserialization: TFRecord -> tf.data.Dataset
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
    """Deserializes TFRecords, returning an unbatched, unpadded tf.data.Dataset.

    split="train" / "val": splits shard files based on the configured val_split after sorting filenames.
    """
    files = sorted(glob.glob(os.path.join(cfg.data.tfrecord_dir, "shard_*.tfrecord")))
    if not files:
        raise FileNotFoundError(
            "No TFRecord files found; please generate them first using: python -m spice_pre.data.dataset build"
        )

    n_files = len(files)
    # Compute the number of validation files based on val_split ratio (at least 1 file), and allocate the remaining files for training
    n_val = max(1, int(round(n_files * cfg.train.val_split)))
    # Edge-case safety guard when very few files exist: guarantees at least 1 file for training
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
    """Computes the total number of records across all TFRecords (used to determine epoch steps)."""
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
    ap.add_argument("--config", default=None, help="Path to YAML configuration file")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Override data.max_shards (for debugging)")
    ap.add_argument("--structures", type=int, default=None,
                    help="Override data.structures_per_shard (for debugging)")
    ap.add_argument("--keep-env-all", action="store_true",
                    help="Override and set use_env_filtered=False (loads all structures with default environmental conditions)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Clean and serialize shards into TFRecords")
    sub.add_parser("stats", help="Print total record count in TFRecords")
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
