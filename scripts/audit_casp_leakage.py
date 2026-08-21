#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import polars as pl

from spice_pre.eval_casp import CASP_TARGETS

REPO = "SPICE-Protein/spice_protein"


def _hf_endpoint() -> str:
    try:
        import google.colab  # noqa: F401
        return "https://huggingface.co"
    except ImportError:
        if os.path.isdir("/kaggle"):
            return "https://huggingface.co"
        return "https://hf-mirror.com"


def _entries_shards(cache_dir: str, endpoint: str) -> list[str]:
    os.environ["HF_ENDPOINT"] = endpoint
    import huggingface_hub.constants as hf_constants
    hf_constants.ENDPOINT = endpoint  
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("entries_shard_") and f.endswith(".parquet"))
    print(f"[audit] {len(files)} entries shards in repo")
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for f in files:
        local = os.path.join(cache_dir, os.path.basename(f))
        if not os.path.exists(local):
            print(f"[audit]   download {f}", flush=True)
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=cache_dir)
        paths.append(local)
    return paths


def _cached_entries(cache_dir: str) -> list[str]:
    snap = glob.glob(
        "data/cache/datasets--SPICE-Protein--spice_protein/snapshots/*/entries_shard_*.parquet"
    )
    if snap:
        return sorted(snap)
    return sorted(glob.glob(os.path.join(cache_dir, "entries_shard_*.parquet")))


def _rcsb_dates(pdb: str) -> dict:
    import urllib.request

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        acc = d.get("rcsb_accession_info", {})
        return {"pdb": pdb,
                "deposit_date": acc.get("deposit_date"),
                "initial_release_date": acc.get("initial_release_date")}
    except Exception as e:  # noqa: BLE001
        return {"pdb": pdb, "deposit_date": None, "initial_release_date": None,
                "note": f"rcsb lookup failed: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="data/casp_leakage",
                    help="dir for downloaded entries shards (default data/casp_leakage)")
    ap.add_argument("--out", default="", help="write JSON audit report")
    args = ap.parse_args()

    targets = [code.lower() for _, code in CASP_TARGETS]
    print(f"[audit] CASP14 targets checked: {', '.join(targets)}")

    paths = []
    try:
        paths = _entries_shards(args.cache, _hf_endpoint())  
    except Exception as e:  # noqa: BLE001
        print(f"[audit] full download failed ({e}); falling back to cached shards")
        paths = _cached_entries(args.cache)
    print(f"[audit] entries shards loaded: {len(paths)}")

    frames = [pl.read_parquet(p).select("pdb_id") for p in paths]
    all_id = pl.concat(frames)
    n_total = all_id.height
    all_id = all_id.with_columns(pl.col("pdb_id").str.to_lowercase().alias("_id"))
    n_unique = all_id["_id"].n_unique()
    print(f"[audit] training entries: {n_total} rows, {n_unique} unique PDB ids")

    hit = all_id.filter(pl.col("_id").is_in(targets)).select("_id").to_series().to_list()
    print(f"[audit] overlap with CASP14 targets: {len(hit)} -> {hit}")

    dates = [_rcsb_dates(c) for _, c in CASP_TARGETS]
    for d in dates:
        print(f"[audit]   {d['pdb']}: deposited {d.get('deposit_date')}, "
              f"released {d.get('initial_release_date')}")

    report = {
        "targets_checked": targets,
        "n_training_entries": int(n_total),
        "n_unique_pdb_ids": int(n_unique),
        "direct_overlap_pdb_ids": hit,
        "pass_no_direct_leakage": len(hit) == 0,
        "rcsb_dates": dates,
    }
    status = "PASS" if report["pass_no_direct_leakage"] else "FAIL"
    print(f"\n[audit] RESULT: {status} (no CASP14 target PDB id in the training corpus)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[audit] report -> {args.out}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
