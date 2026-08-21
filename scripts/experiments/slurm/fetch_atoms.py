#!/usr/bin/env python3
import sys

if sys.version_info[0] < 3:
    sys.exit("需要 Python 3：先 conda activate spice，或用 ~/miniconda3/envs/spice/bin/python")

import argparse
import os
import shutil

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import polars as pl
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "SPICE-Protein/spice_protein"


def _find_atoms_shard(pdb_id: str, repo_files) -> str:
    target = pdb_id.upper()
    for f in repo_files:
        if not f.startswith("entries_shard_") or not f.endswith(".parquet"):
            continue
        try:
            df = pl.read_parquet(hf_hub_download(REPO, f, repo_type="dataset"))
        except Exception:  # noqa: BLE001
            continue
        row = df.filter(pl.col("pdb_id") == target)
        if row.height:
            if "_src" in df.columns:
                return str(row["_src"][0]).replace("entries_shard_", "atoms_shard_")
            return f.replace("entries_shard_", "atoms_shard_")
    raise SystemExit(f"[error] pdb_id {target} 不在任何 entries shard")


def _scan_atoms_for(pdb_id: str, repo_files, out) -> str | None:
    target = pdb_id.upper()
    for f in sorted(repo_files):
        if not f.startswith("atoms_shard_") or not f.endswith(".parquet"):
            continue
        dst = os.path.join(out, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                p = hf_hub_download(REPO, f, repo_type="dataset")
                shutil.copy(p, dst)
            except Exception:  # noqa: BLE001
                continue
        try:
            if pl.read_parquet(dst, columns=["pdb_id"]).filter(
                    pl.col("pdb_id") == target).height:
                return dst
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-id", action="append", default=[])
    ap.add_argument("--list", default=None, help="每行一个 pdb_id 的文件")
    ap.add_argument("--out", default="data/parquet_hpc")
    args = ap.parse_args()

    ids = list(args.pdb_id)
    if args.list:
        with open(args.list) as f:
            ids += [ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")]

    os.makedirs(args.out, exist_ok=True)

    def _local_shard(pid: str):
        for f in sorted(os.listdir(args.out)):
            if f.startswith("atoms_shard_") and f.endswith(".parquet"):
                try:
                    df = pl.read_parquet(os.path.join(args.out, f), columns=["pdb_id"])
                    if df.filter(pl.col("pdb_id") == pid.upper()).height:
                        return f
                except Exception:  # noqa: BLE001
                    pass
        return None

    need_network = []
    for pid in ids:
        f = _local_shard(pid)
        if f:
            print(f"[local] {pid}: 已在 {f}，跳过联网")
        else:
            need_network.append(pid)

    if need_network:
        try:
            repo_files = list_repo_files(REPO, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"[error] 集群访问不了 HF（{e}）。请先在本地跑 stage_atoms.sh 拉好 shard 上传。")
            return 1
        for pid in need_network:
            shard = _find_atoms_shard(pid, repo_files)
            dst = os.path.join(args.out, os.path.basename(shard))
            if os.path.exists(dst):
                print(f"[skip] {pid}: {os.path.basename(shard)} 已存在")
                continue
            p = hf_hub_download(REPO, shard, repo_type="dataset")
            shutil.copy(p, dst)
            try:
                ok = pl.read_parquet(dst, columns=["pdb_id"]).filter(
                    pl.col("pdb_id") == pid.upper()).height > 0
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                os.remove(dst)
                print(f"[warn] {pid} 不在推导的 {shard}，全量扫描 atoms shard…")
                found = _scan_atoms_for(pid, repo_files, args.out)
                if not found:
                    print(f"[error] {pid} 的 atoms 数据没找到（可换 HF_ENDPOINT=https://huggingface.co 重试）")
                    continue
                dst = found
            print(f"[ok]   {pid} -> {dst}")
    print(f"[done] parquet dir: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
