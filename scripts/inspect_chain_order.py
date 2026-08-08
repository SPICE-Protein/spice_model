"""检查 atoms parquet 里残基顺序是否沿肽链骨架。

打印几个结构的前 30 个 (chain_id, res_seq, res_name) 及相邻 Cα 距离。
若顺序正确，相邻距离应 ~3.8Å；若被打乱，会出现大间距。
同时统计每个结构有多少条链（链合并可能是乱序来源）。

用法：
    python -m scripts.inspect_chain_order [--shard 0000] [--npdb 3]
"""
from __future__ import annotations

import argparse
import os

import polars as pl

from spice_pre.config import load_config
from spice_pre.data.dataset import list_shards, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0000", help="shard 序号，如 0000")
    ap.add_argument("--npdb", type=int, default=3, help="检查前 N 个结构")
    args = ap.parse_args()

    cfg = load_config("configs/pretrain.yaml")
    entries_f = f"entries_shard_{args.shard}.parquet"
    atoms_f = f"atoms_shard_{args.shard}.parquet"

    # 通过 config 的端点下载/解析（本地走 hf-mirror）
    ep = resolve_path(cfg, entries_f)
    apath = resolve_path(cfg, atoms_f)
    print(f"entries: {ep}\natoms:   {apath}")

    en = pl.read_parquet(ep)
    at = pl.read_parquet(apath).filter(pl.col("is_ca"))
    print(f"entries 行数: {en.height}, CA 原子行数: {at.height}")
    print(f"atoms 列: {at.columns}")
    print(f"res_seq dtype: {at.schema['res_seq']}")

    for pid in en["pdb_id"].to_list()[: args.npdb]:
        sub = at.filter(pl.col("pdb_id") == pid).sort(["chain_id", "res_seq"])
        chains = sub["chain_id"].unique().to_list()
        print(f"\n===== pdb_id={pid} | 链数={len(chains)} | 链={chains} =====")
        rows = sub.head(30).to_dicts()
        for i, r in enumerate(rows):
            line = (
                f"  [{i:>3}] chain={r['chain_id']:<3} res_seq={r['res_seq']:>4} "
                f"{r['res_name']:<3} "
            )
            if i > 0:
                px, py, pz = rows[i - 1]["x"], rows[i - 1]["y"], rows[i - 1]["z"]
                d = float(((r["x"] - px) ** 2 + (r["y"] - py) ** 2 + (r["z"] - pz) ** 2) ** 0.5)
                line += f"| ΔCA={d:6.2f}Å"
            print(line)
        # 相邻距离统计
        c = sub.select(["x", "y", "z"]).to_numpy().astype(float)
        adj = ((c[1:] - c[:-1]) ** 2).sum(axis=1) ** 0.5
        import numpy as np

        print(f"  全链相邻ΔCA: 中位 {np.median(adj):.2f}Å | "
              f"3.3~4.3Å占比 {np.mean((adj>=3.3)&(adj<=4.3))*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
