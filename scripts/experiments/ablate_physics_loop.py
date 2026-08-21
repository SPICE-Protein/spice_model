#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

_PH_LO, _PH_HI, _T_HI = 4.0, 9.0, 340.0


def _is_beyond(ph, temp):
    return ph < _PH_LO or ph > _PH_HI or temp > _T_HI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coverage", default="runs/posttrain/coverage.csv")
    ap.add_argument("--pdb-cloud", default="data/entries_all.parquet",
                    help="全量 PDB 分布（可选，用于背景对比）")
    ap.add_argument("--out", default="runs/ablation/physics_loop.csv")
    args = ap.parse_args()

    rows = []
    rows.append({
        "variant": "off (supervised-only prior)",
        "n_beyond_pdb_points": 0,
        "note": ("coverage 上界=训练分布；无稳定性信号，beyond-PDB 点恒为 0"
                 "（架构性结论，无需运行）"),
    })
    print("[off] supervised-only prior: beyond-PDB 点 = 0（架构上界=数据分布）")

    if os.path.exists(args.coverage):
        cov = pl.read_csv(args.coverage)
        tot = cov.height
        pts = [(float(r["ph"]), float(r["temp"])) for r in cov.to_dicts()]
        n_beyond = sum(1 for ph, t in pts if _is_beyond(ph, t))
        rows.append({
            "variant": "on (physics loop)",
            "n_beyond_pdb_points": n_beyond,
            "note": f"coverage.csv {tot} 个探索点，其中 beyond-PDB {n_beyond} 个"
                    f"（pH<{_PH_LO} 或 >{_PH_HI}，或 T>{_T_HI}）",
        })
        print(f"[on] physics loop: {tot} 个探索点 | beyond-PDB {n_beyond} 个")
    else:
        rows.append({
            "variant": "on (physics loop)",
            "n_beyond_pdb_points": "N/A",
            "note": f"未找到 {args.coverage}（先跑 RL 生成 coverage.csv）",
        })
        print(f"[on] 未找到 coverage.csv: {args.coverage}")

    if os.path.exists(args.pdb_cloud):
        pdb = pl.read_parquet(args.pdb_cloud)
        n_b = pdb.filter(
            (pl.col("ph") < _PH_LO) | (pl.col("ph") > _PH_HI) | (pl.col("temperature") > _T_HI)
        ).height
        rows.append({
            "variant": "PDB 分布（背景）",
            "n_beyond_pdb_points": n_b,
            "note": f"全量 PDB 中 beyond-PDB 条件的结构数（应≈0，证明 PDB 无覆盖）",
        })
        print(f"[pdb] 全量 {pdb.height} 个结构中 beyond-PDB {n_b} 个")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "n_beyond_pdb_points", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
