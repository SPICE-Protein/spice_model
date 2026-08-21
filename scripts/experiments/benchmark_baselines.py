#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _parse_mutations(mut_str: str, wt: str):
    out = []
    if not mut_str:
        return out
    for part in str(mut_str).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            pos, chg = part.split(":")
            old, new = chg.split(">")
            out.append((int(pos) - 1, old, new))  
        except Exception:  # noqa: BLE001
            continue
    return out


def _apply_mut(wt: str, muts):
    seq = list(wt)
    for i, old, new in muts:
        if 0 <= i < len(seq) and seq[i] == old:
            seq[i] = new
    return "".join(seq)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="runs/posttrain/pathb_candidates.csv")
    ap.add_argument("--out", default="runs/ablation/baselines")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.candidates):
        print(f"[skip] 未找到 {args.candidates}（先跑 RL 生成 pathb_candidates.csv）")
        return 1

    df = pl.read_csv(args.candidates)
    df = df.sort("fitness", descending=True).unique(subset=["tag", "mutations"], keep="first")

    wt_by_tag = {}
    for r in df.to_dicts():
        wt_by_tag[r["tag"]] = r["mut_seq"]  
    wt_cache = {}
    for r in df.to_dicts():
        tag = r["tag"]
        if tag in wt_cache:
            continue
        muts = _parse_mutations(r["mutations"], r["mut_seq"])
        seq = list(r["mut_seq"])
        for i, old, new in muts:  
            if 0 <= i < len(seq) and seq[i] == new:
                seq[i] = old
        wt_cache[tag] = "".join(seq)

    ready = []
    for r in df.to_dicts():
        wt = wt_cache.get(r["tag"], "")
        ready.append({
            "tag": r["tag"], "mutations": r["mutations"],
            "wt_seq": wt, "mut_seq": r["mut_seq"],
            "fitness": r["fitness"], "q": r["q"], "survived": r["survived"],
            "ph": r["ph"], "temp": r["temp"],
        })

    ready_path = os.path.join(args.out, "ready.csv")
    with open(ready_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ready[0].keys()))
        w.writeheader()
        for r in ready:
            w.writerow(r)
    print(f"-> {ready_path} ({len(ready)} 个唯一突变)")

    foldx = shutil.which("foldx") or os.environ.get("FOLDX")
    if foldx:
        print("[foldx] 检测到，跑 BuildModel（需 .pdb 结构 + 序列映射；见文档）")
    else:
        print("[foldx] 未安装（需 Structure + foldx），已导出 ready.csv，可在超算跑")

    rosetta = shutil.which("ddg_monomer") or shutil.which("cartesian_ddg")
    if rosetta:
        print("[rosetta] 检测到，跑 ddg（需 structure + resfile；见文档）")
    else:
        print("[rosetta] 未安装，已导出 ready.csv")

    print("下一步：把 ready.csv 喂给 FoldX/Rosetta（或实验室），"
          "再把 ΔΔG 与 SPICE fitness 合成 summary.csv 画对比图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
