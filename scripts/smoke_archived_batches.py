#!/usr/bin/env python3
"""归档突变批次冒烟测试（2026-08-14）。

对 first_mut / second_mut 两批伪标签做质量体检 + FireProtDB 交叉参照：
- 逐株加载 .npz（seq/env/coords）
- 与野生型（默认 data/7QF3.cif 的 Cα 序列，按 label_seq_id 去重 altloc）对出真实突变
- 几何体检：长度、Rg、最小非相邻距离（局部冲突）、native-contact Q（fold 保留）
- 交叉参照 FireProtDB：7QF3 是否在实验库里、有没有该结构的实验突变

注意：序列参照必须是真实 WT（PDB Cα 序列），不要用 log 里的存活体序列当 WT
（2026-08-14 踩坑：n45000 log 第 0 个存活体自带 62:D>I/86:L>P，误当亲本会造出假突变）。

用法：
    /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
        scripts/smoke_archived_batches.py
    # 自定义路径：
    ... --first-glob "/path/to/first_mut/pseudo_7qf3_*_20.npz" \
        --second-glob "/path/to/second_mut/pseudo_7qf3_*_20.npz" \
        --wt-cif data/7QF3.cif --fire-db data/stability_benchmark/fireprotdb/fireprotdb.sqlite3
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import numpy as np

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_cif_ca(cif_path: str):
    """返回 (seq, ca Nx3)，按 label_seq_id 去重 altloc。列名 strip 尾随空格。"""
    seq_rows: dict = {}
    ca: dict = {}
    cols: dict = {}
    in_atom_loop = False
    with open(cif_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("loop_"):
                in_atom_loop = True
                cols = {}
                continue
            if in_atom_loop and s.startswith("_atom_site."):
                cols[s.split(".")[1].strip()] = len(cols)
                continue
            if in_atom_loop and s and not s.startswith("#") and cols:
                flds = s.split()
                if len(flds) >= len(cols):
                    def g(name):
                        return flds[cols[name]]
                    if g("group_PDB") == "ATOM" and g("label_atom_id") == "CA":
                        try:
                            sid = int(g("label_seq_id"))
                        except ValueError:
                            continue
                        if sid not in ca:
                            ca[sid] = np.array(
                                [float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z"))],
                                np.float32,
                            )
                            seq_rows[sid] = AA3.get(g("label_comp_id"), "?")
                continue
            if s.startswith("_"):
                in_atom_loop = False
    ids = sorted(ca)
    return "".join(seq_rows[i] for i in ids), np.array([ca[i] for i in ids], np.float32)


def native_contact_q(coords, reference, cutoff=8.0):
    ref = np.asarray(reference, np.float32)
    c = np.asarray(coords, np.float32)
    L = min(len(ref), len(c))
    pairs = [(i, j) for i in range(L) for j in range(i + 2, L)
             if np.linalg.norm(ref[i] - ref[j]) < cutoff]
    if not pairs:
        return 0.0
    kept = sum(1 for (i, j) in pairs if np.linalg.norm(c[i] - c[j]) < cutoff)
    return kept / len(pairs)


def rg(c):
    c = np.asarray(c, np.float32)
    return float(np.sqrt(np.mean(np.sum((c - c.mean(0)) ** 2, axis=1))))


def min_nonadj(c):
    c = np.asarray(c, np.float32)
    best = 1e9
    for i in range(len(c)):
        for j in range(i + 2, len(c)):
            best = min(best, float(np.linalg.norm(c[i] - c[j])))
    return best


def mutations(seq, wt):
    return [f"{i + 1}:{b}>{a}" for i, (a, b) in enumerate(zip(seq, wt)) if a != b]


def smoke(name, globpat, wt_seq, wt_ca):
    print(f"######## {name} ########")
    files = sorted(glob.glob(globpat))
    if not files:
        print("  (无文件)")
        return
    for f in files:
        d = np.load(f, allow_pickle=True)
        seq = str(d["seq"])
        env = np.asarray(d["env"], np.float32)
        coords = np.asarray(d["coords"], np.float32)
        q = native_contact_q(coords, wt_ca)
        print(
            f"  {os.path.basename(f)}: pH{env[0]:.1f}/T{env[1]:.0f} | L={len(seq)} "
            f"| Rg={rg(coords):.2f} min_nonadj={min_nonadj(coords):.2f} Q={q:.3f} "
            f"| muts={mutations(seq, wt_seq)}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-glob", default="/Users/redelectricity/Documents/Projects/SPICE/data/first_mut/pseudo_7qf3_*_20.npz")
    ap.add_argument("--second-glob", default="/Users/redelectricity/Documents/Projects/SPICE/data/second_mut/pseudo_7qf3_*_20.npz")
    ap.add_argument("--wt-cif", default="data/7QF3.cif")
    ap.add_argument("--fire-db", default="data/stability_benchmark/fireprotdb/fireprotdb.sqlite3")
    args = ap.parse_args()

    wt_seq, wt_ca = parse_cif_ca(args.wt_cif)
    print(f"WT 参照: {len(wt_seq)} aa, Cα {wt_ca.shape}（{args.wt_cif}）\n")

    smoke("first_mut", args.first_glob, wt_seq, wt_ca)
    print()
    smoke("second_mut", args.second_glob, wt_seq, wt_ca)

    print("\n######## FireProtDB 交叉参照 ########")
    if not os.path.exists(args.fire_db):
        print(f"  （fire-db 不存在: {args.fire_db}）")
        return 0
    con = sqlite3.connect(args.fire_db)
    con.row_factory = sqlite3.Row
    r = con.execute(
        "SELECT id, wwpdb, afdb FROM structure WHERE wwpdb=? OR afdb=?",
        ("7QF3", "7QF3"),
    ).fetchall()
    print("structure 表里 7QF3:", dict(r[0]) if r else "未收录 ❌（该蛋白无实验 ΔΔG 覆盖）")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
