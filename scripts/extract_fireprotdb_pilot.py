#!/usr/bin/env python3
"""FireProtDB 回顾式盲测 pilot 抽取 v2（2026-08-14）。

从 fireprotdb.sqlite3 抽"实验 DDG + PDB 结构"的干净、多样、可建变子集：
- 只取 measurement.type='DDG'（实验 ΔΔG），天然排除 *_ML 预测
- 能经 sequence→residue→chain→assembly→structure 接到 PDB
- **按 protein 去重**（经 protein_sequence → protein_id，同一蛋白只留一个 PDB）
- **struct_pos 均匀采样**（避免只取 N 端前 N 个突变）
- 校验 residue.amino_acid == substitution.source_aa（res_aa / aa_match）
- 附带 pH/T、文献年/DOI、res_aa（建突变用 struct_pos）

用法：
  /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
      scripts/extract_fireprotdb_pilot.py --out /tmp/fpdb_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
import sqlite3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/stability_benchmark/fireprotdb/fireprotdb.sqlite3")
    ap.add_argument("--out", default="/tmp/fpdb_pilot.csv")
    ap.add_argument("--max-proteins", type=int, default=10)
    ap.add_argument("--max-mut-per-pdb", type=int, default=400)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    # 每个 PDB×protein：DDG 突变体数（pilot 结构选择，按 protein 去重）
    top = con.execute(
        """
        SELECT st.wwpdb, ps.protein_id, COUNT(DISTINCT m.mutant_id) AS n_mut
        FROM measurement m
        JOIN substitution sub ON sub.mutant_id=m.mutant_id
        JOIN experiment e ON e.id=m.experiment_id
        JOIN mutant mu ON mu.id=m.mutant_id
        JOIN sequence_residue_mapping srm ON srm.sequence_id=mu.source_id
        JOIN residue r ON r.id=srm.residue_id
        JOIN chain c ON c.id=r.chain_id
        JOIN assembly a ON a.id=c.assembly_id
        JOIN structure st ON st.id=a.structure_id
        JOIN protein_sequence ps ON ps.sequence_id=mu.source_id
        WHERE m.type='DDG' AND m.num_value IS NOT NULL AND st.wwpdb IS NOT NULL
        GROUP BY st.wwpdb, ps.protein_id
        ORDER BY n_mut DESC
        """
    ).fetchall()
    # 按 protein 去重，取该蛋白突变最多/最靠前的 PDB
    seen_protein = set()
    pdbs = []
    for r in top:
        if r["protein_id"] in seen_protein:
            continue
        seen_protein.add(r["protein_id"])
        pdbs.append(r["wwpdb"])
        if len(pdbs) >= args.max_proteins:
            break
    print("pilot 结构（按蛋白去重）:", pdbs, flush=True)

    ph = ", ".join("?" * len(pdbs))
    q = f"""
    SELECT
        st.wwpdb,
        c.name AS chain,
        r.struct_position AS struct_pos,
        sub."position" AS seq_pos,
        sub.source_aa, sub.target_aa,
        r.amino_acid AS res_aa,
        ROUND(m.num_value, 3) AS ddg,
        MAX(ea_ph.num_value) AS ph,
        MAX(ea_t.num_value) AS temp,
        pub.year, pub.doi, pub.title
    FROM measurement m
    JOIN substitution sub ON sub.mutant_id=m.mutant_id
    JOIN experiment e ON e.id=m.experiment_id
    JOIN publication pub ON pub.id=e.publication_id
    JOIN mutant mu ON mu.id=m.mutant_id
    JOIN sequence_residue_mapping srm ON srm.sequence_id=mu.source_id
    JOIN residue r ON r.id=srm.residue_id
    JOIN chain c ON c.id=r.chain_id
    JOIN assembly a ON a.id=c.assembly_id
    JOIN structure st ON st.id=a.structure_id
    LEFT JOIN experiment_annotation ea_ph ON ea_ph.experiment_id=m.experiment_id AND ea_ph.type='PH'
    LEFT JOIN experiment_annotation ea_t ON ea_t.experiment_id=m.experiment_id AND ea_t.type='EXP_TEMPERATURE'
    WHERE m.type='DDG' AND m.num_value IS NOT NULL AND st.wwpdb IN ({ph})
    GROUP BY m.id, st.wwpdb, c.name, r.struct_position, sub."position",
             sub.source_aa, sub.target_aa, r.amino_acid, pub.year, pub.doi, pub.title
    """
    rows = con.execute(q, pdbs).fetchall()
    print(f"原始行: {len(rows)}", flush=True)

    # 只留可建变的（source_aa == res_aa，PDB 残基与声明 WT 一致）——
    # FireProtDB 的 seq→结构映射是按位置偏移，非序列一致性；不匹配的不能可靠放上 PDB。
    clean = [dict(r) for r in rows if r["source_aa"] == r["res_aa"]]
    print(f"可建变（source_aa==res_aa）: {len(clean)}/{len(rows)}", flush=True)

    # 每结构：按 struct_pos 排序，均匀采样（避免只取 N 端）
    picked = []
    for pdb in pdbs:
        sub_rows = sorted(
            [r for r in clean if r["wwpdb"] == pdb],
            key=lambda r: (int(r["struct_pos"]), int(r["seq_pos"])),
        )
        if len(sub_rows) > args.max_mut_per_pdb:
            step = len(sub_rows) / args.max_mut_per_pdb
            idxs = sorted({int(i * step) for i in range(args.max_mut_per_pdb)})
            sub_rows = [sub_rows[i] for i in idxs]
        picked.extend(sub_rows)
    print(f"pilot 突变体: {len(picked)}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
        w.writeheader()
        w.writerows(picked)

    from collections import Counter
    cnt = Counter(r["wwpdb"] for r in picked)
    print(f"\n输出: {args.out}")
    for pdb, n in sorted(cnt.items(), key=lambda x: -x[1]):
        ph_ok = sum(1 for r in picked if r["wwpdb"] == pdb and r["ph"] is not None)
        print(f"  {pdb}: {n} 突变（含 pH {ph_ok}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
