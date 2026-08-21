#!/usr/bin/env python3
"""ΔΔG 代理：对归档突变批次做引擎稳定性余量排名（2026-08-14）。

把 Q"闸门"升级成"排名器"：在每株自己的崩溃条件下建结构，引擎 equilibrate 后：
1. 基线下跑 20 步（零偏置力）→ 存活步数 + 平均势能 <U>
2. set_temperature 向上扫描（+10…+80K）→ 找到还能撑满 20 步的最高温度 T_crit
3. ΔΔG 代理：
   - ΔT_margin = T_crit(突变体) − T_crit(同条件 WT)   ← 主排名（热稳定性余量）
   - ΔU        = <U>(突变体) − <U>(WT)                ← 次级能量代理（kcal/mol，更负=更稳）

用法（spice 环境）：
    /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
        scripts/compute_ddg_proxy.py \
        --npz-glob "/Users/redelectricity/Documents/Projects/SPICE/data/first_mut/pseudo_7qf3_*_20.npz" \
        --wt-cif data/7QF3.cif --out /tmp/ddg_proxy_first.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spice_engine as se  # noqa: E402
from spice_rl.env.structure import load_structure_with_atoms, structure_from_atoms  # noqa: E402
from spice_rl.train_post import _mutant_atoms, build_mutant_structure_from_ca  # noqa: E402

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

STEPS = 20          # 存活窗口（与 RL 一致）
FORCE_DIM = 16      # bias-force 基向量维度
PRESSURE = 1.0
IONIC = 0.0
RELAX = 50
TOL = 2.0
T_LEVELS = [0, 10, 20, 30, 40, 60, 80, 100, 120]  # 相对基线的 ΔT 扫描（+120 防 T_crit 饱和）


def cif_to_base_atoms(cif_path: str):
    atoms = {}
    cols = {}
    in_loop = False
    with open(cif_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("loop_"):
                in_loop = True
                cols = {}
                continue
            if in_loop and s.startswith("_atom_site."):
                cols[s.split(".")[1].strip()] = len(cols)
                continue
            if in_loop and s and not s.startswith("#") and cols:
                flds = s.split()
                if len(flds) >= len(cols):
                    def g(n):
                        return flds[cols[n]]
                    if g("group_PDB") == "ATOM":
                        try:
                            sid = int(g("label_seq_id"))
                        except ValueError:
                            continue
                        key = (sid, g("label_atom_id"))
                        if key not in atoms:
                            atoms[key] = (
                                g("type_symbol"), g("label_comp_id"),
                                np.array([float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z"))], np.float32),
                            )
                continue
            if s.startswith("_"):
                in_loop = False
    groups = {}
    for (sid, aname), (elem, rname, xyz) in atoms.items():
        groups.setdefault(sid, []).append((aname, elem, rname, xyz))
    names, elems, resseq, resnames, coords = [], [], [], [], []
    for sid in sorted(groups):
        for aname, elem, rname, xyz in groups[sid]:
            names.append(aname)
            elems.append(elem)
            resseq.append(sid)
            resnames.append(rname)
            coords.append(xyz)
    return {
        "atom_names": names, "elements": elems, "res_seq": resseq,
        "res_names": resnames, "coords": np.array(coords, np.float32),
    }


def run_window(eng, temp, steps=STEPS):
    """在给定温度下跑 steps 步（零偏置力），返回 (存活步数, 平均势能)。"""
    eng.set_temperature(float(temp))
    survived = 0
    u_sum = 0.0
    for _ in range(steps):
        out = eng.step(np.zeros(FORCE_DIM, np.float32))
        survived += 1
        u_sum += float(eng.u_t_kcal())
        if isinstance(out, dict) and out.get("crashed"):
            break
    return survived, u_sum / max(1, survived)


def evaluate(struct, ph, t_base):
    """建引擎、equilibrate、基线窗口 + 温度扫描。返回指标 dict。"""
    t0 = time.time()
    eng = se.Engine.build(struct, float(ph), float(t_base), PRESSURE, IONIC, RELAX, TOL)
    eng.equilibrate()
    surv_base, u_base = run_window(eng, t_base)
    t_crit = t_base if surv_base >= STEPS else float("nan")
    for dt in T_LEVELS[1:]:
        if surv_base < STEPS:
            break
        surv, _ = run_window(eng, t_base + dt)
        if surv >= STEPS:
            t_crit = t_base + dt
        else:
            break  # 已崩，不再扫更高
    print(f"    [eval] ph={ph} T={t_base} | 基线存活={surv_base}/{STEPS} "
          f"<U>={u_base:.0f} kcal/mol | T_crit={t_crit:.0f}K | {time.time()-t0:.0f}s", flush=True)
    return {"surv_base": surv_base, "u_base": u_base, "t_crit": t_crit}


def cys_sg_geometry(base_atoms: dict, seq: str, wt_seq: str):
    """重建突变体全原子，返回引入 Cys 的最小 SG-SG 距离 + 配对列表。

    真二硫 SG-SG ≈ 2.05 Å；构建结构里 <4.5 Å 为强候选，4.5-6 Å 可能，>6 Å 不太可能。
    仅统计从 WT 新引入的 Cys（seq != wt_seq 且新 AA=C）。无/单 Cys 返回 (None, []).
    """
    introduced = [i for i, (a, b) in enumerate(zip(seq, wt_seq)) if a != b and a == "C"]
    if len(introduced) < 2:
        return None, []
    _names, _elems, seqs, _resnames, coords = _mutant_atoms(base_atoms, seq)
    uniq = []
    for rs in seqs:
        if rs not in uniq:
            uniq.append(rs)
    sg_by = {}
    for name, rs, xyz in zip(_names, seqs, coords):
        if name == "SG":
            sg_by.setdefault(rs, []).append(xyz)
    pairs = []
    for i in range(len(introduced)):
        for j in range(i + 1, len(introduced)):
            ra, rb = uniq[introduced[i]], uniq[introduced[j]]
            if ra in sg_by and rb in sg_by:
                for xa in sg_by[ra]:
                    for xb in sg_by[rb]:
                        d = float(np.linalg.norm(np.asarray(xa) - np.asarray(xb)))
                        pairs.append((introduced[i] + 1, introduced[j] + 1, round(d, 2)))
    if not pairs:
        return None, []
    return min(p[2] for p in pairs), pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--wt-cif", default="data/7QF3.cif",
                    help="WT mmCIF（--wt-parquet-dir 未给时用）")
    ap.add_argument("--wt-parquet-dir", default=None,
                    help="优先：parquet atoms 目录，与 RL 同源同链选择")
    ap.add_argument("--wt-pdb-id", default=None,
                    help="parquet 下的 PDB id（缺省从 npz 文件名 tag 推断）")
    ap.add_argument("--out", default="/tmp/ddg_proxy.csv")
    ap.add_argument("--candidates-csv", default=None,
                    help="可选：pathb_candidates.csv（按 tag+seq 关联 SPICE Q，报告里算 Q↔dT_margin 相关）")
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        print("无 npz 文件:", args.npz_glob)
        return 1

    q_map = {}
    if args.candidates_csv and os.path.exists(args.candidates_csv):
        with open(args.candidates_csv) as _f:
            for _r in csv.DictReader(_f):
                if _r.get("survived") == "1":
                    q_map[(_r["tag"], _r["mut_seq"])] = float(_r["q"])
        print(f"关联 SPICE Q: {len(q_map)} 条存活候选\n")

    if args.wt_parquet_dir:
        pdb = (args.wt_pdb_id or "").upper()
        if not pdb:
            m = re.search(r"pseudo_([a-z0-9]+)_ep", os.path.basename(files[0]))
            pdb = m.group(1).upper() if m else None
        wt_struct, base_atoms = load_structure_with_atoms(args.wt_parquet_dir, pdb)
        print(f"WT: parquet {pdb}（{len(base_atoms['res_seq'])} 残基）\n")
    else:
        base_atoms = cif_to_base_atoms(args.wt_cif)
        wt_struct = structure_from_atoms(
            base_atoms["atom_names"], base_atoms["elements"], base_atoms["res_seq"],
            base_atoms["res_names"], base_atoms["coords"],
        )
    wt_seq = "".join(AA3.get(r, "?") for r in base_atoms["res_names"]) if False else None
    # WT 序列：按 res_seq 每残基一个
    wt_seq = ""
    prev = None
    for sid, rname in zip(base_atoms["res_seq"], base_atoms["res_names"]):
        if sid != prev:
            wt_seq += AA3.get(rname, "?")
            prev = sid
    print(f"WT: {len(wt_seq)} aa（{args.wt_cif}）\n")

    # 先算 WT 在各条件的基准
    wt_cache = {}
    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        seq = str(d["seq"])
        env = np.asarray(d["env"], np.float32)
        ph, t_base, ionic = float(env[0]), float(env[1]), float(env[2])
        key = round(ph, 2)
        if key not in wt_cache:
            print(f"[WT @ pH{ph:.2f}]", flush=True)
            wt_cache[key] = evaluate(wt_struct, ph, t_base)
        wt = wt_cache[key]

        print(f"[mut {os.path.basename(f)}] pH{ph:.1f}/T{t_base:.0f} 构建中…", flush=True)
        try:
            m_struct = build_mutant_structure_from_ca(base_atoms, seq)
            sg_min, sg_pairs = cys_sg_geometry(base_atoms, seq, wt_seq)
            m = evaluate(m_struct, ph, t_base)
        except Exception as _e:  # noqa: BLE001
            print(f"[mut {os.path.basename(f)}] 评估失败，跳过: {_e}", flush=True)
            continue

        muts = []
        for i, (a, b) in enumerate(zip(seq, wt_seq)):
            if a != b:
                muts.append(f"{i+1}:{b}>{a}")
        dt_margin = m["t_crit"] - wt["t_crit"]
        du = m["u_base"] - wt["u_base"]
        _m = re.search(r"pseudo_([a-z0-9]+)_ep", os.path.basename(f))
        _tag = _m.group(1) if _m else ""
        rows.append({
            "file": os.path.basename(f), "pH": round(ph, 2), "T_base": round(t_base, 1),
            "muts": ";".join(muts), "surv_base": m["surv_base"], "u_base": round(m["u_base"], 1),
            "T_crit": round(m["t_crit"], 1), "WT_T_crit": round(wt["t_crit"], 1),
            "dT_margin": round(dt_margin, 1), "dU": round(du, 1),
            "q": q_map.get((_tag, seq), ""),
            "cys_sg_min": (f"{sg_min:.2f}" if sg_min is not None else ""),
            "cys_pairs": ";".join(f"{a}-{b}:{d:.1f}" for a, b, d in sg_pairs),
        })

    # 输出
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== ΔΔG 代理排名（按 dT_margin 降序）=== {args.out}")
    print(f"{'file':<22}{'muts':<34}{'surv':>4}{'dT_margin':>10}{'dU':>9}")
    for r in sorted(rows, key=lambda x: (-(x["dT_margin"] if x["dT_margin"] == x["dT_margin"] else -1e9),)):
        print(f"{r['file']:<22}{r['muts']:<34}{r['surv_base']:>4}{r['dT_margin']:>10.1f}{r['dU']:>9.1f}")

    # === 汇总报告（含二硫候选升级判定 + 可选 Q 关联）===
    def _fl(x):
        try:
            return float(x)
        except Exception:  # noqa: BLE001
            return None

    dts = [x for x in (_fl(r["dT_margin"]) for r in rows) if x is not None]
    n_surv = sum(1 for r in rows if (_fl(r["surv_base"]) or 0) >= STEPS)
    n_pos = sum(1 for x in dts if x > 0)
    strong = [r for r in rows if _fl(r["cys_sg_min"]) is not None and _fl(r["cys_sg_min"]) < 4.5]
    cand = [r for r in rows if _fl(r["cys_sg_min"]) is not None and _fl(r["cys_sg_min"]) < 6.0]
    lines = ["=== ΔΔG 代理报告 ===",
             f"来源: {args.npz_glob}",
             f"评估 {len(rows)} 个存活体 | 满窗存活 {n_surv} | dT_margin>0 {n_pos}/{len(rows)}"
             + (f" | 中位 dT_margin {sorted(dts)[len(dts)//2]}" if dts else "")]
    qpairs = [(_fl(r["q"]), _fl(r["dT_margin"])) for r in rows
              if _fl(r["q"]) is not None and _fl(r["dT_margin"]) is not None]
    if len(qpairs) >= 5:
        try:
            from scipy import stats  # noqa: PLC0415
            rho, p = stats.spearmanr([a for a, _ in qpairs], [b for _, b in qpairs])
            lines.append(f"SPICE Q ↔ dT_margin: Spearman ρ={rho:.3f} (p={p:.3g}, n={len(qpairs)})")
        except Exception:  # noqa: BLE001
            pass
    lines.append("")
    lines.append("— 二硫候选（SG-SG <4.5Å 强 / <6Å 候选；空=Cα-only 未测）—")
    if not strong and not cand:
        lines.append("  无")
    for r in sorted(strong + cand, key=lambda x: _fl(x["cys_sg_min"])):
        kind = "STRONG" if _fl(r["cys_sg_min"]) < 4.5 else "CAND"
        lines.append(f"  [{kind}] {r['file']}  SG-SG_min={r['cys_sg_min']}Å  "
                     f"pairs={r['cys_pairs']}  dT_margin={r['dT_margin']}")
    rep = os.path.splitext(args.out)[0] + "_report.txt"
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[report] -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
