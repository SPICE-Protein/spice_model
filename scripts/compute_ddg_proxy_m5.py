#!/usr/bin/env python3
"""ΔΔG 代理 v2：用引擎 m5（电荷平衡）信号给归档突变排名（2026-08-14）。

方案 2（短跑，MBP 友好，不拉长窗口）：
- 复用 quick_check 协议（build + equilibrate + 20 步零偏置力短跑）
- 主信号：m5_mean（RL 判 Env_fail 的电荷平衡指示器，越高越不稳）
- 打分：
    m5_ratio = m5_mean(突变体) / m5_mean(同条件 WT)   ← 主排名（<1 = 比 WT 更稳）
    margin   = 存活步数 / 20                          ← 次信号
    ΔU       = <U>(突变体) − <U>(WT)                  ← 能量次信号
- 顺带验证信号：WT 在良性条件(298K/pH7.5) vs 崩溃条件的 m5_mean，确认 m5 确实随应力上升

用法（spice 环境）：
    /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
        scripts/compute_ddg_proxy_m5.py \
        --npz-glob "/Users/redelectricity/Documents/Projects/SPICE/data/second_mut/pseudo_7qf3_*_20.npz" \
        --wt-cif data/7QF3.cif --out /tmp/ddg_m5_second.csv
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

from spice_rl.env.quick_check import quick_check  # noqa: E402
from spice_rl.env.structure import load_structure_with_atoms, structure_from_atoms  # noqa: E402
from spice_rl.env.mutant import _mutant_atoms, build_mutant_structure_from_ca  # noqa: E402

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

RELAX = 50       # 与生产 posttrain.yaml 一致（M5 Pro 轻量）
TOL = 2.0
PRESSURE = 1.0
IONIC = 0.0
N_STEPS = 20     # 短跑窗口（不拉长，MBP 散热）
ANCHOR_PH = 7.5
ANCHOR_T = 298.0


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


def sprint(struct, ph, temp, label):
    t0 = time.time()
    r = quick_check(
        struct, ph=ph, temp=temp, pressure=PRESSURE, ionic=IONIC,
        relax_iters=RELAX, tolerance=TOL, n_steps=N_STEPS,
    )
    if r.get("margin") is None:
        # build/equilibrate 失败：打印真实 reason，避免 None 格式化掩盖错误
        print(f"    [{label}] pH{ph}/T{temp} | FAIL {r.get('reason')} | {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"    [{label}] pH{ph}/T{temp} | ok={r['ok']} margin={r['margin']:.2f} "
              f"m5_mean={r['m5_mean']:.4f} m5_peak={r['m5_peak']:.4f} "
              f"U={r['u']:.0f} | {time.time()-t0:.0f}s", flush=True)
    return r


def cys_sg_geometry(base_atoms: dict, seq: str, wt_seq: str):
    """重建突变体全原子，返回引入 Cys 的最小 SG-SG 距离 + 配对列表。"""
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


def _eval_mutant(task):
    """Worker（进程级）：构建突变体 + 短跑 + SG-SG。task=(ba, seq, wt_seq, ph, t_base, fname)。"""
    ba, seq, wt_seq, ph, t_base, fname = task
    try:
        m_struct = build_mutant_structure_from_ca(ba, seq)
        sg_min, sg_pairs = cys_sg_geometry(ba, seq, wt_seq)
        m = sprint(m_struct, ph, t_base, fname)
        return {"ok": True, "m": m, "sg_min": sg_min, "sg_pairs": sg_pairs}
    except Exception as _e:  # noqa: BLE001
        return {"ok": False, "err": str(_e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--wt-cif", default="data/7QF3.cif",
                    help="WT mmCIF（--wt-parquet-dir 未给时用）")
    ap.add_argument("--wt-parquet-dir", default=None, help="优先：parquet atoms 目录，与 RL 同源")
    ap.add_argument("--wt-pdb-id", default=None, help="parquet 下的 PDB id（缺省从 npz 文件名推断）")
    ap.add_argument("--candidates-csv", default=None, help="可选：pathb_candidates.csv → 关联 SPICE Q")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发评估突变体的进程数（fork；1=顺序）。多核加速，引擎单线程故进程级并行")
    ap.add_argument("--out", default="/tmp/ddg_m5.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(args.npz_glob))
    if not files:
        print("无 npz 文件:", args.npz_glob)
        return 1

    if args.wt_parquet_dir:
        pdb = (args.wt_pdb_id or "").upper()
        if not pdb:
            m = re.search(r"pseudo_([a-z0-9]+)_ep", os.path.basename(files[0]))
            pdb = m.group(1).upper() if m else None
        wt_struct, ba = load_structure_with_atoms(args.wt_parquet_dir, pdb)
        print(f"WT: parquet {pdb}（{len(ba['res_seq'])} 残基）\n")
    else:
        ba = cif_to_base_atoms(args.wt_cif)
        wt_struct = structure_from_atoms(
            ba["atom_names"], ba["elements"], ba["res_seq"], ba["res_names"], ba["coords"],
        )
    wt_seq = ""
    prev = None
    for sid, rname in zip(ba["res_seq"], ba["res_names"]):
        if sid != prev:
            wt_seq += AA3.get(rname, "?")
            prev = sid

    q_map = {}
    if args.candidates_csv and os.path.exists(args.candidates_csv):
        with open(args.candidates_csv) as _f:
            for _r in csv.DictReader(_f):
                if _r.get("survived") == "1":
                    q_map[(_r["tag"], _r["mut_seq"])] = float(_r["q"])
        print(f"关联 SPICE Q: {len(q_map)} 条存活候选\n")
    print(f"WT: {len(wt_seq)} aa\n")

    # 解析文件（纯 numpy，不碰引擎/TF —— 必须先于 fork）
    tasks = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        seq = str(d["seq"])
        env = np.asarray(d["env"], np.float32)
        ph, t_base = float(env[0]), float(env[1])
        tasks.append((f, seq, ph, t_base))

    payloads = [(ba, seq, wt_seq, ph, t_base, os.path.basename(f))
                for f, seq, ph, t_base in tasks]

    results = {}
    wt_cache = {}
    if args.workers and args.workers > 1:
        import multiprocessing as _mp
        # ⚠️ fork 必须在父进程起任何引擎线程之前（本脚本已 TF-free：mutant 模块纯 numpy）。
        #   否则子进程继承活线程状态 → worker 里引擎调用死锁（实测 15min 无进展）。
        #   父进程的 WT 短跑在 fork 之后才碰引擎，与 worker 并行（map_async 立即投递全部任务）。
        ctx = _mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            async_res = pool.map_async(_eval_mutant, payloads, chunksize=4)
            print("[信号校验] WT 短跑:", flush=True)
            anchor = sprint(wt_struct, ANCHOR_PH, ANCHOR_T, "WT@benign")
            print(f"  → 良性 anchor m5_mean = {anchor['m5_mean']:.4f}\n", flush=True)
            for f, seq, ph, t_base in tasks:
                key = round(ph, 2)
                if key not in wt_cache:
                    wt_cache[key] = sprint(wt_struct, ph, t_base, f"WT@pH{ph:.2f}")
            got = async_res.get()
        for (f, _s, _p, _t), res in zip(tasks, got):
            results[f] = res
    else:
        # 顺序（单进程）：先 WT 缓存再评估突变体
        print("[信号校验] WT 短跑:", flush=True)
        anchor = sprint(wt_struct, ANCHOR_PH, ANCHOR_T, "WT@benign")
        print(f"  → 良性 anchor m5_mean = {anchor['m5_mean']:.4f}\n", flush=True)
        for f, seq, ph, t_base in tasks:
            key = round(ph, 2)
            if key not in wt_cache:
                wt_cache[key] = sprint(wt_struct, ph, t_base, f"WT@pH{ph:.2f}")
        for (f, _s, _p, _t), task in zip(tasks, payloads):
            results[f] = _eval_mutant(task)

    rows = []
    for f, seq, ph, t_base in tasks:
        wt = wt_cache[round(ph, 2)]
        res = results[f]
        if not res["ok"]:
            print(f"  [评估失败] {os.path.basename(f)}: {res['err']}", flush=True)
            continue
        m = res["m"]
        sg_min, sg_pairs = res["sg_min"], res["sg_pairs"]
        muts = []
        for i, (a, b) in enumerate(zip(seq, wt_seq)):
            if a != b:
                muts.append(f"{i+1}:{b}>{a}")
        m5_ratio = m["m5_mean"] / wt["m5_mean"] if wt["m5_mean"] > 0 else float("nan")
        _m = re.search(r"pseudo_([a-z0-9]+)_ep", os.path.basename(f))
        _tag = _m.group(1) if _m else ""
        rows.append({
            "file": os.path.basename(f), "pH": round(ph, 2), "T": round(t_base, 1),
            "muts": ";".join(muts),
            "margin": m["margin"], "m5_mean": round(m["m5_mean"], 4),
            "WT_m5_mean": round(wt["m5_mean"], 4), "m5_ratio": round(m5_ratio, 3),
            "U": round(m["u"], 1), "WT_U": round(wt["u"], 1), "dU": round(m["u"] - wt["u"], 1),
            "q": q_map.get((_tag, seq), ""),
            "cys_sg_min": (f"{sg_min:.2f}" if sg_min is not None else ""),
            "cys_pairs": ";".join(f"{a}-{b}:{d:.1f}" for a, b, d in sg_pairs),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== m5 稳定性排名（m5_ratio 升序，<1 = 比 WT 更稳）=== {args.out}")
    print(f"{'file':<22}{'muts':<34}{'margin':>7}{'m5_r':>6}{'m5_wt':>8}{'m5_mut':>8}{'dU':>9}")
    for r in sorted(rows, key=lambda x: x["m5_ratio"] if x["m5_ratio"] == x["m5_ratio"] else 1e9):
        print(f"{r['file']:<22}{r['muts']:<34}{r['margin']:>7.2f}{r['m5_ratio']:>6.2f}"
              f"{r['WT_m5_mean']:>8.3f}{r['m5_mean']:>8.3f}{r['dU']:>9.1f}")

    # === 汇总报告 ===
    def _fl(x):
        try:
            return float(x)
        except Exception:  # noqa: BLE001
            return None
    ratios = [x for x in (_fl(r["m5_ratio"]) for r in rows) if x is not None]
    n_stab = sum(1 for x in ratios if x < 1.0)
    strong = [r for r in rows if _fl(r["cys_sg_min"]) is not None and _fl(r["cys_sg_min"]) < 4.5]
    cand = [r for r in rows if _fl(r["cys_sg_min"]) is not None and _fl(r["cys_sg_min"]) < 6.0]
    lines = ["=== m5 ΔΔG 代理报告 ===",
             f"来源: {args.npz_glob}",
             f"评估 {len(rows)} 个存活体 | m5_ratio<1(比WT稳) {n_stab}/{len(rows)}"
             + (f" | 中位 m5_ratio {sorted(ratios)[len(ratios)//2]:.3f}" if ratios else "")]
    qpairs = [(_fl(r["q"]), _fl(r["m5_ratio"])) for r in rows
              if _fl(r["q"]) is not None and _fl(r["m5_ratio"]) is not None]
    if len(qpairs) >= 5:
        try:
            from scipy import stats  # noqa: PLC0415
            rho, p = stats.spearmanr([a for a, _ in qpairs], [b for _, b in qpairs])
            lines.append(f"SPICE Q ↔ m5_ratio: Spearman ρ={rho:.3f} (p={p:.3g}, n={len(qpairs)})")
        except Exception:  # noqa: BLE001
            pass
    lines.append("")
    lines.append("— 二硫候选（SG-SG <4.5Å 强 / <6Å 候选）—")
    if not strong and not cand:
        lines.append("  无")
    for r in sorted(strong + cand, key=lambda x: _fl(x["cys_sg_min"])):
        kind = "STRONG" if _fl(r["cys_sg_min"]) < 4.5 else "CAND"
        lines.append(f"  [{kind}] {r['file']}  SG-SG_min={r['cys_sg_min']}Å  pairs={r['cys_pairs']}")
    rep = os.path.splitext(args.out)[0] + "_report.txt"
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[report] -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
