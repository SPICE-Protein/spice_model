#!/usr/bin/env python3
"""原始汤物理筛原型（soup_screen）：随机 aa 序列 → 引擎纯物理 quick_check。

验证"物理是第一选择者"假说：如果物理势能在**没有进化史/没有折叠语法**的
随机序列上也能筛选出（哪怕极少）稳定、类折叠的幸存者，则"物理先于数据/先于选择"
从比喻变成可测 claim。若随机池筛不出任何折叠体，则 PDB 的进化偏置在偷偷干活，
论文只能守"筛子在已可行折叠邻域内选择"的诚实版。

两种模式：
  --mode backbone : 真实折叠骨架（如 7QF3）+ 随机序列侧链。build 必成功（真实骨架），
                    引擎筛"随机组成在可行骨架上稳不稳"。干净、便宜，作第一信号。
  --mode denovo   : 扩展链（phi/psi 延展）骨架 + 随机侧链 = 更接近"原汤全新链"。
                    起始构象是高位能随机卷曲，equilibrate 会大量爆炸，属预期（即物理拒绝）。

用法（spice 环境）：
  PYTHONPATH=. /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
      scripts/experiments/soup_screen.py --mode backbone --n 20 --len 116 \
      --wt-cif model/data/7QF3.cif --out /tmp/soup.csv --workers 4
  全量 HPC：~100-200 条（多数快速崩溃），2-4 核时。基线自动对比天然 WT + 洗牌 WT。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spice_rl.env.quick_check import quick_check  # noqa: E402
from spice_rl.env.structure import structure_from_atoms  # noqa: E402
from spice_rl.env.mutant import _mutant_atoms  # noqa: E402

# quick_check 参数（与 m5 代理一致）
ANCHOR_PH, ANCHOR_T = 7.5, 298.0
PRESSURE, IONIC = 0.0, 0.0
RELAX, TOL, N_STEPS = 200, 2.0, 20

# 前生命偏差字母表（G/A 丰度最高；V/D/E/P/S/T 常见；R/K 少量）
PREBIOTIC = "GAVDESPTRK"
FULL20 = "ACDEFGHIKLMNPQRSTVWY"

# 扩展链几何（内坐标）
L_NCA, L_CAC, L_CN, L_CO = 1.458, 1.523, 1.329, 1.231
A_NCAC, A_CACN, A_CNCA, A_CA_CO = 111.0, 116.5, 122.0, 117.0
PHI, PSI, OMEGA = -139.0, 135.0, 180.0


def _place_atom(p3, p2, p1, length, angle_deg, dih_deg):
    """由 3 个前一原子放置新原子：键长 p1-p0，键角 p2-p1-p0，二面角 p3-p2-p1-p0。"""
    u = p2 - p1
    u = u / np.linalg.norm(u)
    v = p3 - p2
    v = v / np.linalg.norm(v)
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    ang = np.radians(angle_deg)
    d = u * np.cos(ang) + np.cross(n, u) * np.sin(ang)          # 面内方向
    dih = np.radians(dih_deg)
    d = d * np.cos(dih) + np.cross(u, d) * np.sin(dih) + u * np.dot(u, d) * (1 - np.cos(dih))
    return p1 + d * length


def extended_backbone_base_atoms(n_res: int, coil: bool = True, rng=None):
    """denovo 起点骨架（N/CA/C/O）。

    coil=True：每残基随机 phi/psi（均匀 [-180,180]）→ 链自己卷成紧凑随机卷曲。
      溶剂盒小 ~10×（vs 全伸展），对随机链更公平（不是人工撑直的高位能杆），也更接近汤里形态。
    coil=False：固定扩展 phi/psi → 全伸展杆（旧版，慢且偏置）。
    返回 base_atoms dict 供 _mutant_atoms 放随机侧链。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = np.zeros((n_res, 3))
    CA = np.zeros((n_res, 3))
    C = np.zeros((n_res, 3))
    O = np.zeros((n_res, 3))
    # 种子前 3 原子
    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = np.array([L_NCA, 0.0, 0.0])
    ang = np.radians(180.0 - A_NCAC)
    C[0] = CA[0] + np.array([L_CAC * np.cos(ang), L_CAC * np.sin(ang), 0.0])
    # O(0)：CA-C-O 角 117°，二面角 N-CA-C-O = 180
    O[0] = _place_atom(N[0], CA[0], C[0], L_CO, A_CA_CO, 180.0)
    for i in range(n_res - 1):
        psi_i = float(rng.uniform(-180.0, 180.0)) if coil else PSI
        phi_next = float(rng.uniform(-180.0, 180.0)) if coil else PHI
        N[i + 1] = _place_atom(N[i], CA[i], C[i], L_CN, A_CACN, psi_i)         # psi
        CA[i + 1] = _place_atom(CA[i], C[i], N[i + 1], L_NCA, A_CNCA, OMEGA)    # omega=trans
        C[i + 1] = _place_atom(C[i], N[i + 1], CA[i + 1], L_CAC, A_NCAC, phi_next)  # phi
        O[i + 1] = _place_atom(N[i + 1], CA[i + 1], C[i + 1], L_CO, A_CA_CO, 180.0)
    names, elems, resseq, resnames, coords = [], [], [], [], []
    for i in range(n_res):
        for name, elem, xyz in (("N", "N", N[i]), ("CA", "C", CA[i]),
                                ("C", "C", C[i]), ("O", "O", O[i])):
            names.append(name)
            elems.append(elem)
            resseq.append(i + 1)
            # base 用 GLY：G 位"未突变"时 GLY 天然无侧链=合法；其余 AA 由 _mutant_atoms 重建侧链
            resnames.append("GLY")
            coords.append(xyz)
    return {"atom_names": names, "elements": elems, "res_seq": resseq,
            "res_names": resnames, "coords": np.asarray(coords, np.float32)}


def rand_seq(n: int, alphabet: str, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list(alphabet), size=n))


def _run_one(ba, seq, label):
    """建结构 + quick_check，返回 dict。TF-free，可作 worker。"""
    t0 = time.time()
    try:
        names, elems, seqs, resnames, coords = _mutant_atoms(ba, seq)
        struct = structure_from_atoms(names, elems, seqs, resnames, coords)
        r = quick_check(struct, ph=ANCHOR_PH, temp=ANCHOR_T, pressure=PRESSURE,
                        ionic=IONIC, relax_iters=RELAX, tolerance=TOL, n_steps=N_STEPS)
        out = {
            "ok": True, "seq": seq, "label": label,
            "margin": r.get("margin"), "survive": r.get("survived"),
            "m5": r.get("m5_mean"), "u": r.get("u"), "rg": r.get("rg"),
            "atoms": (names, elems, seqs, resnames, coords),  # 供 --out-mmifs 写 RL 亲本
            "sec": time.time() - t0,
        }
        print(f"  [{label}] margin={out['margin']} m5={out['m5']} U={out['u']} "
              f"surv={out['survive']}/{N_STEPS} | {out['sec']:.0f}s", flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"  [{label}] FAIL {type(e).__name__}: {str(e)[:80]} | {time.time()-t0:.0f}s", flush=True)
        return {"ok": False, "seq": seq, "label": label, "err": str(e),
                "margin": None, "survive": 0, "m5": None, "u": None, "rg": None}


def _run_task(task):
    """pool worker 入口：解包 (ba, seq, label)。"""
    return _run_one(*task)


def _write_mmcif(path, names, elems, seqs, resnames, coords):
    """最小 mmCIF writer（_atom_site loop），供 train_post --structure 当亲本加载。"""
    cols = ["_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
            "_atom_site.label_atom_id", "_atom_site.label_comp_id",
            "_atom_site.label_asym_id", "_atom_site.label_seq_id",
            "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
            "_atom_site.occupancy", "_atom_site.B_iso_or_equiv"]
    with open(path, "w") as f:
        f.write("data_soup\n#\nloop_\n")
        for c in cols:
            f.write(c + "\n")
        f.write("#\n")
        for i, (nm, el, rs, rn, xyz) in enumerate(zip(names, elems, seqs, resnames, coords)):
            f.write(f"ATOM  {i+1:5d} {el:>2s} {nm:<4s} {rn:>3s} A {int(rs):4d} "
                    f"{float(xyz[0]):8.3f} {float(xyz[1]):8.3f} {float(xyz[2]):8.3f}  1.00  0.00\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["backbone", "denovo"], default="backbone")
    ap.add_argument("--n", type=int, default=20, help="随机序列条数")
    ap.add_argument("--len", type=int, default=116, help="序列长度（backbone 模式自动用 WT 长度）")
    ap.add_argument("--wt-cif", default="model/data/7QF3.cif", help="backbone 模式：天然折叠骨架")
    ap.add_argument("--alphabet", choices=["prebiotic", "full20"], default="prebiotic")
    ap.add_argument("--start", choices=["coil", "extended"], default="coil",
                    help="denovo 起点：coil=紧凑随机卷曲（默认，便宜+公平）/ extended=全伸展杆（慢）")
    ap.add_argument("--shuffle-wt", action="store_true", help="加一条 WT 洗牌序列作对照")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="/tmp/soup.csv")
    ap.add_argument("--out-mmifs", default=None,
                    help="把物理接受的稳定幸存者写成 mmCIF（RL 亲本）→ 之后 train_post --structure 让 SPICE 玩")
    ap.add_argument("--min-survive", type=int, default=N_STEPS,
                    help="写 mmCIF 的最低存活步数（默认=N_STEPS，只要完全稳定者当 RL 亲本）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alphabet = PREBIOTIC if args.alphabet == "prebiotic" else FULL20

    # 纯 numpy 准备（不碰引擎/线程，必须先于 fork）
    if args.mode == "backbone":
        # 从 cif 读 WT 骨架
        import re
        ba = None
        for path in (args.wt_cif, os.path.join(os.getcwd(), args.wt_cif)):
            if os.path.exists(path):
                ba = _cif_to_base_atoms(path)
                break
        if ba is None:
            print("找不到 WT cif:", args.wt_cif)
            return 1
        n_res = len(set(ba["res_seq"]))
        wt_seq = "".join(_AA3_1.get(ba["res_names"][i], "?") for i in _first_idx(ba["res_seq"]))
        base_label = os.path.basename(args.wt_cif).split(".")[0]
        seqs = [rand_seq(n_res, alphabet, rng) for _ in range(args.n)]
        labels = [f"rand{i}" for i in range(args.n)]
        baseline = [("WT", wt_seq), ("shuffleWT", _shuffle(wt_seq, rng))] if args.shuffle_wt \
            else [("WT", wt_seq)]
    else:
        ba = extended_backbone_base_atoms(args.len, coil=(args.start == "coil"), rng=rng)
        n_res = args.len
        seqs = [rand_seq(n_res, alphabet, rng) for _ in range(args.n)]
        labels = [f"denovo{i}" for i in range(args.n)]
        baseline = []

    # fork 池（此刻单线程，TF-free）→ worker 里建结构 + quick_check
    import multiprocessing as _mp
    ctx = _mp.get_context("fork")
    payloads = [(ba, seq, label) for seq, label in zip(seqs, labels)]
    results = {}
    if args.workers and args.workers > 1:
        with ctx.Pool(args.workers) as pool:
            got = pool.map_async(_run_task, payloads, chunksize=4).get()
    else:
        got = [_run_one(*p) for p in payloads]
    for p, g in zip(payloads, got):
        results[p[2]] = g

    # 基线：天然 WT（+ 洗牌）在父进程跑（fork 后，不碰并发）
    base_results = []
    for blabel, bseq in baseline:
        if len(bseq) == n_res:
            base_results.append((blabel, _run_one(ba, bseq, blabel)))
        else:
            print(f"  [skip] {blabel} 长度 {len(bseq)} != 骨架 {n_res}")

    # 汇总
    rows = [g for g in results.values()]
    survived = [g for g in rows if g["ok"] and g.get("survive")]
    rate = len(survived) / max(1, len(rows))
    print(f"\n=== 原始汤物理筛（{args.mode} / {alphabet}）===")
    print(f"随机链 {len(rows)} 条 | 存活(20步) {len(survived)} 条 | 存活率 {rate:.3f}")
    if survived:
        ms = [g["m5"] for g in survived if g["m5"] is not None]
        print(f"  存活者 m5 中位: {float(np.median(ms)):.4f}" if ms else "  存活者无 m5")
    margins = [g["margin"] for g in rows if g.get("margin") is not None]
    if margins:
        print(f"  全体 margin 中位: {float(np.median(margins)):.3f}（<1=部分崩溃）")
    for blabel, b in base_results:
        print(f"  基线 {blabel}: margin={b['margin']} m5={b['m5']} survive={b['survive']}/{N_STEPS}")

    # 写 CSV
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["label", "seq", "ok", "margin", "survive", "m5", "u", "rg", "sec"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for g in rows:
            w.writerow({k: g.get(k) for k in fieldnames})
    print(f"\n[out] -> {args.out}")

    # 写 RL 亲本 mmCIF（物理接受的稳定幸存者）→ train_post --structure 让 SPICE 玩
    if args.out_mmifs:
        os.makedirs(args.out_mmifs, exist_ok=True)
        n_written = 0
        for g in rows:
            if g.get("ok") and g.get("survive", 0) >= args.min_survive and g.get("atoms"):
                names, elems, seqs, resnames, coords = g["atoms"]
                path = os.path.join(args.out_mmifs, f"{g['label']}_s{g['survive']}.cif")
                _write_mmcif(path, names, elems, seqs, resnames, coords)
                n_written += 1
        print(f"[mmCIF] 写出 {n_written} 个 RL 亲本 -> {args.out_mmifs}/")
    return 0


# ---- 轻量 mmCIF → base_atoms（骨架模式用；只取 N/CA/C/O 重原子）----
def _first_idx(res_seq):
    seen, out = set(), []
    for i, s in enumerate(res_seq):
        if s not in seen:
            seen.add(s)
            out.append(i)
    return out


_AA3_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


def _cif_to_base_atoms(path):
    atoms = {}
    in_loop, cols = False, {}
    for line in open(path):
        s = line.strip()
        if s.startswith("loop_"):
            in_loop, cols = True, {}
            continue
        if in_loop and s.startswith("_atom_site."):
            cols[s.split(".")[1].strip()] = len(cols)
            continue
        if in_loop and s and not s.startswith("#") and cols:
            flds = s.split()
            if len(flds) >= len(cols):
                def g(n):
                    return flds[cols[n]]
                # 保留全部重原子（N/CA/C/O + 侧链），否则 strict_incomplete build 失败
                if g("group_PDB") == "ATOM" and g("type_symbol") != "H":
                    try:
                        sid = int(g("label_seq_id"))
                    except ValueError:
                        continue
                    key = (sid, g("label_atom_id"))
                    if key not in atoms:
                        atoms[key] = (g("type_symbol"), g("label_comp_id"),
                                      np.array([float(g("Cartn_x")), float(g("Cartn_y")),
                                                float(g("Cartn_z"))], np.float32))
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
    return {"atom_names": names, "elements": elems, "res_seq": resseq,
            "res_names": resnames, "coords": np.array(coords, np.float32)}


def _shuffle(seq: str, rng: np.random.Generator) -> str:
    arr = list(seq)
    rng.shuffle(arr)
    return "".join(arr)


if __name__ == "__main__":
    raise SystemExit(main())
