#!/usr/bin/env python3
"""FireProtDB 引擎相关性 pilot v2（2026-08-14，适配新 SE）。

在 1-2 个蛋白上验证"引擎稳定性信号 ↔ 实验 ΔΔG"的桥。v2 用 SE 新 API：
- `Engine.mutate_with_solvent_reuse()`：WT 引擎建一次，突变体复用其溶剂盒（<0.5s vs ~30s）
- `metrics()['stability_margin']`：>0 稳定 / ≤0 不稳（跨 m1-m5 的组合分数），作预测信号
- 若安装的引擎无新 API，回退旧路径（每个突变完整 build + ΔU）

预测信号：
- 主：margin_mut（突变体自身稳定性余量，越高越稳）
- 次：Δmargin = margin_mut − margin_wt、ΔU = <U_mut> − <U_WT>

用法（spice 环境，async）：
  /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python \
      scripts/compute_fpdb_ddg_corr.py --pdb 1L63 --pdb 1BNI --n 10 --out /tmp/fpdb_corr.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spice_engine as se  # noqa: E402
from spice_rl.env.structure import structure_from_atoms  # noqa: E402
from spice_rl.train_post import build_mutant_structure_from_ca  # noqa: E402

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
CACHE = "/tmp/fpdb_pdbs"
PRESSURE, IONIC, RELAX, TOL, STEPS = 1.0, 0.0, 50, 2.0, 20


def download_cif(pdb: str, cif_dir: str) -> str:
    os.makedirs(cif_dir, exist_ok=True)
    path = os.path.join(cif_dir, f"{pdb}.cif")
    if not os.path.exists(path):
        print(f"  下载 {path} ...", flush=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.cif", path)
    return path


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
                            occ = float(g("occupancy")) if "occupancy" in cols else 0.0
                        except ValueError:
                            continue
                        key = (sid, g("label_atom_id"))
                        # 2026-08-14：altloc 按 occupancy 选（引擎原生解析同款），
                        # 否则取"第一个"会选中低 occupancy 坏构象 → 平衡爆炸（1BNI 实测）
                        if key not in atoms or occ > atoms[key][3]:
                            atoms[key] = (
                                g("type_symbol"), g("label_comp_id"),
                                np.array([float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z"))], np.float32),
                                occ,
                            )
                continue
            if s.startswith("_"):
                in_loop = False
    groups = {}
    for (sid, aname), (elem, rname, xyz, _occ) in atoms.items():
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


def run_engine(eng, steps=STEPS):
    """跑 steps 步，返回 (margin, u_mean, m5_mean, crashed, n)。margin 读 metrics()。"""
    u_sum = m5_sum = 0.0
    n = 0
    crashed = False
    for _ in range(steps):
        out = eng.step(None)
        u_sum += float(out.get("u_t_kcal") or 0.0)
        m5_sum += float(out.get("m5") or 0.0)
        n += 1
        if out.get("crashed"):
            crashed = True
            break
    margin = None
    try:
        m = eng.metrics()
        if isinstance(m, dict) and "stability_margin" in m:
            margin = float(m["stability_margin"])
    except Exception:  # noqa: BLE001
        pass
    return margin, u_sum / max(1, n), m5_sum / max(1, n), crashed, n


def _rankdata(x):
    """平均秩（1-based，并列取均值），纯 numpy。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _betacf(a, b, x, itmax=200, eps=3e-12):
    """不完全 beta 的连分式。"""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """正则化不完全 beta I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_twosided_p(t, df):
    """Student t 双尾 p 值（df 自由度），替代 scipy.stats。"""
    t = float(t)
    if not math.isfinite(t) or df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return float(_betai(0.5 * df, 0.5, x))


def _pearson(x, y):
    """Pearson r + 双尾 p，纯 numpy。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(r):
        return r, float("nan")
    df = n - 2
    t = r * math.sqrt(df / max(1e-12, 1.0 - r * r))
    return r, _t_twosided_p(abs(t), df)


def _spearman(x, y):
    """Spearman（秩 Pearson）+ 双尾 p，纯 numpy。"""
    return _pearson(_rankdata(x), _rankdata(y))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/stability_benchmark/fpdb_pilot.csv")
    ap.add_argument("--pdb", action="append", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ph", type=float, default=6.5)
    ap.add_argument("--temp", type=float, default=298.0)
    ap.add_argument("--no-reuse", action="store_true",
                    help="跳过 solvent-reuse（当前会炸），直接用完整 build")
    ap.add_argument("--cif-dir", default="/tmp/fpdb_pdbs",
                    help="CIF 缓存目录（登录节点无外网时先下载到持久目录，再传此参数）")
    ap.add_argument("--out", default="/tmp/fpdb_corr.csv")
    args = ap.parse_args()

    import csv as _csv
    rows = list(_csv.DictReader(open(args.csv)))
    print(f"pilot 行: {len(rows)} | 引擎新API(solvent_reuse)={hasattr(se.Engine, 'mutate_with_solvent_reuse')}",
          flush=True)

    out_rows = []
    for pdb in args.pdb:
        rows_pdb = [r for r in rows if r["wwpdb"] == pdb]
        rows_pdb.sort(key=lambda r: int(r["struct_pos"]))
        if len(rows_pdb) > args.n:
            step = len(rows_pdb) / args.n
            rows_pdb = [rows_pdb[int(i * step)] for i in range(args.n)]
        print(f"\n===== {pdb}: {len(rows_pdb)} 突变 =====", flush=True)

        ba = cif_to_base_atoms(download_cif(pdb, args.cif_dir))
        wt_seq = ""
        # PDB 残基编号 → 序列索引 映射。struct_pos 是 PDB 残基编号（可能不从 1 起，
        # 如 1BNI 从 3 起），必须按编号查，不能当 1-based 序列索引用。
        prev = None
        resnum_to_idx = {}
        for sid, rname in zip(ba["res_seq"], ba["res_names"]):
            if sid != prev:
                resnum_to_idx[sid] = len(wt_seq)
                wt_seq += AA3.get(rname, "?")
                prev = sid
        wt_struct = structure_from_atoms(
            ba["atom_names"], ba["elements"], ba["res_seq"], ba["res_names"], ba["coords"],
        )
        print(f"  WT {len(wt_seq)} aa, {len(ba['atom_names'])} atoms", flush=True)

        t0 = time.time()
        try:
            wt_eng = se.Engine.build(wt_struct, args.ph, args.temp, PRESSURE, IONIC, RELAX, TOL)
            wt_eng.equilibrate()
            margin_wt, u_wt, m5_wt, cr_wt, _ = run_engine(wt_eng)
            print(f"  WT: margin={margin_wt} U={u_wt:.1f} m5={m5_wt:.4f} crashed={cr_wt} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {pdb}: WT build/equilibrate 失败，跳过该蛋白: {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            continue

        for r in rows_pdb:
            sp = int(r["struct_pos"])
            src, tgt = r["source_aa"], r["target_aa"]
            # 按 PDB 残基编号映射到序列位置（不是 sp-1 索引）
            idx = resnum_to_idx.get(sp)
            if idx is None or wt_seq[idx] != src:
                got = wt_seq[idx] if idx is not None else "?"
                print(f"  [skip] sp={sp} 残基={got} != {src}", flush=True)
                continue
            mut_seq = wt_seq[:idx] + tgt + wt_seq[idx + 1:]
            try:
                m_struct = build_mutant_structure_from_ca(ba, mut_seq)
            except Exception as e:  # noqa: BLE001
                print(f"  [build失败] {src}{sp}{tgt}: {type(e).__name__}: {e}", flush=True)
                continue
            t1 = time.time()
            m_eng = None
            if hasattr(wt_eng, "mutate_with_solvent_reuse") and not args.no_reuse:
                try:
                    m_eng = wt_eng.mutate_with_solvent_reuse(
                        m_struct, args.ph, args.temp, PRESSURE, IONIC, RELAX, TOL
                    )
                    m_eng.equilibrate()
                except Exception as e:  # noqa: BLE001
                    # 2026-08-14：溶剂复用对带电突变会炸（复用离子数未按新净电荷重中和）
                    print(f"  [solvent_reuse失败→完整build] {src}{sp}{tgt}: "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                    m_eng = None
            if m_eng is None:
                m_eng = se.Engine.build(m_struct, args.ph, args.temp, PRESSURE, IONIC, RELAX, TOL)
                m_eng.equilibrate()
            margin_m, u_m, m5_m, cr_m, _ = run_engine(m_eng)
            out_rows.append({
                "pdb": pdb, "mut": f"{src}{sp}{tgt}", "ddg_exp": float(r["ddg"]),
                "margin_mut": margin_m, "margin_wt": margin_wt,
                "d_margin": (margin_m - margin_wt) if (margin_m is not None and margin_wt is not None) else None,
                "dU": (u_m - u_wt), "m5_mut": m5_m, "crashed": cr_m,
                "ph": args.ph, "temp": args.temp,
            })
            print(f"  {src}{sp}{tgt}: ddg={r['ddg']} margin={margin_m} ΔU={u_m-u_wt:.0f} "
                  f"m5={m5_m:.4f} cr={cr_m} ({time.time()-t1:.0f}s)", flush=True)

    # 相关性（纯 numpy，避免 HPC spice 环境缺 scipy）
    # 注：margin/Δmargin 在良性条件(6.5/298K)全存活=粗信号，实测 1EM7 零相关；
    #     m5（电荷平衡）与 ΔU 是 1EM7 上方向正确的两个信号，一并报告
    for sig, name in [("margin_mut", "margin"), ("d_margin", "Δmargin"),
                      ("m5_mut", "m5"), ("dU", "ΔU")]:
        valid = [r for r in out_rows if r[sig] is not None]
        if len(valid) >= 5:
            ddg = np.array([r["ddg_exp"] for r in valid])
            s = np.array([r[sig] for r in valid])
            sp_r, sp_p = _spearman(ddg, s)
            pe_r, pe_p = _pearson(ddg, s)
            print(f"{name}: n={len(valid)} Spearman={sp_r:.3f}(p={sp_p:.4f}) Pearson={pe_r:.3f}(p={pe_p:.4f})",
                  flush=True)

    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["pdb"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n输出: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
