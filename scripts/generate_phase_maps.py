#!/usr/bin/env python3
"""生成 pH-T 稳定相图（§3.3 Path A，Figure 3a）。

对每个蛋白：加载 WT 结构（parquet 优先，回退 cif）→ scan_stability_ranges 扫 pH-T 网格
→ save_phase_map npz（stable/crashed/build_failed）+ 打印摘要 + 渲染 PNG（--plot）。

用法（HPC/本地，spice 环境）：
  PYTHONPATH=. python scripts/generate_phase_maps.py \
      --parquet-dir ~/spice/data/parquet_hpc --proteins 7QF3 6QQE 8D8F 1JVT \
      --ph 2 12 1 --temp 260 360 20 --repeats 3 \
      --out ~/spice/model/runs/posttrain/phase_maps --plot

  成本：~11×6=66 点 × repeats，~11-33 min/蛋白（单线程），4 蛋白 ~4-8 核时。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spice_rl.env.structure import load_structure_with_atoms, structure_from_atoms  # noqa: E402
from spice_rl.env.phase_map import scan_phase_map, summarize_phase_map, save_phase_map  # noqa: E402

PRESSURE, IONIC = 1.0, 0.0


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


def _load_structure(parquet_dir, pdb, wt_cif):
    if parquet_dir:
        struct, _ba = load_structure_with_atoms(parquet_dir, pdb)
        return struct
    ba = _cif_to_base_atoms(wt_cif)
    return structure_from_atoms(ba["atom_names"], ba["elements"], ba["res_seq"],
                                ba["res_names"], ba["coords"])


def _run_protein(task):
    """worker（fork）：扫一个蛋白的相图。task=(pdb, parquet_dir, wt_cif, ph_rng, t_rng, cfg, out_dir, plot)。"""
    pdb, parquet_dir, wt_cif, (ph0, ph1, dph), (t0, t1, dt), cfg, out_dir, plot = task
    t_start = time.time()
    try:
        struct = _load_structure(parquet_dir, pdb, wt_cif)
    except Exception as e:  # noqa: BLE001
        print(f"[{pdb}] 结构加载失败: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return {"ok": False, "pdb": pdb}
    print(f"[{pdb}] 扫 pH {ph0}-{ph1}/Δ{dph} × T {t0}-{t1}/Δ{dt}K（{len(struct.sequence())} aa）...", flush=True)
    pts = scan_phase_map(
        struct,
        temp_range=(t0, t1, dt), ph_range=(ph0, ph1, dph),
        pressure=PRESSURE, ionic=IONIC,
        n_steps=cfg["n_steps"], equil_steps=cfg["equil_steps"], repeats=cfg["repeats"],
        relax_iters=cfg["relax_iters"], tolerance=cfg["tolerance"],
    )
    s = summarize_phase_map(pts)
    out_npz = os.path.join(out_dir, f"{pdb}_phase_map.npz")
    save_phase_map(pts, out_npz)
    print(f"[{pdb}] {len(pts)} 点 | stable={s['n_stable']} boundary={s['n_boundary']} "
          f"crashed={s['n_crashed']} build_failed={s['n_build_failed']} | {time.time()-t_start:.0f}s "
          f"-> {out_npz}", flush=True)
    if plot:
        try:
            _render_png(pdb, out_dir, t0, t1, dt, ph0, ph1, dph)
        except Exception as e:  # noqa: BLE001
            print(f"[{pdb}] 渲染 PNG 失败（不影响 npz）: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return {"ok": True, "pdb": pdb, "summary": s}


def _render_png(pdb, out_dir, t0, t1, dt, ph0, ph1, dph):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(os.path.join(out_dir, f"{pdb}_phase_map.npz"))
    stable = np.asarray(d["stable"]).astype(bool)
    crashed = np.asarray(d.get("crashed", np.zeros_like(stable))).astype(bool)
    build_failed = np.asarray(d.get("build_failed", np.zeros_like(stable))).astype(bool)
    phs = np.arange(ph0, ph1 + 1e-9, dph)
    temps = np.arange(t0, t1 + 1e-9, dt)
    cls = np.zeros((len(temps), len(phs)), int)
    for p in range(len(d["temp"])):
        ti = int(round((d["temp"][p] - t0) / dt))
        pi = int(round((d["ph"][p] - ph0) / dph))
        if 0 <= ti < len(temps) and 0 <= pi < len(phs):
            if stable[p]:
                cls[ti, pi] = 1
            elif crashed[p]:
                cls[ti, pi] = 2
            elif build_failed[p]:
                cls[ti, pi] = 3
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(phs, temps, cls, cmap="RdYlGn", shading="auto", vmin=0, vmax=3)
    ax.set_xlabel("pH"); ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Stability phase map {pdb}  (stable/boundary/crashed/build_failed)")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="gray", label="boundary"),
               mpatches.Patch(color="green", label="stable"),
               mpatches.Patch(color="red", label="crashed"),
               mpatches.Patch(color="purple", label="build_failed")]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{pdb}_phase_map.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[{pdb}] 图 -> {out_png}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", default=None, help="parquet atoms 目录（HPC: ~/spice/data/parquet_hpc）")
    ap.add_argument("--wt-cif", default=None, help="回退：单个蛋白 mmCIF（--proteins 单值）")
    ap.add_argument("--proteins", nargs="+", default=["7QF3", "6QQE", "8D8F", "1JVT"])
    ap.add_argument("--ph", nargs=3, type=float, default=[2.0, 12.0, 1.0], help="ph0 ph1 dph")
    ap.add_argument("--temp", nargs=3, type=float, default=[260.0, 360.0, 20.0], help="t0 t1 dt")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--equil-steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--relax-iters", type=int, default=200)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="默认 1：蛋白顺序跑。⚠️ 勿用 fork 池并行——scan_stability_ranges 引擎内部用 rayon 并列表格，fork+rayon 会死锁（CPU→0）。引擎内部自带列并行+重优化，给足 RAYON 线程即可")
    ap.add_argument("--plot", action="store_true", help="渲染 PNG（本地建议；HPC 容器 matplotlib 易卡）")
    ap.add_argument("--out", default="runs/posttrain/phase_maps")
    args = ap.parse_args()

    if args.parquet_dir is None and args.wt_cif is None:
        print("必须给 --parquet-dir 或 --wt-cif")
        return 1
    os.makedirs(args.out, exist_ok=True)
    cfg = {"n_steps": args.n_steps, "equil_steps": args.equil_steps, "repeats": args.repeats,
           "relax_iters": args.relax_iters, "tolerance": args.tolerance}
    tasks = [(p, args.parquet_dir, args.wt_cif, tuple(args.ph), tuple(args.temp),
              cfg, args.out, args.plot) for p in args.proteins]

    import multiprocessing as _mp
    ctx = _mp.get_context("fork")
    if args.workers and args.workers > 1:
        print("[warn] --workers>1 会 fork+rayon 死锁，强制改回顺序")
        args.workers = 1
    got = [_run_protein(t) for t in tasks]   # 顺序：引擎内部 rayon 并列表格
    n_ok = sum(1 for g in got if g.get("ok"))
    print(f"\n完成 {n_ok}/{len(tasks)} 蛋白 -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
