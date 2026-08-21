#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
from functools import partial

import numpy as np

from spice_rl.env import scan_phase_map
from spice_rl.env.structure import structure_from_mmcif

PUBLISHED = [
    {
        "pdb": "2LYZ", "name": "hen egg-white lysozyme", "n_res": 129,
        "tm_c": 75.0, "tm_ph": 5.0, "ph_window": (2.0, 11.0),
        "note": "Tm ≈ 74–77 °C at pH 4–7; broad pH stability (optimum ~pH 5). [TODO @author: ref]",
    },
    {
        "pdb": "7RSA", "name": "bovine pancreatic ribonuclease A", "n_res": 124,
        "tm_c": 63.0, "tm_ph": 7.0, "ph_window": (2.0, 9.0),
        "note": "Tm ≈ 62–64 °C at pH 7; strongly pH-dependent (Tm drops to ~35–40 °C at pH 2–3). [TODO @author: ref]",
    },
    {
        "pdb": "1UBQ", "name": "human ubiquitin", "n_res": 76,
        "tm_c": 95.0, "tm_ph": 3.0, "ph_window": (2.0, 10.0),
        "note": "Exceptionally thermostable (Tm ≈ 90–100 °C); unusually stable at low pH. [TODO @author: ref]",
    },
    {
        "pdb": "2CI2", "name": "chymotrypsin inhibitor 2", "n_res": 65,
        "tm_c": 76.0, "tm_ph": 7.0, "ph_window": (3.0, 9.0),
        "note": "Classic two-state folder; Tm ≈ 74–77 °C near neutral pH. [TODO @author: ref]",
    },
]
_PUB = {p["pdb"].upper(): p for p in PUBLISHED}


def _structure(pdb_id: str, cache_dir: str, local_cif_dir: str = ""):
    pdb_id = pdb_id.upper()
    if local_cif_dir:
        p = os.path.join(local_cif_dir, f"{pdb_id}.cif")
        if os.path.exists(p):
            return structure_from_mmcif(p)
    path = os.path.join(cache_dir, f"{pdb_id}.cif")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(cache_dir, exist_ok=True)
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        tmp = f"{path}.tmp-{os.getpid()}"
        print(f"  download {pdb_id} <- {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return structure_from_mmcif(path)


def _stable_grid(points, ph, temp):
    if not points:
        return False
    best = min(points, key=lambda p: (p["ph"] - ph) ** 2 + (p["temp"] - temp) ** 2)
    return bool(best["stable"])


def _signals(points):
    stable = [p for p in points if p["stable"]]
    native = _stable_grid(points, 7.0, 298.0)
    max_t = max((p["temp"] for p in stable), default=float("nan"))  
    near = min(set(p["temp"] for p in points), key=lambda t: abs(t - 298.0))
    phs_298 = [p["ph"] for p in stable if abs(p["temp"] - near) < 1e-6]
    ph_lo = min(phs_298) if phs_298 else float("nan")
    ph_hi = max(phs_298) if phs_298 else float("nan")
    low_pH = _stable_grid(points, 3.0, 298.0)   
    return {"native": native, "max_t": max_t, "ph_lo": ph_lo, "ph_hi": ph_hi, "low_pH": low_pH}


def _init_worker(rayon_threads: int) -> None:
    if rayon_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)


def _process_one(pdb: str, args) -> dict:
    pub = _PUB.get(pdb, {})
    print(f"\n===== {pdb} ({pub.get('name', '?')}) =====", flush=True)
    s = _structure(pdb, args.cache_dir, getattr(args, "local_cif_dir", ""))
    print(f"  residues: {s.residue_count()} (published n_res≈{pub.get('n_res')})", flush=True)
    pts = scan_phase_map(
        s, tuple(args.temp_range), tuple(args.ph_range),
        n_steps=args.n_steps, repeats=args.repeats, relax_iters=int(args.relax_iters),
    )
    sig = _signals(pts)
    return {"pdb": pdb, **sig, **{k: pub.get(k) for k in ("tm_c", "tm_ph", "ph_window")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-list", default="", help="comma-separated PDB ids; empty = all in table")
    ap.add_argument("--temp-range", nargs=3, type=float, default=(280.0, 350.0, 15.0))
    ap.add_argument("--ph-range", nargs=3, type=float, default=(3.0, 9.0, 2.0))
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--relax-iters", type=int, default=50)
    ap.add_argument("--cache-dir", default="data/validation")
    ap.add_argument("--local-cif-dir", default="",
                    help="optional dir of local mmCIF files (e.g. md_cal/data/test); else RCSB")
    ap.add_argument("--workers", type=int, default=1, help="number of proteins to scan in parallel")
    ap.add_argument("--threads", type=int, default=0,
                    help="MD threads per worker (0 = auto: cpus // workers)")
    ap.add_argument("--out", default="", help="write comparison table to CSV")
    args = ap.parse_args()

    ids = [p["pdb"] for p in PUBLISHED]
    if args.pdb_list:
        ids = [x.strip().upper() for x in args.pdb_list.split(",") if x.strip()]
    for i in ids:
        if i not in _PUB:
            print(f"  warning: {i} not in published table; using empty reference")

    print("Validation of predicted stability vs published experimental data")
    print(f"  grid: T {args.temp_range} K, pH {args.ph_range}, n_steps={args.n_steps}, "
          f"repeats={args.repeats}, relax_iters={args.relax_iters}, workers={args.workers}")
    if args.workers > 1:
        threads = args.threads or max(1, (os.cpu_count() or 4) // args.workers)
        print(f"  parallel: {args.workers} workers, RAYON_NUM_THREADS={threads} each")
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(threads,)) as pool:
            rows = pool.map(partial(_process_one, args=args), ids)
    else:
        rows = [_process_one(p, args) for p in ids]

    pred_rank = sorted(rows, key=lambda r: -(r["max_t"] if not np.isnan(r["max_t"]) else 1e9))
    print("\n===== COMPARISON =====")
    print(f"  {'PDB':<6}{'native(7,298)':<14}{'pred maxT':<12}{'pub Tm':<8}{'pH window@298':<14}{'stable@pH3':<12}")
    for r in rows:
        pw = "–" if np.isnan(r["ph_lo"]) else f"{r['ph_lo']:.0f}-{r['ph_hi']:.0f}"
        print(f"  {r['pdb']:<6}{str(r['native']):<14}{r['max_t'] if not np.isnan(r['max_t']) else '–':<12}"
              f"{r['tm_c']:<8.0f}{pw:<14}{str(r['low_pH']):<12}")
    print("\n  Predicted max-stable-T ranking (desc):",
          [f"{r['pdb']}({r['max_t']:.0f})" for r in pred_rank])

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (v if not isinstance(v, tuple) else ";".join(map(str, v)))
                            for k, v in r.items()})
        print(f"\n  saved: {args.out}")

    print("\n  Note: absolute predicted Tm is conservative (short screening window); "
          "check native stability, relative Tm ranking, and pH-sensitivity direction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
