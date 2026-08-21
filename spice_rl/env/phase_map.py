from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


def scan_phase_map(
    structure,
    temp_range: Tuple[float, float, float],
    ph_range: Tuple[float, float, float],
    pressure: float = 1.0,
    ionic: float = 0.0,
    n_steps: int = 20,
    equil_steps: int = 10,
    repeats: int = 3,
    relax_iters: int = 200,
    tolerance: float = 2.0,
) -> List[dict]:
    import spice_engine as se

    pts = se.scan_stability_ranges(
        structure,
        tuple(float(v) for v in temp_range),
        tuple(float(v) for v in ph_range),
        (float(pressure), float(pressure), 1.0),
        (float(ionic), float(ionic), 1.0),
        n_steps=int(n_steps),
        equil_steps=int(equil_steps),
        repeats=int(repeats),
        relax_iters=int(relax_iters),
        tolerance=float(tolerance),
    )
    return [dict(p) for p in pts]


def summarize_phase_map(points: List[dict]) -> dict:
    stable = [p for p in points if p["stable"]]
    boundary = [p for p in points if not p["stable"] and not p["build_failed"]]
    crashed = [p for p in points if p["crashed"]]
    build_failed = [p for p in points if p["build_failed"]]
    return {
        "n_points": len(points),
        "n_stable": len(stable),
        "n_boundary": len(boundary),
        "n_crashed": len(crashed),
        "n_build_failed": len(build_failed),
        "stable": stable,
        "boundary": boundary,
        "crashed": crashed,
        "build_failed": build_failed,
    }


def save_phase_map(points: List[dict], path: str) -> None:
    if not points:
        return
    keys = ["temp", "ph", "pressure", "ionic", "stable", "crashed", "build_failed"]
    arr = {
        k: np.array([p.get(k) for p in points]) for k in keys
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arr)


def load_phase_map(path: str) -> List[dict]:
    d = np.load(path, allow_pickle=True)
    n = len(d["temp"])
    return [
        {
            "temp": float(d["temp"][i]),
            "ph": float(d["ph"][i]),
            "pressure": float(d["pressure"][i]),
            "ionic": float(d["ionic"][i]),
            "stable": bool(d["stable"][i]),
            "crashed": bool(d["crashed"][i]),
            "build_failed": bool(d["build_failed"][i]),
        }
        for i in range(n)
    ]
