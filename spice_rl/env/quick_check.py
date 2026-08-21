from __future__ import annotations

from math import isfinite
from typing import Optional


def quick_check(
    structure,
    ph: float = 7.0,
    temp: float = 298.0,
    pressure: float = 0.0,  
    ionic: float = 0.0,
    relax_iters: int = 200,
    tolerance: float = 2.0,
    n_steps: int = 20,
    strict_incomplete: bool = True,
    equilibrate: bool = True,
) -> dict:
    import spice_engine as se

    try:
        eng = se.Engine.build(
            structure, float(ph), float(temp), float(pressure),
            float(ionic), int(relax_iters), float(tolerance),
            strict_incomplete=strict_incomplete,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"build_failed: {e}", "u": None, "margin": None, "survived": 0}

    if equilibrate:
        try:
            eng.equilibrate()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": f"equilibrate_failed: {e}", "u": None, "margin": None, "survived": 0}

    u = 0.0
    m1_peak = m2_peak = m3_peak = m4_peak = m5_peak = 0.0
    m1_sum = m2_sum = m3_sum = m4_sum = m5_sum = 0.0
    n = max(1, int(n_steps))
    survived = 0
    crashed = False
    n_clamped_total = 0
    max_accel = 0.0
    t_kin_last = 0.0
    for i in range(n):
        out = eng.step(None)
        u = float(out["u_t_kcal"])
        m1 = float(out.get("m1") or 0.0)
        m2 = float(out.get("m2") or 0.0)
        m3 = float(out.get("m3") or 0.0)
        m4 = float(out.get("m4") or 0.0)
        m5 = float(out.get("m5") or 0.0)
        if not (isfinite(m1) and isfinite(m2)):
            m1 = m2 = 0.0
        m1_peak = max(m1_peak, m1)
        m2_peak = max(m2_peak, m2)
        m3_peak = max(m3_peak, m3)
        m4_peak = max(m4_peak, m4)
        m5_peak = max(m5_peak, m5)
        m1_sum += m1
        m2_sum += m2
        m3_sum += m3
        m4_sum += m4
        m5_sum += m5
        n_clamped_total += int(out.get("n_clamped") or 0)
        max_accel = max(max_accel, float(out.get("max_accel_clamped") or 0.0))
        t_kin_last = float(out.get("t_kin") or 0.0)
        survived += 1
        if out["crashed"]:
            crashed = True
            break
    margin = survived / n  
    return {"ok": not crashed, "reason": "crashed" if crashed else "ok",
            "u": u, "margin": margin, "survived": survived,
            "m1_peak": m1_peak, "m2_peak": m2_peak,
            "m3_peak": m3_peak, "m4_peak": m4_peak, "m5_peak": m5_peak,
            "m1_mean": m1_sum / max(1, survived), "m2_mean": m2_sum / max(1, survived),
            "m3_mean": m3_sum / max(1, survived), "m4_mean": m4_sum / max(1, survived),
            "m5_mean": m5_sum / max(1, survived),
            "n_clamped": n_clamped_total, "max_accel_clamped": max_accel,
            "t_kin": t_kin_last}


def quick_check_env(structure, cfg, ph: Optional[float] = None, temp: Optional[float] = None,
                    n_steps: Optional[int] = None, equilibrate: bool = True) -> dict:
    return quick_check(
        structure,
        ph=ph if ph is not None else cfg.ph_default if hasattr(cfg, "ph_default") else 7.0,
        temp=temp if temp is not None else 298.0,
        pressure=cfg.pressure,
        ionic=cfg.ionic_default,
        relax_iters=cfg.relax_iters,
        tolerance=cfg.tolerance,
        n_steps=n_steps if n_steps is not None else getattr(cfg, "quick_check_steps", 20),
        strict_incomplete=getattr(cfg, "strict_incomplete", True),
        equilibrate=equilibrate,
    )
