#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _casp(config, weights):
    r = subprocess.run(
        [sys.executable, "-m", "spice_pre.eval_casp",
         "--config", config, "--weights", weights],
        cwd=ROOT, capture_output=True, text=True)
    return r.stdout + r.stderr


def _parse(t):
    auc = re.search(r"AUC\s+([\d.]+)", t)
    p5 = re.search(r"P@L/5\s+([\d.]+)%", t)
    gdt = re.search(r"GDT-TS\s+([\d.]+)", t)
    return (
        float(auc.group(1)) if auc else float("nan"),
        float(p5.group(1)) if p5 else float("nan"),
        float(gdt.group(1)) if gdt else float("nan"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--before", default="checkpoints/pretrain/best_weights.weights.h5")
    ap.add_argument("--after", default="checkpoints/pretrain/finetuned.weights.h5")
    ap.add_argument("--out", default="runs/ablation/reflow.csv")
    args = ap.parse_args()

    rows = []
    for label, w in (("before(pretrain-only)", args.before),
                     ("after(physics-reflow)", args.after)):
        if not os.path.exists(w):
            print(f"[skip] 权重不存在: {w}")
            continue
        auc, p5, gdt = _parse(_casp(args.config, w))
        rows.append((label, auc, p5, gdt))
        print(f"[{label}] CASP AUC={auc:.3f} | P@L/5={p5:.1f}% | GDT-TS={gdt:.3f}")

    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            csv.writer(f).writerows([("model", "casp_auc", "casp_p5", "gdt_ts")] + rows)
        print("->", args.out)
    print("判定：after 的 held-out AUC 不显著掉 + 坐标/GDT 升 => 物理回流教会折叠（seedable prior）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
