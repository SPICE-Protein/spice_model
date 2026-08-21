#!/usr/bin/env python
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTRIES = "data/parquet/entries_all.parquet"
COV = "runs/posttrain/coverage.csv"
PHASE = "runs/posttrain/phase_maps"


def main() -> int:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    if os.path.exists(ENTRIES):
        en = pl.read_parquet(ENTRIES, columns=["ph", "temperature"]).drop_nulls()
        pdb = en.to_numpy()
        pdb = pdb[(pdb[:, 0] >= 0) & (pdb[:, 0] <= 14) &
                  (pdb[:, 1] > 50) & (pdb[:, 1] < 400)]
        print(f"PDB 元数据点: {len(pdb)}")
        ax.hexbin(pdb[:, 0], pdb[:, 1], gridsize=(30, 24), cmap="Greys",
                  mincnt=1, alpha=0.7, zorder=1)
    else:
        print(f"[warn] 缺 {ENTRIES}（先跑 data 准备）")

    colors = {"anchor": "#2c7bb6", "pathA_plus": "#1a9850",
              "pathA_minus": "#66bd63", "env_fail": "#d73027"}
    if os.path.exists(COV):
        cov = pl.read_csv(COV)
        for kind, grp in cov.group_by("kind"):
            k = kind[0]
            ax.scatter(grp["ph"], grp["temp"], label=k, s=46,
                       color=colors.get(k, "#fdae61"), edgecolors="black",
                       linewidths=0.5, zorder=3)
        print(f"SPICE 探索点: {cov.height}")
    else:
        print(f"[warn] 缺 {COV}（先跑 train_post）")

    phase_files = sorted(glob.glob(os.path.join(PHASE, "*.npz")))
    for f in phase_files[-3:]:
        d = np.load(f, allow_pickle=True)
        ax.scatter(np.asarray(d["ph"]), np.asarray(d["temp"]), s=8,
                   alpha=0.25, color="#fdae61", zorder=2)
    if phase_files:
        print(f"相图文件: {len(phase_files)}")

    ax.set_xlabel("pH")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Coverage: PDB structures vs SPICE-explored conditions")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 14)
    plt.tight_layout()
    os.makedirs("runs/posttrain", exist_ok=True)
    out = "runs/posttrain/coverage_map.png"
    plt.savefig(out, dpi=150)
    print(f"coverage map -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
