#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "runs/posttrain/pathb_candidates.csv"
AA = "ACDEFGHIKLMNPQRSTVWY"


def parse_muts(s: str):
    out = []
    for tok in str(s).split(";"):
        tok = tok.strip()
        if not tok or ">" not in tok or ":" not in tok:
            continue
        pos, rest = tok.split(":", 1)
        w, m = rest.split(">", 1)
        out.append((int(pos), w, m))
    return out


def plot_group(df, tag: str, ep: int, outdir: str = "runs/posttrain"):
    rows = []
    for r in df.iter_rows(named=True):
        for pos, w, m in parse_muts(r["mutations"]):
            rows.append({"pos": pos, "aa": m, "fitness": r["fitness"], "q": r["q"],
                         "survived": r["survived"], "n_mut": len(parse_muts(r["mutations"]))})
    if not rows:
        return
    d = pl.DataFrame(rows)
    agg = d.group_by(["pos", "aa"]).agg(
        pl.col("fitness").max().alias("best_fitness"),
        pl.col("survived").max().alias("any_survived"),
        pl.len().alias("n_cand"),
    ).sort("pos")

    fig, ax = plt.subplots(figsize=(max(8, len(AA) * 0.5), 6))
    sc = ax.scatter(
        agg["pos"], [AA.index(a) for a in agg["aa"]],
        c=agg["best_fitness"], cmap="YlOrRd", s=90, edgecolors="black",
        linewidths=0.5, zorder=3, vmin=0,
    )
    surv = agg.filter(pl.col("any_survived") == 1)
    if surv.height:
        ax.scatter(surv["pos"], [AA.index(a) for a in surv["aa"]],
                   marker="*", s=220, facecolors="none", edgecolors="#1a9850",
                   linewidths=1.5, zorder=4, label="survivor")
    ax.set_yticks(range(20), list(AA))
    ax.set_xlabel("Residue position")
    ax.set_ylabel("Target amino acid")
    ax.set_title(f"Path-B mutation landscape {tag} ep{ep} (fitness=steps×Q; * = survived)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("best fitness (steps×Q)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"pathb_landscape_{tag}_ep{ep}.png")
    plt.savefig(out, dpi=150)
    print(f"landscape -> {out}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ep", type=int, default=None)
    a = ap.parse_args()
    if not os.path.exists(CSV):
        print(f"[warn] 缺 {CSV}（先跑 train_post，path_b 触发后才有）")
        return 0
    df = pl.read_csv(CSV)
    if a.tag is not None:
        df = df.filter(pl.col("tag") == a.tag)
    if a.ep is not None:
        df = df.filter(pl.col("ep") == a.ep)
    print(f"候选记录: {df.height} 行（含存活 {df['survived'].sum()}）")
    for (tag, ep), g in df.group_by(["tag", "ep"]):
        plot_group(g, str(tag), int(ep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
