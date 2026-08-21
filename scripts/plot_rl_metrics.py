#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default="runs/posttrain/metrics.csv")
    ap.add_argument("--out", default="runs/posttrain/plots")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"没找到 {args.csv}（先跑 train_post 生成 metrics.csv）")
    df = pl.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)
    ep = df["ep"].to_list()

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ep, df["alpha"].to_list())
    ax[0, 0].set_title("SAC alpha (entropy temperature)")
    ax[0, 0].set_xlabel("episode")
    ax[0, 1].plot(ep, df["buffer"].to_list())
    ax[0, 1].set_title("replay buffer size")
    ax[0, 1].set_xlabel("episode")
    ax[1, 0].plot(ep, df["critic_loss"].to_list(), label="critic")
    ax[1, 0].plot(ep, df["actor_loss"].to_list(), label="actor")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("SAC losses (NaN=还没攒够 batch)")
    ax[1, 0].legend()
    ax[1, 0].set_xlabel("episode")
    ax[1, 1].plot(ep, df["a_survive"].to_list())
    ax[1, 1].set_title("path A survive steps (out of max)")
    ax[1, 1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rl_curves.png"), dpi=150)

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
    ax2[0].bar(ep, df["a_crashed"].to_list())
    ax2[0].set_title("path A crashed per episode")
    ax2[0].set_xlabel("episode")
    ax2[1].bar(ep, df["n_survivors"].to_list())
    ax2[1].set_title("path B survivors per episode")
    ax2[1].set_xlabel("episode")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.out, "rl_survival.png"), dpi=150)

    print(f"已保存: {args.out}/rl_curves.png, {args.out}/rl_survival.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
