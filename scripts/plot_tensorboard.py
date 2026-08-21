#!/usr/bin/env python
"""解析 TensorBoard tfevents -> 论文级图表 + 源数据 CSV。

用法：
  python scripts/plot_tensorboard.py \
      --logdir runs/ablation/data_efficiency/n10/runs \
      --tags val/rmsd,val/dist \
      --out runs/figures/val_curves.png \
      --csv  runs/figures/val_curves.csv \
      --title "n10 ablation: val metrics" --ylabel "RMSD (A) / CE"

  # 多条 run 对比（数据效率消融 / 不同 seed），自动图例=目录名：
  python scripts/plot_tensorboard.py \
      --logdir runs/ablation/data_efficiency/n10/runs \
      --logdir runs/ablation/data_efficiency/n100/runs \
      --tags val/rmsd \
      --out runs/figures/ablation_val_rmsd.png

特性：
  - 同时支持 scalar 与 tensor 两种事件存储（auto 检测）
  - 可选 EMA 平滑（--smooth，论文常用）
  - 300dpi 出版级样式 + 源数据 CSV（Nature 等要求 source data）
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _decode_tensor(tp) -> float:
    if tp.float_val:
        return float(tp.float_val[0])
    if tp.tensor_content:
        return float(np.frombuffer(tp.tensor_content, np.float32)[0])
    # 多元素 tensor：取均值（罕见，兜底）
    import tensorflow as tf
    t = tf.make_ndarray(tp)
    return float(np.asarray(t).reshape(-1).mean())


def _load_series(event_dir: str, tag: str):
    """返回 (steps, values)；tag 不存在返回 None。自动处理 scalar / tensor 事件。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(event_dir, "*.tfevents*")))
    if not files:
        return None
    ea = EventAccumulator(event_dir, size_guidance={
        "scalars": 0, "tensors": 0, "histograms": 0,
    })
    ea.Reload()
    tags = ea.Tags()
    steps, vals = [], []
    if tag in tags.get("scalars", []):
        for e in ea.Scalars(tag):
            steps.append(e.step)
            vals.append(e.value)
    elif tag in tags.get("tensors", []):
        for e in ea.Tensors(tag):
            steps.append(e.step)
            vals.append(_decode_tensor(e.tensor_proto))
    else:
        return None
    return np.array(steps, np.int64), np.array(vals, np.float64)


def _ema(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= window:
        return values
    k = np.ones(window) / window
    return np.convolve(values, k, mode="same")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", action="append", required=True,
                    help="tfevents 目录（可重复，多条曲线）")
    ap.add_argument("--tags", required=True, help="逗号分隔的 tag 列表")
    ap.add_argument("--out", default="runs/figures/tb.png")
    ap.add_argument("--csv", default="", help="导出源数据 CSV（可选）")
    ap.add_argument("--title", default="")
    ap.add_argument("--ylabel", default="value")
    ap.add_argument("--xlabel", default="step")
    ap.add_argument("--smooth", type=int, default=0, help="EMA 平滑窗口（0=不平滑）")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figsize", default="6,4")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    fig, axes = plt.subplots(
        len(tags), 1, figsize=tuple(float(x) for x in args.figsize.split(",")),
        sharex=True, squeeze=False,
    )
    rows = []  # (run, tag, step, value)

    # 一个 tag 一个子图
    for i, tag in enumerate(tags):
        ax = axes[i][0]
        for ld in args.logdir:
            name = os.path.basename(os.path.normpath(ld))
            res = _load_series(ld, tag)
            if res is None:
                print(f"[skip] {ld}: tag '{tag}' 不存在")
                continue
            steps, vals = res
            y = _ema(vals, args.smooth) if args.smooth else vals
            ax.plot(steps, y, label=name, lw=1.4)
            for s, v in zip(steps, vals):
                rows.append((name, tag, int(s), float(v)))
        ax.set_ylabel(f"{tag}\n{args.ylabel}" if len(tags) == 1 else tag, fontsize=9)
        ax.grid(True, alpha=0.3)
        if len(args.logdir) > 1:
            ax.legend(fontsize=8)
    if args.title:
        fig.suptitle(args.title, fontsize=11)
    axes[-1][0].set_xlabel(args.xlabel, fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[fig] -> {args.out}")

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "tag", "step", "value"])
            w.writerows(rows)
        print(f"[csv] -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
