#!/usr/bin/env python
"""E 实验 SMACOF 口径：Head A before/after 的 held-out CASP14 坐标质量。

eval_casp 默认用 naive top-3 MDS 重建（截断局部几何 → GDT 被拉低，论文 §3.2 明示）。
本脚本改用 **bond-length 加权 metric-stress SMACOF**：相邻残基 (|i-j|=1) 加权拉回 ~3.8Å
虚拟键，其余按预测距离——复现论文 §3.2 的 SMACOF 口径（held-out GDT ~0.044-0.046）。

用法（spice 环境）：
  python scripts/experiments/eval_reflow_smacof.py \
      --before <ckpt> --after <finetuned> --out runs/ablation/reflow_smacof_n45000.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import tensorflow as tf  # noqa: E402

from spice_pre.config import load_config  # noqa: E402
from spice_pre.data.preprocessing import normalize_env, seq_to_tokens  # noqa: E402
from spice_pre.eval_casp import (  # noqa: E402
    CASP_TARGETS_CLEAN, _bin_edges, _softmax, ensure_pdb, gdt_ts, parse_pdb_ca, tm_score,
)
from spice_pre.eval_contacts import CONTACT  # noqa: E402
from spice_pre.eval_distogram import mds_reconstruct  # noqa: E402
from spice_pre.losses.kabsch_rmsd import expected_dists_from_distogram  # noqa: E402
from spice_pre.models import SPICEPretrainModel  # noqa: E402


def weighted_smacof(d, w_adj=20.0, n_iter=400):
    """bond-length 加权 metric-stress SMACOF。

    目标 σ = Σ_{i<j} w_ij (‖x_i−x_j‖ − d_ij)²；相邻残基权重 w_adj 拉回虚拟键。
    从 naive MDS 起步（保全局拓扑），按中位非相邻距离缩放后迭代。
    """
    d = np.asarray(d, np.float64)
    L = d.shape[0]
    d = np.clip(d, 3.0, None)
    np.fill_diagonal(d, 0.0)
    w = np.ones((L, L), np.float64)
    for i in range(L - 1):
        w[i, i + 1] = w_adj
        w[i + 1, i] = w_adj
    X = mds_reconstruct(d.astype(np.float32)).astype(np.float64)
    if X.shape[0] != L:
        X = np.zeros((L, 3), np.float64)
    X = X - X.mean(axis=0)
    ii, jj = np.triu_indices(L, 1)
    na = np.abs(ii - jj) > 1
    if na.any():
        md = np.median(d[ii[na], jj[na]])
        if md > 0:
            mx = np.median(np.linalg.norm(X[ii[na]] - X[jj[na]], axis=1))
            if mx > 0:
                X = X * (md / mx)
    for _ in range(n_iter):
        D = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1), 1e-12))
        ratio = np.where(D > 1e-9, d / D, 0.0)
        dirv = (X[:, None, :] - X[None, :, :]) / D[:, :, None]
        delta = d[:, :, None] * dirv
        numer = w[:, :, None] * (X[None, :, :] + delta)
        denom = np.maximum(w.sum(axis=1), 1e-12)[:, None]
        Xnew = numer.sum(axis=1) / denom
        X = Xnew - Xnew.mean(axis=0)
    return X


def eval_weights(cfg, model, weights, casp_dir, contact_bins):
    print(f"\n[权重] {os.path.basename(weights)}")
    model.load_weights(weights)
    default_env = normalize_env(None, None, None, cfg.data.default_env)
    gdts, tms, maes = [], [], []
    for target, code in CASP_TARGETS_CLEAN:
        try:
            p = ensure_pdb(code, casp_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  {code}: 下载失败 {e}")
            continue
        parsed = parse_pdb_ca(p)
        if parsed is None:
            continue
        seq, coords = parsed
        L = len(seq)
        tokens = seq_to_tokens(seq)[None]
        out = model({"tokens": tokens, "env": default_env[None],
                     "mask": np.ones((1, L), np.float32)}, training=False)
        d_pred = expected_dists_from_distogram(
            out["dist_logits"], cfg.model.dist_bins,
            cfg.model.dist_min, cfg.model.dist_max,
        )[0].numpy()
        d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        d_true = np.sqrt(d2)
        ii, jj = np.triu_indices(L, 1)
        mae = float(np.mean(np.abs(d_pred[ii, jj] - d_true[ii, jj])))
        X = weighted_smacof(d_pred)
        gdt = gdt_ts(X, coords)
        tms_ = tm_score(X, coords)
        gdts.append(gdt)
        tms.append(tms_)
        maes.append(mae)
        print(f"  {code}: L={L} | dMAE {mae:5.2f}A | GDT-TS {gdt:.3f} (SMACOF) TM {tms_:.3f}")
    if not gdts:
        return None
    print(f"  -> SMACOF 重建 GDT-TS 均值 {np.mean(gdts):.3f}  TM {np.mean(tms):.3f}")
    return {"gdt": float(np.mean(gdts)), "tm": float(np.mean(tms)),
            "mae": float(np.mean(maes))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--casp-dir", default="data/casp")
    ap.add_argument("--before", default="checkpoints/pretrain/best_weights.weights.h5")
    ap.add_argument("--after", default="checkpoints/pretrain/finetuned.weights.h5")
    ap.add_argument("--out", default="runs/ablation/reflow_smacof.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SPICEPretrainModel(cfg.model)
    model({"tokens": tf.zeros([1, 8], tf.int32), "env": tf.zeros([1, 3]),
           "mask": tf.ones([1, 8])}, training=False)
    edges = _bin_edges(cfg)
    upper = np.concatenate([edges[1:], [np.inf]])
    contact_bins = np.where(upper <= CONTACT)[0]

    results = {}
    for label, w in (("before(pretrain-only)", args.before),
                     ("after(physics-reflow)", args.after)):
        if not os.path.exists(w):
            print(f"[skip] 权重不存在: {w}")
            continue
        results[label] = eval_weights(cfg, model, w, args.casp_dir, contact_bins)

    if len(results) >= 2:
        b, a = results["before(pretrain-only)"], results["after(physics-reflow)"]
        if b and a:
            dg = a["gdt"] - b["gdt"]
            print(f"\n[Δ] GDT-TS {b['gdt']:.3f} -> {a['gdt']:.3f}  ({dg:+.3f}, "
                  f"{100*dg/max(b['gdt'],1e-9):+.0f}%)")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                f.write("model,gdt_smacof,tm_smacof,dist_mae\n")
                f.write(f"before,{b['gdt']:.4f},{b['tm']:.4f},{b['mae']:.4f}\n")
                f.write(f"after,{a['gdt']:.4f},{a['tm']:.4f},{a['mae']:.4f}\n")
            print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
