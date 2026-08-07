"""评估 Pre-train 是否学到拓扑：从预测 distogram 用经典 MDS 重建坐标，Kabsch 对齐后算 RMSD。

「学到拓扑」判据：MDS-RMSD 显著低于回旋半径 Rg（~20Å）。若 ~10Å 以下说明抓到二级结构/粗略拓扑，
RL 只需局部微调；若 ~Rg 则是"紧凑 blob"（没学到折叠）。

用法（Colab，需已训练出 best_weights.weights.h5）：
    python -m spice_pre.eval_distogram --config configs/pretrain.yaml --samples 16

输出：每个样本 len / true_Rg / mds_rmsd（从预测距离重建）/ 以及全样本平均。
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from spice_pre.config import load_config
from spice_pre.data.dataset import load_tfrecord_dataset
from spice_pre.keras_utils import setup_gpu
from spice_pre.losses.kabsch_rmsd import expected_dists_from_distogram
from spice_pre.models import SPICEPretrainModel


def mds_reconstruct(d: np.ndarray) -> np.ndarray:
    """经典 MDS：距离矩阵 [L,L] -> 坐标 [L,3]（numpy）。"""
    n = d.shape[0]
    d2 = d.astype(np.float64) ** 2
    row = d2.mean(1, keepdims=True)
    col = d2.mean(0, keepdims=True)
    grand = d2.mean()
    b = -0.5 * (d2 - row - col + grand)
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1][:3]
    lam = np.sqrt(np.maximum(eigvals[idx], 0.0))
    return (eigvecs[:, idx] * lam).astype(np.float32)


def kabsch_rmsd_np(pred: np.ndarray, true: np.ndarray) -> float:
    pc = pred - pred.mean(0)
    tc = true - true.mean(0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    uf = u.copy()
    uf[:, -1] *= d
    r = vt.T @ uf.T
    rp = pc @ r.T
    return float(np.sqrt(np.mean(np.sum((rp - tc) ** 2, axis=1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--weights", default="checkpoints/pretrain/best_weights.weights.h5")
    args = ap.parse_args()
    cfg = load_config(args.config)

    setup_gpu(cfg.train.use_gpu, cfg.train.gpu_mem_growth, cfg.train.gpu_devices)
    model = SPICEPretrainModel(cfg.model)
    model(
        {"tokens": tf.zeros([1, 8], tf.int32), "env": tf.zeros([1, 3]),
         "mask": tf.ones([1, 8])},
        training=False,
    )
    model.load_weights(args.weights)

    mds_rmsds = []
    print(f"[eval] 预测距离 -> MDS 重建 -> Kabsch RMSD（前 {args.samples} 个 val 样本）:")
    ds = load_tfrecord_dataset(cfg, "val").take(args.samples)
    for i, (x, y) in enumerate(ds):
        n = int(tf.reduce_sum(x["mask"]).numpy())
        inputs = {"tokens": x["tokens"][None], "env": x["env"][None],
                  "mask": x["mask"][None]}
        out = model(inputs, training=False)
        d_pred = expected_dists_from_distogram(
            out["dist_logits"], cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max
        )[0, :n, :n].numpy()
        true = y[:n].numpy()
        m = x["mask"][:n].numpy()
        valid = np.where(m > 0.5)[0]

        true_rg = float(np.sqrt(np.mean(np.sum(
            (true[valid] - true[valid].mean(0)) ** 2, axis=1))))
        pred_ca = mds_reconstruct(d_pred[np.ix_(valid, valid)])
        true_ca = true[valid]
        r = kabsch_rmsd_np(pred_ca, true_ca)
        mds_rmsds.append(r)
        print(f"  s{i}: L={n:>4d} | true_Rg={true_rg:7.3f} | MDS-RMSD={r:7.3f} Å")

    if mds_rmsds:
        print(f"\n[eval] 平均 MDS-RMSD = {np.mean(mds_rmsds):.3f} Å")
        print("  ~10Å 以下 → 学到拓扑（二级结构/粗略折叠）✅ |  ~Rg(~20Å) → 仍是紧凑 blob ❌")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
