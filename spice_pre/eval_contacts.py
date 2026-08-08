"""评估 Pre-train 是否学到拓扑：接触预测精度（contact AUC / top-L precision）。

比 MDS 更直接的判据：
  - MDS 用「期望距离」重建，24 箱粗量化会把长程接触抹平，低估模型真实水平；
  - contact AUC / precision 直接问：模型预测的接触对（Cα<8Å）是不是真接触。

指标（每样本 + 平均）：
  - contact AUC：随机=0.5，>0.7 说明真在学接触，>0.85 单序列很强
  - precision@L/5、@L/2、@L：top k 个预测接触对的精度（随机≈接触密度~4%）

用法（Colab，需已训练出 best_weights.weights.h5）：
    python -m spice_pre.eval_contacts --config configs/pretrain.yaml --samples 32
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from spice_pre.config import load_config
from spice_pre.data.dataset import load_tfrecord_dataset
from spice_pre.keras_utils import setup_gpu
from spice_pre.models import SPICEPretrainModel

CONTACT = 8.0  # Å


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _bin_edges(cfg) -> np.ndarray:
    """bin 下边界（Å）：linspace(min,max,N-1)，与 distogram_ce_loss 一致。"""
    return np.linspace(cfg.model.dist_min, cfg.model.dist_max,
                       cfg.model.dist_bins - 1).astype(np.float64)


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U 统计实现的 AUC（仅 numpy）。

    rank 1 = 最低分（升序），与公式 sum(正例rank) 的约定一致；
    之前用 argsort(-score) 把 rank 1 给了最高分 → AUC 变成 1-真实值（反向）。
    """
    score = np.asarray(score)
    label = np.asarray(label)
    order = np.argsort(score, kind="mergesort")   # 升序：rank 1 = 最低分
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((rank[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sample_metrics(p_contact: np.ndarray, true_contact: np.ndarray,
                   mask: np.ndarray):
    """masked 上三角有效对 -> (auc, p@L/5, p@L/2, p@L)。"""
    m = mask > 0.5
    iu = np.triu_indices(len(m), k=1)
    keep = m[iu[0]] & m[iu[1]]
    if not keep.any():
        return None
    s = p_contact[iu][keep]
    t = true_contact[iu][keep]
    auc = roc_auc(s, t)
    order = np.argsort(-s)
    L = int(m.sum())
    out = {"auc": auc}
    for k in (5, 2, 1):
        topk = max(L // k, 1)
        out[f"p@L/{k}"] = float(t[order[:topk]].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--weights", default="checkpoints/pretrain/best_weights.weights.h5")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_gpu(cfg.train.use_gpu, cfg.train.gpu_mem_growth, cfg.train.gpu_devices)

    model = SPICEPretrainModel(cfg.model)
    model({"tokens": tf.zeros([1, 8], tf.int32), "env": tf.zeros([1, 3]),
           "mask": tf.ones([1, 8])}, training=False)
    model.load_weights(args.weights)

    edges = _bin_edges(cfg)
    # contact bins = 上边界 <= 8Å 的 bin（bin k 上边界 = edges[k+1] 或 +inf）
    upper = np.concatenate([edges[1:], [np.inf]])
    contact_bins = np.where(upper <= CONTACT)[0]

    agg = []
    ds = load_tfrecord_dataset(cfg, "val").take(args.samples)
    print(f"[eval] 接触预测精度（接触= Cα 距离 <{CONTACT}Å，前 {args.samples} 个 val 样本）:")
    for i, (x, y) in enumerate(ds):
        n = int(tf.reduce_sum(x["mask"]).numpy())
        inputs = {"tokens": x["tokens"][None], "env": x["env"][None],
                  "mask": x["mask"][None]}
        out = model(inputs, training=False)
        logits = out["dist_logits"][0, :n, :n].numpy()
        probs = _softmax(logits)  # [L,L,N]
        p_contact = probs[:, :, contact_bins].sum(axis=-1)  # [L,L]
        p_contact = (p_contact + p_contact.T) / 2.0
        true = y[:n].numpy()
        d2 = np.sum((true[:, None, :] - true[None, :, :]) ** 2, axis=-1)
        true_contact = d2 < CONTACT ** 2
        m = x["mask"][:n].numpy()
        res = sample_metrics(p_contact, true_contact, m)
        if res is None:
            continue
        agg.append(res)
        print(f"  s{i}: L={n:>4} | AUC {res['auc']:.3f} | "
              f"P@L/5 {res['p@L/5']*100:5.1f}% P@L/2 {res['p@L/2']*100:5.1f}% "
              f"P@L/1 {res['p@L/1']*100:5.1f}%")

    if not agg:
        print("没有有效样本")
        return 1
    print("\n===== 平均 =====")
    for k in ("auc", "p@L/5", "p@L/2", "p@L/1"):
        v = np.mean([a[k] for a in agg])
        print(f"  {k:<6} {v:.3f}" + ("（随机≈0.5）" if k == "auc" else "（随机≈接触密度~4%）"))
    auc = np.mean([a["auc"] for a in agg])
    print(f"\n判读: AUC={auc:.3f}  "
          f"({'✅ 真在学接触(拓扑可学)' if auc >= 0.7 else '⚠️ 接近随机，接触没学到'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
