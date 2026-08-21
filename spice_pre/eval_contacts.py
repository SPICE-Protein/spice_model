from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from spice_pre.config import load_config
from spice_pre.data.dataset import load_tfrecord_dataset
from spice_pre.keras_utils import setup_gpu
from spice_pre.models import SPICEPretrainModel

CONTACT = 8.0  


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _bin_edges(cfg) -> np.ndarray:
    return np.linspace(cfg.model.dist_min, cfg.model.dist_max,
                       cfg.model.dist_bins - 1).astype(np.float64)


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score)
    label = np.asarray(label)
    order = np.argsort(score, kind="mergesort")   
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((rank[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sample_metrics(p_contact: np.ndarray, true_contact: np.ndarray,
                   mask: np.ndarray):
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


LONG_RANGE_SEP = 24   


def long_range_metrics(p_contact: np.ndarray, true_contact: np.ndarray,
                       mask: np.ndarray, min_sep: int = LONG_RANGE_SEP):
    m = mask > 0.5
    iu = np.triu_indices(len(m), k=1)
    keep = m[iu[0]] & m[iu[1]] & ((iu[1] - iu[0]) >= min_sep)
    if not keep.any():
        return {"lr_auc": float("nan"), "lr_p@L/5": float("nan")}
    s = p_contact[iu][keep]
    t = true_contact[iu][keep]
    auc = roc_auc(s, t)
    order = np.argsort(-s)
    L = int(m.sum())
    topk = max(L // 5, 1)
    return {"lr_auc": auc, "lr_p@L/5": float(t[order[:topk]].mean())}


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
    upper = np.concatenate([edges[1:], [np.inf]])
    contact_bins = np.where(upper <= CONTACT)[0]

    agg = []
    lr_agg = []
    ds = load_tfrecord_dataset(cfg, "val").take(args.samples)
    print(f"[eval] Contact prediction (contact = Cα dist < {CONTACT} A; first {args.samples} val samples):")
    for i, (x, y) in enumerate(ds):
        n = int(tf.reduce_sum(x["mask"]).numpy())
        inputs = {"tokens": x["tokens"][None], "env": x["env"][None],
                  "mask": x["mask"][None]}
        out = model(inputs, training=False)
        logits = out["dist_logits"][0, :n, :n].numpy()
        probs = _softmax(logits)  
        p_contact = probs[:, :, contact_bins].sum(axis=-1)  
        p_contact = (p_contact + p_contact.T) / 2.0
        true = y[:n].numpy()
        d2 = np.sum((true[:, None, :] - true[None, :, :]) ** 2, axis=-1)
        true_contact = d2 < CONTACT ** 2
        m = x["mask"][:n].numpy()
        res = sample_metrics(p_contact, true_contact, m)
        if res is None:
            continue
        agg.append(res)
        lr = long_range_metrics(p_contact, true_contact, m)
        lr_agg.append(lr)
        print(f"  s{i}: L={n:>4} | AUC {res['auc']:.3f} | "
              f"P@L/5 {res['p@L/5']*100:5.1f}% P@L/2 {res['p@L/2']*100:5.1f}% "
              f"P@L/1 {res['p@L/1']*100:5.1f}% | "
              f"LR_AUC(>=24) {lr['lr_auc']:.3f}")

    if not agg:
        print("No valid samples")
        return 1
    print("\n===== AGGREGATE =====")
    for k in ("auc", "p@L/5", "p@L/2", "p@L/1"):
        v = np.mean([a[k] for a in agg])
        print(f"  {k:<6} {v:.3f}" + (" (random ~ 0.5)" if k == "auc" else " (random ~ contact density ~4%)"))
    lr_auc = np.nanmean([a["lr_auc"] for a in lr_agg])
    lr_p5 = np.nanmean([a["lr_p@L/5"] for a in lr_agg])
    print(f"  LR_AUC {lr_auc:.3f} (long-range, seq sep >= {LONG_RANGE_SEP}; random ~ 0.5)")
    print(f"  LR_P@L/5 {lr_p5*100:.1f}% (long-range, random ~ contact density ~4%)")
    auc = np.mean([a["auc"] for a in agg])
    _v = "contacts genuinely learned (topology learnable)" if auc >= 0.7 else "near random, contacts not learned"
    print(f"\nVerdict: AUC={auc:.3f} -> {_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
