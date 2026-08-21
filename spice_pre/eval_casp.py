from __future__ import annotations

import argparse
import glob
import os
import urllib.request
from collections import Counter

import numpy as np
import tensorflow as tf

from spice_pre.config import load_config
from spice_pre.data.preprocessing import normalize_env, res_names_to_seq, seq_to_tokens
from spice_pre.eval_contacts import CONTACT, _bin_edges, _softmax, roc_auc, sample_metrics
from spice_pre.eval_distogram import mds_reconstruct
from spice_pre.keras_utils import setup_gpu
from spice_pre.losses.kabsch_rmsd import expected_dists_from_distogram
from spice_pre.models import SPICEPretrainModel

CASP_TARGETS = [
    ("T1024", "6t1z"), ("T1025", "6uv6"), ("T1029", "6uf2"), ("T1030", "6poo"),
    ("T1032", "6n64"), ("T1038", "6ya2"), ("T1049", "6y4f"), ("T1056", "6yj1"),
    ("T1064", "7jtl"), ("T1082", "6x6o"), ("T1099", "6ygh"),
]
CASP_TARGETS_CLEAN = [t for t in CASP_TARGETS if t[1].lower() not in ("6y4f", "6yj1")]


def ensure_pdb(code: str, casp_dir: str) -> str:
    path = os.path.join(casp_dir, f"{code}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(casp_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{code}.pdb"
    print(f"  download {code} <- {url}")
    urllib.request.urlretrieve(url, path)
    return path


def parse_pdb_ca(path: str, max_len: int = 512, chain: str | None = None):
    records = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                rn = line[17:20].strip()
                ch = line[21]
                rs = line[22:26].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            records.append((ch, rs, rn, x, y, z))
    if not records:
        return None
    if chain is None:
        chain = Counter(r[0] for r in records).most_common(1)[0][0]
    recs = [r for r in records if r[0] == chain]

    def _num(r):
        try:
            return int(float(r[1]))
        except (TypeError, ValueError):
            return 0

    recs.sort(key=lambda r: (_num(r), r[1]))
    seen, out = set(), []
    for r in recs:
        if r[1] in seen:
            continue
        seen.add(r[1]); out.append(r)

    res_names = [r[2] for r in out]
    seq = res_names_to_seq(res_names)
    coords = np.array([[r[3], r[4], r[5]] for r in out], dtype=np.float32)
    if not (40 <= len(seq) <= max_len):
        return None
    return seq, coords


def kabsch_align(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pc = pred - pred.mean(0)
    tc = true - true.mean(0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    uf = u.copy()
    uf[:, -1] *= d
    r = vt.T @ uf.T
    return pc @ r.T + true.mean(0)


def gdt_ts(pred: np.ndarray, true: np.ndarray) -> float:
    a = kabsch_align(pred, true)
    d = np.sqrt(np.sum((a - true) ** 2, axis=1))
    return float(np.mean([np.mean(d <= t) for t in (1.0, 2.0, 4.0, 8.0)]))


def tm_score(pred: np.ndarray, true: np.ndarray) -> float:
    a = kabsch_align(pred, true)
    L = pred.shape[0]
    d = np.sqrt(np.sum((a - true) ** 2, axis=1))
    d0 = max(1.24 * (L - 15) ** (1 / 3) - 1.8, 0.5)
    return float(np.mean(1.0 / (1 + (d / d0) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--casp_dir", default="data/casp")
    ap.add_argument("--weights", default="checkpoints/pretrain/best_weights.weights.h5")
    ap.add_argument("--pdb-list", default="", help="comma-separated PDB codes; empty = all")
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
    default_env = normalize_env(None, None, None, cfg.data.default_env)

    targets = CASP_TARGETS_CLEAN
    if args.pdb_list:
        wanted = {c.strip().lower() for c in args.pdb_list.split(",") if c.strip()}
        targets = [t for t in CASP_TARGETS if t[1].lower() in wanted]

    ready = []
    for target, code in targets:
        try:
            p = ensure_pdb(code, args.casp_dir)
            ready.append((target, code, p))
        except Exception as e:
            print(f"  {target}/{code}: download failed {type(e).__name__}: {str(e)[:60]}")

    agg = []
    print(f"[CASP14] {len(ready)} targets | contact < {CONTACT} A | random AUC ~ 0.5")
    for target, code, fp in ready:
        parsed = parse_pdb_ca(fp)
        if parsed is None:
            print(f"  {target} ({code}): skipped (invalid length / no CA)")
            continue
        seq, coords = parsed
        L = len(seq)
        tokens = seq_to_tokens(seq)[None]
        env = default_env[None]
        mask = np.ones((1, L), np.float32)
        out = model({"tokens": tokens, "env": env, "mask": mask}, training=False)
        logits = out["dist_logits"][0].numpy()
        probs = _softmax(logits)
        p_contact = (probs[:, :, contact_bins].sum(axis=-1) +
                     probs[:, :, contact_bins].sum(axis=-1).T) / 2.0
        d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
        true_contact = d2 < CONTACT ** 2
        d_true = np.sqrt(d2)
        d_pred = expected_dists_from_distogram(
            out["dist_logits"], cfg.model.dist_bins,
            cfg.model.dist_min, cfg.model.dist_max,
        )[0].numpy()
        ii, jj = np.triu_indices(L, 1)
        dist_mae = float(np.mean(np.abs(d_pred[ii, jj] - d_true[ii, jj])))
        pred_c = (p_contact > 0.5)[ii, jj]
        true_c = true_contact[ii, jj]
        cprec = float((pred_c & true_c).sum() / pred_c.sum()) if pred_c.sum() else float("nan")
        mds = mds_reconstruct(d_pred)
        gdt = gdt_ts(mds, coords)
        tms = tm_score(mds, coords)
        res = sample_metrics(p_contact, true_contact, mask[0])
        if res is None:
            continue
        res["mae"] = dist_mae
        res["cprec"] = cprec
        res["gdt"] = gdt
        res["tm"] = tms
        agg.append(res)
        print(f"  {target} ({code}): L={L:>4} | AUC {res['auc']:.3f} | "
              f"P@L/5 {res['p@L/5']*100:5.1f}% P@L/1 {res['p@L/1']*100:5.1f}% | "
              f"dMAE {dist_mae:5.2f}Å | Prec {cprec*100:5.1f}% | "
              f"GDT-TS {gdt:.3f} TM {tms:.3f}")

    if not agg:
        print("No evaluable targets")
        return 1
    auc = np.mean([a["auc"] for a in agg])
    p5 = np.mean([a["p@L/5"] for a in agg])
    mae = np.nanmean([a["mae"] for a in agg])
    cprec = np.nanmean([a["cprec"] for a in agg])
    gdt = np.mean([a["gdt"] for a in agg])
    tms = np.mean([a["tm"] for a in agg])
    print("\n===== CASP14 AGGREGATE =====")
    print(f"  AUC       {auc:.3f}    (in-domain ~0.90)")
    print(f"  P@L/5     {p5*100:.1f}%   (random ~ contact density ~4%)")
    print(f"  Dist MAE  {mae:5.2f} A    (predicted Cα dist vs true; random ~25 A, ~3.8 A = adjacent)")
    print(f"  Prec      {cprec*100:.1f}%  (fraction of predicted contacts that are real; random ~4%)")
    print(f"  GDT-TS    {gdt:.3f}    (MDS-rebuilt vs native; <0.3 ~ blob, 0.5 ~ rough topology, >0.8 ~ high)")
    print(f"  TM-score  {tms:.3f}    (same interpretation)")
    _g = ("small gap -> genuine generalization, no significant leakage" if auc >= 0.75
          else "large gap -> reliance on training-set memory / possible leakage")
    print(f"\nVerdict: CASP AUC={auc:.3f} vs in-domain ~0.90 -> {_g}")
    _t = ("topology learned; RL only needs local refinement" if gdt >= 0.35
          else ("blob-like" if gdt < 0.25 else "topology hints, but still rough"))
    print(f"         GDT-TS={gdt:.3f} -> {_t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

