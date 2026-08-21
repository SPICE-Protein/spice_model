from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import tensorflow as tf

CONTACT_CUTOFF = 8.0   


def _parse_example(proto: tf.Tensor):
    feats = {
        "tokens": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "coords": tf.io.FixedLenFeature([], tf.string),
        "env": tf.io.FixedLenFeature([3], tf.float32),
    }
    ex = tf.io.parse_single_example(proto, feats)
    tokens = tf.io.decode_raw(ex["tokens"], tf.int32)
    mask = tf.io.decode_raw(ex["mask"], tf.float32)
    coords = tf.reshape(tf.io.decode_raw(ex["coords"], tf.float32), (-1, 3))
    return tokens, mask, coords, ex["env"]


def sample_stats(tokens, coords):
    L = len(tokens)
    if L < 2:
        return None
    c = coords.astype(np.float64)
    center = c - c.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum(center ** 2, axis=-1))))

    d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(L, k=1)
    dists = np.sqrt(d2[iu])
    seq_gap = np.abs(iu[0] - iu[1])

    contact = dists < CONTACT_CUTOFF
    n_pairs = len(dists)
    contact_density = float(contact.mean()) if n_pairs else 0.0

    local = contact & (seq_gap <= 4)
    longr = contact & (seq_gap >= 12)
    local_frac = float(local.sum() / max(contact.sum(), 1))
    longr_frac = float(longr.sum() / max(contact.sum(), 1))

    if L >= 2:
        adj = np.sqrt(np.sum((c[1:] - c[:-1]) ** 2, axis=-1))
        adj_frac = float(np.mean((adj >= 3.3) & (adj <= 4.3)))
        adj_med = float(np.median(adj))
    else:
        adj_frac, adj_med = 0.0, 0.0

    unk_frac = float((tokens == 21).mean())  
    return {
        "L": L,
        "rg": rg,
        "contact_density": contact_density,
        "local_frac": local_frac,
        "longr_frac": longr_frac,
        "adj_frac": adj_frac,
        "adj_med": adj_med,
        "unk_frac": unk_frac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord_dir", default="data/tfrecords")
    ap.add_argument("--max", type=int, default=0, help="只看前 N 条（0=全部）")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tfrecord_dir, "shard_*.tfrecord")))
    if not files:
        print(f"没有找到 TFRecord: {args.tfrecord_dir}/shard_*.tfrecord")
        return 1
    print(f"TFRecord 文件: {len(files)} 个 | {files}")

    stats = []
    for fp in files:
        ds = tf.data.TFRecordDataset(fp)
        for proto in ds:
            tokens, mask, coords, env = _parse_example(proto)
            tokens = tokens.numpy()
            coords = coords.numpy()
            s = sample_stats(tokens, coords)
            if s is not None:
                stats.append(s)
            if args.max and len(stats) >= args.max:
                break
        if args.max and len(stats) >= args.max:
            break

    if not stats:
        print("没有有效样本")
        return 1

    n = len(stats)
    Ls = np.array([s["L"] for s in stats])
    rgs = np.array([s["rg"] for s in stats])
    cds = np.array([s["contact_density"] for s in stats])
    lfs = np.array([s["local_frac"] for s in stats])
    rfs = np.array([s["longr_frac"] for s in stats])
    uks = np.array([s["unk_frac"] for s in stats])
    afs = np.array([s["adj_frac"] for s in stats])
    ams = np.array([s["adj_med"] for s in stats])

    print(f"\n===== 数据诊断: {n} 个样本 =====")
    print(f"{'指标':<18}{'均值':>10}{'中位':>10}{'p10':>10}{'p90':>10}")
    def row(name, arr):
        print(f"{name:<18}{arr.mean():>10.2f}{np.median(arr):>10.2f}"
              f"{np.percentile(arr,10):>10.2f}{np.percentile(arr,90):>10.2f}")
    row("长度 L", Ls)
    row("Rg (Å)", rgs)
    row("接触密度 (<8Å)", cds)
    row("局部接触占比", lfs)
    row("长程接触占比", rfs)
    row("相邻Cα 正常占比", afs)
    row("相邻Cα 中位间距", ams)
    row("UNK 占比", uks)

    print("\n===== 判读 =====")
    print(f"UNK 中位数 {np.median(uks)*100:.1f}%  "
          f"({'⚠️ 序列重建有损' if np.median(uks) > 0.05 else '✅ 序列干净'})")
    print(f"接触密度中位数 {np.median(cds)*100:.1f}%  "
          f"({'✅ 有真实接触(可学)' if np.median(cds) > 0.08 else '⚠️ 接触极少(可能是无规链)'})")
    print(f"长程接触占比中位数 {np.median(rfs)*100:.1f}%  "
          f"({'✅ 存在长程接触=有折叠拓扑' if np.median(rfs) > 0.2 else '⚠️ 几乎只有局部接触'})")
    adj_n = np.median(afs)
    if adj_n > 0.8:
        print(f"相邻Cα 正常占比 {adj_n*100:.1f}%  ✅ 骨架顺序正确（i,i+1≈3.8Å），拓扑可学")
    elif adj_n > 0.5:
        print(f"相邻Cα 正常占比 {adj_n*100:.1f}%  ⚠️ 部分顺序可能有问题")
    else:
        print(f"相邻Cα 正常占比 {adj_n*100:.1f}%  🚨 顺序被打乱！i,i+1 间距非 3.8Å → 序列↔坐标不对齐 → 模型学不出拓扑")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
