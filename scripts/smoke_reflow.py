#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from spice_rl.config import load_config
from spice_rl.pseudo_labels import (
    load_pseudo_labels,
    make_finetune_dataset,
    write_pseudo_tfrecord,
)

cfg = load_config("configs/posttrain.yaml")
AA = "ACDEFGHIKLMNPQRSTVWY"
rng = np.random.default_rng(0)

pseudo_dir = cfg.post.pseudo_label_dir
os.makedirs(pseudo_dir, exist_ok=True)
for i, steps in enumerate([200, 150, 80]):
    L = int(rng.integers(30, 60))
    seq = "".join(rng.choice(list(AA), L))
    coords = rng.normal(size=(L, 3)).astype(np.float32)
    env = np.array([7.0, 300.0, 0.15], np.float32)
    np.savez(os.path.join(pseudo_dir, f"pseudo_{i}_{steps}.npz"),
             seq=seq, env=env, coords=coords)

recs = load_pseudo_labels(pseudo_dir)
print("load_pseudo_labels:", len(recs), "entries (weight =", [r["weight"] for r in recs], ")")

n = write_pseudo_tfrecord(pseudo_dir, cfg.post.pseudo_tfrecord_path, 512,
                          weight_repeat=8, survive_steps=200)
print("write_pseudo_tfrecord ->", n, "entries (weighted repeat)")

ds = make_finetune_dataset(cfg, cfg.post.pseudo_tfrecord_path, 512)
padded_shapes = ({"tokens": [None], "mask": [None], "env": [3], "coords": [None, 3]}, [None, 3])
pad_values = ({"tokens": 0, "mask": 0.0, "env": 0.0, "coords": 0.0}, 0.0)
ds = ds.padded_batch(8, padded_shapes, pad_values)
for x, y in ds.take(1):
    print("batch: tokens", x["tokens"].shape, "| env", x["env"].shape,
          "| mask", x["mask"].shape, "| coords(label)", y.shape)
    print("valid mask sum:", float(tf.reduce_sum(x["mask"][0])), "coords finite:",
          bool(np.all(np.isfinite(y[0].numpy()))))
print("REFLOW SMOKE PASSED")
