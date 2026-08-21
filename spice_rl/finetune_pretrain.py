from __future__ import annotations

import argparse
import os
import logging

import numpy as np
import tensorflow as tf

from spice_rl.config import Config, load_config, setup_logging
from spice_rl.pseudo_labels import make_finetune_dataset, write_pseudo_tfrecord

logger = logging.getLogger("spice")


def finetune(cfg: Config) -> None:
    from spice_pre.config import load_config as pre_load
    from spice_pre.models import SPICEPretrainModel
    from spice_pre.train_pretrain import _WarmupCosineSchedule, train_step

    pre_cfg = pre_load(cfg.post.pretrain_config)
    model = SPICEPretrainModel(pre_cfg.model, heads=("A",))
    model(
        {
            "tokens": tf.zeros([1, 8], tf.int32),
            "env": tf.zeros([1, 3]),
            "mask": tf.ones([1, 8]),
        },
        training=False,
    )
    if os.path.exists(cfg.post.pretrain_ckpt):
        model.load_weights(cfg.post.pretrain_ckpt)
        logger.info(f"加载 Pre-train 权重: {cfg.post.pretrain_ckpt}")

    pseudo_tf = write_pseudo_tfrecord(
        cfg.post.pseudo_label_dir,
        cfg.post.pseudo_tfrecord_path,
        pre_cfg.data.max_seq_len,
        weight_repeat=cfg.post.pseudo_weight_repeat,
        survive_steps=cfg.es.fitness_survive_steps,
    )
    if pseudo_tf == 0:
        logger.info("没有伪标签，跳过微调（先跑 train_post 让路径 B 产生伪标签）")
        return

    ds = make_finetune_dataset(cfg, cfg.post.pseudo_tfrecord_path, pre_cfg.data.max_seq_len)
    ds = ds.filter(lambda x, y: (
        tf.reduce_all(tf.math.is_finite(x["coords"])) &
        tf.reduce_all(tf.math.is_finite(x["env"]))
    ))
    padded_shapes = (
        {"tokens": [None], "mask": [None], "env": [3], "coords": [None, 3]},
        [None, 3],
    )
    pad_values = ({"tokens": 0, "mask": 0.0, "env": 0.0, "coords": 0.0}, 0.0)
    ds = ds.padded_batch(
        cfg.post.finetune_batch_size, padded_shapes, pad_values, drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)

    lr = _WarmupCosineSchedule(cfg.post.finetune_lr, 100, max(1000, cfg.post.finetune_epochs * 1000))
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1.0e-4)

    gs = tf.Variable(0, dtype=tf.int64)   # train_step 内部用 step 张量算暖身权重
    steps = 0
    for epoch in range(cfg.post.finetune_epochs):
        for x, y in ds:
            gs.assign_add(1)
            # ⚠️ 2026-08-14: 旧调用漏传 pair_warmup_steps/global_step/chirality/clash →
            #   TypeError 跑不起来 + clash 权重(现 3.0)没生效。回流是微调 → 短暖身 50 步，
            #   让坐标 Kabsch + clash 监督立刻生效（否则 <3000 步时 pair_weight≈0 白训）。
            loss = train_step(
                model, opt, x, y,
                pre_cfg.train.grad_clip,
                False,
                pre_cfg.train.dist_weight,
                pre_cfg.model.dist_bins,
                pre_cfg.model.dist_min,
                pre_cfg.model.dist_max,
                pre_cfg.train.pair_weight,
                50,
                gs,
                pre_cfg.train.frame_chirality_weight,
                pre_cfg.train.frame_clash_weight,
                pre_cfg.train.coord_max_len,
                pre_cfg.train.frame_consistency_weight,
            )
            steps += 1
            if steps % 50 == 0:
                logger.info(f"[finetune ep {epoch}] step {steps} loss {float(loss[0]):.4f} Å")

    os.makedirs(os.path.dirname(cfg.post.finetune_out), exist_ok=True)
    model.save_weights(cfg.post.finetune_out)
    logger.info(f"微调完成，保存: {cfg.post.finetune_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/posttrain.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.post.log_dir, "finetune.log")
    if args.epochs is not None:
        cfg.post.finetune_epochs = args.epochs
    finetune(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
