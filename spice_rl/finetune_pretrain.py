"""环节四：伪标签回流 → Pre-train 微调。

把路径 B 存活的突变体伪标签（.npz）转成 TFRecord，与 Pre-train 原 TFRecord
合并，用 Kabsch RMSD 微调 Head A（路径 A 坐标头），产出微调权重
`finetuned.weights.h5`（作为下一轮 RL 的起点，形成闭环）。

用法：
    python -m spice_rl.finetune_pretrain --config configs/posttrain.yaml [--epochs N]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import tensorflow as tf

from spice_rl.config import Config, load_config
from spice_rl.pseudo_labels import make_finetune_dataset, write_pseudo_tfrecord


def finetune(cfg: Config) -> None:
    from spice_pre.config import load_config as pre_load
    from spice_pre.models import SPICEPretrainModel
    from spice_pre.train_pretrain import _WarmupCosineSchedule, train_step

    pre_cfg = pre_load(cfg.post.pretrain_config)
    model = SPICEPretrainModel(pre_cfg.model, heads=("A",))
    # Keras 3 惰性构建
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
        print(f"加载 Pre-train 权重: {cfg.post.pretrain_ckpt}")

    # 伪标签 → TFRecord（置信度权重重复）
    pseudo_tf = write_pseudo_tfrecord(
        cfg.post.pseudo_label_dir,
        cfg.post.pseudo_tfrecord_path,
        pre_cfg.data.max_seq_len,
        weight_repeat=cfg.post.pseudo_weight_repeat,
        survive_steps=cfg.es.fitness_survive_steps,
    )
    if pseudo_tf == 0:
        print("没有伪标签，跳过微调（先跑 train_post 让路径 B 产生伪标签）")
        return

    # 合并数据集（原 + 伪）
    ds = make_finetune_dataset(cfg, cfg.post.pseudo_tfrecord_path, pre_cfg.data.max_seq_len)
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

    steps = 0
    for epoch in range(cfg.post.finetune_epochs):
        for x, y in ds:
            loss = train_step(model, opt, x, y, 1.0)
            steps += 1
            if steps % 50 == 0:
                print(f"[finetune ep {epoch}] step {steps} loss {float(loss):.4f} Å", flush=True)

    os.makedirs(os.path.dirname(cfg.post.finetune_out), exist_ok=True)
    model.save_weights(cfg.post.finetune_out)
    print(f"微调完成，保存: {cfg.post.finetune_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/posttrain.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.post.finetune_epochs = args.epochs
    finetune(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
