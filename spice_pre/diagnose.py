"""SPICE Pre-train 训练诊断。

在 Colab 里跑（需先同步 spice_pre/ 与 configs/）：
    python -m spice_pre.diagnose --config configs/pretrain.yaml

逐个验证：
  [1] lr 调度是否真在给数
  [2] 数据：真实坐标尺度 / Rg / mask 与坐标长度是否对齐、env 是否正常
  [3-4] loss：TF 的 kabsch_rmsd vs 独立 numpy 实现（手动对拍）
  [4] 预测坐标是否塌缩（pred_Rg << true_Rg 即塌缩 → RMSD 梯度死锁，loss 不动的元凶）
  [5] 梯度是否流通到各模块（encoder / AdaLN / head_a / dist_head / embedding）
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from spice_pre.config import load_config
from spice_pre.data.dataset import load_tfrecord_dataset
from spice_pre.keras_utils import setup_gpu
from spice_pre.losses.kabsch_rmsd import kabsch_rmsd, pairwise_coord_loss
from spice_pre.models import SPICEPretrainModel


def manual_kabsch_rmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """独立 numpy 版 Kabsch RMSD（对拍 TF 实现）。输入已按 mask 过滤。"""
    pc = pred - pred.mean(0)
    tc = true - true.mean(0)
    h = pc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))   # det(V U^T) 修正
    uf = u.copy()
    uf[:, -1] *= d
    r = vt.T @ uf.T                          # R = V U_fix^T
    rp = pc @ r.T
    return float(np.sqrt(np.mean(np.sum((rp - tc) ** 2, axis=1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(args.config)

    setup_gpu(cfg.train.use_gpu, cfg.train.gpu_mem_growth, cfg.train.gpu_devices)
    if cfg.train.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[diag] mixed_float16 ON")

    # [1] lr 调度
    from spice_pre.train_pretrain import build_lr_schedule

    sched = build_lr_schedule(cfg)
    print(f"\n[1] lr 调度（warmup={cfg.train.warmup_steps}, base={cfg.train.lr:.1e}）:")
    for s in (0, cfg.train.warmup_steps // 2, cfg.train.warmup_steps,
              cfg.train.warmup_steps * 2, 2000):
        print(f"    step {s:>5d}: lr = {float(sched(s)):.3e}")

    # [2-4] 数据 + 模型前向 + loss 对拍 + 塌缩检查
    model = SPICEPretrainModel(cfg.model)
    model(
        {"tokens": tf.zeros([1, 8], tf.int32), "env": tf.zeros([1, 3]),
         "mask": tf.ones([1, 8])},
        training=False,
    )
    print(f"\n[2][3][4] 前 {args.samples} 个 val 样本（TF_RMSD vs numpy_RMSD 应一致）:")
    ds = load_tfrecord_dataset(cfg, "val").take(args.samples)
    for i, (x, y) in enumerate(ds):
        n = int(tf.reduce_sum(x["mask"]).numpy())
        inputs = {"tokens": x["tokens"][None], "env": x["env"][None],
                  "mask": x["mask"][None]}
        out = model(inputs, training=False)
        pred = out["coords"][0, :n].numpy()
        true = y[:n].numpy()
        m = x["mask"][:n].numpy()

        pred_rg = float(np.sqrt(np.mean(np.sum((pred - pred.mean(0)) ** 2, axis=1))))
        true_rg = float(np.sqrt(np.mean(np.sum((true - true.mean(0)) ** 2, axis=1))))
        tf_rmsd = float(kabsch_rmsd(out["coords"][0:1], y[None], x["mask"][None])[0])
        np_rmsd = manual_kabsch_rmsd(pred[m > 0.5], true[m > 0.5])
        pair_loss = float(pairwise_coord_loss(out["coords"][0:1], y[None], x["mask"][None]))
        print(
            f"    s{i}: L={n:>4d} | pred_Rg={pred_rg:7.3f} | true_Rg={true_rg:7.3f} "
            f"| TF_RMSD={tf_rmsd:7.4f} | numpy_RMSD={np_rmsd:7.4f} | coord_pairwise={pair_loss:.1f}"
        )
        print(
            f"        coords pred[min={pred.min():.3f} max={pred.max():.3f}] "
            f"true[min={true.min():.3f} max={true.max():.3f}] | env={x['env'].numpy()}"
        )

    # [5] 梯度流通检查（eager 一次）
    print("\n[5] 梯度是否流通到各模块（ZERO/None 即该模块没收到梯度）:")
    opt = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=cfg.train.weight_decay)
    if cfg.train.use_mixed_precision:
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    opt.build(model.trainable_variables)

    x, y = next(iter(load_tfrecord_dataset(cfg, "val").take(1)))
    inputs = {"tokens": x["tokens"][None], "env": x["env"][None], "mask": x["mask"][None]}
    yb, mb = y[None], x["mask"][None]
    with tf.GradientTape() as tape:
        out = model(inputs, training=True)
        loss = tf.reduce_mean(kabsch_rmsd(out["coords"], yb, mb))
        loss = loss + cfg.train.dist_weight * pairwise_coord_loss(out["coords"], yb, mb)
    grads = tape.gradient(loss, model.trainable_variables)
    for v, g in zip(model.trainable_variables, grads):
        if g is None:
            print(f"    [ZERO/None!!] {v.name}")
        else:
            gn = float(tf.norm(g))
            tag = "OK " if gn > 0 else "ZERO!!"
            print(f"    [{tag}] {v.name} grad_norm={gn:.4e}")

    print("\n[diag] 判读：")
    print("  - TF_RMSD 与 numpy_RMSD 不一致 → loss 实现有 bug（假设 4）")
    print("  - pred_Rg << true_Rg（如 0.1 vs 20）→ 预测塌缩 → RMSD 梯度死锁 → 需要坐标两两距离 aux loss")
    print("  - 某模块 grad_norm=0/None → 梯度没到那里（假设 5 / 网络断链）")
    print("  - 一切正常但 loss 仍平 → 查 lr / 模型容量（假设 1 / 3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
