"""SPICE Phase 1：Pre-train 训练入口。

只训练 Head A（坐标头）：输入 (Seq, Env)，输出 Cα 坐标，Kabsch RMSD 监督。

用法：
    # 1) 先构建 TFRecord（从 hf-mirror 下载并清洗）
    python -m spice_pre.data.dataset --config configs/pretrain.yaml build

    # 2) 训练
    python -m spice_pre.train_pretrain --config configs/pretrain.yaml

    调试（只取 1 个 shard、每 shard 200 条）：
    python -m spice_pre.data.dataset --config configs/pretrain.yaml build
    # 然后临时改 config 里 max_shards / structures_per_shard，或：
    python -m spice_pre.train_pretrain --config configs/pretrain.yaml --epochs 1
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tensorflow as tf

from spice_pre.config import Config, load_config
from spice_pre.data.dataset import load_tfrecord_dataset
from spice_pre.keras_utils import setup_gpu
from spice_pre.losses.kabsch_rmsd import (
    distogram_ce_loss,
    kabsch_rmsd,
    pairwise_coord_loss,
)
from spice_pre.models import SPICEPretrainModel

# TensorBoard 可选：未安装时自动降级为只打印日志
try:
    import tensorboard  # noqa: F401

    _HAS_TB = True
except ImportError:
    _HAS_TB = False

# tqdm 可选：有则显示进度条；没有则退回逐 50 步打印
try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# 学习率调度：线性 warmup + cosine decay
# ---------------------------------------------------------------------------
class _WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """线性 warmup 后 cosine 衰减到 0.1*lr。"""

    def __init__(self, lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)

    def __call__(self, step):
        lr = tf.constant(self.lr, tf.float32)
        step_f = tf.cast(step, tf.float32)
        if self.warmup_steps <= 0:
            return lr
        warmup = tf.constant(self.warmup_steps, tf.float32)
        total = tf.constant(max(self.total_steps, self.warmup_steps + 1), tf.float32)
        progress = tf.clip_by_value((step_f - warmup) / (total - warmup), 0.0, 1.0)
        decay_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + tf.cos(np.pi * progress)))
        warmup_lr = lr * step_f / warmup
        return tf.where(step_f < warmup, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


def build_lr_schedule(cfg: Config):
    total = cfg.train.max_steps or 1_000_000
    return _WarmupCosineSchedule(cfg.train.lr, cfg.train.warmup_steps, total)


# ---------------------------------------------------------------------------
# 数据批处理
# ---------------------------------------------------------------------------
def _bucket_length_fn(x, y=None):
    """bucket_by_sequence_length 的分桶 key：按 tokens 实际长度分桶。

    注意：TF 会解包元素为位置参数调用本函数（官方测试亦为此签名），
    y 用不到，仅适配元组元素。
    """
    return tf.shape(x["tokens"])[0]


def _make_padded_dataset(cfg: Config, split: str):
    ds = load_tfrecord_dataset(cfg, split=split)
    padded_shapes = (
        {"tokens": [None], "mask": [None], "env": [3], "coords": [None, 3]},
        [None, 3],
    )
    padding_values = (
        {"tokens": 0, "mask": 0.0, "env": 0.0, "coords": 0.0},
        0.0,
    )
    if split == "train":
        # 长度分桶 + pad 到固定桶边界：只有 ~max_len/64 个固定形状，
        # 避免变长 padding 导致 train_step 反复 retrace（每次重编译要几秒），
        # 同时保留变长的 O(L^2) 收益。
        # 顶部多加 max_seq_len+1 的桶，保证 pad_to_bucket_boundary 的
        # "元素长度必须 < max(bucket_boundaries)" 约束永远满足（长度==512 的序列也不会进最后一个桶）。
        b_step = 64
        boundaries = list(range(b_step, cfg.data.max_seq_len, b_step)) + [
            cfg.data.max_seq_len + 1
        ]
        n_buckets = len(boundaries) + 1
        batch_sizes = [cfg.train.batch_size] * n_buckets
        # 长序列桶降 batch：控制 [B,L,L,N_BINS] 距离张量 + attention 内存，避免最长桶 OOM
        if n_buckets >= 4:
            batch_sizes[-3] = max(16, cfg.train.batch_size // 2)  # 384~448 桶
            batch_sizes[-2] = max(8, cfg.train.batch_size // 4)   # 448~512 桶
        ds = ds.bucket_by_sequence_length(
            element_length_func=_bucket_length_fn,
            bucket_boundaries=boundaries,
            bucket_batch_sizes=batch_sizes,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            pad_to_bucket_boundary=True,
            drop_remainder=True,
        )
        if cfg.train.cache_train:
            # 预存整个 epoch 的 batch 到内存：首次遍历做全部解析/分桶，
            # 之后每个 epoch 直接从 RAM 读，管线不再拖累计算。
            # 注意：cache 必须在 bucket 之后，这样缓存的是分好桶的 batch。
            ds = ds.cache()
            # 缓存后再做 batch 级 shuffle（reshuffle_each_iteration 保证每 epoch 顺序不同）
            ds = ds.shuffle(2000, reshuffle_each_iteration=True)
    else:
        ds = ds.padded_batch(
            cfg.train.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=False,
        )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _count_split_records(cfg: Config, split: str) -> int:
    """统计某个 split 的记录数（用于估算每 epoch 步数 / 进度条 total）。"""
    return sum(1 for _ in load_tfrecord_dataset(cfg, split=split))


# ---------------------------------------------------------------------------
# 训练 / 验证步骤
# ---------------------------------------------------------------------------
@tf.function
def train_step(model, optimizer, x, y, grad_clip, use_loss_scale,
               dist_weight, dist_bins, dist_min, dist_max, pair_weight):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        # 主目标：binned distogram CE（学距离分布 = 接触 = 拓扑）+ 小权重坐标距离 aux。
        # Kabsch 坐标回归路径病态（塌缩梯度死锁），从训练梯度移除，仅作 val 质量指标。
        unscaled_loss = (
            dist_weight * distogram_ce_loss(
                out["dist_logits"], y, x["mask"], dist_bins, dist_min, dist_max
            )
            + pair_weight * pairwise_coord_loss(out["coords"], y, x["mask"])
        )
        rmsd = tf.reduce_mean(kabsch_rmsd(out["coords"], y, x["mask"]))
        loss = optimizer.scale_loss(unscaled_loss)
    grads = tape.gradient(loss, model.trainable_variables)
    if use_loss_scale:
        # 梯度已被 loss scale 放大：clip 阈值要按 scale 同步放大，
        # 否则 apply_gradients 内部 unscale 后有效更新会缩小 ~scale 倍 → 模型不学习。
        scale = optimizer.dynamic_scale
        grads, _ = tf.clip_by_global_norm(grads, grad_clip * scale)
    else:
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return unscaled_loss, rmsd


@tf.function(reduce_retracing=True)
def val_step(model, x, y, dist_bins, dist_min, dist_max):
    out = model(x, training=False)
    rmsd = tf.reduce_mean(kabsch_rmsd(out["coords"], y, x["mask"]))
    dist = distogram_ce_loss(
        out["dist_logits"], y, x["mask"], dist_bins, dist_min, dist_max
    )
    return rmsd, dist


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def train(cfg: Config) -> None:
    tf.random.set_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    setup_gpu(cfg.train.use_gpu, cfg.train.gpu_mem_growth, cfg.train.gpu_devices)
    # 混合精度：必须在创建任何模型/层之前设置全局 policy
    if cfg.train.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("混合精度: mixed_float16 (ON)")
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU 开关: {'ON ' + str(gpus) if cfg.train.use_gpu and gpus else 'OFF (CPU)'}")

    model = SPICEPretrainModel(cfg.model)
    # Keras 3 惰性构建：先用 dummy 输入前向一次创建变量
    model(
        {
            "tokens": tf.zeros([1, 8], tf.int32),
            "env": tf.zeros([1, 3]),
            "mask": tf.ones([1, 8]),
        },
        training=False,
    )
    lr_schedule = build_lr_schedule(cfg)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=cfg.train.weight_decay
    )
    # 混合精度需要 LossScaleOptimizer 做梯度缩放（防 fp16 下溢）
    if cfg.train.use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    lr_opt = optimizer.inner_optimizer if cfg.train.use_mixed_precision else optimizer
    # Keras 3 优化器的动量/速度槽是惰性创建的：在 tf.function 外预先 build，
    # 避免 apply_gradients 在图内创建变量触发
    # "tf.function only supports singleton tf.Variables created on the first call"
    optimizer.build(model.trainable_variables)

    # 可训练参数统计
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"SPICE Pre-train model 参数量: {n_params:,}")

    # checkpoint 恢复
    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step=tf.Variable(0, dtype=tf.int64)
    )
    manager = tf.train.CheckpointManager(
        ckpt, cfg.train.ckpt_dir, max_to_keep=3
    )
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"已恢复 checkpoint: {manager.latest_checkpoint}")
    global_step = int(ckpt.step)

    train_ds = _make_padded_dataset(cfg, "train")
    val_ds = _make_padded_dataset(cfg, "val")

    # 进度条（可选）：有 tqdm 时显示 step / loss / lr / ETA
    pbar = None
    if _HAS_TQDM:
        if cfg.train.max_steps:
            pbar_total = cfg.train.max_steps
        else:
            steps_per_epoch = _count_split_records(cfg, "train") // cfg.train.batch_size
            pbar_total = max(cfg.train.epochs * steps_per_epoch, 1)
        pbar = tqdm(
            total=pbar_total, unit="step", desc="SPICE pretrain", dynamic_ncols=True
        )

    writer = tf.summary.create_file_writer(cfg.train.log_dir) if _HAS_TB else None
    best_val = float("inf")
    start = time.time()

    def _log_scalar(tag, value, step):
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    for epoch in range(cfg.train.epochs):
        print(f"== epoch {epoch}/{cfg.train.epochs} 开始 ==", flush=True)
        ep_loss = 0.0
        ep_steps = 0
        for x, y in train_ds:
            dist_loss, rmsd = train_step(
                model, optimizer, x, y, cfg.train.grad_clip,
                cfg.train.use_mixed_precision,
                cfg.train.dist_weight,
                cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max,
                cfg.train.pair_weight,
            )
            global_step += 1
            ckpt.step.assign(global_step)
            ep_loss += float(dist_loss)
            ep_steps += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(dist_loss):.4f}")

            if global_step % cfg.train.log_every == 0:
                # Keras 3：optimizer.learning_rate 是属性，返回当前 LR 张量（schedule 自动按 iterations 求值）
                lr_now = float(lr_opt.learning_rate)
                line = (
                    f"[epoch {epoch}] step {global_step} | dist {float(dist_loss):.4f} "
                    f"| rmsd {float(rmsd):.4f} Å | lr {lr_now:.2e} | "
                    f"{time.time()-start:.0f}s"
                )
                # 始终用普通 print 输出完整一行（tqdm 会在进度条下方自动重绘，不会吞掉）
                print(line, flush=True)
                _log_scalar("train/dist", dist_loss, global_step)
                _log_scalar("train/rmsd", rmsd, global_step)
                _log_scalar("train/lr", lr_now, global_step)

            if global_step % cfg.train.ckpt_every == 0:
                manager.save()
            if cfg.train.max_steps and global_step >= cfg.train.max_steps:
                break

        # 验证（val 文件数可能为 0，例如只有 1 个 TFRecord 时）
        avg_train = ep_loss / max(ep_steps, 1)
        val_dists, val_rmsds, val_n = 0.0, 0.0, 0
        for x, y in val_ds:
            vr, vd = val_step(
                model, x, y, cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max
            )
            b = int(tf.shape(x["tokens"])[0])
            val_dists += float(vd) * b
            val_rmsds += float(vr) * b
            val_n += b
        if val_n > 0:
            val_dist = val_dists / val_n
            val_rmsd = val_rmsds / val_n
            print(
                f"== epoch {epoch} done | train_dist {avg_train:.4f} | "
                f"val_dist {val_dist:.4f} | val_rmsd {val_rmsd:.4f} Å =="
            )
            _log_scalar("val/dist", val_dist, global_step)
            _log_scalar("val/rmsd", val_rmsd, global_step)
            if val_dist < best_val:
                best_val = val_dist
                model.save_weights(os.path.join(cfg.train.ckpt_dir, "best_weights.weights.h5"))
        else:
            val_dist = None
            print(f"== epoch {epoch} done | train_dist {avg_train:.4f} | 无验证集 ==")
        manager.save()

        if cfg.train.max_steps and global_step >= cfg.train.max_steps:
            break

    if pbar is not None:
        pbar.close()

    if best_val != float("inf"):
        print(f"训练完成。最优验证距离 loss: {best_val:.4f}")
    else:
        print("训练完成（未评估验证集，仅 1 个 TFRecord 文件时会出现）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
