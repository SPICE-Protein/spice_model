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
from contextlib import nullcontext

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


def build_lr_schedule(cfg: Config, total_steps: int | None = None):
    """LR 余弦退火按真实训练步数衰减。

    之前 total 默认 1e6，而实际只跑 ~2 万步 → cosine 退化成常数 1e-3，
    全程不退火，模型在盆地边缘震荡不收敛。传入真实 total_steps 修复。
    """
    total = total_steps or cfg.train.max_steps or 1_000_000
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


def _make_padded_dataset(cfg: Config, split: str, batch_size: int | None = None):
    """按 split 构建数据管线。batch_size 缺省用 cfg.train.batch_size
    （多卡 MirroredStrategy 时传 per-replica batch）。"""
    bs = batch_size or cfg.train.batch_size
    ds = load_tfrecord_dataset(cfg, split=split)
    # 过滤含非有限值（NaN/Inf）的样本：脏坐标或脏 env 都会把 loss 污染成 NaN。
    # ⚠️ 之前只查 coords、漏了 env——env 走 AdaLN 逐层注入，一个 NaN 就能毒掉整批。
    ds = ds.filter(lambda x, y: (
        tf.reduce_all(tf.math.is_finite(x["coords"])) &
        tf.reduce_all(tf.math.is_finite(x["env"]))
    ))
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
        batch_sizes = [bs] * n_buckets
        # 长序列桶降 batch：控制 [B,L,L,N_BINS] 距离张量 + attention 内存。
        # 内存 ~ B·L²：L≈383 若用满 batch 96 会产生 [96,383,383,24]（~1.4GB fp32）
        # 在显存较紧的 GPU（如 Kaggle）上 LogSoftmax OOM。按桶上界递减：
        #   ≥256 折半、≥320 四分之一、≥448 八分之一。
        for i, hi in enumerate(boundaries):
            if hi >= 448:
                batch_sizes[i] = max(2, bs // 8)
            elif hi >= 320:
                batch_sizes[i] = max(4, bs // 4)
            elif hi >= 256:
                batch_sizes[i] = max(8, bs // 2)
        batch_sizes[-1] = max(2, bs // 8)  # 溢出桶（≤max_seq_len 时不会到）
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
            bs,
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
        # 分开返回 ce / pair：总 loss 被 pair aux 噪声主导看不出趋势，日志直接看纯 CE。
        ce = dist_weight * distogram_ce_loss(
            out["dist_logits"], y, x["mask"], dist_bins, dist_min, dist_max
        )
        pair = pair_weight * pairwise_coord_loss(out["coords"], y, x["mask"])
        unscaled_loss = ce + pair
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
    # NaN/Inf 安全：把非有限梯度替换为 0。单个坏样本 / fp16 溢出时，
    # 若让 NaN 梯度进 apply_gradients，权重会被污染成 NaN 且永不恢复（后续 step 全 NaN）。
    grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return unscaled_loss, ce, pair, rmsd


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

    # 多卡数据并行：>1 张 GPU 时启用 MirroredStrategy（单卡/CPU 走原路径，行为不变）
    strategy = None
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"多卡训练: MirroredStrategy（{strategy.num_replicas_in_sync} 张 GPU，"
              f"全局 batch={cfg.train.batch_size}）")

    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
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
        # 每 epoch 步数 & 总步数：LR cosine 按真实训练时长退火（max_steps=0 时不再退化成常数 1e-3）
        steps_per_epoch = _count_split_records(cfg, "train") // cfg.train.batch_size
        total_steps = cfg.train.max_steps or max(steps_per_epoch * cfg.train.epochs, 1)
        lr_schedule = build_lr_schedule(cfg, total_steps=total_steps)
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

    # checkpoint 恢复（自动跳过 NaN 污染 / 版本不兼容的 checkpoint：从最新往回找第一个可用的）
    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step=tf.Variable(0, dtype=tf.int64)
    )
    manager = tf.train.CheckpointManager(
        ckpt, cfg.train.ckpt_dir, max_to_keep=3
    )
    restored = False
    if manager.latest_checkpoint:

        def _weights_finite(m):
            """全部可训练权重都有限（NaN/Inf 出现在任意一层都视为坏 checkpoint）。"""
            return all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy())
                       for v in m.trainable_variables)

        for cp in reversed(manager.checkpoints):   # checkpoints 旧→新，反转 = 从最新往回
            # 1) 尝试完整恢复（模型 + 优化器动量 + 步数）
            try:
                ckpt.restore(cp).expect_partial()
                if _weights_finite(model):
                    print(f"已恢复 checkpoint: {cp}（权重+优化器 ✅）")
                    restored = True
                    break
                print(f"⚠️ 跳过 NaN 污染 checkpoint: {cp}（权重含非有限值）")
                continue
            except Exception as e:
                # 完整恢复失败（多为优化器 dtype/版本不兼容，如 RestoreV2 step_counter）
                print(f"⚠️ 完整恢复 {cp} 失败（{type(e).__name__}: {e}），尝试仅恢复权重")
            # 2) 退化为仅恢复模型权重 + 全局步数（丢弃优化器动量，lr 重新 warmup）
            try:
                ckpt_fb = tf.train.Checkpoint(model=model, step=ckpt.step)
                ckpt_fb.restore(cp).expect_partial()
                if _weights_finite(model):
                    print(f"已恢复 checkpoint: {cp}（仅权重+步数，动量丢弃 ✅）")
                    restored = True
                    break
                print(f"⚠️ 跳过 NaN 污染 checkpoint: {cp}")
            except Exception as e2:
                print(f"⚠️ 跳过不可用 checkpoint {cp}: {type(e2).__name__}: {e2}")
        if not restored:
            print("⚠️ 没有任何可用的 checkpoint（全被 NaN/版本污染）——从随机初始化开始")
    global_step = int(ckpt.step)

    if strategy is not None:
        train_ds = strategy.distribute_datasets_from_function(
            lambda input_context: _make_padded_dataset(
                cfg, "train",
                batch_size=input_context.get_per_replica_batch_size(cfg.train.batch_size),
            )
        )
        val_ds = strategy.distribute_datasets_from_function(
            lambda input_context: _make_padded_dataset(
                cfg, "val",
                batch_size=input_context.get_per_replica_batch_size(cfg.train.batch_size),
            )
        )
    else:
        train_ds = _make_padded_dataset(cfg, "train")
        val_ds = _make_padded_dataset(cfg, "val")

    # 断点续训：从上次 global_step 对应的 epoch 继续，跳过已训练步数（不重走前面数据）
    start_epoch = min(global_step // max(steps_per_epoch, 1), cfg.train.epochs)
    skip_in_epoch = global_step - start_epoch * steps_per_epoch
    if start_epoch >= cfg.train.epochs:
        print(f"== 已跑满 {cfg.train.epochs} epoch（global_step={global_step}），无需再训练 ==")
        return
    if global_step > 0:
        print(f"== 断点续训：从 epoch {start_epoch} 继续（跳过本 epoch 前 {skip_in_epoch} 步）==", flush=True)

    # 进度条（可选）：有 tqdm 时显示 step / loss / lr / ETA
    pbar = None
    if _HAS_TQDM:
        if cfg.train.max_steps:
            pbar_total = max(cfg.train.max_steps - global_step, 1)
        else:
            pbar_total = max(
                (cfg.train.epochs - start_epoch) * steps_per_epoch - skip_in_epoch, 1
            )
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

    # 训练/验证 step 分发：多卡走 strategy.run + 归并；单卡走原逻辑（输入为 (x,y) 或 PerReplica）。
    # ⚠️ 不要在这里给闭包加 @tf.function——捕获 cfg/model/optimizer/strategy 的闭包会触发
    #    AutoGraph closure-mismatch bug（"requested ('cfg','model','strategy')... "→ TypeError）。
    #    train_step/val_step 本身已是模块级 @tf.function，strategy.run 分发时会复用其 trace，性能不掉。
    if strategy is not None:

        def run_train_step(dist_inputs):
            x, y = dist_inputs
            per_replica = strategy.run(
                train_step,
                args=(model, optimizer, x, y, cfg.train.grad_clip,
                      cfg.train.use_mixed_precision, cfg.train.dist_weight,
                      cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max,
                      cfg.train.pair_weight),
            )
            n = strategy.num_replicas_in_sync
            return tuple(strategy.reduce("SUM", v, axis=None) / n for v in per_replica)

        def run_val_step(dist_inputs):
            x, y = dist_inputs
            per_replica = strategy.run(
                val_step,
                args=(model, x, y, cfg.model.dist_bins,
                      cfg.model.dist_min, cfg.model.dist_max),
            )
            n = strategy.num_replicas_in_sync
            return tuple(strategy.reduce("SUM", v, axis=None) / n for v in per_replica)
    else:

        def run_train_step(inputs):
            x, y = inputs
            return train_step(
                model, optimizer, x, y, cfg.train.grad_clip,
                cfg.train.use_mixed_precision, cfg.train.dist_weight,
                cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max,
                cfg.train.pair_weight,
            )

        def run_val_step(inputs):
            x, y = inputs
            return val_step(model, x, y, cfg.model.dist_bins,
                            cfg.model.dist_min, cfg.model.dist_max)

    for epoch in range(start_epoch, cfg.train.epochs):
        print(f"== epoch {epoch}/{cfg.train.epochs} 开始 ==", flush=True)
        ep_loss = 0.0
        ep_steps = 0
        ep_nan_steps = 0
        if epoch == start_epoch and skip_in_epoch > 0:
            # 续训的第一个 epoch：跳掉已训练的 skip_in_epoch 个 batch（其余 epoch 正常完整迭代）
            it = iter(train_ds)
            for _ in range(skip_in_epoch):
                next(it)
            step_iter = it
        else:
            step_iter = iter(train_ds)
        for dist_inputs in step_iter:
            dist_loss, ce_loss, pair_loss, rmsd = run_train_step(dist_inputs)
            global_step += 1
            ckpt.step.assign(global_step)
            ep_steps += 1
            # NaN 安全：单个坏 batch 的 loss 会是 NaN（梯度已置零、权重无恙），
            # 但若不剔除，会污染整个 epoch 的均值（mean(…, NaN)=NaN）。这里只累加有限值，另计 NaN 步数。
            dl = float(dist_loss)
            if np.isfinite(dl):
                ep_loss += dl
            else:
                ep_nan_steps += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(dist_loss):.4f}", ce=f"{float(ce_loss):.2f}")

            if global_step % cfg.train.log_every == 0:
                # Keras 3：optimizer.learning_rate 是属性，返回当前 LR 张量（schedule 自动按 iterations 求值）
                lr_now = float(lr_opt.learning_rate)
                line = (
                    f"[epoch {epoch}] step {global_step} | dist {float(dist_loss):.4f} "
                    f"| ce {float(ce_loss):.4f} | pair {float(pair_loss):.4f} "
                    f"| rmsd {float(rmsd):.4f} Å | lr {lr_now:.2e} | "
                    f"{time.time()-start:.0f}s"
                )
                # 始终用普通 print 输出完整一行（tqdm 会在进度条下方自动重绘，不会吞掉）
                print(line, flush=True)
                _log_scalar("train/dist", dist_loss, global_step)
                _log_scalar("train/ce", ce_loss, global_step)
                _log_scalar("train/pair", pair_loss, global_step)
                _log_scalar("train/rmsd", rmsd, global_step)
                _log_scalar("train/lr", lr_now, global_step)

            if global_step % cfg.train.ckpt_every == 0:
                manager.save()
            if cfg.train.max_steps and global_step >= cfg.train.max_steps:
                break

        # 验证（val 文件数可能为 0，例如只有 1 个 TFRecord 时）
        avg_train = ep_loss / max(ep_steps - ep_nan_steps, 1)
        val_dists, val_rmsds, val_n = 0.0, 0.0, 0
        for dist_inputs in val_ds:
            vr, vd = run_val_step(dist_inputs)
            if strategy is None:
                b = int(tf.shape(dist_inputs[0]["tokens"])[0])
            else:
                b = cfg.train.batch_size
            val_dists += float(vd) * b
            val_rmsds += float(vr) * b
            val_n += b
        if val_n > 0:
            val_dist = val_dists / val_n
            val_rmsd = val_rmsds / val_n
            nan_note = f" | ⚠️NaN步 {ep_nan_steps}/{ep_steps}" if ep_nan_steps else ""
            print(
                f"== epoch {epoch} done | train_dist {avg_train:.4f} | "
                f"val_dist {val_dist:.4f} | val_rmsd {val_rmsd:.4f} Å{nan_note} =="
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
