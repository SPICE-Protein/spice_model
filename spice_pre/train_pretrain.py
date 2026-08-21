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
from spice_pre.losses.frame import (
    frame_chirality_loss,
    frame_clash_loss,
    frame_dist_consistency_loss,
    frame_kabsch_loss,
)
from spice_pre.losses.kabsch_rmsd import distogram_ce_loss, kabsch_rmsd
from spice_pre.models import SPICEPretrainModel

try:
    import tensorboard  # noqa: F401

    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    from tqdm.auto import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class _WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

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
    total = total_steps or cfg.train.max_steps or 1_000_000
    return _WarmupCosineSchedule(cfg.train.lr, cfg.train.warmup_steps, total)


def _bucket_length_fn(x, y=None):
    return tf.shape(x["tokens"])[0]


def _make_padded_dataset(cfg: Config, split: str, batch_size: int | None = None):
    bs = batch_size or cfg.train.batch_size
    ds = load_tfrecord_dataset(cfg, split=split)
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
        b_step = 64
        boundaries = list(range(b_step, cfg.data.max_seq_len, b_step)) + [
            cfg.data.max_seq_len + 1
        ]
        n_buckets = len(boundaries) + 1
        batch_sizes = [bs] * n_buckets
        for i, hi in enumerate(boundaries):
            if hi >= 448:
                batch_sizes[i] = max(2, bs // 8)
            elif hi >= 320:
                batch_sizes[i] = max(4, bs // 4)
            elif hi >= 256:
                batch_sizes[i] = max(8, bs // 2)
        batch_sizes[-1] = max(2, bs // 8)  
        # ⚠️ Shuffle moved before bucketing (raw elements are ~10KB each; buffer 2000 uses only ~20MB RAM) —
        # Do not place shuffle after the on-disk cache. The cache has only 1231 batches (< buffer size), 
        # which would pull the entire cache into RAM and trigger OOM.
        ds = ds.shuffle(2000, reshuffle_each_iteration=True)
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
            if cfg.train.cache_path:
                # ⚠️ On-disk caching: stream and serialize the bucketed dataset to disk during the first epoch,
                # then read directly from the cached files in subsequent epochs.
                # This prevents holding the entire dataset in RAM (resolving RAM OOM) and avoids re-parsing TFRecords
                # (substantially faster than re-reading every epoch).
                # TF will generate <path>.index and <path>.data-* files; warning warnings are known TF false positives and can be safely ignored.
                _cp = os.path.expanduser(cfg.train.cache_path)
                os.makedirs(os.path.dirname(_cp) or ".", exist_ok=True)
                ds = ds.cache(_cp)
            else:
                ds = ds.cache()
            # ⚠️ After caching on disk, disable large shuffle buffers (which would load the entire cache back into RAM).
            # Keep a small buffer window (128) to ensure inter-epoch shuffling (~50MB overhead); 
            # primary shuffling has already been performed on raw elements upstream.
            ds = ds.shuffle(128, reshuffle_each_iteration=True)
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
    return sum(1 for _ in load_tfrecord_dataset(cfg, split=split))


def _count_train_batches(cfg: Config) -> int:
    """Actual training batches after bucketing (used for LR annealing total_steps).

    🔴 Cannot use raw_records // batch_size: bucketing scales down batch size by length, 
      includes drop_remainder, and pads inputs. This makes the actual batch count 3x to 4.6x larger 
      than raw//batch_size. Using raw records previously caused cosine annealing to hit the minimum LR 
      prematurely around epoch 6, leaving zero learning rate for the latter half, which stagnated CE loss 
      at a high plateau (empirically confirmed in production, 2026-08-13).
    """
    if cfg.train.cache_train and cfg.train.cache_path:
        n = build_train_cache(cfg)     # Run full verification traversal + return actual bucketed batch count (fast via on-disk cache)
        if n > 0:
            return n
    ds = _make_padded_dataset(cfg, "train")
    n = 0
    for _ in ds:
        n += 1
    return max(n, 1)


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, x, y, grad_clip, use_loss_scale,
               dist_weight, dist_bins, dist_min, dist_max, pair_weight,
               pair_warmup_steps, global_step,
               frame_chirality_weight, frame_clash_weight, coord_max_len=0,
               frame_consistency_weight=0.0):
    with tf.GradientTape() as tape:
        out = model(x, training=True)
        ce = dist_weight * distogram_ce_loss(
            out["dist_logits"], y, x["mask"], dist_bins, dist_min, dist_max
        )
        # ⚠️ Dynamic warmup weights must be computed inside the graph using the step tensor — 
        # do not pass changing Python constants into tf.function (which triggers step-level retracing, 
        # causing 17s/step lag and dynamic graph cumulative leaks, identified on 2026-08-13).
        pw = pair_weight * tf.minimum(
            1.0, tf.cast(global_step, tf.float32) / tf.cast(pair_warmup_steps, tf.float32))
        # 🔴 Length Gating (decisive fix, 2026-08-13): long-chain cumulative errors in coordinate reconstruction (cumsum) 
        #   diverge, generating extreme gradients that poison the encoder and starve distogram learning.
        #   Coordinate loss is restricted to chains <= coord_max_len; distogram is trained on all chains (long-chain distances remain learnable, verified).
        if coord_max_len and coord_max_len > 0:
            n_res = tf.reduce_sum(x["mask"], axis=1)
            gate = tf.cast(n_res <= tf.cast(coord_max_len, tf.float32), tf.float32)
        else:
            gate = None
        kab = pw * frame_kabsch_loss(out["coords"], y, x["mask"], gate)
        chi = frame_chirality_weight * frame_chirality_loss(
            out["coords"], y, x["mask"], gate
        )
        cla = frame_clash_weight * frame_clash_loss(out["coords"], x["mask"], gate=gate)
        # 🔄 Coordinate-distogram self-consistency loss (2026-08-13): uses the model's own distogram 
        #    (well-learned via cross-entropy across all chain lengths) as a self-supervised teacher for coordinates, 
        #    bypassing the length gate. This provides structural coordinate signals to the Frame head and recycling 
        #    for long chains. The relative error in the logarithmic domain remains bounded, avoiding coordinate 
        #    divergence and encoder poisoning typical of native Kabsch RMSD.
        consist = 0.0
        if frame_consistency_weight and frame_consistency_weight > 0:
            consist = frame_consistency_weight * frame_dist_consistency_loss(
                out["coords"], out["dist_logits"], x["mask"],
                dist_bins, dist_min, dist_max)
        pair = kab + chi + cla + consist
        unscaled_loss = ce + pair
        rmsd = tf.reduce_mean(kabsch_rmsd(out["coords"], y, x["mask"]))
        loss = optimizer.scale_loss(unscaled_loss)
    grads = tape.gradient(loss, model.trainable_variables)
    if use_loss_scale:
        # ⚠️ Keras 3.15+ removed dynamic_scale (present in TF 2.20 -> preserve behavior; fallback to initial_scale on 2.21+)
        scale = getattr(optimizer, "dynamic_scale", None)
        if scale is None:
            scale = getattr(optimizer, "initial_scale", 1.0)
        grads, _ = tf.clip_by_global_norm(grads, grad_clip * scale)
    else:
        grads, _ = tf.clip_by_global_norm(grads, grad_clip)
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


def build_train_cache(cfg: Config, verbose: bool = True) -> int:
    """Pre-build and serialize the bucketed training dataset cache to disk; called prior to training launch.

    Returns the number of batches; reuses the cache if it already exists without rebuilding.
    Usage: python -m spice_pre.train_pretrain --config configs/pretrain.yaml --build-cache
    Or call build_train_cache(cfg) before train(cfg) in notebooks.
    """
    if not cfg.train.cache_train or not cfg.train.cache_path:
        if verbose:
            print("[cache] cache_train is disabled or cache_path is unset, skipping pre-building")
        return 0
    _cp = os.path.expanduser(cfg.train.cache_path)
    os.makedirs(os.path.dirname(_cp) or ".", exist_ok=True)
    # ⚠️ Do not rely solely on "file existence" — a prior crash could leave a corrupted/partial cache file,
    # causing TF to silently rebuild it mid-training (slowing down epoch 0).
    # Always perform a full traversal: if the cache is complete, validation is extremely fast (within minutes);
    # if missing or incomplete, it explicitly rebuilds before training starts.
    print(f"[cache] Traversing training cache (validating if complete, rebuilding if missing/partial)...", flush=True)
    ds = _make_padded_dataset(cfg, "train")
    n = 0
    for _ in ds:
        n += 1
    if verbose:
        print(f"[cache] ✅ Validation/building completed: {n} batches -> {_cp}.* (streamed directly during training)", flush=True)
    return n


def train(cfg: Config) -> None:
    tf.random.set_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    setup_gpu(cfg.train.use_gpu, cfg.train.gpu_mem_growth, cfg.train.gpu_devices)
    if cfg.train.use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: mixed_float16 (ON)")
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU Switch: {'ON ' + str(gpus) if cfg.train.use_gpu and gpus else 'OFF (CPU)'}")

    strategy = None
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Multi-GPU: MirroredStrategy ({strategy.num_replicas_in_sync} GPUs, "
              f"global batch_size={cfg.train.batch_size})")

    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        model = SPICEPretrainModel(cfg.model)
        model(
            {
                "tokens": tf.zeros([1, 8], tf.int32),
                "env": tf.zeros([1, 3]),
                "mask": tf.ones([1, 8]),
            },
            training=False,
        )
        steps_per_epoch = _count_train_batches(cfg)
        total_steps = cfg.train.max_steps or max(steps_per_epoch * cfg.train.epochs, 1)
        lr_schedule = build_lr_schedule(cfg, total_steps=total_steps)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=cfg.train.weight_decay
        )
        if cfg.train.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        lr_opt = optimizer.inner_optimizer if cfg.train.use_mixed_precision else optimizer
        optimizer.build(model.trainable_variables)

    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"SPICE Pre-train model parameters: {n_params:,}")

    ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step=tf.Variable(0, dtype=tf.int64)
    )
    manager = tf.train.CheckpointManager(
        ckpt, cfg.train.ckpt_dir, max_to_keep=3
    )
    restored = False
    if manager.latest_checkpoint:

        def _weights_finite(m):
            return all(bool(tf.reduce_all(tf.math.is_finite(v)).numpy())
                       for v in m.trainable_variables)

        for cp in reversed(manager.checkpoints):   
            try:
                ckpt.restore(cp).expect_partial()
                if _weights_finite(model):
                    print(f"Restored checkpoint: {cp} (weights + optimizer ✅)")
                    restored = True
                    break
                print(f"⚠️ Skipping NaN-polluted checkpoint: {cp} (weights contain non-finite values)")
                continue
            except Exception as e:
                print(f"⚠️ Full restoration of {cp} failed ({type(e).__name__}: {e}), attempting weights-only recovery")
            try:
                ckpt_fb = tf.train.Checkpoint(model=model, step=ckpt.step)
                ckpt_fb.restore(cp).expect_partial()
                if _weights_finite(model):
                    print(f"Restored checkpoint: {cp} (weights + step only, momentum discarded ✅)")
                    restored = True
                    break
                print(f"⚠️ Skipping NaN-polluted checkpoint: {cp}")
            except Exception as e2:
                print(f"⚠️ Skipping unusable checkpoint {cp}: {type(e2).__name__}: {e2}")
        if not restored:
            print("⚠️ No usable checkpoint found (all NaN-polluted or version-mismatched) — starting from random initialization")
    global_step = int(ckpt.step)

    # (On-disk cache has been fully validated/built during _count_train_batches for LR calculation; no redundant traversal needed)

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

    start_epoch = min(global_step // max(steps_per_epoch, 1), cfg.train.epochs)
    skip_in_epoch = global_step - start_epoch * steps_per_epoch
    if start_epoch >= cfg.train.epochs:
        print(f"== Already completed {cfg.train.epochs} epochs (global_step={global_step}), training skipped ==")
        return
    if global_step > 0:
        print(f"== Resuming from checkpoint: starting at epoch {start_epoch} (skipping the first {skip_in_epoch} steps of this epoch) ==", flush=True)

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
    plateau_count = 0      # 🛑 平台期检测：连续"无显著改善"的 epoch 数
    start = time.time()

    def _log_scalar(tag, value, step):
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar(tag, value, step=step)

    if strategy is not None:

        def run_train_step(dist_inputs, pair_weight, pair_warmup_steps, global_step):
            x, y = dist_inputs
            per_replica = strategy.run(
                train_step,
                args=(model, optimizer, x, y, cfg.train.grad_clip,
                      cfg.train.use_mixed_precision, cfg.train.dist_weight,
                      cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max,
                      pair_weight, pair_warmup_steps, global_step,
                      cfg.train.frame_chirality_weight, cfg.train.frame_clash_weight,
                      cfg.train.coord_max_len, cfg.train.frame_consistency_weight),
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

        def run_train_step(inputs, pair_weight, pair_warmup_steps, global_step):
            x, y = inputs
            return train_step(
                model, optimizer, x, y, cfg.train.grad_clip,
                cfg.train.use_mixed_precision, cfg.train.dist_weight,
                cfg.model.dist_bins, cfg.model.dist_min, cfg.model.dist_max,
                pair_weight, pair_warmup_steps, global_step,
                cfg.train.frame_chirality_weight, cfg.train.frame_clash_weight,
                cfg.train.coord_max_len, cfg.train.frame_consistency_weight,
            )

        def run_val_step(inputs):
            x, y = inputs
            return val_step(model, x, y, cfg.model.dist_bins,
                            cfg.model.dist_min, cfg.model.dist_max)

    for epoch in range(start_epoch, cfg.train.epochs):
        print(f"== epoch {epoch}/{cfg.train.epochs} started ==", flush=True)
        ep_loss = 0.0
        ep_steps = 0
        ep_nan_steps = 0
        if epoch == start_epoch and skip_in_epoch > 0:
            it = iter(train_ds)
            for _ in range(skip_in_epoch):
                next(it)
            step_iter = it
        else:
            step_iter = iter(train_ds)
        for dist_inputs in step_iter:
            # ⚠️ Warmup weights are computed inside train_step using a step tensor — 
            # do not pass dynamically changing Python constants into tf.function 
            # (which triggers step-level retracing, causing a 17s/step lag and dynamic graph cumulative leaks, identified on 2026-08-13).
            dist_loss, ce_loss, pair_loss, rmsd = run_train_step(
                dist_inputs, cfg.train.pair_weight, cfg.train.pair_warmup_steps,
                tf.constant(global_step, tf.int64))
            global_step += 1
            ckpt.step.assign(global_step)
            ep_steps += 1
            dl = float(dist_loss)
            if np.isfinite(dl):
                ep_loss += dl
            else:
                ep_nan_steps += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=f"{float(dist_loss):.4f}", ce=f"{float(ce_loss):.2f}")

            if global_step % cfg.train.log_every == 0:
                lr_now = float(lr_opt.learning_rate)
                line = (
                    f"[epoch {epoch}] step {global_step} | dist {float(dist_loss):.4f} "
                    f"| ce {float(ce_loss):.4f} | pair {float(pair_loss):.4f} "
                    f"| rmsd {float(rmsd):.4f} Å | lr {lr_now:.2e} | "
                    f"{time.time()-start:.0f}s"
                )
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
            nan_note = f" | ⚠️ NaN steps {ep_nan_steps}/{ep_steps}" if ep_nan_steps else ""
            print(
                f"== epoch {epoch} done | train_dist {avg_train:.4f} | "
                f"val_dist {val_dist:.4f} | val_rmsd {val_rmsd:.4f} Å{nan_note} =="
            )
            _log_scalar("val/dist", val_dist, global_step)
            _log_scalar("val/rmsd", val_rmsd, global_step)
            if val_dist < best_val:
                best_val = val_dist
                plateau_count = 0
                model.save_weights(os.path.join(cfg.train.ckpt_dir, "best_weights.weights.h5"))
            else:
                # 🛑 Plateau detection: alert when val_dist fails to improve significantly for `plateau_patience` epochs (can be manually halted)
                plateau_count += 1
                if plateau_count >= cfg.train.plateau_patience:
                    print(
                        f"🛑 [PLATEAU] val_dist has not improved significantly for {plateau_count} epochs "
                        f"(best={best_val:.4f}) — convergence reached, training can be stopped (best weights saved)",
                        flush=True,
                    )
                    if cfg.train.plateau_auto_stop:
                        print("🛑 [PLATEAU] plateau_auto_stop=true -> early stopping triggered", flush=True)
                        break
        else:
            val_dist = None
            print(f"== epoch {epoch} done | train_dist {avg_train:.4f} | no validation set ==")
        manager.save()

        if cfg.train.max_steps and global_step >= cfg.train.max_steps:
            break

    if pbar is not None:
        pbar.close()

    if best_val != float("inf"):
        print(f"Training completed. Best validation distance loss: {best_val:.4f}")
    else:
        print("Training completed (no validation set evaluated, typically happens when only 1 TFRecord file is present)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/pretrain.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs in configuration")
    ap.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    ap.add_argument("--build-cache", action="store_true",
                    help="Only pre-build the on-disk dataset cache and exit without training")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.build_cache:
        build_train_cache(cfg)
        return 0
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
