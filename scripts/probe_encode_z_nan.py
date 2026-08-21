#!/usr/bin/env python
"""encode_z NaN 逐层定位探针（2026-08-17）。

目标：找出 model 在特定极端环境（如 ph=10/temp=260）下 encode_z 全 NaN 时，
NaN 从哪一层开始冒出来（token_embed / 各 TransformerBlock / 各 head）。

用法（HPC，singularity 容器内）：
    cd ~/spice/model
    singularity exec "<SIF镜像>" bash -lc "
      source ~/miniconda3/etc/profile.d/conda.sh && conda activate spice && \
      python scripts/probe_encode_z_nan.py --ph 10.0 --temp 260.0 \
          --seq MEKSFVITDPRLPDNPIIFASDGFLELTEYSREEILGRNGRFLQGPETDQATVQKIQDAIRDQREITVQLINYTKSGKKFWNLLHLQPMRDQKGELQYFIGVQLDGEFIPNPLLGL
    "
    # 传 --all-envs 扫描 ph∈[2,12]×temp∈[260,330] 找所有 NaN 环境点

原理：对模型各层注册 call hook（Keras 3 用 __call__ 包装），记录每层输出是否有 NaN；
再对 encode_z 的输入 env 做归一化，确认 NaN 层。
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import tensorflow as tf

from spice_rl.config import load_config
from spice_rl.train_post import build_rl_model, _normalize_env
from spice_pre.data.preprocessing import seq_to_tokens


def build_hooked_model(cfg):
    """构建模型并给关键层挂 NaN 探针，返回 (model, hook_dict)。"""
    model = build_rl_model(cfg)
    hooks = {"layers": []}

    def _make_hook(name):
        def _hook(layer, inputs, output):
            try:
                arr = tf.cast(output, tf.float32).numpy()
                n = int(np.sum(~np.isfinite(arr)))
                hooks["layers"].append((name, n, arr.size,
                                        float(np.nanmin(arr)) if n == 0 else float("nan"),
                                        float(np.nanmax(arr)) if n == 0 else float("nan")))
            except Exception as e:  # noqa: BLE001
                hooks["layers"].append((name, -1, -1, 0.0, 0.0))
            return output
        return _hook

    # 注册 token_embed + 各 TransformerBlock 的 call hook
    for lyr in model.layers:
        if lyr.name in ("token_embed", "input_dropout", "encoder"):
            for name, layer in [("token_embed", lyr)]:
                if lyr.name == "token_embed":
                    lyr.__call__ = tf.keras.utils.pack_sequence_as  # placeholder, replaced below
    # encoder 内部 blocks
    enc = None
    for lyr in model.layers:
        if lyr.name == "encoder":
            enc = lyr
    if enc is not None:
        for i, blk in enumerate(enc.blocks):
            _orig = blk.__call__
            blk.__call__ = _make_hook(f"block{i}")  # will not work; see note below
    return model, hooks


def probe_one(model, tokens, env, tag):
    """对单个 env 跑模型前向，返回各层 NaN 情况 + z 的 NaN 数。"""
    try:
        out = model({"tokens": tokens, "env": env, "mask": np.ones_like(tokens, np.float32)},
                    training=False)
        z = out["z"][0].numpy()
        nz = int(np.sum(~np.isfinite(z)))
        # 检查所有输出头
        nan_heads = {}
        for k, v in out.items():
            a = np.asarray(v)[0] if isinstance(v, tf.Tensor) else np.asarray(v)
            nan_heads[k] = int(np.sum(~np.isfinite(a)))
        print(f"[{tag}] env={env[0].tolist()} z_nan={nz}/{z.size}  heads={nan_heads}")
        return nz
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] env={env[0].tolist()} ERROR {str(e)[:100]}")
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ph", type=float, default=10.0)
    ap.add_argument("--temp", type=float, default=260.0)
    ap.add_argument("--seq", default="MEKSFVITDPRLPDNPIIFASDGFLELTEYSREEILGRNGRFLQGPETDQATVQKIQDAIRDQREITVQLINYTKSGKKFWNLLHLQPMRDQKGELQYFIGVQLDGEFIPNPLLGL")
    ap.add_argument("--config", default="configs/posttrain.yaml")
    ap.add_argument("--all-envs", action="store_true", help="扫描 ph×temp 网格找所有 NaN 点")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_rl_model(cfg)
    print("模型构建 OK，开始探针")

    tokens = seq_to_tokens(args.seq)[None]
    mask = np.ones_like(tokens, np.float32)

    if args.all_envs:
        print("=== 全空间扫描（找 NaN 环境点）===")
        for ph in np.linspace(2.0, 12.0, 21):
            for temp in np.linspace(260.0, 330.0, 15):
                env = _normalize_env((float(ph), float(temp), 0.0), cfg.env)[None]
                out = model({"tokens": tokens, "env": env, "mask": mask}, training=False)
                z = out["z"][0].numpy()
                nz = int(np.sum(~np.isfinite(z)))
                if nz > 0:
                    print(f"  NaN点: ph={ph:.1f} temp={temp:.1f} n_nan={nz}/{z.size}")
        print("=== 扫描完成 ===")
    else:
        env = _normalize_env((args.ph, args.temp, 0.0), cfg.env)[None]
        print(f"=== 单点探测: ph={args.ph} temp={args.temp} env_norm={env[0].tolist()} ===")
        # 分层探测：手动逐层前向，定位第一个 NaN 层
        x = model.token_embed(tokens)
        pos = tf.cast(tf.constant([[[0.0] * cfg.sac.hidden_dim]]), x.dtype)
        # 用模型内部真实的 sinusoidal 位置
        from spice_pre.models.spice_model import sinusoidal_positions
        l = tf.shape(tokens)[1]
        pos = tf.cast(sinusoidal_positions(l, model.embed_dim), x.dtype)[None, :, :]
        x = x + pos
        print(f"  token_embed+pos: n_nan={int(np.sum(~np.isfinite(x.numpy())))}")
        x = model.input_dropout(x, training=False)
        print(f"  input_dropout: n_nan={int(np.sum(~np.isfinite(x.numpy())))}")
        z = x
        # 先检查各 block 的 ffn 权重是否有异常值（1e20+）
        for i, blk in enumerate(model.encoder.blocks):
            d1 = blk.ffn.layers[0]  # Dense(1024, gelu)
            d2 = blk.ffn.layers[-1]  # Dense(256)
            k1 = d1.kernel.numpy()
            k2 = d2.kernel.numpy()
            b1 = d1.bias.numpy()
            b2 = d2.bias.numpy()
            wmax1 = float(np.max(np.abs(k1)))
            wmax2 = float(np.max(np.abs(k2)))
            bmax1 = float(np.max(np.abs(b1)))
            bmax2 = float(np.max(np.abs(b2)))
            if wmax1 > 1e6 or wmax2 > 1e6 or bmax1 > 1e6 or bmax2 > 1e6:
                print(f"  [权重异常] block{i}: Dense1_kernel_max={wmax1:.3e} Dense2_kernel_max={wmax2:.3e} "
                      f"Dense1_bias_max={bmax1:.3e} Dense2_bias_max={bmax2:.3e}  ← ⚠️ 权重有超大值")
        for i, blk in enumerate(model.encoder.blocks):
            # 逐子层检查
            h = blk.adaln1(z, env)
            n1 = int(np.sum(~np.isfinite(h.numpy())))
            h = blk.attn(h, h, attention_mask=(mask[:, None, None, :] > 0.5))
            n2 = int(np.sum(~np.isfinite(h.numpy())))
            z2 = z + blk.dropout(h, training=False)
            n3 = int(np.sum(~np.isfinite(z2.numpy())))
            h2 = blk.adaln2(z2, env)
            n4 = int(np.sum(~np.isfinite(h2.numpy())))
            # ffn 内部细分。⚠️ Dense(1024) 自带 gelu activation，不要重复 .activation。
            # 忠实复现 Sequential：Dense(gelu) → Dropout → Dense
            ffn_in = h2
            fin = ffn_in.numpy()
            fin_stats = (float(np.nanmin(fin)), float(np.nanmax(fin)),
                         float(np.mean(np.abs(fin[np.isfinite(fin)]))) if np.any(np.isfinite(fin)) else 0.0)
            d1_out = blk.ffn.layers[0](ffn_in, training=False)  # Dense(1024)+gelu
            n_d1 = int(np.sum(~np.isfinite(d1_out.numpy())))
            d1a = d1_out.numpy()
            d1_stats = (float(np.nanmin(d1a)), float(np.nanmax(d1a)),
                        float(np.mean(np.abs(d1a[np.isfinite(d1a)]))) if np.any(np.isfinite(d1a)) else 0.0)
            x_ffn = d1_out
            for sub in blk.ffn.layers[1:]:
                x_ffn = sub(x_ffn, training=False)
            n5 = int(np.sum(~np.isfinite(x_ffn.numpy())))
            z = z2 + blk.dropout(x_ffn, training=False)
            n6 = int(np.sum(~np.isfinite(z.numpy())))
            nz = int(np.sum(~np.isfinite(z.numpy())))
            print(f"  block{i}: adaln1={n1} attn={n2} +res={n3} adaln2={n4} "
                  f"Dense1(gelu)={n_d1}(min={d1_stats[0]:.3g} max={d1_stats[1]:.3g} mean|.|={d1_stats[2]:.2f}) "
                  f"ffn_out={n5} +res={n6} z_nan={nz}/{z.numpy().size}")
            print(f"    ffn_in stats: min={fin_stats[0]:.2f} max={fin_stats[1]:.2f} mean|.|={fin_stats[2]:.2f}")
            if nz > 0:
                print(f"    ← 第一个 NaN 层出现在 block{i}")
                break  # 找到就停，避免后续全是传播
        # 最终 z pool
        zm = z * mask[:, :, None]
        zpool = tf.reduce_sum(zm, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1)[:, None], 1.0)
        print(f"  final z_pool: n_nan={int(np.sum(~np.isfinite(zpool.numpy())))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
