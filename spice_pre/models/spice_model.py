from __future__ import annotations

import tensorflow as tf

from spice_pre.config import ModelConfig
from spice_pre.keras_utils import register_keras_serializable
from spice_pre.models.structure import FrameStructureHead
from spice_pre.models.transformer import TransformerEncoder, sinusoidal_positions


@register_keras_serializable(package="spice_pre")
class SPICEPretrainModel(tf.keras.Model):
    def __init__(self, config: ModelConfig, heads: tuple = ("A",), **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        self.heads = tuple(heads)
        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.env_dim = config.env_dim
        self.pos_max_len = config.pos_max_len

        self.token_embed = tf.keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="token_embed"
        )
        self.input_dropout = tf.keras.layers.Dropout(config.dropout)

        self.encoder = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            env_dim=config.env_dim,
            dropout=config.dropout,
            name="transformer_encoder",
        )

        self.head_a = FrameStructureHead(
            config.embed_dim,
            bond_length=getattr(config, "bond_length", 3.8),
            dropout=config.dropout,
            recycle_steps=getattr(config, "frame_recycle_steps", 0),
            refine_dim=getattr(config, "frame_refine_dim", 64),
            refine_heads=getattr(config, "frame_refine_heads", 4),
            name="head_a",
        )

        self.dist_proj = tf.keras.layers.Dense(config.dist_dim, name="dist_proj")
        self.dist_bin_weights = self.add_weight(
            name="dist_bin_weights",
            shape=(config.dist_bins, config.dist_dim),
            initializer="glorot_normal",
            trainable=True,
        )

        self.head_b = (
            tf.keras.layers.Dense(20, name="head_b_mutation") if "B" in self.heads else None
        )
        # Head B' 已由独立 Dense(3) 直接坐标回归 → 移除：改为 Head A 对(突变)序列的折叠
        # （frame 结构头 + SE(3) recycling，键长由构造保证 3.8Å）。见 call() 中 coords_mut 别名。
        self.head_c = (
            tf.keras.layers.Dense(2, name="head_c_env_offset") if "C" in self.heads else None
        )
        self.head_d = (
            tf.keras.layers.Dense(2, activation="sigmoid", name="head_d_confidence")
            if "D" in self.heads
            else None
        )

    def build(self, input_shape=None):
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        tokens = inputs["tokens"]
        env = inputs["env"]
        mask = inputs["mask"]

        b, l = tf.shape(tokens)[0], tf.shape(tokens)[1]
        x = self.token_embed(tokens)                       
        pos = tf.cast(
            sinusoidal_positions(l, self.embed_dim), x.dtype
        )[None, :, :]                                      
        x = x + pos
        x = self.input_dropout(x, training=training)

        z = self.encoder(x, env, mask, training=training)  

        out = {"coords": self.head_a(z, mask, training=training), "z": z}           

        if getattr(self.cfg, "distogram_fp16", False):
            # ⚠️ fp16 distogram（护栏版）：先把 u/w 按 batch 归一化到健康区间（~O(1)），
            # fp16 einsum（T4 tensor core 约 8× 于 fp32），再按比例还原。
            # 误差≈fp16 精度（0.1-1%），且 logits16 ≤ dist_dim 永不溢出；CE 仍 fp32 稳定。
            u32 = tf.cast(self.dist_proj(z), tf.float32)
            w32 = tf.cast(self.dist_bin_weights, tf.float32)
            u_scale = tf.maximum(tf.reduce_max(tf.abs(u32), axis=[1, 2]), 1e-6)  # [B]
            w_scale = tf.maximum(tf.reduce_max(tf.abs(w32)), 1e-6)               # scalar
            u_n = u32 / u_scale[:, None, None]
            w_n = w32 / w_scale
            uw = tf.einsum("bid,kd->bikd", tf.cast(u_n, tf.float16),
                           tf.cast(w_n, tf.float16))
            logits = tf.einsum("bikd,bjd->bijk", uw, tf.cast(u_n, tf.float16))
            # 还原：logits_orig = u_scale² · w_scale · logits_normed
            logits = tf.cast(logits, tf.float32) * (
                u_scale * u_scale * w_scale)[:, None, None, None]
        else:
            u = tf.cast(self.dist_proj(z), tf.float32)                
            w = tf.cast(self.dist_bin_weights, tf.float32)            
            uw = tf.einsum("bid,kd->bikd", u, w)                      
            logits = tf.einsum("bikd,bjd->bijk", uw, u)              
        out["dist_logits"] = logits + tf.transpose(logits, perm=[0, 2, 1, 3])

        if self.head_d is not None:
            out["conf"] = self.head_d(tf.reduce_mean(z, axis=1))
        if self.head_b is not None:
            out["mutation"] = self.head_b(z)
        if "Bp" in self.heads:
            # Head B' = 突变序列经 Head A(frame 结构头+recycle) 的折叠：键长由构造保证 3.8Å，
            # 实测 Rg≈13Å 真实折叠；替代旧 Dense(3) 直接回归的 blob（曾致建突变体 equil 全炸）。
            out["coords_mut"] = out["coords"]
        if self.head_c is not None:
            out["env_offset"] = self.head_c(tf.reduce_mean(z, axis=1))
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            model_cfg=self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else vars(self.cfg),
            heads=list(self.heads),
        )
        return cfg

    @classmethod
    def from_config(cls, cfg):
        from spice_pre.config import ModelConfig as _MC

        mc = _MC(**cfg["model_cfg"])
        return cls(mc, heads=tuple(cfg.get("heads", ("A",))))
