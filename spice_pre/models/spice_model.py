"""SPICE Pre-train 模型：动态 Transformer + AdaLN + 双路四头（当前启用 Head A）。

Phase 1（Pre-train）只训练 Head A（坐标头）：输入 (Seq, Env)，输出 Cα 坐标 [L, 3]。
Head B / B' / C / D 为 Phase 2（RL）预留，默认注册但不参与本阶段 loss。
"""
from __future__ import annotations

import tensorflow as tf

from spice_pre.config import ModelConfig
from spice_pre.keras_utils import register_keras_serializable
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

        # token embedding + 位置编码
        self.token_embed = tf.keras.layers.Embedding(
            config.vocab_size, config.embed_dim, mask_zero=True, name="token_embed"
        )
        self.input_dropout = tf.keras.layers.Dropout(config.dropout)

        # 动态 Transformer 编码器（AdaLN 环境注入）
        self.encoder = TransformerEncoder(
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            env_dim=config.env_dim,
            dropout=config.dropout,
            name="transformer_encoder",
        )

        # Head A：坐标头（路径 A）—— Pre-train 唯一监督头
        self.head_a = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(config.embed_dim, activation="gelu"),
                tf.keras.layers.Dropout(config.dropout),
                tf.keras.layers.Dense(3, name="head_a_coords"),
            ],
            name="head_a",
        )

        # 辅助头：binned distogram（预测每对残基距离分布 [B,L,L,N_BINS]）
        # 因子化双线性 + 对称化：logits[b,i,j,k] = sum_d u[i,d]·W[k,d]·u[j,d]
        self.dist_proj = tf.keras.layers.Dense(config.dist_dim, name="dist_proj")
        self.dist_bin_weights = self.add_weight(
            name="dist_bin_weights",
            shape=(config.dist_bins, config.dist_dim),
            initializer="glorot_normal",
            trainable=True,
        )

        # ---- 预留头（Phase 2 RL 使用；Pre-train 默认不实例化）----
        # Head B：突变概率矩阵 [L, 20]
        self.head_b = (
            tf.keras.layers.Dense(20, name="head_b_mutation") if "B" in self.heads else None
        )
        # Head B'：突变后坐标 [L, 3]
        self.head_bp = (
            tf.keras.layers.Dense(3, name="head_bp_coords") if "Bp" in self.heads else None
        )
        # Head C：环境偏移 [ΔpH, ΔT]
        self.head_c = (
            tf.keras.layers.Dense(2, name="head_c_env_offset") if "C" in self.heads else None
        )
        # Head D：双路置信度 [0,1] x2
        self.head_d = (
            tf.keras.layers.Dense(2, activation="sigmoid", name="head_d_confidence")
            if "D" in self.heads
            else None
        )

    def build(self, input_shape=None):
        """Keras 3 要求 subclass model 显式声明 build（层都在 __init__ 构建）。"""
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        """inputs: dict {tokens:[B,L] int, env:[B,C] float, mask:[B,L] float}"""
        tokens = inputs["tokens"]
        env = inputs["env"]
        mask = inputs["mask"]

        b, l = tf.shape(tokens)[0], tf.shape(tokens)[1]
        x = self.token_embed(tokens)                       # [B, L, D]
        # mixed_float16 下 x 是 fp16：位置编码按 x.dtype 生成，避免 fp16+fp32 报错
        pos = tf.cast(
            sinusoidal_positions(l, self.embed_dim), x.dtype
        )[None, :, :]                                      # [1, L, D]
        x = x + pos
        x = self.input_dropout(x, training=training)

        z = self.encoder(x, env, mask, training=training)  # [B, L, D]

        out = {"coords": self.head_a(z), "z": z}           # Head A

        # 辅助 distogram：binned 距离分布 logits [B,L,L,N_BINS]，对称化（fp32 避免溢出/混合精度 dtype 冲突）
        u = tf.cast(self.dist_proj(z), tf.float32)                # [B,L,D2]
        w = tf.cast(self.dist_bin_weights, tf.float32)            # [N,D2]
        uw = tf.einsum("bid,kd->bikd", u, w)                      # [B,L,N,D2]
        logits = tf.einsum("bikd,bjd->bijk", uw, u)              # [B,L,L,N]
        out["dist_logits"] = logits + tf.transpose(logits, perm=[0, 2, 1, 3])

        # 预留头（RL 阶段启用）
        if self.head_d is not None:
            out["conf"] = self.head_d(tf.reduce_mean(z, axis=1))
        if self.head_b is not None:
            out["mutation"] = self.head_b(z)
        if self.head_bp is not None:
            out["coords_mut"] = self.head_bp(z)
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
