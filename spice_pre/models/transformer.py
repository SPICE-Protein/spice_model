"""动态 Transformer 编码器（可变长序列 + AdaLN 环境注入）。

支持 batch 内不同长度序列：通过 padding + Attention Mask 批处理，
环境向量经 AdaLN 逐层注入，输出序列-环境融合表征 z。
"""
from __future__ import annotations

import math

import tensorflow as tf

from spice_pre.keras_utils import register_keras_serializable
from spice_pre.models.adaln import AdaLN


def sinusoidal_positions(seq_len: int, embed_dim: int, dtype=tf.float32) -> tf.Tensor:
    """正弦位置编码，支持任意长度（无需训练参数）。返回 [seq_len, embed_dim]。"""
    pos = tf.range(seq_len, dtype=tf.float32)[:, None]          # [L, 1]
    i = tf.range(embed_dim, dtype=tf.float32)[None, :]          # [1, D]
    even = tf.cast(tf.math.floormod(i, 2) == 0, tf.float32)
    omega = 1.0 / tf.pow(10000.0, (i // 2) / (embed_dim / 2.0))
    angle = pos * omega
    out = tf.where(even == 1.0, tf.sin(angle), tf.cos(angle))
    return tf.cast(out, dtype)


@register_keras_serializable(package="spice_pre")
class TransformerBlock(tf.keras.layers.Layer):
    """单层：AdaLN → MHA → residual → AdaLN → FFN → residual"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        env_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.env_dim = env_dim
        self.dropout_rate = dropout

        self.adaln1 = AdaLN(embed_dim, env_dim)
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        self.adaln2 = AdaLN(embed_dim, env_dim)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ffn_dim, activation="gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, env: tf.Tensor, mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: [B, L, D], env: [B, C], mask: [B, L] (1=有效)
        # Keras MultiHeadAttention 的 attention_mask 是布尔掩码（True=可 attend），
        # 不是加性偏置。直接把有效位转成 bool，避免掩码反相（只 attend 到 padding）。
        attn_mask = mask[:, None, None, :] > 0.5   # [B, 1, 1, L] bool，True=有效残基
        h = self.adaln1(x, env)
        h = self.attn(h, h, attention_mask=attn_mask)
        x = x + self.dropout(h, training=training)

        h = self.adaln2(x, env)
        h = self.ffn(h, training=training)
        x = x + self.dropout(h, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            env_dim=self.env_dim,
            dropout=self.dropout_rate,
        )
        return cfg


@register_keras_serializable(package="spice_pre")
class TransformerEncoder(tf.keras.layers.Layer):
    """堆叠 N 个 TransformerBlock，输出序列-环境融合表征 z [B, L, D]。"""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        env_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.env_dim = env_dim
        self.dropout_rate = dropout

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ffn_dim, env_dim, dropout)
            for _ in range(num_layers)
        ]

    def call(self, x: tf.Tensor, env: tf.Tensor, mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        for blk in self.blocks:
            x = blk(x, env, mask, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            env_dim=self.env_dim,
            dropout=self.dropout_rate,
        )
        return cfg
