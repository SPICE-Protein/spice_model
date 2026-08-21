from __future__ import annotations

import math

import tensorflow as tf

from spice_pre.keras_utils import register_keras_serializable
from spice_pre.models.adaln import AdaLN


def sinusoidal_positions(seq_len: int, embed_dim: int, dtype=tf.float32) -> tf.Tensor:
    pos = tf.range(seq_len, dtype=tf.float32)[:, None]          
    i = tf.range(embed_dim, dtype=tf.float32)[None, :]          
    even = tf.cast(tf.math.floormod(i, 2) == 0, tf.float32)
    omega = 1.0 / tf.pow(10000.0, (i // 2) / (embed_dim / 2.0))
    angle = pos * omega
    out = tf.where(even == 1.0, tf.sin(angle), tf.cos(angle))
    return tf.cast(out, dtype)


@register_keras_serializable(package="spice_pre")
class TransformerBlock(tf.keras.layers.Layer):

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
        attn_mask = mask[:, None, None, :] > 0.5   
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
