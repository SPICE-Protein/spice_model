from __future__ import annotations

import tensorflow as tf

from spice_pre.keras_utils import register_keras_serializable


@register_keras_serializable(package="spice_pre")
class AdaLN(tf.keras.layers.Layer):

    def __init__(self, hidden_dim: int, env_dim: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.env_dim = env_dim
        self.eps = eps
        self.ln = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.gamma_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False)
        self.beta_proj = tf.keras.layers.Dense(hidden_dim, use_bias=False)

    def call(self, x: tf.Tensor, env: tf.Tensor) -> tf.Tensor:
        g = self.gamma_proj(env)[:, None, :]   
        b = self.beta_proj(env)[:, None, :]    
        return (1.0 + g) * self.ln(x) + b

    def get_config(self):
        cfg = super().get_config()
        cfg.update(hidden_dim=self.hidden_dim, env_dim=self.env_dim, eps=self.eps)
        return cfg
