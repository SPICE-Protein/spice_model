"""SPICE-SAC 网络：双头 Actor + TwinCritic（特权信息）。

定制 2（混合动作解耦输出头）：
- 连续头：偏置力基元系数 [M=16] + 环境偏移 [ΔpH, ΔT] → 各向同性高斯（重参数化）
- 离散头：突变位置 [L] + 目标氨基酸类型 [20] → Categorical，Gumbel-Softmax（τ=1）采样，
  输出 soft one-hot（可微）；两个动作拼接为平坦向量输入 Critic。

定制 3（特权信息）：
- Actor 输入：仅 z + env（不含物理指标 M）
- Critic 输入：z + M + u_hist(10) + 连续动作 + 离散动作 + mutation_mask

定制 5（分层动作时序）：
- 每步都采样完整动作向量；`mutation_allowed` 决定离散头是否"生效"。
  buffer 存原始输出 + mutation_mask，Critic 依掩码学习（Q 目标中离散熵项 × mask）。
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from spice_rl.config import SACConfig
from spice_rl.keras_utils import register_keras_serializable

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def gumbel_softmax(logits, tau: float, hard: bool = False):
    """Gumbel-Softmax（τ 温度，hard 时 straight-through one-hot）。"""
    gumbels = -tf.math.log(
        -tf.math.log(tf.random.uniform(tf.shape(logits), dtype=logits.dtype) + 1e-20) + 1e-20
    )
    y = tf.nn.softmax((logits + gumbels) / tau)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=-1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def _gaussian_log_prob(action, mean, log_std):
    std = tf.exp(log_std)
    log_prob = -0.5 * tf.square((action - mean) / std) - log_std - 0.5 * np.log(2 * np.pi)
    return tf.reduce_sum(log_prob, axis=-1)


@register_keras_serializable(package="spice_rl")
class SacActor(tf.keras.Model):
    """双头 Actor：连续（高斯）+ 离散（Gumbel-Softmax 突变）。"""

    def __init__(
        self,
        cfg: SACConfig,
        cont_dim: int,
        z_dim: int,
        env_dim: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cont_dim = cont_dim               # 连续动作维度（force + env offset）
        self.z_dim = z_dim
        self.env_dim = env_dim
        self.disc_pos_dim = cfg.discrete_position_dim  # L_max
        self.aa_dim = cfg.aa_dim

        # 连续头
        self.cont_trunk = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            ],
            name="actor_cont_trunk",
        )
        self.mean_head = tf.keras.layers.Dense(cont_dim, name="actor_mean")
        self.log_std_head = tf.keras.layers.Dense(cont_dim, name="actor_log_std")

        # 离散头（突变）：z_pool → 位置分布 + 氨基酸类型分布
        self.disc_trunk = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            ],
            name="actor_disc_trunk",
        )
        self.pos_logits_head = tf.keras.layers.Dense(
            self.disc_pos_dim, name="actor_mut_pos_logits"
        )
        self.aa_logits_head = tf.keras.layers.Dense(self.aa_dim, name="actor_mut_aa_logits")

    # ---------------- 连续头 ----------------
    def cont_dist(self, z, env):
        h = self.cont_trunk(tf.concat([z, env], axis=-1))
        mean = self.mean_head(h)
        log_std = tf.clip_by_value(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample_cont(self, z, env):
        mean, log_std = self.cont_dist(z, env)
        action = mean + tf.exp(log_std) * tf.random.normal(tf.shape(mean))
        return action, _gaussian_log_prob(action, mean, log_std)

    # ---------------- 离散头 ----------------
    def disc_logits(self, z, z_mask):
        """z_mask: [B, L_max] float（1=有效残基）→ (pos_logits, aa_logits)。"""
        h = self.disc_trunk(z)
        pos_logits = self.pos_logits_head(h)              # [B, L_max]
        mask = tf.cast(z_mask, pos_logits.dtype)
        pos_logits = tf.where(mask > 0.5, pos_logits, -1e9)
        aa_logits = self.aa_logits_head(h)                # [B, 20]
        return pos_logits, aa_logits

    def sample_disc(self, z, z_mask):
        """Gumbel-Softmax 采样 soft one-hot 离散动作 + 离散 log_pi（交叉熵）。"""
        pos_logits, aa_logits = self.disc_logits(z, z_mask)
        pos_soft = gumbel_softmax(pos_logits, self.cfg.gumbel_tau)
        aa_soft = gumbel_softmax(aa_logits, self.cfg.gumbel_tau)
        log_pi = (
            tf.reduce_sum(tf.nn.log_softmax(pos_logits) * pos_soft, axis=-1)
            + tf.reduce_sum(tf.nn.log_softmax(aa_logits) * aa_soft, axis=-1)
        )
        action_disc = tf.concat([pos_soft, aa_soft], axis=-1)  # [B, L_max + 20]
        return action_disc, log_pi

    # ---------------- 完整采样 ----------------
    def sample(self, z, env, z_mask, mutation_allowed=None):
        """返回混合动作与总 log_pi。

        mutation_allowed: [B] bool/float。False 时离散熵项不计入 log_pi
        （分层动作：该步不允许突变）。
        """
        action_cont, log_pi_cont = self.sample_cont(z, env)
        action_disc, log_pi_disc = self.sample_disc(z, z_mask)
        if mutation_allowed is not None:
            m = tf.cast(mutation_allowed, log_pi_disc.dtype)
            log_pi = log_pi_cont + m * log_pi_disc
        else:
            log_pi = log_pi_cont + log_pi_disc
        return action_cont, action_disc, log_pi, log_pi_cont, log_pi_disc

    def sample_deterministic(self, z, env, z_mask):
        mean, _ = self.cont_dist(z, env)
        pos_logits, aa_logits = self.disc_logits(z, z_mask)
        pos_hard = tf.one_hot(tf.argmax(pos_logits, axis=-1), self.disc_pos_dim)
        aa_hard = tf.one_hot(tf.argmax(aa_logits, axis=-1), self.aa_dim)
        return mean, tf.concat([pos_hard, aa_hard], axis=-1)


@register_keras_serializable(package="spice_rl")
class TwinCritic(tf.keras.Model):
    """双 Q 网络（取 min 防高估）。

    输入：z + M + u_hist + 连续动作 + 离散动作 + mutation_mask。
    """

    def __init__(
        self,
        cfg: SACConfig,
        cont_dim: int,
        z_dim: int,
        metric_dim: int = 5,
        u_window: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cont_dim = cont_dim
        self.z_dim = z_dim
        self.metric_dim = metric_dim
        self.u_window = u_window
        self.disc_dim = cfg.discrete_position_dim + cfg.aa_dim
        self.mask_dim = 1 if cfg.track_mutation_mask else 0
        self.q1 = self._q_net("q1")
        self.q2 = self._q_net("q2")

    def _q_net(self, name):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.cfg.hidden_dim, activation="relu", name=f"{name}_d0"),
                tf.keras.layers.Dense(self.cfg.hidden_dim, activation="relu", name=f"{name}_d1"),
                tf.keras.layers.Dense(1, name=f"{name}_out"),
            ],
            name=name,
        )

    def _feats(self, z, M, u_hist, action_cont, action_disc, mutation_mask):
        parts = [z, M, u_hist, action_cont, action_disc]
        if mutation_mask is not None:
            parts.append(mutation_mask)
        return tf.concat(parts, axis=-1)

    def call(self, z, M, u_hist, action_cont, action_disc, mutation_mask=None):
        x = self._feats(z, M, u_hist, action_cont, action_disc, mutation_mask)
        return tf.squeeze(self.q1(x), -1), tf.squeeze(self.q2(x), -1)

    def min_q(self, z, M, u_hist, action_cont, action_disc, mutation_mask=None):
        q1, q2 = self.call(z, M, u_hist, action_cont, action_disc, mutation_mask)
        return tf.minimum(q1, q2)
