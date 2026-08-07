"""SPICE-SAC 训练器（微观环）——实现文档全部五项定制。

1. 异步收集与批量更新 —— 收集满 `update_every_steps` 触发一次 batch 更新
2. 混合动作解耦输出头 —— Actor 双头（连续高斯 + 离散 Gumbel-Softmax），
   动作拼接平坦向量输入 Critic
3. 特权信息 —— Actor 仅 z+env；Critic 用 z+M+u_hist+action(+mask)
4. 自适应熵系数 —— 目标熵 = -(cont_dim + 2) * factor（离散为 2 个 categorical）
5. 分层动作时序 —— 每步采样完整动作，mutation_mask 决定离散头是否生效；
   buffer 存原始输出+掩码，Critic 依掩码学习
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from spice_rl.config import SACConfig
from spice_rl.sac.buffer import ReplayBuffer
from spice_rl.sac.networks import SacActor, TwinCritic

# 离散动作的熵维度 = 位置 categorical(1) + 氨基酸 categorical(1)
DISC_ENTROPY_DIM = 2


class SACTrainer:
    def __init__(
        self,
        cfg: SACConfig,
        z_dim: int,
        cont_dim: int,
        env_dim: int = 3,
        u_window: int = 10,
    ):
        self.cfg = cfg
        self.z_dim = z_dim
        self.cont_dim = cont_dim                      # force[16] + env_offset[2]
        self.disc_dim = cfg.discrete_position_dim + cfg.aa_dim
        self.env_dim = env_dim
        self.u_window = u_window

        self.actor = SacActor(cfg, cont_dim, z_dim, env_dim)
        self.critic = TwinCritic(
            cfg, cont_dim, z_dim, metric_dim=5, u_window=u_window
        )
        self.critic_target = TwinCritic(
            cfg, cont_dim, z_dim, metric_dim=5, u_window=u_window
        )
        self.critic_target.set_weights(self.critic.get_weights())

        self.log_alpha = tf.Variable(0.0, dtype=tf.float32, name="log_alpha")
        self.target_entropy = -(float(cont_dim) + DISC_ENTROPY_DIM) * cfg.target_entropy_factor

        self.opt_actor = tf.keras.optimizers.Adam(cfg.lr_actor)
        self.opt_critic = tf.keras.optimizers.Adam(cfg.lr_critic)
        self.opt_alpha = tf.keras.optimizers.Adam(cfg.lr_alpha)

        self.buffer = ReplayBuffer(
            capacity=cfg.buffer_capacity,
            z_dim=z_dim,
            cont_dim=cont_dim,
            disc_dim=self.disc_dim,
            pos_dim=cfg.discrete_position_dim,
            env_dim=env_dim,
            metric_dim=5,
            u_window=u_window,
        )
        self._steps_since_update = 0

    # ---------------- 收集接口（定制 1：异步） ----------------
    def collect(self, transition: dict) -> bool:
        """收集一条经验；返回 True 表示达到批量更新阈值（应触发 update）。"""
        self.buffer.add(transition)
        self._steps_since_update += 1
        if self._steps_since_update >= self.cfg.update_every_steps:
            self._steps_since_update = 0
            return True
        return False

    # ---------------- 动作采样 ----------------
    def act(self, z, env, z_mask, deterministic: bool = False):
        """返回 (action_cont, action_disc)。输入均为 numpy [D]/[3]/[L_max]。"""
        z = tf.constant(np.asarray(z, np.float32)[None])
        env = tf.constant(np.asarray(env, np.float32)[None])
        z_mask = tf.constant(np.asarray(z_mask, np.float32)[None])
        if deterministic:
            a_cont, a_disc = self.actor.sample_deterministic(z, env, z_mask)
        else:
            a_cont, a_disc, _, _, _ = self.actor.sample(z, env, z_mask)
        return a_cont.numpy()[0], a_disc.numpy()[0]

    # ---------------- 更新 ----------------
    def update(self, z_mask) -> dict:
        """从 buffer 采样 batch 更新 critic/actor/alpha。

        z_mask: [L_max] 当前（固定）序列掩码——本批经验同属当前序列时用它；
               跨序列混合时从 buffer 采样得到（见 _update_tf）。
        """
        if not self.buffer.can_sample(self.cfg.batch_size):
            return {"critic_loss": float("nan"), "actor_loss": float("nan"),
                    "alpha_loss": float("nan"), "alpha": self.alpha()}
        b = self.buffer.sample(self.cfg.batch_size)
        b = {k: tf.constant(v) for k, v in b.items()}
        # buffer 已存 per-transition z_mask，用它（支持跨序列共享）
        losses = self._update_tf(b)
        self.soft_update()
        return losses

    @tf.function
    def _update_tf(self, b: dict) -> dict:
        gamma = self.cfg.gamma

        # ---------- Critic ----------
        with tf.GradientTape() as tape:
            q1, q2 = self.critic(
                b["z"], b["M"], b["u_hist"],
                b["action_cont"], b["action_disc"], b["mutation_mask"],
            )
            alpha = tf.exp(self.log_alpha)
            # 下一状态动作：离散熵项按掩码（分层动作）
            a_next_cont, a_next_disc, log_pi_next, _, _ = self.actor.sample(
                b["next_z"], b["next_env"], b["z_mask"], b["mutation_mask"]
            )
            q_tgt = self.critic_target.min_q(
                b["next_z"], b["next_M"], b["next_u_hist"],
                a_next_cont, a_next_disc, b["mutation_mask"],
            )
            y = b["reward"] + gamma * (1.0 - b["done"]) * (q_tgt - alpha * log_pi_next)
            critic_loss = tf.reduce_mean(tf.square(q1 - y) + tf.square(q2 - y))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
        self.opt_critic.apply_gradients(zip(grads, self.critic.trainable_variables))

        # ---------- Actor ----------
        with tf.GradientTape() as tape:
            a_cont, a_disc, log_pi, _, _ = self.actor.sample(
                b["z"], b["env"], b["z_mask"], b["mutation_mask"]
            )
            q = self.critic.min_q(
                b["z"], b["M"], b["u_hist"], a_cont, a_disc, b["mutation_mask"]
            )
            actor_loss = tf.reduce_mean(alpha * log_pi - q)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
        self.opt_actor.apply_gradients(zip(grads, self.actor.trainable_variables))

        # ---------- Alpha（自适应熵，定制 4） ----------
        with tf.GradientTape() as tape:
            _, _, log_pi_det, _, _ = self.actor.sample(
                b["z"], b["env"], b["z_mask"], b["mutation_mask"]
            )
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * (tf.stop_gradient(log_pi_det) + self.target_entropy)
            )
        grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.opt_alpha.apply_gradients(zip(grads, [self.log_alpha]))

        return {
            "critic_loss": tf.reduce_mean(critic_loss),
            "actor_loss": tf.reduce_mean(actor_loss),
            "alpha_loss": tf.reduce_mean(alpha_loss),
            "alpha": alpha,
        }

    def soft_update(self):
        tau = self.cfg.tau
        for t, s in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            t.assign(tau * s + (1.0 - tau) * t)

    def alpha(self) -> float:
        return float(np.exp(float(self.log_alpha.numpy())))
