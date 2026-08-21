from __future__ import annotations

import numpy as np
import tensorflow as tf

from spice_rl.config import SACConfig, set_eager_mode
from spice_rl.sac.buffer import ReplayBuffer
from spice_rl.sac.networks import SacActor, TwinCritic

DISC_ENTROPY_DIM = 2


def _trace_tensor(t, name: str):
    """Layer/Tensor diagnostic printer under @tf.function: logs shape, finiteness, bad count, absmax, and mean."""
    tf.print(f"[TRACE:{name}]", tf.shape(t),
             "finite", tf.reduce_all(tf.math.is_finite(t)),
             "nbad", tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(t)), tf.int64)),
             "absmax", tf.reduce_max(tf.where(tf.math.is_finite(t), tf.abs(t), tf.zeros_like(t))),
             "mean", tf.reduce_mean(tf.where(tf.math.is_finite(t), t, tf.zeros_like(t))))


class SACTrainer:
    def __init__(
        self,
        cfg: SACConfig,
        z_dim: int,
        cont_dim: int,
        env_dim: int = 3,
        u_window: int = 10,
        trace_layers: bool | None = None,
    ):
        self.cfg = cfg
        self.z_dim = z_dim
        self.trace_layers = cfg.trace_layers if trace_layers is None else trace_layers
        self.cont_dim = cont_dim                      
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

        if self.trace_layers:
            from spice_rl.sac.networks import install_layer_trace
            install_layer_trace(self.actor, "actor")
            install_layer_trace(self.critic, "critic")
            install_layer_trace(self.critic_target, "critic_target")

        # 2026-08-19 Eager Mode Fallback Switch: if containerized TF graph mode exhibits numerical instability, force eager execution (including train_post and encode_z).
        if getattr(cfg, "eager_update", False):
            set_eager_mode(True)

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
        self.last_losses = None   

    def ensure_built(self):
        z = tf.zeros([1, self.z_dim], tf.float32)
        env = tf.zeros([1, self.env_dim], tf.float32)
        z_mask = tf.ones([1, self.cfg.discrete_position_dim], tf.float32)   
        M = tf.zeros([1, 5], tf.float32)
        u_hist = tf.zeros([1, self.u_window], tf.float32)
        a_cont = tf.zeros([1, self.cont_dim], tf.float32)
        a_disc = tf.zeros([1, self.disc_dim], tf.float32)
        mm = tf.zeros([1, 1], tf.float32)
        self.actor.sample(z, env, z_mask)
        self.critic(z, M, u_hist, a_cont, a_disc, mutation_mask=mm)
        self.critic_target(z, M, u_hist, a_cont, a_disc, mutation_mask=mm)
        for _m in (self.actor, self.critic, self.critic_target):
            _m.built = True

    def save(self, ckpt_dir: str, tag: str = "sac"):
        import os
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ensure_built()
        self.actor.save_weights(os.path.join(ckpt_dir, f"{tag}_actor.weights.h5"))
        self.critic.save_weights(os.path.join(ckpt_dir, f"{tag}_critic.weights.h5"))
        self.critic_target.save_weights(os.path.join(ckpt_dir, f"{tag}_critic_target.weights.h5"))
        np.save(os.path.join(ckpt_dir, f"{tag}_log_alpha.npy"), self.log_alpha.numpy())

    def load(self, ckpt_dir: str, tag: str = "sac"):
        import os
        self.actor.load_weights(os.path.join(ckpt_dir, f"{tag}_actor.weights.h5"))
        self.critic.load_weights(os.path.join(ckpt_dir, f"{tag}_critic.weights.h5"))
        self.critic_target.load_weights(os.path.join(ckpt_dir, f"{tag}_critic_target.weights.h5"))
        self.log_alpha.assign(np.load(os.path.join(ckpt_dir, f"{tag}_log_alpha.npy")))

    def collect(self, transition: dict) -> bool:
        self.buffer.add(transition)
        self._steps_since_update += 1
        if self._steps_since_update >= self.cfg.update_every_steps:
            self._steps_since_update = 0
            return True
        return False

    def act(self, z, env, z_mask, deterministic: bool = False):
        if self.trace_layers:
            _za = np.asarray(z, np.float32)
            _nb = int(np.sum(~np.isfinite(_za)))
            _am = float(np.nanmax(np.abs(_za))) if np.any(np.isfinite(_za)) else float("nan")
            print(f"[TRACE:act] z in  shape={_za.shape} finite={bool(np.all(np.isfinite(_za)))} "
                  f"nbad={_nb} absmax={_am:.4f}")
        z = tf.constant(np.asarray(z, np.float32)[None])
        env = tf.constant(np.asarray(env, np.float32)[None])
        z_mask = tf.constant(np.asarray(z_mask, np.float32)[None])
        if deterministic:
            a_cont, a_disc = self.actor.sample_deterministic(z, env, z_mask)
        else:
            a_cont, a_disc, _, _, _ = self.actor.sample(z, env, z_mask)
        return a_cont.numpy()[0], a_disc.numpy()[0]

    def update(self, z_mask) -> dict:
        if not self.buffer.can_sample(self.cfg.batch_size):
            losses = {"critic_loss": float("nan"), "actor_loss": float("nan"),
                      "alpha_loss": float("nan"), "alpha": self.alpha()}
            self.last_losses = losses
            return losses
        b = self.buffer.sample(self.cfg.batch_size)
        b = {k: tf.constant(v) for k, v in b.items()}
        if getattr(self.cfg, "eager_update", False):
            losses = self._update_eager(b)   # 2026-08-19 Containerized TF graph mode has issues -> bypass via eager mode
        else:
            losses = self._update_tf(b)
        self.soft_update()
        self.last_losses = {k: float(v) for k, v in losses.items()}
        return losses

    def _update_impl(self, b: dict) -> dict:
        """Core SAC update block (excluding @tf.function). Shared between the graph wrapper (_update_tf) 
        and the eager update wrapper (_update_eager); executes via direct eager call on 2026-08-19 when containerized TF graph mode is unstable."""
        gamma = self.cfg.gamma

        # 2026-08-15 Root-cause mitigation: input-level finite value (isfinite) sanitization. 
        # Empirical observations on HPC clusters reveal that latent embeddings (z) and replay buffer data can occasionally 
        # contain NaNs under specific environments (unreproducible using local random inputs). 
        # Once a NaN enters the pipeline, it propagates globally, resulting in NaNs across critic, actor, and alpha losses, 
        # which stalls entropy adjustment. Setting all non-finite values in input tensors to 0.0 guarantees bounded gradients, 
        # ensuring robust training initialization.
        b = {k: tf.where(tf.math.is_finite(v), v, tf.zeros_like(v)) for k, v in b.items()}

        mutation_mask = tf.squeeze(b["mutation_mask"], axis=-1)

        if self.trace_layers:
            for _k, _v in b.items():
                if _v.dtype.is_floating:
                    _trace_tensor(_v, f"update/input/{_k}")

        # 2026-08-15 Root-cause probe: aggregates and reports the count of cleaned NaN/Inf values per update to assist in debugging.
        # Uses tf.print under @tf.function graph mode (conditioned to trigger on the first detected anomaly).
        _n_nonfinite = tf.reduce_sum([
            tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(v)), tf.int64))
            for v in b.values() if v.dtype != tf.int32
        ])
        tf.cond(
            _n_nonfinite > 0,
            lambda: tf.print("[probe] _update_tf sanitized", _n_nonfinite,
                             "non-finite values (batch_size=", tf.shape(b['z'])[0], ")"),
            lambda: tf.no_op(),
        )

        with tf.GradientTape() as tape:
            q1, q2 = self.critic(
                b["z"], b["M"], b["u_hist"],
                b["action_cont"], b["action_disc"], b["mutation_mask"],
            )
            alpha = tf.exp(self.log_alpha)
            a_next_cont, a_next_disc, log_pi_next, _, _ = self.actor.sample(
                b["next_z"], b["next_env"], b["z_mask"], mutation_mask
            )
            q_tgt = self.critic_target.min_q(
                b["next_z"], b["next_M"], b["next_u_hist"],
                a_next_cont, a_next_disc, b["mutation_mask"],
            )
            reward = tf.squeeze(b["reward"], axis=-1)
            done = tf.squeeze(b["done"], axis=-1)
            # 2026-08-15 Critic infinity protection: during the initial updates, the randomly initialized critic_target 
            # can produce exploded Q-values (q_tgt), yielding non-finite temporal difference targets (y) that poison critic loss with infinity.
            # We clip the TD target to a safe numerical range.
            y = reward + gamma * (1.0 - done) * (q_tgt - alpha * log_pi_next)
            if self.trace_layers:
                _trace_tensor(q1, "update/q1_pre")
                _trace_tensor(q2, "update/q2_pre")
                _trace_tensor(q_tgt, "update/q_tgt")
                _trace_tensor(log_pi_next, "update/log_pi_next")
                _trace_tensor(alpha, "update/alpha")
                _trace_tensor(y, "update/y_pre")
            y = tf.clip_by_value(y, -1e6, 1e6)
            y = tf.where(tf.math.is_finite(y), y, tf.zeros_like(y))  # 2026-08-18 Ultimate fallback: prevents NaNs from leaking into loss calculation
            if self.trace_layers:
                _trace_tensor(y, "update/y_post")
            q1 = tf.clip_by_value(q1, -1e6, 1e6)  # 2026-08-18 Robustness reinforcement: unbounded critic outputs cause loss divergence
            q2 = tf.clip_by_value(q2, -1e6, 1e6)
            critic_loss = tf.reduce_mean(tf.square(q1 - y) + tf.square(q2 - y))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
        self.opt_critic.apply_gradients(zip(grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            a_cont, a_disc, log_pi, _, _ = self.actor.sample(
                b["z"], b["env"], b["z_mask"], mutation_mask
            )
            q = self.critic.min_q(
                b["z"], b["M"], b["u_hist"], a_cont, a_disc, b["mutation_mask"]
            )
            q = tf.clip_by_value(q, -1e6, 1e6)  # 2026-08-15 Critic infinity protection
            if self.trace_layers:
                _trace_tensor(log_pi, "update/log_pi_actor")
            log_pi = tf.clip_by_value(log_pi, -100.0, 0.0)  # 2026-08-18 Robustness reinforcement: bounded entropy terms
            actor_loss = tf.reduce_mean(alpha * log_pi - q)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, self.cfg.grad_clip)
        self.opt_actor.apply_gradients(zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            _, _, log_pi_det, _, _ = self.actor.sample(
                b["z"], b["env"], b["z_mask"], mutation_mask
            )
            log_pi_det = tf.clip_by_value(log_pi_det, -100.0, 0.0)  # 2026-08-18 Robustness reinforcement
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * (tf.stop_gradient(log_pi_det) + self.target_entropy)
            )
        grads = tape.gradient(alpha_loss, [self.log_alpha])
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        self.opt_alpha.apply_gradients(zip(grads, [self.log_alpha]))

        return {
            "critic_loss": tf.reduce_mean(critic_loss),
            "actor_loss": tf.reduce_mean(actor_loss),
            "alpha_loss": tf.reduce_mean(alpha_loss),
            "alpha": alpha,
        }

    @tf.function
    def _update_tf(self, b: dict) -> dict:
        return self._update_impl(b)

    def _update_eager(self, b: dict) -> dict:
        return self._update_impl(b)

    def soft_update(self):
        tau = self.cfg.tau
        for t, s in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            t.assign(tau * s + (1.0 - tau) * t)

    def alpha(self) -> float:
        return float(np.exp(float(self.log_alpha.numpy())))
