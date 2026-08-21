from __future__ import annotations

import numpy as np
import tensorflow as tf

from spice_rl.config import SACConfig
from spice_rl.keras_utils import register_keras_serializable

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def gumbel_softmax(logits, tau: float, hard: bool = False):
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
    log_prob = tf.reduce_sum(log_prob, axis=-1)
    # 2026-08-17 Prevent -inf/inf/NaN propagation from extreme continuous values
    return tf.where(tf.math.is_finite(log_prob), log_prob, tf.zeros_like(log_prob))


@register_keras_serializable(package="spice_rl")
class SacActor(tf.keras.Model):

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
        self.cont_dim = cont_dim               
        self.z_dim = z_dim
        self.env_dim = env_dim
        self.disc_pos_dim = cfg.discrete_position_dim  
        self.aa_dim = cfg.aa_dim

        self.cont_trunk = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
                tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            ],
            name="actor_cont_trunk",
        )
        self.mean_head = tf.keras.layers.Dense(cont_dim, name="actor_mean")
        self.log_std_head = tf.keras.layers.Dense(cont_dim, name="actor_log_std")

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

    def cont_dist(self, z, env):
        z = tf.clip_by_value(z, -1e3, 1e3)  # 2026-08-19 Tertiary guard: prevents amplification of extreme latent z values
        h = self.cont_trunk(tf.concat([z, env], axis=-1))
        mean = self.mean_head(h)
        log_std = tf.clip_by_value(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample_cont(self, z, env):
        mean, log_std = self.cont_dist(z, env)
        action = mean + tf.exp(log_std) * tf.random.normal(tf.shape(mean))
        return action, _gaussian_log_prob(action, mean, log_std)

    def disc_logits(self, z, z_mask):
        z = tf.clip_by_value(z, -1e3, 1e3)  # 2026-08-19 Tertiary guard: prevents amplification of extreme latent z values
        h = self.disc_trunk(z)
        pos_logits = self.pos_logits_head(h)              
        mask = tf.cast(z_mask, pos_logits.dtype)
        pos_logits = tf.where(mask > 0.5, pos_logits, -1e9)
        aa_logits = self.aa_logits_head(h)                
        return pos_logits, aa_logits

    def sample_disc(self, z, z_mask):
        pos_logits, aa_logits = self.disc_logits(z, z_mask)
        pos_soft = gumbel_softmax(pos_logits, self.cfg.gumbel_tau)
        aa_soft = gumbel_softmax(aa_logits, self.cfg.gumbel_tau)
        
        # 2026-08-17 Gumbel-Softmax NaN Trap Fix:
        # Masked logits have -1e9, so log_softmax returns -inf. Under IEEE 754 math,
        # multiplying -inf by 0.0 (from pos_soft) results in NaN.
        # Washing log_softmax of any non-finite values to 0.0 eliminates this trap.
        log_pos = tf.nn.log_softmax(pos_logits)
        log_pos = tf.where(tf.math.is_finite(log_pos), log_pos, tf.zeros_like(log_pos))
        
        log_aa = tf.nn.log_softmax(aa_logits)
        log_aa = tf.where(tf.math.is_finite(log_aa), log_aa, tf.zeros_like(log_aa))

        log_pi = (
            tf.reduce_sum(log_pos * pos_soft, axis=-1)
            + tf.reduce_sum(log_aa * aa_soft, axis=-1)
        )
        action_disc = tf.concat([pos_soft, aa_soft], axis=-1)  
        return action_disc, log_pi

    def sample(self, z, env, z_mask, mutation_allowed=None):
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
        u_ref = getattr(self.cfg, "u_ref", 1e5)
        if u_ref and u_ref != 1.0:
            u_hist = u_hist / tf.cast(u_ref, u_hist.dtype)
        
        # 2026-08-17 Robust clipping of physical indicators to prevent outlier-driven gradient spikes
        M = tf.clip_by_value(M, -5.0, 5.0)
        # 2026-08-19 Secondary guard (born-NaN fix): direct clipping of extreme z and action_cont values
        # protects the critic's forward pass against unnormalized histories (e.g. legacy replay buffers with unclipped z) or extreme actor outputs.
        z = tf.clip_by_value(z, -1e3, 1e3)
        action_cont = tf.clip_by_value(action_cont, -1e3, 1e3)

        parts = [z, M, u_hist, action_cont, action_disc]
        if mutation_mask is not None:
            m = tf.cast(mutation_mask, action_disc.dtype)
            # Force [batch, 1] shape on mask to guarantee reliable broadcasting and eliminate gradient noise
            if len(m.shape) == 1:
                m = tf.expand_dims(m, axis=-1)
            elif len(m.shape) == 0:
                m = tf.reshape(m, [1, 1])
            parts[4] = action_disc * m
            
            m_append = mutation_mask
            if len(m_append.shape) == 1:
                m_append = tf.expand_dims(m_append, axis=-1)
            elif len(m_append.shape) == 0:
                m_append = tf.reshape(m_append, [1, 1])
            parts.append(m_append)
        return tf.concat(parts, axis=-1)

    def call(self, z, M, u_hist, action_cont, action_disc, mutation_mask=None):
        x = self._feats(z, M, u_hist, action_cont, action_disc, mutation_mask)
        return tf.squeeze(self.q1(x), -1), tf.squeeze(self.q2(x), -1)

    def min_q(self, z, M, u_hist, action_cont, action_disc, mutation_mask=None):
        q1, q2 = self.call(z, M, u_hist, action_cont, action_disc, mutation_mask)
        return tf.minimum(q1, q2)


def install_layer_trace(root, tag: str):
    """Recursively wraps the `.call` method of all sub-layers in root (e.g. SacActor, TwinCritic, or other Keras Models),
    logging shape, finiteness, non-finite counts, and absolute maximum of finite values for each layer's output.
    Used for layer-by-layer troubleshooting of NaN occurrences (e.g. during 6QQE born-NaN investigation) without modifying network logic.
    Since tf.print is a graph-compatible op, this works seamlessly in both eager and @tf.function graph modes."""
    seen = set()

    def _trace(layer, full_name):
        orig = layer.call

        def traced(*args, **kwargs):
            out = orig(*args, **kwargs)
            if isinstance(out, tf.Tensor) and out.dtype.is_floating:
                _fin = tf.reduce_all(tf.math.is_finite(out))
                _nbad = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(out)), tf.int64))
                _amax = tf.reduce_max(
                    tf.where(tf.math.is_finite(out), tf.abs(out), tf.zeros_like(out)))
                tf.print(f"[TRACE:{full_name}] out", tf.shape(out),
                         "finite", _fin, "nbad", _nbad, "absmax", _amax)
            return out

        layer.call = traced

    def _walk(mod, path):
        for l in getattr(mod, "layers", []):
            if id(l) in seen:
                continue
            seen.add(id(l))
            _full = f"{path}/{l.name}"
            if isinstance(l, tf.keras.layers.Layer):
                _trace(l, _full)
            if getattr(l, "layers", None):
                _walk(l, _full)

    _walk(root, tag)
