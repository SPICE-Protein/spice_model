"""Frame Structure Head: predicts residue-wise 3D rotation frames + enforces a rigid virtual bond length of 3.8 Å,
vectorially integrating them into physical Cα coordinates.

Why we replaced the legacy Head A (direct Cartesian coordinate MLP regression):
- The legacy head_a was supervised only by pairwise_coord_loss (pairwise distance matrix matching), which lacks 
  direct constraints on bond lengths, bond angles, or local chirality. This led to serious geometric self-intersections 
  (steric clashes yielding equilibration potential energy up to +4e15 kcal/mol).
- This structure head mathematically guarantees adjacent Cα distances = bond_length (default 3.8 Å, the standard virtual 
  Cα-Cα bond distance) by construction. It predicts SO(3) rotation frames (using 6D representation) per residue to rotate 
  local relative directions into global axes.
  The global displacements g_i = R_i @ (bond * dir_i), and the coordinates are resolved via cumsum(g) - g (fully vectorized, 
  avoiding slow tf.scan/sequential operations).
  By architecture, bond lengths are structurally correct, coordinates remain fully differentiable, and training is 
  highly efficient and compatible with MirroredStrategy multi-GPU distribution. Only requires Cα coordinates for training.

Output out["coords"] has shape [B, L, 3], matching the interface of the legacy head_a exactly to prevent downstream disruptions (eval/RL).
"""
from __future__ import annotations

import tensorflow as tf

from spice_pre.keras_utils import register_keras_serializable


def rot_from_6d(v: tf.Tensor) -> tf.Tensor:
    """6D rotation representation (Zhou et al. 2019) -> SO(3) matrices [..., 3, 3].

    Performs Gram-Schmidt orthonormalization on two predicted vectors, taking their cross product for the third axis.
    Adds a small epsilon to prevent NaNs in case of zero vectors (padding).
    """
    a1 = v[..., 0:3]
    a2 = v[..., 3:6]
    e1 = tf.math.l2_normalize(a1, axis=-1, epsilon=1e-6)
    u2 = a2 - tf.reduce_sum(e1 * a2, axis=-1, keepdims=True) * e1
    e2 = tf.math.l2_normalize(u2, axis=-1, epsilon=1e-6)
    e3 = tf.linalg.cross(e1, e2)
    return tf.stack([e1, e2, e3], axis=-1)


@register_keras_serializable(package="spice_pre")
class RecycleStructureModule(tf.keras.layers.Layer):
    """Distance-aware SE(3) recycling refinement over Cα positions.

    Mitigates cumulative errors in frame head integration (cumsum): integration-based reconstruction is a local-to-global mapping,
    where errors accumulate along the sequence.
    This module uses the 3D distances of the current predicted structure to create attention biases (residues prioritize 
    attention to their 3D physical neighbors, supplying missing global geometric constraints to latent z). It aggregates 
    sequence features and local-frame neighbor vectors (SE(3)-equivariant) to predict updates for rotation frames (6D) and local translations 
    per residue, applying them back to coordinates. Iterated for `steps` rounds, allowing the model to perform direct geometric correction 
    based on intermediate structures, breaking strict error propagation (lightweight implementation of AlphaFold2 recycling concept).

    SE(3)-equivariance: relative neighbor vectors and translations are represented in each residue's local frame, ensuring 
    coordinates transform correctly under global rotation and translation.
    Fully vectorized (via tf.einsum/tf.matmul, without tf.scan) -> compatible with MirroredStrategy multi-GPU scaling.
    """

    def __init__(self, refine_dim: int = 64, num_heads: int = 4, steps: int = 2,
                 rbf_k: int = 16, rbf_max: float = 40.0, dropout: float = 0.0,
                 trans_scale: float = 0.2, name: str = "recycle_module", **kwargs):
        super().__init__(name=name, **kwargs)
        self.refine_dim = int(refine_dim)
        self.num_heads = int(num_heads)
        self.steps = int(steps)
        self.rbf_k = int(rbf_k)
        self.rbf_max = float(rbf_max)
        self.dropout_rate = float(dropout)
        self.trans_scale = float(trans_scale)
        self.head_dim = max(int(refine_dim) // int(num_heads), 1)

        self.q = tf.keras.layers.Dense(self.refine_dim, name="recycle_q")
        self.k = tf.keras.layers.Dense(self.refine_dim, name="recycle_k")
        self.v = tf.keras.layers.Dense(self.refine_dim, name="recycle_v")
        self.dist_mlp = tf.keras.layers.Dense(self.num_heads, name="recycle_dist")
        self.upd_proj = tf.keras.layers.Dense(self.refine_dim, activation="gelu",
                                              name="recycle_upd_mlp")
        self.upd_out = tf.keras.layers.Dense(9, name="recycle_upd")   # rot6 + t3
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def _rbf(self, d: tf.Tensor) -> tf.Tensor:
        c = tf.linspace(0.0, self.rbf_max, self.rbf_k)              # [K]
        sigma = self.rbf_max / max(self.rbf_k - 1, 1)
        return tf.exp(-((d[..., None] - c) ** 2) / (2.0 * sigma ** 2))  # [B,L,L,K]

    def build(self, input_shape):
        # Compatible with both single shapes and lists of shapes from Keras; sub-layers are lazily built during the first call.
        # Here we just mark self.built = True to suppress the "unbuilt state" warning; sub-layers manage their own builds.
        self.built = True

    def call(self, z, coords, R, mask, training: bool = False) -> tf.Tensor:
        z = tf.cast(z, tf.float32)
        coords = tf.cast(coords, tf.float32)
        R = tf.cast(R, tf.float32)
        m = tf.cast(mask, tf.float32)                              # [B,L]
        B, L = tf.shape(coords)[0], tf.shape(coords)[1]
        eye3 = tf.eye(3, dtype=tf.float32)
        H, Dh = self.num_heads, self.head_dim

        for _ in range(self.steps):
            # 1) 3D pairwise distances -> Radial Basis Functions (RBF) -> Attention Bias (3D spatial prior)
            d2 = tf.reduce_sum((coords[:, :, None, :] - coords[:, None, :, :]) ** 2, axis=-1)
            d = tf.sqrt(tf.maximum(d2, 1e-12))                     # [B,L,L]
            dist_bias = self.dist_mlp(self._rbf(d))                # [B,L,L,H]

            # 2) Sequence-based self-attention (bias modulated by pairwise distances -> enforces strong localized coupling)
            q = tf.reshape(self.q(z), (B, L, H, Dh))
            k = tf.reshape(self.k(z), (B, L, H, Dh))
            v = tf.reshape(self.v(z), (B, L, H, Dh))
            logits = tf.einsum("blhd,bmhd->bhlm", q, k) / tf.sqrt(float(Dh))
            logits = logits + tf.transpose(dist_bias, (0, 3, 1, 2))   # [B,H,L,L]
            valid = m[:, None, :, None] * m[:, None, None, :]         # [B,1,L,L]
            logits = tf.where(valid > 0.5, logits, -1e9)
            attn = tf.nn.softmax(logits, axis=-1)                     # [B,H,L,L]

            # 3) Sequence feature aggregation + local-frame neighbor aggregation (SE(3)-equivariant)
            ctx_seq = tf.reshape(tf.einsum("bhlm,bmhd->blhd", attn, v),
                                 (B, L, self.refine_dim))
            rel = coords[:, :, None, :] - coords[:, None, :, :]       # [B,L,L,3]
            rel_local = tf.einsum("bijc,bica->bija", rel, R)          # R[i]ᵀ @ rel
            point = tf.reshape(tf.einsum("bhij,bija->bhia", attn, rel_local),
                               (B, L, H * 3))

            # 4) Predict and apply frame rotation and translation updates (padding residues remain stationary)
            upd_in = self.ln(tf.concat([ctx_seq, point, z], axis=-1))  # LayerNorm for numerical stabilization
            upd = self.upd_out(self.dropout(self.upd_proj(upd_in), training=training))
            R_up = rot_from_6d(upd[..., :6])                          # [B,L,3,3]
            R_new = tf.matmul(R_up, R)                                # Composition
            # 🔴 Translations must be small (tanh restricts to [-1, 1], scaled by trans_scale): acts as near-identity initially to prevent coordinates from dispersing
            t_loc = tf.tanh(upd[..., 6:9]) * self.trans_scale
            t_abs = tf.einsum("birc,bic->bir", R_new, t_loc)          # R_new @ t_loc
            coords = coords + t_abs * m[..., None]                    # Displacement (mask ensures padding remains frozen)
            R = tf.where(m[..., None, None] > 0.5, R_new, eye3)
        return coords

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            refine_dim=self.refine_dim,
            num_heads=self.num_heads,
            steps=self.steps,
            rbf_k=self.rbf_k,
            rbf_max=self.rbf_max,
            dropout=self.dropout_rate,
            trans_scale=self.trans_scale,
        )
        return cfg


@register_keras_serializable(package="spice_pre")
class FrameStructureHead(tf.keras.layers.Layer):
    """Synthesizes Cα coordinates [B, L, 3] from residue-wise relative rotation frames and direction vectors.

    When `recycle_steps > 0`, runs RecyclingStructureModule to refine the coordinates obtained via cumsum integration:
    Leverages SE(3)-equivalence + distance-aware self-attention to perform iterative corrections, mitigating 
    cumulative errors along long chains (implemented 2026-08-13).
    """

    def __init__(self, embed_dim: int, bond_length: float = 3.8,
                 dropout: float = 0.0, recycle_steps: int = 0,
                 refine_dim: int = 64, refine_heads: int = 4,
                 refine_dropout: float = 0.0, name: str = "head_a", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed_dim = int(embed_dim)
        self.bond_length = float(bond_length)
        self.dropout_rate = float(dropout)
        self.recycle_steps = int(recycle_steps)
        self.refine_dim = int(refine_dim)
        self.refine_heads = int(refine_heads)
        self.refine_dropout = float(refine_dropout)

        self.dir_proj = tf.keras.layers.Dense(embed_dim, activation="gelu",
                                              name="frame_dir_mlp")
        self.dir_out = tf.keras.layers.Dense(3, name="frame_dir")
        self.rot_proj = tf.keras.layers.Dense(embed_dim, activation="gelu",
                                              name="frame_rot_mlp")
        self.rot_out = tf.keras.layers.Dense(6, name="frame_rot6")
        self.dropout = tf.keras.layers.Dropout(dropout)
        if self.recycle_steps > 0:
            self.recycle = RecycleStructureModule(
                refine_dim=self.refine_dim, num_heads=self.refine_heads,
                steps=self.recycle_steps, dropout=self.refine_dropout,
                name="recycle_refine")
        else:
            self.recycle = None

    def call(self, z, mask, training: bool = False) -> tf.Tensor:
        z = tf.cast(z, tf.float32)
        m = tf.cast(mask, tf.float32)[..., None]          # [B, L, 1]

        d = self.dir_out(self.dropout(self.dir_proj(z), training=training))
        d = tf.cast(d, tf.float32)
        d = tf.math.l2_normalize(d, axis=-1, epsilon=1e-6)
        # Directions at padding positions are fixed to +x (ensures bounded synthesis; coordinates are masked in downstream loss)
        d = tf.where(m > 0.5, d, tf.constant([1.0, 0.0, 0.0], tf.float32))

        r6 = self.rot_out(self.dropout(self.rot_proj(z), training=training))
        r6 = tf.cast(r6, tf.float32)
        r = rot_from_6d(r6)                               # [B, L, 3, 3]
        eye3 = tf.eye(3, dtype=tf.float32)
        r = tf.where(m[..., None] > 0.5, r, eye3)         # m=[B,L,1] → [B,L,1,1]

        # ⚠️ Fully vectorized integration (avoiding tf.scan / while_loop / TensorArray -> compatible with MirroredStrategy multi-GPU scaling):
        # Global displacement g_i = R_i @ (bond_length * dir_i); position x_0 = 0, x_i = sum_{k<i} g_k (via tf.cumsum)
        g = tf.matmul(r, (d * self.bond_length)[..., None])[..., 0]   # [B, L, 3]
        coords = tf.cumsum(g, axis=1) - g                   # x_0 = 0, residue-wise accumulation

        if self.recycle is not None:
            coords = self.recycle(z, coords, r, mask, training=training)
        return coords

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            bond_length=self.bond_length,
            dropout=self.dropout_rate,
            recycle_steps=self.recycle_steps,
            refine_dim=self.refine_dim,
            refine_heads=self.refine_heads,
            refine_dropout=self.refine_dropout,
        )
        return cfg
