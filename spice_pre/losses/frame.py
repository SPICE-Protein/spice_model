"""Loss functions for the Frame structure head: primary supervision (Kabsch Cα RMSD), chirality constraint, and clash prevention.

- frame_kabsch_loss: Mean Squared Error (MSE) of aligned Cα coordinates (primary structural supervision, translationally and rotationally invariant).
- frame_chirality_loss: Normalized signed volume of adjacent Cα quadruplets to enforce native chirality 
  (forces the model to learn the natural right-handed topology of the L-amino acid backbone and prevents mirror/topological inversion).
- frame_clash_loss: Hinge penalty for non-adjacent (|i-j| >= 3) Cα pairs with distances < 3.5 Å, 
  directly resolving the geometric self-intersection issue in the legacy head_a 
  (bond lengths are structurally guaranteed by frame construction, eliminating the need for bond-length loss).

Coordinates are all Cα representations [B, L, 3] output by the Frame head, scaled identically to native Cα (absolute in Å).
"""
from __future__ import annotations

import tensorflow as tf

from spice_pre.losses.kabsch_rmsd import (
    _pairwise_dist,
    expected_dists_from_distogram,
    kabsch_rmsd,
)


def frame_kabsch_loss(
    pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor,
    gate: tf.Tensor | None = None,
) -> tf.Tensor:
    per = kabsch_rmsd(pred, target, mask)          # [B]
    if gate is not None:
        per = per * tf.cast(gate, tf.float32)       # Length gating: sets coordinate loss of long chains to zero
    return tf.reduce_mean(per)


def _normalized_tetra_volume(coords: tf.Tensor) -> tf.Tensor:
    """Computes the signed volume of four consecutive Cα atoms normalized by bond³, returning [B, L-3] (scale-invariant)."""
    p0 = coords[:, :-3]
    p1 = coords[:, 1:-2]
    p2 = coords[:, 2:-1]
    p3 = coords[:, 3:]
    vol = tf.reduce_sum(tf.linalg.cross(p1 - p0, p2 - p0) * (p3 - p0), axis=-1)
    return vol / (3.8 ** 3)


def frame_chirality_loss(
    pred: tf.Tensor, target: tf.Tensor, mask: tf.Tensor,
    gate: tf.Tensor | None = None,
) -> tf.Tensor:
    m = tf.cast(mask, tf.float32)
    vm = m[:, :-3] * m[:, 1:-2] * m[:, 2:-1] * m[:, 3:]
    vp = _normalized_tetra_volume(pred)
    vt = _normalized_tetra_volume(target)
    diff = (vp - vt) ** 2
    n = tf.maximum(tf.reduce_sum(vm, axis=-1), 1.0)
    per = tf.reduce_sum(diff * vm, axis=-1) / n    # [B]
    if gate is not None:
        per = per * tf.cast(gate, tf.float32)       # Length gating
    return tf.reduce_mean(per)


def frame_clash_loss(
    pred: tf.Tensor, mask: tf.Tensor, min_dist: float = 3.5,
    gate: tf.Tensor | None = None,
) -> tf.Tensor:
    m = tf.cast(mask, tf.float32)
    l = tf.shape(pred)[1]
    d = _pairwise_dist(pred)                              # [B, L, L]
    idx = tf.range(l)
    sep = tf.abs(idx[:, None] - idx[None, :])
    valid = tf.cast(sep >= 3, tf.float32)                 # Skip adjacent (i-1, i-2) pairs
    nb = valid * m[:, :, None] * m[:, None, :]
    viol = tf.maximum(min_dist - d, 0.0) ** 2
    n = tf.maximum(tf.reduce_sum(nb, axis=[1, 2]), 1.0)
    per = tf.reduce_sum(viol * nb, axis=[1, 2]) / n       # [B]
    if gate is not None:
        per = per * tf.cast(gate, tf.float32)             # Length gating
    return tf.reduce_mean(per)


def frame_dist_consistency_loss(
    pred: tf.Tensor, dist_logits: tf.Tensor, mask: tf.Tensor,
    num_bins: int, min_bin: float, max_bin: float,
) -> tf.Tensor:
    """Self-consistency loss between pairwise coordinate distances and expected distances from the distogram (relative error in logarithmic domain).

    - No native labels required: uses the model's own distogram (well-learned by cross-entropy over the entire chain) as the teacher.
    - Applied to all chains (including long chains) -> provides long-chain coordinate signals to the Frame head and recycling (a secondary channel bypassing the gate).
    - Logarithmic domain: scale-invariant. When long chains diverge (d_pred >> d_exp), the pairwise loss remains bounded by ~log²(ratio), 
      preventing extreme gradient propagation (hundreds of Å) from destabilizing the encoder, unlike native Kabsch RMSD.
      logarithmic domain scales nicely without gradient explosions.
    - Stop-gradient on d_exp: treats the distogram as a fixed reference to train only the coordinate side, avoiding reverse contamination of the distance head.
    """
    d_pred = _pairwise_dist(pred)                          # [B, L, L]
    d_exp = tf.stop_gradient(
        expected_dists_from_distogram(dist_logits, num_bins, min_bin, max_bin)
    )                                                      # [B, L, L]
    eps = 1e-3
    ratio = tf.math.log((d_pred + eps) / (d_exp + eps))    # 相对误差（尺度不变）
    per_pair = ratio ** 2
    m = tf.cast(mask, tf.float32)
    m = m[:, :, None] * m[:, None, :]
    l = tf.shape(mask)[1]
    diag = tf.range(l)[None, :, None] == tf.range(l)[None, None, :]
    m = m * (1.0 - tf.cast(diag, tf.float32))
    n = tf.maximum(tf.reduce_sum(m, axis=[1, 2]), 1.0)
    return tf.reduce_mean(tf.reduce_sum(per_pair * m, axis=[1, 2]) / n)
