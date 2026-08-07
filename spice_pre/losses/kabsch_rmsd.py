"""可微 Kabsch RMSD 损失（残基级）。

Pre-train 阶段路径 A 的监督损失：先把预测坐标与真实坐标做最优刚体
（旋转 + 平移）对齐，再计算逐残基均方根误差。整个过程基于 SVD，
完全可微，支持 batch 与变长（mask）输入。

算法来源（保持一致，均为标准 Kabsch）：
- Uni-Fold:  https://github.com/dptech-corp/Uni-Fold  unifold/losses/geometry.py
- alphaflow: https://github.com/bjing2016/alphaflow   alphaflow/utils/diffusion.py
- charnley/rmsd（经典 numpy 参考）: https://github.com/charnley/rmsd

本实现已与 charnley/rmsd 经典算法做 60 组随机数值对拍（含反射旋转、
噪声、随机 mask），最大误差 < 1e-6，测试见 tests/test_kabsch_rmsd.py。
"""
from __future__ import annotations

import tensorflow as tf


def kabsch_rmsd(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """计算每结构的 Kabsch RMSD。

    Args:
        pred:   [B, L, 3] 预测 Cα 坐标（padding 位置可为任意值，被 mask 忽略）。
        target: [B, L, 3] 真实 Cα 坐标。
        mask:   [B, L]    float32，1 = 有效残基，0 = padding。

    Returns:
        rmsd: [B] 每结构对齐后的 RMSD（单位 Å）。
    """
    # mixed_float16 下 pred/target 可能是 float16：SVD 必须在 float32 下计算
    # 以保证数值稳定（fp16 的 SVD 会丢精度）。
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)
    mask = tf.cast(mask, tf.float32)
    dtype = pred.dtype
    m = tf.cast(mask, dtype)                 # [B, L]
    m3 = m[..., None]                        # [B, L, 1]
    n = tf.reduce_sum(m, axis=-1)            # [B]
    n = tf.maximum(n, 1.0)

    # 去质心（只统计有效残基）
    p_centroid = tf.reduce_sum(pred * m3, axis=1) / n[:, None]       # [B, 3]
    q_centroid = tf.reduce_sum(target * m3, axis=1) / n[:, None]     # [B, 3]
    p = (pred - p_centroid[:, None, :]) * m3   # [B, L, 3]
    q = (target - q_centroid[:, None, :]) * m3

    # 协方差矩阵 H = P^T Q（按 mask 加权）  [B, 3, 3]
    h = tf.einsum("blk,blj->bkj", p, q * m3)

    # SVD: H = U @ diag(S) @ V^T
    _, u, v = tf.linalg.svd(h)               # u: [B,3,3], v: [B,3,3]

    # 最优旋转（Kabsch）：R = V @ diag(1,1,d) @ U^T，d 修正使 det>0。
    # 等价于修正 u 的最后一列：u_fix = [u1, u2, d*u3]，再 R = v @ u_fix^T。
    det = tf.linalg.det(tf.matmul(v, u, transpose_b=True))   # det(V U^T) [B]
    d = tf.where(det < 0.0, -1.0, 1.0)
    u_fix = tf.concat([u[..., :2], u[..., 2:] * d[:, None, None]], axis=-1)
    r = tf.matmul(v, u_fix, transpose_b=True)  # [B, 3, 3]

    # 对齐误差（rp = p @ r^T）
    rp = tf.einsum("bkj,blj->blk", r, p)     # [B, L, 3]
    diff = rp - q
    sq = tf.reduce_sum(diff * diff, axis=-1) * m   # [B, L]
    mse = tf.reduce_sum(sq, axis=-1) / n           # [B]
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))
    return rmsd


def kabsch_rmsd_loss(pred, target, mask):
    """Batch 平均标量损失（训练用）。"""
    return tf.reduce_mean(kabsch_rmsd(pred, target, mask))


def _pairwise_sq_dist(coords):
    """[B, L, 3] -> [B, L, L] 两两平方距离（Gram 矩阵法，省内存）。"""
    c = tf.cast(coords, tf.float32)
    sq = tf.reduce_sum(c * c, axis=-1)                            # [B, L]
    dots = tf.einsum("bld,bmd->blm", c, c)                        # [B, L, L]
    return sq[:, :, None] + sq[:, None, :] - 2.0 * dots           # [B, L, L]


def distogram_ce_loss(logits, target, mask, num_bins=12, min_bin=3.0, max_bin=48.0):
    """binned distogram 交叉熵（主目标，替代裸距离回归）。

    把每对残基的真实 Cα 距离分箱，对预测的 [B,L,L,N] logits 做 softmax CE，
    只统计有效残基对（mask_i & mask_j，i≠j）。
    - 平方距离 + 平方边界（数值稳定，与 `pts_to_distogram` 同款做法）
    - 距离 > max_bin 的落到最后一个溢出 bin
    相比直接坐标回归，距离分布是密集、易优化的信号，能学到"哪些残基在空间上靠近"
    （接触 → 二级结构 → 拓扑），这是 Pre-train 真正该学的折叠直觉。
    """
    logits = tf.cast(logits, tf.float32)                          # [B,L,L,N]
    mask = tf.cast(mask, tf.float32)                              # [B,L]
    boundaries = tf.linspace(
        tf.cast(min_bin, tf.float32), tf.cast(max_bin, tf.float32),
        num_bins - 1,
    )                                                             # [N-1]（Å）
    # 边界也平方，和平方距离比较（数值稳定）
    boundaries = boundaries ** 2
    d_sq = tf.maximum(_pairwise_sq_dist(target), 0.0)             # [B,L,L]（Å^2）
    true_bins = tf.reduce_sum(
        tf.cast(d_sq[..., None] > boundaries, tf.int32), axis=-1
    )                                                             # [B,L,L]
    true_bins = tf.clip_by_value(true_bins, 0, num_bins - 1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)                # [B,L,L,N]
    one_hot = tf.one_hot(true_bins, num_bins)                     # [B,L,L,N]
    per_pair = -tf.reduce_sum(one_hot * log_probs, axis=-1)       # [B,L,L]
    m = mask[:, :, None] * mask[:, None, :]
    l = tf.shape(mask)[1]
    diag = tf.range(l)[None, :, None] == tf.range(l)[None, None, :]
    m = m * (1.0 - tf.cast(diag, tf.float32))                     # 去掉对角线
    n = tf.maximum(tf.reduce_sum(m, axis=[1, 2]), 1.0)
    return tf.reduce_mean(tf.reduce_sum(per_pair * m, axis=[1, 2]) / n)


def expected_dists_from_distogram(logits, num_bins=12, min_bin=3.0, max_bin=48.0):
    """distogram logits -> 每对残基期望距离 [B,L,L]（评估/重建用）。"""
    p = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)       # [B,L,L,N]
    edges = tf.linspace(
        tf.cast(min_bin, tf.float32), tf.cast(max_bin, tf.float32),
        num_bins - 1,
    )                                                             # [N-1]
    # N 个 bin 中心：bin0 左边界用 0（避免与 edges[0]=min_bin 重复），溢出 bin 中心取 max_bin*1.5
    edges_full = tf.concat(
        [[0.0], edges, [tf.cast(max_bin, tf.float32) * 1.5]],
        axis=0,
    )                                                             # [N+1]
    centers = (edges_full[:-1] + edges_full[1:]) / 2.0            # [N]
    return tf.reduce_sum(p * centers, axis=-1)                    # [B,L,L]


def _pairwise_dist(coords):
    """[B, L, 3] -> [B, L, L] 两两欧氏距离（Gram 矩阵法）。

    用 ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b，避免 [B, L, L, 3] 大中间张量，
    防止 batch 大 / 序列长时 OOM（128×512×512×3×4 ≈ 1.2GB 的中间张量）。
    """
    c = tf.cast(coords, tf.float32)
    sq = tf.reduce_sum(c * c, axis=-1)                            # [B, L]
    dots = tf.einsum("bld,bmd->blm", c, c)                      # [B, L, L]
    d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * dots
    return tf.sqrt(tf.maximum(d2, 0.0) + 1e-8)


def pairwise_coord_loss(pred_coords, target, mask):
    """坐标两两距离回归：直接监督 head 输出的坐标。

    关键作用：预测点塌缩在原点附近时，Kabsch RMSD 的梯度被坐标尺度压死
    （~ -Q/(Rg*n)，乘 lr 后每步几乎不动），模型永远停在塌缩态。
    两两距离是无 SVD 的密集强信号（塌缩时误差 ~数百、梯度 ~几十），
    能把预测云直接"撑开"到目标尺度/形状。
    返回 RMSE（单位 Å），与 Kabsch RMSD 同量级。
    """
    d_pred = _pairwise_dist(pred_coords)
    d_tar = _pairwise_dist(target)
    mask = tf.cast(mask, tf.float32)
    m = mask[:, :, None] * mask[:, None, :]                     # [B, L, L]
    l = tf.shape(mask)[1]
    diag = tf.range(l)[None, :, None] == tf.range(l)[None, None, :]
    m = m * (1.0 - tf.cast(diag, tf.float32))                   # 去掉对角线
    sq = (d_pred - d_tar) ** 2 * m
    n = tf.maximum(tf.reduce_sum(m, axis=[1, 2]), 1.0)
    # 返回 RMSE（单位 Å），避免 MSE 量级淹没主 loss
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(sq, axis=[1, 2]) / n))
