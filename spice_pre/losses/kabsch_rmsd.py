from __future__ import annotations

import tensorflow as tf


def kabsch_rmsd(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)
    mask = tf.cast(mask, tf.float32)
    dtype = pred.dtype
    m = tf.cast(mask, dtype)                 
    m3 = m[..., None]                        
    n = tf.reduce_sum(m, axis=-1)            
    n = tf.maximum(n, 1.0)

    p_centroid = tf.reduce_sum(pred * m3, axis=1) / n[:, None]       
    q_centroid = tf.reduce_sum(target * m3, axis=1) / n[:, None]     
    p = (pred - p_centroid[:, None, :]) * m3   
    q = (target - q_centroid[:, None, :]) * m3

    h = tf.einsum("blk,blj->bkj", p, q * m3)

    _, u, v = tf.linalg.svd(h)               

    det = tf.linalg.det(tf.matmul(v, u, transpose_b=True))   
    d = tf.where(det < 0.0, -1.0, 1.0)
    u_fix = tf.concat([u[..., :2], u[..., 2:] * d[:, None, None]], axis=-1)
    r = tf.matmul(v, u_fix, transpose_b=True)  

    rp = tf.einsum("bkj,blj->blk", r, p)     
    diff = rp - q
    sq = tf.reduce_sum(diff * diff, axis=-1) * m   
    mse = tf.reduce_sum(sq, axis=-1) / n           
    rmsd = tf.sqrt(tf.maximum(mse, 1e-12))
    return rmsd


def kabsch_rmsd_loss(pred, target, mask):
    return tf.reduce_mean(kabsch_rmsd(pred, target, mask))


def _pairwise_sq_dist(coords):
    c = tf.cast(coords, tf.float32)
    sq = tf.reduce_sum(c * c, axis=-1)                            
    dots = tf.einsum("bld,bmd->blm", c, c)                        
    return sq[:, :, None] + sq[:, None, :] - 2.0 * dots           


def distogram_ce_loss(logits, target, mask, num_bins=12, min_bin=3.0, max_bin=48.0):
    logits = tf.cast(logits, tf.float32)                          
    mask = tf.cast(mask, tf.float32)                              
    boundaries = tf.linspace(
        tf.cast(min_bin, tf.float32), tf.cast(max_bin, tf.float32),
        num_bins - 1,
    )                                                             
    boundaries = boundaries ** 2
    d_sq = tf.maximum(_pairwise_sq_dist(target), 0.0)             
    true_bins = tf.reduce_sum(
        tf.cast(d_sq[..., None] > boundaries, tf.int32), axis=-1
    )                                                             
    true_bins = tf.clip_by_value(true_bins, 0, num_bins - 1)
    # ⚠️ 速度优化（2026-08-12）：fused sparse CE —— 消灭 one_hot([B,L,L,bins]) 物化，
    # 软max+CE 一步融合（数学与 log_softmax+one_hot+reduce 完全等价），损失段 ~2-3× 提速
    per_pair = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_bins, logits=logits
    )                                                             
    m = mask[:, :, None] * mask[:, None, :]
    l = tf.shape(mask)[1]
    diag = tf.range(l)[None, :, None] == tf.range(l)[None, None, :]
    m = m * (1.0 - tf.cast(diag, tf.float32))                     
    n = tf.maximum(tf.reduce_sum(m, axis=[1, 2]), 1.0)
    return tf.reduce_mean(tf.reduce_sum(per_pair * m, axis=[1, 2]) / n)


def expected_dists_from_distogram(logits, num_bins=12, min_bin=3.0, max_bin=48.0):
    p = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)       
    edges = tf.linspace(
        tf.cast(min_bin, tf.float32), tf.cast(max_bin, tf.float32),
        num_bins - 1,
    )                                                             
    edges_full = tf.concat(
        [[0.0], edges, [tf.cast(max_bin, tf.float32) * 1.5]],
        axis=0,
    )                                                             
    centers = (edges_full[:-1] + edges_full[1:]) / 2.0            
    return tf.reduce_sum(p * centers, axis=-1)                    


def _pairwise_dist(coords):
    c = tf.cast(coords, tf.float32)
    sq = tf.reduce_sum(c * c, axis=-1)                            
    dots = tf.einsum("bld,bmd->blm", c, c)                      
    d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * dots
    return tf.sqrt(tf.maximum(d2, 0.0) + 1e-8)


def pairwise_coord_loss(pred_coords, target, mask):
    d_pred = _pairwise_dist(pred_coords)
    d_tar = _pairwise_dist(target)
    mask = tf.cast(mask, tf.float32)
    m = mask[:, :, None] * mask[:, None, :]                     
    l = tf.shape(mask)[1]
    diag = tf.range(l)[None, :, None] == tf.range(l)[None, None, :]
    m = m * (1.0 - tf.cast(diag, tf.float32))                   
    sq = (d_pred - d_tar) ** 2 * m
    n = tf.maximum(tf.reduce_sum(m, axis=[1, 2]), 1.0)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(sq, axis=[1, 2]) / n))
