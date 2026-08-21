#!/usr/bin/env python
"""Hygon CPU × TF/oneDNN matmul 静默损坏的极简复现（给集群管理员用）。
背景：SCNet 某节点 CPU = Hygon C86 7185（国产 AMD-Zen 兼容，AVX2+FMA，无 AVX512）。
该 CPU 上 TF/oneDNN 的 AVX2/FMA matmul 内核算错（batch>1），SSE41 路径正确。
本脚本无模型、无容器外依赖，直接 python 跑即可复现。

用法（集群登录/计算节点，用装有 TF 的 python）：
    python repro_hygon_matmul.py
    或容器内：singularity exec <SIF> /opt/conda/envs/spice/bin/python scripts/experiments/slurm/repro_hygon_matmul.py

判读：
    numpy 对 + TF 错      → oneDNN 特定 bug（软件层，可压 SSE41 规避/升级 oneDNN 修复）
    numpy 也错            → 该节点 AVX2/FMA 执行有问题（需隔离节点/查硬件）
    TF 在 SSE41 下对       → 证实是 AVX2/FMA 内核路径的错
"""
import os
import numpy as np

# 固定随机种子，可复现
rng = np.random.default_rng(7)
A = rng.normal(size=(32, 256)).astype(np.float32)
B = rng.normal(size=(256, 256)).astype(np.float32)

print("==", "CPU 型号:", (open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[1].strip()) if os.path.exists("/proc/cpuinfo") else "?", "==")

# ---- 1) numpy matmul（OpenBLAS/MKL）vs float64 参考 ----
C = A @ B
C64 = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
err = float(np.abs(C - C64).max())
print(f"[numpy ] matmul absmax={abs(C).max():.4f}  max|err|={err:.3e}  finite={bool(np.all(np.isfinite(C)))}")
numpy_ok = np.all(np.isfinite(C)) and err < 1e-2 * max(1.0, abs(C).max())

# ---- 2) TF matmul（oneDNN，默认 ISA）----
import tensorflow as tf
y = tf.matmul(A, B).numpy()
tf_ok = bool(tf.reduce_all(tf.math.is_finite(y))) and abs(y).max() < 100
print(f"[TF    ] matmul absmax={abs(y).max():.4e}  finite={bool(tf.reduce_all(tf.math.is_finite(y)))}  "
      f"(正常应 ~4 且有限)")

# ---- 3) TF matmul（强制 SSE41，绕开 AVX2/FMA 坏内核）----
os.environ["ONEDNN_MAX_CPU_ISA"] = "SSE41"
os.environ["DNNL_MAX_CPU_ISA"] = "SSE41"
y2 = tf.matmul(A, B).numpy()
sse_ok = bool(tf.reduce_all(tf.math.is_finite(y2))) and abs(y2).max() < 100
print(f"[TF    ] matmul @SSE41 absmax={abs(y2).max():.4f}  finite={bool(tf.reduce_all(tf.math.is_finite(y2)))}")

print()
print("结论:")
print(f"  numpy 正确   : {numpy_ok}")
print(f"  TF 默认路径  : {'正确' if tf_ok else '✗ 损坏（AVX2/FMA 内核算错）'}")
print(f"  TF @SSE41    : {'正确' if sse_ok else '✗ 仍坏'}")
if numpy_ok and not tf_ok and sse_ok:
    print("  → oneDNN 对 Hygon 的 AVX2/FMA 内核 bug：软件层问题，压 ONEDNN_MAX_CPU_ISA=SSE41 可规避；")
    print("    同节点其他用 TF/PyTorch/oneDNN 的任务也可能受影响，建议升级 oneDNN 或配全局 ISA 上限。")
elif not numpy_ok:
    print("  → numpy 也错：该节点 AVX2/FMA 执行疑似硬件级问题，建议隔离节点排查。")
elif not tf_ok:
    print("  → TF 坏但 SSE41 也坏：需要进一步定位（可能 Eigen/其它路径）。")
else:
    print("  → 当前环境全部正确（可能在非 Hygon CPU 上）。")
