"""Compatibility layer for Keras serialization registration.

Under the TF 2.21 + Keras 3 standalone layout, `tf.keras.saving` is no longer
exposed; `register_keras_serializable` lives in `keras.saving`. Fall back
through the available options in order.
"""
from __future__ import annotations

try:  # Keras 3 (standalone package)
    from keras.saving import register_keras_serializable
except ImportError:  # legacy TF-bundled Keras 2
    from tensorflow.keras.saving import register_keras_serializable  # type: ignore


# ---------------------------------------------------------------------------
# GPU 启用开关
# ---------------------------------------------------------------------------
def setup_gpu(
    use_gpu: bool = True,
    mem_growth: bool = True,
    devices: str = "",
) -> None:
    """配置 TensorFlow 的 GPU 使用策略。

    参数：
        use_gpu: True 用 GPU（若有）；False 强制只用 CPU。
        mem_growth: True 显存按需增长（默认）；False 一次性占用。
        devices: 逗号分隔的 GPU 编号白名单（如 "0" / "0,1"），空串 = 全部。

    必须在创建任何张量/模型之前调用才有效。
    """
    import os

    import tensorflow as tf

    if use_gpu and devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    if not use_gpu:
        # 隐藏所有 GPU，强制走 CPU
        tf.config.set_visible_devices([], "GPU")
        return

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, mem_growth)
        except (ValueError, RuntimeError):
            pass  # 设备已在更早阶段初始化，忽略
