from __future__ import annotations

try:  
    from keras.saving import register_keras_serializable
except ImportError:  
    from tensorflow.keras.saving import register_keras_serializable  # type: ignore


def setup_gpu(
    use_gpu: bool = True,
    mem_growth: bool = True,
    devices: str = "",
) -> None:
    import os

    import tensorflow as tf

    if use_gpu and devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
        return

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, mem_growth)
        except (ValueError, RuntimeError):
            pass  
