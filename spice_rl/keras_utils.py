"""Compatibility layer for Keras serialization registration (RL package).

Under the TF 2.21 + Keras 3 standalone layout, `tf.keras.saving` is no longer
exposed; `register_keras_serializable` lives in `keras.saving`. Fall back
through the available options in order.
"""
from __future__ import annotations

try:  # Keras 3 (standalone package)
    from keras.saving import register_keras_serializable
except ImportError:  # legacy TF-bundled Keras 2
    from tensorflow.keras.saving import register_keras_serializable  # type: ignore
