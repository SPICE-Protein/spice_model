from __future__ import annotations

from contextlib import contextmanager

try:  
    from keras.saving import register_keras_serializable
except ImportError:  
    from tensorflow.keras.saving import register_keras_serializable  # type: ignore


@contextmanager
def silence_stdout_stderr():
    """A dummy context manager that does nothing, ensuring 100% thread safety and compatibility."""
    yield
