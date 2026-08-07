"""SPICE — Sequence-Protein Interaction under Conditional Environments.

Python training and decision core (spice-rl).
Current module: Phase 1 Pre-train (dynamic Transformer + AdaLN + Head A coordinate prediction).
"""
from spice_pre.config import Config, DataConfig, ModelConfig, TrainConfig, load_config

__version__ = "0.1.0"
__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
]
