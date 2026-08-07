"""SPICE Phase 2+: post-training (Post-train / RL).

Macro ES (searching the joint mutation + environment space) + micro SAC
(finetuning conformations with a fixed sequence), directly interfacing with
the Rust engine `spice_engine` (all-atom MD + potential energy + five-dim M).
"""
from spice_rl.config import Config as RLConfig  # noqa: F401  (compat)

__version__ = "0.2.0"
