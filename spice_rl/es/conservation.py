from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_AA_CONSERVATION = {
    "W": 0.95, "C": 0.90, "G": 0.85, "F": 0.85, "Y": 0.85,
    "P": 0.75, "H": 0.70, "D": 0.55, "E": 0.55, "N": 0.50,
    "Q": 0.50, "K": 0.45, "R": 0.45, "M": 0.40, "I": 0.40,
    "L": 0.35, "V": 0.35, "A": 0.30, "T": 0.30, "S": 0.30,
}


def conservation_vector(
    base_seq: str, external: Optional[np.ndarray] = None
) -> np.ndarray:
    if external is not None and len(external) == len(base_seq):
        return np.asarray(external, np.float32)
    return np.array([_AA_CONSERVATION.get(a, 0.5) for a in base_seq], np.float32)


def conserved_mask(cons: Sequence[float], threshold: float = 0.80) -> np.ndarray:
    return np.asarray(cons, np.float32) >= float(threshold)


def rejects_masked(base_seq: str, mut_seq: str, mask: np.ndarray) -> bool:
    if len(base_seq) != len(mut_seq):
        return True
    for i in range(len(base_seq)):
        if mask[i] and base_seq[i] != mut_seq[i]:
            return True
    return False


def load_external_conservation(path: str, seq_len: int) -> Optional[np.ndarray]:
    import os

    if not path or not os.path.exists(path):
        return None
    try:
        if path.endswith(".npz"):
            d = np.load(path)
            v = d["conservation"] if "conservation" in d else d[list(d.keys())[0]]
        else:
            v = np.load(path)
        v = np.asarray(v, np.float32).reshape(-1)
        return v if len(v) == seq_len else None
    except Exception:  # noqa: BLE001
        return None
