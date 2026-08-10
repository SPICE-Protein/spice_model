from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

_SKIP_OFFSET = 1


def native_contact_map(
    coords: np.ndarray, cutoff: float = 8.0
) -> Set[Tuple[int, int]]:
    coords = np.asarray(coords, dtype=np.float64)
    L = coords.shape[0]
    pairs: Set[Tuple[int, int]] = set()
    for i in range(L):
        for j in range(i + 1, L):
            if j - i <= _SKIP_OFFSET:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < cutoff:
                pairs.add((i, j))
    return pairs


def native_contact_q(
    coords: np.ndarray,
    native_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    cutoff: float = 8.0,
    reference: Optional[np.ndarray] = None,
) -> float:
    if native_pairs is None:
        ref = reference if reference is not None else coords
        native_pairs = list(native_contact_map(ref, cutoff))
    if not native_pairs:
        return 1.0
    coords = np.asarray(coords, dtype=np.float64)
    kept = 0
    for i, j in native_pairs:
        if float(np.linalg.norm(coords[i] - coords[j])) < cutoff:
            kept += 1
    return kept / len(native_pairs)


def per_residue_rmsf(coords_history: Sequence[np.ndarray]) -> np.ndarray:
    if not coords_history:
        return np.zeros(0, dtype=np.float32)
    stack = np.stack([np.asarray(c, dtype=np.float64) for c in coords_history])
    mean = stack.mean(axis=0)  
    var = np.mean((stack - mean[None, :, :]) ** 2, axis=0).sum(axis=1)
    return np.sqrt(var).astype(np.float32)


def track_rmsf(
    coords_history: List[np.ndarray],
    coords: np.ndarray,
    maxlen: int,
) -> List[np.ndarray]:
    if maxlen <= 0:
        return []
    coords_history.append(np.asarray(coords, dtype=np.float32))
    if len(coords_history) > maxlen:
        del coords_history[0]
    return coords_history
