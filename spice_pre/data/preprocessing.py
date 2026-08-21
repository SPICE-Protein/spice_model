"""Preprocessing module: sequence tokenization, environmental normalization, and coordinate extraction.

Maintains a residue naming mapping consistent with download_pdb.py, ensuring strict alignment 
between reconstructed sequence tokens and Cα coordinates (sequence reconstructed from Cα atom res_names, 
and coordinates sorted and collected by (chain_id, res_seq)).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Amino Acid Tokenization
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}  # 1..20
PAD_IDX = 0
UNK_IDX = 21  # X / Unknown residue
VOCAB_SIZE = 23

# 3-letter -> 1-letter (including protonated variants, aligned with download_pdb.py)
RES_NAME_TO_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "XAA": "X",
    "UNK": "X", "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C",
    "CYM": "C", "GLH": "E", "ASH": "D", "LYN": "K", "TYM": "Y",
}

# ---------------------------------------------------------------------------
# Environmental Normalization Ranges (consistent with download_pdb.py)
# ---------------------------------------------------------------------------
PH_MIN, PH_MAX = 0.0, 14.0
TEMP_MIN, TEMP_MAX = 150.0, 400.0
IONIC_REF_M = 1.0   # Reference concentration for logarithmic mapping


def aa_to_idx(aa: str) -> int:
    return AA2IDX.get(aa.upper(), UNK_IDX)


def seq_to_tokens(seq: str) -> np.ndarray:
    """Converts 1-letter amino acid sequence to int32 token array."""
    return np.array([aa_to_idx(ch) for ch in seq], dtype=np.int32)


def res_names_to_seq(res_names: Sequence[str]) -> str:
    """Converts 3-letter residue name sequence to 1-letter amino acid sequence."""
    return "".join(RES_NAME_TO_AA.get(str(n).upper(), "X") for n in res_names)


# ---------------------------------------------------------------------------
# Environmental Normalization
# ---------------------------------------------------------------------------
def normalize_pH(ph: Optional[float]) -> float:
    if ph is None:
        return 0.5  # Normalized value for default pH 7.0
    return float(np.clip((ph - PH_MIN) / (PH_MAX - PH_MIN), 0.0, 1.0))


def normalize_temp(temp: Optional[float]) -> float:
    if temp is None:
        return (298.0 - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    return float(np.clip((temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0.0, 1.0))


def normalize_ionic(ionic_m: Optional[float], default_m: float = 0.15) -> float:
    """Logarithmic normalization of ionic strength: maps 1 mM - 1 M range to [0.0, 1.0]."""
    if ionic_m is None:
        ionic_m = default_m
    ionic_m = float(np.clip(ionic_m, 1e-3, 1.0))
    return float(np.clip(np.log10(ionic_m / 1e-3) / np.log10(1.0 / 1e-3), 0.0, 1.0))


def normalize_env(
    ph: Optional[float],
    temp: Optional[float],
    ionic_m: Optional[float],
    default_env: Tuple[float, float, float] = (7.0, 298.0, 0.15),
) -> np.ndarray:
    """Returns [pH_norm, T_norm, ionic_norm] float32 array."""
    d_ph, d_t, d_i = default_env
    return np.array(
        [
            normalize_pH(ph if ph is not None else d_ph),
            normalize_temp(temp if temp is not None else d_t),
            normalize_ionic(ionic_m if ionic_m is not None else d_i),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def center_coords(coords: np.ndarray) -> np.ndarray:
    """Subtracts centroid coordinates (essential for numerical stability, also performed inside Kabsch RMSD)."""
    return coords - coords.mean(axis=0, keepdims=True)


def coords_to_tensor(coords: np.ndarray) -> np.ndarray:
    """[L, 3] float64/float32 -> float32。"""
    return np.asarray(coords, dtype=np.float32)
