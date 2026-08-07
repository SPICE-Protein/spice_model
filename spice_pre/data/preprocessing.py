"""Preprocessing：序列编码、环境归一化、坐标提取。

与 download_pdb.py 保持一致的残基命名映射，保证 `seq` 与 Cα 坐标严格对齐
（序列从 CA 原子行的 res_name 重建，坐标按 (chain_id, res_seq) 排序收集）。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 氨基酸编码
# ---------------------------------------------------------------------------
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}  # 1..20
PAD_IDX = 0
UNK_IDX = 21  # X / 未知残基
VOCAB_SIZE = 23

# 3-letter -> 1-letter（含质子化变体，与 download_pdb.py 对齐）
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
# 环境归一化范围（与 download_pdb.py 一致）
# ---------------------------------------------------------------------------
PH_MIN, PH_MAX = 0.0, 14.0
TEMP_MIN, TEMP_MAX = 150.0, 400.0
IONIC_REF_M = 1.0   # log 映射参考浓度


def aa_to_idx(aa: str) -> int:
    return AA2IDX.get(aa.upper(), UNK_IDX)


def seq_to_tokens(seq: str) -> np.ndarray:
    """一字母序列 -> int32 token 数组。"""
    return np.array([aa_to_idx(ch) for ch in seq], dtype=np.int32)


def res_names_to_seq(res_names: Sequence[str]) -> str:
    """3-letter 残基名序列 -> 一字母序列。"""
    return "".join(RES_NAME_TO_AA.get(str(n).upper(), "X") for n in res_names)


# ---------------------------------------------------------------------------
# 环境归一化
# ---------------------------------------------------------------------------
def normalize_pH(ph: Optional[float]) -> float:
    if ph is None:
        return 0.5  # 默认 pH 7.0 的归一化值
    return float(np.clip((ph - PH_MIN) / (PH_MAX - PH_MIN), 0.0, 1.0))


def normalize_temp(temp: Optional[float]) -> float:
    if temp is None:
        return (298.0 - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
    return float(np.clip((temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN), 0.0, 1.0))


def normalize_ionic(ionic_m: Optional[float], default_m: float = 0.15) -> float:
    """离子强度对数归一化：1mM~1M 映射到 ~0~1。"""
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
    """返回 [pH_norm, T_norm, ionic_norm] float32。"""
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
# 坐标
# ---------------------------------------------------------------------------
def center_coords(coords: np.ndarray) -> np.ndarray:
    """减去质心（数值稳定，Kabsch 内部也会去质心）。"""
    return coords - coords.mean(axis=0, keepdims=True)


def coords_to_tensor(coords: np.ndarray) -> np.ndarray:
    """[L, 3] float64/float32 -> float32。"""
    return np.asarray(coords, dtype=np.float32)
