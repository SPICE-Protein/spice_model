"""TF-free 突变体全原子构建（从 train_post 抽出，供 fork 并发脚本在子进程安全使用）。

train_post 顶层 import tensorflow，任何进程只要 import 它就会拉起 TF 线程。
fork 一个带活线程的进程 → 子进程继承被"死线程"持有的锁 → 引擎/线程池死锁。
所以把纯 numpy 的 `_mutant_atoms` / `build_mutant_structure_from_ca` 挪到这里，
让并发脚本（compute_ddg_proxy_m5.py 等）在 fork 前不触碰 TF。
"""

from __future__ import annotations

import numpy as np

from spice_rl.env.sidechain import _BB, _element_of, place_sidechain
from spice_rl.env.structure import structure_from_atoms

_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

_AA_NORM: dict = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "ASH": "ASP", "GLH": "GLU",
    "CYX": "CYS", "CYM": "CYS", "SEC": "CYS",
    "LYN": "LYS", "HYP": "PRO",
}


def _mutant_atoms(base_atoms: dict, mut_seq: str, ca_coords: np.ndarray = None):
    base_seq = base_atoms["res_seq"]
    base_names = base_atoms["atom_names"]
    base_elems = base_atoms["elements"]
    base_resnames = base_atoms["res_names"]
    base_coords = base_atoms["coords"]

    groups: list = []
    cur = object()
    for i, r in enumerate(base_seq):
        if r != cur:
            cur = r
            groups.append([])
        groups[-1].append(i)

    names, elems, seqs, resnames, coords = [], [], [], [], []
    ca_idx = 0
    for gi, idxs in enumerate(groups):
        if gi >= len(mut_seq):
            raise ValueError(
                f"infeasible mutation: mut_seq({len(mut_seq)}) shorter than "
                f"structure({len(groups)} residues)"
            )
        new_aa = mut_seq[gi]
        mutant_3 = _AA3.get(new_aa)
        if mutant_3 is None:
            raise ValueError(f"infeasible mutation at residue {gi}: unknown AA '{new_aa}'")

        wild_res = base_resnames[idxs[0]].upper()
        wild_norm = _AA_NORM.get(wild_res, wild_res)
        mutated = wild_norm != mutant_3

        if not mutated:
            for i in idxs:
                if base_elems[i] == "H":
                    continue
                name = base_names[i]
                x, y, z = base_coords[i]
                if name == "CA" and ca_coords is not None and ca_idx < len(ca_coords):
                    x, y, z = ca_coords[ca_idx]
                    ca_idx += 1
                names.append(name)
                elems.append(base_elems[i])
                seqs.append(base_seq[i])
                resnames.append(mutant_3)
                coords.append([x, y, z])
            continue

        present = {}
        for i in idxs:
            if base_elems[i] == "H":
                continue
            name = base_names[i]
            if name in _BB:
                x, y, z = base_coords[i]
                if name == "CA" and ca_coords is not None and ca_idx < len(ca_coords):
                    x, y, z = ca_coords[ca_idx]
                    ca_idx += 1
                present[name] = np.asarray([x, y, z], np.float64)
        idxs_set = set(idxs)
        others = np.asarray(
            [
                base_coords[i]
                for i in range(len(base_coords))
                if i not in idxs_set and base_elems[i] != "H"
            ],
            np.float32,
        ).reshape(-1, 3)
        side = place_sidechain(present, mutant_3, others=others)

        for i in idxs:
            if base_elems[i] == "H":
                continue
            name = base_names[i]
            if name in _BB:
                names.append(name)
                elems.append(base_elems[i])
                seqs.append(base_seq[i])
                resnames.append(mutant_3)
                coords.append(list(present[name]))

        for name, coord in side:
            names.append(name)
            elems.append(_element_of(name))
            seqs.append(base_seq[idxs[0]])
            resnames.append(mutant_3)
            coords.append(list(coord))

    return names, elems, seqs, resnames, np.asarray(coords, np.float32)


def build_mutant_structure(base_atoms: dict, mut_seq: str):
    names, elems, seqs, resnames, coords = _mutant_atoms(base_atoms, mut_seq)
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def build_mutant_structure_from_ca(base_atoms: dict, mut_seq: str, pred_ca: np.ndarray = None):
    names, elems, seqs, resnames, coords = _mutant_atoms(base_atoms, mut_seq, pred_ca)
    return structure_from_atoms(names, elems, seqs, resnames, coords)
