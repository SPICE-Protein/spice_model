from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("spice")


def _engine():
    import spice_engine

    return spice_engine


def _parse_mmcif_atoms(path: str):
    """纯 python 解析 _atom_site loop → (names, elements, res_seq, res_names, coords float32)。
    忽略 # 行以兼容：标准文件（# 在数据后）与自产旧文件（# 在列头和数据之间）。"""
    cols: list[str] = []
    rows: list[list[str]] = []
    in_atom = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s == "#":
                continue  # 空行与 # 忽略：标准文件在数据后、自产旧文件在列头和数据之间
            if s.startswith("loop_"):
                in_atom = False
            elif s.startswith("_atom_site."):
                if not in_atom:
                    cols, in_atom = [], True
                cols.append(s)
            elif s.startswith("_"):
                in_atom = False
            elif in_atom:
                rows.append(s.split())
    if not cols:
        raise ValueError(f"mmCIF has no _atom_site loop: {path}")
    col_idx = {name: i for i, name in enumerate(cols)}

    def gi(row: list[str], name: str):
        i = col_idx.get(name)
        if i is None or i >= len(row):
            return None
        return row[i]

    names, elems, resseq, resnm, coords = [], [], [], [], []
    for r in rows:
        if gi(r, "_atom_site.group_PDB") != "ATOM":
            continue
        nm = gi(r, "_atom_site.label_atom_id")
        el = gi(r, "_atom_site.type_symbol")
        rn = gi(r, "_atom_site.label_comp_id")
        rs = gi(r, "_atom_site.label_seq_id")
        xyz = [gi(r, f"_atom_site.Cartn_{c}") for c in ("x", "y", "z")]
        if None in (nm, el, rn, rs, *xyz):
            continue
        try:
            rs_i = int(rs)
            xyz_f = [float(v) for v in xyz]
        except (TypeError, ValueError):
            continue
        names.append(nm)
        elems.append(el)
        resseq.append(rs_i)
        resnm.append(rn)
        coords.append(xyz_f)
    if not names:
        raise ValueError(f"no ATOM rows parsed from mmCIF: {path}")
    return names, elems, resseq, resnm, np.asarray(coords, dtype=np.float32)


def base_atoms_from_mmcif(path: str) -> dict:
    """从 mmCIF 直接构造 base_atoms（路径 B 突变重建所需），与 load_structure_with_atoms
    的返回格式一致：{atom_names, elements, res_seq, res_names, coords float32[N,3]}。
    引擎 Structure 不暴露原子数组，故必须从源文件重解析（纯 python，含兜底解析）。"""
    names, elems, resseq, resnm, coords = _parse_mmcif_atoms(path)
    return {
        "atom_names": names,
        "elements": elems,
        "res_seq": resseq,
        "res_names": resnm,
        "coords": coords,
    }


def _structure_from_mmcif_py(path: str):
    """纯 python 兜底：引擎 from_mmcif 需要完整 RCSB 头（_entity/_struct_asym 等 category），
    而自产的 soup 亲本 mmCIF 只有 _atom_site loop → 引擎能解析但读到 0 残基。
    兜底按 _atom_site 列名取 ATOM 行 → from_atoms（与 verify notebook 的
    protein_structure_from_mmcif 同套路，能读自产 12/22 列与过滤后的蛋白 mmCIF）。"""
    names, elems, resseq, resnm, coords = _parse_mmcif_atoms(path)
    return structure_from_atoms(names, elems, resseq, resnm, coords)


def structure_from_mmcif(path: str):
    se = _engine()
    try:
        struct = se.Structure.from_mmcif(path)
        # 引擎解析成功且读到残基 → 直接用（完整 RCSB 文件）
        if struct.residue_count() > 0:
            return struct
        # 0 残基（自产 soup 亲本：只有 _atom_site loop，引擎缺头信息）→ 兜底
        logger.info(f"[structure] engine from_mmcif 读到 0 残基，走纯 python 解析: {path}")
    except Exception as e:  # noqa: BLE001
        logger.info(f"[structure] engine from_mmcif 失败({type(e).__name__}: {e})，走纯 python 解析: {path}")
    return _structure_from_mmcif_py(path)


def structure_from_atoms(
    atom_names: Sequence[str],
    elements: Sequence[str],
    res_seq: Sequence[int],
    res_names: Sequence[str],
    coords: np.ndarray,
    occupancy: Optional[np.ndarray] = None,
):
    se = _engine()
    kwargs = dict(
        atom_names=list(atom_names),
        elements=list(elements),
        res_seq=[int(i) for i in res_seq],
        res_names=list(res_names),
        coords=np.asarray(coords, dtype=np.float32),
    )
    if occupancy is not None:
        kwargs["occupancy"] = np.asarray(occupancy, dtype=np.float32)
    return se.Structure.from_atoms(**kwargs)


def structure_from_dataframe(df, pdb_id: Optional[str] = None):
    if pdb_id is not None and "pdb_id" in df.columns:
        df = df.filter(df["pdb_id"] == pdb_id) if hasattr(df, "filter") else df[df["pdb_id"] == pdb_id]
    if "chain_id" in df.columns:
        order_cols = ["chain_id", "res_seq"]
        if hasattr(df, "sort"):
            df = df.sort(order_cols)
        else:
            df = df.sort_values(order_cols)
    coords = np.stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()], axis=1)
    occ = None
    if "occupancy" in df.columns:
        occ = np.asarray(df["occupancy"].to_numpy(), dtype=np.float32)
    return structure_from_atoms(
        df["atom_name"].to_list(),
        df["element"].to_list(),
        df["res_seq"].to_list(),
        df["res_name"].to_list(),
        coords,
        occupancy=occ,
    )


def structure_from_parquet(
    parquet_dir: str,
    pdb_id: str,
    shard_fname: Optional[str] = None,
):
    import glob

    pdb_id = pdb_id.upper()  

    if shard_fname is None:
        matches = glob.glob(os.path.join(parquet_dir, "atoms_shard_*.parquet"))
        if not matches:
            raise FileNotFoundError(f"no atoms shards in {parquet_dir}")
        import polars as pl

        for m in sorted(matches):
            df = pl.read_parquet(m, columns=["pdb_id", "atom_name", "element", "res_seq", "res_name", "x", "y", "z"])
            sub = df.filter(pl.col("pdb_id") == pdb_id)
            if sub.height:
                return structure_from_dataframe(sub)
        raise KeyError(f"pdb_id {pdb_id} not found in {parquet_dir}")
    df = __import__("polars").read_parquet(os.path.join(parquet_dir, shard_fname))
    return structure_from_dataframe(df, pdb_id=pdb_id)


def load_structure_with_atoms(
    parquet_dir: str,
    pdb_id: str,
    max_residues: Optional[int] = None,
    chain_id: Optional[str] = None,
) -> tuple:
    import glob

    import polars as pl

    pdb_id = pdb_id.upper()  
    for shard in sorted(glob.glob(os.path.join(parquet_dir, "atoms_shard_*.parquet"))):
        df = pl.read_parquet(
            shard,
            columns=["pdb_id", "chain_id", "atom_name", "element", "res_seq",
                     "res_name", "x", "y", "z"],
        )
        sub = df.filter(pl.col("pdb_id") == pdb_id)
        if sub.height == 0:
            continue
        chains = sorted(
            sub.group_by("chain_id")
            .agg(pl.col("res_seq").n_unique().alias("n_res"))
            .iter_rows()
        )
        if chain_id is not None:
            if chain_id not in [c for c, _ in chains]:
                raise ValueError(
                    f"pdb_id {pdb_id} 无 chain {chain_id}"
                    f"（实际 chains: {[c for c, _ in chains]}）"
                )
            pick = chain_id
        else:
            pick = max(chains, key=lambda c: c[1])[0]  
            if len(chains) > 1:
                logger.warning(
                    f"{pdb_id} 多链 {chains}，取最长链 {pick} "
                    "（path B 按 res_seq 分组需单链；传 chain_id 可指定）"
                )
        sub = sub.filter(pl.col("chain_id") == pick).sort("res_seq")
        struct = structure_from_dataframe(sub)
        if max_residues is not None and struct.residue_count() > max_residues:
            raise ValueError(
                f"pdb_id {pdb_id} 过长: {struct.residue_count()} aa > "
                f"max_residues={max_residues}（RL 只收短蛋白）"
            )
        base_atoms = {
            "atom_names": sub["atom_name"].to_list(),
            "elements": sub["element"].to_list(),
            "res_seq": sub["res_seq"].to_list(),
            "res_names": sub["res_name"].to_list(),
            "coords": np.stack(
                [sub["x"].to_numpy(), sub["y"].to_numpy(), sub["z"].to_numpy()], axis=1
            ).astype(np.float32),
        }
        return struct, base_atoms
    raise KeyError(f"pdb_id {pdb_id} not found in {parquet_dir}")
