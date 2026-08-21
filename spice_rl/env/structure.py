from __future__ import annotations

import os
import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("spice")


def _engine():
    import spice_engine

    return spice_engine


def structure_from_mmcif(path: str):
    se = _engine()
    return se.Structure.from_mmcif(path)


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
