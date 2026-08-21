from __future__ import annotations

import argparse
import os
import time

if os.environ.get("SPICE_RL_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

import logging
from spice_rl.config import Config, load_config, setup_logging

logger = logging.getLogger("spice")
from spice_rl.env import MDSimulationEnv
from spice_rl.env import (
    quick_check_env,
    save_phase_map,
    scan_phase_map,
    summarize_phase_map,
)
from spice_rl.confidence import ConfidenceHeadTrainer
from spice_rl.pseudo_labels import write_pseudo_tfrecord
from spice_rl.sac import SACTrainer
from spice_rl.metrics import MetricsLogger
from spice_rl.env.sidechain import place_sidechain, _BB, _element_of
from spice_rl.keras_utils import silence_stdout_stderr

AA20 = "ACDEFGHIKLMNPQRSTVWY"

_TB_WARNED = [False]


def _tb_scalar(writer, name, value, step):
    if writer is None:
        return
    try:
        with writer.as_default():
            tf.summary.scalar(name, value, step=step)
    except Exception as e:  # noqa: BLE001
        if not _TB_WARNED[0]:
            logger.warning(f"[warn] TensorBoard write failed, metric logging skipped (CSV remains unaffected): {e}")
            _TB_WARNED[0] = True


def build_rl_model(cfg: Config, max_seq_len: int = 512):
    from spice_pre.config import load_config as pre_load
    from spice_pre.models import SPICEPretrainModel

    pre_cfg = pre_load(cfg.post.pretrain_config or "configs/pretrain.yaml")
    model = SPICEPretrainModel(
        pre_cfg.model, heads=("A", "B", "Bp", "C", "D")
    )
    model(
        {
            "tokens": tf.zeros([1, 8], tf.int32),
            "env": tf.zeros([1, 3]),
            "mask": tf.ones([1, 8]),
        },
        training=False,
    )
    ckpt = cfg.post.pretrain_ckpt
    if os.path.exists(ckpt):
        model.load_weights(ckpt, skip_mismatch=True)
        logger.info(f"Successfully loaded Pre-train weights: {ckpt}")
    # Head B' is Head A's folded structure of mutated sequences (aliases, no independent weights) -> warm-start not required.
    return model


def _encode_z_pool(model, tokens, env, mask):
    out = model({"tokens": tokens, "env": env, "mask": mask}, training=False)
    z = out["z"][0]                      
    m = mask[0][:, None]                 
    # 2026-08-17 Defensive fix: block non-finite values (NaN/Inf) prior to multiplying by the mask, preventing NaN * 0.0 = NaN from polluting the entire chain
    z_clean = tf.where(tf.math.is_finite(z), z, tf.zeros_like(z))
    z_pool = tf.reduce_sum(z_clean * m, axis=0) / tf.maximum(tf.reduce_sum(m), 1.0)
    # 2026-08-19: Initialization-level NaN mitigation (empirically confirmed in HPC cluster smoke tests).
    # The 45k pre-trained model exhibits an empirical max|z| up to 1e35 (compared to ~35 for older local weights).
    # This amplifies the actor's first-layer activations to 1e17-1e35, causing immediate global NaN propagation 
    # during the first training update (actor kernel 66304/66304).
    # We first clip the latent code to a safe range to prevent squaring-induced overflow (from 1e35), 
    # then perform RMS normalization to O(1) while preserving directionality:
    # Normal z (~35) -> ~1.0, exploded z (1e35) -> ~1.0, achieving scale-robustness across all regimes.
    z_pool = tf.clip_by_value(z_pool, -1e6, 1e6)
    _rms = tf.sqrt(tf.reduce_mean(tf.square(z_pool)) + 1e-8)
    return z_pool / tf.maximum(_rms, 1e-6)


# 2026-08-19 Graph Mode Version (default). If containerized TF graph mode exhibits numerical instability (smoke Check 6),
# eager_mode() directs execution to the eager implementation above, bypassing @tf.function graph optimization.
_encode_z_pool_tf = tf.function(_encode_z_pool, reduce_retracing=True)


def encode_z(model, tokens, env, mask) -> np.ndarray:
    from spice_rl.config import eager_mode
    _pool = _encode_z_pool if eager_mode() else _encode_z_pool_tf
    z_pool = _pool(
        model,
        tf.constant(np.asarray(tokens, np.int32)[None]),
        tf.constant(np.asarray(env, np.float32)[None]),
        tf.constant(np.asarray(mask, np.float32)[None]),
    )
    return z_pool.numpy()


def tokens_from_seq(seq: str, max_len: int) -> tuple:
    from spice_pre.data.preprocessing import seq_to_tokens

    tokens = seq_to_tokens(seq[:max_len])
    L = tokens.shape[0]
    mask = np.ones(L, np.float32)
    return tokens, mask


_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
def predict_mutant_coords(model, tokens, env, mask) -> np.ndarray:
    out = model(
        {
            "tokens": tf.constant(tokens.astype(np.int32)[None]),
            "env": tf.constant(np.asarray(env, np.float32)[None]),
            "mask": tf.constant(np.asarray(mask, np.float32)[None]),
        },
        training=False,
    )
    # Head B' is Head A's frame-based folding for the mutated sequence (3.8 Å bond lengths + recycling),
    # where coords_mut is a realistic fold rather than a diffuse blob. However, this coarse fold contains numerous local clashes < 3 Å
    # (empirically ~100+ pairs), which would explode during all-atom construction (U = 1e11-1e27 / NaN) -> _sane_ca/_rescale 
    # now includes localized clash checking, falling back to the wild-type backbone if unbuildable (fixed the 0-survivor regression on 2026-08-14).
    if "coords_mut" in out:
        return out["coords_mut"][0].numpy()
    return out["coords"][0].numpy()


_AA_HEAVY: dict = {
    "ALA": {"CB"},
    "ARG": {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"CB", "CG", "OD1", "ND2"},
    "ASP": {"CB", "CG", "OD1", "OD2"},
    "CYS": {"CB", "SG"},
    "GLN": {"CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"CB", "CG", "CD", "OE1", "OE2"},
    "GLY": set(),
    "HIS": {"CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "LYS": {"CB", "CG", "CD", "CE", "NZ"},
    "MET": {"CB", "CG", "SD", "CE"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"CB", "CG", "CD"},
    "SER": {"CB", "OG"},
    "THR": {"CB", "OG1", "CG2"},
    "TRP": {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"CB", "CG1", "CG2"},
}
_AA_NORM: dict = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "ASH": "ASP", "GLH": "GLU",
    "CYX": "CYS", "CYM": "CYS", "SEC": "CYS",
    "LYN": "LYS", "HYP": "PRO",
}
_BB = {"N", "CA", "C", "O", "OXT"}


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
    from spice_rl.env.structure import structure_from_atoms

    names, elems, seqs, resnames, coords = _mutant_atoms(base_atoms, mut_seq)
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def build_mutant_structure_from_ca(base_atoms: dict, mut_seq: str, pred_ca: np.ndarray = None):
    from spice_rl.env.structure import structure_from_atoms

    names, elems, seqs, resnames, coords = _mutant_atoms(base_atoms, mut_seq, pred_ca)
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def _n_local_clash(ca, cut=3.0):
    """Number of clashes (< cut Å) for non-adjacent Cα pairs. Stable buildable folds should have near-zero clashes;
    Although Head A's Cα folded bond lengths/Rg may look normal, local regions often contain dense clashing (empirically ~118 pairs < 3 Å)
    -> All-atom construction would explode (U = 1e11-1e27 / NaN). This count is used to intercept unbuildable predicted 
    folds and fall back to the wild-type backbone."""
    ca = np.asarray(ca, np.float32)
    L = len(ca)
    n = 0
    for i in range(L):
        for j in range(i + 2, L):
            if np.linalg.norm(ca[i] - ca[j]) < cut:
                n += 1
    return n


def _sane_ca(pred_ca, wild_ca, min_ratio=0.4, max_ratio=2.5,
             max_clash=4, clash_cut=3.0):
    if pred_ca is None or len(pred_ca) == 0:
        return False
    if not np.all(np.isfinite(pred_ca)):
        return False
    c_p = np.asarray(pred_ca, np.float32)
    c_w = np.asarray(wild_ca, np.float32)
    if len(c_w) == 0:
        return False
    rg = lambda c: float(np.sqrt(np.mean(np.sum((c - c.mean(0)) ** 2, axis=1))))  # noqa: E731
    rgp, rgw = rg(c_p), rg(c_w)
    if rgw <= 0.0:
        return False
    if not (min_ratio <= rgp / rgw <= max_ratio):
        return False
    # Local clash: Rg/bond lengths are normal but localized clashing occurs -> all-atom construction would explode, unusable
    if _n_local_clash(c_p, clash_cut) > max_clash:
        return False
    return True


def _rescale_pred_ca(pred_ca, wild_ca, min_ratio=0.15, max_ratio=8.0,
                     max_clash=4, clash_cut=3.0):
    if pred_ca is None or wild_ca is None:
        return None
    if len(pred_ca) != len(wild_ca):
        return None
    if not np.all(np.isfinite(pred_ca)):
        return None
    c = np.asarray(pred_ca, np.float32)
    w = np.asarray(wild_ca, np.float32)
    rg = lambda x: float(np.sqrt(np.mean(np.sum((x - x.mean(0)) ** 2, axis=1))))  # noqa: E731
    rgp, rgw = rg(c), rg(w)
    if rgp <= 0.0 or rgw <= 0.0:
        return None
    ratio = rgp / rgw
    if not (min_ratio <= ratio <= max_ratio):
        return None
    c_s = (c - c.mean(0)) * (rgw / rgp) + w.mean(0)
    if not (0.95 <= rg(c_s) / rgw <= 1.05):
        return None
    if _n_local_clash(c_s, clash_cut) > max_clash:
        return None  # Scaling cannot resolve local clashes -> fall back to the wild-type backbone
    return c_s.astype(np.float32)


def _native_ca(base_atoms) -> np.ndarray:
    """Extracts wild-type Cα coordinates (deduplicates altloc by res_seq, aligning with the engine's dedup_altloc).

    2026-08-12 Fix: The original implementation fetched all CA atoms (including duplicate altloc, e.g., 7QF3 had 129), 
    whereas the engine builds 116 residues -> length mismatch -> native_contact_q is always 0.0 -> survival gate q >= 0.5 
    never passed -> the "0 survivors" observed in cluster runs was a measurement bug (2LYZ succeeded because it has no altloc).
    Deduplicating resolves the issue, making q normal (WT ≈ 0.93). Supports single-chain structures only (unique res_seq).
    """
    seen = {}
    for i, name in enumerate(base_atoms["atom_names"]):
        if name == "CA":
            rs = base_atoms["res_seq"][i]
            if rs not in seen:
                seen[rs] = base_atoms["coords"][i]
    return np.array([seen[k] for k in sorted(seen)], np.float32)


def extract_base_atoms(structure) -> dict:
    raise NotImplementedError("Please provide all-atom arrays directly from the data pipeline; see path_b_search parameters")


def _sac_nan_report(sac) -> str:
    """SAC weights NaN watchdog diagnostic (2026-08-18): returns location description of the first non-finite weight; empty string if all are finite."""
    try:
        for _tag, _m in (
            ("actor", sac.actor),
            ("critic", sac.critic),
            ("critic_target", sac.critic_target),
        ):
            for _w in _m.trainable_variables:
                _arr = _w.numpy()
                if not np.all(np.isfinite(_arr)):
                    _nn = int(np.sum(~np.isfinite(_arr)))
                    return f"{_tag}/{_w.name} shape={list(_arr.shape)} non-finite {_nn}/{_arr.size}"
        if not np.isfinite(sac.log_alpha.numpy()):
            return f"log_alpha = {sac.log_alpha.numpy()}"
        return ""
    except Exception as _e:  # noqa: BLE001
        return f"Weight check exception: {_e}"


def run_path_a(
    model,
    sac: SACTrainer,
    env: MDSimulationEnv,
    tokens, mask, z_mask,
    n_steps: int,
    m5_anchor: float = 0.0,
    recovery_mode: bool = False,
):
    state = env.state()
    crashed = False
    survive_steps = n_steps
    m5_sum = 0.0
    m5_n = 0
    z = encode_z(model, tokens, state["env"], mask)
    # 2026-08-15 Root-cause probe: checks if encode_z output contains NaN (prime suspect for SAC loss full of NaNs on HPC clusters).
    # Only prints once when NaNs are encountered to prevent log flooding.
    if not np.all(np.isfinite(z)):
        logger.warning(f"[probe] ⚠️ encode_z contains non-finite values! env={state['env']} "
                       f"ph={env.env_ph:.3f} temp={env.env_temp:.1f} "
                       f"n_nan={np.sum(~np.isfinite(z))}/{z.size} n_inf={np.sum(np.isinf(z))}/{z.size}")
        z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
    for step_idx in range(n_steps):
        action_cont, action_disc = sac.act(
            z, state["env"], z_mask, deterministic=False
        )
        if recovery_mode:
            # 2026-08-18 Env Escape: freeze env-offset (zero offset) during the recovery period to keep the environment mild.
            # Runs the parent structure for full steps to populate the SAC buffer with healthy transitions, while training the env-offset head back toward 0.
            action_cont = np.asarray(action_cont, np.float32).copy()
            action_cont[env.force_dim: env.force_dim + env.env_offset_dim] = 0.0
        next_state, reward, done, info = env.step(action_cont)
        m5 = info.get("m5")
        if m5 is not None:
            m5_sum += float(m5)
            m5_n += 1

        next_z = encode_z(model, tokens, next_state["env"], mask)
        if sac.collect(
            {
                "z": z, "env": state["env"], "M": state["M"], "u_hist": state["u_hist"],
                "action_cont": action_cont, "action_disc": action_disc,
                "mutation_mask": 0.0,
                "z_mask": z_mask,
                "reward": reward, "done": done,
                "next_z": next_z, "next_env": next_state["env"],
                "next_M": next_state["M"], "next_u_hist": next_state["u_hist"],
            }
        ):
            sac.update(z_mask)
        state = next_state
        z = next_z     
        if done:
            crashed = bool(info["crashed"])
            survive_steps = step_idx + 1
            break
    m5_unstable = False
    if m5_anchor > 0 and m5_n > 0:
        m5_mean = m5_sum / m5_n
        thr = getattr(env.cfg, "m5_ratio_threshold", 1.3)
        m5_unstable = m5_mean >= thr * m5_anchor
    env_fail = env.current_env() if (crashed or m5_unstable) else None
    return crashed or m5_unstable, env_fail, survive_steps


def path_b_search(
    model,
    sac,
    es,
    base_seq: str,
    env_fail,
    tokens, mask, z_mask,
    base_atoms,               
    pseudo_label_dir: str,
    survive_steps: int,
    env_cfg,
    tag: str = "",           
    ep: int = -1,             
    cand_log: str = "",      
    explosive: dict = None,   # Restart Gate shared state across episodes (fully automated, 2026-08-16)
):
    import spice_engine as se
    from spice_pre.data.preprocessing import seq_to_tokens

    os.makedirs(pseudo_label_dir, exist_ok=True)
    env_norm = _normalize_env(env_fail, env_cfg)

    # Restart Gate (2026-08-16, fully automated discovery): no pre-configured blacklist; RL learns dynamically.
    # `explosive` is a shared dictionary across episodes containing:
    #   counts: {(pos,wt,mut,env_bucket): crash_count}  blacklist: keys meeting threshold
    # Sources of collapse: (1) numerical explosion during equilibration (U -> 1e8+ / NaN); (2) early simulation step collapse (steps <= min_steps).
    # Blacklisted candidates: skip build (saving core hours) and set fitness = penalty (worse than 0 survivors, directing the ES to avoid them).
    # 2026-08-17: env_bucket division (acid/neutral/base) — collapses only accumulate/trigger within the same bucket to prevent cross-environmental false positives.
    _ex = explosive or {}
    _ex_on = bool(_ex.get("on", False))
    _ex_thr = int(_ex.get("threshold", 2))
    _ex_min = int(_ex.get("min_steps", 3))
    _ex_pen = float(_ex.get("penalty", -1.0))
    _ex_counts = _ex.setdefault("counts", {})
    _ex_black = _ex.setdefault("blacklist", set())
    _ex_use_bucket = bool(_ex.get("env_bucket", True))

    def _env_bucket(env):
        """Environment bucket division: acid (pH < 5) / neutral (5 <= pH <= 8) / base (pH > 8). Collapses are only accumulated within the same bucket."""
        ph = float(env[0])
        if ph < 5.0:
            return "acid"
        if ph > 8.0:
            return "base"
        return "neutral"

    _ex_bucket = _env_bucket(env_fail) if _ex_use_bucket else ""

    def _mut_set(ms):
        """Set of single-point mutations {(pos, wt, mut)} relative to parent sequence (1-based position, mutation-level resolution)."""
        return {(i + 1, base_seq[i], ms[i])
                for i in range(len(base_seq)) if base_seq[i] != ms[i]}

    def _ex_key(ms_k):
        """(pos, wt, mut) -> blacklist key (including env_bucket)."""
        return ms_k + (_ex_bucket,) if _ex_use_bucket else ms_k

    def _ex_hit(ms):
        if not _ex_on:
            return False
        return bool({_ex_key(k) for k in _mut_set(ms)} & _ex_black)

    def _ex_register(ms, reason):
        """Register collapsed candidates: increments mutation components count +1; keys >= threshold are blacklisted.
        New blacklisted entries are exported to explosive_blacklist.csv with their respective env_bucket."""
        if not _ex_on:
            return
        prev = set(_ex_black)
        for k in _mut_set(ms):
            kk = _ex_key(k)
            _ex_counts[kk] = _ex_counts.get(kk, 0) + 1
        for kk, n in list(_ex_counts.items()):
            if n >= _ex_thr:
                _ex_black.add(kk)
        new = _ex_black - prev
        if new:
            logger.warning(f"[restart-gate] Added {len(new)} items to explosive blacklist ({reason}, bucket={_ex_bucket or 'global'}): "
                           f"{sorted(f'{p}:{w}>{m}' for p, w, m, *_ in new)}")
            _ex_csv = _ex.get("csv_path", "")
            if _ex_csv:
                import csv as _csv
                _newf = not os.path.exists(_ex_csv)
                os.makedirs(os.path.dirname(_ex_csv), exist_ok=True)
                with open(_ex_csv, "a", newline="") as _f:
                    _w = _csv.writer(_f)
                    if _newf:
                        _w.writerow(["ep", "tag", "pos", "wt", "mut", "env_bucket", "count", "reason"])
                    for item in sorted(new):
                        p, w, m = item[0], item[1], item[2]
                        _w.writerow([ep, tag, p, w, m, _ex_bucket, _ex_counts[item], reason])

    candidates = es.propose_mutations(base_seq, tokens, env_norm, mask)

    from spice_rl.env.observables import native_contact_q
    from spice_rl.es.conservation import (
        conservation_vector, conserved_mask, rejects_masked, load_external_conservation,
    )
    _wt_ca = _native_ca(base_atoms)   # deduplicate altloc -> 116, matching engine dedup (2026-08-12 fix)
    _q_cutoff = float(getattr(env_cfg, "q_cutoff", 8.0))
    _q_gate = float(getattr(es.cfg, "q_gate", 0.5))
    _cmask = None
    if getattr(es.cfg, "conservation_mask", True):
        _ext = load_external_conservation(
            getattr(es.cfg, "conservation_external", "") or "", len(base_seq))
        _cons = conservation_vector(base_seq, external=_ext)
        _cmask = conserved_mask(_cons, getattr(es.cfg, "conservation_threshold", 0.80))

    # Parent Engine (solvent reuse, 2026-08-14): bulk candidates for the same parent are built using `mutate_with_solvent_reuse`
    # which reuses the parent solvent box (reducing build time from 30s -> <0.5s, including #7 dynamic re-neutralization and #8 water pruning).
    # Requires a newer engine version; legacy engines automatically fall back to candidates-wise full builds.
    _parent_eng = None
    if hasattr(se.Engine, "mutate_with_solvent_reuse"):
        try:
            _pstruct = build_mutant_structure_from_ca(base_atoms, base_seq)
            _parent_eng = se.Engine.build(
                _pstruct, float(env_fail[0]), float(env_fail[1]),
                float(env_cfg.pressure), float(env_cfg.ionic_default),
                int(env_cfg.relax_iters), float(env_cfg.tolerance),
            )
            logger.info(f"Parent engine ready (solvent reuse path enabled, {len(base_seq)} aa)")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Parent engine build failed, falling back to candidates-wise full builds: {e}")
            _parent_eng = None

    def _build_engine(struct_):
        if _parent_eng is not None:
            return _parent_eng.mutate_with_solvent_reuse(
                struct_, float(env_fail[0]), float(env_fail[1]),
                float(env_cfg.pressure), float(env_cfg.ionic_default),
                int(env_cfg.relax_iters), float(env_cfg.tolerance),
            )
        return se.Engine.build(
            struct_, float(env_fail[0]), float(env_fail[1]),
            float(env_cfg.pressure), float(env_cfg.ionic_default),
            int(env_cfg.relax_iters), float(env_cfg.tolerance),
        )

    survivors = []
    _seen_surv: set = set()   # 2026-08-14: deduplication, prevents the same mutation from being written multiple times (duplicates seen in ep1 1:M>K logging)
    fitness = np.zeros(len(candidates), np.float32)
    conf_samples = []
    for j, (mut_seq, _k, _strategy) in enumerate(candidates):
        if mut_seq == base_seq or not _validate(mut_seq):
            fitness[j] = 0.0
            continue
        if _cmask is not None and rejects_masked(base_seq, mut_seq, _cmask):
            fitness[j] = 0.0
            continue
        _mut_str = ";".join(f"{i + 1}:{base_seq[i]}>{mut_seq[i]}"
                            for i in range(len(base_seq)) if base_seq[i] != mut_seq[i])
        if _ex_hit(mut_seq):
            fitness[j] = _ex_pen
            logger.info(f"[restart-gate] Blacklist hit, skipping build (fitness={_ex_pen}): {_mut_str}")
            continue
        try:
            tok_mut = seq_to_tokens(mut_seq)
            Lm = tok_mut.shape[0]
            mask_mut = np.ones(Lm, np.float32)
            z_mut = encode_z(model, tok_mut, env_norm, mask_mut)
            z_mask_mut = np.zeros(z_mask.shape, np.float32)
            z_mask_mut[:Lm] = mask_mut
            pred_ca = predict_mutant_coords(model, tok_mut, env_norm, mask_mut)
            if _sane_ca(pred_ca, _wt_ca):
                use_pred_ca = pred_ca
            else:
                use_pred_ca = _rescale_pred_ca(pred_ca, _wt_ca)
                if use_pred_ca is not None:
                    logger.info(f"[note] pred_ca Rg anomaly, scaling to wild Rg before use: {_mut_str}")
                else:
                    use_pred_ca = None
                    logger.warning(f"pred_ca geometric anomaly (unrecoverable via scaling), falling back to wild Cα backbone: {_mut_str}")
            try:
                struct = build_mutant_structure_from_ca(base_atoms, mut_seq, use_pred_ca)
                eng = _build_engine(struct)
            except Exception:
                try:
                    struct = build_mutant_structure_from_ca(base_atoms, mut_seq)
                    eng = _build_engine(struct)
                except Exception as e_fb:  # noqa: BLE001
                    logger.error(f"Mutation build failed (both pred_ca and wild-type failed): {e_fb}")
                    fitness[j] = 0.0
                    continue
            try:
                eng.equilibrate()
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(f"mutant equilibrate failed: {e2}") from e2
            steps = 0
            _early_crash = False
            for _ in range(survive_steps):
                a_cont, _ = sac.act(z_mut, env_norm, z_mask_mut, deterministic=True)
                _f = a_cont[: env_cfg.force_dim].astype(np.float32)
                _fclamp = float(getattr(env_cfg, "force_clamp", 0.5))
                if _fclamp > 0:
                    _f = np.clip(_f, -_fclamp, _fclamp)
                out = eng.step(_f)
                steps += 1
                if out["crashed"]:
                    _early_crash = steps <= _ex_min
                    break
            pseudo = np.asarray(eng.pseudo_labels(), np.float32) if steps >= 1 else None
            q_mut = 0.0
            _q_skip_reason = None
            if pseudo is not None and len(_wt_ca) == pseudo.shape[0]:
                q_mut = native_contact_q(pseudo, reference=_wt_ca, cutoff=_q_cutoff)
            elif pseudo is not None:
                # 2026-08-14: pseudo_labels empty or length-mismatched (probabilistic numerical explosion/NaNs, unreproducible locally,
                # SE untouched -> non-version issue). Rescue high-quality survivors by falling back to the current coords_ca frame:
                # Only accept if Q passes the gate (conservative, avoids injecting bad labels); record as failures otherwise.
                _fb = np.asarray(eng.coords_ca(), np.float32)
                if len(_fb) == len(_wt_ca) and np.all(np.isfinite(_fb)):
                    q_mut = native_contact_q(_fb, reference=_wt_ca, cutoff=_q_cutoff)
                    if q_mut >= _q_gate:
                        pseudo = _fb
                        logger.info(f"[note] pseudo_labels empty, falling back to coords_ca frame (q={q_mut:.2f}): {_mut_str}")
                if pseudo is None or len(_wt_ca) != pseudo.shape[0]:
                    _q_skip_reason = (f"q_skip: _wt_ca({len(_wt_ca)}) != "
                                      f"pseudo({pseudo.shape[0] if pseudo is not None else 0})")
            if _q_skip_reason is not None:
                # Safety net: explicitly alert if still unusable (2026-08-12: silent q=0 once led to two rounds of 0 survivors)
                logger.warning(f"{_q_skip_reason} (altloc or missing residues; please inspect this protein): {_mut_str}")
                if cand_log:  # 2026-08-14: record to failures CSV instead of silent discard (q=0 loses survivors)
                    import csv as _csv
                    _fl = cand_log.rsplit(".csv", 1)[0] + "_failures.csv"
                    _new = not os.path.exists(_fl)
                    os.makedirs(os.path.dirname(_fl), exist_ok=True)
                    with open(_fl, "a", newline="") as _f:
                        _w = _csv.writer(_f)
                        if _new:
                            _w.writerow(["ep", "tag", "mutations", "mut_seq", "strategy", "reason"])
                        _w.writerow([ep, tag, _mut_str, mut_seq, _strategy, _q_skip_reason])
            fitness[j] = float(steps) * q_mut
            if _ex_on and _early_crash:
                _ex_register(mut_seq, f"Early collapse (steps={steps})")
                fitness[j] = _ex_pen
            if cand_log:
                import csv as _csv
                _new = not os.path.exists(cand_log)
                os.makedirs(os.path.dirname(cand_log), exist_ok=True)
                with open(cand_log, "a", newline="") as _f:
                    _w = _csv.writer(_f)
                    if _new:
                        _w.writerow(["ep", "tag", "mutations", "mut_seq", "strategy", "fitness", "q", "survived", "ph", "temp"])
                    _w.writerow([ep, tag, _mut_str, mut_seq, _strategy, round(float(fitness[j]), 3),
                                 round(float(q_mut), 3),
                                 int(steps >= survive_steps and q_mut >= _q_gate),
                                 round(float(env_fail[0]), 3), round(float(env_fail[1]), 3)])
            conf_b = min(1.0, steps / max(1, survive_steps))
            conf_samples.append((z_mut, np.array([0.0, conf_b], np.float32)))
            if steps >= survive_steps and q_mut >= _q_gate:
                if mut_seq in _seen_surv:
                    # 2026-08-14: deduplication (prevents double writing as seen in ep1 1:M>K), avoiding double-weighting upon replay
                    logger.warning(f"Duplicate survivor in this episode, skipping duplicate write: {mut_seq}")
                    continue
                _seen_surv.add(mut_seq)
                _pre = f"{tag}_" if tag else ""
                # 🔴 2026-08-14 Overwrite Bug Fix: the survivors list inside path_b_search used to reset each episode,
                # meaning len(survivors) started from 0 -> pseudo_{tag}{i}_{steps}.npz was overwritten by subsequent episodes,
                # leaving pseudo_label_dir in a mixed state (cross-environmental/cross-parent mixtures, unusable as a batch).
                # Adding `ep` to the filename ensures global uniqueness: format is pseudo_{tag}_ep{ep:03d}_{idx}_{steps}.npz
                fn = os.path.join(
                    pseudo_label_dir,
                    f"pseudo_{_pre}ep{ep:03d}_{len(survivors)}_{steps}.npz",
                )
                np.savez(fn, seq=mut_seq, env=np.array(env_fail), coords=pseudo)
                survivors.append((mut_seq, steps))
                logger.info(f"Survivor mutant: {mut_seq} (steps={steps}, Q={q_mut:.2f}, {_strategy}) pseudo-label -> {fn}")
        except Exception as e:
            logger.error(f"Mutation evaluation failed: {e}")
            fitness[j] = 0.0
            if _ex_on and "equilibrate failed" in str(e):
                _ex_register(mut_seq, "equilibrate explosion")
                fitness[j] = _ex_pen
            if cand_log:
                import csv as _csv
                _fl = cand_log.rsplit(".csv", 1)[0] + "_failures.csv"
                _new = not os.path.exists(_fl)
                os.makedirs(os.path.dirname(_fl), exist_ok=True)
                with open(_fl, "a", newline="") as _f:
                    _w = _csv.writer(_f)
                    if _new:
                        _w.writerow(["ep", "tag", "mutations", "mut_seq", "strategy", "reason"])
                    _w.writerow([ep, tag, _mut_str, mut_seq, _strategy, str(e)[:200]])
    best = max(survivors, key=lambda x: x[1], default=None)
    return survivors, (best[0] if best else base_seq), fitness, conf_samples


def _normalize_env(env_raw, env_cfg):
    ph_denom = max(env_cfg.ph_max - env_cfg.ph_min, 1e-5)
    t_denom = max(env_cfg.temp_max - env_cfg.temp_min, 1e-5)
    ph_n = (env_raw[0] - env_cfg.ph_min) / ph_denom
    t_n = (env_raw[1] - env_cfg.temp_min) / t_denom
    i_n = float(np.clip(np.log10(max(env_raw[2], 1e-3) / 1e-3) / 3.0, 0.0, 1.0))
    return np.array([ph_n, t_n, i_n], np.float32)


def _validate(seq: str) -> bool:
    import spice_engine as se

    try:
        se.validate_sequence(seq)
        return True
    except Exception:
        return False


def _coverage_log(path: str, rows) -> None:
    import csv as _csv

    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as _f:
        _w = _csv.writer(_f)
        if new:
            _w.writerow(["ep", "ph", "temp", "kind"])
        for _r in rows:
            _w.writerow(_r)


def train(cfg: Config, structure, base_seq: str, base_atoms: dict = None, tag: str = ""):
    os.makedirs(cfg.post.log_dir, exist_ok=True)
    os.makedirs(cfg.post.ckpt_dir, exist_ok=True)

    # 2026-08-19 Eager fallback: when containerized TF graph mode has issues, execute both SAC updates and encode_z in eager mode
    from spice_rl.config import set_eager_mode
    set_eager_mode(getattr(cfg.sac, "eager_update", False))

    if len(base_seq) > cfg.sac.discrete_position_dim:
        raise SystemExit(
            f"Sequence too long: {len(base_seq)} aa > discrete_position_dim="
            f"{cfg.sac.discrete_position_dim} (truncation would mismatch sequence and coordinates)"
        )
    if len(base_seq) < getattr(cfg.post, "min_seq_len", 80):
        raise SystemExit(
            f"Sequence too short: {len(base_seq)} aa < min_seq_len="
            f"{cfg.post.min_seq_len} (too short to learn folded structure, skipping)"
        )

    import inspect as _insp
    try:
        from spice_rl.env.md_env import MDSimulationEnv as _MD
        from spice_rl.sac.networks import TwinCritic as _TC

        def _src(_obj) -> str:
            # For @tf.function decorated objects (e.g., _update_tf), use python_function to retrieve original source code
            _raw = getattr(_obj, "python_function", _obj)
            return _insp.getsource(_raw)

        _md_step = _src(_MD.step)
        _has_norm = "reward_ref" in _md_step
        _has_rclip = "np.clip(reward, -10.0, 10.0)" in _md_step
        _has_clamp = "env_offset_clamp" in _insp.getsource(_MD)
        _has_ps = "place_sidechain" in _insp.getsource(_mutant_atoms)
        _has_pred = "pred_ca: np.ndarray = None" in _insp.getsource(build_mutant_structure_from_ca)
        _has_u_ref = "u_ref" in _insp.getsource(_TC._feats)
        # 2026-08-19: sac.py refactors _update_tf into _update_impl (the actual safeguard block) + a @tf.function wrapper 
        # (for eager_update). Self-check must inspect _update_impl to avoid false positives about "missing safeguards". 
        # Falls back to _update_tf if _update_impl is absent.
        _sac_src = _src(getattr(SACTrainer, "_update_impl", None) or SACTrainer._update_tf)
        _has_sac_clean = "tf.math.is_finite(v)" in _sac_src
        _has_sac_clip = "tf.clip_by_value(y" in _sac_src
        _has_sac_qclip = "tf.clip_by_value(q" in _sac_src
        _has_sac_ywash = "tf.math.is_finite(y)" in _sac_src
        _has_sac_ngrad = "tf.math.is_finite(g)" in _sac_src
        logger.info(f"[self-check] reward normalization: {_has_norm} | reward ±10 clip: {_has_rclip} | clamp A+B: {_has_clamp} | place_sidechain: {_has_ps} | pred_ca default: {_has_pred}")
        logger.info(f"[self-check] sac safeguards: u_hist normalization: {_has_u_ref} | input sanitization: {_has_sac_clean} | y/q clip: {_has_sac_clip} | q clip: {_has_sac_qclip} | y fallback: {_has_sac_ywash} | NaN gradient zeroing: {_has_sac_ngrad}")
        if not _has_norm:
            logger.warning("md_env.step lacks reward_ref normalization -> critic_loss 1e10+, SAC stalls! Please upload the latest md_env.py")
        if not _has_rclip:
            logger.warning("md_env.step lacks reward ±10 clip (2026-08-17) -> reward during energy spikes can reach ±100, potentially exploding the critic! Please upload the latest md_env.py")
        if not _has_clamp:
            logger.warning("md_env lacks env_offset_clamp -> environment offsets may exceed engine limits! Please upload the latest md_env.py")
        if not _has_ps:
            logger.warning("_mutant_atoms lacks place_sidechain -> legacy conservative filtering is active! Please upload the latest train_post.py")
        if not _has_pred:
            logger.warning("pred_ca lacks default value parameter -> legacy train_post.py active! Please upload the latest train_post.py")
        if not _has_u_ref:
            logger.warning("networks.py _feats lacks u_hist normalization (u_ref) -> critic_loss 1e10+! Please upload the latest networks.py")
        if not _has_sac_clean:
            logger.warning("sac.py _update_tf lacks input isfinite sanitization (2026-08-15) -> NaN values propagate, destroying SAC! Please upload the latest sac.py")
        if not _has_sac_clip:
            logger.warning("sac.py _update_tf lacks y/q clip (2026-08-15) -> critic targets can explode to 1e12+, leading to divergence! Please upload the latest sac.py")
        if not _has_sac_qclip:
            logger.warning("sac.py _update_tf lacks actor-branch q clipping (2026-08-18) -> critic output diverges unboundedly! Please upload the latest sac.py")
        if not _has_sac_ywash:
            logger.warning("sac.py _update_tf lacks y isfinite validation fallback (2026-08-18) -> NaNs might leak into loss calculation! Please upload the latest sac.py")
        if not _has_sac_ngrad:
            logger.warning("sac.py _update_tf lacks NaN gradient zeroing -> a single NaN gradient poisons all weights! Please upload the latest sac.py")
    except Exception as _e:  # noqa: BLE001
        logger.info(f"[self-check] Skipped ({_e})")

    check = quick_check_env(
        structure, cfg.env, cfg.post.anchor_ph, cfg.post.anchor_temp
    )
    if not check["ok"]:
        logger.error(f"Quick-check failed ({check['reason']}), terminating. Please choose a different starting structure.")
        return
    logger.info(f"Quick-check passed: U={check['u']:.1f} kcal/mol, initiating dual-path exploration")

    model = build_rl_model(cfg)
    L_max = cfg.sac.discrete_position_dim
    cont_dim = cfg.env.force_dim + cfg.env.env_offset_dim
    cfg.sac.u_ref = getattr(cfg.env, "reward_ref", 1e5)
    sac = SACTrainer(
        cfg.sac, z_dim=cfg_sac_z_dim(model), cont_dim=cont_dim, u_window=cfg.env.u_window
    )
    from spice_rl.es import ESEvolver

    es = ESEvolver(model, cfg.es)
    conf_trainer = ConfidenceHeadTrainer(model, lr=cfg.post.conf_lr)

    tokens, mask = tokens_from_seq(base_seq, L_max)
    z_mask = np.zeros(L_max, np.float32)
    z_mask[: len(mask)] = mask

    wt = base_seq
    start = time.time()
    env_plus = MDSimulationEnv(
        structure, cfg.env,
        ph=cfg.post.anchor_ph + cfg.post.env_delta_ph,
        temp=cfg.post.anchor_temp + cfg.post.env_delta_T,
        ionic=cfg.env.ionic_default,
        reuse_engine=cfg.post.reuse_engine,
    )
    env_minus = MDSimulationEnv(
        structure, cfg.env,
        ph=cfg.post.anchor_ph - cfg.post.env_delta_ph,
        temp=cfg.post.anchor_temp - cfg.post.env_delta_T,
        ionic=cfg.env.ionic_default,
        reuse_engine=cfg.post.reuse_engine,
    )
    _cov_path = os.path.join(cfg.post.log_dir, "coverage.csv")
    _coverage_log(_cov_path, [
        (-1, float(cfg.post.anchor_ph), float(cfg.post.anchor_temp), "anchor"),
        (-1, float(env_plus.env_ph), float(env_plus.env_temp), "pathA_plus"),
        (-1, float(env_minus.env_ph), float(env_minus.env_temp), "pathA_minus"),
    ])
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None
    _pbar = (
        _tqdm(total=cfg.post.max_episodes, desc="RL post-train", unit="ep",
              dynamic_ncols=True, leave=True)
        if _tqdm is not None
        else None
    )
    try:
        import tensorboard  # noqa: F401
        _tb_ok = True
    except ImportError:
        _tb_ok = False
        logger.warning("tensorboard is not installed, skipping TensorBoard curves (CSV metrics continue logging normally)")
    tb_writer = (
        tf.summary.create_file_writer(os.path.join(cfg.post.log_dir, "tensorboard"))
        if _tb_ok else None
    )
    metrics = MetricsLogger(
        os.path.join(cfg.post.log_dir, "metrics.csv"),
        fields=["ep", "t", "alpha", "buffer", "a_survive", "a_crashed",
                "n_survivors", "conf_loss", "critic_loss", "actor_loss", "alpha_loss"],
    )
    m5_anchor = 0.0
    try:
        aq = quick_check_env(
            structure, cfg.env, cfg.post.anchor_ph, cfg.post.anchor_temp, n_steps=60
        )
        if aq["ok"]:
            m5_anchor = aq.get("m5_mean") or 0.0
            logger.info(f"[anchor] m5 baseline: {m5_anchor:.3f}")
        else:
            logger.warning(f"anchor m5 probe failed ({aq.get('reason', 'unknown')}) -> m5 triggering disabled, "
                           f"Path B relies solely on physical collapse triggering (m5_anchor=0.0)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"anchor m5 baseline measurement failed (m5 triggering disabled): {e}")

    # Parent history stack (2026-08-15 Scheme B): when drifting parent sequences exhibit continuous 0 survival 
    # under a fixed Env_fail, backtrack to the last parental generation that produced survivors, rather than 
    # misinterpreting saturation of iterations as "unrescuable parent" and aborting prematurely.
    parent_history: list = []

    # Restart Gate cross-episode state (2026-08-16, fully automated): no pre-configured blacklist; 
    # RL compiles collapse statistics automatically (equilibration explosions / early simulation collapses). 
    # Accumulations >= explosive_threshold are blacklisted, skipping structure builds and setting fitness = explosive_penalty.
    # 2026-08-17: Blacklist subdivided by environment buckets (explosive_env_bucket) — early collapse/explosions often occur 
    # in specific pH ranges (e.g., pH 10 base collapse). Without buckets, "collapse in alkaline environment only" would be misregistered as global.
    explosive_state = {
        "on": bool(getattr(cfg.post, "restart_gate", True)),
        "threshold": int(getattr(cfg.post, "explosive_threshold", 2)),
        "min_steps": int(getattr(cfg.post, "explosive_min_steps", 3)),
        "penalty": float(getattr(cfg.post, "explosive_penalty", -1.0)),
        "env_bucket": bool(getattr(cfg.post, "explosive_env_bucket", True)),
        "counts": {},
        "blacklist": set(),
        "csv_path": os.path.join(cfg.post.log_dir, "explosive_blacklist.csv"),
    }

    def _rebuild_parent(new_wt):
        """Reconstruct parent state: structure + env_plus/env_minus + tokens/mask/z_mask.
        Returns (structure, env_plus, env_minus, tokens, mask, z_mask) if successful, None otherwise (preserves legacy state)."""
        try:
            n_tokens, n_mask = tokens_from_seq(new_wt, L_max)
            n_z_mask = np.zeros(L_max, np.float32)
            n_z_mask[: len(n_mask)] = n_mask
            env_norm_anchor = _normalize_env(
                (cfg.post.anchor_ph, cfg.post.anchor_temp, cfg.env.ionic_default),
                cfg.env,
            )
            n_pred_ca = predict_mutant_coords(model, n_tokens, env_norm_anchor, n_mask)
            if base_atoms is not None:
                _wt_ca = _native_ca(base_atoms)  # deduplicate altloc (2026-08-12 fix)
                if not _sane_ca(n_pred_ca, _wt_ca):
                    logger.warning("Parent pred_ca is anomalous, falling back to wild backbone")
                    n_pred_ca = None
            n_struct = build_mutant_structure_from_ca(base_atoms, new_wt, n_pred_ca)
            n_plus = MDSimulationEnv(
                n_struct, cfg.env,
                ph=cfg.post.anchor_ph + cfg.post.env_delta_ph,
                temp=cfg.post.anchor_temp + cfg.post.env_delta_T,
                ionic=cfg.env.ionic_default,
                reuse_engine=cfg.post.reuse_engine,
            )
            n_minus = MDSimulationEnv(
                n_struct, cfg.env,
                ph=cfg.post.anchor_ph - cfg.post.env_delta_ph,
                temp=cfg.post.anchor_temp - cfg.post.env_delta_T,
                ionic=cfg.env.ionic_default,
                reuse_engine=cfg.post.reuse_engine,
            )
            return n_struct, n_plus, n_minus, n_tokens, n_mask, n_z_mask
        except Exception as _e:  # noqa: BLE001
            logger.error(f"Parent structure reconstruction failed: {_e}")
            return None

    no_survivor_streak = 0
    # Env Escape & Recovery (2026-08-18): prevents deadlocks when the agent pushes Path A into an unstable "basin of immediate collapse" 
    # (e.g., pH and temperature limits), leading to step-1 crashes. This causes the replay buffer to add only 1 transition per episode, 
    # stalling SAC updates and deadlocking training.
    # If the agent suffers "early collapse and 0 survivors" for `escape_after` consecutive episodes, training switches to a mild environment 
    # (anchor ± recovery_delta_*) with frozen environment offsets for `recovery_episodes` episodes. This feeds healthy transitions into 
    # the replay buffer, breaking the deadlock before resuming active environmental exploration.
    early_crash_streak = 0
    recovery_remaining = 0
    _esc_thr = int(getattr(cfg.post, "escape_step_threshold", 5))
    _esc_after = int(getattr(cfg.post, "escape_after", 3))
    _rec_eps = int(getattr(cfg.post, "recovery_episodes", 4))
    _rec_dph = float(getattr(cfg.post, "recovery_delta_ph", 1.0))
    _rec_dT = float(getattr(cfg.post, "recovery_delta_T", 5.0))

    def _make_envs(ph_delta, T_delta):
        """Constructs env_plus/env_minus by ± offset (uses current structure; automatically updates with the latest upon parent adoption)."""
        _plus = MDSimulationEnv(
            structure, cfg.env,
            ph=cfg.post.anchor_ph + ph_delta,
            temp=cfg.post.anchor_temp + T_delta,
            ionic=cfg.env.ionic_default,
            reuse_engine=cfg.post.reuse_engine,
        )
        _minus = MDSimulationEnv(
            structure, cfg.env,
            ph=cfg.post.anchor_ph - ph_delta,
            temp=cfg.post.anchor_temp - T_delta,
            ionic=cfg.env.ionic_default,
            reuse_engine=cfg.post.reuse_engine,
        )
        return _plus, _minus

    for ep in range(cfg.post.max_episodes):
        # Recovery epoch wrap-up: checks if this episode is in recovery mode (frozen env-offset); returns to standard exploration env once finished
        in_recovery = recovery_remaining > 0
        if recovery_remaining > 0:
            recovery_remaining -= 1
            if recovery_remaining == 0:
                try:
                    env_plus, env_minus = _make_envs(cfg.post.env_delta_ph, cfg.post.env_delta_T)
                    logger.info(f"[recovery] Recovery epoch concluded, reverting to standard exploration environments (ΔpH={cfg.post.env_delta_ph})")
                except Exception as _e:  # noqa: BLE001
                    logger.error(f"[recovery] Recovery epoch concluded but failed to rebuild standard environments, continuing with recovery envs: {_e}")
        env_fail = None
        conf_a_steps = 0
        n_survivors = 0
        last_conf_loss = float("nan")
        for sign in (+1, -1):
            env = env_plus if sign > 0 else env_minus
            try:
                env.reset()
            except Exception as e:  # noqa: BLE001
                logger.error(f"[ep {ep}] Failed to build parent structure, skipping sign={sign:+d}: {e}")
                continue
            crashed, fail, survive = run_path_a(
                model, sac, env, tokens, mask, z_mask,
                n_steps=cfg.env.episode_max_steps,
                m5_anchor=m5_anchor,
                recovery_mode=in_recovery,
            )
            conf_a_steps = max(conf_a_steps, survive)
            if crashed:
                env_fail = fail
                logger.warning(f"[ep {ep}] Path A became unstable (sim collapsed / m5 threshold exceeded), recording Env_fail: {env_fail}")
                _coverage_log(_cov_path, [(ep, float(env_fail[0]), float(env_fail[1]), "env_fail")])
                break
        z_wt = encode_z(
            model, tokens,
            _normalize_env(
                (cfg.post.anchor_ph, cfg.post.anchor_temp, cfg.env.ionic_default),
                cfg.env,
            ),
            mask,
        )
        conf_trainer.add(
            z_wt,
            np.array(
                [conf_a_steps / max(1, cfg.env.episode_max_steps), 0.0], np.float32
            ),
        )
        if len(sac.buffer) >= cfg.sac.batch_size:
            sac.update(z_mask)
        # 2026-08-18 NaN Watchdog: SAC weights contain non-finite values -> reconstruct SAC (resets buffer to prevent deadlocks in later training stages).
        # Upgrade 2026-08-18: logs which network/weight exhibited non-finite values + the most recent loss, facilitating root-cause analysis.
        if getattr(cfg.post, "nan_watchdog", True):
            _nan_diag = _sac_nan_report(sac)
            if _nan_diag:
                _last = getattr(sac, "last_losses", None)
                logger.warning(
                    f"[NaN-watchdog] ep {ep} SAC weights are non-finite: {_nan_diag} | "
                    f"last_losses={_last} -> Reconstructing SAC (resetting replay buffer)"
                )
                sac = SACTrainer(
                    cfg.sac, z_dim=cfg_sac_z_dim(model), cont_dim=cont_dim, u_window=cfg.env.u_window
                )
                sac.ensure_built()

        if cfg.post.phase_map_interval > 0 and ep % cfg.post.phase_map_interval == 0:
            try:
                pts = scan_phase_map(
                    structure,
                    cfg.post.phase_map_temp_range,
                    cfg.post.phase_map_ph_range,
                    pressure=cfg.env.pressure,
                    ionic=cfg.env.ionic_default,
                    relax_iters=cfg.env.relax_iters,
                    tolerance=cfg.env.tolerance,
                )
                s = summarize_phase_map(pts)
                out = os.path.join(cfg.post.phase_map_dir, f"phase_{ep:05d}.npz")
                save_phase_map(pts, out)
                logger.info(
                    f"[ep {ep}] Phase map: stable={s['n_stable']}/{s['n_points']} "
                    f"boundary={s['n_boundary']} crashed={s['n_crashed']} -> {out}"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"[ep {ep}] Phase map scanning failed: {e}")

        if env_fail is not None and base_atoms is not None:
            _wt_before = wt
            survivors, wt, fitness, conf_samples = path_b_search(
                model, sac, es, wt, env_fail, tokens, mask, z_mask,
                base_atoms, cfg.post.pseudo_label_dir, cfg.es.fitness_survive_steps,
                cfg.env, tag, ep, os.path.join(cfg.post.log_dir, "pathb_candidates.csv"),
                explosive=explosive_state,
            )
            n_survivors = len(survivors)
            for z_m, c in conf_samples:
                conf_trainer.add(z_m, c)
            if survivors:
                if wt != _wt_before:
                    parent_history.append(_wt_before)
                _rebuilt = _rebuild_parent(wt)
                if _rebuilt is not None:
                    structure, env_plus, env_minus, tokens, mask, z_mask = _rebuilt
                    recovery_remaining = 0  # Revert to standard exploration environments after adopting a new parent structure
                    logger.info(f"New parent structure reconstructed successfully: {wt[:20]}... (n_res={structure.residue_count()})")
                else:
                    logger.warning("Failed to reconstruct new parent structure (continuing with legacy structure/state)")
                try:
                    write_pseudo_tfrecord(
                        cfg.post.pseudo_label_dir,
                        cfg.post.pseudo_tfrecord_path,
                        L_max,
                        weight_repeat=cfg.post.pseudo_weight_repeat,
                        survive_steps=cfg.es.fitness_survive_steps,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Pseudo-label recycling failed: {e}")
            es.evolve(fitness)

        # Env Escape Trigger (2026-08-18): Path A exhibits continuous early collapses + 0 survivors = deadlocks.
        # Fall back to mild environment and freeze env-offsets to run a recovery epoch, feeding healthy transitions to break deadlocks before resuming exploration.
        if env_fail is not None and conf_a_steps < _esc_thr and n_survivors == 0:
            early_crash_streak += 1
        else:
            early_crash_streak = 0
        if early_crash_streak >= _esc_after and recovery_remaining <= 0:
            early_crash_streak = 0
            recovery_remaining = _rec_eps
            try:
                env_plus, env_minus = _make_envs(_rec_dph, _rec_dT)
                logger.warning(f"[recovery] Path A suffered early collapse (step < {_esc_thr}) and 0 survivors for {_esc_after} consecutive episodes"
                               f" -> falling back to mild environment (anchor ± {_rec_dph} pH / ± {_rec_dT} K) and freezing env-offset"
                               f" for a recovery period of {_rec_eps} episodes")
            except Exception as _e:  # noqa: BLE001
                recovery_remaining = 0
                logger.error(f"[recovery] Failed to construct mild environment, skipping recovery: {_e}")

        if env_fail is not None and n_survivors == 0:
            no_survivor_streak += 1
        else:
            no_survivor_streak = 0
        _abort_eps = getattr(cfg.post, "no_survivor_abort", 15)
        if no_survivor_streak >= _abort_eps:
            # 2026-08-15 Scheme B: 0 survivors/saturation != unrescuable parent. Attempt backtracking to the last parent 
            # generation that successfully produced survivors (LIFO parent history stack); real runtime error only raised when history is empty.
            if parent_history:
                old_wt = parent_history.pop()
                logger.warning(f"[ep {ep}] Path B suffered {_abort_eps} consecutive episodes of 0 survivors -> backtracking to parent structure "
                               f"{old_wt[:20]}... ({len(parent_history)} remaining in stack)")
                _rebuilt = _rebuild_parent(old_wt)
                if _rebuilt is not None:
                    structure, env_plus, env_minus, tokens, mask, z_mask = _rebuilt
                    wt = old_wt
                    no_survivor_streak = 0
                    logger.info(f"✓ Backtracked successfully to parent {wt[:20]}... (n_res={structure.residue_count()})")
                else:
                    raise RuntimeError(
                        f"Unrescuable parent or structural explosion: Path B suffered {_abort_eps} consecutive episodes of 0 survivors "
                        f"(Env_fail={env_fail}, wt={wt[:20]}...), and backtracking to parent {old_wt[:20]} failed during reconstruction."
                    )
            else:
                raise RuntimeError(
                    f"Unrescuable parent or structural explosion: Path B suffered {_abort_eps} consecutive episodes of 0 survivors "
                    f"(Env_fail={env_fail}, wt={wt[:20]}...), and the parent history stack is empty. "
                    f"Please choose a starting protein with milder stability boundaries and re-run."
                )

        if cfg.post.conf_train_interval > 0 and ep % cfg.post.conf_train_interval == 0:
            if len(conf_trainer) >= cfg.post.conf_batch:
                cl = conf_trainer.update(cfg.post.conf_batch)
                last_conf_loss = float(cl["conf_loss"])
                logger.info(f"[ep {ep}] Head D Confidence loss: {cl['conf_loss']:.4f}")

        if ep % cfg.post.log_every == 0:
            logger.info(
                f"[ep {ep}] alpha={sac.alpha():.3f} buffer={len(sac.buffer)} "
                f"| wt={wt[:20]} | {time.time()-start:.0f}s"
                + (" | RECOVERY" if recovery_remaining > 0 else "")
            )

        sac_losses = sac.last_losses or {}
        _tb_scalar(tb_writer, "alpha", sac.alpha(), step=ep)
        _tb_scalar(tb_writer, "buffer", len(sac.buffer), step=ep)
        _tb_scalar(tb_writer, "pathA_survive", conf_a_steps, step=ep)
        _tb_scalar(tb_writer, "pathA_crashed", 1 if env_fail is not None else 0, step=ep)
        _tb_scalar(tb_writer, "pathB_survivors", n_survivors, step=ep)
        _tb_scalar(tb_writer, "critic_loss", sac_losses.get("critic_loss", float("nan")), step=ep)
        _tb_scalar(tb_writer, "actor_loss", sac_losses.get("actor_loss", float("nan")), step=ep)
        _tb_scalar(tb_writer, "alpha_loss", sac_losses.get("alpha_loss", float("nan")), step=ep)
        _tb_scalar(tb_writer, "conf_loss", last_conf_loss, step=ep)
        metrics.add(
            ep=ep,
            t=round(time.time() - start, 1),
            alpha=sac.alpha(),
            buffer=len(sac.buffer),
            a_survive=conf_a_steps,
            a_crashed=1 if env_fail is not None else 0,
            n_survivors=n_survivors,
            conf_loss=last_conf_loss,
            critic_loss=sac_losses.get("critic_loss", float("nan")),
            actor_loss=sac_losses.get("actor_loss", float("nan")),
            alpha_loss=sac_losses.get("alpha_loss", float("nan")),
        )
        metrics.flush()

        if ep % cfg.post.ckpt_every == 0 or ep == cfg.post.max_episodes - 1:
            try:
                os.makedirs(cfg.post.ckpt_dir, exist_ok=True)
                if not model.built:
                    model(
                        {
                            "tokens": tf.zeros([1, 8], tf.int32),
                            "env": tf.zeros([1, 3]),
                            "mask": tf.ones([1, 8]),
                        },
                        training=False,
                    )
                model.save_weights(os.path.join(cfg.post.ckpt_dir, f"model_ep{ep:04d}.weights.h5"))
                sac.save(cfg.post.ckpt_dir, tag=f"sac_ep{ep:04d}")
                logger.info(f"[ep {ep}] checkpoint -> {cfg.post.ckpt_dir}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"[ep {ep}] checkpoint save failed: {e}")

        if _pbar is not None:
            if recovery_remaining > 0:
                _pbar.set_postfix(
                    buffer=len(sac.buffer),
                    alpha=f"{sac.alpha():.3f}",
                    crash="Y" if env_fail is not None else "N",
                    wt=wt[:8],
                    rec=recovery_remaining,
                )
            else:
                _pbar.set_postfix(
                    buffer=len(sac.buffer),
                    alpha=f"{sac.alpha():.3f}",
                    crash="Y" if env_fail is not None else "N",
                    wt=wt[:8],
                )
            _pbar.update(1)
    if _pbar is not None:
        _pbar.close()
    metrics.save()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


def cfg_sac_z_dim(model) -> int:
    return model.embed_dim


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/posttrain.yaml")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--parquet-dir", default=None,
                    help="Directory of all-atom Parquet files from data pipeline (production entry)")
    ap.add_argument("--pdb-id", default=None,
                    help="Target pdb_id to load from the Parquet files")
    ap.add_argument("--tag", default=None,
                    help="Output label tag used as prefix for pseudo-labels and rows in pathb_candidates. "
                         "Defaults to pdb_id.lower(); must be provided for multi-protein loops to avoid overwriting outputs")
    ap.add_argument("--structure", default=None, help="(Debugging) Path to input mmCIF structure file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.post.log_dir, "train.log")
    if args.max_episodes is not None:
        cfg.post.max_episodes = args.max_episodes

    if args.parquet_dir and args.pdb_id:
        from spice_rl.env import load_structure_with_atoms

        struct, base_atoms = load_structure_with_atoms(
            args.parquet_dir, args.pdb_id, max_residues=cfg.post.max_seq_len
        )
        seq = struct.sequence()
        logger.info(f"Loaded starting structure from data pipeline: {seq[:30]}... n_res={struct.residue_count()}")
        train(cfg, struct, seq, base_atoms=base_atoms,
              tag=(args.tag or args.pdb_id.lower()))
    elif args.structure:
        from spice_rl.env import structure_from_mmcif

        struct = structure_from_mmcif(args.structure)
        if struct.residue_count() > cfg.post.max_seq_len:
            raise SystemExit(
                f"Structure is too long: {struct.residue_count()} aa > max_seq_len="
                f"{cfg.post.max_seq_len} (RL is configured to process only sequences <= {cfg.post.max_seq_len} aa)"
            )
        seq = struct.sequence()
        logger.info(f"(Debugging) Loaded mmCIF starting structure: {seq[:30]}... n_res={struct.residue_count()}")
        train(cfg, struct, seq)
    else:
        raise SystemExit(
            "Execution requires either --parquet-dir + --pdb-id (data pipeline entry) or --structure (debugging mmCIF entry)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
