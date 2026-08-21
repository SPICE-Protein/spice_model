from __future__ import annotations

import collections
from typing import Dict, Optional, Tuple

import numpy as np

from spice_rl.config import EnvConfig
from spice_rl.env.observables import (
    native_contact_q,
    per_residue_rmsf,
    track_rmsf,
)

METRIC_KEYS = ["m1", "m2", "m3", "m4", "m5"]

TERMINAL_CRASH_REWARD = -100.0


def _engine():
    import spice_engine

    return spice_engine


class MDSimulationEnv:

    def __init__(
        self,
        structure,
        cfg: EnvConfig,
        ph: float = 7.0,
        temp: float = 298.0,
        ionic: float = 0.0,
        pressure: float = 0.0,  
        reuse_engine: bool = True,
        sequence: Optional[str] = None,
    ):
        self.structure = structure
        self.cfg = cfg
        self.se = _engine()
        self.reuse_engine = reuse_engine
        self._built_env: Optional[tuple] = None   
        self._needs_rebuild = False               
        self.force_dim = cfg.force_dim
        self.env_offset_dim = cfg.env_offset_dim
        self.act_dim = cfg.force_dim + cfg.env_offset_dim  
        self.mutation_every = cfg.mutation_every           
        self.ionic = ionic
        self.pressure = pressure
        self.sequence = sequence
        self.env_ph = float(ph)
        self.env_temp = float(temp)
        self.env_ionic = float(ionic)
        self._ph_dirty = False
        self._ionic_dirty = False
        self.n_episode_steps = 0
        self.engine = None  
        self._cached_metrics = None  
        self.u_history: collections.deque = collections.deque(
            maxlen=cfg.u_window
        )
        self._native_coords = None          
        self._coords_history: list = []     
        self.strain_total = 0               
        self.max_accel_clamped = 0.0        
        self.t_kin_last = 0.0               

    def _build(self, ph: float, temp: float, ionic: float):
        return self.se.Engine.build(
            self.structure,
            float(ph),
            float(temp),
            float(self.pressure),
            float(ionic),
            int(self.cfg.relax_iters),
            float(self.cfg.tolerance),
            strict_incomplete=getattr(self.cfg, "strict_incomplete", True),
        )

    def reset(self, ph=None, temp=None, ionic=None):
        if ph is not None:
            self.env_ph = float(ph)
        if temp is not None:
            self.env_temp = float(temp)
        if ionic is not None:
            self.env_ionic = float(ionic)
        needs = (
            not self.reuse_engine
            or self.engine is None
            or self._needs_rebuild
            or self._built_env is None
            or self.env_temp != self._built_env[1]
            or self.env_ionic != self._built_env[2]
            or self._ph_drift() >= self.cfg.ph_rebuild_threshold
        )
        if needs:
            try:
                engine = self._build(self.env_ph, self.env_temp, self.env_ionic)
                engine.equilibrate()
            except Exception as e:  # noqa: BLE001
                self.engine = None
                self._built_env = None
                self._needs_rebuild = True
                raise RuntimeError(f"Engine build/equilibrate failed: {e}") from e
            self.engine = engine
            self._built_env = (self.env_ph, self.env_temp, self.env_ionic)
            self._native_coords = self.engine.coords_ca()
            self._ph_dirty = False
            self._needs_rebuild = False
        else:
            self.engine.reset_velocities()
        if self.sequence is None and hasattr(self.engine, "sequence"):
            self.sequence = self.engine.sequence()
        self.n_episode_steps = 0
        self.u_history.clear()
        self._coords_history = []
        self.strain_total = 0
        self.max_accel_clamped = 0.0
        self.t_kin_last = 0.0
        self._cached_metrics = None
        self._ph_dirty = False
        self._ionic_dirty = False
        return self.state()

    def _ph_drift(self) -> float:
        built_ph = self._built_env[0] if self._built_env is not None else self.env_ph
        return abs(self.env_ph - built_ph)

    def apply_env_offset(self, dpH: float, dT: float) -> None:
        if not np.isfinite(dpH):
            dpH = 0.0
        if not np.isfinite(dT):
            dT = 0.0
        if getattr(self.cfg, "env_offset_clamp", True):
            dph_max = getattr(self.cfg, "env_dph_clamp", 2.0)
            dt_max = getattr(self.cfg, "env_dT_clamp", 20.0)
            dpH = float(np.clip(dpH, -dph_max, dph_max))
            dT = float(np.clip(dT, -dt_max, dt_max))
        self.env_ph = float(
            np.clip(self.env_ph + dpH, self.cfg.ph_min, self.cfg.ph_max)
        )
        self.env_temp = float(
            np.clip(self.env_temp + dT, self.cfg.temp_min, self.cfg.temp_max)
        )
        if getattr(self.cfg, "env_abs_window", True):
            self.env_ph = float(np.clip(
                self.env_ph,
                getattr(self.cfg, "env_ph_min", 2.0),
                getattr(self.cfg, "env_ph_max", 10.0),
            ))
            self.env_temp = float(np.clip(
                self.env_temp,
                getattr(self.cfg, "env_temp_min", 260.0),
                getattr(self.cfg, "env_temp_max", 330.0),
            ))
        self.engine.set_temperature(self.env_temp)
        self._ph_dirty = self._ph_drift() >= self.cfg.ph_rebuild_threshold

    def rebuild_if_dirty(self) -> None:
        if self._ph_dirty or self._ionic_dirty:
            self.engine = self._build(self.env_ph, self.env_temp, self.env_ionic)
            self._built_env = (self.env_ph, self.env_temp, self.env_ionic)
            self._ph_dirty = False
            self._ionic_dirty = False

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        force = action[: self.force_dim] if action.size >= self.force_dim else None
        _fclamp = float(getattr(self.cfg, "force_clamp", 0.5))
        if force is not None and _fclamp > 0:
            force = np.clip(force, -_fclamp, _fclamp)
        result = self.engine.step(
            None if force is None else force.astype(np.float32)
        )

        self.strain_total += int(result.get("n_clamped") or 0)
        self.max_accel_clamped = max(
            self.max_accel_clamped, float(result.get("max_accel_clamped") or 0.0)
        )
        self.t_kin_last = float(result.get("t_kin") or 0.0)
        if getattr(self.cfg, "q_track", True) and self._native_coords is not None:
            result["q"] = native_contact_q(
                self.engine.coords_ca(),
                reference=self._native_coords,
                cutoff=getattr(self.cfg, "q_cutoff", 8.0),
            )
        w = getattr(self.cfg, "rmsf_window", 0)
        if w and w > 0:
            track_rmsf(self._coords_history, self.engine.coords_ca(), int(w))

        if all(k in result for k in METRIC_KEYS):
            self._cached_metrics = {k: float(result[k]) for k in METRIC_KEYS}
        else:
            self._cached_metrics = None

        u_kj = float(result["u_t_kj"])
        if not np.isfinite(u_kj) or abs(u_kj) > 1e7:
            u_kj = 0.0
            result["crashed"] = True
        reward = -u_kj / getattr(self.cfg, "reward_ref", 1e5)   
        lam = getattr(self.cfg, "strain_reward_lambda", 0.0)
        if lam and lam > 0.0:
            norm = getattr(self.cfg, "strain_norm_ref", 10.0)
            reward -= lam * (float(result.get("n_clamped") or 0.0) / max(norm, 1e-9))
        # 2026-08-17 Reward clipping: protect against extreme potential energy fluctuations
        reward = float(np.clip(reward, -10.0, 10.0))
        self.u_history.append(u_kj)
        self.n_episode_steps += 1

        mutation_allowed = bool(
            self.n_episode_steps % self.mutation_every == 0
        )
        result["mutation_allowed"] = mutation_allowed

        if action.size >= self.act_dim:
            self.apply_env_offset(float(action[self.force_dim]), float(action[self.force_dim + 1]))

        if mutation_allowed:
            self.rebuild_if_dirty()

        crashed = bool(result["crashed"])
        done = crashed or self.n_episode_steps >= self.cfg.episode_max_steps
        if crashed:
            reward += TERMINAL_CRASH_REWARD
            self._needs_rebuild = True    
        return self.state(), reward, done, result

    def state(self) -> Dict[str, np.ndarray]:
        if getattr(self, "_cached_metrics", None) is not None:
            m = self._cached_metrics
        else:
            m = self.engine.metrics()
        M = np.array([m[k] for k in METRIC_KEYS], dtype=np.float32)
        if not np.all(np.isfinite(M)):
            M = np.where(np.isfinite(M), M, np.zeros_like(M)).astype(np.float32)
        # 2026-08-17 Range clipping: prevent physical metric outliers from polluting neural networks
        M = np.clip(M, -5.0, 5.0)
        u_hist = np.zeros(self.cfg.u_window, dtype=np.float32)
        vals = np.asarray(list(self.u_history), dtype=np.float32)
        if vals.size:
            k = min(vals.size, self.cfg.u_window)
            u_hist[-k:] = vals[-k:]
        if not np.all(np.isfinite(u_hist)):
            u_hist = np.where(np.isfinite(u_hist), u_hist, 0.0).astype(np.float32)
        ph_denom = max(self.cfg.ph_max - self.cfg.ph_min, 1e-5)
        temp_denom = max(self.cfg.temp_max - self.cfg.temp_min, 1e-5)
        env = np.array(
            [
                (self.env_ph - self.cfg.ph_min) / ph_denom,
                (self.env_temp - self.cfg.temp_min) / temp_denom,
                np.clip(np.log10(max(self.env_ionic, 1e-3) / 1e-3) / 3.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        out = {
            "M": M,
            "u_hist": u_hist,
            "coords_ca": np.asarray(self.engine.coords_ca(), dtype=np.float32),
            "env": env,
            "n_steps": np.int32(self.n_episode_steps),
        }
        if getattr(self.cfg, "q_track", True) and self._native_coords is not None:
            out["q"] = native_contact_q(
                out["coords_ca"],
                reference=self._native_coords,
                cutoff=getattr(self.cfg, "q_cutoff", 8.0),
            )
        out["strain"] = np.float32(self.strain_total)
        out["max_accel_clamped"] = np.float32(self.max_accel_clamped)
        out["t_kin"] = np.float32(self.t_kin_last)
        if getattr(self.cfg, "rmsf_window", 0) and self._coords_history:
            out["rmsf"] = per_residue_rmsf(self._coords_history)
        return out

    def native_q(self) -> float:
        if self._native_coords is None:
            return float("nan")
        return native_contact_q(
            self.engine.coords_ca(),
            reference=self._native_coords,
            cutoff=getattr(self.cfg, "q_cutoff", 8.0),
        )

    def per_residue_rmsf(self) -> np.ndarray:
        return per_residue_rmsf(self._coords_history)

    def strain(self) -> Tuple[int, float, float]:
        return self.strain_total, self.max_accel_clamped, self.t_kin_last

    def metrics(self) -> Dict:
        return self.engine.metrics()

    def pseudo_labels(self) -> np.ndarray:
        return np.asarray(self.engine.pseudo_labels(), dtype=np.float32)

    def coords_ca(self) -> np.ndarray:
        return np.asarray(self.engine.coords_ca(), dtype=np.float32)

    def set_temperature(self, k: float) -> None:
        self.env_temp = float(k)
        self.engine.set_temperature(float(k))

    def reset_velocities(self) -> None:
        self.engine.reset_velocities()

    def reset_pseudo_labels(self) -> None:
        self.engine.reset_pseudo_labels()

    def current_env(self) -> Tuple[float, float, float]:
        return self.env_ph, self.env_temp, self.env_ionic
