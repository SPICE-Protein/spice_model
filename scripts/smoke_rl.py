#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from spice_rl.config import load_config

cfg = load_config("configs/posttrain.yaml")
print("RL config ok:", cfg.env.force_dim, cfg.sac.gumbel_tau, cfg.es.population)

import tensorflow as tf  # noqa: E402
from spice_rl.sac import SACTrainer  # noqa: E402

sac = SACTrainer(
    cfg.sac,
    z_dim=256,
    cont_dim=cfg.env.force_dim + cfg.env.env_offset_dim,
    u_window=cfg.env.u_window,
)
print("SACTrainer built; target_entropy =", round(sac.target_entropy, 3))

L = cfg.sac.discrete_position_dim


def rnd_tr(i):
    return {
        "z": np.random.randn(256).astype(np.float32),
        "env": np.random.rand(3).astype(np.float32),
        "M": np.random.rand(5).astype(np.float32),
        "u_hist": np.random.randn(10).astype(np.float32),
        "action_cont": np.random.randn(18).astype(np.float32),
        "action_disc": np.random.rand(L + 20).astype(np.float32),
        "mutation_mask": 1.0 if i % 20 == 0 else 0.0,
        "z_mask": np.concatenate([np.ones(30, np.float32), np.zeros(L - 30, np.float32)]),
        "reward": float(np.random.randn()),
        "done": False,
        "next_z": np.random.randn(256).astype(np.float32),
        "next_env": np.random.rand(3).astype(np.float32),
        "next_M": np.random.rand(5).astype(np.float32),
        "next_u_hist": np.random.randn(10).astype(np.float32),
    }


for i in range(600):
    sac.collect(rnd_tr(i))
print("buffer size:", len(sac.buffer))

z_mask = np.concatenate([np.ones(30, np.float32), np.zeros(L - 30, np.float32)])
losses = sac.update(z_mask)
print("SAC update losses:", {k: round(float(v), 4) if isinstance(v, float) else v for k, v in losses.items()})
print("alpha:", round(sac.alpha(), 4))

a_cont, a_disc = sac.act(
    np.random.randn(256).astype(np.float32),
    np.random.rand(3).astype(np.float32),
    z_mask,
)
print("act shapes:", a_cont.shape, a_disc.shape)

from spice_rl.train_post import build_rl_model, cfg_sac_z_dim  # noqa: E402

model = build_rl_model(cfg)
print("RL model built; embed_dim =", cfg_sac_z_dim(model))

from spice_rl.es import ESEvolver  # noqa: E402

es = ESEvolver(model, cfg.es)
print("ESEvolver ok; evolvable vars (Head-B/C + policy):", len(es.head_vars))

from spice_rl.train_post import encode_z, tokens_from_seq  # noqa: E402

_tk, _mk = tokens_from_seq("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", L)
_z = encode_z(model, _tk, np.array([0.5, 0.5, 0.5], np.float32), _mk)
assert _z.shape == (256,), _z.shape
_c = es.propose_mutations(
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", _tk,
    np.array([0.5, 0.5, 0.5], np.float32), _mk,
)
print("masked z-pool ok:", _z.shape, "| ES candidates:", len(_c))

from spice_rl.confidence import ConfidenceHeadTrainer  # noqa: E402

ct = ConfidenceHeadTrainer(model, lr=1e-4)
for _ in range(64):
    ct.add(np.random.randn(256).astype(np.float32), np.array([0.5, 0.8], np.float32))
cl = ct.update(32)
print("Head D conf loss:", round(float(cl["conf_loss"]), 4))
pred = ct.predict(np.random.randn(256).astype(np.float32))
print("Head D predict shape:", pred.shape, "in [0,1]:", bool(np.all((pred >= 0) & (pred <= 1))))
print("ALL RL SMOKE TESTS PASSED (no engine run)")
