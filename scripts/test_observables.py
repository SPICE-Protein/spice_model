#!/usr/bin/env python
"""Smoke test for env/observables.py (pure numpy; no engine needed)."""
import numpy as np
from spice_rl.env.observables import (
    native_contact_map,
    native_contact_q,
    per_residue_rmsf,
    track_rmsf,
)

rng = np.random.default_rng(0)
L = 40
# Native: a compact folded chain (random walk with small steps -> contacts).
native = np.zeros((L, 3))
for i in range(1, L):
    native[i] = native[i - 1] + rng.normal(0, 1.0, 3)

pairs = native_contact_map(native, cutoff=8.0)
assert len(pairs) > 0, "folded chain should have contacts"
print(f"native contacts (8A): {len(pairs)}")

# Q(native, native) should be ~1.0
q0 = native_contact_q(native, pairs, cutoff=8.0)
print(f"Q(native vs native) = {q0:.3f}  (expect ~1.0)")
assert q0 > 0.95

# Q(unfolded) should be low: stretch the chain into a line
unfolded = np.zeros((L, 3))
unfolded[:, 0] = np.arange(L) * 12.0  # 12A apart -> few contacts
qu = native_contact_q(unfolded, pairs, cutoff=8.0)
print(f"Q(unfolded vs native) = {qu:.3f}  (expect < 0.2)")
assert qu < 0.2

# RMSF: residues 5..10 fluctuate strongly, others static
hist = []
for t in range(50):
    c = native.copy()
    for i in range(5, 11):
        c[i] += rng.normal(0, 2.0, 3) * (1 + t * 0.0)
    c += rng.normal(0, 0.05, c.shape)  # small global noise
    hist.append(c)
rmsf = per_residue_rmsf(hist)
print(f"RMSF[5:10] mean = {rmsf[5:10].mean():.2f} (expect ~2-3), "
      f"RMSF[20:25] mean = {rmsf[20:25].mean():.2f} (expect ~0.1)")
assert rmsf[5:10].mean() > 1.5
assert rmsf[20:25].mean() < 0.5

# track_rmsf bounded window
acc = []
for _ in range(10):
    acc = track_rmsf(acc, native, 5)
assert len(acc) == 5, "window should be bounded at 5"

print("OBSERVABLES_OK")
