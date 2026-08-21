#!/usr/bin/env python
"""Restart Gate Smoke Test (2026-08-16).

Verifies the full lifecycle of the Restart Gate within `path_b_search`:
  1. Candidate A collapses during equilibration -> crash count increments +1
  2. Candidate A collapses again (across episodes, sharing the same explosive dict) -> count >= threshold -> auto-blacklisted
  3. Candidate A appears a third time (or a compound mutation containing component A) -> hits blacklist -> skips structure building and sets fitness = penalty
  4. CSV export to explosive_blacklist.csv records correctly
  5. Negative control: different target residues at the same position (stable mutations) are not falsely blacklisted

Uses a monkeypatch on the real `path_b_search` internals (such as spice_engine.Engine and model functions)
to run without executing actual Molecular Dynamics (achieving deterministic, fast runs). 
This validates the plumbing and integration of Restart Gate logic rather than engine physics.

Run: python scripts/smoke_restart_gate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import numpy as np

# ---- Patch spice_engine.Engine first: enables construction and mock equilibration collapses ----
import spice_engine as _se
import spice_rl.train_post as tp

# Current mut_seq under evaluation (recorded by build_mutant_structure_from_ca, used by fake engine to determine collapse)
_CUR_MUT = {"seq": ""}


class _FakeEngine:
    """Mock physics engine: raises RuntimeError if the current mut_seq is in `_CRASH_MUTS` during equilibration."""

    _CRASH_MUTS = set()

    def __init__(self, *a, **k):
        self._crashed_steps = 0

    @staticmethod
    def build(*a, **k):
        return _FakeEngine()

    def mutate_with_solvent_reuse(self, *a, **k):
        return _FakeEngine()

    def equilibrate(self):
        if _CUR_MUT["seq"] in _FakeEngine._CRASH_MUTS:
            raise RuntimeError("mutant equilibrate failed: U=1.5e9 (smoke)")

    def step(self, f):
        self._crashed_steps += 1
        return {"crashed": False, "u": -1.0}

    def pseudo_labels(self):
        n = len(_CUR_MUT["seq"]) or 20
        return np.zeros((n, 3), np.float32)

    def coords_ca(self):
        n = len(_CUR_MUT["seq"]) or 20
        return np.zeros((n, 3), np.float32)


def _install_fake_engine():
    # `import spice_engine as se` inside path_b_search resolves to the same module object -> patch attributes directly
    _se.Engine = _FakeEngine
    _se.mutate_sequence = lambda seq, p, to: (seq[:p] + to + seq[p + 1:] if 0 <= p < len(seq) else seq)
    _se.validate_sequence = lambda seq: None
    _FakeEngine.build = staticmethod(_FakeEngine.build)
    _FakeEngine.mutate_with_solvent_reuse = staticmethod(_FakeEngine.mutate_with_solvent_reuse)


# ---- Model-level Mocking (encode_z / predict_mutant_coords / build_mutant_structure_from_ca) ----
class _DummyModel:
    pass


def _encode_z_ok(model, tok, env, mask):
    return np.zeros((256,), np.float32)


def _predict_coords_ok(model, tok, env, mask):
    return None  # Fall back to wild backbone


def _build_struct(base_atoms, mut_seq, pred_ca=None):
    from spice_rl.env.structure import structure_from_atoms
    _CUR_MUT["seq"] = mut_seq
    n = len(mut_seq)
    names = ["CA"] * n
    elems = ["C"] * n
    seqs = list(range(1, n + 1))
    resnames = list(mut_seq)
    coords = np.zeros((n, 3), np.float32)
    for i in range(n):
        coords[i, 0] = i * 3.8
    return structure_from_atoms(names, elems, seqs, resnames, coords)


def main():
    _install_fake_engine()
    tp.encode_z = _encode_z_ok
    tp.predict_mutant_coords = _predict_coords_ok
    tp.build_mutant_structure_from_ca = _build_struct
    # _sane_ca requires base_atoms; provide dummy wild_ca
    tp._native_ca = lambda ba: np.zeros((20, 3), np.float32)
    tp._sane_ca = lambda pc, wc, *a, **k: False  # Route all to rescale/wild-type backfalls
    tp._rescale_pred_ca = lambda pc, wc, *a, **k: None
    tp.seq_to_tokens = lambda s: np.arange(len(s), dtype=np.int32)

    base_seq = "MEKSFVITDPRLPDNPIIFASDGFLELTEYSREEILGRNGRFLQGPETDQATVQKIQDAIRDQREITVQLINYTKSGKKFWNLLHLQPMRDQKGELQYFIGVQLDGEFIPNPLLGL"
    L = len(base_seq)
    # _native_ca must match base_seq length, otherwise Q-metric calculations yield q_skip and set fitness = 0 (a mock issue, not logic issue)
    tp._native_ca = lambda ba: np.zeros((L, 3), np.float32)
    tokens = np.arange(L, dtype=np.int32)
    mask = np.ones(L, np.float32)
    z_mask = np.ones(L, np.float32)
    base_atoms = {"atom_names": ["CA"] * 20, "elements": ["C"] * 20,
                  "res_seq": list(range(1, 21)), "res_names": ["A"] * 20,
                  "coords": np.zeros((20, 3), np.float32)}

    from spice_rl.config import EnvConfig, ESConfig

    env_cfg = EnvConfig()
    es_cfg = ESConfig()

    # Dummy ES: returns fixed mutation candidates list
    class _FakeES:
        def __init__(self, cands):
            self.cands = cands
            self.cfg = es_cfg

        def propose_mutations(self, base_seq, tokens, env_norm, mask):
            return self.cands

    # Dummy SAC
    class _FakeSAC:
        def act(self, z, env, z_mask, deterministic=False):
            return (np.zeros((16 + 2,), np.float32), np.zeros((0,), np.float32))

    # Dummy Model
    model = _DummyModel()

    tmp = tempfile.mkdtemp()
    cand_log = os.path.join(tmp, "pathb_candidates.csv")
    explosive = {"on": True, "threshold": 2, "min_steps": 3, "penalty": -1.0,
                 "env_bucket": True,
                 "counts": {}, "blacklist": set(),
                 "csv_path": os.path.join(tmp, "explosive_blacklist.csv")}

    # Candidates: A = 104:L>W (collapses during equilibration), B = 104:L>E (stable)
    mutA = base_seq[:103] + "W" + base_seq[104:]
    mutB = base_seq[:103] + "E" + base_seq[104:]

    # ---- ep7: Candidate A only, collapses -> count=1, fitness=-1.0, not blacklisted yet ----
    _FakeEngine._CRASH_MUTS = {mutA}
    es = _FakeES([(mutA, 0, "conservative")])
    survivors, wt, fitness, _ = tp.path_b_search(
        model, _FakeSAC(), es, base_seq, (2.0, 330.0, 0.1),
        tokens, mask, z_mask, base_atoms, tmp, 20, env_cfg,
        tag="7qf3", ep=7, cand_log=cand_log, explosive=explosive,
    )
    print(f"[ep7] A collapses once: fitness={fitness[0]:.1f} (expected -1.0)  blacklist={sorted(explosive['blacklist'])} (expected empty)  counts={dict(explosive['counts'])}")
    assert fitness[0] == -1.0, "equilibrate collapse fitness must be -1.0"
    assert explosive["counts"].get((104, "L", "W", "acid")) == 1, "A count must be 1 in acid bucket"
    assert not explosive["blacklist"], "Threshold not met; should not be blacklisted yet"

    # ---- ep34: Candidate A collapses again -> threshold met -> auto-blacklisted and exported to CSV ----
    es = _FakeES([(mutA, 0, "conservative")])
    tp.path_b_search(
        model, _FakeSAC(), es, base_seq, (2.0, 330.0, 0.1),
        tokens, mask, z_mask, base_atoms, tmp, 20, env_cfg,
        tag="7qf3", ep=34, cand_log=cand_log, explosive=explosive,
    )
    print(f"[ep34] A collapses twice: blacklist={sorted(explosive['blacklist'])} (expected {{(104,'L','W','acid')}})")
    assert (104, "L", "W", "acid") in explosive["blacklist"], "A must be blacklisted in acid bucket"
    csv_rows = open(explosive["csv_path"]).read().strip().splitlines()
    print(f"  CSV({len(csv_rows)-1} rows): {csv_rows}")
    assert any("104" in r and ",L,W,acid," in r for r in csv_rows), "CSV must contain 104,L,W,acid"

    # ---- ep35: A hits blacklist -> skips build (_CUR_MUT remains empty/base_seq instead of mutA) -> fitness=-1.0 ----
    _CUR_MUT["seq"] = ""  # Reset; if blacklisted, build_mutant_structure_from_ca(mutA) should never be called
    es = _FakeES([(mutA, 0, "conservative")])
    survivors, wt, fitness, _ = tp.path_b_search(
        model, _FakeSAC(), es, base_seq, (2.0, 330.0, 0.1),
        tokens, mask, z_mask, base_atoms, tmp, 20, env_cfg,
        tag="7qf3", ep=35, cand_log=cand_log, explosive=explosive,
    )
    # Note: The parent engine builds parent base_seq once before loop starts (_CUR_MUT=base_seq), 
    # but mutA itself must be skipped and never constructed.
    print(f"[ep35] A hits blacklist: fitness={fitness[0]:.1f} (expected -1.0)  not_built_for_mutA={_CUR_MUT['seq']!=mutA} (expected True)  survivors={len(survivors)} (expected 0)")
    assert fitness[0] == -1.0, "Blacklist hit fitness must be -1.0"
    assert _CUR_MUT["seq"] != mutA, "Blacklisted candidate must skip build to save core hours"
    assert len(survivors) == 0

    # ---- ep36: Candidate B (104:L>E) is stable -> evaluated normally (survives), no false positive ----
    _FakeEngine._CRASH_MUTS = set()  # B does not collapse
    es = _FakeES([(mutB, 0, "conservative")])
    survivors, wt, fitness, _ = tp.path_b_search(
        model, _FakeSAC(), es, base_seq, (2.0, 330.0, 0.1),
        tokens, mask, z_mask, base_atoms, tmp, 20, env_cfg,
        tag="7qf3", ep=36, cand_log=cand_log, explosive=explosive,
    )
    print(f"[ep36] B(104:L>E) control: fitness={fitness[0]:.1f} (expected >0)  survivors={len(survivors)}  blacklist remains={sorted(explosive['blacklist'])}")
    assert fitness[0] > 0.0, "B evaluated normally must yield fitness > 0"
    assert (104, "L", "E", "acid") not in explosive["blacklist"], "B must not be blacklisted"

    # ---- ep37: Cross-bucket isolation: A is blacklisted in acid bucket, but collapses only once in base bucket (pH 10) -> no blacklisting in base bucket ----
    _FakeEngine._CRASH_MUTS = {mutA}  # A collapses in base bucket as well (independent count)
    es = _FakeES([(mutA, 0, "conservative")])
    survivors, wt, fitness, _ = tp.path_b_search(
        model, _FakeSAC(), es, base_seq, (10.0, 330.0, 0.1),
        tokens, mask, z_mask, base_atoms, tmp, 20, env_cfg,
        tag="7qf3", ep=37, cand_log=cand_log, explosive=explosive,
    )
    print(f"[ep37] A in base bucket: fitness={fitness[0]:.1f} (expected -1.0, collapsed but not blacklisted yet)  blacklist={sorted(explosive['blacklist'])}")
    assert explosive["counts"].get((104, "L", "W", "base")) == 1, "A in base bucket must be counted independently"
    assert (104, "L", "W", "base") not in explosive["blacklist"], "Base bucket count below threshold; should not be blacklisted"
    assert (104, "L", "W", "acid") in explosive["blacklist"], "Acid bucket blacklist must be preserved"
    # Although A collapsed on this step, it should have normally undergone structure build since it wasn't blacklisted in base bucket (no false cross-bucket suppression)
    assert _CUR_MUT["seq"] != "", "A must undergo build in base bucket (cross-bucket isolation)"

    print("\n✅ Restart Gate smoke tests passed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
