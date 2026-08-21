# SPICE: Sequence-Protein Interaction under Conditional Environments

> **Physics-as-teacher protein engineering.** A multi-head dynamic Transformer pre-trains a *folding prior* (accurate distance map + coarse topology = "clay"), then a zero-label RL stage (SAC micro-loop + ES macro-loop) supervised by an all-atom MD engine (`spice_engine`) explores stability boundaries and mutation rescue — coverage bounded by compute, not by data.

Two phases:
- **Phase 1 Pre-train** (`spice_pre/`): dynamic Transformer + AdaLN + Head A (**frame structure decoder** with **SE(3) recycling**) + distogram head, supervised by binned distogram cross-entropy + frame coordinate losses.
- **Phase 2 Post-train / RL** (`spice_rl/`): dual-path loop (Path A SAC stability exploration / Path B ES mutation rescue + pseudo-label reflow), backed by the Rust engine `spice_engine`.

---

## Environment

```bash
conda activate spice
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

HuggingFace is reached through the `hf-mirror.com` mirror (already set in `configs/pretrain.yaml`; override with the `HF_ENDPOINT=https://hf-mirror.com` environment variable).

---

## Phase 1 Pre-train (`spice_pre`)

### Model architecture

A **dynamic Transformer encoder** (6 layers, 8 heads, embed 256, FFN 1024) with **AdaLN** (environment injected per layer) produces a sequence–environment embedding `z`, consumed by:

| Head | Output | Role |
| :--- | :--- | :--- |
| **Head A** (coordinate) | Cα coords `[L,3]` | **FrameStructureHead**: per-residue 6D rotation → SO(3) frame, fixed **3.8 Å Cα–Cα virtual bond by construction**, coordinates integrated by cumsum; chirality + clash regularized. Optionally wrapped in **RecycleStructureModule** (distance-aware SE(3) refinement) |
| **Distogram head** | binned dist `[L,L,48]` | main pre-train target (48 bins, 3–48 Å, factorized bilinear, **fp16 einsum**) |
| Head B/B'/C/D | mutation / mutant coords / env offset / confidence | B/B'/C/D instantiate via `heads=("A","B","Bp","C","D")`; **B' has no weights — mutant coords = Head A's fold of the mutant sequence** (`coords_mut = coords`, 2026-08-14) |

**RecycleStructureModule** (`frame_recycle_steps=2`, refine dim 64, 4 heads): each step computes a 3D distance RBF over the current Cα coords → distance-modulated sequence attention → SE(3)-equivariant point features in each local frame → updates each residue frame by rot6 + tanh-bounded translation. Lifts long-chain coordinates from unusable divergence to a usable coarse prior (see below).

### Losses

- **Distogram cross-entropy** (main): squared distances vs squared bin boundaries, symmetrized logits, softmax CE over valid residue pairs. Trained on the **full 45k corpus** (chains to 512 residues).
- **Frame coordinate loss**: Kabsch-aligned Cα RMSD (`pair_weight=1.0`, warm-up 0→1 over `pair_warmup_steps=3000`) + chirality (supervise native handedness) + clash (non-adjacent Cα < 3.5 Å, `frame_clash_weight=3.0` — raised 2026-08-14: 1.0 was numerically too weak, Head A still produced ~10² clashes).

### Two decisive fixes (2026-08-13)

1. **fp16 freeze → fp32 model + fp16 einsum.** Mixed-precision fp16 silently freezes the deep encoder (gradient underflow → grad_norm ≈ 0, CE stuck at random). The model runs **full fp32**; only the distogram bilinear einsum is fp16 (tensor-core accelerated, safe). `use_mixed_precision=false` + `distogram_fp16=true`.
2. **Length-gated coordinate supervision.** FrameHead's cumsum accumulates orientation error with chain length; long chains (200–400 aa = 65% of the data) explode coordinates and poison the shared encoder, starving the distogram. Fix: gate coordinate losses to chains ≤ 200 aa (`coord_max_len=200`) — distogram learns the full corpus, coordinate losses only on short chains, and **RecycleStructureModule** gives long chains a coarse 20–25 Å prior instead of garbage.

### Precision strategy (validated single-protein)

| Setting | Result |
| :--- | :--- |
| fp32 model, fp16 einsum | ✅ trains (2.3 Å) |
| fp16 model (mixed precision) | ❌ freezes (CE stuck ≈ 43, grad_norm ≈ 0) |

### Judging a run

- **Look at `ce`, not `rmsd`.** Random 48-bin baseline is $\ln 48 \approx 3.87$; a healthy run breaks below it early in epoch 0 and continues toward 2–3.
- `pair` only appears on short-chain batches (bucketing + gate); `pair=0` on long-chain batches is expected.
- `rmsd` stays high for long chains **by design** (not supervised) — never use it as a success criterion.

### Data pipeline

The `SPICE-Protein/spice_protein` dataset shares the schema of `download_pdb.py` output:
- `entries_shard_*.parquet`: one row per structure (`ph / temperature / ionic_strength_m / has_env / seq`)
- `atoms_shard_*.parquet`: one row per atom (`is_ca / x / y / z`)

Cleaning:
1. Keep only Cα atoms, sort by `(chain_id, res_seq)` and dedupe (NMR multi-conformers take the first set).
2. **Rebuild the sequence from CA atoms** (3-letter → 1-letter) so `seq` stays strictly aligned with coordinates.
3. Environment normalization: pH→[0,1], T→[0,1], ionic strength→log-mapped [0,1].
4. Length filtering (`min_seq_len=40`, `max_seq_len=512`); `use_env_filtered=false` keeps the full ~45k (missing env filled with defaults — pretraining wants data volume).
5. Write TFRecords (one-time preprocessing, fast reads during training).

```bash
python -m spice_pre.data.dataset --config configs/pretrain.yaml build
python -m spice_pre.data.dataset --config configs/pretrain.yaml stats
```

> Local parquet files (`download_pdb.py` output): set `data.source: local` and `data.local_dir: data/parquet` in `configs/pretrain.yaml`.

### Commands

```bash
# Build TFRecords (downloads via hf-mirror by default; lower max_shards for debugging)
python -m spice_pre.data.dataset --config configs/pretrain.yaml build
python -m spice_pre.data.dataset --config configs/pretrain.yaml stats

# Train
python -m spice_pre.train_pretrain --config configs/pretrain.yaml

# Verify topology: MDS-reconstruct coords from predicted distogram + held-out CASP/contact eval
python -m spice_pre.eval_distogram --config configs/pretrain.yaml --samples 32
python -m spice_pre.eval_casp     --config configs/pretrain.yaml --weights checkpoints/pretrain/best_weights.weights.h5
python -m spice_pre.eval_contacts --config configs/pretrain.yaml --weights checkpoints/pretrain/best_weights.weights.h5

tensorboard --logdir runs/pretrain
```

Outputs: `checkpoints/pretrain/` (latest checkpoint + `best_weights.weights.h5`), `runs/pretrain/` (TensorBoard).

### Cloud notebooks

- `pretrain_kaggle_offline.ipynb` — **Kaggle Commit** (Save & Run All, 12 h budget, `/kaggle/working` auto-saved). Cell ③ forces the official HF endpoint + fp32 model + fp16 einsum regardless of GPU.
- `pretrain_colab.ipynb` — Colab equivalent.

---

## Phase 2 Post-train / RL (`spice_rl`)

Implements the design doc's **SPICE-SAC (five customizations)** + **macro ES**, directly backed by `spice_engine` (PyO3).

### Layout

```
spice_rl/
├── config.py            # RL config (env / sac / es / post)
├── env/
│   ├── structure.py     # all-atom Structure (from_atoms / mmCIF / parquet; multi-chain → longest chain)
│   ├── md_env.py        # MDSimulationEnv: engine step/reward/metrics/pseudo-labels; env-offset clamps
│   ├── quick_check.py   # Step 2: MD-sprint physical validity gate + edge screening
│   ├── phase_map.py     # Step 3: pH–T stability phase maps
│   ├── observables.py   # physical observables (m1..m5, Rg, contacts)
│   └── sidechain.py     # mutant sidechain rebuilding (place_sidechain)
├── sac/
│   ├── networks.py      # dual-head Actor (continuous Gaussian + discrete Gumbel-Softmax) + TwinCritic
│   ├── buffer.py        # global ReplayBuffer (continuous + discrete + hierarchical mask)
│   └── sac.py           # SPICE-SAC trainer (five customizations)
├── es/
│   ├── es.py            # macro ES: mutation search over Head-B/C weight space + policy head
│   └── conservation.py  # functional-retention: conservative-residue mask + optional external MSA vector
├── confidence.py        # Head D confidence (survival-fraction MSE)
├── pseudo_labels.py     # Step 4: survivors → confidence-weighted pseudo-label TFRecord
├── finetune_pretrain.py # reflow: merge pseudo-labels with pre-train set → fine-tune Head A
└── train_post.py        # two-path loop (Path A SAC / Path B ES + reflow)
```

### SPICE-SAC five customizations

1. **Batched collection**: `collect()` fires one update once `update_every_steps` (default 200) is reached.
2. **Decoupled heads for mixed actions**: continuous (bias force [16] + env offset [ΔpH,ΔT]) + discrete (mutation position + type, Gumbel-Softmax τ=1); actions concatenated into a flat vector for the Critic.
3. **Privileged information**: Actor sees z+env; Critic uses z + M + u_hist(10) + action + mask.
4. **Adaptive entropy**: target entropy = -(cont+2) × 0.5.
5. **Temporally layered actions**: mutations take effect every `mutation_every` (20) steps; the buffer stores raw outputs + mask and the Critic learns from the mask.

### Engine integration (confirmed API)

- `Structure.from_atoms(...)` / `from_mmcif(path)` → `Engine.build(structure, ph, temp, pressure, ionic, relax_iters, tolerance, strict_incomplete=True)`
- `engine.step(action=[16])` → `{u_t_kcal, u_t_kj, coords_ca, crashed, m1..m5, rg}`; `metrics()` / `pseudo_labels()` (time-averaged Cα)
- `mutate_sequence(seq, pos, to)` / `validate_sequence(seq)`
- `scan_stability(_ranges)(structure, ...)` / `scan_radial(...)`: stability-domain scans (Path A explores env boundaries / phase maps)

### Safety rails (stability is a discriminant, not a license to destroy function)

- **Env-offset clamps** (`env_offset_clamp`): per-step ΔpH/ΔT clamp + absolute physical window (pH 2–10, T 260–330 K) — the agent cannot push the engine into non-physical dead zones.
- **Functional retention A+B**: Q-gate (survivors must retain ≥ 50% native contacts, `q_gate`) + conservative-residue mask (W/C/G/F/Y or external MSA vector) — prevents "stable but broken" mutations such as active-site W→P.
- **`_sane_ca` sanity guard (Rg + local-clash)**: Head B' = Head A's frame fold of the mutant — a genuine fold, but still carries local Cα clashes (≈10² pairs < 3 Å) that explode all-atom construction (equilibration U → 10¹¹–10²⁷); `_sane_ca`/`_rescale` reject non-buildable predictions (Rg ratio *and* local-clash count) and fall back to the wild-type Cα backbone (2026-08-14 clash check; fixed a 0-survivor regression). Construction uses wild-type backbone + sidechain rebuilding (`sidechain.py`).
- **`no_survivor_abort`**: consecutive zero-survivor episodes abort the protein (caught by the top-K loop, moves to the next).

### Running

Structure input goes through the **data pipeline** (all-atom parquet → `Structure.from_atoms`; the engine adds hydrogens automatically, no mmCIF needed):

```bash
# Two-path post-training (production: data-pipeline all-atom)
python -m spice_rl.train_post --config configs/posttrain.yaml \
    --parquet-dir data/parquet --pdb-id 1ABC --max-episodes 10 [--tag foo]

# Debugging: mmCIF file
python -m spice_rl.train_post --config configs/posttrain.yaml \
    --structure /path/to/2LYZ.cif --max-episodes 10

# Multi-protein top-K loop (4b): --tag namespaces pseudo-labels per protein
#   (train_post.py's path_b_search writes pseudo_{tag}_{i}_{steps}.npz)

# Pseudo-label reflow + fine-tune
python -m spice_rl.finetune_pretrain --config configs/posttrain.yaml --epochs 2

# Smoke tests (no engine)
PYTHONPATH=. python scripts/smoke_rl.py
PYTHONPATH=. python scripts/smoke_reflow.py
```

Input proteins: **80–150 aa** (`min_seq_len` / `max_seq_len`), X-ray, buildable + edge-margin 0.4–0.9 (Step 2 screening; too-stable ≥ 0.95 and too-fragile < 0.1 are skipped). Multi-chain PDBs use the longest chain.

### Three-step closed loop

- **Step 2 · Quick screening** (`quick_check.py`): short MD sprint (`quick_check_steps`) rejects build failures/crashes on the spot; edge screening (`margin` / m5-ratio) selects rescue-worthy proteins in the 0.4–0.9 band.
- **Step 3 · Phase maps** (`phase_map.py`): engine `scan_stability_ranges` → stable-region/collapse-boundary npz (`runs/posttrain/phase_maps/`).
- **Step 4 · Reflow** (`pseudo_labels.py` + `finetune_pretrain.py`): survivors' time-averaged coords → confidence-weighted TFRecord → merged with the pre-train set → fine-tune Head A → `finetuned.weights.h5` (next RL round's start). (2026-08-14: fixed `finetune_pretrain.py` — it omitted `pair_warmup_steps`/`global_step`/chirality/clash when calling `train_step` (TypeError); now passes all args with a short 50-step warm-up so coordinate + clash supervision bite immediately.)

### Logs & metrics

- `runs/posttrain/metrics.csv` — per-episode alpha / buffer / pathA_survive / pathA_crashed / pathB_survivors / critic / actor / alpha / conf losses (plot: `scripts/plot_rl_metrics.py`)
- `runs/posttrain/coverage.csv` — anchor / ±Δ / env_fail (ph, temp) points for the beyond-PDB coverage map (`scripts/plot_coverage_map.py`)
- `runs/posttrain/pathb_candidates.csv` — every Path-B candidate (mutations, fitness = steps×Q, q, survived, env) (`scripts/plot_pathb_landscape.py`)

### Doc-alignment notes

- **Path B evaluation**: mutant-survival evaluation **reuses Path A's SAC Actor frozen** (its bias force drives the engine), not a plain engine short run (`path_b_search`).
- **ES policy-selection vector**: besides the mutation-probability matrix `[L,20]`, the macro head outputs an "aggressive mutation / conservative tweak" policy choice (`es.policy_head`), evolved in the ES weight space too.
- **Head D confidence head**: `confidence.py` regresses two-path confidence supervised by "MD survival steps / max steps" (trained periodically via `conf_train_interval`).

---

## Directory layout (top level)

```
model/
├── configs/pretrain.yaml          # Phase 1 hyperparameters
├── configs/posttrain.yaml         # Phase 2 hyperparameters
├── spice_pre/                     # Phase 1 (models, losses, data, train, eval)
├── spice_rl/                      # Phase 2 (env, sac, es, reflow)
├── scripts/                       # diagnostics + plotting + experiments (incl. slurm/ for SCNet)
├── data/                          # tfrecords, train_cache, pseudo_labels, parquet
├── checkpoints/                   # pretrain/ + posttrain/ weights
├── runs/                          # tensorboard + csv metrics
├── pretrain_kaggle_offline.ipynb  # Kaggle Commit notebook
├── pretrain_colab.ipynb           # Phase 1 Colab notebook
└── posttrain_colab.ipynb          # Phase 2 Colab notebook
```
