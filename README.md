# SPICE: Sequence-Protein Interaction under Conditional Environments

> The repo has two phases:
> - **Phase 1 Pre-train** (`spice_pre/`): dynamic Transformer + AdaLN + Head A (Cα coords), supervised by binned distogram cross-entropy
> - **Phase 2 Post-train / RL** (`spice_rl/`): two paths (SAC micro + ES macro), directly backed by the Rust engine `spice_engine`

## Environment

```bash
conda activate spice
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

HuggingFace is reached through the `hf-mirror.com` mirror (already set in `configs/pretrain.yaml`; you can also override it with the `HF_ENDPOINT=https://hf-mirror.com` environment variable).

## Data pipeline

The `SPICE-Protein/spice_protein` dataset has the same schema as `download_pdb.py` output:
- `entries_shard_*.parquet`: one row per structure (with `ph / temperature / ionic_strength_m / has_env / seq`)
- `atoms_shard_*.parquet`: one row per atom (with `is_ca / x / y / z`)

Cleaning logic:
1. Keep only Cα atoms, sort by `(chain_id, res_seq)` and dedupe (NMR multi-conformers take the first set).
2. **Rebuild the sequence from CA atoms** (3-letter → 1-letter) so `seq` stays strictly aligned with the coordinates.
3. Environment normalization: pH→[0,1], T→[0,1], ionic strength→log-mapped [0,1].
4. Filter by `has_env=True` (default; corresponds to the design doc's "keep only environment-labeled data") plus length filtering.
5. Write TFRecords (one-time preprocessing, fast reads during training).

```bash
# Generate TFRecords (downloads via hf-mirror by default; lower max_shards/structures_per_shard in the yaml for debugging)
python -m spice_pre.data.dataset --config configs/pretrain.yaml build
python -m spice_pre.data.dataset --config configs/pretrain.yaml stats
```

> If you already have local parquet files (`download_pdb.py` output): set `data.source` to `local` in `configs/pretrain.yaml` and point `data.local_dir` at `data/parquet`.

## Training

```bash
python -m spice_pre.train_pretrain --config configs/pretrain.yaml
```

- Loss: **binned distogram cross-entropy** (squared distances vs squared bin boundaries, symmetrized logits, softmax CE over valid residue pairs) as the main objective, plus a small pairwise-coordinate RMSE auxiliary term. Direct Kabsch RMSD coordinate regression is kept only as a **validation metric** — as a training target it is pathological (the prediction collapses toward the origin).
- Learning rate: linear warmup + cosine decay; optimizer AdamW.
- Outputs: `checkpoints/pretrain/` (latest checkpoint + `best_weights.weights.h5`), `runs/pretrain/` (TensorBoard).

Check whether topology is actually learned by MDS-reconstructing coordinates from the predicted distogram:

```bash
python -m spice_pre.eval_distogram --config configs/pretrain.yaml --samples 32
```

(MDS-RMSD < 10 Å → topology learned; ~Rg → still a blob. High precision is intentionally left to the RL phase + MD-engine feedback.)

```bash
tensorboard --logdir runs/pretrain
```

## Directory layout

```
model/
├── configs/pretrain.yaml          # hyperparameters (mirror, data, model, training)
├── spice_pre/
│   ├── config.py                  # dataclass config
│   ├── data/
│   │   ├── preprocessing.py       # AA mapping / tokenization / env normalization
│   │   └── dataset.py             # HF/local loading → cleaning → TFRecord → tf.data
│   ├── models/
│   │   ├── adaln.py               # adaptive layer norm (env injected per layer)
│   │   ├── transformer.py         # dynamic Transformer (variable length + mask)
│   │   └── spice_model.py         # SPICE model (Head A + reserved B/B'/C/D)
│   ├── losses/kabsch_rmsd.py      # distogram CE + pairwise RMSE + Kabsch RMSD (val)
│   └── train_pretrain.py          # training entry
└── requirements.txt
```

## Phase 2 Post-train / RL (spice_rl)

Implements the design doc's **SPICE-SAC (five customizations)** + **macro ES**, directly backed by `spice_engine` (PyO3).

### Layout

```
spice_rl/
├── config.py            # RL config (env / sac / es / post)
├── env/
│   ├── structure.py     # all-atom Structure construction (from_atoms / mmCIF / parquet)
│   └── md_env.py        # MDSimulationEnv: wraps engine step/reward/metrics/pseudo-labels
├── sac/
│   ├── networks.py      # dual-head Actor (continuous Gaussian + discrete Gumbel-Softmax) + TwinCritic
│   ├── buffer.py        # global ReplayBuffer (continuous + discrete actions + hierarchical mask)
│   └── sac.py           # SPICE-SAC trainer (five customizations)
├── es/
│   └── es.py            # macro ES: searches mutations in the Head-B/C weight space
└── train_post.py        # two-path post-training loop (Path A SAC / Path B ES + pseudo-label reflow)
```

### SPICE-SAC five customizations

1. **Asynchronous collection + batched updates**: `collect()` triggers one batch update once `update_every_steps` is reached
2. **Decoupled heads for mixed actions**: continuous head (bias force [16] + env offset [ΔpH,ΔT]) + discrete head (mutation position + type, Gumbel-Softmax τ=1); actions are concatenated into a flat vector for the Critic
3. **Privileged information**: Actor sees only z+env; Critic uses z+M+u_hist(10)+action+mask
4. **Adaptive entropy coefficient**: target entropy = -(cont+2) × 0.5
5. **Hierarchical action timing**: mutations take effect every `mutation_every` (20) steps; the buffer stores raw outputs + mask and the Critic learns from the mask

### Engine integration (confirmed API)

- `Structure.from_atoms(...)` / `from_mmcif(path)` → `Engine.build(structure, ph, temp, pressure, ionic, relax_iters, tolerance)`
- `engine.step(action=[16])` → `{u_t_kcal, u_t_kj, coords_ca, crashed, m1..m5, rg}`; `metrics()` / `pseudo_labels()` (time-averaged Cα)
- `mutate_sequence(seq, pos, to)` / `validate_sequence(seq)`
- `scan_stability(_ranges)(structure, ...)` / `scan_radial(...)`: stability-domain scans (Path A explores env boundaries / phase maps)

### Running

Structure input goes through the **data pipeline** (all-atom parquet → `Structure.from_atoms`; the engine adds hydrogens automatically, no mmCIF needed):

```bash
# Two-path post-training (production: data-pipeline all-atom; --pdb-id is the initial wild-type structure)
python -m spice_rl.train_post --config configs/posttrain.yaml \
    --parquet-dir data/parquet --pdb-id 1ABC --max-episodes 10

# Debugging: you can also use an mmCIF file
python -m spice_rl.train_post --config configs/posttrain.yaml \
    --structure /path/to/2LYZ.cif --max-episodes 10

# Lightweight smoke test (no engine): validates the SAC/ES/model pipeline
PYTHONPATH=. python scripts/smoke_rl.py
```

> The engine's SAC-specific API (`ForceAction`/`ActionMask`/`EnvDelta` in `actions.rs`) already handles bias-force mapping, hierarchical masking, and env-offset timing; Python only needs `engine.step(action)`. Hydrogen placement is the engine's job (heavy atoms suffice). Path B's **post-mutation structure = the Cα coordinates output by model Head B'** (overwrites the backbone CA) + the engine rebuilds side chains from AA templates (`build_mutant_structure_from_ca`).

### Three-step closed loop (quick screening / phase maps / reflow)

- **Step 2 · Quick screening**: `spice_rl/env/quick_check.py` uses a short MD run (`quick_check_steps`) to check physical validity; build failures/crashes are rejected on the spot.
- **Step 3 · Phase maps**: `spice_rl/env/phase_map.py` calls the engine's `scan_stability_ranges` to scan the pH-T plane; `save_phase_map` produces stable-region/collapse-boundary npz files (`runs/posttrain/phase_maps/`).
- **Step 4 · Reflow**: `spice_rl/pseudo_labels.py` (npz → TFRecord, confidence-weighted repetition) + `spice_rl/finetune_pretrain.py` (merges the original Pre-train TFRecords to fine-tune Head A → `finetuned.weights.h5`, the starting point for the next RL round).

```bash
# Pseudo-label reflow + fine-tune (after train_post has produced pseudo-labels)
python -m spice_rl.finetune_pretrain --config configs/posttrain.yaml --epochs 2

# Reflow-pipeline smoke test (no engine)
PYTHONPATH=. python scripts/smoke_reflow.py
```

### Doc-alignment notes (three)

- **Path B evaluation**: mutant-survival evaluation **reuses Path A's SAC Actor in a frozen way** (its bias force drives the engine), instead of a plain engine short run (`path_b_search`).
- **ES policy-selection vector**: besides the mutation-probability matrix `[L,20]`, the macro head outputs an "aggressive mutation / conservative tweak" policy choice (`es.policy_head`), which is also evolved in the ES weight space.
- **Head D confidence head**: `spice_rl/confidence.py` regresses two-path confidence supervised by "MD survival steps / max steps" (trained periodically via `conf_train_interval`).
