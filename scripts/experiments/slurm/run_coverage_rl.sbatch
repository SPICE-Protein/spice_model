#!/bin/bash
#SBATCH --job-name=spice_cov
#SBATCH --array=10,45000
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check (varies by center, e.g., kshcnormal/xahcnormal)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2              # ← Run probe_threads.py first to determine core count (MD engines are mostly single-threaded serial, don't guess 8)
#SBATCH --mem=6G                     # SCNet limits mem:CPU <= DefMemPerCPU (16G/2-core is rejected); 6G/2-core=3G/core is sufficient for RL (SAC buffer uses ~1.2GB)
#SBATCH --time=24:00:00                # ← Empirically ~29min/episode (7QF3 collapses -> Path B triggers 8x build); MAX_EPISODES * K * 29min; 15 episodes * 3 structures (2 independent sequences) ≈ 22h
# ⚠️ Submit from $MODEL_ROOT ($HOME/spice/model) so relative paths in -o/-e resolve correctly.
#    The log directory slurm_logs/ MUST exist prior to submission (Slurm does not automatically create parent directories for -o, causing immediate failure without logs):
#    cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_coverage_rl.sbatch
#SBATCH --output=slurm_logs/cov_%A_%a.out
#SBATCH --error=slurm_logs/cov_%A_%a.err

# ===== Data Efficiency Ablation · Coverage Axis =====
# Each SLURM_ARRAY_TASK_ID = one prior scale (culled to 10 and 45000 on 2026-08-14, 
# cutting out intermediate 100/1000 scales since the coverage axis aims to prove "small priors can also reach beyond-PDB coverage" 
# — small scale is the protagonist, large scale is the control; held-out AUC for 100/1000 is already in the paper, skipping redundant RL runs).
# Run fixed RL loop for the same set of proteins across each scale -> only prior scale varies, keeping RL configurations completely fixed.
#
# Usage:    sbatch scripts/experiments/slurm/run_coverage_rl.sbatch
# Prereqs:  ① posttrain.yaml has been generated for each scale (via ablate_data_efficiency.py --phase rl-gen)
#           ② engine wheel (spice_engine) is installed in the env; checkpoints exist at data/data_efficiency/n*/checkpoints/
#           ③ selected proteins (margin 0.4-0.9, matching the Kaggle full prior group) are available
#           ④ run probe_threads.py first to determine cpus-per-task and protein runtimes (avoid over-allocating cores, which wastes budget)
#
# Outputs:  runs/posttrain/coverage.csv + metrics.csv + pathb_candidates.csv for each scale
#           (use ablate_physics_loop.py to count beyond-PDB points -> compares coverage across scales)

set -euo pipefail

# ---- Env (SCNet: setup_cluster.sh completes install; ⚠️ runs inside singularity container as node glibc 2.17 cannot support TF 2.21) ----
CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
# ⚠️ Container image: select glibc >= 2.28 (TF 2.21 requires glibc 2.27, numpy 2.5.2 requires 2.28).
#   Defaults to ubuntu20.04-mkl (glibc 2.31 + MKL); override SIF= if changed
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

# ---- Data ----
PARQUET_DIR=${MODEL_ROOT}/../data/parquet_hpc       # fetch_atoms.py fetches atoms shard here
PROTEINS="7qf3 6qqe 8d8f"   # P1 final: 3 structures / 2 independent sequences, margin 0.553/0.850/0.864 (all build successfully, inside responsive zone; 6QQE/8D8F are two conformations of the same Lysozyme sequence; 3EMS at 0.909 is too stable and has been removed)
# Empirically ~29min/episode: coverage points accumulate rapidly (a few episodes are enough), but SAC requires ~13-20 episodes to fill buffer and begin training ->
# setting too low prevents any learning (0 survivors); setting too high exceeds time limits. P1 runs 15 episodes as an initial probe, adjust after analyzing results.
MAX_EPISODES=${MAX_EPISODES:-15}

# ===== Task Body =====
SCALE=$SLURM_ARRAY_TASK_ID                          # array id = scale N

# Run python inside the container (conda env is in shared HOME, singularity binds by default; cwd=model root ensures correct relative paths for yaml)
run_py() {
  singularity exec "$SIF" bash -lc "
    source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$MODEL_ROOT' && \
    export PYTHONPATH='$MODEL_ROOT' RAYON_NUM_THREADS='${SLURM_CPUS_PER_TASK:-2}' \
           HF_ENDPOINT='https://hf-mirror.com' POLARS_SKIP_CPU_CHECK='1' CUDA_VISIBLE_DEVICES='-1' \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           ONEDNN_DEFAULT_FPMATH_MODE=FP32 TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
    $1
  "
}
cd "$MODEL_ROOT"    # ⚠️ yaml uses relative paths (../data/...), must execute from model root directory

POST_YAML="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/posttrain.yaml"
COV_CSV="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/runs/posttrain/coverage.csv"
echo "[task ${SLURM_ARRAY_TASK_ID}] scale=n${SCALE}  config=${POST_YAML}"
test -f "$POST_YAML" || { echo "Missing ${POST_YAML} (run rl-gen first)"; exit 1; }
mkdir -p slurm_logs

# Self-heal: isolates output directory for this scale (prevents 4 concurrent scales running same proteins from overwriting each other). Idempotent.
run_py "python scripts/experiments/slurm/self_heal.py '$POST_YAML' '$SCALE'" || \
  { echo "[warn] self-heal failed (yaml might already be in isolated format), continuing"; }

# Sync atoms shards (idempotent; flock prevents race conditions when 4 concurrent scales attempt to download the same shard)
mkdir -p "$PARQUET_DIR"
(
  flock 9
  for PDB in $PROTEINS; do
    run_py "python scripts/experiments/slurm/fetch_atoms.py --pdb-id '$PDB' --out '$PARQUET_DIR'" || true
  done
) 9>"$PARQUET_DIR/.fetch.lock"

# Run fixed RL loop sequentially for each protein on this scale (--tag avoids overwriting pseudo-labels and candidate tables).
# ⚠️ Skip individual protein failures (0 survivor abort or build failure) so the entire scale job doesn't fail (core hours are expensive).
NOK=0
for PDB in $PROTEINS; do
  echo "=== [n${SCALE}] ${PDB} RL start $(date) ==="
  if run_py "python -m spice_rl.train_post --config '$POST_YAML' --parquet-dir '$PARQUET_DIR' --pdb-id '${PDB^^}' --tag '$PDB' --max-episodes '$MAX_EPISODES'"; then
    echo "=== [n${SCALE}] ${PDB} RL done  $(date) ==="
  else
    echo "=== [n${SCALE}] ${PDB} RL FAILED (skipping, continuing to next) $(date) ==="
    NOK=$((NOK+1))
  fi
done
if [ "$NOK" -gt 0 ]; then echo "[warn] n${SCALE}: ${NOK} proteins failed"; fi

echo "[task ${SLURM_ARRAY_TASK_ID}] n${SCALE} completed -> coverage: ${COV_CSV}"
