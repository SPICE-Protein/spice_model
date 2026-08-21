#!/bin/bash
#SBATCH --job-name=spice_n10
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as formal runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=48:00:00                # ← Fixed RL loop: same duration as n45000 40 episodes * 4 structures (3 independent sequences)
#SBATCH --output=slurm_logs/10_%A.out
#SBATCH --error=slurm_logs/10_%A.err

# ===== Data Efficiency Coverage Axis Decisive Experiment: n10 (10-chain minimal seed / "primordial soup" limit) * 4 structures (3 independent sequences) =====
# Purpose: n10 prior has virtually no protein folding grammar (CASP AUC 0.475 ≈ random, P@L/5 8.3% ≈ 2x random contact density, GDT-TS 0.024 = blob).
#          If a fixed RL loop starting from n10 can achieve beyond-PDB coverage, viable mutants, and pseudo-label feedback identical to n45000,
#          then folding and stability are driven by physical organization (MD potentials) rather than data distribution — the strongest proof of physics-as-teacher (paper §3.4).
# Fixed RL loop = identical posttrain.yaml to n45000 (only post.pretrain_ckpt points to the n10 checkpoint).
# Submit:   cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_10.sbatch
# Success:  obtaining viable mutants + pseudo-labels + coverage points for the same set of proteins (7QF3 + Lysozymes 6QQE/8D8F + RNase A 1JVT);
#           coverage boundaries (pH/T range of env_fail) match the n45000 baseline.
# Prereqs:  Upload latest RL code (sac.py numerical safeguards + train_post.py watchdog + Env Escape + config.py + posttrain.yaml),
#           n10 prior (../data/data_efficiency/n10/checkpoints/best_weights.weights.h5) + n10/pretrain.yaml;
#           Generate n10 RL configs first: ablate_data_efficiency.py --phase rl-gen --scales 10 --post-config configs/posttrain.yaml
# Pro tip:  To rerun, override with PROTEINS="1jvt" to run only newly added proteins, saving core hours.
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

PARQUET_DIR=${MODEL_ROOT}/../data/parquet_hpc
PROTEINS=${PROTEINS:-"7qf3 6qqe 8d8f 1jvt"}   # 4 structures / 3 independent sequences: 7QF3 + Lysozyme (6QQE/8D8F conformations) + RNase A 1JVT
# Run only new proteins when rerunning: PROTEINS="1jvt" sbatch scripts/experiments/slurm/run_10.sbatch
# Fixed RL loop: MAX_EPISODES matches n45000 (default 40)
MAX_EPISODES=${MAX_EPISODES:-40}

run_py() {
  singularity exec "$SIF" bash -lc "
    source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$MODEL_ROOT' && \
    export PYTHONPATH='$MODEL_ROOT' RAYON_NUM_THREADS='${SLURM_CPUS_PER_TASK:-2}' \
           HF_ENDPOINT='https://hf-mirror.com' POLARS_SKIP_CPU_CHECK='1' CUDA_VISIBLE_DEVICES='' \
           OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
           ONEDNN_DEFAULT_FPMATH_MODE=FP32 ONEDNN_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' \
           DNNL_MAX_CPU_ISA='${DNNL_MAX_CPU_ISA:-SSE41}' TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
    $1
  "
}

cd "$MODEL_ROOT"
SCALE=10
POST_YAML="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/posttrain.yaml"
COV_CSV="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/runs/posttrain/coverage.csv"
mkdir -p slurm_logs

echo "[n10] config=${POST_YAML}  max_episodes=${MAX_EPISODES}"
test -f "$POST_YAML" || { echo "Missing ${POST_YAML}"; exit 1; }

# Self-heal: isolates output directory for this scale (idempotent)
run_py "python scripts/experiments/slurm/self_heal.py '$POST_YAML' '$SCALE'" || \
  { echo "[warn] self-heal failed (yaml might already be in isolated format)"; }

# Sync atoms shards (idempotent; flock prevents concurrent download race conditions)
mkdir -p "$PARQUET_DIR"
(
  flock 9
  for PDB in $PROTEINS; do
    run_py "python scripts/experiments/slurm/fetch_atoms.py --pdb-id '$PDB' --out '$PARQUET_DIR'" || true
  done
) 9>"$PARQUET_DIR/.fetch.lock"

# Run fixed RL loop for same set of proteins under n10 scale sequentially (--tag prevents overwrites)
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
echo "[n10] completed -> coverage: ${COV_CSV}"
echo "Find survivors: grep 'Survivor mutant' slurm_logs/10_*.out ; pseudo-labels are stored at ${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/pseudo_labels/"
