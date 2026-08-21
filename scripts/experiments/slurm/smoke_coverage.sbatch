#!/bin/bash
#SBATCH --job-name=spice_smoke
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as formal runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/smoke_%A.out
#SBATCH --error=slurm_logs/smoke_%A.err

# ===== P1 Smoke Test: 1 protein * n10 * 2 episodes, verifies full submission execution pathway =====
# Submit:   cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/smoke_coverage.sbatch
# Success:  Logs verify: [self-check] reward normalization: True | clamp A+B: True | place_sidechain: True | pred_ca default: True
#           + completes 2 episodes + n10/runs/posttrain/coverage.csv is created
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

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

cd "$MODEL_ROOT"
SCALE=10
POST_YAML="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/posttrain.yaml"
PARQUET_DIR="${MODEL_ROOT}/../data/parquet_hpc"
PDB=7qf3
mkdir -p slurm_logs

echo "[smoke] n${SCALE} ${PDB} max-episodes=2"
test -f "$POST_YAML" || { echo "Missing $POST_YAML"; exit 1; }

# 1) Self-heal (output isolation)
run_py "python scripts/experiments/slurm/self_heal.py '$POST_YAML' '$SCALE'" || echo "[warn] self-heal failed"

# 2) Sync shard (should execute [local] already exists)
run_py "python scripts/experiments/slurm/fetch_atoms.py --pdb-id '$PDB' --out '$PARQUET_DIR'" || true

# 3) Run 2 episodes (real train_post, incorporating self-check + build + MD simulation)
echo "=== [smoke] ${PDB} RL start $(date) ==="
run_py "python -m spice_rl.train_post --config '$POST_YAML' --parquet-dir '$PARQUET_DIR' --pdb-id '${PDB^^}' --tag '${PDB}_smoke' --max-episodes 2"
echo "=== [smoke] ${PDB} RL done  $(date) ==="

echo "[smoke] Outputs:"
ls -la "${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/runs/posttrain/" 2>/dev/null || true
tail -20 "${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/runs/posttrain/coverage.csv" 2>/dev/null || echo "(coverage.csv unavailable yet)"
