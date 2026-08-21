#!/bin/bash
#SBATCH --job-name=soup
#SBATCH --partition=<PARTITION>        # ← check via sinfo -s, use same partition as other RL runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/soup_%A.out
#SBATCH --error=slurm_logs/soup_%A.err

#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/soup_%A.out
#SBATCH --error=slurm_logs/soup_%A.err

# ===== Primordial Soup Physical Screening (soup_screen): random aa sequence -> engine physical quick_check =====
# Validates the "physics as the first selector" hypothesis (sampling survey framework). Since most random chains collapse rapidly -> highly cost-effective (hundreds of chains = a few core-hours).
# Submit:   cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_soup_screen.sbatch
# Override: N=500 LEN=80 MODE=denovo START=coil sbatch ...    # Default run configuration
#           MODE=backbone WT_CIF=data/7QF3.cif sbatch ...     # Natural backbone + random sidechains
# Thread rule: HPC thread allocation must be explicitly specified (never set to "auto"). WORKERS * THREADS == --cpus-per-task (default 2 * 2 = 4c)
# Prereqs:  Upload latest soup_screen.py (TF-free, safe to run even without SSE41); mode=backbone requires data/7QF3.cif
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

MODE=${MODE:-denovo}
N=${N:-200}          # Sample survey scale (Rule of Three: N=200, 0 survivors -> upper bound on acceptance rate is ~1.5%)
LEN=${LEN:-80}
ALPHABET=${ALPHABET:-prebiotic}
START=${START:-coil}
WT_CIF=${WT_CIF:-data/7QF3.cif}
WORKERS=${WORKERS:-2}   # process concurrency (forks); WORKERS * THREADS == cpus-per-task
THREADS=${THREADS:-2}   # rayon engine threads per worker (explicitly locked, never auto-detected)
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-${MODEL_ROOT}/../data/data_efficiency/soup_screen}
mkdir -p "$OUT_DIR" slurm_logs

# ⚠️ Hygon C86 CPUs require forcing SSE41 (though soup_screen is TF-free and doesn't run matmul, this is for consistency)
run_py() {
  singularity exec "$SIF" bash -lc "
    source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$MODEL_ROOT' && \
    export PYTHONPATH='$MODEL_ROOT' RAYON_NUM_THREADS='$THREADS' \
           HF_ENDPOINT='https://hf-mirror.com' POLARS_SKIP_CPU_CHECK='1' CUDA_VISIBLE_DEVICES='' \
           OMP_NUM_THREADS='$THREADS' MKL_NUM_THREADS='$THREADS' OPENBLAS_NUM_THREADS='$THREADS' NUMEXPR_NUM_THREADS='$THREADS' \
           ONEDNN_DEFAULT_FPMATH_MODE=FP32 ONEDNN_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' \
           DNNL_MAX_CPU_ISA='${DNNL_MAX_CPU_ISA:-SSE41}' TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
    $1
  "
}

OUT="$OUT_DIR/soup_${MODE}_n${N}_len${LEN}_${ALPHABET}.csv"
MMIFS="${SOUP_MMIFS:-${OUT_DIR}/mmifs}"   # physically accepted survivors -> mmCIF, fed into run_soup_rl.sbatch for SPICE optimization
echo "=== Primordial Soup Screen ${MODE} n=${N} len=${LEN} alpha=${ALPHABET} start=${START} (${WORKERS} workers x ${THREADS} threads = $((WORKERS*THREADS)) cores) start $(date) ==="
ARGS="--mode '$MODE' --n '$N' --alphabet '$ALPHABET' --workers '$WORKERS' --seed '$SEED' --out '$OUT' --out-mmifs '$MMIFS'"
if [ "$MODE" = "denovo" ]; then
  ARGS="$ARGS --len '$LEN' --start '$START'"
else
  ARGS="$ARGS --wt-cif '$WT_CIF'"
fi
if run_py "python scripts/experiments/soup_screen.py $ARGS"; then
  echo "=== done -> $OUT ==="
  echo "Check results: cat $OUT | head; compare m5 with WT (~0.047) to see signal strength"
  echo "Evolve via RL: sbatch scripts/experiments/slurm/run_soup_rl.sbatch (SOUP_MMIFS=$MMIFS)"
else
  echo "=== FAILED (check soup_*.err) ==="
  exit 1
fi
