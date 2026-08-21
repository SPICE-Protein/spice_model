#!/bin/bash
#SBATCH --job-name=ddg_proxy
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G         # 2 workers × ~2-3G; if memory limits are lower, set WORKERS=1 (with 2c/3G allocation)
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/ddg_%A.out
#SBATCH --error=slurm_logs/ddg_%A.err

# ===== ΔΔG Proxy (m5 Charge Balance Edition): stability rankings for survivors of n10/n45000 scale across 4 proteins (C-experiment bedrock) =====
# Method: Replaces the legacy temperature-margin scanning (which saturates to 105/105 survivors at mild conditions on HPC). 
#         The m5 charge balance ratio (m5_ratio = mutant/WT, <1 is more stable) is precisely the signal RL uses to detect collapse. 
#         Short sprints resolve stability boundaries without expensive temperature grid scans, achieving ~9x speedups.
#         Concurrency: --workers uses fork-multiprocessing to evaluate mutant sequences in parallel within each protein. 
#         ⚠️ HPC thread assignment must be explicitly specified (never set to "auto" as it oversubscribes CPU cores on login/compute nodes).
#         Thread model: each worker binds RAYON_NUM_THREADS=THREADS (default 2). WORKERS * THREADS == --cpus-per-task (default 2 * 2 = 4c) to avoid oversubscription.
# Inputs:  pseudo-label npz files (mutant sequences + respective env_fail conditions) + parquet atoms (WT baseline, matching RL source and chain)
# Outputs: OUT_DIR/ddg_proxy_n{scale}_{pdb}.csv (ranked by m5_ratio / margin / dU) + _report.txt (highlighting SG-SG disulfide candidates)
# Submit:  cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_ddg_proxy.sbatch
# Concurrency (recommended: run both scales in parallel as two separate jobs):
#     SCALES="10"    sbatch scripts/experiments/slurm/run_ddg_proxy.sbatch   # n10
#     SCALES="45000" sbatch scripts/experiments/slurm/run_ddg_proxy.sbatch   # n45000
# Override: SCALES="45000" PROTEINS="1jvt" WORKERS=4 THREADS=2 sbatch scripts/experiments/slurm/run_ddg_proxy.sbatch  # 4 * 2 = 8c
# Prereqs:  Upload latest compute_ddg_proxy_m5.py (incorporates --wt-parquet-dir and --workers); n10/n45000 pseudo_labels must exist on cluster.
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

PARQUET_DIR=${PARQUET_DIR:-${MODEL_ROOT}/../data/parquet_hpc}
SCALES=${SCALES:-"10 45000"}
PROTEINS=${PROTEINS:-"7qf3 6qqe 8d8f 1jvt"}
WORKERS=${WORKERS:-2}   # mutant evaluation concurrency (via process forks); WORKERS * THREADS == --cpus-per-task (2 * 2 = 4c)
THREADS=${THREADS:-2}   # rayon engine threads per worker (explicitly locked, never auto-detected)
OUT_DIR=${OUT_DIR:-${MODEL_ROOT}/../data/data_efficiency/ddg_proxy}
mkdir -p "$OUT_DIR" slurm_logs

# ⚠️ Hygon C86 CPUs must suppress AVX2/FMA (avoids a silent tf/oneDNN matmul bug where batch_size > 1 yields global NaNs) by forcing SSE41
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

# Sync atoms shards (idempotent; flock prevents concurrent download race conditions)
(
  flock 9
  for PDB in $PROTEINS; do
    run_py "python scripts/experiments/slurm/fetch_atoms.py --pdb-id '${PDB^^}' --out '$PARQUET_DIR'" || true
  done
) 9>"$PARQUET_DIR/.fetch.lock"

NOK=0
for SCALE in $SCALES; do
  PSEUDO_DIR="$MODEL_ROOT/../data/data_efficiency/n${SCALE}/pseudo_labels"
  test -d "$PSEUDO_DIR" || { echo "[skip] Missing pseudo-label directory: $PSEUDO_DIR"; continue; }
  for PDB in $PROTEINS; do
    N=$(ls "$PSEUDO_DIR"/pseudo_${PDB}_ep*.npz 2>/dev/null | wc -l | tr -d ' ')
    if [ "${N:-0}" -eq 0 ]; then echo "[skip] n${SCALE} ${PDB} has no surviving pseudo-labels"; continue; fi
    OUT="$OUT_DIR/ddg_proxy_n${SCALE}_${PDB}.csv"
    echo "=== [n${SCALE}] ${PDB} ΔΔG Proxy (${N} survivors) start $(date) ==="
    if run_py "python scripts/compute_ddg_proxy_m5.py \
        --npz-glob '$PSEUDO_DIR/pseudo_${PDB}_ep*.npz' \
        --wt-parquet-dir '$PARQUET_DIR' --wt-pdb-id '${PDB^^}' \
        --candidates-csv '$MODEL_ROOT/../data/data_efficiency/n${SCALE}/runs/posttrain/pathb_candidates.csv' \
        --workers '$WORKERS' \
        --out '$OUT'"; then
      echo "=== [n${SCALE}] ${PDB} done -> $OUT ==="
    else
      echo "=== [n${SCALE}] ${PDB} FAILED ==="
      NOK=$((NOK+1))
    fi
  done
done
if [ "$NOK" -gt 0 ]; then echo "[warn] ${NOK} proteins failed (check ddg_*.err)"; fi
echo "[ddg_proxy] completed -> $OUT_DIR"
echo "Summary: cat $OUT_DIR/ddg_proxy_n10_*.csv $OUT_DIR/ddg_proxy_n45000_*.csv"
