#!/bin/bash
#SBATCH --job-name=fpdb_corr
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/fpdb_%A.out
#SBATCH --error=slurm_logs/fpdb_%A.err

# ===== FPDB Correlation Pilot: engine stability signals ↔ FireProtDB experimental ΔΔG (C-experiment validation layer) =====
# Establishes the bridge between "engine stability metrics ↔ experimental ΔΔG" on selected proteins (via compute_fpdb_ddg_corr.py).
# Evaluates correlation between FireProtDB experimental ΔΔG and metrics()['stability_margin'] + Δmargin + ΔU.
# Can be submitted in parallel with run_ddg_proxy.sbatch (independent MD runs).
# Submit:   cd ~/spice/model && sbatch scripts/experiments/slurm/run_fpdb_corr.sbatch
# Parallel runs (recommended: submit one job per protein):
#     FPDB_PROTEINS="1L63" sbatch scripts/experiments/slurm/run_fpdb_corr.sbatch
#     FPDB_PROTEINS="1BNI" sbatch scripts/experiments/slurm/run_fpdb_corr.sbatch
#     FPDB_PROTEINS="1EM7" sbatch scripts/experiments/slurm/run_fpdb_corr.sbatch
# Override: FPDB_PROTEINS="1L63 1BNI" N=100 sbatch scripts/experiments/slurm/run_fpdb_corr.sbatch
# Prereqs:  Upload latest compute_fpdb_ddg_corr.py + data/stability_benchmark/fpdb_pilot.csv
# ⚠️ Compute nodes lack internet access (DNS resolution fails) -> download CIF files in advance to CIF_DIR on login node:
#     mkdir -p ~/spice/data/fpdb_pdbs && cd ~/spice/data/fpdb_pdbs
#     for p in 1L63 1BNI 1EM7; do wget -q "https://files.rcsb.org/download/$p.cif" -O $p.cif; done
#     (If login nodes also lack internet, download mmCIF from RCSB website manually and upload)
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

FPDB_PROTEINS=${FPDB_PROTEINS:-"1L63 1BNI 1EM7"}
N=${N:-50}   # number of mutations to evaluate per protein. At n=10, 1EM7 yields r~0.6 but is not statistically significant (p~0.06) -> raised to 50 to gain statistical power (~1-2min/mutation, 3 proteins in parallel)
OUT_DIR=${OUT_DIR:-$MODEL_ROOT/../data/data_efficiency/fpdb_corr}
CIF_DIR=${CIF_DIR:-$MODEL_ROOT/../data/fpdb_pdbs}
mkdir -p "$OUT_DIR" "$CIF_DIR" slurm_logs

# ⚠️ Hygon C86 CPUs must suppress AVX2/FMA (avoids a silent tf/oneDNN matmul bug where batch_size > 1 yields global NaNs) by forcing SSE41
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

NOK=0
for P in $FPDB_PROTEINS; do
  OUT="$OUT_DIR/fpdb_corr_${P}.csv"
  echo "=== FPDB ${P} (n=${N}) start $(date) ==="
  if run_py "python scripts/compute_fpdb_ddg_corr.py --pdb '$P' --n '$N' --no-reuse --cif-dir '$CIF_DIR' --out '$OUT'"; then
    echo "=== FPDB ${P} done -> $OUT ==="
  else
    echo "=== FPDB ${P} FAILED ==="
    NOK=$((NOK+1))
  fi
done
if [ "$NOK" -gt 0 ]; then echo "[warn] ${NOK} proteins failed (check fpdb_*.err)"; fi
echo "[fpdb_corr] completed -> $OUT_DIR"
