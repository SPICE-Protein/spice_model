#!/bin/bash
# 6QQE born-NaN SAC cluster smoke test — login node direct execution version (no sbatch required)
# Usage (execute from login node using normal bash, bypassing sbatch):
#   cd ~/spice/model && bash scripts/experiments/slurm/smoke_6qqe_login.sh
# Overrides: MODEL_ROOT / SIF / CONDA_SH / POST_YAML environment variables.
set -uo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
POST_YAML=${POST_YAML:-${MODEL_ROOT}/../data/data_efficiency/n45000/posttrain.yaml}

module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }
echo "[smoke] runtime = singularity"
echo "[smoke] Login node direct run (non-sbatch)"
echo "[smoke] MODEL_ROOT = $MODEL_ROOT"
echo "[smoke] POST_YAML  = $POST_YAML"
echo "[smoke] SIF        = $SIF"
test -f "$POST_YAML" || { echo "[smoke] Missing $POST_YAML — verify n45000 dataset is correctly deployed"; exit 1; }

# Identical environment settings to run_py in run_45000.sbatch (forces CPU execution and deterministic TF flags)
# Pass through "$@" parameters (e.g. --check6-only container gateway tests)
singularity exec "$SIF" bash -lc "
  source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$MODEL_ROOT' && \
  export PYTHONPATH='$MODEL_ROOT' RAYON_NUM_THREADS=2 \
         HF_ENDPOINT='https://hf-mirror.com' POLARS_SKIP_CPU_CHECK='1' \
         CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
         OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ONEDNN_DEFAULT_FPMATH_MODE=FP32 \
         ONEDNN_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' DNNL_MAX_CPU_ISA='${DNNL_MAX_CPU_ISA:-SSE41}' \
         TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
  python scripts/smoke_6qqe_login.py $*
"
ret=$?
echo "[smoke] exit=$ret"
exit $ret
