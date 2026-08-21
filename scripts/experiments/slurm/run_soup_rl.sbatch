#!/bin/bash
#SBATCH --job-name=soup_rl
#SBATCH --partition=<PARTITION>        # ← check via sinfo -s, use same partition as other RL runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/soup_rl_%A.out
#SBATCH --error=slurm_logs/soup_rl_%A.err

#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/soup_rl_%A.out
#SBATCH --error=slurm_logs/soup_rl_%A.err

# ===== Primordial Soup RL: evolving physically accepted random survivors via SPICE =====
# Pipeline: soup_screen --out-mmifs (extracts stable random chains -> mmCIF) -> runs train_post for each parent 
#           (Path A explores collapse boundaries + Path B mutations) to see if physical seeds can be evolved into highly stable mutants with functional preservation.
# Submit:   cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_soup_rl.sbatch
# Override: SOUP_MMIFS=<dir> POST_CONFIG=configs/HPC/posttrain.yaml MAX_EPISODES=10 sbatch ...
# Prereqs:  ① soup_screen.py (--out-mmifs option) has generated mmCIFs; ② latest spice_engine wheel (deadlocks fixed) is installed in the container; ③ posttrain config + pretrain checkpoint are in place
# Threads:  Evaluates parents sequentially (one RL run at a time), with THREADS=4 to saturate the engine's rayon thread pool (locked threads, never auto-detected)
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

SOUP_MMIFS=${SOUP_MMIFS:-${MODEL_ROOT}/../data/data_efficiency/soup_screen/mmifs}
POST_CONFIG=${POST_CONFIG:-configs/HPC/posttrain.yaml}
MAX_EPISODES=${MAX_EPISODES:-10}
THREADS=${THREADS:-4}
mkdir -p slurm_logs

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

shopt -s nullglob
CIFS=("$SOUP_MMIFS"/*.cif)
if [ ${#CIFS[@]} -eq 0 ]; then echo "[warn] No mmCIF files found under $SOUP_MMIFS (run soup_screen --out-mmifs first)"; exit 1; fi
echo "=== Primordial Soup RL: ${#CIFS[@]} physically accepted parents | config=$POST_CONFIG | max_ep=$MAX_EPISODES | THREADS=$THREADS start $(date) ==="
NOK=0
for CIF in "${CIFS[@]}"; do
  TAG="soup_$(basename "$CIF" .cif)"
  echo "=== Running RL on $CIF (tag=$TAG) start $(date) ==="
  if run_py "python -m spice_rl.train_post --config '$POST_CONFIG' --structure '$CIF' --tag '$TAG' --max-episodes '$MAX_EPISODES'"; then
    echo "=== $TAG done ==="
  else
    echo "=== $TAG FAILED (check soup_rl_*.err) ==="
    NOK=$((NOK+1))
  fi
done
echo "[soup_rl] completed ${NOK} failures / ${#CIFS[@]} parents"
echo "Check survivors: grep -E 'n_survivors|Survivor' slurm_logs/soup_rl_*.out; pseudo-labels are stored in the config pseudo_label_dir (isolated by tag)"
