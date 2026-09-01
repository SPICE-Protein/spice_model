#!/bin/bash
#SBATCH --job-name=soup_prebio
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/soup_prebio_%A.out
#SBATCH --error=slurm_logs/soup_prebio_%A.err

# ===== prebiotic 子集对照（mutation_alphabet=GAVDESPTRK，便宜的决定性测试）=====
# 前置: build_prebiotic_subset.py 已生成 <SUBSET_ROOT>/{low,high}/*.cif + posttrain_soup.yaml
# 提交（每桶一个任务，约 4 亲本 × MAX_EPISODES 集，~5-8h/4 核 ≈ ¥3/桶）:
#   SUBSET_ROOT=$HOME/spice/data/data_efficiency/soup_rl_prebiotic/subset \
#   BUCKET=low  MAX_EPISODES=15 sbatch scripts/experiments/slurm/run_soup_prebiotic_subset.sbatch
#   SUBSET_ROOT=... BUCKET=high MAX_EPISODES=15 sbatch ...
# 判定: 看 <SUBSET_ROOT>/<BUCKET>/runs/posttrain/pathb_candidates.csv 是否有 survived=1
#   + pseudo_labels/*.npz 数量。full20 同批亲本在同样窗口大量出幸存者 → prebiotic 出/不出即结论。
# 验证 n=0: grep -c "Successfully loaded Pre-train weights" slurm_logs/soup_prebio_*.out  # 必须 0
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

SUBSET_ROOT=${SUBSET_ROOT:-$HOME/spice/data/data_efficiency/soup_rl_prebiotic/subset}
BUCKET=${BUCKET:-low}
SUBSET_DIR="$SUBSET_ROOT/$BUCKET"
POST_CONFIG=${POST_CONFIG:-$SUBSET_ROOT/$BUCKET/posttrain_soup.yaml}
MAX_EPISODES=${MAX_EPISODES:-15}
THREADS=${THREADS:-4}
mkdir -p slurm_logs

run_py() {
  singularity exec "$SIF" bash -lc "
    source '$CONDA_SH' && conda activate '$CONDA_ENV' && cd '$MODEL_ROOT' && \
    export PYTHONPATH='$MODEL_ROOT' RAYON_NUM_THREADS='$THREADS' \
           HF_ENDPOINT='https://hf-mirror.com' POLARS_SKIP_CPU_CHECK='1' CUDA_VISIBLE_DEVICES='' \
           OMP_NUM_THREADS='$THREADS' MKL_NUM_THREADS='$THREADS' OPENBLAS_NUM_THREADS='$THREADS' NUMEXPR_NUM_THREADS='$THREADS' \
           ONEDNN_DEFAULT_FPMATH_MODE=FP32 ONEDNN_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' \
           DNNL_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
    $1
  "
}

shopt -s nullglob
CIFS=("$SUBSET_DIR"/*.cif)
if [ ${#CIFS[@]} -eq 0 ]; then echo "[warn] 无 mmCIF in $SUBSET_DIR（先跑 build_prebiotic_subset.py）"; exit 1; fi
echo "=== soup prebiotic subset [$BUCKET] ${#CIFS[@]} parents | config=$POST_CONFIG | max_ep=$MAX_EPISODES | $(date) ==="
NOK=0
for CIF in "${CIFS[@]}"; do
  TAG="soup_$(basename "$CIF" .cif)"
  echo "=== RL on $CIF (tag=$TAG) start $(date) ==="
  if run_py "python -m spice_rl.train_post --config '$POST_CONFIG' --structure '$CIF' --tag '$TAG' --max-episodes '$MAX_EPISODES'"; then
    echo "=== $TAG done ==="
  else
    echo "=== $TAG FAILED ==="
    NOK=$((NOK+1))
  fi
done
echo "[soup_prebio] $BUCKET ${NOK} failures / ${#CIFS[@]} parents | done $(date)"
