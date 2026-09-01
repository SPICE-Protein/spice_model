#!/bin/bash
#SBATCH --job-name=soup_rl_full
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as RL jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/soup_rl_full_%A_%a.out
#SBATCH --error=slurm_logs/soup_rl_full_%A_%a.err
#SBATCH --array=0-9

# ===== Primordial Soup RL 全量（m5 中位切 low/high 两桶 × shard 数组作业，n=0 纯物理）=====
# 前置: split_soup_parents.py 已生成 <SOUP_ROOT>/{low,high}/shard_XX/ + posttrain_soup.yaml（输出已按桶隔离）
# 提交（每桶一个数组作业，1 任务 = 1 shard = 10 亲本）:
#   cd ~/spice/model && mkdir -p slurm_logs
#   BUCKET=low  sbatch --array=0-9 run_soup_rl_full.sbatch
#   BUCKET=high sbatch --array=0-9 run_soup_rl_full.sbatch
# 覆盖: SOUP_ROOT=... BUCKET=low|high MAX_EPISODES=10 THREADS=4（线程钉死，绝不 auto）
# 验证 n=0（随机初始化，不碰 45k 先验）: grep -c "Successfully loaded Pre-train weights" slurm_logs/soup_rl_full_*.out  # 必须全 0
# 收数: 存活突变体/伪标签在各自桶 pseudo_labels/；metrics 在 <bucket>/runs/posttrain/metrics.csv
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

SOUP_ROOT=${SOUP_ROOT:-$HOME/spice/data/data_efficiency/soup_rl/parents}
BUCKET=${BUCKET:-low}
SHARD=$(printf "shard_%02d" "${SLURM_ARRAY_TASK_ID}")
SOUP_MMIFS="$SOUP_ROOT/$BUCKET/$SHARD"
POST_CONFIG=${POST_CONFIG:-$SOUP_ROOT/$BUCKET/posttrain_soup.yaml}
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
           DNNL_MAX_CPU_ISA='${ONEDNN_MAX_CPU_ISA:-SSE41}' TF_ENABLE_ONEDNN_OPTS=0 TF_XLA_FLAGS='--tf_xla_auto_jit=0' && \
    $1
  "
}

shopt -s nullglob
CIFS=("$SOUP_MMIFS"/*.cif)
if [ ${#CIFS[@]} -eq 0 ]; then echo "[warn] 无 mmCIF in $SOUP_MMIFS（先跑 split_soup_parents.py）"; exit 1; fi
echo "=== soup RL full [$BUCKET/$SHARD] ${#CIFS[@]} parents | config=$POST_CONFIG | max_ep=$MAX_EPISODES | THREADS=$THREADS | $(date) ==="
NOK=0
for CIF in "${CIFS[@]}"; do
  TAG="soup_$(basename "$CIF" .cif)"
  echo "=== RL on $CIF (tag=$TAG) start $(date) ==="
  if run_py "python -m spice_rl.train_post --config '$POST_CONFIG' --structure '$CIF' --tag '$TAG' --max-episodes '$MAX_EPISODES'"; then
    echo "=== $TAG done ==="
  else
    echo "=== $TAG FAILED（重启可重跑该 shard；train_post 有 restart-gate 幂等）==="
    NOK=$((NOK+1))
  fi
done
echo "[soup_rl_full] $BUCKET/$SHARD ${NOK} failures / ${#CIFS[@]} parents | done $(date)"
