#!/bin/bash
#SBATCH --job-name=spice_45k
#SBATCH --partition=<PARTITION>        # ← sinfo -s to check, use the same partition as formal runs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=48:00:00                # ← Full 30 episodes * 3 structures (2 independent sequences) can take ~45h (with NaN safeguards, each episode runs actual MD for ~30min); if partition limits to 24h, it will be killed mid-way with partial results preserved
#SBATCH --output=slurm_logs/45k_%A.out
#SBATCH --error=slurm_logs/45k_%A.err

# ===== Production 45k: runs n45000 (full prior) * 4 structures (3 independent sequences: 7QF3 + Lysozyme 6QQE/8D8F conformations + RNase A 1JVT); outputs survivors, pseudo-labels, and coverage =====
# Submit:   cd ~/spice/model && mkdir -p slurm_logs && sbatch scripts/experiments/slurm/run_45000.sbatch
# Success:  6QQE/8D8F (two structural entries for the same Lysozyme sequence, margin ~0.85/0.86, most representative stable proteins) generate viable mutants + pseudo-labels;
#           7QF3, even with 0 survivors, still records coverage points (beyond-PDB evidence); 1JVT (RNase A) provides the 4-protein validation matching paper §3.5 (native-condition agreement) — α+β fold + native acid/cold denaturation, having clear collapse boundaries for RL exploration.
# Prereqs:  Upload latest es.py (NaN safeguards) + md_env.py (env isfinite) + train_post.py (pred_ca scaling fallback + Restart Gate).
# Pro tip:  Override with PROTEINS="1jvt" to run only new proteins when rerunning, avoiding duplicate calculations.
set -euo pipefail

CONDA_SH=${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=spice
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
# Container Runtime: singularity (FUSE mounts for apptainer are unavailable on this cluster, use singularity)
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

PARQUET_DIR=${MODEL_ROOT}/../data/parquet_hpc
PROTEINS=${PROTEINS:-"7qf3 6qqe 8d8f 1jvt"}   # 4 structures / 3 independent sequences: 7QF3 + Lysozyme (6QQE/8D8F conformations) + RNase A 1JVT
# Run only new proteins when rerunning: PROTEINS="1jvt" sbatch scripts/experiments/slurm/run_45000.sbatch
# Full 40 episodes (SAC requires ~13-20 episodes to fill buffer and start training; setting too low prevents any learning); override with MAX_EPISODES=20 if needed
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
SCALE=45000
POST_YAML="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/posttrain.yaml"
COV_CSV="${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/runs/posttrain/coverage.csv"
mkdir -p slurm_logs

echo "[45k] config=${POST_YAML}  max_episodes=${MAX_EPISODES}"
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

# Run fixed RL loop for same set of proteins under n45000 scale sequentially (--tag prevents overwrites)
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
if [ "$NOK" -gt 0 ]; then echo "[warn] n${SCALE}: {NOK} proteins failed"; fi
echo "[45k] completed -> coverage: ${COV_CSV}"
echo "Find survivors: grep 'Survivor mutant' slurm_logs/45k_*.out ; pseudo-labels are stored at ${MODEL_ROOT}/../data/data_efficiency/n${SCALE}/pseudo_labels/"
