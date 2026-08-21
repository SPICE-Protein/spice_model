#!/bin/bash
set -euo pipefail

SIF=${SIF:-/public/software/apps/DeepLearning/singularity/ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-mkl-py3.10-v0.1.sif}
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

echo "== Checking container glibc version =="
singularity exec "$SIF" bash -c "ldd --version | head -1"
echo "== Executing setup_cluster.sh inside the container (installs TF 2.21, numpy 2.5.2, engine wheels, dependencies + runs validation) =="
singularity exec "$SIF" bash ~/spice/model/scripts/experiments/slurm/setup_cluster.sh

echo "[done] Container environment is ready. Ensure that SIF= is pointing to this image inside run_coverage_rl.sbatch before submission."
