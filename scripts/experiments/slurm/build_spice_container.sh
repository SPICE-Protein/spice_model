#!/bin/bash
# Fetches the SPICE custom container: downloads directly from the Sylabs Container Library, using smoke Check 6 as a gatekeeper.
# Usage (run on login node via normal bash, bypassing local build):
#   bash scripts/experiments/slurm/build_spice_container.sh
# Overrides:
#   LIB  URI of the image repository (defaults to library://redelectricity/spice/spice-tf)
#   OUT  path to output SIF image (defaults to $HOME/spice/spice-tf.sif)
#
# Description: The container is built from scratch with ubuntu:22.04 + miniconda (/opt/conda, py3.12, numpy 2.5.2) + TF + cp312 engine wheel
#   (using the recipe in spice-tf.def, which incorporates a build-time matmul gate). Local rebuilding is bypassed in favor of 
#   pulling the pre-built image. To rebuild locally, execute `singularity build spice-tf.sif spice-tf.def` manually.
set -uo pipefail

LIB=${LIB:-library://redelectricity/spice/spice-tf}
MODEL_ROOT=${MODEL_ROOT:-$HOME/spice/model}
SPICE_ROOT=${SPICE_ROOT:-$HOME/spice}
OUT=${OUT:-$SPICE_ROOT/spice-tf.sif}

# ---- Container Runtime: singularity (FUSE mounts for apptainer are unavailable on this cluster, use singularity) ----
module load singularity/3.7.3 2>/dev/null || true
command -v singularity >/dev/null || { echo "singularity unavailable (module load singularity/3.7.3)"; exit 1; }

# ---- Pull SIF image (library:// pulls default to container_name.sif, then moves to OUT) ----
OUT_DIR=$(dirname "$OUT")
mkdir -p "$OUT_DIR"
echo "== singularity pull -F $LIB =="
( cd "$OUT_DIR" && singularity pull -F "$LIB" ) || {
  echo "[error] Pull failed. Common causes: ① Image is private -> run singularity remote login first;"
  echo "        ② Sylabs endpoint connection failed or repository URI is incorrect;"
  echo "        ③ Network connectivity issues."
  exit 1
}
if [ -f "$OUT_DIR/spice-tf.sif" ]; then
  mv -f "$OUT_DIR/spice-tf.sif" "$OUT"
fi
[ -f "$OUT" ] || { echo "[error] SIF image not found at expected location: $OUT"; ls -la "$OUT_DIR"; exit 1; }
echo "[pull] OK → $OUT"

# ---- Gating test: run smoke Check 6 inside the new SIF using its internal Python (/opt/conda) ----
CONDA_SH=/opt/conda/etc/profile.d/conda.sh \
SIF="$OUT" \
bash "$MODEL_ROOT/scripts/experiments/slurm/smoke_6qqe_login.sh" --check6-only
ret=$?
if [ $ret -eq 0 ]; then
  echo ""
  echo "✅ Container gate tests passed successfully! To run RL workloads, configure:"
  echo "   export CONDA_SH=/opt/conda/etc/profile.d/conda.sh SIF=$OUT"
  echo "   bash $MODEL_ROOT/scripts/experiments/slurm/smoke_6qqe_login.sh      # Full smoke test"
  echo "   bash $MODEL_ROOT/scripts/experiments/slurm/run_45000.sbatch         # Production run"
else
  echo "❌ Container gate tests failed: Check 6 indicates matmul is still buggy -> image is unusable, please rebuild/re-push."
fi
exit $ret
