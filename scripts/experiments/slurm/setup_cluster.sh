#!/bin/bash
set -euo pipefail

MINICONDA_DIR=${MINICONDA_DIR:-$HOME/miniconda3}
CONDA_SH="$MINICONDA_DIR/etc/profile.d/conda.sh"
ENV_NAME=${ENV_NAME:-spice}
SPICE_ROOT=${SPICE_ROOT:-$HOME/spice}
MODEL_ROOT="$SPICE_ROOT/model"

[d "$MODEL_ROOT/spice_rl" ] || { echo "[error] Missing $MODEL_ROOT (unpack spice_cluster.tar.gz into ~ first)"; exit 1; }
WHEEL=$(ls "$SPICE_ROOT"/spice_engine-*.whl 2>/dev/null | head -1 || true)
[ -n "$WHEEL" ] || echo "[warn] No engine wheel found; remember to pip install spice_engine-*.whl later"

echo "== 1/5 Miniconda =="
if [ -x "$MINICONDA_DIR/bin/conda" ]; then
  echo "  Exists, skipping"
else
  echo "  Downloading Miniconda from Tsinghua TUNA mirror (faster)..."
  wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_25.7.0-2-Linux-x86_64.sh --no-check-certificate -O /tmp/miniconda.sh \
    || { echo "  TUNA mirror failed, falling back to official source..."; wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.7.0-2-Linux-x86_64.sh --no-check-certificate -O /tmp/miniconda.sh; }
  bash /tmp/miniconda.sh -b -f -p "$MINICONDA_DIR"
fi
source "$CONDA_SH"
set +u; conda activate base; set -u

echo "== 2/5 Configuring Tsinghua TUNA mirror for Conda (~/.condarc) =="
cat > "$HOME/.condarc" <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
conda clean -i -y 2>/dev/null || true   

echo "== 3/5 Environment $ENV_NAME (Python 3.12) =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "  Exists, skipping"
else
  conda create -n "$ENV_NAME" python=3.12 -y \
    || conda create -n "$ENV_NAME" python=3.12 -y -c conda-forge --override-channels
fi
set +u; conda activate "$ENV_NAME"; set -u

echo "== 4/5 Configuring Tsinghua TUNA mirror for pip =="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn 2>/dev/null || true

echo "== 5/5 Engine Wheel & Dependencies =="
MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.aliyun.com/pypi/simple"
  "https://mirrors.huaweicloud.com/repository/pypi/simple"
  "https://pypi.org/simple"
)
install_numpy() {
  local want="$1"
  for m in "${MIRRORS[@]}"; do
    echo "  [numpy] Attempting $m ($want)"
    if pip install --only-binary=:all: --no-cache-dir -i "$m" "$want"; then return 0; fi
  done
  return 1
}
numpy_ok=0
if install_numpy "numpy==2.5.2"; then
  numpy_ok=1
elif install_numpy "numpy>=2.0"; then
  echo "  [warn] Cluster mirror does not host numpy 2.5.2; installed latest available 2.x (engine ABI validation follows)"
  numpy_ok=1
fi
[ "$numpy_ok" = 1 ] || { echo "[error] Failed to install numpy"; exit 1; }

[ -n "$WHEEL" ] && pip install --no-deps "$WHEEL"

ok=0
for m in "${MIRRORS[@]}"; do
  echo "  [deps] Attempting $m"
  if pip install --only-binary=:all: --no-cache-dir -r "$MODEL_ROOT/requirements.txt" -i "$m"; then ok=1; break; fi
done
[ "$ok" = 1 ] || { echo "[error] No mirrors could satisfy the dependency requirements (particularly tensorflow==2.21)"; exit 1; }

echo "== Validation =="
cd "$MODEL_ROOT"
PYTHONPATH="$MODEL_ROOT" python - <<'PY'
import sys
import tensorflow as tf
import keras
print("TF", tf.__version__, "| Keras", keras.__version__)
try:
    import spice_engine
    print("spice_engine OK ->", spice_engine.__file__)
except Exception as e:
    print("spice_engine FAIL:", e)
    sys.exit(1)
import spice_rl  # noqa: F401
print("spice_rl import OK")
PY
echo
echo "[done] Environment is ready. Next steps:"
echo "  1) sinfo -s   # check partition (e.g. kshcnormal), insert into run_coverage_rl.sbatch --partition"
echo "  2) Edit run_coverage_rl.sbatch PROTEINS=... (margin 0.4~0.9 proteins)"
echo "  3) (Optional) Run interactive probe before submission: bash scripts/experiments/slurm/probe_threads.sh"
echo "  4) cd ~/spice/model && sbatch scripts/experiments/slurm/run_coverage_rl.sbatch"
