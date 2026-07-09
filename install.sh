#!/usr/bin/env bash
# ============================================================================
# Grimoire — one-shot environment setup (verified 2026-07 on CUDA 11.8 / Tesla T4)
#
# Creates the `SVG` conda env, installs CUDA-enabled PyTorch 2.0.1, builds diffvg
# from source with GPU support, and installs the pinned Python deps.
#
# Usage:
#   bash install.sh [ENV_NAME] [DIFFVG_DIR]
#     ENV_NAME    conda env to create      (default: SVG)
#     DIFFVG_DIR  path to a diffvg checkout (default: ./diffvg, cloned if absent)
#
# Requires: conda/mamba on PATH, a CUDA-capable GPU (sm_75+), internet access.
# ============================================================================
set -euo pipefail

ENV_NAME="${1:-SVG}"
DIFFVG_DIR="${2:-$PWD/diffvg}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/5] Creating conda env '$ENV_NAME' (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> [2/5] Installing PyTorch 2.0.1 (CUDA 11.8) — must NOT come from default PyPI"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

echo "==> [3/5] Installing CUDA 11.8 toolchain + system libs for the diffvg build"
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit \
    -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake ninja cairo pango

echo "==> [4/5] Building diffvg (pydiffvg) from source for this env (GPU-enabled)"
if [ ! -d "$DIFFVG_DIR/.git" ]; then
    echo "    cloning diffvg into $DIFFVG_DIR"
    git clone https://github.com/BachiLi/diffvg.git "$DIFFVG_DIR"
fi
cd "$DIFFVG_DIR"
git submodule update --init --recursive
rm -rf build
CUDA_HOME="$CONDA_PREFIX" \
CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
TORCH_CUDA_ARCH_LIST="7.5" \
DIFFVG_CUDA=1 \
python setup.py install
cd "$REPO_DIR"

echo "==> [5/5] Installing pinned Python requirements"
pip install -r "$REPO_DIR/requirements.txt"

echo "==> Verifying the environment"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import torch, pydiffvg
print(f"  torch {torch.__version__} | cuda avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  diffvg GPU: {pydiffvg.get_use_gpu()}")
import pytorch_lightning, transformers, vector_quantize_pytorch, x_transformers, kornia, cairosvg  # noqa
print("  project deps import OK")
PY

echo ""
echo "DONE. Activate with:  conda activate $ENV_NAME"
echo "Always export before running:  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib"
