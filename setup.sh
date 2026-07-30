#!/bin/bash
# =============================================================================
# Mixtral MoRE - Complete Setup Script
# =============================================================================
# Prerequisites:
#   - NVIDIA Hopper GPU (H100/H200) with CUDA 12.9+ drivers
#   - Python 3.12+
# Usage: bash setup.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

echo "=============================================="
echo "  Mixtral MoRE - Setup Script"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 1: System dependencies
# -----------------------------------------------------------------------------
echo -e "\n[1/7] Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y unzip curl git
elif command -v yum &> /dev/null; then
    yum install -y unzip curl git
fi
echo "Done."

# -----------------------------------------------------------------------------
# Step 2: Verify prerequisites (GPU, Python)
# -----------------------------------------------------------------------------
echo -e "\n[2/7] Checking prerequisites..."

# GPU / CUDA
#
# This must be a hard stop, not a warning. Step 4 installs torch from the
# CUDA 12.9 wheel index, which publishes no macOS builds and no CPU builds,
# and step 5 compiles SonicMoE's Hopper-only kernels. Without an NVIDIA GPU
# both fail, and under `set -e` the reader gets a bare pip resolver error
# instead of being told their machine cannot run this path.
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found."
    echo ""
    echo "setup.sh is the TRAINING path. It needs an NVIDIA Hopper GPU"
    echo "(H100/H200) with CUDA 12.9+ drivers, because it installs a CUDA-only"
    echo "build of PyTorch and compiles SonicMoE's Hopper kernels."
    echo ""
    echo "To inspect the models instead -- builds and runs on any machine,"
    echo "no GPU required:"
    echo "    pip install -r requirements.txt"
    echo "    python apply_patch.py --model deepseek_v3"
    echo ""
    echo "Then see the Quickstart in README.md."
    exit 1
fi

# Python 3.12+ (check both `python` and `python3`)
PYTHON_BIN=""
for candidate in python python3; do
    if command -v "$candidate" &> /dev/null; then
        MINOR=$("$candidate" -c 'import sys; print(sys.version_info.minor)')
        MAJOR=$("$candidate" -c 'import sys; print(sys.version_info.major)')
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 12 ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.12+ required but not found."
    echo "  python3 -> $(python3 --version 2>/dev/null || echo 'not found')"
    echo "  python  -> $(python --version 2>/dev/null || echo 'not found')"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python $PYTHON_VERSION OK (using '$PYTHON_BIN')"

# -----------------------------------------------------------------------------
# Step 3: Create virtual environment
# -----------------------------------------------------------------------------
echo -e "\n[3/7] Creating virtual environment at ${VENV_DIR}..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists, activating..."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python --version

# -----------------------------------------------------------------------------
# Step 4: Install Python dependencies
# -----------------------------------------------------------------------------
echo -e "\n[4/7] Installing Python dependencies..."
pip install --upgrade pip

# PyTorch with CUDA 12.9, pinned.
#
# TORCH_VERSION matches the reference container
# (runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204). Do not leave this
# unpinned: an unpinned install pulls the newest cu129 build, which
# tightened Tensor.__dlpack__ to reject tensors that require grad. The
# pinned SonicMoE revision calls it on exactly such tensors, so training
# dies with:
#     BufferError: Can't export tensors that require gradient
#
# torchvision and torchaudio are deliberately NOT installed. Nothing here
# uses them, and a torchvision built for a different torch makes
# transformers fail to import MixtralForCausalLM with
#     RuntimeError: operator torchvision::nms does not exist
TORCH_VERSION="2.9.0"
pip3 install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu129/

# All other dependencies
pip install -r requirements.txt

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo "Done."

# -----------------------------------------------------------------------------
# Step 5: Install SonicMoE (from source)
# -----------------------------------------------------------------------------
# SonicMoE requires Hopper GPUs (H100/H200) and builds Triton/CuTe kernels.
# Ref: https://github.com/Dao-AILab/sonic-moe
echo -e "\n[5/7] Installing SonicMoE..."

# Pinned, not tracking main. Our two patches below were authored against this
# revision; SonicMoE has since restructured both files they touch, so an
# unpinned clone silently loses the patches. Verified: both apply cleanly at
# this commit and at no later one.
SONICMOE_COMMIT="66f6ced475259d9aa9ebe854652784b2a45409bd"

if [ ! -d "sonic-moe" ]; then
    git clone https://github.com/Dao-AILab/sonic-moe.git
fi
cd sonic-moe
git fetch -q origin
git checkout -q "$SONICMOE_COMMIT" || {
    echo "ERROR: could not check out SonicMoE $SONICMOE_COMMIT"
    exit 1
}
echo "SonicMoE pinned at $(git rev-parse --short HEAD)"
pip install -r requirements.txt
pip install -e .
cd "$SCRIPT_DIR"
echo "SonicMoE installed."

# -----------------------------------------------------------------------------
# Apply our SonicMoE patches
# -----------------------------------------------------------------------------
# apply_sonicmoe_patch <name> <what breaks without it>
#
# Distinguishes three outcomes, because conflating them is how these patches
# silently stopped being applied: `git apply --check` failing was reported as
# "already applied", so a fresh clone looked fine while Mixtral training
# crashed later with AttributeError: 'MoE' object has no attribute
# '_routing_acc_probs'.
apply_sonicmoe_patch() {
    local name="$1" breaks="$2"
    local patch="$SCRIPT_DIR/patches/sonicmoe_${name}.patch"
    echo "Applying SonicMoE ${name} patch..."
    cd sonic-moe
    if git apply --reverse --check "$patch" 2>/dev/null; then
        echo "  already applied, skipping"
    elif git apply "$patch" 2>/dev/null; then
        echo "  applied"
    else
        echo ""
        echo "  *** PATCH FAILED: sonicmoe_${name}.patch ***"
        echo "  Cause: the SonicMoE checkout does not match what this patch"
        echo "         expects (pinned commit ${SONICMOE_COMMIT:0:7})."
        echo "  Impact: ${breaks}"
        echo "  Setup continues -- the DeepSeek-v3 path does not use SonicMoE"
        echo "  and is unaffected. See patches/README.md."
        echo ""
        SONICMOE_PATCH_FAILED=1
    fi
    cd "$SCRIPT_DIR"
}

SONICMOE_PATCH_FAILED=0
apply_sonicmoe_patch mixed_precision \
    "bf16/autocast training with the SonicMoE kernel will fail on dtype mismatch."
apply_sonicmoe_patch grouped_loss \
    "Mixtral training with use_grouped_aux_loss=true will crash: 'MoE' object has no attribute '_routing_acc_probs'."

# -----------------------------------------------------------------------------
# Step 6: Patch HuggingFace Transformers
# -----------------------------------------------------------------------------
echo -e "\n[6/7] Patching HuggingFace Transformers..."
# All three families live in separate directories and never collide, so we
# patch them all. apply_patch.py backs up the originals and asserts the
# transformers version first. No `|| true`: a version mismatch must stop
# setup, not scroll past. Undo any time with:
#     python apply_patch.py --restore
for model in deepseek_v3 qwen3_moe mixtral; do
    python apply_patch.py --model "$model"
done
python apply_patch.py --check

# -----------------------------------------------------------------------------
# Step 7: Convert tokenizer to HuggingFace format
# -----------------------------------------------------------------------------
echo -e "\n[7/7] Preparing tokenizer..."
if [ ! -f "c4_tokenizer/tokenizer.json" ]; then
    # The SentencePiece model comes from the MoEUT codebase (MIT) and ships
    # with this repo. Convert it to HuggingFace format so
    # from_pretrained("c4_tokenizer/") works.
    if [ ! -f "c4_tokenizer/tokenizer.model" ]; then
        echo "ERROR: c4_tokenizer/tokenizer.model not found."
        echo "It ships with this repo; your checkout may be incomplete."
        exit 1
    fi
    python -c "
from transformers import LlamaTokenizerFast
tokenizer = LlamaTokenizerFast(vocab_file='c4_tokenizer/tokenizer.model', clean_up_tokenization_spaces=False)
tokenizer.pad_token = '<unk>'
tokenizer.save_pretrained('c4_tokenizer/')
print(f'Tokenizer saved (vocab_size={len(tokenizer)})')
"
else
    echo "c4_tokenizer/ already in HuggingFace format, skipping"
fi

# -----------------------------------------------------------------------------
# Model configs
# -----------------------------------------------------------------------------
echo -e "\nExample configs are in configs/example/"
echo "To regenerate or customize, run:"
echo "  python scripts/gen_conf_deepseek_v3.py --preset more-114m --out <dir>"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo -e "\n=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
if [ "$SONICMOE_PATCH_FAILED" = "1" ]; then
    echo ""
    echo "WARNING: at least one SonicMoE patch did not apply (see above)."
    echo "         Mixtral + SonicMoE training will not work. The DeepSeek-v3"
    echo "         path is unaffected."
    echo ""
fi

echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source ${VENV_DIR}/bin/activate"
echo ""
echo "  2. Verify the install with the Quickstart in README.md, which builds"
echo "     a model from configs/example/ and runs a forward pass."
echo ""
echo "  3. Train. This repo ships no dataset -- see README, 'Data', for the"
echo "     format your own tokenized data must be in:"
echo "     python scripts/train_c4_sonic_single.py \\"
echo "       --model_name_or_path configs/example/more-deepseek_v3-114m \\"
echo "       --train_data /path/to/train --val_data /path/to/val"
echo ""
