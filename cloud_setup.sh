#!/bin/bash
# ECG-Agent Cloud Server Setup Script
# For NVIDIA RTX PRO 6000 / A6000 / similar high-VRAM GPUs
set -e

echo "=== ECG-Agent Cloud Environment Setup ==="

# 1. Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n ecg-agent python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ecg-agent

# 2. Install PyTorch (CUDA 12.x)
echo "[2/4] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install base requirements
# Note: unsloth and bitsandbytes may not be on all mirrors, so use official PyPI
echo "[3/6] Installing Python dependencies..."
pip install transformers datasets peft trl xformers wandb accelerate
pip install neurokit2 wfdb ecg_plot scipy scikit-learn pandas
# These packages require official PyPI (not mirrors)
pip install unsloth bitsandbytes -i https://pypi.org/simple/
# For China users: if official PyPI is slow, try:
# pip install unsloth bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
# If tsinghua also doesn't have it, fallback to official

# 4. Try flash-attn (may fail, Unsloth falls back to xformers)
echo "[4/4] Installing flash-attn (optional)..."
pip install ninja 2>/dev/null || true
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn failed, will use xformers instead"

# 5. Create ptbxl_files symlink (needed by preprocess scripts)
echo "[5/6] Setting up paths..."
ln -sf src/ptbxl_files ptbxl_files

# 6. Install core dependencies (langchain, etc.)
echo "[6/6] Installing remaining dependencies..."
pip install langchain_core wandb -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 2>/dev/null || pip install langchain_core wandb

# 7. Login to HuggingFace
echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. hf auth login  (or: export HF_TOKEN='your_token')"
echo "  2. Accept Meta license at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
echo "  3. Install fairseq-signals for classification tool:"
echo "     cd fairseq-signals && pip install --editable ."
echo "  4. Run LLM finetuning:"
echo "     python src/finetune_ecg_dialogue_unsloth.py \\"
echo "       --model unsloth/Llama-3.2-3B-Instruct \\"
echo "       --output-dir ./ecg-dialogue-finetune/Llama-3.2-3B-Instruct \\"
echo "       --batch-size 8 --grad-accum 4"
