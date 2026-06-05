#!/bin/bash
# DK-2500 ECG-Agent 一键安装脚本
# 运行环境: Ubuntu 24.04, Intel Core Ultra 5 225U, 8GB RAM, 无独立 GPU
# 运行方式: bash setup_dk2500.sh
# 日志输出: setup_dk2500.log

set -euo pipefail
LOG="setup_dk2500.log"
exec > >(tee -a "$LOG") 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"   # ~/workspace/ECG-Agent
FAIRSEQ_DIR="$WORKSPACE/fairseq-signals"
CHECKPOINT="$WORKSPACE/checkpoints/checkpoint_best.pt"
ADAPTER_DIR="$WORKSPACE/ecg-dialogue-finetune/Llama-3.2-3B-Instruct"

echo "============================================"
echo " DK-2500 ECG-Agent 安装脚本"
echo " 工作目录: $WORKSPACE"
echo " 日志: $LOG"
echo "============================================"

# ── 0. 检查 conda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[错误] 未找到 conda。请先安装 Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "[OK] conda: $(conda --version)"

# ── 1. 创建 conda 环境 ────────────────────────────────────────────────────────
ENV_NAME="ecg-bedside"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[跳过] conda 环境 '${ENV_NAME}' 已存在"
else
    echo "[1/7] 创建 conda 环境 ${ENV_NAME} (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# 激活环境（在子 shell 中持续有效）
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "[OK] Python: $(python --version)"

# ── 2. 安装 PyTorch (CPU-only) ────────────────────────────────────────────────
echo "[2/7] 安装 PyTorch (CPU)..."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --quiet

# ── 3. 安装 Python 依赖 ───────────────────────────────────────────────────────
echo "[3/7] 安装核心依赖..."
pip install --quiet \
    transformers>=4.46 \
    peft>=0.13 \
    accelerate \
    bitsandbytes \
    datasets \
    neurokit2 \
    scipy \
    numpy \
    pandas \
    scikit-image \
    langchain-core \
    bleak \
    tqdm

# llama-cpp-python（CPU 版，用于 GGUF 推理）
echo "[3/7] 安装 llama-cpp-python (CPU)..."
CMAKE_ARGS="-DGGML_BLAS=OFF" pip install llama-cpp-python --quiet || \
    echo "[警告] llama-cpp-python 安装失败，GGUF 后端不可用（transformers 后端仍可用）"

# ── 4. 安装 fairseq-signals ───────────────────────────────────────────────────
echo "[4/7] 安装 fairseq-signals..."
if [ ! -d "$FAIRSEQ_DIR" ]; then
    echo "[错误] 未找到 fairseq-signals 目录: $FAIRSEQ_DIR"
    echo "  请确认 fairseq-signals 已克隆到 $FAIRSEQ_DIR"
    exit 1
fi
pip install -e "$FAIRSEQ_DIR" --quiet
echo "[OK] fairseq-signals 安装完成"

# ── 5. 验证关键文件 ───────────────────────────────────────────────────────────
echo "[5/7] 验证关键文件..."

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -sh "$1" | cut -f1)
        echo "  [OK] $1 ($SIZE)"
    else
        echo "  [缺失] $1"
        echo "  请按部署指南传输该文件"
        MISSING=1
    fi
}

MISSING=0
check_file "$CHECKPOINT"
check_file "$ADAPTER_DIR/adapter_config.json"
check_file "$ADAPTER_DIR/adapter_model.safetensors"

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "[警告] 部分文件缺失，请传输后重新运行此脚本验证"
fi

# ── 6. 下载 base model（可选，需要 HuggingFace 访问）────────────────────────
echo "[6/7] 检查 base model 缓存..."
BASE_MODEL="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"

python - <<EOF
from huggingface_hub import try_to_load_from_cache
from pathlib import Path
cached = try_to_load_from_cache("unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit", "config.json")
if cached is None:
    print("  base model 未缓存，将在首次运行时自动下载（约 2GB）")
else:
    print(f"  [OK] base model 已缓存")
EOF

# ── 7. 冒烟测试 ──────────────────────────────────────────────────────────────
echo "[7/7] 冒烟测试..."

python - <<'PYEOF'
import sys, os

# fairseq-signals
try:
    import fairseq_signals
    print("  [OK] fairseq_signals")
except ImportError as e:
    print(f"  [FAIL] fairseq_signals: {e}")
    sys.exit(1)

# neurokit2
try:
    import neurokit2 as nk
    print("  [OK] neurokit2")
except ImportError as e:
    print(f"  [FAIL] neurokit2: {e}")
    sys.exit(1)

# scipy
try:
    import scipy.io, scipy.signal
    print("  [OK] scipy")
except ImportError as e:
    print(f"  [FAIL] scipy: {e}")
    sys.exit(1)

# transformers + peft
try:
    from transformers import AutoTokenizer
    from peft import PeftModel
    print("  [OK] transformers + peft")
except ImportError as e:
    print(f"  [FAIL] transformers/peft: {e}")
    sys.exit(1)

# bleak
try:
    import bleak
    print("  [OK] bleak (BLE)")
except ImportError as e:
    print(f"  [WARN] bleak: {e} (BLE 采集不可用)")

# llama-cpp-python
try:
    import llama_cpp
    print("  [OK] llama-cpp-python (GGUF 后端)")
except ImportError:
    print("  [INFO] llama-cpp-python 未安装（GGUF 后端不可用，transformers 后端可用）")

print("\n冒烟测试通过！")
PYEOF

echo ""
echo "============================================"
echo " 安装完成！"
echo ""
echo " 激活环境:"
echo "   conda activate $ENV_NAME"
echo ""
echo " 分析 Lepod CSV:"
echo "   cd $SCRIPT_DIR"
echo "   python bedside_agent.py --ecg ~/workspace/ecg_YYYYMMDD.csv"
echo ""
echo " 实时采集并分析:"
echo "   python bedside_agent.py --live 30"
echo ""
echo " 详细说明见: DK-2500_ECG-Agent_部署指南.md"
echo "============================================"
