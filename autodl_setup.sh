#!/bin/bash
# ECG-Agent AutoDL Setup — RTX PRO 6000 (48 GB)
# 用法：在项目根目录下执行 bash autodl_setup.sh
set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=== [0/5] 检查 GPU ==="
nvidia-smi | grep -E "Driver|CUDA|RTX|PRO" || true
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"

# ---------- 1. 核心训练依赖 ----------
echo "=== [1/5] 安装核心依赖 ==="
pip install -q \
  transformers datasets peft "trl==0.12.0" accelerate \
  -i "$PIP_MIRROR"

pip install -q \
  neurokit2 wfdb scipy scikit-learn pandas \
  -i "$PIP_MIRROR"

pip install -q langchain_core wandb -i "$PIP_MIRROR"

# ---------- 2. unsloth + bitsandbytes ----------
echo "=== [2/5] 安装 unsloth / bitsandbytes ==="
pip install -q unsloth bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- 3. flash-attn（编译失败不阻断，Unsloth 自动回退 xformers） ----------
echo "=== [3/5] 尝试安装 flash-attn ==="
pip install -q ninja -i "$PIP_MIRROR"
pip install flash-attn --no-build-isolation 2>/dev/null \
  && echo "flash-attn 安装成功" \
  || echo "flash-attn 编译失败，将使用 xformers（不影响训练）"

# ---------- 4. fairseq-signals（从本地副本安装） ----------
echo "=== [4/5] 安装 fairseq-signals ==="
pip install -q setuptools -i "$PIP_MIRROR"
cd "$PROJ_DIR/fairseq-signals"
pip install -q --editable . -i "$PIP_MIRROR"
cd "$PROJ_DIR"

# ---------- 5. 路径 & 环境变量 ----------
echo "=== [5/5] 配置路径 ==="

# ptbxl_files 软链（preprocess 脚本要求根目录可见）
ln -sfn "$PROJ_DIR/src/ptbxl_files" "$PROJ_DIR/ptbxl_files"

# HuggingFace 镜像（国内加速）
RCFILE="$HOME/.bashrc"
grep -q "HF_ENDPOINT" "$RCFILE" 2>/dev/null \
  || echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> "$RCFILE"
export HF_ENDPOINT="https://hf-mirror.com"

echo ""
echo "=== Setup 完成 ==="
echo ""
echo "接下来："
echo "  1. 登录 HuggingFace（下载 Llama 需要授权）："
echo "     huggingface-cli login"
echo "     # 或：export HF_TOKEN='your_token'"
echo ""
echo "  2. 如果 .mat 文件尚未上传，在本地执行 rsync 上传数据："
echo "     rsync -avz --progress -e 'ssh -p <PORT>' \\"
echo "       data/ptbxl_mat/ root@<HOST>:$(basename "$PROJ_DIR")/data/ptbxl_mat/"
echo ""
echo "  3. 启动训练："
echo "     bash autodl_train.sh"
