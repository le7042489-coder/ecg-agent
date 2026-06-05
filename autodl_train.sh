#!/bin/bash
# ECG-Agent 训练 Pipeline — RTX PRO 6000 (48 GB)
# 用法：bash autodl_train.sh [--skip-llm] [--skip-cls]
# 依赖：autodl_setup.sh 已执行完毕，HF_TOKEN 已设置
set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
export PROJ_DIR
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # 默认禁用 wandb，设 WANDB_API_KEY 后自动启用
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

SKIP_LLM=false
SKIP_CLS=false
NUM_EPOCHS=1
for arg in "$@"; do
  case "$arg" in
    --skip-llm) SKIP_LLM=true ;;
    --skip-cls) SKIP_CLS=true ;;
    --epochs=*) NUM_EPOCHS="${arg#*=}" ;;
  esac
done

# ------------------------------------------------------------------ #
# 路径配置
# ------------------------------------------------------------------ #
ECG_FM_CKPT="$PROJ_DIR/weights/mimic_iv_ecg_physionet_pretrained.pt"
MANIFEST_DIR="$PROJ_DIR/data/ptbxl_mat/manifest"
LLM_OUT="$PROJ_DIR/ecg-dialogue-finetune/Llama-3.2-3B-Instruct"
CLS_CKPT_DIR="$PROJ_DIR/checkpoints"

echo "=========================================="
echo " ECG-Agent Training Pipeline"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo " VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "=========================================="

# ------------------------------------------------------------------ #
# Step 0: 下载 ecg-fm 预训练权重（已验证兼容 fairseq-signals）
# ------------------------------------------------------------------ #
if [ ! -f "$ECG_FM_CKPT" ]; then
  echo ""
  echo "[0/3] 下载 ecg-fm checkpoint..."
  mkdir -p "$PROJ_DIR/weights"
  python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download
dest = os.path.join(os.environ["PROJ_DIR"], "weights")
try:
    path = hf_hub_download(
        repo_id="wanglab/ecg-fm",
        filename="mimic_iv_ecg_physionet_pretrained.pt",
        local_dir=dest,
    )
    print(f"  已下载至 {path}")
except Exception as e:
    print(f"  下载失败: {e}")
    print("  请手动下载后放到 weights/mimic_iv_ecg_physionet_pretrained.pt")
    sys.exit(1)
PYEOF
fi

# ------------------------------------------------------------------ #
# Step 1: LLM 微调（Llama 3.2 3B + Unsloth LoRA 4-bit）
# ------------------------------------------------------------------ #
if [ "$SKIP_LLM" = false ]; then
  echo ""
  echo "[1/3] LLM 微调 —— 预计 2-4 小时"
  echo "  model : unsloth/Llama-3.2-3B-Instruct"
  echo "  output: $LLM_OUT"
  echo "  batch : 8 x grad_accum 4 = effective 32  epochs: $NUM_EPOCHS"
  echo ""

  SECONDS=0
  python src/finetune_ecg_dialogue_unsloth.py \
    --model "unsloth/Llama-3.2-3B-Instruct" \
    --output-dir "$LLM_OUT" \
    --batch-size 8 \
    --grad-accum 4 \
    --num-epochs "$NUM_EPOCHS"
  echo "  LLM 微调完成，耗时 $((SECONDS/60)) 分钟"
else
  echo "[1/3] 跳过 LLM 微调（--skip-llm）"
fi

# ------------------------------------------------------------------ #
# Step 2: 生成 PTB-XL manifest（如已存在则跳过）
# ------------------------------------------------------------------ #
if [ ! -f "$MANIFEST_DIR/train.tsv" ]; then
  echo ""
  echo "[2/3] 生成 PTB-XL manifest..."
  python src/preprocess/manifest_ptbxl_10s.py \
    --root "$PROJ_DIR" \
    --data-dir data/ptbxl_mat \
    --dest data/ptbxl_mat/manifest \
    --valid-percent 0.05 --test-percent 0.1 --seed 42
  echo "  manifest 生成完毕"
else
  echo "[2/3] manifest 已存在，跳过生成"
fi

# ------------------------------------------------------------------ #
# Step 2.5: 写入 71 类 SCP-ECG 标签到 .mat 文件（首次运行 ~5 分钟）
# ------------------------------------------------------------------ #
if [ "$SKIP_CLS" = false ]; then
  FIRST_MAT=$(ls "$PROJ_DIR/data/ptbxl_mat/"*.mat 2>/dev/null | head -1)
  if [ -n "$FIRST_MAT" ] && python -c "import scipy.io; d=scipy.io.loadmat('$FIRST_MAT'); exit(0 if 'label' in d else 1)" 2>/dev/null; then
    echo "[2.5/3] .mat 标签已存在，跳过"
  else
    echo ""
    echo "[2.5/3] 向 .mat 文件写入 71 类诊断标签（~5 分钟）..."
    python "$PROJ_DIR/src/preprocess/add_labels_to_mat.py"
    echo "  标签写入完成"
  fi
fi

# ------------------------------------------------------------------ #
# Step 3: 分类工具微调（ecg-fm → PTB-XL 71类诊断）
# ------------------------------------------------------------------ #
if [ "$SKIP_CLS" = false ]; then
  echo ""
  echo "[3/3] 分类工具微调 —— 预计 1-3 小时"
  echo "  起点  : $ECG_FM_CKPT"
  echo "  数据  : $MANIFEST_DIR"
  echo "  输出  : $CLS_CKPT_DIR"
  echo ""

  SECONDS=0
  fairseq-hydra-train \
    model.model_path="$ECG_FM_CKPT" \
    +task.data="$MANIFEST_DIR" \
    criterion.weights_file=null \
    checkpoint.save_dir="$CLS_CKPT_DIR" \
    --config-dir "$PROJ_DIR/fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
    --config-name diagnosis
  echo "  分类工具微调完成，耗时 $((SECONDS/60)) 分钟"
  echo "  最优模型: $CLS_CKPT_DIR/checkpoint_best.pt"
else
  echo "[3/3] 跳过分类工具微调（--skip-cls）"
fi

# ------------------------------------------------------------------ #
# 完成
# ------------------------------------------------------------------ #
echo ""
echo "=========================================="
echo " 训练完成！接下来运行推理："
echo ""
echo "  python src/inference_ecg_dialogue.py"
echo ""
echo "  或批量提取工具输出："
echo "  python src/extract_tool_outputs.py --tool measurements \\"
echo "    --ecg_dir data/ptbxl_mat --output_dir results/measurements"
echo "  python src/extract_tool_outputs.py --tool classification \\"
echo "    --ecg_dir data/ptbxl_mat \\"
echo "    --model_path $CLS_CKPT_DIR/checkpoint_best.pt \\"
echo "    --output_dir results/classification"
echo "=========================================="
