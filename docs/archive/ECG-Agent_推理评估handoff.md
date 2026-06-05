# ECG-Agent 推理评估 Handoff

**日期**：2026-05-26（更新）  
**前置阶段**：模型训练（AutoDL RTX PRO 6000，已完成）  
**本文档目标**：指导在云 GPU 环境完成端到端推理和评估

---

## 1. 已完成的训练成果

| 产物 | 本地路径 | 说明 |
|------|----------|------|
| LLM LoRA adapter | `ecg-dialogue-finetune/Llama-3.2-3B-Instruct/` | Llama 3.2 3B + Unsloth LoRA 4-bit，在 ECG-MTD 对话数据集上微调 |
| 分类工具 checkpoint | `checkpoints/checkpoint_best.pt` | ecg-fm → PTB-XL 71类诊断，fairseq-signals ECGTransformerClassifier |

> 路径均相对于 `~/workspace/ECG-Agent/`（ECG-Agent 项目根目录的**上一级**）

---

## 2. 已完成的推理准备（本地 RTX 4070 Laptop 验证）

| 步骤 | 状态 | 输出 |
|------|------|------|
| Step 1: 测量工具输出 | ✅ 完成 | `results/ecg_analysis_measurements.csv`（21,799 行，749K） |
| Step 2: 分类工具输出 | ✅ 完成 | `results/ecg_analysis_classification.csv`（21,799 行，958K） |
| Step 3: LLM 推理（3 样本测试） | ✅ 验证通过 | 3/3 样本 action 预测全部正确 |
| Step 3: LLM 推理（完整 2,174 样本） | ⏳ 待云执行 | 本地 RTX 4070 ~36s/样本，全集约 21 小时 |

---

## 3. 代码修复记录

### 3.1 分类工具 — checkpoint 路径硬编码问题

**文件**：`src/medrax/tools/classification.py`

**问题**：`checkpoint_best.pt` 内嵌 `model_path=/root/ECG-Agent/weights/mimic_iv_ecg_physionet_pretrained.pt`（AutoDL 训练时的路径）。加载时 fairseq-signals 会尝试读取该路径的预训练权重，在本地报 `PermissionError`。

**修复**：传入 `model_overrides={"no_pretrained_weights": True}`，跳过预训练权重加载（finetuned checkpoint 已包含完整模型权重，通过 `load_state_dict` 加载）。

```python
# 修复后
model, cfg, task = checkpoint_utils.load_model_and_task(
    model_path,
    model_overrides={"no_pretrained_weights": True},
)
```

### 3.2 推理脚本 — ECG 文件名不匹配

**文件**：`src/inference_ecg_dialogue.py`

**问题**：ECG-MTD 数据集引用文件名为 `HR00025.mat`（零填充 5 位），但 .mat 文件实际命名为 `HR25.mat`（无零填充）。导致 `get_precomputed_tool_output` 查找 CSV 时匹配失败。

**修复**：添加 `normalize_ecg_filename()` 函数，在查找前将 `HR00025.mat` → `HR25.mat`。

```python
def normalize_ecg_filename(name):
    """Normalize HR00025.mat -> HR25.mat (strip leading zeros from numeric part)."""
    import re
    m = re.match(r'(HR)0*(\d+)(\.mat)', name)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else name
```

### 3.3 推理脚本 — 死代码

**文件**：`src/inference_ecg_dialogue.py`

移除 `get_precomputed_tool_output` 中 `return json.dumps(measurements)` 后不可达的 `return "[]"`。

---

## 4. 项目文件结构（关键路径）

```
~/workspace/ECG-Agent/
├── ecg-dialogue-finetune/
│   └── Llama-3.2-3B-Instruct/      # LLM adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── checkpoint-1500/         # 中间 checkpoint（可忽略）
├── checkpoints/
│   └── checkpoint_best.pt           # 分类工具最优权重
└── ECG-Agent/                       # 代码仓库
    ├── src/
    │   ├── inference_ecg_dialogue.py  # 已修复文件名匹配+死代码
    │   ├── extract_tool_outputs.py
    │   ├── eval/                      # 评估脚本
    │   │   ├── next_action_prediction.py  # 本地可跑（无需 API）
    │   │   ├── faithfulness.py            # 需要 GOOGLE_API_KEY
    │   │   ├── accuracy_completeness.py   # 需要 GOOGLE_API_KEY
    │   │   └── naturalness_cefr.py        # 需要 GOOGLE_API_KEY
    │   └── medrax/tools/classification.py # 已修复 checkpoint 加载
    ├── results/
    │   ├── ecg_analysis_measurements.csv     # ✅ 21,799 行
    │   └── ecg_analysis_classification.csv   # ✅ 21,799 行
    ├── data/ptbxl_mat/              # 21,799 个 .mat 文件（9.9 GB）
    └── fairseq-signals/
```

---

## 5. 云服务器推理步骤

### 前置条件

两个 CSV 已生成（如果传输到云服务器，可跳过 Step 1/2）：
- `results/ecg_analysis_measurements.csv`
- `results/ecg_analysis_classification.csv`

### Step 3: LLM 完整推理

```bash
cd ~/workspace/ECG-Agent/ECG-Agent
python src/inference_ecg_dialogue.py \
  --base-model-path unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit \
  --adapter-path ../ecg-dialogue-finetune/Llama-3.2-3B-Instruct \
  --inference-mode without_gt
```

| 参数 | 说明 |
|------|------|
| `--base-model-path` | 使用 `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`（与 adapter 训练时的 base model 一致） |
| `--adapter-path` | LoRA adapter 路径 |
| `--inference-mode` | `without_gt`（用模型生成的历史）或 `with_gt`（用标注历史） |
| `--max-samples N` | 限制样本数（调试用） |
| `--filter-action X` | 只跑含特定 action 的样本（如 `response_fail`） |

**预计耗时**：
- RTX 4070 Laptop (8 GB)：~36s/样本 → 2,174 样本约 21 小时
- RTX PRO 6000 (48 GB)：预计 5-8 小时

**输出格式**：`inference_*.jsonl`，每行一个样本，包含 `turns`（逐轮对齐）、`generated_dialogue`、`ground_truth_dialogue`。结果逐条追加写入，中断后可续跑。

### Step 4: 评估

```bash
# 1. Next Action Prediction（本地，无需 API）
python src/eval/next_action_prediction.py \
  --llama_3b_file inference_Llama-3.2-3B-Instruct_*.jsonl

# 2. Faithfulness（需要 GOOGLE_API_KEY）
export GOOGLE_API_KEY=your_key
python src/eval/faithfulness.py \
  --llama_3b_file inference_Llama-3.2-3B-Instruct_*.jsonl \
  --output_filename faithfulness_scores.json

# 3. Accuracy & Completeness（需要 GOOGLE_API_KEY）
python src/eval/accuracy_completeness.py \
  --llama_3b_file inference_Llama-3.2-3B-Instruct_*.jsonl

# 4. Naturalness & CEFR（需要 GOOGLE_API_KEY）
python src/eval/naturalness_cefr.py \
  --llama_3b_file inference_Llama-3.2-3B-Instruct_*.jsonl
```

---

## 6. 环境依赖

```bash
pip install transformers peft trl unsloth datasets bitsandbytes
pip install neurokit2 wfdb scipy pandas
pip install langchain_core

# fairseq-signals（分类工具必须）
cd ~/workspace/ECG-Agent/ECG-Agent/fairseq-signals
pip install -e .
```

> 本地实测：fairseq-signals 和 unsloth 在同一 conda 环境无冲突。

---

## 7. 已知问题和注意事项

| 问题 | 处置方式 |
|------|----------|
| base model 需从 HF 下载 | `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` 约 2 GB |
| 分类工具 71 类标签 | 见 `src/medrax/tools/classification.py` CLASS_LABELS 列表 |
| 两个 CSV 路径硬编码 | `inference_ecg_dialogue.py` 第 18-19 行，需在同一目录执行 |
| checkpoint 内嵌预训练路径 | 已通过 `no_pretrained_weights=True` 修复，无需额外文件 |
| 文件名零填充不匹配 | 已通过 `normalize_ecg_filename()` 修复 |
| 评估脚本（3/4）需 Gemini API | 设置 `GOOGLE_API_KEY` 环境变量 |

---

## 8. 需要传到云的文件

```
必须传输：
├── ECG-Agent/src/                              # 含所有修复
├── ECG-Agent/results/ecg_analysis_*.csv        # 两个预计算 CSV（共 1.7 MB）
├── ecg-dialogue-finetune/Llama-3.2-3B-Instruct/ # LoRA adapter（~100 MB）
└── checkpoints/checkpoint_best.pt               # 分类 checkpoint（1.1 GB）

可选传输（如需重跑 Step 1/2）：
└── ECG-Agent/data/ptbxl_mat/                    # 21,799 .mat 文件（9.9 GB）
```
