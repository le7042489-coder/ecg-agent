# ECG-Agent 本地部署指南

> **环境假设：** 单张消费级 GPU（RTX 3090/4090，24 GB VRAM）、Ubuntu / WSL2、CUDA ≥ 12.1  
> **目标模型：** Llama-3.2-3B-Instruct（未微调基座）  
> **部署范围：** 完整 pipeline = 分类工具 + 测量工具 + 对话 Agent

---

## 0. 整体架构速览

ECG-Agent 不是一个端到端模型，而是一个 **LLM + 3 个外部工具** 的 Agent 系统：

| 组件 | 作用 | 关键依赖 |
|------|------|----------|
| **对话 Agent（LLM）** | 理解用户问题 → 决定调用哪个工具 → 将工具结果转化为自然语言回答 | Llama 3.2 3B + Unsloth + Transformers |
| **分类工具** | 对 ECG 信号做心律失常分类（71 个 PTB-XL 诊断类） | fairseq-signals（W2V+CMSC+RLM 预训练 → PTB-XL 微调） |
| **测量工具** | 提取 PQRST 波段的精确时间间隔 | NeuroKit2（纯 Python，无需 GPU） |
| **解释工具** | 用 SpectralX 提供时频域分类可解释性（仅限单导联） | gustmd0121/Time_is_not_Enough |

> ⚠️ **关键现实：** 微调版 LLM（HuggingFace 上的 `gustmd0121/llama-3.2-3B-ecg-tool-calling-v1`）返回 401，无法下载。使用未微调的 Llama 3.2 3B 意味着 **LLM 不会自动生成论文中定义的 tool-calling 格式**（`action` / `thought` / `tool_output` 结构）。你需要通过以下方式之一弥补：
>
> - **方案 A（推荐）：自己在 ECG-MTD 数据集上微调。** 数据集是公开的，代码仓库提供了 `finetune_ecg_dialogue_unsloth.py`，在单张 3090/4090 上使用 LoRA + 4-bit 量化完全可行（论文实验在 RTX A6000 上完成，VRAM 需求相近）。
> - **方案 B：用 prompt engineering 模拟 tool-calling。** 在 system prompt 中定义工具调用格式，让原版 Llama 3.2 3B 按格式输出。效果会明显低于微调版。

---

## 1. 基础环境准备

### 1.1 创建 conda 环境

```bash
conda create -n ecg-agent python=3.10 -y
conda activate ecg-agent
```

### 1.2 安装 PyTorch（匹配你的 CUDA 版本）

```bash
# 以 CUDA 12.1 为例；根据 nvidia-smi 输出调整
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 克隆 ECG-Agent 仓库

```bash
git clone https://github.com/gustmd0121/ECG-Agent.git
cd ECG-Agent
pip install -r requirements.txt
```

> ⚠️ `requirements.txt` 中指定了 `flash_attn==2.8.3`，安装时需要 CUDA toolkit 和 `ninja`：
> ```bash
> pip install ninja
> pip install flash-attn --no-build-isolation
> ```
> 如果编译失败，可暂时跳过（Unsloth 在 4-bit 模式下可以不依赖 flash-attn）。

### 1.4 登录 Hugging Face（下载 Llama 需要授权）

```bash
pip install huggingface_hub
huggingface-cli login
# 输入你的 HF token
```

你需要先在 https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct 页面接受 Meta 的许可协议。

---

## 2. 下载数据集

### 2.1 PTB-XL 数据集（ECG 原始信号）

分类工具和测量工具都需要 PTB-XL 的原始 ECG 波形。

```bash
# 方式一：通过 PhysioNet 下载
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# 方式二：使用 wfdb
pip install wfdb
python -c "import wfdb; wfdb.dl_database('ptb-xl', './data/ptb-xl')"
```

### 2.2 ECG-MTD 对话数据集（微调 Agent 用）

如果选择方案 A（自行微调），需要下载 ECG-MTD 数据集：

```bash
# 12 导联版本（临床标准）
# 直接用 datasets 库加载
python -c "
from datasets import load_dataset
ds = load_dataset('gustmd0121/12-lead-ecg-mtd-dataset')
print(ds)
"
```

可用的数据集：

| 配置 | 数据集 ID |
|------|-----------|
| 12 导联 | `gustmd0121/12-lead-ecg-mtd-dataset` |
| Lead I | `gustmd0121/single-lead-I-ecg-mtd-dataset` |
| Lead II | `gustmd0121/single-lead-II-ecg-mtd-dataset` |
| 对应的 Ground-Truth 版本 | 在上述 ID 后加 `-gt` |

---

## 3. 部署分类工具（Classification Tool）

这是整个 pipeline 中最复杂的部分——需要在 fairseq-signals 框架下完成"预训练 → 微调"。

### 3.1 安装 fairseq-signals

```bash
cd ~
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals
pip install --editable .
```

> 实践验证无依赖冲突，可直接安装在 `ecg-agent` 主环境中。

### 3.2 预处理 PhysioNet 2021 数据（用于预训练）

按 fairseq-signals 仓库的 [uni-modal tasks 文档](https://github.com/Jwoo5/fairseq-signals#uni-modal-tasks) 操作：

```bash
# 下载 PhysioNet/CinC Challenge 2021 数据
# 然后生成 manifest 文件
python fairseq_signals/data/ecg/preprocess/preprocess_physionet2021.py \
    /path/to/physionet2021/ \
    --dest /path/to/physionet2021_processed/
```

### 3.3 W2V+CMSC+RLM 预训练

按 [W2V+CMSC+RLM 指南](https://github.com/Jwoo5/fairseq-signals/blob/master/examples/w2v_cmsc/README.md) 操作：

```bash
fairseq-hydra-train \
    task.data=/path/to/physionet2021_processed/cmsc \
    --config-dir examples/w2v_cmsc/config/pretraining \
    --config-name w2v_cmsc_rlm
```

> ⏱ **预训练耗时很长**（论文使用多卡训练）。在单张 3090 上可能需要数天到一周。
>
> 💡 **替代方案（已验证）：** [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) 的预训练权重（`mimic_iv_ecg_physionet_pretrained.pt`）经 fairseq-signals 加载成功，可直接跳过预训练步骤，作为 PTB-XL 微调的起点。

### 3.4 预处理 PTB-XL 并微调分类器

```bash
# Step 2a: 预处理 PTB-XL
# 注意：PhysioNet wget 下载后目录名含版本号，如
# ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/
python src/preprocess/preprocess_ptbxl.py \
    /path/to/ptb-xl/records500/ \
    --dest /path/to/ptbxl_processed/

# Step 2b: 生成 manifest
python src/preprocess/manifest_ptbxl_10s.py /path/to/ptbxl_processed/

# Step 2c: 微调分类器
fairseq-hydra-train \
    model.model_path=/path/to/pretrained_model/checkpoints/checkpoint_last.pt \
    +task.data=/path/to/ptbxl_processed/ptbxl_10s_manifest \
    --config-dir examples/w2v_cmsc/config/finetuning/ecg_transformer \
    --config-name diagnosis
```

微调完成后，你会得到 `checkpoint_best.pt`——这就是分类工具的模型。

### 3.5 提取分类工具输出

```bash
python src/extract_tool_outputs.py \
    --tool classification \
    --ecg_dir /path/to/ecg/files \
    --model_path /path/to/checkpoint_best.pt \
    --output_dir ./results/classification
```

---

## 4. 部署测量工具（Measurement Tool）

这是最简单的部分，纯 Python 实现，无需额外模型。

### 4.1 安装依赖

```bash
pip install neurokit2 wfdb
```

### 4.2 提取测量输出

```bash
python src/extract_tool_outputs.py \
    --tool measurements \
    --ecg_dir /path/to/ecg/files \
    --output_dir ./results/measurements
```

这会自动用 NeuroKit2 对 ECG 信号做 PQRST 波群检测，输出各个波段的时间间隔。

---

## 5. 部署解释工具（Explanation Tool）——可选

解释工具基于 SpectralX，**仅支持单导联**（Lead I 或 Lead II），不支持 12 导联。

```bash
git clone https://github.com/gustmd0121/Time_is_not_Enough.git
cd Time_is_not_Enough
pip install -r requirements.txt
# 按该仓库的 README 生成解释输出
```

> 如果你的场景以 12 导联为主，可以先跳过此步。

---

## 6. 部署对话 Agent（LLM）

### 方案 A（推荐）：用 ECG-MTD 数据集自行微调

仓库已提供微调脚本 `finetune_ecg_dialogue_unsloth.py`，使用 Unsloth + LoRA + 4-bit 量化。

#### 6A.1 安装 Unsloth

```bash
pip install unsloth
```

#### 6A.2 修改微调脚本的配置

打开 `finetune_ecg_dialogue_unsloth.py`，检查并修改以下关键参数：

```python
# 基座模型（改为 3B）
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# LoRA 参数（论文设定）
lora_rank = 16
lora_alpha = 16

# 训练参数
learning_rate = 2e-4
weight_decay = 0.01
num_train_epochs = 3
max_seq_length = 4096
per_device_train_batch_size = 2    # 根据 VRAM 调整
gradient_accumulation_steps = 64   # 使有效 batch size = 128
```

#### 6A.3 运行微调

```bash
python finetune_ecg_dialogue_unsloth.py
```

> 在 RTX 3090/4090 上，3B 模型 + LoRA 4-bit 的 VRAM 占用约 8–12 GB，可以顺利运行。
> 预计训练时间：几小时到半天（取决于数据集大小和 epoch 数）。

#### 6A.4 运行推理

```bash
python inference_ecg_dialogue.py
```

---

### 方案 B：直接使用未微调模型 + Prompt Engineering

如果不想微调，可以加载原版 Llama 3.2 3B，通过 system prompt 定义工具调用格式。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 定义工具调用的 system prompt（模拟论文中的 action 格式）
system_prompt = """You are ECG-Agent, a medical assistant that analyzes ECG data.
You have access to three tools:
1. classification_tool: Classifies cardiac arrhythmias from ECG signals.
2. measurement_tool: Extracts PQRST interval measurements from ECG signals.
3. explanation_tool: Provides time-frequency explanations for classification results.

When a user asks about their ECG, respond with:
- [ACTION]: one of {call_classification, call_measurement, call_explanation, response}
- [THOUGHT]: your reasoning for choosing this action
- [CONTENT]: your response to the user (after tool results are available)

Tool outputs will be provided to you in CSV format after you make a tool call.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What does my ECG show? Are there any arrhythmias?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> ⚠️ 方案 B 的局限性：原版 Llama 3.2 3B 没有在 ECG-MTD 上训练，工具调用行为不可靠。仅适合快速验证 pipeline 是否打通，不适合正式使用。

---

## 7. 端到端运行流程

将所有组件串联的完整工作流：

```
用户提问
    ↓
[对话 Agent] 分析意图 → 决定调用哪个工具
    ↓
┌─ call_classification → 运行分类模型 → 返回诊断类别
├─ call_measurement   → 运行 NeuroKit2  → 返回 PQRST 测量值
└─ call_explanation   → 运行 SpectralX  → 返回可解释性分析
    ↓
[对话 Agent] 将工具输出整合为自然语言回答
    ↓
返回用户
```

实际推理时，`inference_ecg_dialogue.py` 会处理这个循环：读取用户输入 → LLM 生成 action → 如果是 tool call 则执行对应工具 → 将工具输出拼接回对话历史 → LLM 生成最终回复。

---

## 8. 常见问题与注意事项

**Q: 单卡 3090 能跑通整个 pipeline 吗？**
对话 Agent 和测量工具没问题。分类工具的预训练步骤（W2V+CMSC+RLM）在单卡上非常慢，建议找预训练好的 checkpoint 跳过此步，只做 PTB-XL 微调。

**Q: 为什么 requirements.txt 中有 `openai` 和 `sglang`？**
`openai` 可能用于调用 GPT 做评估（LLM-as-Judge），`sglang` 用于高效推理。部署时如果只做推理，这两个不是必须的。

**Q: ECG-MTD 的对话数据集格式是什么样的？**
每条数据是一个多轮对话，每个 turn 包含 `action`（动作类型）、`thought`（推理过程）、`content/tool_output`（内容或工具输出）。LLM 通过 SFT 学习按此格式生成输出。

**Q: 我只有 Lead I 的可穿戴设备数据怎么办？**
完全可以。论文支持 12-lead、Lead I、Lead II 三种配置。下载对应的 ECG-MTD 数据集即可（`single-lead-I-ecg-mtd-dataset`）。注意 Lead I 配置下解释工具可用，但分类工具的诊断类别会受限（非所有 71 类都能从单导联检测）。

---

## 9. 评估结果（5% 测试集验证，2026-05-27）

基于本地 RTX 4070 Laptop 对 ECG-MTD 5%（109 样本）的推理和评估结果。详细数据见 `ECG-Agent_评估结果.md`。

### 9.1 工具层（全量 21,799 条 PTB-XL）

| 工具 | 处理数 | 成功率 | 耗时 |
|------|--------|--------|------|
| 测量工具 (NeuroKit2, CPU) | 21,799 | 100%（HR 覆盖 99.6%，QRS 覆盖 92.3%） | ~16 min |
| 分类工具 (ECG-FM, GPU) | 21,799 | 100%（31/71 类出现） | ~6.5 min |

### 9.2 LLM 推理评估（Llama 3.2 3B + LoRA, without_gt）

| 评估维度 | 指标 | 分数 |
|----------|------|------|
| Next Action Prediction | Overall Accuracy | **97.05%** |
| Faithfulness | Alignment Rate | **85.5%** |
| Accuracy (Post-Classification) | Avg (1-5) | **4.43** |
| Accuracy (Post-Measurement) | Avg (1-5) | **4.12** |
| Accuracy (Direct Response) | Avg (1-5) | **5.00** |
| Completeness (avg across categories) | Avg (1-5) | **3.87** |
| Naturalness | Avg (1-5) | **3.95** |
| CEFR Adherence | Avg (1-5) | **3.92** |

> 评估 LLM：DeepSeek V4 Flash。全量推理（2,174 样本）待云 GPU 执行后重新评估。

### 9.3 代码修复记录

| 修复 | 文件 | 说明 |
|------|------|------|
| checkpoint 预训练路径硬编码 | `src/medrax/tools/classification.py` | 传入 `model_overrides={"no_pretrained_weights": True}` |
| ECG 文件名零填充不匹配 | `src/inference_ecg_dialogue.py` | 添加 `normalize_ecg_filename()` 函数 |
| 不可达死代码 | `src/inference_ecg_dialogue.py` | 移除 `return "[]"` |
| 评估脚本 Gemini → DeepSeek | `src/eval/faithfulness.py` 等 3 个文件 | `google.generativeai` → `openai` (DeepSeek V4 Flash) |
| HF 数据集不可用 fallback | `src/eval/accuracy_completeness.py` | 从 JSONL 的 `ground_truth_dialogue` 提取 GT |
| CEFR 加载失败 non-fatal | `src/eval/naturalness_cefr.py` | HF 失败时默认 CEFR='B' |
| 报告解析 KeyError | `src/eval/naturalness_cefr.py` | `generate_report` 跳过缺少 score 的条目 |

---

## 10. 推荐执行顺序（按优先级排列）

1. ✅ 环境搭建 + 安装依赖（第 1 节）
2. ✅ 测量工具先行——最简单，验证 ECG 数据读取正确（第 4 节）
3. ✅ 下载 ECG-MTD 数据集，微调 Agent LLM（第 2 + 6A 节）
4. ✅ 分类工具微调完成（第 3 节）
5. ✅ 5% 测试集推理 + 四项评估验证通过（第 9 节）
6. ⏳ 云 GPU 全量推理（2,174 样本）+ 重新评估
7. ➕ 按需添加解释工具（第 5 节）
