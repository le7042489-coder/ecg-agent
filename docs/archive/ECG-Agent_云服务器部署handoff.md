# ECG-Agent 云服务器部署 Handoff

> 目标 GPU：NVIDIA RTX PRO 6000（48GB+ VRAM）  
> 最后更新：2026-05-25

---

## 1. 已完成（本地）

| 步骤 | 状态 | 备注 |
|------|------|------|
| 环境搭建 | ✅ | conda `ecg-agent`，PyTorch 2.10+cu128，CUDA 可用 |
| PTB-XL 数据集 | ✅ | 下载并解压，`ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`（3.2GB，21,799 条记录） |
| ECG-MTD 对话数据集 | ✅ | HuggingFace `datasets` 自动下载（32MB，73K 条对话） |
| 预处理（WFDB→.mat） | ✅ | 21,799 个 `.mat` 文件，`ECG-Agent/data/ptbxl_mat/`（9.9GB） |
| 测量工具验证 | ✅ | NeuroKit2，0.09s/条，PQRST + 心率正常 |
| fairseq-signals | ✅ | 已安装，与当前环境无冲突 |
| ecg-fm checkpoint 验证 | ✅ | `wanglab/ecg-fm` 已下载，经 fairseq-signals 加载成功 |
| PTB-XL manifest | ✅ | 已生成于 `ECG-Agent/data/ptbxl_mat/manifest/`（train/valid/test = 18,553/1,081/2,165） |
| 解释工具 | ✅ | `Time_is_not_Enough` 已 clone（未安装依赖，避免 torch 冲突） |
| LLM 微调脚本 | ✅ | `src/finetune_ecg_dialogue_unsloth.py`，已适配 transformers v5 |
| cloud_setup.sh | ✅ | 一键环境安装脚本 |
| ecg-agent-cloud.tar.gz | ✅ | 云传输包（265MB）：含 `src/` + 2,918 个 .mat 样本（全集 13%），**不含** manifest |

---

## 2. 需要传到云服务器的文件

```
ECG-Agent/                                 # 仓库根目录（git clone 后 cd 进入此目录）
├── src/
│   ├── finetune_ecg_dialogue_unsloth.py   # LLM 微调脚本（已适配）
│   ├── inference_ecg_dialogue.py          # LLM 推理脚本
│   ├── extract_tool_outputs.py            # 工具批量推理（注意在 src/ 下）
│   ├── ptbxl_files/
│   │   └── ptbxl_database.csv             # PTB-XL 索引（21,837 条）
│   ├── preprocess/
│   │   ├── preprocess_ptbxl.py            # WFDB→.mat 转换
│   │   └── manifest_ptbxl_10s.py          # PTB-XL manifest 生成（已修复索引 bug）
│   └── medrax/                            # Agent 工具代码
├── cloud_setup.sh                         # 一键环境安装（需先修改 HF token）
├── requirements.txt                       # pip 依赖参考
├── ecg-agent-cloud.tar.gz                 # 云传输包（265MB）：含 src/ + 2,918 个 .mat 样本
├── data/
│   └── ptbxl_mat/                         # 预处理的 .mat 文件（9.9GB，21,799 个）
│       └── manifest/                      # 已生成的 train/valid/test splits
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/  # 原始 PTB-XL（3.2GB）
├── fairseq-signals/                       # fairseq-signals 框架（已安装）
└── Time_is_not_Enough/                    # 解释工具源码（可选）
```

> ECG-MTD 数据集无需传输，云服务器上 `datasets` 库直接拉取。  
> `ecg-agent-cloud.tar.gz` 内含 2,918 个 .mat 样本（全集 13%），**不含** manifest；若要传完整数据集，需单独打包 `data/ptbxl_mat/`（含 manifest 子目录，共 9.9GB）。  
> `src/ptbxl_files/` 已通过 `cloud_setup.sh` 中的 `ln -sf src/ptbxl_files ptbxl_files` 做了根目录软链，云上会自动处理。

---

## 3. 云服务器操作步骤

### 3.1 环境搭建

```bash
bash cloud_setup.sh
# 完成后登录 HF
hf auth login
```

### 3.2 安装 fairseq-signals（如未传输已安装的目录）

```bash
git clone --depth 1 https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals && pip install --editable .
```

### 3.3 重新生成 PTB-XL manifest（如 .mat 文件在云上重新生成）

```bash
python src/preprocess/manifest_ptbxl_10s.py \
  --root "$(pwd)" \
  --data-dir data/ptbxl_mat \
  --dest data/ptbxl_mat/manifest \
  --valid-percent 0.05 --test-percent 0.1 --seed 42
```

> 如果完整传输了 `data/ptbxl_mat/`（含 manifest 子目录），此步可跳过。若通过 `ecg-agent-cloud.tar.gz` 传输，manifest **未包含**，必须执行此步。

### 3.4 LLM 微调（Section 6A）

```bash
python src/finetune_ecg_dialogue_unsloth.py \
  --model unsloth/Llama-3.2-3B-Instruct \
  --output-dir ./ecg-dialogue-finetune/Llama-3.2-3B-Instruct \
  --batch-size 8 --grad-accum 4
```

| 参数 | RTX PRO 6000 建议值 | 说明 |
|------|---------------------|------|
| `--batch-size` | 8 | 48GB VRAM 可尝试 16 |
| `--grad-accum` | 4 | effective batch ≈ 32 |
| `--max-seq-length` | 4096 | 默认值 |

预计耗时：**2-4 小时**。

### 3.5 分类工具——跳过预训练方案（Section 3）

ecg-fm 预训练权重在本地已验证可加载，云服务器上只需：

```bash
# 下载 ecg-fm 预训练权重
hf download wanglab/ecg-fm mimic_iv_ecg_physionet_pretrained.pt

# PTB-XL 微调
fairseq-hydra-train \
  model.model_path=/path/to/mimic_iv_ecg_physionet_pretrained.pt \
  +task.data=data/ptbxl_mat/manifest \
  --config-dir fairseq-signals/examples/w2v_cmsc/config/finetuning/ecg_transformer \
  --config-name diagnosis
```

预计耗时：**1-3 小时**。

### 3.6 端到端推理验证（Section 7）

```bash
# 测量工具（就绪）
python src/extract_tool_outputs.py --tool measurements \
  --ecg_dir data/ptbxl_mat --output_dir results/measurements

# 分类工具（微调完成后）
python src/extract_tool_outputs.py --tool classification \
  --ecg_dir data/ptbxl_mat --model_path /path/to/checkpoint_best.pt \
  --output_dir results/classification

# LLM 推理（微调完成后）
python src/inference_ecg_dialogue.py
```

---

## 4. 微调脚本关键修改

相比原版 `finetune_ecg_dialogue_unsloth.py` 的改动：

| 改动 | 原因 |
|------|------|
| `load_in_4bit=True` | 4-bit 量化加载 |
| 预分词 + `skip_prepare_dataset=True` | 避免 Unsloth × datasets 多进程 pickle 错误 |
| `eos_token='<\|eot_id\|>'`, `pad_token` 显式设置 | 兼容 transformers v5 `TokenizersBackend` |
| `report_to="none"` | 无 wandb 时报错修复 |
| `processing_class=tokenizer` | 绕过 TRL 对 tokenizer 的错误封装 |
| 新增 `--batch-size` / `--grad-accum` CLI 参数 | 按硬件灵活调整 |
| special tokens `<EOS_TOKEN>`/`<PAD_TOKEN>` | 解决 SFTConfig.to_dict() 序列化问题 |

---

## 5. Bug 修复记录

### manifest_ptbxl_10s.py — patient_id 索引错误

**问题：** `data['patient_id'][0][0]` 对 scipy 1D 数组取到字符串首字符，导致 21,799 条记录只有 9 个"患者"。

**修复：** 改为 `data['patient_id'].flatten()[0]`，并用 `int(float(pid))` 兼容 `'15709.0'` 格式。

```python
# 修复前
def get_patient_id(file_path):
    data = scipy.io.loadmat(file_path)
    return int(data['patient_id'][0][0])  # BUG: 对字符串取 [0] 返回首字符

# 修复后
def get_patient_id(file_path):
    data = scipy.io.loadmat(file_path)
    pid = data['patient_id'].flatten()[0]
    return int(float(pid))
```

### finetune_ecg_dialogue_unsloth.py — 多项兼容性修复

见第 4 节表格。

---

## 6. 尚未完成（需在云服务器执行）

| 项目 | 说明 | 预计耗时 |
|------|------|----------|
| LLM 微调 | 脚本就绪 | 2-4 小时 |
| 分类器 PTB-XL 微调 | ecg-fm 作起点，fairseq-hydra-train | 1-3 小时 |
| 端到端推理 | 等微调完成 | 几分钟 |
| 解释工具 | 可选，仅单导联 | — |

---

## 7. 环境注意事项

- **transformers v5**：Unsloth tokenizer 被包装为 `TokenizersBackend`，已在脚本中适配
- **flash-attn**：编译失败不影响，Unsloth 回退 xformers
- **fairseq-signals**：当前环境无冲突，已直接安装在 ecg-agent 中
- **Time_is_not_Enough**：其 requirements.txt 含 torch 2.2.1，建议单独 conda 环境安装
- **sglang / tensorflow**：requirements.txt 中包含但推理不需要，可跳过
