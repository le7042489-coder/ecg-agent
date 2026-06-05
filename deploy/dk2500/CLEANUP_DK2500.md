# DK-2500 床旁项目整理指南（给设备上的 Claude Code 执行）

> 这份文档是在**开发机**上写的，交给 **DK-2500 设备上的 Claude Code** 按步骤执行。
> 你（设备上的 Claude Code）没有开发机那边的上下文，下面把背景和原则都讲清楚了。

## 背景

- 开发机（用户的 WSL 工作站）这边的项目刚做过一轮整理：提交了所有代码、删掉可再生的大产物（释放 ~7.5G）、理顺了 git 远端、并**把全部源码/文档/部署码备份到了用户自己的 GitHub**：
  `https://github.com/le7042489-coder/ecg-agent` （分支 `local-work`，`origin`=用户库，`upstream`=官方只读库）。
- **本设备（DK-2500）是部署目标**，不是开发仓库。它通过 `ecg-agent-dk2500.tar.gz` 解压得到模型+源码+部署脚本，运行 `bedside_agent.py` 做床旁 ECG 分析对话。
- 目标：**把这台设备也整理干净、腾出磁盘**，但**绝不能弄坏正在工作的床旁系统**。

## ⚠️ 最高优先级原则（务必遵守）

1. **这是正在使用的医疗设备**（Core Ultra 5 225U · 8GB RAM · 128GB SSD · CPU-only）。删任何东西前先**诊断看清楚**，不确定就**停下来问人**，不要猜着删。
2. **绝不自动删除患者 ECG 记录**：`~/workspace/ecg_*.csv`、`*.mat` 很可能是真实采集的患者数据。是否归档/删除**交给人决定**。
3. **不要动**：conda 环境 `ecg-bedside`、模型权重（见第 3 节保留清单）、`~/workspace/lepod_ecg.py`（采集脚本）。
4. 删之前**先列清单**，可再生/重复/缓存类可以删，**贵的或不确定的先问人**；删完**报告腾出了多少空间**。
5. 整理完先跑一次冒烟测试，确认 `bedside_agent.py` 还能正常加载再收工。

## 第 0 步：诊断（只读，先看清楚再动手）

```bash
echo "===== 磁盘总览（128GB SSD 用了多少）====="
df -h /

echo "===== 项目目录体积分布 ====="
du -sh ~/workspace/ECG-Agent/* 2>/dev/null | sort -rh | head -20
du -sh ~/workspace/* 2>/dev/null | sort -rh | head -20

echo "===== 内存 ====="
free -h

echo "===== conda 环境 ====="
conda env list

echo "===== HuggingFace 缓存（路径A transformers 会用到的 base model）====="
du -sh ~/.cache/huggingface 2>/dev/null || echo "无 HF 缓存"

echo "===== 是否 git 仓库（决定能否 diff 出设备侧改动）====="
git -C ~/workspace/ECG-Agent/ECG-Agent rev-parse --is-inside-work-tree 2>/dev/null || echo "ECG-Agent/ECG-Agent 不是 git 仓库"
git -C ~/workspace/ECG-Agent rev-parse --is-inside-work-tree 2>/dev/null || echo "顶层不是 git 仓库"

echo "===== 患者数据（采集输出，先数数量、别删）====="
ls -lh ~/workspace/ecg_*.csv ~/workspace/ecg_*.mat 2>/dev/null | tail -10
echo "CSV 数量: $(ls ~/workspace/ecg_*.csv 2>/dev/null | wc -l)"
```

把上面结果先看一遍，对照下面的清单再决定。

## 第 1 步：确认本机在用哪条 LLM 路径（最大的空间机会）

设备上跑 LLM 有两条路，**只会用其中一条**，另一条的 ~2GB 大文件就是死重：

| 路径 | 需要的大文件 | 不需要 |
|------|-------------|--------|
| A：transformers 4-bit | `~/.cache/huggingface` 里的 base model（~2GB） | gguf |
| B：gguf + llama-cpp（部署指南推荐） | `ecg_agent_llama3b_q4km.gguf`（~2GB） | HF base 缓存 |

**怎么判断在用哪条**：回忆/询问平时怎么启动 `bedside_agent.py`（有没有 `--backend llama-cpp --gguf ...`），或问设备使用者。
- 确定用 **B（gguf）** → HF base 缓存可清（腾 ~2GB）。
- 确定用 **A（transformers）** → gguf 文件可清（腾 ~2GB）。
- **两条都不确定就都先留着**，问人之后再清。

## 第 2 步：可安全清理（诊断确认后逐项删，删前报大小）

| 候选 | 预计大小 | 前提/说明 |
|------|---------|-----------|
| `ecg-agent-dk2500.tar.gz` | ~1.06G | **确认已解压**（checkpoints/、ecg-dialogue-finetune/ 都在）后即可删——解压后它没用了。最大的稳妥一删。 |
| 未使用那条 LLM 路径的大文件 | ~2G | 见第 1 步，二选一 |
| conda / pip 缓存 | 常有几个 G | `conda clean -a -y` ＋ `pip cache purge` |
| `__pycache__` / `*.pyc` | 小 | `find ~/workspace/ECG-Agent -type d -name __pycache__ -prune -exec rm -rf {} +` |
| `checkpoint-1500/`、`checkpoint-1525/` | ~316M | 训练中间 checkpoint，推理不需要（部署指南说本就不该传，确认在的话可删） |
| `Miniconda3-latest-Linux-x86_64.sh` | ~150M | 安装包，装完即可删（若还在 ~ 或 ~/workspace） |

> 删 tar 包前务必先确认解压完整，命令示例：
> ```bash
> ls -lh ~/workspace/ECG-Agent/checkpoints/checkpoint_best.pt \
>        ~/workspace/ECG-Agent/ecg-dialogue-finetune/Llama-3.2-3B-Instruct/adapter_model.safetensors
> # 两个都在且大小正常，才删 tar 包：
> # rm ~/workspace/ECG-Agent/ecg-agent-dk2500.tar.gz
> ```

## 第 3 步：保留（别碰）

- `checkpoints/checkpoint_best.pt`（分类编码器，1.1G，**不可再生**）
- `ecg-dialogue-finetune/.../adapter_model.safetensors` + tokenizer（LoRA 微调成果）
- **在用**那条 LLM 路径的模型（gguf 或 HF base 缓存，二选一保留）
- 患者 ECG 记录 `ecg_*.csv` / `*.mat`（医疗数据，**人来决定**）
- conda 环境 `ecg-bedside`、`~/workspace/lepod_ecg.py`

## 第 4 步：捕获设备上的本地修改（重要——别让设备侧的修复丢了）

部署到本设备的 `ECG-Agent/src/` 是当时打包的**快照**。如果后来**在设备上改过代码**让它在 CPU/8GB 下跑通（部署指南提到过的：`classification.py` 的 `no_pretrained_weights`、`lepod2mat.py` 列名处理、`bedside_agent.py` 的后端逻辑等），这些改动可能**只在本设备上、开发机仓库里没有**。

- 开发机代码已备份在 GitHub：`https://github.com/le7042489-coder/ecg-agent`（分支 `local-work`）。
- **做法**：
  - 若本设备能联网：`git clone` 上面那个仓库到临时目录，和设备上的 `ECG-Agent/src/`（及 `dk2500_deploy/`）逐文件 `diff`，把**设备独有的修复**整理出来。能 push 就提交回 `local-work` 并 push；不能就把差异文件交给人带回开发机。
  - 若不能联网：把设备上改过的关键文件（至少 `classification.py`、`lepod2mat.py`、`bedside_agent.py`）拷出来（U 盘），交给人在开发机比对合并。
- 这一步**只读+导出**，不改设备上正在用的代码。

## 第 5 步：理顺结构（可选，低优先级）

- 设备的目标目录布局见同目录《DK-2500_ECG-Agent_部署指南.md》第 5 节。
- 若发现重复/套娃（例如多份 `fairseq-signals/`、或类似开发机那种 `ECG-Agent/ECG-Agent` 嵌套、解压残留的重复目录），用同一原则：**留权威的一份、删重复的**，删前确认哪份在被实际引用（`bedside_agent.py` / `classification.py` 实际 import 的那份）。
- 可选：给 `~/workspace/ECG-Agent` 做 `git init` 或直接从 GitHub clone 一份，方便以后追踪设备侧改动（但注意 8GB/128GB 资源，别把大模型也纳入 git）。

## 收尾：报告

- `df -h /` 前后对比，**腾出多少 GB**。
- 列出：删了什么、保留了什么、第 1 步判定用的是哪条 LLM 路径。
- **第 4 步是否发现设备独有的代码修改**——如果有，明确告诉人需要带回开发机/已 push。
- 跑一次冒烟测试确认 `bedside_agent.py` 仍能加载（不要在没有患者在用时之外的时间打断正在进行的采集）。

---

*整理方法论和开发机那轮一致：诊断 → 保护在用系统 → 分清「可再生/重复/缓存（删）」vs「成果/患者数据（留）」→ 删前列清单、贵的先问人 → 报告腾出的空间。稳妥第一，这是医疗设备。*
