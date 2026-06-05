# ECG-Agent 评估结果报告

**日期**：2026-05-29（Accuracy & Completeness 已重建为论文式临床参考 GT）  
**模型**：Llama 3.2 3B + LoRA (Unsloth 4-bit)  
**测试集**：ECG-MTD 10% 采样（217 / 2,174 样本）  
**导联配置**：仅 **12-lead**（论文另有 Lead I / Lead II 两种单导配置，本报告未拆分；见下方说明）  
**推理模式**：`without_gt`（模型生成历史，非 ground truth 历史）  
**评估 LLM**：DeepSeek V4 Pro（Faithfulness / Accuracy & Completeness / Naturalness & CEFR）  
**硬件**：RTX 4070 Laptop (8 GB VRAM)

---

## ⚠️ 与论文评测的 Ground Truth 差异（重要）

本报告各指标与论文（ECG-Agent.pdf）的 Ground Truth 对齐情况如下。**Accuracy & Completeness 已重建为与论文同源的临床参考 GT**（PTB-XL 诊断码 + PTB-XL+ Uni-G 测量值）；仅 CEFR 与裁判 LLM 仍有差异。

| 评估维度 | 论文 Ground Truth | 本报告 Ground Truth | 可比性 |
|----------|------------------|--------------------|--------|
| Next Action Prediction | ECG-MTD 测试集 action 标注（exact match） | 同源 | ✅ 可比 |
| Faithfulness | 工具输出 ↔ 回复 一致性（二分类） | 概念一致（工具输出本地重算） | ⚠️ 概念一致 |
| **Accuracy & Completeness** | **外部临床参考**：分类用 PTB-XL 医生标注诊断码；测量用 PTB-XL+ 的 **Uni-G** PQRST 测量值，再由裁判 LLM 生成 GT 回复 | **同源**：分类← PTB-XL `scp_codes`；测量← PTB-XL+ Uni-G PQRST；直接响应无临床参考，沿用 ECG-MTD 回复（论文同此） | ✅ **基准已对齐**（仅裁判 LLM 不同） |
| **CEFR Adherence** | 每条对话**真实标注的 CEFR 等级** | 全部固定为 'B' | ❌ 基准不同 |
| 裁判 LLM | Gemini-2.5-Pro | DeepSeek V4 Pro | ⚠️ 绝对分值不可比 |

---

## 导联配置说明（仅 12-lead）

论文将 ECG-Agent 拆成 **12-lead / Lead I / Lead II 三种独立训练+评测的系统**（各用对应导联的对话数据集 + 各自的单导分类模型，单导可检测的诊断类别更少）。**本报告只覆盖 12-lead 这一支**：

- 信号为 12 导 PTB-XL（`.mat` 形状 `(12, 5000)`）；分类工具为 71 类 12 导模型；微调 agent 也在 12 导对话上训练。
- 因此本报告各表**只能与论文的 12-lead 行对比**，不涉及 Lead I/II 行。

**为何暂不扩单导**：补齐 Lead I/II 需在两种单导上各重训分类器（现有 71 类模型只接受 12 通道）+ 理想情况下重训 agent，工作量约等于把训练流程再做两轮（数据非瓶颈，HF 上 `single-lead-I/II-ecg-mtd-dataset` 可获取）。

**对部署的影响**：DK-2500/Lepod 为 8 通道（V6, I, II, V1–V5），含 I、II 可线性推导 III/aVR/aVL/aVF，加上现成 V1–V6 → **可重建完整 12 导**，故 12-lead 配置在床旁设备上可直接部署，单导配置非部署必需。

---

## 1. 工具层验证（全量 21,799 条 PTB-XL 记录）

### 1.1 测量工具（NeuroKit2，CPU）

| 指标 | 总记录 | 有效数 | 覆盖率 | 均值 | 标准差 | 范围 |
|------|--------|--------|--------|------|--------|------|
| Heart Rate (bpm) | 21,799 | 21,717 | 99.6% | 74.7 | 17.4 | 36.9–197.4 |
| PR Interval (ms) | 21,799 | 20,853 | 95.7% | 132.6 | 33.1 | 80.0–370.0 |
| QRS Duration (ms) | 21,799 | 20,112 | 92.3% | 122.6 | 33.5 | 80.0–240.0 |
| QTc (ms) | 21,799 | 21,123 | 96.9% | 422.8 | 44.3 | 300.1–597.8 |

- 处理时间：~16 分钟（CPU）
- 失败率：0%（所有文件均被处理，部分指标因信号质量无法提取）

### 1.2 分类工具（ECG-FM + fairseq-signals，GPU）

| 指标 | 数值 |
|------|------|
| 总记录 | 21,799 |
| 有效预测 | 21,799 (100%) |
| 唯一 top-1 类别数 | 31 / 71 |
| 处理时间 | ~6.5 分钟（GPU） |

**Top-5 最常见 top-1 诊断类别：**

| 类别 | 数量 | 占比 |
|------|------|------|
| SR (窦性心律) | 11,940 | 54.8% |
| NORM (正常) | 4,312 | 19.8% |
| AFIB (房颤) | 1,159 | 5.3% |
| ASMI (前间壁心梗) | 623 | 2.9% |
| LAFB (左前分支阻滞) | 605 | 2.8% |

---

## 2. LLM 推理统计

| 指标 | 数值 |
|------|------|
| 测试样本 | 217 / 2,174（10%） |
| GT 评估轮次 | 926 turns |
| 平均轮次/样本（助手） | 4.2 |
| 推理速度 | ~42 s/样本（RTX 4070 Laptop 8 GB） |
| 总推理耗时 | ~2 小时 33 分 |

---

## 3. Next Action Prediction（本地评估，无需 API）

论文 Table 4 格式：w/o GT 为真实场景（自生成历史），w/ GT 为上界（使用标注历史）。

| 动作类型 | GT 总数 | w/o GT | w/ GT |
|----------|---------|--------|-------|
| call_classification_tool | 124 | **99.19%** | **99.19%** |
| call_measurement_tool | 96 | **89.58%** | **90.62%** |
| response | 274 | **98.54%** | **98.91%** |
| response_followup | 203 | **98.52%** | **100.00%** |
| system_bye | 217 | **98.16%** | **98.62%** |
| response_fail | 12 | 58.33% | 66.67% |
| **Overall** | **926** | **97.08%** | **97.84%** |
| *response_fail（全量补测）* | *110* | ***62.73%*** | ***66.36%*** |

> `response_fail` 在 10% 采样里仅 12 条，准确率不稳，故补做全量评估。
>
> **response_fail 全量补测**：全测试集 2,174 条中含 `response_fail` 的样本共 **110 个**（每样本 1 条，是最稀有动作）。用 `--filter-action response_fail` 对这 110 条本地全量推理（without_gt + with_gt 并发，RTX 4070 Laptop 8GB，约 78 分钟），NAP 准确率 **62.73%（w/o GT）/ 66.36%（w/ GT）**，比 12 样本的 58.33%/66.67% 更可靠。该补测行不计入上方 Overall（Overall 仍为 10% 采样的 926 轮）。产物：`results/inference_3B_response_fail_{without,with}_gt.jsonl`。

---

## 4. Faithfulness（DeepSeek V4 Pro 评估）

| 指标 | 评估轮次 | 分数 |
|------|---------|------|
| Faithfulness | 209 | **0.909** (90.9%) |

> 评估的是模型回复与工具输出之间的一致性。1 = 一致，0 = 不一致。

---

## 5. Accuracy & Completeness（论文式临床参考 GT，DeepSeek V4 Pro 评估，1–5 分制）

**Ground Truth 与论文同源**：分类问题用 PTB-XL 心脏科医生标注诊断码（`scp_codes` + `scp_statements`），测量问题用 PTB-XL+ 的 **Uni-G** PQRST 区间测量值（`HR/PR/QRS/QT/QTc`），由裁判 LLM 据此生成 GT 回复，再对模型回复打分。直接响应无临床参考，沿用 ECG-MTD 回复（论文同此处理）。

| 类别 | 评估数 | Avg Accuracy | Avg Completeness |
|------|--------|--------------|------------------|
| Post-Classification | 124 | **3.76** | **2.63** |
| Post-Measurement | 84 | **2.25** | **2.50** |
| Direct Response | 66 | **4.77** | **3.86** |

> GT 来源：post_classification ← PTB-XL 诊断码（124 条全命中）；post_measurement ← PTB-XL+ Uni-G 测量（84 条全命中）；direct_response ← ECG-MTD 回复（66 条）。共 274 条，0 解析失败。
> 
> **与论文的可比性**：GT 基准已对齐论文（Table 1/2）；唯一差异是裁判 LLM（本报告 DeepSeek V4 Pro，论文 Gemini-2.5-Pro），故绝对分值仍有偏移。
>
> **为何比旧的自参照分数低**：旧版用 ECG-MTD 合成对话回复当 GT（模型本就照着同分布训练，分数偏高）。改用权威临床参考后，Post-Measurement 降幅最大——因为模型回复的测量值来自 NeuroKit2，与 Uni-G 仪器测量存在系统偏差（例如 PR 100ms vs 114ms），按 Uni-G 评判即被判为不准确。这更真实地反映了模型相对临床基准的准确度。
>
> 产物：`results/accuracy_completeness_paper_gt/`（`scored_items.jsonl` 含每条参考数据/GT/模型回复/分数，`evaluation_report.md` 汇总）。脚本：`src/eval/accuracy_completeness_paper_gt.py`。

---

## 6. Naturalness & CEFR Adherence（DeepSeek V4 Pro 评估，1–5 分制）

| 指标 | 有效评估数 | 解析失败数 | 平均分 |
|------|-----------|-----------|--------|
| Naturalness | 195 | 5 | **4.56** |
| CEFR Adherence | 195 | 5 | **4.11** |

> ⚠️ **基准与论文不同**：CEFR level 全部默认为 'B'（HF 数据集不可用，无法获取每条对话的原始 CEFR 标注）。论文按各对话真实标注的 CEFR 等级评判，因此本表 CEFR Adherence **不可与论文 Table 5 直接对比**。

---

## 7. 结果汇总

| 评估维度 | 指标 | 分数 |
|----------|------|------|
| Next Action Prediction (w/o GT) | Overall Accuracy | **97.08%** |
| Next Action Prediction (w/ GT) | Overall Accuracy | **97.84%** |
| Faithfulness | Alignment Rate | **90.9%** |
| Accuracy (Post-Classification) | Avg Score (1-5) | **3.76** |
| Accuracy (Post-Measurement) | Avg Score (1-5) | **2.25** |
| Accuracy (Direct Response) | Avg Score (1-5) | **4.77** |
| Completeness (Post-Classification) | Avg Score (1-5) | **2.63** |
| Completeness (Post-Measurement) | Avg Score (1-5) | **2.50** |
| Completeness (Direct Response) | Avg Score (1-5) | **3.86** |
| Naturalness | Avg Score (1-5) | **4.56** |
| CEFR Adherence | Avg Score (1-5) | **4.11** |

> Accuracy / Completeness 已用论文式临床参考 GT（PTB-XL + PTB-XL+ Uni-G）评测，基准与论文 Table 1/2 对齐，仅裁判 LLM 不同（DeepSeek vs Gemini-2.5-Pro）。NAP 同源可比。CEFR Adherence 仍因 CEFR 标注缺失固定为 'B'，不可与论文 Table 5 直接对比。

---

## 8. 产出文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| 测量工具输出 | `results/ecg_analysis_measurements.csv` | 21,799 行 |
| 分类工具输出 | `results/ecg_analysis_classification.csv` | 21,799 行 |
| LLM 推理结果 (w/o GT) | `results/inference_3B_10pct_without_gt.jsonl` | 217 样本 |
| LLM 推理结果 (w/ GT) | `results/inference_3B_10pct_with_gt.jsonl` | 217 样本 |
| response_fail 全量推理 (w/o GT) | `results/inference_3B_response_fail_without_gt.jsonl` | 110 样本 |
| response_fail 全量推理 (w/ GT) | `results/inference_3B_response_fail_with_gt.jsonl` | 110 样本 |
| Faithfulness 分数 | `results/faithfulness_scores_10pct.json` | |
| Accuracy & Completeness（论文式 GT） | `results/accuracy_completeness_paper_gt/` | 含 scored_items.jsonl、reference_gt_cache.json、evaluation_report.md（**当前采用**） |
| Naturalness & CEFR | `results/naturalness_cefr_10pct/` | 含 llm_eval_dialogue_quality.json |
| 论文式 GT 评估脚本 | `src/eval/accuracy_completeness_paper_gt.py` | 两阶段、并发、可断点续跑 |
| Uni-G 测量参考 | `data/ptbxl_plus/unig_features.csv` | PTB-XL+ 1.0.1（PhysioNet，21,795 条） |

---

## 9. 待办：全量推理

全量推理（2,174 样本）建议在云 GPU 上执行：

```bash
cd ~/workspace/ECG-Agent/ECG-Agent
python src/inference_ecg_dialogue.py \
  --base-model-path unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit \
  --adapter-path ../ecg-dialogue-finetune/Llama-3.2-3B-Instruct \
  --inference-mode without_gt
```

预计耗时：RTX PRO 6000 约 5–8 小时。完成后用相同的评估脚本重新跑四项评估。
