# DK-2500 床旁 ECG-Agent 部署指南

**硬件**: Intel Core Ultra 5 225U · 8GB DDR5 · 128GB SSD · Ubuntu 24.04  
**ECG 设备**: Lepod Pro (ER3)，BLE，8 通道，250Hz  
**目标**: 采集 ECG → 自动分析 → 自然语言对话

---

## 0. 整体架构

```
Lepod Pro (BLE)
      │ lepod_ecg.py（已有）
      ▼
ecg_YYYYMMDD.csv   ─── lepod2mat.py ───▶  ecg_YYYYMMDD.mat
  8通道, 250Hz                              12导联, 500Hz, 5000采样
                                                    │
                                      ┌─────────────┤
                                      │  bedside_agent.py
                                      │  ┌──────────────────────────┐
                                      │  │ 分类工具 (fairseq-signals) │ CPU
                                      │  │ 测量工具 (NeuroKit2)       │ CPU
                                      │  │ LLM (Llama 3.2 3B + LoRA) │ CPU
                                      │  └──────────────────────────┘
                                      │           │
                                      └───────────▶ 终端对话
```

**12 导联重建**（Lepod 8ch → 标准 12 导联）：

| 导联 | 来源 |
|------|------|
| I, II, V1–V6 | Lepod 直接输出 |
| III | II − I |
| aVR | −(I + II) / 2 |
| aVL | I − II / 2 |
| aVF | II − I / 2 |

---

## 1. 开发机侧：准备传输文件

在你的**开发机**（AutoDL 或本地笔记本）上执行，打包需要传到 DK-2500 的文件。

### 1.1 模型文件清单

```
需要传到 DK-2500 的文件（共约 1.3 GB）：
├── checkpoints/checkpoint_best.pt                           1.1 GB  分类工具权重
├── ecg-dialogue-finetune/Llama-3.2-3B-Instruct/
│   ├── adapter_config.json                                   ~4 KB
│   ├── adapter_model.safetensors                            ~93 MB  LoRA 权重
│   ├── tokenizer.json                                        ~17 MB
│   ├── tokenizer_config.json                                 ~52 KB
│   └── chat_template.jinja                                   ~4 KB
├── fairseq-signals/                                          ~14 MB  分类框架
├── ECG-Agent/src/                                            ~33 MB  推理源码
└── dk2500_deploy/                                           ~96 KB  部署脚本
```

> **不需要传输**：PTB-XL 数据（9.9 GB）、预计算 CSV、训练脚本、
> `checkpoint-1500/`、`checkpoint-1525/`（中间 checkpoint，共 316MB，推理不需要）。

### 1.2 打包

```bash
# 在开发机上，进入 ~/workspace/ECG-Agent
cd ~/workspace/ECG-Agent

tar -czf ecg-agent-dk2500.tar.gz \
    checkpoints/checkpoint_best.pt \
    ecg-dialogue-finetune/Llama-3.2-3B-Instruct/adapter_config.json \
    ecg-dialogue-finetune/Llama-3.2-3B-Instruct/adapter_model.safetensors \
    ecg-dialogue-finetune/Llama-3.2-3B-Instruct/tokenizer.json \
    ecg-dialogue-finetune/Llama-3.2-3B-Instruct/tokenizer_config.json \
    ecg-dialogue-finetune/Llama-3.2-3B-Instruct/chat_template.jinja \
    fairseq-signals/ \
    ECG-Agent/src/ \
    dk2500_deploy/

echo "打包完成：$(du -sh ecg-agent-dk2500.tar.gz)"
# 预期约 1.3 GB（checkpoint_best.pt 占大头）
```

### 1.3 传输到 DK-2500

DK-2500 地址：`user@10.157.225.11`

```bash
# 从开发机推送（需与 DK-2500 在同一局域网）
scp ecg-agent-dk2500.tar.gz user@10.157.225.11:~/workspace/ECG-Agent/

# 如果 ~/workspace/ECG-Agent/ 目录不存在，先建：
ssh user@10.157.225.11 "mkdir -p ~/workspace/ECG-Agent"
scp ecg-agent-dk2500.tar.gz user@10.157.225.11:~/workspace/ECG-Agent/
```

或通过 U 盘离线传输：
```bash
# 复制到 U 盘
cp ecg-agent-dk2500.tar.gz /media/$USER/<U盘名>/

# DK-2500 上从 U 盘解压
cp /media/user/<U盘名>/ecg-agent-dk2500.tar.gz ~/workspace/ECG-Agent/
```

---

## 2. DK-2500 侧：安装部署

所有操作在 **DK-2500** 上执行（通过 SSH 或直连键盘）。

### 2.1 安装 Miniconda（如未安装）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2.2 解压文件

```bash
cd ~/workspace/ECG-Agent
tar -xzf ecg-agent-dk2500.tar.gz

# 验证关键文件
ls -lh checkpoints/checkpoint_best.pt
ls -lh ecg-dialogue-finetune/Llama-3.2-3B-Instruct/adapter_model.safetensors
ls dk2500_deploy/
# 应看到: setup_dk2500.sh  bedside_agent.py  lepod2mat.py  DK-2500_ECG-Agent_部署指南.md
```

### 2.4 运行安装脚本

```bash
cd ~/workspace/ECG-Agent/dk2500_deploy
chmod +x setup_dk2500.sh
bash setup_dk2500.sh
```

脚本自动完成：
- 创建 conda 环境 `ecg-bedside`
- 安装 PyTorch CPU 版、transformers、fairseq-signals 等全部依赖
- 验证关键文件
- 冒烟测试

> 首次运行约需 **10–20 分钟**（主要是下载 PyTorch CPU 包，约 150 MB）。

---

## 3. LLM 准备（二选一）

8GB RAM 下有两条路，**推荐路径 B**（更稳定，推理更快）。

### 路径 A：transformers 4-bit（开箱即用，内存较紧）

无需额外操作，`bedside_agent.py --backend transformers` 会自动加载。

- 首次运行时从 HuggingFace 下载 base model（`unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`，约 2 GB）
- 内存占用：base model ~2 GB + adapter ~200 MB + 工具 ~500 MB + OS ~2 GB = 约 **4.7 GB**
- 推理速度：约 **1–3 token/s**（Core Ultra 5 225U CPU）

> **注意**：`bitsandbytes` 的 4-bit 量化在 CPU 上是实验性的。如果报错 `CUDA not available`，需要改用路径 B，或者手动将 `load_in_4bit` 改为 `torch_dtype=torch.float32`（内存需求约 12GB，8GB 下会 OOM）。

如果 4-bit 加载失败，脚本会自动回退到 float16 CPU 加载，此时需要约 6.4GB RAM for weights alone，在 8GB 下非常紧张。这种情况下强烈建议切换到路径 B。

### 路径 B：GGUF + llama-cpp（推荐，内存小，速度快）

**在开发机上执行**（需要较大内存或 GPU 做合并）：

#### B.1 合并 LoRA adapter 到 base model

```bash
conda activate ecg-agent  # 或你的训练环境

python - <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
adapter = "../ecg-dialogue-finetune/Llama-3.2-3B-Instruct"
out_dir = "./merged_llama3b_ecg"

print("加载 base model...")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base)

print("合并 LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()

print(f"保存合并后模型到 {out_dir} ...")
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print("完成")
EOF
```

#### B.2 转换为 GGUF

```bash
# 安装 llama.cpp（用于转换）
pip install gguf

# 下载 llama.cpp 转换脚本
git clone https://github.com/ggerganov/llama.cpp --depth=1
cd llama.cpp
pip install -r requirements.txt

# 转换并量化（Q4_K_M 是质量/大小最佳平衡）
python convert_hf_to_gguf.py ../merged_llama3b_ecg \
    --outfile ecg_agent_llama3b_q4km.gguf \
    --outtype q4_K_M

echo "GGUF 文件大小: $(du -sh ecg_agent_llama3b_q4km.gguf)"
# 预期约 2.0 GB
```

#### B.3 传输 GGUF 到 DK-2500

```bash
scp ecg_agent_llama3b_q4km.gguf user@10.157.225.11:~/workspace/ECG-Agent/
```

#### B.4 DK-2500 上使用 GGUF

```bash
conda activate ecg-bedside
cd ~/workspace/ECG-Agent/dk2500_deploy

python bedside_agent.py \
    --ecg ~/workspace/ecg_20260527_143022.csv \
    --backend llama-cpp \
    --gguf ~/workspace/ECG-Agent/ecg_agent_llama3b_q4km.gguf
```

- 内存占用：GGUF ~2 GB + 工具 ~500 MB + OS ~2 GB = 约 **4.5 GB**
- 推理速度：约 **3–8 token/s**（使用所有 CPU 核）

---

## 4. 日常使用

激活环境：

```bash
conda activate ecg-bedside
cd ~/workspace/ECG-Agent/dk2500_deploy
```

### 场景 1：分析已有 Lepod CSV

```bash
python bedside_agent.py --ecg ~/workspace/ecg_20260527_143022.csv
```

### 场景 2：实时采集 + 立即分析

```bash
# 采集 30s ECG，然后自动进入对话
python bedside_agent.py --live 30
```

> 需要 Lepod Pro 开机并靠近 DK-2500（BLE 有效距离 ~5m）

### 场景 3：直接分析 .mat 文件

```bash
# 如果已经转换好了 .mat，跳过 CSV 步骤
python bedside_agent.py --mat /path/to/ecg.mat
```

### 场景 4：单独转换 CSV→.mat（不启动对话）

```bash
python lepod2mat.py ~/workspace/ecg_20260527_143022.csv
# 输出: ~/workspace/ecg_20260527_143022.mat
```

### 场景 5：网页界面（推荐演示/床旁）

```bash
# 起本机网页服务（数据源参数与 CLI 一致：--mat / --ecg / --live）
python web_server.py --mat /path/to/ecg.mat \
  --backend llama-cpp --gguf ~/workspace/ECG-Agent/ecg_agent_llama3b_q4km.gguf
# 然后在 DK-2500 上用 Firefox 打开 http://127.0.0.1:8000
```

界面三块：**12 导联波形** + **分析发现面板**（分类/测量/电轴/信号质量）+ **流式问答**（逐字显示）。

- 想从**另一台机器**（如笔记本）访问：加 `--host 0.0.0.0`，浏览器开 `http://<DK-2500_IP>:8000`。
- 端口默认 8000，可用 `--port` 改；transformers 后端同理（`--backend transformers`，不需 `--gguf`）。
- 纯标准库实现（`http.server` + SSE），无额外依赖、可离线运行，无需联网取 CDN。

**语音（后期，已留钩子）**：
- 🔊「朗读回答」走浏览器 TTS——装上 `speech-dispatcher` 语音并接扬声器后即出声，无需改代码。
- 🎤 麦克风暂禁用：Firefox **无离线语音识别**，需后期接**服务端 STT**（如 whisper.cpp）到 `web_server.py` 的 `/api/stt` 占位路由。

> CLI 与网页共用同一套核心（`agent_core.py`），回答逻辑完全一致。

### 典型对话示例

```
ECG-Agent 已就绪。输入问题开始对话，输入 'exit' 退出。
================================================================

您: 我的心跳正常吗？

ECG-Agent: Your heart rate is 74 bpm, which falls within the normal range of 60–100 bpm. 
           Your rhythm appears to be sinus rhythm (SR), which is a normal, regular heartbeat 
           pattern. Overall, your heart rate looks good!

您: PR 间期和 QRS 时限怎么样？

ECG-Agent: Your PR interval is 148 ms (normal: 120–200 ms) and your QRS duration is 95 ms 
           (normal: 80–120 ms). Both measurements are within normal limits, suggesting normal 
           conduction through your heart's electrical system.

您: 谢谢
ECG-Agent: You're welcome! Take care and feel free to ask if you have any other questions 
           about your ECG readings.
```

---

## 5. 文件结构

安装完成后，DK-2500 的工作目录：

```
~/workspace/ECG-Agent/
├── dk2500_deploy/                     ← 本目录
│   ├── setup_dk2500.sh               一键安装脚本
│   ├── bedside_agent.py              CLI 入口
│   ├── agent_core.py                 可复用核心（CLI/网页共用）
│   ├── web_server.py                 网页服务（http.server + SSE）
│   ├── web/index.html               网页界面（单文件，离线）
│   ├── lepod2mat.py                  数据格式转换器
│   ├── test_streaming.py            门控/转圈单测
│   ├── test_agent_core.py           编排单测
│   └── DK-2500_ECG-Agent_部署指南.md 本文件
├── checkpoints/
│   └── checkpoint_best.pt            分类工具权重 (1.1 GB)
├── ecg-dialogue-finetune/
│   └── Llama-3.2-3B-Instruct/
│       ├── adapter_config.json
│       └── adapter_model.safetensors  LoRA adapter (~170 MB)
├── ECG-Agent/
│   ├── src/                          ECG-Agent 源码（含修复）
│   └── fairseq-signals/              分类工具框架
└── ecg_agent_llama3b_q4km.gguf      (路径 B 需要，~2 GB)

~/workspace/
├── lepod_ecg.py                      Lepod 采集脚本（已有）
└── ecg_YYYYMMDD_HHMMSS.csv          采集输出
```

---

## 6. 性能预期

| 阶段 | 耗时（Core Ultra 5 225U） |
|------|--------------------------|
| 模型加载（首次，含下载） | 5–15 分钟 |
| 模型加载（已缓存） | 30–60 秒 |
| CSV → .mat 转换 | < 1 秒 |
| 分类工具推理（CPU） | 3–8 秒/条 |
| 测量工具推理（CPU） | 1–3 秒/条 |
| LLM 生成（路径 A, transformers） | 1–3 token/s |
| LLM 生成（路径 B, llama-cpp） | 3–8 token/s |
| 完整一次问答（含思考）| 约 30–90 秒 |

> 每次启动 `bedside_agent.py` 时，工具和 LLM 一次性加载，后续对话不再重新加载。

---

## 7. 常见问题

**Q: `bitsandbytes` 报 `CUDA not available`**  
A: Core Ultra 5 225U 无独立 GPU，bitsandbytes 4-bit 在纯 CPU 上支持有限。改用路径 B（GGUF）是最稳妥的方案。临时解决：在 `bedside_agent.py` 中将 `load_in_4bit=True` 改为 `torch_dtype=torch.float32`，但需要 ~12GB RAM，8GB 下会 OOM。

**Q: 运行 `bedside_agent.py` 时 OOM**  
A: 8GB RAM 紧张，先关掉所有其他程序（浏览器、桌面）。  
```bash
# 查看当前内存占用
free -h
# 可以加 --no-gui 参数让 Ubuntu 切换到无桌面模式
sudo systemctl set-default multi-user.target && reboot
```
重启后以命令行模式运行，省出约 1–1.5 GB。长期方案：加装 16GB DDR5 到 32GB。

**Q: 分类工具报 `PermissionError` 或 `no_pretrained_weights`**  
A: 已在 `classification.py` 修复（`model_overrides={"no_pretrained_weights": True}`）。如果还报错，确认 fairseq-signals 使用的是 `~/workspace/ECG-Agent/fairseq-signals/` 而不是系统全局安装版本。

**Q: Lepod 连接失败**  
A: 确认 Intel AX201NGW 蓝牙正常：
```bash
bluetoothctl show | grep Powered   # 应显示 yes
hciconfig hci0                     # 应显示 UP RUNNING
```
如果不行，重启蓝牙服务：
```bash
sudo systemctl restart bluetooth
```

**Q: CSV 转换后分类结果异常（全是 NORM 或 SR）**  
A: 检查 CSV 列名是否被正确识别。运行：
```bash
python lepod2mat.py ~/workspace/ecg_test.csv  # 会打印识别到的列名
```
如果列名不对，在 `lepod_ecg.py` 中确认 8 个通道的输出顺序（V6, I, II, V1, V2, V3, V4, V5），并在 `lepod2mat.py` 的 `load_lepod_csv()` 中手动指定正确的列名列表。

**Q: 测量工具报 `Too few R-peaks`**  
A: 可能原因：  
1. ECG 片段太短（< 10 秒）  
2. 信号质量差（Lepod 佩戴不良）  
3. Lead II 通道噪声过大  
尝试重新采集 30 秒以上的清晰 ECG。

**Q: 推理速度极慢（< 1 token/s）**  
A: 确认使用了全部 CPU 核心：
```bash
htop  # 推理时应看到多核利用率高
```
如果使用路径 B（llama-cpp），`n_threads=os.cpu_count()` 已设置。  
如果使用路径 A（transformers），加载时间长是正常现象，推理速度受 RAM 带宽限制。

---

## 8. 升级路线

当前为"先跑通"配置，后续可按需升级：

| 优先级 | 升级项 | 预期收益 |
|--------|--------|----------|
| ⭐⭐⭐ | RAM 升级至 32GB DDR5 | 可直接加载 float16 模型，推理更快 |
| ⭐⭐ | 改用 GGUF 路径 B | 内存降低，推理速度提升 2–3x |
| ⭐⭐ | OpenVINO NPU 加速 | 利用 Intel AI Boost 12 TOPS NPU，需模型转换 |
| ⭐ | 添加实时波形显示 | 基于 matplotlib 动画滚动 ECG 波形 |
| ⭐ | Web UI 界面 | Flask/FastAPI + 简单前端，替代终端对话 |

---

## 9. 与原 inference_ecg_dialogue.py 的关键差异

| 维度 | 原脚本 | bedside_agent.py |
|------|--------|-----------------|
| 数据来源 | HuggingFace ECG-MTD 数据集 | Lepod CSV / .mat 文件 |
| 工具输出 | 预计算 CSV 查表 | 现场实时推理 |
| 运行模式 | 批量评估（2174 样本） | 单次交互对话 |
| LLM 后端 | transformers float16 | transformers 4-bit 或 llama-cpp GGUF |
| 设备假设 | GPU（RTX 系列） | CPU（Core Ultra 5 225U） |
