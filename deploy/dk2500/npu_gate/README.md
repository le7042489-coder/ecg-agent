# NPU 异常门控 + 滚动窗口（DK-2500 床旁）

NPU 常驻的「正常/异常」异常门控：用 full-spec TSRNet（OpenVINO，跑在 Intel NPU 上）给 10s/12 导
ECG 窗打异常分，**滚动窗口**做持久判据，检出持续异常时自动唤醒床旁 Agent（`web_server.py` /
`bedside_agent.py`）。门控本身轻量常驻、低功耗，正中 NPU「小模型 + 常驻」本命。

## 文件

| 文件 | 作用 |
|------|------|
| `rolling_gate.py` | 滚动窗口决策核（纯 numpy，torch/OpenVINO 无关）：`slide_windows`（连续录制→重叠 10s 窗）+ `RollingGate`（中值平滑 → M-of-N 持久判据 → 不应期/迟滞状态机）+ `run_stream` |
| `ecg_gate_npu.py` | 生产门控：NPU 前向（OpenVINO）+ numpy 打分 + 滚动窗口决策 + **唤醒真 Agent**。torch-free |
| `test_rolling_gate.py` | 决策核单测（7 项，纯 numpy，无需权重/NPU）|

> **模型权重不在本仓库。** `ecg_gate_npu.py` 需要外部提供的 `tsrnet_spec.onnx`（TSRNet full-spec 导出）
> 和一份校准过的阈值 JSON。TSRNet（ISBI 2024, UARK-AICV）**仓库无 license**，仅供研究/比赛使用并引用，
> **请勿二次分发其模型代码或权重**。本目录只含我方编写的门控/滚动窗口代码（不含 TSRNet 模型源码）。

## 滚动窗口设计

把门控从「离散 10s 块 + N 连续去抖」升级为**流式滚动窗口**：

1. **信号级滑窗** `slide_windows`：连续录制切成**重叠** 10s 窗，步长 `--hop`（默认 2.5s / 75% 重叠）。
   等价于设备上维护 10s 环形缓冲、每 2.5s 重打一次分 → 检测延迟低、每个异常事件投票更多。
   `--mat` 现在滑过**整段**录制（旧版只看前 10s）。
2. **决策级滚动窗** `RollingGate`：
   - 可选**中值平滑**（`--smooth`）——压掉单窗运动伪迹尖峰；
   - **M-of-N 持久判据**（`--m`/`--n`，默认 3/5）——最近 N 窗里 ≥M 窗异常才唤醒，容忍持续异常中夹
     一个干净窗（重叠采样下常见），比「连续 N 窗」稳；
   - **不应期 + 迟滞**（`--refractory`，默认 30s）——醒后需时间到且 buffer 沉静才重新布防，
     一次异常事件只唤醒一次，不刷屏 Agent。
   - `--debounce d` 向后兼容 = `--m d --n d`。

为何重要：Lepod 正常分均值 ~0.22、阈值 ~0.28，余量很窄，偶发正常窗会越界；平滑 + M-of-N 压掉这些
误唤醒，重叠又保证真异常不漏。

## 唤醒接线

检出持续异常时，门控把**触发的那个 10s 窗**写成 `lepod2mat.py` 兼容的 `.mat`（`feats (12,5000)` mV +
`curr_sample_rate`，分类工具两者都要）。两种把它交给 Agent 的方式：

**A. 网页常开、门控推报警（推荐，`--notify-url`）**——`web_server.py` 常驻当床旁界面；门控每窗 POST
实时态到 `/api/gate`（页面顶栏「门控 ● 监测中 score/阈」实时跳），检出异常时 POST 报警，服务端把
agent **热切到异常窗**（重算工具缓存、不重载 LLM）并经 SSE 让页面弹报警横幅 + 波形/findings 自动切到
该窗：

```bash
# 终端1：常驻网页（先起一个初始 .mat）
python ../web_server.py --mat init.mat --backend llama-cpp --gguf $G --host 0.0.0.0 --port 8000
# 终端2：门控滑窗 + 推送
python ecg_gate_npu.py --onnx tsrnet_spec.onnx --thr gate_thr_lepod.json --mat ecg.mat \
    --hop 2.5 --m 3 --n 5 --notify-url http://127.0.0.1:8000
```

**B. 平时无界面、异常时拉起一个新进程（`--wake`）**：

- `--wake web` → `web_server.py`（网页界面）
- `--wake cli` → `bedside_agent.py`（CLI）
- `--wake-cmd "...{mat}..."` → 自定义命令（`{mat}` = 保存的异常窗）
- `--wake-dry` → 只打印命令、不启动（验证接线用）

启动是**完全 detached**（独立会话 + 输出落日志），门控作为常驻 daemon 不被启动的 Agent 拖住。
`--deploy-dir`/`--gguf`/`--host`/`--port` 覆盖启动细节。

> 前端接入（A）改动：`web_server.py` 加 `POST /api/gate`（收门控事件）+ `GET /api/gate/stream`（SSE
> 广播给浏览器）；`agent_core.py` 加 `load_mat()` 热切；`web/index.html` 加门控状态栏 + 报警横幅 + 自动
> 刷新。设备 E2E 实测：91 条 status + 2 条 wake/ecg_ready，`/api/ecg` 热切到异常窗（分类/测量/波形齐全，
> LLM 不重载）。plumbing 冒烟见 `../test_gate_web.py`。

## 用法

```bash
# 设备上必须先激活 env（NPU 需要 ZE_ENABLE_ALT_DRIVERS，已写进 ecg-bedside env vars）
conda activate ecg-bedside

# 1) 用目标域正常样本校准阈值（建议 Lepod 正常录制；on-device、跑 NPU）
python ecg_gate_npu.py --onnx tsrnet_spec.onnx --calibrate lepod_normals.npy --thr-out gate_thr_lepod.json

# 2) 对真 Lepod .mat 滑窗常驻，检出持续异常自动唤醒网页 Agent
python ecg_gate_npu.py --onnx tsrnet_spec.onnx --thr gate_thr_lepod.json --mat ecg.mat \
    --hop 2.5 --m 3 --n 5 --refractory 30 --wake web

# 回放/验证带标签集（m=n=2 == 旧「2 连续」去抖）
python ecg_gate_npu.py --onnx tsrnet_spec.onnx --thr gate_thr_lepod.json \
    --npy windows.npy --labels labels.npy --m 2 --n 2

# 决策核单测（纯 numpy）
python test_rolling_gate.py
```

## 设备实测（DK-2500, full-spec TSRNet @ Intel NPU）

- 决策核 7 项单测在设备上全过。
- **90 个真 Lepod 正常窗**（阈值 0.284）：朴素 per-window 阈值会误唤醒 **2** 次（孤立越界窗），
  滚动 M-of-N（3/5）→ 误唤醒 **0**。滚动窗口把残余误报压到 0。
- `--mat` 滑窗：240s 连续 → **93 窗 / 4.33s**（≈47ms/窗 e2e，常驻 hop 2.5s 下 <2% 占空）。
- 唤醒接线端到端：门控产出异常窗 `.mat` → 启动真 `web_server.py` → `/api/ecg` 返回该窗的
  分类 + 测量 + 形态 + 波形（HTTP 200）。
