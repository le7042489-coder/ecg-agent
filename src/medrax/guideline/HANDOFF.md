xian# ECG Guideline RAG — 接手文档 / 项目状态

子项目：为 ECG-Agent 接入一个**版本化、仅 ECG**的指南检索工具。
本文件是给人看的"现状 + 怎么接手"；**构建/运行命令见同目录 `README.md`**。

## TL;DR
- 阶段：**M0 纵切完成**；**M0.5 的工程项本轮全部清掉**——代码已并入真仓库、补了测试套件、修了一个检索器降级 bug、床旁自动 grounding 已接入（**默认关闭**）。
- **稠密索引推迟到 M1**：当前语料仅 10 条占位记录，hybrid-vs-BM25 对比没有信号，且建索引要装 ~2GB 的 torch+sentence-transformers。等真语料到位再建（一次性验证 build→ship→load 管线）。
- 工程优先；发论文是既定目标，但推迟到工具端到端可用之后。
- 工作地点：**在笔记本开发，在 DK-2500 部署/验证**。

## 代码位置（重要 —— 先读这段，别踩坑）
工作区目录 `~/workspace/ECG-Agent/` **本身不是 git 仓库**，它是个同名工作区，里面套着真正的 git 仓库：

```
~/workspace/ECG-Agent/                 ← 工作区（非 git）
├── ECG-Agent/                         ← 真正的 git 仓库 (origin=gustmd0121, branch main)
│   └── src/medrax/
│       ├── guideline/                 ← 本子项目就在这里
│       └── tools/guideline_retrieval.py
└── dk2500_deploy/bedside_agent.py     ← 床旁入口，sys.path 指向 上面那个 ECG-Agent/src
```

- **所有相对路径（README 里的 `src/medrax/...`）都相对“真仓库根” `ECG-Agent/`**。在笔记本上跑命令前先 `cd ~/workspace/ECG-Agent/ECG-Agent`。
- 设备(DK-2500)上布局是**平的**：仓库直接在 `~/workspace/ECG-Agent/`，`dk2500_deploy/` 是它的兄弟目录。`bedside_agent.py` 用 `SCRIPT_DIR.parent/"ECG-Agent"/src` 解析 `medrax`，两台机器都能命中。
- 历史坑：本子项目最初被误建在**工作区顶层** `~/workspace/ECG-Agent/src/medrax/`（一个游离副本，床旁/部署都 import 不到它）。2026-06-04 已整体搬进真仓库并删除顶层副本。**别再往顶层 `src/` 放代码。**

## 两台机器的分工
| 机器 | 标识 | 角色 |
|---|---|---|
| 笔记本（开发机） | `yushanhui@DESKTOP-LI27D6M`，仓库在 `~/workspace/ECG-Agent/ECG-Agent` | 写代码、建稠密索引、扩语料、跑实验/基线 |
| DK-2500（设备） | hostname `dk2500`，Core Ultra 5 225U，7.3GB，纯 CPU，`192.168.137.2` | 部署+验证：接收构建好的 `index/`、测 RAM/延迟、床旁 `--guideline` 端到端、Lepod 采集 |

- 同步：两机之间用 `scp`（2026-06-04 已这么做），或自建 fork/remote。
- **不要 `git push origin`**：`origin = https://github.com/gustmd0121/ECG-Agent` 是上游作者(KAIST)的仓库，不是你的。
- 建索引是**离线**活（笔记本）；把 `index/` 传到设备，设备运行时只编码 query。零依赖的 BM25 路径已在真实 7.3GB 设备上验证（M0）。

## 当前状态
已建于 `src/medrax/guideline/`（+ `../tools/guideline_retrieval.py`）：
- schema / 分词 / hybrid 检索器（BM25 + 可选稠密 + RRF）/ 离线建索引 / CLI。
- 版本感知：默认只检索现行版 + 命中旧版时 forward-resolution。
- 每条结果带引用（学会/版本/Class/LOE）。
- LangChain `BaseTool` 封装，已在真仓库 `tools/__init__.py` 导出（`from .guideline_retrieval import *`）。

**本轮（2026-06-04）新增**：
- **搬进真仓库** —— 从工作区顶层游离副本整体迁移到 `ECG-Agent/src/medrax/`；`tools/__init__.py` 已接线；真仓库 `.gitignore` 已加 `/src/medrax/guideline/index/`。
- **测试套件** —— `guideline/tests/test_retriever.py`，**纯标准库 17 个用例全绿**（裸 python3，无需 pytest/numpy）：版本过滤、forward-resolution 提示、topic/society 过滤、BM25 相关性、schema 往返、分词、以及下面这个 bug 的回归测试。
  - 跑：`cd ECG-Agent && PYTHONPATH=src python -m unittest medrax.guideline.tests.test_retriever`
- **修了检索器降级 bug** —— `search(mode="dense")` 在无稠密索引（设备 BM25-only）或 embedder 加载失败时，原会返回 k 条 score=0 的任意结果并误标 `hybrid`；现已以 BM25 兜底（保留 sparse，只有显式要 dense 且 dense 真出了排序时才丢掉 sparse）。
- **床旁自动 grounding 已接入**（`dk2500_deploy/bedside_agent.py`，**默认关闭**）：
  - `--guideline` 开启；可选 `--guideline-index DIR` 或 `ECG_GUIDELINE_INDEX` 环境变量，缺省用随仓库占位语料 BM25-only。
  - 实现：基于预算好的分类发现检索现行版指南，**并入分类工具的 `Tool_Output`**（走训练时同一通道，对微调 3B 最低风险），应答 turn 一并阅读。
  - 任何加载/检索失败都静默关闭，不影响会话；空发现/无命中不注入。
  - **为何默认关**：占位语料引用是「占位-待核」，默认注入会让模型把占位内容讲给患者，违反“勿用于临床”。

**尚未完成**：
- **稠密索引未建（已知，推迟到 M1）** —— 见 TL;DR 理由。在真语料到位前，排序是 BM25-only（偏粗，已能用）。
- **语料是开发占位** —— `corpus/ecg_guidelines_cn.jsonl` 为改写/示意，引用是 `占位-待核`。真实策展 + 版权签字 = M1。
- **设备端验证未做** —— `index/` 还没 scp 到设备；RAM/延迟没测；真机上 `--guideline` 端到端（含微调 3B 的实际反应）没跑。床旁的 grounding 接线只在笔记本用 torch 桩 + 占位语料验证过函数级行为。

## 下一步
- **设备验证（DK-2500）**：真机上跑 `bedside_agent.py --guideline`（先用随仓库占位语料），确认接线/降级/输出格式 OK，量一下加载 BM25 的 RAM/延迟。
- **M1**：真实 CN 指南语料（≥3 组版本对）+ ECG grounding/QA 基准；版权签字；（可选）加英文。真语料到位后再用 bge-small-zh 建一次稠密索引，scp 到设备，确认 hybrid 优于 BM25-only。
- **findings→query 映射（M1 配套）**：分类工具吐的是英文/缩写标签，占位语料是中文；BM25 跨语种命中差。真语料阶段要把标签映射成检索 query（或用双语/跨语 embedder）。
- **论文（推迟）**：版本感知 + 引用忠实的指南 grounding（on-device ECG agent）+ 双语版本化基准；基线含 MedRAG / RAG²。详见项目笔记。

## 关键决策
- 工程优先；先中文单语（schema 留 `lang`）。
- 存储 v1 = 内存 + numpy + 纯 Python BM25（零硬依赖；jieba/sentence-transformers/numpy 有则自动用）。LanceDB/bm25s 推迟。
- 设备 embedder = bge-small-zh-v1.5（~95MB）或 int8 ONNX；稠密可选 + BM25 兜底（8GB 预算）。
- 轻量版本化：每条带 version + only_current；`effective_date`/`superseded_by` 预留给将来跨版本对比。
- **（2026-06-04）代码归位真仓库**：床旁/部署只 import `ECG-Agent/src` 这一棵树，且 README 的集成路径(`src/medrax/agent/agent.py`)也只在真仓库存在；顶层游离副本是死路，故迁移。
- **（2026-06-04）床旁 grounding 默认关 + 走 Tool_Output 通道**：占位语料不进临床回答；不引入训练时没见过的消息类型。
- **（2026-06-04）稠密索引推迟到真语料**：占位 10 条建索引低价值、且要装重依赖。

## 坑（务必注意）
- **两棵 `medrax` 树** —— 真仓库在嵌套的 `ECG-Agent/`，工作区顶层别再放 `src/`（见上“代码位置”）。
- `origin` 是上游 —— 别往那 push。
- 8GB 设备：GGUF(2GB)+工具+OS ≈4.5GB；检索要轻（小模型/int8 或 BM25-only）。
- `index/` 和 `__pycache__/` 是构建产物 —— 别提交（真仓库 `.gitignore` 已覆盖）。
- 语料是占位内容 —— 暂不可用于临床/再分发；床旁 `--guideline` 默认关也是这个原因。
