"""
DK-2500 床旁 ECG-Agent 交互入口

用法:
  # 分析已有 Lepod CSV
  python bedside_agent.py --ecg /path/to/ecg_20260527.csv

  # 分析已有 .mat 文件（跳过转换）
  python bedside_agent.py --mat /path/to/ecg.mat

  # 先采集 30s ECG，再分析（需要 lepod_ecg.py 在同目录）
  python bedside_agent.py --live 30

模型后端:
  --backend transformers   (默认，需要 ~4GB RAM，加载较慢)
  --backend llama-cpp      (推荐低内存场景，需要 GGUF 文件，加载快)
"""

import argparse
import itertools
import json
import os
import re
import sys
import subprocess
import textwrap
import threading
from pathlib import Path

import numpy as np
import torch

# ─── 路径配置 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent / "ECG-Agent"
SRC_DIR = REPO_ROOT / "src"
FAIRSEQ_DIR = SCRIPT_DIR.parent / "fairseq-signals"
CHECKPOINT_PATH = SCRIPT_DIR.parent / "checkpoints" / "checkpoint_best.pt"
ADAPTER_PATH = SCRIPT_DIR.parent / "ecg-dialogue-finetune" / "Llama-3.2-3B-Instruct"

# 向 sys.path 添加必要路径
for p in [str(SRC_DIR), str(FAIRSEQ_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ─── System Prompt（与 inference_ecg_dialogue.py 保持一致）─────────────────
SYSTEM_PROMPT = """Instruction:
You will be provided with part of a dialogue between the user and the system. The dialogue is conducted with a ECG wearable device user and the system, therefore the system should provide direct answers to the user's inquries without acting like a medical professional.
In this dialogue, the user is inquiring about an ECG reading, and the system will either respond directly or make appropriate tool calls to retrieve the necessary information to respond.
Your task is to generate appropriate thoughts, actions, and responses based on the dialogue history and the user's most recent utterance.

<Action list>
- call_classification_tool: Use this to identify arrhythmias, abnormalities, and findings from the provided ECG data
- call_measurement_tool: Use this to measure and output the heart rate, PR interval, QRS duration, and QTc interval.
- response: Provide comprehensive answer using results from tool outputs, combining technical findings with clinical interpretation
- response_fail: Indicate that the requested analysis cannot be performed due to tool limitations or because the question is not within the scope of the System and requires medical professional consultation.
- response_followup: Responds to the user's followup questions by providing additional information, clarification, or related insights about previous discussed ECG findings without requiring new tool calls.
- system_bye: Acknowledges the user's gratitude end the conversation politely

<General Rules>
1. Each generated dialogue (except for system_bye) is during the conversation with the user, therefore the responses should only address the user's last utterance and should not be too long.
2. Generated dialogue should be like a natural conversation between a user and ECG wearable assistant.

Based on the rules above, you will get an input as below:
Input:
<Dialogue history>
User: <User's last utterance> or Assistant: <Assistant's last Tool_Output>

And here is the format for what you should return in two different cases:

Case1. When a tool must be called based on the user's inquiry:
Action: <Assistant's chosen action, one of the tools>
Thought: <Assistant's reasoning process>
Tool_Output: <This will be provided externally.>

Case2. When providing a response based on previous tool or any other action not requiring a tool call:
Action: <Assistant's chosen action>
Thought: <Assistant's reasoning process>
Content: <Assistant's response>

Now, generate appropriate reasoning trace and responding message to user. Only generate in the proper format above, without indicating the chosen case."""


# ─── 模型加载 ─────────────────────────────────────────────────────────────────

def load_llm_transformers(base_model_path: str, adapter_path: str):
    """加载 LLM（transformers 后端）。

    8GB RAM 策略：尝试 4-bit 量化，失败则回退到 fp16（可能 OOM）。
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
    from peft import PeftModel

    print(f"[LLM] 加载 base model: {base_model_path}")
    print(f"[LLM] 加载 LoRA adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先尝试 4-bit 量化（~2GB RAM）
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("[LLM] 4-bit 量化加载成功")
    except Exception as e:
        print(f"[LLM] 4-bit 量化失败（{e}），回退到 float16 CPU 加载...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    return model, tokenizer, gen_config


def load_llm_llamacpp(gguf_path: str):
    """加载 LLM（llama-cpp 后端，推荐低内存场景）。"""
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python 未安装。运行: pip install llama-cpp-python"
        )

    print(f"[LLM] llama-cpp 加载 GGUF: {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_threads=os.cpu_count(),
        verbose=False,
    )
    return llm


def generate_transformers_stream(model, tokenizer, gen_config, messages: list, on_token=None) -> str:
    """流式生成（transformers 后端）。

    逐 token 产出：每段新文本调用 on_token(text)，同时累积并返回完整字符串
    （供 parse_response 解析 Action/Thought/Content）。
    """
    from transformers import TextIteratorStreamer
    from threading import Thread

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {**inputs, "generation_config": gen_config, "streamer": streamer}

    def _run_generate():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = Thread(target=_run_generate)
    thread.start()

    full = []
    for text in streamer:
        full.append(text)
        if on_token:
            on_token(text)
    thread.join()
    return "".join(full).strip()


def generate_llamacpp_stream(llm, messages: list, on_token=None) -> str:
    """流式生成（llama-cpp 后端）。逐 delta 产出并累积返回完整字符串。"""
    full = []
    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.0,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            full.append(delta)
            if on_token:
                on_token(delta)
    return "".join(full).strip()


def generate_transformers(model, tokenizer, gen_config, messages: list) -> str:
    """非流式封装：消费流式生成后返回完整字符串。"""
    return generate_transformers_stream(model, tokenizer, gen_config, messages, on_token=None)


def generate_llamacpp(llm, messages: list) -> str:
    """非流式封装：消费流式生成后返回完整字符串。"""
    return generate_llamacpp_stream(llm, messages, on_token=None)


# ─── ECG 工具（现算模式）────────────────────────────────────────────────────

def load_tools(checkpoint_path: str):
    """加载分类工具、测量工具、形态学工具和信号质量工具。"""
    from medrax.tools.classification import ECGClassifierTool, ECGAnalysisTool, ECGMorphologyTool
    from medrax.tools.signal_quality import ECGSignalQualityTool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Tools] 工具运行设备: {device}")

    print(f"[Tools] 加载分类模型: {checkpoint_path}")
    classifier = ECGClassifierTool(model_path=str(checkpoint_path), device=device)
    analyzer = ECGAnalysisTool(device=device)
    morphology = ECGMorphologyTool(device=device)
    signal_quality = ECGSignalQualityTool(device=device)
    return classifier, analyzer, morphology, signal_quality


def run_classification(classifier, mat_path: str) -> str:
    results, meta = classifier._run(mat_path)
    if "error" in results:
        return f"[分类失败: {results['error']}]"
    # 返回 top-5 类别（概率 > 0.3）
    sorted_r = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top = [f"{label}({prob:.2f})" for label, prob in sorted_r if prob > 0.05][:10]
    return str(top)


def run_measurement(analyzer, mat_path: str) -> str:
    result = analyzer._run(mat_path)
    if result.get("analysis_status") == "failed":
        return f"[测量失败: {result.get('note', '')}]"
    return json.dumps({
        "heart_rate": result.get("Heart_Rate"),
        "pr_interval": result.get("PR_Interval_ms"),
        "qrs_duration": result.get("QRS_Duration_ms"),
        "qtc_interval": result.get("QTc_ms"),
    })


def run_morphology(morphology, mat_path: str) -> str:
    result = morphology._run(mat_path)
    if result.get("analysis_status") == "failed":
        return f"[形态学分析失败: {result.get('note', '')}]"
    # 定量形态学（多导联 ST/T/R/S + QRS 电轴），紧凑 JSON 供 LLM 使用
    return json.dumps({
        "leads_used": result.get("leads_used"),
        "ST_deviation_mV": result.get("ST_deviation_mV"),
        "T_amplitude_mV": result.get("T_amplitude_mV"),
        "T_polarity": result.get("T_polarity"),
        "R_amplitude_mV": result.get("R_amplitude_mV"),
        "S_amplitude_mV": result.get("S_amplitude_mV"),
        "RS_ratio": result.get("RS_ratio"),
        "QRS_axis_deg": result.get("QRS_axis_deg"),
        "axis_interpretation": result.get("axis_interpretation"),
    }, ensure_ascii=False)


def run_signal_quality(signal_quality, mat_path: str) -> str:
    result = signal_quality._run(mat_path)
    if result.get("analysis_status") == "failed":
        return f"[信号质量分析失败: {result.get('note', '')}]"
    # 逐导记录质量汇总（整体 + 不合格导联 + 待查电极），紧凑 JSON 供 LLM 使用
    return json.dumps({
        "overall_quality": result.get("overall_quality"),
        "acceptable_lead_count": result.get("acceptable_lead_count"),
        "total_leads_assessed": result.get("total_leads_assessed"),
        "unacceptable_leads": result.get("unacceptable_leads"),
        "suspect_electrodes": result.get("suspect_electrodes"),
    }, ensure_ascii=False)


# ─── 指南检索（版本化，自动 grounding；可选特性）────────────────────────────

def load_guideline_retriever(index_dir=None):
    """加载版本化 ECG 指南检索器（可选）。

    优先用预构建索引（dense + RRF），否则回退到随仓库语料（纯 BM25，无需下载模型）。
    任何失败都不应中断床旁会话——返回 None 即静默关闭该特性。
    """
    try:
        from medrax.guideline import GuidelineRetriever

        idx = index_dir or os.environ.get("ECG_GUIDELINE_INDEX")
        if idx and Path(idx).is_dir():
            retr = GuidelineRetriever.from_index(idx, enable_dense=True)
            print(f"[指南] 已加载预构建索引: {idx}  (corpus={retr.corpus_version})")
        else:
            corpus = SRC_DIR / "medrax" / "guideline" / "corpus" / "ecg_guidelines_cn.jsonl"
            retr = GuidelineRetriever.from_corpus(corpus, enable_dense=False)
            print(f"[指南] 未提供索引，使用随仓库语料 BM25-only  (corpus={retr.corpus_version})")
            print("[指南] ⚠ 随仓库语料为 AI 起草、未经临床核实（verified=false），仅供比赛/打通流程，勿用于临床。")
        return retr
    except Exception as e:
        print(f"[指南] 加载失败（{e}），本次会话关闭指南 grounding。")
        return None


def guideline_context(retr, findings_text: str, k: int = 3) -> str:
    """按 ECG 发现检索相关指南，返回供 LLM 阅读的上下文块。

    无检索器 / 无发现 / 无命中 / 异常 → 返回空串（即不注入任何内容）。
    """
    if retr is None or not findings_text:
        return ""
    try:
        from medrax.guideline import ground_by_findings
        # findings_text 是分类器输出（英文 SCP 码）；映射成中文+按主题限定的 query，
        # 才能命中中文语料（否则跨语种零命中）。无可映射发现 → 空 → 不注入。
        hits = ground_by_findings(retr, findings_text, k=k)
    except Exception as e:
        print(f"[指南] 检索异常（{e}），跳过本轮 grounding。")
        return ""
    if not hits:
        return ""
    lines = [f"[相关指南 · corpus={retr.corpus_version}]"]
    for h in hits:
        r = h.record
        lines.append(
            f"- {r.text}（{r.society} {r.version}, 推荐类别 {r.recommendation_class}"
            f"/证据级别 {r.level_of_evidence}, {r.citation}）"
        )
        if h.change_note:
            lines.append(f"  注意: {h.change_note}")
    return "\n".join(lines)


def run_selftest(mat_path: str):
    """重训前的工具自检（--selftest）。

    不加载 LLM、不依赖分类 checkpoint，只验证免模型的 measurement + morphology +
    signal_quality 工具能跑通并产生格式正确的输出——即新工具的「工具 + 输出格式」是否就绪。
    分类工具与对话 dispatch 的端到端验证请在设备上用 --force-action 进行。
    """
    from medrax.tools.classification import ECGAnalysisTool, ECGMorphologyTool
    from medrax.tools.signal_quality import ECGSignalQualityTool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[selftest] 设备: {device}  文件: {mat_path}")
    analyzer = ECGAnalysisTool(device=device)
    morphology = ECGMorphologyTool(device=device)
    signal_quality = ECGSignalQualityTool(device=device)

    print("\n[selftest] call_measurement_tool →")
    print("  " + run_measurement(analyzer, mat_path))
    print("\n[selftest] call_morphology_tool →")
    print("  " + run_morphology(morphology, mat_path))
    print("\n[selftest] call_signal_quality_tool →")
    print("  " + run_signal_quality(signal_quality, mat_path))
    print("\n[selftest] ✅ 工具与输出格式正常（未加载 LLM / 分类模型）。")


# ─── 响应解析 ────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    action_m = re.search(r"(?im)^\s*\[?\s*Action\s*[:\-]\s*([^\n\r\]]+)", text)
    thought_m = re.search(
        r"^\s*\[?Thought:\s*(.*?)(?=\n\s*\[?(?:Action|Content|Tool_Output):|$)",
        text, re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    content_m = re.search(r"^\s*\[?Content:\s*(.*)", text, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    return {
        "action": action_m.group(1).strip() if action_m else "",
        "thought": thought_m.group(1).strip() if thought_m else "",
        "content": content_m.group(1).strip() if content_m else "",
    }


def format_assistant_turn(turn: dict) -> str:
    s = f"Action: {turn.get('action', '')}\nThought: {turn.get('thought', '')}\n"
    if "tool_output" in turn:
        s += f"Tool_Output: {turn['tool_output']}"
    elif "content" in turn:
        s += f"Content: {turn.get('content', '')}"
    return s.strip()


# ─── 流式显示：只把最终应答（Content: 之后）实时打印给用户 ────────────────────

class ContentStreamer:
    """在流式生成过程中，只把「给用户看的最终应答」实时打印出来。

    模型输出是结构化的 Action/Thought/Content 文本——只有 `Content:` 之后的内容
    才是给用户的回答。本类在 token 流上：
      - 跳过 Action / Thought 脚手架（不显示）；
      - 检测到 tool-call 类 Action 时整段静默（该次没有 Content，交给工具+二次生成）；
      - 越过 `Content:` 标记后逐 token 实时打印；首个非空字符出现时才打印前缀，
        避免在无内容时留下一个空的「ECG-Agent: 」。
    `enabled=False` 时全程静默（用于 --force-action 调试路径的首次生成）。

    只负责「显示」；完整原文仍由调用方交给 parse_response 解析，数据流不受影响。
    """

    _ACTION = re.compile(r"(?im)^[ \t]*\[?[ \t]*Action[ \t]*[:\-][ \t]*([A-Za-z_]+)")
    _CONTENT = re.compile(r"(?im)^[ \t]*\[?[ \t]*Content[ \t]*:[ \t]*")
    _TOOL_ACTIONS = {
        "call_classification_tool", "call_measurement_tool",
        "call_morphology_tool", "call_signal_quality_tool",
    }

    def __init__(self, prefix="ECG-Agent: ", enabled=True, out=None, on_first=None):
        self.prefix = prefix
        self.enabled = enabled
        self.out = out if out is not None else sys.stdout
        self.on_first = on_first  # 首个应答字符到达前调用（用于停掉等待转圈并清行）
        self.buf = ""
        self.in_content = False   # 已越过 Content: 标记，后续 token 直接打印
        self.started = False      # 已打印过至少一个非空字符（真正显示了内容）
        self.suppressed = False   # 检测到 tool-call，本段不显示

    def feed(self, token: str):
        if not self.enabled or self.suppressed:
            return
        if self.in_content:
            self._emit(token)
            return
        self.buf += token
        am = self._ACTION.search(self.buf)
        if am and am.group(1) in self._TOOL_ACTIONS:
            self.suppressed = True
            return
        cm = self._CONTENT.search(self.buf)
        if cm:
            self.in_content = True
            self._emit(self.buf[cm.end():])

    def _emit(self, text: str):
        if not self.started:
            text = text.lstrip()      # 跳过 Content: 后的前导空白/换行
            if not text:
                return
            if self.on_first is not None:
                self.on_first()       # 先停掉等待转圈并清行，再打印应答（避免争抢同一行）
            self.out.write(self.prefix)
            self.started = True
        self.out.write(text)
        self.out.flush()

    def finish(self):
        if self.started:
            self.out.write("\n\n")
            self.out.flush()


# ─── 等待动画：模型「思考/prompt-eval」静默期给用户反馈 ───────────────────────

class Spinner:
    """终端等待转圈。仅在交互式终端（isatty）显示，被管道/重定向时整体降级为
    no-op（不画、不清行），避免污染日志。

    start() 起一个后台线程持续刷新同一行；stop() 幂等——停止后台线程并清行。
    与 ContentStreamer 配合：应答首字到达前调用 stop()（join 掉后台线程后再清行），
    确保转圈线程先停，再由主线程打印应答，二者不会争抢同一行。
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message="正在分析…", interval=0.12, out=None):
        self.message = message
        self.interval = interval
        self.out = out if out is not None else sys.stdout
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _enabled(self):
        try:
            return bool(self.out.isatty())
        except Exception:
            return False

    def start(self):
        if not self._enabled():
            return
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

    def _spin(self):
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            self.out.write(f"\r{self.message} {frame} ")
            self.out.flush()
            self._stop.wait(self.interval)

    def stop(self):
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        self._stop.set()
        thread.join(timeout=1.0)
        if self._enabled():
            self.out.write("\r\033[K")  # 回到行首并清除到行尾
            self.out.flush()


# ─── 主对话循环 ───────────────────────────────────────────────────────────────

def run_bedside_session(mat_path: str, backend: str, args):
    # 1. 加载工具
    classifier, analyzer, morphology, signal_quality = load_tools(str(CHECKPOINT_PATH))

    # 1b.（可选）版本化指南检索器，用于基于分类发现的自动 grounding（默认关闭）
    gretr = (load_guideline_retriever(getattr(args, "guideline_index", None))
             if getattr(args, "guideline", False) else None)

    # 2. 预计算工具输出（避免对话过程中重复推理）
    print("\n[Agent] 预计算分类结果...")
    cached_classification = run_classification(classifier, mat_path)
    print(f"  分类结果: {cached_classification}")

    print("[Agent] 预计算测量结果...")
    cached_measurement = run_measurement(analyzer, mat_path)
    print(f"  测量结果: {cached_measurement}")

    print("[Agent] 预计算形态学结果...")
    cached_morphology = run_morphology(morphology, mat_path)
    print(f"  形态学结果: {cached_morphology}")

    print("[Agent] 预计算信号质量结果...")
    cached_signal_quality = run_signal_quality(signal_quality, mat_path)
    print(f"  信号质量结果: {cached_signal_quality}")

    # 基于分类发现预检索相关指南（仅当 --guideline 开启；空串表示不注入）
    cached_guideline = ""
    if gretr is not None:
        print("[Agent] 预检索相关指南（基于分类发现）...")
        cached_guideline = guideline_context(gretr, cached_classification)
        print(f"  指南上下文: {cached_guideline or '（无命中）'}")

    # 3. 加载 LLM
    print("\n[Agent] 加载 LLM（首次加载较慢，请稍候）...")
    if backend == "llama-cpp":
        llm = load_llm_llamacpp(args.gguf)
        stream_generate = lambda msgs, on_token: generate_llamacpp_stream(llm, msgs, on_token)
    else:
        model, tokenizer, gen_config = load_llm_transformers(
            args.base_model, str(ADAPTER_PATH)
        )
        stream_generate = lambda msgs, on_token: generate_transformers_stream(
            model, tokenizer, gen_config, msgs, on_token
        )

    def gen_streamed(msgs, live=True, spinner=None):
        """流式生成并实时显示最终应答；返回 (完整原文, 是否已显示内容)。

        live=False 时全程静默——仅用于 --force-action 调试路径的首次生成：
        该次输出可能被强制 action 覆盖，提前显示会与二次生成重复。
        spinner：传入则在首个应答字符到达前 stop()（清除等待转圈行）。
        """
        printer = ContentStreamer(prefix="ECG-Agent: ", enabled=live,
                                  on_first=(spinner.stop if spinner is not None else None))
        raw = stream_generate(msgs, printer.feed)
        printer.finish()
        return raw, printer.started

    # 4. 对话循环
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n" + "="*60)
    print("ECG-Agent 已就绪。输入问题开始对话，输入 'exit' 退出。")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "退出", "q"):
            print("ECG-Agent: 感谢使用，再见！")
            break

        messages.append({"role": "user", "content": user_input})

        # 本轮等待动画：从用户回车起转圈，应答首字到达时由 ContentStreamer 清除。
        # 工具问答跨 gen-1(静默)+工具+gen-2，spinner 一路转到答案首字；直接应答转到 gen-1 首字。
        print()  # 与上一轮之间留一空行
        spinner = Spinner("正在分析…")
        spinner.start()
        try:
            # LLM 第一次生成（决定 action）。直接应答（response 等）会在此流式显示；
            # 工具调用类 action 不含 Content:，本次静默，留待二次生成。
            # --force-action 时本次完全静默（live=False），避免被强制覆盖后重复输出。
            raw, shown = gen_streamed(messages, spinner=spinner,
                                      live=(getattr(args, "force_action", None) is None))
            parsed = parse_response(raw)
            action = parsed["action"]

            # 临时验证 hack（重训前）：强制 action，验证工具+接线+输出格式跑通。
            # 模型尚未在 call_morphology_tool 上训练，靠它自主选择不可靠，故先手动强制。
            # 重训并把新 action 写入 system prompt 后，删除此分支即可。
            if getattr(args, "force_action", None):
                spinner.stop()  # 调试信息要占整行，先停转圈
                print(f"[force-action] 强制 action = {args.force_action}（覆盖模型输出 '{action}'）")
                action = args.force_action

            if action in ("call_classification_tool", "call_measurement_tool",
                          "call_morphology_tool", "call_signal_quality_tool"):
                # 使用预计算结果
                tool_output = {
                    "call_classification_tool": cached_classification,
                    "call_measurement_tool": cached_measurement,
                    "call_morphology_tool": cached_morphology,
                    "call_signal_quality_tool": cached_signal_quality,
                }[action]
                # 自动 grounding：把相关指南并入分类工具输出，让应答 turn 一并阅读
                #（走 Tool_Output 通道，与微调训练格式一致；默认关闭，见 --guideline）
                if action == "call_classification_tool" and cached_guideline:
                    tool_output = f"{tool_output}\n{cached_guideline}"
                tool_turn = {
                    "role": "assistant",
                    "action": action,
                    "thought": parsed["thought"],
                    "tool_output": tool_output,
                }
                messages.append({"role": "assistant", "content": format_assistant_turn(tool_turn)})

                # LLM 第二次生成（基于工具输出作答）——流式显示给用户
                raw2, shown2 = gen_streamed(messages, live=True, spinner=spinner)
                parsed2 = parse_response(raw2)
                response_turn = {
                    "role": "assistant",
                    "action": parsed2.get("action", "response"),
                    "thought": parsed2.get("thought", ""),
                    "content": parsed2.get("content", raw2),
                }
                messages.append({"role": "assistant", "content": format_assistant_turn(response_turn)})
                # 流式已实时打印；仅当未显示（输出缺 Content: 等异常）才回退整段打印
                if not shown2:
                    spinner.stop()
                    print(f"ECG-Agent: {response_turn['content']}\n")
            else:
                # 直接回复（response / response_followup / system_bye / response_fail）
                direct_turn = {
                    "role": "assistant",
                    "action": action,
                    "thought": parsed["thought"],
                    "content": parsed.get("content", raw),
                }
                messages.append({"role": "assistant", "content": format_assistant_turn(direct_turn)})
                # 首次生成已流式打印；仅当未显示（被 --force-action 静默或缺 Content:）才回退
                if not shown:
                    spinner.stop()
                    print(f"ECG-Agent: {direct_turn['content']}\n")

                if action == "system_bye":
                    break
        finally:
            spinner.stop()  # 确保任何路径（含异常/break）都清除转圈


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def acquire_live_ecg(duration: int) -> str:
    """调用 lepod_ecg.py 采集实时 ECG，返回生成的 CSV 路径。"""
    lepod_script = Path.home() / "workspace" / "lepod_ecg.py"
    if not lepod_script.exists():
        raise FileNotFoundError(
            f"找不到采集脚本: {lepod_script}\n"
            "请确认 lepod_ecg.py 在 ~/workspace/ 目录下。"
        )
    print(f"[采集] 开始 {duration}s ECG 采集（Lepod BLE）...")
    result = subprocess.run(
        [sys.executable, str(lepod_script), str(duration)],
        cwd=str(lepod_script.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("lepod_ecg.py 采集失败")

    # 找到最新生成的 CSV
    workspace = lepod_script.parent
    csvs = sorted(workspace.glob("ecg_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError("采集完成但未找到 CSV 文件")
    return str(csvs[-1])


def main():
    parser = argparse.ArgumentParser(description="DK-2500 床旁 ECG-Agent")

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--ecg", metavar="CSV", help="Lepod CSV 文件路径")
    src_group.add_argument("--mat", metavar="MAT", help="已转换的 .mat 文件路径")
    src_group.add_argument("--live", metavar="SEC", type=int,
                           help="实时采集 N 秒（需要 Lepod 连接）")

    parser.add_argument("--backend", choices=["transformers", "llama-cpp"],
                        default="transformers", help="LLM 推理后端")
    parser.add_argument("--base-model",
                        default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
                        help="transformers 后端：base model 路径或 HF ID")
    parser.add_argument("--gguf", default=None,
                        help="llama-cpp 后端：GGUF 文件路径")
    parser.add_argument("--lepod-rate", type=int, default=250,
                        help="Lepod 采样率（默认 250Hz）")
    parser.add_argument("--force-action", default=None,
                        choices=["call_classification_tool", "call_measurement_tool",
                                 "call_morphology_tool", "call_signal_quality_tool"],
                        help="重训前验证 hack：强制每轮首个 action（绕过模型自主选择），"
                             "用于在花重训成本前跑通工具+接线+输出格式")
    parser.add_argument("--guideline", action="store_true",
                        help="启用指南自动 grounding：基于分类发现检索版本化指南，注入分类工具输出。"
                             "默认关闭——随仓库语料为开发占位内容（引用『占位-待核』），勿用于临床。")
    parser.add_argument("--guideline-index", default=None,
                        help="指南预构建索引目录（dense+RRF）；缺省读 ECG_GUIDELINE_INDEX 环境变量，"
                             "再缺省用随仓库占位语料 BM25-only。")
    parser.add_argument("--selftest", action="store_true",
                        help="工具自检：运行 measurement+morphology 并打印输出后退出"
                             "（不加载 LLM / 分类 checkpoint）")

    args = parser.parse_args()

    # 参数检验
    if args.backend == "llama-cpp" and not args.gguf:
        parser.error("--backend llama-cpp 需要 --gguf 指定 GGUF 文件路径")

    # 确定 .mat 文件
    if args.mat:
        mat_path = args.mat
    else:
        # 需要先获取 CSV
        if args.live:
            csv_path = acquire_live_ecg(args.live)
        else:
            csv_path = args.ecg

        # 转换 CSV → .mat
        from lepod2mat import convert as csv2mat
        mat_path = csv2mat(csv_path, src_rate=args.lepod_rate)

    if not os.path.exists(mat_path):
        print(f"错误：.mat 文件不存在: {mat_path}")
        sys.exit(1)

    # 重训前的工具自检：跑通新工具与输出格式，不加载 LLM / 分类模型
    if args.selftest:
        run_selftest(mat_path)
        return

    print(f"\n[Agent] ECG 文件: {mat_path}")
    run_bedside_session(mat_path, args.backend, args)


if __name__ == "__main__":
    main()
