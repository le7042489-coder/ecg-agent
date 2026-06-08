"""可复用的床旁 ECG-Agent 核心（UI 无关）。

CLI(`bedside_agent.py`) 与网页(`web_server.py`) 共用同一份「加载一次 → 提问 → 流式应答」
逻辑。重活（工具加载、预算、生成、门控）全部复用 bedside_agent 里的函数/类，本模块只做
编排：把结构化输出 (Action/Thought/Content + 工具路由) 串成一个产出「应答正文」的生成器。

注：bedside_agent.run_bedside_session 目前仍保留自己的同款编排循环（CLI 历史路径）。
两者的工具路由逻辑相同，后续可让 CLI 也改用本类；改时两处一起动。
"""

import json
import re
import threading

import numpy as np
import scipy.io

import bedside_agent as B

# 12 导联标准顺序（lepod2mat 写出的 feats 行顺序）
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

TOOL_ACTIONS = (
    "call_classification_tool", "call_measurement_tool",
    "call_morphology_tool", "call_signal_quality_tool",
)


def _parse_classification(cached: str):
    """run_classification 返回的是 Python list 字符串，如 "['PACE(0.49)', 'AFIB(0.29)']"。
    解析成 [{"label": "PACE", "prob": 0.49}, ...]；失败则原样塞进 note。"""
    import ast
    import re
    try:
        items = ast.literal_eval(cached)
        out = []
        for it in items:
            m = re.match(r"\s*([A-Za-z0-9_/+-]+)\s*\(([\d.]+)\)", str(it))
            if m:
                out.append({"label": m.group(1), "prob": float(m.group(2))})
        return out, None
    except Exception:
        return [], cached


class BedsideAgent:
    """加载一次（工具 + 预算 + 可选指南 + LLM），之后可多轮 ask_stream / 取 summary / waveform。"""

    def __init__(self, mat_path, *, backend="llama-cpp", gguf=None,
                 base_model="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
                 guideline=False, guideline_index=None, force_action=None,
                 fast_route=True):
        self.mat_path = mat_path
        self.backend = backend
        self.force_action = force_action
        self.fast_route = fast_route
        self._lock = threading.Lock()
        self.last_action = ""
        self.last_content = ""

        # 1. 工具
        self.classifier, self.analyzer, self.morphology, self.signal_quality = \
            B.load_tools(str(B.CHECKPOINT_PATH))

        # 1b. 可选指南检索器
        self._gretr = B.load_guideline_retriever(guideline_index) if guideline else None

        # 2. 预算工具输出（缓存），并打印一遍便于 CLI 启动时/服务端日志查看
        print("[core] 预计算工具输出...")
        self._budget(mat_path)

        # 3. LLM + 后端 token 迭代器（**kw 透传 max_tokens / stop）
        print("[core] 加载 LLM...")
        if backend == "llama-cpp":
            if not gguf:
                raise ValueError("backend=llama-cpp 需要 gguf 路径")
            self.llm = B.load_llm_llamacpp(gguf)
            self._iter = lambda msgs, **kw: B.iter_llamacpp(self.llm, msgs, **kw)
        else:
            self.model, self.tokenizer, self.gen_config = \
                B.load_llm_transformers(base_model, str(B.ADAPTER_PATH))
            self._iter = lambda msgs, **kw: B.iter_transformers(
                self.model, self.tokenizer, self.gen_config, msgs, **kw)

        # 3b. 预热：先把不变的系统提示喂进 KV 缓存 + 触发各种惰性初始化，
        #     这样首轮提问不必再冷 prefill 整段系统提示，显著缩短首轮首 token。
        print("[core] 预热模型（预填系统提示）...")
        try:
            warm = [{"role": "system", "content": B.SYSTEM_PROMPT},
                    {"role": "user", "content": "hi"}]
            for _ in self._iter(warm, max_tokens=1):
                break
        except Exception as e:
            print(f"[core] 预热跳过（{e}）")

        # 4. 对话历史
        self.messages = [{"role": "system", "content": B.SYSTEM_PROMPT}]

    # ── 数据源预算 / 热切 ─────────────────────────────────────────────────────

    def _budget(self, mat_path):
        """（重新）预算 4 个工具输出 + 可选指南，缓存给 summary()/gen-2。用已加载的工具，不碰 LLM。
        __init__ 与 load_mat 共用。"""
        self.mat_path = mat_path
        self.cached_classification = B.run_classification(self.classifier, mat_path)
        print(f"  分类: {self.cached_classification}")
        self.cached_measurement = B.run_measurement(self.analyzer, mat_path)
        print(f"  测量: {self.cached_measurement}")
        self.cached_morphology = B.run_morphology(self.morphology, mat_path)
        print(f"  形态学: {self.cached_morphology}")
        self.cached_signal_quality = B.run_signal_quality(self.signal_quality, mat_path)
        print(f"  信号质量: {self.cached_signal_quality}")
        self.cached_guideline = (B.guideline_context(self._gretr, self.cached_classification)
                                 if self._gretr is not None else "")
        if self.cached_guideline:
            print(f"  指南: {self.cached_guideline}")

    def load_mat(self, mat_path):
        """热切到新 .mat（如门控唤醒的异常窗）：重算工具缓存、重置对话历史；保留已载 LLM/工具。
        线程安全：与 ask_stream 共用 _lock 串行化（生成中则等其结束）。"""
        with self._lock:
            print(f"[core] 切换 ECG → {mat_path}")
            self._budget(mat_path)
            self.messages = [{"role": "system", "content": B.SYSTEM_PROMPT}]
            self.last_action = ""
            self.last_content = ""

    # ── 给 UI 的只读数据 ──────────────────────────────────────────────────────

    def summary(self) -> dict:
        """findings 面板用：分类 top 列表 + 测量/形态学/信号质量结构化结果 + 指南。"""
        cls, cls_note = _parse_classification(self.cached_classification)

        def _loads(s):
            try:
                return json.loads(s)
            except Exception:
                return {"note": s}

        return {
            "classification": cls,
            "classification_note": cls_note,
            "measurement": _loads(self.cached_measurement),
            "morphology": _loads(self.cached_morphology),
            "signal_quality": _loads(self.cached_signal_quality),
            "guideline": self.cached_guideline,
        }

    def waveform(self, max_points_per_lead: int = 1250) -> dict:
        """波形面板用：读 .mat 的 feats(12,N)，按需降采样后返回各导联数据。"""
        try:
            mat = scipy.io.loadmat(self.mat_path)
            feats = np.asarray(mat["feats"], dtype=float)        # (12, N) mV
            fs = float(np.ravel(mat.get("curr_sample_rate", [[500]]))[0])
        except Exception as e:
            return {"error": f"读取波形失败: {e}", "leads": []}

        if feats.ndim != 2 or feats.shape[0] < len(LEAD_NAMES):
            return {"error": f"feats 形状异常: {feats.shape}", "leads": []}

        n = feats.shape[1]
        step = max(1, n // max_points_per_lead)
        leads = []
        for i, name in enumerate(LEAD_NAMES):
            data = feats[i, ::step]
            leads.append({"name": name, "data": [round(float(v), 4) for v in data]})
        return {
            "fs": fs,
            "fs_display": fs / step,
            "duration_s": round(n / fs, 2),
            "n_points": int(np.ceil(n / step)),
            "leads": leads,
        }

    # ── 快速路由：能直接判明工具就跳过 gen-1（省一整趟 LLM，首 token 的主要来源）──
    _ROUTE_RULES = (
        ("call_measurement_tool",
         re.compile(r"心率|心跳|脉搏|\bHR\b|bpm|\bPR\b|\bQRS\b|\bQTc?\b|间期|多少下|快慢|节律快不快", re.I)),
        ("call_signal_quality_tool",
         re.compile(r"信号质量|信号|电极|导联.*(脱落|接触|质量)|脱落|干扰|噪声|noise|quality|lead.?off", re.I)),
        ("call_morphology_tool",
         re.compile(r"\bST\b|ST段|T波|T 波|电轴|\baxis\b|抬高|压低|波形态|形态", re.I)),
        ("call_classification_tool",
         re.compile(r"什么病|诊断|异常|心律失常|房颤|早搏|房早|室早|传导|arrhythmia|abnormal|finding|"
                    r"正常吗|有没有(问题|毛病)|怎么样|结果如何|分析一下", re.I)),
    )
    # 像在问预后/医疗建议（可能该 response_fail / 转诊）→ 不路由，交回模型判断
    _OOS_HINT = re.compile(r"严重吗|危险|要紧|会不会死|致命|能活|寿命|需要.*(看|找|挂).*医生|"
                           r"吃.*药|怎么治|治疗|手术|住院", re.I)
    _ROUTED_THOUGHT = "用户的问题需要调用该工具，从这段心电图取得量化结果后再作答。"
    _TOOL_LABEL = {
        "call_classification_tool": "分类", "call_measurement_tool": "测量",
        "call_morphology_tool": "形态学", "call_signal_quality_tool": "信号质量",
    }

    def _route(self, question: str):
        """无模型的意图路由：命中工具关键词→返回该 tool action；否则 None（回退 gen-1）。"""
        if self._OOS_HINT.search(question):
            return None
        for action, pat in self._ROUTE_RULES:
            if pat.search(question):
                return action
        return None

    def _tool_status(self, action: str) -> str:
        return f"正在结合{self._TOOL_LABEL.get(action, '分析')}结果作答…"

    # ── 提问：产出「给用户看的应答正文」流 ────────────────────────────────────

    def ask_stream(self, question: str, status_cb=None):
        """逐片 yield 应答正文（已过滤 Action/Thought）。串行化（锁）以保护模型与历史。

        status_cb(str): 可选阶段提示回调——在首字到达前给 UI 推进度文案（CLI 转圈/网页 SSE），
        缓解「首 token 等待期」的死等观感。真实延迟不变，但用户能看到在动。
        """
        def status(s):
            if status_cb:
                try:
                    status_cb(s)
                except Exception:
                    pass

        with self._lock:
            self.messages.append({"role": "user", "content": question})

            # 快速路由 / force-action：能直接判明工具就跳过 gen-1（省一整趟 LLM）
            routed = self.force_action or (
                self._route(question) if getattr(self, "fast_route", True) else None)
            if routed in TOOL_ACTIONS:
                self.last_action = routed
                status(self._tool_status(routed))
                yield from self._run_gen2(routed, self._ROUTED_THOUGHT)
                return

            # 回退：让模型自主决定 action（gen-1）。stop=Tool_Output: 让工具路径选完即刹车，
            # 不再白生成被门控丢弃的 Tool_Output/Content；直接应答路径不含该串、照常流式。
            status("正在分析问题…")
            gate1 = B.ContentGate()
            for tok in self._iter(self.messages, stop=["Tool_Output:"]):
                chunk = gate1.feed(tok)
                if chunk:
                    yield chunk
            parsed = B.parse_response(gate1.full_text())
            action = parsed["action"]
            self.last_action = action

            if action in TOOL_ACTIONS:
                status(self._tool_status(action))
                yield from self._run_gen2(action, parsed["thought"])
            else:
                direct_turn = {
                    "role": "assistant", "action": action,
                    "thought": parsed["thought"], "content": parsed["content"] or gate1.full_text(),
                }
                self.messages.append({"role": "assistant",
                                      "content": B.format_assistant_turn(direct_turn)})
                self.last_content = direct_turn["content"]
                if not gate1.started and direct_turn["content"]:
                    yield direct_turn["content"]

    def _run_gen2(self, action: str, thought: str):
        """已确定工具 → 构造工具轮（用预算好的缓存输出）→ 流式作答（gen-2）。

        routed（跳过 gen-1）与 fallback（gen-1 选了工具）两条路径共用此编排，单一事实源。
        """
        tool_output = {
            "call_classification_tool": self.cached_classification,
            "call_measurement_tool": self.cached_measurement,
            "call_morphology_tool": self.cached_morphology,
            "call_signal_quality_tool": self.cached_signal_quality,
        }[action]
        if action == "call_classification_tool" and self.cached_guideline:
            tool_output = f"{tool_output}\n{self.cached_guideline}"
        tool_turn = {"role": "assistant", "action": action,
                     "thought": thought, "tool_output": tool_output}
        self.messages.append({"role": "assistant",
                              "content": B.format_assistant_turn(tool_turn)})

        gate2 = B.ContentGate()
        for tok in self._iter(self.messages):
            chunk = gate2.feed(tok)
            if chunk:
                yield chunk
        raw2 = gate2.full_text()
        parsed2 = B.parse_response(raw2)
        response_turn = {
            "role": "assistant",
            "action": parsed2.get("action", "response"),
            "thought": parsed2.get("thought", ""),
            "content": parsed2["content"] or raw2,  # 缺 Content: 时回退整段
        }
        self.messages.append({"role": "assistant",
                              "content": B.format_assistant_turn(response_turn)})
        self.last_action = response_turn["action"]
        self.last_content = response_turn["content"]
        # 门控未产出任何正文（异常/缺 Content:）时，回退把整段正文吐出去
        if not gate2.started and response_turn["content"]:
            yield response_turn["content"]
