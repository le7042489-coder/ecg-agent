"""可复用的床旁 ECG-Agent 核心（UI 无关）。

CLI(`bedside_agent.py`) 与网页(`web_server.py`) 共用同一份「加载一次 → 提问 → 流式应答」
逻辑。重活（工具加载、预算、生成、门控）全部复用 bedside_agent 里的函数/类，本模块只做
编排：把结构化输出 (Action/Thought/Content + 工具路由) 串成一个产出「应答正文」的生成器。

注：bedside_agent.run_bedside_session 目前仍保留自己的同款编排循环（CLI 历史路径）。
两者的工具路由逻辑相同，后续可让 CLI 也改用本类；改时两处一起动。
"""

import json
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
                 guideline=False, guideline_index=None, force_action=None):
        self.mat_path = mat_path
        self.backend = backend
        self.force_action = force_action
        self._lock = threading.Lock()
        self.last_action = ""
        self.last_content = ""

        # 1. 工具
        self.classifier, self.analyzer, self.morphology, self.signal_quality = \
            B.load_tools(str(B.CHECKPOINT_PATH))

        # 1b. 可选指南检索器
        self._gretr = B.load_guideline_retriever(guideline_index) if guideline else None

        # 2. 预算工具输出（缓存）
        print("[core] 预计算工具输出...")
        self.cached_classification = B.run_classification(self.classifier, mat_path)
        self.cached_measurement = B.run_measurement(self.analyzer, mat_path)
        self.cached_morphology = B.run_morphology(self.morphology, mat_path)
        self.cached_signal_quality = B.run_signal_quality(self.signal_quality, mat_path)
        self.cached_guideline = (
            B.guideline_context(self._gretr, self.cached_classification)
            if self._gretr is not None else ""
        )

        # 3. LLM + 后端 token 迭代器
        print("[core] 加载 LLM...")
        if backend == "llama-cpp":
            if not gguf:
                raise ValueError("backend=llama-cpp 需要 gguf 路径")
            self.llm = B.load_llm_llamacpp(gguf)
            self._iter = lambda msgs: B.iter_llamacpp(self.llm, msgs)
        else:
            self.model, self.tokenizer, self.gen_config = \
                B.load_llm_transformers(base_model, str(B.ADAPTER_PATH))
            self._iter = lambda msgs: B.iter_transformers(
                self.model, self.tokenizer, self.gen_config, msgs)

        # 4. 对话历史
        self.messages = [{"role": "system", "content": B.SYSTEM_PROMPT}]

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

    # ── 提问：产出「给用户看的应答正文」流 ────────────────────────────────────

    def ask_stream(self, question: str):
        """逐片 yield 应答正文（已过滤 Action/Thought）。串行化（锁）以保护模型与历史。"""
        with self._lock:
            self.messages.append({"role": "user", "content": question})

            # gen-1：决定 action（直接应答会在此产出正文）
            gate1 = B.ContentGate()
            emit1 = self.force_action is None  # force-action 调试：首次静默，避免与二次重复
            for tok in self._iter(self.messages):
                chunk = gate1.feed(tok)
                if chunk and emit1:
                    yield chunk
            raw = gate1.full_text()
            parsed = B.parse_response(raw)
            action = self.force_action or parsed["action"]
            self.last_action = action

            if action in TOOL_ACTIONS:
                tool_output = {
                    "call_classification_tool": self.cached_classification,
                    "call_measurement_tool": self.cached_measurement,
                    "call_morphology_tool": self.cached_morphology,
                    "call_signal_quality_tool": self.cached_signal_quality,
                }[action]
                if action == "call_classification_tool" and self.cached_guideline:
                    tool_output = f"{tool_output}\n{self.cached_guideline}"
                tool_turn = {
                    "role": "assistant", "action": action,
                    "thought": parsed["thought"], "tool_output": tool_output,
                }
                self.messages.append({"role": "assistant",
                                      "content": B.format_assistant_turn(tool_turn)})

                # gen-2：读工具结果作答（流式产出正文）
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
            else:
                direct_turn = {
                    "role": "assistant", "action": action,
                    "thought": parsed["thought"], "content": parsed["content"] or raw,
                }
                self.messages.append({"role": "assistant",
                                      "content": B.format_assistant_turn(direct_turn)})
                self.last_content = direct_turn["content"]
                if not gate1.started and emit1 and direct_turn["content"]:
                    yield direct_turn["content"]
