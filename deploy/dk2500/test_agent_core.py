"""agent_core.BedsideAgent.ask_stream 编排测试（不依赖模型/权重）。

运行： python test_agent_core.py

stub 掉 numpy/torch/scipy，用 __new__ 绕过重型 __init__，注入一个「脚本化」的
token 迭代器，验证编排：直接应答 / 工具路径(gen-1静默+gen-2作答) / force-action /
缺 Content 的回退，以及对话历史增长正确。
"""
import os
import sys
import threading
import types

for name in ("numpy", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
_scipy = types.ModuleType("scipy"); _scipy_io = types.ModuleType("scipy.io")
_scipy.io = _scipy_io
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.io", _scipy_io)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent_core as AC

_fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <<< {extra}"))
    if not cond:
        _fails.append(name)


def make_agent(iter_fn, force_action=None):
    """绕过重型 __init__，手工装配一个最小可测的 agent。"""
    a = AC.BedsideAgent.__new__(AC.BedsideAgent)
    a._lock = threading.Lock()
    a.force_action = force_action
    a.last_action = ""
    a.last_content = ""
    a.cached_classification = "['AFIB(0.90)']"
    a.cached_measurement = '{"heart_rate": 72}'
    a.cached_morphology = "{}"
    a.cached_signal_quality = "{}"
    a.cached_guideline = ""
    a._iter = iter_fn
    a.messages = [{"role": "system", "content": "sys"}]
    return a


def scripted(*responses):
    """返回一个 _iter(messages)：第 k 次调用逐字符 yield responses[k]（超出则用最后一条）。"""
    state = {"i": 0}
    def _it(_messages):
        r = responses[min(state["i"], len(responses) - 1)]
        state["i"] += 1
        for ch in r:
            yield ch
    return _it


def main():
    # 1) 直接应答：gen-1 即产出正文
    a = make_agent(scripted("Action: response\nThought: t\nContent: Hello there."))
    out = "".join(a.ask_stream("hi"))
    check("1 direct yields content", out == "Hello there.", repr(out))
    check("1 direct last_content", a.last_content == "Hello there.", repr(a.last_content))
    check("1 direct history", len(a.messages) == 3, len(a.messages))  # sys+user+assistant

    # 2) 工具路径：gen-1 静默（工具调用无 Content），gen-2 作答
    a = make_agent(scripted(
        "Action: call_measurement_tool\nThought: m\nTool_Output: x",
        "Action: response\nThought: t2\nContent: Your HR is 72 bpm.",
    ))
    out = "".join(a.ask_stream("hr?"))
    check("2 tool yields gen2 only", out == "Your HR is 72 bpm.", repr(out))
    check("2 tool history", len(a.messages) == 4, len(a.messages))  # sys+user+tool+response
    check("2 tool last_action", a.last_action == "response", repr(a.last_action))
    # tool_turn 里应带上预算的测量结果
    check("2 tool_output cached", '"heart_rate": 72' in a.messages[2]["content"], a.messages[2]["content"])

    # 3) force-action：gen-1 内容被静默，强制走工具，仅 gen-2 流式
    a = make_agent(scripted(
        "Action: response\nThought: t\nContent: should-not-emit",
        "Action: response\nThought: t2\nContent: forced answer.",
    ), force_action="call_measurement_tool")
    out = "".join(a.ask_stream("q"))
    check("3 force_action streams gen2 only", out == "forced answer.", repr(out))

    # 4) 直接路径缺 Content 的回退：门控没产出 → 回退把整段当正文吐出
    a = make_agent(scripted("Action: response\nThought: t\n(no content marker here)"))
    out = "".join(a.ask_stream("q"))
    check("4 direct fallback when no Content", out.strip().endswith("(no content marker here)")
          or "no content marker" in out, repr(out))

    print()
    print("ALL PASS" if not _fails else f"{len(_fails)} FAILED: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
