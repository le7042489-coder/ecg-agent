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


def make_agent(iter_fn, force_action=None, fast_route=False):
    """绕过重型 __init__，手工装配一个最小可测的 agent。

    fast_route 默认 False：让历史用例(1/2/4)仍走 gen-1 两趟编排；路由用例显式开 True。
    """
    a = AC.BedsideAgent.__new__(AC.BedsideAgent)
    a._lock = threading.Lock()
    a.force_action = force_action
    a.fast_route = fast_route
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
    def _it(_messages, **_kw):  # 吞掉 stop / max_tokens 等后端参数
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

    # 3) force-action：跳过 gen-1，直接用强制工具的缓存输出走 gen-2（只一趟生成）
    a = make_agent(scripted(
        "Action: response\nThought: t2\nContent: forced answer.",
    ), force_action="call_measurement_tool")
    out = "".join(a.ask_stream("q"))
    check("3 force_action skips gen1", out == "forced answer.", repr(out))
    check("3 force_action history", len(a.messages) == 4, len(a.messages))  # sys+user+tool+response
    check("3 force_action tool cached", '"heart_rate": 72' in a.messages[2]["content"], a.messages[2]["content"])

    # 4) 直接路径缺 Content 的回退：门控没产出 → 回退把整段当正文吐出
    a = make_agent(scripted("Action: response\nThought: t\n(no content marker here)"))
    out = "".join(a.ask_stream("q"))
    check("4 direct fallback when no Content", out.strip().endswith("(no content marker here)")
          or "no content marker" in out, repr(out))

    # 5) 快速路由：问心率→关键词命中→跳过 gen-1，首个脚本即 gen-2 作答（只一趟）
    a = make_agent(scripted(
        "Action: response\nThought: t\nContent: Your HR is 72 bpm.",
    ), fast_route=True)
    out = "".join(a.ask_stream("我的心率是多少？"))
    check("5 route skips gen1", out == "Your HR is 72 bpm.", repr(out))
    check("5 route history", len(a.messages) == 4, len(a.messages))  # sys+user+tool+response
    check("5 route picks measurement", "heart_rate" in a.messages[2]["content"], a.messages[2]["content"])

    # 6) 预后类提问（命中 OOS 提示）→ 不路由，回退 gen-1，让模型自己选 response_fail
    a = make_agent(scripted(
        "Action: response_fail\nThought: oos\nContent: 这类问题建议咨询医生。",
    ), fast_route=True)
    out = "".join(a.ask_stream("这个房颤严重吗？"))
    check("6 oos falls back to gen1", out == "这类问题建议咨询医生。", repr(out))
    check("6 oos action", a.last_action == "response_fail", repr(a.last_action))
    check("6 oos history", len(a.messages) == 3, len(a.messages))  # sys+user+assistant(direct)

    print()
    print("ALL PASS" if not _fails else f"{len(_fails)} FAILED: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
