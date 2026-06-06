"""ContentGate + Spinner 回归测试（不依赖模型/权重）。

运行： python test_streaming.py

顶层只 import numpy/torch（函数内才真正用），stub 成空模块即可导入真实的
ContentGate / Spinner / parse_response，无需任何 ML 依赖或权重。

覆盖：门控只放行 Content: 之后正文、屏蔽 Action/Thought、tool-call 整段静默、
跨 token 的标记/动作名拼接、前导空白、与 parse_response 一致、逐字符流、full_text；
Spinner 的 tty 门控/清行/幂等。
"""
import os
import sys
import time
import types

sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bedside_agent as B


class FakeTTY:
    """可控 isatty 的输出流，用于测试 Spinner 的终端门控与清行。"""
    def __init__(self, isatty=True):
        self._isatty = isatty
        self._buf = []
    def write(self, s): self._buf.append(s)
    def flush(self): pass
    def isatty(self): return self._isatty
    def getvalue(self): return "".join(self._buf)


def gate(tokens):
    """喂入 token 序列，返回 (放行的正文拼接, gate 对象)。"""
    g = B.ContentGate()
    out = "".join(g.feed(t) for t in tokens)
    return out, g


_fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <<< {extra}"))
    if not cond:
        _fails.append(name)


def main():
    # 1) 直接应答：放行 Content 正文，屏蔽 Action/Thought
    out, g = gate(["Action", ": ", "response", "\n", "Thought", ": reason", "\n",
                   "Content", ": ", "Your ", "HR ", "is ", "72 bpm."])
    check("1 direct content", out == "Your HR is 72 bpm." and g.started, repr(out))
    check("1 no scaffold", ("Thought" not in out) and ("Action" not in out), repr(out))

    # 2) 工具调用：整段静默 + suppressed
    out, g = gate(["Action: call_classification_tool\n", "Thought: c\n", "Tool_Output: x"])
    check("2 toolcall silent", out == "" and g.suppressed and not g.started, repr(out))

    # 3) 工具 action 名跨 token 拼接，仍判定为 tool
    out, g = gate(["Action: ", "call", "_measurement", "_tool", "\nThought: x\n"])
    check("3 toolname split", out == "" and g.suppressed, repr(out))

    # 4) Content: 标记跨 token 拼接
    out, g = gate(["Action: response\n", "Cont", "ent", ":", " hi", " there"])
    check("4 marker split", out == "hi there" and g.started, repr(out))

    # 5) Content: 后前导换行/空格被跳过
    out, g = gate(["Action: response\nContent:", "\n", "  ", "Hello", " world"])
    check("5 leading ws", out == "Hello world", repr(out))

    # 6) system_bye 也放行正文
    out, g = gate(["Action: system_bye\nContent: 感谢使用，再见！"])
    check("6 system_bye", out == "感谢使用，再见！", repr(out))

    # 7) response_fail（含 response 前缀，不可误判为 tool）
    out, g = gate(["Action: response_fail\nThought: oos\nContent: 这超出我的能力范围。"])
    check("7 response_fail", out == "这超出我的能力范围。" and not g.suppressed, repr(out))

    # 8) 逐字符喂入（最严格的跨边界）
    full = "Action: response\nThought: r\nContent: 一切正常，无需担心。"
    out, g = gate(list(full))
    check("8 char-by-char", out == "一切正常，无需担心。", repr(out))

    # 9) full_text 完整
    check("9 full_text", g.full_text() == full, repr(g.full_text()))

    # 10) 与 parse_response 一致
    out, g = gate([full])
    check("10 consistency", out == B.parse_response(full)["content"],
          repr((out, B.parse_response(full)["content"])))

    # 11) Spinner：tty 下画转圈、含消息、stop 清行且幂等
    ft = FakeTTY(isatty=True)
    sp = B.Spinner("正在分析…", interval=0.01, out=ft)
    sp.start(); time.sleep(0.05); sp.stop()
    v = ft.getvalue()
    check("11 spinner draws", "正在分析…" in v and "\r" in v, repr(v[:50]))
    check("11 spinner clears", v.endswith("\r\033[K"), repr(v[-8:]))
    sp.stop()
    check("11 spinner idempotent", ft.getvalue() == v, "二次 stop 不应再写")

    # 12) Spinner：非 tty 整体 no-op
    ft2 = FakeTTY(isatty=False)
    sp2 = B.Spinner("x", interval=0.01, out=ft2)
    sp2.start(); time.sleep(0.03); sp2.stop()
    check("12 spinner noop non-tty", ft2.getvalue() == "", repr(ft2.getvalue()))

    print()
    print("ALL PASS" if not _fails else f"{len(_fails)} FAILED: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
