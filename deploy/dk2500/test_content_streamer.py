"""ContentStreamer + Spinner 回归测试（不依赖模型/权重）。

运行： python test_content_streamer.py

顶层只是 import numpy/torch（函数内部才真正用到），测试里 stub 成空模块即可
导入真实的 ContentStreamer / Spinner / parse_response，无需任何 ML 依赖或权重。

覆盖：只显示 Content: 之后的正文、屏蔽 Action/Thought、tool-call 整段静默、
跨 token 的标记/动作名拼接、--force-action 静默（enabled=False）、前导空白、
与 parse_response 一致、逐字符流、on_first 钩子、Spinner（tty 门控/清行/幂等）。
"""
import io
import os
import sys
import time
import types

# 顶层只 import numpy/torch（函数内部才用），stub 成空模块即可导入本模块
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bedside_agent as B

PREFIX = "ECG-Agent: "


class FakeTTY:
    """可控 isatty 的输出流，用于测试 Spinner 的终端门控与清行。"""
    def __init__(self, isatty=True):
        self._isatty = isatty
        self._buf = []
    def write(self, s):
        self._buf.append(s)
    def flush(self):
        pass
    def isatty(self):
        return self._isatty
    def getvalue(self):
        return "".join(self._buf)


def run(tokens, enabled=True, on_first=None):
    buf = io.StringIO()
    cs = B.ContentStreamer(prefix=PREFIX, enabled=enabled, out=buf, on_first=on_first)
    for t in tokens:
        cs.feed(t)
    cs.finish()
    return buf.getvalue(), cs.started


def body(out):
    """去掉前缀与收尾换行，取真正显示给用户的正文。"""
    return out.replace(PREFIX, "", 1).rstrip("\n")


_fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <<< {extra}"))
    if not cond:
        _fails.append(name)


def main():
    # 1) 直接应答：流式显示 Content，屏蔽 Action/Thought
    out, shown = run(["Action", ": ", "response", "\n", "Thought", ": reasoning here", "\n",
                      "Content", ": ", "Your ", "heart ", "rate ", "is ", "72 bpm."])
    check("1 direct: shown", shown)
    check("1 direct: body", body(out) == "Your heart rate is 72 bpm.", repr(out))
    check("1 direct: no scaffold",
          ("Thought" not in out) and ("Action" not in out) and ("reasoning" not in out), repr(out))
    check("1 direct: prefix once", out.count("ECG-Agent:") == 1, repr(out))

    # 2) 工具调用（gen-1）：无 Content，整段静默
    out, shown = run(["Action", ": ", "call_classification_tool", "\n", "Thought", ": classify", "\n",
                      "Tool_Output", ": <ext>"])
    check("2 toolcall: silent", out == "" and not shown, repr(out))

    # 3) 工具 action 名跨 token 拼接，仍判定为 tool → 静默
    out, shown = run(["Action: ", "call", "_classification", "_tool", "\nThought: x\n"])
    check("3 toolname split: silent", out == "" and not shown, repr(out))

    # 4) Content: 标记跨 token 拼接
    out, shown = run(["Action: response\n", "Cont", "ent", ":", " hi", " there"])
    check("4 marker split", shown and body(out) == "hi there", repr(out))

    # 5) enabled=False（--force-action 首次生成）：全程静默
    out, shown = run(["Action: response\nContent: should not show"], enabled=False)
    check("5 disabled: silent", out == "" and not shown, repr(out))

    # 6) Content: 后前导换行/空格被跳过，前缀只在首个非空字符前出现
    out, shown = run(["Action: response\nContent:", "\n", "  ", "Hello", " world"])
    check("6 leading ws", shown and out == "ECG-Agent: Hello world\n\n", repr(out))

    # 7) system_bye 也是直接应答，应显示
    out, shown = run(["Action: system_bye\nContent: 感谢使用，再见！"])
    check("7 system_bye", shown and body(out) == "感谢使用，再见！", repr(out))

    # 8) response_fail（名字含 response 前缀，不能误判为 tool）
    out, shown = run(["Action: response_fail\nThought: oos\nContent: 这超出我的能力范围。"])
    check("8 response_fail shown", shown and body(out) == "这超出我的能力范围。", repr(out))

    # 9) 与 parse_response 一致：整段一次喂入，显示正文 == parse_response 的 content
    full = "Action: response\nThought: t\nContent: 你的心率为 72 次/分，属正常范围。"
    out, shown = run([full])
    check("9 consistency", body(out) == B.parse_response(full)["content"],
          repr((body(out), B.parse_response(full)["content"])))

    # 10) 逐字符喂入（最严格的跨边界测试）
    full10 = "Action: response\nThought: reasoning\nContent: 一切正常，无需担心。"
    out, shown = run(list(full10))
    check("10 char-by-char", shown and body(out) == "一切正常，无需担心。", repr(out))

    # 11) on_first：有内容时恰好回调一次，且在打印前缀前
    calls = []
    out, shown = run(["Action: response\nContent: hi"], on_first=lambda: calls.append(1))
    check("11 on_first once", shown and calls == [1], repr((out, calls)))

    # 12) on_first：工具调用静默时不回调
    calls = []
    out, shown = run(["Action: call_measurement_tool\nThought: m\n"], on_first=lambda: calls.append(1))
    check("12 on_first not called when suppressed", (not shown) and calls == [], repr((out, calls)))

    # 13) Spinner：tty 下会画转圈、含消息、stop 清行且幂等
    ft = FakeTTY(isatty=True)
    sp = B.Spinner("正在分析…", interval=0.01, out=ft)
    sp.start(); time.sleep(0.05); sp.stop()
    v = ft.getvalue()
    check("13 spinner draws msg", "正在分析…" in v and "\r" in v, repr(v[:60]))
    check("13 spinner clears line", v.endswith("\r\033[K"), repr(v[-10:]))
    sp.stop()  # 幂等：二次 stop 不抛错、不重复清行
    check("13 spinner stop idempotent", ft.getvalue() == v, "二次 stop 不应再写入")

    # 14) Spinner：非 tty（管道/重定向）整体 no-op，不污染输出
    ft2 = FakeTTY(isatty=False)
    sp2 = B.Spinner("x", interval=0.01, out=ft2)
    sp2.start(); time.sleep(0.03); sp2.stop()
    check("14 spinner noop when not tty", ft2.getvalue() == "", repr(ft2.getvalue()))

    print()
    print("ALL PASS" if not _fails else f"{len(_fails)} FAILED: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
