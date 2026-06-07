"""DK-2500 床旁 ECG-Agent 网页界面（纯标准库 http.server，零额外依赖、可离线）。

设计目标：设备本机起服务，Firefox 打开 http://127.0.0.1:8000 即用；为后期加
麦克风/扩音器/显示器做准备（语音见前端 index.html 与下方 /api/stt、/api/tts 钩子）。

路由：
  GET  /              → 单页界面（web/index.html，自带 CSS/JS，无 CDN）
  GET  /api/ecg       → JSON：波形 + findings（分类/测量/形态学/信号质量/指南）
  GET  /api/ask?q=..  → SSE：逐片流式应答（text/event-stream）
  POST /api/stt       → 501 占位：后期接服务端语音识别（whisper.cpp 等；Firefox 无离线 STT）
  POST /api/tts       → 501 占位：后期接服务端语音合成（可选；前端已先用浏览器 speechSynthesis）

用法（与 bedside_agent.py 一致的数据源参数）：
  python web_server.py --mat ecg.mat --backend llama-cpp --gguf model.gguf
  python web_server.py --ecg ecg.csv --backend llama-cpp --gguf model.gguf --host 0.0.0.0 --port 8000
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).parent.resolve()
WEB_DIR = SCRIPT_DIR / "web"


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, agent):
        super().__init__(addr, handler)
        self.agent = agent


class Handler(BaseHTTPRequestHandler):
    server_version = "ECGAgentWeb/1.0"

    # 把默认那串噪声日志收敛成简短一行
    def log_message(self, fmt, *args):
        sys.stderr.write("[web] %s - %s\n" % (self.address_string(), fmt % args))

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body, content_type, code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── 路由 ──────────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/" or route == "/index.html":
            self._serve_index()
        elif route == "/api/ecg":
            self._serve_ecg()
        elif route == "/api/ask":
            q = (parse_qs(parsed.query).get("q") or [""])[0].strip()
            self._serve_ask(q)
        elif route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, code=404)

    def do_POST(self):
        route = urlparse(self.path).path
        # 语音占位：实现前先明确告知未启用，前端据此优雅降级
        if route == "/api/stt":
            self._send_json({"error": "STT 未启用", "hint": "后期接服务端语音识别（如 whisper.cpp）；"
                             "Firefox 无离线浏览器 STT，故走服务端"}, code=501)
        elif route == "/api/tts":
            self._send_json({"error": "服务端 TTS 未启用", "hint": "前端当前用浏览器 speechSynthesis 朗读；"
                             "如需服务端合成（piper 等）后期在此实现"}, code=501)
        else:
            self._send_json({"error": "not found"}, code=404)

    # ── 处理器 ────────────────────────────────────────────────────────────────

    def _serve_index(self):
        index = WEB_DIR / "index.html"
        try:
            body = index.read_bytes()
        except FileNotFoundError:
            self._send_json({"error": f"缺少 {index}"}, code=500)
            return
        self._send_bytes(body, "text/html; charset=utf-8")

    def _serve_ecg(self):
        agent = self.server.agent
        try:
            data = {"summary": agent.summary(), "waveform": agent.waveform(),
                    "ecg_file": os.path.basename(agent.mat_path)}
        except Exception as e:
            self._send_json({"error": f"取 ECG 数据失败: {e}"}, code=500)
            return
        self._send_json(data)

    def _serve_ask(self, q):
        if not q:
            self._send_json({"error": "缺少问题参数 q"}, code=400)
            return
        agent = self.server.agent
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")  # 关掉可能的反代缓冲
        self.end_headers()

        # 阶段提示：首字到达前推 {"status": ...}，前端据此更新「正在…」文案而非冻屏。
        # status_cb 在生成器内同线程被调用，与下面 _sse 写同一 wfile，无并发问题。
        gen = agent.ask_stream(q, status_cb=lambda s: self._sse({"status": s}))
        try:
            for chunk in gen:
                self._sse({"t": chunk})
            self._sse({"done": True, "action": agent.last_action})
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端断开
        except Exception as e:
            try:
                self._sse({"error": str(e)})
            except Exception:
                pass
        finally:
            gen.close()  # 触发 ask_stream 内 with-lock 释放

    def _sse(self, obj):
        payload = json.dumps(obj, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


def resolve_mat(args):
    """与 bedside_agent.py 一致的数据源解析：--mat / --ecg(转换) / --live(采集后转换)。"""
    import bedside_agent as B
    if args.mat:
        return args.mat
    if args.live:
        csv_path = B.acquire_live_ecg(args.live)
    else:
        csv_path = args.ecg
    from lepod2mat import convert as csv2mat
    return csv2mat(csv_path, src_rate=args.lepod_rate)


def main():
    parser = argparse.ArgumentParser(description="DK-2500 床旁 ECG-Agent 网页界面")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ecg", metavar="CSV", help="Lepod CSV 文件路径")
    src.add_argument("--mat", metavar="MAT", help="已转换的 .mat 文件路径")
    src.add_argument("--live", metavar="SEC", type=int, help="实时采集 N 秒")

    parser.add_argument("--backend", choices=["transformers", "llama-cpp"], default="llama-cpp")
    parser.add_argument("--base-model", default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--gguf", default=None, help="llama-cpp 后端：GGUF 路径")
    parser.add_argument("--lepod-rate", type=int, default=250)
    parser.add_argument("--guideline", action="store_true", help="启用指南自动 grounding（默认关）")
    parser.add_argument("--guideline-index", default=None)
    parser.add_argument("--no-fast-route", action="store_true",
                        help="关闭快速路由（默认开）：开启时用关键词直接判明工具、跳过 gen-1，缩短首 token。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机；外部访问用 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.backend == "llama-cpp" and not args.gguf:
        parser.error("--backend llama-cpp 需要 --gguf 指定 GGUF 文件路径")

    mat_path = resolve_mat(args)
    if not os.path.exists(mat_path):
        print(f"错误：.mat 文件不存在: {mat_path}")
        sys.exit(1)

    from agent_core import BedsideAgent
    print(f"[web] 加载 agent（数据源 {mat_path}）...")
    agent = BedsideAgent(
        mat_path, backend=args.backend, gguf=args.gguf, base_model=args.base_model,
        guideline=args.guideline, guideline_index=args.guideline_index,
        fast_route=not args.no_fast_route,
    )

    httpd = AgentHTTPServer((args.host, args.port), Handler, agent)
    print(f"[web] 就绪 → http://{args.host}:{args.port}  (Ctrl-C 退出)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 退出")
        httpd.shutdown()


if __name__ == "__main__":
    main()
