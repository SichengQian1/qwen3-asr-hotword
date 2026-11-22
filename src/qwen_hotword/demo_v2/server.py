"""One control session and a private recording URL."""

from __future__ import annotations

import argparse
import fcntl
import hmac
import json
import os
import secrets
import socket
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from qwen_hotword.demo.runtime import CHUNK
from qwen_hotword.demo_v2.session import SharedStudio
from qwen_hotword.training.stage_progress import stage_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--max-cases", default=150, type=int, help="每次自动验证的候选上限；0=仅麦克风"
    )
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    if args.max_cases < 0:
        parser.error("--max-cases 必须为非负整数")
    # Don't initialize CUDA and run a search only to discover an occupied port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.bind, args.port))
    root = Path.cwd()
    static = root / "examples/zh_rag_demo_v2"
    (root / "outputs").mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="zh_rag_demo_v2_", dir=root / "outputs"))
    token = secrets.token_urlsafe(24)
    lock = Lock()
    session = ""
    seen = 0.0
    print("本次输出：", out, flush=True)
    with stage_progress("验证资产，加载 GPU 3 检索与双路共享的语音解码模型"):
        demo = SharedStudio(root, out, 3, Path(os.environ["DEMO_DECODER_PY"]))
    try:
        if args.max_cases > 0:
            with (
                stage_progress("实测中文录屏样例（最多保留十条）"),
                (root / "outputs" / "zh_rag_v2_hunt.lock").open("a") as lockfile,
            ):
                fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
                demo.hunt(args.max_cases)
        demo.configure("", {})
        demo.init_studio()
        if demo.clips:
            demo.select(0)
        else:
            print("本次未验证出成功样例；麦克风仍可使用。", flush=True)

        (out / "startup.json").write_text(
            json.dumps(
                dict(
                    status="ready",
                    gpu=3,
                    checkpoint_sha256=demo.checkpoint,
                    epoch=25,
                    classes=183,
                    verified_cases=len(demo.clips),
                    checked=len(demo.checked),
                    case_limit=args.max_cases,
                    chunk_seconds=2,
                ),
                indent=2,
            )
        )

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(15)

            def send(self, data: Any, code: int = 200, mime: str = "application/json") -> None:
                body = (
                    data
                    if isinstance(data, bytes)
                    else json.dumps(data, ensure_ascii=False).encode()
                )
                self.send_response(code)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript"),
                    "/mic.js": ("mic.js", "text/javascript"),
                }
                if path in files:
                    name, mime = files[path]
                    self.send((static / name).read_bytes(), mime=mime)
                else:
                    self.send({}, 404)

            def do_POST(self) -> None:
                nonlocal session, seen
                if not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token
                ):
                    self.send({"error": "请打开终端输出的完整专属地址"}, 403)
                    return
                with lock:
                    try:
                        op = self.path
                        n = int(self.headers.get("Content-Length", "0"))
                        if not 0 <= n <= CHUNK:
                            raise ValueError("请求超长")
                        data = self.rfile.read(n)
                        if len(data) != n:
                            raise ValueError("音频传输中断")
                        params = {} if op == "/step" else json.loads(data or b"{}")
                        if session and time.monotonic() - seen > 120:
                            session = ""
                        if op == "/state":
                            result = demo.snapshot()
                        elif op == "/clip":
                            self.send(demo.replay(), mime="application/octet-stream")
                            return
                        elif op in ("/start", "/add", "/remove", "/select") and session:
                            self.send({"error": "已有音频在识别，请先结束。"}, 409)
                            return
                        elif op == "/select":
                            i = int(params["i"])
                            demo.select(i)
                            result = {}
                        elif op == "/add":
                            result = demo.register(str(params["word"]).strip())
                        elif op == "/remove":
                            result = demo.remove(str(params["word"]))
                        elif op == "/start":
                            demo.begin_pass(params["kind"])
                            session = secrets.token_urlsafe(16)
                            seen = time.monotonic()
                            result = {"session": session}
                        elif op in ("/step", "/abort"):
                            if not session or self.headers.get("X-Session") != session:
                                raise ValueError("录音会话已失效")
                            if op == "/abort":
                                session = ""
                                result = {}
                            else:
                                if int(self.headers["X-Seq"]) != demo.seq:
                                    raise ValueError("音频块顺序不一致")
                                final = self.headers.get("X-Final") == "1"
                                result = demo.record_step(data, final)
                                seen = time.monotonic()
                                if final:
                                    session = ""
                        else:
                            self.send({}, 404)
                            return
                        self.send(result)
                    except Exception as exc:
                        traceback.print_exc()
                        self.send({"error": str(exc)}, 400)

        with ThreadingHTTPServer((args.bind, args.port), Handler) as http:
            ip = socket.gethostbyname(socket.gethostname())
            print(
                f"已就绪：实测录屏样例 {len(demo.clips)} 条。\n"
                f"电脑端单独开终端：ssh -N -L {args.port}:{ip}:{args.port} star@starpc\n"
                f"浏览器打开：http://localhost:{args.port}/#{token}",
                flush=True,
            )
            http.serve_forever()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
