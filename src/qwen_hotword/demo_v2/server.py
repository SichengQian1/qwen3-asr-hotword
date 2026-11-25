"""Shared LAN console with one active recording at a time."""

from __future__ import annotations

import argparse
import fcntl
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

from qwen_hotword.training.stage_progress import stage_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=3, type=int, help="物理 GPU 卡号，默认 3")
    parser.add_argument("--https", action="store_true", help="开启HTTPS，默认自动生成内网证书")
    parser.add_argument("--public-host", default="10.44.168.92")
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument(
        "--max-cases", default=0, type=int, help="每次自动验证的候选上限；0=仅麦克风（默认）"
    )
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu 必须为非负整数")
    if args.max_cases < 0:
        parser.error("--max-cases 必须为非负整数")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须为1到65535")
    if (args.tls_cert or args.tls_key) and not args.https:
        parser.error("指定证书时必须加 --https")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Set visibility before importing the inference stack or spawning the Decoder.
    from qwen_hotword.demo.runtime import CHUNK
    from qwen_hotword.demo_v2.session import SharedStudio

    # Don't initialize CUDA and run a search only to discover an occupied port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((args.bind, args.port))
    root = Path.cwd()
    tls = None
    if args.https:
        from qwen_hotword.demo_v2.https import certificate_context

        tls = certificate_context(
            root / "outputs/demo_tls", args.public_host, args.tls_cert, args.tls_key
        )
    static = root / "examples/zh_rag_demo_v2"
    (root / "outputs").mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="zh_rag_demo_v2_", dir=root / "outputs"))
    lock = Lock()
    session = ""
    seen = 0.0
    print("本次输出：", out, flush=True)
    with stage_progress(f"验证资产，加载 GPU {args.gpu} 检索与双路共享的语音解码模型"):
        demo = SharedStudio(root, out, args.gpu, Path(os.environ["DEMO_DECODER_PY"]))
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
        demo.select(-1)
        print("默认进入自由麦克风；同事共享热词和历史，轮流录音。", flush=True)

        (out / "startup.json").write_text(
            json.dumps(
                dict(
                    status="ready",
                    gpu=args.gpu,
                    https=args.https,
                    public_url=f"https://{args.public_host}:{args.port}/" if tls else None,
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
                            result.update(gpu=args.gpu, occupied=bool(session))
                        elif op == "/vocabulary":
                            result = {
                                "words": sorted({e.surface for e in demo.entries}),
                                "extras": sorted(demo.extras),
                            }
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
            if tls:
                http.socket = tls.wrap_socket(http.socket, server_side=True)
                (root / "outputs/demo_https_endpoint.json").write_text(
                    json.dumps(
                        {
                            "container": socket.gethostname(),
                            "host": args.public_host,
                            "port": args.port,
                            "output_dir": str(out),
                        }
                    ),
                    encoding="utf-8",
                )
                print(
                    f"内网访问：https://{args.public_host}:{args.port}/（直接进入，无访问口令）\n"
                    "若容器无端口映射，在宿主机另开终端执行：\n"
                    f"python3 scripts/expose_zh_rag_demo.py --container {socket.gethostname()} "
                    f"--host-ip {args.public_host} --port {args.port}",
                    flush=True,
                )
            if not tls:
                ip = socket.gethostbyname(socket.gethostname())
                print(
                    f"已就绪：实测录屏样例 {len(demo.clips)} 条。\n"
                    f"电脑端单独开终端：ssh -N -L {args.port}:{ip}:{args.port} star@starpc\n"
                    f"浏览器打开：http://localhost:{args.port}/",
                    flush=True,
                )
            http.serve_forever()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
