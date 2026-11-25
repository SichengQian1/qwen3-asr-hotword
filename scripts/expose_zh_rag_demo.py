#!/usr/bin/env python3
"""Run on the Docker HOST: forward one LAN port to the existing TLS demo container."""

from __future__ import annotations

import argparse
import json
import select
import socket
import socketserver
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", help="默认读取最近一次HTTPS启动记录")
    parser.add_argument("--host-ip")
    parser.add_argument("--port", type=int)
    parser.add_argument("--container-port", type=int, help="默认与Demo启动记录或--port一致")
    parser.add_argument("--container-ip", help="仅多网络容器需要指定")
    args = parser.parse_args()
    endpoint = {}
    if not args.container:
        path = Path(__file__).resolve().parents[1] / "outputs/demo_https_endpoint.json"
        if not path.exists():
            parser.error("请先在容器启动HTTPS Demo并等候就绪，或指定 --container")
        endpoint = json.loads(path.read_text(encoding="utf-8"))
        args.container = endpoint["container"]
    args.host_ip = args.host_ip or endpoint.get("host", "10.44.168.92")
    args.port = args.port or endpoint.get("port", 8444)
    target_port = args.container_port or endpoint.get("port", args.port)
    info = json.loads(subprocess.check_output(["docker", "inspect", args.container]))[0]
    if info["HostConfig"]["NetworkMode"] == "host":
        print("容器已使用host网络，无需转发；直接访问Demo打印的HTTPS地址。")
        return
    addresses = {
        n["IPAddress"] for n in info["NetworkSettings"]["Networks"].values() if n.get("IPAddress")
    }
    target = args.container_ip or (next(iter(addresses)) if len(addresses) == 1 else None)
    if target not in addresses:
        parser.error(f"请用 --container-ip 从当前容器地址中选择：{sorted(addresses)}")

    class Relay(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                with socket.create_connection((target, target_port), timeout=10) as upstream:
                    upstream.settimeout(None)
                    while True:
                        ready, _, _ = select.select([self.request, upstream], [], [], 180)
                        if not ready:
                            return
                        for source in ready:
                            chunk = source.recv(65536)
                            if not chunk:
                                return
                            (upstream if source is self.request else self.request).sendall(chunk)
            except OSError as exc:
                print(f"连接结束：{exc}", flush=True)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((args.host_ip, args.port), Relay) as server:
        print(
            f"HTTPS透传：{args.host_ip}:{args.port} → {target}:{target_port}；Ctrl+C停止",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
