"""The host relay only transports bytes; it does not terminate or weaken TLS."""

import json
import socket
import socketserver
import sys
import threading

from scripts import expose_zh_rag_demo as relay


def test_existing_container_port_relay_roundtrip(monkeypatch):
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            data = self.request.recv(4096)
            self.request.sendall(data)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo) as upstream:
        port = upstream.server_address[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            host_port = probe.getsockname()[1]
        source_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        source_thread.start()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "relay",
                "--container",
                "current",
                "--host-ip",
                "127.0.0.1",
                "--port",
                str(host_port),
                "--container-port",
                str(port),
            ],
        )

        def inspect(command):
            assert command == ["docker", "inspect", "current"]
            return json.dumps(
                [
                    {
                        "HostConfig": {"NetworkMode": "bridge"},
                        "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "127.0.0.1"}}},
                    }
                ]
            ).encode()

        monkeypatch.setattr(relay.subprocess, "check_output", inspect)
        ready, instances, errors = threading.Event(), [], []
        base = socketserver.ThreadingTCPServer

        class ObservedServer(base):
            def serve_forever(self):
                instances.append(self)
                ready.set()
                super().serve_forever(poll_interval=0.05)

        monkeypatch.setattr(relay.socketserver, "ThreadingTCPServer", ObservedServer)

        def run():
            try:
                relay.main()
            except Exception as exc:
                errors.append(exc)
                ready.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            assert ready.wait(5)
            assert not errors
            with socket.create_connection(("127.0.0.1", host_port), timeout=5) as client:
                data = b"\x16\x03\x01opaque-tls-payload\x00\xff"
                client.sendall(data)
                assert client.recv(4096) == data
        finally:
            if instances:
                instances[0].shutdown()
            thread.join(5)
            upstream.shutdown()
            source_thread.join(5)
        assert not errors
