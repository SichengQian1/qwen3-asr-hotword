"""Local demo TLS assets; private keys never leave the server."""

from __future__ import annotations

import ipaddress
import os
import ssl
import subprocess
from pathlib import Path


def certificate_context(
    directory: Path, host: str, cert: Path | None = None, key: Path | None = None
) -> ssl.SSLContext:
    if (cert is None) != (key is None):
        raise ValueError("--tls-cert 和 --tls-key 必须一起提供")
    if cert is None:
        ipaddress.ip_address(host)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        cert, key = directory / f"{host}.crt", directory / f"{host}.key"
        ca, ca_key = directory / "demo-ca.crt", directory / "demo-ca.key"

        def openssl(*args: str) -> None:
            result = subprocess.run(["openssl", *args], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("HTTPS证书生成失败：" + result.stderr.strip())

        old_umask = os.umask(0o077)
        try:
            if not ca.exists() and not ca_key.exists():
                openssl(
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-sha256",
                    "-days",
                    "3650",
                    "-subj",
                    "/CN=Qwen Demo Internal CA",
                    "-addext",
                    "basicConstraints=critical,CA:TRUE",
                    "-addext",
                    "keyUsage=critical,keyCertSign,cRLSign",
                    "-addext",
                    "subjectKeyIdentifier=hash",
                    "-keyout",
                    str(ca_key),
                    "-out",
                    str(ca),
                )
            if not ca.exists() or not ca_key.exists():
                raise ValueError("CA文件不完整；请检查证书目录，不自动替换已有CA")
            valid = (
                cert.exists()
                and key.exists()
                and subprocess.run(
                    ["openssl", "x509", "-checkend", "86400", "-noout", "-in", str(cert)],
                    capture_output=True,
                ).returncode
                == 0
            )
            if not valid:
                csr, ext = directory / f"{host}.csr", directory / f"{host}.ext"
                ext.write_text(
                    f"subjectAltName=IP:{host}\nbasicConstraints=CA:FALSE\n"
                    "keyUsage=digitalSignature,keyEncipherment\n"
                    "extendedKeyUsage=serverAuth\n"
                    "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n",
                    encoding="utf-8",
                )
                openssl(
                    "req",
                    "-new",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-sha256",
                    "-subj",
                    f"/CN={host}",
                    "-keyout",
                    str(key),
                    "-out",
                    str(csr),
                )
                openssl(
                    "x509",
                    "-req",
                    "-in",
                    str(csr),
                    "-CA",
                    str(ca),
                    "-CAkey",
                    str(ca_key),
                    "-CAcreateserial",
                    "-days",
                    "365",
                    "-sha256",
                    "-extfile",
                    str(ext),
                    "-out",
                    str(cert),
                )
            public_ca = directory.parent / "demo_ca.crt"
            public_ca.write_bytes(ca.read_bytes())
            public_ca.chmod(0o644)
            print(f"首次使用：同事电脑信任公开CA证书 {public_ca}；不要传递任何.key文件", flush=True)
        finally:
            os.umask(old_umask)
    assert key is not None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context
