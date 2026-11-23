"""Resume interrupted immutable downloads; only SHA-verified files become final."""

from __future__ import annotations

import http.client
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress


def download(url: str, path: Path, digest: str, size: int, *, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        pin(path, digest)
        return
    pending = path.with_suffix(path.suffix + ".pending")
    failures = 0
    checksum_restarts = 0
    with stage_progress(
        "Download " + path.name + " (bytes)", total=size, every=8 * 1024 * 1024
    ) as tick:
        while True:
            offset = pending.stat().st_size if pending.exists() else 0
            tick(offset)
            if offset >= size:
                if offset == size and sha(pending) == digest:
                    pending.replace(path)
                    return
                if checksum_restarts:
                    raise ValueError("download SHA256 differs from pinned input: " + str(pending))
                pending.replace(pending.with_suffix(pending.suffix + ".bad-sha"))
                checksum_restarts += 1
                print("Invalid completed download; retrying once from byte 0", flush=True)
                continue
            # Bounded ranges avoid restarting a large transfer after a proxy cuts the response.
            end = min(size - 1, offset + 8 * 1024 * 1024 - 1)
            request = urllib.request.Request(
                url + ("&" if "?" in url else "?") + f"download=true&range_start={offset}",
                headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
            )
            before = offset
            try:
                with urllib.request.urlopen(
                    request, context=ssl._create_unverified_context(), timeout=60
                ) as response:
                    status = response.status
                    mode = "ab"
                    if status == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                        )
                        if (
                            not match
                            or int(match[1]) != offset
                            or int(match[3]) != size
                            or not offset <= int(match[2]) <= end
                        ):
                            raise ValueError(
                                "invalid HTTP Content-Range; refusing incorrect append"
                            )
                        limit = int(match[2]) + 1
                    elif status == 200:
                        # Range ignored: replace, never append the full body to partial bytes.
                        mode, offset, limit = "wb", 0, size
                    else:
                        raise ValueError(f"unexpected HTTP download status: {status}")
                    with pending.open(mode) as stream:
                        while chunk := response.read(min(1024 * 1024, limit - offset + 1)):
                            if offset + len(chunk) > limit:
                                raise ValueError("HTTP body exceeds declared range")
                            stream.write(chunk)
                            offset += len(chunk)
                            tick(offset)
                if offset <= before:
                    raise OSError("download made no forward progress")
                failures = 0
            except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
                failures += 1
                if failures >= retries:
                    raise RuntimeError(
                        f"Download interrupted after {retries} retries; "
                        f"partial bytes retained: {pending}"
                    ) from error
                print(f"Download interrupted; resuming ({failures}/{retries}): {error}", flush=True)
                time.sleep(min(failures, 3))
