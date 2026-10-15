"""Read-only progress display for the existing English preparation workflow."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def read_object(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return result if isinstance(result, dict) else {}


class PreparationProgress:
    """Count actual completed rows; never infer MFA percent or completion from silence."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.offsets: dict[Path, tuple[int, int]] = {}
        self.started = time.monotonic()
        self.phase_started = self.started
        self.phase = 0
        self.log_path: Path | None = None
        self.log_offset = 0
        self.log_pending = ""

    def line_count(self, path: Path) -> int:
        try:
            previous, count = self.offsets.get(path, (0, 0))
            if path.stat().st_size < previous:
                previous, count = 0, 0
            with path.open("rb") as handle:
                handle.seek(previous)
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    count += chunk.count(b"\n")
                self.offsets[path] = handle.tell(), count
            return count
        except OSError:
            return 0

    def log_updates(self) -> list[str]:
        paths = list(self.output.glob("mfa_attempt_*/mfa.log"))
        if not paths:
            return []
        latest = max(paths, key=lambda p: p.parent.stat().st_mtime_ns)
        if latest != self.log_path:
            self.log_path, self.log_offset, self.log_pending = latest, 0, ""
        try:
            with latest.open("rb") as handle:
                if latest.stat().st_size < self.log_offset:
                    self.log_offset = 0
                handle.seek(self.log_offset)
                content = handle.read(1024 * 1024)
                self.log_offset = handle.tell()
        except OSError:
            return []
        text = self.log_pending + ANSI.sub("", content.decode("utf-8", errors="replace"))
        lines = re.split(r"[\r\n]", text)
        self.log_pending = lines.pop()[-2000:]
        # Native MFA substage progress remains native; it is not total preparation percent.
        return [line.strip()[:300] for line in lines if line.strip()][-3:]

    def snapshot(self) -> dict[str, Any]:
        root = self.output
        report = read_object(root / "report.json")
        source = read_object(root / "source/report.json")
        total = source.get("source_records", 0)
        detail = ""
        done, progress = False, None
        if report.get("status") == "data_prepared":
            phase, name, done = 7, "数据准备完成", True
            en = report.get("selection", {}).get("en", {})
            detail = f"英语已选 {en.get('hours', '?')} 小时；后续单独运行 Encoder cache"
        elif not (root / "preparation_contract.json").exists():
            phase, name = 1, "输入身份核验"
            detail = "读取源JSON和原固定资产SHA；此阶段未提供总进度"
        elif not (root / "source/sha256.txt").exists():
            phase, name = 2, "三国训练JSON转换"
            detail = f"已写 {self.line_count(root / 'source/source_rows.jsonl'):,} 条训练记录"
        elif not (root / "words/report.json").exists():
            phase, name = 3, "全量训练词表提取"
            detail = f"从全部 {total:,} 条训练记录统计唯一词"
        elif not (root / "g2p_report.json").exists():
            phase, name = 4, "MFA G2P"
            words = read_object(root / "words/report.json").get("unique_words", "?")
            detail = f"{words} 个唯一词；未提供可靠总百分比，以下显示MFA原始日志"
        elif not (root / "manifest/sha256.txt").exists():
            phase, name = 5, "全量逐句CTC标签构建"
            checked = sum(
                read_object(p).get("source_records", 0)
                for p in (root / "manifest/shard_reports").glob("shard-*.json")
            )
            progress = (checked, total) if total else None
            detail = "按已完成shard统计；本阶段还包含生成文件的SHA核验"
        else:
            phase, name = 6, "去重、480小时选择、最终音频字节核验"
            attempts = list(root.glob("frozen_*"))
            if attempts:
                latest = max(attempts, key=lambda p: p.stat().st_mtime_ns)
                count = self.line_count(latest / "audio_fingerprints.jsonl")
                capacity = read_object(latest / "capacity_report.json")
                hours = capacity.get("available", {}).get("hours")
                detail = (
                    f"最终字节核验已写 {count:,} 个文件指纹"
                    if count
                    else "完整释放池字节扫描/去重；等待已核验的容量与选样结果"
                )
                if hours is not None:
                    detail += f"；合格唯一英语 {hours:.3f} 小时"
        if phase != self.phase:
            self.phase, self.phase_started = phase, time.monotonic()
        return {
            "phase": phase,
            "name": name,
            "detail": detail,
            "progress": progress,
            "done": done,
            "observed_seconds": time.monotonic() - self.started,
            "phase_observed_seconds": time.monotonic() - self.phase_started,
        }


def duration(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def render(state: dict[str, Any], tick: int) -> str:
    counted = state["progress"]
    if counted is not None:
        completed, total = counted
        fraction = min(1.0, completed / total)
        filled = int(fraction * 20)
        bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
        indicator = f"{bar} {completed:,}/{total:,} ({fraction:.1%})"
    else:
        indicator = "完成" if state["done"] else "|/-\\"[tick % 4]
    return (
        f"[{state['phase']}/7 {state['name']}] {indicator} "
        f"| 本阶段已观察 {duration(state['phase_observed_seconds'])} "
        f"| 监看用时 {duration(state['observed_seconds'])}\n  {state['detail']}"
    )
