#!/usr/bin/env python3
"""Show live preparation progress; optionally launch the unchanged prepare command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.english_refresh_progress import PreparationProgress, render


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start prepare; otherwise only observe")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_countries_refresh.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 1 or args.interval > 60:
        parser.error("--interval must be between 1 and 60 seconds")
    if args.resume and not args.run:
        parser.error("--resume only applies with --run; default watch mode changes no state")
    root = args.root.resolve()
    config_path = (root / args.config).resolve()
    cfg = json.loads(config_path.read_text())
    output = (root / (args.output_dir or cfg["output"])).resolve()
    process = None
    if args.run:
        command = [
            sys.executable,
            "-u",
            "-B",
            str(root / "scripts/run_english_refresh.py"),
            "prepare",
            "--root",
            str(root),
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--workers",
            str(args.workers),
            "--jobs",
            str(args.jobs),
        ]
        if args.resume:
            command.append("--resume")
        process = subprocess.Popen(command, cwd=root)
    else:
        print("只读监看：不会启动或重跑任务；文件进度不能单独证明进程仍在运行。", flush=True)
    monitor = PreparationProgress(output)
    print("阶段编号不是总耗时百分比；MFA未报告百分比时只显示活动指示与已观察时间。", flush=True)
    try:
        tick = 0
        while True:
            state = monitor.snapshot()
            print(render(state, tick), flush=True)
            for line in monitor.log_updates():
                print("  MFA日志> " + line, flush=True)
            if process is not None:
                code = process.poll()
                if code is not None:
                    print(
                        f"prepare进程已退出，exit_code={code}；以原任务JSON和报告判定结果。",
                        flush=True,
                    )
                    return code
            elif state["done"]:
                return 0
            tick += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if process is None:
            print("\n监看已退出，原任务不受影响。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
