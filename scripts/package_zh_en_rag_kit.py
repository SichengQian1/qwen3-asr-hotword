"""Build the portable bilingual 183-class retrieval kit from verified work-zone assets."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import suppress
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read


def runtime_sources(entry: Path, src: Path) -> set[Path]:
    """Include local imports (also lazy imports) plus their package initializers."""
    todo = [entry]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        mods = []
        for n in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if isinstance(n, ast.ImportFrom):
                if n.level:
                    raise ValueError(f"unhandled relative import: {path}:{n.lineno}")
                mod = n.module or ""
                mods.extend([mod, *(mod + "." + a.name for a in n.names)])
            elif isinstance(n, ast.Import):
                mods.extend(a.name for a in n.names)
        for mod in mods:
            if not mod.startswith("qwen_hotword"):
                continue
            obj = src.joinpath(*mod.split("."))
            for item in (obj.with_suffix(".py"), obj / "__init__.py"):
                if item.is_file():
                    todo.append(item)
                    p = item.parent
                    while p != src:
                        if (p / "__init__.py").exists():
                            todo.append(p / "__init__.py")
                        p = p.parent
    return seen - {entry}


def main() -> None:
    cfg = read(ROOT / "configs/zh_light_eval.workzone.json")
    run = ROOT / cfg["output"]
    if not (run / "report.json").is_file():
        raise SystemExit(f"需要工区本轮已完成的 report.json：{run}")
    report, plan = read(run / "report.json"), read(run / "run_config.json")
    model = plan["model"]
    cc = read(ROOT / cfg["checkpoint_config"])
    if (
        report["status"],
        report["run_sha256"],
        report["model"],
        report["table_sha256"],
        report["active_surfaces"],
        report["pronunciations"],
        model["checkpoint_sha256"],
        model["vocab_sha256"],
        model["checkpoint_epoch"],
    ) != (
        "completed",
        digest(plan),
        model,
        plan["table_sha256"],
        2411,
        2412,
        cc["required_checkpoint_sha256"],
        cc["vocab_sha256"],
        25,
    ):
        raise ValueError("expected the completed 480h epoch-25 joint2411 run")
    assets = {
        "ctc_head_best.pt": (
            ROOT / cc["training_run"] / "head/ctc_head_best.pt",
            model["checkpoint_sha256"],
        ),
        "vocab.json": (
            ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json",
            model["vocab_sha256"],
        ),
        "joint2411.jsonl": (run / "joint2411.jsonl", plan["table_sha256"]),
    }
    template = ROOT / "examples/zh_en_rag_kit"
    src = ROOT / "src"
    with stage_progress("Verify assets and package runtime, without loading the model"):
        files = {n: template / n for n in ("run_delivery.py", "README.md")}
        for name, (path, checksum) in assets.items():
            pin(path, checksum)
            files["assets/" + name] = path
        for p in runtime_sources(template / "run_delivery.py", src):
            files["src/" + p.relative_to(src).as_posix()] = p
        for n, checksum in model["code_sha256"].items():
            rel = "src/qwen_hotword/" + n
            if rel in files:
                pin(files[rel], checksum)
        # The evaluator records the retrieval loop separately from model.code_sha256.
        pin(src / "qwen_hotword/evaluation/zh_en_delivery.py", plan["code_sha256"][2])
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        target = Path(tempfile.mkdtemp(dir=run.parent, prefix="zh_en183_rag_kit_"))
        with tempfile.TemporaryDirectory(dir=target) as temp:
            staging = Path(temp)
            versions = dict(model["versions"])
            for name in ("accelerate", "PyYAML"):
                versions[name] = metadata.version(name)
            with suppress(metadata.PackageNotFoundError):
                versions["rapidfuzz"] = metadata.version("rapidfuzz")
            req = staging / "requirements.txt"
            req.write_text(
                "".join(f"{n}=={v}\n" for n, v in sorted(versions.items())), encoding="utf8"
            )
            files["requirements.txt"] = req
            identity = dict(
                revision=revision,
                source_run_sha256=report["run_sha256"],
                checkpoint_epoch=25,
                base_model_files=model["model_identity"],
                files={n: sha(p) for n, p in sorted(files.items())},
            )
            pending = staging / "kit.zip"
            with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("zh_en_rag/manifest.json", json.dumps(identity, indent=2))
                for n, p in sorted(files.items()):
                    z.write(p, "zh_en_rag/" + n)
            with zipfile.ZipFile(pending) as z:
                if z.testzip() is not None:
                    raise ValueError("kit CRC failed")
            result = target / "zh_en_rag_kit.zip"
            shutil.move(pending, result)
    print(f"请发给下游：{result}\nSHA256: {sha(result)}", flush=True)


if __name__ == "__main__":
    main()
