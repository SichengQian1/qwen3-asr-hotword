"""Append the November Mandarin business vocabulary, then seal an independent Ascend kit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from package_zh_en_rag_kit import runtime_sources

from qwen_hotword.evaluation import aishell_run as ar
from qwen_hotword.evaluation.zh_en_delivery import canonical
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows, save_rows


def requested(text: str, previous: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    inventory = [w.strip() for line in text.splitlines() for w in line.split("、") if w.strip()]
    original = {canonical(r["surface"]) for r in previous}
    first: dict[str, str] = {}
    repeats = []
    reused = []
    added = []
    for w in inventory:
        k = canonical(w)
        if k in first:
            repeats.append([w, first[k]])
            continue
        first[k] = w
        if k in original:
            reused.append(w)
        else:
            added.append(w)
    return added, dict(
        listed=len(inventory), distinct=len(first), repeated=repeats, reused=reused, added=added
    )


def spoken(word: str) -> str:
    # Surface stays unchanged; these are pronunciation inputs only.
    return (
        word.casefold()
        .replace("agentos", " agent oh ess ")
        .replace("tv", " tee vee ")
        .replace("2026", "二零二六")
        .replace("1.5", "一点五")
        .replace("2", "二")
        .replace("1", "一")
    ).strip()


def main() -> None:
    cfg = read(ROOT / "configs/zh_light_eval.workzone.json")
    parent = ROOT / cfg["output"]
    src = ROOT / "src"
    tpl = ROOT / "examples/zh_en_ascend"
    shared = ROOT / "examples/zh_en_rag_kit/run_delivery.py"
    with stage_progress("校验原2411表、183类和epoch25模型"):
        plan, report = read(parent / "run_config.json"), read(parent / "report.json")
        cc = read(ROOT / cfg["checkpoint_config"])
        m = plan["model"]
        if (
            report["status"],
            report["run_sha256"],
            report["model"],
            report["table_sha256"],
            m["checkpoint_epoch"],
            m["checkpoint_sha256"],
            m["vocab_sha256"],
        ) != (
            "completed",
            digest(plan),
            m,
            plan["table_sha256"],
            25,
            cc["required_checkpoint_sha256"],
            cc["vocab_sha256"],
        ):
            raise ValueError("wrong original run/model")
        old = parent / "joint2411.jsonl"
        pin(old, plan["table_sha256"])
        table = list(rows(old))
        train = read(ROOT / cc["training_config"])
        f = read(ROOT / train["foundation_config"])
        vp = ROOT / f["vocab"]
        pt = ROOT / cc["training_run"] / "head/ctc_head_best.pt"
        pin(pt, m["checkpoint_sha256"])
        pin(vp, m["vocab_sha256"])
        from qwen_hotword.training.zh_en_foundation import verify_assets

        verify_assets(ROOT, f)
        vocab = load_phoneme_vocab(vp)
        words, audit = requested(
            (ROOT / "configs/zh_en_ascend_additions.txt").read_text(encoding="utf8"), table
        )
        if (
            len(table),
            len({canonical(r["surface"]) for r in table}),
            audit["listed"],
            audit["distinct"],
            len(words),
        ) != (2411, 2411, 304, 293, 285):
            raise ValueError("business vocabulary coverage differs")
    out = Path(tempfile.mkdtemp(dir=parent.parent, prefix="zh_en_ascend2696_"))
    print("本轮工作目录：", out, flush=True)
    spoken_inputs = list(dict.fromkeys(spoken(w) for w in words))
    with stage_progress("先生成新词MFA发音并更新联合词表"):
        entries = ar.pronunciations(
            ROOT,
            out,
            spoken_inputs,
            train,
            f,
            vocab,
            digest([spoken_inputs, f, train["frontend_packages"]]),
        )
        by_spoken = {e.surface: e for e in entries}
        if set(by_spoken) != set(spoken_inputs):
            raise ValueError("incomplete pronunciations")
        for w in words:
            e = by_spoken[spoken(w)]
            table.append(
                dict(
                    surface=w,
                    pronunciations=[
                        dict(
                            phoneme_tokens=e.phoneme_tokens,
                            token_ids=e.token_ids,
                            languages=["zh", "us"]
                            if any(c.isascii() and c.isalpha() for c in w)
                            else ["zh"],
                            sources=["business_20261128"],
                        )
                    ],
                )
            )
        new = out / "joint2696.jsonl"
        save_rows(new, table)
        duplicates = Counter(tuple(p["token_ids"]) for r in table for p in r["pronunciations"])
        audit.update(
            spoken_inputs={w: spoken(w) for w in words if spoken(w) != w},
            original_table_sha256=sha(old),
            table_sha256=sha(new),
            surfaces=len(table),
            pronunciations=sum(duplicates.values()),
            shared_pronunciation_sequences=sum(n > 1 for n in duplicates.values()),
        )
        if (audit["surfaces"], audit["pronunciations"]) != (2696, 2697):
            raise ValueError("unexpected expanded counts")
        audit_path = out / "word_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    with stage_progress("封装独立昇腾运行包和同表H200对照入口"):
        files = {
            p.name: p
            for p in [tpl / "run_ascend.py", tpl / "compare_delivery.py", tpl / "README.md"]
        }
        files.update(
            {
                "common_delivery.py": shared,
                "assets/head.pt": pt,
                "assets/vocab.json": vp,
                "assets/joint2696.jsonl": new,
                "assets/word_audit.json": audit_path,
            }
        )
        for p in runtime_sources(shared, src) | runtime_sources(tpl / "run_ascend.py", src):
            files["src/" + p.relative_to(src).as_posix()] = p
        for name, checksum in m["code_sha256"].items():
            dest = "src/qwen_hotword/" + name
            if dest in files:
                pin(files[dest], checksum)
        pin(src / "qwen_hotword/evaluation/zh_en_delivery.py", plan["code_sha256"][2])
        dist = metadata.distribution("qwen-asr")
        if dist.version != "0.0.6":
            raise ValueError("package in the original qwen-asr 0.0.6 environment")
        for name in [
            "configuration_qwen3_asr.py",
            "modeling_qwen3_asr.py",
            "processing_qwen3_asr.py",
        ]:
            files["kit_qwen/" + name] = Path(
                str(dist.locate_file("qwen_asr/core/transformers_backend/" + name))
            )
        (license_path,) = [
            p for p in dist.files or [] if p.name == "LICENSE" and ".dist-info" in str(p)
        ]
        files["kit_qwen/LICENSE"] = Path(str(dist.locate_file(license_path)))
        identity = dict(
            files={n: sha(p) for n, p in sorted(files.items())},
            base_model_files=m["model_identity"],
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_run=report["run_sha256"],
            checkpoint_epoch=25,
        )
        pending = out / "kit.pending"
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("zh_en_ascend/manifest.json", json.dumps(identity, indent=2))
            for n, p in sorted(files.items()):
                z.write(p, "zh_en_ascend/" + n)
        with zipfile.ZipFile(pending) as z:
            if z.testzip():
                raise ValueError("kit CRC failed")
        final = out / "zh_en_ascend2696.zip"
        pending.rename(final)
    print(f"新增285词，2696词面 / 2697发音。请发给下游：{final}\nSHA256: {sha(final)}", flush=True)


if __name__ == "__main__":
    main()
