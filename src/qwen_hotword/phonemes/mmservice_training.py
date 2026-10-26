"""CPU-only C++ G2P compatibility audit over the actual frozen training manifest."""

from __future__ import annotations

import json
import math
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mmservice_compact import _parts
from qwen_hotword.phonemes.mmservice_full import _checked
from qwen_hotword.phonemes.mmservice_process import execute
from qwen_hotword.phonemes.mmservice_smoke import _cpp_rows, _save, _sha, inspect_model
from qwen_hotword.phonemes.mmservice_training_delivery import export_differences
from qwen_hotword.phonemes.mmservice_training_labels import (
    LANGUAGES,
    candidates,
    compare_words,
    dictionary_labels,
    inventory,
    sentence_effect,
    source_key,
)
from qwen_hotword.training.stage_progress import stage_progress


def _read(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return value


def run(
    root: Path,
    config: Path,
    output: Path,
    *,
    timeout: float = 7200,
    release_config: Path | None = None,
    binary: Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
    inspector: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("choose a NEW training G2P diagnostic output directory")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    if (release_config is None) != (binary is None):
        raise ValueError("--release-config and --binary must be supplied together")
    identities: dict[str, Any] = {}
    sources = {}
    release: dict[str, Any] = {"name": "legacy_v1"}
    arguments: dict[str, list[str]] = {}
    with stage_progress("Verify frozen training manifest, original dictionaries and reviewed C++"):
        cfg = _read(_checked(config, identities))
        full = _read(_checked(root / cfg["full_comparison_config"], identities))
        cpp = _read(_checked(root / full["cpp_config"], identities, full["cpp_config_sha256"]))
        table = _read(
            _checked(root / full["table_config"], identities, full["table_config_sha256"])
        )
        manifest = _checked(root / cfg["manifest"], identities, cfg["manifest_sha256"])
        vocab_path = _checked(root / cpp["vocab"], identities, cpp["vocab_sha256"])
        vocab = load_phoneme_vocab(vocab_path)
        if release_config is not None:
            release = _read(_checked(root / release_config, identities))
            arguments = release["args_by_language"]
            if arguments != {lang: ["--lang", lang] for lang in LANGUAGES}:
                raise ValueError("reviewed release requires explicit --lang en/es/pt")
            pins = release["source_sha256"]
            if not isinstance(release.get("name"), str) or not release["name"]:
                raise ValueError("reviewed release requires a name")
            if (
                not isinstance(pins, dict)
                or not pins
                or any(
                    not isinstance(name, str)
                    or Path(name).name != name
                    or name in {".", ".."}
                    or not isinstance(digest, str)
                    for name, digest in pins.items()
                )
            ):
                raise ValueError("reviewed source names must be plain file names")
            assert binary is not None
            binary = _checked(root / binary, identities)
        else:
            pins = cpp["source_sha256"]
            binary = _checked(root / full["binary"], identities, full["binary_sha256"])
        with binary.open("rb") as handle:
            header = handle.read(20)
        if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
            raise ValueError("reviewed x86-64 binary required")
        if not os.access(binary, os.X_OK):
            raise ValueError("reviewed binary must be executable")
        for name, digest in pins.items():
            _checked(binary.parent / name, identities, digest)
        models = {
            lang: _checked(root / item["path"], identities, item["sha256"])
            for lang, item in table["models"].items()
        }
        if set(cfg["sources"]) != set(LANGUAGES) or set(cfg["expected_counts"]) != set(LANGUAGES):
            raise ValueError("exactly EN/ES/PT training sources and counts required")
        for lang, paths in cfg["sources"].items():
            for relative in paths:
                build_path = _checked(root / relative, identities)
                build = _read(build_path)
                key = source_key(build["tsv"]["path"], root)
                if key in sources:
                    raise ValueError(f"duplicate source TSV build configuration: {key}")
                # Historical build configs can have size/mtime only. Do not claim they
                # pin the dictionary SHA; independently verify against stored sentence IDs.
                dictionary = _checked(
                    Path(build["dictionary"]["path"]), identities, build["dictionary"].get("sha256")
                )
                print(f"Read existing {lang} dictionary: {dictionary}", flush=True)
                if Path(build["vocab"]["path"]).resolve() != vocab_path.resolve():
                    raise ValueError("source build config uses a different phoneme vocabulary")
                sources[key] = {
                    "language": lang,
                    "build_config": str(build_path),
                    "dictionary": str(dictionary),
                    "labels": dictionary_labels(dictionary, vocab),
                }
        for path in (
            Path(__file__),
            Path(__file__).with_name("mmservice_training_labels.py"),
            Path(__file__).with_name("mmservice_training_delivery.py"),
            Path(__file__).with_name("mmservice_process.py"),
            Path(__file__).with_name("mmservice_smoke.py"),
            Path(__file__).with_name("mmservice_full.py"),
            Path(__file__).with_name("mmservice_pipeline.py"),
            Path(__file__).with_name("mmservice_compact.py"),
            Path(__file__).with_name("coverage.py"),
            root / "src/qwen_hotword/training/g2p_prep.py",
            root / "src/qwen_hotword/training/spanish_mfa_repair.py",
            root / "scripts/compare_mmservice_g2p_training.py",
        ):
            _checked(path, identities)
        if any(
            output.is_relative_to(Path(p).parent) or Path(p).is_relative_to(output)
            for p in identities
        ):
            raise ValueError("output overlaps protected inputs")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "actual_training_manifest_g2p_label_compatibility",
        "output_dir": str(output),
        "inputs": identities,
        "by_language": {},
        "processes": [],
        "cpp_release": release,
        "cpp_args_by_language": arguments,
        "binary_sha256": _sha(binary),
        "audio_read": False,
        "gpu_used": False,
        "test_set_used": False,
        "official_g2p_rerun": False,
        "ctc_or_dictionary_modified": False,
        "source_tsv_root": str(root),
        "source_tsv_path_policy": "absolute paths retained; relative paths joined to explicit "
        "repository root; lexical normalization only, no source TSV stat/read/resolve",
        "replacement_verified": False,
        "reference": "existing source-specific MFA dictionaries; validated against ACTUAL frozen "
        "sentence labels; no majority merging of different source pronunciations",
        "input_policy": "project extract_word_tokens (NFC/casefold and lexical connectors); "
        "C++ internal preprocessing remains active; not original whole-sentence input",
        "limitations": [
            "Measures compatibility, not pronunciation accuracy, CTC PER, retrieval recall or WER.",
            "No fresh official G2P replay; C++ internal preprocessing remains active.",
            "Actual selected training samples only; not unused source words or sealed test data.",
            "Word-source labels and token occurrences are separate denominators.",
            "Sentence equality compares concatenated isolated-word outputs to stored labels; "
            "it does not claim whole-sentence C++ input tokenization equivalence.",
            "Dictionary identities without historical SHA are recorded as current bytes; "
            "baseline reconstruction is required for attributing word differences to training.",
        ],
    }
    call = runner or execute
    inspect = inspector or inspect_model
    examples: list[dict[str, Any]] = []
    try:
        total = sum(cfg["expected_counts"].values())
        data = inventory(manifest, sources, vocab, total, root=root)
        frequencies = data["frequencies"]
        if {k: v["samples"] for k, v in data["stats"].items()} != cfg["expected_counts"]:
            raise ValueError("per-language frozen training counts differ")
        report["baseline_examples"] = data["baseline_examples"]
        decoded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for lang in LANGUAGES:
            words = sorted(
                {w for (source, w) in frequencies if sources[source]["language"] == lang}
            )
            directory = output / lang
            directory.mkdir()
            word_path = directory / "words.txt"
            word_path.write_text("\n".join(words) + "\n", encoding="utf-8")
            with stage_progress(f"Inspect unchanged official model for C++ {lang}"):
                model = inspect(models[lang], directory / "model")
            report["by_language"][lang] = {
                **data["stats"][lang],
                "unique_words": len(words),
                "word_reference_attribution_verified": data["stats"][lang]["baseline_not_verified"]
                == 0,
                "model": model,
            }
            print(
                f"{lang}: {len(words):,} unique ACTUAL training words; C++ native output",
                flush=True,
            )
            process = call(
                [
                    str(binary),
                    str(word_path),
                    str(directory / "model/model.fst"),
                    str(directory / "model/phones.sym"),
                    *arguments.get(lang, []),
                ],
                directory / "cpp.jsonl",
                directory / "cpp.stderr.log",
                label=f"C++ G2P ALL {lang} training words",
                rows_total=len(words),
                timeout=timeout,
                env={**os.environ},
            )
            report["processes"].append(process)
            if process["returncode"] != 0 or process.get("timed_out"):
                raise RuntimeError(f"{lang} C++ process failed; preserve output and see stderr")
            native = _cpp_rows(directory / "cpp.jsonl", words, process)
            decoded[lang] = {row["word"]: candidates(row, vocab, lang) for row in native}
            report["by_language"][lang]["cpp_unique_word_ok"] = sum(row["ok"] for row in native)
        with stage_progress(
            "Compare source-specific word labels and occurrence-weighted differences"
        ):
            counts, examples = compare_words(frequencies, sources, decoded, output / "words.jsonl")
            for lang in LANGUAGES:
                report["by_language"][lang]["word_label_comparison"] = counts[lang]
                report["by_language"][lang]["source_word_pairs"] = sum(
                    sources[source]["language"] == lang for source, _ in frequencies
                )
        sentences = sentence_effect(
            manifest, sources, decoded, vocab, total, output / "samples.jsonl", root=root
        )
        for lang in LANGUAGES:
            report["by_language"][lang]["sentence_label_comparison"] = sentences[lang]
        if release_config is not None:
            report["delivery"] = export_differences(output, report["by_language"])
        with stage_progress("Recheck recorded input identities after CPU-only comparison"):
            for name, item in identities.items():
                if _sha(Path(name)) != item["sha256"]:
                    raise ValueError(f"input changed during comparison: {name}")
        report["status"] = "completed"
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        ImportError,
        KeyboardInterrupt,
    ) as error:
        report.update(status="failed", error=str(error) or type(error).__name__)
    _save(output / "report.json", report)
    _save(output / "examples.json", examples)
    with stage_progress("Package SMALL summary and examples; create <=32KiB text parts"):
        names = ["report.json", "examples.json"]
        _save(
            output / "source_evidence.json",
            {
                name: {"sha256": _sha(output / name), "bytes": (output / name).stat().st_size}
                for name in ("words.jsonl", "samples.jsonl")
                if (output / name).is_file()
            },
        )
        names.append("source_evidence.json")
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(output / name)}  {name}\n" for name in names), encoding="utf-8"
        )
        archive = output / "return_summary.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for name in [*names, "sha256.txt"]:
                handle.write(output / name, name)
        parts = _parts(archive, 32 * 1024)
    return {
        "status": report["status"],
        "error": report.get("error"),
        "output_dir": str(output),
        "by_language": {
            lang: {key: value for key, value in item.items() if key != "model"}
            for lang, item in report["by_language"].items()
        },
        "official_g2p_rerun": False,
        "replacement_verified": False,
        "cpp_release": release["name"],
        "binary_sha256": report["binary_sha256"],
        "delivery": report.get("delivery", {}),
        "return_file": str(archive),
        "zip_bytes": archive.stat().st_size,
        "text_parts": parts,
    }
