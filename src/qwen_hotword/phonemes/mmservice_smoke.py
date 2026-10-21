"""Keep raw official FSTs and C++ source unchanged; distinguish failures from equality."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    normalize_ipa_for_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.stage_progress import stage_progress


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path, inputs: dict[str, Any], expected: str | None = None) -> Path:
    path = path.resolve()
    sha = _sha(path)
    if expected is not None and sha != expected:
        raise ValueError(f"SHA256 differs from the reviewed input: {path}")
    inputs[str(path)] = {"sha256": sha, "bytes": path.stat().st_size}
    return path


def inspect_model(archive: Path, directory: Path) -> dict[str, Any]:
    """Read and copy named official members only; never attach external symbol tables."""
    directory.mkdir()
    copied = {}
    with zipfile.ZipFile(archive) as source:
        for name in ("model.fst", "phones.sym", "meta.json", "graphemes.sym"):
            member = f"{archive.stem}/{name}"
            occurrences = source.namelist().count(member)
            if occurrences == 0 and name == "graphemes.sym":
                continue
            if occurrences != 1 or source.getinfo(member).file_size > 128 * 1024 * 1024:
                raise ValueError(f"missing/duplicate/oversized official model member: {member}")
            path = directory / name
            path.write_bytes(source.read(member))
            copied[name] = _sha(path)
    metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    fst = importlib.import_module("pynini").Fst.read(str(directory / "model.fst"))
    symbols = fst.input_symbols()
    return {
        "metadata": metadata,
        "member_sha256": copied,
        "fst_type": fst.fst_type(),
        "arc_type": fst.arc_type(),
        "num_states": fst.num_states(),
        "input_symbols_embedded": symbols is not None,
        "embedded_input_symbol_count": symbols.num_symbols() if symbols is not None else 0,
        "external_graphemes_present": "graphemes.sym" in copied,
        "cpp_inferred_input_mode": "bigram" if symbols is not None else "unicode_codepoint",
        "fst_modified": False,
        "external_symbols_attached": False,
    }


def _command(
    command: list[str], stdout: Path, log: Path, *, timeout: float, env: dict[str, str]
) -> dict[str, Any]:
    started = time.monotonic()
    with stdout.open("x", encoding="utf-8") as out, log.open("x", encoding="utf-8") as err:
        try:
            result = subprocess.run(
                command, stdout=out, stderr=err, env=env, timeout=timeout, check=False
            )
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    return {
        "command": command,
        "returncode": code,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "stderr_log": str(log),
    }


def _cpp_rows(path: Path, cases: list[str], process: dict[str, Any]) -> list[dict[str, Any]]:
    if process["returncode"] != 0:
        return [{"word": word, "ipa": "", "ok": False, "process_failed": True} for word in cases]
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if [r.get("word") for r in rows] != cases:
        raise ValueError("C++ output count/order/original surface differs from requested cases")
    if any(
        type(r.get("ok")) is not bool
        or not isinstance(r.get("ipa"), str)
        or (not r["ok"] and r["ipa"])
        for r in rows
    ):
        raise ValueError("invalid C++ result schema")
    return rows


def _phones(ipa: str | None, vocab: PhonemeVocab) -> dict[str, Any]:
    if ipa is None:
        return {
            "usable": False,
            "raw_ipa": None,
            "normalized_ipa": None,
            "token_ids": [],
            "oov_units": [],
        }
    converted = tokenize_ipa_to_vocab(ipa, vocab)
    return {
        "usable": bool(converted.token_ids) and not converted.oov_units,
        "raw_ipa": ipa,
        "normalized_ipa": converted.normalized_ipa,
        "token_ids": converted.token_ids,
        "oov_units": converted.oov_units,
    }


def compare_cases(
    cases: list[str],
    dictionary: Path,
    raw: list[dict[str, Any]],
    normalized: list[dict[str, Any]],
    vocab: PhonemeVocab,
) -> list[dict[str, Any]]:
    phones: dict[str, set[str]] = {}
    for word, pronunciations in load_mfa_dictionary(dictionary).items():
        phones.setdefault(normalize_training_text(word), set()).update(pronunciations)
    results = []
    for surface, cpp_raw, cpp_final in zip(cases, raw, normalized, strict=True):
        words = extract_word_tokens(surface)
        blocked = not words or any(c.isdigit() for c in surface)
        missing = sorted({w for w in words if not phones.get(w)})
        ambiguous = sorted({w for w in words if len(phones.get(w, ())) > 1})
        reference = (
            None
            if blocked or missing or ambiguous
            else " ".join(next(iter(phones[w])) for w in words)
        )
        mfa = _phones(reference, vocab)
        candidate_raw = _phones(cpp_raw["ipa"] if cpp_raw["ok"] else None, vocab)
        candidate = _phones(cpp_final["ipa"] if cpp_final["ok"] else None, vocab)
        results.append(
            {
                "surface": surface,
                "project_words": words,
                "numeric_or_empty_input": blocked,
                "mfa_missing_words": missing,
                "mfa_ambiguous_words": ambiguous,
                "mfa": mfa,
                "cpp_raw": {**cpp_raw, **candidate_raw},
                "cpp_normalized": {**cpp_final, **candidate},
                "raw_exact": reference is not None
                and cpp_raw["ok"]
                and reference == cpp_raw["ipa"],
                "token_ids_equal": mfa["usable"]
                and candidate["usable"]
                and mfa["token_ids"] == candidate["token_ids"],
                "cpp_postprocess_matches_project_string": cpp_raw["ok"]
                and cpp_final["ok"]
                and normalize_ipa_for_vocab(cpp_raw["ipa"]) == cpp_final["ipa"],
            }
        )
    return results


def run(
    root: Path,
    config: Path,
    binary: Path,
    output: Path,
    *,
    mfa: str,
    jobs: int = 1,
    timeout: float = 300,
    runner: Callable[..., dict[str, Any]] | None = None,
    inspector: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"preserve existing output; choose a new directory: {output}")
    if jobs <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("jobs and process timeout must be positive")
    inputs: dict[str, Any] = {}
    with stage_progress("Verify reviewed C++ sources, executable and official model identities"):
        cfg = json.loads(_identity(config, inputs).read_text(encoding="utf-8"))
        official = json.loads(_identity(root / cfg["official_config"], inputs).read_text())
        binary = _identity(binary, inputs)
        with binary.open("rb") as handle:
            header = handle.read(20)
        if (
            len(header) != 20
            or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[18:20], "little") != 62
        ):
            raise ValueError("the supplied binary must be the rebuilt x86-64 ELF executable")
        if not os.access(binary, os.X_OK):
            raise ValueError("rebuilt binary is not executable")
        for name, sha in cfg["source_sha256"].items():
            _identity(binary.parent / name, inputs, sha)
        models = {
            lang: _identity(root / asset["path"], inputs, asset["sha256"])
            for lang, asset in official["models"].items()
        }
        vocab = load_phoneme_vocab(_identity(root / cfg["vocab"], inputs, cfg["vocab_sha256"]))
        if set(models) != {"en", "es", "pt"} or set(cfg["cases"]) != set(models):
            raise ValueError("exactly the three language-specific models and case lists required")
        for cases in cfg["cases"].values():
            if (
                not cases
                or len(set(cases)) != len(cases)
                or any(
                    not isinstance(s, str)
                    or not s.strip()
                    or s != s.strip()
                    or any(c in s for c in "\r\n\t")
                    for s in cases
                )
            ):
                raise ValueError("cases must be distinct nonempty single-line surfaces")
        for path in (
            Path(__file__),
            Path(__file__).parents[3] / "scripts/compare_mmservice_g2p_smoke.py",
            Path(__file__).with_name("coverage.py"),
            Path(__file__).parents[1] / "training/g2p_prep.py",
            Path(__file__).parents[1] / "training/mfa_audit.py",
        ):
            _identity(path, inputs)
        if any(
            Path(p).is_relative_to(output)
            or (Path(p).parent != root and output.is_relative_to(Path(p).parent))
            for p in inputs
        ):
            raise ValueError("output overlaps protected source/model/config inputs")
    output.mkdir(parents=True)
    print(f"New diagnostic output: {output}", flush=True)
    evidence = [output / "input_cases.json"]
    _save(evidence[0], cfg["cases"])
    versions: dict[str, str | None] = {}
    for name in ("montreal-forced-aligner", "pynini"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report: dict[str, Any] = {
        "status": "running",
        "scope": "three_language_small_list_unmodified_cpp_smoke",
        "output_dir": str(output),
        "inputs": inputs,
        "by_language": {},
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "versions": versions,
        },
        "replacement_verified": False,
        "full_corpus_compared": False,
        "gpu_used": False,
        "return_file": str(output / "return_bundle.zip"),
        "limitations": [
            "MFA agreement measures compatibility, not independently correct pronunciation.",
            "Raw official FST copies unchanged; no attached symbols or Spanish repair.",
            "MFA uses project-normalized isolated words; phrases concatenate per-word Top1.",
            "Numeric/empty lexical inputs are unresolved under the project label policy.",
            "Process times include startup/model loading; not throughput speedup estimates.",
        ],
    }
    comparisons: list[dict[str, Any]] = []
    execute = runner or _command
    try:
        for lang, model in models.items():
            directory = output / lang
            directory.mkdir()
            cases = cfg["cases"][lang]
            with stage_progress(f"Inspect unmodified official FST and symbol tables: {lang}"):
                inventory = (inspector or inspect_model)(model, directory / "model")
                for name, digest in inventory["member_sha256"].items():
                    _identity(directory / "model" / name, inputs, digest)
                _save(directory / "model_inventory.json", inventory)
            words = sorted(
                {
                    w
                    for s in cases
                    if not any(c.isdigit() for c in s)
                    for w in extract_word_tokens(s)
                }
            )
            word_path, case_path = directory / "words.txt", directory / "surfaces.txt"
            word_path.write_text("\n".join(words) + "\n", encoding="utf-8")
            case_path.write_text("\n".join(cases) + "\n", encoding="utf-8")
            dictionary = directory / "mfa_raw.dict"
            command = [
                mfa,
                "g2p",
                "--num_jobs",
                str(jobs),
                "--num_pronunciations",
                "1",
                str(word_path),
                str(model),
                str(dictionary),
            ]
            env = {**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work" / lang)}
            with stage_progress(
                f"Official MFA Top1 {lang}: {len(words)} unique words; see mfa logs"
            ):
                reference = execute(
                    command,
                    directory / "mfa.stdout.log",
                    directory / "mfa.stderr.log",
                    timeout=timeout,
                    env=env,
                )
            if reference["returncode"] != 0 or not dictionary.is_file():
                raise RuntimeError(f"official MFA baseline failed for {lang}; see {directory}")
            processes = {}
            rows = {}
            for mode in ("raw", "normalized"):
                destination = directory / f"cpp_{mode}.jsonl"
                command = [
                    str(binary),
                    str(case_path),
                    str(directory / "model/model.fst"),
                    str(directory / "model/phones.sym"),
                ]
                if mode == "raw":
                    command.append("--no-postprocess")
                with stage_progress(f"Unmodified C++ {lang} {mode}: {len(cases)} surfaces"):
                    process = execute(
                        command,
                        destination,
                        directory / f"cpp_{mode}.stderr.log",
                        timeout=timeout,
                        env=dict(os.environ),
                    )
                processes[mode] = process
                rows[mode] = _cpp_rows(destination, cases, process)
            result = compare_cases(cases, dictionary, rows["raw"], rows["normalized"], vocab)
            comparisons.extend({"language": lang, **r} for r in result)
            metrics = {
                "cases": len(result),
                "mfa_usable": sum(r["mfa"]["usable"] for r in result),
                "cpp_ok": sum(r["cpp_normalized"]["ok"] for r in result),
                "cpp_usable": sum(r["cpp_normalized"]["usable"] for r in result),
                "both_usable": sum(
                    r["mfa"]["usable"] and r["cpp_normalized"]["usable"] for r in result
                ),
                "input_policy_blocked": sum(r["numeric_or_empty_input"] for r in result),
                "raw_exact": sum(r["raw_exact"] for r in result),
                "token_ids_equal": sum(r["token_ids_equal"] for r in result),
                "both_unusable": sum(
                    not r["mfa"]["usable"] and not r["cpp_normalized"]["usable"] for r in result
                ),
            }
            report["by_language"][lang] = {
                "counts": metrics,
                "model": inventory,
                "mfa_process": reference,
                "cpp_processes": processes,
            }
            print(f"{lang}: {metrics}", flush=True)
        with stage_progress("Recheck immutable inputs and package small comparison evidence"):
            for filename, identity in inputs.items():
                if _sha(Path(filename)) != identity["sha256"]:
                    raise ValueError(f"input changed during smoke comparison: {filename}")
            report["status"] = "completed"
            report["all_cases_usable_and_token_ids_equal"] = all(
                r["token_ids_equal"] for r in comparisons
            )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        report.update(status="failed", error=str(error))
    report["totals"] = dict(
        Counter(
            {
                key: sum(g["counts"][key] for g in report["by_language"].values())
                for key in (
                    "cases",
                    "mfa_usable",
                    "cpp_ok",
                    "cpp_usable",
                    "both_usable",
                    "input_policy_blocked",
                    "raw_exact",
                    "token_ids_equal",
                    "both_unusable",
                )
            }
        )
    )
    comparison_path = output / "comparison.jsonl"
    comparison_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in comparisons), encoding="utf-8"
    )
    _save(output / "report.json", report)
    for lang in models:
        directory = output / lang
        if directory.is_dir():
            evidence.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    evidence.extend([comparison_path, output / "report.json"])
    checks = "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in evidence)
    (output / "sha256.txt").write_text(checks, encoding="utf-8")
    with zipfile.ZipFile(
        output / "return_bundle.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in [*evidence, output / "sha256.txt"]:
            archive.write(path, path.relative_to(output).as_posix())
    summary = {
        k: report[k]
        for k in ("status", "output_dir", "totals", "replacement_verified", "return_file")
    }
    summary["by_language"] = {
        lang: {
            "counts": group["counts"],
            "architecture": group["model"]["metadata"]["architecture"],
            "unicode_decomposition": group["model"]["metadata"].get("unicode_decomposition"),
            "input_symbols_embedded": group["model"]["input_symbols_embedded"],
            "cpp_inferred_input_mode": group["model"]["cpp_inferred_input_mode"],
        }
        for lang, group in report["by_language"].items()
    }
    if "error" in report:
        summary["error"] = report["error"]
    return summary
