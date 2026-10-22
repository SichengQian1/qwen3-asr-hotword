"""Compare saved native C++ results with explicitly named project G2P policies."""

from __future__ import annotations

import csv
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    normalize_ipa_for_vocab,
)
from qwen_hotword.phonemes.mmservice_smoke import _command, _phones, _save, _sha
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_mfa_repair import (
    finalize_spanish_mfa_repair,
    prepare_spanish_mfa_repair,
    repair_spanish_pronunciation,
)
from qwen_hotword.training.stage_progress import stage_progress


def _identity(path: Path, identities: dict[str, str], expected: str | None = None) -> None:
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"input SHA256 changed: {path}")
    identities[str(path.resolve())] = digest


def _dictionary(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for word, phones in load_mfa_dictionary(path).items():
        result.setdefault(normalize_training_text(word), set()).update(phones)
    return result


def verify_source(
    root: Path, source: Path, cfg: dict[str, Any], identities: dict[str, str]
) -> dict[str, Any]:
    """Check the returned evidence identity and every protected original input."""
    _identity(source / "report.json", identities, cfg["source_report_sha256"])
    _identity(source / "sha256.txt", identities, cfg["source_manifest_sha256"])
    members: set[str] = set()
    for line in (source / "sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = (source / name).resolve()
        if name in members or not path.is_relative_to(source) or path == source:
            raise ValueError("invalid/duplicate source evidence member")
        members.add(name)
        _identity(path, identities, digest)
    required = {"report.json", "input_cases.json", "comparison.jsonl"}
    required.update(
        f"{lang}/{name}"
        for lang in ("en", "es", "pt")
        for name in ("words.txt", "mfa_raw.dict", "cpp_raw.jsonl", "cpp_normalized.jsonl")
    )
    if not required <= members:
        raise ValueError("source evidence manifest is incomplete")
    report: dict[str, Any] = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "completed"
        or report.get("scope") != "three_language_small_list_unmodified_cpp_smoke"
        or Path(report["output_dir"]).resolve() != source
        or set(report["by_language"]) != {"en", "es", "pt"}
    ):
        raise ValueError("source must be the completed, unchanged three-language smoke run")
    for name, item in report["inputs"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"source input is outside the requested repository: {path}")
        _identity(path, identities, item["sha256"])
    return report


def compare_pipeline(
    cases: list[str],
    dictionary: Path,
    cpp_rows: list[dict[str, Any]],
    vocab: PhonemeVocab,
    language: str,
    provenance: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Native output and an explicit caller-adapter diagnostic remain separate."""
    phones = _dictionary(dictionary)
    if [row.get("word") for row in cpp_rows] != cases:
        raise ValueError("saved C++ surfaces/count/order differ from frozen cases")
    result = []
    for surface, cpp in zip(cases, cpp_rows, strict=True):
        words = extract_word_tokens(surface)
        reasons = []
        if not words:
            reasons.append("empty_lexical_input")
        if any(c.isdigit() for c in surface):
            reasons.append("unresolved_digit")
        missing = sorted({w for w in words if not phones.get(w)})
        ambiguous = sorted({w for w in words if len(phones.get(w, ())) > 1})
        if missing:
            reasons.append("missing_word")
        if ambiguous:
            reasons.append("ambiguous_word")
        ipa = None if reasons else " ".join(next(iter(phones[w])) for w in words)
        reference = _phones(ipa, vocab)
        if ipa is not None and not reference["usable"]:
            reasons.append("oov_or_empty_pronunciation")
        native = _phones(cpp["ipa"] if cpp["ok"] else None, vocab)
        # This is diagnostic output only. It is never substituted for native output.
        cleanup: dict[str, int] = {}
        adapted_ipa = cpp["ipa"] if cpp["ok"] else None
        if language == "es" and adapted_ipa is not None:
            adapted_ipa, changes = repair_spanish_pronunciation(adapted_ipa, ())
            cleanup = dict(changes)
        adapted = _phones(adapted_ipa, vocab)
        comparisons = {}
        for name, candidate in (("native", native), ("existing_es_cleanup", adapted)):
            comparisons[name] = {
                # Compare literal native/default IPA, before the shared tokenizer
                # can normalize it again and conceal a postprocessor discrepancy.
                "ipa_equal": reference["usable"]
                and candidate["usable"]
                and normalize_ipa_for_vocab(str(ipa)) == candidate["raw_ipa"],
                "token_ids_equal": reference["usable"]
                and candidate["usable"]
                and reference["token_ids"] == candidate["token_ids"],
            }
        result.append(
            {
                "language": language,
                "surface": surface,
                "project_words": words,
                "reference_policy_reasons": reasons,
                "missing_words": missing,
                "ambiguous_words": ambiguous,
                "word_provenance": [
                    {
                        "word": w,
                        **(provenance or {}).get(
                            w,
                            {
                                "resolution": "base_exact"
                                if len(phones.get(w, ())) == 1
                                else ("ambiguous" if phones.get(w) else "missing")
                            },
                        ),
                    }
                    for w in words
                ],
                "reference": reference,
                "cpp_ok": cpp["ok"],
                "cpp_process_failed": bool(cpp.get("process_failed", False)),
                "cpp_native": native,
                "cpp_with_existing_es_cleanup": adapted,
                "diagnostic_cleanup": cleanup,
                "cpp_accepts_policy_blocked_input": cpp["ok"]
                and any(r in reasons for r in ("unresolved_digit", "empty_lexical_input")),
                "comparison": comparisons,
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "cases": len(rows),
        "reference_usable": sum(r["reference"]["usable"] for r in rows),
        "cpp_ok": sum(r["cpp_ok"] for r in rows),
        "cpp_native_usable": sum(r["cpp_native"]["usable"] for r in rows),
        "native_ipa_equal": sum(r["comparison"]["native"]["ipa_equal"] for r in rows),
        "native_token_ids_equal": sum(r["comparison"]["native"]["token_ids_equal"] for r in rows),
        "reference_usable_native_not_equal": sum(
            r["reference"]["usable"] and not r["comparison"]["native"]["token_ids_equal"]
            for r in rows
        ),
        "after_existing_es_cleanup_token_ids_equal": sum(
            r["comparison"]["existing_es_cleanup"]["token_ids_equal"] for r in rows
        ),
        "cpp_accepts_policy_blocked_input": sum(
            r["cpp_accepts_policy_blocked_input"] for r in rows
        ),
    }


def probe_spanish(model_dir: Path, words_file: Path, output: Path) -> dict[str, Any]:
    """Run only in aligner: official rewriter, fixed FST/symbols, one flag varied."""
    version = importlib.metadata.version("montreal-forced-aligner")
    if version != "3.4.0":
        raise ValueError(f"reviewed probe requires MFA 3.4.0; found {version}")
    generator = importlib.import_module("montreal_forced_aligner.g2p.generator")
    pynini = importlib.import_module("pynini")
    pywrapfst = importlib.import_module("pywrapfst")
    metadata = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("architecture") != "phonetisaurus" or not metadata.get("unicode_decomposition"):
        raise ValueError("expected the reviewed Spanish Phonetisaurus model/metadata")
    source_file = Path(str(generator.__file__))
    source_sha = _sha(source_file)
    original = pynini.Fst.read(str(model_dir / "model.fst"))
    embedded = original.input_symbols()
    external = pywrapfst.SymbolTable.read_text(str(model_dir / "graphemes.sym"))
    phones = pywrapfst.SymbolTable.read_text(str(model_dir / "phones.sym"))
    symbol_differences = []
    for line in (model_dir / "graphemes.sym").read_text(encoding="utf-8").splitlines():
        symbol, number = line.rsplit(maxsplit=1)
        found = None if embedded is None else embedded.find(symbol)
        if found != int(number):
            symbol_differences.append(
                {"symbol": symbol, "external_id": int(number), "embedded_id": found}
            )
    words = words_file.read_text(encoding="utf-8").splitlines()
    rows = []
    with stage_progress("Official rewriter flag probe", total=2 * len(words), every=1) as update:
        for decomposition in (True, False):
            fst = pynini.Fst.read(str(model_dir / "model.fst"))
            # Official setup attaches these tables in memory; disk assets stay intact.
            fst.set_input_symbols(external)
            fst.set_output_symbols(phones)
            rewriter = generator.PhonetisaurusRewriter(
                fst,
                external,
                phones,
                num_pronunciations=1,
                grapheme_order=metadata["grapheme_order"],
                sequence_separator=metadata.get("sequence_separator", "|"),
                graphemes=metadata["graphemes"],
                unicode_decomposition=decomposition,
            )
            for word in words:
                cleaned, missing = generator.clean_up_word(
                    word, metadata["graphemes"], unicode_decomposition=decomposition
                )
                rewrite_input = unicodedata.normalize("NFKD", cleaned) if decomposition else cleaned
                row = {
                    "word": word,
                    "unicode_decomposition": decomposition,
                    "cleaned_word": cleaned,
                    "missing_graphemes": sorted(missing),
                    "rewrite_input": rewrite_input,
                    "rewrite_codepoints": [f"U+{ord(c):04X}" for c in rewrite_input],
                }
                try:
                    hypotheses = rewriter(cleaned)
                    row["pronunciations"] = [p.pronunciation for p in hypotheses]
                    row["status"] = "ok" if hypotheses else "empty"
                except generator.rewrite.Error as error:
                    row.update(status="error", error_type=type(error).__name__, error=str(error))
                rows.append(row)
                update(len(rows))
    if _sha(source_file) != source_sha:
        raise ValueError("official rewriter source changed during probe")
    result = {
        "status": "completed",
        "mfa_version": version,
        "official_generator_source_sha256": source_sha,
        "official_generator_source": str(source_file),
        "changed_factor": "unicode_decomposition only; official external tables held fixed",
        "metadata_default": True,
        "external_symbol_count": external.num_symbols(),
        "embedded_symbol_count": None if embedded is None else embedded.num_symbols(),
        "symbol_tables_equal": embedded is not None
        and embedded.num_symbols() == external.num_symbols()
        and not symbol_differences,
        "symbol_differences": symbol_differences,
        "rows": rows,
    }
    _save(output, result)
    return result


def run(
    root: Path,
    config: Path,
    output: Path,
    *,
    mfa: str,
    jobs: int = 1,
    timeout: float = 300,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"preserve existing output; choose a new directory: {output}")
    if jobs <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("jobs and timeout must be positive")
    identities: dict[str, str] = {}
    with stage_progress("Verify sealed smoke evidence and original inputs (no audio)"):
        _identity(config, identities)
        cfg = json.loads(config.read_text(encoding="utf-8"))
        source = (root / cfg["source_run"]).resolve()
        if not source.is_relative_to(root) or output.is_relative_to(source):
            raise ValueError("invalid/overlapping source and output")
        saved = verify_source(root, source, cfg, identities)
        for relative in (
            "src/qwen_hotword/phonemes/mmservice_pipeline.py",
            "scripts/compare_mmservice_g2p_pipeline.py",
            "src/qwen_hotword/training/spanish_mfa_repair.py",
        ):
            _identity(root / relative, identities)
        cases = json.loads((source / "input_cases.json").read_text(encoding="utf-8"))
        official_path = root / "configs/joint_4000_mfa.workzone.json"
        official = json.loads(official_path.read_text(encoding="utf-8"))
        vocab_path = root / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
        vocab = load_phoneme_vocab(vocab_path)
    output.mkdir(parents=True)
    print(f"New pipeline diagnostic output: {output}", flush=True)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "saved_27_cases_project_policy_vs_native_cpp",
        "source_run": str(source),
        "source_report_sha256": _sha(source / "report.json"),
        "output_dir": str(output),
        "inputs": identities,
        "replacement_verified": False,
        "full_corpus_compared": False,
        "gpu_used": False,
        "cpp_rerun": False,
        "reference_policy": {
            "common": "project normalization, isolated MFA Top1, phrase concatenation, IPA/vocab",
            "es": "existing prepare/finalize_spanish_mfa_repair; exact before missing-word proxy",
            "input_rejection": "numeric/empty lexical input, missing/ambiguous/OOV pronunciation",
            "not_training_row_audit": "excludes corpus-only h/connector/audio/CTC-length checks",
        },
        "cpp_variants": {
            "native": "saved default C++ IPA; no extra language repair or policy filtering",
            "existing_es_cleanup": "diagnostic adapter only: ES tilde removal, no proxy rules",
        },
        "limitations": [
            "Small fixed list; agreement is compatibility, not real pronunciation accuracy.",
            "No existing training dictionary or hotword table is replaced or revalidated here.",
            "The caller-adapter result cannot be reported as native C++ compatibility.",
            "This compares reconstructed project rules, not all historical dictionary bytes.",
            "Shared ID conversion normalizes again; literal final IPA equality is also reported.",
        ],
    }
    execute = runner or _command
    all_rows: list[dict[str, Any]] = []
    try:
        with stage_progress("Reuse small saved evidence; prepare existing Spanish repair plan"):
            for line in (source / "sha256.txt").read_text(encoding="utf-8").splitlines():
                _, name = line.split(maxsplit=1)
                destination = output / "source_evidence" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)
            shutil.copyfile(source / "sha256.txt", output / "source_evidence/sha256.txt")
            counts = Counter(w for s in cases["es"] for w in extract_word_tokens(s))
            count_path = output / "diagnostic_word_counts.tsv"
            count_path.write_text(
                "word\tcount\n" + "".join(f"{w}\t{n}\n" for w, n in sorted(counts.items())),
                encoding="utf-8",
            )
            plan = prepare_spanish_mfa_repair(
                source / "es/words.txt", count_path, source / "es/mfa_raw.dict", output / "es_plan"
            )
        proxy_dictionary = output / "es_proxy_raw.dict"
        with stage_progress(
            f"MFA Top1 only for {plan['proxy_words_for_mfa']} missing-word proxies"
        ):
            if plan["proxy_words_for_mfa"]:
                command = [
                    mfa,
                    "g2p",
                    "--num_jobs",
                    str(jobs),
                    "--num_pronunciations",
                    "1",
                    plan["proxy_words_path"],
                    str(root / official["models"]["es"]["path"]),
                    str(proxy_dictionary),
                ]
                process = execute(
                    command,
                    output / "es_proxy.stdout.log",
                    output / "es_proxy.stderr.log",
                    timeout=timeout,
                    env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work")},
                )
                report["proxy_process"] = process
                if process["returncode"] != 0 or not proxy_dictionary.is_file():
                    raise RuntimeError("Spanish proxy MFA failed; see proxy logs")
            else:
                proxy_dictionary.write_text("", encoding="utf-8")
        with stage_progress(
            "Finalize diagnostic dictionary with existing Spanish production rules"
        ):
            final = finalize_spanish_mfa_repair(
                source / "es/words.txt",
                count_path,
                source / "es/mfa_raw.dict",
                Path(plan["repair_plan_path"]),
                proxy_dictionary,
                vocab_path,
                output / "es_final",
            )
            report["spanish_repair"] = {"prepare": plan, "finalize": final}
            with Path(plan["repair_plan_path"]).open(encoding="utf-8", newline="") as handle:
                provenance = {row["word"]: row for row in csv.DictReader(handle, delimiter="\t")}
        script = root / "scripts/compare_mmservice_g2p_pipeline.py"
        with stage_progress("Official Spanish NFC/NFKD single-factor probe; see child word counts"):
            command = [
                sys.executable,
                "-B",
                str(script),
                "--probe-worker",
                "--probe-model",
                str(source / "es/model"),
                "--probe-words",
                str(source / "es/words.txt"),
                "--probe-output",
                str(output / "es_unicode_probe.json"),
            ]
            process = execute(
                command,
                output / "probe.stdout.log",
                output / "probe.stderr.log",
                timeout=timeout,
                env=dict(os.environ),
            )
            report["probe_process"] = process
            if process["returncode"] != 0 or not (output / "es_unicode_probe.json").is_file():
                raise RuntimeError("official rewriter probe failed; see probe logs")
            probe = json.loads((output / "es_unicode_probe.json").read_text(encoding="utf-8"))
            probe_words = (source / "es/words.txt").read_text(encoding="utf-8").splitlines()
            expected_probe = {(word, flag) for word in probe_words for flag in (True, False)}
            actual_probe = [(r["word"], r["unicode_decomposition"]) for r in probe["rows"]]
            if (
                probe.get("status") != "completed"
                or len(actual_probe) != len(expected_probe)
                or set(actual_probe) != expected_probe
            ):
                raise ValueError("official rewriter probe returned incomplete/duplicate cases")
            base_es = _dictionary(source / "es/mfa_raw.dict")
            defaults = [r for r in probe["rows"] if r["unicode_decomposition"]]
            report["probe_default_reproduces_saved_mfa"] = all(
                set(row.get("pronunciations", [])) == base_es.get(row["word"], set())
                for row in defaults
            )
            report["probe_summary"] = {
                "symbol_tables_equal": probe["symbol_tables_equal"],
                "by_decomposition": {
                    str(flag): {
                        "cases": sum(r["unicode_decomposition"] == flag for r in probe["rows"]),
                        "ok": sum(
                            r["unicode_decomposition"] == flag and r["status"] == "ok"
                            for r in probe["rows"]
                        ),
                    }
                    for flag in (True, False)
                },
            }
        with stage_progress(
            "Compare project final IPA and IDs against saved native C++",
            total=sum(len(group) for group in cases.values()),
            every=1,
        ) as update:
            report["by_language"] = {}
            for lang in ("en", "es", "pt"):
                cpp = [
                    json.loads(s)
                    for s in (source / lang / "cpp_normalized.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                dictionary = (
                    Path(final["dictionary_path"])
                    if lang == "es"
                    else (source / lang / "mfa_raw.dict")
                )
                rows = compare_pipeline(
                    cases[lang], dictionary, cpp, vocab, lang, provenance if lang == "es" else None
                )
                all_rows.extend(rows)
                report["by_language"][lang] = summarize(rows)
                update(len(all_rows))
        with stage_progress("Recheck protected source/model/code identities"):
            for name, digest in identities.items():
                if _sha(Path(name)) != digest:
                    raise ValueError(f"input changed during pipeline comparison: {name}")
        report["totals"] = summarize(all_rows)
        report["source_totals"] = saved["totals"]
        report["status"] = "completed"
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        report.update(status="failed", error=str(error))
    _save(output / "report.json", report)
    (output / "comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8"
    )
    with stage_progress("Package small pipeline comparison evidence (no models)"):
        files = sorted(
            p
            for p in output.rglob("*")
            if p.is_file() and "mfa_work" not in p.relative_to(output).parts
        )
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files),
            encoding="utf-8",
        )
        with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in [*files, output / "sha256.txt"]:
                archive.write(path, path.relative_to(output).as_posix())
    return {key: report[key] for key in ("status", "output_dir", "replacement_verified")} | {
        "by_language": report.get("by_language", {}),
        "probe_summary": report.get("probe_summary"),
        "probe_default_reproduces_saved_mfa": report.get("probe_default_reproduces_saved_mfa"),
        "error": report.get("error"),
        "return_file": str(output / "return_bundle.zip"),
    }
