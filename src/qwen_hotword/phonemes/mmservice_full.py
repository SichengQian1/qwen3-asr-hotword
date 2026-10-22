"""Read-only G2P comparison over every retained variant of the frozen joint table."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    normalize_ipa_for_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.phonemes.mmservice_pipeline import _dictionary, compare_pipeline
from qwen_hotword.phonemes.mmservice_process import execute
from qwen_hotword.phonemes.mmservice_smoke import _cpp_rows, _phones, _save, _sha, inspect_model
from qwen_hotword.training.g2p_prep import extract_word_tokens
from qwen_hotword.training.spanish_capacity import _verified
from qwen_hotword.training.spanish_mfa_repair import (
    finalize_spanish_mfa_repair,
    prepare_spanish_mfa_repair,
)
from qwen_hotword.training.stage_progress import stage_progress

LANGUAGES = ("en", "es", "pt")
ALIASES = {"en": "en", "es": "es", "pt": "pt", "pt-BR": "pt"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _checked(path: Path, identities: dict[str, Any], expected: str | None = None) -> Path:
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    identities[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return path


def inventory(
    table: dict[str, Any],
    vocab: PhonemeVocab,
    surfaces: int,
    variants: int,
) -> list[dict[str, Any]]:
    seen = set()
    rows: list[dict[str, Any]] = []
    for entry in table["entries"]:
        key = entry["normalized"]
        if (
            not key
            or key != " ".join(normalize_match_words(entry["surface"]))
            or key in seen
            or not entry["variants"]
        ):
            raise ValueError("invalid/duplicate source surface or empty variants")
        seen.add(key)
        for variant in entry["variants"]:
            surface = variant["surface"]
            if (
                variant["language"] not in ALIASES
                or variant["normalized"] != key
                or " ".join(normalize_match_words(surface)) != key
                or any(c in surface for c in "\t\r\n")
                or variant["decision"] != "retained"
                or variant.get("role", "target") not in {"target", "filler"}
            ):
                raise ValueError(f"invalid source language/surface/variant: {surface}")
            if surface != surface.strip():
                raise ValueError(
                    "source surface has leading/trailing whitespace; do not rewrite it"
                )
            ids = variant.get("token_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(i) is not int or i < 2 or i >= len(vocab.tokens) for i in ids)
            ):
                raise ValueError(f"invalid saved variant token IDs: {surface}")
            ipa = variant.get("pronunciation")
            if not isinstance(ipa, str) or not ipa.strip():
                raise ValueError(f"missing saved pronunciation: {surface}")
            converted = tokenize_ipa_to_vocab(ipa, vocab)
            if converted.oov_units or converted.token_ids != ids:
                raise ValueError(f"saved pronunciation/ID inconsistency: {surface}")
            rows.append(
                {
                    "variant_index": len(rows),
                    "language": ALIASES[variant["language"]],
                    "declared_language": variant["language"],
                    "surface": surface,
                    "normalized": entry["normalized"],
                    "role": variant.get("role", "target"),
                    "saved_ipa": ipa,
                    "saved_token_ids": ids,
                    "source_groups": variant.get("source_groups", []),
                    "source": variant.get("source"),
                }
            )
    if len(rows) != variants:
        raise ValueError(f"expected {variants} retained variants; found {len(rows)}")
    if len(seen) != surfaces or {r["language"] for r in rows} != set(LANGUAGES):
        raise ValueError("source surface count or language coverage differs")
    return rows


def _reference(
    surface: str, dictionary: dict[str, set[str]], vocab: PhonemeVocab
) -> dict[str, Any]:
    words = extract_word_tokens(surface)
    blocked = not words or any(c.isdigit() for c in surface)
    missing = sorted({w for w in words if not dictionary.get(w)})
    ambiguous = sorted({w for w in words if len(dictionary.get(w, ())) > 1})
    raw = (
        None
        if blocked or missing or ambiguous
        else " ".join(next(iter(dictionary[w])) for w in words)
    )
    return {**_phones(raw, vocab), "blocked": blocked, "missing": missing, "ambiguous": ambiguous}


def compare_language(
    language: str,
    saved: list[dict[str, Any]],
    directory: Path,
    project_dictionary: Path,
    vocab: PhonemeVocab,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    surfaces = (directory / "surfaces.txt").read_text(encoding="utf-8").splitlines()
    words = (directory / "words.txt").read_text(encoding="utf-8").splitlines()
    cpp_raw = _cpp_rows(directory / "cpp_raw.jsonl", surfaces, {"returncode": 0})
    cpp_final = _cpp_rows(directory / "cpp_final.jsonl", surfaces, {"returncode": 0})
    cpp_words = _cpp_rows(directory / "cpp_words.jsonl", words, {"returncode": 0})
    official_words = [
        json.loads(s)
        for s in (directory / "exact/words.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [r.get("word") for r in official_words] != words:
        raise ValueError("official exact-input word output count/order differs")
    word_rows = []
    for official, cpp in zip(official_words, cpp_words, strict=True):
        a, b = (
            _phones(official["ipa"] if official["ok"] else None, vocab),
            _phones(cpp["ipa"] if cpp["ok"] else None, vocab),
        )
        word_rows.append(
            {
                "language": language,
                "word": official["word"],
                "official_exact": a,
                "cpp_raw": b,
                "official_status": official["status"],
                "raw_equal": official["ok"] and cpp["ok"] and official["ipa"] == cpp["ipa"],
                "ids_equal": a["usable"] and b["usable"] and a["token_ids"] == b["token_ids"],
                "both_failed": not official["ok"] and not cpp["ok"],
            }
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in saved:
        groups[variant["normalized"]].append(variant)
    case_keys = {r["surface"]: r["normalized"] for r in saved}
    exact = _dictionary(directory / "exact/raw.dict")
    default = _dictionary(directory / "mfa_default.dict")
    provenance = None
    if language == "es" and (directory / "plan/repair_plan.tsv").is_file():
        with (directory / "plan/repair_plan.tsv").open(encoding="utf-8", newline="") as handle:
            provenance = {row["word"]: row for row in csv.DictReader(handle, delimiter="\t")}
    project = compare_pipeline(surfaces, project_dictionary, cpp_final, vocab, language, provenance)
    by_word = {r["word"]: r for r in word_rows}
    cases = []
    for row, raw in zip(project, cpp_raw, strict=True):
        surface = row["surface"]
        core = _reference(surface, exact, vocab)
        normal = _reference(surface, default, vocab)
        native, adapted = row["cpp_native"], row["cpp_with_existing_es_cleanup"]
        refs = groups[case_keys[surface]]
        snapshots = []
        for variant in refs:
            snapshots.append(
                {
                    "variant_index": variant["variant_index"],
                    "native_ids_equal": native["usable"]
                    and native["token_ids"] == variant["saved_token_ids"],
                    "after_es_cleanup_ids_equal": adapted["usable"]
                    and adapted["token_ids"] == variant["saved_token_ids"],
                    "project_rules_ids_equal": row["reference"]["usable"]
                    and row["reference"]["token_ids"] == variant["saved_token_ids"],
                }
            )
        same_input_raw = (
            core["raw_ipa"] is not None and raw["ok"] and (core["raw_ipa"] == raw["ipa"])
        )
        postprocess = (
            raw["ok"]
            and row["cpp_ok"]
            and (normalize_ipa_for_vocab(raw["ipa"]) == native["raw_ipa"])
        )
        flags = []
        if not row["cpp_ok"]:
            flags.append("cpp_failed")
        elif not native["usable"]:
            flags.append("cpp_label_oov_or_empty")
        if raw["ok"] and row["cpp_ok"] and not postprocess:
            flags.append("cpp_postprocess_string_diff")
        if normal["missing"]:
            flags.append("official_default_missing_word")
        if row["cpp_accepts_policy_blocked_input"]:
            flags.append("cpp_accepts_policy_blocked_input")
        if core["raw_ipa"] is not None and not same_input_raw:
            flags.append("surface_path_raw_diff")
        if all(by_word[w]["raw_equal"] for w in row["project_words"] if w in by_word) and (
            core["raw_ipa"] is not None and not same_input_raw
        ):
            flags.append("canonical_word_engine_agrees_but_surface_path_diff")
        if row["reference"]["usable"] and not row["comparison"]["native"]["token_ids_equal"]:
            flags.append("project_rules_native_ids_diff")
        if not any(s["native_ids_equal"] for s in snapshots):
            flags.append("saved_table_native_ids_diff")
        cases.append(
            {
                **row,
                "normalized": case_keys[surface],
                "official_default": normal,
                "official_same_input": core,
                "cpp_raw_ipa": raw["ipa"] if raw["ok"] else None,
                "same_input_surface_raw_equal": same_input_raw,
                "same_input_surface_final_ids_equal": core["usable"]
                and native["usable"]
                and core["token_ids"] == native["token_ids"],
                "cpp_postprocess_string_equal": postprocess,
                "saved_variants": snapshots,
                "difference_flags": flags,
            }
        )
    return cases, word_rows


def summarize(
    cases: list[dict[str, Any]],
    words: list[dict[str, Any]],
    saved: list[dict[str, Any]],
) -> dict[str, Any]:
    case_map = {(r["language"], r["surface"]): r for r in cases}
    variant_rows = []
    for variant in saved:
        case = case_map[(variant["language"], variant["surface"])]
        match = next(
            s for s in case["saved_variants"] if s["variant_index"] == variant["variant_index"]
        )
        variant_rows.append({**variant, **match})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["language"], case["normalized"])].append(case)
    return {
        "raw_surface_inputs": len(cases),
        "language_normalized_surfaces": len(grouped),
        "saved_variant_rows": len(saved),
        "eligible_inputs": sum(not r["official_same_input"]["blocked"] for r in cases),
        "cpp_ok": sum(r["cpp_ok"] for r in cases),
        "cpp_label_usable": sum(r["cpp_native"]["usable"] for r in cases),
        "official_default_label_usable": sum(r["official_default"]["usable"] for r in cases),
        "project_rules_label_usable": sum(r["reference"]["usable"] for r in cases),
        "official_same_input_raw_available": sum(
            r["official_same_input"]["raw_ipa"] is not None for r in cases
        ),
        "same_input_surface_raw_equal": sum(r["same_input_surface_raw_equal"] for r in cases),
        "same_input_surface_final_ids_equal": sum(
            r["same_input_surface_final_ids_equal"] for r in cases
        ),
        "cpp_postprocess_string_equal": sum(r["cpp_postprocess_string_equal"] for r in cases),
        "project_rules_native_ids_equal": sum(
            r["comparison"]["native"]["token_ids_equal"] for r in cases
        ),
        "project_rules_after_es_cleanup_ids_equal": sum(
            r["comparison"]["existing_es_cleanup"]["token_ids_equal"] for r in cases
        ),
        "saved_variant_native_ids_equal": sum(r["native_ids_equal"] for r in variant_rows),
        "saved_variant_after_es_cleanup_ids_equal": sum(
            r["after_es_cleanup_ids_equal"] for r in variant_rows
        ),
        "all_input_forms_match_some_saved_variant": sum(
            all(any(s["native_ids_equal"] for s in r["saved_variants"]) for r in group)
            for group in grouped.values()
        ),
        "all_input_forms_match_some_saved_variant_after_es_cleanup": sum(
            all(any(s["after_es_cleanup_ids_equal"] for s in r["saved_variants"]) for r in group)
            for group in grouped.values()
        ),
        "word_engine": {
            "unique_words": len(words),
            "official_ok": sum(r["official_status"] == "ok" for r in words),
            "cpp_ok": sum(r["cpp_raw"]["raw_ipa"] is not None for r in words),
            "raw_equal": sum(r["raw_equal"] for r in words),
            "ids_equal": sum(r["ids_equal"] for r in words),
            "both_failed": sum(r["both_failed"] for r in words),
        },
        "difference_flags": dict(Counter(flag for r in cases for flag in r["difference_flags"])),
        "by_role_variant_rows": {
            role: {
                "rows": sum(r["role"] == role for r in variant_rows),
                "native_ids_equal": sum(
                    r["role"] == role and r["native_ids_equal"] for r in variant_rows
                ),
                "after_es_cleanup_ids_equal": sum(
                    r["role"] == role and r["after_es_cleanup_ids_equal"] for r in variant_rows
                ),
            }
            for role in sorted({r["role"] for r in variant_rows})
        },
    }


def run(
    root: Path,
    config: Path,
    output: Path,
    *,
    mfa: str,
    jobs: int = 1,
    timeout: float = 1800,
    runner: Callable[..., dict[str, Any]] | None = None,
    inspector: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"preserve previous output; choose a new directory: {output}")
    if jobs < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("jobs and timeout must be positive")
    identities: dict[str, Any] = {}
    with stage_progress("Verify frozen 4000 table, all variants, C++ build and official models"):
        cfg = _json(_checked(config, identities))
        table_cfg = _json(
            _checked(root / cfg["table_config"], identities, cfg["table_config_sha256"])
        )
        cpp_cfg = _json(_checked(root / cfg["cpp_config"], identities, cfg["cpp_config_sha256"]))
        source = (root / table_cfg["source_run"]).resolve()
        _checked(source / "sha256.txt", identities)
        table_path = _checked(
            _verified(source, "reviewed_keyword_table.json", identities),
            identities,
            table_cfg["table_sha256"],
        )
        old_report = _json(_verified(source, "report.json", identities))
        review = old_report["review"]
        if (
            old_report.get("status") != "completed"
            or old_report.get("candidate_scope") != "joint_multilingual"
            or review.get("expanded_table_sha256") != table_cfg["table_sha256"]
            or review.get("active_unique_surfaces_per_audio") != table_cfg["expected_surfaces"]
        ):
            raise ValueError("completed sealed joint-table source identity differs")
        vocab_path = _checked(root / cpp_cfg["vocab"], identities, cpp_cfg["vocab_sha256"])
        vocab = load_phoneme_vocab(vocab_path)
        saved = inventory(
            _json(table_path), vocab, table_cfg["expected_surfaces"], cfg["expected_variants"]
        )
        binary = _checked(root / cfg["binary"], identities, cfg["binary_sha256"])
        with binary.open("rb") as binary_handle:
            header = binary_handle.read(20)
        if (
            header[:6] != b"\x7fELF\x02\x01"
            or len(header) != 20
            or int.from_bytes(header[18:20], "little") != 62
            or not os.access(binary, os.X_OK)
        ):
            raise ValueError("reviewed executable x86-64 binary required")
        for name, digest in cpp_cfg["source_sha256"].items():
            _checked(binary.parent / name, identities, digest)
        if set(table_cfg["models"]) != set(LANGUAGES):
            raise ValueError("exactly three fixed official models required")
        models = {
            lang: _checked(root / asset["path"], identities, asset["sha256"])
            for lang, asset in table_cfg["models"].items()
        }
        for relative in (
            "src/qwen_hotword/phonemes/mmservice_full.py",
            "src/qwen_hotword/phonemes/mmservice_rewriter.py",
            "src/qwen_hotword/phonemes/mmservice_process.py",
            "src/qwen_hotword/phonemes/mmservice_pipeline.py",
            "src/qwen_hotword/phonemes/mmservice_smoke.py",
            "src/qwen_hotword/phonemes/coverage.py",
            "src/qwen_hotword/training/g2p_prep.py",
            "src/qwen_hotword/training/mfa_audit.py",
            "src/qwen_hotword/training/spanish_mfa_repair.py",
            "src/qwen_hotword/training/spanish_capacity.py",
            "src/qwen_hotword/training/stage_progress.py",
            "src/qwen_hotword/inference/hotword_prompt.py",
            "scripts/compare_mmservice_g2p_full.py",
        ):
            _checked(root / relative, identities)
        if any(
            output.is_relative_to(Path(p).parent) or Path(p).is_relative_to(output)
            for p in identities
        ):
            raise ValueError("output overlaps protected source/config/build/model assets")
    output.mkdir(parents=True)
    print(f"New full-table G2P output: {output}", flush=True)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "frozen_joint_4000_all_retained_variants",
        "output_dir": str(output),
        "inputs": identities,
        "processes": [],
        "by_language": {},
        "source_unique_surfaces": table_cfg["expected_surfaces"],
        "saved_variant_rows": len(saved),
        "replacement_verified": False,
        "gpu_used": False,
        "audio_read": False,
        "ctc_or_table_modified": False,
        "comparison_layers": {
            "word_engine": "same submitted project word and FST/input/phone tables; "
            "official exact-token Top1 vs C++ raw (C++ internal preprocessing remains active)",
            "official_default": "unaltered MFA CLI policy; exact dictionary key lookup",
            "project_rules": "default CLI plus existing missing-word ES proxy/cleanup rules",
            "saved_table": "every frozen retained pronunciation; variant and any-variant counts",
            "adapter": "diagnostic extra ES cleanup on native output; never native equivalence",
        },
        "limitations": [
            "Full current hotword table, not full training dictionaries or independent accuracy.",
            "Raw strings can differ by separators or tie-breaking; ID and failure counts separate.",
            "Both failures or OOV/partial labels are never counted as valid ID agreement.",
            "Snapshot disagreements can originate in old labels or input/proxy policies.",
            "Canonical-word differences alone do not prove an inference-core defect: "
            "C++ still preprocesses submitted words; its internal tokens are not observed here.",
            "Timings include process startup/loading and are not a production speed benchmark.",
        ],
    }
    call = runner or execute
    all_cases: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    def step(
        command: list[str],
        directory: Path,
        name: str,
        stdout_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        process = call(
            command,
            directory / (stdout_name or f"{name}.stdout.log"),
            directory / f"{name}.stderr.log",
            timeout=timeout,
            **kwargs,
        )
        report["processes"].append(process)
        if process["returncode"] != 0:
            raise RuntimeError(f"{name} failed/timeout; see logs in {directory}")

    try:
        with (output / "input_inventory.jsonl").open("x", encoding="utf-8") as handle:
            for row in saved:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for lang in LANGUAGES:
            directory = output / lang
            directory.mkdir()
            selected = [r for r in saved if r["language"] == lang]
            surfaces = sorted({r["surface"] for r in selected})
            words = sorted(
                {
                    w
                    for s in surfaces
                    if not any(c.isdigit() for c in s)
                    for w in extract_word_tokens(s)
                }
            )
            for name, items in (("surfaces", surfaces), ("words", words)):
                (directory / f"{name}.txt").write_text(
                    "\n".join(items) + ("\n" if items else ""), encoding="utf-8"
                )
            print(
                f"{lang}: {len(selected)} variants; {len(surfaces)} raw surfaces; "
                f"{len(words)} unique words",
                flush=True,
            )
            with stage_progress(f"{lang}: inspect/copy unchanged official model"):
                model = (inspector or inspect_model)(models[lang], directory / "model")
                for name, digest in model["member_sha256"].items():
                    _checked(directory / "model" / name, identities, digest)
                _save(directory / "model_inventory.json", model)
            for mode, filename, inputs in (
                ("raw", "cpp_raw", "surfaces"),
                ("final", "cpp_final", "surfaces"),
                ("words", "cpp_words", "words"),
            ):
                command = [
                    str(binary),
                    str(directory / f"{inputs}.txt"),
                    str(directory / "model/model.fst"),
                    str(directory / "model/phones.sym"),
                ]
                if mode != "final":
                    command.append("--no-postprocess")
                # Native stdout is JSONL; store it as evidence and show actual written rows.
                step(
                    command,
                    directory,
                    filename,
                    stdout_name=f"{filename}.jsonl",
                    env=dict(os.environ),
                    label=f"{lang}: C++ {mode} (rows written)",
                    rows_total=len(words) if inputs == "words" else len(surfaces),
                )
            command = [
                sys.executable,
                "-B",
                str(root / "scripts/compare_mmservice_g2p_full.py"),
                "--rewriter-worker",
                "--model-dir",
                str(directory / "model"),
                "--words",
                str(directory / "words.txt"),
                "--worker-output",
                str(directory / "exact"),
            ]
            step(
                command,
                directory,
                "exact",
                env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work" / f"exact_{lang}")},
                live=True,
                label=f"{lang}: official exact-input Top1 (child progress)",
            )
            exact_report = _json(directory / "exact/report.json")
            if exact_report.get("status") != "completed" or exact_report.get("words") != len(words):
                raise ValueError("official exact-input worker is incomplete")
            command = [
                mfa,
                "g2p",
                "--num_jobs",
                str(jobs),
                "--num_pronunciations",
                "1",
                str(directory / "words.txt"),
                str(models[lang]),
                str(directory / "mfa_default.dict"),
            ]
            if words:
                step(
                    command,
                    directory,
                    "mfa_default",
                    label=f"{lang}: MFA default Top1, {len(words)} words",
                    env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work" / lang)},
                    live=True,
                )
            else:
                (directory / "mfa_default.dict").write_text("", encoding="utf-8")
            project_dictionary = directory / "mfa_default.dict"
            default_words = _dictionary(project_dictionary)
            if lang == "es":
                counts = Counter(w for s in surfaces for w in extract_word_tokens(s))
                count_path = directory / "diagnostic_counts.tsv"
                count_path.write_text(
                    "word\tcount\n" + "".join(f"{w}\t{n}\n" for w, n in sorted(counts.items())),
                    encoding="utf-8",
                )
                with stage_progress("es: prepare existing missing-word proxy plan"):
                    plan = prepare_spanish_mfa_repair(
                        directory / "words.txt", count_path, project_dictionary, directory / "plan"
                    )
                proxy_dict = directory / "proxy_raw.dict"
                if plan["proxy_words_for_mfa"]:
                    command = [
                        mfa,
                        "g2p",
                        "--num_jobs",
                        str(jobs),
                        "--num_pronunciations",
                        "1",
                        plan["proxy_words_path"],
                        str(models[lang]),
                        str(proxy_dict),
                    ]
                    step(
                        command,
                        directory,
                        "proxy",
                        env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work/proxy")},
                        live=True,
                        label=f"es: MFA for {plan['proxy_words_for_mfa']} missing-word proxies",
                    )
                else:
                    proxy_dict.write_text("", encoding="utf-8")
                with stage_progress("es: finalize existing proxy/cleanup diagnostic dictionary"):
                    final = finalize_spanish_mfa_repair(
                        directory / "words.txt",
                        count_path,
                        project_dictionary,
                        Path(plan["repair_plan_path"]),
                        proxy_dict,
                        vocab_path,
                        directory / "final",
                    )
                    project_dictionary = Path(final["dictionary_path"])
                    report["spanish_repair"] = {"prepare": plan, "finalize": final}
            with stage_progress(f"{lang}: compare saved variants and all reference layers"):
                cases, word_rows = compare_language(
                    lang, selected, directory, project_dictionary, vocab
                )
                all_cases.extend(cases)
                all_words.extend(word_rows)
                report["by_language"][lang] = summarize(cases, word_rows, selected)
                report["by_language"][lang]["exact_input_worker"] = exact_report
                report["by_language"][lang]["mfa_default_word_inventory"] = {
                    "input_unique_words": len(words),
                    "output_unique_keys": len(default_words),
                    "missing_input_words": sorted(set(words) - default_words.keys()),
                    "unexpected_output_keys": sorted(default_words.keys() - set(words)),
                    "ambiguous_input_words": sorted(
                        w for w in words if len(default_words.get(w, ())) > 1
                    ),
                }
        with stage_progress("Recheck every frozen source/model/build/code identity"):
            for name, item in identities.items():
                if _sha(Path(name)) != item["sha256"]:
                    raise ValueError(f"input changed during full-table comparison: {name}")
        report["totals"] = summarize(all_cases, all_words, saved)
        report["status"] = "completed"
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
        KeyboardInterrupt,
    ) as error:
        report.update(status="failed", error=str(error) or type(error).__name__)
    _save(output / "report.json", report)
    for name, rows in (("case_comparison", all_cases), ("word_engine_comparison", all_words)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    with (output / "differences.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "language",
                "surface",
                "flags",
                "saved_variants",
                "project_rules_ipa",
                "cpp_native_ipa",
                "cpp_after_es_cleanup_ipa",
            ]
        )
        for row in all_cases:
            if row["difference_flags"]:
                writer.writerow(
                    [
                        row["language"],
                        row["surface"],
                        ";".join(row["difference_flags"]),
                        ",".join(str(s["variant_index"]) for s in row["saved_variants"]),
                        row["reference"]["raw_ipa"],
                        row["cpp_native"]["raw_ipa"],
                        row["cpp_with_existing_es_cleanup"]["raw_ipa"],
                    ]
                )
    with stage_progress("Package full-table text evidence (no FST, model ZIP, audio or cache)"):
        files = sorted(
            p
            for p in output.rglob("*")
            if p.is_file() and not {"model", "mfa_work"}.intersection(p.relative_to(output).parts)
        )
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files),
            encoding="utf-8",
        )
        with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for path in [*files, output / "sha256.txt"]:
                archive.write(path, path.relative_to(output).as_posix())
    return {
        "status": report["status"],
        "error": report.get("error"),
        "output_dir": str(output),
        "by_language": report["by_language"],
        "totals": report.get("totals"),
        "replacement_verified": False,
        "return_file": str(output / "return_bundle.zip"),
    }
