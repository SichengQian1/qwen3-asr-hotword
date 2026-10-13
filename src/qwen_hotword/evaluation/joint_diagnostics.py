"""Paired shortlist/ranking/gate audit from saved retrieval results; no inference."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.joint_diagnostic_inputs import check_pair, load_run, validate_row
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import LANGUAGES
from qwen_hotword.inference.external_keyword_retrieval import _write_json
from qwen_hotword.inference.external_keyword_topk_replay import _select, _validated_source_gate
from qwen_hotword.training.spanish_capacity import _sha, _verified

COUNT_FIELDS = (
    "sample_count",
    "expected_target_sample_pairs",
    "gated_target_hits",
    "raw_target_hits",
    "selected_all_words",
    "selected_words_absent_from_transcript",
)


def gate_failures(match: dict[str, Any], gate: dict[str, Any]) -> list[str]:
    return [
        reason
        for reason, failed in (
            ("score", match["score"] < gate["threshold"]),
            ("edit_ratio", match["edit_ratio"] > gate["maximum_edit_ratio"]),
            ("posterior", match["posterior_confidence"] < gate["minimum_posterior_confidence"]),
        )
        if failed
    ]


def target_state(
    identifier: str, row: dict[str, Any], selected: set[str], gate: dict[str, Any]
) -> str:
    if identifier in selected:
        return "selected"
    match = next((m for m in row["raw_ranked_matches"] if m["hotword_id"] == identifier), None)
    if match is None:
        return "not_in_saved_shortlist"
    failures = gate_failures(match, gate)
    return "gate:" + "+".join(failures) if failures else "qualified_below_topk"


def _profile() -> dict[str, Any]:
    return {
        "old": Counter(),
        "new": Counter(),
        "paired": Counter(),
        "current_target_states": Counter(),
        "lost_target_destinations": Counter(),
        "current_absent_by_origin_language": Counter(),
        "added_absent_by_origin_language": Counter(),
        "added_absent_phone_lengths": Counter(),
        "added_absent_edit_ratio_zero": 0,
        "current_absent_words": Counter(),
        "added_absent_words": Counter(),
    }


def _counts(replay: dict[str, Any], targets: set[str], k: int) -> Counter[str]:
    expected = set(replay["expected_hotword_ids"])
    selected = set(replay[f"top{k}_selected_hotword_ids"])
    return Counter(
        {
            "sample_count": 1,
            "expected_target_sample_pairs": len(expected & targets),
            "gated_target_hits": len(selected & expected & targets),
            "raw_target_hits": len(set(replay[f"raw_top{k}_hotword_ids"]) & expected & targets),
            "selected_all_words": len(selected),
            "selected_words_absent_from_transcript": len(selected - expected),
        }
    )


def _match_info(identifier: str, row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_ranked_matches"]
    rank = next((i for i, m in enumerate(raw, 1) if m["hotword_id"] == identifier), None)
    entry = run["canonical"][identifier]
    if rank is None:
        return {
            "word": entry.surface,
            "raw_rank": None,
            "target_phonemes": list(entry.phoneme_tokens),
        }
    match = raw[rank - 1]
    variant = run["index"][match["matched_variant_id"]]
    start, end = match.get("decoded_start"), match.get("decoded_end")
    tokens = row["decoded_tokens"]
    valid = isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(tokens)
    return {
        "word": entry.surface,
        "raw_rank": rank,
        **{
            key: match.get(key)
            for key in (
                "score",
                "edit_ratio",
                "posterior_confidence",
                "matched_language",
                "matched_pronunciation",
                "decoded_start",
                "decoded_end",
                "start_step",
                "end_step",
            )
        },
        "query_phonemes": list(variant.phoneme_tokens),
        "matched_ctc_tokens": tokens[start:end] if valid else None,
        "ctc_context": tokens[max(0, start - 4) : end + 4] if valid else None,
    }


def _update(
    profile: dict[str, Any],
    old_row: dict[str, Any],
    new_row: dict[str, Any],
    old_replay: dict[str, Any],
    new_replay: dict[str, Any],
    targets: set[str],
    new: dict[str, Any],
    fillers: set[str],
    gate: dict[str, Any],
    k: int,
) -> tuple[set[str], set[str], set[str]]:
    a = set(old_replay[f"top{k}_selected_hotword_ids"])
    b = set(new_replay[f"top{k}_selected_hotword_ids"])
    expected = set(new_replay["expected_hotword_ids"])
    wanted = targets & expected
    lost, gained = (a - b) & wanted, (b - a) & wanted
    old_bad, new_bad = a - expected, b - expected
    added_bad, removed_bad = new_bad - old_bad, old_bad - new_bad
    profile["old"].update(_counts(old_replay, targets, k))
    profile["new"].update(_counts(new_replay, targets, k))
    profile["paired"].update(
        {
            "lost_targets": len(lost),
            "gained_targets": len(gained),
            "retained_target_hits": len(a & b & wanted),
            "added_absent_candidates": len(added_bad),
            "removed_absent_candidates": len(removed_bad),
            "raw_topk_target_losses": len(
                (
                    set(old_replay[f"raw_top{k}_hotword_ids"])
                    - set(new_replay[f"raw_top{k}_hotword_ids"])
                )
                & wanted
            ),
            "raw_topk_target_gains": len(
                (
                    set(new_replay[f"raw_top{k}_hotword_ids"])
                    - set(old_replay[f"raw_top{k}_hotword_ids"])
                )
                & wanted
            ),
        }
    )
    old_matches = {m["hotword_id"]: m for m in old_row["raw_ranked_matches"]}
    new_matches = {m["hotword_id"]: m for m in new_row["raw_ranked_matches"]}
    for identifier in wanted & old_matches.keys() & new_matches.keys():
        before, after = old_matches[identifier], new_matches[identifier]
        profile["paired"]["targets_present_in_both_saved_lists"] += 1
        profile["paired"]["target_score_changed"] += int(before["score"] != after["score"])
        profile["paired"]["target_ctc_span_changed"] += int(
            (before.get("decoded_start"), before.get("decoded_end"))
            != (after.get("decoded_start"), after.get("decoded_end"))
        )
    for identifier in wanted:
        state = target_state(identifier, new_row, b, gate)
        profile["current_target_states"][state] += 1
        if identifier in lost:
            profile["lost_target_destinations"][state] += 1
    # Conditional replay of the SAVED list only, not an Anchor query or a new run.
    retained_only = {
        m["hotword_id"]
        for m in _select(
            [m for m in new_row["raw_ranked_matches"] if m["hotword_id"] not in fillers],
            gate=gate,
            top_k=k,
        )
    }
    profile["paired"]["lost_targets_recovered_removing_fillers_from_saved_list"] += len(
        lost & retained_only
    )
    matches = {m["hotword_id"]: m for m in new_row["raw_ranked_matches"]}
    for identifier in new_bad:
        match = matches[identifier]
        variant = new["index"][match["matched_variant_id"]]
        lang = "pt" if variant.language == "pt-BR" else variant.language
        origin = "filler" if identifier in fillers else "reviewed_target"
        category = f"{origin}::{lang}"
        profile["current_absent_by_origin_language"][category] += 1
        profile["current_absent_words"][identifier] += 1
        if identifier in added_bad:
            profile["added_absent_by_origin_language"][category] += 1
            profile["added_absent_words"][identifier] += 1
            length = len(variant.token_ids)
            bin_name = (
                "1-3" if length <= 3 else "4-5" if length <= 5 else "6-8" if length <= 8 else "9+"
            )
            profile["added_absent_phone_lengths"][bin_name] += 1
            profile["added_absent_edit_ratio_zero"] += int(match["edit_ratio"] == 0)
    return lost, gained, added_bad


def _finish(profile: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    result = dict(profile)
    for key in ("old", "new"):
        counts = dict(profile[key])
        denom, selected = counts["expected_target_sample_pairs"], counts["selected_all_words"]
        counts.update(
            target_recall=counts["gated_target_hits"] / denom if denom else None,
            selected_reference_precision=(
                selected - counts["selected_words_absent_from_transcript"]
            )
            / selected
            if selected
            else None,
        )
        result[key] = counts
    for key in ("current_absent_words", "added_absent_words"):
        result[key + "_top10"] = [
            {"word": new["canonical"][identifier].surface, "count": count}
            for identifier, count in sorted(profile[key].items(), key=lambda p: (-p[1], p[0]))[:10]
        ]
        del result[key]
    return result


def diagnose_joint_runs(
    old_dir: Path, new_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identities: dict[str, Any] = {}
    old, new = load_run(old_dir, identities), load_run(new_dir, identities)
    check_pair(old, new)
    fillers = new["canonical"].keys() - old["canonical"].keys()
    summaries: dict[str, dict[str, Any]] = {}
    by_dataset: dict[str, Any] = {}
    example_slots: dict[tuple[str, str], dict[str, Any]] = {}
    decoded_checks: dict[str, Counter[str]] = defaultdict(Counter)
    for name, group in sorted(new["plan"]["groups"].items()):
        lang = group["language"]
        gate = _validated_source_gate(group["config"])
        paths = [
            _verified(
                r["directory"], f"languages/{lang}/{name}/retrieval_details.jsonl", identities
            )
            for r in (old, new)
        ]
        profiles = {f"top{k}": _profile() for k in (5, 7)}
        for k in (5, 7):
            summaries.setdefault(lang, {}).setdefault(f"top{k}", _profile())
        with paths[0].open() as a, paths[1].open() as b:
            left = (json.loads(s) for s in a if s.strip())
            right = (json.loads(s) for s in b if s.strip())
            for before, after, record in zip_longest(left, right, group["records"]):
                if before is None or after is None or record is None:
                    raise ValueError(f"saved sample count differs: {name}")
                a_replay = validate_row(before, record, old, name)
                b_replay = validate_row(after, record, new, name)
                if set(a_replay["expected_hotword_ids"]) != set(b_replay["expected_hotword_ids"]):
                    raise ValueError("fillers unexpectedly match a reference or targets changed")
                decoded_checks[lang]["samples"] += 1
                equal = before["decoded_token_ids"] == after["decoded_token_ids"]
                decoded_checks[lang][
                    "identical_token_sequences" if equal else "changed_token_sequences"
                ] += 1
                decoded_checks[lang]["identical_confidence_sequences"] += int(
                    before["decoded_confidences"] == after["decoded_confidences"]
                )
                for k in (5, 7):
                    for profile in (profiles[f"top{k}"], summaries[lang][f"top{k}"]):
                        lost, gained, added_bad = _update(
                            profile,
                            before,
                            after,
                            a_replay,
                            b_replay,
                            set(group["target_ids"]),
                            new,
                            fillers,
                            gate,
                            k,
                        )
                    if k == 7:
                        categories: dict[str, set[str]] = defaultdict(set)
                        for identifier in lost:
                            categories[
                                "lost:"
                                + target_state(
                                    identifier,
                                    after,
                                    set(b_replay["top7_selected_hotword_ids"]),
                                    gate,
                                )
                            ].add(identifier)
                        if gained:
                            categories["gained_target"].update(gained)
                        if added_bad & fillers:
                            categories["new_absent_filler"].update(added_bad & fillers)
                        for category, words in categories.items():
                            slot = (lang, category)
                            if slot in example_slots:
                                continue
                            identifier = sorted(words)[0]
                            example_slots[slot] = {
                                "dataset": name,
                                "sample_id": record["sample_id"],
                                "category": category,
                                "reference": record["reference_text"],
                                "audio_path": record["audio_path"],
                                "tokens_unchanged": equal,
                                "word_before": _match_info(identifier, before, old)
                                if identifier in old["canonical"]
                                else None,
                                "word_after": _match_info(identifier, after, new),
                                "old_top7": [
                                    _match_info(i, before, old)
                                    for i in a_replay["top7_selected_hotword_ids"]
                                ],
                                "new_top7": [
                                    _match_info(i, after, new)
                                    for i in b_replay["top7_selected_hotword_ids"]
                                ],
                            }
        # Independently reproduce all six count fields from the returned source reports.
        for key, profile in profiles.items():
            for label, run in (("old", old), ("new", new)):
                saved = run["report"]["groups"][name]["current"][key]
                if any(profile[label][field] != saved[field] for field in COUNT_FIELDS):
                    raise ValueError(f"source report counts do not reproduce: {name}/{key}/{label}")
        by_dataset[name] = {key: _finish(v, new) for key, v in profiles.items()}
        print(f"Checked {name}: {len(group['records'])} paired samples", flush=True)
    by_language = {
        lang: {key: _finish(v, new) for key, v in profiles.items()}
        for lang, profiles in summaries.items()
    }
    for lang, profiles in by_language.items():
        for key, profile in profiles.items():
            for label, run in (("old", old), ("new", new)):
                saved = run["report"]["by_language"][lang]["current"][key]
                if any(profile[label][field] != saved[field] for field in COUNT_FIELDS):
                    raise ValueError("language aggregate counts do not reproduce")
    report = {
        "status": "completed",
        "old_run": str(old_dir),
        "new_run": str(new_dir),
        "old_surfaces": len(old["canonical"]),
        "new_surfaces": len(new["canonical"]),
        "new_fillers": len(fillers),
        "inputs": identities,
        "by_language": by_language,
        "by_dataset": by_dataset,
        "ctc_sequence_comparison": dict(decoded_checks),
        "source_metrics_reproduced": True,
        "model_loaded": False,
        "audio_read": False,
        "source_outputs_modified": False,
        "retrieval_rerun": False,
        "example_rule": (
            "first saved sample per language/category in sorted dataset order; lowest word ID"
        ),
        "limitations": [
            "Not-in-shortlist cannot distinguish no Anchor hits from pre-Top64 pruning.",
            "Top64 counts pronunciation variants; surface collapse can leave fewer words.",
            "Gate failures form disjoint combinations; qualified_below_topk passes all gates.",
            "Removing fillers replays the saved list only; it cannot recover pruned targets.",
            "Candidate language is winning pronunciation provenance, not detected spoken language.",
            "CTC local spans have no word boundaries; matching a subword is not confirmed here.",
            "Reference-absent candidates are not proof of final ASR hallucination or audio truth.",
            "Changed CTC sequences/confidences limit attribution solely to candidate competition.",
            "Recorded audio/Head/metadata hashes checked; no audio or model bytes reopened.",
            "Original runs did not hash model weight shards; equal backbone weights are assumed.",
        ],
        "diagnostic_code_sha256": {
            p.name: _sha(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("joint_diagnostic_inputs.py"),
            )
        },
    }
    return report, [example_slots[k] for k in sorted(example_slots)]


def write_diagnostics(report: dict[str, Any], examples: list[dict[str, Any]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if any(output.resolve().is_relative_to(Path(report[k])) for k in ("old_run", "new_run")):
        raise ValueError("diagnostic output must be outside source runs")
    # Recheck the small artifacts read during the scan; never rehash external audio/model assets.
    for filename, identity in report["inputs"].items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError(f"source artifact changed during diagnostics: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "report.json", report)
    _write_json(output / "examples.json", examples)
    compact = {
        key: report[key]
        for key in (
            "status",
            "old_surfaces",
            "new_surfaces",
            "new_fillers",
            "ctc_sequence_comparison",
            "source_metrics_reproduced",
        )
    }
    compact["by_language_top7"] = {
        lang: {
            k: v
            for k, v in report["by_language"][lang]["top7"].items()
            if not k.endswith("_words_top10")
        }
        for lang in LANGUAGES
    }
    _write_json(output / "summary.json", compact)
    seal(output)
