"""Freeze small English failure/control examples from two sealed joint runs."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.joint_diagnostic_inputs import check_pair, load_run, validate_row
from qwen_hotword.evaluation.joint_diagnostics import COUNT_FIELDS, _counts, target_state
from qwen_hotword.evaluation.retrieval_case_trace import make_index, trace_case
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.inference.external_keyword_retrieval import _write_json
from qwen_hotword.inference.external_keyword_topk_replay import _validated_source_gate
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified

QUOTAS = {"not_in_saved_shortlist": 2, "gate_score_only": 2, "gate_other": 1, "selected_control": 1}
SEED = "english-failure-cases-20261013-v1"
REPLAY_CODE = (
    "hotwords/anchor_index.py",
    "hotwords/scoring.py",
    "training/edit_distance.py",
    "inference/reviewed_joint.py",
    "inference/external_keyword_topk_replay.py",
    "inference/hotword_prompt.py",
    "phonemes/coverage.py",
)


def stratum(state: str) -> str | None:
    if state == "not_in_saved_shortlist":
        return state
    if state == "gate:score":
        return "gate_score_only"
    if state.startswith("gate:"):
        return "gate_other"
    return "selected_control" if state == "selected" else None


def keep_example(slots: list[dict[str, Any]], item: dict[str, Any], count: int) -> None:
    """Smallest hash per distinct audio ID, then smallest hashes in the stratum."""
    by_sample = {v["sample_id"]: v for v in slots}
    previous = by_sample.get(item["sample_id"])
    if previous is None or item["selection_key"] < previous["selection_key"]:
        by_sample[item["sample_id"]] = item
    slots[:] = sorted(by_sample.values(), key=lambda v: v["selection_key"])[:count]


def check_local_code(run: dict[str, Any], identities: dict[str, Any]) -> None:
    package = Path(__file__).resolve().parents[1]
    for relative in REPLAY_CODE:
        _checked(package / relative, identities, run["plan"]["code_sha256"][relative])
    # Older run manifests did not record this dependency. Save its current
    # identity and require actual candidate/ranking reproduction on every case.
    _checked(package / "hotwords/exact_automaton.py", identities)


def collect_cases(
    root: Path,
    old_dir: Path,
    new_dir: Path,
    seed: str = SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identities: dict[str, Any] = {}
    old, new = load_run(old_dir, identities), load_run(new_dir, identities)
    check_pair(old, new)
    for run in (old, new):
        check_local_code(run, identities)
    vocab_path = (root / new["plan"]["configuration"]["vocab"]).resolve()
    vocab = load_phoneme_vocab(
        _checked(vocab_path, identities, new["plan"]["inputs"][str(vocab_path)]["sha256"])
    )
    for run in (old, new):
        for entry in run["variants"]:
            if tuple(vocab.tokens[i] for i in entry.token_ids) != entry.phoneme_tokens:
                raise ValueError("indexed phonemes do not agree with vocabulary IDs")
    summaries: dict[str, Any] = {}
    chosen: list[dict[str, Any]] = []
    aggregate: dict[str, Counter[str]] = {label: Counter() for label in ("old", "new")}
    for name, group in sorted(new["plan"]["groups"].items()):
        if group["language"] != "en":
            continue
        gate = _validated_source_gate(group["config"])
        paths = [
            _verified(run["directory"], f"languages/en/{name}/retrieval_details.jsonl", identities)
            for run in (old, new)
        ]
        slots: dict[str, list[dict[str, Any]]] = {s: [] for s in QUOTAS}
        states: Counter[str] = Counter()
        pools: Counter[str] = Counter()
        sample_pools: dict[str, set[str]] = defaultdict(set)
        counts: dict[str, Counter[str]] = {label: Counter() for label in ("old", "new")}
        targets = set(group["target_ids"])
        with paths[0].open() as left, paths[1].open() as right:
            a = (json.loads(s) for s in left if s.strip())
            b = (json.loads(s) for s in right if s.strip())
            for before, after, record in zip_longest(a, b, group["records"]):
                if before is None or after is None or record is None:
                    raise ValueError(f"saved sample count differs: {name}")
                replays = [
                    validate_row(row, record, run, name)
                    for row, run in ((before, old), (after, new))
                ]
                if set(replays[0]["expected_hotword_ids"]) != set(
                    replays[1]["expected_hotword_ids"]
                ):
                    raise ValueError("fillers matched reference or target identity changed")
                for row in (before, after):
                    ids, phones, confidence = (
                        row[k]
                        for k in ("decoded_token_ids", "decoded_tokens", "decoded_confidences")
                    )
                    if any(not isinstance(i, int) or not 0 <= i < len(vocab.tokens) for i in ids):
                        raise ValueError("decoded token ID outside vocabulary")
                    if [vocab.tokens[i] for i in ids] != phones or any(
                        not math.isfinite(p) or not 0 <= p <= 1 for p in confidence
                    ):
                        raise ValueError("decoded phones/confidences invalid")
                for label, replay in zip(("old", "new"), replays, strict=True):
                    counts[label].update(_counts(replay, targets, 7))
                selected = [set(r["top7_selected_hotword_ids"]) for r in replays]
                for identifier in sorted(set(after["expected_hotword_ids"]) & targets):
                    state = target_state(identifier, after, selected[1], gate)
                    states[state] += 1
                    bucket = stratum(state)
                    if bucket is None:
                        continue
                    pools[bucket] += 1
                    sample_pools[bucket].add(record["sample_id"])
                    identity = [seed, name, bucket, record["sample_id"], identifier]
                    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                    keep_example(
                        slots[bucket],
                        {
                            "dataset": name,
                            "sample_id": record["sample_id"],
                            "target_id": identifier,
                            "stratum": bucket,
                            "selection_key": key,
                            "old_state": target_state(identifier, before, selected[0], gate),
                            "new_state": state,
                            "before": before,
                            "after": after,
                            "replays": replays,
                        },
                        QUOTAS[bucket],
                    )
        for label, run in (("old", old), ("new", new)):
            saved = run["report"]["groups"][name]["current"]["top7"]
            if any(counts[label][f] != saved[f] for f in COUNT_FIELDS):
                raise ValueError(f"English source report does not reproduce: {name}/{label}")
            aggregate[label].update(counts[label])
        summaries[name] = {
            "counts": counts,
            "target_states": states,
            "strata": {
                s: {
                    "eligible_pairs": pools[s],
                    "eligible_audio_ids": len(sample_pools[s]),
                    "selected": len(slots[s]),
                    "requested": n,
                    "shortfall": n - len(slots[s]),
                }
                for s, n in QUOTAS.items()
            },
        }
        chosen.extend(item for s in QUOTAS for item in slots[s])
        print(f"Checked {name}: {counts['new']['sample_count']} paired samples", flush=True)
    if not summaries:
        raise ValueError("no English groups in source run")
    for label, run in (("old", old), ("new", new)):
        saved = run["report"]["by_language"]["en"]["current"]["top7"]
        if any(aggregate[label][f] != saved[f] for f in COUNT_FIELDS):
            raise ValueError("English aggregate report does not reproduce")
    examples = []
    indexes: dict[tuple[str, str], Any] = {}
    for item in chosen:
        name, identifier = item["dataset"], item["target_id"]
        group = new["plan"]["groups"][name]
        gate = _validated_source_gate(group["config"])
        example = {k: v for k, v in item.items() if k not in {"before", "after", "replays"}}
        example.update(
            word=new["canonical"][identifier].surface,
            reference=item["after"]["reference_text"],
            audio_path=item["after"]["audio_path"],
            audio_sha256=item["after"]["audio_sha256"],
            gate=gate,
            ctc_sequences_identical=all(
                item["before"][k] == item["after"][k]
                for k in ("decoded_token_ids", "decoded_tokens", "decoded_confidences")
            ),
        )
        for number, (label, run, row_key) in enumerate(
            (("old", old, "before"), ("new", new, "after"))
        ):
            config = run["plan"]["groups"][name]["config"]
            index_key = (label, json.dumps(config["retrieval"], sort_keys=True))
            if index_key not in indexes:
                indexes[index_key] = make_index(run, config)
            row = item[row_key]
            print(f"Tracing {name}/{item['sample_id']}: {example['word']} ({label})", flush=True)
            trace = trace_case(run, row, identifier, config, gate, indexes[index_key])
            selected_ids = set(item["replays"][number]["top7_selected_hotword_ids"])
            expected = set(row["expected_hotword_ids"])
            trace.update(
                source_row_sha256=row["row_sha256"],
                decoded_tokens=row["decoded_tokens"],
                decoded_confidences=row["decoded_confidences"],
                competitors=[
                    {
                        **m,
                        "raw_rank": i,
                        "selected_top7": m["hotword_id"] in selected_ids,
                        "present_in_reference": m["hotword_id"] in expected,
                    }
                    for i, m in enumerate(row["raw_ranked_matches"], 1)
                    if i <= 10 or m["hotword_id"] in selected_ids
                ],
            )
            example[label] = trace
        examples.append(example)
    report = {
        "status": "completed",
        "scope": "english_saved_ctc_failure_case_audit",
        "old_run": str(old_dir),
        "new_run": str(new_dir),
        "seed": seed,
        "quotas_per_dataset": QUOTAS,
        "by_dataset": summaries,
        "aggregate_top7": aggregate,
        "example_count": len(examples),
        "unique_audio_count": len({(e["dataset"], e["sample_id"]) for e in examples}),
        "inputs": identities,
        "source_metrics_reproduced": True,
        "model_loaded": False,
        "audio_read": False,
        "existing_outputs_modified": False,
        "sampling_policy": (
            "SHA256(seed,dataset,stratum,sample_id,target_id); distinct audio per stratum"
        ),
        "shortfall_policy": "take all available distinct audio IDs; no cross-stratum backfill",
        "code_sha256": {
            p.name: _sha(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("retrieval_case_trace.py"),
                Path(__file__).with_name("joint_diagnostic_inputs.py"),
                Path(__file__).with_name("joint_diagnostics.py"),
                Path(__file__).resolve().parents[3] / "scripts/extract_english_failure_cases.py",
            )
        },
        "limitations": [
            "Stratified target/audio pairs are examples, not population prevalence estimates.",
            "Audio may recur across strata; old/new comparisons always use the same target/audio.",
            "No audio listening, new MFA labels, CTC inference, or final ASR decoding performed.",
            "Uncapped Anchor replay explains sampled omissions, not all population misses.",
            "Full-start target-only search keeps existing width limits; no new global ranking.",
            "Best matching CTC span may be another word; it is not an acoustic alignment.",
            "Edits are relative to table pronunciation, not proof of model/label mistakes.",
            "No real audio timestamps reconstructed; all new positions are decoded-token indices.",
            "Head/audio hashes checked as recorded only; original backbone shards not hashed.",
            "Original runs did not hash exact-automaton code; sampled replay must reproduce.",
        ],
    }
    return report, examples


def write_cases(report: dict[str, Any], examples: list[dict[str, Any]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if any(
        output.resolve().is_relative_to(Path(report[k]).resolve()) for k in ("old_run", "new_run")
    ):
        raise ValueError("case output must be outside source runs")
    for filename, identity in report["inputs"].items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError(f"source changed during audit: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "summary.json", report)
    _write_json(output / "examples.json", examples)
    lines = [
        "# 英语漏召回配对样本",
        "",
        "先读参考、词表音素，再看保存匹配和诊断匹配。",
        "位置均为CTC token下标，不是时间或真实词边界。全起点搜索不是线上结果。",
        "",
    ]
    for e in examples:
        lines.extend(
            [
                f"## {e['dataset']} / {e['sample_id']} / {e['word']}",
                "",
                f"分层：{e['stratum']}；旧→新：{e['old_state']} → {e['new_state']}",
                f"参考：{e['reference']}",
                f"音频：{e['audio_path']}",
                "",
            ]
        )
        for label in ("old", "new"):
            trace = e[label]
            lines.extend([f"### {label}", "", "CTC：" + " ".join(trace["decoded_tokens"]), ""])
            for variant in trace["variants"]:
                lines.extend(
                    [
                        f"词表音素（{variant['language']}）：" + " ".join(variant["phonemes"]),
                        f"Anchor：{variant['anchor_state']}；"
                        f"截断前排名：{variant['anchor_rank_before_cap']}",
                    ]
                )
                for title, key in (
                    ("保存的局部匹配", "saved_guided_match"),
                    ("诊断全起点匹配", "diagnostic_full_start_search"),
                ):
                    m = variant[key]
                    if m is None:
                        lines.append(f"{title}：无")
                        continue
                    lines.extend(
                        [
                            f"{title}：{' '.join(m['ctc_span'])}；score={m['score']:.6f}，"
                            f"edit={m['edit_ratio']:.6f}，posterior={m['posterior_confidence']:.6f}",
                            f"失败门控：{m['failed_gates']}；编辑："
                            + json.dumps(m["edits_table_to_ctc"], ensure_ascii=False),
                            "",
                        ]
                    )
            lines.extend(["前列竞争词（*为最终Top7，参考存在按整表词面判断）：", ""])
            for m in trace["competitors"]:
                lines.append(
                    f"- {m['raw_rank']}. {m['surface']}"
                    f"{' *' if m['selected_top7'] else ''}，score={m['score']:.6f}，"
                    f"参考存在={m['present_in_reference']}"
                )
            lines.append("")
    (output / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "target_terms.txt").write_text(
        "\n".join(sorted({e["word"] for e in examples})) + "\n", encoding="utf-8"
    )
    seal(output)
    with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "summary.json",
            "examples.json",
            "examples.md",
            "target_terms.txt",
            "sha256.txt",
        ):
            archive.write(output / name, name)
