"""Freeze a small final-ASR prompt experiment from sealed English Top7 deliveries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.english_keywords import DATASETS, checked
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.training.spanish_capacity import _sha, _verified

GROUPS = tuple("ABCDEFGH")
QUOTAS = {"target_hit": 10, "target_miss": 5, "no_target": 5}
PLAIN = (
    "The following terms are spelling references. Use a term only when supported by the audio. "
    "Do not insert a term that is not spoken.\n{terms}"
)
PRIORITY = (
    "The following terms are spelling references. Priority indicates preference only when the "
    "audio supports multiple candidates. Do not insert a term that is not spoken.\n{terms}"
)
PAIRS = (
    ("A", "B"),
    ("B", "C"),
    ("B", "D"),
    ("C", "D"),
    ("E", "F"),
    ("D", "E"),
    ("G", "H"),
    ("A", "G"),
    ("B", "G"),
)


def key(text: str) -> str:
    return " ".join(normalize_match_words(text))


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def rank(seed: int, *values: str) -> str:
    return digest([seed, *values])


def deduplicate(items: list[dict[str, str]]) -> list[dict[str, str]]:
    result, seen = [], set()
    for item in items:
        surface = item["word"]
        if not key(surface) or any(c in surface for c in ("\n", "\r", "<|", "|>")):
            raise ValueError(f"invalid prompt term: {surface!r}")
        if key(surface) not in seen:
            result.append(dict(item))  # Keep the original surface and pronunciation bytes.
            seen.add(key(surface))
    return result


def oracle_terms(reference: str, targets: list[dict[str, str]]) -> list[dict[str, str]]:
    words = normalize_match_words(reference)
    present = []
    for item in targets:
        term = normalize_match_words(item["word"])
        starts = [i for i in range(len(words) - len(term) + 1) if words[i : i + len(term)] == term]
        if starts:
            present.append((starts[0], key(item["word"]), item["word"], item))
    return [dict(row[3]) for row in sorted(present, key=lambda row: row[:3])]


def conditions(sample: dict[str, Any], seed: int) -> dict[str, Any]:
    b = deduplicate(sample["top7"])
    d = sorted(
        b,
        key=lambda item: (
            rank(seed, "permutation", sample["uid"], key(item["word"])),
            key(item["word"]),
        ),
    )
    g = sample["oracle"]
    lists = {
        "A": [],
        "B": b,
        "C": list(reversed(b)),
        "D": d,
        "E": d,
        "F": d,
        "G": g,
        "H": list(reversed(g)),
    }
    result = {}
    for group, items in lists.items():
        entries: list[dict[str, Any]] = [dict(item) for item in items]
        for item in entries:
            item["priority"] = (
                "high"
                if group == "F" and key(item["word"]) == key(b[0]["word"])
                else "normal"
                if group in "EF"
                else None
            )
        lines = [
            f"{item['word']} — priority: {item['priority']}" if group in "EF" else item["word"]
            for item in entries
        ]
        prompt = (PRIORITY if group in "EF" else PLAIN).format(terms="\n".join(lines))
        result[group] = {"candidates": entries, "context": prompt if entries else ""}
    return result


def select_samples(
    pool: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take quotas first, then fill deficits in fixed stratum order, without replacement."""
    selected, summary = [], {}
    for dataset in DATASETS:
        buckets = {
            s: sorted(
                [r for r in pool if r["dataset"] == dataset and r["stratum"] == s],
                key=lambda r: (rank(seed, "selection", r["uid"]), r["uid"]),
            )
            for s in QUOTAS
        }
        taken = []
        for s, quota in QUOTAS.items():
            for row in buckets[s][:quota]:
                taken.append({**row, "requested_stratum": s, "fallback": False})
        used = {r["uid"] for r in taken}
        deficits = {s: max(0, n - len(buckets[s])) for s, n in QUOTAS.items()}
        for requested, count in deficits.items():
            for _ in range(count):
                candidates = [r for s in QUOTAS for r in buckets[s] if r["uid"] not in used]
                if not candidates:
                    break
                row = candidates[0]
                taken.append({**row, "requested_stratum": requested, "fallback": True})
                used.add(row["uid"])
        summary[dataset] = {
            "available": {s: len(rows) for s, rows in buckets.items()},
            "requested": QUOTAS,
            "initial_deficits": deficits,
            "selected_by_actual_stratum": dict(Counter(r["stratum"] for r in taken)),
            "selected": len(taken),
            "remaining_shortfall": 20 - len(taken),
        }
        selected.extend(sorted(taken, key=lambda r: (rank(seed, "execution", r["uid"]), r["uid"])))
    return selected, summary


def _model_files(model: Path, cfg: dict[str, Any], identities: dict[str, Any]) -> None:
    checked(model / "config.json", identities, cfg["model_config_sha256"])
    index = _json(
        checked(model / "model.safetensors.index.json", identities, cfg["model_index_sha256"])
    )
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("empty model weight index")
    required = {"preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"}
    for filename in sorted(required | shards):
        path = model / filename
        if not path.resolve().is_relative_to(model.resolve()):
            raise ValueError("model index points outside snapshot")
        print(f"Verifying model asset: {filename}", flush=True)
        checked(path, identities)
    # Include optional generation/tokenizer/template assets as part of the frozen model identity.
    for path in sorted(model.iterdir()):
        if (
            path.is_file()
            and path.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken"}
            and str(path) not in identities
        ):
            checked(path, identities)


def code_identity(root: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    names = [
        "evaluation/prompt_order.py",
        "evaluation/prompt_order_metrics.py",
        "inference/prompt_order_runtime.py",
        "inference/prompt_order_experiment.py",
        "inference/hotword_prompt.py",
        "inference/external_keyword_retrieval.py",
        "evaluation/english_keywords.py",
        "evaluation/wave_keywords.py",
        "evaluation/sd_keywords.py",
        "training/spanish_capacity.py",
        "training/edit_distance.py",
    ]
    paths = [root / "src/qwen_hotword" / n for n in names]
    paths.append(root / "scripts/run_english_prompt_order.py")
    return {str(p.relative_to(root)): _sha(p) for p in paths}


def make_plan(root: Path, config: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    cfg = _json(config)
    if cfg["language"] != "English" or cfg["max_new_tokens"] != 512:
        raise ValueError("this experiment fixes English and max_new_tokens=512")
    identities: dict[str, Any] = {}
    checked(config, identities)
    retrieval, tables, model = (root / cfg[k] for k in ("retrieval", "tables", "model"))
    old = _json(_verified(retrieval, "run_config.json", identities))
    old_report = _json(_verified(retrieval, "report.json", identities))
    source_count = cfg.get("source_samples_per_dataset", 2000)
    if (
        old_report.get("status") != "completed"
        or old_report.get("sample_count") != source_count * 3
        or old_report.get("table_sha256") != cfg["table_sha256"]
        or old_report.get("checkpoint_sha256") != cfg["retrieval_head_sha256"]
        or old_report.get("top7_exact_replay") is not True
        or "audio_root" in old["configuration"]
    ):
        raise ValueError("expected the completed clean English retrieval run")
    table_path = _verified(tables, "keyword_bias_phoneme.json", identities)
    checked(table_path, identities, cfg["table_sha256"])
    targets_path = _verified(tables, "targets_by_dataset.json", identities)
    checked(targets_path, identities, cfg["targets_sha256"])
    table, target_keys = _json(table_path), _json(targets_path)
    entries = deduplicate(
        [
            {"word": w, "phoneme": table["keyword_phonemes"][w]}
            for w in table["keyword_sets"]["all_keywords"]
        ]
    )
    if len(entries) != cfg.get("keyword_count", 3000) or set(target_keys) != set(DATASETS):
        raise ValueError("expected the fixed 3000 table and three target sets")
    by_key = {key(e["word"]): e for e in entries}
    pool, targets_by_dataset = [], {}
    for name in DATASETS:
        targets = [by_key[t] for t in target_keys[name]]
        targets_by_dataset[name] = targets
        source = root / cfg["data_root"] / name / "en"
        transcript = checked(source / "transcripts.txt", identities)
        if identities[str(transcript)]["sha256"] != old["inputs"][str(transcript)]["sha256"]:
            raise ValueError(f"{name}: reference text identity changed")
        records, _ = external.build_external_dataset(
            (external.SourceSpec(name, source / "wav", transcript),), language="en"
        )
        if (
            len(records) != source_count
            or [r.to_dict() for r in records] != old["groups"][name]["records"]
        ):
            raise ValueError(f"{name}: clean audio/reference mapping changed")
        delivery_path = _verified(retrieval, f"delivery/{name}_top7.json", identities)
        delivery = _json(delivery_path)
        if set(delivery) != {r.sample_id for r in records}:
            raise ValueError(f"{name}: Top7 IDs differ from audio IDs")
        for r in records:
            raw = delivery[r.sample_id]
            if not isinstance(raw, list) or len(raw) > 7:
                raise ValueError("Top7 delivery must contain at most seven entries")
            top7 = deduplicate(raw)
            for item in top7:
                if by_key.get(key(item["word"])) != item:
                    raise ValueError(f"Top7 entry differs from fixed table: {item}")
            oracle = oracle_terms(r.reference_text, targets)
            expected = {key(t["word"]) for t in oracle}
            hit = bool(expected & {key(t["word"]) for t in top7})
            pool.append(
                {
                    "uid": f"{name}/{r.sample_id}",
                    "dataset": name,
                    "sample_id": r.sample_id,
                    "audio_path": str(r.audio_path),
                    "reference": r.reference_text,
                    "transcript_path": str(transcript),
                    "candidate_source": str(delivery_path),
                    "candidate_duplicates_removed": len(raw) - len(top7),
                    "top7": top7,
                    "oracle": oracle,
                    "stratum": "target_hit" if hit else "target_miss" if oracle else "no_target",
                }
            )
    samples, sampling = select_samples(pool, cfg["seed"])
    repeat_uids = []
    for name in DATASETS:
        choices = [s for s in samples if s["dataset"] == name]
        informative = [s for s in choices if len(s["top7"]) >= 2]
        if choices:
            repeat_uids.append(
                min(informative or choices, key=lambda s: rank(cfg["seed"], "repeat", s["uid"]))[
                    "uid"
                ]
            )
    for sample in samples:
        path = Path(sample["audio_path"])
        checked(path, identities, old["inputs"][str(path)]["sha256"])
        sample["audio_sha256"] = identities[str(path)]["sha256"]
        sample["reference_sha256"] = hashlib.sha256(sample["reference"].encode()).hexdigest()
        sample["conditions"] = conditions(sample, cfg["seed"])
        sample["execution_order"] = sorted(
            GROUPS, key=lambda g: rank(cfg["seed"], sample["uid"], g)
        )
        sample["request_seed"] = int(rank(cfg["seed"], "request", sample["uid"])[:8], 16)
        sample["repeat_b"] = sample["uid"] in repeat_uids
    _model_files(model, cfg, identities)
    return {
        "schema_version": 1,
        "ready": len(samples) == 60,
        "configuration": cfg,
        "git_commit": external._git_commit(),
        "code_sha256": code_identity(root),
        "inputs": identities,
        "runtime_preflight": runtime,
        "sampling": sampling,
        "sampling_policy": "quota first; fallback target_hit,target_miss,no_target; no reuse",
        "oracle_tie_rule": "first normalized token position, normalized surface, original surface",
        "permutation_policy": "SHA256(seed,permutation,uid,normalized term) order; ties lexical",
        "repeat_policy": "one seeded sample per dataset, prefer >=2 Top7; same seed and B request",
        "targets_by_dataset": targets_by_dataset,
        "samples": samples,
        "main_calls": len(samples) * 8,
        "extra_repeat_calls": len(repeat_uids),
        "templates": {"plain": PLAIN, "priority": PRIORITY},
        "metric_normalization": "NFKC, casefold, Unicode punctuation to spaces, whitespace tokens",
        "new_asr_predictions_used_for_selection": False,
    }


def preparation_report(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "status": "frozen" if plan["ready"] else "insufficient_samples",
        "output_dir": str(output),
        "sample_count": len(plan["samples"]),
        "main_calls": plan["main_calls"],
        "extra_repeat_calls": plan["extra_repeat_calls"],
        "sampling": plan["sampling"],
        "model_loaded": False,
        "top7_less_than_two": sum(len(s["top7"]) < 2 for s in plan["samples"]),
        "oracle_less_than_two": sum(len(s["oracle"]) < 2 for s in plan["samples"]),
        "oracle_more_than_seven": sum(len(s["oracle"]) > 7 for s in plan["samples"]),
        "plan_sha256": digest(plan),
    }


def freeze(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing experiment directory: {output}")
    folder = output / "freeze"
    folder.mkdir(parents=True, exist_ok=False)
    external._write_json(folder / "plan.json", plan)
    external._write_jsonl(folder / "samples.jsonl", plan["samples"])
    external._atomic_write_text(
        folder / "sample_ids.txt", "".join(s["uid"] + "\n" for s in plan["samples"])
    )
    report = preparation_report(plan, output)
    external._write_json(folder / "preparation_report.json", report)
    seal(folder)
    return report
