"""Find saved paired ASR gains and export unchanged WAVs; never run inference."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.inference.hotword_prompt import normalize_match_words, strict_phrase_match
from qwen_hotword.training.stage_progress import stage_progress

TABLE_SHA256 = "6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b"
TABLE_RELATIVE = "outputs/reviewed_joint_4000_retrieval_480h_v1/reviewed_keyword_table.json"
RESULT_NAMES = {
    "sample_results.jsonl",
    "predictions.jsonl",
    "asr_results.jsonl",
    "results.jsonl",
    "baseline_predictions.jsonl",
    "retrieved_predictions.jsonl",
}
METADATA_NAMES = ("run_config.json", "report.json", "runtime.json", "config.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object: {path}:{number}")
                result.append(row)
    return result


def table_digests(value: Any, location: str = "") -> list[tuple[str, str]]:
    """Accept SHA only in table/hotword input identity fields, not arbitrary text."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{location}/{key}"
            if isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item):
                if any(word in path.lower() for word in ("table", "hotword", "keyword")):
                    found.append((path, item))
            elif isinstance(item, dict):
                found.extend(table_digests(item, path))
    return found


def discover(scan_roots: list[Path]) -> list[Path]:
    import os

    directories = set()
    for root in scan_roots:
        if not root.is_dir():
            continue
        with stage_progress(f"Locate saved ASR results under {root}"):
            for folder, children, files in os.walk(root, followlinks=False):
                # Only aggregate predictions are consumed, not model caches,
                # millions of training rows or individual inference shards.
                children[:] = [
                    name
                    for name in children
                    if name not in {"cache", "features", "shards", "samples", ".git"}
                    and not name.startswith("hotword_recovered_cases_")
                ]
                if RESULT_NAMES.intersection(files):
                    directories.add(Path(folder).resolve())
    return sorted(directories)


def metadata(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged: dict[str, Any] = {}
    identities = []
    for name in METADATA_NAMES:
        path = directory / name
        if path.is_file() and path.stat().st_size < 16 * 1024 * 1024:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                # Run config takes precedence over report/runtime defaults.
                merged = {**obj, **merged}
                identities.append({"path": str(path), "sha256": sha(path)})
    return merged, identities


def verify_saved_hashes(directory: Path, paths: list[Path]) -> str:
    manifest = directory / "sha256.txt"
    if not manifest.exists():
        return "no_saved_checksum_manifest; export records current source identities"
    recorded = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            recorded[name.lstrip(" *")] = digest
    covered = 0
    for path in paths:
        name = path.relative_to(directory).as_posix()
        if name in recorded:
            if sha(path) != recorded[name]:
                raise ValueError(f"saved result checksum mismatch: {path}")
            covered += 1
    return f"verified {covered}/{len(paths)} source files against saved checksum manifest"


def paired_rows(
    directory: Path,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], list[Path]]:
    base = directory / "baseline_predictions.jsonl"
    hot = directory / "retrieved_predictions.jsonl"
    if base.exists() and hot.exists():
        files = [base, hot]
        left, right, label = rows(base), rows(hot), "offline baseline/retrieved"
    else:
        files = [directory / name for name in sorted(RESULT_NAMES) if (directory / name).is_file()]
        files = [p for p in files if p.name not in {base.name, hot.name}]
        if len(files) != 1:
            raise ValueError("no unambiguous supported paired prediction files")
        all_rows = rows(files[0])
        groups = {r.get("experiment_group") for r in all_rows}
        if {"C", "D"} <= groups:
            a, b, label = "C", "D", "streaming C/D"
        elif {"A", "B"} <= groups:
            a, b, label = "A", "B", "offline A/B"
        else:
            raise ValueError("no known baseline/retrieved groups; schema needs explicit adapter")
        left = [r for r in all_rows if r.get("experiment_group") == a]
        right = [r for r in all_rows if r.get("experiment_group") == b]

    def index(items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        lookup = {}
        for item in items:
            if any(
                not isinstance(item.get(field), str) or not item[field]
                for field in ("case_id", "sample_id")
            ):
                raise ValueError("paired ASR rows require nonempty case_id and sample_id strings")
            key = (str(item["case_id"]), str(item["sample_id"]))
            if key in lookup:
                raise ValueError(f"duplicate paired ASR case: {key}")
            lookup[key] = item
        return lookup

    aa, bb = index(left), index(right)
    return [(aa[k], bb[k], label) for k in sorted(aa.keys() & bb.keys())], files


def support_samples(config: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = {}
    for name in ("validation", "validation_manifest", "boundary_manifest"):
        identity = config.get("inputs", {}).get(name, {})
        if not isinstance(identity, dict) or not identity.get("path"):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))):
            continue
        path = Path(identity["path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            continue
        if sha(path) != identity["sha256"]:
            raise ValueError(f"audio/reference manifest checksum mismatch: {path}")
        for row in rows(path):
            for key in (row.get("case_id"), row.get("sample_id"), row.get("id")):
                if isinstance(key, str):
                    if key in support and support[key].get("audio_path") != row.get("audio_path"):
                        raise ValueError(f"ambiguous sample-to-audio mapping: {key}")
                    support[key] = row
    return support


def language(value: str) -> str:
    lowered = value.casefold()
    for code, names in {
        "en": {"english", "en"},
        "es": {"spanish", "es"},
        "pt": {"portuguese", "pt"},
    }.items():
        if lowered in names or lowered.startswith(code + "-"):
            return code
    raise ValueError(f"unsupported or missing ASR language: {value}")


def candidate(
    base: dict[str, Any],
    hot: dict[str, Any],
    config: dict[str, Any],
    table: dict[tuple[str, ...], str],
    support: dict[str, dict[str, Any]],
    root: Path,
    table_sha256: str = TABLE_SHA256,
) -> dict[str, Any] | None:
    for item in (base, hot):
        if item.get("error") or item.get("status") in {"failed", "error"}:
            raise ValueError("failed ASR request")
        if item.get("generation_truncated") or item.get("reached_max_new_tokens"):
            raise ValueError("truncated ASR request")
        if any(digest != table_sha256 for _, digest in table_digests(item)):
            raise ValueError("row-level table identity conflicts with latest table")
        if "oracle" in str(item.get("mode", "")).casefold():
            raise ValueError("oracle prompting is not retrieved-hotword evidence")
    reference = base.get("reference_text")
    if not isinstance(reference, str) or reference != hot.get("reference_text"):
        raise ValueError("paired references missing or different")
    before, after = base.get("prediction"), hot.get("prediction")
    if not isinstance(before, str) or not isinstance(after, str):
        raise ValueError("final ASR prediction strings are missing")
    if base.get("injected_hotwords") or base.get("actual_context"):
        raise ValueError("baseline already contains injected hotwords")
    injected = hot.get("injected_hotwords")
    if not isinstance(injected, list) or any(not isinstance(w, str) for w in injected):
        raise ValueError("recorded injected-hotword strings are missing")
    gained = sorted(
        {
            table[key]
            for word in injected
            if (key := normalize_match_words(word)) in table
            and strict_phrase_match(reference, table[key])
            and not strict_phrase_match(before, table[key])
            and strict_phrase_match(after, table[key])
        }
    )
    if not gained:
        return None
    source = support.get(str(base["case_id"]), support.get(str(base["sample_id"]), {}))
    if (
        source.get("leading_silence_sec", 0)
        or base.get("leading_silence_sec", 0)
        or hot.get("leading_silence_sec", 0)
    ):
        raise ValueError(
            "ASR used synthetic leading silence; original file differs from evaluated waveform"
        )
    paths = [item.get("audio_path", source.get("audio_path")) for item in (base, hot)]
    if not all(isinstance(p, str) and p for p in paths):
        raise ValueError("audio path absent; no authenticated supporting sample mapping")
    resolved = []
    for path_string in paths:
        if not isinstance(path_string, str):
            raise ValueError("invalid original audio path")
        path_value = Path(path_string)
        resolved.append((path_value if path_value.is_absolute() else root / path_value).resolve())
    if resolved[0] != resolved[1]:
        raise ValueError("baseline/retrieved audio paths differ")
    path = resolved[0]
    if path.suffix.casefold() != ".wav" or not path.is_file():
        raise ValueError(f"original WAV missing: {path}")
    with path.open("rb") as handle:
        header = handle.read(12)
    if header[:4] not in {b"RIFF", b"RIFX", b"RF64"} or header[8:12] != b"WAVE":
        raise ValueError(f"audio file does not have a WAV header: {path}")
    digest = sha(path)
    recorded = [item.get("audio_sha256") for item in (base, hot, source)]
    if any(value is not None and value != digest for value in recorded):
        raise ValueError("current audio bytes differ from saved ASR audio identity")
    codes = [
        language(str(item.get("language", config.get("language", "")))) for item in (base, hot)
    ]
    if codes[0] != codes[1]:
        raise ValueError("baseline/retrieved languages differ")
    return {
        "case_id": base["case_id"],
        "sample_id": base["sample_id"],
        "language": codes[0],
        "source_audio": str(path),
        "audio_sha256": digest,
        "historical_audio_sha_recorded": any(value is not None for value in recorded),
        "reference_text": reference,
        "without_hotwords": before,
        "with_hotwords": after,
        "recovered_hotwords": gained,
        "injected_hotwords": injected,
    }


def choose(cases: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    # Deterministic language round-robin; no quota fabrication when a language
    # has too few examples. Deduplicate source bytes across all runs/conditions.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(cases, key=lambda c: (c["language"], c["source_run"], c["case_id"])):
        groups[case["language"]].append(case)
    selected: list[dict[str, Any]] = []
    used = set()
    while any(groups.values()) and len(selected) < count:
        for code in ("en", "es", "pt"):
            while groups[code]:
                case = groups[code].pop(0)
                if case["audio_sha256"] not in used:
                    used.add(case["audio_sha256"])
                    selected.append(case)
                    break
            if len(selected) == count:
                break
    return selected


def write_package(output: Path, selected: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    if output.exists():
        raise FileExistsError(f"export requires a NEW directory: {output}")
    output.mkdir(parents=True)
    (output / "audio").mkdir()
    lines = [
        "# 加热词后恢复的转写案例",
        "",
        f"请求 {report['requested_cases']} 条，实际导出 **{len(selected)} 条不同原始音频**。",
        "",
        f"状态：`{report['status']}`；不足时不补造案例。",
        "",
        "筛选条件：该热词在参考文本及实际注入候选中，无热词转写未命中，加热词转写命中；"
        "采用规范化后的整词／连续短语匹配。大小写和标点变化不单独算恢复。",
        "",
        "这是已保存配对结果的成功案例集，不代表总体提升率或每次运行都能恢复。参考内容尚未人工听审。",
        "",
        "每例只确认列出的热词恢复，不表示整句转写的所有变化都更好。",
        "",
        f"联合4k词表 SHA256：`{report['table_sha256']}`。WAV保持原字节，不裁剪或重采样。",
        "",
    ]
    for number, case in enumerate(selected, 1):
        source = Path(case["source_audio"])
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:48]
        name = f"{number:03d}_{case['language']}_{stem}_{case['audio_sha256'][:8]}.wav"
        target = output / "audio" / name
        shutil.copyfile(source, target)
        if sha(target) != case["audio_sha256"]:
            raise ValueError(f"audio changed during copy: {source}")
        case["exported_audio"] = "audio/" + name
        lines.extend(
            [
                f"## {number:02d} · {name}",
                "",
                f"- 音频：[下载／播放]({case['exported_audio']})",
                f"- 语言：{case['language']}",
                "- 恢复热词：" + "、".join(f"**{w}**" for w in case["recovered_hotwords"]),
                f"- 原路径：`{source}`",
                f"- 对照来源：`{case['source_run']}`；模式 `{case['comparison']}`；"
                f"case `{case['case_id']}`",
                "",
            ]
        )
        for label, key in (
            ("参考文本", "reference_text"),
            ("不加热词", "without_hotwords"),
            ("加热词后", "with_hotwords"),
        ):
            # Prefix every line, so transcript text cannot break document fences.
            lines.extend([label + "：", "", *("> " + s for s in case[key].splitlines()), ""])
    (output / "案例说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in selected), encoding="utf-8"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "sha256.txt").write_text(
        "".join(f"{sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8"
    )
    archive = output / "return_bundle.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zip_file:
        for path in [*files, output / "sha256.txt"]:
            zip_file.write(path, arcname=path.relative_to(output).as_posix())
    return archive


def export_saved_cases(
    root: Path,
    scan_roots: list[Path],
    table_path: Path,
    output: Path,
    *,
    count: int = 50,
    expected_table_sha256: str = TABLE_SHA256,
) -> dict[str, Any]:
    if count <= 0 or count > 1000:
        raise ValueError("count must be between 1 and 1000")
    if output.exists():
        raise FileExistsError(f"export requires a NEW directory: {output}")
    actual_table_sha = sha(table_path)
    if actual_table_sha != expected_table_sha256:
        raise ValueError("table is not the pinned latest joint 4k table")
    table_json = json.loads(table_path.read_text(encoding="utf-8"))
    table = {normalize_match_words(e["surface"]): e["surface"] for e in table_json["entries"]}
    inventory = []
    cases = []
    rejected: Counter[str] = Counter()
    directories = discover(scan_roots)
    with stage_progress(
        "Read saved paired ASR transcripts (CPU; no inference)", total=len(directories)
    ) as tick:
        for number, directory in enumerate(directories, 1):
            item: dict[str, Any] = {"directory": str(directory)}
            try:
                config, identities = metadata(directory)
                digests = table_digests(config)
                if not digests or {digest for _, digest in digests} != {actual_table_sha}:
                    raise ValueError(
                        "latest table use unverified, old table, or ambiguous table identities"
                    )
                pairs, files = paired_rows(directory)
                item["source_checksum_check"] = verify_saved_hashes(
                    directory, [*files, *(Path(i["path"]) for i in identities)]
                )
                item["sources"] = identities + [{"path": str(p), "sha256": sha(p)} for p in files]
                item["saved_model_identity"] = {
                    key: config[key]
                    for key in ("model", "model_path", "checkpoint_sha256", "git_commit")
                    if key in config
                }
                item["saved_checkpoint_identity"] = config.get("inputs", {}).get("checkpoint")
                support = support_samples(config, root)
                item.update(status="paired_transcripts_found", paired_rows=len(pairs), eligible=0)
                for base, hot, comparison in pairs:
                    try:
                        case = candidate(base, hot, config, table, support, root, actual_table_sha)
                        if case is not None:
                            case.update(source_run=str(directory), comparison=comparison)
                            cases.append(case)
                            item["eligible"] += 1
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        rejected[str(error)] += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                item.update(status="not_usable", reason=str(error))
            inventory.append(item)
            tick(number)
    selected = choose(cases, count)
    report = {
        "status": "completed" if len(selected) == count else "insufficient_verified_cases",
        "requested_cases": count,
        "exported_cases": len(selected),
        "shortfall": max(0, count - len(selected)),
        "table_path": str(table_path),
        "table_sha256": actual_table_sha,
        "scan_roots": [str(p) for p in scan_roots],
        "candidate_run_directories": len(directories),
        "run_inventory": inventory,
        "rejected_pair_reasons": dict(rejected),
        "by_language": dict(Counter(c["language"] for c in selected)),
        "source_code_sha256": sha(Path(__file__)),
        "gpu_used": False,
        "inference_performed": False,
        "audio_listened_to": False,
        "limitation": (
            "Only supported saved ASR schemas in scanned directories; "
            "other directories or formats may contain additional results."
        ),
    }
    archive = write_package(output, selected, report)
    return {
        key: report[key]
        for key in (
            "status",
            "requested_cases",
            "exported_cases",
            "shortfall",
            "by_language",
            "candidate_run_directories",
        )
    } | {"return_bundle": str(archive)}
