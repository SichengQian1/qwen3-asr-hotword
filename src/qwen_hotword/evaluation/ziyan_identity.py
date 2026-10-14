"""Read and package sealed Ziyan identities before interpreting retrieval errors."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

NAME = "ziyan_en_noise10db"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seal(directory: Path) -> dict[str, str]:
    hashes = {}
    for line in (directory / "sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, separator, name = line.partition("  ")
        path = (directory / name).resolve()
        if (
            not separator
            or len(expected) != 64
            or name in hashes
            or not path.is_relative_to(directory.resolve())
            or not path.is_file()
        ):
            raise ValueError(f"invalid seal entry: {line}")
        if digest(path) != expected:
            raise ValueError(f"sealed file hash differs: {path}")
        hashes[name] = expected
    if not hashes:
        raise ValueError(f"empty seal: {directory}")
    return hashes


def inspect(root: Path, config: Path, assets: Path, source: Path, output: Path) -> dict[str, Any]:
    """No model, MFA, source audio, candidate query, or scoring is executed."""
    root, config, assets, source, output = (
        p.resolve() for p in (root, config, assets, source, output)
    )
    if output.exists():
        raise FileExistsError(f"existing output refused: {output}")
    cfg = read_json(config)
    copies = {
        "assets_report.json": assets / "report.json",
        "assets_config.json": assets / "config.json",
        "assets_sha256.txt": assets / "sha256.txt",
        "targets.json": assets / "targets.json",
        "retrieval_report.json": source / "report.json",
        "run_config.json": source / "run_config.json",
        "retrieval_sha256.txt": source / "sha256.txt",
        "inspection_config.json": config,
    }
    for directory in (assets, source):
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("output must be separate from original asset/retrieval directories")
    seals = {"assets": verify_seal(assets), "retrieval": verify_seal(source)}
    for directory, label in ((assets, "assets"), (source, "retrieval")):
        required = {"report.json", "keyword_table.json"}
        required |= (
            {"config.json", "targets.json", "english_raw.dict", "dataset_manifest.jsonl"}
            if label == "assets"
            else {
                "run_config.json",
                f"{NAME}/retrieval_details.jsonl",
                *(f"delivery/{NAME}_top{k}.json" for k in (5, 7)),
            }
        )
        if not required <= seals[label].keys():
            raise ValueError(f"required files absent from seal: {directory}")
    asset_report, report, plan = (
        read_json(copies[k])
        for k in ("assets_report.json", "retrieval_report.json", "run_config.json")
    )
    if (
        asset_report.get("status") != "completed"
        or asset_report.get("table_ready") is not True
        or report.get("status") != "completed"
        or report.get("candidate_scope") != "joint_multilingual"
        or report.get("qwen_decoder_used") is not False
        or plan["configuration"] != cfg
        or read_json(assets / "config.json") != cfg
    ):
        raise ValueError("completed fixed assets/retrieval and identical configuration required")
    inputs = {}
    for key in ("dataset", "targets", "vocab", "checkpoint"):
        path = (root / cfg[key]).resolve()
        if output.is_relative_to(path.parent):
            raise ValueError(f"output inside input directory: {path.parent}")
        actual = digest(path)
        if actual != cfg[f"{key}_sha256"] or plan["inputs"][str(path)]["sha256"] != actual:
            raise ValueError(f"configured/recorded/actual input SHA differs: {key}")
        inputs[key] = {"path": str(path), "sha256": actual}
    table_sha = seals["assets"]["keyword_table.json"]
    if (
        table_sha != seals["retrieval"]["keyword_table.json"]
        or report["table_sha256"] != table_sha
        or report["checkpoint_sha256"] != inputs["checkpoint"]["sha256"]
        or report["table_replacement"] != asset_report["selection"]
    ):
        raise ValueError("report/table/Head identities differ")
    for key, sha in (
        ("ctc_checkpoint", inputs["checkpoint"]["sha256"]),
        ("vocab", inputs["vocab"]["sha256"]),
        ("keyword_bias", table_sha),
    ):
        if plan["group"]["config"][key]["sha256"] != sha:
            raise ValueError(f"recorded detector identity differs: {key}")
    if plan["inputs"][str(assets / "keyword_table.json")]["sha256"] != table_sha:
        raise ValueError("recorded input table identity differs")
    targets = read_json(assets / "targets.json")
    if (
        targets != (root / cfg["targets"]).read_text(encoding="utf-8").splitlines()
        or targets != asset_report["targets"]
        or len(targets) != cfg["expected_targets"]
        or len(read_json(assets / "keyword_table.json")["entries"]) != report["keyword_count"]
    ):
        raise ValueError("target list/table count differs")
    code = {}
    for relative, expected in plan["code_sha256"].items():
        path = (
            root / relative
            if relative.startswith("scripts/")
            else root / "src/qwen_hotword" / relative
        )
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"recorded/current retrieval code differs: {relative}")
        code[relative] = actual
    details = source / NAME / "retrieval_details.jsonl"
    records = plan["records"]
    frozen = [json.loads(s) for s in (assets / "dataset_manifest.jsonl").read_text().splitlines()]
    dataset = read_json(Path(inputs["dataset"]["path"]))
    references = {Path(r["audios"][0]).stem: (r["audios"][0], r["response"]) for r in dataset}
    if (
        frozen != records
        or len(records) != cfg["expected_samples"]
        or len(references) != len(records)
        or len(dataset) != len(records)
    ):
        raise ValueError("frozen/source JSON record identity differs")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fields: dict[str, int] = defaultdict(int)
    first = None
    candidate_row = None
    count = 0
    with details.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if count >= len(records) or any(row.get(k) != v for k, v in records[count].items()):
                raise ValueError("detail/frozen record identity differs")
            if references[row["sample_id"]] != (row["audio_path"], row["reference_text"]):
                raise ValueError("detail/source JSON reference differs")
            body = {k: v for k, v in row.items() if k != "row_sha256"}
            row_sha = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())
            if row_sha.hexdigest() != row["row_sha256"]:
                raise ValueError("saved row checksum differs")
            if row["audio_sha256"] != plan["inputs"][row["audio_path"]]["sha256"]:
                raise ValueError("recorded audio identity differs")
            for key in row:
                fields[key] += 1
            groups[row["audio_sha256"]].append(row)
            first = row if first is None else first
            if candidate_row is None and row.get("raw_variant_ranked_matches"):
                candidate_row = row
            count += 1
    if count != len(records) or count != report["sample_count"]:
        raise ValueError("detail count differs")
    extras = count - len(groups)
    if extras != plan["duplicate_audio_file_hashes"]:
        raise ValueError("recorded duplicate count differs")
    duplicates = [
        {
            "audio_sha256": sha,
            "members": [
                {k: row[k] for k in ("sample_id", "audio_path", "reference_text")} for row in rows
            ],
            "raw_references_identical": len({r["reference_text"] for r in rows}) == 1,
            "ctc_sequence_identical": all(
                r.get(k) == rows[0].get(k)
                for r in rows[1:]
                for k in ("decoded_token_ids", "decoded_confidences")
            ),
        }
        for sha, rows in sorted(groups.items())
        if len(rows) > 1
    ]
    assert first is not None
    schema_row = candidate_row or first
    summary = {
        "status": "identity_verified",
        "sample_count": count,
        "target_count": len(targets),
        "keyword_count": report["keyword_count"],
        "actual_inputs": inputs,
        "table_sha256": table_sha,
        "source_git_commit": plan["group"]["config"].get("git_commit"),
        "source_group_config": plan["group"]["config"],
        "code_sha256": code,
        "seal_file_counts": {k: len(v) for k, v in seals.items()},
        "row_field_presence": dict(sorted(fields.items())),
        "candidate_item_fields": {
            k: sorted(schema_row[k][0]) if schema_row.get(k) else []
            for k in ("raw_ranked_matches", "raw_variant_ranked_matches")
        },
        "candidate_schema_sample_id": schema_row["sample_id"],
        "reported_metrics_not_recomputed": report["metrics"],
        "duplicate_extra_records": extras,
        "duplicate_hash_groups": duplicates,
        "audio_hours_from_saved_metadata": sum(
            m["duration_seconds"] for m in plan["audio_metadata"]
        )
        / 3600,
        "model_loaded": False,
        "audio_read": False,
        "inference_rerun": False,
        "candidate_or_gate_replay": False,
        "limitations": [
            "Audio hashes reused from original run; audio bytes, SNR and spoken words not checked.",
            "Retrieval metrics copied, not independently recomputed at this identity stage.",
            "Up to two saved schema rows are packaged; these are not selected failure cases.",
            "Backbone metadata/weight bytes not reopened; original weight shards were not hashed.",
        ],
    }
    # Recheck every source seal after the scan; publication never changes original outputs.
    if any(verify_seal(d) != seals[k] for k, d in (("assets", assets), ("retrieval", source))):
        raise ValueError("source seal changed during inspection")
    contents = {name: path.read_bytes() for name, path in copies.items()}
    contents.update(
        {
            "summary.json": (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode(),
            "first_saved_row.json": (
                json.dumps(first, ensure_ascii=False, indent=2) + "\n"
            ).encode(),
            "candidate_schema_row.json": (
                json.dumps(schema_row, ensure_ascii=False, indent=2) + "\n"
            ).encode(),
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        (output / name).write_bytes(content)
    manifest = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in sorted(contents.items())
    )
    (output / "sha256.txt").write_text(manifest, encoding="utf-8")
    with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.name != "return_bundle.zip":
                archive.write(path, path.name)
    return {
        "status": summary["status"],
        "sample_count": count,
        "target_count": len(targets),
        "keyword_count": summary["keyword_count"],
        "duplicate_extra_records": extras,
        "return_file": str(output / "return_bundle.zip"),
        "bundle_sha256": digest(output / "return_bundle.zip"),
        "bundle_bytes": (output / "return_bundle.zip").stat().st_size,
    }
