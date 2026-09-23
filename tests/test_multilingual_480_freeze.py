from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.ctc_overfit import load_experiment_records
from qwen_hotword.training.en_pt_480_plan import prepare_plan
from qwen_hotword.training.multilingual_480_freeze import freeze_plan, portuguese_protected
from qwen_hotword.training.spanish_capacity import _sha


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data))


def jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def checks(root: Path) -> None:
    (root / "sha256.txt").write_text(
        "".join(
            f"{_sha(p)}  {p.name}\n"
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "sha256.txt"
        )
    )


def setup(root: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {"pools": {}, "target_hours": 8 / 3600, "seed": 1}
    for lang, tag in (("en", "en-US"), ("es", "es"), ("pt", "pt-BR")):
        pool = root / lang
        pool.mkdir()
        data: dict[str, list[dict[str, Any]]] = {}
        all_rows = []
        for split, count, split_hash in (
            ("train", 4, 0.1),
            ("validation", 1, 0.97),
            ("test", 1, 0.99),
        ):
            data[split] = []
            for i in range(count):
                name = f"{lang}_{split}_{i}"
                audio = root / f"{name}.wav"
                audio.write_bytes(name.encode())
                row = {
                    "id": name,
                    "audio_path": str(audio),
                    "text": "fixture",
                    "split": split,
                    "language": tag,
                    "speaker_id": name if lang != "pt" else None,
                    "duration_seconds": 4.0,
                    "source_corpus": "corpus",
                    "release_source": "original_ready",
                    "split_hash": split_hash,
                    "phoneme_token_ids": [1, 2, 3],
                    "label_length": 3,
                    "ctc_minimum_input_length": 3,
                    "estimated_ctc_input_length": 5,
                    "ctc_time_upsampling_factor": 2,
                    "effective_ctc_input_length": 10,
                    "effective_ctc_target_ratio": 0.3,
                    "training_ready": True,
                    "label_status": "ready",
                    "issues": [],
                }
                data[split].append(row)
                all_rows.append(row)
            # Deliberately no sealed test manifest: it must never be opened.
            if split != "test":
                jsonl(pool / f"full_ctc_{split}.jsonl", data[split])
        summary = {
            "status": "pass",
            "test_set_sealed": True,
            "test_set_used": False,
            "split_records": {s: len(rs) for s, rs in data.items()},
            "split_audio_hours": {s: len(rs) * 4 / 3600 for s, rs in data.items()},
            "split_fractions": {"train": 0.96, "validation": 0.02, "test": 0.02},
            "manifest_paths": {s: str(pool / f"full_ctc_{s}.jsonl") for s in data},
            "manifest_sha256": {s: _sha(pool / f"full_ctc_{s}.jsonl") for s in data if s != "test"}
            | {"test": "f" * 64},
            "corpus_metrics": {
                "corpus": {
                    f"split_{s}": {"records": len(rs), "hours": len(rs) * 4 / 3600}
                    for s, rs in data.items()
                }
            },
        }
        dump(pool / "split_summary.json", summary)
        if lang == "pt":
            sources = root / "pt_sources"
            sources.mkdir()
            raw = [
                {k: v for k, v in r.items() if k not in {"text", "phoneme_token_ids"}}
                for r in all_rows
            ]
            recovery = raw.pop()  # One heldout test row exercises 2x recovery reconstruction.
            recovery.update(
                training_ready=False,
                label_status="needs_review",
                issues=[{"reason": "ctc_length_infeasible"}],
                ctc_minimum_input_length=6,
            )
            jsonl(sources / "ready.jsonl", raw)
            jsonl(sources / "review.jsonl", [recovery])
            dump(
                pool / "split_config.json",
                {
                    "split_strategy": "existing_stable_split_hash",
                    "split_fractions": summary["split_fractions"],
                    "release_policy": {"time_upsampling_factor": 2, "maximum_effective_ratio": 0.9},
                    "inputs": {
                        "corpus": {
                            kind: {"path": str(sources / name), "sha256": _sha(sources / name)}
                            for kind, name in (
                                ("ready_manifest", "ready.jsonl"),
                                ("review_manifest", "review.jsonl"),
                            )
                        }
                    },
                },
            )
        else:
            source = pool / "source.tsv"
            source.write_text(
                "audio\tspeaker_id\tsource_split\n"
                + "".join(f"{r['audio_path']}\t{r['speaker_id']}\t{r['split']}\n" for r in all_rows)
            )
            (pool / "speaker_split_assignments.tsv").write_text(
                "speaker_id\tsplit\n"
                + "".join(f"{r['speaker_id']}\t{r['split']}\n" for r in all_rows)
            )
            dump(
                pool / "split_config.json",
                {
                    "inputs": {
                        "corpus": {"source_tsv": {"path": str(source), "sha256": _sha(source)}}
                    }
                },
            )
        checks(pool)
        if lang != "es":
            cfg["pools"][lang] = {
                "pool": str(pool),
                "sources": ["corpus"],
                "manifest_sha256": summary["manifest_sha256"],
                "records": 4,
                "hours": 16 / 3600,
            }
        else:
            cfg["frozen_es"] = {
                "train_manifest": str(pool / "full_ctc_train.jsonl"),
                "sha256": summary["manifest_sha256"]["train"],
                "records": 4,
                "hours": 16 / 3600,
            }
    plan = root / "plan"
    report = prepare_plan(cfg, plan)
    return {
        "plan_dir": str(plan),
        "target_hours": cfg["target_hours"],
        "es_holdout_pool": str(root / "es"),
        "proposed_ids_sha256": {
            lang: report["languages"][lang]["proposed_ids_sha256"] for lang in ("en", "pt")
        },
    }


def chosen_path(root: Path, lang: str) -> Path:
    row = json.loads((root / "plan" / f"proposed_ids_{lang}.jsonl").read_text().splitlines()[0])
    return Path(row["audio_path"])


def test_full_freeze_preserves_ids_labels_es_and_training_format(tmp_path: Path) -> None:
    cfg = setup(tmp_path)
    es = tmp_path / "es/full_ctc_train.jsonl"
    original = es.read_bytes()
    out = tmp_path / "out"
    result = freeze_plan(cfg, out, workers=1)
    assert result["status"] == "completed"
    assert result["artifacts"]["combined"]["records"] == 8
    assert result["audio_identity_audit"]["checked_files"] == 14
    assert result["audio_identity_audit"]["unique_train_file_hashes"] == 8
    assert not result["test_manifest_content_read"]
    assert result["test_audio_bytes_read_for_identity_only"]
    assert es.read_bytes() == original
    assert not list(tmp_path.glob("*/full_ctc_test.jsonl"))
    rows = [json.loads(x) for x in (out / "full_ctc_train.jsonl").read_text().splitlines()]
    assert {r["balanced_language_bucket"] for r in rows} == {"en", "es", "pt"}
    assert all(r["phoneme_token_ids"] == [1, 2, 3] and r["text"] == "fixture" for r in rows)
    # Exercise the actual feature/training manifest parser without importing/loading a model.
    assert (
        len(
            load_experiment_records(
                out / "full_ctc_train.jsonl", num_classes=90, expected_experiment="full-ctc-v1"
            )
        )
        == 8
    )
    for line in (out / "sha256.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert digest == _sha(out / name)


@pytest.mark.parametrize("other", ["es_train_0", "pt_validation_0", "pt_test_0"])
def test_cross_language_and_heldout_byte_conflicts_block(tmp_path: Path, other: str) -> None:
    cfg = setup(tmp_path)
    chosen_path(tmp_path, "en").write_bytes((tmp_path / f"{other}.wav").read_bytes())
    result = freeze_plan(cfg, tmp_path / "out", workers=1)
    assert result["status"] == "blocked_audio_conflicts"
    assert not result["training_manifest_ready"]
    assert not list((tmp_path / "out").glob("full_ctc_train*"))
    assert (tmp_path / "out/en.pending.jsonl").exists()


@pytest.mark.parametrize("mutation", ["ids", "source_sha", "counts", "es_sha"])
def test_identity_and_metadata_reconstruction_guards(tmp_path: Path, mutation: str) -> None:
    cfg = setup(tmp_path)
    if mutation == "ids":
        cfg["proposed_ids_sha256"]["en"] = "changed"
    elif mutation == "source_sha":
        (tmp_path / "pt_sources/ready.jsonl").write_text("changed")
    elif mutation == "es_sha":
        (tmp_path / "es/full_ctc_train.jsonl").write_text("changed")
    else:
        summary = json.loads((tmp_path / "pt/split_summary.json").read_text())
        summary["corpus_metrics"]["corpus"]["split_test"]["records"] = 2
        with pytest.raises(ValueError, match="source-derived"):
            portuguese_protected(tmp_path / "pt", summary, {})
        return
    with pytest.raises(ValueError):
        freeze_plan(cfg, tmp_path / "out", workers=1)
    assert not (tmp_path / "out/full_ctc_train.jsonl").exists()


def test_failed_plan_and_existing_outputs_never_reused(tmp_path: Path) -> None:
    cfg = setup(tmp_path)
    with pytest.raises(FileExistsError):
        freeze_plan(cfg, tmp_path / "plan")
    (tmp_path / "plan/FAILED.txt").write_text("failed")
    with pytest.raises(ValueError, match="failed"):
        freeze_plan(cfg, tmp_path / "out")
