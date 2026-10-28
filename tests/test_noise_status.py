from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import pytest

from qwen_hotword.training.noise_status import (
    inspect,
    inspect_replacements,
    terminal_summary,
    write_diagnostic,
)
from qwen_hotword.training.spanish_capacity import _sha


def _fixture(tmp_path: Path, *, mismatch: str | None = None) -> tuple[Path, dict]:
    torch = pytest.importorskip("torch")
    run = tmp_path / "noise_run"
    run.mkdir()
    cache = tmp_path / "original_cache"
    (cache / "shards").mkdir(parents=True)
    (cache / "cache_index.json").write_text(json.dumps({"shard_count": 2}))
    order = [0, 1]
    random.Random(123 + 1).shuffle(order)
    bad = order[0]
    parts = []
    for shard in range(2):
        ids = [f"sample-{shard}-{i}" for i in range(2)]
        folder = run / "noise/train" / f"part-{shard:06d}"
        (folder / "features/shards").mkdir(parents=True)
        rows = [
            {
                "id": sample_id,
                "selected_30": True,
                "phoneme_token_ids": [2],
                "language": "en",
                "audio_path": f"/original/{sample_id}.wav",
            }
            for sample_id in ids
        ]
        manifest = folder / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (folder / "mix_audit.json").write_text(
            json.dumps(
                [{"id": sample_id, "source_samples": 16000, "snr_db": 10} for sample_id in ids]
            )
        )
        for noise in (False, True):
            frames = 4 if noise and shard == bad and mismatch == "frames" else 3
            label = 3 if noise and shard == bad and mismatch == "labels" else 2
            path = (
                folder / "features/shards/shard-000000.pt"
                if noise
                else cache / "shards" / f"shard-{shard:06d}.pt"
            )
            torch.save(
                {
                    "hidden_states": torch.zeros(frames + 3, 1024, dtype=torch.bfloat16),
                    "hidden_offsets": torch.tensor([0, frames, frames + 3]),
                    "token_ids": torch.tensor([label, 2]),
                    "token_offsets": torch.tensor([0, 1, 2]),
                },
                path,
            )
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "sample_ids": ids,
                        "feature_bytes": path.stat().st_size,
                        "feature_sha256": _sha(path),
                    }
                )
            )
        (folder / "features/cache_summary.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "sample_count": 2,
                    "feature_bytes": (folder / "features/shards/shard-000000.pt").stat().st_size,
                    "extraction_seconds": 1.5,
                }
            )
        )
        parts.append(
            {
                "split": "train",
                "base_shard": shard,
                "records": 2,
                "manifest": manifest.relative_to(run).as_posix(),
                "sha256": _sha(manifest),
            }
        )
    plan = {
        "train_cache": str(cache),
        "parts": parts,
        "input_sha256": {},
        "configuration": {"seed": 123, "training": {"epochs": 10}},
    }
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "prepared.json").write_text(
        json.dumps(
            {
                "plan_sha256": _sha(run / "plan.json"),
                "status": "prepared",
                "noisy_hours": {"30": {"en": 144}, "50": {"en": 240}},
            }
        )
    )
    (run / "cache_completed.json").write_text(
        json.dumps(
            {
                "status": "noise_cache_completed",
                "parts": 2,
                "mixing": {"snr_db_min": 10},
            }
        )
    )
    head = run / "head_30"
    head.mkdir()
    torch.save(
        {
            "completed_epoch": 0,
            "best_epoch": 0,
            "training_seconds": 0,
            "optimizer_state_dict": {"state": {}},
            "initial_validation": {"loss": 0.2},
            "initial_validation_by_group": {"en_original": {"phoneme_error_rate": 0.05}},
            "best_validation": {"loss": 0.2},
            "history": [],
        },
        head / "training_state_latest.pt",
    )
    return run, plan


def _snapshot(tmp_path: Path) -> dict[str, str]:
    return {str(p): _sha(p) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mismatch", ["labels", "frames"])
def test_diagnostic_separates_mismatch_and_never_modifies_run(
    tmp_path: Path, mismatch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    run, _ = _fixture(tmp_path, mismatch=mismatch)
    before = _snapshot(tmp_path)
    original_load = torch.load
    mmap_calls = []

    def load(path: Path, **kwargs: object) -> object:
        if "shards" in str(path):
            assert kwargs["mmap"] is True
            mmap_calls.append(str(path))
        assert kwargs["map_location"] == "cpu" and kwargs["weights_only"] is True
        return original_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    report = inspect(run, check_replacements=True)
    assert _snapshot(tmp_path) == before
    assert len(mmap_calls) == 2
    assert report["preparation"]["status"] == "prepared"
    assert report["cache"]["completed_parts_observed"] == 2
    assert report["cache"]["by_split"]["train"]["extraction_seconds_sum"] == 3
    assert report["training"]["30"]["persisted_epoch"] == 0
    assert report["training"]["30"]["initial_validation"]["loss"] == 0.2
    assert report["training"]["50"]["status"] == "not_started"
    assert report["evaluation"]["status"] == "not_started"
    details = report["replacement_check"]
    assert details["checked_shards"] == 1
    assert details["label_mismatches"] == int(mismatch == "labels")
    assert details["frame_mismatches"] == int(mismatch == "frames")
    assert details["examples"][0]["mix_audit"]["snr_db"] == 10
    result = write_diagnostic(report, tmp_path / "inspection")
    summary = terminal_summary(result)
    assert "original_label_ids" not in json.dumps(summary)
    assert "examples" not in summary["replacement_check"]
    for name, digest in before.items():
        assert _sha(Path(name)) == digest
    with zipfile.ZipFile(result["return_bundle"]) as archive:
        assert set(archive.namelist()) == {"report.json", "sha256.txt"}
    with pytest.raises(ValueError, match="NEW directory outside"):
        write_diagnostic(report, run / "inspection")


def test_limit_and_full_scan_are_distinguished(tmp_path: Path) -> None:
    run, plan = _fixture(tmp_path)
    report = inspect_replacements(run, plan, ratio=30, epoch=1, max_shards=1)
    assert report["checked_replacements"] == 2
    assert report["all_training_shards_checked"] is False
    assert "subset" in report["conclusion"]
    report = inspect_replacements(run, plan, ratio=50, epoch=1, max_shards=0)
    assert report["checked_replacements"] == 4
    assert report["all_training_shards_checked"] is True


def test_changed_plan_is_rejected_without_touching_run(tmp_path: Path) -> None:
    run, _ = _fixture(tmp_path)
    (run / "plan.json").write_text((run / "plan.json").read_text() + "\n")
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="plan SHA mismatch"):
        inspect(run)
    assert _snapshot(tmp_path) == before


def test_missing_cache_and_code_drift_are_not_claimed_complete(tmp_path: Path) -> None:
    run, plan = _fixture(tmp_path)
    (run / plan["parts"][0]["manifest"]).parent.joinpath("features/shards/shard-000000.pt").unlink()
    code = tmp_path / "source.py"
    code.write_text("# changed\n")
    plan["input_sha256"][str(code)] = "historical-sha"
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "prepared.json").write_text(json.dumps({"plan_sha256": _sha(run / "plan.json")}))
    report = inspect(run)
    assert report["cache"]["completion_marker_present"] is True
    assert report["cache"]["completed_parts_observed"] == 1
    assert report["cache"]["invalid_parts"] == 1
    assert report["source_code_drift"][0]["path"] == str(code)
    report = inspect(run, check_replacements=True, max_shards=0)
    assert report["replacement_check"]["status"] == "inspection_error"
    assert report["cache"]["completed_parts_observed"] == 1
    assert write_diagnostic(report, tmp_path / "diagnostic_after_error")["return_bundle"]
