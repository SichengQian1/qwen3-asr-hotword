from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_english_refresh import prepared_inputs as prepared_inputs

from qwen_hotword.training import english_refresh as original
from qwen_hotword.training import english_refresh_compat as compatibility
from qwen_hotword.training import english_refresh_fast as fast
from qwen_hotword.training import english_refresh_gpu as gpu_stages
from qwen_hotword.training.spanish_480_plan import _rows, candidate
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def staged(prepared_inputs, monkeypatch):
    root, cfg, mfa_calls = prepared_inputs

    def stop(*args):
        raise RuntimeError("stopped before byte scanning")

    monkeypatch.setattr(original, "freeze", stop)
    with pytest.raises(RuntimeError, match="stopped before"):
        original.prepare(root, cfg, root / "original", workers=2, jobs=2)
    assert (root / "original/manifest/sha256.txt").exists()
    return root, cfg, mfa_calls


def test_reuses_completed_labels_without_opening_any_audio(staged, monkeypatch):
    root, cfg, mfa_calls = staged
    before = {p: _sha(p) for p in (root / "original").rglob("*") if p.is_file()}
    open_path = Path.open

    def no_audio(path, *args, **kwargs):
        assert path.suffix.lower() not in {".wav", ".flac"}, "WAV reads must be skipped"
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_audio)
    result = fast.prepare(root, cfg, root / "original", root / "fast")
    assert result["status"] == "data_prepared" and not result["audio_byte_audit_performed"]
    assert result["full_english"]["source_records"] == 12
    assert len(mfa_calls) == 1
    assert result["selection"]["en"]["hours"] >= cfg["target_hours"]
    frozen = json.loads((root / "fast/frozen/report.json").read_text())
    assert frozen["audio_identity_audit"]["status"] == "not_performed"
    assert not frozen["audio_identity_audit"]["byte_isolation_verified"]
    for lang in ("es", "pt"):
        assert (
            _sha(root / f"fast/frozen/full_ctc_train_{lang}.jsonl")
            == cfg["fixed_languages"][lang]["sha256"]
        )
    baseline = json.loads((root / "config/baseline.json").read_text())
    training = json.loads((root / "fast/training_config_path_only.json").read_text())
    assert training["training_options"] == baseline["training_options"]
    assert training["audio_isolation_policy"] == fast.POLICY
    for path, sha in before.items():
        assert _sha(path) == sha
    for row in _rows(root / "fast/frozen/full_ctc_train_en.jsonl"):
        assert "audio_file_sha256" not in row
    with pytest.raises(FileExistsError):
        fast.prepare(root, cfg, root / "original", root / "fast")


def test_path_metadata_rules_match_original_candidate(staged):
    root, cfg, _ = staged
    fast.prepare(root, cfg, root / "original", root / "fast")
    for row in _rows(root / "fast/frozen/released_english_pool.jsonl"):
        assert fast.light_candidate(row) == candidate(
            row, row["source_corpus"], row["release_source"]
        )


def test_path_overlap_and_conflicts_quarantined_but_different_paths_not_deduped(staged):
    root, cfg, _ = staged
    fast.prepare(root, cfg, root / "original", root / "fast")
    rows = list(_rows(root / "fast/frozen/released_english_pool.jsonl"))[:5]
    rows[1]["audio_path"] = rows[0]["audio_path"]
    rows[2]["audio_path"] = rows[0]["audio_path"]
    rows[2]["text"] = "conflicting transcript"
    pool = root / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    candidates, excluded = fast.path_candidates(pool, {rows[3]["audio_path"]}, 1, 5)
    assert len(candidates) == 1
    assert excluded["same_path_conflicting_labels"] == 3
    assert excluded["fixed_train_or_heldout_path_overlap"] == 1
    assert candidates[0]["audio_path"] == rows[4]["audio_path"]


def test_gpu_stage_delegates_with_explicit_policy_and_unchanged_options(staged, monkeypatch):
    root, cfg, _ = staged
    fast.prepare(root, cfg, root / "original", root / "fast")
    calls = []

    def run(config, output, stage, gpu):
        calls.append((config, output, stage, gpu))
        return {"status": "mock_only"}

    monkeypatch.setattr(gpu_stages, "run_stage", run)
    monkeypatch.chdir(root)
    assert fast.run_stage(root, root / "fast", "smoke", "4")["status"] == "mock_only"
    assert calls[0][0]["audio_isolation_policy"] == fast.POLICY
    assert calls[0][2:] == ("smoke", "4")


def test_incomplete_labels_or_changed_original_inputs_refused(staged):
    root, cfg, _ = staged
    with pytest.raises(ValueError, match="separate"):
        fast.prepare(root, cfg, root / "original", root / "original/new")
    (root / "original/manifest/train_ready.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        fast.prepare(root, cfg, root / "original", root / "fast")
    assert not (root / "fast").exists()


def test_fast_insufficient_capacity_preserves_labels_and_does_not_publish(
    prepared_inputs, monkeypatch
):
    root, cfg, _ = prepared_inputs
    cfg["target_hours"] = 1

    def stop(*args):
        raise RuntimeError("before byte scan")

    monkeypatch.setattr(original, "freeze", stop)
    with pytest.raises(RuntimeError):
        original.prepare(root, cfg, root / "original", workers=2, jobs=2)
    with pytest.raises(ValueError, match="insufficient eligible capacity"):
        fast.prepare(root, cfg, root / "original", root / "fast")
    assert (root / "original/manifest/sha256.txt").exists()
    assert (root / "fast/frozen/capacity_report.json").exists()
    assert not (root / "fast/training_config_path_only.json").exists()


def legacy_contract(output, digest):
    path = output / "preparation_contract.json"
    saved = json.loads(path.read_text())
    saved["input_sha256"][str(compatibility.FAST_MODULE)] = digest
    path.write_text(json.dumps(saved))
    seal = output / "sha256.txt"
    lines = []
    for line in seal.read_text().splitlines():
        old, relative = line.split("  ", 1)
        lines.append(f"{_sha(path) if relative == path.name else old}  {relative}\n")
    seal.write_text("".join(lines))


def test_reviewed_compatibility_target_matches_delivered_fast_module():
    assert _sha(compatibility.FAST_MODULE) == compatibility.REVIEWED_TARGET


@pytest.mark.parametrize("old_sha", sorted(compatibility.REVIEWED_SOURCES))
def test_existing_preparation_survives_reviewed_code_update(staged, monkeypatch, old_sha):
    root, cfg, mfa_calls = staged
    output = root / "fast"
    fast.prepare(root, cfg, root / "original", output)
    legacy_contract(output, old_sha)
    before = {p: _sha(p) for p in output.rglob("*") if p.is_file()}
    monkeypatch.chdir(root)
    monkeypatch.setattr(gpu_stages, "run_stage", lambda *args: {"status": "mock_only"})
    for _ in range(2):
        result = fast.run_stage(root, output, "smoke", "4")
        change = result["preparation_code_compatibility"][0]
        assert change["recorded_sha256"] == old_sha
        assert change["current_sha256"] == compatibility.REVIEWED_TARGET
        for path, digest in before.items():
            assert _sha(path) == digest
    assert len(mfa_calls) == 1
    audits = list(output.glob("code_compatibility_*.json"))
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text())
    assert audit["dataset_checks_relaxed"] is False
    assert audit["preparation_contract_sha256"] == _sha(output / "preparation_contract.json")


@pytest.mark.parametrize("problem", ["data", "unknown_source", "target", "other_code", "seal"])
def test_compatibility_never_bypasses_other_identity_changes(staged, monkeypatch, problem):
    root, cfg, _ = staged
    output = root / "fast"
    fast.prepare(root, cfg, root / "original", output)
    legacy_contract(output, next(iter(compatibility.REVIEWED_SOURCES)))
    contract_path = output / "preparation_contract.json"
    if problem == "data":
        (root / "original/manifest/train_ready.jsonl").write_text("changed labels")
    elif problem == "unknown_source":
        legacy_contract(output, "0" * 64)
    elif problem == "target":
        sha = compatibility._sha
        monkeypatch.setattr(
            compatibility,
            "_sha",
            lambda path: "f" * 64 if path == compatibility.FAST_MODULE else sha(path),
        )
    else:
        saved = json.loads(contract_path.read_text())
        if problem == "other_code":
            other = str(Path(original.__file__))
            saved["input_sha256"][other] = next(iter(compatibility.REVIEWED_SOURCES))
        else:
            saved["unexpected"] = "edited without updating original seal"
        contract_path.write_text(json.dumps(saved))
        if problem == "other_code":
            legacy_contract(output, next(iter(compatibility.REVIEWED_SOURCES)))
    with pytest.raises(ValueError):
        compatibility.verify_preparation_inputs(output)
    assert not list(output.glob("code_compatibility_*.json"))
