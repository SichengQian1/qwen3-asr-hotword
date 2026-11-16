from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwen_hotword.phonemes.arabic_native import caphi_units, egyptian_units
from qwen_hotword.phonemes.arabic_pronunciation import (
    PronunciationFrontend,
    fixed_caphi,
    restore_surface,
    validate_policy,
    word_pronunciation,
)
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.arabic_refresh import (
    assign_split,
    export_review,
    recording_key,
    waterfill,
)
from qwen_hotword.training.arabic_refresh_train import ReuseOrExtract
from qwen_hotword.training.ctc_overfit import CachedSample, ExperimentRecord
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, FeatureShardDescriptor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy() -> dict:
    return json.loads((ROOT / "configs/arabic_pronunciation_policy.json").read_text())


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("f_i_y", ["f", "iː"]),
        ("2_a_b_u_w", ["ʔ", "a", "b", "uː"]),
        ("h_i_y_a", ["h", "i", "j", "a"]),
        ("3_a_l_i_y_y", ["ʕ", "a", "l", "i", "jː"]),
        ("b_a_y_t", ["b", "a", "j", "t"]),
        ("b_u_w_w", ["b", "u", "wː"]),
    ],
)
def test_caphi_long_vowel_context_and_counterexamples(source: str, expected: list[str]) -> None:
    assert fixed_caphi(source, True)[0] == expected


def test_legacy_frontend_keeps_old_identity_and_behavior() -> None:
    assert caphi_units("f_i_y") == ["f", "i", "j"]
    with pytest.raises(ValueError, match="unsupported_Arabic_letter"):
        egyptian_units("ما_بَقاش")


@pytest.mark.parametrize(
    ("surface", "diac", "expected"),
    [
        ("بيت", "بَيت", ["b", "eː", "t"]),
        ("يوم", "يَوم", ["j", "oː", "m"]),
        ("بيت", "بَيْت", ["b", "a", "j", "t"]),
        ("بيي", "بَيَّ", ["b", "a", "jː", "a"]),
        ("ياخد", "ياخُذ", ["j", "aː", "x", "u", "d"]),
        ("أكتر", "أَكْثَر", ["ʔ", "a", "k", "t", "a", "r"]),
        ("يعني", "يَعْنِي", ["j", "a", "ʕ", "n", "i"]),
    ],
)
def test_document_examples_are_explicit_automatic_rule_cases(
    policy: dict, surface: str, diac: str, expected: list[str]
) -> None:
    assert word_pronunciation(surface, {"diac": diac}, "ar_egy", policy)["phones"] == expected


def test_boundary_recovery_has_trace_and_does_not_claim_human_review(policy: dict) -> None:
    result = word_pronunciation("مبقاش", {"diac": "ما_بَقاش"}, "ar_egy", policy)
    assert result["baseline_issue"] == "unsupported_Arabic_letter:_"
    assert "egy_clitic_boundary" in result["rules"]
    assert "_" not in result["phones"]


def test_two_monophthongs_do_not_shift_other_phones(policy: dict) -> None:
    result = word_pronunciation("بيتوم", {"diac": "بَيتَوم"}, "ar_egy", policy)
    assert result["phones"] == ["b", "eː", "t", "oː", "m"]


def test_do_not_rewrite_an_unaligned_analysis() -> None:
    assert restore_surface("حَأَظُنّ", "حضن") == ("حَأَظُنّ", False)


def test_pronominal_h_is_morphology_gated(policy: dict) -> None:
    assert (
        word_pronunciation("شفته", {"diac": "شُفتُه", "enc0": "3ms_dobj"}, "ar_egy", policy)[
            "phones"
        ][-1]
        == "u"
    )
    assert (
        word_pronunciation("شفته", {"diac": "شُفتُه", "enc0": "0"}, "ar_egy", policy)["phones"][-1]
        == "h"
    )


def test_exceptions_need_review_and_exact_analysis_scope(policy: dict) -> None:
    validate_policy(policy)
    result = word_pronunciation("ده", {"diac": "دَه"}, "ar_egy", policy)
    assert result["exception"] is None and result["exception_candidates"]
    approved = copy.deepcopy(policy)
    entry = approved["exceptions"][1]
    entry["status"] = "approved"
    with pytest.raises(ValueError, match="reviewer"):
        validate_policy(approved)
    entry["reviewer"] = "fixture_reviewer"
    validate_policy(approved)
    assert word_pronunciation("ده", {"diac": "دَه"}, "ar_egy", approved)["phones"] == ["d", "a"]
    assert word_pronunciation("ده", {"diac": "دِه"}, "ar_egy", approved)["exception"] is None


def test_pause_and_link_rules_are_contextual_not_global(policy: dict) -> None:
    policy["rules"]["linked_article"] = True
    a = {"diac": "الكتاب", "caphi": "2_a_l_k_i_t_aa_b"}
    assert word_pronunciation("الكتاب", a, "ar_msa", policy)["phones"][:2] == ["ʔ", "a"]
    assert (
        word_pronunciation("الكتاب", a, "ar_msa", policy, linked=True, previous_phone="iː")[
            "phones"
        ][0]
        == "l"
    )
    a["diac"] = "أحمد"
    assert word_pronunciation("أحمد", a, "ar_msa", policy, linked=True)["phones"][0] == "ʔ"
    policy["rules"]["msa_pause_endings"] = True
    a = {
        "diac": "كِتابًا",
        "caphi": "k_i_t_aa_b_a_n",
        "pos": "noun",
        "cas": "a",
        "stt": "i",
        "enc0": "0",
    }
    assert word_pronunciation("كتابا", a, "ar_msa", policy, pause_after=True)["phones"][-1] == "aː"
    a["stt"] = "c"
    assert word_pronunciation("كتابا", a, "ar_msa", policy, pause_after=True)["phones"][-2:] == [
        "a",
        "n",
    ]


def test_sentence_and_downstream_share_rules_and_punctuation_boundaries(policy: dict) -> None:
    frontend = object.__new__(PronunciationFrontend)
    frontend.policy = policy
    frontend.group = "ar_msa"
    tokens = json.loads((ROOT / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json").read_text())[
        "tokens"
    ]
    frontend.ids = {t: i for i, t in enumerate(tokens)}
    analyses = [
        SimpleNamespace(analyses=[SimpleNamespace(analysis=a)])
        for a in (
            {"diac": "فِي", "caphi": "f_i_y", "source": "lex"},
            {"diac": "الكتاب", "caphi": "2_a_l_k_i_t_aa_b", "source": "lex"},
        )
    ]
    frontend.model = Mock()
    frontend.model.disambiguate.return_value = analyses
    policy["rules"]["linked_article"] = True
    linked = frontend.label("في الكتاب")
    paused = frontend.label("في، الكتاب")
    assert linked["word_traces"][1]["phones"][0] == "l"
    assert paused["word_traces"][1]["phones"][0] == "ʔ"
    assert linked["human_pronunciation_review_completed"] is False
    assert linked["phones"] == frontend.label("في الكتاب", mode="isolated")["phones"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/a/video__abc__seg0002.wav", "video:video"),
        ("/b/video__def__seg0003.wav", "video:video"),
        ("/b/uuid_spk-002_seg-123:456.wav", "recording:uuid"),
        (
            "/x/20001102_1200_1400_VOA_ARB_459.986_467.766.wav",
            "broadcast:20001102_1200_1400_VOA_ARB",
        ),
        (
            "/x/ALJZ_TODHARV_ARB_20070319_145800_1636375_1657050.wav",
            "broadcast:ALJZ_TODHARV_ARB_20070319_145800",
        ),
        ("/x/spk000053-Arabic917-917217.wav", "biaobei-speaker:spk000053"),
    ],
)
def test_recording_identity_is_not_segment_identity(path: str, expected: str) -> None:
    assert recording_key(path) == expected


def test_unknown_recording_layout_fails_instead_of_claiming_disjointness() -> None:
    with pytest.raises(ValueError, match="unknown_recording"):
        recording_key("/audio/random.wav")


def test_waterfill_and_deterministic_recording_partition() -> None:
    assert waterfill({"small": 10, "large": 100, "other": 100}, 100) == {
        "small": 10,
        "large": 45,
        "other": 45,
    }
    with pytest.raises(ValueError, match="capacity"):
        waterfill({"small": 10}, 11)
    c = {"seed": 1, "test_percent": 5, "validation_percent": 10}
    assert assign_split(recording_key("/a/video__id__seg1.wav"), c) == assign_split(
        recording_key("/b/video__id__seg2.wav"), c
    )


def test_feature_reuse_replaces_only_targets_and_rejects_path_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen_hotword.training import arabic_refresh_train as m

    path = tmp_path / "shard.pt"
    path.write_bytes(b"fixture")
    audio = tmp_path / "audio.wav"
    descriptor = FeatureShardDescriptor(0, path, path, 1, 4, 1, 7, sha(path), ("sample",), 2)
    cache = DiskFeatureCache(
        "train", tmp_path, 1, 1, 7, 317477504, "fixture", True, 2, (descriptor,)
    )
    hidden = SimpleNamespace(shape=(4, 1024))
    monkeypatch.setattr(
        m, "load_feature_shard", lambda *a, **kw: [CachedSample("sample", hidden, (72,))]
    )
    lookup = ReuseOrExtract([cache], {"sample": str(audio)}, None)
    record = ExperimentRecord("sample", audio, "text", "ar", (73, 74), 2, 2)
    results, parameters, _ = lookup([record], None)
    assert results[0].hidden_states is hidden
    assert results[0].token_ids == (73, 74) and parameters == 317477504
    assert lookup.wrapper is None
    with pytest.raises(ValueError, match="path identity"):
        lookup([replace(record, audio_path=tmp_path / "other.wav")], None)


def test_review_delivery_contains_real_traces_not_reviewed_claims(
    tmp_path: Path, policy: dict
) -> None:
    trace = word_pronunciation("في", {"diac": "فِي", "caphi": "f_i_y"}, "ar_msa", policy)
    row = {
        "source": "/source.json",
        "row": 1,
        "group": "ar_msa",
        "text": "في",
        "audio_path": "/real/path.wav",
        "word_traces": [trace],
    }
    (tmp_path / "labels.jsonl").write_text(json.dumps(row) + "\n")
    write_json(tmp_path / "report.json", {"labels_sha256": sha(tmp_path / "labels.jsonl")})
    result = export_review(tmp_path)
    assert result["changed_rows_vs_legacy_rules"] == 1
    assert result["human_pronunciation_review_completed"] is False
    item = json.loads((tmp_path / "review_delivery/ar_msa_top300.json").read_text())[0]
    assert item["surface"] == "في" and item["review_status"] == "pending"
    listening = tmp_path / "review_delivery/listening.json"
    reviewed = json.loads(listening.read_text())
    reviewed[0]["reviewer"] = "human_notes_must_survive"
    listening.write_text(json.dumps(reviewed))
    export_review(tmp_path)
    assert json.loads(listening.read_text())[0]["reviewer"] == "human_notes_must_survive"


def test_freeze_stratifies_resumes_and_preserves_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen_hotword.training import arabic_refresh as m
    from scripts import arabic_diagnostic_common as common

    source, labels, output = (tmp_path / name for name in ("source", "labels", "new"))
    source.mkdir()
    labels.mkdir()
    config = {
        "source_run": str(source),
        "train_hours": 0.012,
        "heldout_hours_per_group": 0.0008,
        "test_hours_per_group": 0.0008,
        "seed": 42,
        "validation_percent": 10,
        "test_percent": 5,
        "metadata_workers": 2,
        "quality_exclusions": None,
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    rows = []
    for group in ("ar_egy", "ar_msa"):
        for s in range(4):
            for split, count in (
                ("train", 5 if group == "ar_egy" else 20),
                ("validation", 3),
                ("test", 3),
            ):
                i, found = 0, 0
                while found < count:
                    name = f"{group}-{s}-{split}-{i}__hash__seg0001.wav"
                    i += 1
                    if assign_split(recording_key(name), config) != split:
                        continue
                    audio = tmp_path / name
                    audio.write_bytes(name.encode())
                    found += 1
                    rows.append(
                        {
                            "source": f"/{group}-{s}.json",
                            "row": len(rows) + 1,
                            "group": group,
                            "text": "fixture",
                            "audio_path": str(audio),
                            "duration_seconds": 1.0,
                            "ids": [72],
                            "word_traces": [],
                            "ctc_minimum_input_length": 1,
                        }
                    )
    (labels / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(labels / "contract.json", {})
    write_json(
        labels / "report.json",
        {
            "status": "automatic_labels_completed",
            "identity": {},
            "num_classes": 167,
            "vocab_sha256": "fixture",
            "labels_sha256": sha(labels / "labels.jsonl"),
        },
    )
    manifests = {}
    for lang in ("en", "ar"):
        for split in ("train", "validation"):
            path = source / f"{lang}-{split}.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": f"{lang}-{split}",
                        "audio_sha256": f"{lang}-{split}",
                        "source": "/unused.json",
                        "source_row": 0,
                    }
                )
                + "\n"
            )
            manifests[f"{lang}/{split}"] = {"path": str(path), "sha256": sha(path), "samples": 1}
    Path(rows[2]["audio_path"]).write_bytes(Path(rows[1]["audio_path"]).read_bytes())
    protected = rows[0]
    old = {
        **protected,
        "id": "en-ar-v1-" + sha(Path(protected["audio_path"])),
        "audio_sha256": sha(Path(protected["audio_path"])),
        "source_row": protected["row"],
        "language": "ar",
        "source_group": "ar_egy",
        "split": "validation",
        "phoneme_token_ids": [72],
        "ctc_time_upsampling_factor": 2,
        "experiment": "full-ctc-v1",
    }
    old_path = Path(manifests["ar/validation"]["path"])
    old_path.write_text(json.dumps(old) + "\n")
    manifests["ar/validation"]["sha256"] = sha(old_path)
    checkpoint = source / "old_head.pt"
    checkpoint.write_bytes(b"fixture")
    baseline = {
        "identity": {},
        "plan": {"manifests": manifests, "vocabularies": {"ar": {"sha256": "fixture"}}},
        "head": {"best_checkpoint_path": str(checkpoint)},
    }
    monkeypatch.setattr(common, "completed_run", lambda _: baseline)
    monkeypatch.setattr(common, "recheck", lambda _: None)
    monkeypatch.setattr(m, "read_audio_metadata", lambda _: SimpleNamespace(duration_seconds=1.0))
    result = m.freeze(tmp_path, config_path, labels, output, False)
    assert result["status"] == "prepared"
    selected = {
        split: list(m.iter_jsonl(output / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }
    assert sum(r["duration_seconds"] for r in selected["train"]) >= 43.2
    assert result["manifests"]["ar/paired_validation"]["samples"] == 1
    assert result["rejections"]["duplicate_audio"] >= 1
    assert all(r["audio_sha256"] != old["audio_sha256"] for part in selected.values() for r in part)
    for split in ("validation", "test"):
        assert len({r["source"] for r in selected[split]}) == 8
    recording_splits = {}
    for split, records in selected.items():
        for row in records:
            assert recording_splits.setdefault(row["recording_key"], split) == split
    assert (output / "english/train.jsonl").read_bytes() == (source / "en-train.jsonl").read_bytes()
    assert m.freeze(tmp_path, config_path, labels, output, True) == result
    config["seed"] += 1
    write_json(config_path, config)
    with pytest.raises(ValueError, match="identity changed"):
        m.freeze(tmp_path, config_path, labels, output, True)
