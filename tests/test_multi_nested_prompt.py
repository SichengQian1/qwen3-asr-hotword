from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.hotwords.multi_nested import (
    load_multi_nested_case_scores,
    load_multi_nested_cases,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table, write_hotword_table
from qwen_hotword.inference.multi_nested_prompt import (
    DEFAULT_GROUP_QUOTAS,
    run_multi_nested_prompt_eval,
    select_multi_nested_prompt_samples,
)
from qwen_hotword.inference.prompt_smoke import load_validation_manifest
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab


def _tokens(index: int) -> tuple[int, ...]:
    return tuple(((index // (4**position)) % 4) + 1 for position in range(4))


def _match(hotword_id: str, surface: str, score: float) -> dict[str, object]:
    return {
        "hotword_id": hotword_id,
        "surface": surface,
        "language": "pt-BR",
        "score": score,
        "edit_similarity": score,
        "edit_distance": 0,
        "edit_ratio": 0.0,
        "posterior_confidence": 0.95,
        "decoded_start": 0,
        "decoded_end": 4,
        "start_step": 0,
        "end_step": 5,
    }


def _assets(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_asr"}\n')
    (model / "model.safetensors.index.json").write_text('{"weight_map":{}}\n')
    vocab = PhonemeVocab(
        tokens=("<blank>", "a", "b", "c", "d"),
        phone_tokens=("a", "b", "c", "d"),
        token_to_id={"<blank>": 0, "a": 1, "b": 2, "c": 3, "d": 4},
    )
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"tokens": list(vocab.tokens)}))
    entries = [
        HotwordEntry(
            hotword_id=f"h{index:03d}",
            language="pt-BR",
            surface=f"termo{index:03d}",
            normalized=f"termo{index:03d}",
            words=(f"termo{index:03d}",),
            pronunciation=" ".join(vocab.tokens[item] for item in _tokens(index)),
            phoneme_tokens=tuple(vocab.tokens[item] for item in _tokens(index)),
            token_ids=_tokens(index),
            source="test",
            validation_occurrences=1,
        )
        for index in range(100)
    ]
    hotwords_path = tmp_path / "hotwords.jsonl"
    write_hotword_table(hotwords_path, entries)
    surfaces = {entry.hotword_id: entry.surface for entry in entries}
    all_ids = [entry.hotword_id for entry in entries]
    manifest_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    case_index = 0
    expected_by_group = {
        "single_hotword": 1,
        "two_independent": 2,
        "three_independent": 3,
        "nested_short_only": 1,
        "nested_long_present": 1,
        "nested_family_plus_two": 3,
        "negative": 0,
    }
    for group, count in DEFAULT_GROUP_QUOTAS.items():
        for _ in range(count):
            sample_id = f"sample-{case_index:03d}"
            case_id = f"case-{case_index:03d}"
            base = (case_index * 3) % 90
            independent = tuple(
                f"h{base + offset:03d}" for offset in range(expected_by_group[group])
            )
            family_ids: tuple[str, ...] = ()
            containment = independent
            longest = independent
            if group in {"nested_short_only", "nested_long_present", "nested_family_plus_two"}:
                short_id = f"h{base:03d}"
                long_id = f"h{base + 1:03d}"
                family_id = f"family-{case_index:03d}"
                family_ids = (family_id,)
                family_rows.append(
                    {
                        "family_id": family_id,
                        "short_hotword_id": short_id,
                        "long_hotword_id": long_id,
                        "short_surface": surfaces[short_id],
                        "long_surface": surfaces[long_id],
                    }
                )
                if group == "nested_short_only":
                    containment = longest = (short_id,)
                    independent = ()
                elif group == "nested_long_present":
                    containment = (short_id, long_id)
                    longest = (long_id,)
                    independent = ()
                else:
                    extra = (f"h{base + 2:03d}", f"h{base + 3:03d}")
                    containment = (short_id, long_id, *extra)
                    longest = (long_id, *extra)
                    independent = extra
            reference = " ".join(surfaces[item] for item in containment) or "frase sem alvo"
            words = reference.split()
            spans = {
                item: (words.index(surfaces[item]), words.index(surfaces[item]) + 1)
                for item in containment
            }
            manifest_rows.append(
                {
                    "id": sample_id,
                    "split": "validation",
                    "language": "pt-BR",
                    "audio_path": str(tmp_path / f"{sample_id}.wav"),
                    "text": reference,
                }
            )
            case_rows.append(
                {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "audio_path": str(tmp_path / f"{sample_id}.wav"),
                    "reference_text": reference,
                    "normalized_reference_text": reference,
                    "language": "pt-BR",
                    "primary_group": group,
                    "expected_hotword_ids": list(containment),
                    "expected_surfaces": [surfaces[item] for item in containment],
                    "expected_word_spans": {key: list(value) for key, value in spans.items()},
                    "containment_expected_ids": list(containment),
                    "longest_match_expected_ids": list(longest),
                    "active_hotword_ids": all_ids,
                    "nested_family_ids": list(family_ids),
                    "hard_negative_ids": [],
                    "independent_expected_ids": list(independent),
                    "selection_reason": "test",
                }
            )
            ranked_ids = list(dict.fromkeys((*containment, "h099", "h098", "h097", "h096")))[:5]
            operating_ids = list(dict.fromkeys((*containment, "h099")))
            score_rows.append(
                {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "primary_group": group,
                    "effective_time_steps": 20,
                    "decoded_token_count": 5,
                    "ranking_top5": [
                        _match(item, surfaces[item], 0.99 - i * 0.01)
                        for i, item in enumerate(ranked_ids)
                    ],
                    "operating_matches": [
                        _match(item, surfaces[item], 0.95 - i * 0.01)
                        for i, item in enumerate(operating_ids)
                    ],
                }
            )
            case_index += 1
    paths = {
        "model": model,
        "vocab": vocab_path,
        "hotwords": hotwords_path,
        "families": tmp_path / "families.jsonl",
        "manifest": tmp_path / "validation.jsonl",
        "cases": tmp_path / "cases.jsonl",
        "scores": tmp_path / "scores.jsonl",
        "report": tmp_path / "ctc_report.json",
    }
    for key, rows in (
        ("families", family_rows),
        ("manifest", manifest_rows),
        ("cases", case_rows),
        ("scores", score_rows),
    ):
        paths[key].write_text("".join(json.dumps(row) + "\n" for row in rows))
    paths["report"].write_text(
        json.dumps(
            {
                "test_set_used": False,
                "scoring_config": {
                    "threshold": 0.86,
                    "top_k": 5,
                    "maximum_edit_ratio": 0.35,
                    "posterior_weight": 0.25,
                    "minimum_posterior_confidence": 0.0,
                    "minimum_top1_margin": 0.0,
                    "minimum_phonemes": 4,
                    "time_axis": "temporal_upsample_2x_only",
                },
            }
        )
    )
    return paths


class _FakeWrapper:
    backend = "transformers"

    def __init__(self, references: dict[str, str]) -> None:
        self.references = references
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        sample_id = Path(str(kwargs["audio"])).stem
        if kwargs["context"]:
            return [SimpleNamespace(text=self.references[sample_id])]
        return [SimpleNamespace(text="frase sem alvo")]


def test_fixed_selection_and_fake_prompt_run(tmp_path: Path) -> None:
    paths = _assets(tmp_path)
    vocab = load_phoneme_vocab(paths["vocab"])
    selected = select_multi_nested_prompt_samples(
        load_validation_manifest(paths["manifest"]),
        load_hotword_table(paths["hotwords"], vocab=vocab),
        load_multi_nested_cases(paths["cases"]),
        load_multi_nested_case_scores(paths["scores"]),
    )
    assert len(selected) == 50
    assert {
        group: sum(item.primary_group == group for item in selected)
        for group in DEFAULT_GROUP_QUOTAS
    } == DEFAULT_GROUP_QUOTAS
    assert len({item.rag_sample.audio_path for item in selected}) == 50
    assert any(item.redundant_family_ids for item in selected)

    references = {
        row.sample_id: row.reference_text
        for row in load_validation_manifest(paths["manifest"]).values()
    }
    wrappers: list[_FakeWrapper] = []

    def loader(_: object) -> _FakeWrapper:
        wrapper = _FakeWrapper(references)
        wrappers.append(wrapper)
        return wrapper

    output = tmp_path / "output"
    report = run_multi_nested_prompt_eval(
        model_path=paths["model"],
        validation_manifest_path=paths["manifest"],
        vocab_path=paths["vocab"],
        hotword_table_path=paths["hotwords"],
        families_path=paths["families"],
        cases_path=paths["cases"],
        ctc_case_scores_path=paths["scores"],
        ctc_report_path=paths["report"],
        output_dir=output,
        model_loader=loader,
        print_progress=False,
    )
    assert len(wrappers) == 1
    assert report["inference"]["model_load_count"] == 1
    assert report["selection"]["total_cases"] == 50
    assert report["selection"]["positive_cases"] == 40
    assert report["retrieved_prompt_metrics"]["hotword_recall"] == 1.0
    assert report["oracle_prompt_metrics"]["hotword_recall"] == 1.0
    assert report["prompt_safety"]["redundant_family_candidates_injected"] > 0
    assert all(
        (output / name).is_file()
        for name in (
            "sample_selection.json",
            "baseline_predictions.jsonl",
            "retrieved_predictions.jsonl",
            "oracle_predictions.jsonl",
            "multi_nested_prompt_report.json",
        )
    )
    with pytest.raises(FileExistsError):
        run_multi_nested_prompt_eval(
            model_path=paths["model"],
            validation_manifest_path=paths["manifest"],
            vocab_path=paths["vocab"],
            hotword_table_path=paths["hotwords"],
            families_path=paths["families"],
            cases_path=paths["cases"],
            ctc_case_scores_path=paths["scores"],
            ctc_report_path=paths["report"],
            output_dir=output,
            model_loader=loader,
            print_progress=False,
        )
