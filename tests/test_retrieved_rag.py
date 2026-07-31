from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.hotwords.registry import HotwordEntry, write_hotword_table
from qwen_hotword.hotwords.simulation import SimulatedHotwordCase
from qwen_hotword.inference.prompt_smoke import load_validation_manifest
from qwen_hotword.inference.retrieved_rag import (
    load_ctc_case_scores,
    run_retrieved_rag,
    select_ctc_matches,
    select_retrieved_rag_samples,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab


def _vocab() -> PhonemeVocab:
    tokens = ("<blank>", "a", "b", "c", "d")
    return PhonemeVocab(
        tokens=tokens,
        phone_tokens=tokens[1:],
        token_to_id={token: index for index, token in enumerate(tokens)},
    )


def _entry(
    hotword_id: str,
    surface: str,
    token_ids: tuple[int, ...],
) -> HotwordEntry:
    vocab = _vocab()
    return HotwordEntry(
        hotword_id=hotword_id,
        language="pt-BR",
        surface=surface,
        normalized=surface.casefold(),
        words=tuple(surface.casefold().split()),
        pronunciation=" ".join(vocab.tokens[token_id] for token_id in token_ids),
        phoneme_tokens=tuple(vocab.tokens[token_id] for token_id in token_ids),
        token_ids=token_ids,
        source="test",
        validation_occurrences=2,
    )


def _assets(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3_asr"}\n',
        encoding="utf-8",
    )
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{}}\n',
        encoding="utf-8",
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"tokens": list(_vocab().tokens)}),
        encoding="utf-8",
    )
    hotwords = [
        _entry("hw-short", "sol", (1, 2, 3, 4)),
        _entry("hw-medium", "mercado", (1, 2, 3, 4, 1, 2, 3, 4)),
        _entry(
            "hw-long",
            "computador moderno",
            (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1),
        ),
        _entry("hw-wrong", "fantasma", (4, 3, 2, 1, 4, 3)),
    ]
    hotword_path = tmp_path / "hotwords.jsonl"
    write_hotword_table(hotword_path, hotwords)
    all_ids = tuple(entry.hotword_id for entry in hotwords)
    cases = [
        SimulatedHotwordCase(
            case_id="case-p0",
            sample_id="sample-p0",
            case_type="positive_confusable",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-short",),
        ),
        SimulatedHotwordCase(
            case_id="case-p1",
            sample_id="sample-p1",
            case_type="positive_confusable",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-medium",),
        ),
        SimulatedHotwordCase(
            case_id="case-p2",
            sample_id="sample-p2",
            case_type="positive_confusable",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-long",),
        ),
        SimulatedHotwordCase(
            case_id="case-n0",
            sample_id="sample-n0",
            case_type="negative",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=(),
        ),
        SimulatedHotwordCase(
            case_id="case-n1",
            sample_id="sample-n1",
            case_type="negative",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=(),
        ),
    ]
    references = {
        "sample-p0": "Eu vi o sol.",
        "sample-p1": "O mercado abriu.",
        "sample-p2": "Comprei um computador moderno.",
        "sample-n0": "Esta é uma frase comum.",
        "sample-n1": "Outra frase sem palavra especial.",
    }
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "id": sample_id,
                    "split": "validation",
                    "language": "pt-BR",
                    "audio_path": str(tmp_path / f"{sample_id}.wav"),
                    "text": reference,
                },
                ensure_ascii=False,
            )
            + "\n"
            for sample_id, reference in references.items()
        ),
        encoding="utf-8",
    )
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    ranked = {
        "case-p0": [
            _match("hw-short", "sol", score=0.95, edit_ratio=0.1),
        ],
        "case-p1": [
            _match("hw-medium", "mercado", score=0.94, edit_ratio=0.1),
            _match("hw-wrong", "fantasma", score=0.90, edit_ratio=0.2),
        ],
        "case-p2": [
            _match("hw-long", "computador moderno", score=0.85, edit_ratio=0.1),
        ],
        "case-n0": [
            _match("hw-wrong", "fantasma", score=0.93, edit_ratio=0.2),
        ],
        "case-n1": [
            _match("hw-short", "sol", score=0.99, edit_ratio=0.5),
        ],
    }
    score_path = tmp_path / "scores.jsonl"
    score_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "sample_id": case.sample_id,
                    "case_type": case.case_type,
                    "active_hotword_ids": list(case.active_hotword_ids),
                    "expected_hotword_ids": list(case.expected_hotword_ids),
                    "ranked_matches": ranked[case.case_id],
                },
                ensure_ascii=False,
            )
            + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    return model, manifest, vocab, hotword_path, case_path, score_path


def _match(
    hotword_id: str,
    surface: str,
    *,
    score: float,
    edit_ratio: float,
) -> dict[str, object]:
    return {
        "hotword_id": hotword_id,
        "surface": surface,
        "score": score,
        "edit_ratio": edit_ratio,
        "posterior_confidence": 0.9,
    }


class _FakeWrapper:
    backend = "transformers"
    max_new_tokens = 256
    max_inference_batch_size = 1

    def __init__(self, references: dict[str, str]) -> None:
        self.references = references
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        sample_id = Path(str(kwargs["audio"])).stem
        context = str(kwargs["context"])
        if not context:
            prediction = (
                self.references[sample_id]
                if sample_id.startswith("sample-n")
                else "transcrição sem alvo"
            )
        elif sample_id == "sample-n0":
            prediction = "Esta é uma frase comum com fantasma."
        else:
            prediction = self.references[sample_id]
        return [SimpleNamespace(text=prediction)]


def _fake_loader(
    manifest: Path,
) -> tuple[list[object], list[_FakeWrapper], Any]:
    references = {
        str(row["id"]): str(row["text"])
        for row in (json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines())
    }
    configs: list[object] = []
    wrappers: list[_FakeWrapper] = []

    def loader(config: object) -> _FakeWrapper:
        configs.append(config)
        wrapper = _FakeWrapper(references)
        wrappers.append(wrapper)
        return wrapper

    return configs, wrappers, loader


def test_match_selection_applies_edit_guard_and_margin(tmp_path: Path) -> None:
    *_, score_path = _assets(tmp_path)
    scores = {item.case_id: item for item in load_ctc_case_scores(score_path)}

    assert [
        item.hotword_id
        for item in select_ctc_matches(
            scores["case-p1"],
            threshold=0.86,
            top_k=3,
            maximum_edit_ratio=0.35,
            minimum_posterior_confidence=0.0,
            minimum_top1_margin=0.0,
        )
    ] == ["hw-medium", "hw-wrong"]
    assert not select_ctc_matches(
        scores["case-n1"],
        threshold=0.86,
        top_k=3,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
    )
    assert not select_ctc_matches(
        scores["case-p1"],
        threshold=0.86,
        top_k=3,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.05,
    )


def test_selection_is_deterministic_stratified_and_enriches_ctc_false_positives(
    tmp_path: Path,
) -> None:
    _, manifest, vocab_path, hotword_path, case_path, score_path = _assets(tmp_path)
    from qwen_hotword.hotwords.registry import load_hotword_table
    from qwen_hotword.hotwords.simulation import load_simulated_cases
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab

    records = load_validation_manifest(manifest)
    vocab = load_phoneme_vocab(vocab_path)
    hotwords = load_hotword_table(hotword_path, vocab=vocab)
    cases = load_simulated_cases(case_path)
    scores = load_ctc_case_scores(score_path)
    first = select_retrieved_rag_samples(
        records,
        hotwords,
        cases,
        scores,
        positive_count=3,
        negative_count=2,
        seed=17,
    )
    second = select_retrieved_rag_samples(
        records,
        hotwords,
        cases,
        scores,
        positive_count=3,
        negative_count=2,
        seed=17,
    )

    assert first == second
    positive_reasons = " ".join(
        sample.selection_reason for sample in first if sample.case_type == "positive"
    )
    assert "short_4_7" in positive_reasons
    assert "medium_8_12" in positive_reasons
    assert "long_13_plus" in positive_reasons
    negative = [sample for sample in first if sample.case_type == "negative"]
    assert any("ctc_false_positive_stress" in sample.selection_reason for sample in negative)


def test_retrieved_rag_loads_once_reuses_no_prompt_and_reports_attribution(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, manifest, vocab, hotwords, cases, scores = _assets(tmp_path)
    configs, wrappers, loader = _fake_loader(manifest)
    output = tmp_path / "output"

    report = run_retrieved_rag(
        model_path=model,
        validation_manifest_path=manifest,
        vocab_path=vocab,
        hotword_table_path=hotwords,
        cases_path=cases,
        ctc_case_scores_path=scores,
        output_dir=output,
        positive_count=3,
        negative_count=2,
        seed=17,
        model_loader=loader,
    )

    assert len(configs) == 1
    assert len(wrappers) == 1
    assert len(wrappers[0].calls) == 11
    assert all(call["context"] == "" for call in wrappers[0].calls[:5])
    assert all(call["return_time_stamps"] is False for call in wrappers[0].calls)
    assert "completed=11/11" in capsys.readouterr().out

    assert report["test_set_used"] is False
    assert report["ctc_retrieval_used"] is True
    assert report["baseline_metrics"]["hotword_recall"] == 0.0
    assert report["retrieved_prompt_metrics"]["hotword_recall"] == pytest.approx(2 / 3)
    assert report["oracle_prompt_metrics"]["hotword_recall"] == 1.0
    attribution = report["pipeline_attribution"]
    assert attribution["wrong_candidates_injected"] == 2
    assert attribution["wrong_injected_candidates_written"] == 1
    assert attribution["newly_written_wrong_candidates_vs_baseline"] == 1
    assert len(attribution["hotword_rescue_cases_vs_baseline"]) == 2
    assert report["inference"]["model_load_count"] == 1
    assert report["inference"]["retrieved_baseline_reuses"] == 2

    retrieved_rows = [
        json.loads(line)
        for line in (output / "retrieved_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(retrieved_rows) == 5
    assert sum(row["inference_reused_from_baseline"] for row in retrieved_rows) == 2
    assert any(len(row["injected_hotwords"]) > 1 for row in retrieved_rows)
    assert all(
        (output / filename).is_file()
        for filename in (
            "sample_selection.json",
            "baseline_predictions.jsonl",
            "retrieved_predictions.jsonl",
            "oracle_predictions.jsonl",
            "retrieved_rag_report.json",
        )
    )

    with pytest.raises(FileExistsError, match="refusing to .*overwrite"):
        run_retrieved_rag(
            model_path=model,
            validation_manifest_path=manifest,
            vocab_path=vocab,
            hotword_table_path=hotwords,
            cases_path=cases,
            ctc_case_scores_path=scores,
            output_dir=output,
            positive_count=3,
            negative_count=2,
            model_loader=loader,
        )
    assert len(configs) == 1
