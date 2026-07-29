from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.hotwords.registry import HotwordEntry, write_hotword_table
from qwen_hotword.hotwords.simulation import SimulatedHotwordCase
from qwen_hotword.inference.hotword_prompt import (
    DEFAULT_PT_BR_PROMPT_TEMPLATE,
    build_hotword_prompt,
    strict_phrase_match,
)
from qwen_hotword.inference.prompt_smoke import (
    load_validation_manifest,
    run_prompt_smoke,
    select_prompt_smoke_samples,
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
) -> tuple[Path, Path, Path, Path, Path, list[HotwordEntry], list[SimulatedHotwordCase]]:
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_asr"}\n', encoding="utf-8")
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
        _entry("hw-short-a", "sol", (1, 2, 3, 4)),
        _entry("hw-short-b", "luz", (2, 3, 4, 1, 2)),
        _entry("hw-medium-a", "mercado", (1, 2, 3, 4, 1, 2, 3, 4)),
        _entry("hw-medium-b", "central", (2, 3, 4, 1, 2, 3, 4, 1, 2)),
        _entry(
            "hw-long-a",
            "computador moderno",
            (1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1),
        ),
        _entry(
            "hw-long-b",
            "telefone antigo",
            (2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3),
        ),
        _entry("hw-control", "fantasma", (3, 4, 1, 2, 3, 4)),
    ]
    hotword_path = tmp_path / "hotwords.jsonl"
    write_hotword_table(hotword_path, hotwords)
    all_ids = tuple(entry.hotword_id for entry in hotwords)
    cases = [
        SimulatedHotwordCase(
            case_id="case-p0",
            sample_id="sample-p0",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-short-a",),
        ),
        SimulatedHotwordCase(
            case_id="case-p1",
            sample_id="sample-p1",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-short-a", "hw-short-b"),
        ),
        SimulatedHotwordCase(
            case_id="case-p2",
            sample_id="sample-p2",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-medium-a",),
        ),
        SimulatedHotwordCase(
            case_id="case-p3",
            sample_id="sample-p3",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-medium-a", "hw-medium-b"),
        ),
        SimulatedHotwordCase(
            case_id="case-p4",
            sample_id="sample-p4",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-long-a",),
        ),
        SimulatedHotwordCase(
            case_id="case-p5",
            sample_id="sample-p5",
            case_type="positive",
            language="pt-BR",
            active_hotword_ids=all_ids,
            expected_hotword_ids=("hw-long-a", "hw-long-b"),
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
        "sample-p0": "Eu vi o sol hoje.",
        "sample-p1": "A luz veio depois do sol.",
        "sample-p2": "O mercado abriu.",
        "sample-p3": "O mercado central abriu.",
        "sample-p4": "Comprei um computador moderno.",
        "sample-p5": "O computador moderno fica perto do telefone antigo.",
        "sample-n0": "Esta frase não contém a palavra de controle.",
        "sample-n1": "Outra frase comum para avaliação.",
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
    return model, manifest, vocab, hotword_path, case_path, hotwords, cases


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
                else "transcrição sem palavra alvo"
            )
        elif sample_id == "sample-n0":
            prediction = context.rsplit(": ", maxsplit=1)[-1]
        elif sample_id == "sample-n1":
            prediction = self.references[sample_id]
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


def test_strict_matching_normalizes_case_punctuation_and_complete_words() -> None:
    assert strict_phrase_match("Falamos de SÃO—PAULO hoje.", "são paulo")
    assert strict_phrase_match("Coisa!", "coisa")
    assert not strict_phrase_match("coisas", "coisa")
    assert not strict_phrase_match("relacionamentos", "relacionamento")
    assert not strict_phrase_match("mercado muito central", "mercado central")


def test_prompt_builder_handles_empty_and_multiple_hotwords() -> None:
    assert build_hotword_prompt(()) == ""
    prompt = build_hotword_prompt(("Mercado Central", "telefone antigo"))

    assert prompt == DEFAULT_PT_BR_PROMPT_TEMPLATE.format(
        hotwords="Mercado Central, telefone antigo"
    )
    assert build_hotword_prompt(("sol", "SOL")) == DEFAULT_PT_BR_PROMPT_TEMPLATE.format(
        hotwords="sol"
    )
    with pytest.raises(ValueError, match="placeholder"):
        build_hotword_prompt(("sol",), template="sem campo")


def test_selection_is_deterministic_stratified_and_has_single_and_multi(
    tmp_path: Path,
) -> None:
    _, manifest, _, _, _, hotwords, cases = _assets(tmp_path)
    records = load_validation_manifest(manifest)

    first = select_prompt_smoke_samples(
        records,
        hotwords,
        cases,
        positive_count=6,
        negative_count=2,
        seed=17,
    )
    second = select_prompt_smoke_samples(
        records,
        hotwords,
        cases,
        positive_count=6,
        negative_count=2,
        seed=17,
    )

    assert first == second
    positives = [sample for sample in first if sample.case_type == "positive"]
    assert any(len(sample.expected_hotword_ids) == 1 for sample in positives)
    assert any(len(sample.expected_hotword_ids) > 1 for sample in positives)
    reasons = " ".join(sample.selection_reason for sample in positives)
    assert "short_4_7" in reasons
    assert "medium_8_12" in reasons
    assert "long_13_plus" in reasons
    for sample in first:
        if sample.case_type == "negative":
            assert sample.negative_control_surface is not None
            assert not strict_phrase_match(
                sample.reference_text,
                sample.negative_control_surface,
            )


def test_prompt_smoke_loads_once_builds_three_paths_and_computes_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, manifest, vocab, hotwords, cases, _, _ = _assets(tmp_path)
    configs, wrappers, loader = _fake_loader(manifest)
    output = tmp_path / "output"

    report = run_prompt_smoke(
        model_path=model,
        validation_manifest_path=manifest,
        vocab_path=vocab,
        hotword_table_path=hotwords,
        cases_path=cases,
        output_dir=output,
        positive_count=6,
        negative_count=2,
        seed=17,
        model_loader=loader,
    )

    assert len(configs) == 1
    assert len(wrappers) == 1
    assert len(wrappers[0].calls) == 16
    assert all(call["context"] == "" for call in wrappers[0].calls[:8])
    assert all(call["return_time_stamps"] is False for call in wrappers[0].calls)
    oracle_calls = wrappers[0].calls[8:14]
    assert all(call["context"] for call in oracle_calls)
    assert any("," in str(call["context"]) for call in oracle_calls)
    negative_calls = wrappers[0].calls[14:]
    assert all(call["context"] for call in negative_calls)
    assert "completed=16/16" in capsys.readouterr().out

    baseline_metrics = report["baseline_metrics"]
    oracle_metrics = report["oracle_prompt_metrics"]
    negative_metrics = report["negative_prompt_control_metrics"]
    assert isinstance(baseline_metrics, dict)
    assert isinstance(oracle_metrics, dict)
    assert isinstance(negative_metrics, dict)
    assert baseline_metrics["hotword_recall"] == 0.0
    assert oracle_metrics["hotword_recall"] == 1.0
    assert oracle_metrics["additional_correct_hotwords_vs_baseline"] == 9
    assert negative_metrics["prompt_hallucination_rate"] == 0.5
    assert report["test_set_used"] is False
    assert report["ctc_retrieval_used"] is False
    assert all(
        (output / filename).is_file()
        for filename in (
            "sample_selection.json",
            "baseline_predictions.jsonl",
            "oracle_predictions.jsonl",
            "negative_prompt_predictions.jsonl",
            "prompt_smoke_report.json",
        )
    )

    oracle_rows = [
        json.loads(line)
        for line in (output / "oracle_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["actual_prompt"] for row in oracle_rows)
    assert all(row["mode"] == "oracle_prompt" for row in oracle_rows)
    assert any(len(row["injected_hotwords"]) > 1 for row in oracle_rows)

    with pytest.raises(FileExistsError, match="refusing to .*overwrite"):
        run_prompt_smoke(
            model_path=model,
            validation_manifest_path=manifest,
            vocab_path=vocab,
            hotword_table_path=hotwords,
            cases_path=cases,
            output_dir=output,
            positive_count=6,
            negative_count=2,
            model_loader=loader,
        )
    assert len(configs) == 1


@pytest.mark.parametrize("split", ["train", "test"])
def test_prompt_manifest_rejects_non_validation_and_sealed_test(
    tmp_path: Path,
    split: str,
) -> None:
    manifest = tmp_path / f"{split}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "split": split,
                "language": "pt-BR",
                "audio_path": "/audio.wav",
                "text": "texto",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    expected = "sealed test" if split == "test" else "not formal validation"
    with pytest.raises(ValueError, match=expected):
        load_validation_manifest(manifest)


def test_progress_flag_does_not_change_selected_outputs(tmp_path: Path) -> None:
    model, manifest, vocab, hotwords, cases, _, _ = _assets(tmp_path)
    _, _, loader_a = _fake_loader(manifest)
    _, _, loader_b = _fake_loader(manifest)

    run_prompt_smoke(
        model_path=model,
        validation_manifest_path=manifest,
        vocab_path=vocab,
        hotword_table_path=hotwords,
        cases_path=cases,
        output_dir=tmp_path / "with-progress",
        positive_count=6,
        negative_count=2,
        seed=31,
        model_loader=loader_a,
        print_progress=True,
    )
    run_prompt_smoke(
        model_path=model,
        validation_manifest_path=manifest,
        vocab_path=vocab,
        hotword_table_path=hotwords,
        cases_path=cases,
        output_dir=tmp_path / "without-progress",
        positive_count=6,
        negative_count=2,
        seed=31,
        model_loader=loader_b,
        print_progress=False,
    )

    for filename in (
        "sample_selection.json",
        "baseline_predictions.jsonl",
        "oracle_predictions.jsonl",
        "negative_prompt_predictions.jsonl",
    ):
        left = _without_timing(tmp_path / "with-progress" / filename)
        right = _without_timing(tmp_path / "without-progress" / filename)
        assert left == right


def _without_timing(path: Path) -> object:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row.pop("inference_seconds", None)
    return rows
