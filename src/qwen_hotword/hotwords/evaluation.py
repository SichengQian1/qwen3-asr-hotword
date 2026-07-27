from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    score_hotwords,
)
from qwen_hotword.hotwords.simulation import SimulatedHotwordCase
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    load_feature_shard,
)


@dataclass(frozen=True)
class HotwordCaseScore:
    case_id: str
    sample_id: str
    case_type: str
    active_hotword_ids: tuple[str, ...]
    expected_hotword_ids: tuple[str, ...]
    effective_time_steps: int
    decoded_token_count: int
    ranked_matches: tuple[HotwordMatch, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "case_type": self.case_type,
            "active_hotword_ids": list(self.active_hotword_ids),
            "expected_hotword_ids": list(self.expected_hotword_ids),
            "effective_time_steps": self.effective_time_steps,
            "decoded_token_count": self.decoded_token_count,
            "ranked_matches": [match.to_dict() for match in self.ranked_matches],
        }


@dataclass(frozen=True)
class HotwordThresholdMetrics:
    threshold: float
    positive_cases: int
    negative_cases: int
    expected_hotwords: int
    selected_hotwords: int
    true_positive_hotwords: int
    false_positive_hotwords: int
    positive_case_hits: int
    positive_top1_hits: int
    negative_false_positive_cases: int
    precision: float
    recall: float
    f1: float
    positive_case_hit_rate: float
    positive_top1_accuracy: float
    negative_case_false_positive_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HotwordRankingMetrics:
    k: int
    positive_cases: int
    expected_hotwords: int
    retrieved_expected_hotwords: int
    positive_case_hits: int
    recall_at_k: float
    positive_case_hit_rate_at_k: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_hotword_ranking(
    case_scores: list[HotwordCaseScore],
    *,
    ks: tuple[int, ...] = (1, 3, 5),
    expected_hotword_ids: set[str] | None = None,
) -> list[HotwordRankingMetrics]:
    if not ks or any(k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ranking ks must contain unique positive values")
    results: list[HotwordRankingMetrics] = []
    for k in sorted(ks):
        positive_cases = 0
        expected_total = 0
        retrieved_total = 0
        positive_case_hits = 0
        for case in case_scores:
            expected = set(case.expected_hotword_ids)
            if expected_hotword_ids is not None:
                expected.intersection_update(expected_hotword_ids)
            if not expected:
                continue
            positive_cases += 1
            expected_total += len(expected)
            ranked_ids = {
                match.hotword_id for match in case.ranked_matches[:k]
            }
            retrieved = len(expected & ranked_ids)
            retrieved_total += retrieved
            if retrieved:
                positive_case_hits += 1
        results.append(
            HotwordRankingMetrics(
                k=k,
                positive_cases=positive_cases,
                expected_hotwords=expected_total,
                retrieved_expected_hotwords=retrieved_total,
                positive_case_hits=positive_case_hits,
                recall_at_k=_safe_ratio(retrieved_total, expected_total),
                positive_case_hit_rate_at_k=_safe_ratio(
                    positive_case_hits,
                    positive_cases,
                ),
            )
        )
    return results


def evaluate_hotword_threshold(
    case_scores: list[HotwordCaseScore],
    *,
    threshold: float,
    top_k: int,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
    minimum_top1_margin: float,
) -> HotwordThresholdMetrics:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    positive_cases = 0
    negative_cases = 0
    expected_total = 0
    selected_total = 0
    true_positive_total = 0
    positive_case_hits = 0
    positive_top1_hits = 0
    negative_false_positive_cases = 0
    for case in case_scores:
        expected = set(case.expected_hotword_ids)
        if expected:
            positive_cases += 1
            expected_total += len(expected)
        else:
            negative_cases += 1
        qualified = [
            match
            for match in case.ranked_matches
            if match.score >= threshold
            and match.edit_ratio <= maximum_edit_ratio
            and match.posterior_confidence >= minimum_posterior_confidence
        ]
        if (
            len(qualified) > 1
            and qualified[0].score - qualified[1].score < minimum_top1_margin
        ):
            selected: list[HotwordMatch] = []
        else:
            selected = qualified[:top_k]
        selected_ids = {match.hotword_id for match in selected}
        true_positives = len(selected_ids & expected)
        selected_total += len(selected_ids)
        true_positive_total += true_positives
        if expected and true_positives:
            positive_case_hits += 1
        if expected and selected and selected[0].hotword_id in expected:
            positive_top1_hits += 1
        if not expected and selected:
            negative_false_positive_cases += 1

    false_positive_total = selected_total - true_positive_total
    precision = _safe_ratio(true_positive_total, selected_total)
    recall = _safe_ratio(true_positive_total, expected_total)
    return HotwordThresholdMetrics(
        threshold=threshold,
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        expected_hotwords=expected_total,
        selected_hotwords=selected_total,
        true_positive_hotwords=true_positive_total,
        false_positive_hotwords=false_positive_total,
        positive_case_hits=positive_case_hits,
        positive_top1_hits=positive_top1_hits,
        negative_false_positive_cases=negative_false_positive_cases,
        precision=precision,
        recall=recall,
        f1=_safe_ratio(2.0 * precision * recall, precision + recall),
        positive_case_hit_rate=_safe_ratio(positive_case_hits, positive_cases),
        positive_top1_accuracy=_safe_ratio(positive_top1_hits, positive_cases),
        negative_case_false_positive_rate=_safe_ratio(
            negative_false_positive_cases,
            negative_cases,
        ),
    )


def evaluate_hotword_scoring(
    checkpoint_path: str | Path,
    cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    hotwords: list[HotwordEntry],
    cases: list[SimulatedHotwordCase],
    output_dir: str | Path,
    *,
    device: Any,
    batch_size: int = 128,
    thresholds: tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ),
    top_k: int = 3,
    minimum_phonemes: int = 4,
    maximum_edit_ratio: float = 0.35,
    posterior_weight: float = 0.25,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.03,
    target_precision: float = 0.90,
    maximum_negative_case_false_positive_rate: float = 0.03,
    ranking_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        ctc_head_config,
    )

    if batch_size <= 0:
        raise ValueError("hotword evaluation batch size must be positive")
    if cache.split != "validation":
        raise ValueError("hotword threshold tuning accepts only validation features")
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("threshold sweep values must be non-empty and within [0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("threshold sweep values must be unique")
    if (
        not ranking_ks
        or any(k <= 0 for k in ranking_ks)
        or len(set(ranking_ks)) != len(ranking_ks)
    ):
        raise ValueError("ranking ks must contain unique positive values")
    for name, value in (
        ("target_precision", target_precision),
        (
            "maximum_negative_case_false_positive_rate",
            maximum_negative_case_false_positive_rate,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    checkpoint = Path(checkpoint_path).expanduser()
    payload = _load_checkpoint(checkpoint, vocab)
    head = build_ctc_head_from_checkpoint(payload)
    if not isinstance(head, TemporalUpsampleCtcHead) or head.time_upsampling_factor != 2:
        raise ValueError(
            "deployment hotword evaluation requires the temporal_upsample Head "
            "with time_upsampling_factor=2"
        )
    head = head.to(device=device, dtype=torch.float32)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    entry_by_id = {entry.hotword_id: entry for entry in hotwords}
    if len(entry_by_id) != len(hotwords):
        raise ValueError("hotword table contains duplicate IDs")
    case_by_sample: dict[str, SimulatedHotwordCase] = {}
    for case in cases:
        missing_ids = set(case.active_hotword_ids) - set(entry_by_id)
        if missing_ids:
            raise ValueError(f"case {case.case_id} refers to unknown hotwords: {missing_ids}")
        if case.sample_id in case_by_sample:
            raise ValueError(f"duplicate case sample ID: {case.sample_id}")
        case_by_sample[case.sample_id] = case
    cached_ids = {sample_id for shard in cache.shards for sample_id in shard.sample_ids}
    missing_samples = set(case_by_sample) - cached_ids
    if missing_samples:
        raise ValueError(
            f"{len(missing_samples)} simulated cases are absent from the validation cache"
        )

    scoring_config = HotwordScoringConfig(
        score_threshold=0.0,
        top_k=max(len(hotwords), 1),
        minimum_phonemes=minimum_phonemes,
        maximum_edit_ratio=1.0,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
    )
    scores: list[HotwordCaseScore] = []
    with torch.no_grad():
        for descriptor in cache.shards:
            wanted = set(descriptor.sample_ids) & set(case_by_sample)
            if not wanted:
                continue
            samples = [
                sample
                for sample in load_feature_shard(
                    descriptor,
                    num_classes=len(vocab.tokens),
                )
                if sample.sample_id in wanted
            ]
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden_states = pad_sequence(
                    [sample.hidden_states for sample in batch],
                    batch_first=True,
                    padding_value=0.0,
                ).to(device=device, dtype=torch.float32)
                input_lengths = torch.tensor(
                    [sample.hidden_states.shape[0] for sample in batch],
                    dtype=torch.long,
                    device=device,
                )
                logits = head(hidden_states, input_lengths=input_lengths)
                effective_lengths = head.output_lengths(input_lengths)
                for row, sample in enumerate(batch):
                    case = case_by_sample[sample.sample_id]
                    active = [entry_by_id[item] for item in case.active_hotword_ids]
                    result = score_hotwords(
                        logits[row],
                        input_length=int(effective_lengths[row].item()),
                        hotwords=active,
                        config=scoring_config,
                        blank_id=0,
                    )
                    scores.append(
                        HotwordCaseScore(
                            case_id=case.case_id,
                            sample_id=case.sample_id,
                            case_type=case.case_type,
                            active_hotword_ids=case.active_hotword_ids,
                            expected_hotword_ids=case.expected_hotword_ids,
                            effective_time_steps=result.effective_time_steps,
                            decoded_token_count=len(result.decoded_token_ids),
                            ranked_matches=result.ranked_matches,
                        )
                    )
            del samples
    if len(scores) != len(cases):
        raise RuntimeError(
            f"hotword evaluation scored {len(scores)} cases, expected {len(cases)}"
        )
    scores.sort(key=lambda item: item.case_id)

    ranking_metrics = evaluate_hotword_ranking(scores, ks=ranking_ks)
    length_groups = {
        "phonemes_4_7": {
            entry.hotword_id
            for entry in hotwords
            if 4 <= len(entry.token_ids) <= 7
        },
        "phonemes_8_12": {
            entry.hotword_id
            for entry in hotwords
            if 8 <= len(entry.token_ids) <= 12
        },
        "phonemes_13_18": {
            entry.hotword_id
            for entry in hotwords
            if 13 <= len(entry.token_ids) <= 18
        },
        "phonemes_19_24": {
            entry.hotword_id
            for entry in hotwords
            if 19 <= len(entry.token_ids) <= 24
        },
    }
    ranking_by_length = {
        name: [
            metrics.to_dict()
            for metrics in evaluate_hotword_ranking(
                scores,
                ks=ranking_ks,
                expected_hotword_ids=hotword_ids,
            )
        ]
        for name, hotword_ids in length_groups.items()
        if hotword_ids
    }
    sweep = [
        evaluate_hotword_threshold(
            scores,
            threshold=threshold,
            top_k=top_k,
            maximum_edit_ratio=maximum_edit_ratio,
            minimum_posterior_confidence=minimum_posterior_confidence,
            minimum_top1_margin=minimum_top1_margin,
        )
        for threshold in sorted(thresholds)
    ]
    qualified = [
        metrics
        for metrics in sweep
        if metrics.precision >= target_precision
        and metrics.negative_case_false_positive_rate
        <= maximum_negative_case_false_positive_rate
    ]
    pool = qualified or sweep
    recommended = max(
        pool,
        key=lambda metrics: (
            metrics.recall if qualified else metrics.f1,
            metrics.precision,
            -metrics.negative_case_false_positive_rate,
            -metrics.threshold,
        ),
    )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    case_scores_path = destination / "hotword_case_scores.jsonl"
    report_path = destination / "hotword_scoring_report.json"
    _write_jsonl(case_scores_path, [score.to_dict() for score in scores])
    report: dict[str, object] = {
        "purpose": "simulated_validation_hotword_scoring_and_false_positive_control",
        "checkpoint_path": str(checkpoint),
        "head_config": ctc_head_config(head),
        "validation_cache_dir": str(cache.root),
        "validation_case_count": len(scores),
        "hotword_count": len(hotwords),
        "active_hotwords_per_case": sorted(
            {len(case.active_hotword_ids) for case in cases}
        ),
        "scoring_config": {
            "top_k": top_k,
            "minimum_phonemes": minimum_phonemes,
            "maximum_edit_ratio": maximum_edit_ratio,
            "posterior_weight": posterior_weight,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_top1_margin": minimum_top1_margin,
        },
        "control_targets": {
            "minimum_precision": target_precision,
            "maximum_negative_case_false_positive_rate": (
                maximum_negative_case_false_positive_rate
            ),
        },
        "ranking_metrics": {
            "definition": (
                "expected hotword occurrences found in the first K scored active "
                "candidates; no score threshold or ambiguity suppression is applied"
            ),
            "threshold_applied": False,
            "overall": [metrics.to_dict() for metrics in ranking_metrics],
            "by_phoneme_length": ranking_by_length,
        },
        "threshold_sweep": [metrics.to_dict() for metrics in sweep],
        "recommended_operating_point": {
            **recommended.to_dict(),
            "meets_control_targets": bool(qualified),
        },
        "case_scores_path": str(case_scores_path),
        "time_axis_policy": "temporal_upsample_2x_only",
        "evaluation_scope": "simulated hotwords on formal validation cases",
        "test_set_used": False,
        "status": "pass",
    }
    _write_json(report_path, report)
    return report


def _load_checkpoint(path: Path, vocab: PhonemeVocab) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"CTC Head checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"CTC Head checkpoint is invalid: {path}")
    if payload.get("input_dimension") != 1024:
        raise ValueError("CTC Head checkpoint input dimension is not 1024")
    if payload.get("num_classes") != len(vocab.tokens):
        raise ValueError("CTC Head checkpoint class count differs from the vocabulary")
    if payload.get("blank_id") != 0 or payload.get("vocab_tokens") != list(vocab.tokens):
        raise ValueError("CTC Head checkpoint vocabulary identity does not match")
    return payload


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
