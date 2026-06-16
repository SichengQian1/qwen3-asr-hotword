from __future__ import annotations

import random
from collections import defaultdict
from difflib import SequenceMatcher

from qwen_hotword.evaluation.config import CaseConfig
from qwen_hotword.evaluation.records import EvalCase, Hotword, Utterance
from qwen_hotword.evaluation.text import (
    contains_token_sequence,
    language_code_for_id,
    tokenize_words,
)


def build_eval_cases(
    utterances: list[Utterance],
    hotwords: list[Hotword],
    config: CaseConfig,
    *,
    seed: int,
    eval_stage: str,
) -> list[EvalCase]:
    rng = random.Random(seed)
    hotwords_by_language: dict[str, list[Hotword]] = defaultdict(list)
    utterances_by_language: dict[str, list[Utterance]] = defaultdict(list)
    for hotword in hotwords:
        hotwords_by_language[hotword.language].append(hotword)
    for utterance in utterances:
        utterances_by_language[utterance.language].append(utterance)

    cases: list[EvalCase] = []
    for language in sorted(utterances_by_language):
        language_utterances = utterances_by_language[language]
        language_hotwords = hotwords_by_language.get(language, [])
        if not language_utterances:
            continue
        expected_by_utt = _expected_hotwords_by_utterance(
            language_utterances,
            language_hotwords,
        )
        counts = _case_counts(config)
        cases.extend(
            _build_language_cases(
                language=language,
                utterances=language_utterances,
                hotwords=language_hotwords,
                expected_by_utt=expected_by_utt,
                counts=counts,
                config=config,
                eval_stage=eval_stage,
                rng=rng,
            )
        )
    return sorted(cases, key=lambda item: item.case_id)


def _expected_hotwords_by_utterance(
    utterances: list[Utterance],
    hotwords: list[Hotword],
) -> dict[str, list[str]]:
    hotword_sequences = [
        (hotword.hotword_id, tuple(tokenize_words(hotword.normalized)))
        for hotword in hotwords
    ]
    expected: dict[str, list[str]] = {}
    for utterance in utterances:
        tokens = tokenize_words(utterance.text)
        matches = [
            hotword_id
            for hotword_id, sequence in hotword_sequences
            if contains_token_sequence(tokens, sequence)
        ]
        if matches:
            expected[utterance.utt_id] = matches[:3]
    return expected


def _case_counts(config: CaseConfig) -> dict[str, int]:
    total = config.cases_per_language
    positive = int(total * config.positive_ratio)
    negative = int(total * config.negative_ratio)
    confusable = int(total * config.confusable_ratio)
    no_hotword = max(total - positive - negative - confusable, int(total * config.no_hotword_ratio))
    return {
        "positive": positive,
        "negative": negative,
        "confusable": confusable,
        "no_hotword": no_hotword,
    }


def _build_language_cases(
    *,
    language: str,
    utterances: list[Utterance],
    hotwords: list[Hotword],
    expected_by_utt: dict[str, list[str]],
    counts: dict[str, int],
    config: CaseConfig,
    eval_stage: str,
    rng: random.Random,
) -> list[EvalCase]:
    positive_pool = [
        utterance for utterance in utterances if utterance.utt_id in expected_by_utt
    ]
    negative_pool = [
        utterance for utterance in utterances if utterance.utt_id not in expected_by_utt
    ]
    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)
    cases: list[EvalCase] = []
    case_index = 1

    for case_type, pool_name in (
        ("positive", "positive"),
        ("confusable", "positive"),
        ("negative", "negative"),
        ("no_hotword", "negative"),
    ):
        target_count = counts[case_type]
        pool = positive_pool if pool_name == "positive" else negative_pool
        if not pool:
            continue
        for utterance in _cycle_sample(pool, target_count):
            expected_ids = expected_by_utt.get(utterance.utt_id, [])
            active_ids = _active_hotwords(
                case_type=case_type,
                expected_ids=expected_ids,
                hotwords=hotwords,
                count=config.active_hotwords_per_case,
                rng=rng,
            )
            distractors = [
                hotword_id for hotword_id in active_ids if hotword_id not in expected_ids
            ]
            cases.append(
                EvalCase(
                    case_id=(
                        f"{eval_stage}_{language_code_for_id(language)}_"
                        f"{case_type}_{case_index:06d}"
                    ),
                    eval_stage=eval_stage,
                    case_type=case_type,
                    utt_id=utterance.utt_id,
                    dataset=utterance.dataset,
                    split=utterance.split,
                    language=utterance.language,
                    audio_path=utterance.audio_path,
                    reference_text=utterance.text,
                    active_hotword_ids=active_ids,
                    expected_hotword_ids=expected_ids,
                    distractor_hotword_ids=distractors,
                )
            )
            case_index += 1
    return cases


def _cycle_sample(items: list[Utterance], count: int) -> list[Utterance]:
    if count <= 0:
        return []
    output: list[Utterance] = []
    while len(output) < count:
        output.extend(items[: count - len(output)])
    return output


def _active_hotwords(
    *,
    case_type: str,
    expected_ids: list[str],
    hotwords: list[Hotword],
    count: int,
    rng: random.Random,
) -> list[str]:
    if case_type == "no_hotword" or count <= 0:
        return []
    hotword_by_id = {hotword.hotword_id: hotword for hotword in hotwords}
    active: list[str] = list(expected_ids)
    if case_type == "confusable" and expected_ids:
        active.extend(_confusable_ids(expected_ids[0], hotwords, hotword_by_id, limit=count))
    available = [
        hotword.hotword_id
        for hotword in hotwords
        if hotword.hotword_id not in set(active)
    ]
    rng.shuffle(available)
    active.extend(available[: max(count - len(active), 0)])
    return active[:count]


def _confusable_ids(
    expected_id: str,
    hotwords: list[Hotword],
    hotword_by_id: dict[str, Hotword],
    *,
    limit: int,
) -> list[str]:
    expected = hotword_by_id.get(expected_id)
    if expected is None:
        return []
    scored: list[tuple[float, str]] = []
    for hotword in hotwords:
        if hotword.hotword_id == expected_id:
            continue
        score = SequenceMatcher(
            None,
            expected.normalized,
            hotword.normalized,
        ).ratio()
        scored.append((score, hotword.hotword_id))
    scored.sort(reverse=True)
    return [hotword_id for _, hotword_id in scored[: max(limit - 1, 0)]]
