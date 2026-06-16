from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from qwen_hotword.evaluation.config import HotwordConfig
from qwen_hotword.evaluation.records import Hotword, Utterance
from qwen_hotword.evaluation.text import (
    language_code_for_id,
    normalize_text,
    stopwords_for_language,
    tokenize_words,
)


@dataclass
class _Candidate:
    tokens: tuple[str, ...]
    language: str
    source_dataset: str
    frequency: int = 0
    source_utt_ids: list[str] = field(default_factory=list)


def build_hotwords(
    utterances: list[Utterance],
    config: HotwordConfig,
    *,
    phonemizer_backend: str,
    require_ipa: bool,
) -> list[Hotword]:
    candidates = _collect_candidates(utterances, config)
    selected: list[Hotword] = []
    by_language: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates.values():
        by_language[candidate.language].append(candidate)

    for language in sorted(by_language):
        ranked = sorted(by_language[language], key=_candidate_sort_key)
        for index, candidate in enumerate(ranked[: config.max_per_language], start=1):
            surface = " ".join(candidate.tokens)
            ipa, phoneme_tokens, phoneme_source = phonemize_hotword(
                surface,
                language=language,
                backend=phonemizer_backend,
                require_ipa=require_ipa,
            )
            lang_code = language_code_for_id(language)
            selected.append(
                Hotword(
                    hotword_id=f"hw_{lang_code}_{index:06d}",
                    language=language,
                    surface=surface,
                    normalized=normalize_text(surface),
                    ipa=ipa,
                    phoneme_tokens=phoneme_tokens,
                    phoneme_source=phoneme_source,
                    source_dataset=candidate.source_dataset,
                    source_utt_ids=candidate.source_utt_ids[: config.source_utt_limit],
                    hotword_type="phrase" if len(candidate.tokens) > 1 else "word",
                    frequency=candidate.frequency,
                )
            )
    return selected


def phonemize_hotword(
    text: str,
    *,
    language: str,
    backend: str,
    require_ipa: bool,
) -> tuple[str | None, list[str], str]:
    if backend == "none":
        if require_ipa:
            raise RuntimeError("IPA was required, but phonemizer backend is 'none'")
        return None, _fallback_tokens(text), "normalized_char_fallback"
    if backend != "espeak":
        raise ValueError(f"unsupported phonemizer backend: {backend}")
    try:
        from phonemizer import phonemize
    except ImportError as error:
        if require_ipa:
            raise RuntimeError(
                "phonemizer is not installed; install phonemizer and espeak-ng "
                "or run with --phonemizer none for a dry run"
            ) from error
        return None, _fallback_tokens(text), "normalized_char_fallback"

    ipa = phonemize(
        text,
        language=_espeak_language(language),
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
    )
    ipa = str(ipa).strip()
    if require_ipa and not ipa:
        raise RuntimeError(f"empty IPA output for hotword: {text}")
    return ipa or None, _ipa_tokens(ipa), "espeak"


def _collect_candidates(
    utterances: list[Utterance],
    config: HotwordConfig,
) -> dict[tuple[str, tuple[str, ...]], _Candidate]:
    candidates: dict[tuple[str, tuple[str, ...]], _Candidate] = {}
    for utterance in utterances:
        tokens = tokenize_words(utterance.text)
        if not tokens:
            continue
        seen_in_utterance: set[tuple[str, ...]] = set()
        for width in range(config.min_words, config.max_words + 1):
            if width > len(tokens):
                continue
            for start in range(len(tokens) - width + 1):
                phrase = tuple(tokens[start : start + width])
                if not _is_candidate_phrase(phrase, utterance.language, config):
                    continue
                if phrase in seen_in_utterance:
                    continue
                seen_in_utterance.add(phrase)
                key = (utterance.language, phrase)
                candidate = candidates.setdefault(
                    key,
                    _Candidate(
                        tokens=phrase,
                        language=utterance.language,
                        source_dataset=utterance.dataset,
                    ),
                )
                candidate.frequency += 1
                candidate.source_utt_ids.append(utterance.utt_id)
    return {
        key: candidate
        for key, candidate in candidates.items()
        if candidate.frequency <= config.max_occurrences
    }


def _is_candidate_phrase(
    phrase: tuple[str, ...],
    language: str,
    config: HotwordConfig,
) -> bool:
    stopwords = stopwords_for_language(language)
    if all(token in stopwords for token in phrase):
        return False
    if phrase[0] in stopwords or phrase[-1] in stopwords:
        return False
    char_count = sum(len(token) for token in phrase)
    if char_count < config.min_chars:
        return False
    return not (len(phrase) == 1 and phrase[0] in stopwords)


def _candidate_sort_key(candidate: _Candidate) -> tuple[float, str]:
    token_bonus = 6.0 * (len(candidate.tokens) - 1)
    char_bonus = min(sum(len(token) for token in candidate.tokens), 30) / 3.0
    rarity_penalty = candidate.frequency * 0.15
    score = token_bonus + char_bonus - rarity_penalty
    return (-score, " ".join(candidate.tokens))


def _fallback_tokens(text: str) -> list[str]:
    return [char for char in normalize_text(text) if not char.isspace()]


def _ipa_tokens(ipa: str) -> list[str]:
    return [char for char in ipa if not char.isspace()]


def _espeak_language(language: str) -> str:
    lowered = language.lower().replace("_", "-")
    if lowered.startswith("pt"):
        return "pt-br"
    return "en-us"
