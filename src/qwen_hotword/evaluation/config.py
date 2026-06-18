from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class EvalConfigError(ValueError):
    """Raised when an evaluation asset config is invalid."""


@dataclass(frozen=True)
class SourceConfig:
    name: str
    dataset: str
    language: str
    root: Path
    include_splits: list[str] = field(default_factory=list)
    include_locales: list[str] = field(default_factory=list)
    use_for_hotwords: bool = True
    use_for_cases: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    max_utterances_per_source: int
    seed: int


@dataclass(frozen=True)
class HotwordConfig:
    max_per_language: int
    min_chars: int
    min_words: int
    max_words: int
    max_occurrences: int
    source_utt_limit: int


@dataclass(frozen=True)
class CaseConfig:
    cases_per_language: int
    active_hotwords_per_case: int
    positive_ratio: float
    negative_ratio: float
    confusable_ratio: float
    no_hotword_ratio: float


@dataclass(frozen=True)
class EvalAssetConfig:
    output_dir: Path
    sampling: SamplingConfig
    hotwords: HotwordConfig
    cases: CaseConfig
    sources: list[SourceConfig]


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvalConfigError(f"{field} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise EvalConfigError(f"missing required field: {field}.{key}")
    return mapping[key]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvalConfigError("expected a list of strings")
    return [str(item) for item in value]


def _bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise EvalConfigError("expected a boolean value")


def load_eval_asset_config(path: str | Path) -> EvalAssetConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise EvalConfigError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    root = _mapping(raw, "root")
    sampling = _mapping(root.get("sampling", {}), "sampling")
    hotwords = _mapping(root.get("hotwords", {}), "hotwords")
    cases = _mapping(root.get("cases", {}), "cases")

    source_values = root.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise EvalConfigError("sources must be a non-empty list")

    sources: list[SourceConfig] = []
    for index, source_value in enumerate(source_values):
        source = _mapping(source_value, f"sources[{index}]")
        sources.append(
            SourceConfig(
                name=str(_required(source, "name", f"sources[{index}]")),
                dataset=str(_required(source, "dataset", f"sources[{index}]")),
                language=str(_required(source, "language", f"sources[{index}]")),
                root=Path(str(_required(source, "root", f"sources[{index}]"))).expanduser(),
                include_splits=_string_list(source.get("include_splits")),
                include_locales=_string_list(source.get("include_locales")),
                use_for_hotwords=_bool(source.get("use_for_hotwords"), default=True),
                use_for_cases=_bool(source.get("use_for_cases"), default=True),
            )
        )

    return EvalAssetConfig(
        output_dir=Path(str(root.get("output_dir", "outputs/eval_assets"))).expanduser(),
        sampling=SamplingConfig(
            max_utterances_per_source=int(sampling.get("max_utterances_per_source", 2000)),
            seed=int(sampling.get("seed", 20260616)),
        ),
        hotwords=HotwordConfig(
            max_per_language=int(hotwords.get("max_per_language", 800)),
            min_chars=int(hotwords.get("min_chars", 4)),
            min_words=int(hotwords.get("min_words", 1)),
            max_words=int(hotwords.get("max_words", 4)),
            max_occurrences=int(hotwords.get("max_occurrences", 40)),
            source_utt_limit=int(hotwords.get("source_utt_limit", 20)),
        ),
        cases=CaseConfig(
            cases_per_language=int(cases.get("cases_per_language", 1000)),
            active_hotwords_per_case=int(cases.get("active_hotwords_per_case", 10)),
            positive_ratio=float(cases.get("positive_ratio", 0.50)),
            negative_ratio=float(cases.get("negative_ratio", 0.25)),
            confusable_ratio=float(cases.get("confusable_ratio", 0.15)),
            no_hotword_ratio=float(cases.get("no_hotword_ratio", 0.10)),
        ),
        sources=sources,
    )
