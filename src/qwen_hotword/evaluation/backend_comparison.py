"""Small final-ASR backend comparison: immutable samples, shared prompts and honest metrics."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

LANGUAGES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
CONDITIONS = ("original", "noise10db")
PROMPTS = {
    "en": "The following terms are spelling references. Use them only when supported by the audio. "
    "Do not add a term that was not spoken.\n{words}",
    "es": "Los siguientes términos son referencias de escritura. Úsalos solo si el audio los "
    "respalda. No añadas términos que no se hayan pronunciado.\n{words}",
    "pt": "Os termos abaixo são referências de escrita. Use-os apenas quando houver apoio no "
    "áudio. Não acrescente termos que não foram falados.\n{words}",
}


@contextmanager
def block_legacy_qwen_vllm_registration() -> Iterator[None]:
    """Disable ONLY qwen-asr0.0.6's optional legacy vLLM import in a child process.

    That optional import is wrapped by upstream in try/except. Stopping it before
    execution also prevents a newer vLLM import from changing HF config registries
    during a Transformers-only run. No installed source file is modified.
    """

    class BlockOptionalBackend(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None,
            target: ModuleType | None = None,
        ) -> importlib.machinery.ModuleSpec | None:
            if fullname.startswith("qwen_asr.core.vllm_backend"):
                raise ImportError("legacy qwen-asr vLLM import disabled for the Transformers child")
            return None

    blocker = BlockOptionalBackend()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    """One process owns a run. Equal completed files can be reused, never replaced."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing artifact differs; choose a NEW output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def public_wrapper_eos_ids(generate: Any) -> list[int]:
    """Qwen0.0.6's public generate overrides the thinker's config EOS by default."""
    eos = inspect.signature(generate).parameters["eos_token_id"].default
    result = list(eos) if isinstance(eos, list) else [eos]
    if not result or any(not isinstance(token, int) or token < 0 for token in result):
        raise ValueError("cannot determine the public Qwen wrapper's EOS policy")
    return result


def context(language: str, words: list[str]) -> str:
    if (
        language not in LANGUAGES
        or len(words) > 7
        or any(
            not word.strip()
            or any(token in word for token in ("\n", "\r", "<|", "|>", "<asr_text>"))
            for word in words
        )
    ):
        raise ValueError("invalid saved Top7 surfaces")
    return PROMPTS[language].format(words="\n".join(words)) if words else ""


def normalize(text: str) -> tuple[str, ...]:
    """Ignore punctuation/case/spacing only. Retain accents, ñ, digits and other letters."""
    value = unicodedata.normalize("NFC", text).casefold()
    return tuple(
        "".join(" " if unicodedata.category(c).startswith("P") else c for c in value).split()
    )


def contains(text: str, word: str) -> bool:
    tokens, target = normalize(text), normalize(word)
    return bool(target) and any(
        tokens[start : start + len(target)] == target
        for start in range(len(tokens) - len(target) + 1)
    )


def distance(reference: tuple[str, ...], prediction: tuple[str, ...]) -> int:
    previous = list(range(len(prediction) + 1))
    for row, token in enumerate(reference, 1):
        current = [row]
        for column, other in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(token != other),
                )
            )
        previous = current
    return previous[-1]


def saved_source_rows(source: Path, cfg: dict[str, Any]) -> Iterable[dict[str, Any]]:
    evaluation = read(source / "evaluation_inputs.json")
    report = read(source / "retrieval/report.json")
    if (
        report.get("status") != "completed"
        or evaluation["checkpoints"]["new70"] != cfg["source_head_sha256"]
        or evaluation["new_index_sha256"] != cfg["source_index_sha256"]
    ):
        raise ValueError("requires completed, matching latest 100-class/new70 retrieval results")
    # File hashes give a fixed order independent of transcripts or whether a term
    # was recovered. Read only until the six small buckets are full.
    for path in sorted((source / "retrieval/cells").glob("*.json")):
        row = read(path)
        identity = row.get("identity", {})
        if identity.get("tag") != "new70":
            continue
        if row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"}):
            raise ValueError(f"saved retrieval cell was modified: {path}")
        if identity["checkpoint_sha256"] != cfg["source_head_sha256"]:
            raise ValueError("new70 cell checkpoint mismatch")
        yield {**row, "source_cell": str(path), "source_cell_sha256": sha(path)}


def compare_rows(
    requests: list[dict[str, Any]],
    outputs: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    details = []
    expected = {request["request_id"] for request in requests}
    if any(set(outputs[backend]) != expected for backend in ("transformers", "vllm")):
        raise ValueError("backend request IDs are incomplete or differ")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        a, b = (outputs[backend][request["request_id"]] for backend in ("transformers", "vllm"))
        for row in (a, b):
            if row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"}):
                raise ValueError("backend result cell changed")
            if row["request_sha256"] != digest(request) or row["generation_limit_reached"]:
                raise ValueError("request identity differs or generation was truncated")
        if a["input_token_ids"] != b["input_token_ids"]:
            raise ValueError("actual Prompt IDs differ; this is not a comparable backend pair")
        if a["waveform_sha256"] != b["waveform_sha256"]:
            raise ValueError("normalized waveforms differ")
        exact = a["text"] == b["text"]
        normalized = normalize(a["text"]) == normalize(b["text"])
        reference = request["reference_text"]
        targets = [word for word in request["words"] if contains(reference, word)]
        row = {
            "request_id": request["request_id"],
            "language": request["language"],
            "condition": request["condition"],
            "mode": request["mode"],
            "audio_path": request["audio_path"],
            "words": request["words"],
            "reference_text": reference,
            "transformers": a["text"],
            "vllm": b["text"],
            "classification": "exact"
            if exact
            else ("format_only" if normalized else "word_difference"),
            "generated_token_ids_equal": a["content_token_ids"] == b["content_token_ids"],
            "reference_words": len(normalize(reference)),
            "transformers_word_errors": distance(normalize(reference), normalize(a["text"])),
            "vllm_word_errors": distance(normalize(reference), normalize(b["text"])),
            "reference_supported_injected_words": targets,
            "transformers_target_hits": [word for word in targets if contains(a["text"], word)],
            "vllm_target_hits": [word for word in targets if contains(b["text"], word)],
        }
        details.append(row)
        for group in (
            "overall",
            f"language/{request['language']}",
            f"mode/{request['mode']}",
            f"{request['language']}/{request['condition']}/{request['mode']}",
        ):
            groups[group].append(row)
    summaries = {}
    for group, rows in groups.items():
        ref_words = sum(row["reference_words"] for row in rows)
        summaries[group] = {
            "pairs": len(rows),
            "exact": sum(row["classification"] == "exact" for row in rows),
            "format_only": sum(row["classification"] == "format_only" for row in rows),
            "word_difference": sum(row["classification"] == "word_difference" for row in rows),
            "exact_rate": sum(row["classification"] == "exact" for row in rows) / len(rows),
            "token_equal": sum(row["generated_token_ids_equal"] for row in rows),
            "transformers_wer": sum(row["transformers_word_errors"] for row in rows) / ref_words
            if ref_words
            else None,
            "vllm_wer": sum(row["vllm_word_errors"] for row in rows) / ref_words
            if ref_words
            else None,
            "target_hits_differ": sum(
                row["transformers_target_hits"] != row["vllm_target_hits"] for row in rows
            ),
        }
    return details, {
        "status": "completed",
        "scope": "small_H200_final_ASR_backend_comparison",
        "by_group": summaries,
        "normalization": "NFC/casefold/punctuation/spacing; retain accents and ñ",
        "references_sent_to_model": False,
        "ctc_rerun": False,
        "training_started": False,
        "not_a_310p_or_two_stage_cache_validation": True,
        "all_equal_is_not_assumed": True,
    }
