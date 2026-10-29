"""Real official Phonetisaurus rewriting in a checked existing MFA environment."""

from __future__ import annotations

import importlib.metadata
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, runtime_versions, sha, write_json
from qwen_hotword.training.stage_progress import stage_progress


def missing_graphemes(word: str, meta: dict[str, Any], symbols: set[str]) -> list[str]:
    """Mirror the official preprocessing BEFORE its potentially lossy filtering."""
    rewritten = unicodedata.normalize("NFKD", word) if meta["unicode_decomposition"] else word
    return sorted({c for c in rewritten if c not in set(meta["graphemes"]) or c not in symbols})


def input_profile(
    word: str,
    meta: dict[str, Any],
    symbols: set[str],
    *,
    allow_exact_nfc: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Use NFC only if every composed letter exists in BOTH pinned alphabets.

    This is an explicit input adapter, not accent removal or a model-file edit.
    Unsupported letters are never filtered. Both checks are retained in output.
    """
    published_missing = missing_graphemes(word, meta, symbols)
    exact = {**meta, "unicode_decomposition": False}
    if (
        allow_exact_nfc
        and meta["unicode_decomposition"]
        and published_missing
        and not missing_graphemes(word, exact, symbols)
    ):
        return exact, published_missing
    return meta, published_missing


def run(config: dict[str, Any], assets: Path, root: Path, request: Path, output: Path) -> None:
    # Runtime imports intentionally stay outside the default ASR Python environment.
    import pynini  # type: ignore[import-not-found]
    import pywrapfst  # type: ignore[import-not-found]
    from montreal_forced_aligner.g2p import generator  # type: ignore[import-not-found]
    from montreal_forced_aligner.g2p.generator import (  # type: ignore[import-not-found]
        PhonetisaurusRewriter,
    )

    version = importlib.metadata.version("montreal-forced-aligner")
    if version not in runtime_versions(config):
        raise ValueError(f"unreviewed MFA runtime {version}; no environment changes attempted")
    pin(root / config["vocab"], config["vocab_sha256"])
    vocab = load_phoneme_vocab(root / config["vocab"])
    rows = [json.loads(s) for s in request.read_text(encoding="utf-8").splitlines() if s.strip()]
    if any(r["language"] not in config["models"] or not isinstance(r["word"], str) for r in rows):
        raise ValueError("invalid worker input")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output.open("x", encoding="utf-8") as stream:
        for lang, spec in config["models"].items():
            selected = [r for r in rows if r["language"] == lang]
            if not selected:
                continue
            for relative, identity in spec["files"].items():
                pin(assets / lang / relative, identity["sha256"])
            model_dir = assets / lang / "g2p" / spec["dialect"]
            meta = json.loads((model_dir / "meta.json").read_text())
            if meta["architecture"] != "phonetisaurus":
                raise ValueError("new models must use the declared Phonetisaurus architecture")
            suffix = spec["symbol_suffix"]
            phones = pywrapfst.SymbolTable.read_text(str(model_dir / f"phones.{suffix}"))
            graphemes = pywrapfst.SymbolTable.read_text(str(model_dir / f"graphemes.{suffix}"))
            symbols = {graphemes.find(i) for i in range(graphemes.num_symbols())}
            fst = pynini.Fst.read(str(model_dir / "model.fst"))
            fst.set_input_symbols(graphemes)
            fst.set_output_symbols(phones)
            rewriters = {}
            for decomposition in {True, False}:
                rewriters[decomposition] = PhonetisaurusRewriter(
                    fst,
                    graphemes,
                    phones,
                    num_pronunciations=3,
                    grapheme_order=meta["grapheme_order"],
                    sequence_separator=meta["sequence_separator"],
                    graphemes=set(meta["graphemes"]),
                    strict=True,
                    unicode_decomposition=decomposition,
                )
            with stage_progress(f"{lang} real G2P words", total=len(selected), every=25) as update:
                for number, row in enumerate(selected, 1):
                    word = unicodedata.normalize("NFC", row["word"]).casefold()
                    effective, published_missing = input_profile(
                        word,
                        meta,
                        symbols,
                        allow_exact_nfc=config.get("exact_nfc_adapter", {}).get(lang, False),
                    )
                    bad = missing_graphemes(word, effective, symbols)
                    result: dict[str, Any] = {
                        "language": lang,
                        "word": row["word"],
                        "input_word": word,
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "missing_graphemes": bad,
                        "published_missing_graphemes": published_missing,
                        "published_unicode_decomposition": meta["unicode_decomposition"],
                        "effective_unicode_decomposition": effective["unicode_decomposition"],
                        "candidates": [],
                        "status": "unsupported_graphemes" if bad else "empty",
                        "input_policy": "strict alphabets; exact NFC adapter recorded; no proxies",
                    }
                    if not word or any(c.isspace() or c.isdigit() for c in word):
                        result["status"] = "unsupported_word_shape"
                    elif not bad:
                        try:
                            for hypothesis in rewriters[effective["unicode_decomposition"]](word):
                                ipa, score = hypothesis.pronunciation, float(hypothesis.score)
                                converted = tokenize_ipa_to_vocab(ipa, vocab)
                                if not math.isfinite(score):
                                    raise ValueError("non-finite G2P score")
                                result["candidates"].append(
                                    {
                                        "ipa": ipa,
                                        "score": score,
                                        "tokens": converted.tokens,
                                        "token_ids": converted.token_ids,
                                        "oov_units": converted.oov_units,
                                        "usable": bool(converted.token_ids)
                                        and not converted.oov_units,
                                    }
                                )
                            if result["candidates"]:
                                result["status"] = (
                                    "ok"
                                    if any(c["usable"] for c in result["candidates"])
                                    else "phone_oov"
                                )
                        except Exception as error:
                            # One bad word is diagnostic data, never a partial pronunciation.
                            result.update(status="rewrite_error", error=str(error), candidates=[])
                    counts[lang + ":" + result["status"]] += 1
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    update(number)
    write_json(
        output.with_suffix(output.suffix + ".report.json"),
        {
            "status": "completed",
            "mfa_version": version,
            "python_version": sys.version,
            "pynini_version": importlib.metadata.version("pynini"),
            "official_generator_sha256": sha(Path(generator.__file__)),
            "counts": dict(counts),
            "gpu_used": False,
            "encoder_loaded": False,
            "pronunciation_accuracy_verified": False,
        },
    )
