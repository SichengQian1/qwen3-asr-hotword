"""Official rewrite algorithm on exact input tokens, separate from MFA CLI policy."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mmservice_smoke import _save, _sha
from qwen_hotword.training.stage_progress import stage_progress


def rewrite_exact_words(model_dir: Path, words_file: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing exact-input worker output: {output}")
    if importlib.metadata.version("montreal-forced-aligner") != "3.4.0":
        raise ValueError("exact-input probe is reviewed against MFA 3.4.0")
    generator = importlib.import_module("montreal_forced_aligner.g2p.generator")
    pynini = importlib.import_module("pynini")
    pywrapfst = importlib.import_module("pywrapfst")
    metadata = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    inputs = {str(p): _sha(p) for p in model_dir.iterdir() if p.is_file()}
    source_file = Path(str(generator.__file__))
    inputs[str(source_file)] = _sha(source_file)
    fst = pynini.Fst.read(str(model_dir / "model.fst"))
    phones = pywrapfst.SymbolTable.read_text(str(model_dir / "phones.sym"))
    fst.set_output_symbols(phones)
    embedded = fst.input_symbols()
    common = {"num_pronunciations": 1, "graphemes": None, "unicode_decomposition": False}
    symbol_differences = []
    if metadata["architecture"] == "phonetisaurus":
        if embedded is None or metadata.get("grapheme_order") != 2:
            raise ValueError("reviewed C++ bigram input contract differs from official model")
        for line in (model_dir / "graphemes.sym").read_text(encoding="utf-8").splitlines():
            symbol, number = line.rsplit(maxsplit=1)
            if embedded.find(symbol) != int(number):
                symbol_differences.append(symbol)
        external = pywrapfst.SymbolTable.read_text(str(model_dir / "graphemes.sym"))
        tables_equal: bool | None = (
            not symbol_differences and embedded.num_symbols() == external.num_symbols()
        )
        rewriter = generator.PhonetisaurusRewriter(
            fst,
            embedded,
            phones,
            **common,
            grapheme_order=2,
            sequence_separator=metadata.get("sequence_separator", "|"),
        )
    elif metadata["architecture"] == "pynini":
        if embedded is not None:
            raise ValueError("reviewed C++ Unicode input contract differs from official model")
        tables_equal = None
        rewriter = generator.Rewriter(fst, "utf8", phones, **common)
    else:
        raise ValueError("unknown official G2P architecture")
    words = words_file.read_text(encoding="utf-8").splitlines()
    output.mkdir()
    successes = 0
    with (
        (output / "words.jsonl").open("x", encoding="utf-8") as records,
        (output / "raw.dict").open("x", encoding="utf-8") as dictionary,
        stage_progress(
            "Official same-input words (no character filtering/decomposition)",
            total=len(words),
            every=100,
        ) as update,
    ):
        for index, word in enumerate(words, 1):
            row: dict[str, Any] = {"word": word, "input": word}
            try:
                hypotheses = rewriter(word)
                if len(hypotheses) > 1:
                    raise ValueError("official Top1 returned multiple pronunciations")
                ipa = hypotheses[0].pronunciation if hypotheses else ""
                row.update(ok=bool(ipa), ipa=ipa, status="ok" if ipa else "empty")
            except generator.rewrite.Error as error:
                row.update(ok=False, ipa="", status="error", error=str(error))
            if row["ok"]:
                successes += 1
                dictionary.write(f"{word}\t{row['ipa']}\n")
            records.write(json.dumps(row, ensure_ascii=False) + "\n")
            update(index)
    for name, digest in inputs.items():
        if _sha(Path(name)) != digest:
            raise ValueError(f"worker input changed: {name}")
    report = {
        "status": "completed",
        "words": len(words),
        "ok": successes,
        "architecture": metadata["architecture"],
        "inputs": inputs,
        "same_embedded_input_symbols_as_cpp": True,
        "embedded_external_symbols_equal": tables_equal,
        "embedded_external_symbol_differences": symbol_differences,
        "policy": "exact submitted tokens; no MFA clean_up_word or grapheme filtering; no NFKD",
        "output_policy": "official Top1 decoding; common IPA normalization is compared separately",
    }
    _save(output / "report.json", report)
    return report
