import csv
import json
from pathlib import Path

from qwen_hotword.training.spanish_mfa_diagnostics import diagnose_spanish_mfa_audit


def _write_tsv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)


def test_diagnose_spanish_mfa_audit_classifies_missing_words_and_oov(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "mfa_audit_v1"
    audit_dir.mkdir()
    dictionary = tmp_path / "spanish.dict"
    dictionary.write_text(
        "mas m a s\nanos a n o s\nnormal n o ɹ m a l\n", encoding="utf-8"
    )
    (audit_dir / "summary.json").write_text(
        json.dumps({"dictionary_path": str(dictionary)}), encoding="utf-8"
    )
    _write_tsv(
        audit_dir / "missing_words.tsv",
        ["word", "corpus_count"],
        [["más", 10], ["años", 4], ["todavía", 2]],
    )
    _write_tsv(
        audit_dir / "oov_phone_counts.tsv",
        ["oov_unit", "dictionary_count", "corpus_weighted_count"],
        [["\u0303", 3, 12]],
    )
    _write_tsv(
        audit_dir / "words_with_oov_phones.tsv",
        ["word", "corpus_count", "pronunciation", "oov_units"],
        [["banco", 5, "b ã n k o", "\u0303"]],
    )
    _write_tsv(
        audit_dir / "extra_dictionary_words.tsv",
        ["word", "corpus_count"],
        [["extra", 0]],
    )

    report = diagnose_spanish_mfa_audit(audit_dir, max_items=5)

    assert report["acute_recoverable_unique_words"] == 1
    assert report["acute_recoverable_corpus_tokens"] == 10
    assert report["all_marks_recoverable_unique_words"] == 1
    assert report["all_marks_recoverable_corpus_tokens"] == 4
    assert report["unrecovered_unique_words"] == 1
    assert report["unrecovered_corpus_tokens"] == 2
    assert report["missing_categories"] == {"acute_accent": 2, "enye": 1}
    assert report["oov_units"] == [
        {
            "unit": "\u0303",
            "unicode": [{"codepoint": "U+0303", "name": "COMBINING TILDE"}],
            "dictionary_count": 3,
            "corpus_weighted_count": 12,
        }
    ]
    assert report["top_extra_dictionary_words"] == ["extra"]
    assert (audit_dir / "spanish_diagnostics.json").is_file()
