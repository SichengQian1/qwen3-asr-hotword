import json
from pathlib import Path

from qwen_hotword.training.swift_json import (
    clean_swift_asr_text,
    convert_swift_json_to_tsv,
)


def test_clean_swift_asr_text_removes_qwen_prefix() -> None:
    assert (
        clean_swift_asr_text("language Portuguese<asr_text>  quando   ia sendo")
        == "quando ia sendo"
    )
    assert clean_swift_asr_text("sem prefixo") == "sem prefixo"


def test_convert_swift_json_uses_response_text(tmp_path: Path) -> None:
    source = tmp_path / "swift.json"
    source.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<audio>"},
                        {
                            "role": "assistant",
                            "content": "language Portuguese<asr_text> texto antigo",
                        },
                    ],
                    "audios": ["/data/audio/a.flac"],
                    "response": "language Portuguese<asr_text> texto certo",
                    "language": "Portuguese",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_tsv = tmp_path / "out" / "source.tsv"

    summary = convert_swift_json_to_tsv(
        source,
        output_tsv,
        expected_language="Portuguese",
    )

    assert summary.status == "pass"
    assert summary.source_records == 1
    assert summary.written_records == 1
    assert output_tsv.read_text(encoding="utf-8").splitlines() == [
        "audio\ttext",
        "/data/audio/a.flac\ttexto certo",
    ]


def test_convert_swift_json_falls_back_to_assistant_message(tmp_path: Path) -> None:
    source = tmp_path / "swift.json"
    source.write_text(
        json.dumps(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<audio>"},
                        {
                            "role": "assistant",
                            "content": "language Portuguese <asr_text> gritou gloria",
                        },
                    ],
                    "audios": ["/data/audio/b.flac"],
                    "language": "Portuguese",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_tsv = tmp_path / "source.tsv"

    convert_swift_json_to_tsv(source, output_tsv)

    assert output_tsv.read_text(encoding="utf-8").splitlines() == [
        "audio\ttext",
        "/data/audio/b.flac\tgritou gloria",
    ]
