from __future__ import annotations

import csv
import io
import json
import tarfile
import unicodedata
from pathlib import Path

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.zh_en_foundation import (
    inspect_chinese,
    inspect_file,
    inspect_json,
    inspect_tar,
    verify_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def test_approved_183_inventory_covers_all_source_symbols() -> None:
    path = ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json"
    spec = json.loads(path.read_text())
    vocab = load_phoneme_vocab(path)
    assert len(vocab.tokens) == len(set(vocab.tokens)) == 183
    assert vocab.tokens[:2] == ("<blank>", "<unk>")
    assert vocab.tokens[-6:] == ("fʷ", "vʷ", "əw", "ɛː", "ɜ", "ɜː")
    with (path.parent / spec["mapping_file"]).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        if row["disposition"] == "fst_epsilon":
            assert row["token_id"] == ""
            continue
        for form in ("NFC", "NFD"):
            result = tokenize_ipa_to_vocab(unicodedata.normalize(form, row["source_unit"]), vocab)
            assert result.token_ids == [int(row["token_id"])]
    assert tokenize_ipa_to_vocab("a not_a_phone", vocab).token_ids == []
    for left, right in [("i", "iː"), ("k", "kʰ"), ("a˥", "a˧˥"), ("ɹ", "ɻ")]:
        assert vocab.tokens.index(left) != vocab.tokens.index(right)


def test_json_probe_is_bounded_and_keeps_source_accent(tmp_path: Path) -> None:
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"header probe does not decode audio")
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"audio": "x.wav", "accent": "UK", "text": "hello"}] * 20))
    result = inspect_json(source, rows=2)
    assert result["sampled_rows"] == 2
    assert result["total_rows"] is None
    assert result["examples"][0]["accent"] == "UK"
    assert result["audio_paths"][0]["exists"]
    assert result["accent_inferred_from_filename"] is False


def make_tar(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for name, data in [
            ("001.wav", b"not decoded"),
            ("001.txt", "导航到北京".encode()),
            ("test/002.txt", b"heldout content"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("link.txt")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)


def test_tar_probe_pairs_members_without_extracting_or_following_links(tmp_path: Path) -> None:
    archive = tmp_path / "shard.tar"
    make_tar(archive)
    result = inspect_tar(archive)
    assert result["webdataset_compatible_pairs_observed"]
    assert result["same_stem_audio_text_examples"] == {"001": [".wav", ".txt"]}
    assert result["members"][1]["text_preview"] == "导航到北京"
    assert "text_preview" not in result["members"][2]
    assert "text_preview" not in result["members"][3]
    assert not (tmp_path / "001.txt").exists()
    assert inspect_tar(archive, limit=1)["members_inspected"] == 1


def test_chinese_layout_probe_retains_unknown_formats_and_skips_known_test(tmp_path: Path) -> None:
    make_tar(tmp_path / "shard.tar")
    (tmp_path / "data.arrow").write_bytes(b"ARROW1")
    (tmp_path / "test").mkdir()
    (tmp_path / "test/secret.txt").write_text("not sampled")
    result = inspect_chinese(tmp_path)
    assert result["file_types_observed"] == {".arrow": 1, "tar": 1}
    assert len(result["examples"]) == 2
    assert not result["split_provenance_verified"]


def test_unmounted_or_malformed_inputs_are_reported(tmp_path: Path) -> None:
    assert "error" in inspect_chinese(tmp_path / "absent")
    assert "error" in inspect_file(tmp_path / "absent.json")
    source = tmp_path / "bad.json"
    source.write_text('{"text":\n "hello"}')
    result = inspect_file(source)
    assert "schema_parse_error" in result
    assert "text_preview" in result


def test_asset_verifier_rejects_changed_vocabulary_before_use() -> None:
    config = json.loads((ROOT / "configs/zh_en_foundation.workzone.json").read_text())
    config["vocab_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="vocabulary SHA256 mismatch"):
        verify_assets(ROOT, config)
