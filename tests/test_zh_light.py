import json
from collections import Counter
from pathlib import Path

import pytest

from qwen_hotword.evaluation import zh_light as light
from qwen_hotword.training.zh_en_common import rows

ROOT = Path(__file__).resolve().parents[1]


def test_word_inventory() -> None:
    prior = list(rows(ROOT / "configs/zh_en_joint2351.v1.jsonl"))
    words = (ROOT / "configs/zh_light_hotwords.v1.txt").read_text().splitlines()
    new = light.additions(words, prior)
    assert (len(prior), len(new)) == (2351, 60)
    assert set(words) - set(new) == {"哪吒之魔童闹海", "灯带", "繁花"}
    assert set(light.SPOKEN) <= set(new)


def test_gold_join(tmp_path: Path) -> None:
    wav = tmp_path / "wav_16k"
    wav.mkdir()
    files = [wav / name for name in ["a.wav", "b.wav", "c.wav"]]
    for f in files:
        f.touch()
    labels = tmp_path / "meta_16k.jsonl"
    samples = [
        {"wav": "wav_16k/a.wav", "keywords": ["暂停", "暂停"]},
        {"wav": "b.wav", "text": "QQ音乐"},
        {"audio_path": str(files[2]), "targets": []},
    ]

    def save(records: list[dict[str, object]]) -> None:
        labels.write_text("\n".join(json.dumps(r) for r in records))

    save(samples)
    assert light.metadata(wav, None) == labels

    def read_gold() -> list[list[str]]:
        return [r[-1] for r in light.light_gold(wav, files, labels, ["暂停", "QQ音乐"])]

    assert read_gold() == [["暂停"], ["QQ音乐"], []]
    save(samples[:-1])
    with pytest.raises(ValueError, match="missing annotations"):
        read_gold()
    save([*samples, samples[0]])
    with pytest.raises(ValueError, match="duplicate WAV"):
        read_gold()
    save([{**samples[0], "keywords": ["缺词"]}, *samples[1:]])
    with pytest.raises(ValueError, match="missing/unknown"):
        read_gold()
    save([samples[0], {"wav": "b.wav", "text": "请暂停一下"}, samples[2]])
    with pytest.raises(ValueError, match="missing/unknown"):
        read_gold()


def test_combined_counts() -> None:
    counts = {g: {"raw": Counter(target_pairs=12, target_hits=8)} for g in light.GROUPS}
    counts[light.base.AIS]["raw"]["target_pairs"] = 1234
    assert light.aggregate(counts)["raw"] == Counter(target_pairs=72, target_hits=48)


def test_expand_preserves_bilingual_entries_and_mixed_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from qwen_hotword.hotwords.registry import HotwordEntry
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    cli = importlib.import_module("evaluate_zh_light_hotwords")
    cfg = json.loads((ROOT / "configs/zh_light_eval.workzone.json").read_text())
    words = (ROOT / cfg["word_list"]).read_text().splitlines()
    vocab = load_phoneme_vocab(ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json")

    def fake_generate(root, output, inputs, train, foundation, vocab, identity):
        return [
            HotwordEntry(
                str(i), "zh", w, w, (w,), vocab.tokens[2], (vocab.tokens[2],), (2,), "test", 0
            )
            for i, w in enumerate(inputs)
        ]

    monkeypatch.setattr(cli.ar, "pronunciations", fake_generate)
    entries, owners = cli.expand(cfg, tmp_path, words, {"frontend_packages": []}, {}, vocab)
    table = list(rows(tmp_path / "joint2411.jsonl"))
    assert len(entries) == len(owners) == 2412
    assert len(table) == 2411 and table[:2351] == list(rows(ROOT / cfg["previous_table"]))
    assert next(r for r in table if r["surface"] == "QQ音乐")["pronunciations"][0]["languages"] == [
        "zh",
        "us",
    ]
    assert all(owners[e.hotword_id].surface == e.surface for e in entries)
