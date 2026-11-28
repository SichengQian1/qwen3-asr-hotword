import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation import zh_en_delivery as historical
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import sha
from qwen_hotword.training.zh_en_common import digest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples/zh_en_rag_kit"), str(ROOT / "scripts")]
import package_zh_en_rag_kit as pack  # noqa: E402
import run_delivery as runtime  # noqa: E402


def test_portable_registry_preserves_historical_ids_order_and_aliases() -> None:
    vocab = load_phoneme_vocab(ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json")
    table = ROOT / "configs/zh_en_joint2351.v1.jsonl"
    assert runtime.registry(table, vocab) == historical.registry(table, vocab)


def test_build_extract_check_and_reject_wrong_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write(p: Path, v: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v))

    for name in ["src", "examples"]:
        (tmp_path / name).symlink_to(ROOT / name)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/phonemes").symlink_to(ROOT / "configs/phonemes")
    monkeypatch.setattr(pack, "ROOT", tmp_path)
    monkeypatch.setattr(pack.subprocess, "check_output", lambda *a, **k: "fixture")
    monkeypatch.setattr(pack.metadata, "version", lambda n: "1.0")
    out = tmp_path / "outputs/run"
    out.mkdir(parents=True)
    ck = tmp_path / "train/head/ctc_head_best.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"fixture")
    vocab = ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json"
    words = (ROOT / "configs/zh_en_joint2351.v1.jsonl").read_text().splitlines()
    word = json.loads(words[0])
    for i in range(60):
        words.append(
            json.dumps(dict(surface=f"fixture{i}", pronunciations=word["pronunciations"][:1]))
        )
    table = out / "joint2411.jsonl"
    table.write_text("\n".join(words))
    snapshot = tmp_path / "base"
    write(snapshot / "config.json", {})
    cfg = dict(output="outputs/run", checkpoint_config="configs/head.json")
    write(tmp_path / "configs/zh_light_eval.workzone.json", cfg)
    write(
        tmp_path / "configs/head.json",
        dict(training_run="train", required_checkpoint_sha256=sha(ck), vocab_sha256=sha(vocab)),
    )
    model = dict(
        checkpoint_epoch=25,
        checkpoint_sha256=sha(ck),
        vocab_sha256=sha(vocab),
        model_identity={"config.json": sha(snapshot / "config.json")},
        versions={},
        code_sha256={},
    )
    plan = dict(
        model=model,
        table_sha256=sha(table),
        code_sha256=["", "", sha(ROOT / "src/qwen_hotword/evaluation/zh_en_delivery.py")],
    )
    write(out / "run_config.json", plan)
    write(
        out / "report.json",
        dict(
            status="completed",
            run_sha256=digest(plan),
            model=model,
            table_sha256=sha(table),
            active_surfaces=2411,
            pronunciations=2412,
        ),
    )
    pack.main()
    (path,) = out.parent.glob("zh_en183_rag_kit_*/zh_en_rag_kit.zip")
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp_path / "destination")
    dest = tmp_path / "destination/zh_en_rag"
    cmd = [sys.executable, "-I", str(dest / "run_delivery.py"), "--model", str(snapshot), "--check"]
    a = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert a.returncode == 0, a.stderr
    write(snapshot / "config.json", {"other": 1})
    assert subprocess.run(cmd, cwd=tmp_path, capture_output=True).returncode != 0


def test_downstream_delivery_keeps_both_modes_empty_steps_and_language_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = [dict(word=str(i), phoneme="a", score=1 - i / 10) for i in range(7)]
    monkeypatch.setattr(runtime, "retrieve", lambda *args: {"steps": [[], items]})
    vocab = load_phoneme_vocab(ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json")
    samples = [
        dict(sample_id="shared", key="set@@" + lang, language=lang, duration_seconds=2.2)
        for lang in ("zh", "en")
    ]
    output = tmp_path / "demo.zip"
    runtime.deliver(samples, None, vocab, (), {}, output)
    with zipfile.ZipFile(output) as z:
        ctx = json.loads(z.read("delivery/stream_contexts.json"))
        for lang in ("zh", "en"):
            for mode, k in runtime.MODES.items():
                assert ctx[mode]["set@@" + lang]["shared"] == [[], items[:k]]
