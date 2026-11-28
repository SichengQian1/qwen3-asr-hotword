import importlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha
from qwen_hotword.training.zh_en_common import digest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "examples/zh_en_ascend")]
import compare_delivery as comp  # noqa: E402
import package_zh_en_ascend as build  # noqa: E402


def test_package_and_extracted_tower(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if importlib.util.find_spec("qwen_asr") is None:
        pytest.skip("requires the qwen-asr application environment")
    # Real approved vocab and original input surfaces, synthetic cached weights/pronunciations.
    for sub in ("src", "examples", "configs"):
        (tmp_path / sub).symlink_to(ROOT / sub)
    cfg = json.loads((ROOT / "configs/zh_light_eval.workzone.json").read_text())
    ck = json.loads((ROOT / cfg["checkpoint_config"]).read_text())
    out = tmp_path / cfg["output"]
    out.mkdir(parents=True)

    def put(p: Path, v: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v))

    head = tmp_path / ck["training_run"] / "head/ctc_head_best.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"test weight")
    table = [json.loads(line) for line in (ROOT / cfg["previous_table"]).read_text().splitlines()]
    old = {r["surface"].casefold() for r in table}
    for w in (ROOT / cfg["word_list"]).read_text().splitlines():
        if w.casefold() not in old:
            table.append(dict(surface=w, pronunciations=table[0]["pronunciations"][:1]))
    (out / "joint2411.jsonl").write_text("\n".join(json.dumps(r) for r in table))
    real_read = build.read

    def fixture_read(path: Path) -> dict[str, Any]:
        v = real_read(path)
        if path.name == Path(cfg["checkpoint_config"]).name:
            v["required_checkpoint_sha256"] = sha(head)
        return v

    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "read", fixture_read)
    monkeypatch.setattr(build.subprocess, "check_output", lambda *a, **kw: "fixture")
    monkeypatch.setattr("qwen_hotword.training.zh_en_foundation.verify_assets", lambda *a: None)
    model = dict(
        checkpoint_epoch=25,
        checkpoint_sha256=sha(head),
        vocab_sha256=ck["vocab_sha256"],
        model_identity={},
        code_sha256={},
    )
    tablehash = sha(out / "joint2411.jsonl")
    plan = dict(
        model=model,
        table_sha256=tablehash,
        code_sha256=["", "", sha(ROOT / "src/qwen_hotword/evaluation/zh_en_delivery.py")],
    )
    put(out / "run_config.json", plan)
    put(
        out / "report.json",
        dict(status="completed", run_sha256=digest(plan), model=model, table_sha256=tablehash),
    )

    def fake_g2p(
        root: Path, output: Path, spoken: list[str], train: Any, f: Any, vocab: Any, identity: str
    ) -> list[Any]:
        assert "一点五倍速" in spoken and "下一曲" in spoken and "启动 agent oh ess" in spoken
        return [
            SimpleNamespace(surface=s, phoneme_tokens=[vocab.tokens[2]], token_ids=[2])
            for s in spoken
        ]

    monkeypatch.setattr(build.ar, "pronunciations", fake_g2p)
    build.main()
    (archive,) = out.parent.glob("zh_en_ascend2696_*/zh_en_ascend2696.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path / "dst")
    kit = tmp_path / "dst/zh_en_ascend"
    audited = json.loads((kit / "assets/word_audit.json").read_text())
    expanded = [
        json.loads(line) for line in (kit / "assets/joint2696.jsonl").read_text().splitlines()
    ]
    assert expanded[:2411] == table
    assert (len(audited["repeated"]), len(audited["reused"]), len(audited["added"])) == (11, 8, 285)
    assert {"清醒点！姐妹", "兜兜转转，原配才是心安归宿", "有点刺眼，再暗一些", "下1曲"} <= set(
        audited["added"]
    )
    for n, h in json.loads((kit / "manifest.json").read_text())["files"].items():
        assert sha(kit / n) == h
    monkeypatch.syspath_prepend(str(kit))
    rt = importlib.import_module("run_ascend")
    import torch
    from kit_qwen.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from kit_qwen.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    from safetensors.torch import save_file

    # Exercise the *packaged* eager loader against a tiny real upstream tower.
    # Include an unrelated decoder tensor and check only encoder keys load.
    c = Qwen3ASRAudioEncoderConfig(
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        downsample_hidden_size=4,
        num_mel_bins=16,
        output_dim=32,
    )
    c._attn_implementation = "eager"
    ref = Qwen3ASRAudioEncoder(c).eval()
    state = {"thinker.audio_tower." + k: v for k, v in ref.state_dict().items()}
    state["decoder.dummy"] = torch.zeros(1)
    snapshot = tmp_path / "snapshot"
    put(snapshot / "config.json", dict(thinker_config=dict(audio_config=c.to_dict())))
    save_file(state, str(snapshot / "m.safetensors"))
    put(
        snapshot / "model.safetensors.index.json",
        dict(weight_map=dict.fromkeys(state, "m.safetensors")),
    )
    clone = rt.load_tower(snapshot, "cpu", torch.float32)
    data = torch.randn(16, 48)
    lens = torch.tensor([48])
    with torch.no_grad():
        torch.testing.assert_close(
            ref(data, feature_lens=lens).last_hidden_state,
            clone(data, feature_lens=lens).last_hidden_state,
        )


def test_paired_rank_and_identity(tmp_path: Path) -> None:
    def make(name: str, words: list[str], audio: str) -> Path:
        p = tmp_path / (name + ".zip")
        items = [dict(word=w, score=0.95 - i / 100, phoneme="a") for i, w in enumerate(words)]
        ctx = {m: {"g@@zh": {"u": [[], items[:k]]}} for m, k in comp.MODES.items()}
        with zipfile.ZipFile(p, "w") as z:
            for n, v in [
                ("stream_contexts", ctx),
                ("durations", {"g@@zh": {"u": 3.0}}),
                (
                    "audio_manifest",
                    {
                        "g@@zh": {
                            "u": dict(
                                audio_sha256=audio, frames=48000, sample_rate=16000, step_count=2
                            )
                        }
                    },
                ),
            ]:
                z.writestr(f"delivery/{n}.json", json.dumps(v))
        Path(str(p) + ".run.json").write_text(
            json.dumps(
                dict(
                    backend=name,
                    delivery_sha256=sha(p),
                    dtype="bfloat16",
                    attention="eager",
                    table_sha256="t",
                    kit_sha256="k",
                )
            )
        )
        return p

    h = make("cuda", list("abcdefg"), "a")
    n = make("npu", list("abcdegf"), "a")
    r = comp.compare(h, n)
    assert r["RAG_top5_stream"]["ordered_word_matches"] == 2
    v = r["RAG_top7_stream"]
    assert (v["ordered_word_matches"], v["same_word_sets"], v["both_empty"]) == (1, 2, 1)
    assert v["differences"][0]["step"] == 1
    assert v["paired_steps"][1]["end_seconds"] == 3
    assert v["paired_steps"][1]["h200"][-1]["word"] == "g"
    make("npu", list("abcdegf"), "other")
    with pytest.raises(ValueError, match="audio bytes"):
        comp.compare(h, n)
