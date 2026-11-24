"""Export the seven Chinese datasets from the completed 2,411-word retrieval run, CPU only."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_zh_light_hotwords import inventory
from run_zh_en_480_eleven import cached, load_result

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_en_eleven as sets
from qwen_hotword.evaluation import zh_light as light
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read


def export(
    out: Path, groups: dict[str, list[dict[str, Any]]], run_id: str, metrics: dict[str, Any]
) -> Path:
    if set(groups) != set(light.GROUPS) or not all(groups.values()):
        raise ValueError("需要七个非空中文数据集")
    contexts: dict[str, Any] = {m: {} for m in MODES}
    durations: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    total = sum(map(len, groups.values()))
    n = 0
    with stage_progress("校验缓存并恢复每个2秒前缀的真实提示词", total=total, every=250) as tick:
        for group in light.GROUPS:
            key = group + "@@zh"
            manifest[key], durations[key] = {}, {}
            for mode in MODES:
                contexts[mode][key] = {}
            totals: dict[str, dict[str, Counter[str]]] = {}
            for sample in groups[group]:
                uid = sample["sample_id"]
                if (sample["group"], sample["language"], sample["key"]) != (
                    group,
                    "zh",
                    key,
                ) or uid in manifest[key]:
                    raise ValueError(f"样本身份冲突: {key}/{uid}")
                result = load_result(cached(out, sample), run_id, sample)
                steps = result["steps"]
                if not isinstance(steps, list) or not all(
                    isinstance(s, list) and len(s) <= 7 for s in steps
                ):
                    raise ValueError(f"非法门控候选: {key}/{uid}")
                durations[key][uid] = sample["duration_seconds"]
                manifest[key][uid] = {k: v for k, v in sample.items() if k != "targets"}
                for mode, k in MODES.items():
                    contexts[mode][key][uid] = [s[:k] for s in steps]
                base.score(result, sample, totals)
                n += 1
                tick(n)
            if {m: base.rates(c) for m, c in totals[group].items()} != metrics[group]:
                raise ValueError(f"缓存与已完成评测报告不一致: {group}")
    validation = validate_contexts(contexts, durations)
    print("流式格式验证：", validation, flush=True)
    result_path = out / "delivery.zip"
    with (
        stage_progress("打包并校验下游交付文件"),
        tempfile.TemporaryDirectory(dir=out, prefix=".delivery-") as staging,
    ):
        tmp = Path(staging) / "delivery.zip"
        with zipfile.ZipFile(tmp, "w") as archive:
            for name, value in [
                ("durations", durations),
                ("audio_manifest", manifest),
                ("stream_contexts", contexts),
            ]:
                member = zipfile.ZipInfo(f"delivery/{name}.json")  # stable timestamp for re-export
                member.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(
                    member,
                    json.dumps(
                        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode(),
                )
        with zipfile.ZipFile(tmp) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP 内容校验失败")
        # Never replace an unrelated delivery, including in concurrent exports.
        try:
            os.link(tmp, result_path)
        except FileExistsError:
            pin(result_path, sha(tmp))
    return result_path


def main() -> None:
    cfg = read(ROOT / "configs/zh_light_eval.workzone.json")
    out = ROOT / cfg["output"]
    if not (out / "report.json").is_file():
        raise SystemExit(f"缺少七组评测完成报告：{out}/report.json。请先完成本轮评测。")
    report, plan = read(out / "report.json"), read(out / "run_config.json")
    checkpoint = read(ROOT / cfg["checkpoint_config"])
    model = plan["model"]
    run_id = digest(plan)
    if (
        report["status"],
        report["run_sha256"],
        report["model"],
        report["active_surfaces"],
        report["table_sha256"],
        report["light_label_sha256"],
    ) != ("completed", run_id, model, 2411, plan["table_sha256"], plan["label_sha256"]) or (
        model["checkpoint_sha256"],
        model["checkpoint_epoch"],
        model["vocab_sha256"],
    ) != (checkpoint["required_checkpoint_sha256"], 25, checkpoint["vocab_sha256"]):
        raise ValueError("报告、模型、词表或评测身份不符；无法交付")
    pin(out / "joint2411.jsonl", plan["table_sha256"])
    labels = Path(report["light_labels"])
    pin(labels, plan["label_sha256"])
    old = ROOT / cfg["previous_table"]
    pin(old, cfg["previous_table_sha256"])
    ner = ROOT / cfg["cached_aishell_ner"]
    if not ner.is_file():
        ner = out / "aishell_ner_transcript.test.txt"
    pin(ner, sets.NER_SHA)
    wavs = Path(cfg["audio_root"])
    files = sorted(p for p in wavs.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav")
    extra = light.light_gold(
        wavs, files, labels, (ROOT / cfg["word_list"]).read_text(encoding="utf-8").splitlines()
    )
    groups = inventory(read(ROOT / cfg["sources_config"]), old, ner, extra)
    if digest(groups) != plan["groups"]:
        raise ValueError("音频或标注清单已变化，不能导出为本轮结果")
    print("请发给下游：", export(out, groups, run_id, report["by_dataset"]), flush=True)


if __name__ == "__main__":
    main()
