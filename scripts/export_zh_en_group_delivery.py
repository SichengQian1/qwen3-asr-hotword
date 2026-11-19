"""Package five English subsets of the original gated, same-step 480h delivery."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from report_zh_en_480_groups import RUN, TABLE, grouped_inputs
from run_zh_en_480_eleven import cached, compact, load_result

from qwen_hotword.evaluation.english_group_recall import GROUPS, Label
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.mfa_v03_assets import sha
from qwen_hotword.training.stage_progress import stage_progress

EXPECTED = {
    "regular": (13, 59),
    "cross": (73, 355),
    "review": (21, 98),
    "navigation": (525, 50306),
    "multi": (284, 27665),
}


def split_groups(
    samples: list[dict[str, Any]], labels: dict[str, Label]
) -> dict[str, dict[str, list[tuple[dict[str, Any], list[str]]]]]:
    grouped: dict[str, dict[str, list[tuple[dict[str, Any], list[str]]]]] = {
        group: defaultdict(list) for group in GROUPS
    }
    seen = set()
    for s in samples:
        key, uid = s["key"], s["sample_id"]
        if (key, uid) in seen or s["language"] != "en" or key != s["group"] + "@@en":
            raise ValueError(f"invalid or repeated English sample: {key}/{uid}")
        seen.add((key, uid))
        selected: dict[str, list[str]] = defaultdict(list)
        for w in sorted(set(s["targets"])):
            lab = labels[w]
            selected[lab.group].append(w)
            if lab.multi:
                selected["multi"].append(w)
        for group, ws in selected.items():
            grouped[group][key].append((s, ws))
    return grouped


def export_one(
    group: str,
    groups: dict[str, list[tuple[dict[str, Any], list[str]]]],
    labels: dict[str, Label],
    source: Path,
    work: Path,
) -> dict[str, Any]:
    folder = work / group
    folder.mkdir()
    utterances = sum(len(rows) for rows in groups.values())
    counts: Counter[str] = Counter()
    boundaries = 0
    with ExitStack() as stack:
        targets = stack.enter_context((folder / "evaluation_targets.jsonl").open("wb"))
        streams = {
            mode: stack.enter_context((folder / (mode + ".json")).open("wb")) for mode in MODES
        }
        for stream in streams.values():
            stream.write(b"{")
        durations: dict[str, dict[str, float]] = {}
        manifest: dict[str, dict[str, Any]] = {}
        with stage_progress("Export " + GROUPS[group], total=utterances, every=500) as tick:
            n = 0
            for j, (key, records) in enumerate(groups.items()):
                durations[key], manifest[key] = {}, {}
                for stream in streams.values():
                    stream.write((b"," if j else b"") + compact(key) + b":{")
                for i, (s, words) in enumerate(records):
                    uid = s["sample_id"]
                    steps = load_result(cached(source, s), RUN, s)["steps"]
                    duration = s["duration_seconds"]
                    timelines = {
                        m: {key: {uid: [step[:k] for step in steps]}} for m, k in MODES.items()
                    }
                    validate_contexts(timelines, {key: {uid: duration}})
                    if len(steps) != s["step_count"]:
                        raise ValueError(f"cached boundaries differ from manifest: {key}/{uid}")
                    for mode, stream in streams.items():
                        stream.write(
                            (b"," if i else b"")
                            + compact(uid)
                            + b":"
                            + compact(timelines[mode][key][uid])
                        )
                    durations[key][uid] = duration
                    manifest[key][uid] = {k: v for k, v in s.items() if k != "targets"}
                    counts.update(words)
                    targets.write(
                        compact(
                            dict(
                                key=key,
                                sample_id=uid,
                                targets=[
                                    {
                                        "surface": labels[w].surface,
                                        "accepted_aliases": [labels[w].alias]
                                        if labels[w].alias
                                        else [],
                                    }
                                    for w in words
                                ],
                            )
                        )
                        + b"\n"
                    )
                    boundaries += len(steps)
                    n += 1
                    tick(n)
                for stream in streams.values():
                    stream.write(b"}")
        for stream in streams.values():
            stream.write(b"}")
    path = folder / "delivery.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("delivery/durations.json", compact(durations))
        z.writestr("delivery/audio_manifest.json", compact(manifest))
        with z.open("delivery/stream_contexts.json", "w", force_zip64=True) as dst:
            dst.write(b"{")
            for i, mode in enumerate(MODES):
                dst.write((b"," if i else b"") + compact(mode) + b":")
                with (folder / (mode + ".json")).open("rb") as src:
                    shutil.copyfileobj(src, dst)
            dst.write(b"}")
    with zipfile.ZipFile(path) as z:
        if z.testzip():
            raise ValueError(f"invalid delivery archive: {path}")
    return dict(
        name=GROUPS[group],
        utterances=utterances,
        target_words=len(counts),
        target_pairs=counts.total(),
        prefixes=boundaries,
        delivery_sha256=sha(path),
    )


README = """# 英文分组 Context Learning 交付

每组目录中的 delivery.zip 可以独立解包使用。delivery/stream_contexts.json 提供
RAG_top5_stream、RAG_top7_stream 两个模式，每 2 秒一个累计音频边界，same-step 生效。
durations.json 为秒数；audio_manifest.json 沿用工区音频路径、原始数据集 key 和 utterance ID。
保留每步完整门控候选、顺序、分数、发音及空列表。所有候选来自同一 2351 词库。

regular 常规英文；cross 跨语言专名；review 边界词及其他品牌；navigation 导航实体；
multi 数字／缩写及列明多读法。按句内标注目标入组。多目标句可进多个包，每包内每句一次。
multi 是重叠标签。无目标负句未分入五包。汇总音频数时按原 key 和 ID 去重。

每组另附 evaluation_targets.jsonl，仅供离线评分；请勿送入提示词。按 key 和 sample_id
关联，只计算本组 targets。词面计分使用 surface；实体计分额外认可 accepted_aliases。
五包共用原始推理，允许跨组复用同一模式、key、ID 的下游结果。
"""


def main() -> None:
    source, labels, samples, _, _ = grouped_inputs()
    grouped = split_groups(samples, labels)
    out = Path(tempfile.mkdtemp(prefix="zh_en480_group_delivery_", dir=ROOT / "outputs"))
    summary: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(dir=out) as temp:
        work = Path(temp)
        with zipfile.ZipFile(out / "grouped_delivery.pending", "w") as bundle:
            for group, records in grouped.items():
                data = export_one(group, records, labels, source, work)
                if (data["target_words"], data["target_pairs"]) != EXPECTED[group]:
                    raise ValueError(
                        f"{group} gold coverage differs from completed group report: {data}"
                    )
                summary[group] = data
                for filename in ("delivery.zip", "evaluation_targets.jsonl"):
                    bundle.write(
                        work / group / filename,
                        group + "/" + filename,
                        compress_type=zipfile.ZIP_STORED
                        if filename.endswith(".zip")
                        else zipfile.ZIP_DEFLATED,
                    )
                print(GROUPS[group], data, flush=True)
                shutil.rmtree(work / group)
            bundle.writestr("README.md", README.encode())
            bundle.writestr(
                "summary.json",
                json.dumps(
                    dict(run=RUN, table_sha256=TABLE, groups=summary), ensure_ascii=False, indent=2
                ).encode(),
            )
    with zipfile.ZipFile(out / "grouped_delivery.pending") as bundle:
        if bundle.testzip():
            raise ValueError("invalid grouped delivery")
    final = out / "grouped_delivery.zip"
    (out / "grouped_delivery.pending").replace(final)
    print("请交给下游：", final, flush=True)


if __name__ == "__main__":
    main()
