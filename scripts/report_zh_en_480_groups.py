"""Score the completed bilingual 480h cache and package listening examples on CPU."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_zh_en_480_eleven import cached, load_result

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation.english_group_recall import (
    GROUPS,
    VIEWS,
    Label,
    Word,
    policy,
    summary,
    views,
    worst,
)
from qwen_hotword.evaluation.zh_en_eleven import PROBE, probe_gold
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows

RUN = "37b0d91c93abc9b9920336a45fa06f27deaf6455cdc498aa85102381987d52a0"
TABLE = "33e73fa9df358475d766b7cd3c5f2fa07071a5ffc25ed8e87923e2effaae5e40"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.2f}%"


def grouped_inputs() -> tuple[
    Path, dict[str, Label], list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    source = ROOT / "outputs/zh_en183_480_joint2351_eleven_v1"
    table = ROOT / "configs/zh_en_joint2351.v1.jsonl"
    rules = ROOT / "configs/zh_en_english_eval_groups.v1.tsv"
    plan, report = read(source / "run_config.json"), read(source / "report.json")
    if (digest(plan), report["plan_sha256"], plan["table_sha256"], report["status"]) != (
        RUN,
        RUN,
        TABLE,
        "completed",
    ):
        raise ValueError("expected returned 480h epoch-25 run and unchanged joint table")
    pin(table, TABLE)
    labels = policy(rules, list(rows(table)))
    cfg = read(ROOT / "configs/zh_en_stream2351_probe.workzone.json")
    with stage_progress("Join original gold to the saved delivery audio manifest"):
        with (
            zipfile.ZipFile(source / "delivery.zip") as z,
            z.open("delivery/audio_manifest.json") as f,
        ):
            manifest = json.load(f)
        index = Path(cfg["work"]) / cfg["en_index"]
        pin(index, plan["audit"][PROBE]["index_sha256"])
        probe, audit = probe_gold(
            index,
            Path(cfg["en_wav"]),
            {lab.surface for lab in labels.values() if lab.group == "navigation"},
        )
        if audit != plan["audit"][PROBE] or probe.keys() != manifest[PROBE + "@@en"].keys():
            raise ValueError("TTS index coverage differs from original evaluation")
        common = []
        for group, uid, path, ws in base.inputs(cfg):
            s = manifest[group + "@@" + group[:2]][uid]
            if s["sample_id"] != uid or s["audio_path"] != str(path):
                raise ValueError(f"changed home/aishell manifest: {uid}")
            common.append({**s, "targets": sorted({base.canonical(w) for w in ws})})
        if digest(common) != plan["common_samples_sha256"]:
            raise ValueError("original gold/manifest digest mismatch")
    items = [s for s in common if s["language"] == "en"]
    items.extend({**manifest[PROBE + "@@en"][uid], "targets": gold} for uid, gold in probe.items())
    return source, labels, items, cfg, report


def main() -> None:
    source, labels, items, cfg, report = grouped_inputs()
    rules = ROOT / "configs/zh_en_english_eval_groups.v1.tsv"
    home_meta = {r["id"]: r for r in rows(Path(cfg["home_meta"])) if r["id"].startswith("val_")}
    words: dict[str, Word] = defaultdict(Word)
    totals: dict[str, Word] = defaultdict(Word)
    negatives: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with stage_progress(
        "Read signed final-prefix rankings; no model inference", total=len(items), every=500
    ) as tick:
        for i, s in enumerate(items, 1):
            group, uid = s["group"], s["sample_id"]
            lists = views(load_result(cached(source, s), RUN, s))
            if not s["targets"] and group != PROBE:
                neg = negatives[group]
                neg["n"] += 1
                neg.update({v: int(bool(r)) for v, r in zip(VIEWS, lists, strict=True)})
            for w in s["targets"]:
                lab = labels[w]
                words[w].add(w, lab.alias, lists)
                totals[group].add(w, "", lists)
                if lab.group == "cross":
                    examples[w].append(
                        dict(
                            sample=uid,
                            group=group,
                            path=s["audio_path"],
                            sha256=s["audio_sha256"],
                            text=home_meta[uid]["text"],
                            ranked=lists,
                        )
                    )
            tick(i)
    for group in {s["group"] for s in items}:
        for v in VIEWS:
            ref = report["by_dataset"][group][v]
            if (totals[group].n, totals[group].strict[v]) != (
                ref["target_pairs"],
                ref["target_hits"],
            ):
                raise ValueError(f"strict scores differ from original: {group}/{v}")
            if group != PROBE and (negatives[group]["n"], negatives[group][v]) != (
                ref["negative_utterances"],
                ref["negative_with_candidate"],
            ):
                raise ValueError(f"negative counts differ from original: {group}/{v}")
    out = Path(tempfile.mkdtemp(prefix="zh_en480_english_groups_", dir=ROOT / "outputs"))
    lines = ["分组\t口径\t平均\t实测词数\t目标句词对\tTop5前\tTop5后\tTop7前\tTop7后"]
    for group, name in GROUPS.items():
        sample = [
            cnt
            for w, cnt in words.items()
            if labels[w].group == group or (group == "multi" and labels[w].multi)
        ]
        for metric, title in (("strict", "词面"), ("entity", "实体")):
            for macro in (False, True):
                rates = summary(sample, metric, macro)
                lines.append(
                    "\t".join(
                        [
                            name,
                            title,
                            "逐词" if macro else "逐目标",
                            str(len(sample)),
                            str(sum(w.n for w in sample)),
                            *map(pct, rates),
                        ]
                    )
                )
    comparison = []
    for label_group in ("regular", "cross"):
        bucket = [cnt for word, cnt in words.items() if labels[word].group == label_group]
        comparison.append(summary(bucket, "strict", False))
    if all(x is not None for rates in comparison for x in rates):
        gap = [
            f"{(a - b) * 100:+.2f}"
            for a, b in zip(*comparison, strict=True)
            if a is not None and b is not None
        ]
        lines.append("\t".join(["常规减跨语言", "词面差(百分点)", "逐目标", "—", "—", *gap]))
    if "hongyadong" in words:
        count = words["hongyadong"]
        for metric, title in (("strict", "词面"), ("entity", "实体")):
            lines.append(
                "\t".join(
                    [
                        "Hongyadong／洪崖洞",
                        title,
                        "逐目标",
                        "1",
                        str(count.n),
                        *map(pct, summary([count], metric, False)),
                    ]
                )
            )
    for group, neg in sorted(negatives.items()):
        if neg["n"]:
            lines.append(
                "\t".join(
                    [
                        group,
                        "误提示率",
                        "逐负句",
                        "—",
                        f"{neg['n']}负句",
                        *(pct(neg[v] / neg["n"]) for v in VIEWS),
                    ]
                )
            )
    grid = "\n".join(lines) + "\n"
    (out / "groups.tsv").write_text(grid, encoding="utf-8-sig")
    print(grid, flush=True)
    hearing = [
        "# 跨语言专名听审",
        "",
        "按末次门控后 Top7 词面召回率升序，同率先列目标数多的词，再按词面排序。"
        "每词优先选两条不同音频，先列实体也漏的样例，其次仅别名命中，最后词面命中。",
        "",
    ]
    word_lines = [
        "词面\t组\t认可的库内别名\t目标对\t"
        + "\t".join(f"{m}_{v}" for m in ("strict", "entity") for v in VIEWS)
    ]
    for w, counts in sorted(words.items()):
        lab = labels[w]
        word_lines.append(
            "\t".join(
                [
                    lab.surface,
                    GROUPS[lab.group],
                    lab.alias,
                    str(counts.n),
                    *(str(getattr(counts, m)[v]) for m in ("strict", "entity") for v in VIEWS),
                ]
            )
        )
    (out / "words.tsv").write_text("\n".join(word_lines) + "\n", encoding="utf-8-sig")
    top = worst(words, labels)
    with (
        stage_progress(
            "Package original WAVs for the ten lowest-recall cross-language names",
            total=len(top),
            every=1,
        ) as tick,
        zipfile.ZipFile(out / "listening.pending", "w", zipfile.ZIP_DEFLATED) as z,
    ):
        for rank, w in enumerate(top, 1):
            lab, counts = labels[w], words[w]
            details = (
                f"{rank}. {lab.surface}　词面 {counts.strict[VIEWS[-1]]}/{counts.n}"
                f"　实体 {counts.entity[VIEWS[-1]]}/{counts.n}"
            )
            print(details, flush=True)
            hearing.extend(["## " + details, ""])
            cases = sorted(
                examples[w],
                key=lambda r: (w in r["ranked"][-1], lab.alias in r["ranked"][-1], r["sample"]),
            )
            used = set()
            for case in cases:
                if case["sha256"] in used:
                    continue
                used.add(case["sha256"])
                p = Path(case["path"])
                pin(p, case["sha256"])
                name = f"audio/{rank:02}_{case['sample']}.wav"
                z.write(p, name)
                hearing.extend(
                    [
                        f"- [{case['sample']}]({name})　{case['text']}",
                        f"  - 原始 Top7：{'、'.join(case['ranked'][2]) or '空'}",
                        f"  - 门控 Top7：{'、'.join(case['ranked'][3]) or '空'}",
                        f"  - 工区原音频：`{p}`",
                    ]
                )
                print("  ", case["sample"], p, flush=True)
                if len(used) == 2:
                    break
            hearing.append("")
            tick(rank)
        (out / "listen.md").write_text("\n".join(hearing) + "\n", encoding="utf-8")
        for name in ("groups.tsv", "words.tsv", "listen.md"):
            z.write(out / name, name)
        z.write(rules, rules.name)
        z.writestr(
            "identity.json",
            json.dumps(
                dict(
                    plan=RUN,
                    table=TABLE,
                    policy=sha(rules),
                    scope="English final prefix, fixed candidates; explicit alias credit",
                )
            ),
        )
    with zipfile.ZipFile(out / "listening.pending") as z:
        if z.testzip() is not None:
            raise ValueError("listening archive integrity check failed")
    (out / "listening.pending").replace(out / "listening.zip")
    print("下载听审包：", out / "listening.zip", flush=True)


if __name__ == "__main__":
    main()
