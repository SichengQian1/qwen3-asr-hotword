"""Read the completed 480h run; replay at most four final prefixes per home split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_zh_en_480_eleven import cached, load_result

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_en_english_audit as audit
from qwen_hotword.evaluation.aishell_run import check_gpu, identity
from qwen_hotword.evaluation.zh_en_eleven import PROBE
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.stream_context_contract import audio_prefixes, candidates
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows

HOME = ("en-smarthome-val_v1", "en-smarthome-val_v2")
HEAD = "955e31483b54040c5fce9da6e9287f604545087cdddf15db434a2ce7557991d1"


def scan(
    source: Path,
    out: Path,
    plan: dict[str, Any],
    report: dict[str, Any],
    cfg: dict[str, Any],
    table: list[dict[str, Any]],
) -> None:
    run_id = report["plan_sha256"]
    with zipfile.ZipFile(source / "delivery.zip") as z, z.open("delivery/audio_manifest.json") as f:
        manifest = json.load(f)
    near = audit.neighbors(table, PROBE)
    declared = {
        base.canonical(r["surface"]): {tuple(p["phoneme_tokens"]) for p in r["pronunciations"]}
        for r in table
    }
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = defaultdict(list)
    rival_pairs: Counter[tuple[str, str]] = Counter()
    per_word: dict[str, Counter[str]] = defaultdict(Counter)
    splits: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for_group = manifest[PROBE + "@@en"]
    index = Path(cfg["work"]) / cfg["en_index"]
    pin(index, plan["audit"][PROBE]["index_sha256"])
    with stage_progress(
        "Audit saved English TTS Top7 and source labels", total=len(for_group), every=500
    ) as tick:
        for row in rows(index):
            uid = hashlib.md5(row["audio_path"].encode()).hexdigest()[:16]
            if uid not in for_group or uid in seen:
                continue
            seen.add(uid)
            sample = for_group[uid]
            saved = load_result(cached(source, sample), run_id, sample)
            ranked = [base.canonical(w) for w in saved["raw_last"]]
            if sample["sample_id"] != uid or sample["group"] != PROBE:
                raise ValueError(f"incorrect probe cache join: {uid}")
            ents = {base.canonical(e["surface"]): e for e in row["entities"]}
            gold = near.keys() & ents.keys()
            for word in sorted(gold):
                counts: Counter[str] = Counter()
                rivals = audit.count_probe(word, gold, ranked, near[word], counts)
                rivals_label = "with_neighbor" if near[word] else "without_neighbor"
                stats[rivals_label].update(counts)
                per_word[word].update(counts)
                rival_pairs.update((word, other) for other in rivals)
                if rivals and len(examples["non_gold_neighbor"]) < 15:
                    examples["non_gold_neighbor"].append(
                        dict(
                            id=uid,
                            word=word,
                            gold=sorted(gold),
                            ranked=ranked,
                            non_gold_neighbors=rivals,
                        )
                    )
                ent = ents[word]
                labels["category"][str(ent.get("category") or "unspecified")] += 1
                text = row.get("text") or ""
                ipa = unicodedata.normalize("NFC", ent.get("ipa") or "").split()
                same = tuple(ipa) in declared[word]
                state = "missing" if not ipa else ("same_tokens" if same else "different_tokens")
                labels["index_ipa_vs_registry"][state] += 1
                present = audit.text_contains(text, word)
                labels["literal_surface_in_tts_text"][str(present)] += 1
                key = "text_form_review" if not present else "ipa_form_review"
                if (not present or not same) and len(examples[key]) < 15:
                    examples[key].append(
                        dict(
                            id=uid,
                            word=word,
                            text=text,
                            index_ipa=ipa,
                            registry=sorted(declared[word]),
                            ipa_comparison=state,
                        )
                    )
            tick(len(seen))
    if seen != for_group.keys():
        raise ValueError("index and delivery English probe identities differ")
    pool = []
    # Reconstruct the original common inventory digest from gold sources + saved metadata.
    common = []
    for group, uid, path, words in base.inputs(cfg):
        s = manifest[group + "@@" + group[:2]][uid]
        if str(path) != s["audio_path"] or uid != s["sample_id"]:
            raise ValueError(f"changed audio mapping for {uid}")
        gold = {base.canonical(w) for w in words}
        common.append({**s, "targets": sorted(gold)})
        if group not in HOME:
            continue
        saved = load_result(cached(source, s), run_id, s)
        ranked = [base.canonical(w) for w in saved["raw_last"]]
        gated = {base.canonical(m["word"]) for m in saved["steps"][-1]}
        splits[group].update(
            pairs=len(gold), hits=len(gold.intersection(ranked)), gated_hits=len(gold & gated)
        )
        for word in gold.intersection(ranked) - gated:
            pool.append(
                dict(group=group, sample_id=uid, target=word, rank=ranked.index(word) + 1, sample=s)
            )
    if digest(common) != plan["common_samples_sha256"]:
        raise ValueError("source sentence targets differ from original inventory")
    meta = {r["id"]: r for r in rows(Path(cfg["home_meta"])) if r["id"].startswith("val_")}
    for group in HOME:
        for uid in manifest[group + "@@en"]:
            r = meta[uid]
            for w in set(map(base.canonical, r["keywords"])):
                present = audit.text_contains(r.get("text") or "", w)
                labels[group + "_surface_in_spoken_text"][str(present)] += 1
                written = audit.text_contains(r.get("label") or "", w)
                labels[group + "_spoken_written_presence"][
                    f"spoken={present} written={written}"
                ] += 1
                if not present and len(examples[group]) < 10:
                    examples[group].append(
                        dict(id=uid, word=w, text=r.get("text"), label=r.get("label"))
                    )
    picked = audit.choose_cases(pool)
    for case in picked:
        r = meta[case["sample_id"]]
        case.update(text=r.get("text"), label=r.get("label"))
    total: Counter[str] = Counter()
    for counts in stats.values():
        total.update(counts)
    for group, counts in [(PROBE, total), *splits.items()]:
        ref = report["by_dataset"][group]
        if (counts["pairs"], counts["hits"]) != (
            ref["raw_top7_last"]["target_pairs"],
            ref["raw_top7_last"]["target_hits"],
        ):
            raise ValueError(f"saved hits/denominator differ: {group}")
        if group in HOME and counts["gated_hits"] != ref["gated_top7_last"]["target_hits"]:
            raise ValueError(f"saved gated hits differ: {group}")
    write_json(
        out / "cpu.json",
        dict(
            plan=run_id,
            probe_pairs=total,
            observed_probe_words=len(per_word),
            thresholds=base.GATE,
            by_neighbor_group={
                key: dict(
                    **cnt,
                    recall=cnt["hits"] / cnt["pairs"],
                    covered_words=sum(bool(near[w]) == (key == "with_neighbor") for w in per_word),
                )
                for key, cnt in stats.items()
            },
            by_word=per_word,
            top_rivals=[(w, n, c) for (w, n), c in rival_pairs.most_common(25)],
            source_checks=labels,
            examples=examples,
            home_pool=dict(Counter(c["group"] for c in pool)),
            selected=picked,
            definition="Whole registered pronunciation edit distance / longer length <= 0.20. "
            "A non-gold neighbor in raw Top7 is co-occurrence; target rank below 7 was not saved. "
            "Text form and source IPA comparisons are not listening-based verdicts.",
        ),
    )


def diagnose(gpu: int) -> None:
    cfg = read(ROOT / "configs/zh_en_stream2351_probe.workzone.json")
    source = ROOT / "outputs/zh_en183_480_joint2351_eleven_v1"
    plan, report = read(source / "run_config.json"), read(source / "report.json")
    if (
        report["status"],
        report["plan_sha256"],
        report["checkpoint_sha256"],
        plan["model_identity"]["checkpoint_sha256"],
        plan["gate"],
    ) != ("completed", digest(plan), HEAD, HEAD, base.GATE):
        raise ValueError("requires completed original epoch 25 joint run and unchanged gate")
    for name, checksum in plan["additional_code"].items():
        pin(ROOT / "src/qwen_hotword" / (name + ".py"), checksum)
    t = ROOT / "configs/zh_en_joint2351.v1.jsonl"
    pin(t, plan["table_sha256"])
    run_key = digest(
        [
            report["plan_sha256"],
            sha(Path(__file__)),
            sha(Path(audit.__file__)),
            sha(Path(cfg["home_meta"])),
            sha(Path(cfg["work"]) / cfg["en_index"]),
        ]
    )
    out = ROOT / "outputs" / ("zh_en480_english_diagnosis_" + run_key[:12])
    with exclusive_feature_cache_run(out):
        with stage_progress("Read original delivery and signed caches"):
            if not (out / "cpu.json").is_file():
                scan(source, out, plan, report, cfg, list(rows(t)))
            cpu = read(out / "cpu.json")
            print("Saved en-TTS raw Top7:", cpu["probe_pairs"], flush=True)
        selected = cpu["selected"]
        replayed = []
        detector = None
        with stage_progress(
            "Replay selected home utterances (last prefix only)", total=len(selected), every=1
        ) as tick:
            for i, case in enumerate(selected, 1):
                s = case["sample"]
                path = out / (case["sample_id"] + ".json")
                if not path.exists():
                    if detector is None:
                        original = read(ROOT / "configs/aishell_original_retrieval.workzone.json")
                        model, _, _, vocab = identity(
                            ROOT, original, read(ROOT / original["review_policy"])
                        )
                        if model != plan["model_identity"]:
                            raise ValueError("replay model/code identity differs")
                        entries, owners = base.registry(t, vocab)
                        check_gpu(gpu, 10)
                        detector = external._load_detector(
                            model=Path(model["model_path"]),
                            checkpoint=Path(model["checkpoint_path"]),
                            vocab=vocab,
                            hotwords=entries,
                            device="cuda:0",
                            dtype="bfloat16",
                            language="English",
                        )
                    saved = load_result(cached(source, s), report["plan_sha256"], s)
                    pin(Path(s["audio_path"]), s["audio_sha256"])
                    for prefix in audio_prefixes(s["audio_path"], s):
                        if prefix["source_end_frame"] == s["frames"]:
                            break
                    else:
                        raise ValueError(f"missing final audio prefix: {path}")
                    result = external._retrieve_waveform(
                        detector, prefix["waveform"], vocab, saved_raw_rank_depth=64
                    )
                    ranked = collapse_variants(result, entries, owners, base.GATE)[
                        "raw_ranked_matches"
                    ]
                    gated = candidates(_select(ranked, gate=base.GATE, top_k=7))
                    if [m["surface"] for m in ranked[:7]] != saved["raw_last"] or gated != saved[
                        "steps"
                    ][-1]:
                        raise ValueError(
                            f"replay differs from saved last prefix: {case['sample_id']}"
                        )
                    match = ranked[case["rank"] - 1]
                    failures = audit.failed_gates(match)
                    if base.canonical(match["surface"]) != case["target"] or not failures:
                        raise ValueError("replayed target rejection not explained")
                    tokens: Any = result["decoded_tokens"]
                    write_json(
                        path,
                        dict(
                            **case,
                            match=match,
                            failed=failures,
                            ctc_match=tokens[match["decoded_start"] : match["decoded_end"]],
                            saved_top7=saved["raw_last"],
                            saved_final=gated,
                        ),
                    )
                replayed.append(read(path))
                print(case["sample_id"], case["target"], replayed[-1]["failed"], flush=True)
                tick(i)
        write_json(
            out / "home.json",
            dict(
                gate=base.GATE,
                cases=replayed,
                failed_counts=Counter(f for r in replayed for f in r["failed"]),
            ),
        )
        tmp = out / "bundle.pending"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("cpu.json", "home.json"):
                z.write(out / name, name)
        tmp.replace(out / "return_bundle.zip")
        print("Return:", out / "return_bundle.zip", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    diagnose(args.gpu)
