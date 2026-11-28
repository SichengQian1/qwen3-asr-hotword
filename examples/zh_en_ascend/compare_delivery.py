"""Paired gated Top5/Top7 parity for the same utterances on H200 and Ascend."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts  # noqa: E402

Identity = dict[tuple[str, str], tuple[str, int, int, int]]


def read_run(p: Path, backend: str) -> tuple[dict[str, Any], dict[str, Any], Identity]:
    receipt = json.loads(Path(str(p) + ".run.json").read_text())
    with p.open("rb") as f:
        h = hashlib.file_digest(f, "sha256").hexdigest()
    if receipt["backend"] != backend or receipt["delivery_sha256"] != h:
        raise ValueError(f"{backend} receipt/zip mismatch")
    with zipfile.ZipFile(p) as z:
        ctx, manifest, durations = [
            json.loads(z.read("delivery/" + s + ".json"))
            for s in ("stream_contexts", "audio_manifest", "durations")
        ]
    validate_contexts(ctx, durations)
    identity = {
        (key, uid): (r["audio_sha256"], r["frames"], r["sample_rate"], r["step_count"])
        for key, records in manifest.items()
        for uid, r in records.items()
    }
    if identity.keys() != {(key, uid) for key, byid in durations.items() for uid in byid}:
        raise ValueError("manifest and timelines differ")
    return receipt, ctx, identity


def compare(left: Path, right: Path) -> dict[str, Any]:
    h, x, ids = read_run(left, "cuda")
    n, y, other = read_run(right, "npu")
    for key in ("kit_sha256", "table_sha256", "dtype", "attention"):
        if h[key] != n[key]:
            raise ValueError(f"unpaired {key}: {h[key]} / {n[key]}")
    if not ids or ids != other:
        raise ValueError("audio bytes, ID, group, frames or step count differ")
    result: dict[str, Any] = {}
    for mode in MODES:
        total = ordered = sets = empty = 0
        shared = 0
        max_diff = 0.0
        pairs = []
        diffs = []
        for key, uid in sorted(ids):
            for i, (left_items, right_items) in enumerate(
                zip(x[mode][key][uid], y[mode][key][uid], strict=True)
            ):
                a = {v["word"]: v for v in left_items}
                b = {v["word"]: v for v in right_items}
                total += 1
                ordered += list(a) == list(b)
                sets += a.keys() == b.keys()
                empty += not a and not b
                for w in a.keys() & b.keys():
                    shared += 1
                    max_diff = max(max_diff, abs(a[w]["score"] - b[w]["score"]))
                row = dict(
                    group=key,
                    id=uid,
                    step=i,
                    end_seconds=min(2 * (i + 1), ids[key, uid][1] / ids[key, uid][2]),
                    h200=left_items,
                    ascend=right_items,
                )
                pairs.append(row)
                if list(a) != list(b) or any(
                    a[w]["phoneme"] != b[w]["phoneme"] for w in a.keys() & b.keys()
                ):
                    diffs.append(dict(group=key, id=uid, step=i))
        result[mode] = dict(
            steps=total,
            ordered_word_matches=ordered,
            same_word_sets=sets,
            both_empty=empty,
            shared_word_comparisons=shared,
            max_abs_score_delta_common_word=max_diff if shared else None,
            differences=diffs,
            paired_steps=pairs,
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h200", type=Path)
    p.add_argument("ascend", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = compare(a.h200, a.ascend)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    for m, v in r.items():
        print(m, {k: x for k, x in v.items() if k not in ("differences", "paired_steps")})
    print("完整差异:", a.out)


if __name__ == "__main__":
    main()
