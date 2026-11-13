#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.arabic_per_diagnostics import decoded_summary
from qwen_hotword.training.ctc_diagnostics import (
    diagnose_ctc_checkpoint,
)
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import (
    completed_run,
    load_cache,
    manifest_rows,
    protect_output,
    protected_run_directories,
    recheck,
    rows,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose frozen-encoder CTC checkpoints on the validation cache. "
            "The sealed test split is intentionally not accepted."
        )
    )
    parser.add_argument("--validation-cache")
    parser.add_argument("--validation-manifest")
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--grouping", action="append", default=[])
    parser.add_argument("--per-sample-output", type=Path)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--group-column",
        help="Optional validation manifest field used for one-pass grouped PER metrics.",
    )
    parser.add_argument(
        "--expected-groups",
        default="",
        help="Optional comma-separated exact group set, for example en,es,pt.",
    )
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    args = parser.parse_args()
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be nonnegative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.source_run and (
        args.validation_cache
        or args.validation_manifest
        or args.checkpoint
        or args.skip_cache_sha256_verification
    ):
        parser.error("--source-run derives verified caches, manifests and best checkpoint")
    if not args.source_run and not all(
        (args.validation_cache, args.validation_manifest, args.checkpoint)
    ):
        parser.error("supply --source-run or cache, manifest and checkpoint")

    try:
        output = Path(args.output).expanduser()
        if args.per_sample_output and output.resolve() == args.per_sample_output.resolve():
            raise ValueError("summary and per-sample output must be different files")
        for path in (output, args.per_sample_output):
            if path and path.exists():
                raise FileExistsError(f"diagnostic output exists: {path}")
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics were requested but CUDA is unavailable")
        run = completed_run(args.source_run) if args.source_run else None
        if run:
            for destination in (output, args.per_sample_output):
                if destination:
                    protect_output(destination, protected_run_directories(run))
            args.vocab = run["plan"]["vocabularies"]["ar"]["path"]
            args.checkpoint = [run["head"]["best_checkpoint_path"]]
            cache = load_cache(run, "validation", output.parent)
            metadata = manifest_rows(run, "validation", ("en", "ar"))
            args.group_column = args.group_column or "language"
        else:
            cache = load_disk_feature_cache(
                args.validation_cache,
                expected_split="validation",
                source_manifest_path=args.validation_manifest,
                vocab_path=args.vocab,
                verify_sha256=not args.skip_cache_sha256_verification,
            )
            metadata = list(rows(Path(args.validation_manifest)))
        vocab = load_phoneme_vocab(args.vocab)
        by_id = {r["id"]: r for r in metadata}
        if any(r.get("split") != "validation" for r in metadata):
            raise ValueError("diagnostics accept validation rows only")
        if len(by_id) != len(metadata) or set(by_id) != {
            v for s in cache.shards for v in s.sample_ids
        }:
            raise ValueError("validation metadata/cache IDs disagree")
        expected_groups = tuple(
            value.strip() for value in args.expected_groups.split(",") if value.strip()
        )
        if expected_groups and not args.group_column:
            raise ValueError("--expected-groups requires --group-column")
        sample_groups = (
            {k: r[args.group_column] for k, r in by_id.items()} if args.group_column else None
        )
        if (
            expected_groups
            and sample_groups
            and set(sample_groups.values()) != set(expected_groups)
        ):
            raise ValueError("validation groups differ from expected set")
        groupings = {}
        for specification in args.grouping:
            name, sep, field = specification.partition("=")
            if not sep or not name or not field or name in groupings:
                raise ValueError("grouping must be unique NAME=MANIFEST_FIELD")
            groupings[name] = {k: str(r[field]) for k, r in by_id.items()}
        reports = []
        all_samples = []
        for checkpoint in args.checkpoint:
            checkpoint_sha = sha(Path(checkpoint))
            report: dict[str, Any] = diagnose_ctc_checkpoint(
                checkpoint,
                cache,
                vocab,
                device=args.device,
                batch_size=args.batch_size,
                sample_groups=sample_groups,
                sample_groupings=groupings,
                include_sample_metrics=bool(args.per_sample_output or groupings),
            )
            pin(Path(checkpoint), checkpoint_sha)
            report["checkpoint_sha256"] = checkpoint_sha
            if run:
                if report["checkpoint_epoch_metrics"]["epoch"] != run["head"]["best_epoch"]:
                    raise ValueError("best Head epoch disagrees with source report")
                if args.group_column == "language":
                    for group, expected in run["head"]["best_validation_by_group"].items():
                        actual = report["validation_by_group"][group]
                        if (
                            actual["errors"],
                            actual["reference_tokens"],
                            actual["sample_count"],
                        ) != (
                            expected["phoneme_errors"],
                            expected["reference_phonemes"],
                            expected["sample_count"],
                        ):
                            raise ValueError(
                                f"validation decode does not reproduce source {group} counts"
                            )
            samples = report.pop("sample_metrics", [])
            for row in samples:
                row.update(
                    {k: v for k, v in by_id[row["sample_id"]].items() if k != "phoneme_token_ids"}
                )
                row["checkpoint_sha256"] = checkpoint_sha
            if samples:
                with stage_progress("Summarize exact/folded PER and error distributions"):
                    report["sample_breakdown"] = decoded_summary(samples)
                all_samples.extend(samples)
            reports.append(report)
        if run:
            recheck(run)
        result = {
            "purpose": "frozen_encoder_ctc_validation_diagnostics",
            "validation_samples": cache.sample_count,
            "group_column": args.group_column,
            "expected_groups": list(expected_groups),
            "test_set_used": False,
            "checkpoints": reports,
            "status": "pass",
            "source_identity": run["identity"] if run else None,
            "fold_note": (
                "Arabic-only symmetric folds are alternative metrics, not causal contributions; "
                "exact PER retained."
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.per_sample_output:
            args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
            with args.per_sample_output.open("x", encoding="utf-8") as stream:
                for row in all_samples:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            result["per_sample_sha256"] = sha(args.per_sample_output)
        write_json(output, result)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"CTC DIAGNOSTICS FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
