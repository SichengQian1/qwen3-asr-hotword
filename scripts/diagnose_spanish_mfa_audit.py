#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_mfa_diagnostics import diagnose_spanish_mfa_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose systematic Spanish MFA missing words and invisible phone OOV "
            "units from one or more dictionary-audit directories."
        )
    )
    parser.add_argument(
        "--audit-dir",
        action="append",
        required=True,
        help="MFA audit directory; repeat to diagnose multiple corpora.",
    )
    parser.add_argument("--max-items", type=int, default=30)
    args = parser.parse_args()

    reports = []
    try:
        for audit_dir in args.audit_dir:
            reports.append(
                diagnose_spanish_mfa_audit(audit_dir, max_items=args.max_items)
            )
    except (FileNotFoundError, KeyError, OSError, UnicodeError, ValueError) as error:
        print(f"SPANISH MFA DIAGNOSTIC FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
