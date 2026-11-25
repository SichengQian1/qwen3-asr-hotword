#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/start_zh_rag_demo_v2.sh --https --port 8444 --public-host 10.44.168.92 --max-cases 0 "$@"
