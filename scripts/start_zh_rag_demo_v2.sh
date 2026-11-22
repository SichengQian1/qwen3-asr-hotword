#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PLUGINS= VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src" DEMO_DECODER_PY="${DEMO_DECODER_PY:-$PWD/.venv/qwen_asr_vllm023_cu129/bin/python}"
if [[ ! -x "$DEMO_DECODER_PY" ]]; then echo '缺少已经验证的 vLLM 0.23 解释器，可用 DEMO_DECODER_PY 指向已有环境。' >&2; exit 1; fi
exec python -u -B -m qwen_hotword.demo_v2.server "$@"
