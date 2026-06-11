#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ENV="qwen3-asr-hotword"
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"

if [[ "${ACTIVE_ENV}" != "${EXPECTED_ENV}" ]]; then
  echo "ERROR: activate Conda environment '${EXPECTED_ENV}' first." >&2
  echo "Current environment: '${ACTIVE_ENV:-none}'" >&2
  exit 2
fi

python -m pip install -e ".[workzone]"

if [[ ! -f configs/workzone.local.yaml ]]; then
  cp configs/workzone.example.yaml configs/workzone.local.yaml
  echo "Created configs/workzone.local.yaml"
  echo "Edit its model and persistent-storage paths before running diagnostics."
else
  echo "Keeping existing configs/workzone.local.yaml"
fi

echo "Work-zone bootstrap complete."
