#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: this bootstrap script targets Apple Silicon macOS." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${REPO_ROOT}/.conda"
PYTHON="${ENV_PREFIX}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  conda create -p "${ENV_PREFIX}" -c conda-forge \
    "python=3.12.12" \
    pip \
    av \
    ffmpeg \
    pkg-config \
    "libblas=*=*accelerate" \
    -y
else
  conda install -p "${ENV_PREFIX}" -c conda-forge \
    "python=3.12.12" \
    av \
    ffmpeg \
    pkg-config \
    "libblas=*=*accelerate" \
    -y
fi

"${PYTHON}" -m pip install --no-deps \
  "qwen-asr==0.0.6" \
  "qwen-omni-utils==0.0.9"

"${PYTHON}" -m pip install \
  -c "${REPO_ROOT}/requirements/constraints-macos.txt" \
  "torch==2.10.0" \
  "transformers==4.57.6" \
  "accelerate==1.12.0" \
  "librosa==0.11.0" \
  "soundfile==0.13.1" \
  "sox>=1.5" \
  "nagisa==0.2.11" \
  "soynlp==0.0.493" \
  "pillow>=10" \
  "pytz" \
  "flask" \
  "gradio==6.17.3"

"${PYTHON}" -m pip install -e "${REPO_ROOT}[dev,workzone]"
"${PYTHON}" -m pip check

echo "Local environment ready: ${ENV_PREFIX}"
echo "Activate with: conda activate \"${ENV_PREFIX}\""
