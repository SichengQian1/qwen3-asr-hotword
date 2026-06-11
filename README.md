# Qwen3-ASR 1.7B Hotword Biasing

This repository implements a phoneme-based English and Brazilian Portuguese
hotword-biasing extension for `Qwen/Qwen3-ASR-1.7B`.

The repository is split between:

- a non-secure development zone for code, tests, and documentation;
- a secure H200 work zone for model weights, business data, training, and
  sample-level evaluation.

Large audio corpora, model weights, checkpoints, and machine-local
configuration stay outside Git. Small synthetic or approved diagnostic
artifacts may be committed when they improve reproducibility.

## v0.1 Work-zone checks

Install the repository inside the existing `qwen3-asr-hotword` Conda
environment:

```bash
conda activate qwen3-asr-hotword
pip install -e ".[workzone]"
cp configs/workzone.example.yaml configs/workzone.local.yaml
```

Edit only `configs/workzone.local.yaml`, which is ignored by Git. Then run:

```bash
python scripts/doctor.py --config configs/workzone.local.yaml
python scripts/inspect_qwen.py --config configs/workzone.local.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_qwen.py \
  --config configs/workzone.local.yaml
```

The first two commands do not load model weights. The smoke test loads the
local 1.7B model and runs a synthetic-audio transcription, so it should only be
run after a GPU has been allocated.

## Local development

See `docs/LOCAL_DEVELOPMENT.md` for the pinned Python 3.12 environment and
official 1.7B snapshot.

```bash
python -m pytest
ruff check .
mypy src
```
