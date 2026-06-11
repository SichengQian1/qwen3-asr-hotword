# Work-zone Runbook

## Repository boundary

The Git repository contains code, configuration templates, documentation,
synthetic test data, and approved experiment summaries.

Do not commit:

- large audio corpora;
- model weights, feature caches, or checkpoints;
- credentials, tokens, proxy settings, or local configuration.

## First installation

Inside the existing training container:

```bash
conda activate qwen3-asr-hotword
git clone <private-repository-url>
cd <repository-directory>
bash scripts/bootstrap_workzone.sh
```

Edit `configs/workzone.local.yaml`. Keep the file inside the work zone; it is
ignored by Git.

## Diagnostics without GPU allocation

```bash
python scripts/doctor.py \
  --config configs/workzone.local.yaml \
  --output reports/environment.json

python scripts/inspect_qwen.py \
  --config configs/workzone.local.yaml
```

`doctor.py` reports a CUDA failure when the container was launched without GPU
access. That is expected only for CPU-only diagnostic sessions.

## Single-GPU smoke test

Run this only after a GPU has been assigned:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_qwen.py \
  --config configs/workzone.local.yaml
```

The test uses generated audio and does not require business data. It verifies
that the local `Qwen3-ASR-1.7B` weights can be loaded and used for inference.

## Returning results to the development zone

Return the generated JSON reports and any approved error-analysis artifacts.
Detailed paths and sample-level outputs may be enabled when they are useful for
debugging, while credentials and access tokens must remain machine-local.
