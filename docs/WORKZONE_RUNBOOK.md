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

## Build evaluation assets

The hotword evaluation set can be built before model training because it only
uses local open-source corpus paths and transcripts.

```bash
cp configs/eval_sources.example.yaml configs/eval_sources.local.yaml
```

Edit `configs/eval_sources.local.yaml` if any H200 data path differs. Then run
the dry build first:

```bash
python scripts/build_eval_assets.py \
  --config configs/eval_sources.local.yaml \
  --output-dir outputs/eval_assets_v1 \
  --phonemizer none
```

Inspect:

```bash
cat outputs/eval_assets_v1/summary.json
head -3 outputs/eval_assets_v1/utterances.jsonl
head -3 outputs/eval_assets_v1/hotwords.jsonl
head -3 outputs/eval_assets_v1/ctc_eval_cases.jsonl
```

For final IPA generation, install the optional Python wrapper and confirm the
container has the `espeak-ng` binary:

```bash
python -m pip install -e ".[eval]"
espeak-ng --version
python scripts/build_eval_assets.py \
  --config configs/eval_sources.local.yaml \
  --output-dir outputs/eval_assets_v1 \
  --phonemizer espeak \
  --require-ipa
```

## Returning results to the development zone

Return the generated JSON reports and any approved error-analysis artifacts.
Detailed paths and sample-level outputs may be enabled when they are useful for
debugging, while credentials and access tokens must remain machine-local.
