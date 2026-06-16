# Evaluation Set And Hotword Table

This project evaluates two separate questions:

1. whether the CTC decoder retrieves the correct hotwords and ranks them high;
2. whether injecting retrieved hotwords into Qwen3-ASR changes the final text in
   the right direction.

The evaluation assets are generated in the work zone from local open-source
corpora. Audio files remain in their original directories and are never copied
into Git.

## Output Files

`scripts/build_eval_assets.py` writes:

- `utterances.jsonl`: normalized audio/transcript manifest;
- `hotwords.jsonl`: hotword table with normalized text and optional IPA;
- `ctc_eval_cases.jsonl`: cases for retrieval/ranking evaluation;
- `asr_injection_eval_cases.jsonl`: cases for baseline-vs-hotword ASR evaluation;
- `summary.json`: counts and build settings.

Each utterance record has:

```json
{
  "utt_id": "librispeech_test-clean_123-456-0001",
  "dataset": "librispeech",
  "split": "test-clean",
  "language": "en",
  "audio_path": "/home/z00841352/27A/data/LibriSpeech/...",
  "text": "REFERENCE TRANSCRIPT",
  "duration_sec": null,
  "metadata": {}
}
```

Each hotword record has:

```json
{
  "hotword_id": "hw_en_000001",
  "language": "en",
  "surface": "saint michael",
  "normalized": "saint michael",
  "ipa": "seɪnt ˈmaɪkəl",
  "phoneme_tokens": ["s", "e", "ɪ", "n", "t", "m", "a", "ɪ", "k", "ə", "l"],
  "phoneme_source": "espeak",
  "source_dataset": "librispeech",
  "source_utt_ids": ["librispeech_test-clean_..."],
  "hotword_type": "phrase",
  "frequency": 2
}
```

Each CTC or ASR-injection case has:

```json
{
  "case_id": "ctc_en_positive_000001",
  "eval_stage": "ctc",
  "case_type": "positive",
  "utt_id": "librispeech_test-clean_...",
  "language": "en",
  "audio_path": "/home/z00841352/27A/data/LibriSpeech/...",
  "reference_text": "REFERENCE TRANSCRIPT",
  "active_hotword_ids": ["hw_en_000001", "hw_en_000099"],
  "expected_hotword_ids": ["hw_en_000001"],
  "distractor_hotword_ids": ["hw_en_000099"]
}
```

## Work-zone Build

Copy the template once:

```bash
cp configs/eval_sources.example.yaml configs/eval_sources.local.yaml
```

Edit paths only if the H200 data directories differ from the template.

Dry run without IPA:

```bash
python scripts/build_eval_assets.py \
  --config configs/eval_sources.local.yaml \
  --output-dir outputs/eval_assets_v1 \
  --phonemizer none
```

Final run with IPA requires `phonemizer` and the `espeak-ng` binary:

```bash
python -m pip install phonemizer
espeak-ng --version
python scripts/build_eval_assets.py \
  --config configs/eval_sources.local.yaml \
  --output-dir outputs/eval_assets_v1 \
  --phonemizer espeak \
  --require-ipa
```

If `espeak-ng` is missing in the container, keep the dry-run assets and report
the error. We will either install the binary in the image or add a separate IPA
generation step.

## What To Inspect First

After the first run, check:

```bash
cat outputs/eval_assets_v1/summary.json
head -3 outputs/eval_assets_v1/utterances.jsonl
head -3 outputs/eval_assets_v1/hotwords.jsonl
head -3 outputs/eval_assets_v1/ctc_eval_cases.jsonl
```

The first acceptance gate is not model accuracy. It is whether the manifests
contain the expected languages, valid audio paths, real reference text, and
reasonable hotword candidates.

