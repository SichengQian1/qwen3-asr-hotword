# AGENTS.md

This repository is a technical exploration project for hotword-aware adaptation of
Qwen3-ASR-1.7B. It is not a generic ASR fine-tuning repo. Future agents should
preserve the project direction below unless the user explicitly changes it.

## Project Goal

The target system augments Qwen3-ASR-1.7B with multilingual, phoneme-based
hotword support that can be updated online. The intended runtime path is:

```text
audio
  -> Qwen3-ASR audio encoder
  -> lightweight CTC hotword detection branch
  -> top-k hotword candidates
  -> prompt injection into Qwen3-ASR LLM decoder
  -> hotword-enhanced transcription
```

The project should optimize for practical performance on the H200 work-zone
environment, while keeping local development reproducible on the non-work-zone
machine where possible.

## Fixed Target Model

- The target model is `Qwen3-ASR-1.7B`.
- Do not design against Qwen3-ASR 0.6B unless the user explicitly asks for a
  separate compatibility experiment.
- The verified work-zone model path is:

```text
/glusterfs_103/models/Qwen3-ASR-1.7B
```

This path was confirmed in the work zone on 2026-07-15 and contains the
complete local Qwen3-ASR-1.7B snapshot, including `config.json`,
`model.safetensors.index.json`, model shards, tokenizer files, and
`preprocessor_config.json`.

The model inspection also passed on 2026-07-15 and confirmed:

```text
Architecture: Qwen3ASRForConditionalGeneration
Audio encoder layers: 24
Audio encoder hidden dimension: 1024
Audio projected dimension: 2048
CTC tap module: thinker.audio_tower.ln_post
CTC tap dimension: 1024
Audio tower parameters: 317,477,504
Total model parameters: 2,349,217,408
Manifest validation: pass, no mismatches
```

## Architecture Direction

Prefer additions around the existing Qwen3-ASR architecture instead of invasive
rewrites. The blue-path/base model should remain recognizable:

- audio encoder / `audio_tower`
- projector
- tokenizer
- LLM decoder

The main project additions are:

- CTC branch attached near the audio encoder output;
- phoneme/IPA representation for multilingual hotwords;
- hotword registry and reloadable runtime index;
- prompt builder that injects selected candidates into Qwen3-ASR inference;
- optional LoRA or adapter training only after the lightweight path is working.

## Implementation Phases

### 1. Model Loading And Inspection

Keep scripts that verify the Qwen3-ASR-1.7B model can be loaded and inspected.
Important facts to preserve:

- architecture class and config metadata;
- audio encoder hidden dimension;
- projected dimension;
- candidate CTC tap module;
- parameter counts;
- weight index/hash summaries where useful.

This phase exists to prevent accidental development against the wrong model
shape.

### 2. CTC Hotword Branch

The CTC branch is the first training target. Its job is not to produce the final
ASR transcript. Its job is to score whether a given audio segment likely contains
configured hotwords.

Expected modules:

- CTC head over audio encoder hidden states;
- decoder for phoneme/token posterior sequences;
- top-k candidate selection;
- confidence scoring;
- false-positive controls.

Start conservatively:

```text
freeze Qwen3-ASR backbone
train only CTC head
```

Only consider encoder adapters or LoRA after this path works.

### 3. Phoneme-Based Multilingual Hotword Representation

Hotword matching should be based on pronunciation as much as possible, not only
surface text. English and Portuguese are the immediate target languages.

Expected behavior:

- normalize surface forms;
- generate or load IPA/phoneme forms;
- keep language tags such as `en`, `pt-BR`, or `pt-PT`;
- support fallback representations when IPA tooling is unavailable;
- avoid hard-coding a single language's spelling rules into shared code.

### 4. Online Hotword Updates

The final product should support updating hotwords without retraining and,
ideally, without restarting the service.

Expected components:

- hotword registry;
- config loader;
- phoneme conversion;
- searchable runtime index;
- version tracking;
- reload or refresh API.

Hotwords should be treated as runtime data, not model weights.

### 5. Prompt Injection Into Qwen3-ASR LLM

After CTC detection, selected hotwords should be injected into the Qwen3-ASR
prompt as references, not as forced transcript content.

The prompt path should:

- rank candidates by confidence;
- limit top-k size;
- avoid injecting low-confidence candidates;
- reduce hallucination risk;
- support baseline inference and hotword-enhanced inference side by side.

### 6. Training Plan

Training should proceed from least invasive to most invasive:

1. frozen Qwen3-ASR plus trainable CTC head;
2. CTC head plus small encoder adapter if needed;
3. projector or audio-side LoRA if needed;
4. LLM LoRA only after prompt injection and CTC quality justify it;
5. end-to-end tuning only as a later experiment.

Avoid full-model fine-tuning as a first step.

### 7. Streaming And Latency

Streaming optimization is important but should come after the basic offline
pipeline works. Future streaming work may include:

- fixed chunk processing;
- dynamic buffer;
- VAD;
- encoder cache reuse;
- parallel CTC and LLM inference;
- tail latency metrics.

## Evaluation Assets

Evaluation-set and hotword-table construction exist in this repository, but do
not confuse them with the main adaptation architecture. They are support tooling
for measuring:

- CTC hotword recall/ranking;
- final ASR changes after prompt injection.

When changing evaluation tooling, keep it separate from model/runtime code.

## Work-Zone Assumptions

The user operates in two environments:

- non-work-zone local machine: discussion, coding, GitHub updates, lightweight
  tests;
- H200 work zone: model/data access, heavy training, realistic inference.

The work zone may use Docker containers and mounted shared storage. Prefer
mounting large datasets directly into containers instead of copying millions of
small files between shared filesystems.

Known project paths used in the work zone:

```text
Inside container:
  /host_home/star/q00933266/qwen3-asr-hotword
Outside container:
  /home/star/q00933266/qwen3-asr-hotword
```

Confirmed work-zone diagnostics as of 2026-07-15:

```text
Python: 3.12.12
CUDA visible: true
GPUs: 8 x NVIDIA H200, about 139.8 GiB each
Torch: 2.10.0, CUDA 12.8
qwen-asr: 0.0.6
transformers: 4.57.6
accelerate: 1.12.0
datasets: 5.0.0
librosa: 0.11.0
soundfile: 0.13.1
jiwer: 4.0.0
flash-attn: not installed
vllm: not installed
```

The work-zone `doctor.py` check has `overall_status: pass`. The configured
`cache` and `runs` directories have been created. The earlier broken editable
`qwen-asr` installation was replaced, and `qwen_asr` now imports from the active
Conda environment's `site-packages` directory.

The single-H200 synthetic-audio smoke test also passed on 2026-07-15. The
complete Qwen3-ASR-1.7B checkpoint loaded successfully and transcribed a
one-second 440 Hz synthetic input, returning one result. The test was launched
with physical GPU 4 exposed as logical `cuda:0` via `CUDA_VISIBLE_DEVICES=4`.
The recurring `pynvml` deprecation message is a non-fatal telemetry dependency
warning and did not affect model loading or inference.

The runtime CTC tap probe passed on 2026-07-15 and confirmed the following real
`thinker.audio_tower.ln_post` behavior on H200:

```text
1.0 s audio: input_features [128, 100] -> ln_post [13, 1024]
2.0 s audio: input_features [128, 200] -> ln_post [26, 1024]
Device/dtype: logical cuda:0, torch.bfloat16
Length validation: pass, no errors
```

Although the wrapper received two audio samples together, the official
`get_audio_features` path invoked `audio_tower` once per sample. This matches
the upstream implementation, which loops over input audios to preserve
precision. The first training implementation should therefore retain explicit
per-sample lengths and avoid assuming that `ln_post` directly returns a padded
`[B, T, 1024]` tensor.

The padded Encoder batch probe also passed on H200 on 2026-07-15:

```text
Processor input_features: [2, 128, 200]
Processor feature lengths: [100, 200]
Padded ln_post hidden states: [2, 26, 1024]
CTC input lengths: [13, 26]
Encoder attention mask: [2, 26], valid counts [13, 26]
Backbone gradients enabled: false
Device/dtype: logical cuda:0, torch.bfloat16
Status: pass, no errors
```

This validates the first frozen-backbone training boundary: the extractor can
turn variable-length audio into padded `[B, T_max, 1024]` states while retaining
the exact per-sample lengths needed by CTC loss.

The minimal frozen-backbone CTC training smoke test passed on H200 on
2026-07-15:

```text
CTC head: Linear(1024, 90), 92,250 trainable parameters
Encoder states: [2, 26, 1024], input lengths [13, 26]
CTC logits: [2, 26, 90]
CTC log probabilities: [26, 2, 90]
Synthetic targets: English "cat" length 3; Portuguese "bom dia" length 5
Float32 CTC loss: 17.65545654296875
Head gradient norm before clipping: 22.5
Head weights changed after optimizer step: true
Frozen audio-tower parameters: 317,477,504
Audio-tower parameters with gradients: 0
Status: pass, no errors
```

The loss and gradient magnitude above come from random Head initialization,
synthetic sine-wave audio, and artificial labels, so they validate mechanics
only and must not be used as quality baselines.

Known data-copy target, if used:

```text
/glusterfs_103/q00933266/data
```

Prefer direct mounts for original data paths when possible.

Known work-zone corpus paths currently recorded in
`configs/eval_sources.workzone.yaml` and reused by the evaluation/G2P scanners:

```text
English:
  LibriSpeech:
    /host_home/z00841352/27A/data/LibriSpeech
  Common Voice English:
    /host_home/z00841352/27A/data/Common_Voice_Scripted_Speech_25.0/cv-corpus-25.0-2026-03-09/en
  FLEURS English:
    /host_home/z00841352/27A/data/Fleurs/fleurs_wav/English

Brazilian Portuguese:
  Common Voice Portuguese:
    /host_home/z00841352/27A/data/Common_Voice_Scripted_Speech_25.0/cv-corpus-25.0-2026-03-09/pt
  FLEURS Portuguese:
    /host_home/z00841352/27A/data/Fleurs/fleurs_wav/Portuguese
  MLS Portuguese:
    /host_home/z00841352/27A/data/MLS_MultiLingual_LibriSpeech/mls_portuguese
```

As of July 2026, the checked-in work-zone source config does not yet contain a
Spanish corpus path. Do not assume Spanish data is available until the user
provides or confirms the `es-419`/Spanish training path.

The first real CTC training corpus confirmed on 2026-07-15 is a 500-hour
Brazilian Portuguese conversational dataset:

```text
TSV inside container:
  /host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/500小时巴西葡萄牙语口语化语音数据.tsv
Audio root inside container:
  /host_home/z00841352/27A/data/Noah_espt/noah_pt
TSV columns:
  audio, text
Language:
  pt-BR
```

The TSV `audio` values are relative paths such as
`APY.../data/category/...wav`; resolve them as `audio_root / audio`. Paths and
texts contain non-ASCII characters, so readers must use UTF-8 and `pathlib`
rather than manual path concatenation. This corpus has now been processed with
the Brazilian Portuguese MFA G2P model described below. Earlier FLEURS G2P
reports remain separate and must not be treated as coverage evidence for this
corpus.

The first 1,000-row Noah Portuguese TSV audit passed on 2026-07-15:

```text
Rows/audio/text: 1000/1000/1000
Resolved audio files: 1000
Missing audio files: 0
Duplicate audio values: 0
Absolute audio values: 0
Text characters: min 3, mean 80.829, max 242
Status: pass, no errors
```

MFA is available in the work zone through the Conda environment `aligner`:

```bash
conda run -n aligner mfa ...
```

The downloaded Brazilian Portuguese G2P model is:

```text
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/g2p/portuguese_brazil_mfa.zip
```

The previously verified MFA invocation is:

```bash
conda run -n aligner mfa g2p \
  --num_pronunciations 1 \
  WORDS_TXT \
  G2P_MODEL_ZIP \
  OUTPUT_DICT
```

`--num_pronunciations 1` keeps the top-1 pronunciation per unique word. The
Noah corpus must first produce its own normalized unique-word list; do not reuse
the earlier FLEURS word list or its G2P coverage report.

The complete Noah Portuguese word-list and MFA G2P run finished on 2026-07-16:

```text
TSV records: 366,508
Corpus word tokens: 5,348,219
Unique input words: 76,744
MFA dictionary lines: 76,736
MFA runtime: 1,702.057 seconds
```

The eight-line difference must be audited as missing words rather than assumed
to be harmless. The generated dictionary is stored in the ignored work-zone
output directory and should not be committed:

```text
outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
```

The first real-data feasibility run is Experiment A. It intentionally uses only
128 high-confidence Portuguese samples to test whether the frozen Qwen3-ASR
audio encoder plus linear CTC Head can overfit a tiny dataset. The first pass
excludes standalone `h`, unresolved apostrophe/hyphen forms, dictionary misses,
ambiguous pronunciations, phone OOVs, invalid audio, and CTC-length-infeasible
samples. Following manual review, Experiment A also requires every word to
appear at least 100 times in the Noah corpus, rejects non-Portuguese single-letter
tokens while retaining valid forms such as `a`, `à`, `e`, `é`, `o`, `ó`, and
common vocalic variants, and caps the minimum CTC target/input ratio at 0.75.
These restrictions intentionally make the first overfit test easier and reduce
rare-name and code-switch uncertainty. Do not add normalization fallbacks to
this first manifest; connector and code-switch recovery should be evaluated
separately after the clean-label training path is proven.

The corrected Experiment A manifest was regenerated and manually reviewed on
2026-07-17. It contains 128 Brazilian Portuguese samples totaling about 7.21
minutes, with a mean duration of about 3.38 seconds. The build reported 96,563
lexically clean corpus rows and completed with `status: pass`. The reviewed
samples had consistent text, word, phone, token-ID, and CTC-length fields, with
no obvious English terms or names in the first inspection set.

The Experiment A trainer is `scripts/train_experiment_a.py`. It validates the
manifest before model loading, freezes the complete Qwen audio tower, extracts
the 128 `ln_post` feature sequences once, caches those frozen states in CPU
memory, and repeatedly trains only a float32 `Linear(1024, 90)` CTC head. It
records CTC loss and greedy-decoded training PER after every epoch. Expected
outputs under the requested run directory are:

```text
metrics.jsonl
report.json
ctc_head_best.pt
ctc_head_latest.pt
```

`status: completed` means the training program ran successfully;
`overfit_success: true` means the best training PER reached the configured
target. These states must not be conflated.

## Coding Guidelines

- Keep changes scoped and compatible with Qwen3-ASR-1.7B.
- Prefer small, inspectable modules under `src/qwen_hotword/`.
- Add tests for scanner, hotword, CTC, prompt, or training behavior when logic
  changes.
- Avoid hard-coded local absolute paths in committed code. Put paths in config
  files or command-line arguments.
- Do not commit model weights, datasets, generated manifests, caches, or large
  artifacts.
- Keep generated outputs under ignored output directories.
- Use `PYTHONPATH=src` or the project packaging setup when running scripts
  directly in the work zone.

## Current Priority, Excluding Evaluation Tooling

The next major project steps are:

1. stabilize Qwen3-ASR-1.7B load/inference wrapper;
2. expose audio encoder hidden states for a CTC tap point;
3. implement a minimal trainable CTC head;
4. implement phoneme-space decoding and hotword scoring;
5. implement prompt injection wrapper;
6. run a small H200 training smoke test;
7. then expand to LoRA/adapters and streaming optimization.

When unsure, choose the path that gets a minimal Qwen3-ASR-1.7B plus CTC
hotword branch running end to end on H200 first.
