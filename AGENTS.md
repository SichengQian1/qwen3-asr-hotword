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
