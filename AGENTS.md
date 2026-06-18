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
- The known work-zone model path is:

```text
/glutsterfs_103/models/Qwen3-ASR-1.7B
```

Confirm paths in the work zone before relying on them, because shared storage
mount names have appeared with similar spellings.

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

Known project path used in the work zone:

```text
/glusterfs_103/q00933266/qwen3-asr-hotword
```

Known data-copy target, if used:

```text
/glusterfs_103/q00933266/data
```

Prefer direct mounts for original data paths when possible.

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
