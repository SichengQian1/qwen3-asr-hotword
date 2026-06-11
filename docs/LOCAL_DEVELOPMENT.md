# Local Development Environment

The non-work-zone environment mirrors the work-zone Python and Qwen package
versions while using the macOS build of PyTorch.

## Official environment

```text
Environment path: .conda
Python: 3.12.12
torch: 2.10.0 (macOS build)
qwen-asr: 0.0.6
transformers: 4.57.6
accelerate: 1.12.0
av: 17.0.1
gradio: 6.17.3
librosa: 0.11.0
soundfile: 0.13.1
Model: models/Qwen3-ASR-1.7B
```

Activate it with:

```bash
conda activate "$PWD/.conda"
```

Recreate it with:

```bash
bash scripts/bootstrap_local_macos.sh
```

The bootstrap uses the macOS Accelerate BLAS backend. This prevents Conda's
OpenMP OpenBLAS runtime from being loaded alongside the OpenMP runtime bundled
with the PyTorch wheel.

The Mac environment validates configuration, model structure, checkpoint
keys, tensor shapes, data processing, retrieval, and CPU unit tests. CUDA,
FlashAttention, NCCL, distributed training, and H200 BF16 execution remain
work-zone checks.

The current development Mac has limited memory, so do not load the complete
1.7B checkpoint for CPU inference. `scripts/inspect_qwen.py` constructs the
full model on the meta device and inspects the real Safetensors index without
materializing all parameters.

## Model snapshot

The approved model revision and architecture facts are recorded in
`configs/models/qwen3-asr-1.7b.yaml`. The model directory itself is ignored by
Git.

Download the same revision with:

```bash
hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --local-dir models/Qwen3-ASR-1.7B
```

Validate it with:

```bash
python scripts/inspect_qwen.py --config configs/workzone.local.yaml
```
