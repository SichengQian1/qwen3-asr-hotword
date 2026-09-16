# Ascend 310P 外部热词检索调用链与性能核查

更新日期：2026-09-16

## 1. 当前迁移命令的实际任务

当前在 Ascend 310P 上运行的命令为：

```bash
python3 -u scripts/run_external_keyword_retrieval.py \
  --model /home/f00955675/Qwen3-ASR-310P/models/Qwen3-ASR-1.7B \
  --ctc-checkpoint /home/f00955675/asr-rag/models/ctc/ctc_head_best.pt \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --keyword-bias pt_keyword_bias_phoneme.json \
  --keyword-set hard_k266 \
  --source "mls_portuguese=..." \
  --source "delivery_20260706_ptbr=..." \
  --output-dir /home/f00955675/h200_to_310p/results/pt_external_keyword_retrieval_mls_delivery_310p_fp16_audio_eager_v2 \
  --device npu:0 \
  --dtype float16 \
  --resume
```

这条命令运行的是：

```text
完整音频
  -> Qwen3-ASR audio processor
  -> Qwen3-ASR audio_tower
  -> thinker.audio_tower.ln_post 隐状态
  -> temporal-2x CTC Head
  -> CTC greedy decode
  -> Anchor Top-64 shortlist
  -> 音素编辑距离 rerank
  -> 0.75 / posterior 0.5 / Top-5 门控
  -> 每条音频的热词召回结果
```

它不运行 Qwen 文本解码器，不运行 vLLM，也不生成最终 ASR 转写。这是一条完整音频离线 CTC + Anchor 热词检索链路。

两个 `--source` 共用同一个 `pt_keyword_bias_phoneme.json` 中的 `hard_k266` 热词集合。`source` 名称只用于数据读取、结果分组和统计，不会自动为 Delivery 切换另一套热词表。

如果迁移的是此前相同的两个测试集，样本总量应为：

```text
MLS Portuguese: 871
Delivery PT-BR: 2071
合计: 2942
```

## 2. 完整 Python 调用链

### 2.1 命令行入口

```text
scripts/run_external_keyword_retrieval.py
```

职责：

1. 解析模型、CTC checkpoint、音素词表和热词文件路径；
2. 解析一个或多个 `--source NAME=AUDIO_DIR,TRANSCRIPTS`；
3. 调用 `run_external_keyword_retrieval()`；
4. 最终将汇总报告打印到标准输出。

### 2.2 外部数据集和推理调度

```text
src/qwen_hotword/inference/external_keyword_retrieval.py
```

主要流程：

1. 加载 90 类音素词表；
2. 读取 `pt_keyword_bias_phoneme.json` 的 `hard_k266`；
3. 将每个热词的 MFA 音素转换成当前 CTC 词表 token ID；
4. 递归扫描来源目录中的 WAV/FLAC；
5. 读取 transcripts，并通过文件名 stem 关联音频和文本；
6. 检查缺失音频、缺失文本和跨来源重复 ID；
7. 加载模型、CTC Head 和 Anchor Index；
8. 对音频逐条串行执行检索；
9. 每完成一条音频写入一个可恢复的 sample shard；
10. 全部完成后聚合最终结果和评测指标。

### 2.3 模型和 Detector 加载

```text
src/qwen_hotword/inference/streaming_backends.py
  -> load_cumulative_ctc_detector()
```

加载行为：

1. 使用 `torch.load(..., map_location="cpu", weights_only=True)` 读取 CTC checkpoint；
2. 检查 checkpoint 内的 `vocab_tokens` 与请求的 90 类词表完全一致；
3. 根据 checkpoint 构建 `TemporalUpsampleCtcHead`；
4. 验证 Head 是 temporal-2x 结构；
5. 严格加载 Head state dict；
6. 将 CTC Head 移到 `npu:0`，但 dtype 固定为 `torch.float32`；
7. 调用 `load_asr_model()` 加载完整 Qwen3-ASR-1.7B wrapper；
8. 冻结 Qwen `audio_tower`；
9. 为 266 个热词构建 Anchor Index。

Qwen 模型加载位于：

```text
src/qwen_hotword/modeling/qwen_backbone.py
  -> load_asr_model()
```

实际调用为：

```python
Qwen3ASRModel.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="npu:0",
    local_files_only=True,
)
```

虽然后续只调用 `model.thinker.audio_tower`，但当前代码仍通过完整的 `Qwen3ASRModel` wrapper 加载模型。

### 2.4 单条音频处理

单条音频在 `external_keyword_retrieval.py` 中依次执行：

```text
_load_audio()
  -> soundfile 读取 float32 音频
  -> 多声道取均值变为单声道
  -> 必要时由 librosa 重采样到 16 kHz

_retrieve_waveform()
  -> build_audio_prompt(..., "Portuguese")
  -> Qwen processor(text=[prompt], audio=[waveform])
  -> input_features 移到模型设备和模型 dtype
  -> feature_attention_mask 移到模型设备
  -> extract_padded_ln_post()
  -> Encoder hidden state 转为 npu:0 / float32
  -> temporal-2x CTC Head
  -> decode_ctc_posterior()
  -> Anchor query
  -> shortlist rerank 和门控
```

### 2.5 Audio Encoder 特征提取

```text
src/qwen_hotword/modeling/audio_encoder.py
  -> extract_padded_ln_post()
```

这个函数：

1. 在 `audio_tower.ln_post` 上注册 forward hook；
2. 调用 Qwen `audio_tower`；
3. 捕获形状为 `[T, 1024]` 的编码器输出；
4. 记录准确的 CTC input length；
5. 将变长 hidden states pad 成批量张量。

该函数即使收到批量输入，也会在 Python 中逐样本调用一次 `audio_tower`。当前外部检索入口本身每次只传入一条音频，因此有效 batch size 为 1。

### 2.6 Temporal-2x CTC Head

```text
src/qwen_hotword/modeling/ctc_head.py
  -> TemporalUpsampleCtcHead
```

结构为：

```text
[T, 1024] encoder hidden states
  -> LayerNorm
  -> 时间轴 repeat_interleave x2
  -> 1x1 Conv1d: 1024 -> 512
  -> depthwise Conv1d, kernel 5
  -> 1x1 context Conv1d
  -> GELU
  -> Dropout（推理时关闭）
  -> Linear: 512 -> 90
  -> [2T, 90] CTC logits
```

需要注意：即使命令使用 `--dtype float16`，当前 loader 仍然将 CTC Head 固定为 FP32，并将 Encoder 输出转换到 FP32 后再运行 Head。

### 2.7 CTC 解码和 Anchor 检索

CTC 解码位于：

```text
src/qwen_hotword/hotwords/scoring.py
  -> decode_ctc_posterior()
```

流程为：

```text
logits
  -> float32 softmax
  -> argmax token ID 和 posterior confidence
  -> token/confidence 从设备复制到 CPU list
  -> 合并连续重复 token
  -> 删除 CTC blank
```

Anchor 检索涉及：

```text
src/qwen_hotword/hotwords/anchor_index.py
src/qwen_hotword/hotwords/exact_automaton.py
src/qwen_hotword/hotwords/scoring.py
src/qwen_hotword/training/edit_distance.py
```

Anchor Index 使用 2/3/4-gram 稀有位置音素 Anchor，从 266 个词中获得最多 64 个候选，然后只对 shortlist 做局部音素编辑距离评分和门控。

## 3. 当前固定检索参数

外部检索脚本内部固定使用：

```text
language: Portuguese
retrieval mode: operating
retrieval backend: anchor_guided
threshold: 0.75
top_k: 5
minimum phonemes: 1
maximum edit ratio: 0.35
posterior weight: 0.25
minimum posterior confidence: 0.5
minimum top-1 margin: 0

Anchor shortlist size: 64
Anchor start radius: 2
Anchor n-gram sizes: 2,3,4
anchors per entry: 24
anchor offset tolerance: 1
```

因此当前命令对应之前的 k266、threshold 0.75、posterior 0.5、Top-5 口径。

## 4. 运行涉及的代码文件

核心入口和调度：

```text
scripts/run_external_keyword_retrieval.py
src/qwen_hotword/inference/external_keyword_retrieval.py
```

模型、Encoder 和 CTC：

```text
src/qwen_hotword/config.py
src/qwen_hotword/inference/streaming_backends.py
src/qwen_hotword/inference/streaming_core.py
src/qwen_hotword/modeling/qwen_backbone.py
src/qwen_hotword/modeling/audio_encoder.py
src/qwen_hotword/modeling/ctc_head.py
src/qwen_hotword/training/ctc_overfit.py
```

热词和检索：

```text
src/qwen_hotword/hotwords/registry.py
src/qwen_hotword/hotwords/anchor_index.py
src/qwen_hotword/hotwords/exact_automaton.py
src/qwen_hotword/hotwords/scoring.py
src/qwen_hotword/training/edit_distance.py
```

音素和评测辅助：

```text
src/qwen_hotword/phonemes/coverage.py
src/qwen_hotword/inference/hotword_prompt.py
```

## 5. 必须提供的模型、配置和数据

```text
Qwen3-ASR-1.7B 完整本地模型目录
ctc_head_best.pt
configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
pt_keyword_bias_phoneme.json
MLS Portuguese 音频目录和 transcripts.txt
Delivery PT-BR 音频目录和 transcripts.txt
```

Python 运行依赖至少包括：

```text
torch
torch_npu
qwen-asr
transformers
accelerate
numpy
soundfile
librosa
```

这条检索链路不需要 vLLM。

## 6. 输出和 resume 行为

运行过程中，每条已完成音频会写入：

```text
OUTPUT_DIR/sample_shards/<sha256-prefix>.json
```

`--resume` 会验证已有 `run_config` 和输入身份，然后跳过已经存在且身份正确的 shard。它不会加速尚未处理的样本，也不会对剩余样本启用 batching。

全部样本完成后才会生成：

```text
retrieval_output.json
retrieval_details.jsonl
evaluation_summary.json
failure_cases.jsonl
README.md
sha256.txt
```

## 7. 310P 上运行较慢的主要原因

### 7.1 2942 条音频完全串行，batch size 为 1

当前实现的主循环是：

```text
加载一条音频
  -> processor
  -> audio_tower 前向
  -> CTC Head
  -> CTC decode
  -> Anchor 检索
  -> 写一个 JSON
  -> 下一条音频
```

这是最主要的结构性吞吐限制。

### 7.2 完整模型 wrapper 加载

代码通过 `Qwen3ASRModel.from_pretrained()` 加载完整 Qwen3-ASR-1.7B，虽然推理阶段只使用 `audio_tower`。这会增加启动时间和设备内存占用。

### 7.3 Audio Encoder FP16，但 CTC Head 固定 FP32

当前数据类型边界是：

```text
Qwen audio_tower: float16
ln_post hidden states: float16
转换
CTC Head: float32
CTC logits/softmax: float32
```

在 310P eager 模式下，FP32 的 LayerNorm、Conv1d、depthwise Conv、GELU 和 Linear 可能效率较低，具体取决于 Ascend PyTorch 的算子支持和 fallback 情况。

### 7.4 当前同步函数只支持 CUDA，不支持 NPU

源码只在设备字符串以 `cuda` 开头时执行：

```python
torch.cuda.synchronize(device)
```

`npu:0` 不会进入这段逻辑。因此各阶段计时可能发生异步错位：Encoder 或 Head 的真实执行时间可能在后续 `.item()`、`.cpu()` 或 `.tolist()` 时才被同步，并被记入 decode 或 retrieval 时间。

这主要影响阶段级计时的可信度，不等同于推理本身一定变慢。总体墙钟时间、完成速度和 `audio_to_result_seconds` 更值得参考。

### 7.5 每条音频都有 NPU 到 CPU 的同步和复制

CTC decode 会把 token IDs 和 posterior confidences 复制为 CPU Python list，随后 Anchor Index 和音素编辑距离在 CPU 上运行。这会造成每条音频至少一次设备同步。

### 7.6 音频预处理和小文件 I/O

音频读取、单声道转换和重采样均在 CPU 完成。运行还会写入 2942 个小 JSON shard；如果输出目录位于慢速共享盘，小文件 I/O 也可能明显拖慢吞吐。

### 7.7 Anchor 检索通常不是主要瓶颈

当前只有 266 个热词，Anchor 查询后最多对 64 个候选做 rerank。正常情况下，Qwen audio tower 的计算成本应远高于 Anchor 检索成本。

## 8. 不停止当前任务的进度核查

```bash
OUT=/home/f00955675/h200_to_310p/results/pt_external_keyword_retrieval_mls_delivery_310p_fp16_audio_eager_v2

echo "已完成样本数："
find "$OUT/sample_shards" -type f -name '*.json' | wc -l

echo "最近一小时完成数："
find "$OUT/sample_shards" -type f -name '*.json' -mmin -60 | wc -l

echo "最近完成的样本："
find "$OUT/sample_shards" -type f -name '*.json' -printf '%T@ %p\n' |
  sort -n |
  tail -n 5

echo "运行进程："
ps -eo pid,etime,%cpu,%mem,cmd |
  grep '[r]un_external_keyword_retrieval.py'

npu-smi info
```

如果使用的是完整 MLS 和 Delivery 测试集，最终 shard 数量应达到 `2942`。

只要 shard 数量持续增加，说明进程仍在正常推进。中断后可以用完全相同的命令和输出目录继续 `--resume`。

## 9. 已完成部分的紧凑耗时统计

以下命令只读取已完成的 shard，不修改输出：

```bash
OUT=/home/f00955675/h200_to_310p/results/pt_external_keyword_retrieval_mls_delivery_310p_fp16_audio_eager_v2

python3 - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
groups = defaultdict(list)

for path in (root / "sample_shards").glob("*.json"):
    row = json.loads(path.read_text(encoding="utf-8"))
    groups[row["source"]].append(row)

def percentile(values, ratio):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values) - 1) * ratio)]

for source, rows in sorted(groups.items()):
    duration = sum(float(r["audio"]["duration_seconds"]) for r in rows)
    total = sum(float(r["timing"]["audio_to_result_seconds"]) for r in rows)
    sample_times = [float(r["timing"]["audio_to_result_seconds"]) for r in rows]

    print({
        "source": source,
        "completed": len(rows),
        "audio_hours": round(duration / 3600, 3),
        "wall_hours": round(total / 3600, 3),
        "rtf": round(total / duration, 4) if duration else None,
        "mean_seconds_per_sample": round(total / len(rows), 4),
        "p95_seconds_per_sample": round(percentile(sample_times, 0.95), 4),
        "mean_encoder_seconds": round(
            sum(float(r["timing"]["ctc_encoder_seconds"]) for r in rows) / len(rows),
            4,
        ),
        "mean_head_seconds": round(
            sum(float(r["timing"]["ctc_head_seconds"]) for r in rows) / len(rows),
            4,
        ),
        "mean_pure_retrieval_seconds": round(
            sum(float(r["timing"]["pure_retrieval_seconds"]) for r in rows) / len(rows),
            4,
        ),
    })
PY
```

注意：由于当前源码没有 NPU synchronize，阶段级的 Encoder、Head 和 retrieval 时间可能异步错位；总体 `audio_to_result_seconds`、实际完成样本速度和 RTF 更可信。

## 10. 当前结论

如果满足以下现象：

```text
NPU 利用率持续较高
sample shard 数量持续增长
进程没有重复报错或反复加载模型
```

那么运行一晚上仍未完成，很可能主要是当前“2942 条完整音频、单样本串行、Ascend eager 推理”架构的正常低吞吐。

如果出现以下现象：

```text
NPU 利用率长期很低
CPU 占用很高
每小时新增 shard 极少
显存占用或设备进程异常
```

则应优先检查：

1. Qwen audio tower 是否真正位于 `npu:0`；
2. `device_map="npu:0"` 是否被当前 qwen-asr、transformers 和 accelerate 正确支持；
3. 是否有大量算子 fallback 到 CPU；
4. attention、Conv1d、depthwise Conv、LayerNorm 和 softmax 的 Ascend eager 支持情况；
5. 音频目录和输出目录是否位于高延迟共享存储；
6. FP32 CTC Head 是否成为 310P 上的明显瓶颈。

在获得 NPU 利用率、当前 shard 数量、每小时完成量和紧凑耗时统计前，不建议仅凭“运行了一晚上”判断程序卡死或 Anchor 检索过慢。
