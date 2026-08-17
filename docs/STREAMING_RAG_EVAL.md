# Qwen3-ASR 流式端到端热词 RAG 评测

这套评测只改变 Qwen 的推理执行方式，不训练模型，不调整 CTC 阈值、
Top-K 或 Prompt 模板，不读 sealed test。A/B 直接导入既有离线 Retrieved RAG
产物，C/D/E 调用 Qwen 官方 vLLM 流式方法。

## 固定基线

```text
chunk_size_sec = 2.0
unfixed_chunk_num = 2
unfixed_token_num = 5
CTC input = 当前时刻的累计音频
candidate TTL = 0（每轮重新评分）
```

官方实现会把已收到音频累计为 0–2、0–4、0–6 秒重新解码。前两个
chunk 不加历史文本；从第 3 个 chunk 开始，用 `processor.tokenizer`对上轮
raw text 分词，回退 5 个 tokenizer token。尾音由
`finish_streaming_transcribe` 不补零处理。

## 官方接口限制

当前 `qwen-asr` 没有公开的流中动态 context setter。D/E 为了让当前累计
音频的 CTC 候选在同一 chunk 生效，会用公开
`init_streaming_state(context=...)` 创建临时 state，只把版本相关的
`prompt_raw/context/force_language` 复制到活跃 state。这是明示记录的
experimental state refresh，不是官方公开的动态 Prompt API。安装版本不暴露
这三个字段时程序会立即失败，不会静默延迟到下一 chunk。

实时 CTC 与 vLLM 当前是两个模型实例：Transformers 实例提取当前累计
音频的 `ln_post`，vLLM 实例执行官方流式解码。这是评测实现，尚未复用
同一份 Encoder 状态；`run_config.json` 会明确记录。

## 实验组

- A：导入离线 Baseline。
- B：导入离线 CTC RAG。
- C：官方流式，不注入热词。
- D：官方流式，因果累计音频 CTC RAG。
- E：官方流式，只注入该 case 的 Oracle 热词。Oracle 不进入 D。

## 输出

`scripts/run_streaming_rag_evaluation.py` 写入：

```text
run_config.json
sample_results.jsonl
chunk_timeline.jsonl
summary.json
boundary_summary.json
latency_summary.json
failure_cases.jsonl
README.md
sha256.txt
sample_shards/*.json
```

`sample_shards` 是原子写入的可恢复单样本分片。中断后只能在完全相同的
`run_config.json` 下使用 `--resume`；配置或输入 SHA256 不同会拒绝恢复。

原始离线集没有热词声学时间戳，因此该集的最终 Recall/WER/CER 有效，
但以声学结束为起点的 latency 会保持 `null`。边界集只接受
`forced_alignment` 或 `manual_confirmed` 时间，用于生成真正的延迟分布。

## 2 秒边界资产

`scripts/build_streaming_boundary_eval.py` 读取一个人工确认/强制对齐的 JSONL。
每行格式：

```json
{
  "case_id": "source_case_001",
  "sample_id": "validation_id",
  "audio_path": "/absolute/read-only/audio.wav",
  "reference_text": "...",
  "language": "Portuguese",
  "active_hotword_ids": ["hw_001", "hw_002"],
  "expected_hotword_ids": ["hw_001"],
  "coverage_tags": ["multiword_phrase"],
  "hotword_timings": [
    {
      "hotword_id": "hw_001",
      "start_sec": 4.12,
      "end_sec": 4.68,
      "timing_source": "manual_confirmed"
    }
  ]
}
```

生成器只写变体 manifest，在评测时内存中加前置静音，不复制或覆盖
原音频。默认要求中间、边界前、跨边界、边界后、尾音、多词短语、长热词、
多热词和负例全部有覆盖；不满足时拒绝生成完整基线。

## 工作区命令

从交付分支更新：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull origin codex/g2p-coverage-scan
```

先检查官方流式 API 和依赖：

```bash
python - <<'PY'
import importlib.metadata
import importlib.util
from qwen_asr import Qwen3ASRModel

print("qwen-asr:", importlib.metadata.version("qwen-asr"))
print("vllm:", importlib.util.find_spec("vllm"))
for name in (
    "LLM",
    "init_streaming_state",
    "streaming_transcribe",
    "finish_streaming_transcribe",
):
    print(name, hasattr(Qwen3ASRModel, name))
PY
python -m pip check
python scripts/run_streaming_rag_evaluation.py --help
python scripts/build_streaming_boundary_eval.py --help
```

如果 `vllm` 为 `None` 或流式方法缺失，先停止并返回版本/检查输出；不要
在正式环境里盲目升级。

小规模 50 条 A/B/C/D/E smoke：

```bash
EVAL_ROOT=outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100
STREAM_ROOT="$EVAL_ROOT/streaming_rag_2s_u2_t5_smoke50_v1"

CUDA_VISIBLE_DEVICES=5 python scripts/run_streaming_rag_evaluation.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords "$EVAL_ROOT/stratified_hotwords_v2.jsonl" \
  --cases "$EVAL_ROOT/stratified_hotword_cases_v2.jsonl" \
  --offline-rag-dir "$EVAL_ROOT/retrieved_rag_v1" \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --output-dir "$STREAM_ROOT" \
  --groups A,B,C,D,E \
  --max-samples 50 \
  --chunk-size-sec 2.0 \
  --unfixed-chunk-num 2 \
  --unfixed-token-num 5 \
  --threshold 0.86 \
  --top-k 3 \
  --maximum-edit-ratio 0.35 \
  --minimum-top1-margin 0 \
  --gpu-memory-utilization 0.70
```

中断后重跑同一命令并加 `--resume`。先检查时间线：

```bash
sed -n '1,5p' "$STREAM_ROOT/chunk_timeline.jsonl"
python -m json.tool "$STREAM_ROOT/summary.json"
python -m json.tool "$STREAM_ROOT/latency_summary.json"
sed -n '1,20p' "$STREAM_ROOT/failure_cases.jsonl"
```

完整原始 100 条评测使用新目录，去掉 `--max-samples 50`。

边界源 spec 人工确认后：

```bash
BOUNDARY_ROOT="$EVAL_ROOT/streaming_boundary_2s_v1"

python scripts/build_streaming_boundary_eval.py \
  --source-spec "$EVAL_ROOT/streaming_boundary_source_confirmed_v1.jsonl" \
  --output-dir "$BOUNDARY_ROOT/assets" \
  --chunk-size-sec 2.0 \
  --minimum-hotword-start-sec 4.0

CUDA_VISIBLE_DEVICES=5 python scripts/run_streaming_rag_evaluation.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords "$EVAL_ROOT/stratified_hotwords_v2.jsonl" \
  --cases "$EVAL_ROOT/stratified_hotword_cases_v2.jsonl" \
  --offline-rag-dir "$EVAL_ROOT/retrieved_rag_v1" \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --boundary-manifest "$BOUNDARY_ROOT/assets/boundary_cases.jsonl" \
  --output-dir "$BOUNDARY_ROOT/run_v1" \
  --groups C,D,E \
  --threshold 0.86 \
  --top-k 3 \
  --gpu-memory-utilization 0.70
```

主评测和边界评测的 `summary.json`/`boundary_summary.json`/
`latency_summary.json`/`failure_cases.jsonl` 共同用于第一轮结论。
