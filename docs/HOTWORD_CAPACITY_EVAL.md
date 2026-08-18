# 葡语热词库容量评测

本评测只测已经封存的葡语 Temporal 2× CTC Head，不训练模型、不读取sealed test，
也不修改既有v3离线或流式结果。容量指每次查询真正参与排序的
`active_hotword_ids`数量，而不是JSONL文件总行数。

固定容量阶梯：

```text
100 -> 500 -> 1,000 -> 2,000 -> 5,000 -> 10,000
```

固定检索参数：Top-5、threshold 0.86、maximum edit ratio 0.35、posterior weight
0.25、minimum phonemes 4、minimum posterior 0、margin 0。10,000是压力上限，
不是预先承诺的上线容量。

## 1. 环境和输入

在H200容器仓库根目录执行：

```bash
V3_ROOT=outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested
FORMAL100_ROOT="$V3_ROOT/prompt_multi_nested_formal100_top5_v1"
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_ROOT="$CAP_ROOT/assets"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
STREAM_REPLAY_SMOKE_ROOT="$CAP_ROOT/replay_streaming_smoke20_v1"

test -f outputs/noah_pt_full_training_v1/full_ctc_train.jsonl
test -f outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl
test -f outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
test -f "$V3_ROOT/multi_nested_hotwords_v3.jsonl"
test -f "$V3_ROOT/multi_nested_cases_v3.jsonl"
test -f "$FORMAL100_ROOT/sample_selection.json"
test -f outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt

python scripts/build_hotword_capacity_assets.py --help >/dev/null
python scripts/build_hotword_capacity_replay.py --help >/dev/null
python scripts/benchmark_hotword_capacity.py --help >/dev/null
```

所有输出目录都拒绝覆盖非空目录。重复实验必须使用新目录，不能删除或覆盖现有
Operating/Forced评测。

## 2. 构建确定性嵌套词库

该步骤是CPU任务。新增候选来自Noah葡语train-only Manifest中的真实1至4词连续
n-gram；base 100来自既有v3 case。`--selection`把case严格限制为既有Operating
formal100的同一批80正例+20负例。
构建器还会拒绝非`formal100`或明确标记为`forced_topk`的选择文件；2026-08-18
以前生成、尚无`retrieval_mode`字段的旧Operating formal100兼容放行。选择文件中的
最长匹配真值会覆盖原始case真值；正式结果应为100条case、172个
`base_expected_hotwords`。

```bash
python scripts/build_hotword_capacity_assets.py \
  --training-manifest outputs/noah_pt_full_training_v1/full_ctc_train.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --base-hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --base-cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --selection "$FORMAL100_ROOT/sample_selection.json" \
  --sizes 100,500,1000,2000,5000,10000 \
  --candidate-pool-multiplier 3 \
  --output-dir "$ASSET_ROOT"
```

先检查：

```bash
cat "$ASSET_ROOT/asset_summary.json"
cat "$ASSET_ROOT/representative/size_100/summary.json"
cat "$ASSET_ROOT/representative/size_10000/summary.json"
cat "$ASSET_ROOT/hard_negative/size_10000/summary.json"
```

每一级每条case的active数量必须等于目录名；大一级必须包含小一级的全部active IDs。
候选表中的`capacity_train_occurrences`才是扩容词的真实train频次；历史字段
`validation_occurrences=1`只为兼容现有registry schema，不能误读为validation证据。
词库加载时间先在不启用`tracemalloc`时测量；Python heap另做一次带
`tracemalloc`的reload，并分别写为`load_seconds`和
`tracemalloc_profiled_reload_seconds`，避免把内存探针开销混进正式加载时间。

## 3. 生成离线CTC replay

离线replay只加载Validation feature cache和CTC Head，不加载完整Qwen模型。它固定
formal100的完整音频decoded phoneme，使六级词库共享完全相同的CTC输入。

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/build_hotword_capacity_replay.py \
  --mode offline \
  --validation-cache outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --cases "$ASSET_ROOT/representative/size_100/cases.jsonl" \
  --output-dir "$OFFLINE_REPLAY_ROOT" \
  --device cuda:0 \
  --batch-size 64

cat "$OFFLINE_REPLAY_ROOT/summary.json"
```

预期`cases=100`、`final_rows=100`、`test_set_used=false`。

## 4. 离线质量和纯检索性能

先运行Representative主曲线：

```bash
python scripts/benchmark_hotword_capacity.py \
  --assets-root "$ASSET_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 100,500,1000,2000,5000,10000 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --stop-retrieval-p95-seconds 2.0 \
  --output-dir "$CAP_ROOT/benchmark_offline_representative_v1"
```

默认一旦某一级纯检索P95超过2秒就停止更大规模。不要使用
`--continue-after-deadline-failure`绕过保护，除非已经确认资源和预计运行时间。

Representative完成后再单独运行Hard-negative，避免两条曲线相互覆盖：

```bash
python scripts/benchmark_hotword_capacity.py \
  --assets-root "$ASSET_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles hard_negative \
  --sizes 100,500,1000,2000,5000,10000 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --stop-retrieval-p95-seconds 2.0 \
  --output-dir "$CAP_ROOT/benchmark_offline_hard_negative_v1"
```

主检查文件：

```text
quality_summary.json
performance_summary.json
capacity_recommendation.json
query_results.jsonl
summary.json
sha256.txt
```

`capacity_recommendation.json`只以Representative曲线给正式容量建议；Hard-negative
始终是压力诊断。Raw Recall@5下降表示排序碰撞，Raw仍命中但Operating丢失表示门控
问题。

## 5. 2秒累计流式工程smoke

先只生成20条的逐2秒累计CTC replay。该步骤加载完整Qwen Encoder，但不加载vLLM
Decoder，记录processor、Encoder、CTC Head、posterior decode及GPU内存。

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/build_hotword_capacity_replay.py \
  --mode streaming \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --cases "$ASSET_ROOT/representative/size_100/cases.jsonl" \
  --output-dir "$STREAM_REPLAY_SMOKE_ROOT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --language Portuguese \
  --chunk-size-sec 2.0 \
  --max-samples 20

cat "$STREAM_REPLAY_SMOKE_ROOT/summary.json"
```

然后只跑Representative工程曲线：

```bash
python scripts/benchmark_hotword_capacity.py \
  --assets-root "$ASSET_ROOT" \
  --replay "$STREAM_REPLAY_SMOKE_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 100,500,1000,2000,5000,10000 \
  --threshold 0.86 \
  --top-k 5 \
  --stop-retrieval-p95-seconds 2.0 \
  --output-dir "$CAP_ROOT/benchmark_streaming_smoke20_representative_v1"
```

smoke20只能验证链路和发现明显性能拐点，不能作为正式容量结论。通过后把replay命令
的`--max-samples 20`去掉，在新目录生成formal100 streaming replay，再运行同一
Representative benchmark。

## 6. 判定规则

Representative每一级必须同时满足：

```text
Raw Recall@5相对100词下降 <= 1个百分点
Operating Recall@5相对100词下降 <= 1个百分点
负样本FPR <= 3%
纯retrieval P95 <= 100 ms
纯retrieval P99 <= 200 ms
有真实逐chunk source timing时，CTC+retrieval P95 < 2 s
```

`verified_maximum`是实际全部通过的最大规模；`recommended_online_cap`默认退回一个
已测试档位保留工程余量。例如10k通过建议先上线5k，5k通过但10k失败建议先上线2k。

当前matcher是Python逐热词局部编辑距离全扫描再全量排序。若5k或10k失败，先保存
本基线，再考虑音素长度桶/候选索引加精确重排；不得修改本轮结果来掩盖失败。

第一次工作区基线在纯Python动态规划后端上仅100个active热词就得到retrieval
P95约2.070秒，因此按保护策略停止；该结果必须保留为slow-backend基线，不能解释成
模型容量为0。后续实现保持窗口、距离、排序和门控完全不变，只把Levenshtein距离
换成项目已锁定的RapidFuzz 3.x等价C++后端。新benchmark的`run_config.json`和
`summary.json`必须显示`edit_distance_backend=rapidfuzz`，并使用新输出目录。
