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
ctc_prefix_stability.json
rank_displacement_cases.jsonl
rank_displacement_summary.json
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

Top-5仍是封存的产品决策与Prompt注入口径，不随词库扩容改变。从第2步开始所有
容量评测同时保存Raw Top-7和Top-10作为观察范围：记录逐query候选ID、期望热词
命中数、总体Recall和正例case hit rate。Top-7/10只用于判断容量增加后目标词是
“轻度掉出Top-5”还是“候选召回失败”，不用于降低0.86门控，也不扩大本轮
Operating/Prompt的Top-5。

`verified_maximum`是实际全部通过的最大规模；`recommended_online_cap`默认退回一个
已测试档位保留工程余量。例如10k通过建议先上线5k，5k通过但10k失败建议先上线2k。

当前matcher是Python逐热词局部编辑距离全扫描再全量排序。若5k或10k失败，先保存
本基线，再考虑音素长度桶/候选索引加精确重排；不得修改本轮结果来掩盖失败。

第一次工作区基线在纯Python动态规划后端上仅100个active热词就得到retrieval
P95约2.070秒，因此按保护策略停止；该结果必须保留为slow-backend基线，不能解释成
模型容量为0。后续实现保持窗口、距离、排序和门控完全不变，只把Levenshtein距离
换成项目已锁定的RapidFuzz 3.x等价C++后端。新benchmark的`run_config.json`和
`summary.json`必须显示`edit_distance_backend=rapidfuzz`，并使用新输出目录。

## 7. 已冻结的RapidFuzz基线与2k目标

2026-08-19已完成Representative基线。质量数字来自同一份formal100离线replay；
流式工程数字来自20条真实累计音频、67个step（含17个tail flush）。以下结果必须
保留，后续GPU候选器以它们为对照，不能通过改阈值或改Top-5重写基线：

| active词数 | Raw Recall@5 | Operating Recall@5 | 负例FPR | 流式retrieval P95 | CTC+retrieval P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 95.35% | 81.40% | 0% | 109 ms | 514 ms |
| 500 | 94.19% | 81.40% | 0% | 720 ms | 1.051 s |
| 1,000 | 91.86% | 81.40% | 5% | 1.097 s | 1.411 s |
| 2,000 | 88.37% | 80.23% | 20% | 2.893 s | 3.021 s |

因此2k不能只靠增量编辑距离解决：matching时延和Top-5排序/FPR都已经失效。下一版
目标架构固定为“GPU帧级CTC候选召回 -> Top-128/256（Top-512诊断） ->
CPU/GPU精确重排 -> Operating Top-5”。精确重排同时输出Raw Top-7/10观察值，
但实际注入仍只取Operating Top-5。在实现候选器前先完成两项不改语义的证据建设：

1. 从既有decoded replay导出processor/Encoder/Head/decode分阶段分布、累计CTC前缀
   稳定性，以及每个目标词被挤出Top-5或被Operating guard拦下的逐例记录。
2. 重新生成float16帧级`log_softmax` Posterior Replay，供后续GPU packed CTC scorer
   复用；每个分片必须通过SHA256、shape、有效长度、归一化和greedy collapse等价校验。

新benchmark输出会额外包含：

```text
performance_summary.json
  -> performance.source_latency_seconds.{processor,encoder,ctc_head,ctc_decode,source_ctc}
ctc_prefix_stability.json
  -> 每个相邻累计chunk的LCP、被改写后缀长度、append-only率
rank_displacement_cases.jsonl
  -> expected_displaced_below_raw_top5 / raw_top5_rejected_by_operating_guards /
     negative_false_positive等逐例证据
rank_displacement_summary.json
```

排名挤出以完整formal100离线replay为主，不加载Qwen：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_ROOT="$CAP_ROOT/assets"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
STREAM_REPLAY_SMOKE_ROOT="$CAP_ROOT/replay_streaming_smoke20_v1"

python scripts/benchmark_hotword_capacity.py \
  --assets-root "$ASSET_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 100,500,1000,2000 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --stop-retrieval-p95-seconds 2.0 \
  --output-dir "$CAP_ROOT/benchmark_offline_formal100_diagnostics_v2"
```

再对20条流式replay生成分阶段时延和跨chunk前缀稳定性：

```bash
python scripts/benchmark_hotword_capacity.py \
  --assets-root "$ASSET_ROOT" \
  --replay "$STREAM_REPLAY_SMOKE_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 100,500,1000,2000 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --stop-retrieval-p95-seconds 2.0 \
  --output-dir "$CAP_ROOT/benchmark_streaming_smoke20_diagnostics_v2"
```

formal100离线结果是2k排名质量主证据；smoke20只用于补充真实累计
流式的前缀改写和时延证据，不得用其替代完整100条质量结论。

## 8. 生成帧级Posterior Replay

Posterior Replay仍使用0-2/0-4/...累计音频和原Temporal 2× Head。它不改变检测结果，
只在原`ctc_replay.jsonl`旁新增张量资产：

```text
posterior_shards/part-00000.pt ...  # [N,Tmax,90] float16 log-softmax
posterior_index.jsonl               # case/chunk -> shard row
posterior_shards.json               # shape、长度、SHA256和分片元数据
```

先在20条smoke上生成，输出必须使用新目录：

```bash
POSTERIOR_SMOKE_ROOT="$CAP_ROOT/replay_streaming_posterior_smoke20_v2"

CUDA_VISIBLE_DEVICES=4 python scripts/build_hotword_capacity_replay.py \
  --mode streaming \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --cases "$ASSET_ROOT/representative/size_100/cases.jsonl" \
  --output-dir "$POSTERIOR_SMOKE_ROOT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --language Portuguese \
  --chunk-size-sec 2.0 \
  --max-samples 20 \
  --save-log-posteriors \
  --posterior-shard-size 32

cat "$POSTERIOR_SMOKE_ROOT/summary.json"
cat "$POSTERIOR_SMOKE_ROOT/posterior_shards.json"
(cd "$POSTERIOR_SMOKE_ROOT" && sha256sum -c sha256.txt)
```

验收条件：`posterior_replay.status=pass`、`num_classes=90`、
`storage_dtype=float16`、`quantization=argmax_preserving_float16`、
`greedy_equivalence_mismatches=0`、全部SHA256通过，且20条
smoke应仍为67个step/17个tail flush（若输入case未变）。这些资产确认后才进入
AC精确基线与音素Anchor Top-64/128/256候选器；GPU packed Posterior scorer改为
Anchor召回不足时的后备。本阶段不实现候选保留、TTL、阈值调整或轻量reranker。

2026-08-20的首次原始float16 smoke写內容器完成67个step和3个分片，但严格
校验发现1个row的greedy collapse不等价，因而正确拒绝生成
`summary.json`。原因是float32近乎并列的两个frame log posterior转float16后变成
相同值，`argmax`因索引顺序翻转。修复使用显式argmax保持量化：只对发生舍入
并列的帧做一个float16 ULP级调整，并记录`argmax_correction_frames`和
`maximum_abs_quantization_error`。修正帧数可以非零，但greedy token/span必须仍为
100%等价。失败的`replay_streaming_posterior_smoke20_v1`保留为证据，不删除、
不覆盖；修复后使用上述`v2`新目录重跑。

## 9. 4,000词/50 ms目标与Aho-Corasick精确基线

2026-08-21将当前产品目标定义为：已有CTC decoded音素序列到热词候选完成的
纯检索延迟，在4,000个active热词下P95不超过50毫秒。该数字不包含音频
processor、Qwen Encoder、CTC Head或LLM解码。Operating仍使用Top-5/0.86，
Raw Top-7/10只作观察。

Posterior Replay v2已在H200通过：20个case、67个step、17个tail flush、3个
float16分片和7,552个有效时间帧；`argmax_correction_frames=1`、
`greedy_equivalence_mismatches=0`、`quantization=argmax_preserving_float16`，
全部`sha256.txt`条目为OK，且`test_set_used=false`。Step 2因此封存完成。

新增整数音素Aho-Corasick精确基线：

```text
src/qwen_hotword/hotwords/exact_automaton.py
src/qwen_hotword/hotwords/exact_capacity.py
scripts/benchmark_exact_hotword_capacity.py
```

自动机对每个replay step的完整当前greedy音素序列重新扫描，不假设累计CTC前缀
append-only；查询后做最长匹配过滤，再以span平均置信度、音素长度和最小置信度做
确定性排序。报告同时保存未过滤/最长匹配数、Exact availability、Exact
Top-5/7/10、负例FPR、索引节点/转移/构建内存及50毫秒deadline miss。

本地对上传的formal100/2k资产做过只读探针：热词表实际共2,402个唯一entry、
14,916个自动机节点，构建约12.28毫秒；100条完整序列扫描的平均/P95/P99/最大值
约为0.101/0.156/0.201/0.305毫秒。该时间只说明纯AC实现不是50毫秒瓶颈，不能
替代工作区Linux CPU实测。纯精确子串在当前CTC上只覆盖132/172期望词
（76.74%），2k负例有3/20精确误触发，因此AC只是快速通道/诊断基线，不直接
替代近似召回。

旧`assets`不覆盖。为保持旧曲线可比，新资产仍以10k为最大采样池，只在2k与5k
之间新增4k档：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"

python scripts/build_hotword_capacity_assets.py \
  --training-manifest outputs/noah_pt_full_training_v1/full_ctc_train.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --base-hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --base-cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --selection "$FORMAL100_ROOT/sample_selection.json" \
  --sizes 100,500,1000,2000,4000,5000,10000 \
  --candidate-pool-multiplier 3 \
  --output-dir "$ASSET_4K_ROOT"
```

构建后必须先比较旧新共有档的hotwords/cases SHA256；种子、最大容量和输入没变时，
100/500/1k/2k/5k/10k应保持一致。然后用已有offline formal100 replay运行纯AC：

```bash
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
EXACT_4K_ROOT="$CAP_ROOT/benchmark_exact_ac_offline_formal100_4000_v1"

python scripts/benchmark_exact_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 100,500,1000,2000,4000 \
  --deadline-ms 50 \
  --output-dir "$EXACT_4K_ROOT"

cat "$EXACT_4K_ROOT/quality_summary.json"
cat "$EXACT_4K_ROOT/performance_summary.json"
cat "$EXACT_4K_ROOT/summary.json"
(cd "$EXACT_4K_ROOT" && sha256sum -c sha256.txt)
```

这一步用于确认4k下AC纯检索P95距离50毫秒的余量，并封存精确通道的Recall/FPR
上限，不用它直接替代Operating结果。通过后再实现音素Anchor Top-64/128/256
shortlist和现有近似评分精排。

## 10. 位置Anchor候选召回层

Step 3使用音素2/3/4-gram的稀有度加权倒排索引。每个热词默认选最多24个
低document-frequency Anchor，并保留Anchor在热词发音中的位置。查询时按
`query_position - entry_position`聚合证据，允许±1 offset以容纳单次插入/删除；
这样同一批n-gram只有在局部位置一致时才形成强证据，避免把散落在长序列不同位置的
常见音素片段误当成一个热词。Aho-Corasick精确命中无条件并入候选，然后与Anchor
候选形成一条确定性排序，Top-64/128/256只是同一排序的切片，因此严格嵌套。

本阶段只衡量候选召回，不做最终近似精排。`anchor_retrieval_seconds`只覆盖AC与Anchor
查询；为衡量候选是否损失现有质量，同一次运行还会执行原完整CPU近似扫描并保存
Raw Top-5参考，但其`full_scan_reference_seconds`不计入50 ms预算。输出重点包括：

- `expected_recall_at_64/128/256`；
- `positive_case_hit_rate_at_64/128/256`；
- `reference_top5_coverage_at_64/128/256`及正例专用覆盖率；
- 全查询与final查询的`no_anchor_rate`；
- shortlist实际候选数、访问posting数；
- Anchor查询P50/P95/P99/max与50 ms deadline miss；
- 索引构建时间、Python heap和RSS。

正式4k命令：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
ANCHOR_4K_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v1"

test ! -e "$ANCHOR_4K_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64,128,256 \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --output-dir "$ANCHOR_4K_ROOT"

(cd "$ANCHOR_4K_ROOT" && sha256sum -c sha256.txt)
cat "$ANCHOR_4K_ROOT/quality_summary.json"
cat "$ANCHOR_4K_ROOT/performance_summary.json"
cat "$ANCHOR_4K_ROOT/summary.json"
```

正式计时应避开Feature Cache等CPU/存储高负载任务。收到4k结果后先决定Top-256是否
达到候选召回要求；Top-128只作速度消融。下一步才在候选集上复用既有近似评分器并
比较full-current与2/4/6秒lookback，不能在本阶段调整0.86、Top-5或Prompt。

## 11. Anchor v1基线与等价加速复测

4k v1正式结果显示候选质量已接近可用：Top-64/128/256 expected recall分别为
95.35%/97.67%/97.67%，Top-256对完整CPU Raw Top-5覆盖率为99.2%，且无Anchor率
为0。但P50/P95/P99为26.02/87.36/94.83 ms，11%查询超出50 ms，因此v1不能通过
性能验收。Top-128到256没有新增expected hit，4个miss需要逐例诊断。

v2保持完全相同的Anchor、AC并集和排序键，只做等价执行优化：固定±1 offset直接
查询相邻bucket，不再对每个center扫描全部offset；仅用有界heap选择最终Top-256，
不再全排所有Anchor候选；active ID集合只构建一次。输出新增：

```text
diagnostic_summary.json
diagnostic_cases.jsonl
```

其中原因包括`expected_not_in_anchor_top_256`、
`reference_top5_not_in_anchor_top_256`和`anchor_retrieval_over_deadline`，并保存
case、chunk、累计时长、posting访问量和候选rank。v1目录保持只读，v2输出使用
`benchmark_anchor_offline_formal100_4000_v2`。正式计时应等待Feature Cache结束并
保持CPU/存储空闲；否则只能比较质量，不能据此验收50 ms。

## 12. v2诊断与v3两阶段计时

v2质量与v1完全一致，证明等价加速没有改变候选。4个Top-256 expected miss由3个
唯一热词构成，其中`sim_v3_hw_ptbr_0169`出现两次；4项均不在对应完整CPU Raw
Top-5。另4个未覆盖的Raw Top-5项均为非truth distractor。因此Top-256相对现有
完整扫描的正式正确项没有发现新增损失，后续仍以Top-256为主精排候选集。

v2 P50/P90/P95为24.07/39.23/82.21 ms。8个超时查询存在明显非算法性双峰：部分
低posting查询反而比高posting查询慢。v1/v2逐case执行顺序是“完整CPU扫描→Anchor
计时”，3至8秒的慢参考会污染下一次Anchor的GC、分配器和调度状态。

v3改为：

1. 构建索引并warmup；
2. `gc.collect()`；
3. 连续完成全部Anchor计时；
4. Anchor计时全部结束后才执行完整CPU参考；
5. 按case合并质量字段。

该变更不改变任何候选结果，只修复benchmark测量协议。输出显式记录
`timing_protocol=all_anchor_queries_before_full_scan_reference`。v3仍需在CPU空闲时
运行；若P95仍超过50 ms，才允许进入会改变召回的高DF posting cap消融。

## 13. v3结果与统一Top-5/7/10优化历史

v3严格隔离慢参考后，Anchor P50/P90/P95/P99为18.86/30.07/62.73/73.32 ms，
6%查询超过50 ms。候选质量继续与v1/v2一致：Top-64/128/256 expected recall为
95.35%/97.67%/97.67%，Top-256覆盖完整CPU Raw Top-5的99.2%。因此当前核心查询
大多数已进入预算，但P95仍未通过；下一性能消融先定位Python GC/调度尾延迟，不能
直接用posting cap牺牲质量。

后续所有容量阶段统一输出四组指标：

```text
候选器 Raw Top-5/7/10：correct / recall / precision / positive case hit
完整扫描参考 Raw Top-5/7/10：correct / recall / precision / positive case hit
完整扫描 Operating Top-5：correct / recall / precision / negative FPR
性能：P50 / P90 / P95 / P99 / max及deadline miss
```

Raw precision分母是所有final query实际返回的Top-K候选数，包含负例；Operating
precision分母是实际通过threshold、edit ratio、posterior和margin门控的候选数。
Raw Top-7/10只用于判断目标是否轻度掉出Top-5，不能扩大Prompt注入。

`scripts/summarize_hotword_capacity_history.py`接受多个`LABEL=OUTPUT_DIR`，读取各阶段
`query_results.jsonl`并生成统一的`optimization_history.json/tsv`。它兼容旧全扫描、
Exact AC及Anchor输出；旧Anchor只有参考Top-5，因此对应参考Top-7/10和Operating列
保持`null`。新Anchor运行额外保存完整字段后即可填满这些列。工作区命令和固定目录
见`docs/HANDOFF.md` 0.42。

## 14. GC尾延迟A/B

v3/v4在严格两阶段计时下仍出现稳定的约62毫秒P95，但P90只约27至30毫秒。
下一步先检验Python cyclic GC是否与慢查询重叠，不修改Anchor、候选数、Top-5/7/10、
0.86门控或Prompt。`benchmark_anchor_hotword_capacity.py`新增：

```text
--gc-policy normal
--gc-policy defer_during_anchor_pass
```

`normal`保持Python默认GC行为；`defer_during_anchor_pass`只在连续Anchor计时阶段禁用
自动cyclic GC，Anchor阶段结束后立即恢复原状态并在时延口径外显式`gc.collect()`。
这不是永久禁用GC，也不包括完整CPU参考扫描。两组均在warmup后、Anchor计时前
执行一次显式GC，保持起点可比。

新输出：

```text
gc_events.jsonl
  -> 每次GC的generation、耗时、collected/uncollectable及所在query

gc_summary.json
  -> GC策略、计时阶段启用状态、GC/query重叠、超时query中的GC占比，
     以及时延口径外的前/后显式GC耗时
```

`query_results.jsonl`每行同时记录`gc_collections_during_query`、
`gc_seconds_during_query`和`gc_generations_during_query`；超过50毫秒的诊断行也携带这些
证据。默认值仍为`normal`，老命令行为不变。

验收时先要求两组`quality_summary.json`完全相同，再比较P50/P90/P95/P99和
deadline miss。若normal慢query多数与GC重叠，且defer稳定把P95压到50毫秒内，
才能将GC调度视为主要原因；若defer仍约62毫秒，说明尾延迟更可能来自OS调度、
CPU竞争或实现分配峰值，下一步应做process pinning/pyperf或分配调查，不应直接用
posting cap牺牲候选召回。

## 15. GC结论与Anchor小候选精排

工作区A/B确认7条normal慢查询全部与generation 0/1/2 GC重叠，GC耗时43.07至
53.30 ms；defer组质量逐字节一致，Anchor P95从63.56 ms降至19.52 ms，max从
83.46 ms降至22.33 ms，deadline miss从7%降为0。因此Step 3完成，暂不做posting
cap。该结论只证明连续Anchor查询本体达标；生产服务仍需在Step 5验证持续流中的
安全GC调度和RSS，不能把每请求后的同步full GC算到请求外后直接上线。

Step 4使用`scripts/benchmark_anchor_rerank_capacity.py`，顺序固定为：当前因果CTC
序列（或其recent 2/4/6秒frame窗口）→ AC+Anchor shortlist → 仅在shortlist上运行
既有近似音素评分器 → Raw Top-5/7/10与Operating Top-5。窗口由当前replay row的
`effective_time_steps`和`cumulative_audio_sec`计算，不按字符/单词截断，不读取下一
chunk，也不把offline full结果注入streaming。

性能口径为`anchor_seconds + rerank_seconds`；报告同时保留两阶段各自分布。正式
参数继续固定0.86/Top-5/0.35/0.25/min posterior 0/margin 0，Raw Top-7/10只观察，
不能扩大Prompt。历史汇总按每个`window × shortlist`分别生成`anchor_rerank`行，
不得混算不同变体。工作区完整命令与输出检查见`docs/HANDOFF.md` 0.47。

## 16. Step 4结果与Anchor引导局部精排

Step 4证明候选生成已不再是总检索瓶颈。离线formal100中，shortlist 64/128/256的
总检索P95分别为123.45/157.87/446.07 ms，其中rerank P95分别占90.68/133.09/
413.03 ms。三档Operating Top-5 recall都为80.23%，precision都为55.42%，负例FPR
都为25%；扩大候选集没有带来Operating收益。64档Raw Top-5/7/10 recall为
86.63%/90.70%/93.02%，因此后续以64为性能主线、128为质量对照，停止256主线精排。

流式recent窗口必须同时报告两类口径：

- `final_*`：最后一个chunk仍保留的候选，衡量最终窗口状态；
- `any_step_*`：任一因果chunk曾检出同一case/hotword，衡量真实流式发现能力。

只看final会把已经检出、后来滑出recent窗口的热词误算成漏检。`any_step_*`不做TTL、
不向后续chunk注入历史候选，也不改变在线策略，只是对已有逐chunk结果做评测聚合。
Raw Top-5/7/10和Operating Top-5均输出recall、precision与positive case hit；负例还输出
任一step误检率。

下一步`anchor_guided`模式复用同一个近似评分器、分数公式、排序键和门控，只限制每个
候选的滑窗起点：以Anchor的`best_offset`为中心，默认搜索前后2个decoded token
位置。候选没有有效offset时仍回退该候选的完整搜索。报告写入`rerank_mode`和
`anchor_start_radius`，优化历史据此区分full-search与guided，不能把两者当作等价实现。
第一轮只测shortlist 64/128、radius 2；完整工作区命令与验收顺序见
`docs/HANDOFF.md` 0.48。

## 17. Operating Top-5/7/10观察口径

Raw Top-7/10只说明正确热词是否进入相应排名，不能回答现有0.86、edit ratio、posterior
和margin门控之后的Recall/Precision。Anchor rerank报告因此新增Operating@5/@7/@10。
它们共享同一次候选生成、局部精排和排序，只改变门控通过后最多保留的观察数量，额外
统计不进入检索延迟计时。

实际运行仍由`--top-k 5`控制，`operating_ids`继续表示真实Top-5输出；新增
`operating_top7_ids`和`operating_top10_ids`仅用于评测。每档同时输出final与流式
`any_step` correct、recall、precision、positive case hit和负例FPR。容量历史对旧输出
保持兼容：旧阶段只有Operating@5，@7/@10必须为`null`，不得拿Raw@7/@10代替。

固定4k guided基线的重跑命令和输出检查见`docs/HANDOFF.md` 0.49。该轮仍冻结全部检索
参数；结果用于判断“扩大Operating观察上限”本身的Recall收益和Precision代价，不作为
Prompt Top-K变更授权。

## 18. Step 5 Operating门控扫描

门控扫描必须建立在完整保存的精排shortlist上。只保存原Top-20时，如果threshold、edit
ratio或posterior门控拒绝前排候选，第21至64名可能补入Operating结果，因此不能把旧
Top-20报告当作精确门控重放。先用`--saved-ranked-matches 64`生成独立benchmark目录，
再用`sweep_hotword_operating_points.py`扫描：

```text
Operating Top-K
score threshold
maximum edit ratio
minimum posterior confidence
minimum top-1 margin
```

Posterior weight在该重放中保持源benchmark值，因为它参与score并改变完整排序。要扫描
posterior weight必须重新精排，不能在已截断或已排序JSON上近似推断。

每个扫描点同时输出final和any-step的Recall、Precision、正例case hit、负例FPR、返回
候选数与错误候选数。离线formal100用`final`选点；流式验证使用`any_step`解释因果检出。
推荐逻辑先要求源检索P95不超过50 ms和Recall达到目标，再在合格点内最大化Precision、
最小化FPR并偏好较小Top-K。Precision 85%当前是诊断目标，不是硬阻塞条件。
`recommended_config.json`还分别保留Top-5/7/10在原门控下的结果和每个Top-K自己的最优
Recall-first点，便于逐步比较，不只输出一个跨Top-K总推荐。
如果某个Top-K没有任何点达到目标Recall，报告回退为先最大化Recall、再最大化Precision
和最小化FPR；不能把高Precision但极低Recall的点标成该Top-K的Recall-first推荐。

产物包括：

```text
run_config.json
sweep_results.jsonl
pareto_frontier.jsonl
recommended_config.json
summary.json
sha256.txt
```

完整工作区命令见`docs/HANDOFF.md` 0.50。扫描推荐点只用于下一轮端到端诊断；由于选点
和报告来自同一formal100，不得把它当成独立测试集上的最终泛化指标。
