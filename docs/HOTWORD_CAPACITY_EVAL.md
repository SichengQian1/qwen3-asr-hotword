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
