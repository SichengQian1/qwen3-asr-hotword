# 工作交接记录

## 0.19 2026-08-18 葡语100至10k热词库容量评测（代码完成，待H200）

按照已确认的容量阶梯`100/500/1k/2k/5k/10k`新增葡语单语热词库容量评测。
10k只是当前精确扫描实现的压力上限，不预先宣称为上线容量；100k不在本轮范围。
固定复用Temporal 2× CTC Head、Top-5、threshold 0.86、maximum edit ratio 0.35、
posterior weight 0.25、minimum phones 4及既有v3 formal100，不训练、不调参、不读
sealed test。

新增实现：

- `src/qwen_hotword/hotwords/capacity_assets.py`：从Noah葡语train-only Manifest
  确定性采样真实1至4词连续n-gram，校验MFA/phone覆盖，构建Representative与
  Hard-negative两套严格嵌套active词库；支持读取既有formal100
  `sample_selection.json`，强制formal100并拒绝明确的forced选择，同时兼容尚无
  `retrieval_mode`字段的旧Operating formal100；以选择文件中的最长匹配真值覆盖
  原始多真值，防止扩容时悄悄改变case集合或真值口径。
- `src/qwen_hotword/hotwords/capacity_replay.py`：生成完整Validation feature-cache
  离线CTC replay，或真实累计0-2/0-4/...音频的流式CTC replay；后者分段记录
  processor、Encoder、CTC Head、posterior decode和GPU内存，尾部不足2秒保留。
- `src/qwen_hotword/hotwords/capacity_benchmark.py`：同一decoded replay回放各级词库，
  输出Raw Recall@1/3/5/10/20、Operating Recall@5、MRR/rank/margin、负例FPR、
  Top-5 churn、matching/sort/select延迟、P50/P90/P95/P99、2秒deadline、RSS、
  Python heap、GPU峰值及累计音频时长分桶；默认retrieval P95超过2秒后停止扩容。
- `score_decoded_hotwords`和`profile_decoded_hotwords`：与现有logits评分完全复用
  同一个matcher和门控逻辑，只把不可变decoded phoneme作为输入，避免每一级重复
  Encoder造成质量与计时噪声。
- CLI：`build_hotword_capacity_assets.py`、`build_hotword_capacity_replay.py`、
  `benchmark_hotword_capacity.py`。
- 测试：`tests/test_hotword_capacity.py`，覆盖logits/replay等价、严格嵌套active
  数量、train-only候选、Representative/Hard两profile、离线质量/性能输出、容量
  headroom建议及sealed-test拒绝。

输出根目录固定建议为：

```text
outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
```

完整工作区命令和判定口径见`docs/HOTWORD_CAPACITY_EVAL.md`。执行顺序必须是：CPU
资产构建、formal100离线replay、Representative离线benchmark、Hard-negative离线
benchmark、20条流式replay smoke、Representative流式benchmark；smoke通过后再做
formal100流式replay。最终上线建议只取Representative，Hard-negative只作压力诊断。

本地不加载真实H200模型或30.9GB缓存。全仓库Ruff通过；全量pytest 166项通过；
5个任务source模块Mypy strict通过；容量/距离定向pytest通过；3个CLI
`--help`和`git diff --check`通过。全仓库Mypy仍是6个既有Torch/transformers模块的
11项错误，本轮5个模块为0。发布前remote-parent检查仍需在收口阶段执行。

H200第一次Representative运行在`python_dynamic_programming`编辑距离后端上，
100词档Raw Recall@5为164/172（95.35%）、Operating Recall@5为140/172
（81.40%）、负例FPR为0；但retrieval P95/P99为2.070/2.515秒，耗时几乎全部在
matching，因而保护机制在100词档停止。`recommended_online_cap=0`只表示该慢速
参考实现不满足工程SLO，不代表CTC质量或产品容量为0。随后新增RapidFuzz等价距离
后端和随机序列一致性测试；下一轮必须写入新目录，并确认报告中的
`edit_distance_backend=rapidfuzz`后再解释扩容曲线。

## 0.18 2026-08-18 v3 Formal100 Operating/Forced Top-5 实测结论与词库上限计划

同一批v3 `formal100`（80正例、20负例、172个期望热词）已经完成
`0.86 / Operating Top-5`和无门控`Forced Top-5`的离线与2秒流式控制实验。
两轮都只读validation资产，没有读取sealed test；Forced结果保存在新的输出目录，
没有覆盖Operating基线。

正式对比如下（Recall均为最终文本的严格完整词/短语exact recall）：

| 推理组 | 策略 | 正确/期望 | Recall | WER | CER | 负样本错误注入 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 离线B | Operating Top-5，threshold 0.86 | 157/172 | 91.2791% | 8.2329% | 3.3610% | 0/20 |
| 离线B | Forced Raw Top-5，无门控 | 158/172 | 91.8605% | 8.1660% | 3.4078% | 20/20 |
| 流式D | Operating Top-5，threshold 0.86 | 160/172 | 93.0233% | 7.9652% | 3.3453% | 0/20 |
| 流式D | Forced Raw Top-5，无门控 | 159/172 | 92.4419% | 8.6345% | 4.0019% | 20/20 |
| 流式E | Oracle | 161/172 | 93.6047% | 8.0991% | 3.3610% | 0/20 |

Forced离线相对Operating只多正确1个热词（+0.5814个百分点），并把所有20条
负样本都注入了错误候选；共注入302个错误候选，最终严格写出1个，但相对baseline
没有新增严格错误热词。Forced流式相对Operating反而少正确1个热词（-0.5814个
百分点），WER增加0.6693个百分点（119个词错误变为129个，增加10个），CER增加
0.6566个百分点。严格热词幻觉率仍为0不代表错误Prompt无害：普通词错误已经明显
增加。

流式失败证据也支持保留门控：Forced使`ctc_never_detected`从5降为0，但
`ctc_detected_too_late_already_fixed`从3增至9，且20条负样本均成为
`wrong_hotword_injected`。因此无门控只是让更多低置信、随累计音频波动的候选进入
Prompt，并没有提高最终流式Recall。正式2秒流式基线继续固定Top-5、threshold 0.86、
maximum edit ratio 0.35、posterior weight 0.25和5-token rollback；Forced目录只作为
诊断对照封存。早期约99%的数字是另一资产上的CTC raw ranking recall，不是Qwen最终
端到端Recall，不能用来替代本次结论。

Forced离线报告位于：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/
  prompt_multi_nested_formal100_forced_top5_v1/multi_nested_prompt_report.json
```

下一阶段拟先用已经训练完成的葡语Temporal 2× CTC Head测单语热词库上限，分成两条
互不混淆的曲线：

1. 检索质量上限：保持同一批音频、CTC checkpoint、CTC decoded序列、Top-5和
   0.86门控不变，只扩大每条case的active词库，主指标为raw Recall@5，同时报告
   Operating Recall@5、目标热词rank/MRR、负样本FPR和跨chunk候选稳定性。
2. 工程性能上限：每2秒对当前累计音频重新检索，分别记录Encoder、CTC Head、
   posterior decode/CPU copy、hotword matching、Top-K排序和Prompt刷新耗时，并记录
   p50/p90/p95/p99、2秒deadline miss、进程RSS、GPU allocated/reserved/peak、词库
   加载时间及每词内存。

词库规模采用确定性嵌套超集，最终固定先测100、500、1k、2k、5k、10k；10k为本轮
压力上限，不测25k/50k/100k。若某一级retrieval p95已经超过2秒或资源达到预先约定
的上限，则记录首个失败点并停止更大规模。扩展词来自train-only葡语真实1至多词
n-gram，不读取sealed test；每个case
必须排除参考文本中存在但未标注为目标的词，防止把真实热词误算成干扰项。除频率分层
的代表性词库外，还要单列长度匹配、近音、包含/被包含关系等hard-negative词库，避免
随机干扰词高估Recall。

现有实现还不能直接产生可信的工程上限报告：`score_hotwords`会对每个active entry
逐一执行Python局部编辑距离窗口扫描，再对全部match排序；当前v3每条case固定只有
100个active hotwords。流式时间线只记录整条样本`inference_seconds`，没有上述分段
耗时或RSS/GPU峰值。下一轮应先增加独立benchmark资产构建器和分段计时器；质量实验
复用一次生成的离线/逐chunk CTC decoded序列，在所有词库规模上replay，避免重复运行
Encoder造成噪声。随后只在100、性能拐点和最大可接受规模上跑真实全链路流式D组确认。

## 0.17 2026-08-18 v3 Raw/Forced Top-5 无门控消融

正式100条 `0.86 / Operating Top-5` 离线与流式评测均已在H200完成并保持
`status=pass`。最终精确热词Recall A/B/C/D/E = 89.53%/91.28%/89.53%/
93.02%/93.60%；流式D比离线B高1.74个百分点，D与流式Oracle E仅差0.58个百分点。
自然100条没有人工确认或强制对齐时间戳，因此其Recall/WER/CER有效，但
`boundary_summary`为空，声学结束相对延迟为null；这不表示CTC没有检出。

为验证早期无阈值Ranking Top-5的记忆口径，新增显式
`--retrieval-mode forced_topk`。该模式不是把threshold简单设为0，而是离线直接读取
`ranking_top5`、流式每个累计音频step直接使用`ranked_matches[:5]`；0.86 score、
0.35 edit ratio、posterior confidence和margin门控均不参与候选选择。`minimum_phonemes=4`
与`posterior_weight=0.25`仍用于生成原始排名。样本仍为同一formal100，旧Operating
目录不覆盖，新报告显式记录`threshold=null`、`guards_applied=false`和候选来源。

必须先生成独立的离线Forced控制目录，再以它作为流式A/B导入和配置校验来源。
工作区命令见`docs/STREAMING_RAG_EVAL.md`“Raw/Forced Top-5消融”。该消融用于观察
Recall收益和错误注入/WER代价，不替代0.86部署基线，也不读取sealed test。

本地实际验证：Forced/流式定向pytest 24项通过；全量pytest通过；本轮文件Ruff
通过；54个source模块全量Mypy通过；两个CLI `--help`和`git diff --check`通过。
全仓库Ruff仍只有9个既有UP038，不在本轮改动范围。

## 0.16 2026-08-17 v3 Top-5 流式正式100条控制实验（H200 已完成）

首次 Top-3 50条流式smoke已在H200完成，A/B/C/D/E均为50条且
`status=pass`。该次使用v2 `stratified_100/retrieved_rag_v1`，最终精确热词
Recall A/B/C/D/E = 95%/96.25%/96.25%/96.25%/96.25%；D相对B的Recall
变化为0，WER/CER分别增加0.252/0.145个百分点。该smoke仅验证流式
链路，不是正式离线→流式结论；原目录不覆盖，作为参考封存。

用户确认正式控制实验应对齐更早的v3多词/组合/嵌套端到端Top-5，
而不是v2 Top-3。v3固定参数为0.86/Top-5/.35/.25/min posterior 0/
minimum phonemes 4/margin 0/Temporal 2×。旧v3端到端只有50条，因此新增
`formal100`选样profile：把原七组配额严格2倍扩展到80正例+20负例，
且原50条必为新100条的确定性子集。

正式流程分两步：先用`run_multi_nested_prompt_eval.py
--selection-profile formal100`在新目录生成这100条的离线A/B/Oracle；
再用`run_streaming_rag_evaluation.py --offline-format multi_nested_v3`读同一
`sample_selection.json`运行C/D/E。流式入口在加载模型前强制校验Top-5
参数、Prompt/语言/dtype/`max_new_tokens`/qwen-asr版本、输入SHA256、
CTC修正报告及checkpoint SHA256，任意不同立即失败。

同时修正两个smoke暴露的统计问题：注入前已正确不再产生负的
`chunks_from_injection_to_first_correct`，而是单独记录
`correct_before_first_injection`；只有人工或强制对齐证明热词位于尾块时才分类
`tail_flush_failure`。完整H200命令和新输出目录见`docs/STREAMING_RAG_EVAL.md`。

本地实际验证：流式/v3定向pytest 23项通过；全量pytest通过；
本轮文件Ruff通过；54个source模块全量Mypy通过；两个CLI `--help`通过；
`git diff --check`通过。全仓库Ruff在本地新版规则下仍报告9个既有UP038，
位于本轮未修改的multi-nested/retrieved/prompt/temporal历史代码，本轮不做
扩张性机械改写。

## 0.15 2026-08-17 流式端到端热词 RAG 评测（代码完成，待 H200 实跑）

本轮暂停西语数据处理，新增独立的 Qwen3-ASR-1.7B 流式端到端热词
评测，不改训练数据、模型权重、CTC checkpoint、Manifest 或既有离线结果。

已对照 Qwen 官方示例和当前上游源码确认：流式仅支持 vLLM，默认每个
2 秒 chunk 累计重放全部已收到音频；前 2 个 chunk 无文本 prefix，从第 3 个
chunk 开始回退最后 5 个 `processor.tokenizer` token；尾音通过
`finish_streaming_transcribe` 不补零处理。官方接口没有流中动态 context
setter，所以本评测把同轮候选注入明确标记为 experimental state refresh：通过
公开 initializer 构造临时 state，只刷新活跃 state 的 Prompt 元数据。安装版本
字段不兼容时立即失败，不会假装同轮生效或静默延迟。

新增：

- `src/qwen_hotword/inference/streaming_core.py`：2 秒调度、尾音 flush、tokenizer
  5-token rollback、fixed/unfixed 记录、因果候选、同轮 Prompt、逐 chunk
  时间线、延迟与失败分类。
- `src/qwen_hotword/inference/streaming_backends.py`：官方 vLLM streaming adapter，以及
  独立 Transformers Qwen Encoder + 封存 Temporal 2× CTC Head 的累计音频检测器。
- `src/qwen_hotword/inference/streaming_rag.py`：统一 A/B/C/D/E、单样本原子分片
  resume、Recall/WER/CER、边界、延迟、稳定性与失败汇总。
- `src/qwen_hotword/inference/streaming_boundary.py`：只接受强制对齐或人工确认
  时间戳，通过运行时前置静音生成不覆盖原音频的 2 秒相位变体。
- `scripts/run_streaming_rag_evaluation.py`
- `scripts/build_streaming_boundary_eval.py`
- `tests/test_streaming_core.py`
- `tests/test_streaming_boundary.py`
- `tests/test_streaming_rag.py`
- `docs/STREAMING_RAG_EVAL.md`

原始端到端集复用离线 `sample_selection.json` 和 A/B 预测；C/D/E 重新做真实
流式推理。D 每一步只读当前累计音频，没有候选 TTL/永久保留；E 的
Oracle 只来自该 case 的 expected IDs，不进入 D。每个 chunk 记录 CTC Top-K/
置信度、实际注入、Prompt 生效 chunk、fixed prefix、回退 token IDs/文本、
unfixed/完整 partial、热词状态和文本 diff。

无强制对齐的原始集不会伪造声学结束时间，因此该集的 Recall/WER/CER
有效，基于声学结束的 latency 保持 `null`。边界资产默认强制覆盖
chunk 中间、边界前、跨边界、边界后、尾音、多词短语、跨多 chunk 长热词、
多热词和负例；缺类别时拒绝标记为完整基线。

本地不加载完整 Qwen/vLLM。已完成 fake backend 和纯 CPU 单测，覆盖空/
短于/等于/长于 2 秒、尾音、前两 chunk 无 prefix、tokenizer 级 5-token
回退、多字节文本、当轮候选、Oracle 隔离、边界分桶、时间线失败归因和
可恢复分片。H200 尚未完成的项目是安装版本 API 实测、50条 smoke、100条
完整原始集和人工/强制对齐边界集。

本地实际验证：

```text
Ruff（全仓库）: pass
Pytest 定向:    pass, 17 tests
Pytest 全仓库: pass, 153 tests
Mypy 新增四个 source 模块: pass
Mypy 全仓库: 仍有 11 个既有 Torch/类型注解错误，新模块 0 个
CLI --help smoke: pass
git diff --check: pass
```

完整工作区命令、输入路径、边界 spec 格式、输出和恢复规则见
`docs/STREAMING_RAG_EVAL.md`。第一轮必须保留 2/2/5 原始基线，报告后再决定是否
测 1 秒 chunk 或扩大 unfixed token 数。

## 0.14 2026-08-17 阿根廷/拉普拉塔西语：原始结构复核与专用转换器

西语新增两套只读来源，数据根目录为：

```text
/host_home/star/q00933266/data/es_ar_sources_v1
```

SLR61解压目录共有5,919个WAV。三个阿根廷索引文件均为无表头两列TSV，不能用
`csv.DictReader`读取，否则每个文件会少算首行。原始索引规模为female 3,921、
male 1,818、`es-ar` weather 90，共5,829行；实测weather的90个`source_id`已经全部
包含在female索引中，因此去重后是5,739条唯一阿根廷语音。库存中的另外180个WAV
分别是这90条`es-ar` weather的重复副本和90条`es-es`天气语音。转换器必须逐一验证
重复weather的文本和WAV内容相同，并明确排除`es-es`。`extracted/line_index.tsv`是
male索引的重复副本，也不得再次纳入。

Common Voice Rioplatense v26包含train 9,903、dev 266、test 224，共10,393条，和
`clips/`下10,393个MP3一致。该语料的地域标签覆盖阿根廷、乌拉圭、巴拉圭和玻利维亚
东部，不能宣称为纯阿根廷西语。必须保留官方train/dev/test和`client_id`，并审计
跨split说话人重叠；官方test后续保持封存，不能重分到训练集。

新增两个专用转换入口：

```text
scripts/convert_slr61_argentinian_to_tsv.py
scripts/convert_common_voice_rioplatense_to_tsv.py
```

两者输出规范TSV时均保留`source_id`、`speaker_id`、`source_split`、`language`、
`dialect`和来源元数据。主语言仍写`es`；方言只作为来源元数据记录为`argentinian`
或`rioplatense`，不把Rioplatense误标为纯`es-AR`。当前只做转换、全量音频/库存及
split审计；审计结果确认前不运行MFA G2P或完整Manifest。

工区拉取后先记录Common Voice归档SHA256，再运行转换：

```bash
ES_AR_DATA_ROOT=/host_home/star/q00933266/data/es_ar_sources_v1
ES_AR_OUTPUT_ROOT=outputs/es_ar_train_sources_v1

sha256sum \
  "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/downloads/es-Rioplatense.tar.gz" \
  > "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/downloads/common_voice_rioplatense_v26_sha256.txt"

python scripts/convert_slr61_argentinian_to_tsv.py \
  --source-root "$ES_AR_DATA_ROOT/slr61_argentinian_spanish" \
  --output-tsv "$ES_AR_OUTPUT_ROOT/slr61/source.tsv" \
  --check-audio \
  --scan-audio-inventory

python scripts/convert_common_voice_rioplatense_to_tsv.py \
  --corpus-root "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/extracted/es-Rioplatense" \
  --output-tsv "$ES_AR_OUTPUT_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --check-audio \
  --scan-audio-inventory
```

先回传`slr61/slr61_conversion_summary.json`和
`common_voice_rioplatense_v26/common_voice_conversion_summary.json`。SLR61应读取
5,829个原始索引行、精确写入5,739条唯一语音，逐一验证并排除90条相同的`es-ar`
weather副本，再排除90条`es-es`，缺失、内容不一致和其他未索引WAV均为0；Common
Voice应精确写入10,393条、三组split规模不变、缺失/重复/未引用MP3为0。若Common
Voice出现跨split说话人重叠，转换器会返回`warn`；先决定是否沿用官方split或重新做
说话人隔离，不能直接继续G2P和Manifest。

首轮SLR61和Rioplatense MFA字典审计发现大量带重音或`ñ`的输入词没有生成发音，
且phone OOV在终端显示为空白组合符。新增`diagnose_spanish_mfa_audit.py`，用于区分
只去acute accent即可映射、必须去全部组合符才可映射和仍不可恢复的缺词，并把不可见
OOV的Unicode码点写入JSON。诊断只读现有字典和审计TSV，不修改或修复字典：

```bash
python scripts/diagnose_spanish_mfa_audit.py \
  --audit-dir outputs/es_ar_train_sources_v1/slr61/mfa_audit_v1 \
  --audit-dir outputs/es_ar_train_sources_v1/common_voice_rioplatense_v26/mfa_audit_v1
```

每个输入目录会生成`spanish_diagnostics.json`。诊断结果确认前不要构建西语Manifest，
也不要全局删除组合波浪号；葡语鼻元音仍依赖这些组合符。

实际诊断确认phone OOV只有`U+0303 COMBINING TILDE`，来自西语G2P的非音位性
鼻化；该符号后续只在西语修复字典内处理。缺词主要来自MFA模型不接受acute accent、
`ñ`和`ü`输入。acute accent可以用去重音代理词重新G2P，但不能把`ñ`直接折叠成
`n`，否则会把`/ɲ/`错误变成`/n/`。已加入小型代理拼写探针：

```bash
conda run --no-capture-output -n aligner mfa g2p \
  --num_pronunciations 1 \
  configs/phonemes/spanish_latin_america_repair_probe.v1.txt \
  models/mfa/g2p/spanish_latin_america_mfa.zip \
  outputs/spanish_latin_america_repair_probe.v1.dict

cat outputs/spanish_latin_america_repair_probe.v1.dict
```

探针用于比较`ny`/`ni`对`ñ`以及`gw`对`gü`的输出；结果确认前不生成正式修复字典。

## 0.13 2026-08-14 美式英语独立处理：MFA复审通过，待完整Manifest

美式英语 Swift JSON 已在工区独立转换并完成全量音频审计，输出目录为
`outputs/en_external_train_sources_v1/swift_us_english`。原始 JSON 及音频路径为：

```text
Swift JSON:
/host_home/z00841352/27A/data/en/json/swift_en_美式英语.json

Audio prefix:
/host_home/z00841352/27A/data/en/untar_files/美式英语
```

转换和音频审计均通过：389,738条记录全部写入，语言字段全部为`English`，
389,738个WAV全部存在，空audio/text、缺失和重复音频均为0。词表提取共得到
2,940,208个word token和46,041个唯一词，数字片段为0。

候选English US MFA模型和第一版生成字典的SHA256为：

```text
english_us_mfa.zip:
9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02

words.txt:
15546c3ab1dc7136a732bd25524e0681c88a4bd2e484915db1d03b4526d4ecfe

swift_us_english_english_us_mfa.v1.dict:
2dd3c3e045eabaeac4666a4a99cf472bf7a4ef816adf35943a072bd2262c23e2
```

MFA为46,041个输入词生成46,039个唯一发音，无额外词和重复发音；只缺`h`
（corpus count 3）和`lx`（count 1）。初次v0.2共享CTC词表审计还发现590个
发音包含同一个OOV符号`ʷ`，corpus-weighted count为12,460。该符号是MFA附在
辅音后的唇化修饰符，例如`kʷ`、`tʷ`和`ɟʷ`；共享词表已有普通`w`，因此phone
归一化现将`ʷ`展开为`w`，得到`k w`、`t w`和`ɟ w`，无需重新运行MFA G2P。

工区拉取本提交后只需重新运行字典审计：

```bash
python scripts/audit_mfa_dictionary.py \
  --words outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/words.txt \
  --word-counts outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/word_counts.tsv \
  --dictionary outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/swift_us_english_english_us_mfa.v1.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/en_external_train_sources_v1/swift_us_english/mfa_audit_v2
```

工区复审确认phone OOV及其corpus-weighted count均归零；词型和token覆盖率分别为
99.995656%和99.999864%。`training_labels_ready`只因为`h`和`lx`两个缺词保持
false。这4个corpus token不得静默删除，完整Manifest阶段应将包含它们的记录写入
`needs_review`。

英语词表另有1,717个含撇号或连字符的唯一词，共17,671个corpus token；这些词在
第一版字典中全部有且只有一个精确发音，缺失和歧义均为0。完整Manifest构建器因此
增加显式`--allow-exact-dictionary-connectors`：启用后不再仅因连接符把已有唯一
精确发音的英语词送入review，但字典缺失、多发音和phone OOV仍按原规则review。
默认关闭，既有葡语和西语策略不变；该策略写入build config和summary，不能与默认
策略的旧shard混用。

英语完整Manifest命令：

```bash
python scripts/build_full_training_manifest.py \
  --tsv outputs/en_external_train_sources_v1/swift_us_english/source.tsv \
  --audio-root /host_home \
  --dictionary outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/swift_us_english_english_us_mfa.v1.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/en_us_swift_full_manifest_v1 \
  --language en-US \
  --dataset swift_us_english \
  --id-prefix swift_us_english_row \
  --split unsplit \
  --allow-exact-dictionary-connectors \
  --shard-size 5000 \
  --workers 16
```

该构建仍不与西语或葡语合并，也不生成正式96/2/2 split；先回传summary、ready/review
记录数与小时数，再做Temporal 2×可恢复审计。

## 0.12 2026-08-13 英西葡混训前置：西语独立处理第一阶段（代码完成，待工作区审计）

英西葡混训不直接拼接原始数据，先分别完成英语和西语的独立TSV、音频、G2P、
字典和Manifest审计。当前西语已知输入只有两份只读Swift JSON：

```text
MLS Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/mls/swift_librispeech_es.json

Common Voice Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/cv/swift_cv_es.json

Candidate G2P model:
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/g2p/spanish_latin_america_mfa.zip
```

尚未生成或确认西语TSV、MFA字典、覆盖率报告、ready/review Manifest。MLS Spanish
通常偏西班牙来源，Common Voice `es` 可能混合多个地区；当前统一只标记为`es`，
不声明`es-AR`或纯拉美西语。Latin America MFA模型先作为候选标签器，其G2P成功率
不能代替方言适配结论。

Swift转换输出绝对容器音频路径。旧TSV审计默认把任何绝对路径判为失败，已新增
显式`--allow-absolute-audio`：只有传入该开关才允许绝对路径，仍会逐条检查文件
存在性；默认相对路径安全策略不变。原Swift JSON始终只读，所有产物进入新的
`outputs/es_external_train_sources_v1`，不覆盖葡语或原始文件。

工作区按顺序运行，先不要并行：

```bash
python scripts/convert_swift_json_to_tsv.py \
  --input /data/h00911716/code/ms-swift/self_test/datalist/es/mls/swift_librispeech_es.json \
  --output-tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --expected-language Spanish \
  --audio-prefix-rewrite /home_92=/host_home \
  --check-audio \
  --progress-every 10000

python scripts/convert_swift_json_to_tsv.py \
  --input /data/h00911716/code/ms-swift/self_test/datalist/es/cv/swift_cv_es.json \
  --output-tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --expected-language Spanish \
  --audio-prefix-rewrite /home_92=/host_home \
  --check-audio \
  --progress-every 10000

python scripts/audit_training_tsv.py \
  --tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --audio-root /host_home \
  --allow-absolute-audio \
  --max-records 0 \
  --sample-count 5 \
  --output outputs/es_external_train_sources_v1/mls/audio_audit_full.json

python scripts/audit_training_tsv.py \
  --tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --audio-root /host_home \
  --allow-absolute-audio \
  --max-records 0 \
  --sample-count 5 \
  --output outputs/es_external_train_sources_v1/common_voice/audio_audit_full.json

python scripts/prepare_mfa_g2p.py \
  --tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --output-dir outputs/es_external_train_sources_v1/mls/mfa_g2p \
  --text-column text \
  --max-records 0 \
  --minimum-word-count 1

python scripts/prepare_mfa_g2p.py \
  --tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --output-dir outputs/es_external_train_sources_v1/common_voice/mfa_g2p \
  --text-column text \
  --max-records 0 \
  --minimum-word-count 1
```

需要返回两套`swift_json_conversion_summary.json`、`audio_audit_full.json`和
`mfa_g2p/summary.json`。只有两套转换均无unexpected language、音频审计缺失为0，
才继续各自运行MFA G2P、字典覆盖/phone OOV审计及完整Manifest；若路径前缀不匹配，
先根据转换报告修正映射，不得跳过音频检查。下一步是西语第二阶段G2P与Manifest，
不是三语合并或训练。

## 0.11 2026-08-04 v3结果复核与50条多关键词 Prompt 端到端验证（代码完成，待工作区运行）

工作区 v3 CTC 专项资产和评分已完成：210条自然 Validation case、7组目标全部满足、
500个热词、80个嵌套 family，primary audio 全部互异，`status=pass`。固定
`threshold=0.86/top_k=5` 的主要实测结果为：总体 Ranking Recall@5 95.37%；
Operating Precision 96.06%、Recall 83.17%、负例FPR 0%；3个独立热词 Recall@5
93.33%、All-3-Hit@5 80%，未达到本轮95%/85%的工程参考目标；组合非嵌套词
Recall@5 96.40%，高于单词91.67%，当前没有“组合词更难”的证据；short-only长词
Forced Top-5误排20%，但Operating误触发0%；总体 slot crowding loss为0，不过有3条
局部case出现family成员占位并挤掉独立真词。

复核发现旧报告有两处仅影响指标汇总、不影响CTC逐case评分的口径问题：

1. Longest-match Operating Precision 错把同family的短词冗余命中计为false positive；
2. 嵌套short/long专项指标使用全局family成员集合，可能把case里的独立词误归入其他family。

已修正为逐case family口径，并新增CPU-only报告重建工具；它复用已有
`hotword_case_scores_v3.jsonl`，不加载Head、不读取特征、不重复推理，也不覆盖原报告。
原报告保留用于审计，修正版另存：

```bash
python scripts/rebuild_multi_nested_hotword_report.py \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_case_scores_v3.jsonl \
  --base-report outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3.json \
  --output outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3_corrected.json
```

新增固定50条 Qwen3-ASR-1.7B Prompt 端到端评估：40正例+10负例，分组固定为
3独立词10、nested family+2独立词10、nested long 8、2独立词6、nested short 3、
单热词3、负例10。模型只加载一次，依次运行全部Baseline、仅有Operating候选时的
Retrieved Prompt、40条正例Oracle Prompt；无候选case直接复用Baseline，不重复推理。
最终目标采用Longest-match，contained short单独统计为redundant family hit，不当作
错误候选；真正无关候选的注入、写入和新增幻觉另行统计。CTC配置必须仍为固定
0.86/Top-5/.35/.25，不允许在本命令内调参；不读取sealed test。

物理GPU 5运行：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_multi_nested_prompt_eval.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --ctc-case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_case_scores_v3.jsonl \
  --ctc-report outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3_corrected.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/prompt_multi_nested_v1 \
  --device cuda:0 \
  --dtype bfloat16
```

新增/修改文件：

- `src/qwen_hotword/hotwords/multi_nested.py`
- `src/qwen_hotword/inference/multi_nested_prompt.py`
- `scripts/rebuild_multi_nested_hotword_report.py`
- `scripts/run_multi_nested_prompt_eval.py`
- `tests/test_multi_nested_hotwords.py`
- `tests/test_multi_nested_prompt.py`

本地实际验证：Ruff全仓库pass；Pytest全仓库128 tests pass；新模块Mypy pass；
两个CLI `--help` smoke pass；`git diff --check` pass。fake/mock覆盖固定分组选择、
audio-disjoint、模型单次加载、Baseline/Retrieved/Oracle、嵌套冗余与真正错误分离、
原子输出及防覆盖。

预计Prompt输出：`sample_selection.json`、`baseline_predictions.jsonl`、
`retrieved_predictions.jsonl`、`oracle_predictions.jsonl`、
`multi_nested_prompt_report.json`。需要返回修正后的CTC报告和Prompt报告；若报告发现
具体幻觉或异常提升，再按其中case ID返回对应prediction行。当前限制：这是固定50条
Validation小样本，不是生产评估；没有工作区Prompt真实结果前不得宣称Prompt有效。
下一步只检查50条结果，再决定是否进入threshold/top-k调优或扩大正式评估。

## 0.10 2026-08-04 多关键词与组合/嵌套关键词专项评估（代码完成，待工作区运行）

本轮只实现 Validation CTC 专项评估，不训练模型、不提取 Encoder 特征、不读取
sealed test，也不运行 Qwen Prompt 推理。复用固定资产：

```text
Validation manifest: outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl
Validation cache:    outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation
Temporal 2× Head:    outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt
MFA dictionary:      outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
```

新增独立 v3 资产构建器，严格从自然 Validation 文本选择约210条 audio-disjoint
case：1/2/3个独立关键词、nested short-only、long-present、nested family+2个独立词
及负例。组合词必须是连续完整词组；独立词 span 不重叠、无包含关系；嵌套 family
同时保存 containment 和 longest-match Ground Truth。每条 case 固定100个 active
hotwords，并记录规范化文本、真实 word span、family、困难负例和选择理由。输出目录
非空时拒绝覆盖；关键嵌套组少于10条时保留最大自然子集并标记
`smoke_insufficient_data`，不伪造样本或降低标准。无 speaker ID，因此只声明
audio-disjoint。

评分只复用现有 Validation cache 和 Temporal 2× Head，固定参数不可搜索：

```text
top_k=5, threshold=0.86, maximum_edit_ratio=0.35
posterior_weight=0.25, minimum_posterior_confidence=0.0
minimum_phonemes=4, minimum_top1_margin=0.0
time_axis=temporal_upsample_2x_only
```

报告严格区分 Forced Ranking Top-5 与 threshold/edit/posterior guard 后最多Top-5的
Operating 结果。包含总体/分组 Micro Recall@1/3/5、Any/All-Hit、All-3-Hit、Mean
Hits、Raw Precision@5、Operating P/R/F1、正例命中率和负例FPR；按 hotword form、
音素长度及 form×length 分桶；嵌套专项包含 short-only长词误触发、双GT、family
槽位、redundant hit、其他独立词Recall、slot crowding loss及具体归因case。报告明确
说明3个真实词而固定返回5个候选时 Raw Precision@5 的理论上限60%只是计算口径，
不是模型准确率上限。

本轮文件：

- `src/qwen_hotword/hotwords/multi_nested.py`
- `scripts/build_multi_nested_hotword_eval.py`
- `scripts/evaluate_multi_nested_hotwords.py`
- `tests/test_multi_nested_hotwords.py`
- `docs/HANDOFF.md`

本地实际验证：

```text
Ruff（全仓库）: pass
Pytest（全仓库）: pass, 126 tests
Mypy（multi_nested新模块）: pass
两个CLI --help smoke: pass
git diff --check: pass
```

工作区先运行CPU资产构建：

```bash
python scripts/build_multi_nested_hotword_eval.py \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested
```

先检查 `asset_summary_v3.json` 的实际分组数；无论 formal 或 insufficient_data，均可
继续运行固定口径 GPU 评分以获得 smoke 数据。物理 GPU 5 暴露为逻辑 cuda:0：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_multi_nested_hotwords.py \
  --validation-cache outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --asset-summary outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/asset_summary_v3.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested \
  --device cuda:0 \
  --batch-size 128
```

预计生成：

```text
multi_nested_hotwords_v3.jsonl
hotword_families_v3.jsonl
multi_nested_cases_v3.jsonl
sample_selection_v3.json
asset_summary_v3.json
hotword_case_scores_v3.jsonl
multi_nested_evaluation_report_v3.json
```

需要返回 `asset_summary_v3.json` 和 `multi_nested_evaluation_report_v3.json`；若要
逐case核查 crowding 再返回 `hotword_case_scores_v3.jsonl`。当前限制是本机没有真实
Validation/cache/checkpoint，尚未产生或宣称任何真实效果结论。下一步只根据CTC专项
结果决定是否运行50条多热词/组合词 Prompt 端到端验证。

## 0.9 2026-08-04 Temporal 2× 五语料合并训练集（工作区已完成）

Temporal 2× 只读审计已在工作区完成，5套语料合计结果：

```text
原 ready:              380,289 / 588.552262 h
纯时间恢复 ratio<=0.9: 166,757 / 227.142632 h
合并候选总计:          547,046 / 815.694894 h
```

工作区构建已完成并通过：train 525,189条/783.223637小时，validation
10,899条/16.239131小时，sealed test 10,958条/16.232128小时；全局重复ID、重复
音频和跨split overlap均为0，`source_manifests_modified=false`、`test_set_used=false`、
`status=pass`。后续不再把该合并manifest写成“待构建”。

用户决定不再只释放 Noah 500h，而是把以下5套语料的原 ready 与安全时间恢复集
一次性组成新的独立训练数据版本，按已有稳定 `split_hash` 做96/2/2：

```text
Noah 金融 200h
Noah 原 500h
MLS Portuguese
Common Voice Portuguese
FLEURS Portuguese
```

新构建器严格纳入两类记录：

1. 原 `train_ready.jsonl` 的全部 ready 记录；
2. `needs_review.jsonl` 中 issue 集合恰好只有 `ctc_length_infeasible`、Temporal
   2× 后可行且 effective ratio `<=0.90` 的记录。

任何 dictionary/connector/digit/standalone-h/empty-target 等其他问题仍然阻塞；
`(0.90,1.00]` 高压力样本和2×后仍不可行样本不释放。构建器只读源文件，输出
目录非空时拒绝覆盖，写入临时文件后原子替换；全局拒绝重复ID和重复绝对音频
路径，并检查跨split ID/音频重叠为0。输出保留 `source_corpus`、原始 language
和 `release_source`，MLS/CV 的 `pt` 不会被伪改为 `pt-BR`。

所有新记录明确写入：

```text
dataset_version: temporal2x-combined-v1
ctc_time_upsampling_factor: 2
estimated_ctc_input_length: 原 Encoder CTC 帧数
effective_ctc_input_length: 原帧数 * 2
```

feature-cache/训练边界已同步支持这一合约：恢复样本在缓存校验时按2×有效时间轴
判断可行；新缓存记录 time factor，训练时若使用小于数据要求的 Head factor 会
拒绝启动。旧 manifest 未写该字段时默认1，既有旧缓存元数据缺字段也按1兼容。

本轮文件：

- `src/qwen_hotword/training/combined_training.py`
- `scripts/build_temporal2x_combined_training.py`
- `tests/test_combined_training.py`
- `src/qwen_hotword/training/ctc_overfit.py`
- `src/qwen_hotword/training/feature_cache.py`
- `src/qwen_hotword/training/sharded_ctc.py`
- `tests/test_ctc_overfit.py`
- `tests/test_feature_cache.py`
- `docs/data.md`

本地验证：

```text
Ruff（全仓库）: pass
Pytest 定向（combined/feature/loader/sharded CTC）: pass, 24 tests
Pytest 全仓库: pass
Mypy（combined_training 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

工作区不需要GPU，从项目根目录运行：

```bash
python scripts/build_temporal2x_combined_training.py \
  --corpus noah_finance_200h=outputs/noah_pt_finance_200h/full_manifest_v1 \
  --corpus noah_500h=outputs/noah_pt_full_500h \
  --corpus mls=outputs/pt_external_train_sources_v1/mls/full_manifest_v1 \
  --corpus common_voice=outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard \
  --corpus fleurs=outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard \
  --output-dir outputs/pt_combined_temporal2x_v1 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --progress-every 50000
```

预计输出：

```text
outputs/pt_combined_temporal2x_v1/full_ctc_train.jsonl
outputs/pt_combined_temporal2x_v1/full_ctc_validation.jsonl
outputs/pt_combined_temporal2x_v1/full_ctc_test.jsonl
outputs/pt_combined_temporal2x_v1/split_config.json
outputs/pt_combined_temporal2x_v1/split_summary.json
```

首先返回 `split_summary.json`。通过标准：总数547,046、原ready 380,289、恢复
166,757、总时长约815.694894小时、三split均非空、duplicate和cross-split overlap
全部为0、`source_manifests_modified=false`、`test_set_used=false`、status pass。
test输出生成后立即封存，不参与特征缓存、选模或调参。

当前限制：这是按样本自然比例的首个全集版本，尚未做corpus sampling weight；
MLS仍保留Portuguese来源身份，不能宣称为纯巴葡。下一步只在summary通过后缓存
新train/validation的Encoder特征，test不读取。

## 0.8 2026-08-03 Temporal 2× 训练语料恢复审计（工作区已完成）

用户决定先重新审计旧 full manifest 的时间筛选，再决定第一批释放量。本轮只读，
不生成训练 manifest、不修改原 ready/review、不缓存特征、不训练模型。审计顺序：

```text
Noah 金融 200 小时
Noah 原 500 小时
MLS
Common Voice
FLEURS
```

Temporal Head 的实际 `output_lengths` 已确认严格为原 Encoder CTC 长度乘 2。
本轮 effective ratio 定义为：

```text
ctc_minimum_input_length / (estimated_ctc_input_length * 2)
```

为保护 Noah 500 小时已经封存的 test，工具不逐行读取任何
`train_ready.jsonl`；原 ready 记录数/小时只从原 `summary.json` 获取。记录级
扫描只读取从未进入正式 train/validation/test 切分的 `needs_review.jsonl`，报告
显式记录 `ready_manifest_content_read=false`、`sealed_test_content_read=false`。

Review 分类严格互斥：

1. issues 恰好只有 `ctc_length_infeasible`：纯时间问题；
2. 纯时间问题在 2× 后拆为可恢复与仍不可行；
3. 可恢复再拆为 effective ratio `<=0.90` 的首批建议集，以及 `(0.90,1.00]`
   的高压力延后集；
4. 只要包含任何其他 issue，即使 2× 时间可行也归入“其他问题阻塞”，不得进入
   第一批恢复。

每套 corpus 报告包含原 ready/review 记录数与小时、2× 总可恢复、首批建议、
高压力延后、仍不可行、其他 issue 阻塞、effective ratio 分桶、每种 issue
总量、精确 issue 组合和两两交集。所有分类同时记录数量、已知小时和缺 duration
数量；输入 summary/review 记录 SHA256，ready 只记路径和大小、不读取或计算
SHA256。输出目录非空时拒绝覆盖。

首次工作区运行在 Noah 金融 review 第 3,507 行发现合法的“只有
`estimated_ctc_input_length`、没有 `ctc_minimum_input_length`”记录。这是
`empty_ctc_target`/标签组装失败一类记录的正常 partial metadata，不是数据损坏。
审计器已修正：非时间问题的 partial length 计入 ratio unavailable 并继续保持
阻塞；纯 `ctc_length_infeasible` 候选仍强制要求两个长度字段完整。

本轮代码：

- `src/qwen_hotword/training/temporal_recovery.py`
- `scripts/audit_temporal2x_recovery.py`
- `tests/test_temporal_recovery.py`

本地验证：

```text
Ruff（全仓库）: pass
Pytest 定向: pass
Pytest 全仓库: pass
Mypy（temporal_recovery）: pass
CLI --help smoke: pass
git diff --check: pass
```

工作区无需 GPU，按用户指定优先级运行：

```bash
python scripts/audit_temporal2x_recovery.py \
  --corpus noah_finance_200h=outputs/noah_pt_finance_200h/full_manifest_v1 \
  --corpus noah_500h=outputs/noah_pt_full_500h \
  --corpus mls=outputs/pt_external_train_sources_v1/mls/full_manifest_v1 \
  --corpus common_voice=outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard \
  --corpus fleurs=outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard \
  --output-dir outputs/temporal2x_recovery_audit_v1 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --progress-every 50000
```

预计输出：

```text
outputs/temporal2x_recovery_audit_v1/summary.json
outputs/temporal2x_recovery_audit_v1/noah_finance_200h.json
outputs/temporal2x_recovery_audit_v1/noah_500h.json
outputs/temporal2x_recovery_audit_v1/mls.json
outputs/temporal2x_recovery_audit_v1/common_voice.json
outputs/temporal2x_recovery_audit_v1/fleurs.json
```

审计结束后先检查六个小 JSON，再决定第一批释放量。倾向方案保持为“旧训练集 +
纯时间问题、2× 后 effective ratio <=0.90 的恢复集”；本轮不创建该合并版本。

## 0.7 2026-07-31 Retrieved RAG 端到端验证（代码完成，待工作区运行）

本轮把已经完成的两段链路真正接起来：

```text
预生成 validation CTC case scores
  -> 固定 threshold=0.86 / top-k=3 / margin=0
  -> 热词 Prompt
  -> Qwen3-ASR-1.7B 最终转写
  -> Baseline / Retrieved / Oracle 归因
```

只做流程 smoke，不搜索 threshold/top-k，不训练 Encoder/CTC Head，不读 sealed
test，不修改 Qwen 模型。CTC 候选除了 `score >= 0.86`，继续使用正式评分阶段的
`edit_ratio <= 0.35` 与 `minimum_posterior_confidence=0`；不能把阈值简化为只
过滤 score。

现有 500 条 score 文件按本轮固定 `top-k=3` 复算为 Precision 93.47%、
Recall 88.48%、负例 case FPR 2.8%。此前记录的 93.29% / 89.89% 是
`top-k=5` 口径；两者没有冲突，本轮只是先固定 top-k=3 跑通流程。

上一轮 Prompt smoke 已在工作区完成。40 条 validation case 的实际结果为：

```text
Baseline: 44/48, hotword recall 91.67%, positive case hit 30/30
Oracle:   44/48, hotword recall 91.67%, absolute gain 0
Negative Prompt: 0/10 错误热词写入，幻觉率 0
Model load count: 1
```

这证明 `Qwen3ASRModel.transcribe(..., context=prompt)` 接口、固定葡语模板和安全
控制可以运行，但未证明 Oracle Prompt 在当前常见词样本上有收益。Oracle 仍漏
`pra vocês`（2 次）、`pode pausar`、`então vamo`。因此本轮完成标准是链路与
归因正确，不以显著 Recall 提升作为通过条件。

### 实现

- `src/qwen_hotword/inference/retrieved_rag.py`
  - 严格校验 validation manifest、v2 case、hotword table 与 CTC score 一致；
  - 确定性选择 60 正例、40 负例；正例覆盖短/中/长热词和单/多热词；
  - 负例优先纳入全部 threshold 触发 case，再用固定 seed 补足，专门观察错误
    候选注入后的最终转写污染；
  - Baseline 跑全部 100 条；有候选时才跑 Retrieved Prompt，无候选直接复用
    Baseline；60 条正例另跑 Oracle；
  - 模型只加载一次，逐次打印 phase、累计调用、耗时、速度和 ETA；
  - 记录 CTC 检索 Precision/Recall/FPR、最终热词 Recall/case hit、检索漏召回、
    检索正确但 Decoder 未写出、相对 Baseline 的热词救回、错误候选写入、
    文本变化和简单 corpus WER；
  - 五个输出统一原子写入，非空目录拒绝覆盖，记录所有输入 SHA256。
- `scripts/run_retrieved_rag.py`
- `tests/test_retrieved_rag.py`

本地 fake inference 覆盖阈值与 edit guard、margin、确定性分层选样、CTC
误触发负例、模型单次加载、无 Prompt Baseline 复用、多热词 Prompt、三路
Recall、错误候选写入归因、WER、进度和防覆盖。

本地实际验证：

```text
Ruff（全仓库）: pass
Ruff format（本轮文件）: pass
Pytest 定向: pass
Pytest 全仓库: pass
Mypy（retrieved_rag 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

### 工作区运行

先拉取交付分支，然后从项目根目录在物理 GPU 5 运行：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_retrieved_rag.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotwords_v2.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotword_cases_v2.jsonl \
  --ctc-case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/scoring_temporal2x_v2/hotword_case_scores.jsonl \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1 \
  --threshold 0.86 \
  --top-k 3 \
  --minimum-top1-margin 0 \
  --device cuda:0
```

输出：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/sample_selection.json
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/baseline_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/retrieved_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/oracle_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/retrieved_rag_report.json
```

需要优先返回 `retrieved_rag_report.json`；若要逐 case 归因，再返回另外四个小
文件。当前限制：v2 是 validation 模拟常见词，Baseline 很高；负例选样故意
富集 CTC 误触发，所以选中 40 条负例内部的误触发率不能当作无偏 FPR，报告另
保留全 500 case 的正式 FPR。CTC score 是上一阶段离线生成，本轮尚未实现在线
registry/reload。

下一步只检查工作区端到端结果；threshold/top-k 调优留到后续独立实验。

## 0.6 2026-07-30 三套葡语 Swift JSON（第一版独立处理完成）

Noah 金融 200 小时第一版 full manifest 已完成：

```text
Source:          142,985 / 195.3711 h
Training-ready:   86,614 / 119.5436 h
Needs review:     56,371
Status:           pass

ctc_length_infeasible: 53,699
dictionary_missing:     3,390
unresolved_connector:   3,377
standalone_h:               10
empty_ctc_target:            8
```

MFA corpus-token coverage 为 99.8380%，v0.2 phone OOV 为 0。缺词主要与连字符
重合。金融数据先保留这版结果，未来用 2× 时间轴与连接词解析恢复，不阻塞三套新
语料。

FLEURS、MLS、Common Voice 葡语 Swift JSON 已分别完成转换、词表、巴葡 MFA
候选 G2P、v0.2 audit 和完整 manifest。三者独立保存、全部作为 train 候选；
没有修改源 JSON/音频，没有建立 validation/test，也没有合并或缓存 Encoder
特征。

本轮更新 Swift JSON 转换器：

- 支持重复传入 `--audio-prefix-rewrite OLD=NEW`；
- 只匹配完整路径前缀边界；
- 在改写后执行 `--check-audio`；
- summary 记录 rewrite 配置、改写数量、缺失音频和耗时；
- 大文件加载前后打印状态，每 10,000 条打印转换速度和累计结果；
- 检测到语言不一致或缺失音频时 summary 标记 `warn`。

工作区转换与词表实际结果：

```text
Corpus          Records   Audio   Word tokens   Unique words   Digit fragments
FLEURS            2,793   WAV          60,947          7,743               847
MLS              37,533   FLAC      1,261,190         75,392                 0
Common Voice     22,923   MP3         154,407         26,593               845
```

三套均 `status=pass`，全部 63,249 条音频在路径改写后存在，无跳过或语言异常。
FLEURS/Common Voice 中的数字不能从 CTC 标签中静默丢失，因此 full manifest
新增 `unresolved_digit` review 原因；暂不自动决定年份、金额或序数的葡语读法。
该保护只影响后续新 manifest，不改动原始 JSON、TSV 或已有 Noah v1 输出。

MFA audit：

```text
Corpus          Token coverage   Missing words   Duplicate entries   Phone OOV
FLEURS                99.4192%             124                   1           0
MLS                    98.8174%           2,294                 305           0
Common Voice           99.2889%             802                  12           0
```

最终 full manifest：

```text
Corpus          Source records/h       Ready records/h       Review
FLEURS          2,793 / 10.1789 h       1,966 / 6.8475 h         827
MLS            37,533 / 160.9632 h     26,030 / 110.3132 h     11,503
Common Voice   22,923 / 26.4790 h      21,803 / 24.9761 h       1,120
Total          63,249 / 197.6211 h     49,799 / 142.1368 h     13,450
```

正式使用路径：

```text
FLEURS:
outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard

MLS:
outputs/pt_external_train_sources_v1/mls/full_manifest_v1

Common Voice:
outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard
```

FLEURS 的旧 `full_manifest_v1` 未启用数字保护，只保留历史对照，后续不得用于
训练。Common Voice v1 的 ready 数量虽与 v2 相同，正式引用也固定为 v2。

主要遗留问题：

1. FLEURS 的 839 个、Common Voice 的 350 个 `unresolved_digit` issue 在有
   上下文安全的数字读法规则前保留 review。
2. MLS 有 14,698 个 connector issue，导致大量记录受影响；旧拼写和复合词修复
   需新建恢复版本，不能覆盖 v1。
3. MLS 使用巴葡 MFA 仅证明 phone/vocab 技术兼容，不能证明所有说话人是 pt-BR；
   合并到巴葡训练前仍需元数据或跨说话人音频抽查。
4. 三套与 Noah 数据尚未做跨语料音频/文本去重，也未决定训练采样权重。

下一步只在用户决定合并策略后，构建 train-only 合并 manifest；在此之前不要
缓存特征或启动新 CTC 训练。

## 0.5 2026-07-29 Noah 200 小时巴葡金融数据（代码就绪，待首轮审计）

用户确认新增的 200 小时金融领域数据是巴西葡萄牙语，可复用 Noah 500 小时的
巴葡 MFA 和完整 CTC manifest 流程。该数据先独立处理、全部作为 train 候选，
暂不与旧 500 小时数据合并，也不建立新的 validation/test。

用户给出的宿主机 TSV 为：

```text
/home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv
```

容器内候选路径按现有挂载规则暂定为：

```text
TSV:
/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv

Audio root candidate:
/host_home/z00841352/27A/data/Noah_espt/noah_pt
```

现有 full manifest builder 原先写死
`dataset=noah_pt_full_500h`、`id=noah_pt_row_*` 和 `split=unsplit`，直接复用会
导致新旧语料身份错误和 ID 冲突。本轮已给
`scripts/build_full_training_manifest.py` 和
`src/qwen_hotword/training/full_manifest.py` 增加：

```text
--dataset
--id-prefix
--split
```

旧默认值和旧 500 小时 `build_config.json` 的 resume 兼容性保持不变。新数据
固定使用：

```text
dataset:   noah_pt_finance_200h
id prefix: noah_pt_finance_200h_row
language:  pt-BR
split:     train
```

第一步只运行 1,000 行只读审计，不直接开始 MFA 长任务：

```bash
python scripts/audit_training_tsv.py \
  --tsv "/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv" \
  --audio-root /host_home/z00841352/27A/data/Noah_espt/noah_pt \
  --max-records 1000 \
  --sample-count 5 \
  --output outputs/noah_pt_finance_200h/audit_first_1000.json
```

需要返回：

```text
outputs/noah_pt_finance_200h/audit_first_1000.json
```

通过标准：字段存在、1,000 行均有 audio/text、音频解析 1,000/1,000、缺失为
0、绝对 audio 值为 0。若失败，先根据 report 中样本纠正 audio root，不启动
G2P。

本地定向验证：

```text
Ruff: pass
Pytest（full manifest + G2P prep + MFA audit）: pass, 9 tests
CLI --help smoke: pass
git diff --check: pass
```

首轮审计通过后的下一步：全量审计 → 独立 word list → 巴葡 MFA G2P →
dictionary/vocab audit → 独立 full manifest。具体数据边界同步记录在
`docs/data.md`。

## 0.4 2026-07-29 Prompt Injection 最小验证（代码完成，待工作区运行）

本轮目标是关键词 RAG 的第一步，仅在 formal validation 上比较：

```text
Baseline:                40 条音频，不注入热词
Oracle Prompt:           30 条正例，注入该音频真实包含的热词
Negative Prompt Control: 10 条负例，各注入 1 个严格不在参考文本中的热词
```

不读取 CTC `hotword_case_scores`，不接 Retrieved RAG，不使用 sealed test，不训练
Encoder/CTC Head，也不修改 Qwen 模型结构。固定 seed `20260729` 确定性选样；
正例按 4–7、8–12、13+ 音素三档选择，并混合单热词和多热词 case。最终选择完整
写入 `sample_selection.json`。

### 已确认的官方 Prompt 接口

工作区固定目标仍是模型 `Qwen3-ASR-1.7B`；`qwen-asr==0.0.6` 是 Python
推理库版本，不是模型大小。已直接检查该库的真实接口：

```python
Qwen3ASRModel.transcribe(
    audio,
    context="",
    language=None,
    return_time_stamps=False,
)
```

实际参数名为 `context`。官方 `_build_messages` 将 `context` 放入 system
message，将音频放入 user message。本轮通过现有 `load_asr_model` 只加载一次
模型，固定 `language="Portuguese"`、`return_time_stamps=False`，不额外覆盖
beam、sampling 或其他 `generate` 参数；运行报告会记录 wrapper backend、
`max_new_tokens` 和 `max_inference_batch_size` 的实际值。

固定且唯一的葡萄牙语模板为：

```text
As palavras a seguir podem aparecer no áudio e servem apenas como referência de grafia. Use-as somente se forem realmente faladas; não as inclua à força na transcrição: {hotwords}
```

Baseline 传空 `context`。Oracle 和 Negative Control 均使用同一模板；热词只是
拼写参考，不描述为必须输出。

### 实现文件

- `src/qwen_hotword/inference/hotword_prompt.py`
  - NFKC、casefold、去标点、空格规范化和严格完整单词/连续词组匹配。
  - 不扩展单复数、口语 alias；`coisa` 不匹配 `coisas`，
    `relacionamento` 不匹配 `relacionamentos`。
- `src/qwen_hotword/inference/prompt_smoke.py`
  - validation-only 校验、确定性分层选样、三路推理、指标计算、逐条进度和
    原子输出；非空输出目录拒绝覆盖。
  - Baseline 只计算一次并复用于 Oracle/Negative 对照。
  - 记录模型与四个输入文件的路径、大小和 SHA256。
- `scripts/run_hotword_prompt_smoke.py`
- `tests/test_hotword_prompt_smoke.py`

### 本地实际验证

本地测试使用 fake inference，不加载 1.7B 权重：

```text
Ruff（全仓库）: pass
Ruff format（本轮文件）: pass
Pytest 定向: pass, 7 tests
Pytest 全仓库: pass, 108 tests
Mypy（两个 inference 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

覆盖 Baseline 空 Prompt、Oracle 单/多热词 Prompt、负例错误 Prompt、空热词、
模型只加载一次、严格词匹配、Recall/幻觉率、确定性选样、进度开关结果一致、
防覆盖、非 validation split 拒绝和 sealed test 拒绝。

### 工作区运行命令

从项目根目录在物理 GPU 5 运行；它在进程内映射为逻辑 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_hotword_prompt_smoke.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotwords_v2.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotword_cases_v2.jsonl \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1 \
  --device cuda:0
```

每完成一次推理会打印 phase、累计完成数、耗时、cases/s 和 ETA。总计 80 次：
40 次 Baseline、30 次 Oracle、10 次 Negative Control。

需要返回并检查以下五个小文件：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/sample_selection.json
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/baseline_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/oracle_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/negative_prompt_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/prompt_smoke_report.json
```

当前限制：尚无工作区真实推理结果，因此不能宣称 Prompt 有效；这是 40 条
validation Oracle smoke，不是业务验收、Retrieved RAG 或完整误触发评估。

下一步：接入Retrieved RAG小规模评估。

## 0.3 2026-07-29 热词评分阶段收尾（当前状态）

v2 已在工作区完成：100 个热词、250 个正例、250 个负例，每条 case 激活完整
100 词 registry。原始排序结果为 Recall@1/3/5 =
67.98% / 96.63% / 99.44%，Top-1 正例 case 命中率 96.8%。

完整 `hotword_case_scores.jsonl` 复算确认，旧报告在 threshold=0.90 时的
Recall=43.26% 主要不是 Head 或阈值问题，而是
`minimum_top1_margin=0.03` 会在 Top-2 接近时清空整条多热词 case：

```text
旧策略（margin=0.03）: Precision 96.25%, Recall 43.26%, negative FPR 2.4%
关闭 margin，阈值 0.90: Precision 95.41%, Recall 87.64%, negative FPR 2.4%
关闭 margin，阈值 0.86: Precision 93.29%, Recall 89.89%, negative FPR 2.8%
```

margin 共误杀 75 个正例 case、158 个已经过阈值的正确热词；67/75 case 的
Top-1 和 Top-2 都是正确热词，且 margin 没有减少任何负例 case 误触发。因此
多热词默认 margin 已改为 0.0，仍保留 CLI 参数供单标签实验显式启用。默认
threshold sweep 补入 0.86。

评分现会打印 Head 加载、每个 feature shard 的累计 case、耗时、cases/s、ETA
以及输出路径，并在报告中记录 scoring/evaluation wall seconds。

6 个严格词面负例触发均有明确文本来源：`coisa` 匹配 `coisas` 3 次、
`relacionamento` 匹配 `relacionamentos` 2 次、`vamos` 匹配口语 `vamo`
1 次。它们主要是局部音素子串与数据标签口径不一致，并非随机声学误触发。
Top-5 仅漏 `design` 和 `ruim é` 各 1 次。

遗留问题（进入下一阶段时保留）：

1. 正式业务热词、人名/品牌、speaker-disjoint 与困难负例尚未提供；v2 只用于
   validation 开发，不是最终业务验收集。
2. 单复数和口语变体应通过 registry 显式 alias/pronunciation 管理，不能默认
   对所有品牌、人名开放子串匹配。
3. 0.86/0.90 都只是 validation 候选工作点，正式阈值需在未来业务集确认；已
   消耗的 sealed CTC test 不得用于热词调参。
4. 下一阶段按项目路线进入在线 hotword registry/reload 与 Qwen prompt
   injection，再评估最终转写热词命中率、普通词退化和部署延迟。

本轮代码：

- `src/qwen_hotword/hotwords/scoring.py`
- `src/qwen_hotword/hotwords/evaluation.py`
- `scripts/evaluate_hotword_scoring.py`
- `tests/test_hotword_scoring.py`

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy（scoring/evaluation，skip imports）: pass
Pytest 定向: pass, 10 tests
Pytest 全仓库: pass, 101 tests
CLI --help smoke: pass
git diff --check: pass
```

## 0.2 2026-07-27 v2 分层模拟热词与 Recall@K（已完成）

用户确认 v1 的 50 个 validation 模拟热词只完成了链路 smoke test，不能作为
正式热词评估集。v1 的主要偏差是全部热词仅出现一次、音素长度为 14–24，
缺少短词、中等长度词和较高频词。

本轮新增独立 v2，不覆盖 v1：

- `scripts/build_stratified_hotwords.py`
- `build_stratified_hotword_assets` in
  `src/qwen_hotword/hotwords/simulation.py`
- `evaluate_hotword_ranking` and length-bucket Recall@K in
  `src/qwen_hotword/hotwords/evaluation.py`
- `--ranking-ks` in `scripts/evaluate_hotword_scoring.py`

v2 固定构造 100 个热词：

```text
4–7 phonemes:   30
8–12 phonemes:  40
13–18 phonemes: 20
19–24 phonemes: 10
```

每个长度桶混合 occurrence=1、occurrence=2–5 和 occurrence>=6 的候选；不足时
从同长度桶其他频率候选补齐。可通过 `--exclude-hotwords` 排除 v1 的词面和完全
相同发音。v2 输出目录必须为空，任何已有结果都拒绝覆盖。

case 构造会先做 coverage selection，保证 100 个热词都至少进入一个正例 case；
默认生成 500 个 case。每条 case 激活完整 100 词 registry，因此 Recall@1、
Recall@3 和 Recall@5 是在全部 100 个候选上的真实排序指标，不是从较小随机
候选集计算。报告同时输出整体 Recall@K 和四个音素长度桶的 Recall@K。该排名
指标不加 score threshold；原 threshold sweep、Precision 和负例 FPR 报告继续
保留并与排序能力分开解释。

当前数据仍不能保证人名/品牌类别或 speaker-disjoint，因为 formal validation
manifest 没有实体类别和 speaker ID。v2 是普通难度的代表性开发集，不是最终
业务验收集；仍不读取已经消耗的 formal CTC test 集。

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy（本轮两个模块，skip imports）: pass
Pytest 定向: pass, 8 tests
Pytest 全仓库: pass, 99 tests
CLI --help smoke: pass
git diff --check: pass
```

工作区下一步：

1. 在新目录 `simulated_hotword_eval_v2_stratified_100` 构建 v2。
2. 人工查看 100 词表和 summary 的长度/频率分布。
3. 用固定 2x temporal best Head 评估全部 500 个 validation case。
4. 返回 `stratified_hotword_summary_v2.json` 和
   `hotword_scoring_report.json`，重点读取整体及分长度 Recall@1/3/5。

## 0.1 2026-07-25 sealed test PER 一次性评估（已完成）

用户已明确要求获取当前固定 CTC 模型的正式 test PER，因此允许首次打开此前
封存的 `full_ctc_test.jsonl`。此次评估必须保持以下冻结条件：

```text
checkpoint:
  outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/
  ctc_head_best.pt

Head:
  temporal_upsample, hidden=512, kernel=5, dropout=0.1, time axis=2x

decode:
  greedy argmax CTC collapse, blank_id=0
```

不得根据 test PER 重新选择 checkpoint、调整 Head、修改解码策略或调参。后续若有
新模型版本，必须建立新的正式评估版本和新的独立测试协议，不能反复使用本次结果
进行开发。

本轮新增：

- `src/qwen_hotword/training/sealed_test.py`
  - 直接对 test 音频分块提取冻结的 `ln_post` 特征并立即评估。
  - 不写 test feature cache，不保留可用于反复调参的测试特征。
  - 只接受文件名为 `ctc_head_best.pt` 的 2x temporal checkpoint。
  - 记录 test PER、loss、substitution/deletion/insertion、预测/参考长度比、
    blank ratio、高频错误以及 manifest/vocab/model/checkpoint SHA256。
  - 报告已存在时拒绝覆盖。
- `scripts/evaluate_sealed_ctc_test.py`
  - 必须显式传 `--acknowledge-sealed-test-evaluation`。
  - 只接受 `experiment=full-ctc-v1, split=test` manifest。
  - 输出明确标记 `test_set_used=true`、`one_time_evaluation=true` 和
    `checkpoint_selection_or_tuning_permitted=false`。
- `tests/test_sealed_test.py`
  - 覆盖 2x temporal best checkpoint 的一次性流式评估。
  - 覆盖报告防覆盖和 latest checkpoint 拒绝。

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy sealed_test.py: pass
Pytest 定向: pass, 2 tests
Pytest 全仓库: pass, 97 tests
CLI --help smoke: pass
git diff --check: pass
```

工作区已于 2026-07-25 在 GPU 5 完成唯一一次正式评估：

```text
test samples:       4,860
test loss:          0.309711
test PER:           0.0677893 (6.7789%)
validation PER:     0.0676448 (6.7645%)
val-test gap:       0.0144 percentage points
sub/del/ins:        6,510 / 6,646 / 3,256
prediction/reference length ratio: 0.9860
blank frame ratio:  0.4038
status:             pass
```

checkpoint SHA256:
`abaadac43c40daf8e2eee339653c64bfafa44fd0267eb7930449bd8d927de774`。
test manifest SHA256:
`a00f111643d75a33884a73ab7e21f520e7dd4e744f56b09a41c51b20da10dedf`。
Test 与 validation 几乎一致，没有明显过拟合。该 test 已消耗，不得再用于当前
checkpoint 或解码策略的选择和调参。

## 0. 2026-07-22 最新状态（后续交接以本节为准）

### 当前目标与已确认决策

时间上采样冻结 Encoder CTC Head 已完成正式训练和 validation 诊断。
用户已确认：

- 部署时只使用 `time_upsampling_factor=2` 的新 Head 时间轴。
- 原始 1× 线性 Head 只作历史对照和研究分析。
- 暂不修改 CTC 压力分桶报告，不让该报告阻塞产品路径。
- 当前进入 phoneme-space hotword scoring 和误触发控制阶段。
- 领导尚未提供正式热词表，先从 formal validation 文本和 Noah MFA 词典构建
  可复现的 pt-BR 模拟热词表，用于打通评分链路。
- 本阶段仍不读取封存 test 集。模拟 validation 结果只用于开发和阈值初调，
  不得宣称为最终泛化结论。

### 工作区已完成的新 Head 实验

2026-07-21 工作区完整训练结果：

```text
run:              run_temporal_upsample_ctc_h512_k5_lr3e4_v1
head:             temporal_upsample, hidden=512, kernel=5, dropout=0.1, 2×
trainable params: 838,746
best epoch:       24
train loss/PER:   0.246010 / 0.066503
validation loss:  0.305222
validation PER:   0.067645
early stop:       true, validation_loss patience=6
test used:        false
status:           completed
```

与旧线性 Head 的 validation 诊断对比：

```text
linear best:    PER 0.293637, deletion 52,016, substitution 16,052,
                insertion 2,636, prediction/reference length 0.7949
temporal 2×: PER 0.067645, deletion 6,812,  substitution 6,557,
                insertion 2,919, prediction/reference length 0.9838
```

这证明新 Head 的时间对齐能力明显更强，并已达到进入热词评分阶段的标准。

### 本轮已实现（待工作区运行）

- `src/qwen_hotword/hotwords/registry.py`
  - 定义可序列化的热词条目。
  - 校验热词 ID、语种、词面、MFA 发音、phoneme token 与 token ID 一致性。
  - 拒绝 blank、越界 ID、重复 ID 和重复发音。
- `src/qwen_hotword/hotwords/simulation.py`
  - 只接受 `split=validation` 记录。
  - 从 1–2 词 validation 短语构造确定性 pt-BR 模拟热词表。
  - 用 Noah MFA 词典和当前 v0.2 词表生成精确 phoneme token IDs。
  - 生成 positive-confusable 和 negative 验证 case，每条 case 有自己的在线
    active hotword 集合。
- `src/qwen_hotword/hotwords/scoring.py`
  - 在 Head 有效时间轴上做 CTC greedy collapse。
  - 用局部音素编辑距离和 posterior confidence 对热词排序。
  - 支持 score threshold、最大 edit ratio、最低 posterior、top-k 和 top-1
    margin 歧义抑制。
- `src/qwen_hotword/hotwords/evaluation.py`
  - 强制 checkpoint 必须是 2× `TemporalUpsampleCtcHead`。
  - 只从 validation feature cache 取模拟 case，不读 test。
  - 输出 precision、recall、F1、positive case hit/top-1 accuracy、negative case
    false-positive rate 和阈值扫描。
  - 默认控制目标是 precision >= 0.90 且 negative-case FPR <= 0.03；未达标时
    会显式写 `meets_control_targets: false`。
- 新 CLI：
  - `scripts/build_simulated_hotwords.py`
  - `scripts/evaluate_hotword_scoring.py`
- 新测试：
  - `tests/test_hotword_scoring.py`
  - `tests/test_simulated_hotwords.py`

### 本轮本地测试

2026-07-22 实际结果：

```text
Ruff（本轮相关文件）: pass
Pytest 定向:                 pass, 6 tests
Pytest 全仓库:             pass, 91 tests
CLI --help smoke:              pass
git diff --check:              pass
Mypy src/qwen_hotword:         11 existing errors, 0 in new hotword modules
```

Mypy 的 11 个既有错误集中在 `qwen_backbone.py`、`ctc_head.py` 和四个训练器的
Torch 类型注解；本轮未扩大范围修改它们。

### 下一步（当前最高优先级）

1. 在工作区从 formal validation manifest 生成 50 个模拟热词和 200 个
   validation-only case。
2. 人工快速查看 `simulated_hotwords.jsonl` 的词面和发音是否合理。
3. 在 GPU 5 上用新 Head best checkpoint 运行 hotword threshold sweep。
4. 将 `hotword_scoring_report.json` 和必要的失败 case 发回分析。
5. 根据报告固定第一版阈值与误触发策略，再接入可在线 reload 的正式
   hotword registry 和 Qwen prompt injection。

### 工作树与保留修改

本轮开始时实际本地基线为 `main@33fe87a`，用户已确认继续在 `main`
工作。本地 `main` 与 GitHub 发布分支历史仍不同步；发布时应继续采用隔离
worktree，将本阶段独立 commit 移植到 `origin/codex/g2p-coverage-scan`，不应强推
本地 `main`。

以下是其他任务/用户的未提交修改，必须继续保留，不得纳入本阶段 commit：

- `docs/PHONEME_VOCAB.md`
- `docs/WORKZONE_RUNBOOK.md`
- `scripts/scan_g2p_coverage.py`
- `tests/test_g2p_coverage.py`
- `tests/test_phoneme_vocab.py`
- `configs/phonemes/en_es_ptbr_fr_id_precision_ipa_vocab.v0.3.json`
- `work/`

## 1. 上一阶段目标（历史记录）

在 Qwen3-ASR-1.7B Audio Encoder 保持冻结、复用既有 `ln_post` BF16 特征缓存的
前提下，将正式分片 CTC 训练链路从仅支持线性 Head 扩展为可选的时间上采样
Head，以验证更高时间分辨率和局部上下文能否降低验证集 PER，尤其是删除错误。

目标结构已实现：

```text
thinker.audio_tower.ln_post
-> LayerNorm
-> 确定性时间上采样（默认 2 倍 repeat_interleave）
-> 1x1 投影 + 轻量 depthwise 时序卷积 + 1x1 上下文投影
-> GELU + Dropout
-> Linear(hidden_dim, 90)
-> CTC Loss
```

## 2. 实际工作树与交接冲突

2026-07-21 恢复任务时的实际状态：

```text
branch: main
HEAD:   56f2c37 Add unfrozen encoder CTC trainer
remote: main...origin/main [ahead 28, behind 5]
```

上一版 HANDOFF 记录的 `unfrozen-encoder-ctc@d36ba23` 与实际工作树不一致。
`d36ba23` 上的以下诊断文件在当时的 `main` 中不存在：

- `scripts/diagnose_frozen_ctc.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `tests/test_ctc_diagnostics.py`

本轮没有在带未提交修改的情况下切换分支，而是以 `main@56f2c37` 为真实基线，
将必要诊断能力移植到当前工作树，并更新为支持新 Head。未切换、回退、删除
或覆盖其他任务的未提交修改。

## 3. 本轮已完成

- 完成 `TemporalUpsampleCtcHead` 的代码审查与收口：
  - 恢复 `LinearCtcHead` 的构造和输入维度校验。
  - 校验 Head 维度、奇数卷积核、dropout 和上采样倍率。
  - 在卷积前后按有效输出长度遮蔽 padding，避免卷积 bias 污染有效边界。
  - `compute_ctc` 恢复旧线性路径校验，并返回 Head 变换后的有效输入长度。
- 将新 Head 接入 `training/sharded_ctc.py`：
  - 通过 `build_ctc_head` 工厂构建线性或上采样 Head。
  - 训练、验证、贪心解码和 PER 统计统一使用 `CtcComputation.input_lengths`。
  - 训练状态和报告写入完整 `head_config`。
  - 新 Head 配置纳入 resume fingerprint；旧线性训练的 fingerprint 保持不变，
    可继续恢复。
- 更新 checkpoint 兼容性：
  - best/latest checkpoint 保留旧顶层字段，同时新增 `head_config`。
  - 可从新 checkpoint 重建时间上采样 Head。
  - 旧 `LinearCtcHead` checkpoint 仍可重建和加载。
  - Experiment A 的初始 Head 加载会显式拒绝结构不匹配的 checkpoint。
- 更新正式训练 CLI：
  - `--head-type {linear,temporal_upsample}`
  - `--head-hidden-dimension`
  - `--head-kernel-size`
  - `--head-dropout`
  - `--head-time-upsampling-factor`
  - 旧命令为保持向后兼容仍默认 `linear`；新实验必须显式传
    `--head-type temporal_upsample`。
- 恢复并升级验证诊断链路：
  - 按 checkpoint 元数据重建 Head。
  - loss、解码、blank ratio、预测/参考长度比和 CTC 压力分桶均使用上采样后
    长度。
  - 保留删除、插入、替换及高频 token 统计。
- 确认特征缓存无需重建：现有校验要求目标在原始 Encoder 帧长下也可行，
  对上采样 Head 是更严格但安全的超集。

## 4. 本轮修改文件

本阶段代码与测试：

- `src/qwen_hotword/modeling/ctc_head.py`
- `src/qwen_hotword/training/sharded_ctc.py`
- `src/qwen_hotword/training/ctc_overfit.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `scripts/train_full_ctc.py`
- `scripts/diagnose_frozen_ctc.py`
- `tests/test_ctc_head.py`
- `tests/test_ctc_diagnostics.py`
- `tests/test_sharded_ctc.py`
- `docs/HANDOFF.md`

工作树中仍存在以下其他任务的未提交修改，本轮未改动、未删除：

- `docs/PHONEME_VOCAB.md`
- `docs/WORKZONE_RUNBOOK.md`
- `scripts/scan_g2p_coverage.py`
- `tests/test_g2p_coverage.py`
- `tests/test_phoneme_vocab.py`
- `configs/phonemes/en_es_ptbr_fr_id_precision_ipa_vocab.v0.3.json`
- `work/`

## 5. 测试结果

2026-07-21 本地实际结果：

```text
Ruff（本轮相关文件）: pass
Mypy src/qwen_hotword:          pass, 33 source files
Pytest (.conda + torch 2.10):   pass, 86 tests
git diff --check:               pass
CLI --help smoke:               pass
```

新增/更新测试覆盖：

- 线性 Head 形状、长度和旧校验回归。
- 上采样 Head 输出 shape 与精确 2 倍有效长度。
- 改变 padding 区域数值不影响有效卷积输出。
- 上采样 Head 的实际 CPU CTC loss。
- 新 Head checkpoint round-trip 和旧线性 checkpoint 兼容。
- 线性与上采样两种 Head 的分片训练、checkpoint、optimizer/scheduler 状态
  和 epoch resume CPU smoke test。
- 诊断模块从两种 checkpoint 重建 Head，且输入帧统计使用有效输出长度。
- 全仓库回归测试。

基础 Python 环境没有 torch，其张量测试会 skip；最终验证使用仓库 `.conda`
环境的 torch 2.10.0，所有测试均真实执行。

## 6. 仍待完成的工作区实验

代码接入和本地完成标准已达到。下一个最高优先级是在 H200 工作区对真实特征
缓存做受控实验；本机没有该 30.9 GB 缓存，因此本轮未伪造“真实缓存已跑”
的结果。

建议步骤：

1. 在新输出目录中，用少量真实 train/validation shard 跑 1–2 epoch smoke test。
2. 确认 report 与 checkpoint 的 `head_config.head_type` 为 `temporal_upsample`，且 resume
   能从下一 epoch 继续。
3. 使用与线性基线相同的 train/validation 特征缓存完整训练，不读取封存
   test 集。
4. 对线性 best checkpoint 和上采样 best checkpoint 运行
   `scripts/diagnose_frozen_ctc.py`。
5. 比较验证 PER、删除/插入/替换、预测/参考长度比、blank frame ratio 和各 CTC
   压力分桶，再决定是否进入 Encoder adapter/LoRA。

正式命令需显式加入：

```bash
PYTHONPATH=src python scripts/train_full_ctc.py \
  ...现有 train/validation cache 与 manifest 参数... \
  --output-dir outputs/full_ctc_temporal_upsample_v1 \
  --head-type temporal_upsample \
  --head-hidden-dimension 512 \
  --head-kernel-size 5 \
  --head-dropout 0.1 \
  --head-time-upsampling-factor 2
```

不得复用旧线性训练输出目录，也不得用线性 checkpoint 作为新 Head 初始权重。

## 7. 已知风险与已确认决策

- 2026-07-21 用户已确认：本项目后续直接在 `main` 分支继续更新，不再为冻结
  Encoder + 时间上采样 CTC Head 阶段单独创建 `codex/` 分支。
- 当前工作树混有其他任务修改。分支归属虽已确认，但未明确提交范围前仍不应
  整体提交、rebase 或整理其他未提交修改。
- 上采样会提高 Head 中间 activation 和训练时间；需用 H200 smoke test 确认
  `train_batch_size=256` 是否仍合适，必要时只调低 Head 训练 batch size。
- 上采样能增加对齐路径，但不能自动解决 G2P/标签噪声或声学表征不足；必须以
  validation PER 和错误分解为准。
- 当前时间上采样 Head 参数量约取决于 hidden dimension，不再是 92,250 参数的线性
  Head；报告会记录实际可训参数。

## 8. 下一任务需读取的最小文件

1. `AGENTS.md`
2. `docs/HANDOFF.md`
3. `src/qwen_hotword/modeling/ctc_head.py`
4. `src/qwen_hotword/training/sharded_ctc.py`
5. `scripts/train_full_ctc.py`
6. `src/qwen_hotword/training/ctc_diagnostics.py`
7. `scripts/diagnose_frozen_ctc.py`
8. `tests/test_ctc_head.py`
9. `tests/test_sharded_ctc.py`
10. `tests/test_ctc_diagnostics.py`

开始下一步前应再执行 `git status --short --branch`。后续保留在 `main` 分支；
不需要再询问是否创建新的 `codex/` 分支。
