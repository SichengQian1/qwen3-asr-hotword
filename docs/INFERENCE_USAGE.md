# 当前推理入口与迁移使用说明

本文记录仓库截至2026-09-11已经实现并在H200工区验证过的两条推理路径：

1. 完整音频到CTC Anchor热词列表；
2. CTC + Anchor + Prompt + Qwen的2秒流式端到端转写。

两条路径均使用固定目标模型`Qwen3-ASR-1.7B`和三语Temporal 2x CTC Head。本文只说明
现有实现，不把尚未实现的通用单音频入口写成可用功能。讨论中的
`scripts/transcribe_with_anchor_rag.py`当前不存在；它是后续把评测依赖剥离后的纯推理CLI
候选，不应在新机器上直接调用。

## 1. 共同架构与必需产物

共享模型链路：

```text
audio
  -> Qwen3-ASR-1.7B audio encoder / thinker.audio_tower.ln_post
  -> multilingual Temporal 2x CTC Head
  -> greedy phoneme posterior decode
  -> phoneme 2/3/4-gram Anchor Top-64 shortlist
  -> local approximate rerank and operating gate
  -> Top-K hotwords
```

端到端路径在上述链路后继续执行：

```text
Top-K hotwords
  -> spelling-reference prompt
  -> official qwen-asr vLLM streaming decoder
  -> final transcript
```

新机器至少需要：

```text
Git仓库：
  https://github.com/SichengQian1/qwen3-asr-hotword.git
分支：
  codex/g2p-coverage-scan

完整模型目录：
  Qwen3-ASR-1.7B/

三语CTC Head（不在Git）：
  outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
  SHA256: bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0

90类音素词表（在Git）：
  configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
```

当前工区验证环境是Python 3.12、Torch 2.10.0 + CUDA 12.8、`qwen-asr==0.0.6`、
`transformers==4.57.6`和`vllm==0.14.0`。仓库的`workzone` extra不单独安装Torch、
Qwen-ASR或vLLM；迁移时应优先复用已经验证的CUDA基础镜像和依赖版本，不要仅运行
`pip install -e .[workzone]`后就假定完整端到端环境已经齐备。

## 2. 路径一：完整音频Anchor热词检索

### 2.1 适用范围

入口：

```text
scripts/run_external_keyword_retrieval.py
```

该入口递归读取一个或多个音频目录，原生支持WAV和FLAC，在内存中转换为mono、16 kHz、
float32；不会覆盖或转换源音频文件。每个音频完整地经过Qwen audio encoder、三语CTC
Head和Anchor检索，最终输出0至5个`word/phoneme`对象。

这个入口**不运行Qwen LLM decoder**，所以：

- `retrieval_output.json`是热词检索结果，不是ASR转写；
- `final_retrieval_recall`是音频到门控后热词列表的Recall，不是最终文本Recall；
- transcript只在检索完成后建立真值和统计Recall/Precision，不参与候选生成、排序或门控。

当前实现针对`pt_keyword_bias_phoneme.json`中的`hard_k266`完成验证。固定配置为：

```text
threshold:                    0.75
top_k:                        5
maximum_edit_ratio:           0.35
posterior_weight:             0.25
minimum_posterior_confidence: 0.5
minimum_top1_margin:          0
minimum_phonemes:             1
Anchor shortlist:             64
Anchor ngrams:                2,3,4
anchors_per_entry:            24
anchor_offset_tolerance:      1
rerank_start_radius:          2
```

这些门控值目前固化在该入口内部，不是CLI参数。不要把评测入口误认为可任意调参的生产
接口。

### 2.2 输入格式

每个数据源使用：

```text
--source NAME=AUDIO_DIR,TRANSCRIPTS
```

其中音频文件stem必须能和转写记录ID一一对应。程序拒绝缺失转写、无音频转写、源内重复
stem和跨来源重复stem，避免最终JSON的key被覆盖。

热词输入：

```text
--keyword-bias /path/pt_keyword_bias_phoneme.json
--keyword-set hard_k266
```

程序会验证指定set中的词面、MFA音素和90类CTC词表映射；存在OOV时停止，不会静默丢词。

### 2.3 CPU预检

首次运行使用一个不存在的新输出目录：

```bash
cd /path/qwen3-asr-hotword

MODEL=/path/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
CTC_CHECKPOINT=/path/ctc_head_best.pt
KEYWORDS=/path/pt_keyword_bias_phoneme.json
EXTERNAL_OUTPUT=outputs/external_keyword_retrieval_v1

python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$KEYWORDS" \
  --keyword-set hard_k266 \
  --source "dataset_name=/path/audio_dir,/path/transcripts.txt" \
  --output-dir "$EXTERNAL_OUTPUT" \
  --audit-only
```

多个来源重复传入`--source`：

```bash
  --source "mls_portuguese=/path/mls_audio,/path/mls_transcripts.txt" \
  --source "delivery_ptbr=/path/delivery_audio,/path/delivery_transcripts.txt"
```

预检后验证：

```bash
(cd "$EXTERNAL_OUTPUT" && sha256sum -c sha256.txt)

jq '{status, sample_count, unique_sample_ids,
  cross_source_duplicate_sample_ids, sources}' \
  "$EXTERNAL_OUTPUT/dataset_audit.json"

jq '{status, keyword_set, keyword_count, vocabulary_size,
  oov_keyword_count, keywords_below_four_phonemes}' \
  "$EXTERNAL_OUTPUT/keyword_audit.json"
```

### 2.4 H200完整运行和续跑

预检通过后，必须复用同一输出目录并加`--resume`：

```bash
GPU_ID=0
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$KEYWORDS" \
  --keyword-set hard_k266 \
  --source "dataset_name=/path/audio_dir,/path/transcripts.txt" \
  --output-dir "$EXTERNAL_OUTPUT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
```

中断后原样重跑。不要删除`sample_shards`，不要修改配置后强制复用原目录；程序会对模型、
checkpoint、词表、热词、数据源及已有shard做身份验证。

主要输出：

```text
retrieval_output.json       音频stem -> 0至5个word/phoneme对象
evaluation_summary.json     总体和分来源Recall/Precision/FPR及分阶段时延
retrieval_details.jsonl     逐条原始Top-20、分数、CTC音素及门控详情
failure_cases.jsonl         missed/wrong selected案例
run_config.json             完整运行身份和固定参数
sha256.txt                  最终小文件完整性
```

完成后：

```bash
(cd "$EXTERNAL_OUTPUT" && sha256sum -c sha256.txt)

jq '{status, evaluation_scope, gate, retrieval_backend, overall, by_source}' \
  "$EXTERNAL_OUTPUT/evaluation_summary.json"

jq 'to_entries[:3]' "$EXTERNAL_OUTPUT/retrieval_output.json"
```

### 2.5 Top-7精确重放

已经完成的Top-5运行保存了原始Top-20，可以在CPU上把唯一变化限定为Top-K 5到7，无需
重新加载模型或音频：

```bash
TOP7_OUTPUT=outputs/external_keyword_top7_replay_v1

python scripts/replay_external_keyword_top7.py \
  --source-run "$EXTERNAL_OUTPUT" \
  --vocab "$VOCAB" \
  --keyword-bias "$KEYWORDS" \
  --keyword-set hard_k266 \
  --output-dir "$TOP7_OUTPUT"

(cd "$TOP7_OUTPUT" && sha256sum -c sha256.txt)
cat "$TOP7_OUTPUT/topk_comparison.md"
```

主要交付：

```text
mls_retrieval_output.json
delivery_retrieval_output.json
topk_comparison.json
topk_comparison.md
top7_replay_details.jsonl
```

Top-7重放严格要求源运行是已完成、SHA通过的固定D5运行；它不是任意参数扫描器。重放
时延沿用源D5模型运行观测值，不冒充Top-7重新测得的Encoder/Anchor时延。

## 3. 路径二：CTC + Anchor + Prompt + Qwen最终转写

### 3.1 适用范围和限制

入口：

```text
scripts/run_streaming_rag_evaluation.py
```

当前实现是正式评测入口，不是通用单音频生产CLI。它按Manifest和sealed cases运行2秒
流式C/D/E对照：

```text
C: Qwen streaming no-RAG
D: multilingual CTC + Anchor RAG + dynamic prompt
E: Oracle prompt
```

只需要RAG路径时可以传`--groups D`，但仍需要评测样本选择和身份文件。程序当前还不能用
`--audio sample.wav`直接处理任意文件。

除了共同模型、CTC Head和词表，该入口还需要：

```text
validation manifest
standard hotwords.jsonl
cases.jsonl
hotword_families_v3.jsonl
multi_nested_evaluation_report_v3_corrected.json
offline prompt evaluation directory
```

这些文件用于样本选择、active hotword IDs、Oracle定义和配置身份校验。它们是当前评测
入口的依赖，不代表未来生产推理必须保留这些依赖。

### 3.2 单语种端到端命令

以下是当前已验证配置的通用模板：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_streaming_rag_evaluation.py \
  --model /path/Qwen3-ASR-1.7B \
  --validation-manifest /path/full_ctc_validation_LANGUAGE.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords /path/size_4000/hotwords.jsonl \
  --cases /path/size_4000/cases.jsonl \
  --hotword-families /path/hotword_families_v3.jsonl \
  --ctc-report /path/multi_nested_evaluation_report_v3_corrected.json \
  --offline-rag-dir /path/offline_LANGUAGE \
  --offline-format multi_nested_v3 \
  --offline-control-mode selection_only \
  --ctc-checkpoint /path/ctc_head_best.pt \
  --output-dir outputs/streaming_LANGUAGE \
  --groups C,D,E \
  --chunk-size-sec 2.0 \
  --unfixed-chunk-num 2 \
  --unfixed-token-num 5 \
  --retrieval-mode operating \
  --retrieval-backend anchor_guided \
  --anchor-shortlist-size 64 \
  --anchor-start-radius 2 \
  --anchor-ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --anchor-offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --language Portuguese \
  --dtype bfloat16 \
  --device cuda:0 \
  --gpu-memory-utilization 0.18 \
  --resume
```

三语CTC Head是同一个checkpoint，`--language`表示本次单语种推理的Qwen语言上下文，
不是CTC Head支持语言的声明：

```text
English audio:    --language English
Spanish audio:    --language Spanish
Portuguese audio: --language Portuguese
```

当前仓库未实现`--language auto`。Qwen官方接口允许`language=None`执行原生语言识别，
但直接传字符串`auto`不是同一语义；还需要在项目中补充参数映射、热词语言子库选择和
Prompt策略后才能把`auto`作为受支持功能。

### 3.3 当前流式语义

- 每2秒产生一个新chunk；尾部不足2秒时不补零，结束时flush；
- CTC输入是截至当前时刻的累计音频，即0--2、0--4、0--6秒重新提取；
- 当前尚未实现audio encoder cache，因此功能正确但会重复计算历史音频；
- D组每个step重新运行CTC、Anchor和门控，并刷新Top-K spelling-reference Prompt；
- `unfixed_chunk_num=2`，前两个chunk不复用历史文本；之后回退最后5个token再续写；
- Qwen-ASR没有公开的逐step context setter，当前实现通过官方
  `init_streaming_state`取得`prompt_raw/context/force_language`并刷新活动state；缺少字段
  时立即失败，不会静默退化。

### 3.4 显存和进程注意事项

端到端D组会在同一进程中初始化两个模型实例：

```text
Transformers实例：Qwen audio encoder + CTC Head
vLLM实例：Qwen streaming decoder
```

因此显存不能按“只加载一次1.7B模型”估算。开始前应检查目标GPU空闲显存；若vLLM提示
启动时空闲显存低于`gpu_memory_utilization`请求值，应释放同卡其他进程、换空闲GPU，或在
新输出目录上采用经过确认的新显存配置。不要修改参数后对旧目录强行`--resume`。

### 3.5 输出和续跑

`--resume`只复用运行配置与输入身份完全匹配的sample shard。主要输出：

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
```

验证：

```bash
(cd outputs/streaming_LANGUAGE && sha256sum -c sha256.txt)

jq '{status, groups}' outputs/streaming_LANGUAGE/summary.json
jq '.groups.D.compute' outputs/streaming_LANGUAGE/latency_summary.json
```

`retrieval_seconds`严格表示CTC greedy decode + Anchor query + Top-64 shortlist
rerank/gate，不包括CTC processor、Qwen Encoder或CTC Head。完整step和样本推理时延另有
字段，不得把纯检索P95当成整条端到端P95。

## 4. 迁移建议

### 4.1 优先迁移完整仓库

不要手工只复制几个Python文件；推理入口会跨`src/qwen_hotword/`调用模型、音素、热词、
Anchor、评分、Prompt和streaming模块。使用Git迁移：

```bash
git clone https://github.com/SichengQian1/qwen3-asr-hotword.git
cd qwen3-asr-hotword
git checkout codex/g2p-coverage-scan
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

Git只带走代码、配置、测试和文档。模型、CTC checkpoint、数据集、Manifest、评测资产及
`outputs/`通常被忽略，必须单独复制并用SHA256核验。

### 4.2 按目标选择入口

| 目标 | 当前入口 | 是否有Anchor | 是否运行Qwen decoder | 输入粒度 |
| --- | --- | ---: | ---: | --- |
| 给下游返回热词列表 | `run_external_keyword_retrieval.py` | 是 | 否 | 音频目录 + transcripts |
| formal流式端到端对照 | `run_streaming_rag_evaluation.py` | 是 | 是 | Manifest + cases + offline controls |
| 任意单音频直接返回热词和转写 | 尚未实现 | 计划支持 | 计划支持 | 未来`--audio` |

如果新机器要立刻复现已经完成的266词交付，应优先迁移路径一；如果必须获得带Prompt的
最终转写，则迁移路径二的完整评测资产和vLLM环境。不要把路径一的热词JSON格式直接传给
路径二：路径二当前要求标准registry JSONL，而266入口读取带`keyword_sets`的嵌套JSON。

后续纯推理入口应复用现有模块，去掉validation、cases、offline report等评测依赖，并支持
单音频/目录、完整音频/2秒流式、标准registry/266输入适配和稳定的机器可读输出；在该入口
真正实现、测试和H200验证前，不将其列为可部署命令。
