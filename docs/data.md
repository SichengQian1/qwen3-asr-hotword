# 训练数据说明

本文记录 Qwen3-ASR 热词项目当前正式训练数据的来源、处理结果、格式和使用边界。
工作区实际文件、SHA256 与运行报告优先于本文；流程变化时应建立新数据版本并同步
更新本文。

## 1. 原始语料

当前已完成正式处理的语料是 Noah 500 小时巴西葡萄牙语口语数据。

```text
TSV:
/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/500小时巴西葡萄牙语口语化语音数据.tsv

Audio root:
/host_home/z00841352/27A/data/Noah_espt/noah_pt

TSV columns: audio, text
Language: pt-BR
```

`audio` 是相对路径，解析方式为 `audio_root / audio`。路径和文本包含非 ASCII
字符，处理时使用 UTF-8 和 `pathlib`。

```text
TSV records:  366,508
Audio:        496.7013 h
Word tokens:  5,348,219
Unique words: 76,744
```

前 1,000 行审计通过，音频解析 1,000/1,000，缺失和重复均为 0。原始 TSV 没有
可靠 speaker ID，因此只能保证文件级隔离，不能声明 speaker-disjoint。目前也
没有已确认的 Spanish/es-419 正式训练数据路径。

## 2. MFA 与 G2P

MFA 环境和已固定的多语言 G2P 模型：

```text
Conda env: aligner
MFA version: 3.4.0
Model directory:
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/g2p

Brazilian Portuguese:
  portuguese_brazil_mfa.zip
English (US):
  english_us_mfa.zip
Spanish (Latin America):
  spanish_latin_america_mfa.zip
```

英语和拉美西语模型由用户于 2026-08-04 确认已下载到上述目录，与巴葡模型并列
保存。三个模型必须按各自语言使用，不能跨语言复用。开始英语或西语正式 G2P
前仍需记录实际文件大小和 SHA256；拉美西语模型是阿根廷西语第一版的候选基线，
不应表述为专门的阿根廷口音模型。

执行形式：

```bash
conda run -n aligner mfa g2p \
  --num_pronunciations 1 \
  WORDS_TXT \
  G2P_MODEL_ZIP \
  OUTPUT_DICT
```

相关代码：

```text
scripts/prepare_mfa_g2p.py
scripts/audit_mfa_dictionary.py
src/qwen_hotword/training/g2p_prep.py
src/qwen_hotword/training/mfa_audit.py
```

主要产物：

```text
outputs/noah_pt_mfa_g2p/words.txt
outputs/noah_pt_mfa_g2p/word_counts.tsv
outputs/noah_pt_mfa_g2p/character_counts.tsv
outputs/noah_pt_mfa_g2p/fragments_with_digits.tsv
outputs/noah_pt_mfa_g2p/summary.json
outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
```

完整运行结果：

```text
Input unique words:   76,744
MFA dictionary lines: 76,736
Runtime:              1,702.057 s
```

相差的 8 行必须通过 MFA audit 的 `missing_words.tsv` 等文件核对，不能直接视为
无影响。Noah 必须使用自己的 word list，不能用早期 FLEURS 报告证明覆盖率。

文本规范化采用 Unicode NFC、`casefold()`、空格规范化和连接符统一。单词内部
的 `'`、`-` 可保留，数字片段单独统计。

## 3. CTC 音素词表

正式 Noah 数据固定使用：

```text
configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
Classes: 90
blank_id: 0
SHA256: 0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4
```

Formal manifest、token ID 和现有 checkpoint 都绑定 v0.2。更换词表必须建立
新版本，重新完成 G2P/OOV 审计、token ID 生成和模型兼容性验证。

## 4. 完整 Manifest

构建入口和输出目录：

```text
scripts/build_full_training_manifest.py
src/qwen_hotword/training/full_manifest.py
outputs/noah_pt_full_500h
```

主要产物：

```text
train_ready.jsonl
needs_review.jsonl
summary.json
build_config.json
shard_index.json
shards/
reports/
```

每条源 TSV 记录只能进入 `train_ready.jsonl` 或 `needs_review.jsonl`，不能静默
丢弃。Ready 至少要求音频可读、MFA/vocab 覆盖完整、标签非空且 CTC 长度物理
可行。Standalone `h`、连接符问题、dictionary miss、OOV phone、无效音频、
空标签和 CTC 长度不可行等都进入 review，并保留明确 `issues`。

```text
Source:          366,508 / 496.7013 h
Training-ready:  243,876 / 326.8718 h
Needs review:    122,632
Shards:          74
Status:          pass
```

核心记录字段：

```text
id, source_tsv, row_number
audio_relative, audio_path
text, normalized_text, language
words, phonemes, phoneme_token_ids, label_length
word_pronunciations
duration_seconds, sample_rate
estimated_feature_length, estimated_ctc_input_length
ctc_minimum_input_length, ctc_target_ratio
issues, training_ready, label_status, split_hash
```

`ctc_minimum_input_length` 考虑相邻重复 token 所需的 blank。稳定 `split_hash`
基于 `SHA256(audio_relative + "\0" + text)`。构建按 5,000 条分片，默认 16 个
metadata worker，支持原子输出和 resume。

Experiment A/B 的词频、时长和 `CTC ratio <= 0.75` 限制不属于完整数据 pass。

## 5. 正式切分

只从 `train_ready.jsonl` 构建稳定的 96/2/2 切分：

```text
scripts/build_full_training_splits.py
src/qwen_hotword/training/full_training.py

outputs/noah_pt_full_training_v1/full_ctc_train.jsonl
outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl
outputs/noah_pt_full_training_v1/full_ctc_test.jsonl
outputs/noah_pt_full_training_v1/split_config.json
outputs/noah_pt_full_training_v1/split_summary.json
```

```text
Train:       234,181
Validation:    4,835
Test:          4,860
Total:       243,876
```

构建过程拒绝重复 ID、重复音频、跨 split 重叠、无效标签、无效 CTC 长度和意外
覆盖，并在 summary/config 中保存输入输出 SHA256。

Validation manifest SHA256：

```text
56621b5ef3c48a3b8488732fcb5c47ddf69c475c096ae46e1099bf034f880a45
```

## 6. Sealed Test

```text
outputs/noah_pt_full_training_v1/full_ctc_test.jsonl
SHA256: a00f111643d75a33884a73ab7e21f520e7dd4e744f56b09a41c51b20da10dedf
```

该 test 已于 2026-07-25 用于一次固定 CTC checkpoint 正式评估。后续数据开发、
规则分析、样本抽查和调参不得读取其内容，也不得创建 test feature cache。只允许
通过 `split_summary.json` 核对路径、数量和 SHA256。

## 7. 冻结 Encoder 特征缓存

```text
scripts/cache_full_training_features.py
src/qwen_hotword/training/feature_cache.py

outputs/noah_pt_full_training_v1/features_ln_post_bf16/train
outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation
```

```text
Model: Qwen3-ASR-1.7B
Tap: thinker.audio_tower.ln_post
Dimension/dtype: 1024 / bfloat16
Encoder: frozen

Train:      234,181 samples, 458 shards, 14,741,705 frames, 30,289,466,690 bytes
Validation:   4,835 samples,  10 shards,    303,471 frames,    625,356,770 bytes
Test cache:   none
```

缓存由物理 GPU 5 完成，进程内映射为 `cuda:0`。每个 shard 都有 SHA256 校验。
更换模型、tap、manifest、vocab、dtype 或 Encoder 策略时必须建立新缓存。

## 8. 历史小规模数据

Experiment A 是 128 条、约 7.21 分钟的严格清洗过拟合集；Experiment B 是
8 小时 train、1 小时 validation、1 小时 test 的初步泛化集。它们是历史
可行性实验，不是当前正式训练数据。

## 9. 数据版本原则

1. 原始 TSV 和音频只读。
2. 任何源记录都不能静默丢弃。
3. Review/reject 必须保留明确原因。
4. 大型语料、字典、manifest、缓存和模型权重不提交 Git。
5. 新规则写入新版本目录，不覆盖现有 v1。
6. 不读取 sealed test 做开发。
7. 没有 speaker ID 时不声明 speaker-disjoint。
8. 长任务保留 summary、配置、数量、路径和 SHA256。
9. 重跑前检查现有 config、summary、shard index 和 resume 状态。
10. 工作区实际结果与本文冲突时，以实际结果为准并更新本文。

## 10. Noah 200 小时巴葡金融口语数据

用户于 2026-07-29 确认了一套新的巴西葡萄牙语金融领域口语数据。它优先于
MLS 葡语变体审计，按 Noah 500 小时的 MFA、v0.2 音素词表和完整 manifest
流程处理，但使用独立数据版本，暂不与 500 小时数据合并。

用户提供的宿主机 TSV：

```text
/home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv
```

按现有容器挂载规则，第一轮应验证的容器内路径和候选音频根目录是：

```text
TSV:
/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv

Candidate audio root:
/host_home/z00841352/27A/data/Noah_espt/noah_pt
```

首批和全量 TSV 审计均已通过，确认音频根目录正确：

```bash
python scripts/audit_training_tsv.py \
  --tsv "/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv" \
  --audio-root /host_home/z00841352/27A/data/Noah_espt/noah_pt \
  --max-records 1000 \
  --sample-count 5 \
  --output outputs/noah_pt_finance_200h/audit_first_1000.json
```

实际输入和第一版 manifest 结果：

```text
Source records:       142,985
Source audio:         195.3711 h
Valid audio:          142,985
Word tokens:          2,145,885
Unique words:         40,458

MFA dictionary word coverage:   97.1971%
MFA corpus-token coverage:      99.8380%
Missing unique words:           1,134
Words/phones outside v0.2:      0

Training-ready:        86,614 / 119.5436 h
Needs review:          56,371
Completed shards:      29
Status:                pass
```

第一版问题计数允许一条记录命中多个问题：

```text
ctc_length_infeasible: 53,699
dictionary_missing:     3,390
unresolved_connector:   3,377
standalone_h:               10
empty_ctc_target:            8
```

缺词主要是带连字符的高频形式，如 `e-commerce`、`bem-vindo`、`bate-papo`、
`bem-estar` 和 `matéria-prima`。所有 MFA 输出音素均兼容当前 90 类 v0.2
词表。后续恢复重点是按已部署的 2× temporal Head 重算 CTC 可行性，并为连接词
建立显式解析；这些恢复工作不阻塞其他语料首轮处理。

正式 manifest 身份固定为：

```text
dataset:   noah_pt_finance_200h
id prefix: noah_pt_finance_200h_row
language:  pt-BR
split:     train
```

`scripts/build_full_training_manifest.py` 已支持这些可配置字段，同时保留旧 500
小时默认值以兼容已有输出。建议目录：

```text
outputs/noah_pt_finance_200h/mfa_g2p
outputs/noah_pt_finance_200h/mfa_audit
outputs/noah_pt_finance_200h/full_manifest_v1
```

这套数据当前全部作为新增训练候选，不从中建立新的 validation/test。是否与旧
500 小时 train 合并，必须等独立覆盖率、ready/review 比例和音频去重检查完成后
另建合并数据版本。

## 11. FLEURS、MLS、Common Voice 葡语 Swift JSON

三套新语料先分别转换、审计和版本化，全部作为 train 候选，不覆盖 Noah 数据，
也不在本阶段创建 validation/test。

输入 JSON：

```text
FLEURS:
/data/h00911716/code/ms-swift/self_test/datalist/pt/fleurs/swift_fleurs_pt.json

MLS:
/data/h00911716/code/ms-swift/self_test/datalist/pt/mls/swift_librispeech_pt.json

Common Voice:
/data/h00911716/code/ms-swift/self_test/datalist/pt/cv/swift_cv_pt.json
```

每条源记录是顶层 JSON list 中的对象。音频来自 `audios[0]`，文本优先使用
`response`，缺失时回退到最后一条 assistant message，并去掉：

```text
language Portuguese<asr_text>
```

三份输入 JSON 位于 `/data/...`，但 JSON 内的 63,249 条音频路径均以
`/home_92/...` 开头；转换时在容器内确定性改写为 `/host_home/...`。转换器在
改写后检查文件存在性，并将所有行为记录在各自的
`swift_json_conversion_summary.json`。

建议独立目录：

```text
outputs/pt_external_train_sources_v1/fleurs
outputs/pt_external_train_sources_v1/mls
outputs/pt_external_train_sources_v1/common_voice
```

第一阶段只要求三份转换报告确认：全部源记录有音频和文本、语言字段一致、音频
文件全部存在、路径改写数量合理。MLS 官方数据只提供 Portuguese 标识，地区变体
尚未严格确认；在音频/元数据抽查前保留 MLS 来源身份，不把全部说话人强行声明为
pt-BR。FLEURS、Common Voice 也保留各自 corpus provenance，后续分别生成词表和
G2P 报告，再决定最终语言标签与合并策略。

转换和独立词表已于 2026-07-30 完成：

```text
Corpus          Records   Audio   Word tokens   Unique words   Digit fragments
FLEURS            2,793   WAV          60,947          7,743               847
MLS              37,533   FLAC      1,261,190         75,392                 0
Common Voice     22,923   MP3         154,407         26,593               845
Total            63,249
```

三套转换均为 `status=pass`，无跳过、缺失音频或语言异常；63,249 条音频路径全部
完成 `/home_92 → /host_home` 改写并验证存在。MLS 的唯一词比例明显高于另外两
套，符合书籍语料、旧拼写和长尾词较多的特征，不能与口语语料共用覆盖率结论。

FLEURS 和 Common Voice 的数字片段不能被静默忽略。full manifest builder 从此
将任何含数字片段的记录标记为 `unresolved_digit` 并保留到 `needs_review`；
在没有上下文明确的葡语数字规范化方案前，不自动生成年份、金额、序数或缩写的
发音标签。

独立 MFA audit：

```text
Corpus          Token coverage   Missing words   Duplicate entries   Phone OOV
FLEURS                99.4192%             124                   1           0
MLS                    98.8174%           2,294                 305           0
Common Voice           99.2889%             802                  12           0
```

巴葡 MFA 的所有输出 phone 均兼容当前 90 类 v0.2 词表。`training_labels_ready`
为 false 表示并非 100% 词覆盖，不代表审计运行失败。MLS 的覆盖结果只证明技术
兼容，不能代替方言确认。

最终第一版完整 manifest：

```text
Corpus          Source records/h       Ready records/h       Review
FLEURS          2,793 / 10.1789 h       1,966 / 6.8475 h         827
MLS            37,533 / 160.9632 h     26,030 / 110.3132 h     11,503
Common Voice   22,923 / 26.4790 h      21,803 / 24.9761 h       1,120
Total          63,249 / 197.6211 h     49,799 / 142.1368 h     13,450
```

正式引用：

```text
FLEURS:
outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard

MLS:
outputs/pt_external_train_sources_v1/mls/full_manifest_v1

Common Voice:
outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard
```

FLEURS `full_manifest_v1` 在数字保护合入前生成，仅保留历史对照，不能用于训练。
Common Voice v1 虽然 ready 数量与 v2 相同，正式版本仍固定为 v2。

最终问题计数：

```text
FLEURS:
  dictionary_missing:      350
  standalone_h:             21
  unresolved_connector:    316
  unresolved_digit:        839

MLS:
  ctc_length_infeasible:    121
  dictionary_missing:    14,773
  standalone_h:             49
  unresolved_connector: 14,698

Common Voice:
  ctc_length_infeasible:    111
  dictionary_missing:     1,096
  empty_ctc_target:          66
  standalone_h:               1
  unresolved_connector:   1,094
  unresolved_digit:         350
```

Issue 数允许一条记录命中多个原因，不能将计数直接相加当作 review 记录数。

三套数据完成独立 train 候选处理时记录了以下合并检查项：

1. 明确 MLS 的方言兼容边界；
2. 做跨 FLEURS/MLS/Common Voice/Noah 的音频和规范化文本去重；
3. 明确记录是否使用 corpus 采样权重，避免误解样本自然比例；
4. 新建合并 manifest 版本，不覆盖任何独立产物；
5. 合并完成后才创建新的 Encoder feature cache。

## 12. Temporal 2× 五语料合并版本

2026-08-04 用户决定建立新的全集训练版本，不覆盖任何旧清单。数据来源为 Noah
500h、Noah金融200h、MLS、Common Voice和FLEURS；每套都纳入原ready记录，并
额外纳入唯一问题为 `ctc_length_infeasible`、2×后 effective ratio `<=0.90`
的恢复记录。其他标签问题和ratio `(0.90,1.00]` 暂不释放。

审计后的候选规模：

```text
Corpus                 Original ready             Temporal 2x recovery <=0.90
Noah finance 200h       86,614 / 119.543627 h      52,944 / 70.628133 h
Noah 500h              243,876 / 326.871849 h     113,607 / 156.022247 h
MLS                     26,030 / 110.313155 h          98 / 0.382994 h
Common Voice            21,803 /  24.976131 h         108 / 0.109258 h
FLEURS                    1,966 /   6.847500 h           0 / 0 h
Total                   380,289 / 588.552262 h     166,757 / 227.142632 h
Combined                547,046 / 815.694894 h
```

构建入口和新输出目录：

```text
scripts/build_temporal2x_combined_training.py
src/qwen_hotword/training/combined_training.py
outputs/pt_combined_temporal2x_v1
```

使用每条记录已有的稳定 `split_hash` 做96/2/2，保证旧Noah样本不会因合并改变
split。新清单保留来源与原始语言标签，拒绝跨语料重复ID/绝对音频路径；测试清单
生成后封存。所有记录绑定 `ctc_time_upsampling_factor=2`，因此该版本只用于
Temporal 2× Head，不用于线性1× Head。第一版暂按样本自然比例合并，没有实现
corpus重采样权重；MLS的地域变体限制仍需在正式产品结论中注明。

## 13. 西语候选原始数据

2026-08-04 用户提供两份与既有 Swift JSON 数据相同格式的西语候选来源：

```text
MLS Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/mls/swift_librispeech_es.json

Common Voice Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/cv/swift_cv_es.json
```

原始 JSON 只读，第一阶段分别转换并输出到新的西语目录，不覆盖葡语产物。处理
时使用 `spanish_latin_america_mfa.zip` 生成候选发音标签，并继续绑定
`en_es_ptbr_precision_ipa_vocab.v0.2.json` 做覆盖审计。

这两份文件目前只登记为 `Spanish` 候选来源，不能仅凭文件名声明为阿根廷西语：
MLS Spanish 通常属于西班牙来源，Common Voice `es` 也可能包含多个地区。正式
合并前必须根据可用元数据或语料来源确认方言范围；在确认前输出标签使用 `es`，
不使用 `es-AR`，也不把生成字典视为阿根廷口音准确性的证明。

## 14. 阿根廷/拉普拉塔西语增量MFA修复

SLR61 Argentinian与Common Voice Rioplatense v26已分别完成原始词表的MFA G2P，
现有字典不得覆盖。首轮覆盖不足主要来自MFA模型不接受带acute accent、`ñ`或`ü`
的输入；发音中的`U+0303 COMBINING TILDE`在本项目西语输出中为非音位性phone OOV。

修复使用一个跨两套语料去重的增量代理词表，不重跑完整MFA。原始文本和最终字典键
始终保留正确西语拼写；去acute、`ñ -> ni`与`gü -> gw`只发生在G2P代理层。
`ñ`代理产生的`ɲ j`只在相应规则下还原为`ɲ`，`U+0303`只从最终西语候选发音中
删除，不能修改共享CTC词表或葡语鼻元音处理。

修复后每套字典自动运行`audit_mfa_dictionary`。存在任何缺词、多发音、空发音或
phone OOV时保持`training_labels_ready=false`并写入未解析清单；在两套审计均通过
前不构建完整Manifest，也不合并MLS Spanish或通用Common Voice Spanish。

## 15. 西语150小时扩容库存与方言元数据分层

最终目标是至少150小时`train_ready`西语，而不是150小时未经审计的原始音频。为给
过滤、speaker-disjoint划分和测试封存留余量，先基于MLS Spanish与通用Common Voice
Spanish建立约170小时可释放候选池。阿根廷/拉普拉塔核心集优先；通用CV依据原始
元数据分层；MLS因来源和拉美G2P方言不完全匹配只作限量后备。

Swift JSON转换后的通用TSV没有speaker、accent或官方split，因此使用
`audit_spanish_candidate_inventory.py`按Common Voice clip basename关联原始v25
`validated.tsv`以及`train/dev/test.tsv`。分层只表示元数据证据：

```text
argentinian_rioplatense_metadata
latin_american_metadata
mixed_latin_american_peninsular
peninsular_metadata
other_unclassified_metadata
unknown
```

这不是声学口音分类器，不能把通用CV统一标成`es-AR`。训练池按以下优先级输出：

```text
priority_argentinian_rioplatense
priority_latin_american
candidate_mixed / candidate_unknown / candidate_other_unclassified
fallback_peninsular
```

以下情况显式排除：音频不可读、原始CV元数据缺失、locale非`es`、Swift文本与元数据
不一致、官方validation/test，以及与Rioplatense v26核心任一split的clip ID重复。
通用CV v25与Rioplatense v26可能共享同一个`common_voice_es_*` clip ID，因此跨版本
去重必须先于任何选样；核心validation/test重复尤其不得进入训练。

MLS inventory只从MLS文件名提取speaker ID并统计时长，方言tier固定为
`mls_source_unknown_likely_peninsular`。它不能仅凭使用拉美MFA模型就改称拉美语料。
第一轮建议MLS在最终train中最多占10%至15%，具体配额必须等待真实小时数、CV元数据
覆盖和核心重叠审计完成后确定。

## 16. 明确拉美Common Voice辅助池

真实库存证明，只用有明确拉美元数据的Common Voice就足以支撑本轮目标。
通用CV共516.125434小时；排除与Rioplatense核心train重复的15.268736
小时后，已审计的显式元数据中仍有额外Rioplatense 2.110041小时和其他
拉美228.542930小时。因此首版不使用未知口音、半岛CV或MLS。

`select_spanish_auxiliary_pool.py`从已生成的CV inventory读取时长和元数据，
并与原始Swift `source.tsv`按绝对音频路径做一对一校验。它从原始`accent`重新
计算tier，因此修正了库存v1未把`América central`识别为拉美的规则漏项，
不需再读取音频。

选样以speaker为分割边界，按未满的时长比例确定性补足96/2/2，并为普通拉美
speaker设置2小时贡献上限。核心train speaker强制留在train，核心validation/test
speaker整体排除，从而在clip去重之外再防止speaker泄漏。输出`source.tsv`
保留`source_split`/`speaker_id`/元数据tier，可直接用于后续词表提取和
`build_full_training_manifest.py --split-column source_split`。

本轮选170小时辅助池，加核心Temporal 2×候选合计约193.7小时。这是
过滤前的余量设计，不得预先声称最终train已达150小时；必须完成MFA修复、
完整Manifest和Temporal 2×审计后才做最终判定。

工作区实际选样为118,177条、171.594529小时，train/validation/test分别为
164.102775/3.819581/3.672173小时，所有speaker跨split重叠为0。该子集与核心
Temporal 2×候选合计约195.29小时，但最终可训小时数仍以后续MFA、Manifest和
Temporal 2×审计为准。

## 17. 三语平衡集口径

最终产物是英语/西语/葡语`1:1:1`训练集。默认口径为train音频小时和
有效训练曝光一致，不是原始样本条数一致。第一版目标是三种语言各约150小时
train；完整单语种池另行保留，平衡集作为新的可复现派生Manifest。

`1:1:1`不改变独立validation/test的封存边界，也不允许从已封存test回流训练。
三语合并必须等西语MFA/Manifest/Temporal 2×完成后才开始，并在配额输出中显式
记录每语言的records、hours、source corpora和采样策略。

## 18. 西语split保留式Temporal 2×合并

西语三套Manifest的原ready合计180.910621小时；Temporal 2×只读审计在唯一
`ctc_length_infeasible`且有效ratio不超过0.90的策略下，可再释放7,740条、
9.465169小时，总候选约190.375790小时。高压力15条和所有其他issue记录保持隔离。

西语不能复用会按`split_hash`重分全部数据的葡语合并模式。构建时必须把Manifest
音频路径与原`source.tsv`元数据关联：Rioplatense和明确拉美辅助池保留既有
`source_split`；SLR61按完整speaker确定性分配；最终再次验证跨split speaker、
audio和ID overlap均为0。产物绑定Temporal 2×，完整西语池保留，后续另行从train
确定性选择150小时参与英西葡`1:1:1`派生集。

## 19. 美式英语speaker候选键审计

英语Temporal 2×可从51条review中安全恢复47条、0.037675小时；4条其他标签问题
保持隔离。原ready加推荐恢复约549.622421小时。

在切分前，全量验证Swift WAV basename的speaker候选键：stem最后一个下划线段作为
utterance ID，其余前缀作为speaker ID。审计必须与完整Manifest逐音频一一关联，
报告前缀字段分布、speaker规模、每speaker时长、跨shard情况、解析失败和重复
speaker+utterance。只有全部关联与唯一性检查通过后才按speaker整体切分；不能仅凭
三个样例文件名宣称speaker-disjoint。

## 20. US-only英语Temporal 2×池

Swift speaker库存审计确认全部389,738条均可稳定解析，但首段分布为
`US=360,043`、`us=54`、`AU=13,640`、`CN=16,001`。本轮美式英语数据边界只允许
首段大小写归一化后等于`us`的speaker；AU/CN原记录不删除、不改标签，只从派生池
排除。该策略必须记录在`split_config.json`和`split_summary.json`中。

英语派生池以完整Manifest和通过的`speaker_inventory.tsv`为输入：原ready直接
保留；review仅在唯一问题为`ctc_length_infeasible`、Temporal 2×后可行且有效ratio
不超过0.90时释放。然后按完整speaker执行确定性96/2/2，验证speaker、audio和ID
跨split overlap均为0，并封存test。英语train达到完整规模后再单独确定性选择约
150小时参与英西葡1:1:1；不能为了平衡语言而改动validation/test或读取封存test。

工作区结果为360,093条、507.959291小时；train/validation/test为
345,820/7,187/7,086条和487.628442/10.165238/10.165611小时。Temporal 2×在US
范围释放26条；AU/CN排除29,641条，其中21条属于可恢复但方言边界不符的记录；
其余4条review继续隔离。1,622名speaker按1,560/31/31分配，三类cross-split
overlap均为0。

## 21. 英西葡150小时1:1:1派生train

三个Temporal 2×完整train池分别为英语487.628442小时、西语181.868358小时、
葡语783.223637小时。平衡派生集对每种语言选择至少150小时，总计约450小时，
不覆盖完整池。

选择先强制纳入西语`slr61`和`common_voice_rioplatense_v26`全部train，再以稳定
SHA256优先级从拉美辅助池补足；英语和葡语按相同稳定优先级在完整train自然来源
分布上抽样。最终合并Manifest按各语言累计已输出音频时长做deficit interleave，
避免连续的大语言块。每个输入train必须SHA匹配、summary计数一致、全部使用
Temporal 2×；跨语言重复ID/音频直接失败。

validation/test继续使用三个独立单语种池的封存版本。派生器只从split summary
记录其路径和SHA，不打开、复制、重采样或合并test内容。下一阶段的Encoder feature
cache只读取新派生train和另行明确的validation策略。
