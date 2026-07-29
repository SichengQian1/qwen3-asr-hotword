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

MFA 环境和巴西葡萄牙语 G2P 模型：

```text
Conda env: aligner
Model:
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/g2p/portuguese_brazil_mfa.zip
```

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

## 10. 待处理：Noah 200 小时巴葡金融口语数据

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

在首批 TSV 审计实际通过前，候选音频根目录不能视为已经确认。第一轮只运行：

```bash
python scripts/audit_training_tsv.py \
  --tsv "/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv" \
  --audio-root /host_home/z00841352/27A/data/Noah_espt/noah_pt \
  --max-records 1000 \
  --sample-count 5 \
  --output outputs/noah_pt_finance_200h/audit_first_1000.json
```

确认字段为 `audio,text` 且音频正确解析后，再依次执行：

1. 全量 TSV 审计；
2. 为这套语料独立生成唯一词表和词频；
3. 使用 `portuguese_brazil_mfa.zip` 生成独立 MFA 词典；
4. 对词典覆盖率和 v0.2 phone OOV 做独立审计；
5. 构建完整 `train_ready.jsonl` 和 `needs_review.jsonl`。

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
