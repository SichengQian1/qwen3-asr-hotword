# eSpeak-ng 新热词 G2P 选型

## 1. 选型问题

本选型只回答一个问题：为英、西、葡新热词生成音素时，eSpeak-ng 能否替代当前
MFA G2P。它不是 fallback 评测，不训练或修改 CTC Head，不运行 Qwen3-ASR，不接入
4k 热词检索，也不修改英西葡增训数据。

本轮没有预设必须达到的通过阈值。目标是收集可复核证据：

1. eSpeak-ng 和 MFA 对同一批词的转换成功情况；
2. eSpeak-ng 是否产生当前 90 类 CTC 词表外音素；
3. 映射到当前词表后，两者音素序列的一致程度；
4. 哪些语言、词频、词长和拼写类型存在系统性差异。

MFA 是当前训练标签体系的参照，不被视为绝对发音真值。高差异和 OOV 词仍需人工判断
属于表示体系差异、MFA 问题还是 eSpeak-ng 实质性错音。

## 2. 实验边界

- 语言和 eSpeak-ng voice 固定为 `en-us`、`es-419`、`pt-br`；
- 每种语言从已有 train Manifest 中抽取 500 个唯一词，共 1,500 个；
- 只选同时存在唯一 MFA 发音、且 MFA 发音可完整映射到当前 90 类词表的词；
- 输入 Manifest 必须逐行标记为 `split=train`，语言字段必须匹配；
- 当前词表保持
  `configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json`，不因本轮结果修改；
- 输出使用全新忽略目录 `outputs/espeak_mfa_selection_v1`，已存在时拒绝覆盖；
- 不读取 validation/test，不生成训练 Manifest、音频、特征、checkpoint 或模型结果。

抽样种子固定为 `20260829`。每语言 500 个名额按五个主分层各 100 个分配：特殊拼写、
长词、低频、中频和高频。候选不足时只从其余合格词确定性补齐，并在逐词结果中标记
`sampling_stratum=fallback`。所有排序均使用带语言、分层和种子的 SHA256，保证复现。

## 3. 指标和产物

逐词比较先分别把 MFA 和 eSpeak-ng 输出映射到当前 90 类词表，再计算：

- 非空转换、完整词表映射和语言切换标记数量；
- 完全一致率；
- 音素编辑距离、PER 均值/中位数/P90；
- `PER <= 0.1`、`PER <= 0.2`、`PER > 0.3` 比例；
- substitution/insertion/deletion；
- MFA/eSpeak-ng 音素混淆对；
- 词表外 Unicode 单元；
- 按语言、抽样分层、频率、词长和拼写风险聚合的结果。

输出文件：

```text
outputs/espeak_mfa_selection_v1/
  run_config.json
  sampled_words.jsonl
  word_comparisons.jsonl
  summary.json
  language_summary.tsv
  phone_confusions.tsv
  oov_units.tsv
  manual_review.tsv
  sha256.txt
```

`manual_review.tsv`自动包含每种语言 PER 最高的 20 个词，以及全部 OOV/语言切换词；
`review_label`和`reviewer_notes`留给后续人工判断。

## 4. 工作区运行

本任务是 CPU 文本处理，不需要 GPU、完整 Qwen 模型或 MFA 重跑。拉取交付提交后：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

python -m pip install -e ".[eval]"
python -c 'import phonemizer; print(phonemizer.__version__)'
espeak-ng --version
```

`phonemizer`是仓库已有 `eval` extra。若最后一条命令找不到 `espeak-ng`，应先在当前
容器安装系统 eSpeak-ng 包并记录版本；不要用其他 G2P 后端代跑。

固定输入和运行命令：

```bash
BALANCED_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
ESPEAK_SELECTION_ROOT=outputs/espeak_mfa_selection_v1

EN_DICT=outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/swift_us_english_english_us_mfa.v1.dict
ES_DICT=outputs/es_ar_train_sources_v1/repaired_mfa_v1/slr61/slr61_spanish_latin_america_repaired.v1.dict
PT_DICT=outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict

test -f "$BALANCED_ROOT/full_ctc_train_en.jsonl"
test -f "$BALANCED_ROOT/full_ctc_train_es.jsonl"
test -f "$BALANCED_ROOT/full_ctc_train_pt.jsonl"
test -f "$EN_DICT"
test -f "$ES_DICT"
test -f "$PT_DICT"
test ! -e "$ESPEAK_SELECTION_ROOT"

python scripts/compare_espeak_mfa_g2p.py \
  --language-manifest en="$BALANCED_ROOT/full_ctc_train_en.jsonl" \
  --language-manifest es="$BALANCED_ROOT/full_ctc_train_es.jsonl" \
  --language-manifest pt="$BALANCED_ROOT/full_ctc_train_pt.jsonl" \
  --mfa-dictionary en="$EN_DICT" \
  --mfa-dictionary es="$ES_DICT" \
  --mfa-dictionary pt="$PT_DICT" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --sample-size 500 \
  --seed 20260829 \
  --output-dir "$ESPEAK_SELECTION_ROOT"
```

本实验不支持 `--resume`。执行前失败不会创建正式输出目录；写出阶段意外中断时保留
现场并先报告，不要覆盖或手工拼接结果。重新运行必须换一个新目录，或在确认失败目录
只属于本实验后由用户自行处理。

## 5. 结果核验与回传

```bash
(cd "$ESPEAK_SELECTION_ROOT" && sha256sum -c sha256.txt)

jq '{status, total_sampled_words, by_language, source_stats}' \
  "$ESPEAK_SELECTION_ROOT/summary.json"

wc -l \
  "$ESPEAK_SELECTION_ROOT/sampled_words.jsonl" \
  "$ESPEAK_SELECTION_ROOT/word_comparisons.jsonl"
```

两个 JSONL 都应为 1,500 行，summary 应为 `status=completed`、
`test_set_used=false`。本次后续分析需要逐词差异，输出目录整体体积应很小；请回传上述
九个文件，不需要返回任何训练数据、MFA模型、Qwen模型、checkpoint、音频或特征缓存。

## 6. 当前状态

截至 2026-08-29，独立代码、mock 测试、命令和结果契约已经完成；工作区 1,500 词
实测尚未运行。没有 eSpeak-ng 实测结果前，不能得出其能够或不能替代 MFA 的结论。
