# 西语G2P三类差异交接（2026-10-23）

## 交付目的

交付现有比较结果及全部对应标签明细，供接手同事处理。本次不再追加根因实验，不修改C++、模型或原训练标签。详细评估背景见`MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md`；本文件为简版清单交付说明。

比较范围是实际西语480小时训练域：296,399句、92,715个去重单词、144,499个来源×词条、3,517,234次词出现。参考是项目原生产词典标签，C++使用同一套官方西语模型。标签不同不等于C++发音一定错误。

## 三类问题及明细文件

| 类别 | 来源×词条数 | 问题说明 | 完整明细 |
|---|---:|---|---|
| 1. 缺少已有西语音素清洗 | 预期46,291 | 原C++标签不一致/不可用，补项目已有西语音素清洗后与旧标签一致；例如鼻化清洗。此规则还未写回C++。 | `01_cleanup_resolved.jsonl` |
| 2. 原词生成失败 | 9 | 含ć、ś、ń的9个词，训练中共出现11次；历史流程均为mfa_proxy＋strip_acute。C++直接输入原词失败，尚未测试其代理输入。 | `02_cpp_failed.jsonl` |
| 3. 清洗后标签仍不同 | 4,603 | C++能给出可用标签，但补已有西语清洗后仍与原训练标签不同；训练中共出现60,507次。交付全部差异，不预设具体根因。 | `03_labels_different.jsonl` |

第二、三类合计4,612个来源×词条，占144,499的3.1917%；按词出现次数加权为60,518/3,517,234＝1.7206%。

“来源×词条”按`source_tsv + word`区分，不是去重单词数。同一个词在不同来源的标签不能合并或覆盖。第一类完整条数以导出结果核验；三类预期合计50,903条。

每个JSONL保留原逐词记录，重点字段如下：

| 字段 | 含义 |
|---|---|
| `word` | 单词 |
| `source_tsv` / `source_build_config` | 训练来源与构建配置 |
| `occurrences` | 在本次实际训练清单中的出现次数 |
| `saved_reference.raw_ipa` / `token_ids` | 原训练词典IPA及90类ID |
| `cpp.native.raw_ipa` / `token_ids` | 原C++输出IPA及ID |
| `cpp.existing_es_cleanup.raw_ipa` / `token_ids` | 额外补已有ES音素清洗后的IPA及ID |
| `status.native` / `status.existing_es_cleanup` | 对应标签的比较状态 |

`existing_es_cleanup`是诊断适配分支，不包含历史拼写代理输入。第二类原C++发音为空是失败事实，不补造标签。

## 从已有工区结果导出，不重新推理

原目录：`outputs/mmservice_g2p_training_393ce786a9`。完整词级文件在工区，本地小回传包只有汇总与示例，不能用小包冒充全部差异。以下在容器仓库根目录执行，仅需要既有`jq`。

先核验两份固定输入，均显示OK后再执行下一段：

```bash
G2P_RUN=outputs/mmservice_g2p_training_393ce786a9
printf '%s  %s\n' 29d1c2f990500e317cdac75bbf9af3f1cb80d23666ab4717c93b1d557ad5b109 "$G2P_RUN/report.json" 6533f2e153c428251fb80a375f7213ca02932010a5c172377a5d2789da54b74a "$G2P_RUN/words.jsonl" | sha256sum -c -
```

自动新建目录，保留原记录全部字段：

```bash
D=$(mktemp -d outputs/spanish_g2p_handoff_XXXXXXXX)
printf '1/3 导出补清洗后已一致的条目\n'
jq -c 'select(.language=="es" and .status.native!="equal" and .status.existing_es_cleanup=="equal")' "$G2P_RUN/words.jsonl" > "$D/01_cleanup_resolved.jsonl"
printf '2/3 导出生成失败的条目\n'
jq -c 'select(.language=="es" and .status.existing_es_cleanup=="cpp_failed")' "$G2P_RUN/words.jsonl" > "$D/02_cpp_failed.jsonl"
printf '3/3 导出清洗后仍不同的全部标签\n'
jq -c 'select(.language=="es" and .status.existing_es_cleanup=="different")' "$G2P_RUN/words.jsonl" > "$D/03_labels_different.jsonl"
cp docs/MMService_G2P_DIFFERENCES_DELIVERY_2026-10-23.md "$D/README.md"
wc -l "$D"/*.jsonl
printf '交付目录：%s\n' "$D"
```

预期三份文件分别为46,291、9、4,603行。若不符，先保留实际输出，不修改文件凑数。

如同事需要直接查看表格，可再生成对应TSV（JSONL中的完整ID等字段仍保留）：

```bash
for f in "$D"/*.jsonl; do
  jq -nr '["word","occurrences","source_tsv","old_ipa","cpp_native_ipa","cpp_cleaned_ipa"], (inputs | [.word,.occurrences,.source_tsv,.saved_reference.raw_ipa,.cpp.native.raw_ipa,.cpp.existing_es_cleanup.raw_ipa]) | @tsv' "$f" > "${f%.jsonl}.tsv"
done
```

将打印出的整个新目录交给同事即可，包含本说明、三份完整JSONL及可选TSV。无需回传这些大明细给本聊天；如需核验完成状态，只需返回`wc -l`结果。导出不读WAV、不跑G2P/CTC/Encoder、不使用GPU，无需恢复或覆盖任何原输出。
