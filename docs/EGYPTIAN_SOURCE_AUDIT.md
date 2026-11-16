# 纯埃及方言新来源：先核实小时数

2026-11-16。用户将本轮阿语目标改为纯埃及方言，训练和验证均不再用MSA补样。
open-source优先，gaoshan与coarse作为补充。当前只检查来源和时长，等用户确认容量后再决定选样。
历史混合EGY/MSA的label/freeze命令不适用于这个新目标，保留其代码和输出供追溯。

## 当前标注与音素表状态

阿语发音转换修复已在提交035960c中实现，部分规则仍需听审；不是已经在工区完成新标签验证。
167类 `en_ar_native_ipa_vocab.v1.1.json` 没有修改，英语ID也没有修改。
本次检查不执行该前端，不生成音素标签，不下载模型，不运行Encoder或训练。

## 数据范围

配置 `configs/egyptian_sources_audit.workzone.json` 固定用户提供的埃及目录。
扫描以下目录直属的所有 `.json` / `.jsonl`，并核对用户已列出的14份文件是否齐全：

| 组 | 文件数（预期） | 用途 |
| --- | ---: | --- |
| open-source | 4 | 优先训练候选池 |
| gaoshan/train | 4 | 补充候选池 |
| coarse/train | 5 | 补充候选池 |
| gaoshan/test | 1 | 原测试集，单独统计、保留身份 |

新增文件会纳入并报告；缺目录、缺预期文件或解析失败返回`status: review`，不能把不完整统计当全部容量。
`dialect: EGY`和目录只是来源声明，脚本不据此宣称音频已核实为埃及方言。

## 执行

原qwen环境、容器内：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -u -B scripts/audit_egyptian_sources.py --workers 32
```

不需要GPU。输出固定为 `outputs/egyptian_sources_audit_v1`，首次执行拒绝覆盖已有目录。
每个文件显示实际已检查行数与10秒心跳。只读音频头，不解码整段声音、不做音频内容SHA。
JSON文件本身做SHA身份核验，音频头检查每1024行封存一个分片。

中断恢复：

```bash
python -u -B scripts/audit_egyptian_sources.py --workers 32 --resume
```

允许调整worker数量，不允许改变源文件/配置/代码后混用旧检查分片。
续跑会重读文本索引和核验已完成分片，但不重读已完成分片的音频头。
它恢复的是同一轮审计快照；如果后来修了挂载、改了音频、增添了源文件，请用新的`--output-dir`重新审计。
已有原始音频、标签、缓存和训练结果均不改。

## 结果怎么看

`report.json`包含逐文件、逐组与总计：

- `by_group.*.readable_hours`：按每条记录累计的实测音频头时长，**包含重复记录**。
- `totals.readable_hours_including_duplicates_and_original_test`：全部组原始累计时长。
- `totals.path_unique_hours_including_original_test`：按解析后的音频路径去重的总时长，含原测试集。
- `totals.path_unique_train_pool_hours_excluding_original_test`：去路径重复且排除原test路径后的候选池时长。
- `path_unique_train_pool_attribution`：跨来源重复路径优先归属open-source；同优先级采用固定遍历顺序，
  这只是统计归属，不是已经完成质量筛选或训练选样。
- 重复路径额外行数、训练池与原test重合路径数、同路径不同文本数量，以及缺失/坏头文件数量。

这些数字都不是“最终合格训练小时”：尚未按音频内容去重、核实方言/音文一致性、G2P覆盖和CTC长度。
路径不同的同一录音暂不能识别。原test与训练池重合的路径无论出现在哪个来源，都从候选池净时长扣除。

文本方面同时支持带`language Arabic<asr_text>`前缀和纯文本response，保留两份原文，不生成训练用text。
`text_relation_counts/hours`区分完全相同、仅标点空格、阿文拼写归一化后相同、内容不同、字段缺失。
诊断归一化包括hamza/alef、ى/ي、ة/ه等有限替换；它仅用于分类，**不作为发音打标文本**。
`strict_text_agreement_hours`仅表示头可读、文本严格一致且无结构问题，也不是人工正确率。
不使用用户样例中的`lengths`字段推算音频秒数。

## 回传

只需回传：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/egyptian_sources_audit_v1/return_summary.zip
```

里面是`report.json`、输入契约和少量文本差异例子；无模型、无音频。
全量`source_inventory.jsonl`与可恢复分片留工区。也可直接粘贴下面的精简汇总：

```bash
jq '{status,by_group,totals,path_unique_train_pool_attribution}' outputs/egyptian_sources_audit_v1/report.json
```

此阶段只回答“这批来源实际有多少可读时长、哪些数据需要处理”。
确认后再把审核过的新清单接入埃及前端打标和纯埃及划分，不调用旧MSA补样逻辑。
