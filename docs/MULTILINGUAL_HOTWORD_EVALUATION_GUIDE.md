# 三语热词表与检索评测使用说明

更新日期：2026-10-22。适用范围：联合4000词表，英语6000条、西语400条、葡语400条，
合计6800条。本文不包含另做的Ziyan实验。

## 1. 评测范围与版本

当前检索链路为：

```text
音频 → Qwen3-ASR Encoder → CTC音素预测
     → 三语热词检索 → 排序和质量过滤 → Top5 / Top7
```

衡量的是：参考文本中应出现的目标热词，有多少进入了交付候选。
每条音频搜索完整三语联合词表，不按音频语言预先筛成单语词表。

这些检索任务没有运行Qwen的LLM转写。热词检索召回率、CTC音素错误率PER和最终
转写错误率WER是三个不同指标。

当前最新三语完整结果是最终epoch27 Head的带噪前缀检索。历史干净三语整段结果
使用旧480h Head，英语另有epoch21干净预览，不能直接当成epoch27的干净对照。

## 2. 仓库、词表与模型

### 仓库与代码

```text
容器内：/host_home/star/q00933266/qwen3-asr-hotword
容器外：/home/star/q00933266/qwen3-asr-hotword
Git分支：codex/g2p-coverage-scan
```

在对应仓库根目录拉取说明及代码：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

重现实验还应记录该次运行的Git、配置、Head和词表身份；拉取新文档不要求重跑已完成
实验。旧任务的严格恢复以其封存代码/输入身份为准，不应通过删除校验或覆盖目录续跑。

### 最终三语词表

建议以实际评测使用的封存副本为准：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/reviewed_joint_4000_retrieval_480h_v1/reviewed_keyword_table.json
```

对应构表产物：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/reviewed_joint_4000_keywords_v1/keyword_table.json
```

两者应对应同一版内容，固定SHA256为：

```text
6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b
```

共有4000个规范化词面、4012条带语言标签的发音记录。按语言统计为英语1640、
西语1197、葡语1175；跨语言同形词合并后为4000词面。

表中包含目标热词和填充词。填充词参与检索竞争，但不自动进入目标召回率分母。
不要用旧英语3000词表、以前单独的西葡4000词表，或新生成的MFA/C++词典替代这份
固定词表。部分填充词是在这批参考文本上排除重合后选出的，因此这套评测不能直接
视为未知参考条件下的盲词表泛化评估。

### 当前最终Head和音素表

```text
Head：
/host_home/star/q00933266/qwen3-asr-hotword/outputs/english_countries_refresh_newval_v1/head/ctc_head_best.pt

最佳轮次：epoch27
Head SHA256：
872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466

90类音素表：
/host_home/star/q00933266/qwen3-asr-hotword/configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
音素表 SHA256：
0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4

完整Qwen模型：
/glusterfs_103/models/Qwen3-ASR-1.7B
```

完整模型在H200工区使用，不需要下载到本地讨论机器。

## 3. 干净与带噪评测集路径

以下路径来自配置和历史封存记录；工区目录若再次迁移，应以实际运行清单核对。

### 英语干净版：三国各2000条

```text
/host_home/star/两波验证集-可直接推理/en_chile/en/wav
/host_home/star/两波验证集-可直接推理/en_colombia/en/wav
/host_home/star/两波验证集-可直接推理/en_mexico/en/wav
```

参考文本为各自同级的`transcripts.txt`。

### 英语带噪版：三国各2000条

```text
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_chile/en/wav
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_colombia/en/wav
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_mexico/en/wav
```

### 西语与葡语干净版：每个wave、每种语言100条

原封存读取位置在仓库下：

```text
/host_home/star/q00933266/qwen3-asr-hotword/wave1/{es,pt}/wav
/host_home/star/q00933266/qwen3-asr-hotword/wave2/{es,pt}/wav
/host_home/star/q00933266/qwen3-asr-hotword/wave3/{es,pt}/wav
/host_home/star/q00933266/qwen3-asr-hotword/wave4/{es,pt}/wav
```

`{es,pt}`是两个语言目录的简写，不是字面目录名。西语合计400条、葡语合计400条。
参考文本为各语言目录下的`transcripts.txt`。如果干净目录也被重命名，需要先核对
旧运行的`run_config.json`，不能直接套用带噪目录名。

### 西语与葡语带噪版：当前已确认的目录名

```text
/host_home/star/两波验证集-可直接推理/2026-09-29-带噪10dB/wave1_标贝/{es,pt}/wav
/host_home/star/两波验证集-可直接推理/2026-09-29-带噪10dB/wave2_澳鹏/{es,pt}/wav
/host_home/star/两波验证集-可直接推理/2026-09-29-带噪10dB/wave3_海天/{es,pt}/wav
/host_home/star/两波验证集-可直接推理/2026-09-29-带噪10dB/wave4_澳鹏0921/{es,pt}/wav
```

带噪评测沿用原封存音频ID、参考文本和目标集合，不用带噪目录另附的文本或词表重新
定义答案。参考文本只用于资产核验及检索后评分，不输入前缀模型或候选生成。
“10dB”是供方目录标称，本项目没有独立复测实际SNR。

## 4. Top5、Top7召回率口径

先将参考文本与该数据组的固定目标集合做整词/连续短语匹配，得到每条音频的预期
目标；再与最终交付候选比较：

```text
TopK召回率
= 进入最终TopK候选的目标词×音频对数量
÷ 参考中全部预期目标词×音频对数量
```

例如一句话包含3个不同目标词，Top5命中其中2个，则该音频贡献分母3、命中2。
同一目标词在一句话中重复多次，只计一个词×音频对。

| 语言 | 音频数 | 固定目标词×音频对分母 |
|---|---:|---:|
| 英语 | 6000 | 4374 |
| 西语 | 400 | 724 |
| 葡语 | 400 | 736 |

Top5/Top7是通过质量过滤后最多交付5/7个不同词面，不强行凑满。当前主要条件为：

```text
score ≥ 0.75
edit_ratio ≤ 0.35
posterior_confidence ≥ 0.5
```

Anchor先保留最多64个发音候选，再进行词面合并及筛选；64不是最终交付词数。
posterior是匹配片段的音素置信度，不是词级存在概率。比较实验时保持同一评分、
门控、词表和目标集合，不能将过滤前raw召回与过滤后交付召回混用。

## 5. 评测脚本使用方法

以下命令均在容器内仓库根目录、原Qwen运行环境执行：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
```

GPU编号需换成实际分配的物理卡。示例输出目录必须是未存在的新目录，不覆盖旧结果。
这些入口依赖已有封存资产及身份校验，并非只指定任意WAV目录就能运行的通用脚本。

### 5.1 当前主要入口：epoch27、三语带噪前缀检索

```bash
python -B scripts/run_multilingual_stream.py \
  --effect same-step --gpu 4 \
  --output-dir outputs/colleague_ep27_noise_stream_v1
```

配置为`configs/multilingual_stream_contexts.workzone.json`，覆盖全部6800条音频。
每隔2秒使用当前已收到的累计音频前缀重新检索，最后不足2秒的尾段也保留：

```text
0～2秒 → 一组Top5/Top7
0～4秒 → 一组Top5/Top7
……
0～实际结尾 → 最后一组Top5/Top7
```

同一步检索同时导出两个K值，不为Top5、Top7分别跑Encoder。候选可以随时间出现或
消失，不跨步永久累加。`same-step`指当前前缀结果供当前步使用；本脚本生成供下游
回放的热词时间线，不启动下游流式ASR服务。

中断后，在代码、配置及所有输入身份保持一致时，重复原命令并加`--resume`：

```bash
python -B scripts/run_multilingual_stream.py \
  --effect same-step --gpu 4 \
  --output-dir outputs/colleague_ep27_noise_stream_v1 --resume
```

命令自带预检、实际样本/前缀计数、耗时及活动提示。`--audit-only`可单独进行CPU
预检，不写推理结果；正常运行已包含预检，无须重复执行。

### 5.2 epoch27、仅英语带噪整段检索

```bash
python -B scripts/run_english_best_noise.py \
  --gpu 4 \
  --output-dir outputs/colleague_ep27_noise_whole_v1
```

这个入口一次输入完整音频，只测英语6000条，同样搜索完整三语4000词表。
配置为`configs/english_best_noise10db_4k.workzone.json`。中断恢复使用同一命令加
`--resume`，身份不同会拒绝复用。

### 5.3 历史干净三语整段入口

```bash
python -B scripts/run_reviewed_target_retrieval.py \
  --config configs/reviewed_joint_4000_retrieval.workzone.json \
  --gpu 4 \
  --output-dir outputs/colleague_clean_baseline_v1
```

此配置绑定旧480h Head，不是最终epoch27；英语干净集另有epoch21预览记录。
要严格比较干净/带噪效果，需要固定同一Head、词表、目标集合和参数，另建配置及
输出目录。不能直接把现有不同Head结果的差值称为噪声损失。

## 6. 结果位置与查看命令

### 最新已完成的三语带噪结果

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step
```

在仓库根目录查看精简汇总：

```bash
R=outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step
cat "$R/terminal_summary.json"
```

只查看三语末步Top5/Top7指标（需要`jq`）：

```bash
jq '.by_language | map_values({
  top5: .RAG_top5_stream.last_context,
  top7: .RAG_top7_stream.last_context
})' "$R/terminal_summary.json"
```

| 字段 | 含义 |
|---|---|
| `expected_target_sample_pairs` | 目标分母 |
| `target_hits` | 命中的目标词×音频对 |
| `target_recall` | 召回率，乘100得到百分比 |
| `selected_word_sample_pairs` | 实际交付的词×音频对数量 |
| `words_absent_from_full_reference` | 未在完整参考文本中匹配到的交付词数量 |

流式结果有两种视图：

- `last_context`：最后一步实际交付列表的召回率；当前same-step最后一步输入完整
  音频前缀。它本身不能说明热词是否赶在ASR解码对应词之前送达。
- `ever_delivered`：整条时间线上曾经交付过的词的覆盖率。跨步并集可能超过K，
  不能称为固定TopK召回，也不等于及时召回。

完整参考外输出不直接等于下游ASR错误或该前缀的错误。参考内非目标词也不应被
一律判错，不能用“输出数减目标命中数”直接当作误匹配数。

### 历史干净三语整段结果

```bash
R=outputs/reviewed_joint_4000_retrieval_480h_v1
jq '.by_language | map_values(.current)' "$R/report.json"
```

其中`top5/top7.gated_target_recall`是过滤后实际交付召回率，`raw_target_recall`是
过滤前口径。英语独立预览/带噪整段报告的汇总在根字段`.current`，而非
`.by_language.en.current`，例如：

```bash
jq '.current' outputs/english_best_ep27_noise10db_4k_v1/report.json
```

逐组结果看`groups`，身份及输入清单看`run_config.json`。不要将报告中的旧基线
字段当成本轮新结果。新的自定义输出目录应相应替换上述路径。

## 7. 目前已返回的最新三语结果

epoch27、联合4000词表、标称10dB带噪、same-step前缀检索的末步结果：

| 语言 | Top5命中/分母 | Top5召回 | Top7命中/分母 | Top7召回 |
|---|---:|---:|---:|---:|
| 英语 | 3654/4374 | 83.54% | 3698/4374 | 84.55% |
| 西语 | 616/724 | 85.08% | 638/724 | 88.12% |
| 葡语 | 529/736 | 71.88% | 553/736 | 75.14% |

数字来自已返回并归档的工区终端摘要，本地归档为
[`results/2026-10-21_multilingual_stream_ep27_noise10db_terminal_summary.json`](results/2026-10-21_multilingual_stream_ep27_noise10db_terminal_summary.json)。
已复算汇总计数和比率，未据此声称独立核验了工区全部逐样本原件。

Top7增加召回的同时也可能增加干扰词，实际收益需要下游无RAG/Top5/Top7转写对照。
这些检索指标不是最终ASR WER。

## 8. 给下游交付的文件

最新前缀检索结果目录的`delivery/`下包含：

```text
stream_contexts.json
durations.json
audio_manifest.json
```

Top5和Top7都在同一份`stream_contexts.json`中，对应两个顶层字段：

```text
RAG_top5_stream
RAG_top7_stream
```

结构为：

```text
模式 → 数据集 → 音频ID → 各时间点的候选列表
```

`durations.json`提供真实音频时长，`audio_manifest.json`提供音频路径和身份。
`prefix_shards/`保留逐前缀CTC、候选、得分等诊断信息，不是日常下游交付文件。
逐样本计量在`sample_metrics.jsonl`，按数据组的指标也保留在`report.json`中。

历史整段检索按语言和数据组交付`*_top5.json`、`*_top7.json`，结构为音频stem到
`[{word, phoneme}]`的映射，与上述时间线格式不同。同名stem在不同数据组间必须
隔离，不能直接混合成一个字典。

有关前缀时序、下游key和接口契约的细节见[`STREAM_CONTEXTS.md`](STREAM_CONTEXTS.md)；
实验节点、身份和证据等级见[`HANDOFF.md`](HANDOFF.md)的0.170–0.174。
