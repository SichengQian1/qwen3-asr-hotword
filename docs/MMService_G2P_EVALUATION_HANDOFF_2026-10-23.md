# MMService C++ G2P替代评估与交接（2026-10-23）

## 1. 当前结论与任务边界

目标是判断组内C++ G2P程序能否替代本项目现用MFA G2P及其必要的前后处理。
当前比较使用相同的三套官方模型，不是训练一个新的G2P模型，也没有重新训练CTC。

- **英语、巴葡：在本次实际480小时训练清单覆盖的规范单词输入上，C++输出与原训练标签全部一致，整句拼接标签也全部一致。**这是这批有限输入上的兼容性结果。
- **西语：C++原生输出不能直接替换项目现有流程。**额外应用已有西语音素清洗后，来源×词标签差异率由35.2272%降为3.1917%，按词出现次数加权的差异率由23.7538%降为1.7206%。
- 西语剩余差异中，有4,603个来源×词条能生成可用标签但与旧标签不同；另有9个词生成失败。后者的历史路径现已确认：全部使用去附加符号的代理词，再调用MFA生成标签。
- **尚未证明C++接收这9个代理词后能复现原标签；也尚未解释全部4,603个来源×词条的差异。**不能把已经找到的代理规则来源直接当作修复后的实测结果。
- 报告的`replacement_verified: false`保留。尚未进行生产替换，也没有以本次数据证明任意新热词、原始整句前处理或完整服务行为都等价。

本次“全量”指**实际入选训练的三语各约480小时**，不是全部原始语料、未入选池、验证集或测试集。指标是标签兼容性，不是发音准确率、CTC PER、热词检索召回率或最终ASR WER。

## 2. 环境、代码与固定资产

| 项目 | 路径或状态 |
|---|---|
| 仓库 | `https://github.com/SichengQian1/qwen3-asr-hotword` |
| 分支 | `codex/g2p-coverage-scan` |
| 容器内仓库根目录 | `/host_home/star/q00933266/qwen3-asr-hotword` |
| 容器外仓库根目录 | `/home/star/q00933266/qwen3-asr-hotword` |
| 原C++交付目录 | `mm/` |
| 独立x86构建目录 | `outputs/mm_g2p_x86_2LrhDuRR/` |
| 本次成功运行 | `outputs/mmservice_g2p_training_393ce786a9/` |
| 早先失败运行，保留 | `outputs/mmservice_g2p_training_2a281eb184/` |
| 当前审计代码提交 | `b326a6f5fbc27494bbbf567d6eac8727b4317b3d` |

上述SHA是运行审计代码的身份，也是本次归档修改前核实的本地/远端HEAD；**不代表新增本文后的最新提交**。归档提交身份以本轮交付回复及`git log -1 -- docs/MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md`为准。

工区为x86_64，原交付二进制为AArch64，不能直接运行。用户已在新目录以原源码重新编译为x86_64，并验证动态依赖可解析：G++ 13.3.0、OpenFST 1.8.4、`libfst.so.26`；`aligner`环境为MFA 3.4.0、Pynini 2.1.7。未覆盖`mm/`中的原二进制。

| 固定输入 | SHA256 |
|---|---|
| `models/mfa/g2p/english_us_mfa.zip` | `9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02` |
| `models/mfa/g2p/spanish_latin_america_mfa.zip` | `58e0743cf364d8aa2cc8b419e7c95dd9127908268af670f305722c601f726b2a` |
| `models/mfa/g2p/portuguese_brazil_mfa.zip` | `11e7f72ecccbae7905fa064017ac1a1fead9b6ebfdde4e23e506c28edfbcbcc8` |
| `outputs/mm_g2p_x86_2LrhDuRR/mmservice_g2p` | `d6596db29acec77bee00df8fb999e2e2352d3d9f0acc3067c0e324be3c98f094` |
| `configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json` | `0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4` |
| `outputs/english_countries_refresh_fast_v1/frozen/full_ctc_train.jsonl` | `10fdbdbbdb8ef352cd7a143b742fd962627b40a98a44d0b54701311f0b94fbfe` |

模型ZIP、二进制和大训练清单身份来自工区封存报告；未把这些大文件回传或在本地加载。英语模型、西语模型为Phonetisaurus架构，C++使用模型内嵌输入符号表构建单字母/双字母输入；巴葡为Pynini模型，无内嵌输入符号表，C++使用Unicode码点输入。三者的模型文件均未修改。

## 3. 必须区分的比较对象

| 对象 | 定义 | 能回答什么 |
|---|---|---|
| 官方默认流程 | 当前MFA CLI默认前处理、模型推理和Top1输出 | 当前默认调用会得到什么；可能存在缺词或输出键变化 |
| 官方同输入控制支路 | 原MFA重写算法、相同FST/符号，明确关闭该支路的metadata字符过滤和NFKD | 在固定输入政策下与C++进行推理对照；不能称为默认CLI策略 |
| 项目生产参考 | 历史MFA词典，加已有缺词代理、西语发音清洗和90类映射 | 对既有训练标签是否兼容，不自动代表真实发音正确 |
| C++ native | 原C++默认前处理、模型推理、通用IPA后处理后的输出 | 原生程序目前提供的行为 |
| `existing_es_cleanup` | 在C++ native之上额外调用已有西语发音清洗 | 调用方补已有音素清洗后的兼容性；不是C++原生结果 |

C++确实包含文本前处理和IPA后处理，但没有完整复刻项目的Python casefold、数字审核、90类OOV门控、西语缺词代理和专项发音清洗。`--no-postprocess`只关闭IPA后处理，仍保留输入前处理。

本次全训练域审计**没有重新运行官方MFA**。其参考是原生产词典及固定训练标签，而非每个原词重新调用官方模型的结果。审计先用各来源词典逐句重建标签，再与实际训练清单中的ID核对；881,749句全部一致，支持本轮词级参考归属。

审计按语言对项目已规范化的单词去重后调用C++，再按来源词典逐词比较并拼接整句ID。没有直接让C++处理原始整句，因此整句拼接一致也不等于两套原始整句前处理等价。

`existing_es_cleanup`在本轮只补发音清洗，没有把历史`strip_acute`等拼写代理应用到C++输入。巴葡不套西语去鼻化规则。

## 4. 实际三语480小时训练域的全量结果

### 4.1 输入规模与来源归属

| 语言 | 训练句数 | 小时 | 语言内唯一单词 | 来源×词条 | 词出现次数 |
|---|---:|---:|---:|---:|---:|
| 英语 | 263,609 | 480.001615 | 12,844 | 12,844 | 2,177,358 |
| 西语 | 296,399 | 480.001823 | 92,715 | 144,499 | 3,517,234 |
| 巴葡 | 321,741 | 480.000408 | 101,940 | 161,686 | 4,903,520 |
| 合计 | 881,749 | 约1,440 | 207,499（分语种计数之和） | 319,029 | 10,598,112 |

同一个词可以出现在多个来源词典中，所以“来源×词条”不等于去重词数。西语/葡语保留来源区分，不能按多数票合并同词的不同历史标签。输入绑定10份原Manifest构建配置（EN 1、ES 4、PT 5）。

### 4.2 标签差异率

以下差异均包含“不同标签”和“不可用/生成失败”，分母是有效参考标签；本轮所有参考均可用。

| 语言 | native：来源×词差异 | 补现有ES清洗：来源×词差异 | native：按词出现次数加权差异 | 补现有ES清洗：按词出现次数加权差异 |
|---|---:|---:|---:|---:|
| 英语 | 0 / 12,844＝0% | 0 / 12,844＝0% | 0 / 2,177,358＝0% | 0 / 2,177,358＝0% |
| 西语 | 50,903 / 144,499＝35.2272% | 4,612 / 144,499＝3.1917% | 835,476 / 3,517,234＝23.7538% | 60,518 / 3,517,234＝1.7206% |
| 巴葡 | 0 / 161,686＝0% | 0 / 161,686＝0% | 0 / 4,903,520＝0% | 0 / 4,903,520＝0% |

| 语言 | native：整句标签差异/不可用 | 补现有ES清洗：整句标签差异/不可用 |
|---|---:|---:|
| 英语 | 0 / 263,609＝0% | 0 / 263,609＝0% |
| 西语 | 278,403 / 296,399＝93.9285% | 54,052 / 296,399＝18.2362% |
| 巴葡 | 0 / 321,741＝0% | 0 / 321,741＝0% |

这里的句子差异是“只要整句有任何一处ID不同或不可用，就计一条”，不是句子识别错误率。词差异率也不是音素编辑距离PER。

### 4.3 西语差异分解

| 来源×词状态 | native词条数 | 补现有ES清洗后词条数 | native词出现次数 | 补现有ES清洗后词出现次数 |
|---|---:|---:|---:|---:|
| 标签一致 | 93,596 | 139,887 | 2,681,758 | 3,456,716 |
| 可用但标签不同 | 35,642 | 4,603 | 664,998 | 60,507 |
| OOV或空标签 | 15,252 | 0 | 170,467 | 0 |
| C++生成失败 | 9 | 9 | 11 | 11 |

补清洗后整句状态为：242,347句一致、54,041句不同、11句不可用。9个失败词只占全部西语词出现次数的0.000313%，不能解释主要标签差异。

小包包含39条差异示例，全部是西语。其中19条native差异可被现有清洗消除；另20条补清洗后仍不同。这些示例按词频优先选择，不是随机抽样，不能直接外推全部差异的构成。

可被清洗消除的例子：`en`原训练标签为`e n`，C++为`ẽ n`；`con`原标签为`k o n`，C++为`k õ n`。含`ã`的输出还会触发当前90类映射的OOV。

仍不同的例子：`día`旧标签`d̪ j a`、C++清洗后`d̪ i a`；`río`旧标签`r j o`、C++清洗后`r i o`；另见`álbum`的`n/m`及`allí`的`l ʎ/ʝ`差异。此前4k同输入控制实验中，`día`和`río`的官方控制支路输出与当前C++一致，支持“至少部分差异来自历史标签生成政策”的判断；不能把全部4,603项都据此归因。

## 5. 新补齐证据：9个失败词全部来自历史MFA代理路径

用户先从本次`words.jsonl`返回失败词、训练出现次数及原标签，再从历史修复计划返回`resolution/proxy/rules`。9项全部为`mfa_proxy + strip_acute`：

| 原词 | 训练出现次数 | 历史代理词 | 原训练词典IPA |
|---|---:|---|---|
| budućnost | 3 | buducnost | b u ð u ɣ n o s t̪ |
| andrić | 1 | andric | a n d̪ ɾ i k |
| ibrahimović | 1 | ibrahimovic | i β ɾ a j m o β i k |
| kościuszko | 1 | kosciuszko | k o s i u s s k o |
| mihailović | 1 | mihailovic | m i a j l o β i k |
| modrić | 1 | modric | m o ð ɾ i k |
| radić | 1 | radic | r a ð i k |
| stepanović | 1 | stepanovic | s t̪ e p a n o β i k |
| toruń | 1 | torun | t̪ o ɾ u n |

全部来源为`outputs/es_latam_cv_auxiliary_170h_v1/source.tsv`。本次返回的历史`corpus_count`也为3/其余各1，但其定义是历史源词频，不能在其他词上自动当作当前训练词频。

历史修复路径由以下文件定位：

```text
outputs/es_latam_cv_auxiliary_170h_v1/repaired_mfa_v1/es_latam_cv/finalize_summary.json
  -> repair_plan_path
  -> 原词的 resolution=mfa_proxy, proxy=<上表>, rules=strip_acute
```

生产规则位于`src/qwen_hotword/training/spanish_mfa_repair.py`：`spanish_g2p_proxy`对NFD分解后的U+0301做移除，再NFC组合。此规则不仅影响元音，也会把`ć/ś/ń`转换为`c/s/n`。`finalize_spanish_mfa_repair`对`mfa_proxy`从代理词典取发音，完成已有发音清洗及90类审核后，以原词为键保存。

因此这9项实际比较的是：

```text
历史项目：原词 -> 去附加符号代理词 -> 官方MFA -> 已有清洗 -> 原训练标签
当前C++：原词 -> 通用前处理 -> 同一官方FST -> 生成失败
```

**这9项不是同一输入下官方MFA成功、C++失败的直接证据。**C++保留这些字母送入逐词生成；当前证据尚未逐项展示FST符号表/失败内部路径，也没有对这9个代理词重新运行C++。不能把缺少输入补救规则扩大成已确认的FST核心算法缺陷。

9项原IPA及修复计划均来自用户终端转贴，尚无原计划文件SHA回传；本地已实际调用现有代理函数，确认9项代理映射与转贴一致。不要把此核对说成本地重读了工区整份历史计划或已验证代理发音正确。

终端筛选还带回表头和`word ... base_exact word`这一普通词条；后者只是查询把`word`也作为匹配词造成的附带结果，**不是第10个失败词**。

## 6. 与此前三语联合4000热词表审计的关系

此前已完成`outputs/mmservice_g2p_full_v1_553359315a/`，源表为`outputs/reviewed_joint_4000_retrieval_480h_v1/reviewed_keyword_table.json`，SHA为`6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b`。

该表为4,000唯一词面、4,012份语言/发音变体记录，EN 1,640、ES 1,197、PT 1,175。它与本次实际训练词域是两个不同范围，不能把分母混用。

- 原报告同输入词对照4,209项：双方成功4,192项且raw一致；另17项双方失败，失败不算一致。精简回传保留720条组成词记录，其中703成功一致、17双方失败，其余全量成功统计来自原报告。
- 对旧表标签，native不同EN 13、ES 445、PT 28；补已有ES清洗后ES剩47，总剩88/4,012。
- EN仍有带附加符号的外语词失败，PT仍有连字符词失败。因此本次训练域EN/PT全一致，不能覆盖掉旧表已发现的输入边界问题。
- ES残余包含代理词与原词输入政策差异，以及新官方/C++一致而旧表不同的记录；PT有18份成功输出与新官方一致、但与旧表不同的标签。旧标签不是发音真值，不应为追平旧表直接改写C++发音。
- 更早的小测已在固定西语模型/MFA 3.4.0中通过单因素实验观察到`unicode_decomposition=True/False`的成功率差异。此结论限于该版本、模型和输入，不能泛化为所有MFA存在同类故障。

详细分层、精简证据范围与原命令见`docs/HANDOFF.md`的0.179–0.181；无需从27项小测或4k全量重新开始。

## 7. 返回证据与核验范围

本次成功结果的小包为`return_summary.zip`，9,830字节。用户也返回一份`return_part_001(1).txt`，13,265字节；该分片可独立还原相同ZIP。本次归档再次通过Base64还原、包SHA和全部成员SHA核验。

| 证据 | SHA256 |
|---|---|
| 返回ZIP | `43330274a48be115122fa75aa0b7299967d130c8f4815cff7371a64176e46808` |
| 返回TXT | `644cf59f74b7d93ef64f202fc4ecbe0785e7ed93d3b83d898537ca4a2d2cfceb` |
| 包内`report.json` | `29d1c2f990500e317cdac75bbf9af3f1cb80d23666ab4717c93b1d557ad5b109` |
| 工区`words.jsonl`（282,026,053字节） | `6533f2e153c428251fb80a375f7213ca02932010a5c172377a5d2789da54b74a` |
| 工区`samples.jsonl`（161,453,350字节） | `ad7be4b12a30c26737415d94fda2d7ea659ce7af36423702f6c4981e3d4c0283` |

包内为`report.json、examples.json、source_evidence.json、sha256.txt`。本地可重算报告计数/比例、39项示例的IPA到ID映射及状态，核对可用代码/配置/vocab身份。完整逐词/逐句文件仅返回身份与大小，不能声称已在本地独立重放319,029个来源×词条或881,749句。

## 8. 工区入口、只读查询与输出政策

入口文件：

```text
scripts/compare_mmservice_g2p_training.py
configs/mmservice_g2p_training.workzone.json
src/qwen_hotword/phonemes/mmservice_training.py
src/qwen_hotword/phonemes/mmservice_training_labels.py
```

接手先完整读`AGENTS.md`，核对实际Git状态和远端，不覆盖原7份测试修改及未跟踪资料。拉取与文档身份检查：

```bash
git status --short
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
git log -1 -- docs/MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md
```

现有结果无需重跑。以下命令在容器仓库根目录仅读取已完成结果：

```bash
G2P_RUN=outputs/mmservice_g2p_training_393ce786a9
jq '{status,official_g2p_rerun,replacement_verified}' "$G2P_RUN/report.json"
jq -c 'select(.status.existing_es_cleanup=="cpp_failed") |
  {word,occurrences,reference_ipa:.saved_reference.raw_ipa}' "$G2P_RUN/words.jsonl"
```

历史代理来源复核，使用系统`grep`，无需安装`rg`：

```bash
F=outputs/es_latam_cv_auxiliary_170h_v1/repaired_mfa_v1/es_latam_cv/finalize_summary.json
P=$(jq -r '.repair_plan_path' "$F")
grep -E '^(budućnost|andrić|ibrahimović|kościuszko|mihailović|modrić|radić|stepanović|toruń)[[:space:]]' "$P"
sha256sum "$P"
```

SHA命令只能补充当前计划身份；未与先前已封存SHA对应前，不能冒称已验证历史文件从未变化。不要编辑计划、词典或旧输出。

如确需在相同身份输入上重现全训练域审计，原命令如下；**本次归档未重新执行**：

```bash
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_training.py
```

它不读WAV/Encoder/Qwen，不用GPU，不重跑官方MFA。默认新建`outputs/mmservice_g2p_training_<随机后缀>`，没有resume，拒绝已有或与输入嵌套的输出目录；失败目录保留，再试使用新目录。长任务已有阶段、真实计数和活动提示。只回传新目录的小摘要ZIP或全部编号TXT，不回传模型、音频、特征缓存或完整大JSONL。

旧失败的原因是历史`source_tsv`为仓库相对路径，旧审计误要求绝对路径。`b326a6f`已修正：统一以显式仓库root做纯字符串路径归一化，不读TSV/WAV，且整句比较共用同规则。成功运行已证明通过这一阶段，不要再回到旧失败目录补建文件。

## 9. 下一步与替代判断

1. 对已确认的9个历史代理，固定同一西语FST/二进制，分别输入原词与代理词，比较C++生成状态、IPA、90类ID和原训练标签。先验证能否补齐兼容性，不直接改生产规则；此探针尚未交付或执行。
2. 对补清洗后仍不同的4,603个来源×词条，优先按60,507次实际出现频次排序，并按来源、原词/代理词、已有修复规则归类。不能把同词不同来源合并成一个标签，也不能把已发现的少量重音词例子当成全体根因。
3. 对重点差异执行同模型、同输入、同Unicode策略的官方/C++控制对照，将历史标签政策、输入政策与推理差异分开。以`día、río、álbum、allí`等既有例子为起点，保留真实相同对照。
4. 明确替代目标：若要求兼容旧训练标签，需要审核并复现必要的旧规则；若选择保留原拼写并接受不同发音，需要独立审核发音质量及对检索的影响，不能仅以旧标签相等决定对错。
5. EN/PT可继续开展有限范围的工程接入验收，仍需覆盖已知外语字符、连字符、数字、大小写/Unicode、短语和错误返回策略。性能另做相同工作量、启动/加载分开的基准；当前耗时不构成速度提升倍数证据。

本次只完成评估归档。未修改C++、模型、训练标签、热词表、90类、门控或核心检索，未新增训练、Encoder缓存或ASR评测。后续诊断继续使用新表/新目录，保留旧资产；只有出现需要测量下游影响的受控变更，才另行安排检索配对评测。

## 10. 可直接用于新聊天的接手说明

> 请接管MMService C++ G2P替代评估。先完整读取AGENTS.md，再读本文及HANDOFF最新节点。当前成功结果为outputs/mmservice_g2p_training_393ce786a9，实际三语各480小时共881749句，源词典逐句重建与训练ID全部一致。EN/PT在该规范单词训练域标签全一致；ES native来源×词差异35.2272%，补现有音素清洗后3.1917%，按词频加权1.7206%，整句18.2362%。剩4603个来源×词条标签不同、9个词生成失败。9词历史计划已由用户终端确认全部mfa_proxy+strip_acute，代理映射及旧IPA见本文；不能把它们算作同输入核心算法差异，也尚未测试代理词输入C++后的结果。不要重跑已完成全量、不要重训/重提Encoder，不改旧词典和outputs。下一步先对9组原词/代理词做同模型控制，再按频次和来源解释其余差异。比较对象、回传证据与SHA、旧4k边界问题和全部未完成项见本文。新代码按AGENTS验证、更新节点成果并提交推送，报告实际远端SHA。
