# 英语＋纯埃及方言：动态等时长的一条龙训练

## 最新接续方式：每语最多480小时，GPU5

用户已将本轮上限改为480:480h。原动态run已完成标签，在音频指纹阶段因时长比较失败。
旧manifest可能把秒数四舍五入到6位小数，而原检查容差为1e-8秒，精度不匹配；
用户报错未给具体差值，尚不能断定该WAV确实变更或一定只是舍入。

本次使用 `scripts/run_en_egy_480.py` 接续；不再运行下面的旧all命令。
它校验并引用 `outputs/en_egy_dynamic_ctc_v1/labels`，不重新打阿语标签。
英语直接引用r15轮已使用的en/train与en/validation，不扩大英语池、不重新跑英语G2P。
保留纯EGY、8:1:1和英阿1:1；每语H取合格容量与480h的较小值，整段选择向下取，
不会为了凑时长超出480h；不设置最低小时门槛。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
python -u -B scripts/run_en_egy_480.py --gpu 5
```

新输出 `outputs/en_egy_480_ctc_v1`；同命令自动续跑。通过SQLite backup复制旧指纹进度，
原标签与指纹库不修改。旧hash通过size/mtime核验后只补读音频头，不再完整读取同一WAV。
新增头元数据也持久化。只有已知cbae06e版本的数据准备模块允许兼容升级，
其他旧输入/标注代码SHA必须一致，不任意跳过合同检查。

时长舍入容差为0.5微秒（加浮点计算余量），选样采用实测头时长。
更大时长差、已记录的音频SHA变化、实际长度CTC不可行，写入新目录的
`audio_review.jsonl` 并隔离对应候选；保护集仍保留原/现音频hash及录音组，防止回流训练。
无法读取或读取过程中变更仍报明确错误，不能把未知内容计作训练小时。
`audio_duration_report.json`含实际/原时长、差值、分类计数（每类最多20个例子）。
若English validation中发现真实异常，会隔离并记录，不能再无条件宣称原验证集完全一致。

冻结后自动接相同r15 Encoder缓存复用/补提和Head训练。完成回传
`outputs/en_egy_480_ctc_v1/return_summary.zip`。未完成时可先回传
`audio_duration_report.json`、`capacity_report.json`、`prepare_report.json`。

以下保留首次动态方案及机制说明，480h接续策略以上述为准。

## 本轮已确认策略

固定使用 r15/checkpoint-5100-merged 的 Encoder（不再用基模）；冻结 Encoder，
新初始化 ×2 temporal CTC Head。167类 `en_ar_native_ipa_vocab.v1.1.json` 不变，
英语前72个ID空间不变。英语沿用已完成的官方MFA Top1映射，阿语采用已修正的
CALIMA EGY MLE＋埃及发音规则（`arabic_pronunciation_policy.json`）。不调用MSA补样。

这是一轮自动标签实验：文本归一化相同不等于人工核准，来源的EGY属性尚未独立听审。
`ثورة/ده/عشان` 等仅处于candidate状态的词例不自动应用例外；命中待审例外的句子
进入review。其他词按已启用通用规则打标。词义相关同形异音仍使用word-based MLE，
不能称为句级上下文消歧。

### 数据接收

- open_source：仅用户选定的 `arabic_spelling_normalization_equivalent` 子集，审计约591.857h。
- gaoshan/train、coarse/train：exact、标点/空格等价、有限阿文拼写归一化等价三类。
- 采用assistant原正字法打标；response、assistant、原分类、来源和行号一起保留。
- 内容差异、坏音频、明确非EGY声明等不放行。空dialect字符串视为未声明，不能当成明确冲突。
- gaoshan/test、embedded_source中的test/validation/dev保护，不用于训练。
- 英语使用审计中的全部en_egy候选，以及 `en_us_swift_temporal2x_v1/full_ctc_train.jsonl`
  的完整已释放US训练池（487.628h，非过去抽好的480h）。原US validation/test保护。
- 英语验证继续用r15轮的2361条；旧阿语验证整体保护、不回流训练。

### 打标、隔离和动态容量

先打标所有接收候选，再对完整可用标签的候选及保护集计算音频字节SHA256。
32个I/O线程，逐文件持久化到SQLite；重跑通过size/mtime复用已完成指纹。
这一阶段确实读取WAV文件，会有共享盘成本，但中断后不会丢掉已完成扫描。
检查实际音频头时长是否与审计一致；不可行CTC长度、标签OOV/空输出进入review。

完全相同的音频只留一条；不同标签的相同音频整组隔离。跨来源重复优先归属open_source，
随后gaoshan，再coarse。分割先按录音键稳定hash预留3%验证、2%测试，再选训练时长；
识别YouTube片段、QASR/MGB广播片段和coarse的32位录音ID。未知命名只做到文件/字节隔离，
报告中明确标出 `file_only_unknown_recording`，不宣称全体speaker-disjoint或近重复去除。
已知录音的原保护集会保护同录音所有片段。英语保持原文件/字节隔离验证策略。

设完成上述处理后英语训练候选为E，三类阿语训练候选为O/G/C：

```
每语目标时长 H = min(E, O / 0.8, G / 0.1, C / 0.1)
英语 H；阿语 H
阿语：open_source 0.8H、gaoshan 0.1H、coarse 0.1H
```

**不设置最低小时门槛，不强制480h。** 例如open_source剩400h且其他容量足够，
8:1:1可组成阿语500h；若英语只有400h，则最终每语400h。
以整段音频选取，每个来源目标与实际时长的误差不超过一个候选音频时长，记录实际小时数；
不复制音频、不切断句子来凑比例。英语优先选埃及口音，不够再补US。
若必需来源完全没有可用数据或某语言验证为空，属于不可执行的数据条件；会给具体错误，
不为追求“跑上”偷偷改成MSA、打破8:1:1或混入测试数据。

阿语验证仅来自同样接收且合格的EGY候选；按录音分割后的验证池选择最多约4h，
各来源目标3.2/0.4/0.4h，来源不足就保留实际可用量，不挪用训练数据填满。
未选中的验证预留部分仍不回流训练。新test清单封存，不缓存、不评测、不用于选点。

### Encoder与Head

按r15源run核验权重、预处理、tap、dtype与缓存身份。
相同音频及相同r15特征可复用；旧特征按原分片顺序读取，避免随机反复打开大分片。
新标签写入新的缓存分片；缺失特征才加载r15跑Encoder。原缓存、原标签、旧训练输出不动。

新缓存train/validation全部完成后自动开始新Head训练，**不需要人工再敲下一步**。
默认沿用r15轮40epochs上限、初始LR=3e-4、最低LR=5e-6、early-stop delta=0.0005、
patience=6、macro-PER选点、batch256等已记录参数。实际参数封存在training_contract.json。
单卡GPU4；若该卡有其他compute进程则打印PID等待，不终止其他进程。

新阿语验证域和标签已经改变，最终新PER不能直接减去旧EGY+MSA的17.60%作为配对收益。
英语验证维持原集合；本轮报告同时保存两语PER、实际数据时长和输入身份。

## 工区执行

在原有 `qwen3-asr-hotword` 环境与容器内执行，CAMeL隔离环境已由此前setup安装：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
python -u -B scripts/run_en_egy_dynamic.py all --gpu 4 --workers 32
```

后台启动（只启动其中一种，不要同时前台和后台各跑一次）：

```bash
mkdir -p outputs
LOG="outputs/en_egy_dynamic_$(date +%Y%m%d_%H%M%S).log"
nohup python -u -B scripts/run_en_egy_dynamic.py all --gpu 4 --workers 32 > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

`tail -f`按Ctrl+C只退出看日志，不中断nohup训练。输出固定为
`outputs/en_egy_dynamic_ctc_v1`。同一命令自动续跑：跳过完整标签分片、指纹、冻结清单和
Encoder分片；Head从training_state_latest.pt续训。workers/io-workers可调整（1..64），
但运行中的进程不会热修改；没有必要为了调线程中断。进程锁防重复启动。
`prepare`子命令只到冻结清单；本任务用`all`。`--output-dir`只能选新的独立目录，
不能指向旧run/原数据。配置或代码身份改变时拒绝混合续跑。

耗时由新增音频和共享存储吞吐决定，不承诺1小时完成。长阶段有真实计数或10秒心跳；
不在本地加载/下载完整Qwen。

## 结果与回传

容器内根目录：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_egy_dynamic_ctc_v1/`

- `admission_report.json`：接收、排除数量及打标前时长。
- `arabic_labels_summary.json`、`english_labels_report.json`：可用标签容量、失败原因。
- `capacity_report.json`：去重和隔离后的容量；不存在最低小时门槛。
- `prepare_report.json`：最终英阿时长、来源比例、冻结清单SHA。
- `train.jsonl`、`validation.jsonl`、`test.jsonl`：新清单；test不进入训练器。
- `labels/parts/*/labels.jsonl`：完整发音与逐词规则轨迹；`review.jsonl`保留未通过的原记录。
- `cache/train`、`cache/validation`：新特征分片；`feature_reuse.json`给本次调用的复用/新增数。
- `head/report.json`、`head/metrics.jsonl`、`head/ctc_head_best.pt`：训练结果。
- `report.json`：最终总报告。

只需回传 `return_summary.zip`（含小报告，不含音频、模型、标签全集、特征）。
若尚在运行，可先贴 `capacity_report.json` 与 `prepare_report.json`。
