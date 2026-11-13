# 中英183类：完整标签 → 冻结Encoder → 新Head → 验证PER

本轮只做到模型训练完成及验证集 PER。**不建立热词表，不跑热词检索、delivery 或最终 ASR。**
中英与英阿实验分开保存；基模是原始 Qwen3-ASR-1.7B，不加载英阿调整后的Encoder，
也不加载旧三语或英阿Head。用户已确认直接全量推进，不另设小规模训练对照。

## 1. 固定资产与训练条件

| 项目 | 本轮配置 |
|---|---|
| 基模 | `/glusterfs_103/models/Qwen3-ASR-1.7B`，冻结全部参数 |
| 特征位置 | `thinker.audio_tower.ln_post`，1024维，bfloat16缓存 |
| 音素表 | `configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json`，183类，含blank/unk |
| 中文 | `mandarin_china_mfa` G2P 3.0.0 |
| 美式英语 | `english_us_mfa` G2P 3.3.0 |
| 英式英语 | `english_uk_mfa` G2P 3.3.0 |
| Head | 既有 temporal_upsample 结构：×2、hidden=512、kernel=5、dropout=0.1，输出183类 |
| 初始化 | Head随机初始化；不继承任何旧Head权重 |
| 优化 | batch=256，AdamW，初始lr=3e-4，最多40轮，至少5轮；验证指标驱动降学习率和早停 |
| 加噪 | 每个训练来源分别选70%条目，固定10 dB；对应原声被替换，不额外翻倍样本数 |
| 随机种子 | 20261113；同一行的噪声选择及片段偏移固定 |

完整配置：`configs/zh_en_noise70.workzone.json`。模型、表及映射来源见
[基础资源说明](ZH_EN_FOUNDATION_V1.md)。运行开始核验已批准的MFA文件和音素表SHA；
使用各FST匹配的 `.txt` 符号表，不换成曾有编号问题的英式原始 `.sym`。
G2P模型版本与MFA运行程序版本不同：本流程使用现有 `aligner` 环境中的
MFA 3.4.0或3.4.3，检查API后调用Top1；不安装、克隆或升级这个环境。

### 数据量目标

| 来源 | 训练目标 | 原声验证目标 | 另配10 dB验证 |
|---|---:|---:|---:|
| 中文 | 480 h | 10 h | 同一批10 h |
| 美式来源 | 336 h | 5 h | 同一批5 h |
| 英式来源 | 144 h | 5 h | 同一批5 h |

这是中文/英语各480h、英语内部美英70:30的工程起点，**不代表最佳口音比例的实测结论**。
加噪70%按各来源训练条数四舍五入选择；时长比例可能略偏，报告同时记录条数与小时。
不同来源单条时长不同，480:480小时也不保证训练条数相等。

只在已划定的split内选数据，不用重复样本或借用验证/测试凑时长。
不足目标会记录实际小时与缺口；为避免因全量标签失败而悄悄训练一个很小的数据集，
默认训练不足各目标90%、验证不足各目标50%时停止在准备阶段，返回原因。
这些是可配置的工程保护线，不是质量验收标准。修改配置必须使用新输出目录。

## 2. 数据来源与读取方式

- 美式JSON：`/host_home/z00841352/27A/data/en/json/swift_en_美式英语.json`
- 英式JSON：`/host_home/z00841352/27A/data/en/json/swift_en_英国.json`
- 中文tar：`/glusterfs_103/asr-data_imedia_zh-en/zh/ASRCHINESE0025-noah_carrec230-000500H_shards`
- MUSAN：`/data/h00911716/data_noise/musan/noise`

英语JSON按流读取 `audios`、`response`、assistant消息及language；去除
`language English<asr_text>` 包装后保留正文。原路径不存在时，尝试将
`/home_92` 根前缀映射到 `/host_home`，逐条检查实际文件。
用户已经验证抽样美音8/8、英音8/8可映射；这不等于已验证全量。

中文已观察到207个未压缩tar，抽样为同名 `.txt/.wav` 配对。
流程按成员读取，不把整套语料解包；特征阶段凭已记录的成员偏移读取音频，校验字节SHA。
重复成员、缺少配对、异常UTF-8、不可读音频、空/非有限波形均保留失败原因。
原始音频、旧标签、历史outputs均不修改。

所有可用音频转16k单声道后计算精确PCM SHA；同PCM但转写冲突的所有副本都隔离，
同PCM同文本只保留首个。先按speaker/session元数据分组；缺少时用文件名族作为代理，
再合并被相同PCM连接的组，以稳定hash划分96/2/2的train/validation/sealed test。
随后在train、validation内部按固定排序取目标时长。

**边界：文件名族尚未核实为说话人ID；精确PCM去重不等于近重复检测。**
因此不能宣称已经做到严格说话人独立、所有近重复隔离或纯美/英口音。
US/UK表示用户提供的来源及所用G2P分支，不仅凭文件名证明每条实际口音。
sealed test只登记原始身份与文本，不生成音素标签、不提取特征、不选模，也不报告测试PER。

## 3. 新标签如何生成

### 中文

中文FST接受带调拼音，不能直接输入汉字。这一点来自固定模型的meta/graphemes及
[官方说明](https://huggingface.co/MontrealCorpusTools/mandarin_mfa/blob/cfbee14002d514fcf411c16676877a96b7afe6ff/README.md)。
流程为：

```text
原始汉字上下文
 → pypinyin 0.55.0（Style.TONE，上下文词组读音，不另外启用tone_sandhi规则）
 → jieba 0.42.1词边界（HMM=False）
 → 对应词的带调拼音（必要的a/e/o起始音节前加拼音分隔撇号）
 → mandarin_china_mfa Top1
 → 完整MFA单位 → 固定183类ID
```

这里保留的是固定读音词典/规则给出的声调，**不保证全部都是词典本调，也不保证与连续语音
实际调值一致**。词组词典本身可能包含变调读法；例如pypinyin的某些词组读音会改变“一”的声调，
即使未另外启用tone_sandhi。儿化、轻声、多音字和人名仍可能存在自动读音错误。
不额外猜测儿化，也不删除音素上的声调、送气或其他修饰信息。
中文中的Latin片段交给US分支；这不能保证英语单字母/外来词在中文中实际采用同一读法。

两个纯Python依赖按固定URL和SHA下载，解出包目录到本轮 `frontend_runtime/`，
不运行setup.py、不改变现有Conda环境。工区下载按已授权方式局部跳过TLS验证，
SHA不匹配的文件不会使用；下载中断只重取未完成文件。
参考：[pypinyin用法](https://pypinyin.readthedocs.io/zh-cn/master/usage.html)、
[jieba实现](https://github.com/fxsjy/jieba)。

### 英语及失败处理

美式/英式来源分别调用自己的固定G2P，取Top1；保留完整多字符MFA单位。
标点与词边界不作为CTC发声类别；NFC和弯引号规范化不删重音字符。
不确定的阿拉伯数字读法、全大写多字母词/缩写、特殊符号、非支持字符、G2P失败、
音素OOV、空标签、CTC长度不可行等逐条进入 `prepared/quarantine.jsonl`，
不会用部分发音冒充成功。全大写规则也会隔离全大写普通词，失败统计需结合语料书写风格解释。
没有人工读音审核；PER参考是自动生成标签，不能直接等同真实发音错误率或最终ASR质量。

## 4. 加噪、缓存与选模

MUSAN录音按原始文件及精确PCM去重，按来源划分训练/验证噪声池；同一完整噪声录音
不跨两池。近重复噪声仍未声学核验。10 dB沿用项目的活跃区能量定义：原声20 ms窗、
峰值窗能量下20 dB以内算活跃区，按该区域信噪比缩放噪声；若峰值过高，对混合分量
共同缩放，避免削波。不会另存全量带噪WAV。

每512条构成一批，特征逐批写盘。完整原始Qwen权重和预处理配置做身份核验；
所有缓存为本轮新标签，不直接使用旧90/100类缓存中的标签。
特征提取每次处理1条音频，保存真实长度；CTC长度既做事前估计，也做真实帧数检查。
不要求加噪特征与历史原声缓存帧数相等。

训练只读取train和validation特征、float32新Head；不加载LLM进行生成。
每轮保存最新训练状态、最佳Head及指标。选模与早停使用以下四组PER的等权均值：

```text
zh/original、zh/noise10db、en/original、en/noise10db
```

英语各条件内合并US/UK的音素错误数和参考长度。因此不会把“美音”“英音”当成两个语言
在宏平均里给英语双倍权重；英语内部仍受各组参考音素总数影响。
最终最佳Head额外输出zh/us/uk × original/noise10db共六组的PER、删除/插入/替换统计。
这是验证集选模指标，不能当作从未参与选模的独立测试结果。

## 5. 工区运行

分支保持 `codex/g2p-coverage-scan`；使用原有 `qwen3-asr-hotword` 环境，
另需可调用原有 `aligner`。完整Qwen只在H200工区加载。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_zh_en_training.py all --gpu 4
```

`4`为已分配的物理卡号，进程内为cuda:0。不会停止其他任务。
特征阶段要求约35 GiB空闲显存；磁盘按剩余时长估计并检查空间。
以960h训练＋40h配对验证、约13帧/秒、1024维bf16估算，特征约96 GB，
加上全量源清单、标签、SQLite及输出，建议先预留160–220 GiB；实际取决于语料规模。
CPU标注和存储读取可能持续较久；脚本给出阶段、真实已完成数量及每10秒活动提示，不伪造ETA。
长任务建议在已有tmux会话中执行，避免终端断连。

单独观察状态（不加载模型、不占GPU）：

```bash
python -B scripts/run_zh_en_training.py status
```

如需分阶段，可将 `all` 替换为 `prepare`、`cache` 或 `train`；GPU阶段仍带 `--gpu 4`。
这些是同一流水线的分段入口，不是额外实验；直接用all即可。

### 中断恢复

重复原来的 `all --gpu 4`，**不加--resume**。

- 完整准备完成后，只核验源文件、标签和清单身份，跳过完整G2P/音频解码准备。
- 准备中断时按已完成的审计、文本、G2P及标签分批文件恢复；聚合和索引可能重新扫描。
- 缓存按完成分片SHA复用，只提取缺失分片；损坏或身份变化会报错，不静默覆盖。
- 训练从本轮最新已完成epoch恢复Head/优化器/调度器；中断的当前epoch重跑。
- 全部训练完成后复用已保存结果，不再训练；核验仍会读取缓存文件。
- 相同输出目录拒绝实质代码/配置变化；新实验显式 `--output-dir outputs/新的目录`。

## 6. 产物与返回

默认根目录：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en_183_noise70_scratch_v1`

| 文件 | 含义 |
|---|---|
| `report.json` | 当前阶段、完成/失败状态、最终六组验证PER和Head身份 |
| `label_report.json` | 各组入选小时、70%实际条数、失败类型与样例、183类出现次数 |
| `prepared/quarantine.jsonl` | 完整失败条目；不进入训练 |
| `prepared/sealed_test.jsonl` | 留出的原始身份；本次不训练、不评测 |
| `head/ctc_head_best.pt` | 按验证四组宏平均选择的183类新Head |
| `head/training_state_latest.pt` | 本轮恢复训练用状态，不是下游推理权重 |
| `head/metrics.jsonl` | 每轮train/validation指标 |
| `validation_per.json` | 最佳Head六组PER及音素错误诊断 |
| `return_bundle.zip` | 自动打包小型报告、身份、指标；不包含音频、特征或模型权重 |

请返回 `return_bundle.zip`，即使失败也返回这个包（若在最初目录/配置保护检查前失败，
则先给终端错误即可）。最终用途仍需之后用业务热词独立测试；本轮不据PER声称热词召回提升。
