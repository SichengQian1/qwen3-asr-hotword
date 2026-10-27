# 工作交接记录

## 0.186 2026-10-27 联合4k带噪三语时延归因已核验，下一步先查英语解码序列偏短

用户明确本轮关注“英语为什么快几倍”，不将语种差异自动认定为缺陷，也不以是否
满足50ms取代原因分析。0.184的诊断已回传并核验，本节承接其待工区状态；本轮仅
归档结果和下一步计划，不修改生产算法、90类、G2P标签、Head、门控或任何旧outputs。

### 证据、范围与核验边界

工区诊断目录outputs/joint_retrieval_latency_87a7b56815。附件return_bundle(5).zip为
18715字节，SHA256=db53eb7f4b29f4e6b4adef64de8e6d5fa46399472764e5e6ce5696a6c764d86a。
包内report.json（69716字节）及summary.json（3715字节）均与sha256.txt相符：

```text
report:  f9988e9baa041520856da935a811a5a006b03fcd067daf37741678caae212019
summary: 1ad00cfccbe123a935085ad12d1032c7581befd16fd1b7151b47ab472fb31cc3
```

report按原字节归档到
[results/2026-10-27_joint_retrieval_latency_report.json](results/2026-10-27_joint_retrieval_latency_report.json)，
保留原SHA；摘要两种查询范围的纯检索分布与report逐项相同。语种查询数之和26354，
EN22786/ES1706/PT1862；独立音频EN6000/ES400/PT400，两个TopK交付模式不重复计次。
时长分桶计数与分语种查询数逐项一致。报告声明的27份源码SHA与本地文件全部相等，
包含Anchor/scoring/greedy decode和三语运行入口。

原推理源outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step，
报告声明源Git=fd2a460933e024af422dca55d15512ceb80147de，Head为epoch27、SHA
872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466；表4000词面/4012
发音，SHA=6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b。
报告记录原源清单SHA=03ad9ad20771e0cbc54fbd133b9b81bca67e078e2688d31a13d7e7add592ca0e，
消费证据文件26356份（配置/报告/26354前缀）；本地未收到这些原文件，不冒称逐文件
重新复算工区清单。inspection_environment_only为Python3.12.12/rapidfuzz，不能据此
反推原推理时的依赖或GC状态。诊断复用旧计时，没有重新推理或加速；不能将它和
其他环境/字段的15ms、40–50ms数字当成优化前后对照。归档日期不是实际运行日期。

### 已确认的耗时和工作量

以下均为每条音频最后一步（完整音频），一条音频只计一次。候选平均音素长度是
先按查询求均值、再跨查询平均；倒排/音素是总访问次数÷总解码音素数，不是逐句
比值的平均。毫秒为原推理保存时延，音素数为CTC合并后的预测数量，不是真实语音
音素数，更不是PER或发音质量分数。

| 指标均值 | EN | ES | PT |
|---|---:|---:|---:|
| 音频秒数 | 6.5802 | 7.5323 | 8.3033 |
| 解码音素数 | 24.5028 | 77.5225 | 72.3075 |
| 倒排访问数 | 530.6900 | 5766.5950 | 3527.0725 |
| 倒排访问/解码音素 | 21.6583 | 74.3861 | 48.7788 |
| 精排发音候选数 | 63.7385 | 64.0000 | 64.0000 |
| 候选平均音素长度 | 5.1107 | 7.7170 | 7.2073 |
| Anchor ms | 2.3249 | 17.5491 | 10.3463 |
| 精排 ms | 2.6243 | 5.0804 | 4.0353 |
| 纯检索均值 ms | 5.7812 | 23.6470 | 15.2933 |
| 纯检索P95 ms | 8.6212 | 34.3481 | 22.6991 |

1. 主要工作量解释是“序列更长+索引命中更密”。ES相对EN的解码长度约3.164倍、
   每音素倒排访问约3.435倍，乘积为访问量约10.866倍；PT相应为2.951×2.252=6.646倍。
   这是汇总工作量恒等分解，不是耗时线性公式、逐n-gram因果归因或固定成本模型。
   当前实现先访问posting并聚合位置，再截断Top64，词表大小和最终候选数相同不代表
   初筛工作量相同。不能推广为所有英语都更快或把低时延当成识别更准确。
2. 精排候选数接近相同，但西葡候选更长；局部窗口、距离计算和后验求和工作更多，
   与精排约1.94/1.54倍耗时一致。只能说支持该机制，未做候选长度受控的因果实验。
3. 按同范围均值差分解，Anchor占ES-EN差距85.21%、PT-EN差距84.33%，精排分别
   占13.75%/14.83%；全部前缀对应Anchor占81.23%/80.81%。不是相减各阶段P95。
4. 不仅是时长分布：4–6秒前缀桶解码均值EN23.18/ES59.70/PT50.85，倒排访问
   498.63/4398.07/2424.98，差异仍存在。首2秒解码均值3.48/19.33/13.77，提示英语
   内容密度、起始静音或CTC输出行为值得继续取证；现有统计无法在这些解释间定论。

### 未确认项和反证

- 尚不知英语序列短来自实际说话量/静音/语速、转写内容分布，还是CTC漏音/blank偏好。
  首2秒不能用完整参考文本当作已说完的真值；自动G2P也不是人工发音真值。
- 末步候选中EN约95.42%来自英语，ES约11.90%来自PT，PT约22.71%来自ES，可作
  跨语种重叠线索；Top64候选组成不能冒充截断前postings的语言组成或定量因果贡献。
- Anchor耗时与posting量的Pearson相关在原全样本仅约0.28/0.38/0.29，不能宣称近乎
  完全线性。少量大停顿及候选/offset聚合差异会影响耗时，需要和普遍工作量分开。
  例如英语24音素/543次访问的查询Anchor约287.24ms；葡语19音素/560次访问约
  285.41ms。GC/调度为候选解释，未收到GC事件，不能直接认定。剔除最慢项不能用来
  替换验收指标。全26354步超50ms共32步，原结果保留，不静默排除。
- 本地只核验小包和其汇总、源码声明，不重放完整检索、听审音频或加载完整Qwen。

### 下一步计划：先低成本解释英语序列短，暂不启动新推理

1. **已有记录审计（先做CPU统计）**：在同一源计划/原始音频SHA下关联prefix_shards、
   冻结联合基线run_config的完整参考文本及语种/国家/wave。分别统计末步解码长度、
   首2秒空/短输出比例、参考文本单词/字符数量，以及分来源、相近时长/文本量的分布；
   不能把跨语言字符/单词单位当成严格等价音素量，也不把长度比称作PER。输出可复现
   的小摘要和固定样本ID，先判断英语是否本来就说得少、是否由某个来源主导。
2. **分层听审小集**：拟每语种20–30条，英语覆盖三国、西葡覆盖四wave，兼顾短输出
   和正常输出、不只挑异常。核对首2秒/整句是否确有语音、前导静音及转写对应，必要时
   用固定VAD作辅助并标记人工不确定项。仅统计和听审，不裁剪/覆盖生产音频，不把VAD
   当真值。这是工程取证规模，不宣称有总体代表性或已完成标注。
3. **按证据分支**：若文本短/静音可解释，则把数据分布作为本轮原因；若清晰发音仍
   未进入CTC，再对固定小集用同Head检查逐帧blank/重复、参考发音和删除错误。现有
   prefix只存合并后IDs/置信度及候选局部时间，缺全帧posterior和全部音素原始边界，
   不能从这些文件恢复真实blank率/声学对齐。只有该分支需要小规模工区推理，不能
   默认重跑6800条、换Head或重训。没有完整可靠参考标签时不报告PER。
4. **索引密度解释可并行准备CPU方案**：对固定已保存序列按解码长度分桶，分解2/3/4
   gram命中、倒排列表长度、重复位置和命中语言，定位少量高访问片段。相近长度下比较
   计数，保持原24 anchors/Top64/门控及全部音素ID；不先截短序列、按语言过滤或修改
   检索质量。只解释已有索引工作量，不拿本地回放耗时当原H200请求时延。

上述为待实施计划，本轮没有新诊断脚本、运行命令、输出目录或resume任务。
用户后续若指定不同“下一步”目标，应以用户目标调整，不将计划自动等同于生产改动授权。

### 本轮交付

分支codex/g2p-coverage-scan，起点e2edfa774e2c81209f7b5b720ebeea0fa1e26986。
仅提交本节与原字节小报告，保留原7份测试修改及未跟踪资料。推送SHA见本轮回复。
检查包成员SHA、JSON结构、三语/时长桶计数、摘要一致、均值差与工作量分解、27份
源码SHA、归档字节及git diff --check；文档归档不重新运行训练/推理测试。

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
sha256sum docs/results/2026-10-27_joint_retrieval_latency_report.json
```

最后一行应为上面的report SHA；不需要重跑0.184，也无需再次回传已核验小包。

## 0.185 2026-10-26 新版MMService G2P源码审阅及三语实际训练域复测入口（待工区）

用户提供mmservice_g2p_cpp(1).zip，包SHA
2408faf751ac07e4ff83f3d2081a5eabe1a354a36944b968132e523dfb583ff1，含8个源码/头文件/
Makefile/README，没有二进制。只解到临时审阅目录，未覆盖本地mm或旧构建/模型/outputs。
新版--lang es会对所有词做ñ→ni、gü→gw及StripAcute，再推理；ñ代理调用ɲ j合并，
NormalizeIpa仅es删除U+0303，en/pt保留。--no-postprocess仍保留输入代理及ñ滑音合并，
不能当成未处理模型raw。旧项目仅缺词才代理，不能先宣称新版全面复现历史标签。

完整操作见[MMService_G2P_V2_RETEST_2026-10-26.md](MMService_G2P_V2_RETEST_2026-10-26.md)。
新增configs/mmservice_g2p_v2.workzone.json封存8个文件SHA和三语语言参数；沿用
compare_mmservice_g2p_training.py，新增成对--release-config/--binary，旧命令仍固定旧版。
新版动态二进制SHA记录到report，并校验快照源码、x86_64/可执行；明确--lang en/es/pt，
模型/固定训练清单/10份来源词典的原审核与逐句ID重建不变。不是重新跑官方MFA，不读音频。
沿用实际三语各480h共881749句、207499个语言×唯一词输入、319029个来源×词标签对。

新增mmservice_training_delivery.py，在新版成功运行后从全部words导出三语native所有不同/
失败/OOV的JSONL和TSV，保留来源、词频、原标签、native及额外清洗IPA/ID。行数和词频
复核对应report；无差异时空JSONL/仅表头TSV。delivery全文件SHA/大小写入report，不塞进
小回传ZIP。额外ES清洗仍单列，native是本轮主要比较；不沿用旧9/4603当作新结果。

本地/实际远端起点120bf68c6e5dd9cac424909d5f627465ec0feba8，保留0.184时延旁线及
原7份测试修改、未跟踪资料。分支codex/g2p-coverage-scan，推送SHA见本轮回复。
用户把新版8个文件直接放入工区mm/后，在仓库根目录执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
B=$(mktemp -d outputs/mm_g2p_v2_x86_XXXXXXXX)
cp mm/*.cpp mm/*.h mm/Makefile mm/README.md "$B/"
conda run --no-capture-output -n aligner bash -c 'make -C "$1" OPENFST="$CONDA_PREFIX" CXX=g++ CXXFLAGS="-std=c++17 -O2" -j2' _ "$B"
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_training.py \
  --release-config configs/mmservice_g2p_v2.workzone.json \
  --binary "$B/mmservice_g2p"
```

新版复制到全新构建目录，不覆盖outputs/mm_g2p_x86_2LrhDuRR及旧报告。默认新建
outputs/mmservice_g2p_training_<随机后缀>，无resume，拒绝已有/嵌套输出，失败留证据。
命名阶段、真实计数及10秒活动提示；--timeout默认7200秒/语种。旧PT全量约39.6分钟
只是历史值，不据此保证新耗时。无需GPU/Qwen/Encoder/训练或修改90类。

返回新目录return_summary.zip或全部<=32KiB编号TXT（二选一）及终端摘要即可；不回传
完整words/samples/delivery。delivery可直接交同事；核验其SHA对应report.delivery.files。
待返回新报告后再给三语新旧结果表，不能把源码审阅/本地探针当实测结果。

本地新增7项版本回归，相关G2P60项通过；覆盖语言参数实际传递、旧命令保留、源码身份
变化/错语种/缺配置拒绝、完整差异及失败空标签、空差异文件、源输入不改及小包不含大交付。
新增/改动Python Ruff、严格Mypy、CLI help通过。附件8份SHA复核通过，另实际编译纯
C++字符串模块探针（无OpenFST模型）验证ES去鼻化、EN/PT鼻化保留及滑音合并。
全仓863 passed（133.69秒），Python3.10语法、CLI help及git diff --check通过。

## 0.184 2026-10-23 最新联合4k带噪三语时延差异：只读逐前缀归因入口（待工区）

用户观察英语约15ms、西葡约40–50ms，并明确指英语三国6000条、西葡wave1–4各400条
带噪音频、同一最新三语联合4000词表；不是0.75的9月单语formal100结果。当前相关
已归档主运行是0.174的epoch27/noise10db/same-step。该轮只返回总耗时摘要，本地没有
prefix_shards原件，不能把代码推断或本地合成CPU结果称为真实三语耗时归因。

源码核对：三语检索相同联合索引，无按语言选择快速算法的分支。Top64只限制精排
条目数，不能限制截断前的postings/offset聚合。精排对每个候选检查随其音素长度
变化的窗口、重复调用编辑距离和后验求和；Top64相同不保证工作量相同。
原始整段/前缀入口external._retrieve_waveform不显式禁GC，而旧streaming
CumulativeAudioCtcDetector.__call__在Anchor/精排期间禁GC；两条入口不可混用时延。
编辑距离有RapidFuzz/Python回退，当前环境不能替代历史运行后端证据。

新增scripts/diagnose_joint_retrieval_latency.py、evaluation/joint_latency.py和独立测试。
仅读取已封存run_config/report/prefix_shards及固定词表/index；不读音频、模型或权重，
不重新生成CTC、不重放检索、不改90类/标签/门控/生产实现和任何旧outputs。
逐项核对源sha256清单及prefix内部摘要、音频声明SHA/累计截止点、逐样本完整步数、
完整三语覆盖、固定表SHA与原运行声明index SHA。缺失阶段计时、残缺Top64、重复/缺失
前缀及错表立即失败，不静默补0或将Top5当完整候选。

输出分语种decode/Anchor/rerank/sort/gate/merge及纯检索Mean/P50/P95/P99/max；阶段占比
用各阶段累计时间/纯检索累计时间，不相减P95。对照prefix时长、decoded长度、postings、
实际发音候选数及候选音素长度；保留同2秒累计时长桶、分数据组、每语种最慢5步。
全部前缀与每条音频最后一步分开汇总，不把6000个样本和累计前缀查询混用。相关性与
分桶只是描述，语言仍和来源/序列/候选内容/CPU负载混杂，不自动宣布因果或调整阈值。

分支codex/g2p-coverage-scan，初始HEAD为7b6fea695ff933973e419e0f3c60d2fc352c9cb9；
并行文档交付后起点为df9e0ba4f94b0eb5133abbd30e74f5830e07f4fc，完整保留0.183。
本轮仅四个上述诊断/测试/交接文件；原7份测试修改及未跟踪资料保留。推送SHA见回复。
工区在原Qwen运行环境执行，不用GPU，不切到aligner、不安装依赖：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/diagnose_joint_retrieval_latency.py
```

默认源outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step，固定资产源
outputs/reviewed_joint_4000_retrieval_480h_v1；若同一评测另存目录，只改
--run outputs/实际已完成目录，不能指向下游ASR报告或其他格式。默认自动生成
outputs/joint_retrieval_latency_<随机后缀>。--output-dir只允许新且与源分离的目录，
不提供resume；预检/统计失败不生成输出目录，重试仍新目录，不删除任何旧结果。
每1000个文件显示实际完成数，另有10秒活动提示。当前环境的Python/编辑距离后端
仅标inspection_environment_only，不反推历史运行依赖或GC状态。

成功仅返回新目录return_bundle.zip（report.json、summary.json、sha256.txt）及终端
summary即可，不回传旧prefix全量、音频、模型或缓存。不需要重跑6800条推理。
核验新摘要：

```bash
python -m json.tool outputs/实际诊断目录/summary.json
(cd outputs/实际诊断目录 && sha256sum --quiet -c sha256.txt)
```

本地独立诊断测试12 passed；新增三份Python文件Ruff、两份实现严格Mypy、CLI help、
Python3.10语法及git diff --check通过。没有改运行链路，不重复全仓训练/推理测试。
这些检查只证明诊断逻辑，不是H200耗时结论，真实归因仍待上面的小包。

## 0.183 2026-10-23 需求收敛：交付三类现有西语差异，不继续根因实验

用户因时间限制中断进一步历史归因，改为简版分类说明＋各类完整标签明细，交同事处理。
新增[MMService_G2P_DIFFERENCES_DELIVERY_2026-10-23.md](MMService_G2P_DIFFERENCES_DELIVERY_2026-10-23.md)。
以原成功运行outputs/mmservice_g2p_training_393ce786a9为唯一数据源，仅读固定report/
words并用jq导出三类原记录：补ES清洗后已一致（预期46291条）、生成失败9条、补清洗
后仍不同4603条。后两类词频为11/60507，来源×词与去重单词分母不混用。
原词/来源/词频、原IPA/ID、C++native与补清洗IPA/ID全部保留，另有可选TSV命令。
本地小回传包不含完整words，故完整明细需在工区导出，不声称已在本地生成全量文件。

刚才中断的新归因脚本、两份模块及配置共4份未提交草稿已移至本地临时归档，保留字节
及SHA清单，未提交/推送，不作为可运行交付。原7份测试修改、其他用户文件及全部
outputs均保留。此次只交付文档和只读导出命令，不继续G2P控制实验，不需要GPU。

本地/实际远端起点7b6fea695ff933973e419e0f3c60d2fc352c9cb9，分支
codex/g2p-coverage-scan。拉取后按新简版文档先核验两项源SHA，再导出；不重跑0.181。

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

每次mktemp新建outputs/spanish_g2p_handoff_XXXXXXXX，无resume，不覆盖旧文件。
三个命名阶段有开始提示，结束用wc核验46291/9/4603；JSONL可转TSV，整目录交同事。
只需返回行数，无需回传完整明细或大包。此处为导出方案交付，实际全量导出尚待工区。

## 0.182 2026-10-23 实际三语480h标签审计回传核验，西语9词历史代理已确认

用户要求评估后生成交接Markdown。本次新增完整交接文档
[MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md](MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md)，
汇总比较层次、固定资产、全训练域结果、9词代理证据、既有4k边界和后续未完成项。
这是0.181待工区任务的结果归档，不修改C++、生产标签、90类、检索或任何旧outputs。

成功源outputs/mmservice_g2p_training_393ce786a9。实际返回ZIP 9830字节，SHA
43330274a48be115122fa75aa0b7299967d130c8f4815cff7371a64176e46808；
TXT 13265字节，SHA644cf59f74b7d93ef64f202fc4ecbe0785e7ed93d3b83d898537ca4a2d2cfceb，
由其Base64还原相同ZIP，全部成员SHA通过。report SHA
29d1c2f990500e317cdac75bbf9af3f1cb80d23666ab4717c93b1d557ad5b109。
小包为report、39项ES差异示例、source_evidence、sha清单；全words/samples未回传，
只记录SHA与大小，不冒称本地独立重放全量。

- 原词典逐句重建与固定训练ID全部相等：881749句，EN263609/ES296399/PT321741；
  每语种约480h，原清单SHA仍10fdbdbbdb8ef352cd7a143b742fd962627b40a98a44d0b54701311f0b94fbfe。
  EN12844、ES92715、PT101940个语言内唯一词；来源×词分别12844/144499/161686。
- EN/PT native词标签、按词频及整句均全一致，仅限该规范单词训练域。不能覆盖掉4k
  已发现的EN外语字符、PT连字符词等失败，也不证明原始整句前处理完全等价。
- ES native来源×词差50903/144499=35.2272%，按词频835476/3517234=23.7538%；
  补现有ES音素清洗后为4612/144499=3.1917%、60518/3517234=1.7206%。
  后者分解为4603个可用但不同的来源×词（60507次）和9个失败词（11次），无剩余OOV。
  整句native278403/296399=93.9285%，适配后54052/296399=18.2362%，其中54041不同、
  11不可用；这些不是PER、检索召回或WER。official_g2p_rerun与replacement_verified均false。
- 9词全部来自ES辅助CV：budućnost、andrić、ibrahimović、kościuszko、mihailović、
  modrić、radić、stepanović、toruń；全部原标签可用。用户再返回历史repair_plan，
  resolution全为mfa_proxy、rules全为strip_acute，代理依次为buducnost/andric/
  ibrahimovic/kosciuszko/mihailovic/modric/radic/stepanovic/torun。
  这确认历史输入已改变，不能说官方直接输入原词成功而C++同输入失败。9项旧IPA及
  计划目前据用户终端转贴，尚无计划文件SHA独立回传；本地调用现有代理函数复核映射。
  生成失败位于逐词生成阶段，不是90类映射失败；尚未逐项验证FST内部失败路径，
  更未运行9个代理的C++对照，不能宣称已修复。附带匹配的普通word行不是第10个失败词。

下一步仅需对9组原词/代理词固定同模型做小对照，再按来源和实际词频归类其余4603项。
已有día/río同输入官方控制结果与C++一致，说明部分旧标签政策可造成差异，不能扩推全体；
不为追平旧标签盲目替换i/j、u/w或去掉所有外语符号。既有全量无需重跑，后续探针尚未交付。

归档前本地/远端HEAD均b326a6f5fbc27494bbbf567d6eac8727b4317b3d，分支
codex/g2p-coverage-scan；原7份测试修改及未跟踪资料保留。本次仅两份Markdown变更，
文档提交SHA见本轮回复。工区无需计算任务，拉取及核验文档：

```bash
git status --short
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
git log -1 -- docs/MMService_G2P_EVALUATION_HANDOFF_2026-10-23.md
```

本次没有新输出目录、resume或GPU任务，无需再次回传已有包。保留成功/失败旧目录。
原复现入口、可执行只读查询及小文件回传政策均写入交接文档；归档核验只读现有证据，
重算计数/比例和39项示例，并检查文档链接/表格及git diff --check。代码未改，不重复
运行已通过的全仓训练/推理测试。

## 0.181 2026-10-22 联合4k差异分片已核验，扩到实际三语480h训练词汇（待工区）

用户明确本轮“全量”指训练用到的语料。此次是G2P/训练标签差异审计，不是音频热词
检索，不运行CTC/Encoder/Qwen，不把一致率称为发音准确率、PER、召回率或WER。

### 已返回证据与阶段结论

实际读取Downloads/return_part_001.txt至005.txt，5份编号/总数/包SHA一致，Base64
还原ZIP为110627字节，SHA
c44e84a1db30012e6bb907c0d451d294343a184f5c2b7e963dbc3444338868fa。
5个成员及sha256.txt完整通过，原report SHA
c3702627672f5449ecd9c22a0f74b8a671a536614a1be19d7779b7425945c47f；
原清单记录SHA899dce41fe414df03a85a0a8fd76e65b706a35bf85081b9199913bb2c359b117。
源仍outputs/mmservice_g2p_full_v1_553359315a。保留586案例（EN16/ES539/PT31，
各含3条对照）及586旧标签逐项用当前90类复算IPA/ID、native/适配/重建标签比较；
全部difference_flags复算与原report一致。返回720条词结果中703成功raw一致、17双方
失败；全4209词/4192成功一致仍来自原报告，不能说本地读到了省略的全部成功结果。
17份本地可用源码/配置/vocab身份匹配，25份工区大资产或证据未在本地，不冒称重读。

- 与旧表标签native不一致：EN13/1640、ES445/1197、PT28/1175；额外现有ES清理后
  ES剩47，总剩88/4012。EN13是8个含重音/ñ等独立词涉及的词面失败，并非13套不同
  G2P音素替换；EN填充881全部一致。
- ES149条原生不可用均含ã导致的未支持组合波浪号，共153处OOV；额外现有ES清理后
  149全部可用，但不等于全部与旧标签一致。剩47中43条仍不同于项目重建标签，均有
  替代拼写来源（39 strip_acute、3 enye_via_ni、1二者）；另4条Aeroclub Chañar
  Ladeado/Carnaval Porteño/Lasaña/entraña是新官方+C++/项目重建相同而旧表不同。
  新项目参考与适配C++总差44，其中额外1条ñquis已与旧表相同；不要混淆这两个集合。
  数字策略异常Avenida 9 de Julio也与旧表相同，但双方省略数字不算完整词面正确。
  不能为追平旧代理标签把原拼写读音硬改回去，真实读音未另行审核。
- PT28中10词面失败涉及9个连字符词；其余18新官方与C++一致但旧表不同，包含São
  Paulo旧s a w、新s ɐ̃ w̃等实际鼻化差别，亦有其它元音/辅音不同。旧来源字段为null，
  不能直接断言全部历史错误来自某一规则。此结果支持扩查训练标签，不是生产替代许可。

### 全量运行范围及比较方法

新增scripts/compare_mmservice_g2p_training.py、phonemes/mmservice_training.py及
mmservice_training_labels.py、configs/mmservice_g2p_training.workzone.json与测试。
默认读取最新实际训练的固定清单
outputs/english_countries_refresh_fast_v1/frozen/full_ctc_train.jsonl，SHA
10fdbdbbdb8ef352cd7a143b742fd962627b40a98a44d0b54701311f0b94fbfe；
881749条，EN263609/ES296399/PT321741。仅实际入选的每语种约480h，英语为三国
新语料，不使用旧Swift英语；不扩大到未入选池或验证/test。

由配置列出的10份原full-manifest build_config.json，绑定其source TSV路径和对应
MFA dictionary.path（EN1/ES4/PT5）。只读取词典和清单文字，不读取原TSV/WAV；
不按大多数票合并跨来源同词不同发音。词典可能只有历史size/mtime而无SHA，重新
记录当前SHA，并逐句重建词典ID与固定训练清单实际ID核对；无法核对的句子明确列
baseline_not_verified，不从整句音素反猜词边界或把源词典差异归成C++推理错误。
同一来源TSV不允许多份build配置静默覆盖；找不到对应build/源词典则明确停止。
历史训练清单source_tsv可为仓库相对路径，build_config常为绝对路径；两处统一按
显式仓库root作纯字符串拼接/规范化，不按当前cwd或Manifest父目录、不对TSV做
stat/read/resolve。清单扫描和整句比较共用这一规则，原清单字节/SHA不变；每语种
记录source_tsv_absolute_samples与source_tsv_repository_relative_samples。

项目extract_word_tokens先NFC/casefold分词，按语种去重后分别调用原C++默认输出；
这测的是规范单词输入，不证明C++对原始整句前处理完全等价。只生成诊断输出，不改
词典、90类、门控或检索；本轮不重新跑官方MFA。对照为已有生产MFA标签兼容性，
不是新一轮同输入官方核心算法对照。后者若有需要只针对返回的差异词另做控制实验。

分别给出native与existing_es_cleanup两支：逐source×word标签对、按词出现次数
加权（重复词计多次）、句子整体ID相等/不同/不可用。词统计参照不可用单列，不计入
有效参照差异率；word_reference_attribution_verified只有该语种全部句子的词典重建
均与实际训练标签相等才为true。句子直接对实际ID比较，并另列baseline_verified各类，
避免把未经验证的词典归属解释成已确认的训练词标签。PT不套ES去鼻化。

### 工区命令、输出与回传

修改前本地/远端均c669e30ffb9530b59825b7474f65dc8c961cbbdf，其相对e319aa6仅为
说明文档更新，已保留。分支codex/g2p-coverage-scan，推送SHA见本轮回复；原7份测试
修改及未跟踪资料不改。在容器仓库根目录使用既有aligner（pynini/OpenFST），无GPU：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_training.py
```

默认新建outputs/mmservice_g2p_training_<随机后缀>，--output-dir只可指定新且不与
输入目录嵌套的目录；不改旧输出，无resume，失败后保留诊断，新目录重试。预检
失败不建目录；缺路径或源归属错误只返回报错，不猜字典。--timeout默认7200秒/语种
C++子进程；命名阶段、扫描实际完成数/881749、C++实际已写词数、10秒活动提示，
超时/Ctrl+C沿用已测进程组清理。词典转换/模型检查是CPU工作，不读取完整Qwen。

首次工区outputs/mmservice_g2p_training_2a281eb184在清单扫描起始因旧版误要求
source_tsv绝对路径而失败（相对值outputs/es_ar_train_sources_v1/slr61/source.tsv）。
10份词典预检完成11.3秒，尚无C++推理结果，by_language为空；不是词典/训练标签
差异结论。修正后保留该失败目录，拉取并原命令重跑，自动新建目录，无需回传失败包。

完整report.json、words.jsonl（全部源词对，含发音/ID/频次/来源）、samples.jsonl
（每句ID、语种、源、重建核对及两支状态）、分语种words.txt/C++输出/日志保留新目录。
回传return_summary.zip即可：report、按实际词频优先的每语种/分支最多20条差异示例、
完整逐词/逐句文件SHA及包成员清单；不包含大逐样本文件、FST、权重、音频或缓存。
同时生成每份<=32768字节的return_part_NNN.txt，可像本次一样回传全部编号TXT，
不必另外传ZIP。工区真实差异数尚待运行，本地测试不作为全量结果。

本地新增7项训练审计测试，连同既有G2P测试44项通过；全量835 passed（136.61s）。
覆盖实际训练词域/词频分母、同词多来源不同标签不合并、ES适配/PT鼻化、源字典与
实际ID不符时单列、缺路径/身份漂移/test输入/未知TSV来源拒绝、C++失败保留小摘要、
原文件不变及小包/分片校验。原生FST/C++用mock，不声称H200实测。新增模块/脚本/
测试Ruff、Python3.10严格Mypy（follow-imports=silent）、CLI help、3.10语法及
git diff --check通过；未改原全表审计代码或处理无关全局lint/typing问题。
路径兼容另补9项回归，覆盖真实相对路径、build/清单绝对相对混用、改变cwd仍按显式
root关联、整句阶段同规则、词频不拆组、原文件不变及重复归属/非法路径仍拒绝。
修正后相关G2P测试53 passed，全量844 passed（135.39s），新增/改动代码Ruff、
严格Mypy、CLI help、Python3.10语法及git diff --check通过；工区重跑结果仍待返回。

## 0.180 2026-10-22 G2P项目规则小测已核验，联合4k终端返回、精简证据待核验

用户回传outputs/mmservice_g2p_pipeline_v1_07b51fbf3a对应return_bundle(4).zip，
包SHA12d51a06d807cdec2780b411a1c848207121a373345ea1742e1ea1939d7d2006。
52项小证据校验及逐项ID/统计复算通过，代码/配置/vocab身份与bce9d21交付相符。

- 官方西语同FST/字符表只切换unicode_decomposition：True为3/8成功，False为8/8，
  默认True完整复现原MFA；内嵌与外部字符表的283个符号ID全部相同。原5个缺词在
  True下均无未知metadata字符，但发生Composition failure；False全部恢复。此结论
  限于MFA3.4.0与这份固定西语模型，不泛化成所有MFA版本/模型错误。
- 将False的8个孤立词结果按9个原surface词序拼接，和C++raw逐项9/9一致。对比
  现用代理修复规则则native5/9、额外ES清理后7/9：pingüino的ĩ/i、San José的ã/a
  是既有ES鼻化清理差异。剩余2项是同一alegría的NFC/NFD输入，旧代理alegria输出
  a l e ɣ ɾ j a，保留原拼写的官方False及C++均为a l e ɣ ɾ i a。不能以旧代理标签
  为绝对真值强改C++；真实读音仍需审核。EN8个普通输入一致，ﬁre/数字策略差异仍在；
  PT8/8一致。小测未证明全词表替代或发音准确率，也没有改生产规则/标签。

### 扩测范围与参照层

用户明确选择“现用三语联合4000热词表”，不是训练语料全量词典。新增
scripts/compare_mmservice_g2p_full.py、phonemes/mmservice_full.py、
mmservice_rewriter.py、mmservice_process.py及配置/测试。冻结源仍为
outputs/reviewed_joint_4000_retrieval_480h_v1/reviewed_keyword_table.json，SHA
6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b；4000唯一词面，
4012 retained发音变体，目标/填充、跨语言同形词、同词多发音全部保留。核验源封存
report/清单、expanded_table_sha256及变体pronunciation与90类ID一致性。三套ZIP、
vocab仍固定旧SHA；x86二进制仍为outputs/mm_g2p_x86_2LrhDuRR/mmservice_g2p，
SHA d6596db29acec77bee00df8fb999e2e2352d3d9f0acc3067c0e324be3c98f094，
另核验7份编译源码/Makefile。原C++、原小测代码和全部既有outputs均不改。

实际比较分为以下参照，报告中不得混用分母或命名：

1. word_engine：项目分词的同一唯一词，官方重写算法与C++raw用同一原FST、内嵌
   输入字符表/Unicode模式及phones.sym；官方侧明确不做metadata字符过滤或NFKD。
   C++仍保留自身前处理；这里只固定提交词，没有直接观察其内部token，word差异
   不能直接判为纯推理核心缺陷。官方算法用原MFA3.4.0类，内存绑定符号，不改磁盘。
   记录官方/C++各自成功、raw/ID一致和双方失败；双方失败不算一致。原CLI默认的
   字符清理/分解另做一层，不把算法控制支路说成未修改的官方默认策略。
2. official_default与project_rules：三语各自正常MFA CLI Top1，记录缺词、额外输出
   key、歧义，按原词序组装；西语再调用现有prepare/finalize缺词代理及发音清理，
   保存代理计划/原始发音/规则出处。新结果只写诊断目录，不替换训练词典或热词表。
3. saved_table：直接比较4012份已存标签；同时统计每个语言×规范词面是否所有输入
   形式都命中至少一个原有变体，避免把另一个合法发音计作整词失败。逐变体、原始
   输入、语言×规范词面分母分开；target/filler分开统计。表标签是兼容参照而非读音真值。
4. adapter：C++默认输出额外应用“已有ES清理”的结果单列，PT鼻化保留，不能把该
   适配支路当成native。保存raw和final两次C++surface结果，还直接核对其IPA后处理
   与Python字符串规范化是否一致，避免共享ID转换再次规范化掩盖差异。

词面/唯一词推理可复用同一输入，审计仍保留全部发音变体索引与来源。数字/空词生成
策略的拒绝和已有标签比较分别记录，不能用删除数字后的输出冒充完整热词读音。
不读取音频/Encoder/Head/Qwen、不测检索召回或最终WER，不修改90类/门控/检索。

### 工区执行、进度与回传

本地及核实远端起点bce9d21bfb94aa125f0fc62f8db887d3c25c02a0，分支
codex/g2p-coverage-scan；原7份测试修改及未跟踪资料保留。推送SHA见交付回复。
容器内仓库根目录，使用已有aligner，不需GPU或重新编译：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_full.py
```

默认新建outputs/mmservice_g2p_full_v1_<随机后缀>，可用--output-dir指定新目录；
拒绝已有目录，无resume。中断/失败保留诊断，重试使用新目录，不删除旧输出。
--jobs默认1，--timeout默认1800秒/子进程。命名阶段+10秒活动提示；C++显示真实
已写JSONL行数，官方同输入worker每100词显示实际计数，MFA stdout实时转发且完整
stderr留日志。超时或Ctrl+C会终止该子进程组，避免后台孤儿进程；执行中断尽量封存
failed报告及日志包，预检失败不创建目录。耗时含启动/加载，不宣称线上加速倍数。

回传新目录return_bundle.zip与终端摘要。包内report.json、input_inventory.jsonl、
case_comparison.jsonl、word_engine_comparison.jsonl、differences.tsv、各语种词表/
C++输出/官方同输入及默认dict/ES代理与修复dict/日志和sha256.txt；不含模型FST、
ZIP、MFA缓存或音频。可在新目录运行sha256sum --quiet -c sha256.txt复核。
replacement_verified仍为false；正式训练词典全量扫描不在本轮范围内。

用户随后返回outputs/mmservice_g2p_full_v1_553359315a终端completed摘要，尚未返回
实际report/逐词文件。终端称：语言×词输入4209，双方成功4192且raw全一致、双方失败
17（不算一致）；已存标签native一致EN1627/1640、ES752/1197、PT1147/1175，额外
ES清理后ES1150/1197。PT新官方/C++成功1165全一致，但仅1147匹配旧标签，需查18条
旧标签差异；另10条失败。ES有1条策略阻断却被C++接受，不能并入合格输入一致率。
以上只据终端，不能宣称本地独立核验、真实发音正确率或已经可替代。

### 大包无法回传时：只打包已有差异，不重跑G2P

用户反馈return_bundle.zip约900多KB无法传回。新增
scripts/package_mmservice_g2p_result.py及phonemes/mmservice_compact.py；保留全部
带difference_flags或额外ES清理后仍有旧变体不一致的案例，另取每语种最多3条无差异
对照；保留这些案例的全部旧标签/变体来源、组成词官方/C++结果，以及全部raw不一致
或失败词。原report按字节原样保留，不含过程日志和无关成功案例。

仅读取原sha256.txt及四份report/inventory/case/word证据，按原SHA核验这四份文件、
从完整记录复算总计及分语种计数，生成后再核验原文件未变。不会重新读取模型、二进制、
原始音频或重新推理；模型/代码身份是原report中的记录，不冒称重新验证实物。
分支codex/g2p-coverage-scan，修改前本地/远端均548a81c3e5749c252cfe895ec5c275955c957da6，
原7份测试修改和未跟踪资料保留。推送SHA见本次回复。容器内仓库根目录执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/package_mmservice_g2p_result.py outputs/mmservice_g2p_full_v1_553359315a
```

不需要aligner或GPU。默认另建同级outputs/mmservice_g2p_compact_<随机后缀>，不改原
目录/return_bundle；拒绝已有或与原目录嵌套的输出，无resume，失败重试用新目录。
命名阶段及10秒活动提示。打印return_compact.zip实际字节数/SHA和所有文本分片路径；
每份return_part_NNN.txt默认不超过32768字节，可用--part-kib调小（最小4）。
回传精简ZIP，或回传全部编号TXT（二选一）。TXT首行含包SHA及分片编号，余下Base64
按编号拼接后可恢复ZIP；用户无需手动还原。包内sha256.txt核对精简成员，
compact_report.json记录原证据SHA、选择范围、计数复核及省略项。实际压缩大小/分片数
须以工区打印为准，不用本地合成夹具估计实测大小。

精简回传另补4项测试，连同既有G2P测试37项通过：覆盖差异/变体标签保留、多分片按
编号还原与字节上限、原文件不变、证据SHA或计数异常及嵌套输出拒绝。新增脚本/模块
Ruff、Python3.10严格Mypy（follow-imports=silent）、CLI help及3.10语法检查通过；
全量828 passed（132.84s），git diff --check通过。原推理与封存身份代码未改。

本地新增15项测试与既有18项G2P测试共33项通过，全量824 passed（130.56s）。覆盖
跨语同形词/多发音全部保留、逐变体与任一变体分母、ES代理来源/PT鼻化、身份漂移、
空结果/双方失败、子进程失败/超时/中断和原始证据保护；实际子进程超时与中断清理
有本地测试，MFA/原生FST和C++推理明确使用夹具，不宣称工区4k结果。新增代码Ruff、
Python3.10严格Mypy、CLI help、3.10语法及git diff --check通过。全仓仍有原Ruff11项，
全局Mypy仍先被NumPy类型桩与项目3.10目标不兼容阻断，未修无关文件或环境。

## 0.179 2026-10-22 G2P小测回传核验、项目规则与C++原生输出分层对照（新实验待工区）

用户回传return_bundle(3).zip，包SHA为
51fc84a10b269266797e4e1248c94caa6f420bd0442b57c473989170e5f08539。
实际源目录outputs/mmservice_g2p_smoke_v1_82a5fd0563，33项证据SHA全部通过；本地
独立重算逐项计数/音素ID，与终端一致。源report SHA为
ce7a5f73d9008208323e032c6362e0a2bc5ca290670b99104a7327f899d7b2d6，
源sha256.txt SHA为5c0ea2029abad92543a4f4f67880c4aff73faec14a564650841d52c8de391fef。
记录的源码/配置/vocab身份与交付对应，模型ZIP记录仍为三套固定官方SHA；大模型和
二进制未在回传包中，不能称本地重新读取工区模型验证。x86 C++二进制记录SHA为
d6596db29acec77bee00df8fb999e2e2352d3d9f0acc3067c0e324be3c98f094。

- EN10项：双方可用8项，原始发音和最终ID全部一致。ﬁre被Python casefold展开为fire，
  C++未展开而失败；GPT4被测试规则拦在MFA侧，C++丢数字后输出GPT读音，不是官方模型
  失败或C++读对GPT4的证据。PT8项全一致，小样本不能证明全词表替代或真实读音正确。
- ES9个surface对应8个唯一词；官方raw.dict只有casa/guitarra/san，实际缺少
  alegría/josé/mañana/niño/pingüino，并非查找时词名没对上。双方可用2项且一致，
  C++另有6项可用、San José虽ok却因ã的组合波浪号无法完整映射90类。官方san也产生
  s ã n，不能归为C++独有问题。英/西FST实际有内嵌输入符号表，PT没有。
- C++确实内置通用文本前处理、G2P和通用IPA后处理；不包含完整Python casefold、
  数字审核、90类ID/OOV检查或西语专项代理/发音修复。原小测既比较raw IPA，也比较
  通用规范化后的ID，并另核对C++后处理字符串；26个ok输出的后处理字符串检查通过。
  “原版官方输出”是MFA侧未加西语专项修复，不是C++无前后处理或只做错层字符串比较。

### 下一轮对象与实现

用户确认继续判断。新增scripts/compare_mmservice_g2p_pipeline.py、
phonemes/mmservice_pipeline.py及固定源身份配置，不改原C++、原小测、任何训练字典、
检索表、90类vocab或CTC。复用已完成27项及原模型；只新增：

1. 调用生产prepare/finalize_spanish_mfa_repair，原词唯一发音优先；仅缺词才做代理，
   歧义不靠代理消除。预计新增5个代理唯一词的MFA Top1；去鼻化/按代理规则合并ɲ j，
   经90类验证后生成新诊断字典。词频仅来自这批诊断surface，不冒充训练语料词频。
2. 对三语比较项目规则重建的最终IPA/ID与保存的C++默认输出。native结果保持原样；
   另列“额外调用既有ES清理”的调用方适配诊断，不能把适配后结果算成C++原生一致。
   数字/空词/缺词/歧义/OOV不算一致；这是热词文本链路，不套语料专用h/连接符/
   音频/CTC长度审核。它也不是与全部历史训练字典或现有4k表逐字节比对。
3. aligner子进程用MFA3.4.0官方PhonetisaurusRewriter与同一原FST/外部字符表，
   仅切换unicode_decomposition，8个西语唯一词各跑True/False，记录输入码点、
   清理结果、实际输出或rewrite错误，比较内嵌/外部符号ID。模型磁盘不改；官方正常
   setup的字符表只在内存绑定。另查默认True能否复现保存的MFA结果；若不能，先查
   版本/API/符号差异，不能直接把False结果归因到唯一字符分解因素。

字面最终IPA比较发生在共享ID转换再次规范化之前，避免ID相同掩盖C++后处理差异。
replacement_verified仍固定false；真实兼容率/代理结果/字符分解根因尚未返回。

### 工区命令与输出

分支codex/g2p-coverage-scan；本地及实际远端起点均为
e24a1d250d76c3f6c5041da2056621625b04169b。原7项测试修改与用户未跟踪资料保留。
推送SHA见交付回复，容器内仓库根目录执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_pipeline.py
```

无需GPU、Qwen、音频或重新编译。脚本核验固定源report/清单、33项小证据和源报告
记录的模型/二进制/代码身份；已完成源目录不得改名或编辑。默认新建
outputs/mmservice_g2p_pipeline_v1_<随机后缀>，可用--output-dir指定新目录；无resume，
拒绝覆盖，中断或失败保留证据，重试用新目录。--jobs默认1，--timeout默认300秒/
子进程；阶段有计数或10秒活动提示。探针逐词记录在probe.stdout.log，错误在对应
stderr.log。预检失败不创建目录；执行失败尽量形成failed报告及日志包。

回传新目录的return_bundle.zip：report.json、comparison.jsonl、ES代理计划/原始
代理dict/修复诊断dict、es_unicode_probe.json、日志及源小证据。只复制已校验的
源清单成员，不复制未列入清单的附加文件；包不含FST、MFA缓存、Qwen或音频。
可在新目录执行sha256sum --quiet -c sha256.txt核对文件。正式全词典/4k表替代验证
仍待这些差异解释后单独开展，不把小词表一致或调用方适配等同于替代验收。

本地12项新增测试与6项旧小测共18项通过；全量809 passed（134.25s）。定向Ruff、
默认Python3.10目标的严格Mypy、CLI help、3.10语法解析及git diff --check通过。
全仓Ruff仍为原有11项。全仓Mypy被当前NumPy类型桩与3.10目标的语法兼容阻断；
临时以3.12复核仍有19项，均在本次未改动文件，不能称全仓Mypy通过。未改环境或
无关源码。测试明确模拟MFA/原生FST执行，仅生产修复规则与文件封存是真实本地运行，
不冒充工区新发音结果。

## 0.178 2026-10-21 MMService G2P原版C++已在x86编译，三语小词表兼容对照入口（待工区）

用户回到G2P替代验证，确认使用本仓库三套官方MFA模型，不重训CTC、不重算Encoder。
附件mmservice_g2p_cpp.zip只含源码；工区上传目录实际为mm/，其原程序为AArch64且
没有执行权限。工区uname=x86_64、g++13.3、aligner内MFA3.4.0/Pynini2.1.7/OpenFST
1.8.4，具备libfst.so.26和头文件。用户已在新目录
outputs/mm_g2p_x86_2LrhDuRR复制源码并编译成功（13.349s），使用OPENFST=$CONDA_PREFIX、
CXX=g++、CXXFLAGS='-std=c++17 -O2'覆盖原Makefile路径和强制旧ABI参数；源码未修改。
readelf确认新程序是x86-64，ldd无not found且libfst/libstdc++来自aligner。两条
PROJECT_OUTPUT/SymbolTableIterator弃用警告未阻断编译；这些只证明构建/依赖可用，
尚无真实G2P发音一致性结果。原ARM程序、构建目录及旧模型均保留。

### 已确认的模型契约与小测范围

- EN/ES为Phonetisaurus，grapheme_order=2，附有graphemes.sym；PT为Pynini，无该表。
  ES meta的unicode_decomposition=true；C++输入只做手写NFC，未按模型执行NFKD。
  这是待验证差异；还需确认FST是否实际内嵌输入符号表。C++按是否内嵌符号表切换
  bigram/Unicode模式，不能仅凭ZIP里有graphemes.sym认定已读入。
- 新增scripts/compare_mmservice_g2p_smoke.py、phonemes/mmservice_smoke.py、
  configs/mmservice_g2p_smoke.workzone.json及6项测试。27个固定输入为EN10/ES9/PT8，
  包含常用词、warning/warnings、arrive/arrived、大小写、短语、撇号、重音字母及
  NFC/NFD组合字符；GPT4是数字策略边界，不作为正常可用训练标签。
- 核验编译目录7份源码/Makefile与附件原件SHA一致，记录新二进制SHA并要求x86-64
  ELF和执行权限；三套官方ZIP及90类vocab仍使用既有固定SHA。只把指定官方成员
  复制到全新输出目录，不调用extractall、不补字符表、不改FST或执行西语专项修复。
- 同环境调用官方mfa g2p --num_jobs 1 --num_pronunciations 1；输入为项目规范化的
  唯一词，短语按词序拼接孤立词Top1。C++分别运行原始发音和默认后处理，使用同一
  批原始surface，保存完整JSONL及stderr。核对原始词名/顺序/数量，不靠输出行号
  或删重音把MFA变更的key冒充原词。两边缺词、空结果、OOV或失败不会算作一致。
- 对比原始发音字符串及经项目90类映射后的Token ID，另外保存C++后处理与项目
  后处理的字符串一致性。counts含全部输入数、MFA可用、C++ok/可用、双方可用、
  原始一致、ID一致、双方不可用和数字策略阻塞数量；逐词明细保留原始输出与OOV。
  C++某语种进程失败时记录明确process_failed并继续另两语种，MFA基线执行失败则
  停止并返回failed和日志包。completed仅代表取证程序跑完，replacement_verified
  固定为false；小测不能替代全词表/训练词典或独立真实发音质量审核。

### 工区执行与回传

本地/跟踪/实际远端起点1c5b45c5b4c2b4d0cd117f931fa0460d383ebbd9，分支
codex/g2p-coverage-scan，原7份测试修改及未跟踪资料未动。推送SHA见交付回复。
容器内仓库根目录执行，使用已有aligner环境，不需要GPU、Qwen、音频或下载依赖：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_smoke.py \
  --binary outputs/mm_g2p_x86_2LrhDuRR/mmservice_g2p
```

默认新建outputs/mmservice_g2p_smoke_v1_<随机后缀>，打印实际目录；也可显式提供新的
--output-dir。拒绝已有目录，无resume；中断保留原尝试，重跑创建新目录，不删除
旧输出。每个模型读取、MFA调用、C++调用都有阶段/已知输入数量/耗时和10秒活动提示。
默认每个外部进程timeout=300秒，可用--timeout设置；超时保留日志并标记失败。
各进程耗时包含加载/启动，仅诊断用，不直接解释成C++吞吐加速倍数。

回传终端打印的return_file，即该新目录return_bundle.zip。包内为report.json、
comparison.jsonl、input_cases.json、各语言model_inventory.json、词表、官方raw.dict、
C++两份输出/日志及sha256.txt；不含FST模型、模型ZIP、MFA缓存、权重或音频。
即使failed也尽量形成日志包；预检失败未创建目录时只回传错误。下一步依据结果定位
模型装载/字符编码/前后处理/音素解码差异，保留原版对照后再讨论单因素修复，之后
扩到真实训练词表，不直接替换任何生产字典、热词表或CTC标签。

本地6项定向测试通过，覆盖三语同输入与短语拼接、原FST不改、不打包大模型、
源码/官方资产漂移与ARM程序拒绝、双方失败/OOV/数字/变更key不计一致、C++错误
继续另两语种、官方失败日志包、输出缺行/错序和真实子进程超时。全量797 passed
（67.88s），补充官方模型漂移断言后再跑定向测试；新增范围Ruff/strict Mypy、CLI
help和git diff --check通过。
全仓既有Ruff11/Mypy8项未改。所有模型读取/执行在本地测试中为明确夹具边界，
没有在本地执行附件程序、安装OpenFST或宣称三语真实一致率。

## 0.177 2026-10-21 干净英语每地区100条、真实目标数1–3、best27 Top7单文件清单（待工区）

用户临时插入下游交付任务，G2P替代验证暂停。已确认使用原干净英语三组各2000条，
每组只在冻结真实目标数1–3的候选池中抽100条，合计300；使用最终epoch27、完整
三语4000词索引、整段Top7。下游只需新版sample_selection.json。用户贴出的旧文件
对应multi_nested_prompt formal100；旧负例/嵌套配额和优先错候选压力选样不继承。

新增scripts/build_english_sample_selection.py、inference/english_sample_selection.py、
configs/english_sample_selection.workzone.json及8项测试；不修改旧检索、评分、门控、
训练、模型或outputs。起点本地/跟踪/实际origin的codex/g2p-coverage-scan均为
a2190945c90417b4b1a8593e44057b4292e9b3a1。原7份测试修改和未知资料保留，不提交。

### 数据、选样和候选口径

- 只读旧封存outputs/reviewed_joint_4000_retrieval_480h_v1的run_config/report、词表
  快照及发音索引，保留每个英语组的target_ids、2000个ID/参考/干净路径；复算全体
  6000条目标分母须等于4374。按生产strict_phrase_match匹配冻结目标，重复说同一
  词只算一个，不用selected_matches数量决定入选，也不从参考重新构造目标词表。
- 对每组的1–3目标候选统一按seed+source+sample_id的SHA排序取前100，固定seed为
  english-clean-targets1to3-20261021-v1。1/2/3不另设等额配额；不足100则停止，不补
  0目标/4目标或按检索命中重选。元数据扫描全部6000条，只对选中300条读取WAV字节，
  与旧封存SHA核验；不执行全语料字节去重，不声称新增字节/说话人隔离。
- best27通过既有load_completed_best核对训练完成marker、plan/report、权重、词表和
  统计，并固定Head SHA872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466。
  table/vocab完整SHA在新配置中，沿用0.170–0.176；仍搜索全部4000词面/4012发音。
- 正常只对这300条补跑Encoder→best27 Head→原Anchor Top64→词面合并和门控→Top7。
  沿用score>=0.75、edit<=0.35、posterior>=0.5，不调参、不跑最终LLM转写或训练。
  Top7是最多7个合格词面；零候选保留空数组，不补齐、不重选成功音频。
- 支持显式--reuse-run读取已完成、封存的同Head/同表/同参数/同干净音频整段结果；
  验证配置、代码SHA、逐样本参考/路径/音频SHA及Top5重放，再精确提取Top7。
  epoch21、带噪或流式结果不能冒充。当前记录只有epoch21干净6000完成证据，因此
  默认执行300条新检索，不自动扫描/猜测其他outputs。使用reuse时prepare/run都必须
  提供同一个路径，使恢复契约一致。

### 单一交付与工区命令

JSON保留samples内case_id/sample_id/audio_path/reference_text/language、expected_*、
selected_matches及选样原因；候选保留实际hotword_id/surface/score/edit_ratio/
posterior_confidence。新增source区分地区，case_id全局唯一，原sample_id不改。
本次primary_group为one_target/two_targets/three_targets，actual_group_counts据实统计；
不虚构旧嵌套压力分组。containment_expected_ids保留全部冻结命中目标，与expected一致，
不执行旧人工嵌套集的longest-match删减，redundant_family_ids为空；JSON写明此政策。
evaluation_scope为external_english_clean_300_target_count_1_to_3、test_set_used=true，
并内含Head/表/vocab身份、各地区数量、Top7门控和clean条件，不能照抄旧validation标记。

容器内仓库根目录、原Qwen环境执行；分支codex/g2p-coverage-scan，远端SHA见交付回复：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/build_english_sample_selection.py --prepare-only
python -B scripts/build_english_sample_selection.py --gpu 4 --resume
```

第一条Python只CPU核验、选样和冻结plan，打印各组可选/选中数量及1/2/3分布，不发布
带空候选的最终交付。第二条只对选中300条推理并发布JSON；4可换为已分配物理GPU。
也可以在全新目录省略prepare-only，直接带--gpu一步完成。阶段均有命名、真实计数、
耗时和10秒活动提示，不估算虚假ETA。中断后沿用第二条恢复已核验分片；完成后恢复
不再加载Qwen。代码/配置/Head/参考/音频或分片漂移会拒绝，不删除分片绕过。

默认新目录outputs/english_clean_ep27_top7_selection_300_v1，首次拒绝覆盖；所有旧输出
保留。内部run_config.json、sample_shards、report.json只留工区。唯一交下游文件是：

```text
outputs/english_clean_ep27_top7_selection_300_v1/sample_selection.json
```

终端status=completed后确认sample_count=300、三个source各100及delivery_sha256；
若需本地核验，回传小report.json或终端摘要即可，不要求return_bundle、WAV或权重。

本地19项新/旧入口定向测试通过，全量791 passed（68.14s）；新增8项覆盖真值计数、
重复词/子串、0/4目标排除、顺序无关的固定抽样、缺额拒绝、每地区配额、只hash选中
音频、真实Top7与门控/空候选、身份漂移、原资产不变、分片续跑及封存clean复用。
最新进度/复用代码核验补充后再跑新增测试。新增范围Ruff/strict Mypy、CLI help和
git diff --check通过；全仓仍有11个既有Ruff、8个既有Mypy问题，未改相关文件。
本地使用小Head及模拟检索边界；真实300条选择/召回、可用池数量和最终交付SHA待工区，
不把这些本地测试当成已完成H200检索或ASR质量结果。

## 0.176 2026-10-21 best27完整曲线与旧域对照核验完成（本轮收尾）

用户返回ep27_details.tar.gz，要求总结后开始下一任务。本次只读核验小包并归档，
不训练、不加载完整Qwen、不重新推理、调参或覆盖任何outputs；本阶段到此收尾。
接收包6,300bytes，SHA256为
6a355eaa84badd6b3937560fa60ab373e7c3b2ab26e4156ad0d1c6035ba1953e。
白名单检查包恰含3个普通文件，无链接/其他路径；不执行附件内容。原始成员逐字节
存入docs/results/2026-10-21_english_newval_ep27_details/，附verification.json与本地
sha256.txt；原始包未带发件端SHA清单，不把本地生成的SHA说成已跨端核对过包字节。

| 原始成员 | 字节 | SHA256 |
|---|---:|---|
| head/metrics.jsonl | 25584 | 49ba23b76006fc672fd7521d74285e3185b0317a0cb7ffb7fc716760c7271a35 |
| old_domain_control.json | 10924 | 69cb6f4c3dc2baf57e92ce5173f0ffe181ffdcbcf36aa57954bb8e434eb2f0ba |
| training_completed.json | 292 | f129d1758a32f443f8a3a36e5fa9329896b22a39d1df17a2a1f9110425c415a4 |

### 身份链及完整性

marker.report_sha256=57c44…0398，与0.175已保存完整head_report实际SHA完全一致；
marker、旧域对照及0.174检索声明的Head SHA均为872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466。
marker.plan_sha256=dbf8e6cce9bd07fefcb801107cbea0c50a22f3874fbc51c35ace88d2a01fee83，
与此前smoke记录一致，但plan原文件不在包内。marker/control均test_set_used=false；
control用途明确为old_domain_control_only_not_used_for_training_decisions。
本地没有权重、完整训练/验证清单或缓存，不能因此声称逐字节核验这些大资产。

metrics恰好27行、epoch连续1–27，无重复或缺口；逐轮train/validation及三语言
errors/reference均复算PER，组计数加总、宏平均、固定样本数/分母一致。末轮train/
validation与已归档最终报告一致；按macro→worst-group→loss复选best为27，epoch
耗时总和18091.924452942098s与报告一致。旧域control中的checkpoint_epoch_metrics
与新域第27轮validation记录逐字段一致；ES/PT在新旧验证口径的样本/reference/errors
分别完全相同。以上检查和重放明细保存在verification.json。

### 实际学习曲线和早停

以下均为PER百分比，完整27行保留原件，不由抽样节点补造其他epoch：

| epoch | Train PER | 新EN PER | ES PER | PT PER | Macro PER | 本轮使用LR |
|---|---:|---:|---:|---:|---:|---|
| 1 | 13.6765 | 8.3418 | 4.9536 | 11.8978 | 8.3977 | 3e-4 |
| 7 | 7.0887 | 5.9290 | 3.9444 | 9.4120 | 6.4284 | 3e-4 |
| 20 | 6.6267 | 5.5630 | 3.7519 | 9.0005 | 6.1052 | 3e-4 |
| 21 | 6.5009 | 5.4655 | 3.7381 | 8.8346 | 6.0127 | 1.5e-4 |
| 26 | 6.4503 | 5.4288 | 3.8189 | 8.7329 | 5.9935 | 1.5e-4 |
| 27 | 6.3941 | 5.4122 | 3.7235 | 8.7801 | 5.9719 | 7.5e-5 |

按原scheduler在validation loss上以abs threshold0.0005、patience2/factor0.5重放，
仅在epoch20、26结束后降LR，因此epoch21、27开始使用新LR；逐轮记录全部匹配。
metadata记录的是本轮训练使用LR，不能将epoch20当成已经用1.5e-4训练。
新EN最低PER/最小loss/Macro最低均在27，ES单独最低在22，PT单独最低在26；实际
统一Head仍为27，不拼接不同epoch的分语种最低值当同一个checkpoint的性能。

早停按Macro PER、绝对min_delta0.0005（即0.05个百分点）、patience6重放。最后一次
达到显著改善判据在epoch21；22–27逐轮stale为1–6。21→27宏PER实际仍改善约
0.04081675个百分点，低于0.05pp要求，因此第27轮保存新best并同时早停，与原实现
完全一致。下限LR5e-6未达到，最终为7.5e-5。只能说按指定规则正确停止、后期缓慢
改善并有小幅波动，不能说已完全无改善空间；本轮不因此自动续训或改变早停。

### 旧英语域的公平比较终于补齐

旧域对照为原8101条：EN2808/ES2631/PT2662，reference分别77701/129906/163324。
本轮best27在这个旧域实测为：

| 语种（同一旧验证口径） | 原480h best25 | 过渡轮best9 | 本轮best27旧域 | 相对原480h变化pp |
|---|---:|---:|---:|---:|
| EN | 3039/77701 = 3.9111% | 13561/77701 = 17.4528% | 16268/77701 = 20.9367% | +17.0255 |
| ES | 4857/129906 = 3.7389% | 5230/129906 = 4.0260% | 4837/129906 = 3.7235% | -0.0154 |
| PT | 14464/163324 = 8.8560% | 15068/163324 = 9.2258% | 14340/163324 = 8.7801% | -0.0759 |

旧域总体35445/370931=9.5557% PER，旧域Macro11.1467%、loss0.501826831。原模型
对照引用0.141既有记录，本次没有重跑原Head；本轮control的组PER与S/D/I及总体
加和均已本地复算。旧EN替换12177、删除2453、插入1638，合16268错误；这些计数
不能直接证明口音、MFA标签或90类音素是根因。

关键修正：此前新EN5.4122%只说明新来源验证集结果；现在同一个best27在旧EN域为
20.9367%，相比原模型同集3.9111%明显退化，相比过渡轮best9的17.4528%也更差。
新旧验证结果显示明显域差异，不能将本次换英语训练总结成“英语全面提升”。西葡
在保留验证集上基本保持并略好。训练来源、验证选点及预算/早停策略均发生过变化，
本次证据不能把旧域退化单独归为某个训练参数、口音或标签错误。

CTC PER仍不等于热词召回或最终WER。0.170新Head epoch21在固定干净英语4k检索
得到88.2259% Top7，与这次旧域CTC PER并不矛盾，不能拿不同指标抵消已测退化；
0.174 best27带噪流式召回及下游实际转写另按各自口径记录。

### 阶段结论和交接边界

新英语替换训练best27的最终报告、27轮过程、LR/早停及旧域对照现已归档核验闭环。
没有独立核验全部训练/缓存/音频原件，不延伸成数据全链认证或最终ASR验收。本轮
无需用户再补同样文件、运行GPU或重训；已有Head、词表、门控、交付和全部outputs
保留，不自动替换/回滚Head或扩跑实验。下游流式ASR、Ziyan控制实验及其他旁线仍
按各自记录保持未闭环，不能标成由本轮解决。用户表示接下来开始另一任务，待新指令。

本地/跟踪/实际远端归档起点b714b3621940abe19d2cebf57cbf595c435431cb。只新增小结果
原件及文档，保留7个原测试修改和未知资料；验证包成员/原件SHA、报告身份、27轮
算术和选择规则/时长、独立重放scheduler与早停、旧域S/D/I及分组一致性、归档
sha256.txt及git diff --check，不重复与文档归档无关的模型单元测试。
提交推送codex/g2p-coverage-scan，实际远端SHA见交付回复；拉取归档不需要重跑训练。

## 0.175 2026-10-21 新英语同域验证best27最终训练报告归档与三阶段对照（完整核验见0.176）

用户要求补齐“27轮完整训练结果，对比之前”。本次从既有终端附件提取完整正式
head_report，不再把smoke_head epoch1或早先epoch21预览当成最终训练。已归档：

- docs/results/2026-10-21_english_newval_ep27_head_report.json：完整最终head_report，
  SHA57c44ca7d3cebea5696de607efd2af2bb3b26df231e951afde11b9ee84e90398。
- docs/results/2026-10-21_english_refresh_oldval_ep9_head_report.json：同一批新训练数据、
  仍用旧英语validation的过渡轮完整head_report，SHA
  fc2585700aa980a3d558808e8bee0dbcf71ed3a116f7a4a9bd80409eed00d43b。

来源分别为此前b53a3c88附件与a5edc3c4附件，文件SHA为
f1fb358d09d6da23724f34cf7db561cf5993fee0adea9aeebac3a7f074871c7b、
4534f6caa4b5e65071ab110ead4dff38ac4f986253c279a27061b0ba1b552b2e。
日期为本地补归档日期，不推断原训练开始/结束日期。原始480h best25对照继续引用
0.141已归档的错误数/reference，不伪造未收到的该次完整report原件。

关键身份交叉核验：将终端完整head_report保存为JSON后的SHA恰好等于后续0.172
工区smoke回传checkpoint.training_report_sha256（57c44…0398），不是只有百分比相近。
对应smoke终端附件SHA为f159d0617a8ecbbcbfc9b97900a20596903e3f8eb2f30fefae25e101aeacf5b0。
因此完整最终报告内容已归档且与后续声明报告SHA一致；仍未在本地访问工区原路径、
权重或缓存，不把这种跨回传一致性表述为已本地检查全部工区文件。

### 本轮最终正式训练事实

正式目录outputs/english_countries_refresh_newval_v1/head，选用ctc_head_best.pt，
Head SHA872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466由后续
smoke及0.174检索回传共同声明，已固定在检索配置中；非latest/smoke目录。

| 项目 | 已返回最终值 |
|---|---|
| 请求/实际/最佳epoch | 40 / 27 / 27，early_stopped=true |
| 训练样本/分片 | 881749 / 1723，每轮all_samples_once |
| 验证样本/分片 | 7492 / 15；EN2199、ES2631、PT2662 |
| Head/Encoder | Temporal2x，h512/k5/dropout0.1，838746可训练；Encoder317477504参数冻结 |
| 最佳/末轮训练loss、PER | 0.2415060505、6.3940862%；PER为三语训练音素微平均 |
| 最佳/末轮验证loss | 0.2587039483 |
| Macro验证PER / 混合音素PER | 5.9719233% / 6.3635082% |
| 选点/早停指标 | validation_macro_per；worst-group PER/loss作选点后续排序 |
| 初始/最终/下限LR | 3e-4 / 7.5e-5 / 5e-6；未到LR下限仍可触发早停 |
| patience / min_delta | 6 / 0.0005；minimum_epochs=5 |
| batch / weight_decay / clip / seed | 256 / 1e-4 / 5 / 20260825 |
| 报告训练时间 | 18091.92445s，约5.0255h，含各epoch训练/验证；不含Encoder提取 |
| cache_sha256_verified / test_set_used | true / false（报告声明） |

best与final分组PER、loss、训练统计一致，因为最终epoch与best同为27。保存best依据
任何更优的选择排序，而早停计数只有超过min_delta的改善才重置；best27与第27轮
早停不构成矛盾。实际逐轮何时降LR/改善/累计stale仍须读取metrics.jsonl，不能由
最终report倒推出27个epoch的数据或宣布曲线完全收敛。

### 与原480h和过渡轮的比较

| 阶段 | 最佳/实际epoch | 英语验证域 | EN PER（errors/ref） | ES PER（errors/ref） | PT PER（errors/ref） |
|---|---|---|---|---|---|
| 原英语480h，0.141 | 25/26，上限30 | 原英语2808条 | 3.9111%（3039/77701） | 3.7389%（4857/129906） | 8.8560%（14464/163324） |
| 换英语后仍用旧validation，0.169 | 9/15，上限30 | 原英语2808条 | 17.4528%（13561/77701） | 4.0260%（5230/129906） | 9.2258%（15068/163324） |
| 新英语同域validation，本轮 | 27/27，上限40 | 新英语2199条 | 5.4122%（2943/54377） | 3.7235%（4837/129906） | 8.7801%（14340/163324） |

本轮只替换英语训练480h，西葡训练480h及原西葡验证数据/标签/特征保留；新英语
validation从未入选本轮480h的合格池冻结。过渡轮与本轮使用同一881749条训练清单/
已完成Encoder特征，但新validation负责选点、调度、早停，Head从新初始化，且
epoch上限30→40、LR下限1e-5→5e-6、min_delta0.001→0.0005。因此不是只换验证数据
的一项消融，不将差异单独归于新英语语料、训练轮数或早停参数。

ES/PT验证样本数与reference_phonemes三阶段相同，按同组PER可比较数值：本轮相对
原480h，ES少20个错误、下降0.015396个百分点；PT少124个错误、下降0.075923个百分点。
相对过渡轮ES少393错/下降0.302526pp，PT少728错/下降0.445740pp。结论是这两组
保留验证集表现基本保持并略好，不宣称统计显著改善或全部来源/语种均泛化提升。

英语3.9111%与17.4528%使用同旧验证集，可描述过渡轮在旧英语域上的退化；新5.4122%
来自不同新验证集，不能把17.4528%→5.4122%称为同集降低12.04pp，也不能从旧3.9111%
与新5.4122%断定本轮退步。三阶段Macro/overall PER同样因英语验证域变化不能直接
当同集提升。需要best27在旧validation上的old_domain_control才能补齐旧域比较。
CTC PER、0.170/0.174检索召回和最终ASR WER继续分开，不用检索代替训练验证。

### 完整逐轮结果仍需回收，现有文件只读即可

最终head_report已收齐并归档；缺的“完整27轮训练”具体指head/metrics.jsonl，及用于
旧域公平比较的old_domain_control.json。源码execute(run)在返回run_completed前调用
old_english_control，并将文件放入return_bundle；按成功返回路径文件应已产生，但
此前终端JSON只打印head_report，不包含control数值，所以是尚缺回传，不先要求重跑。

在容器内项目根目录，只读现有3个小结果文件打一个独立小包，不执行训练/Encoder，
不读/传权重，不包含大plan或缓存，不覆盖原return_bundle：

```bash
R=outputs/english_countries_refresh_newval_v1
O=$(mktemp -d outputs/english_ep27_details_XXXXXXXX)
tar -czf "$O/ep27_details.tar.gz" -C "$R" head/metrics.jsonl old_domain_control.json training_completed.json
ls -lh "$O/ep27_details.tar.gz"
```

返回这个新小包；若tar报告缺文件，保留现场并返回报错，不为补档自动重训或重算对照。
收到后核对marker的报告/Head SHA与已归档身份、metrics epoch连续1–27及最后/最佳
统计，并据old_domain_control校验best27及旧8101条/EN2808的同域对照，再补曲线和
早停/学习率轨迹。源码/运行参数保持不变；文档更新后不必启动旧run或用新Git恢复训练。

本轮归档起点本地/跟踪/实际远端0430559f2efdbb90e00c8eee301c1bc251d22d0c；原7处
测试修改和未知资料不动。只改文档与小结果JSON，已复算两份报告分组PER/Macro/
overall、样本总数、同集差值及最终报告SHA交叉身份，git diff --check；不重复与归档
无关的模型单元测试。归档提交推送当前codex/g2p-coverage-scan，远端SHA见交付回复。

## 0.174 2026-10-21 三语6,800条带噪因果前缀检索完成（终端结果归档）

2026-10-22补充同事使用说明：
[MULTILINGUAL_HOTWORD_EVALUATION_GUIDE.md](MULTILINGUAL_HOTWORD_EVALUATION_GUIDE.md)。
集中记录联合4k表/epoch27身份、干净及带噪目录、三类评测入口、恢复和新目录政策、
Top5/7分母与报告字段、下游交付；区分历史干净Head与最终带噪Head，仅整理已有
记录，不重跑推理或新增实测。分支codex/g2p-coverage-scan，工区在仓库根目录
git pull --ff-only origin codex/g2p-coverage-scan后阅读，无需返回新的结果文件。

用户要求归档本次最新终端JSON，并核对前面是否有遗漏结果供讨论。本轮只归档已返回
数据、复算指标和梳理状态，不重跑推理/训练、不调门控、不改任何outputs。完整终端
摘要按字段和值保存为docs/results/2026-10-21_multilingual_stream_ep27_noise10db_terminal_summary.json，
本地归档文件SHA256为7d29ccaf104b82adc1709b0c476d63233136ea0d86d19633971537d50836415b。
这是接收日期，不推断工区实际运行起止时间；该SHA是本地保存的JSON文件身份，不冒充
工区terminal_summary.json原字节SHA。

### 范围、身份和证据等级

回传status=completed、effect=same-step：EN6,000/ES400/PT400，共6,800条、26,354次
实际前缀检索。两个mode共享推理，mode_boundary_count=52,708=2×26,354；不是
52,708次Encoder推理。每个时间点从当前累计前缀重新检索，不永久累加候选。
配置仍为原三语联合4,000词面/4,012发音索引、90类vocab及原Anchor/门控，best27：

```text
Head:       872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466
Table:      6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b
Vocab:      0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4
run_config: 47134f2d4489a260ea2bdeb8a7d120b0b4e0658ff687335952846ebe20224743
```

前三项声明SHA与已交付固定配置逐一一致，回传的epoch=27。run_config SHA首次收到，
本地没有该工区文件，不能独立复算。回传没有git_commit，不能把本轮归档起点
fd2a460933e024af422dca55d15512ceb80147de直接称为实际运行代码SHA。该起点经本地/
origin跟踪及实际远端核对一致；7处既有未提交测试修改和未知资料未改动、未纳入提交。

本地只核对终端摘要的结构、计数、比率和声明身份，没有收到本次完整run_config、
delivery、逐样本/前缀、权重或音频原件，不声称已独立验证工区文件字节、实际ASR
消费或词级正确性。schema_validation=pass、downstream_format_validator_passed=true
为工区回传检查结果；不能升级成下游模型已真实联调通过。qwen_decoder_used=false、
word_timestamps_available=false，未测本轮CTC PER、最终ASR WER或及时召回。

音频全部取各自标称10dB目录：英语2026-10-09三国目录，西葡2026-09-29下
wave1_标贝/wave2_澳鹏/wave3_海天/wave4_澳鹏0921。旧干净参考仅作身份/评分，
实际SNR未独立复测；本次摘要只含分语种汇总，不伪造分波/分国家结果。

### Top5/Top7实测汇总

last_context是末步实际交付列表，same-step末步输入完整音频；ever_delivered是跨
时间曾交付词面的并集，每词×音频只计一次，集合可大于K。两者分母均为原冻结目标
词×音频，不能把ever当固定TopK、实时及时召回或永久并集的线上策略。

| 语言/K | 目标分母 | 末步命中（召回） | 曾交付命中（覆盖） | 输出词×音频：末步/曾交付 | 完整参考外：末步/曾交付 |
|---|---:|---:|---:|---:|---:|
| EN5 | 4374 | 3654（83.5391%） | 3732（85.3224%） | 12145/13059 | 8488/9324 |
| EN7 | 4374 | 3698（84.5450%） | 3759（85.9396%） | 12726/13428 | 9025/9666 |
| ES5 | 724 | 616（85.0829%） | 632（87.2928%） | 1879/2927 | 1246/2277 |
| ES7 | 724 | 638（88.1215%） | 645（89.0884%） | 2382/3337 | 1725/2673 |
| PT5 | 736 | 529（71.8750%） | 564（76.6304%） | 1794/2358 | 1248/1776 |
| PT7 | 736 | 553（75.1359%） | 585（79.4837%） | 2160/2539 | 1589/1935 |

全部12项hits/denominator与给定recall一致，样本数/分母/两个mode计数一致；目标命中
不超过参考内输出，Top5不超过Top7，曾交付覆盖不低于末步，汇总关系均通过复算。
参考外只表示未匹配完整参考，不等于模型在该前缀听到了幻觉或下游最终落错字；
参考内非目标输出也不能当成错误，selected-target_hits不能直接作为误匹配数。

阶段结论：本次末步Top7为EN84.55%/ES88.12%/PT75.14%，葡语仍为这批任务中较低者，
这些汇总不定位具体根因。Top7相对Top5新增EN44/ES22/PT24目标命中，同时新增
537/479/341参考外词×音频；不能只按召回选择最终ASR用K。Top7的曾交付覆盖相对
末步再多61/7/32目标对，同时多641/948/346参考外词×音频；这些是曾出现后在末步
消失的计数，不证明热词及时送达或ASR受益。

不能将epoch21干净英语88.2259%与本次best27带噪84.5450%的差直接叫噪声损失或流式
损失（Head和音频同时改变）。0.150旧西葡带噪还使用不同表组成，也不是本轮联合4k
同Head流式对照。不自动重跑这些对照，也不把本次末步结果倒填为另一独立run已完成。

### 耗时、文件与后续取证

prefix_compute_seconds为前缀准备+检索：累加602.8678998s，平均22.8757646ms，最大
521.4525450ms，超过2s为0。平均值与总量/26,354一致。不是含预检、加载、写盘的全程
墙钟，不含ASR解码/网络/等待，不能当端到端实时验收或推广成其他时长音频的保证。

工区输出仍为：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step
```

delivery只有stream_contexts.json、durations.json、audio_manifest.json。Top5/7为
stream_contexts.json的RAG_top5_stream/RAG_top7_stream两个顶层字段；音频对应数组
按2s、4s、…、实际末尾表示时间点，不是分别生成每个时间段文件。prefix_shards与
delivery并列，保存逐前缀完整CTC/候选/得分/截止点及恢复身份；不是第四份交付JSON。
这些结构已按源码核对，尚未本地读取本次工区三文件原件。

本轮已完成，无须为归档再跑GPU或训练。后续优先回收下游无RAG/Top5/Top7的真实
流式转写摘要，确认实际context采用及same-step时序；如要解释参考外输出或词后
召回，再在工区从已存前缀提取少量固定样例，不以汇总替代听审、不要求传大包。

### 前序归档核对与待讨论项

以下区分“已收到但记录不完整”与“尚未收到”；本轮不补造缺失结果，不自动续跑。

| 项目 | 已有记录/证据 | 尚缺或可补记 |
|---|---|---|
| 新英语fast准备、smoke/cache/pilot/formal | 0.168–0.169已有规模、SHA、缓存及旧验证训练结果 | 不是未完成；原章节部分历史文字仍描述当时待执行 |
| 新英语同域validation准备 | 对话已回2199条/4.0023396658h；合格未选池343.5264709325h；EN/ES/PT为2199/2631/2662、合7492 | 0.169尚未独立补齐这次实测prepare结果和三国配额；完整plan/分层/清单SHA未返回 |
| best27正式训练 | 0.175最终报告及0.176完整27轮过程/旧域对照均已归档核验 | 训练结果归档已闭环；完整缓存/plan/音频原件不在回传包，不冒称已独立验过所有大资产 |
| epoch21干净英语6,000条/4k | 0.171已有Top5=3804/4374、Top7=3859/4374及Head SHA | 附件终端分组/原小report尚未单独形成结果文件归档；不是需要重跑的实验 |
| epoch21或best27独立整句带噪英语 | 对话曾安排，0.171有best27入口 | 本聊天未见这两个独立run完成摘要；本次流式末步不替代其独立run身份 |
| 20条流式smoke | 0.173已记20条/68次、身份及耗时 | 完整产物未回；用户已明确取消回大包前置，不再阻塞全量 |
| Ziyan原712条及身份核验 | 0.164记原召回，0.166记两包合计11项SHA通过 | 0.165旧标题“待回传”为历史状态；399漏项完整分解和听审仍未闭环 |
| Ziyan删37词、每句目标≤4 | 0.167有654条/649目标对的固定实验入口 | 未见成功结果回传，不能宣称控制实验已完成或改善 |
| 60条ASR提示顺序/优先级、旧4k竞争、旧英语18例、759目标MFA | 0.164已集中归档，并区分原件核验和终端摘要 | 固定案例听审、13个外语项核验仍未完成；不是新的大样本证据 |
| 三语4k新MFA查看dict、葡语标签/310P旁线、本轮下游流式ASR | 仅有各自历史入口或未闭环记录 | 尚缺成功产物/实测摘要；不把RAG指标称作其最终ASR结果 |

其中可优先补记已在对话存在的新validation prepare和epoch21干净完整摘要；如要讨论
新英语对旧域影响，应先看已有old_domain_control.json的小摘要；如要决定产品使用
Top5还是Top7，应先看本轮下游ASR对照。尚未返回的内容只列待办，不推断已失败/未执行。

本轮文档/小结果JSON变更，验证JSON可解析、声明身份与配置一致、12项比率/计数/
时延均复算，以及git diff --check；不为归档重复运行不相关的模型测试或完整模型。
文档同步仍为git pull --ff-only origin codex/g2p-coverage-scan，实际远端SHA见交付回复；
拉取文档不要求重跑已完成推理，也不要求用新Git恢复旧封存run。

## 0.173 2026-10-20 三语流式全量6,800条与终端精简指标入口（结果见0.174）

0.172工区终端已回传：20条（智利7/哥伦比亚7/墨西哥6）、68次前缀检索，Top5/7
同源、供方validator通过；其utts=40/chunks=136为两个mode累计，不是40条独立音频。
best27实际Head SHA为872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466。
sum前缀准备+检索2.9651s、最大0.6738s、超过2s为0；加载另48.4s。这是终端摘要，
完整文件尚未独立核验，没有实际ASR消费、WER、及时召回或端到端时延结果。

用户明确表示return_bundle过大不能回传，要求直接英语6,000条并补跑之前西葡，各
语种重新测流式指标。该决定替代0.172“先回包/联调再全量”的顺序；不继续要求传大包。
新入口无smoke硬前置，也不冒称下游已确认时序。分支codex/g2p-coverage-scan；起点
本地/跟踪/实际远端均0b4a4bfe46fe0e8a27993a00643352106eaac247。7处既有未提交测试
修改及未知资料保留；新增模块复用旧代码，未修改训练/Anchor/门控/旧配置或旧outputs。

### 范围、指标与交付

新增configs/multilingual_stream_contexts.workzone.json、scripts/run_multilingual_stream.py、
evaluation/multilingual_stream_assets.py、evaluation/stream_retrieval_metrics.py和
inference/multilingual_stream_run.py。完整操作和口径见docs/STREAM_CONTEXTS.md首节。
英语三国各2,000条、合6,000，西葡分别wave1–4各100条、各400，总6,800。英语沿用
2026-10-09-带噪10dB根目录，西葡沿用2026-09-29-带噪10dB/<当前wave目录>/es或pt/wav。
保留各自原ID/参考/target_ids，英语分母4,374，西葡从同一sealed clean4k plan/report
重算并验证各自原分母；不重建目标/词表，不采用带噪目录新参考。完整三语4k/4012
发音及90类vocab和0.75/0.35/0.5原门控不变；Head通过训练marker验证并额外锁定上面
已回传实际SHA。单进程按三语顺序跑，每语种加载一次detector，保留原语种提示。

继续调用0.172原因果音频前缀/检索函数和原供方validator，不改变两种effect定义。
same-step末步为完整音频前缀，next-step最后检索不额外塞尾槽；指标只按实际交付
context评分。每语种/数据集、每K分别给last_context目标召回及ever_delivered曾交付
覆盖率，后者每词×音频只计一次、集合可能>K，不当成固定TopK或及时召回。两者还
给输出词×音频及完整参考外数量；无词级真值，不宣称逐前缀精确率/词后延迟，未跑
LLM/最终ASR WER或CTC PER。参考仅在检索后的独立计量使用，不进入前缀推理。
原参考条件化词表限制仍在；旧Head+clean不是新Head+noise流式配对对照，不自动
把差值归因于噪声或流式。时延只计RAG，不包含ASR/网络及加载，SNR未独立验证。

下游取delivery下stream_contexts.json、durations.json、audio_manifest.json；英语key
保持原样，西葡本次配置声明当前wave目录名@@es/pt，路由尚未
经下游运行确认。逐前缀取证在prefix_shards，逐样本计量在sample_metrics.jsonl。
终端最后直接打印三语精简JSON，保存terminal_summary.json；可直接贴回，不传大包。
summary_bundle.zip仅含summary/report/checkpoint小身份和包内SHA，不包含run_config、
delivery、音频清单或因果取证；完整产物留工区并SHA封存，不回传大包不是阻塞条件。

### 工区运行和恢复

容器内项目根目录、原Qwen环境；完整模型只由用户在H200加载：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_multilingual_stream.py --effect same-step --gpu 4
```

实际远端SHA以交付回复为准，4换成已分配物理卡。正常命令自带完整预检，不必另跑
--audit-only；保留阶段、实际样本/前缀总数、耗时和10秒活动提示。默认独立新目录
outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step，不覆盖旧20条。
中断后同代码/输入重复命令加--resume，校验每个前缀摘要/PCM后只计算缺失；已完成
目录恢复只核验SHA并打印摘要。next-step另建full_next-step，不能混用effect恢复。
独占锁防并行覆盖，不删锁绕过活进程。回传终端最后JSON即可，或：

```bash
cat outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step/terminal_summary.json
```

三份下游文件仍需对齐manifest，在下游运行后才有真实流式转写指标；本次全量入口
只负责RAG生成/计量。新全量尚未在H200实际运行，不能引用本地夹具作为三语召回。

工区补充：9e8bc26版本预检在3.5s因旧wave1/es/wav不存在退出，尚未进入写结果/加载
完整模型。用户明确四目录当前为wave1_标贝、wave2_澳鹏、wave3_海天、wave4_澳鹏0921。
补充wave_directories映射及对应下游key（如wave1_标贝@@es）；原waveN_es/pt逻辑分组、
封存参考/target_ids及评分口径继续保留，不在磁盘重命名、不改旧outputs。拉取后直接
重复上面运行命令即可，不删目录、不需--resume。英/西/葡本轮均从标称10dB目录读音频，
不混用clean WAV；原clean参考仅作身份/评分，实际SNR仍未独立复测。
目录修复定向13 passed（28.55s）、全量783 passed（118.74s）；修改范围Ruff/Mypy及
git diff --check通过，未知文件和7处既有测试修改继续保留。

### 本地验证

新增5项三语测试，与0.170–0.172定向回归共24 passed（50.56s）；全量783 passed
（114.43s）。另从HEAD归档原测试夹具到临时目录，独立验证新三语/旧英语流式共
13 passed（24.87s），确认不依赖本地7处未提交测试修改。覆盖三语提示/完整索引、
原分母、参考不入前缀、曾交付与末步区分、next-step短音频空context、SHA漂移/
坏路径/门控阻断、缺失前缀续跑、完成不重载、旧文件不改及精简包不含大产物。
全量通过后只补齐西葡时长桶到三语汇总，防止继承英语单语数量，并定向复验。
新增文件Ruff/strict Mypy、CLI help、git diff --check通过；全仓既有Ruff11项/Mypy8项
未改。均为CPU小音频/真实小Head及模拟检索边界，不加载完整Qwen，不声称H200实测。

## 0.172 2026-10-20 供方2秒流式context适配：因果前缀与20条smoke入口（终端结果见0.173）

用户提供stream-interface-接口手册.rar，先要求只读解释、随后批准实现。包只含Markdown
手册、示例context、示例durations及校验脚本，是文件交付契约，不是网络服务地址。
原包SHA485d5f8d0cc8d257f1cc593b505c0c3f15bc0017510b61cdb540dfe30cb8e895；
原validator SHA35312007820f382c29448cae1633f9495679c066548bddb40c5b4a8f2ee29cbb。
附带材料只作接口依据，没有执行其中docker命令或联系下游；校验器经源码审阅后原样
保存为tests/fixtures/stream_interface_v1_validator.py.txt用于本地/工区兼容性检查，
样例和来源清单同存tests/fixtures。没有改供方脚本来掩盖接口矛盾。

分支codex/g2p-coverage-scan；起点本地/跟踪/实际远端均
055a6d47f45d04536776565d2b1a65be6f9e6922。既有7个未提交测试修改及未知资料保留，
不覆盖/纳入提交；本次新增模块，不改0.170/0.171及训练源码、旧配置或既有outputs。

### 实现和口径

新增configs/english_stream_contexts.workzone.json、scripts/run_stream_contexts.py、
evaluation/stream_context_assets.py和inference/stream_context_{contract,run}.py。
继续引用0.171最终best27、完整三语4k/4012发音、90类vocab及原Anchor/门控，
通过训练完成marker核验Head/report/plan。音频仍是0.154三国各2,000条10dB目录；
干净原参考只作资产身份核验，不进入前缀模型或候选生成。不重训、不重新MFA、
不使用全句召回复制时间线、不调用LLM/vLLM解码，不声称新增ASR指标。

只读封存clean4k的plan/report/索引/表快照，验证源配置和底层检索代码身份。扫描
6,000条音频头，按固定seed和实际可用时长桶选择三国20条smoke，仅对这20条读字节
hash；full才对6,000条hash。保留重复样本，不强造整数边界，不声称掌握词级时间。
原始采样点决定2秒边界：每次只读新到达的帧，组合当前前缀，单声道/必要重采样也在
截断之后做，再独立跑Encoder→Head→Anchor。旧全句特征不复用，每个前缀最多一次
推理，词面/发音合并后先用原质量门控，再导出Top5和Top7两个mode；不永久并集。

供方key映射en_chile→encl@@en、en_colombia→encol@@en、en_mexico→enmx@@en。
候选为原word/胜出phoneme/score对象，按分数排序，空列表[]。供方手册要求全候选
由ASR截K，但其validator把>K判ERROR；本版一次检索、分别按mode截5/7，完整合格
列表保留在prefix_shards。不做Vocabulary拼接或orig/join_phon/nospace改写。

时序没有默认值，--effect必须明确：same-step使用当前前缀，next-step首步[]、整体
延后一块；均恰好ceil(T/2)槽，尾段保留、整数边界不多槽。next-step最后一次前缀
不被消费，<2秒音频context全空。两种时序分别对应手册第3/6节，**实际主口径仍需
下游确认**。当前same-step命令用于先检验前缀链路，不能说下游已经确认当步使用。
语音末词在next-step可能来不及注入，不可误归为CTC漏召回。

### 运行、结果和恢复

详细契约和操作见docs/STREAM_CONTEXTS.md。容器内项目根目录、原Qwen环境：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_stream_contexts.py smoke --effect same-step --gpu 4
```

实际远端SHA以交付回复为准，4换成已分配物理卡。前缀/样本计数、耗时及10秒活动
提示覆盖准备、加载、推理和封存；正常命令包含完整预检，不必再单独--audit-only。
默认新目录outputs/english_stream_contexts_ep27_noise10db_v1/smoke_same-step。
中断保持同代码/配置/Head/音频/effect，重复上面命令加--resume；每个前缀摘要与PCM
输入身份校验后复用，已完成目录恢复只验SHA、不加载模型；不覆盖旧目录或嵌套输入。

回传该目录return_bundle.zip：含delivery/stream_contexts.json、durations.json、
audio_manifest.json、因果审计和report/config/checkpoint小身份及包内SHA，不含WAV/
权重/全量前缀原件。两个mode共享一次前缀推理，JSON不增加_meta假设。smoke只含
选定20条，下游必须按audio_manifest选同样音频，不能将源目录6,000条当作smoke范围。
内部自检及供方原脚本都会验证，结构通过不等同于ASR采用已生效或实时速度已验收。

先核对smoke、让下游验证真实context采用/部分及最终转写，再用相同effect启动full：

```bash
python -B scripts/run_stream_contexts.py full --effect same-step --gpu 4 \
  --smoke-run outputs/english_stream_contexts_ep27_noise10db_v1/smoke_same-step
```

full要求已完成且同Head/配置/代码/音频/effect的smoke，输出独立full_same-step；不会
自动从smoke扩跑。next-step另建smoke_next-step/full_next-step，不混目录或指标。
prefix_compute_seconds仅RAG前缀准备/计算耗时，不是ASR端到端时延。下游最终WER、
目标命中、参考外落字与实际context采用仍待真实运行。本地没有访问H200完整模型。

### 本地验证

新增8项流式适配测试及旧入口回归共19 passed。使用真实小WAV验证任意采样点、
短句/整2秒/不足2秒尾段、8k双声道先截断再重采样、改变未来1秒不改变前2秒输入；
模拟检索边界验证完整多语索引、每前缀只算一次、Top5/7同源、两种effect、候选可消失
而不永久累计、原校验脚本通过和>K矛盾。还覆盖无输出预检、坏分片拒绝、缺失前缀
续跑、音频漂移阻断、full必须匹配smoke和旧资产不变。全量778 passed（101.70s）；
随后仅收紧汇总内存，不在内存保留全部前缀CTC详情，仍逐前缀写盘并由定向回归验证。
新增文件Ruff/strict Mypy、CLI help、git diff --check通过。全仓既有Ruff11项/Mypy8项
未改。全部模型/检索硬件边界为小型CPU夹具，不能当成工区召回、WER或实时性能。

## 0.171 2026-10-19 最终best第27轮与英语6,000条10dB/联合4k入口（待工区）

用户已回传0.169的run_completed终端JSON，并明确要求用最终best重新测三套带噪英语。
正式head_report显示：请求40轮，实际27轮、early_stopped=true、best_epoch=27；
best路径outputs/english_countries_refresh_newval_v1/head/ctc_head_best.pt。不是
smoke_head的epoch1，也不是0.170复制的epoch21。新英语validation为2943/54377=
5.4122% PER，ES4837/129906=3.7235%、PT14340/163324=8.7801%，宏平均5.9719%、
混合音素PER6.3635%。选择指标validation_macro_per；Head838746参数、Encoder冻结。
这些数字来自终端摘要，实际权重SHA及完整metrics/marker尚未独立回传；不据此推断
带噪检索或最终ASR效果，也没有修改/重启训练或改变早停。

0.170的epoch21干净英语结果也已终端回传：Head SHA
71853594931df390a02dc01d76574ff2b3245fe13d6a0f138c21cee1c84b8bf4；同6,000条/4k表/
4,374分母，Top5=3804（86.9684%）、Top7=3859（88.2259%）；旧Head为3175/3215。
Top7净增644，漏项1159→515；参考外输出8468→9229，不能说误匹配已解决。已复算
分组和汇总，尚未拿到该预览完整返回包，不能冒称独立核验全部输入或逐样本原件。

### 本次范围和身份契约

新增configs/english_best_noise10db_4k.workzone.json、scripts/run_english_best_noise.py、
inference/english_best_noise.py及独立测试；扩展0.170预览封装的checkpoint/preparation
边界以复用原推理/评分/词面合并/Top5/7导出。默认干净预览路径仍要求同音频SHA，
只有明确传入带噪根目录才允许audio_path变化；没有修改旧Anchor、评分、门控或训练
源码/配置。分支codex/g2p-coverage-scan，起点本地/跟踪/实际远端均
d83964e289f4fdbbe61fcd06063a48915dacfb7a。既有7个未提交测试修改和未知资料不动，
不随本次交付提交。

完整三语4k表仍为SHA6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b，
4,000词面/4,012发音变体；90类vocab仍为SHA
0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4。冻结目标和
原参考读取旧clean4k的sealed run_config/report/逐样本/索引与原干净transcripts，
复算旧3175/3215及4374分母，仅作资产/口径核验。带噪目录里的转写和词表不采用。
每组必须与原2,000个ID/参考/语言严格一致，三个带噪WAV目录为：

```text
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_chile/en/wav
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_colombia/en/wav
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/en_mexico/en/wav
```

工区读取training_completed.json，按其中SHA验证head/report.json、plan.json及best.pt；
必须是完成的正式训练、cache_sha256_verified=true、test_set_used=false、宏平均选点、
best27/完成27/请求40及所给best路径。CPU读取Head小权重，核验vocab、Temporal2x
结构/权重、epoch、validation loss/PER/errors/reference及报告分语种汇总。将实际SHA
和报告/plan身份固定进新run_config，结束及恢复时重新检查；源best变化会阻断，不
自动追随新文件或改成latest。无需用户再次手工复制、hash或寻找权重。

只扫描这6,000条带噪WAV，旧clean WAV hash来自封存plan，不重新读取clean WAV。
逐组报告changed_files/identical_files；整组全相同拒绝，部分相同明确保留计数。
snr_verified=false，字节不同不能证明实际10dB或逐条声学对应。冻结模型metadata/
检索代码及所有非音频设置，只允许明确的Head和audio_dir变更。原阈值0.75、edit0.35、
posterior0.5、Anchor Top64发音及词面Top5/7保持，不评测ES/PT音频，不跑LLM或训练。

**比较限制：**本次best27+noise与旧480h Head+clean同时改变两项，旧参考在report中
明确标为reference_clean_old_head，省略召回差值字段，不能把差值叫噪声损失。
epoch21干净88.23%也不是同Head干净对照；本次不自动额外推理6,000条clean或旧Head。
之后如需噪声效应，另行固定best27对clean配对。保持原参考条件化fillers及训练测试
隔离范围，不将目录名或本地mock当实测。

### 工区命令、输出与恢复

容器内项目根目录、原Qwen环境；完整模型只在H200加载。拉取并比对最终回复远端SHA：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_english_best_noise.py --gpu 4
```

4换成用户已分配物理卡；一条命令先核验后推理，自带阶段/实际样本数量/耗时/10秒
活动提示。可选--audit-only（不带GPU）只核验CPU小权重、带噪文件及元数据，不加载
Qwen、不写结果；常规运行已包含同样核验，无须再单独审计一遍。
默认新目录outputs/english_best_ep27_noise10db_4k_v1，拒绝已有目录/嵌套输入；所有
旧输出和epoch21结果保留。中断后保持同Git/配置/Head/音频身份，执行：

```bash
python -B scripts/run_english_best_noise.py --gpu 4 --resume
```

复用已核验样本分片，不重复完成的推理；完成后恢复不再加载模型。注意0.170封装因
本次新增边界SHA发生变化，旧epoch21目录不能用新代码强行--resume；若需恢复旧任务，
使用其原d83964e版本。这个限制不影响读取旧报告，也不覆盖旧文件。

report.json给出真实Head SHA/epoch/验证统计、当前Top5/7同4374分母命中/召回/参考外
输出、三组结果及加噪字节配对摘要。六份下游JSON在languages/en/delivery/；逐样本
仍有CTC/候选/门控取证。完成后回传新目录return_bundle.zip即可，内含report.json、
run_config.json、checkpoint_metadata.json及包内SHA清单，不含权重/WAV/特征张量。

### 本地验证范围

定向11 passed，覆盖旧干净入口身份/恢复回归、新final best真实权重读取、只使用
带噪WAV（clean WAV缺失仍可跑）、忽略带噪自带参考、完整多语索引、同分母、权重/
报告/plan/epoch不符阻断、全同音频拒绝/部分同字节报告、中断续跑不重算和旧资产
字节不改。全量770 passed（85.48s）；修改范围Ruff、strict Mypy、CLI help和
git diff --check通过。全仓仍为既有Ruff11项/Mypy8项，不改未知草稿或7处既有测试修改。
本地仅用小型合成音频/检索边界和真实小Head，不声称已获得best27带噪召回。

## 0.170 2026-10-16 第21轮Head快照：干净英语6,000条/三语4k同口径预览（终端结果见0.171）

用户确认以0.164的联合4k版本严格配对，只测en_chile/en_colombia/en_mexico各2,000条，
不测ES/PT音频，但每条英语音频继续搜索完整EN/ES/PT联合词表。固定目标分母4,374；
旧Top5=3,175/4,374=72.5880%，Top7=3,215/4,374=73.5025%。不能与旧3k表/10,046
分母或Ziyan372目标结果混比。只替换CTC Head，不调门控/核心检索、不运行LLM转写。

当前新验证训练尚在进行，用户已复制只读预览来源
outputs/english_head_preview_hLF3E7KR/ctc_head_best.pt并生成checkpoint.sha256；终端
torch.load成功，epoch_metrics为epoch21、loss0.260812525937281、
PER0.06402920539574865、errors22257/reference347607。这个PER是混合验证音素统计，
不是英语PER或热词召回；该快照也不能称为已完成40轮后的最终模型。具体权重SHA尚未
回传，入口在工区读取上述单文件清单，核对路径/字节/词表/epoch21和Temporal2x权重
结构，并将实际SHA写入新plan/report。不会读取训练中持续变化的best或暂停现有训练。

新增configs/english_head_preview_4k.workzone.json、scripts/run_english_head_preview.py、
inference/english_head_preview.py和独立测试；旧准备、缓存、训练、检索模块均未改。
起点本地/跟踪/实际远端均2b54d31d6faed1238d776617fde22a9c4bba3722；
分支codex/g2p-coverage-scan。本地7个既有未提交测试修改及未知未跟踪资料保留，
不包含在本次提交中。

### 配对契约及执行边界

源目录固定outputs/reviewed_joint_4000_retrieval_480h_v1。只读核验封存run_config、
report、词表快照、indexed_pronunciations、三份英语retrieval_details；词表SHA仍为
6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b；旧Head和vocab
SHA分别保持320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265、
0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4。
直接复用封存发音索引的完整顺序与ID，包含西葡发音，不重建MFA/目标/填充词。
核验原记录/参考/目标ID与6,000条WAV字节，重放旧Top5/Top7并检查各组、汇总及上面
固定命中数。比较旧run_config各组，除git_commit/ctc_checkpoint外字段必须一致；
旧plan记录的检索代码SHA、model config/index必须一致。完整模型分片不重新hash。
缺少/漂移的旧产物会停止，不自动重建旧outputs或放宽校验。

完整Qwen只由用户在H200加载；对这6,000条重新运行Encoder→新Head→原Anchor Top64
发音候选→原词面合并/门控→Top5/7。保留0.75/0.35/0.5与原评分，posterior仍不是
词级存在概率。阶段/真实样本进度/耗时/10秒活动提示覆盖核验、加载和推理。ES/PT
音频不读、不评测；新英语480h训练缓存不属于这批评测输入，不重算训练缓存。

### 工区命令、恢复及回传

容器内/host_home/star/q00933266/qwen3-asr-hotword，原Qwen环境。先拉取并核验本次
最终回复的实际远端SHA；5为示例物理评测卡，换成已分配卡，4号若仍训练则留给训练：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_english_head_preview.py --gpu 5
```

不用再次复制训练中的best。可选CPU预检为同命令去掉--gpu 5并加--audit-only；不会
加载Qwen或写结果，但会读取Head小权重及这6,000条WAV核验身份。正常命令已包含
同样预检，无须重复跑。默认新目录outputs/english_head_preview_4k_ep21_v1，拒绝
覆盖旧目录或嵌套输入；中断后保持同一Git/配置/快照，运行：

```bash
python -B scripts/run_english_head_preview.py --gpu 5 --resume
```

恢复校验plan/代码/输入及逐样本摘要，只计算缺失分片；成功完成后重跑不会再加载
模型。report.json含各组/汇总新旧Top5/7、固定分母、命中、参考外输出、召回百分点
差值及快照身份；六份下游JSON在languages/en/delivery/。回传默认目录下
return_bundle.zip即可，内含report.json、run_config.json、checkpoint_metadata.json
及包内SHA清单，不含权重、WAV、训练缓存或完整逐样本结果。包原子发布后再封存。
这是同口径中途预览；不宣称最终ASR WER，也不新增字节级/说话人级训练测试隔离结论；
原4k填充词依据参考排除的条件保持不变，不能当作盲词表泛化评估。

### 本地验证

当前工作区全量764 passed（76.27s），随后补充的新预测计量用例及ZIP原子发布由
独立定向6 passed覆盖。覆盖完整多语索引、只测英语、同分母
复算、新预测确实改变命中/百分点、中断恢复不重算、坏分片/输入/门控/代码漂移阻断、
旧文件不改及结果包身份。新增文件Ruff/strict Mypy、CLI help、git diff --check通过。
全仓仍有既有Ruff11项/Mypy8项，未碰这些问题或未知测试修改。本地使用小型模拟
音频/检索边界及真实小Head保存/读取，没有加载完整Qwen或产生H200召回实测。

## 0.169 2026-10-16 新英语同域验证与40轮Head训练入口（最终报告见0.175）

用户指出0.168替换英语训练来源后仍用旧英语validation选点，明确批准修正：新英语
validation必须来自未入选本轮480小时的合格池；三个原test.json继续封存。保留原三语
训练清单/特征、原ES/PT验证样本/特征，旧英语只作额外对照。用户进一步要求取消5轮
pilot，保留smoke后直接正式训练，总上限40轮、minimum-learning-rate=5e-6、
early-stopping-min-delta=0.0005。分支codex/g2p-coverage-scan；起点本地/跟踪/实际
远端均613db41eb7bc16fb74a63b8b698c7cf1f56278f3。既有未知未跟踪文件保持原样。

### 已返回的上一轮结果与结论修正

用户终端已确认旧入口smoke_completed、cache_completed、pilot_completed和
formal_completed。run_plan SHA始终76543fea57f1bead99027942e5abe37e34254ee3499c5b42f4d97999f1f02aab；
训练清单SHA10fdbdbbdb8ef352cd7a143b742fd962627b40a98a44d0b54701311f0b94fbfe；
旧validation SHA仍d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499。
全量train881749条/1723shards/138772069023bytes；validation8101条/16shards/
1155707472bytes。Head缓存SHA核验true，test_set_used=false。
上一轮formal从5续训到15、early_stopped=true，三语macro选中epoch9：英语
13561/77701=17.4528%、西语5230/129906=4.0260%、葡语15068/163324=9.2258%；
macro10.2349%。epoch15英语24.9701%、西语3.8282%、葡语8.9950%。这只能说明旧英语
验证域表现，不能宣称新英语同域不收敛/语料坏；当时未建立新的英语validation。
独立权重SHA和完整15轮metrics仍未返回，不伪称已审查完整曲线或H200权重字节。

### 新验证集的构建及身份

新增configs/english_countries_newval.workzone.json、scripts/run_english_newval.py及
training/english_newval{,_cache,_run}.py。没有修改被旧准备契约锁定的训练/准备模块，
也不放宽旧数据身份。新入口核对旧封存report/contract/config、上面三项已返回SHA、
旧run_plan全部输入、两套原缓存index/config/summary、vocab及model config/index。
原完整模型仍仅在H200本地路径加载，不下载或在本机加载。

以实际881749条训练清单为准排除所有语言已训练ID/规范路径，再排除全部旧validation、
旧heldout保护路径和新三国test路径。从已封存released_english_pool.jsonl重用原
original_ready/temporal2x recovery释放规则、原路径冲突隔离和去重优先级。不能把未选中
原始行直接视为合格。只读清单/JSON元数据，不重新执行MFA或全量WAV字节扫描。

新英语validation目标小时数读取旧英语validation真实总时长，不硬编码为4小时或2808条。
对剩余合格池按来源×release×时长×标签密度×CTC压力分层、按可用时长比例分配，
seed20260923，完整音频略超目标；三国必须均有容量且被选中，不足明确停止。新英语
记录保留原ID/路径/文本/音素/来源，只改变split和新数据版本。报告容量、排除原因、
分层配额、实际小时/条数、交集和SHA。训练清单和480小时选样不变。仍为用户确认的
path_only_no_audio_byte_audit_user_requested，不声称不同路径同字节/说话人隔离。

原西葡validation原行另存legacy_es_pt_rows.jsonl，可与旧清单过滤出的字节比较；新
混合validation仅更新版本元数据，ID/音频/标签保持。新目录默认
outputs/english_countries_refresh_newval_v1，拒绝已有目录和嵌套输入目录。准备到封存前
中断的目录保留现场、另选新目录；完整plan可被后续阶段严格复用，不改旧outputs。

### Encoder缓存复用、训练与恢复

原约130GiB训练cache直接引用原位置，不复制、不重算；正式Head trainer仍校验每个
原训练tensor SHA。只对新英语validation调用冻结Qwen Encoder，保持原ln_post1024/
bf16、batch8/512samples分片。完成的英语缓存重跑先验证后复用，不再次加载Qwen。

组合新验证缓存时，从原验证缓存提取原ES/PT的BF16状态和标签，与新英语特征重组为
标准validation分片。只加载所需源分片，首次读取前验证tensor SHA，最多保留4个源
分片在CPU缓存中；不把所有训练特征放内存。smoke也只从现有特征复制小子集，不跑
Encoder。所有新缓存保留manifest/model/vocab身份，完成缓存重复运行校验后返回。

smoke为独立1epoch Head，不用于正式初始化。正式训练从fresh Head直接请求40epoch，
没有5epoch pilot步骤；minimum_epochs=5只是早停的原最小轮数。其余batch256、
LR3e-4、weight_decay1e-4、clip5、seed20260825、h512/k5/dropout0.1/time2、
all_samples_once、macro选点、patience6、scheduler patience2/factor0.5均沿用。
沿用现有trainer参数耦合：early-stopping-min-delta同时作为macro PER早停绝对改善值
和validation loss调度器threshold，本轮均为0.0005；没有改训练器或引入新调度算法。
新三语validation负责选点/调度/早停，旧英语不进入这些决策。早停后不会为凑40轮继续。

训练结束才用新best Head对旧validation cache做一次独立old_domain_control.json，
包含原英语及西葡S/D/I/PER等，不修改checkpoint。完整模型不会在此Head对照中加载。
PER、检索召回、最终ASR WER继续区分，本轮没有运行LLM转写或重跑Ziyan检索。

### 工区命令及小文件回传

先在同一挂载仓库拉取，实际远端SHA见本次最终回复：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录，原Qwen环境。准备仅CPU/清单元数据；run依次执行新增英语cache、
复用ES/PT/小smoke、正式40epoch上限训练、旧域对照，不再中途要求跑pilot：

```bash
python -B scripts/run_english_newval.py prepare
python -B scripts/run_english_newval.py run --gpu 4
```

4保留为显式物理GPU参数，可替换为用户已分配卡。run中断后：

```bash
python -B scripts/run_english_newval.py run --gpu 4 --resume
```

缓存自动复用已验证分片；Head恢复新实验的optimizer/scheduler/epoch状态，不跨验证集
续用旧Head。无Head状态时可从尚未开始的阶段继续；已有状态必须--resume。所有阶段
有stage/实际分片或样本数量/耗时/10秒活动提示。并发锁冲突检查PID，不删锁绕过。
支持显式cache/smoke/train分步诊断；没有pilot子命令。自定义--output-dir后每步保持一致。

prepare返回prepare_bundle.zip（report/plan/包内sha256.txt），包含真实可用/选中小时与
分国家来源摘要；成功后即可继续run。run返回return_bundle.zip（report/plan/cache/
smoke/正式训练report、完整metrics、旧域对照及包内sha256.txt），无需传权重、WAV或
特征张量。当前新验证容量/选中小时及H200训练结果均待回传，不从本地夹具外推。

### 本地验证范围

定向52 passed；全量783 passed（43.85s）。新增测试从真实fast.prepare产物开始，真实
构建新清单/特征分片、运行Head CLI、保存epoch/optimizer状态并恢复，验证正式请求
为40epoch且没有pilot。仅MFA/Encoder输入及CUDA硬件边界替换为合成特征/CPU；没有
本地加载完整Qwen。验证三国来源均覆盖、训练与新验证ID/路径不交叉、进入合格池的
test重叠音频仍被排除、ES/PT复用BF16数值/标签逐值相等、全部旧产物字节不改、
英语提取完成后的中断恢复不重复加载Encoder、源/新缓存篡改阻断、配置/清单漂移
阻断。额外覆盖Head已完成但完成标记尚未发布时中断：恢复只校验并发布结果，不在
早停后多训一轮；新JSON通过完整临时文件的独占硬链接原子发布，不替换已有文件。
修改范围Ruff/strict Mypy、CLI help和git diff --check通过；全仓仍为既有Ruff11项/
Mypy8项，未改未知草稿或被旧契约锁定的模块。本地只证明链路与身份/恢复逻辑，
不证明新语料的实际容量、PER改善或H200性能。

## 0.168 2026-10-15 新英语480h快速准备与GPU训练衔接（准备终端回传，H200待继续）

主线仍为仅替换三语480h中的英语，保留原西语/葡语480h、旧validation8101条、90类
音素及全部训练策略。分支codex/g2p-coverage-scan。新英语原始路径为
/glusterfs_103/z00816262/data/en/{智利,哥伦比亚,墨西哥}，JSON的旧104前缀映射至103；
文本取response中language English<asr_text>之后，与assistant一致。完整模型仅用户在
H200加载/glusterfs_103/models/Qwen3-ASR-1.7B；本地不下载/加载完整模型。

### 已完成资产及口径

旧outputs/english_countries_refresh_v1已处理453826条全量源记录和MFA标签；用户因
第6/7步完整WAV指纹读取太慢，明确选择跳过字节去重/字节隔离。新增独立fast入口复用
该完整标签，不重跑MFA、不覆盖旧目录。原候选WAV指纹未逐条持久化，不将内存中曾
扫描的291840条伪装成可复用缓存。

用户回传fast prepare终端JSON：status=data_prepared，新根为
outputs/english_countries_refresh_fast_v1。以下是终端摘要，完整return_bundle.zip尚未
独立返回核验，不宣称本轮Encoder cache或Head已完成：

| 语言 | 条数 | 小时 |
|---|---:|---:|
| 新英语 | 263609 | 480.001615 |
| 原西语 | 296399 | 480.001823 |
| 原葡语 | 321741 | 480.000408 |

合计881749条/1440.003846h。英语智利81456条/152.089802h、哥伦比亚95396条/
176.951241h、墨西哥86757条/150.960573h。英语original_ready263594条，原规则的
2x recovery15条；保留原释放阈值/分层维度/seed20260923，不因快版改变标签或选样规则。

fast按canonical绝对路径去重、隔离同路径标签冲突、排除固定ES/PT及全部保护路径；
不发现不同路径同内容录音，不声称字节/说话人隔离或独立录音480h。明确政策为
path_only_no_audio_byte_audit_user_requested，audio_byte_audit_performed=false。
原ES/PT清单按原SHA复制，validation保持原SHA；原training_config.json为构建中间件，
GPU入口实际使用training_config_path_only.json，后续run_plan.config继续携带该政策。

### 当前接口与只读核验

scripts/run_english_refresh_fast.py负责prepare/smoke/cache/pilot/formal。快版准备不会
生成frozen/audio_fingerprints.jsonl。GPU前置曾错误读取该严格版产物，根因是原GPU
单测夹具恰好包含该文件，未覆盖真实fast准备产物；不能用先前通过的局部测试声称已
验证整个衔接链路。当前改从原准备契约中声明且SHA锁定的两个实际来源读取保护路径：

- configuration.fixed_freeze/audio_fingerprints.jsonl：旧480h资产中已有的validation/test
  路径和角色，只读取JSONL元数据，不重算WAV字节指纹。
- source_preparation/source/test_audio_paths.jsonl：新三国测试音频路径，不读测试文本。

english_refresh_gpu核验两文件自身封存SHA及其在原preparation_contract中的精确SHA，
检查旧validation覆盖、整个新训练清单与保护路径不重叠，以及保护路径数量与fast
冻结报告一致。protection_provenance写入新run_plan；不在fast目录补造指纹文件。

GPU前置及显式prevalidated_audio_paths模式用字符串比较已规范路径，避免对约89万条
音频重复resolve/stat/readlink。通用加载器默认仍检查音频存在性，只有fast子命令显式
选择上述政策；所有标签、split、ID/路径重叠、计数、输入SHA检查保留。真实音频读取、
可解码性及实际CTC长度仍在Encoder提取时严格核验。用户最新终端已确认训练清单
881749条扫描22.3秒、validation8101条0.2秒；之后因上述错误路径退出，GPU尚未启动。
这只证明本次元数据扫描耗时，不能外推全量Encoder完成时间或质量。

stage_progress显示阶段、真实完成/总数、ASCII进度条和耗时；没有可靠总数时每10秒
显示活动和累计耗时，不编造ETA。覆盖输入SHA、清单/标签、保护路径、空间、Qwen
加载、两split Encoder提取、缓存SHA、验证分组和Head训练；保留原batch/shard/epoch
日志。活动提示只说明进程存在，不证明处理数量持续增长。

旧快版准备契约额外锁定自身源码。english_refresh_compat仅允许实际同一路径的
english_refresh_fast.py从已发布2615e3d文件SHA
5d69e4a99650ec42469c87c274f1975c3a1e18c8d2fd3b6bc0590a5620cb83cf，或6b85bfc文件SHA
59ac7fc038578cfe22481ae337eacb6d865a77a40b72b073cd6d763e0fe5f6b1，升级至已审查文件SHA
ed283d77e9c7443c99bd4e682af51283300f93c217d7691226656667c14821dc。7个准备函数AST
相同，改动限于运行入口。未知代码/数据/标签变化仍阻断；不重写旧contract、seal或
配置哈希，仅新增不可覆盖code_compatibility_<内容摘要>.json及成功摘要中的兼容记录。
本次保护路径修复不再改变该被锁定入口，既有兼容审计仍可幂等复用。

### 拉取、运行与恢复

在同一仓库拉取当前分支，SHA以本轮最终回复/实际远端为准：

```bash
# 容器外（也可在同一挂载仓库的容器内拉取）
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内使用原Qwen环境、已分配物理H200，4为显式GPU示例：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_english_refresh_fast.py smoke --gpu 4
python -B scripts/run_english_refresh_fast.py cache --gpu 4
python -B scripts/run_english_refresh_fast.py pilot --gpu 4
python -B scripts/run_english_refresh_fast.py formal --gpu 4
```

逐步成功才下一步，不自动连跑。本次失败发生在发布run_plan前，修复后直接重新smoke，
复用现有准备结果，不重跑prepare/MFA/选样，不删任何目录。首次新准备才执行
python -B scripts/run_english_refresh_fast.py prepare；默认新根
outputs/english_countries_refresh_fast_v1，拒绝已有目录。准备失败保留现场、另选新根，
不恢复部分选样；--prepared-dir可指向已有完整标签，--output-dir需在后续命令保持一致。

新plan完整核验后原子发布；已有完整plan按原config/input_sha256复用，不重写身份。
若training目录存在但缺plan，保留现场报告。smoke/cache按原分片SHA恢复，smoke Head
按原训练状态恢复；pilot中断显式加--resume，formal从已完成5epoch状态继续。
并发锁冲突先查持有PID及子进程；STAT=T表示暂停、仍持锁，不是已退出。确认是旧任务
才终止，不删除锁文件绕过保护。已处理过的12418为暂停的旧smoke，无子进程。

全量cache仍提取三语全部训练数据及8101条固定validation，不读取sealed test；本版
未优化复用旧ES/PT特征。bf16、ln_post1024、encoder batch8、512样本/shard；缓存前按
估计帧数×2048×1.25+10GiB核验空间。冻结Qwen、新temporal Head h512/k5/dropout0.1/
time2、batch256/LR3e-4、all_samples_once、训练seed20260825、macro PER选点等原参数
保持不变。smoke只1epoch检查流程，不将其Head用于正式初始化；pilot5epoch，formal
总上限30epoch，validation/early stopping策略见原480h配置。

### 验证与待返结果

新增真实产物串联回归：先用实际fast.prepare生成不含新指纹文件的资产，接入旧版本
源码身份，再调用实际cache/Head CLI。以合成Encoder特征和CPU替代Qwen/H200边界，
真实写入/读取特征分片并校验SHA，真实训练smoke1epoch、pilot5epoch、从第5epoch续训
至formal停止；验证完成缓存重用不再次提取、原产物逐字节不改、相对fixed_freeze解析、
旧validation/新test重叠和保护元数据篡改均阻断。仅MFA/Encoder输入及硬件为测试边界，
不伪造Head报告/缓存成功，也不将CPU回归称为H200质量或性能结果。本次定向44 passed。
该小夹具为train4/validation3条，formal报告实际从5续到6并按原策略早停，缓存SHA
核验为true、test_set_used=false；它证明衔接/恢复，不提供真实语料质量证据。
全量775 passed；修改范围Ruff/strict Mypy、CLI help及git diff --check已通过。
全仓仍仅既有Ruff11项/Mypy8项，未知PPT/文档/诊断草稿均未改、未提交。

核验git rev-parse HEAD与回复远端SHA一致；准备根目录内可运行
sha256sum -c sha256.txt核验旧元数据（不证明WAV字节）。返回根return_bundle.zip、
code_compatibility_*.json、training/run_plan.json、training/smoke_head/report.json及终端
摘要；cache完成返回training/cache/feature_cache_report.json；训练返回
training/head/report.json、metrics.jsonl。无需上传音频、特征张量或权重。当前工区仍待
本次修复后smoke、全量Encoder和Head实测；Ziyan诊断/控制检索仍是独立旁线。

## 0.167 2026-10-15 Ziyan删除37英语非目标词、仅保留0–4目标音频的配对检索（待工区）

用户恢复Ziyan检索分析并明确授权：删除讨论的13个疑似残片和24个非目标近邻词，
不再要求300–500个热词；仅跑每条参考包含4个及以下目标词的音频。用户另明确确认本次
只要Top5/Top7检索，不运行最终Qwen LLM转写。训练替换英语的0.166为独立工作，本次
不改训练任务或Head。当前分支codex/g2p-coverage-scan；本次起点HEAD、origin跟踪与
实际远端均为6e2d93b601fb1aa1fb80b9217b144e2aecf4a2d9。未知未跟踪PPT/文档/图片、
葡语配置及两份Ziyan诊断草稿保持原样，不导入、覆盖、提交；tracked起点干净。

### 已确认的原始分布与新口径

用户回传资产report的targets_per_audio，独立复算712条/1100个不同目标词×音频，
413正样本/299无目标样本。超过3目标为93条，占全部13.0618%、正样本22.5182%，
贡献591/1100=53.7273%目标对；这说明微平均召回受高密度音频影响，不证明音文配错。
0目标只指固定372词与参考的交集为空，不表示没有语音/没有全表词，也不直接降低目标召回。

新实验按原冻结372目标、生产strict_phrase_match计不同词，一词重复多次仍计1；
保留0/1/2/3/4目标的299/175/101/44/35条，共654条，其中355正样本、299无目标，
分母649目标对。排除58条/451目标对。Top5/Top7对新目标的理论容量界均100%，不能
据此预言实际召回。筛选使用参考、不使用预测，是参考条件化子集，不是泛化收益。
不去重、不截断音频；原6条额外hash重复仍由源记录保留，新报告重算选中重复数。
372是已有冻结资产，不为本次重新凑词数；新控制工具没有300–500限制。仅在被排除
音频出现的目标仍留在索引，另列active_target_words_in_subset和缺席目标，不静默删目标。

新增configs/ziyan_control_le4_drop37.workzone.json冻结两组词：

- 13残片：ately/connec/contem/cutors/immigra/interna/kets/larly/mation/opportuni/
  sions/sition/sonal。疑似残片仍是审查判断，未证明其原始产生原因。
- 24近邻：warnings/warns/balanced/balances/companies/guarantees/delays/designs/
  highways/perfumes/stations/requested/enjoyed/rained/weathered/fished/contracting/
  advancement/nationally/nationals/booker/prides/thrived/thriving。

这37词均不在372目标中。只删除这些词的英语filler变体；若同词面有ES/PT变体，保留
其他语言，并明确报告仍活跃的跨语言词面。因此完整三语词面数不预先写死为3606-37。
保留的词面/发音不修改，不跑MFA，不改90类音素、Anchor、评分、门控、Top64或Top5/7。

### CPU真实索引重放与结果解释

新增scripts/run_ziyan_control.py及evaluation/ziyan_control{,_replay}.py。复用已完成
outputs/ziyan_english_assets_v1和outputs/ziyan_english_retrieval_480h_v1。源表固定SHA
09244f35b64964313a26fc8a6cfa51b321e17ce02b730f168d1c4c3d6fabc530；Head仍为320123fe…4265，
vocab仍0f4939ba…af4，完整原身份继续由configs/ziyan_english.workzone.json约束。

先调用既有ziyan_identity核验源资产/检索全部SHA、实际JSON/targets/Head/vocab、原代码
和712条详情。读取已有CTC IDs/confidences，对每条选中音频重新构建原表索引查询Top64，
调用原评分/词面合并/门控，复现全部变体候选、词面候选、分数、编辑距离、posterior、
decoded_start/end和Top5选取，除不能恢复的声学帧时间外逐项核对（浮点容差1e-10）。
任一不一致即停止，不发布实验结果。随后对真正删除英语条目的新表重新建Anchor索引，
重算锚点权重/Top64候选/重排/门控，产出新Top5/Top7，绝不是从旧交付或Top64列表删词。

原文件未存逐音素声学帧边界；CPU内部使用离散序号辅助原函数，导出所有start_step/
end_step为null，仅保留真实decoded-token位置，绝不把序号伪装成WAV时间。无需GPU、
Qwen Encoder或Head前向，未打开源音频字节、不加载完整模型；音频hash复用已存记录。
这是一致CTC下的检索配对实验，不提供PER/WER、新H200时延或音文/SNR审核结论。

报告分开给原712条旧指标、同654条旧表指标、新表同654条指标。只有后两项是删词配对；
对照同649分母，另存逐条gained/lost目标、参考外输出和候选数量，避免掩盖一增一减。
by_target_count逐层报告0–4；无目标样本分别记录错误目标触发率与任意输出率，后者可能
含参考中真实出现的非目标词，不能全部叫误报。新表词面数和发音索引规模独立报告。

### 同步、执行、恢复与回传

容器外拉取当前分支（本轮新远端SHA见回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内仓库根目录，现有Python环境执行，纯CPU，无需aligner或GPU参数：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_ziyan_control.py
```

默认新目录outputs/ziyan_control_le4_drop37_v1，拒绝已有目录及旧资产/检索目录内或祖先
输出。可用--output-dir指定另一新目录；固定--config/--root。四阶段进度及每条完成数、
真实百分比、累计时间和sample ID可见。CPU耗时依CTC长度与机器而变，未给虚构ETA。

完全相同配置、源身份和代码中断后：

```bash
python -B scripts/run_ziyan_control.py --resume
```

逐条配对JSON shard有SHA与原行SHA，完成配对才原子保存；未完成音频重放原/新两个索引，
已核验配对复用。代码/配置/源身份漂移拒绝恢复。完成后--resume仅验证、返回原报告。
源身份小包写入中断或shard损坏不删除现场，需使用另一新输出目录。已有生成词表不同
则拒绝，避免覆盖用户修改；原outputs始终只读。部分最终打包可恢复；zip临时文件不
当成完成包，残留尝试保留。归档与报告完成后封存整个新输出。

新产物包括keyword_table.json、english_words.txt、removed_english_words.txt、targets.txt、
subset_manifest.jsonl（654条逐条参考/目标数）、excluded_manifest.jsonl、原JSON选中行、
sample_shards、逐样本详情、paired_changes.jsonl、selection/removal/report及SHA清单。
delivery两份名称仍为ziyan_en_noise10db_top5/top7.json；仅654个stem，格式仍word/phoneme。
屏幕打印return_bundle.zip及SHA；只需返回该小包和终端JSON，不传音频、权重或特征。
包内含新两份交付、英语词清单、移除清单、冻结目标、配对变化与各小报告，便于核验。
另含新run_config的实际输入/代码SHA及源identity summary；源音频字节未重新打开。

完成后在工区核验新输出：

```bash
R=outputs/ziyan_control_le4_drop37_v1
(cd "$R" && sha256sum --quiet -c sha256.txt)
```

本地验证为合成CTC/模拟文件及单元测试，不是工区实测；当前尚无这版召回率或删词收益。
定向32 passed，全量705 passed；新范围Ruff、strict Mypy、CLI help与git diff --check通过。
全仓Ruff仍有原11项E501，Mypy仍有原8项（历史训练3项、未知诊断草稿5项），未改无关
文件。测试覆盖不到300目标的冻结资产、完整原表重放、变体/候选/门控一致、原Top64外
词可在删词重建后进入候选、ES/PT同形词保留、0/4/5边界、重复词去重计数、原文件不改、
无实际音频仍可重放、损坏输出/配置漂移拒绝、逐条中断与最终打包中断的恢复，以及无
伪造声学时间的导出。新代码未修改原检索/训练模块。

## 0.166 2026-10-14 新三国英语全量处理→480h替换→原三语CTC训练入口（待工区）

用户明确暂停0.164–0.165英语检索根因诊断，授权换英语训练语料后重新训练三语CTC；
西语/葡语各480h不变。此前“不重训”限制由本次新请求更新，检索诊断、旧听审和prompt
旁线继续暂停。本轮接续0.13英语G2P/精确连接符、0.125已冻结三语清单、0.126及
0.138–0.141缓存与训练方案，不重建旧ES/PT/validation，不从旧Head续训新数据。
起点本地HEAD/origin跟踪/实际远端一致为448c9d174a9d7d6bda2230c7f6a828004a24134b；
tracked无修改。保留两份未知Ziyan诊断代码及新增未跟踪docs/MFA_POSTPROCESSING_V0_2.md、
PPT/图片/文档/葡语配置，不导入、覆盖或误提交它们。没有本地加载/下载完整模型。
暂停前0.165身份小包已分两包返回，合并11项SHA全部匹配，目标分母/容量独立重算一致；
399漏项全量分解及听审仍未完成，本次不继续展开该分析。

### 固定输入及处理政策

新增configs/english_countries_refresh.workzone.json、scripts/run_english_refresh.py及
training/english_refresh{,_source,_freeze}.py。三国来源均为
/glusterfs_103/z00816262/data/en下的智利、哥伦比亚、墨西哥目录，每目录train.json、
test.json、audios。逐国将JSON内/glusterfs_104/asr-data_imedia_zh-en/en/ASR_shards_data/
English/<国家>前缀替换成实际目录，其余audios/shards_*/WAV后缀保留；也接受已是新
前缀的路径，拒绝未知前缀/跨国/目录越界。只取response的English<asr_text>后正文，
与可用assistant内容核对；流式读取原JSON，每条train保留来源/旧新路径/原response，
不拿test响应造词表或标签，不猜speaker，不把国家文件夹当作已验证口音。

先处理三个train.json的全部行：统一TSV与全量训练词表，新跑既有English US MFA zip
Top1（固定9923b38d…cb02，不下载），再调用原full_manifest，语言en-US是沿用的标签
管线标记，来源保留三国。使用原90类v0.2词表SHA0f4939ba…af4及原IPA后处理；允许
词典唯一精确、无OOV的撇号/连字符。数字、缺词、歧义、OOV、音频问题等进入完整
needs_review，不丢行、不用旧英语词典补标签。词典输入/输出缺失、额外键与歧义另报。
严格分区source_records=ready+review，只在完整分区结束后开始选样。

沿用original_ready及仅ctc_length_infeasible的temporal 2x recovery：有效长度满足
minimum/(2*T_est)<=0.90。其余review不释放。对完整释放池做音频字节hash：同内容同标签
只保留一个，内容相同但文本/标签/元数据冲突整组隔离；排除三国test、旧全部heldout
和固定ES/PT训练音频的路径或hash重合。原文件不删除，排除原因保存在新产物。
旧heldout由0.125已封存audio_fingerprints的roles重建，不重新打开旧test标签。
新test仅解析路径并读取字节用于保护；最终所选三语与保护集合再次做真实字节审计。
SHA不识别重编码/重叠片段，当前没有speaker metadata，不能声称speaker-disjoint。

合格唯一英语按国家来源×release×时长×标签密度×CTC ratio分层，按可用小时比例分配
配额，seed20260923沿用0.118选样策略。总计至少480h，保留完整音频略超；不足停止并
保留完整处理产物/capacity_report，不重复抽样凑时长。原英US训练全部替换。

固定ES/PT来自outputs/multilingual_480h_training_v2_a48f58e452，分别296399/321741条，
SHA3f3abb4c…d0777 / 987e6508…0d25a，完整值见新配置。新的分语言ES/PT文件须与原字节
一致；combined仅更新版本元数据，原音频、ID、文本、音素、选样不变。固定旧8101条
validation及SHA d43d143f…4499全部沿用，包括原英语2808条，以支持旧视图比较；新
英语域上的验证能力需另行评估，本次不使用新test挑checkpoint。

### 新目录及短命令

默认新根outputs/english_countries_refresh_v1，包括source、words、MFA尝试目录、
全量manifest、frozen_<后缀>、新training配置和training运行目录。旧outputs不覆盖。
prepare尚未执行，不能声称新英语可用小时或新480h已完成。固定ES/PT/validation身份
及原freeze完成状态在新处理前核对；代码/配置/六个JSON身份写入preparation_contract。

容器外同步（本轮最终推送SHA见回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内CPU/MFA完整准备，无GPU，不复制音频或提取Encoder特征：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda run --no-capture-output -n aligner python -B scripts/run_english_refresh.py prepare
```

成功status=data_prepared。返回屏幕JSON及打印的return_bundle.zip；包含原源计数、
完整ready/review摘要、MFA缺项、可用/选中小时与分层、去重/保护、固定资产及新配置SHA，
不含全量清单、缓存、音频或权重。失败返回终端错误；容量失败另返回新frozen目录下的
capacity_report.json，最终审计失败返回audio_identity_audit.json。不能使用失败清单训练。

首次拒绝已有根目录；完全一致的配置/JSON/旧资产/代码可显式追加--resume。完成的
source/words/manifest/freeze必须通过SHA清单；full_manifest未完成shards按原机制恢复。
MFA失败新建attempt目录，保留失败输出；freeze失败新建frozen尝试目录，不覆写现场。
若source/words转换中断、无完成清单，保留原根并用--output-dir指定另一新根；所有后续
阶段须传相同--output-dir。不支持配置/字典漂移的原目录恢复，也不删除目录自动重跑。

prepare成功后切回现有Qwen环境；4仅是空闲物理H200示例，用户选择编号：

```bash
python -B scripts/run_english_refresh.py smoke --gpu 4
python -B scripts/run_english_refresh.py cache --gpu 4
python -B scripts/run_english_refresh.py pilot --gpu 4
python -B scripts/run_english_refresh.py formal --gpu 4
```

逐步成功才执行下一步。复用原multilingual_480_run和原cache/train_full_ctc入口，训练
冻结Qwen3-ASR-1.7B的ln_post特征上随机新temporal_upsample Head（h512/k5/dropout0.1/
time2），不使用旧480h或smoke Head初始化正式Head。batch256/LR3e-4、weight_decay1e-4、
clip5、seed20260825、all_samples_once均不变；三语Macro PER选checkpoint，原worst-group/
loss tie-break、min_delta0.001/patience6/minimum5epochs、loss scheduler不变。
smoke1epoch机械检查；全量cache为新三语清单+固定validation，bf16/encoder batch8/
512samples每shard，建议H200至少40GiB空闲，存储门槛由新帧数估计，不套旧140GB为实测。
pilot完整数据5epoch；formal恢复Head/optimizer/scheduler到总上限30，不另加30。
pilot中断明确追加--resume；smoke/cache及formal按原身份和分片机制恢复，未完成epoch
会重跑。早停后不为凑30轮反复formal。训练只读cache，test不参与，PER不等于检索召回/WER。

smoke/cache返回终端JSON；pilot/formal返回新training/head/report.json及metrics.jsonl，
另返回ctc_head_best.pt的SHA即可，不上传权重。新的三语训练、缓存与效果均尚未在H200
发生；本轮仅交付代码与本地合成验证。检索诊断继续暂停，后续评测另立受控对照。

### 2026-10-15 长任务终端进度补充

用户要求后续可能长时间运行的任务都提供可见进度，已写入AGENTS Coding Guidelines：
真实计数、阶段、时间和日志/活动提示；底层没有可靠进度时不编百分比或ETA。
当前prepare的MFA stdout/stderr写入mfa_attempt_*/mfa.log，原命令终端可能长时间安静。
项目仅有葡语76744词/1702.057秒（约28.4分钟）的历史实测，不能当新英语耗时；新英语
唯一词数/吞吐尚未回传，MFA成本按唯一词而非480h音频小时估计，完整prepare还含音频字节扫描。

新增scripts/watch_english_refresh.py及training/english_refresh_progress.py。保留9811190
原prepare/训练入口及其契约哈希覆盖的代码字节不变，避免为显示进度破坏已有任务身份。
此补充在0.166记档，不改变数据、MFA参数、筛选、缓存或训练；不新增训练实验节点。

容器外按上面的分支pull同步后，容器内另开终端观察已启动任务：

```bash
python -B scripts/watch_english_refresh.py
```

默认只读，不启动、不恢复、不覆盖任何任务或产物；Ctrl-C仅退出监看。每5秒显示7个
阶段编号、实际计数/条数进度条、阶段已观察时间和总监看时间，并跟随MFA原始日志。
附加监看的时间从打开监看起算，不能当成原任务已经运行多久；阶段编号不是总耗时百分比。
MFA若没提供真实总百分比，只显示活动指示、词数和日志，不把内部子步骤100%当全程完成。
只读文件变化不能单独证明原进程仍在运行，以原任务终端退出码/JSON和原报告为准。

新任务可把启动与进度合到一条命令（已启动任务不要重复--run）：

```bash
conda run --no-capture-output -n aligner python -B scripts/watch_english_refresh.py --run
```

--run将同一配置/输出/worker/jobs参数交给原scripts/run_english_refresh.py prepare，
保持原返回码；原任务报错不会被监看当成功。已验证配置相同的中断恢复使用--run --resume，
原0.166恢复限制仍然适用。支持--root/--config/--output-dir/--workers/--jobs/--interval。
监看不要求MFA/Qwen环境；只有--run需要aligner。GPU cache/train原有shard/epoch日志继续保留。
本次全量回归692 passed，最终监看定向7 passed（含补充的只读/返回码CLI验证）；新范围
Ruff/Mypy、CLI help、git diff --check通过。全仓既有Ruff11/Mypy8项不变，未改未知文件。

本地新流程8项测试及相关模块定向38 passed，全量687 passed；新增范围Ruff和strict
Mypy（scoped --follow-imports=silent）通过，CLI help、git diff --check通过。
全仓Ruff仍有11个既有/未知文件E501，Mypy仍有8项（既有训练unused-ignore 3项、
未知ziyan_diagnostics 5项），未改无关文件。合成测试包括全量处理后选样、固定ES/PT/
validation及优化参数不变、review保存、字节重复/冲突、holdout隔离、不足阻断、
temporal recovery、MFA失败新尝试及恢复、GPU阶段委托原流程及配置漂移拒绝。

## 0.165 2026-10-14 Ziyan产物身份与实际字段核验入口（待工区回传）

本阶段承接0.164，先核验现有产物，不重新推理或改变检索。完整读取AGENTS及
0.159–0.164、Ziyan入口/构表/检索与旧取证重放实现。开始时HEAD、origin跟踪分支和
git ls-remote真实远端均为f6d956bcf387bf10119a7518078d9cb9fb9401c8，tracked无修改。
本地没有outputs目录，不能把终端报告当成已独立核验的工区文件。

发现未跟踪scripts/diagnose_ziyan_english.py及evaluation/ziyan_diagnostics.py，来源
未确认；本轮只读检查，不覆盖、导入或提交这两份文件。其余PPT/图片/文档/葡语配置
也保持原样。新工具使用独立文件scripts/inspect_ziyan_artifacts.py及
evaluation/ziyan_identity.py，只依赖标准库，避免混用未知诊断草稿。

工具核验资产与检索各自SHA清单全部条目；检查已完成状态、固定配置、JSON/冻结目标/
vocab/Head实际SHA与原运行记录、资产/检索两份表字节、原检索代码SHA。逐行检查712条
详情的行SHA、保存音频SHA、原JSON参考和manifest/run_config记录一致；回传实际字段
出现次数与首条候选字段，列出记录hash重复组及原始参考/保存CTC一致性。
不打开原音频、Qwen权重或运行MFA/Anchor/评分，不独立重算召回/PER/WER。
summary中的reported_metrics_not_recomputed明确是原report值；identity_verified
仅表示这次身份检查通过，不表示已定位399对漏召回原因或验证音文/SNR质量。
原JSON/Head/vocab实际字节验证与音频hash记录复用须分开解释。

容器外同步；本轮最终推送SHA见回复，不沿用上述起点SHA：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，在项目现有Python环境执行一条CPU命令，无GPU参数：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/inspect_ziyan_artifacts.py
```

默认新建outputs/ziyan_identity_v1_<随机后缀>，拒绝已存在目录或源资产/检索目录内部及
其祖先目录。支持--root/--config/--assets-dir/--source-run/--output-dir。没有resume；
中断重跑会生成新目录，旧outputs不覆盖、不删除。打印return_file、bundle_sha256及
bundle_bytes；用户只需返回该return_bundle.zip和屏幕JSON。
包内summary.json、首条first_saved_row.json、首个非空候选行candidate_schema_row.json、
两份原report、run_config、配置、目标和
源SHA清单用于独立核验；不包含全部详情、样本分片、音频、模型或Head。
样例行仅用于确认字段，不作为根因代表案例；全量无非空候选时仍保留首条并报告空字段。

本地定向21 passed，全量679 passed（本地模拟/单元测试，不是工区结果）；新增代码及
修改测试Ruff通过，独立身份模块Mypy通过，git diff --check通过。全量Ruff有11个
既有/未跟踪文件E501：历史扫描/PPT文件5个，两份未知诊断代码6个；全量mypy src
有8项问题：既有训练unused-ignore 3项，未知ziyan_diagnostics 5项，未改这些文件。
使用全局Mypy执行仓库Python3.10配置；.conda内同配置检查被现装NumPy的Python3.12
类型stub阻断，切到3.12又引入依赖类型问题，不将不同检查环境当作历史结果复现。
测试覆盖源文件不改、无原音频/模型仍可核验、精简包、报告/Head/代码/参考篡改拒绝、
已有/源内输出及清单越界拒绝。尚未在工区运行该工具。

如需工区核对新小包（R替换成打印目录）：

```bash
(cd "$R" && sha256sum --quiet -c sha256.txt)
```

收到小包后先验包及原件身份，再交付399对漏项的候选/门控组合/Top7分解与必要CPU
Anchor重放、ants等误例及真实正确对照。此轮尚无新集分解或听审结果，未排除28词、
调门控、改90音素、重训或扩跑LLM提示实验；保持0.164所有待完成状态及解释限制。

## 0.164 2026-10-14 近期实测结果补记与英语检索诊断交接（当前状态）

本节补记0.155、0.157–0.163之后已返回但未独立落档的重要结果；这些历史章节的
“待工区”描述是当时的代码交付状态，当前进度以本节为准。只更新阶段成果、限制与
下一步，不把兼容修复逐条立为节点，不改模型、词表、检索算法或任何已有outputs。

### 仓库与证据等级

本轮完整读取AGENTS.md，检查status、工作区/暂存diff、未跟踪文件及真实远端。
开始时本地HEAD和git ls-remote查到的origin/codex/g2p-coverage-scan一致：
b779a8e448cd3ba6b6ff920999100d9bb4b4d72e；tracked无修改。既有PPT、图片、文档和
configs/pt_quality_whisper_assets.json等未跟踪用户文件保留，不修改或误提交。
本节和交接prompt是文档交付；最终推送SHA通过git/本轮回复核验，不将该起点SHA当作
更新后的提交。完整模型只在H200由用户运行，本地没有下载或推理。

证据必须区分：

- 本地实际附件字节及包内SHA可核验：提示实验1.zip、英语失败案例return_bundle.zip、
  MFA审计return_bundle(1).zip + comparisons.zip、Ziyan原始JSON。
- 用户终端返回：联合表检索/扩表全语种聚合、候选竞争诊断、Ziyan audit/run及交付词计数。
  本地可核对算术和代码定义，但未收到这些工区完整产物，不声称逐文件重新验证了H200。
- 模型发音正确性、音文一致性、真实噪声SNR、失败片段的实际说法，需要音频或额外证据；
  MFA一致、词表OOV为0、命令completed都不能独立认证这些质量。

### 1. 英语最终ASR排序/优先级实验：运行完成，探索结论完成

对应0.155。复用三套干净英语en_chile/en_colombia/en_mexico，每套20条，固定分层
10条target_hit、5条target_miss、5条no_target，无缺额/补选；seed=20261012。
使用原英语3,000词表定义目标及旧Top7交付，只注入候选，不注入整表。
outputs/english_prompt_order_v1完成60×8=480主请求及3次B重复请求；没有重跑CTC。
实际Qwen离线完整音频、English、bf16/eager、贪心；GPU为物理3；返回runtime中
Torch=2.9.1、CUDA=12.8、qwen-asr=0.0.6、transformers=4.57.6，不能用7月环境替代。

| 组 | 条件 | 最终目标命中/85 | 正确注入采用 | 参考外注入词落字 | WER错误/495 |
|---|---|---:|---:|---:|---:|
| A | 无提示 | 81（95.2941%） | 无分母 | 无分母 | 24（4.8485%） |
| B | 检索原序 | 81 | 52/53 | 2/200 | 25（5.0505%） |
| C | 同候选倒序 | 81 | 52/53 | 2/200 | 24（4.8485%） |
| D | 同候选固定随机序 | 81 | 52/53 | 2/200 | 25（5.0505%） |
| E | D同序，全部normal | 79（92.9412%） | 51/53 | 2/200 | 26（5.2525%） |
| F | E同序，仅B首词high | 79 | 51/53 | 2/200 | 26（5.2525%） |
| G | 正确声明目标Oracle首次位置序 | 81 | 81/85 | 无分母 | 23（4.6465%） |
| H | G同候选倒序 | 81 | 81/85 | 无分母 | 23（4.6465%） |

配对结论（使用规范化转写，而非标点/大小写差异）：

- B/C有效54条，有1条转写变化，目标新增/丢失均0，WER少1错误；B/D有效46条，0变化。
  排序在此批音频上偶有影响，没有观察到目标召回收益；不能说排序绝对无影响。
- E/F有57条prompt实际变化，规范转写0变化；原始文本有2条变化。被提升正确词14/14
  在两侧均出现，参考外被提升词0/43在两侧均出现；未观察到优先级提升的采用或误落字收益。
- D/E有2条规范转写变化、目标净丢2个、WER净增加1错误。它改变说明与列表格式，
  不等同于“只修改一个优先级标签”；不能将E/F比B差归因于high标签。
- G/H仅20条含至少2个正确候选，0变化；其他40条不提供排序证据。A/G目标仍81/85，
  仅少1个普通WER错误；Oracle没有在这60条中恢复额外目标，不是线上能力或数学上限。
- 3次B重复请求均无原始/规范转写变化，只是有限重复性检查。
- B/C/D/E/F落字2/200不能直接叫prompt诱发幻觉：配对与A相比未新增这些错误候选，
  也没有音频听审。原目标含A/THE等基础词，实验范围仍是历史声明目标，未追溯更换分母。

本轮重读1.zip，包SHA为c3a2bbce92bff44f5d37993699055c8d3124e7e1880099a4cbdd50b16e4e06a5；
report.json和representative_examples.md均匹配包内清单。包只含这两个结果文件及清单，
清单中其余492个文件未随包返回，不能宣称验证了完整freeze/483条结果原件。
本小样本探索已完成；推广到6,000条、统计效应或产品优化尚未验证，不自动扩跑。

### 2. 人工审核三语联合表及4k扩表：检索完成，扩表竞争诊断完成

对应0.157–0.159。固定干净英语3×2,000 + ES四wave×100 + PT四wave×100，共6,800条。
三语共用整张表，不按音频语言筛表；下游分语言目录，共22份Top5/Top7交付。
审核后表：1,358唯一词面、1,370发音条目，en759/es316/pt295；用户仅删除完整条目，
原标签不改，无fillers。表SHA=89a63763216e9564edffa50014699e8c05f1257fb436ff8bcddc7e8fa720d9a7。
之后增加2,642填充词（EN881/ES881/PT880），成4,000词面；新增词排除全6,800条
参考中完整token、基础/无关短词、低频/短音素/OOV等，使用已存在训练MFA词典精确发音。
不是所有目标重新生成G2P，也不保证没有自然近音或子词声学匹配。
扩表SHA=6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b。
两阶段保持480h Head、原评分/门控和目标分母；不得拿更早包含基础词的旧全目标指标比较。

| 语言 | 同目标分母 | 审核表Top5 | 4k Top5 | 审核表Top7 | 4k Top7 | Top7参考外候选 审核→4k |
|---|---:|---:|---:|---:|---:|---:|
| EN | 4,374 | 3181（72.7252%） | 3175（72.5880%） | 3216（73.5254%） | 3215（73.5025%） | 6634→8468 |
| ES | 724 | 679（93.7845%） | 649（89.6409%） | 685（94.6133%） | 664（91.7127%） | 514→1760 |
| PT | 736 | 595（80.8424%） | 561（76.2228%） | 606（82.3370%） | 594（80.7065%） | 605→1608 |

工区outputs/joint_4000_diagnostics_v1_a7165cd759返回：source_metrics_reproduced=true，
两轮EN6000/ES400/PT400条CTC token和置信度序列完全一致。真实配对最终目标：
EN丢1/新增0，ES丢21/新增0，PT丢12/新增0；从已保存候选移除填充词后分别恢复1/15/11。
这仅回答保存候选内的槽位竞争，不恢复被Top64裁剪的目标。EN新增参考外候选1,901个，
其中filler::en1718/es155/pt28；原审核英语目标自身仍贡献6,519个参考外输出。
英语扩表最终召回基本不变，误匹配更多；“新fillers解释英语全部低召回”不成立。
ES/PT的扩表下降有候选竞争证据，但并不证明其余全部错误都是竞争或多语干扰。
这些全语种数字来自用户终端回传，本地未收到诊断目录全部原件。

### 3. 原英语漏召回取证：18例已核验，音频听审未完成

对应0.160，返回包/Users/qsc/Downloads/return_bundle.zip，SHA为
5e1536ae118c5b8a1822292e2bbbf6f4fddb6043de7eb063da8e2fef611c00f9。
本轮重验包内summary.json、examples.json、examples.md、target_terms.txt四文件SHA，
共18例、18条不同音频，原/新指标复现。旧6000条英语4k目标漏召回1,159对中：
666未进保存候选、485门控失败、8合格但低于Top7；成功3,215/4,374。
这是旧干净英语数据，绝不能直接套到后面的Ziyan712条。

抽到的6个未进候选案例：5个无Anchor且无exact命中，1个Gentleman在截断前排614
超过Top64。Night、Home、Good出现表标签与CTC相邻片段不一致；即使全起点诊断搜索
找到近似段，也可能继续score失败。门控案例包括Nature0.735216、Lucy0.738020、
Dance0.742003、Court0.711700、Great0.701521，均低于0.75；Magic一处音素替换后仍
0.762517成功，说明不是“所有有音素差异的词都失败”。Answer/Concert/Press同时失败
score和edit。score=(1-edit_ratio)*(0.75+0.25*posterior)，例如edit_ratio=0.25时
posterior<1就使score<0.75；这是规则的数学效果，不是posterior表示词级概率。

样本是分层探索，不从5/6推断全量无Anchor比例。不把全起点最佳片段当真实词边界，
不把token下标/旧内部step当音频秒数。没有听审，尚不能把差异归为说话人口音、标签错
或CTC模型错；也没有证据证明90类输出表先天无法表达英语。

听审候选（均在原干净目录/host_home/star/两波验证集-可直接推理/<dataset>/en/wav/）：

| dataset | WAV stem | 目标/用途 |
|---|---|---|
| en_chile | 200638_quite_589244 | Night，无Anchor |
| en_colombia | 170013_quite_034920 | Home，无Anchor |
| en_colombia | 170337_quite_177793 | Good，无Anchor |
| en_colombia | 170637_quite_221463 | Dance，门控失败 |
| en_mexico | 210625_quite_183691 | Court，门控失败 |
| en_mexico | 210659_quite_652449 | Great，成功对照 |

用户已提出要听这些音频；目前本地未收到对应音频、没有听审结论。下一聊天如继续旧集
需复用这批固定案例，不重新挑更容易解释的样本。

### 4. 原759个英语目标MFA一致性：审计完成，有13项未解决

对应0.161，工区outputs/english_target_mfa_audit_v1_077982ff91，
status=completed_with_unresolved。检查目标词，不包含fillers；759条目标、753个独立
MFA输入词，English US官方本地zip Top1，按训练分词及90类规范化比较。

| 比较 | CTC完全一致 | CTC不同 | 未解决 |
|---|---:|---:|---:|
| 当前表 vs 新MFA | 746 | 0 | 13 |
| 当前表 vs 英语训练词典 | 696 | 0 | 63 |
| 新MFA vs 英语训练词典 | 696 | 0 | 63 |

已解析746项原始字符串不同主要为外层/标记，CTC标签一致，不是746个错词。
13项带非ASCII/外语词项存在输出key与输入不对应等问题，需单独语言/词形审计，
不把它们算作“已确认错误发音”，不去重音或强行套英语。Lie等令人疑惑的标签可被
新MFA/训练字典同样复现：证明来源一致，不能证明音频实际发音正确。
目前没有实测支持“批量重新跑MFA即可修复旧英语召回”；未修改生产标签或重新推理。

本轮重读两个分包：return_bundle(1).zip SHA
1db430f4b60f4b6bf50b31fe11b5e3e8846aacfada7641e3c13053b077035366；comparisons.zip SHA
5819f7b7ef75a33dc3705f6f6f57b5cb61635d0c8c9fe34119d47f5085aa284d。
首包7项文件SHA通过，缺的comparisons.json在第二包，SHA
5e8aaac3fcf16b64836a965380c5c744660e46b6548ca2a8c2be142cf11174bb与第一包清单一致。
两包组合补齐8项清单文件。官方Top1一致性不替代发音变体或音频核验。

三语4k重新G2P查看文件（0.162）代码已交付并修正扩表来源身份字段；用户首次被
“completed sealed 4000-table source identity differs”阻断。当前尚无成功dict/report
回传；代码已完成，工区后续完成情况未知，成功回传核验未完成。不要跳过SHA、覆盖旧表，
或将此查看文件自动替换生产表。

### 5. Ziyan10dB英语新目标与检索：工区完成，根因诊断进行中

对应0.163，原JSON与冻结372目标的完整身份和筛选规则见该节。用户指定“只替换原759
英语目标，保留原fillers”，因此没有针对新JSON重新排除旧填充词。新目标全部需新
English US MFA Top1生成并通过vocab检查才发布；保留原ES/PT及旧fillers发音，不补4k。

- 原JSON：/data/c00809914/audio/ms-swift-qwen3-asr/test/en-ziyan_2-5h_noise_10db_test.json。
- 资产：outputs/ziyan_english_assets_v1；运行：outputs/ziyan_english_retrieval_480h_v1。
- 工区audit_pass：712条、3,606词面，实际0.8267253646小时（约49.60分钟），并非名称
  暗示的2–5小时；duplicate_audio_file_hashes=6表示比唯一hash多6条，不必然是6组。
  没有自动删除这些记录；重复组的参考一致性/敏感性分析未完成。
- 372目标从场景内容词选取，282词出现于至少2条参考、90词单次；不按检索结果选词。
  1100目标词×音频，413条有新目标，299条无新目标（不等于无整表词）。
- 41条超过5个目标、22条超过7个目标；容量界Top5=939/1100（85.3636%），
  Top7=1007/1100（91.5455%）。这是固定槽位理想界，不是模型表现。
- 物理GPU1运行完成，交付2份ziyan_en_noise10db_top5/top7.json，原stem→word/phoneme格式。

| 指标 | Top5 | Top7 |
|---|---:|---:|
| 目标命中/1100 | 639（58.0909%） | 701（63.7273%） |
| 门控前TopK目标命中/1100 | 680（61.8182%） | 762（69.2727%） |
| 最终输出候选总数 | 1236 | 1416 |
| 参考外候选数 | 578 | 696 |
| 全表候选参考匹配比例 | 658/1236（53.2362%） | 720/1416（50.8475%） |

Top7比Top5多62目标命中（+5.6364pp）及118参考外输出。门控前后Top7净差61不能
直接等同于“恰好61个目标被门控拒绝”：门控可能从更低排名回填，需配对状态检查。
Top7仍漏399目标对；即使完美排序，槽位容量也至少导致93个漏项，距容量界仍少306命中；
不能把这306直接全归于模型/门控/噪声。不同域、目标集合、噪声、表组成与分母同时改变，
不直接和旧干净英语77.37%判定退化；这里只测检索，不测PER或最终ASR WER。
工区report/sha256尚未作为文件返回，本节指标来源是用户终端audit/run JSON及后续jq。

### 6. Ziyan旧填充词与新参考重合：部分取证完成，影响尚未闭环

用户查询得到原2642填充词中28个非新目标词出现在新测试参考某处：
adjustments、ants、canceled、candles、cheque、conditions、congratulations、contain、
curry、deduction、delivered、drones、eleven、emerald、equals、frozen、goats、guarantees、
including、operating、pagoda、parks、polar、rained、ships、stamps、tastes、traditional。
“不在旧6800条参考”不推出“不在新712条参考”；保留fillers不代表重新完成新参考排除。

交付计数：Top5这28词输出42次，涉及39音频、18种词；Top7输出47次，涉及43音频、
19种词，占1416候选3.3192%，涉及712音频6.0393%。统计单位是词×音频，不是47种词。
该jq仅检查词属于全测试集重合集，没有匹配该条音频自己的参考；不能把它们全算正确。
本轮按生产strict_phrase_match重新读取SHA一致的原JSON，得到：

| 词 | 含词参考音频数 | Top7交付次数（用户返回） | 仅依计数可证明的参考外次数下界 |
|---|---:|---:|---:|
| ants | 1（spoken_languages_244） | 13 | 12 |
| conditions | 1（taxi_129） | 4 | 3 |
| contain | 1（restaurant_80） | 4 | 3 |
| drones | 1（spoken_languages_248） | 3 | 2 |
| guarantees | 1（business_negotiation_81） | 2 | 1 |
| including | 4 | 3 | 0，需逐条核验 |

各交付词在单音频内去重，因此计数下界成立；是否音频真说了该词仍需听审。
本轮没有delivery原件，不能确定那1条真正含ants的音频是否也被检出。
总Top7参考匹配720次减新目标命中701=19次正确非目标输出；这19包含所有背景词，
因此28词的47次交付至少28次参考外。这里的“28次”是输出次数下界，恰巧与“28种
全测试集重合词”同数，概念不同；不能得出恰好28次或全部47次均错。
Top5对应至少42-(658-639)=23次参考外。

这些词不增加新目标召回分子。它们可能占最终TopK槽位，也可能在更早Top64中竞争，
当前统计没有证明挤出哪些目标或总体占比；删除输出中的47项无法模拟删除索引后重跑。
尚未重新筛除28词、调门控、改评分、改90类词表或重训。下一阶段先读已有结果定位，
不要把“置信度很高”“edit=0”解释为该词在音频中出现的校准概率。

### 当前交接主线、待办与暂停项

当前主线：解释英语检索为何漏召回和误匹配，最新优先对象为Ziyan712条带噪英语。
旧6000条干净英语提供已核验的候选/门控案例和MFA一致性基线，但不是新集的已知归因。

下一步按证据顺序（均未完成）：

1. 先核验Ziyan资产/检索report、run_config、SHA清单及字节身份；只让用户返回小文件，
   不要求全outputs、100MB详情或模型。检查重复hash组及其参考，保留原统计另报敏感性。
2. 只读扫描outputs/ziyan_english_retrieval_480h_v1/ziyan_en_noise10db/retrieval_details.jsonl，
   分解399目标漏召回为未进保存候选、score/edit/posterior组合失败、合格但低于Top7；
   候选不存在不能直接区分无Anchor和Top64裁剪，需保存CTC序列重放。
3. 对ants等已确定参考不匹配的交付逐条取参考、目标、获胜音素/语言、score/edit/posterior、
   匹配CTC片段及同音频竞争目标；取少量真实正确对照，不只选误例。保留音频ID/路径，
   给用户短命令提取小报告和少量完整WAV用于听审，不伪造秒级对齐。
4. 区分参考文本/口语发音、CTC预测、标签变体、Anchor过滤、局部评分及门控/TopK。
   先说明已证实/待验证，再决定一个受控修改。若排除28词，必须新表/新输出、同372目标
   和712音频、同Head/参数配对；记录词数变化及参考条件化，不能宣布线上泛化收益。

当前未完成旁线：原6000条固定失败案例听审、13个英语标记的外语项发音来源核验、
三语4k新MFA查看dict成功回传。葡语标签准确性审计没有完成；密度匹配子集不是clean
认证，MFA对齐不能单独证明参考真的说出。现有480h训练和两视图PER结果已记0.141–0.142，
本轮不继续训练。新机器纯推理/Anchor移植仍暂停；310P完整4k检索时延及文档任务是
独立旁线，不在当前英语根因诊断中自动继续，也不宣称它们已被本轮解决。

### 文档同步与后续工程约束

本轮只改HANDOFF及docs/HANDOFF_PROMPT_2026-10-14.md，不改代码或产物；核验附件SHA、
上述算术与git diff --check，Markdown文档变更不重复运行与其无关的模型单元测试。
0.163代码交付的14定向/649全量通过、23跳过及既有Ruff/Mypy问题仅是当时记录，
不表述为本轮重跑。后续若改诊断/检索代码，按AGENTS运行Ruff、定向/全量pytest、Mypy、
diff --check，阶段交付和工区结果分别记档；新实验用新outputs，严格身份一致才resume。

容器外只同步文档，不必重跑完成实验：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

下一阶段若只核查已完成Ziyan产物，在容器内项目根目录执行以下只读短命令：

```bash
R=outputs/ziyan_english_retrieval_480h_v1
(cd "$R" && sha256sum --quiet -c sha256.txt)
jq '{status,sample_count,keyword_count,metrics,target_presence}' "$R/report.json"
```

返回资产report.json、检索report.json/run_config中的必要身份摘要及各自sha256.txt即可，
进一步样本提取命令由下一阶段确认实际JSON字段后提供，不猜接口、不重新跑完整推理。

## 0.163 2026-10-14 Ziyan带噪英语：新目标替换与Top5/Top7入口（待工区）

用户提供新JSON并要求选300–500个英语目标，替换原联合4k表的700多个英语需求词，
新词先跑MFA，再评测全部JSON音频。已明确确认：仅替换原759个英语target，保留
原英语fillers及全部ES/PT条目；不补回4,000整。继续使用三语联合候选范围、480h Head
及原Anchor/门控参数；不重训、不改检索算法、不跑最终Qwen转写。起点本地/实际远端
codex/g2p-coverage-scan均为8be370f47248681ae7868395b68760de935827ec。AGENTS已完整
读取；检查status/diff、近期HANDOFF与构表/检索/交付实现，tracked无修改，既有用户
未跟踪文件保留。旧outputs不覆盖。

### 已核验输入与冻结目标（只基于附件文本，不是工区召回结果）

容器JSON为/data/c00809914/audio/ms-swift-qwen3-asr/test/en-ziyan_2-5h_noise_10db_test.json，
本地附件140,077字节，SHA为
32465f58a3c00d36c7d4d75b9b6c7200962789b9e8de7e4c340feec624cf20e1。
顶层列表712条，每条一个audios绝对路径及一个纯文本response；没有时长、说话人或
口音字段。音频路径来自/glusterfs_103/c00809914/wenet_qwen/evaluate/data/test_zh-en/
data/en-ziyan_2-5h_noise_10db。路径/文件stem无重复，参考不改写；10dB仅按提供方命名，
本地没有音频、未核验SNR或音文准确性。工区JSON字节必须与附件一致。

冻结configs/ziyan_english_targets.v1.txt共372个ASCII单词，SHA为
8200f17b8485f9145ac27c31c4a209e474e9ce8b6cb813337d7d64f6a303340e。
以机场、酒店、医疗、餐饮、购物等内容词为候选；排除功能词、泛化会话词、数字、明显
缺撇号/拼接/错拼碎片。4字母词只保留room/card/menu/taxi等有明确场景含义的词；
282词出现在至少两条参考中，另人工按词义补入90个明确业务词的单次出现项，例如
checkpoint、dehydration、luggage、insurance、wheelchair。不根据CTC/ASR预测或分数挑词。
部分地名/人名属于英语文本中的专名，English US G2P不能保证其实际说法正确。
此次是参考文本条件化的内容词检索实验，不是盲词库测试，也非只评难专名。

按原整词规范化规则独立核对：1100个不同目标词×音频，413条含新目标、299条不含；
41条各含超过5个目标、22条各含超过7个目标。固定TopK槽位的理想容量界为Top5
85.3636%、Top7 91.5455%，不是预测所得成绩；排序/门控/声学误差只能在此之外再损失。
旧fillers可能在新参考中出现，按用户要求不删除，但单独列出重合，不作为新的目标分母。

### 实现与资产

新增scripts/run_ziyan_english.py、evaluation/ziyan_assets.py、inference/ziyan_retrieval.py
及configs/ziyan_english.workzone.json、冻结词清单。prepare只读原4k检索封存表，
SHA仍为6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b，
核对review.expanded_table_sha256及实际4k词面数量。移除原en target变体759条；全部
其他变体原样保留，包括跨语言同形词、新目标与旧filler重合词的旧变体。新目标附加
独立fresh English US MFA Top1标签；同词面最终只占一个TopK槽位，胜出音素用于交付。

prepare调用已有本地models/mfa/g2p/english_us_mfa.zip（SHA9923b38d...），不下载。
372个独立词全部MFA生成后按原90类vocab规范化；缺词、多发音、OOV/空标签导致明确失败，
保留日志和unresolved，但不发布部分词表、不偷偷更换目标或缩小分母。原参考不改。
产物：keyword_table.json、targets.json、words.txt、english_raw.dict、dataset_manifest.jsonl、
config.json、report.json、sha256.txt；MFA临时缓存不纳入交付清单。

audit读取712条JSON明确列出的音频，不扫描目录纳入其他文件；核对已冻结记录/输入SHA、
音频文件SHA和可读头部，报告实际小时数及重复文件hash计数，不把文件字节去重当声学去重。
run绑定物理GPU为逻辑cuda:0，复用完整音频离线CTC检测、三语变体合并与原导出逻辑。
原参数：Anchor64、2/3/4-gram、24 anchors/entry、offset1、radius2，score>=0.75、
edit<=0.35、posterior>=0.5、posterior_weight0.25；Top7从同一完整保存候选精确重放。
报告新目标Top5/Top7分子分母、门控前后召回、参考外候选、文件名前缀分组和TopK容量界。
不同新词、数据域、噪声及总词表规模同时改变，不能与旧英语77.37%直接解释为模型改进。
不衡量CTC PER或最终ASR WER；模型分片不新做全量哈希，记录模型config/index身份。

### 工区命令与输出策略

容器外（远端提交SHA见本轮回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，先CPU/MFA准备，再qwen环境预检与推理，GPU编号由用户填：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda run --no-capture-output -n aligner python -B scripts/run_ziyan_english.py prepare
python -B scripts/run_ziyan_english.py audit
python -B scripts/run_ziyan_english.py run --gpu 4
```

新表目录outputs/ziyan_english_assets_v1；新检索目录
outputs/ziyan_english_retrieval_480h_v1。禁止覆盖旧或已有目录；prepare失败后改用新
--assets-dir（audit/run也传相同参数），或新配置输出名；没有prepare恢复。
run中断可给原命令加--resume，仅在音频、参考、新旧词表、目标、Head、模型元数据、
配置和相关代码身份都一致时复用已校验的sample_shards；损坏行拒绝。物理GPU可换，
逻辑设备仍固定cuda:0；换词表/代码/策略需新输出目录。audit不写文件、不加载模型。

最终交付文件（原word/phoneme列表格式，以JSON音频stem为key）：

```text
outputs/ziyan_english_retrieval_480h_v1/delivery/ziyan_en_noise10db_top5.json
outputs/ziyan_english_retrieval_480h_v1/delivery/ziyan_en_noise10db_top7.json
```

返回终端精简JSON或新检索report.json、sha256.txt即可，必要时再给两份delivery；
不传完整逐样本详情或权重。核验命令：

```bash
R=outputs/ziyan_english_retrieval_480h_v1
(cd "$R" && sha256sum --quiet -c sha256.txt)
jq '{status,sample_count,keyword_count,metrics,target_presence,table_replacement}' "$R/report.json"
```

本地新定向14 passed；全量649 passed/23 skipped；新代码Ruff通过，Mypy无新增问题、
git diff --check通过。全量Ruff仍为既有5个E501，Mypy仍为ctc_overfit:260、
sharded_ctc:989、unfrozen_encoder_ctc:813三个unused-ignore，未改无关文件。
测试仅模拟MFA及检测器，WAV头部用本地stdlib读真实测试文件替代SoundFile依赖；覆盖
填充/外语/同形词保留、MFA失败不发布、JSON契约、TopK容量与重复词计数、Top5/Top7
原格式导出、只读预检、恢复与音频/结果篡改拒绝。未在本地运行真实MFA/Qwen，尚无新召回率。

## 0.162 2026-10-13 三语联合4,000词表重新生成MFA G2P查看文件（待工区）

用户要求对当前三语4k表的全部目标及填充词重新执行MFA G2P，只需查看一份合并dict。
本阶段不改现有标签、检索、阈值、Head或outputs。起点本地/实际远端
codex/g2p-coverage-scan均为60421e914f81f6fbb15d04488a4bde26376aa6cf；完整读取
AGENTS及0.159–0.161、4k构表/封存/分词/G2P实现；tracked无修改，既有用户未跟踪
PPT、文档、配置均保留。本地不运行真实MFA或Qwen，不宣称工区词条生成完整或发音正确。

新增scripts/run_joint_4000_mfa.py、evaluation/joint_mfa_dict.py及
configs/joint_4000_mfa.workzone.json。取已完成检索封存的
outputs/reviewed_joint_4000_retrieval_480h_v1/reviewed_keyword_table.json，固定SHA为
6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b。
检查SHA清单、完成状态、表身份、4,000唯一词面及三语覆盖，包含target和filler。
扩表身份核对review.expanded_table_sha256；review.reviewed_table_sha256保留的是
补词前审核目标表的来源身份，不能与扩表文件SHA混用。校验失败输出具体字段及预期/实际值。
按语言+规范词面去重，跨语言同形词分别保留，因此查看文件行数可超过4,000。

三语分别使用已存在的English US、Spanish Latin America、Portuguese Brazil本地G2P
zip，路径和已知SHA固定在配置中；缺失或身份不符在执行前停止，不联网下载。
使用训练extract_word_tokens分词，NFC/casefold和已有连接符规则，保留重音；逐语言
调用mfa g2p --num_jobs 16 --num_pronunciations 1。每个词组按原词序拼接孤立词Top1，
保留重复词；不修改原词面、不自动判别语言/代码切换、不去重音、不套西语修复规则、
不做CTC音素转换、不用旧标签补缺。数字/无词、缺词和多发音条目标为UNRESOLVED。
同时记录输入缺词及额外输出键，不能因输入/输出数量相同就认定覆盖完整。

最终供用户查看的单一文件是joint_4000_mfa.dict，格式为三列Tab分隔：
language、原词面、MFA原始音素；未解决条目为# UNRESOLVED注释行并给出原因。
它是查看文件，不是可以直接导入MFA或线上检索的词典。原始词级dict、词清单和日志
保存在新目录work/en、work/es、work/pt中；report.json与sha256.txt保留身份及计数。
三语执行成功且重新核验全部输入身份后才发布最终dict；失败保留诊断，不发布部分表。

容器外拉取（推送SHA见本轮回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，无需GPU：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda run --no-capture-output -n aligner python -B scripts/run_joint_4000_mfa.py
```

默认自动新建outputs/joint_4000_mfa_v1_<随机后缀>，打印output_dir及dictionary路径。
支持--root、--config、--output-dir、--jobs；拒绝已有目录和输入资产目录内输出。
没有resume；失败后重跑会新建目录。completed_with_unresolved表示三语命令完成但有
词条未解决，不是全部标签正确。用户只需返回最终dict；需要核验身份时再返回report.json
和sha256.txt，不传模型、MFA缓存或原outputs。

```bash
R=outputs/joint_4000_mfa_v1_实际后缀
(cd "$R" && sha256sum --quiet -c sha256.txt)
less "$R/joint_4000_mfa.dict"
```

本地定向45 passed，全量635 passed/23 skipped；新增文件Ruff通过，全量Mypy无新增
问题，git diff --check通过。全量Ruff仍为原有5个E501（用户PPT脚本2、扫描脚本3），
全量Mypy仍为ctc_overfit:260、sharded_ctc:989、unfrozen_encoder_ctc:813三个
unused-ignore，未改无关文件。新增16项测试仅模拟MFA进程，覆盖全部语言/填充/同形词、
短语顺序/重复/撇号、缺词/重音输出键变化/多发音/数字、身份/覆盖/输出保护，以及执行失败
或途中输入变化时不发布最终dict；包括原审核SHA与扩表SHA不同的真实报告结构，以及
扩表SHA缺失/错误时拒绝执行。真实三语G2P须由用户在工区运行。

## 0.161 2026-10-13 全部英语目标词MFA一致性检查入口（待工区）

用户确认先核查标签基础，不改词表/检索/Head/阈值。本地及实际远端分支
codex/g2p-coverage-scan起点均为0d67866ac09794178268466cea37a3f02e8b12b3；完整读取
AGENTS、0.159–0.160及发音规范化、训练词典、审核词表实现。tracked修改为空，
用户未跟踪PPT/文档/配置保留。本地读取用户return_bundle.zip，包SHA为
5e1536ae118c5b8a1822292e2bbbf6f4fddb6043de7eb063da8e2fef611c00f9；包内4项数据文件
校验通过，诊断及重放代码SHA与本地一致，18例为18条不同音频。案例是分层探索样本，
未听音频；音素编辑不能直接归为模型或标签错误。本阶段只提供一致性检查入口。

### 核查范围与方法

新增scripts/audit_english_target_mfa.py、evaluation/english_target_mfa.py及
configs/english_target_mfa_audit.workzone.json。默认检查已完成目标表联合检索的封存
reviewed_keyword_table.json：outputs/reviewed_joint_target_retrieval_480h_v1。
固定审核表SHA为89a63763216e9564edffa50014699e8c05f1257fb436ff8bcddc7e8fa720d9a7，
只取759个保留的en语言目标，不取填充词，不重新查询音频或CTC。

- 先核对封存表/report/run_config、完成状态/无填充及目标数；检查vocab SHA与已有
  检索使用的coverage.py代码身份，并重新核对每个保存标签的音素/token ID。
- 使用既有English US MFA G2P zip，固定SHA为
  9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02。
  同时读取实际英语训练词典，SHA固定为
  2dd3c3e045eabaeac4666a4a99cf472bf7a4ef816adf35943a072bd2262c23e2。
  只调用本地zip，不下载模型、不加载Qwen、不占GPU。
- 复用full-manifest训练extract_word_tokens分词（NFC/casefold、连接符映射、保留
  词内连接符/重音），词典key同样按normalize_training_text规范化，
  记录实际词序；每个唯一词调用MFA Top1，再按原词序拼接，保留重复词。
  含数字或无有效词的条目保持未解决，不猜读法；缺词、多发音不拼部分标签。
  非ASCII、词组/标点仅标记，不能据此自动确认语言或更改拼写。
- 保存MFA原始输出后，复用原90类CTC规范化规则比较：原标签vs新MFA、原标签vs训练
  词典、新MFAvs训练词典；报告ctc_equal/ctc_different/unresolved及逐音素编辑。
  另计原始字符串不同但CTC相同的项；不把格式差异当成真实音素差异。
- 词组结果是孤立词发音拼接，不是连读标签或音频强制对齐；官方Top1不是正确性认证。
  en源标签不代表实际说英语，不自动切换语言。训练词典无此词仅表示无法比较。
  检查结束重验输入SHA，途中变化报失败，不安装新标签、不生成新召回结论。

### 工区交付

容器外拉取，远端SHA见本轮回复：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内一条命令，不需要GPU参数：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda run --no-capture-output -n aligner python -B scripts/audit_english_target_mfa.py
```

默认自动新建outputs/english_target_mfa_audit_v1_<后缀>，打印路径及report；支持
--root/--config/--output-dir/--jobs（默认16）。拒绝已有目录及源运行/资产目录内输出。
没有resume；失败保留日志/报告，重跑到新目录。输入模型缺失/身份不符在MFA执行前失败。
产物为targets.json、configuration.json、words.txt、mfa_en/raw.dict及mfa.log、
comparisons.json/tsv、report.json、sha256.txt、return_bundle.zip。包只含交付文件和
清单，不含MFA临时缓存或模型；返回打印路径下的return_bundle.zip即可。

```bash
R=outputs/english_target_mfa_audit_v1_实际后缀
(cd "$R" && sha256sum --quiet -c sha256.txt)
```

本地仅模拟MFA进程测试，定向10 passed，全量619 passed/23 skipped；新增Ruff/Mypy通过、
git diff --check通过。全量Ruff仍为原有5个E501，全量Mypy仍为ctc_overfit:260、
sharded_ctc:989、unfrozen_encoder_ctc:813原有3个unused-ignore，无关文件未修改。
测试覆盖原始/CTC差异、短语词序/重复/撇号、缺词/多发音/OOV/数字未解决、填充词拒绝、
身份/数量/输出保护、输入途中变化及MFA失败封存；未在本地运行真实MFA或完整ASR。
工区回传前不宣称759个词发音正确或修改词表能改善召回。

## 0.160 2026-10-13 英语未进候选与门控失败：小样本配对取证入口（待工区）

用户要求先看具体失败样本，再讨论原因和修改；本阶段不改检索算法、词表、Head或阈值。
开始时本地/远端codex/g2p-coverage-scan均为f4d471bcfea96fe3319a169ea63b4643f7465c0d，
已完整读AGENTS、相关HANDOFF和实际候选/门控代码，tracked修改为空，用户未跟踪文件保留。
用户已粘贴0.159工区诊断：英语3215/4374命中、666未进保存候选、485门控失败、
8通过门控但低于Top7；扩表仅丢1个最终目标，原新6000条CTC序列/置信度完全相同。
这些是用户回传的聚合证据，本地未收到原summary/examples字节；尚不能将失败阶段
归因为标签错、模型错或多语干扰。原examples只取扩表后改变的案例，不足以检查持续漏召回。

### 抽样与证据

新增scripts/extract_english_failure_cases.py及evaluation/english_failure_cases.py、
retrieval_case_trace.py，只读旧审核表与新4k表两轮已封存的英语结果：

```text
outputs/reviewed_joint_target_retrieval_480h_v1
outputs/reviewed_joint_4000_retrieval_480h_v1
```

- 对每套英语数据按新结果的目标×音频状态分层：未进候选2例、仅score门控失败2例、
  其他门控失败1例、成功对照1例。固定seed为english-failure-cases-20261013-v1，
  按SHA256(seed,dataset,stratum,sample_id,target_id)排序；层内音频ID不重复。
  每套最多6例、三套最多18例；不足取全部且报告缺口，不跨层补选。层间可重复音频，
  输出唯一音频数量；不是随机总体样本，不从例子数量推断全量根因占比。
- 核对两轮SHA清单、Head/词表/配置/样本/记录音频SHA、目标范围与保留条目；扫描英语
  全部记录，重算旧新Top7六项计数并与原报告核对。校验CTC token ID/符号/置信度，
  复用既有门控判定，保存旧新状态和同音频同目标配对。
- 对选中案例用原Anchor实现/参数和保存的CTC序列作CPU复查，取得截断前完整排序；
  验证其前64变体与原cap查询一致，再复现全部保存变体的评分/排名/匹配token位置。
  区分无Anchor且无exact命中、截断前排名超cap、已进候选后失败；不只从不存在推断。
  当前参与重放的关键代码SHA须与原运行记录一致，不一致或复现失败直接报错。
  原运行未记录exact_automaton.py身份，本次记录其SHA并以样本实际重放一致作为限制条件。
- 每个目标保留全部索引发音变体、获胜发音、旧新前列竞争词及Top7是否选中、参考存在性、
  完整CTC token/confidence、score/edit/posterior、阈值差值和逐音素编辑。
  另用相同评分器对该目标做全起点搜索（仍受原窗口长度范围约束）；它是诊断对照，
  不改变实际候选/门控结果，不生成新的全表排名，更不代表召回必然改善。
- CTC最佳片段可能匹配别的词，不能替代听审/真实词边界。新位置仅为decoded token
  下标；原逐token帧区间未保存，内部占位step不导出为音频时间。此阶段不调用MFA，
  不认证目标发音或模型预测正确，不把表→CTC编辑自动归为模型错误。

### 工区交付与结果返回

容器外拉取（远端SHA见本轮回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
```

容器内执行，不需要GPU或aligner环境，不加载模型/音频：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/extract_english_failure_cases.py
```

自动新建outputs/english_failure_cases_v1_<随机后缀>，支持--root、--old-run、--new-run、
--output-dir、--seed；拒绝已有目录及源运行内部目录。无resume；中断后重跑新目录，
相同资产/seed取相同案例。输出summary.json、examples.json、examples.md、
target_terms.txt、sha256.txt，另压缩为return_bundle.zip便于回传；清单覆盖包内4个
数据文件，zip不递归包含自己。打印包路径和字节数，只需返回该zip，不传完整详情或权重。

```bash
R=outputs/english_failure_cases_v1_实际后缀
(cd "$R" && sha256sum --quiet -c sha256.txt)
```

本地定向测试14 passed，全量609 passed/23 skipped；新增代码Ruff通过、无新增Mypy
问题、git diff --check通过。全量Ruff仍为原有5个E501（用户PPT临时脚本2、
scan_g2p_coverage.py 3）；全量Mypy仍为ctc_overfit.py:260、sharded_ctc.py:989、
unfrozen_encoder_ctc.py:813原有3个unused-ignore，未顺手修改。测试覆盖固定抽样、
短缺、exact单音素命中、Anchor裁剪、门控算术、发音变体、保存结果不一致拒绝、
输入封存身份及输出不覆盖；没有在本地运行H200模型。工区案例尚未返回，不能声称
已经定位英语低召回根因，返回后按案例分别判断，再设计单因素验证。

## 0.159 2026-10-13 联合表1,358→4,000词的候选竞争诊断入口（待工区回传）

用户回传4k检索已完成6,800条/22份交付，要求先检查新增词如何影响候选位置，
不调阈值、不重训。原审核表与4k表的Top7命中回传分别为：
EN 3216→3215 /4374，ES 685→664 /724，PT 606→594 /736。
这些是用户粘贴的汇总；本地尚未读取工区逐样本详情，不把净差当作实际丢失样本数，
也不据此宣称已证明填充词导致全部下降。阶段起点本地与远端分支
codex/g2p-coverage-scan为906a87c572387d705ee88ec3f71fe5872276b368；完整读取AGENTS、
近期HANDOFF及原联合检索/评分/门控实现，检查status/diff/未跟踪文件，用户文件未修改。

### 诊断内容与口径

新增scripts/diagnose_joint_4000_retrieval.py及evaluation/joint_diagnostic_inputs.py、
joint_diagnostics.py。仅CPU读取以下两个已封存的检索目录：

```text
outputs/reviewed_joint_target_retrieval_480h_v1
outputs/reviewed_joint_4000_retrieval_480h_v1
```

- 核对SHA清单中的run_config、report、词表/索引快照、每组retrieval_details；
  检查相同音频的记录SHA、参考、目标集合、Head、模型路径/已记录元数据、检索设置及
  核心评分代码身份。原目标词面/发音条目必须完整保留，新增条目必须声明为filler。
  不重新打开音频或完整模型。原运行未哈希模型权重分片，不能认证其字节相等。
- 重现发音候选→词面去重，再按原规则重放Top5/Top7，重算参考中的整词/短语匹配；
  独立核对每组及各语言原报告的6项计数。新增填充词匹配到参考或身份不一致时失败，
  不输出一份看似成功的诊断。
- 每个真实目标归入互斥状态：已选中、未进入保存的候选、通过门控但排不进TopK、
  或未通过score/edit_ratio/posterior门控（同时失败时记录组合，避免重复计数）。
  “未进入候选”无法区分没有Anchor命中与Top64之前被裁剪；Top64单位是发音变体。
- 配对列出丢失命中、新增命中、保留命中及新增/消失的参考外候选，不能用净差替代。
  统计参考外候选来自新增filler还是原审核需求词、获胜发音语言、音素长度及零编辑比例。
  语言字段表示词条来源，不证明音频说的是该语言；参考外也不等于未经听审的错误声音。
- 记录两轮CTC token/confidence序列是否完全相同；记录两轮都存在的目标分数和局部
  匹配位置是否改变。若CTC本身不同，限制“只由候选竞争导致”的解释。
- 额外从新保存候选列表移除fillers并重放原门控，仅统计已丢目标能恢复多少。
  这是保存列表内的条件检查，不能恢复之前被Top64裁掉的目标，不等于重跑原词表。
- examples.json按语言/失败阶段取首个固定顺序样本，同时含新增目标命中、新增参考外
  filler示例；保存参考、旧新排名/分数/门控字段、获胜词条音素、匹配CTC片段及上下文。
  CTC片段没有真实词边界，不能仅据此确认“截取词内子片段”；不只挑成功案例。

### 工区命令、产物与限制

容器外（远端提交SHA见本轮回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，不需要GPU参数，不运行CTC或Qwen推理：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/diagnose_joint_4000_retrieval.py
```

每次自动新建outputs/joint_4000_diagnostics_v1_<随机后缀>；不修改原检索/模型/词表。
支持--old-run、--new-run、--output-dir；输出目录已存在或位于源检索目录内部则拒绝。
此阶段不需恢复推理；CPU扫描中断可重跑到新目录，不复用半份报告。

产物：summary.json（精简Top7统计）、report.json（Top5/Top7及每数据集明细）、
examples.json（有限配对示例）、sha256.txt。先返回屏幕JSON；再按需返回summary.json、
examples.json、sha256.txt，无需上传完整retrieval_details、outputs或权重。
检查命令（R替换为新生成目录）：

```bash
(cd "$R" && sha256sum --quiet -c sha256.txt)
```

本地使用封存的模拟检索产物测试身份、排名去重、门控回填、失败阶段、配对增减、
跨语言来源、输出不覆盖及原产物不变。工区真实诊断尚未执行，不声称已经定位实际根因。
定向测试9 passed；全量601 passed、23 skipped；新增文件Ruff及diff --check通过。
全量Ruff仍有5项已有E501（用户PPT临时脚本2项、scan_g2p_coverage.py 3项）；
全量Mypy仍有ctc_overfit.py:260、sharded_ctc.py:989、unfrozen_encoder_ctc.py:813
三项已有unused-ignore，新增代码无类型错误；这些无关文件未修改。

### 指定词项的MFA原始发音复核入口

用户另提供仓库根目录mismatch_words_en/es/pt.txt，要求使用既有MFA G2P模型生成
原始结果。首条shell命令的单词预检在合法词组（如el niño）处主动停止，MFA未启动、
未创建输出；不能将此解释为模型或输入文本损坏。新增scripts/run_mismatch_mfa.py：
UTF-8逐行保留原词项与行号，NFC+casefold后按空白拆词；每语言唯一词调用当地语言
G2P Top1，再按原词序拼回（重复词保留）。保留重音和标点，不猜测father s day中的
撇号，不自动识别或切换语言，不应用西语代理/去鼻化及CTC词表规范化。
缺词或多发音歧义时整条phrase pronunciation为null，保留逐词结果及显式原因；
completed_with_unresolved表示流程结束但存在未完整生成的词项，不认证标注正确。
组合结果是孤立词发音拼接，未模拟跨词连读，不是基于测试音频的校准。

容器外在/home/star/q00933266/qwen3-asr-hotword执行
git pull --ff-only origin codex/g2p-coverage-scan；远端SHA见本轮交付。
容器内在/host_home/star/q00933266/qwen3-asr-hotword执行：

```bash
conda run --no-capture-output -n aligner python -B scripts/run_mismatch_mfa.py \
  --output-dir outputs/mismatch_mfa_phrases_v1
```

不下载模型、不使用Qwen/GPU、不改已有表或outputs。模型从models/mfa/g2p中读取
english_us_mfa.zip、spanish_latin_america_mfa.zip、portuguese_brazil_mfa.zip，
报告实际模型/输入SHA、脚本SHA、Python/MFA版本与调用参数，每语言隔离MFA工作目录。
默认--jobs 16；省略--output-dir则自动创建outputs/mismatch_mfa_phrases_<随机后缀>。
拒绝已存在的输出目录，不提供resume；失败时保留日志和失败报告，重跑使用新目录。
结果为en/es/pt子目录的input.txt、words.txt、raw.dict、phrases.json、phrases.tsv、
mfa.log，根目录report.json和sha256.txt。SHA清单覆盖交付文件，不包含mfa_work缓存。
返回report.json、三份phrases.json和sha256.txt即可；可用
(cd outputs/mismatch_mfa_phrases_v1 && sha256sum --quiet -c sha256.txt)核验。
本地仅使用模拟MFA进程测试拆词/拼回、缺词不拼残缺标签、隔离和不覆盖；未运行真实MFA。
定向3 passed、全量604 passed/23 skipped；新增脚本Ruff/Mypy与diff --check通过。
全量Ruff/Mypy仍仅有上文记录的5项E501与3项unused-ignore，无关文件未修改。

## 0.158 2026-10-13 审核后三语联合表扩充4,000词与同目标对照入口（待H200）

用户要求把已审核三语表补到4,000个不同词面，新增词不能出现在任何测试转写中，
并排除无关短词，然后复测原11套干净音频。此阶段只增加候选词，不调整模型、训练、
Anchor、评分、门控、Top5/Top7及目标集合；继续每条音频检索三语整表，交付按语种分开。
开始时本地及远端codex/g2p-coverage-scan均为
89bd2aab40c99b644e1e6c522c5592ca939e32c2。已完整读取AGENTS、0.156–0.157、
英语3000构表和三语共同检索实现，检查Git状态、diff及远端；用户未跟踪文件未修改。

### 固定输入、筛选与解释边界

- 审核表outputs/multilingual_target_review_v1/keyword_table.json SHA固定为
  89a63763216e9564edffa50014699e8c05f1257fb436ff8bcddc7e8fa720d9a7。
  用户回传为1,358个不同词面、1,370个语言发音条目，语言计数en759/es316/pt295。
  本地没有工区真实词表；构表时重验SHA及原审阅记录，保留所有存活条目原样。
  过滤只用于新增词；不再次删除用户已经审核保留的短专名，不改变目标分母。
- 读取已完成outputs/reviewed_joint_target_retrieval_480h_v1的封存run_config/report，
  从实际配置取得11套transcripts路径，要求逐文件SHA、样本ID和文字与上轮一致。
  英语3×2,000，西语4×100，葡语4×100；全体6,800条参考形成统一禁入集合。
  NFKC+casefold+标点转空格，按完整token排除；保留重音，不是子串或听审判断。
- 新增词仅从训练清单文字提取，结合已存在MFA词典的唯一精确词条；不生成新发音。
  en：Swift US训练345,820条的已知SHA清单及既有English US MFA词典。
  es：已冻结480h训练清单（145c16e2…5b300）及Noah修复词典（0a233860…e3269）。
  pt：783h训练池（47214e4b…b3fd4）及outputs/noah_pt_mfa_g2p中的原巴葡词典。
  EN/ES字典及三份train清单SHA配置固定；PT字典没有已核验的历史SHA，本次记录其
  实际SHA并绑定后续加载/恢复，不能把这一点表述为与某历史字典版本独立核验一致。
- 必须是纯拉丁字母单词、至少4字母、至少4个词表音素、训练出现至少5次；排除三语
  基础词规则并集、数字/缩写式标点/连字符/撇号、字典缺失/多发音歧义/OOV条目。
  原全部需求词（包括规则移出和用户删除的词）也在补词禁入名单，不能借另一语言补回。
  这些规则减少短片段和孤立拼写，不认证每个词的业务意义、词性或真实发音。
- 对合格词按固定seed+语言+词面SHA排序，EN→ES→PT轮流选取，跨语言词面全局去重。
  本次预期补EN881、ES881、PT880，共2,642词；最终4,000词面、约4,012个索引发音。
  不足时失败并指出语种，不重复凑数、不自动放宽规则或改变语言配额，不发布半张表。
- 不按检索分数或新预测选词，也不刻意选择/排除自然近音词。排除参考中出现的补词
  是用户指定的受控干扰实验，因此不称盲测；词面不出现在参考中不代表声音没有近音。
  此轮回答“按上述补词策略扩大候选池后表现怎样”，不能泛化成所有4k词表性能。

### 代码与输出

新增scripts/build_reviewed_joint_4000.py、evaluation/joint_fillers.py及两份配置：
configs/reviewed_joint_4000_keywords.workzone.json、
configs/reviewed_joint_4000_retrieval.workzone.json。
复用run_reviewed_target_retrieval.py，扩展仅限加载经校验的补词及同目标基线报告。
旧检索/评分核心不变，原始targets仍来自独立验证过的审核表，补词不进入目标分母。

新构表目录outputs/reviewed_joint_4000_keywords_v1：
keyword_table.json（完整4k表）、reviewed_keyword_table.json（原审核表快照）、
selected_fillers.jsonl（来源、语言、训练频次和音素）、keywords.txt、
excluded_reference_words.json、basic_word_policy.json、config.json、report.json、sha256.txt。
原表及其旧SHA清单保持不变；新表不可直接手工改写后恢复，需新版本构表。

新推理目录outputs/reviewed_joint_4000_retrieval_480h_v1：

```text
languages/en/delivery/  # 3套 × Top5/Top7，6份
languages/es/delivery/  # 4套 × Top5/Top7，8份
languages/pt/delivery/  # 4套 × Top5/Top7，8份
```

共22份stem→[{word,phoneme}]，保留实际候选数和获胜发音，仍有原门控；不是强行输出7词。
逐样本详情、run_config、词表/索引快照、report及sha256继续保存。
报告by_language中的current为本次4k，reviewed_baseline_same_targets为上次1,358词；
baseline_same_retained_targets仍保留更早单语3k/4k在相同目标上的参考数值。
从封存原始详情重算基线，核对相同音频SHA、参考、目标IDs、Head、门控、模型配置，
不需要重跑旧模型；综合比率用计数加总。没有最终LLM转写、WER或CTC PER结论。

### 工区命令与恢复

容器外拉取（远端提交SHA见本轮交付回复）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内先构表（CPU，读取训练文字和词典，不读取训练音频，不加载模型）：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/build_reviewed_joint_4000.py
```

成功后预检与运行，GPU编号由用户选择：

```bash
GPU=4
C=configs/reviewed_joint_4000_retrieval.workzone.json
python -B scripts/run_reviewed_target_retrieval.py --config "$C" --audit-only &&
python -B scripts/run_reviewed_target_retrieval.py --config "$C" --gpu "$GPU"
```

构表拒绝已有目录，没有resume；不需重复构表。推理中断时仅最后一条加--resume。
只有输入、表、音频、参考、Head、配置、代码身份相同才可复用分片；新词表绝不复用
上一轮1,358词结果分片。不要覆盖/删除已有outputs。改构表规则要同时指定新词表和
新推理目录；源码更新后旧任务的严格恢复可能拒绝，不能删除身份检查强行恢复。
模型权重继续只绑定config/index，未逐权重分片hash；保留0.157的这一已知限制。

完成后返回两份report.json及sha256.txt即可，交付件仍按上面的三语目录交给下游：

```bash
K=outputs/reviewed_joint_4000_keywords_v1
R=outputs/reviewed_joint_4000_retrieval_480h_v1
(cd "$K" && sha256sum --quiet -c sha256.txt) &&
(cd "$R" && sha256sum --quiet -c sha256.txt) &&
jq '{status,sample_count,review,by_language,delivery_file_count}' "$R/report.json"
```

### 本地验证

新增16项测试覆盖跨语种参考排除、重音/标点/大小写、短词/短音素/基础词/低频/OOV、
不补回删除目标、均衡全局去重与4k数量、确定性、输入异常拒绝且不生成半成品、
审核目标原样保留、11组22份交付、同目标分母和旧基线、相同运行恢复与篡改拒绝。
定向52 passed；全量592 passed、23 skipped；新增/修改代码Ruff与定向Mypy通过。
全仓Ruff仍5个既有E501（用户PPT临时脚本2个、scan_g2p_coverage.py 3个）；全仓Mypy
仍3个既有unused-ignore（ctc_overfit.py、sharded_ctc.py、unfrozen_encoder_ctc.py），
未修改无关文件。git diff --check通过。所有本地推理均mock；没有本地运行完整模型，
真实可用补词数量及本轮召回必须等待用户工区运行。

## 0.157 2026-10-13 人工审核后三语整表共同检索与22份交付入口（待H200）

用户已在0.156生成的keyword_table.json中删除不需要的完整条目，要求复测干净英语三套
和西/葡语四wave。用户进一步明确：每条音频都使用审核后的整张三语表，不按音频语言
过滤候选；仅交付目录按音频语种区分。此确认覆盖了开发初期准备按语言取子表的假设。
开始时本地及远端codex/g2p-coverage-scan为7b535d3e4267a78fd9915245943fd947b10ec2bf，
已完整读取AGENTS.md、检查Git状态及0.146–0.147、0.152–0.153、0.156记录；用户已有
未跟踪文件均保留。本地无审核后的实际词表或音频，本节只记录代码交付，不报告召回。

### 范围、词表身份和公平比较

入口scripts/run_reviewed_target_retrieval.py；配置configs/reviewed_target_retrieval.workzone.json。
新模块evaluation/reviewed_targets.py、inference/reviewed_retrieval.py及reviewed_joint.py。
不修改既有CTC模型、Anchor索引/评分实现、旧评测入口或任何既有outputs。

- 读取outputs/multilingual_target_review_v1/keyword_table.json作为用户审核的新输入。
  手工删除后旧sha256.txt中的该文件SHA自然不再匹配：不要求重跑构表、不改写旧校验清单。
  另核验原封存report及all_target_decisions.jsonl，重建原始审阅条目，要求新表仅删除
  完整entries成员；剩余条目的词面、语言、来源和音素保持原样。重复、添加、改写或
  JSON破损在模型加载前报错，不静默修复。记录原表SHA和审核后实际SHA及删除数量。
- 三语所有保留词共同参与每条音频的检索。相同规范词面、相同token序列的重复发音只
  建一个索引项；不同发音均参与Anchor检索，保留原始变体排名。按规范词面保留最高
  排名的变体，再按原门控选Top5/Top7；同词面只占一个名额，下游phoneme使用该音频
  实际胜出的变体，原审核总表快照保留全部语言来源。不是每语种使用自己的子表。
- 固定480h Head SHA为320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265；
  模型路径/glusterfs_103/models/Qwen3-ASR-1.7B。沿用完整音频、bf16、各音频语言参数、
  Anchor Top64、门控0.75/edit≤0.35/posterior≥0.5和Top5/Top7。
  Top64按发音变体计数，去重后唯一词面可能不足64；报告明确这项多发音适配语义。
  这不是可与旧单语词表声称“仅删几个词、其他条件完全相同”的实验：表规模、组成、
  三语竞争及部分词的多发音可用性一起变化。评分/门控参数未调优。
- 复用并核对两份已完成干净基线的run_config/report及输入SHA：
  outputs/english_3000_retrieval_480h_v1（3×2,000条），以及
  outputs/wave_4000_retrieval_480h_old_only_v1（4×2×100条）。从旧配置获得精确音频和
  transcripts路径，逐条核对参考、ID、音频SHA、原词表及门控/模型配置，不用带噪目录。
  总计11组、6,800条；相同stem在不同组中独立保存。
- 新指标仅按每组原声明目标与审核保留词的交集计算。并读取旧逐条结果，在同一保留
  目标分母上重新计算baseline_same_retained_targets；不重跑旧模型，也不拿旧全目标
  百分比直接比较。语言综合指标为总命中/总分母；全表误选仍按全部三语候选与参考
  匹配判断。空目标分母保留null。未出现不等于人工确认的幻觉，不测最终WER或CTC PER。

### 工区命令与输出

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内先预检，再运行；GPU编号由用户选空闲卡：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_reviewed_target_retrieval.py --audit-only &&
python -B scripts/run_reviewed_target_retrieval.py --gpu 4
```

不重新运行build_multilingual_target_review.py，不重新封存或覆盖用户审核文件。
默认新输出outputs/reviewed_joint_target_retrieval_480h_v1，文件结构：

```text
report.json / run_config.json / sha256.txt
reviewed_keyword_table.json       # 实际审核文件的内容快照
indexed_pronunciations.jsonl      # 真正参与共同检索的发音索引条目
languages/en/delivery/            # en_chile/en_colombia/en_mexico × Top5/Top7，共6份
languages/es/delivery/            # wave1_es … wave4_es × Top5/Top7，共8份
languages/pt/delivery/            # wave1_pt … wave4_pt × Top5/Top7，共8份
languages/<lang>/<dataset>/       # 每样本shard、详细结果及指标
```

交付JSON仍为stem→[{word,phoneme}]，保持排序、实际不足K不补足，空召回[]。三语合计
22份下游文件。来源语言目录不限制候选语言，英语交付中也可能出现西/葡语词。
词表、基线、音频、参考、模型元数据、Head、配置或代码身份变化不得复用旧shard。
仅同一次中断运行追加--resume；词表再改则用新--output-dir，例如v2。不允许在原审阅
目录或旧推理目录内创建新实验。旧outputs一律不覆盖；源码更新导致恢复身份变化时
明确拒绝。本轮模型权重只绑定config/index，未逐分片hash，报告保留此局限。

```bash
python -B scripts/run_reviewed_target_retrieval.py --gpu 4 --resume
```

完成后校验并返回report.json及sha256.txt两个小文件；22份delivery交下游：

```bash
R=outputs/reviewed_joint_target_retrieval_480h_v1
(cd "$R" && sha256sum --quiet -c sha256.txt)
jq '{status,sample_count,review,by_language,delivery_file_count}' "$R/report.json"
```

### 本地验证

新增15项测试：允许已授权人工删除导致的旧表SHA变化，但绑定剩余原音素；三语共同
检索、相同发音去重、同词Top-K去重及胜出发音交付；11组/22份文件、负样本空列表、
同stem隔离、同目标旧结果重算、恢复仅补缺失样本、词表/音频/参考/Head变化拒绝恢复、
禁止写入原资产目录。定向101项通过；全量576 passed、23 skipped，git diff --check通过。
新增代码Ruff及定向Mypy通过。全仓Ruff仍有既有5个E501（用户PPT脚本2个、
scan_g2p_coverage.py 3个）；全仓Mypy仍有既有3个unused-ignore（ctc_overfit.py、
sharded_ctc.py、unfrozen_encoder_ctc.py），均未修改。所有本地推理为mock；H200实测
仍须用户运行，收到结果前不得声称新词表提高召回。远端提交SHA见本轮回复。

## 0.156 2026-10-13 三语需求热词合并与基础词过滤审阅入口交付（待工区）

用户要求把英语三套数据的773个需求目标、西语wave1–4的316个需求目标、葡语wave1–4
的295个需求目标合并，移出a/the/me/I等基础词后先人工检查。本阶段只交付词表生成器，
不做推理、不改检索或训练、不补充充数词。开始时本地和远端codex/g2p-coverage-scan
均为47baeb86815c51dc3316d45e8896460258e4792d；已核对Git状态及相关HANDOFF资产记录。
原有用户未跟踪图片、PPT、文档及pt_quality_whisper配置均不修改、不提交。
本地没有工区完整词表，因此本节不报告实际移出/保留数量；输入是1,384个语言目标条目，
跨语言合并后的唯一词面数量须由工区生成结果确认。

### 输入与过滤规则

入口scripts/build_multilingual_target_review.py，配置
configs/multilingual_target_review.workzone.json，独立模块evaluation/target_review.py。

- 英语：outputs/english_3000_keywords_v1中的keyword_bias_phoneme.json及
  targets_by_dataset.json。固定表SHA为
  6e2a7016d16bf86cb1f31f3254213e58f5becd4810cc02a166fd604d1cd9b655，目标范围SHA为
  e3b362df7708ece0c2b9a444199997ac0b157235581a909a8d1e2bbb2b97924f。
- 西/葡语：outputs/wave_4000_keywords_old_only_v1中各自的
  <es|pt>/keyword_bias_phoneme.json和targets_by_wave.json。
- 每套输入核对封存sha256.txt、report和目标组名称/数量，只遍历目标清单成员，从已有表
  取回原词面与音素；不会将3k/4k表中的填充词加入新表。西/葡语未另造已知固定SHA，
  以各自已封存清单核验，并将实际输入身份写入新report。
- configs/multilingual_basic_words.v1.json是显式可审阅规则：按语言移出独立冠词、
  代词/限定词、常见介词/连词、常见助动词/系动词及否定词。仅匹配整个规范单词，保留
  重音差异；The Who、El Paso、São Paulo等完整短语不拆删。普通实词不会仅因常见或短
  而删除，短词和短音素只做提示。
- 不将基础词判定冒充词义审核：Will/May/São等同形专名或字母名称可能被规则移出，
  删除清单明确标注歧义。每语种keep_overrides可恢复命中规则的词；修改后用新目录。
- 同规范词面跨语言合并时保留全部语言、原词面、音素和来源，不强选一种发音。
  保留和移出清单仍按语言展开，便于逐项审阅。

### 工区命令、产物与恢复约定

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内（仅CPU，小文件读取，无需GPU或完整模型）：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/build_multilingual_target_review.py
```

默认新目录outputs/multilingual_target_review_v1。输入预检失败不会创建输出；目录已存在
则拒绝覆盖，不提供resume。规则或输入改变后用新名称，例如
`--output-dir outputs/multilingual_target_review_v2`，保留所有旧产物。

- review.md：保留词表，含语言、原音素、来源组及审阅提示；优先给用户阅读。
- removed_basic_words.md：移出词及规则理由，检查误删、同形专名并决定恢复。
- keyword_table.json：合并总词表，保留所有语言发音变体；当前是审阅格式，并非直接可供
  旧推理入口加载的配置。keywords.txt提供唯一显示词面清单。
- all_target_decisions.jsonl：全部输入目标的保留/移出决定，可追溯且不丢失删除项。
- report.json、config.json、basic_word_policy.json及sha256.txt：数量、输入/代码身份、
  规则快照及产物校验。本轮human_approved=false，不宣称发音或业务相关性已获人工确认。

验证与回传：

```bash
cd outputs/multilingual_target_review_v1
sha256sum --quiet -c sha256.txt
cat report.json
```

请返回report.json、review.md、removed_basic_words.md及sha256.txt即可；人工检查后再
确定最终范围。删除基础词会改变评测目标分母，不能把未来新召回率直接与旧指标比较。
本阶段不使用音频、参考转写、检索预测或模型；没有热词召回/最终转写效果结论。

### 本地验证

新增21项测试覆盖目标限定、基础词/短语/重音、同形多语言发音、显式恢复例外、确定性、
源SHA及数量/语言/OOV异常、拒绝覆盖和拒绝写入原词表目录。定向测试106 passed；
全量561 passed、23 skipped。新增代码Ruff与定向Mypy通过；全仓Ruff仍有5个既有E501
（用户PPT临时脚本2个、scan_g2p_coverage.py 3个），全仓Mypy仍有3个既有unused-ignore
（ctc_overfit.py、sharded_ctc.py、unfrozen_encoder_ctc.py），均未顺手修改。
合成小样本CLI生成与产物SHA校验通过，只证明工具行为，不能代替工区真实词表结果。
交付提交SHA见本轮回复及git rev-parse HEAD。

## 0.155 2026-10-12 英语60条最终ASR提示排序/优先级对照交付（待H200）

本阶段响应用户明确的新实验：复用已完成的干净英语Top7，只测完整Qwen3-ASR最终转写。
不运行CTC检索、不训练、不改检索算法/门控，不使用带噪音频或流式状态，不覆盖旧outputs。
开始时本地及origin/codex/g2p-coverage-scan均为980b48ed2609da8a1635d3209ea6bbd6dd16cb77，
已完整读取AGENTS.md及本文件0.151–0.154的英语资产、结果和交付约定。原有用户未跟踪
PPT、图片、报告及pt_quality_whisper配置均不修改、不提交。本节为新实验入口完成节点，
不是H200实测结果；交付commit SHA见本轮回复及git rev-parse HEAD。

### 固定资产与新目录

入口：scripts/run_english_prompt_order.py，配置configs/english_prompt_order.workzone.json。
独立模块：evaluation/prompt_order.py、prompt_order_metrics.py及
inference/prompt_order_experiment.py、prompt_order_runtime.py。

- 三套原始干净数据：/host_home/star/两波验证集-可直接推理/
  <en_chile|en_colombia|en_mexico>/en/{wav,transcripts.txt}，预期各2,000条。
- 已完成检索：outputs/english_3000_retrieval_480h_v1，读取封存run_config、report和
  delivery/<dataset>_top7.json；保留delivery数组原顺序，不重跑、截取新排名。
- 固定表：outputs/english_3000_keywords_v1/keyword_bias_phoneme.json，SHA
  6e2a7016d16bf86cb1f31f3254213e58f5becd4810cc02a166fd604d1cd9b655。
- 固定目标范围：同目录targets_by_dataset.json，SHA
  e3b362df7708ece0c2b9a444199997ac0b157235581a909a8d1e2bbb2b97924f。
  两者与用户此前返回的sha256(2).txt核对；表定义身份/范围，不注入整张3k表。
- 检索报告Head身份须为320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265，
  仅核对来源，不加载CTC Head。
- 完整模型：/glusterfs_103/models/Qwen3-ASR-1.7B。配置中固定已知config/index SHA，
  prepare逐字节计算所有权重分片及tokenizer/processor/模板等资产SHA，后续audit/run重验。
  分词器接受tokenizer.json，或vocab.json+merges.txt；tokenizer_config.json两种布局都
  必须存在，不能独自替代词表。所有实际存在的分词器文件继续参与SHA256和恢复校验。
  用户首次prepare因旧预检错误地强制要求tokenizer.json而停止；
  [官方模型文件清单](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)
  实际采用vocab.json+merges.txt。修复只调整资产布局识别，不改实验定义或模型加载。
  该失败发生在freeze创建目录之前；拉取修复后直接重跑prepare，无需删除输出或--resume。
- 新输出：outputs/english_prompt_order_v1。prepare拒绝已有目录；不同模板、顺序规则或
  输入使用另一个新目录，不能借--resume混入旧结果。

### 先冻结样本，再运行

seed=20261012。每组按目标/参考与旧Top7分层：10条target_hit（参考含本组目标且Top7
至少命中一个）、5条target_miss（参考含本组目标但Top7无正确目标）、5条no_target。
各层按SHA256(seed, selection, dataset/stem)排序取样，不读任何新ASR转写。
若某层不足，先取各层原配额，再按target_hit→target_miss→no_target固定顺序从未选
样本补足。记录原始缺额、requested_stratum、actual stratum、fallback；仍不足20条则
冻结不足报告并拒绝推理。不会把补选样本伪称原配额层，也不会挑既有ASR错误样本。

固定样本ID、来源、原始词面/音素、分层、参考、音频SHA、所有条件context及执行顺序
写入freeze/。保留不足两个候选、空候选、Oracle多于7词的计数。每样本组执行顺序也
固定seed打散，以减少固定执行次序与条件混淆；每次调用前重置相同的该样本随机seed。

| 组 | context内容 |
|---|---|
| A | 空context，无热词说明 |
| B | Top7交付原顺序；不足7保持实际数量 |
| C | B倒序，词集合不变 |
| D | B按seed+sample ID+规范词面哈希确定的随机排列 |
| E | D同序，全部priority: normal |
| F | E同序，仅B的第一个候选priority: high，其余normal |
| G | 本组声明目标与参考的交集，按首次规范token位置排列；全部保留 |
| H | G完全相同候选倒序 |

B/C/D/G/H英文模板完全相同：

```text
The following terms are spelling references. Use a term only when supported by the audio. Do not insert a term that is not spoken.
Term A
Term B
```

E/F英文模板完全相同，仅一个priority值变化：

```text
The following terms are spelling references. Priority indicates preference only when the audio supports multiple candidates. Do not insert a term that is not spoken.
Term A — priority: normal
Term B — priority: high
```

空候选直接使用空context，不虚构词；规范化去重保留原词面及音素，音素只作来源记录，
prompt不注入音素。Oracle首次位置相同以规范词面、原词面依次打破并列；重叠的声明
短语独立保留。G/H可含Top7未召回的正确目标，超过7也不截断，不注入整句参考/普通词。
D允许随机排列恰好等于B/C，明确统计相同prompt，不强制为获得差异而重排。

### 实际Qwen调用与控制变量

已检查公开qwen-asr==0.0.6的源码wheel（仅Python源码、无模型权重），对照官方
[离线接口源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)。
运行前用安装包版本及4个关键源文件SHA核验：inference/qwen3_asr.py、inference/utils.py、
modeling_qwen3_asr.py、processing_qwen3_asr.py。若不一致先失败，不能自动升级或猜API。

实际接口为transcribe(audio, context, language, return_time_stamps)，不接受臆造的
生成参数。使用官方transformers后端，from_pretrained设bf16、eager attention、
max_inference_batch_size=1、max_new_tokens=512；thinker.generation_config固定
贪心do_sample=False、num_beams=1、num_return_sequences=1，其完整实际配置落盘。
language固定English，return_time_stamps=False，完整音频每次独立请求，无上一条对话。
要求音频在(0,1200]秒以避免官方内部长音频切块；每条实际生成次数须恰好1。

固定Python/NumPy/Torch/CUDA seed、deterministic algorithms、禁用TF32、cuDNN benchmark；
CUBLAS_WORKSPACE_CONFIG=:4096:8，离线加载已存在本地模型，不下载。记录包版本、
实际GPU/CUDA/cuDNN、运行参数、完整生成配置，不能宣称跨硬件/版本必然逐bit一致。

调用边界拦截并检查真正传入的context、两条system/user消息、序列化prompt，以及
传给thinker.generate的input_ids。只允许processor把audio占位token展开，其他token
必须与完整prompt一致；改写、截断、额外分块、继承past_key_values或改变解码参数时报错。
保存完整实际context、序列化prompt、token哈希/长度、实际生成参数、转写和截断上限标记。
同时每组数据确定性选1条重复完全相同B请求，3次额外推理与480次主实验分开统计；
若重复已有输出不同，须先考虑运行波动，不能直接把所有配对变化归因于prompt。

### 报告和解释边界

report.json按overall、dataset、实际采样层、dataset×层输出每组计数/分母/比率：
最终声明目标召回、正确注入词采用、参考不存在的注入词落字、无目标样本误落字、WER
及S/D/I。计量为不同规范词×音频对，同词在同一音频多次出现只计一次；合并以总命中
除以总分母，分母0为null。WER统一NFKC/casefold/Unicode标点为空格的词序列编辑距离，
不去音调、不另做数字展开；empty reference单独统计，不套用不同条件的规范化。

配对A/B、B/C、B/D、C/D、E/F、D/E、G/H、A/G、B/G，输出逐条原始/规范转写变化、
正确目标新增/丢失、错误候选新增/消失、WER改善/退化数。错误候选比较在两组参考缺失
候选并集上同时匹配两个输出，避免将A没prompt导致的空集合误当作A完全没说错词。
E/F额外只看被提升的B首词，按参考存在/缺失报告normal/high两侧落字计数。

排序对照仅≥2候选且实际prompt变化的样本计入有效分母，相同/不足2个单列。
代表例按每对比较×unchanged/improved/worsened/changed_equal_wer固定哈希选1条，
不存在的类别显式列空；完整60条配对结果均保留。这里improved/worsened仅指WER，
不替代热词质量判断，不只展示成功样本。

结论限定：这是60条分层探索，不直接代表6,000条；文字匹配错误不等于听审确认的
幻觉；采用率不是prompt造成采用的因果证明；Oracle不是线上能力或严格数学上界。
B/G同时改变候选覆盖、正确性、数量和顺序，不能解释为纯排序效应。实验新提示模板
也不能直接与其他历史ASR流水线指标作单因素比较。

### 工区执行与安全恢复

容器外（分支codex/g2p-coverage-scan）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内首次准备：读取旧交付、音频/模型身份，冻结60条；无模型加载/推理。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_english_prompt_order.py prepare
```

查看冻结摘要后运行。4替换成空闲物理GPU，脚本映射为逻辑cuda:0：

```bash
python -B scripts/run_english_prompt_order.py run --gpu 4
```

run会自动重复完整预检；如只想再次只读核验：

```bash
python -B scripts/run_english_prompt_order.py audit
```

相同实验中断后恢复，不再prepare：

```bash
python -B scripts/run_english_prompt_order.py run --gpu 4 --resume
```

恢复同时比较冻结plan/ID/音频/候选/参考/context、模型权重/配置/tokenizer、相关代码SHA、
Git commit及依赖身份。运行时还要求实际解码配置、GPU和环境相同；不同GPU编号也拒绝
未完成运行的复用，应保持原卡/环境。每个sample×group结果原子保存、带独立摘要；
篡改或缺失已封存结果会失败，不重新覆盖。完成的相同运行只校验返回，不加载模型。
独占进程锁防同时写；中断请求未保存则重做，attempts.jsonl分别计入已开始/已保存请求，
未保存的尝试可能尚未或已经到GPU，不能精确声称每次中断都执行了推理。

输出目录下：freeze/{plan.json,samples.jsonl,sample_ids.txt,preparation_report.json,sha256.txt}；
runtime.json；results/<UID哈希>/<A-H或B_repeat>.json；predictions.jsonl（483条预期）；
paired_results.jsonl（60条，包含8组候选、实际prompt、转写及配对结果）；
representative_examples.md；report.json；sha256.txt。音频/模型不复制。

```bash
(cd outputs/english_prompt_order_v1 && sha256sum -c sha256.txt)
cat outputs/english_prompt_order_v1/report.json
```

返回report.json、sha256.txt、representative_examples.md即可，无需权重或全部旧outputs。
prepare如报告缺额，把freeze/preparation_report.json一并返回；缺额与补选会在新推理前
冻结保存。如果遇到版本/source SHA不符，返回错误和安装版本，不绕过校验或自行升级。

### 本地验证（不等于H200结果）

定向测试31通过，包含8组变量隔离、去重/空/单候选、Oracle>7与并列顺序、分层缺额与
不足阻断、micro/WER/共享错误候选分母、优先词分析、重复请求波动单列、483次mock完整
运行、中断仅补缺、已有产物保护、资产/代码/配置/运行环境变化拒绝恢复、结果篡改拒绝。
Qwen边界用轻量替身测试真实封装逻辑，覆盖context改写、额外历史、token截断/改写、
继承状态、采样参数改动、额外生成调用；未加载任何完整Qwen模型。
全量pytest：530 passed / 23 skipped（本地无Torch等可选运行依赖）。
本轮相关文件Ruff、定向Mypy通过；git diff --check通过。
全仓Ruff仍有5个既有E501：用户PPT辅助脚本2处、scripts/scan_g2p_coverage.py 3处；
全仓Mypy仍有ctc_overfit.py:260、sharded_ctc.py:989、unfrozen_encoder_ctc.py:813的
3个既有unused-ignore。未修改无关代码。本轮尚无排序/优先级有效性的实测结论。

分词器布局回归新增10例：fast/BPE/两者共存、配置独存及BPE缺件拒绝、分词器变化
或新增tokenizer.json时禁止恢复。修复后定向41通过、全量540通过/23跳过；本轮Ruff、
定向Mypy、diff --check通过，全仓仍仅上述既有5个Ruff和3个Mypy问题。
本次仅完善本节点兼容说明，不新立实验阶段，不记录为H200推理成功。

## 0.154 2026-10-12 三套英语10dB带噪Top5/Top7对照入口交付（待H200）

用户要求对三套英语带噪音频生成同格式六份交付。按新父目录解释其路径：
`/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/<en_chile|en_colombia|en_mexico>/en/wav/`。
用户同时复述的无日期旧路径用作干净参考，不用于本次音频推理。目录名10dB是用户
提供的数据说明，当前不测信噪比，不宣称已核验加噪算法或逐条音频内容对应。

新增configs/english_3000_noise10db.workzone.json，沿用已冻结773目标+2227补词的
outputs/english_3000_keywords_v1及480h Head；干净转写和目标JSON均从原目录读取。
原构表配置保持原字节不变，不重新构表，不采纳带噪目录中的新词表/转写。
新结果目录为outputs/english_3000_retrieval_480h_noise10db_v1，干净结果保持不动。

英语评测入口新增audio_root+comparison_run_config成对配置，绑定0.153已完成的
outputs/english_3000_retrieval_480h_v1/run_config.json与report.json，并按原sha256清单
核验这两文件。允许输出目录与音频路径改变；比较其他配置、Head/词表及全部非音频
输入SHA、样本ID/参考文本/语言、目标与分母、检索门控参数，以及底层检索代码SHA。
英文运行封装自身新增本次路径/校验逻辑，允许与旧封装SHA不同；底层检索算法不变。
旧音频SHA从已封存的干净run_config读取作比较，不要求再次读取旧音频字节。

预检要求新音频与旧转写stem严格一一对应；每组统计changed_files和identical_files。
某组全部字节与干净版相同则拒绝；部分相同明确报告，不能据changed_files>0宣称
所有样本都成功加噪。只验证文件字节变化，报告snr_verified=false。
仍使用Top5直接复现、Top7精确重放，完整3k词表、原参数、原输出结构；不调阈值，
不训练、不调用最终Qwen转写。本地只用合成fixture，不报告真实带噪召回结果。

### 工区命令与回传

分支codex/g2p-coverage-scan，交付SHA见回复；容器外拉代码：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内运行，4替换为空闲物理GPU。预检通过才启动推理：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
C=configs/english_3000_noise10db.workzone.json
python -B scripts/run_english_3000.py audit --config "$C" &&
python -B scripts/run_english_3000.py run --config "$C" --gpu 4
```

无需build；带噪配置执行build会明确拒绝。预检不创建结果目录、不加载模型。
中断后保持代码、配置、词表、音频及基线身份不变，恢复同一次带噪运行：

```bash
python -B scripts/run_english_3000.py run \
  --config configs/english_3000_noise10db.workzone.json --gpu 4 --resume
```

新实验拒绝已有输出，不能用--resume复用干净版或其他加噪版结果。成功后验证：

```bash
(cd outputs/english_3000_retrieval_480h_noise10db_v1 && sha256sum -c sha256.txt)
cat outputs/english_3000_retrieval_480h_noise10db_v1/report.json
```

返回新目录report.json和sha256.txt即可。delivery/仍为en_chile_top5/top7.json、
en_colombia_top5/top7.json、en_mexico_top5/top7.json，共六份（各预计2,000个stem），
格式stem→[{word,phoneme}]，空召回[]。回传后按0.153同目标分母比较召回、精确率及
负样本误触发，不把本次本地检查当作H200结果。

定向英语测试34通过；全量pytest 499通过、23跳过。新增测试覆盖带噪真实路径使用、
重新执行检索而非复用旧输出、六文件/重放/空结果既有行为、原产物保护、重复恢复、
音频变更拒绝恢复、缺失/多余/全同音频、基线篡改/缺失、换Head/参数和旧输出目录。
修改文件Ruff、定向Mypy、git diff --check通过。全仓Ruff仍有既有5个E501（用户PPT
脚本2、scan_g2p_coverage3），MYPYPATH=src mypy仍有训练模块3个既有unused-ignore，
未修改无关文件。未跟踪用户文件保持不动。

## 0.153 2026-10-12 三套英语3000词Top5/Top7完成（H200回传）

用户回传`outputs/english_3000_retrieval_480h_v1/report.json`内容，状态completed，
三套各2,000条，共6,000条音频；3,000词，六份delivery文件。Head SHA
320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265与固定480h
Head一致；词表SHA 6e2a7016d16bf86cb1f31f3254213e58f5becd4810cc02a166fd604d1cd9b655
与0.152构表清单一致。top5_reproduced/top7_exact_replay=true，parameters_tuned=false，
qwen_decoder_used=false。未测最终WER或CTC PER，不凭目录名断言说话人口音。

本地对回传JSON重新计算每组召回/精确率/未出现比例及负样本误触发率，全部与报告
一致；核对样本总数和六文件命名。回传为粘贴文本，未收到本轮推理sha256清单及六份
JSON原文件，因此不能宣称本地已独立核验交付文件字节。粘贴附件SHA为
37f59421dc993719d0c64f057582c518be035e539e5870799db988dc936508fd；不作工区report文件SHA。

### 按各数据集声明目标计算的最终门控后召回

计量单位为“不同目标词×音频”，同一音频重复说同词不重复计数；合并用命中数之和
除以分母之和，不平均三组百分比。

| 数据集 | 目标词-音频对 | Top5命中 / 召回 | Top7命中 / 召回 |
|---|---:|---:|---:|
| en_chile | 3,292 | 2,285 / 69.4107% | 2,597 / 78.8882% |
| en_colombia | 3,396 | 2,343 / 68.9929% | 2,636 / 77.6207% |
| en_mexico | 3,358 | 2,246 / 66.8851% | 2,540 / 75.6403% |
| 合计 | 10,046 | 6,874 / 68.4252% | 7,773 / 77.3741% |

Top7比Top5提高8.9488个百分点，多命中899个目标词-音频对。
门控前Top5=6992/10046=69.5998%，Top7=7953/10046=79.1658%；门控分别损失
1.1746/1.7918个百分点。剩余漏召回不能只归因于阈值，CTC/候选筛选/排名仍未拆解。

### 全词表输出质量与限制

| 项目 | Top5 | Top7 |
|---|---:|---:|
| 最终输出词-音频对 | 23,876 | 28,417 |
| 参考中出现的输出 | 6,874 | 7,774 |
| 整表输出精确率 | 28.7904% | 27.3569% |
| 输出未出现在参考中的数量 / 占比 | 17,002 / 71.2096% | 20,643 / 72.6431% |
| 每条音频平均输出 | 3.9793 | 4.7362 |
| 无整表热词样本发生输出 | 931/974=95.5852% | 931/974=95.5852% |

上述未出现与负样本均按参考文字匹配定义，尚未音频听审，不能等同最终ASR幻觉率。
整表expected为10,050，比各组声明目标总数多4（Chile多3、Mexico多1）；完整并集允许
其他组的目标出现在该组参考中，并非补词泄入。Top7整表多1个真阳性，故不得混用
7774/10046。target_precision只计算落在该组目标列表内的预测，不能替代整表精确率。

目前可确认运行和汇总自洽；召回之外，非目标输出及负样本误触发是下一步重点。
先固定此次交付，不据此单独调阈值后宣称同条件提升。需要进一步诊断时，从现有
逐样本结果检查无目标样本、误召回词长度和具体音素/排名，区分参考匹配问题与检测
问题；再检查context prompting的最终效果。本轮补词使用测试转写排除，词表3k、
数据域不同，不与历史ES/PT 4k召回直接归因比较。

交付目录`outputs/english_3000_retrieval_480h_v1/delivery/`包含
en_chile_top5.json、en_chile_top7.json、en_colombia_top5.json、en_colombia_top7.json、
en_mexico_top5.json、en_mexico_top7.json；结构沿用stem→[{word,phoneme}]，空召回[]。
交下游前工区可执行：

```bash
(cd outputs/english_3000_retrieval_480h_v1 && sha256sum -c sha256.txt)
```

无需重新构表或推理；本节仅结果文档更新，分支codex/g2p-coverage-scan。

## 0.152 2026-10-12 英语3000词表构建完成（H200回传，推理待执行）

用户回传`outputs/english_3000_keywords_v1/report.json`全文及sha256清单。
构表入口交付提交为c73c5ed5351c6268f78ccb2441cd72850d526cb3；本节仅记录结果，
不修改源码、配置或已有产物。H200实际运行的git SHA尚未在构表报告中提供。

| 项目 | 回传结果 |
|---|---:|
| status / tables_created | completed / true |
| 三套规范化目标并集 | 773 |
| 训练词典补充词 | 2,227 |
| 最终词表 | 3,000 |
| 三套目标数（Chile / Colombia / Mexico） | 412 / 470 / 445 |
| 三套转写条数 | 各2,000，共6,000 |
| 全部目标保留 / OOV | true / 0 |
| 入选补词与三套转写重合 | 0 |
| 可用补词候选 / 因转写出现而排除的候选 | 38,784 / 2,869 |
| 训练记录 / 训练唯一词 | 345,820 / 43,808 |
| 保留同音组 | 17 |

773+2227=3000；三个目标列表有交集，因此不能把412+470+445当成并集数量。
inputs中训练清单、词典、vocab的SHA与0.151固定配置一致；本地重新计算配置SHA
01278638732702cb7714eec418fa6e16fcc4a829efa46f57c4a57425d685343f及vocab SHA，
均与回传一致。Colombia两份输入SHA也与0.151先前实际读取的样例一致。

用户提供的sha256清单记录（词表文件未回传，不声称已本地重算）：
- keyword_bias_phoneme.json：
  6e2a7016d16bf86cb1f31f3254213e58f5becd4810cc02a166fd604d1cd9b655
- report.json：
  e1bf0650e2e66d3daaab830cc2f0aead62273d83570121e00d289835c875e294

结论限于构表报告的一致性：补词在规范化参考转写中零出现，不代表排除自然近音，
也不证明目标发音已经声学校验。未读取音频或模型，尚无召回率及六份推理交付件。
按转写排除补词的3000词条件与历史4000词条件不同，不能直接当成跨语言能力对照。

下一步无需重建词表或为本节文档更新拉代码。容器内：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_english_3000.py audit &&
python -B scripts/run_english_3000.py run --gpu 4
```

4换成空闲物理GPU。audit核验报告/词表SHA、目标与标签、全部构表输入、固定480h
Head，以及三套音频与转写的一一对应和音频SHA；通过才继续推理。输出为
`outputs/english_3000_retrieval_480h_v1`，delivery下三套各Top5/Top7，共六份JSON。
相同输入和代码中断恢复仍用`run --gpu 4 --resume`，不得改变词表或覆盖旧目录。
返回该目录report.json和sha256.txt后再记录实测召回，验证命令沿用0.151。

## 0.151 2026-10-12 三套英语3000词并集与Top5/Top7入口交付（待H200）

用户新增en_chile/en_colombia/en_mexico三套英语数据，目录均为
`/host_home/star/两波验证集-可直接推理/<数据集>/en/`，包含
`en_keyword_bias_phoneme.json`、`transcripts.txt`、`wav/`。
本阶段准备工具，不训练、不在本地加载完整模型、不读取或修改既有outputs。

### 已核实附件与构表规则

用户提供的一个样例JSON使用keyword_sets.all_test，470个目标；2,000条转写。
本地真实附件解析通过，目标缺失音素/OOV均0，不将这个单例当成三套全量验证。
JSON SHA：050cb9ab01eaab22a583908ad0ec9cb3765b80fd7f27b38640c45fb0afc9a707；
transcripts SHA：7680f3d4de15b9d61b71b4cea6a8fb54a7f2dfce12d1a3c1b1ff170dee4deccb。
附件不提交；未知三套并集数量，不硬编码每组样本数，也不凭目录名认定说话人口音。

新增scripts/run_english_3000.py、configs/english_3000.workzone.json及独立英语构表/评测模块。
先读三份all_test，按NFKC/casefold/标点分词规范取并集；同规范词面同token标签只保留
一个检索条目、记录大小写等alias；不同标签冲突直接报错，不猜发音或丢目标。
保留用户目标的词面及原发音，包含短词、多词短语和西语词面，不重新跑G2P。

补词来自既有英语train全文实际出现的词和同批英语MFA词典，身份固定为：
- outputs/en_us_swift_temporal2x_v1/full_ctc_train.jsonl，SHA
  6882074371edfdc94a5ca66ff4ba487c0f965c5f7527b0026740af9b952f9dc8；
- outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/
  swift_us_english_english_us_mfa.v1.dict，SHA
  2dd3c3e045eabaeac4666a4a99cf472bf7a4ef816adf35943a072bd2262c23e2。

逐行要求train和英语标签；补词仅取至少2个ASCII字母的单词，要求已有唯一MFA发音、
当前音素表无OOV。按三套转写合并后的完整token集合排除补词（大小写、Unicode、标点
统一；不做子串或词干猜测）。目标始终优先保留。固定seed哈希排序补至恰好3,000个
规范唯一条目；不足则报告缺口并停止，不放宽排除条件。保留自然同音/近音，不做声学排除。
选择记录保存来源/训练出现次数，排除记录保存原因；生成目标分组、报告和SHA256。

这是用户明确要求的“按测试转写排除补词”受控实验，报告标记
table_filler_exclusion_used_transcripts=true。不能宣称盲测词表，不把精度变化全部
归因于模型；参考转写中的未出现也不是音频听审结论。在线候选选择仍只用音频。

### 评测与产物约定

复用0.147/0.150的新480h epoch25 Head，SHA
320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265；
vocab SHA 0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4。
仅扩展external keyword loader的英语语言/ID映射；原ES/PT行为不变。
沿用完整音频Anchor检索、64候选、threshold .75、posterior weight/min .25/.5、
max edit .35、margin0、anchor2/3/4、每词24、offset1、radius2、minimum phonemes1。
推理设language=English，单次保留64候选，复现Top5并精确重放Top7；不调用Qwen文本生成。

新目录outputs/english_3000_keywords_v1与outputs/english_3000_retrieval_480h_v1。
构表拒绝已有目录；评测恢复绑定源码SHA/git、输入配置、全部三份目标/转写、训练词典、
词表、Head和音频字节。不能用换表/换数据后的--resume复用旧结果。按样本落盘，已完成
相同运行恢复时核验文件，不重新加载模型。若要另开实验，复制配置到新文件并改新输出
目录，保留旧配置及旧outputs；不得删目录绕过校验。

预检要求各组音频stem与转写严格一一匹配，无缺失、重复或多余。各组独立导出，允许
不同数据集具有相同stem。delivery/下六文件：
en_chile_top5.json、en_chile_top7.json、en_colombia_top5.json、en_colombia_top7.json、
en_mexico_top5.json、en_mexico_top7.json。每个stem→[{word,phoneme}]，无召回保留[]。
目标召回按各组声明的目标词计数；full_3000_view包含整张并集表（补词在参考中零出现）。
不再给新报告写full_4000_view或wave目标字段；旧wave/SD导出默认格式保持不变。

### 工区运行

分支codex/g2p-coverage-scan，推送SHA见交付消息。容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，第一步仅CPU读取文本/词典，不需要GPU、模型下载或MFA运行：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_english_3000.py build
```

要求status=completed、total=3000、all_targets_retained=true、
selected_filler_transcript_overlap=0、oov_count=0。三套真实并集/补词数由这一步报告。
构表成功再预检和推理，4换成空闲物理GPU，映射为cuda:0：

```bash
python -B scripts/run_english_3000.py audit &&
python -B scripts/run_english_3000.py run --gpu 4
```

audit不加载模型、不创建输出；run自带同等预检。中断恢复：

```bash
python -B scripts/run_english_3000.py run --gpu 4 --resume
```

验证及回传（无需传音频、模型或sample_shards）：

```bash
(cd outputs/english_3000_keywords_v1 && sha256sum -c sha256.txt)
(cd outputs/english_3000_retrieval_480h_v1 && sha256sum -c sha256.txt)
cat outputs/english_3000_keywords_v1/report.json
cat outputs/english_3000_retrieval_480h_v1/report.json
```

回传两份report.json和sha256.txt；交下游retrieval目录delivery/六文件。
收到H200真实结果后再单独写结果型HANDOFF，不将本地mock当召回率证据。

本地全量pytest：489通过、23跳过；定向覆盖构表/原wave/SD/外部检索/重放。
24项新增测试覆盖真实容量3,000、三套转写排除、目标alias/标签冲突/OOV、来源身份、
六份交付、Top5/Top7差异、同stem隔离、无结果[]、中断恢复及变更输入拒绝。
修改文件Ruff和定向Mypy通过，git diff --check通过。全仓检查仍有既有5个Ruff E501
（scan_g2p_coverage3处、用户PPT脚本2处）和训练模块3个Mypy unused-ignore，未修改。

## 0.150 2026-10-09 四wave 10dB带噪召回完成（H200回传）

用户回传report状态completed，800条/8组各100条，16份delivery，
outputs/wave_4000_retrieval_480h_old_only_noise10db_v1。
Head SHA320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265
与0.147干净版一致；top5_reproduced/top7_exact_replay=true，parameters_tuned=false，
qwen_decoder_used=false。audio_root为0.149用户指定路径，8组均changed_files100、
identical_files0。报告snr_verified=false，不将目录名10dB当成实测SNR证明。

### 同目标分母的最终门控后召回

按语言汇总命中数/目标词-音频对数（不是四wave百分比平均），不混入补充词召回。
分母ES724/PT736与0.147相同；代码检查各组hits/ref等于gated_target_recall，
整表true_positive+absent等于selected。基线用0.147同Head排除近音表的干净结果。

|语言/K|干净命中|带噪命中|干净召回|带噪召回|变化百分点|
|---|---:|---:|---:|---:|---:|
|ES5|593/724|584/724|81.9061%|80.6630%|-1.2431|
|ES7|651/724|632/724|89.9171%|87.2928%|-2.6243|
|PT5|483/736|463/736|65.6250%|62.9076%|-2.7174|
|PT7|539/736|522/736|73.2337%|70.9239%|-2.3098|

|wave|ES5|ES7|PT5|PT7|
|---|---:|---:|---:|---:|
|1|82.81%|89.06%|55.08%|64.17%|
|2|81.25%|89.06%|48.66%|57.75%|
|3|85.53%|92.45%|76.74%|80.81%|
|4|73.48%|79.01%|72.11%|81.58%|

总量下降1.24–2.72个百分点，不是所有组一致下降：ES wave1 Top5及wave2两K略升；
ES wave4下降最明显（Top5 -6.63pp、Top7 -7.73pp）。葡语全部组两K下降，
wave1/2绝对召回仍最低。不能从一次噪声版本外推普遍噪声鲁棒性或统计显著性。
raw目标命中ES5/7=584/633，PT5/7=465/528；门控仅再损失0/1、2/6个目标对，
本次剩余漏召回多数已不在raw TopK内，不能仅靠放宽门控解决，但未分解上游各原因。

全部输出中未匹配参考文本的比例：ES5 958/1866=51.34%，ES7 1502/2476=60.66%；
PT5 1248/1898=65.75%，PT7 1716/2431=70.59%。target_precision并非全部候选精度。
Top7较Top5新增ES48/PT59目标对，但仍需下游实际转写评估干扰与收益；无WER/PER实测。

交下游该新目录delivery/下16个waveX_lang_topK.json，不与旧干净版同名文件混用。
下一步使用这16文件做带噪音频context prompting，比较无提示/Top5/Top7；当前无需
重跑检索或改训练/门控。本轮附件SHA
5678da4430e6698fd446fe9db03808bc1e1d5ebffc84bdc535da025f1d6bad5c。
收到的是report文本，未收到sha256.txt或16文件本体，不声称本地验证H200输出文件
字节或独立核验词表SHA；词表一致性依赖0.149入口校验及completed报告。
本轮仅阶段结果文档，算术/内部一致性和git diff --check通过，不修改outputs。

## 0.149 2026-10-08 四wave带噪10dB、排除近音旧表补齐评测入口（待H200）

用户提供新音频根目录：
/host_home/star/两波验证集-可直接推理/2026-09-29-带噪10dB。
新增配置configs/wave_retrieval_480h_old_only_noise10db.workzone.json，复用
outputs/wave_4000_keywords_old_only_v1（ES316/PT295目标，各4000词）、新480h
best Head以及0.147全部检索/门控参数。只替换音频，不重建词表、不训练、不跑Qwen文本生成。
对照身份绑定outputs/wave_4000_retrieval_480h_old_only_v1/run_config.json。

wave_retrieval新增受控audio_root及comparison_change=audio。参考转写仍读取仓库
原wave1..4/es,pt/transcripts.txt，并校验原inventory SHA；要求带噪wav/中的
stem与原转写一一对应、每组100条，不自动剥离文件名后缀、不静默漏样本。
只允许音频路径/字节变化，要求Head字节、完整词表字节、目标ID、参考文本、
source/语言及全部门控完全一致。逐组记录changed_files/identical_files；若一组
全部与旧音频字节相同则停止，防止指回干净数据。10dB为用户提供的条件名称，
未测实际SNR；字节变化不证明SNR准确或语音内容一致。

新输出outputs/wave_4000_retrieval_480h_old_only_noise10db_v1。输出16份
wave1_es_top5.json等文件，格式stem→[{word,phoneme}]，无召回保留[]。
旧wave/SD产物不修改。--resume只接受相同代码/输入/配置/音频SHA，不能恢复旧干净运行。

分支codex/g2p-coverage-scan，提交SHA见交付消息。容器外更新：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内（4替换为空闲物理GPU），预检通过才开始推理：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
C=configs/wave_retrieval_480h_old_only_noise10db.workzone.json
python -B scripts/run_wave_keyword_retrieval.py --config "$C" --audit-only &&
python -B scripts/run_wave_keyword_retrieval.py --config "$C" --gpu 4
```

audit_only不创建目录/加载模型，报告8组800条、每语种4000词和音频变化计数。
文件名不匹配时返回错误，不猜映射；先由用户确认实际命名再适配。中断时原run命令
追加--resume；已完成同配置重放会验证输出，不重新推理。不得删除输出绕过校验。

验证与回传：

```bash
R=outputs/wave_4000_retrieval_480h_old_only_noise10db_v1
(cd "$R" && sha256sum -c sha256.txt)
cat "$R/report.json"
```

交下游$R/delivery/16文件；回传report.json及sha256.txt即可，不需音频、权重或
rank详情。本地尚无带噪实测，不声称召回结果。收到H200报告后，与0.147同目标
分母的干净音频召回逐组比较，单独记结果型HANDOFF。

验证：定向31项通过（含带噪路径实用、原转写、16导出、恢复、拒绝相同音频/错误
文件名/换Head/词表/门控/目标）；全量465通过/23跳过。修改文件Ruff、定向Mypy、
git diff --check通过。全仓仍仅既有5个Ruff E501及训练模块3个Mypy unused-ignore，
未修改用户文件或无关模块。本地模型均mock，不代替H200实测。

## 0.148 2026-09-28 SD362扩展4000词、MLS/SD双来源评测入口交付（待H200）

用户明确选择后来的pt_keyword_bias_phoneme_sd.json之all_keywords 362词作为
唯一目标集合，不加入MLS hard_k266，也不用165词in/out版本。保留两套测试音频：
MLS871条、SD/Delivery2071条，总2942条；同一葡语4000表用于两来源。
原始SD表保持不变，既有加载器只剥离词面首尾空格，兼容已核实的Dra. /Dra.映射。
全部目标音素保持不变；标准化重复、OOV、缺失音素、目标数量变化均阻塞。

新增scripts/run_sd_4000.py的build/audit/run三个阶段，配置
configs/sd_4000.workzone.json。build只读SD目标和旧葡语capacity表，不读音频、
转写或预测；以固定seed sd-4000-20260928-v1从旧表确定性补足3638词。
旧表沿用0.146来源，SHA ac8f50fa6f6f14870d8ca97b7028a570f88ace81c31062c41da60228b2a5d4d3。
目标优先；旧表同规范词面发音冲突全部排除；检查语言别名pt/pt-BR、IPA及token一致性。
不使用提供的近音文件作为补词来源，但不声称旧词与这362词不存在天然近音关系。
不把上一轮wave近音清单当作SD362的声学近音定义，也不按本次转写反向筛补词。

固定新480h epoch25 Head，SHA
320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265。
复用已有完整音频Anchor检索和Top5复现/Top7精确重放逻辑，不改模型或门控：
threshold .75，posterior weight/min .25/.5，max edit .35，margin0，shortlist64，
anchor2/3/4、24个/entry、offset1、radius2、minimum phonemes1。保存全部64候选。
报告分开给362目标召回和full_4000_view，并给参考文本未匹配输出词的比例。
共享指标字段declared_wave_target_count在本入口表示SD目标数，而非wave集合。
MLS的目标分母变为SD362，且Head也不同，不能直接与历史MLS266+旧Head的召回作
词表单因素比较。当前未运行H200，无新召回率、WER或PER结论。

### 工区操作

交付分支codex/g2p-coverage-scan，提交SHA以交付消息为准。容器外更新：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录；旧/home_91挂载须可见（配置沿用0.76/0.81确证的两来源路径）：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_sd_4000.py build
python -B scripts/run_sd_4000.py audit
python -B scripts/run_sd_4000.py run --gpu 4
```

4替换为空闲物理GPU，映射为逻辑cuda:0；audit不加载模型、不创建运行目录。
build预期mandatory362、old_table_fillers3638、total4000、OOV0；audit预期
MLS871/Delivery2071，并显示各来源362目标词的文本出现对数，模型加载前核实。
任一步失败即停，不绕过身份检查。完整推理仅在H200由用户执行。

新输出目录（不得删除旧实验或覆盖）：
outputs/sd_4000_keywords_v1 与 outputs/sd_4000_retrieval_480h_v1。
构表拒绝已有目录；运行中断后用同一命令加--resume，逐样本恢复，绑定配置、代码
commit、词表/目标/Head/转写/音频SHA并校验结果行摘要。换词表或Head不能resume。
已完成的相同运行不会再次加载模型。新实验修改配置中的新输出路径，不复用旧目录。

下游四文件位于运行目录delivery/：
mls_portuguese_top5.json、mls_portuguese_top7.json、
delivery_20260706_ptbr_top5.json、delivery_20260706_ptbr_top7.json。
每个音频stem映射到[{word,phoneme}]，无召回保留[]；两来源分开，跨来源同名不丢失。

```bash
(cd outputs/sd_4000_keywords_v1 && sha256sum -c sha256.txt)
(cd outputs/sd_4000_retrieval_480h_v1 && sha256sum -c sha256.txt)
cat outputs/sd_4000_retrieval_480h_v1/report.json
```

回传构表摘要和运行目录report.json/sha256.txt即可；下游取delivery四文件，不传
音频、权重、样本分片或完整rank详情。得到H200结果后再作独立结果型HANDOFF提交。

本地验证：定向33项通过；全量458通过/23跳过；新增模块/CLI/tests Ruff通过，
新增模块Mypy(--follow-imports=silent)通过，git diff --check通过。全仓Ruff仍有
既有5个E501（scan_g2p_coverage3处、用户PPT脚本2处）；全仓Mypy仍有既有训练模块
3个unused-ignore，与本次无关，不修改这些文件。模型调用为mock，仅证明工具链。

## 0.147 2026-09-28 排除提供近音词补词后的四wave召回结果完成（H200回传）

用户附件同时展示run_config摘要、report状态和完整report：tables为
outputs/wave_4000_keywords_old_only_v1，Head为multilingual_480h_run_v1/head/
ctc_head_best.pt；status=completed、800条、16份delivery，top5_reproduced=true、
top7_exact_replay=true、parameters_tuned=false、qwen_decoder_used=false。
输出outputs/wave_4000_retrieval_480h_old_only_v1，本轮确实已完成，无需重跑。
checkpoint SHA与0.145混合表实验完全一致：
320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265。

### 同一新Head、同一目标分母、不同补词表

本地逐组核对sample_count100、目标对分母与0.145一致、hits/denominator等于
gated_target_recall、整表true-positive+absent=selected。以下语言汇总为命中总数
除以目标对总数，不是各wave百分比平均；指标是原wave目标词召回。

| 语言/K | 含近音补词命中/分母 | 排除后命中/分母 | 含近音召回 | 排除后召回 | 提升百分点 |
|---|---:|---:|---:|---:|---:|
| ES Top5 | 227/724 | 593/724 | 31.3536% | 81.9061% | +50.5525 |
| ES Top7 | 329/724 | 651/724 | 45.4420% | 89.9171% | +44.4751 |
| PT Top5 | 214/736 | 483/736 | 29.0761% | 65.6250% | +36.5489 |
| PT Top7 | 309/736 | 539/736 | 41.9837% | 73.2337% | +31.2500 |

| wave | ES Top5 | ES Top7 | PT Top5 | PT Top7 |
|---|---:|---:|---:|---:|
| 1 | 81.77% | 91.67% | 58.82% | 65.78% |
| 2 | 79.69% | 88.02% | 50.80% | 59.36% |
| 3 | 86.79% | 93.71% | 77.91% | 83.14% |
| 4 | 80.11% | 86.74% | 75.79% | 85.26% |

全部8组两K目标召回均比0.145提高，PT wave1/2仍弱于wave3/4。新表Top7相比
Top5多召回ES58个目标对、PT56个目标对，分别+8.0110、+7.6087个百分点。

### 干扰与解释边界

| 语言/K | 含近音absent/selected | 排除后absent/selected | 比例变化 |
|---|---:|---:|---:|
| ES Top5 | 1304/2000 | 949/1880 | 65.20%→50.48% |
| ES Top7 | 1854/2799 | 1503/2502 | 66.24%→60.07% |
| PT Top5 | 1350/2000 | 1241/1912 | 67.50%→64.91% |
| PT Top7 | 1847/2786 | 1723/2460 | 66.30%→70.04% |

absent是规范化参考文本中未匹配到，不是人工听审结论。PT Top7的absent绝对数
减少124，但输出总数减少326，所以比例反而升高，不能声称输出纯度全面改善。
本新表Top5→Top7增加ES622/PT548次选择，其中新增absent ES554/PT482次；
Top7覆盖更高，但新增候选多数未匹配参考，最终提示词收益必须单独测WER。

更关键的是，full_4000_view的参考命中词对因词表改变而减少：ES1872→1081，
PT1941→951。移除的词并非全是假候选，其中也有文本中实际出现的普通词。
目标分母仍ES724/PT736，因此目标召回比较有效；整表召回率分母不同，不能直接
当同一召回任务量化提升。这轮证明词表组成/候选竞争强烈影响目标排名，不能把
全部50.55/36.55点提升都归因于音近混淆；还包含真实普通词竞争减少及替换补词的
效应。需要逐样本shortlist/rank才能分辨具体原因，暂不修改评分或门控。
新表整表负样本ES2/PT5，均有输出，数量少且负样本定义随词表变化，不外推普遍FPR。

### 交付与下一步

16份文件位于outputs/wave_4000_retrieval_480h_old_only_v1/delivery，文件名和
音频stem→[{word,phoneme}]格式沿用0.144。可交下游，标明“新480h Head＋排除提供
近音词补词”，不能与0.145目录下的同名文件混用。当前无需再重跑该实验。
下一步优先固定两套结果，评估无提示/Top5/Top7的最终转写及目标词正确率，关注
PT wave1/2和额外候选的负面效果；新Head/旧Head在排除表上的直接对照尚未进行，
不能借换表收益宣称新Head在此表比旧Head更优。无新训练或自动参数调优。

证据附件SHA256：bdbfa3294a8818beaec25db46198b5a31112c2eb3a8fa96c8c0201f0fc73ba4c。
本地核对用户回传report内部一致性及前轮报告；未收到独立sha256.txt/16文件本体，
不声称本地完成交付文件字节验证。H200可按0.144既有sha256sum -c命令核验。
本轮仅阶段结果文档提交，指标算术核验和git diff --check通过，不修改outputs。

## 0.146 2026-09-28 排除提供近音词的旧表补词版本构建完成（H200回传）

用户运行0.144构表命令返回status=completed、tables_created=true，目录
outputs/wave_4000_keywords_old_only_v1。政策old_only_exclude_supplied，
seed仍wave-4000-20260923-v1，不读取音频/转写/预测，不加载模型。

| 语言 | 总词数 | 保留目标 | 旧表补词 | 排除旧表重合词面 | 可用旧表补词池 | 最终补词重合 |
|---|---:|---:|---:|---:|---:|---:|
| es | 4000 | 316 | 3684 | 168 | 4225 | 0 |
| pt | 4000 | 295 | 3705 | 171 | 4229 | 0 |

两语all_targets_retained=true、OOV=0、neighbor来源0、optional冲突0，旧表身份
校验问题计数均空。ES剩余541个未选可用补词，PT524个，无需扩充来源或回退。
近音文件归一化词面并集ES3451/PT2187；目标中ES14/PT12个也出现在近音清单，
按目标全保留规则保留，不计入补词重合。PT excluded_supplied_neighbor有173条
来源记录，对应171个唯一词面，两个统计单位不同，并非遗漏排除。

旧表语言记录ES4401个es，PT3949个pt加454个pt-BR。ES规范化更新3次，primary
去组合tilde累计288次，与旧表构建一致；不能把来源重复次数当唯一目标数。
保留相同标签序列组ES1/PT3（如entre rios/entre ríos、belem/belém），因此本实验
应称“排除提供近音词的补词对照”，不是所有自然近音或同音都不存在的词表。

本次证据为用户直接粘贴终端摘要，未收到独立report.json/sha256.txt，未在本地
重算H200词表文件SHA。摘要中316+3684=4000、295+3705=4000及候选余量一致。
已有评测入口会在模型加载前核验该目录sha256.txt及与0.145的固定实验输入。

下一步在容器内项目根目录运行，4换为空闲物理GPU，无需重新构表或修改配置：

```bash
python -B scripts/run_wave_keyword_retrieval.py --gpu 4 \
  --config configs/wave_retrieval_480h_old_only.workzone.json
```

固定同一新480h epoch25 Head、四wave各ES/PT100条、Top5/Top7和全部门控参数，
只改变词表。输出outputs/wave_4000_retrieval_480h_old_only_v1，delivery/仍16份
同格式JSON，旧混合表结果保留。中断后同命令追加--resume，身份变化则拒绝恢复。
校验及返回根report.json/sha256.txt按0.144；当前尚无这套新词表的召回实测。
对照0.145：新Head混合表ES Top5/7=31.3536%/45.4420%，PT=29.0761%/41.9837%。
本轮仅结果型文档更新，git diff --check通过，不改代码或outputs。

## 0.145 2026-09-24 新480h Head配原近音混合4000表评测完成（H200回传）

用户回传命令使用configs/wave_retrieval_480h.workzone.json，输出为
outputs/wave_4000_retrieval_480h_v1。此为0.143“仅换Head”实验，仍含ES1842/PT1279
条neighbor来源，不是0.144排除近音词实验。后者构建和评测结果均尚未返回。
status=completed，8组×100=800条，报告列出16份delivery文件；top5_reproduced=true、
top7_exact_replay=true、qwen_decoder_used=false、parameters_tuned=false。
本次报告首次返回新Head SHA：
320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265。
这是H200入口计算的权重身份；本地未获取权重或独立sha256.txt，不声称本地重算。

### 与0.137旧Head同词表目标召回对比

原始证据附件SHA256：
13f5cc729a042154c953800d22692ffc3558fd13e2058a8de265b2dda056e29c。
本地读取0.137原附件和本附件，核对每组各K的目标对分母一致、hits/denominator
与gated_target_recall一致、整表true-positive加absent等于selected。以下均为
原wave目标召回，语言汇总按命中总数/目标对总数，不是四wave百分比简单平均。

| 语言/K | 旧Head命中/目标对 | 新Head命中/目标对 | 旧召回 | 新召回 | 变化百分点 |
|---|---:|---:|---:|---:|---:|
| ES Top5 | 207/724 | 227/724 | 28.5912% | 31.3536% | +2.7624 |
| ES Top7 | 310/724 | 329/724 | 42.8177% | 45.4420% | +2.6243 |
| PT Top5 | 219/736 | 214/736 | 29.7554% | 29.0761% | -0.6793 |
| PT Top7 | 309/736 | 309/736 | 41.9837% | 41.9837% | 0 |

| 组 | 旧Top5→新Top5 | 旧Top7→新Top7 |
|---|---:|---:|
| wave1 ES | 24.48%→27.60% | 38.54%→38.02% |
| wave2 ES | 21.88%→26.56% | 35.42%→38.02% |
| wave3 ES | 40.88%→44.65% | 54.72%→58.49% |
| wave4 ES | 29.28%→28.73% | 44.75%→49.72% |
| wave1 PT | 20.32%→16.04% | 27.27%→24.06% |
| wave2 PT | 14.97%→17.11% | 19.79%→22.46% |
| wave3 PT | 43.60%→44.77% | 61.63%→61.05% |
| wave4 PT | 41.05%→39.47% | 60.53%→61.58% |

整表候选未在参考转写中匹配到的比例：ES Top5 64.90%→65.20%，Top7
66.1654%→66.2379%；PT Top5 65.75%→67.50%，Top7 65.7574%→66.2958%。
新Head PT Top5多35次absent选择；Top7多15次。该口径是规范化转写短语匹配，
不等于人工声学误触发认证；不能用接近1的target_precision表示整表输出可靠。

### 结论与当前动作

西语目标召回改善；葡语Top5略降、Top7持平，部分wave升部分降。此前PT PER约
8.4%相对下降没有转化为这份固定混合词表/门控上的目标召回提高，因此新Head
尚不能在葡语热词任务上替换旧基线。不能凭汇总归因近音竞争、概率校准或门控；
需词表对照及必要的逐样本排名分析。未测最终WER，不推断下游转写收益。

本次16文件在outputs/wave_4000_retrieval_480h_v1/delivery，可作为“新Head＋
含专门近音补词”的独立交付；不得标成排除近音词版本。无需重跑这个已完成实验。
下一步按0.144先CPU构建，保证旧表与提供近音词重合的补词也被排除、目标保留；
若容量足够再用同新Head评测。命令和新目录沿用0.144，两次实验仅词表不同。
本轮只新增结果记录；算术核验及git diff --check通过，不改模型/词表/代码/outputs。

## 0.144 2026-09-24 目标全保留、旧表补词排除提供近音词的4000词对照（待H200构建/评测）

用户最新要求覆盖先前“重合可保留”的临时回答：ES316/PT295目标全部保留，
剩余仅从原旧表选择，并排除旧表中与提供近音词重合的补词。采用
old_only_exclude_supplied，不提供“词数不足时重新加近音词”的自动回退。
此前混合表ES1842、PT1279条neighbor来源；本轮neighbor来源必须0。

### 固定规则和边界

- 原primary四wave并集不变，ES316/PT295；沿用已有ES音素清理、PT原标签和
  标准化词面去重。目标优先级最高，即使目标出现在近音文件也保留。
- 四wave的phonetic_neighbors_phoneme.json中全部neighbors[parent]候选word
  规范化后构成排除集合，不限parent是否属于本轮目标，不依赖phoneme有效或sim值。
  近音文件仅用于词面排除及计数，绝不作为发音/补词来源。
- 旧表路径/SHA、语言别名/vocab一致性、冲突隔离、排序seed仍与原构建相同。
  非目标旧表词面命中排除集合即剔除；剩余按固定哈希顺序选取，不看转写/模型预测。
  成功必须ES316+3684、PT295+3705，均4000唯一词面。
- summary记录excluded_old_supplied_neighbor_surfaces、mandatory重合数及
  selected_optional_supplied_neighbor_overlap（必须0），selection_audit逐旧表
  记录excluded_supplied_neighbor。这不是“音素相似度绝对为零”的认证：未被
  提供词面清单列出的自然近音/同音仍可能存在，不额外改动音素等价规则。
- 先扫描两语，再决定发布；任一语种不足，stdout status=insufficient_capacity、
  tables_created=false、两语容量/缺口汇总，退出1、不创建输出、不发布残缺表。
  不重复凑词、不用近音fallback、不修改旧表。真实容量待H200，不能预先保证够数。

### 构建、评测和导出

交付分支codex/g2p-coverage-scan。新增两份独立配置，旧配置不改，最终commit见回复。
容器外项目目录拉取：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

先在容器内项目目录构建（只用CPU）：

```bash
python -B scripts/build_wave_4000_keywords.py \
  --config configs/wave_4000_keywords_old_only.workzone.json
```

新词表目录outputs/wave_4000_keywords_old_only_v1，已有目录拒绝覆盖，构建不支持
resume。成功应tables_created=true、两语total4000、mandatory316/295、可选重合0、
selected_by_source仅primary和old_table。返回report.json/sha256.txt；若不足，只返回
终端两语容量汇总，先解决缺口，不继续GPU评测。

仅在构建成功后运行；4替换为空闲物理GPU：

```bash
python -B scripts/run_wave_keyword_retrieval.py --gpu 4 \
  --config configs/wave_retrieval_480h_old_only.workzone.json
```

该对照要求上一轮新Head近音表运行已产生
outputs/wave_4000_retrieval_480h_v1/run_config.json。使用同一新480h epoch25 best
权重字节，复核旧运行全部输入、音频/转写、语言/模型/门控/检索参数，仅允许
keyword_bias与代码版本不同。每wave目标词面/发音必须相同。词表位置型hotword_id
允许随排序改变，因此比较目标词面和token序列，按新表ID正确统计，不拿旧索引套新表。
此入口不修改仍在运行的旧实验输出；如旧实验需resume，请先用其原代码完成再更新。

新结果outputs/wave_4000_retrieval_480h_old_only_v1；中断后同命令追加--resume，
校验固定输入/代码/权重身份，仅补缺失样本。--audit-only可选预检，不创建输出。
delivery/仍16份wave1_es_top5.json……wave4_pt_top7.json，每份100个音频stem，
值为[{"word": ..., "phoneme": ...}]，空召回保留[]；新旧目录分开交付避免同名混淆。

```bash
(cd outputs/wave_4000_keywords_old_only_v1 && sha256sum -c sha256.txt)
(cd outputs/wave_4000_retrieval_480h_old_only_v1 && sha256sum -c sha256.txt)
```

评测返回根report.json/sha256.txt，16份delivery文件给下游；对比同新Head的混合表
结果，观察每wave目标Top5/Top7召回和整表干扰，不将换表后的提升归因于Head改进。
本地仅测试，未生成真实H200词表或召回结果。用户返回后再独立提交成果记录。

新增12项测试覆盖补词交集排除/目标例外、非目标parent及无效近音IPA、容量不足
两语汇总且不写文件、4000唯一且确定性、源文件保留/哈希、目标ID重排、16文件格式、
Head/目标发音/门控变化阻断。定向88 passed，全量454 passed/23 skipped；改动范围
Ruff、strict Mypy及git diff --check通过。全仓仍既有5处E501/3处unused-ignore。

## 0.143 2026-09-24 新480h Head复测四wave并导出16份Context Learning文件（待H200执行）

用户授权使用新的CTC Head重跑昨日四wave。新增独立配置
configs/wave_retrieval_480h.workzone.json，复用run_wave_keyword_retrieval.py，
新目录outputs/wave_4000_retrieval_480h_v1，绝不覆盖昨日wave_4000_retrieval_v1。
本地未加载完整Qwen、未执行真实GPU评测；新Head召回结果待用户返回。

### 固定对照与身份

- 仅换multilingual_480h_run_v1/head/ctc_head_best.pt，训练报告best_epoch必须25、
  status=completed、cache_sha256_verified=true、test_set_used=false、best路径
  与配置一致，训练/验证数量须与run_plan一致，vocab SHA不变。
- run_plan SHA固定d146173589b915a862344df12d6d3725ff5779067530a75cfd9d700ac8a32e99。
  尚未收到新权重外部SHA，因此首次在H200对指定best文件计算SHA，写入run_config
  和report；这是本轮首次身份绑定，不声称已有独立可信权重摘要。resume必须保持
  权重、训练report、run_plan、输入及代码身份相同，变化即拒绝。
- 核对昨日outputs/wave_4000_retrieval_v1/run_config.json的全部输入SHA，包括
  两份4000词表、目标列表、800条音频、转写、模型元数据、旧配置和旧Head。
  每组配置除git_commit/ctc_checkpoint外须与昨日相同；包括gate、retrieval、
  audio、language/device/dtype，样本记录和target IDs也必须一致。没有调参数。
- 原三语Head默认配置不修改。仍4wave×2语言×Top5/7=16文件；全音频冻结Encoder
  加新CTC Head与Anchor，完整Qwen仅在H200加载，不执行最终文本decoder。
- 依旧复现D5并精确重放Top7。门控threshold0.75、posterior weight0.25/min0.5、
  edit ratio0.35、shortlist64等沿用0.136；新Head概率校准差异是固定门控实验的一部分。

### 执行和回传

交付分支codex/g2p-coverage-scan，最终远端commit SHA见回复。容器外项目目录：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录，4换成空闲物理GPU编号：

```bash
python -B scripts/run_wave_keyword_retrieval.py --gpu 4 \
  --config configs/wave_retrieval_480h.workzone.json
```

若中断，在同一命令后追加--resume；固定身份校验后只补缺失样本。新实验可用
--output-dir新目录，禁止删除旧产物或把旧Head结果目录传给新Head继续写入。
首次命令自动完成预检；可选--audit-only在CPU检查且不创建输出，不需要先跑一次。

输出delivery/内wave1_es_top5.json、wave1_es_top7.json……wave4_pt_top7.json。
每文件100个音频stem，值为[{"word": ..., "phoneme": ...}]，空召回保留[]，
与昨日下游格式一致。新旧目录必须分别交付，不能因同名文件混淆模型版本。

```bash
(cd outputs/wave_4000_retrieval_480h_v1 && sha256sum -c sha256.txt)
```

用户返回根report.json和sha256.txt两个小文件；delivery/16文件交下游。
收到后与0.137逐wave目标Top5/Top7召回、整表未匹配转写比例对照，不将填充词
命中当目标提升，不将CTC指标改善代替最终WER结论。当前未生成任何H200新结果。

新增8项测试覆盖新Head身份绑定、16文件schema/空列表、恢复及权重变化阻断、
训练epoch/path/plan漂移、音频/词表/门控对照变化、两配置差异范围。定向wave及
旧检索/replay测试31 passed；全量442 passed/23 skipped。改动范围Ruff和strict
Mypy、git diff --check通过；全仓仍既有5处E501/3处unused-ignore，无新增问题。

## 0.142 2026-09-24 480h Head在固定密度匹配葡语子集上改善（H200回传）

用户返回diagnose_frozen_ctc.py的JSON：status=pass、test_set_used=false、
validation_samples=2631。两个checkpoint路径分别为原150h三语基线和
multilingual_480h_run_v1/head/ctc_head_best.pt；保存epoch分别24、25，结构均
temporal_upsample/h512/k5/dropout0.1/time2。此处当前评测指标取validation，
checkpoint_epoch_metrics是保存时旧三语验证集指标，不能误读为本子集成绩。

两模型均2631条、131492参考音素、375838有效输入帧。旧Head PER=8.207343%，
精确复现0.98同子集结果；新Head PER=7.519089%。少905个音素错误，绝对下降
0.688255个百分点、相对错误率降低8.385841%。本地已验证S+D+I、hyp=ref-D+I、
PER=errors/ref、各压力桶计数求和与总体一致。

| 指标 | 原150h三语Head | 新480h三语Head | 减少错误数 |
|---|---:|---:|---:|
| 总PER / errors | 8.207343% / 10792 | 7.519089% / 9887 | 905 |
| substitution | 3.719618% / 4891 | 3.364463% / 4424 | 467 |
| deletion | 2.849603% / 3747 | 2.582667% / 3396 | 351 |
| insertion | 1.638122% / 2154 | 1.571959% / 2067 | 87 |

替换、删除、插入分别贡献净错误减少的51.60%、38.78%、9.61%，这是算术分解，
不是数据/标签/架构各因素的因果贡献。长度比0.987885→0.989893，validation loss
0.340272→0.298787。并非通过增加插入抵消删除，也不是仅高密度样本改善。
minimum CTC ratio<=0.50桶2571条：8.124907%→7.450712%；(0.50,0.75]桶60条：
10.688106%→9.576732%。这些是minimum CTC ratio桶，不是ref/frame三等分，
也不等于original_ready/recovery划分；高压力桶仅60条，不外推全部困难样本。

历史旧葡语专用Head同子集7.617954%，本轮新三语低约0.098865个百分点，但旧
专用Head未在本次再次执行，且训练数据/域/预算不同，不宣称统计显著或多语架构
优于单语。仍有a/ɐ、e/i、e/ɛ等参考/预测混淆，仅凭排行不能判断真实发音或错标。
例如新Head的i删除398次高于旧382次，不能把总体改善说成每个音素都改善。

### 当前结论与下一步

结合0.141旧集葡语9.662%→8.856%，新集8.207%→7.519%，两个固定视图都约
8.3%相对错误率下降，扩量方案的CTC指标收益在这两个视图上方向一致。它们并非
被证明完全独立的测试集，未提供逐样本配对置信区间；保留标签准确性未认证边界。
不能归因为“所有剩余错误都是模型错误”，也不能将PER下降等同最终WER收益。

下一阶段建议固定0.135两份4000词表、0.136检索与门控参数和全部wave音频，
仅替换为本轮最佳Head，比较0.137基线的目标Top5/Top7召回及整表干扰，再向下游
传新旧结果评估最终转写。该热词实验尚未执行，不在此轮修改门控/词表或继续训练。

证据为用户粘贴JSON附件，SHA256：
94d53f11aef7bcc3a14ca7b04f6a1e98e3a647a1716db880eac1c0ab1cb30cc2。
本次未收到checkpoints.sha256，不声称已验证新权重字节身份；报告也未包含原始
manifest/cache指纹，当前同集证据是固定命令上下文、聚合计数及旧结果复现。
本轮仅结果型文档更新，指标算术检查及git diff --check通过，不改已有outputs。

## 0.141 2026-09-24 三语480h训练早停完成，最佳epoch25（H200回传）

用户返回formal_completed，resumed_from_epoch=5、epochs_requested=30、
epochs_completed=26、early_stopped=true、best_epoch=25。选择ctc_head_best.pt，
不是epoch26的ctc_head_latest.pt。run_plan SHA仍为
d146173589b915a862344df12d6d3725ff5779067530a75cfd9d700ac8a32e99。
Head/采样/训练与验证数量均延续0.139，cache_sha256_verified=true、test_set_used=false。

| 语言 | 旧150h基线PER | 480h最佳PER | 降低百分点 | 相对错误率降低 |
|---|---:|---:|---:|---:|
| en | 4.736% | 3.9111% | 0.8249 | 17.42% |
| es | 3.910% | 3.7389% | 0.1711 | 4.38% |
| pt | 9.662% | 8.8560% | 0.8060 | 8.34% |

差值基于既有四舍五入基线。best分语言错误/reference为en3039/77701、
es4857/129906、pt14464/163324；样本en2808/es2631/pt2662。Macro PER=5.5020%，
overall PER=6.0281%，validation loss=0.233729。epoch26的pt PER=8.8793%，
Macro PER=5.5217%，应使用epoch25作为本轮选定候选。葡语比旧专用9.032%也低约
0.176个百分点，但数据量/域/预算不同，不能据此证明多语结构更优。

这次三语均改善，葡语较基线约8.34%相对错误率下降；仍明显弱于英西。不从PER
推断热词召回或最终WER必然改善，也不能把扩量实验全部收益归因于唯一数据多样性。
总training_seconds=15544.30（累计约4.32小时，含最初5轮；不含特征提取），
最终LR=3.75e-5。stale=6达到patience6，早停min_delta为Macro PER的0.001；
最佳出现在25而26早停不矛盾：微小改善可保存best却不重置早停计数。不建议为凑
30轮反复执行formal。无完整metrics.jsonl，不能复原全部epoch曲线或断言无提升空间。

### 验证集身份与下一步

本报告仍是在旧固定8101条验证集上（葡语2662条），并非新选的西语密度匹配
葡语2631条，也不是16小时全池。此前0.98新子集原三语PER=8.207343%、旧葡语
7.617954%；本次480h Head在新子集尚无实测值，禁止将8.8560%与8.207343%跨集
直接判断变差。新子集也未获得标签准确性认证。

下一步复用已完成evaluation_oja96zhl/validation_cache，在同一个新子集同时
评估旧三语基线及新480h最佳Head。已有diagnose_frozen_ctc.py接受多个checkpoint，
核验所有cache shard SHA，输出PER/S/D/I和混淆；无需提取Qwen特征或重新训练。
容器内项目根目录执行以下命令，4替换为空闲物理GPU：

```bash
R=outputs/pt_es_density_match_v1
H=outputs/multilingual_480h_run_v1/head
B=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1
O=$(mktemp -d outputs/pt_480h_matched_eval_XXXXXXXX)
sha256sum "$H/ctc_head_best.pt" "$B/ctc_head_best.pt" > "$O/checkpoints.sha256"
CUDA_VISIBLE_DEVICES=4 python -B scripts/diagnose_frozen_ctc.py \
  --validation-cache "$R/evaluation_oja96zhl/validation_cache" \
  --validation-manifest "$R/full_ctc_validation.jsonl" \
  --checkpoint "$B/ctc_head_best.pt" --checkpoint "$H/ctc_head_best.pt" \
  --output "$O/report.json"
```

每次独立新目录，不覆盖旧outputs。执行后返回该目录report.json和checkpoints.sha256。
本次只核对用户训练回传，尚未运行该新子集评测；H200命令由用户执行。
回传附件SHA256：a7124402fc898b8a36251082943e3ede1e5692ae066f0fe791fd4914c40dc4ad。
本地已验证重复head_report一致、分语言PER算术；未收到独立checkpoint SHA及完整
metrics，不声称已验证H200权重字节。仅文档里程碑记录，git diff --check通过。

## 0.140 2026-09-23 三语480h首5epoch训练完成（H200回传）

用户返回pilot_completed，head报告status=completed、epochs_requested=5、
epochs_completed=5、early_stopped=false、best_epoch=5；因此本次是预设5轮预算
用完正常结束，不是早停、报错或已证明收敛。输出仍在
outputs/multilingual_480h_run_v1/head。run_plan SHA仍为
d146173589b915a862344df12d6d3725ff5779067530a75cfd9d700ac8a32e99。
回传cache_sha256_verified=true，test_set_used=false，all_samples_once，
每epoch958519条，train shards1873/validation shards16，validation8101条。
Head838746参数，Encoder冻结317477504参数，temporal2x/h512/k5/dropout0.1。

| 语言 | 原150h三语基线PER | 本轮epoch5 PER | 变化（百分点） | 错误数/参考音素数 |
|---|---:|---:|---:|---:|
| en | 4.736% | 4.6743% | -0.0617 | 3632/77701 |
| es | 3.910% | 4.1053% | +0.1953 | 5333/129906 |
| pt | 9.662% | 9.6391% | -0.0229 | 15743/163324 |

差值使用既有四舍五入基线；本轮分语言样本数en2808/es2631/pt2662与旧固定验证
一致。葡语基本持平，英语略好、西语略差，尚未证明扩量能明显降低葡语PER。
该对比是训练中间checkpoint与旧最终基线，不是等epoch/算力对照，不更换正式基线。
Macro PER=6.1396%，音素加权overall PER=6.6611%；两种聚合权重不同，不能混用。
train loss=0.264741、train PER=6.9654%、validation loss=0.253881。
training_seconds=4147.59（约69.13分钟）；epoch5=815.3秒，LR仍3e-4。

日志stale=1而best_epoch=5不矛盾：checkpoint可保存更小的Macro PER，但早停
重置要求超过min_delta=0.001（0.1个百分点）的改善。本轮未早停，最佳在最后一轮，
可按原配置继续观察；附件只有epoch5日志及两份一致汇总，没有完整metrics.jsonl，
不能宣称前5轮单调下降或推断最终能达到的PER。报告无S/D/I，不能归因删除改善。

本地核对重复head_report一致性、分语言errors/reference计算、Macro平均、样本数
及run_plan身份。用户文本附件SHA256：
0a69a4769f5651407fa05b6222f1a0c0d18732a9699b2045d10a014c3f19e270。
尚未收到checkpoint SHA或独立metrics文件，不声称已读取H200权重字节。

下一步在容器内项目根目录用已交付代码继续，4替换为空闲物理GPU编号：

```bash
python -B scripts/run_multilingual_480h.py formal --gpu 4
```

自动恢复第5轮Head/optimizer/scheduler，从第6轮开始，最多总计30轮，可按原规则
提前停止，不另加30轮、不重新初始化、不重提特征。中断后同一命令恢复。此次仅
文档结果提交，不需为执行已有formal入口更新代码。完成返回head/report.json和
head/metrics.jsonl，再评估三语PER和葡语收益；不以CTC PER代替最终ASR WER。
本次指标算术检查及git diff --check通过，未改训练配置/代码或现有outputs。

## 0.139 2026-09-23 全量缓存验收后交付5epoch训练及续训短入口

缓存实测完成结果已在独立提交15028ce/0.138记录。用户授权进入训练下一阶段。
本次仅扩展scripts/run_multilingual_480h.py的pilot/formal执行入口，调用原有
train_full_ctc.py，不改Head、CTC算法、采样、冻结配置或既有outputs。
分支codex/g2p-coverage-scan，最终推送SHA见回复。

先在容器外项目目录拉取：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录，4替换为用户分配的空闲物理H200编号：

```bash
python -B scripts/run_multilingual_480h.py pilot --gpu 4
```

这里pilot是完整958519条/1440小时数据训练5epoch，不是小样本smoke。使用
0.126固定的随机新Head、temporal2x/h512/k5/dropout0.1、batch256、LR3e-4、
seed20260825、all_samples_once，每轮每条一次；验证8,101条，三语Macro PER
选择checkpoint，原LR scheduler/early stopping不变。不加载完整Qwen、不做新
Encoder推理，只读取既有冻结缓存；test不参与。首次缓存SHA核验需读取约140GB，
在进入训练前有存储I/O等待属于预期。随后逐shard读取，单GPU，显存建议20–30GiB。
未实测本次训练耗时，不以缓存提取耗时推算训练完成时间。

入口要求既有run_plan及配置/所有输入SHA一致、smoke完成、cache根报告和两split
status/count/path匹配、test_set_used=false。原trainer继续严格核验所有shard
SHA、manifest/vocab身份、分片完成度和恢复指纹；不传skip校验选项。使用新的
outputs/multilingual_480h_run_v1/head/，与smoke_head/隔离。已有head目录首次命令
拒绝，不自动覆盖或随机重启；中断且存在training_state_latest.pt时明确恢复：

```bash
python -B scripts/run_multilingual_480h.py pilot --gpu 4 --resume
```

恢复Head、optimizer和scheduler，从最近完整epoch继续；未完成epoch会重跑。
若尚无state文件则恢复拒绝，保留现场诊断，不删除输出。正式5epoch完成后返回
pilot_completed及head_report，检查分语言PER、Macro PER、loss和耗时，再决定继续。
5epoch结果不能直接用作最终扩量有效/无效结论。后续经结果审查使用：

```bash
python -B scripts/run_multilingual_480h.py formal --gpu 4
```

formal要求head状态文件及已完成至少5epoch的report，自动传--resume，总预算30epoch
（不是另加30）。中断后同一formal命令恢复，不重新初始化；不自动从pilot跳转formal。
配置文件未改，所以既有run_plan和缓存可复用。每次训练前后复核已冻结输入身份。

用户完成pilot后只需返回以下小文件，不上传缓存或权重：

- outputs/multilingual_480h_run_v1/head/report.json
- outputs/multilingual_480h_run_v1/head/metrics.jsonl

可一并回传身份摘要：

```bash
sha256sum outputs/multilingual_480h_run_v1/head/{report.json,metrics.jsonl,ctc_head_best.pt}
```

新增9项用例验证fresh/resume、正式阶段前置、缓存数量/状态/路径、test标记、输入
与配置漂移、运行后复验及不调用Encoder；定向19 passed，全量434 passed/23 skipped。
修改范围Ruff/strict Mypy、CLI help及git diff --check通过。全仓Ruff仍5处既有E501，
全仓Mypy仍3处既有unused-ignore；无新增错误。本地为模拟流程测试，不冒充H200
真实训练结果。下一次用户回传后再单独记录训练实测里程碑。

## 0.138 2026-09-23 三语480h全量特征缓存完成（H200回传）

用户返回status=cache_completed，根目录outputs/multilingual_480h_run_v1，
run_plan SHA为d146173589b915a862344df12d6d3725ff5779067530a75cfd9d700ac8a32e99。
训练清单SHA eae271e209a79ee6aff64356eb75a7e7b5f94c7069bd9e857cd8de00cb740c82、
验证清单SHA d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499，
均与0.126冻结配置一致。以下是用户终端报告值，未在本地读取H200缓存字节。

| split | 样本 | shards | 基础帧 | 标签tokens | feature bytes | 提取秒数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 958519 | 1873 | 67562173 | 45517837 | 138751395901 | 16232.49 |
| validation | 8101 | 16 | 562779 | 370931 | 1155707472 | 151.22 |

两split status=pass；encoder batch8、每shard512条、resumed_shards=0。
单GPU进程/logical cuda:0，冻结Encoder参数317477504，test_set_used=false。
合计特征139907103373 bytes（约139.91GB/130.30GiB），提取约4.55小时。
total_frames为缓存基础帧，Head仍2x，不能直接把tokens/base_frames当有效CTC密度。
formal_training_started=false；这证明特征准备完成，不是新模型PER改善证据。

下一阶段复用cache/train及cache/validation，正式Head写入独立head/，不复用
smoke_head权重。固定0.126参数，先5epoch完整数据训练，再审查分语言PER及收敛；
继续时恢复同一Head/optimizer/scheduler到总计最多30epoch。正式入口必须重新
核验清单、配置及每个cache shard SHA，不重提Encoder特征、不访问test。

## 0.137 2026-09-23 wave两语4000词表检索完成，16份下游文件已导出（H200回传）

用户执行0.136入口 `python -B scripts/run_wave_keyword_retrieval.py --gpu 4`，
回传status=completed，8组各100条，共800条，输出目录
`outputs/wave_4000_retrieval_v1`。交付入口代码提交为
`19205b188851b0d538ad25967b9865ae3b2341bf`；本节是独立结果记录，不改检索代码。
报告列出16份delivery JSON，top5_reproduced=true、top7_exact_replay=true、
qwen_decoder_used=false、parameters_tuned=false。Head仍为原三语基线，回传SHA：
`bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0`。

证据是用户终端附件，SHA256为
`b8cf7850ac71050fef9fca383caf5c9890c901d1b7a9e0c84422238c607ee4f4`。
本地核对了组数、样本数、16个不重复导出文件名、Head身份和指标算术；尚未收到
独立report.json/sha256.txt或交付JSON，不声称本地重新核验了H200输出文件字节。

### 原wave目标词召回结果

以下为门控后gated_target_recall，分母是参考文本中的唯一“目标词×音频”对，
同一音频重复说同一词只计一次。不是4000词总表召回，也不是CTC PER或ASR WER。

| 组 | 目标对数 | Top5命中 | Top5召回 | Top7命中 | Top7召回 |
|---|---:|---:|---:|---:|---:|
| wave1 ES | 192 | 47 | 24.48% | 74 | 38.54% |
| wave2 ES | 192 | 42 | 21.88% | 68 | 35.42% |
| wave3 ES | 159 | 65 | 40.88% | 87 | 54.72% |
| wave4 ES | 181 | 53 | 29.28% | 81 | 44.75% |
| wave1 PT | 187 | 38 | 20.32% | 51 | 27.27% |
| wave2 PT | 187 | 28 | 14.97% | 37 | 19.79% |
| wave3 PT | 172 | 75 | 43.60% | 106 | 61.63% |
| wave4 PT | 190 | 78 | 41.05% | 115 | 60.53% |

按命中数/目标对数汇总（非四wave百分比简单平均）：ES Top5=207/724=28.59%，
Top7=310/724=42.82%，增加14.23个百分点；PT Top5=219/736=29.76%，
Top7=309/736=41.98%，增加12.23个百分点。两语汇总接近，PT wave1/2明显低于
wave3/4，不能据此解释为统一的葡语基础能力差距。回传中的wave4 ES Top7
raw命中80、gated命中81允许成立：门控后可用后续排名候选补位。

### 候选干扰及结论边界

- Top5每条均输出5词；Top7平均ES 6.9825词、PT 6.965词。输出数接近上限。
- 全部候选中，按现有规范化短语匹配未在参考文本出现的比例：ES Top5
  1298/2000=64.90%、Top7 1848/2793=66.17%；PT Top5 1315/2000=65.75%、
  Top7 1832/2786=65.76%。这是文本匹配口径的干扰指标，不是人工听审结论。
- target_precision只对被选中的原wave目标计算，不包含其余填充词。因此它接近
  94%至98%的语言汇总值，不能说明整份输出列表有相同精度。整表精度约34%至35%。
- 全表口径负样本只有PT wave3一条，不能据此估计一般无热词音频的误触发率。
- 当前证明运行完成、Top7提高目标覆盖；尚不能证明最终转写改善，也不能将
  低召回原因直接归于CTC、近音词竞争或门控。未在这批测试上调参数或改词表。

### 交付和下一步

`outputs/wave_4000_retrieval_v1/delivery/`包含wave1至wave4、es/pt、top5/top7
组合共16份JSON，命名如`wave1_es_top5.json`、`wave4_pt_top7.json`。格式保持
音频stem到`[{"word": ..., "phoneme": ...}]`，可交给后续Context Learning流程。
在容器内项目根目录校验现有产物，无需重跑GPU：

```bash
(cd outputs/wave_4000_retrieval_v1 && sha256sum -c sha256.txt)
```

后续固定这批输出比较无热词提示、Top5提示、Top7提示的最终转写和目标词正确率，
同时观察额外候选引入的错误。若单独诊断检索，可使用已保存的最多64候选排序
区分shortlist漏召、排名和门控，不需先重跑Encoder。用户回传根report.json和
sha256.txt即可补全文件身份记录，无需上传权重、音频或特征。
本次仅更新阶段成果；指标一致性检查和git diff --check通过，未改代码或outputs。

## 0.136 2026-09-23 wave Top5/Top7推理及16份下游导出入口交付（待H200运行）

第一步词表实测成果见0.135（独立结果提交d47f78a）。第二步代码交付：
configs/wave_retrieval.workzone.json、scripts/run_wave_keyword_retrieval.py、
inference/wave_retrieval.py，复用external_keyword_retrieval和既有Top7精确重放函数。
本地只跑合成数据/模拟检索回归，尚无本轮H200召回结果，不下载或运行完整Qwen。

### 本轮运行契约

- 使用已完成outputs/wave_4000_keywords_v1_b566974527的两份4000表，每语言四wave
  共用同表。首先以sha256.txt核验report、词表和targets_by_wave；词面重分词检查
  恰好4000、全部目标保留（ES316/PT295）。旧inventory报告也核验后绑定转写SHA。
- 全部8组各100条的音文ID检查、所有800条音频文件SHA及输入身份检查完成后才
  加载模型。音频字节身份写入run_config，resume重新比对；不依赖文件名判断未变。
  模型config/index记录SHA，不声称对数GB全部模型权重重算SHA；基线Head/vocab精确绑定。
- 固定原三语Head：en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/
  ctc_head_best.pt，SHA bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0。
  不使用失败original-only候选或尚未完成的480h新Head。模型路径仍
  /glusterfs_103/models/Qwen3-ASR-1.7B；vocab仍v0.2。
- 完整音频→冻结Encoder→temporal2x Head→Anchor检索，不运行Qwen文本decoder。
  ES/Spanish与PT/Portuguese显式传入数据、词表ID和processor prompt。共享旧入口
  默认仍pt-BR，旧PT ID/schema保持兼容；未改旧Top7脚本的MLS/Delivery来源映射。
- Anchor ngrams2/3/4、每词24、offset1、start radius2、shortlist64；threshold0.75、
  posterior weight0.25/minimum0.5、edit ratio0.35、margin0、minimum_phonemes1。
  Top5/7来自同一次计算，不重跑Encoder或调门控。新wave保存完整shortlist排序
  最多64条（旧入口默认20不变），仅增加记录量以保证Top7门控后排序精确重放。
- 同语言四wave只加载一次模型，逐音频推理；两语顺序运行并释放前一语种模型。
  用户用--gpu指定空闲物理卡，脚本在导入模型前设置CUDA_VISIBLE_DEVICES并使用
  cuda:0。不自动挑卡，不与训练缓存任务共享指定卡。resume可换空闲物理卡，逻辑
  device/dtype及全部输入/代码身份必须一致。
- 每wave分别导出Top5与Top7，共16文件。样本ID保留stem，跨wave同名互不覆盖；
  schema仍stem -> [{"word": ..., "phoneme": ...}]，空召回保留[]。Top5逐样本及
  聚合指标必须与原选择完全复现，Top7保存排名充分性验证通过后才导出。

### 评测口径

每组top5/top7分别报告原wave目标的raw_target_recall和gated_target_recall；分母
为“目标词×音频”唯一匹配对，不是词表总词数，也不将同一音频重复说词算多次。
原目标词列表与参考transcript只用于检索后计算指标，不参与候选产生/筛选。
单列full_4000_view，不把普通补词命中当成目标改善。误召回根据整张词表在转写中
是否实际出现判断，不能因为某补词不在原目标清单就把正确召回算成误触发。
本轮不测最终ASR WER或CTC PER；标签质量边界和同音不同词保留规则见0.135。

### 拉取、执行和恢复

分支codex/g2p-coverage-scan，最终远端SHA见交付回复。容器外项目目录：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目目录，以下4仅为示例，请替换为空闲物理GPU编号：

```bash
python -B scripts/run_wave_keyword_retrieval.py --gpu 4
```

可选仅CPU预检（不加载模型/不写输出），正式命令本身也会先完成同一预检：

```bash
python -B scripts/run_wave_keyword_retrieval.py --audit-only
```

默认新输出outputs/wave_4000_retrieval_v1。首次拒绝已有目录；明确恢复时：

```bash
python -B scripts/run_wave_keyword_retrieval.py --gpu 4 --resume
```

resume绑定代码commit、配置、词表、Head、vocab、transcript、音频SHA，只跳过完整
且行内容SHA正确的逐样本分片；已存在导出文件比较一致才复用，不覆盖内容不同产物。
中断发生在8组任何位置都可以续跑；新实验用--output-dir新目录，不删除旧输出。
代码更新导致身份变化时不能强行resume，应保留原目录并选择新目录。

### 下游文件与结果回传

outputs/wave_4000_retrieval_v1/delivery/内恰好16份：
wave1_es_top5.json、wave1_es_top7.json……wave4_pt_top5.json、wave4_pt_top7.json。
每份100个stem，可直接供后续Context Learning读取。每组目录保留逐样本结果、排序、
原4000口径summary、topk_metrics；根report汇总8组两K，sha256覆盖全部输出。

请返回根report.json和sha256.txt两个小文件；16份delivery JSON交给下游流程。
无需上传音频、特征缓存或权重。验证命令：

```bash
(cd outputs/wave_4000_retrieval_v1 && sha256sum -c sha256.txt)
```

10项新增流程测试覆盖8组/16文件、跨wave重名、空召回、原目标与补词分母分离、
Top5/Top7精确重放、双语ID/语言绑定、全组预检、SHA漂移、断点恢复、输出拒绝覆盖。
与旧外部检索/Top7回归合计23 passed；全仓425 passed/23 skipped。新增Ruff、定向
strict Mypy、CLI help、git diff --check通过。全仓仍既有5处E501、3处unused-ignore。
用户真实运行后，再以实测结果更新阶段进度，不把模拟推理数值当H200结果。

## 0.135 2026-09-23 wave两语4000词表构建完成（H200回传）

用户在bb427aadc068804341cb78839e14dff5108b105a代码运行成功，输出目录：
outputs/wave_4000_keywords_v1_b566974527。status=completed、tables_created=true。
本次收到终端摘要，未收到独立report.json/sha256.txt附件；不能声称已在本地重新
计算工作区词表SHA。后续推理入口必须先读取该目录sha256.txt核验所需文件。

| 语言 | 唯一词面 | 必留目标并集 | 近音词 | 旧库补充 | OOV |
|---|---:|---:|---:|---:|---:|
| ES | 4000 | 316 | 1842 | 1842 | 0 |
| PT | 4000 | 295 | 1279 | 2426 | 0 |

两语all_targets_retained=true。PT近音可用1279，不足目标1852，按冻结规则以旧库
补足，未为凑近音配额添加未知词。PT有9个可选词面发音冲突，57条来源记录隔离，
不涉及删除必留目标；ES可选冲突0。两语言干扰词组成不同，不能仅比较召回率就
推断跨语言基础能力。ignored_non_target_parent_occurrences是未关联本轮目标的
parent出现次数，不是丢失目标词数。

旧库ES4401条均es，PT3949条pt+454条pt-BR，解释上一轮pt/3949的元数据兼容失败。
两语old_table_validation_issue_counts均空；ES3条词面规范化更新。ES主词来源
记录累计去除U+0303共288次，包含重复wave和本来可映射的鼻化元音，不等于288个
唯一词；PT清理计数空，未调用西语去鼻化规则。输入SHA绑定、原文件不改。

ES11组、PT21组不同词面具有相同提供音素序列，均保留。它们是“标签序列相同”
而非人工确认的真实同音，例如capital/capitão、mal/mau/mão不能据此认证发音；
本任务使用固定提供词表进行检索压力测试，不在此轮按识别错误修改发音或门控。

audio_read=false、model_loaded=false，第一步两份词表完成，第二步8组Top5/Top7
及16份Context Learning文件尚未执行。下一阶段继续原三语基线Head和0.127参数，
分别报告每wave原目标词召回与整张4000词表指标，不能将填充词命中混为目标提升。
后续HANDOFF按用户要求只记录阶段交付/实测成果，不再为每次小修复新增章节。

## 0.134 2026-09-23 修复wave主词表语言名称兼容（待重跑）

0.133用户回传结果已独立提交2622eff。本轮仅修改primary语言元数据校验：
ES允许es/es-419/Spanish，PT允许pt/pt-BR/Portuguese，忽略大小写及首尾空白。
语言不匹配、未知值、空值及非字符串仍阻断；报错现在包括wave、目标语言、实际值
和允许别名。缺少language字段沿用原有由显式lang参数限定的行为。
不改变neighbors/旧库校验、SHA绑定、IPA处理、配比或选词顺序，不改用户源文件。
测试fixture改成实测Spanish/Portuguese，避免今后只测省略language的合成输入。

交付分支codex/g2p-coverage-scan（推送SHA见回复）。容器外项目根目录：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录重跑原命令，仅CPU：

```bash
python -B scripts/build_wave_4000_keywords.py
```

仍自动创建outputs/wave_4000_keywords_v1_<随机后缀>，无resume、拒绝覆盖既有目录。
成功后回传return_files指定report.json与sha256.txt；在新目录执行
sha256sum -c sha256.txt核验。失败则返回完整短报错，不更改原JSON以绕过SHA。
两份4000表及16份推理文件仍未在H200成功生成；本轮仅修复已证实的元数据兼容错误。

39项定向测试通过（含17项新增别名、交叉语言拒绝和旧行为回归），全量402 passed、
23 skipped。本轮文件Ruff、定向strict Mypy、git diff --check通过；全仓仍为既有
5处E501和3处unused-ignore，无新增，未修改这些无关文件。

## 0.133 2026-09-23 wave构建首跑失败：主词表使用英文语言名称

用户在9868a52版本运行build_wave_4000_keywords.py，返回status=failed、
error="wave1: primary language mismatch"。随后只读jq核对全部8份primary文件：
wave1–4/es均为language="Spanish"，wave1–4/pt均为language="Portuguese"。
这证明主词表元数据与脚本只接受语言代码的限制不兼容，不是词表混入另一种语言的
证据，也不是音素质量结论。此前neighbors元数据使用es/pt，不应由它推断primary
必定使用相同写法。失败在发布目录创建之前，未生成4000表或修改现有输出。
本次用户仅返回终端内容；没有新的H200报告文件SHA可复核。
下一提交限定修复primary语言别名兼容，并增强报错展示实际值及预期语言；词表配比、
IPA处理、输入SHA校验、检索参数保持0.132规则。仍待H200成功构建后再进入Top5/7。

## 0.132 2026-09-23 两语固定4000词表构建入口（待H200 CPU运行）

0.131库存结果单独提交4203474。本轮实现evaluation/wave_keywords.py、
scripts/build_wave_4000_keywords.py、configs/wave_4000_keywords.workzone.json。
当前仅在本地合成数据测试；未读取H200真实wave数据、未创建真实4000表、未运行模型。

### 固定构建规则

- 绑定0.130 inventory目录wave_keyword_inventory_v1_b704685dc1，先核验其report
  对应sha256.txt，再以report内SHA核验8份primary和8份neighbors输入；v0.2 vocab
  与旧补充库SHA另外固定在配置。所有输入在写输出前再次核验，漂移即失败。
- 四wave所有keyword_sets取规范化并集，必须得到ES316/PT295，全部保留。以现有
  normalize_match_words规范化词面。相同词面处理后token序列不同则阻断；OOV、
  无发音等不能静默删掉目标词。规范化计数不同也阻断，避免错用其他批次。
- ES所有主词/补充词统一复用repair_spanish_pronunciation(..., rules=())，只用
  现有西语训练去U+0303口径，不启用enye代理规则，不变更词面、共享vocab或源文件。
  PT不调用西语清理，保留鼻元音。输出采用vocab规范token序列；保存原IPA、清理后
  IPA、token IDs、清理计数及来源，兼容性通过不代表发音准确性得到人工认证。
- 只取parent规范化词面属于本轮目标并集的近音词；用每项phoneme，不用根部
  keyword_phonemes替代。邻词已有sim/count仅保留为来源信息，不按它们排序，
  ES清理后不宣称原sim是重新计算的相似度。未知结构阻断，无效候选显式隔离。
- 补充名额先一半近音词（ES1842、PT1852），再从旧库补满；旧库不足再用剩余近音
  词。若近音词不足，由旧库填补；两者仍不足则失败，不能少于4000条发布。
  候选按SHA256(seed:language:normalized)排序；seed=wave-4000-20260923-v1。
  这是固定压力测试配比，不代表已证明最优或两语言难度完全相同。
- 旧库读取实际4401/4403行，而非假定4000；验证语言、原IPA、token IDs、tokens和
  词面规范化一致。可选词面出现发音冲突时所有可选变体隔离；目标词优先于补充库
  同词面记录，差异写审计。同音异词全部保留，不能按发音去重而丢目标词。
- 两语言都通过才开始发布，每语言恰好4000个唯一规范化词面。选择不读transcript、
  音频、预测或召回结果。测试集目标词由用户明确提供作为运行词库，不据错误调词库。

### 运行与回传

交付分支codex/g2p-coverage-scan，容器外项目目录（最终远端SHA见交付消息）：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/build_wave_4000_keywords.py
```

仅CPU，无模型下载/加载，不占训练缓存GPU。新目录
outputs/wave_4000_keywords_v1_<随机后缀>；无resume，显式--output-dir拒绝已有目录，
不会覆盖任何旧输出。仅发布后生成sha256.txt；发生失败不得使用任何不完整目录。
每语种文件：

- keyword_bias_phoneme.json：all_keywords恰好4000条及对应可映射IPA；供后续检索。
- targets_by_wave.json：原四wave各自规范化目标词，用于独立目标召回分母。
- hotwords.jsonl：规范化词、IPA/token IDs、来源与来源记录。
- selection_audit.jsonl：逐输入词变换、未选中、无效/冲突/目标优先记录。

根目录config.json、report.json、sha256.txt。report报告必留数、两种填充来源数量、
候选冲突/异常、清理计数和同音词组；终端打印小摘要。请返回新目录report.json与
sha256.txt，无需上传词库全集、音频、tensor或权重。可在输出目录核验：

```bash
sha256sum -c sha256.txt
```

下一阶段仍为原三语基线Head和0.127参数的8组Top5+Top7，输出16份Context Learning
JSON。检索旧入口仍有PT硬编码，不要直接拿它运行ES；本轮不修改检索核心、不运行
Qwen decoder。后续入口须显式支持ES/wave，并分别报告原目标词召回及4000词表结果。

### 本地验证

15项新增测试加7项inventory回归通过；覆盖ES清理/PT鼻化保留、必留主词异常阻断、
发音冲突、相同词面优先级、无效补词、确定性顺序、近音词不足/旧库不足双向补齐、
真实4000数量测试、同音异词保留、输入SHA漂移、只读输入和拒绝覆盖。全量385 passed、
23 skipped（本地缺Torch等依赖）；新增文件Ruff、定向strict Mypy、CLI help及
git diff --check通过。全仓Ruff仍有原有5处E501，Mypy仍有原有3处unused-ignore，
本轮未改这些无关文件；不以合成测试冒充H200词表已发布。

## 0.131 2026-09-23 wave近音词结构和旧补充库实测库存

用户从0.130同一inventory报告回传字段预览，本地粘贴附件SHA256：
8cab964f817e398dd002e075eb3fa5f4040a635d06c4c143a958207911912660。
这是终端摘录，未取得完整report/sha256文件，本地不能重新计算工作区源文件哈希。

- ES/PT近音词均为neighbors[parent]列表，每项含word/sim/count/phoneme；候选词
  发音必须取项内phoneme。根keyword_phonemes是目标词发音，不能当候选词映射。
- wave1 ES共有908个parent、n_vocab4116；PT834个parent、n_vocab2664。这些不是
  四wave去重后可用补充词数量；其余wave仍须构建时逐文件检查相同结构。
- 旧capacity_es/representative/size_4000/hotwords.jsonl实际4401行，SHA256
  19d23505ad7719a404662b13b97481750bca549637028a7efc13de8a7ca08553。
  PT对应文件4403行，SHA256
  ac8f50fa6f6f14870d8ca97b7028a570f88ace81c31062c41da60228b2a5d4d3。
  均位于outputs/en_es_pt_streaming_e2e_4k_formal100_v1下；目录名不代表实际行数。
- 旧记录含surface/language/pronunciation/phoneme_tokens/token_ids等，构建时应
  重新核对共享vocab和语言，不能仅复制或靠目录名推断4000条。

现已具备实现词表构建的信息。目标仍为ES316/PT295全部必留，余量从相关近音词和
旧库确定性补齐；尚未创建两份4000表或运行模型，本记录与后续实现分开提交。

## 0.130 2026-09-23 wave v2回传：ES并集316、59词含U+0303；已有西语训练规则可复用

用户回传outputs/wave_keyword_inventory_v1_b704685dc1终端摘要，非完整report。
本地粘贴附件SHA为6c6395187745578a2c20964524538bd9667961ce89f0d6bfa453e54e6b40ab25。
8组仍各100条音频/转写，无ID对应错误；两语言的四个primary文件都已完整扫描。

- ES规范化并集316，补到4000缺3684；59个规范化词至少有一项不可映射，唯一
  OOV单元为U+0303 COMBINING TILDE。127是所有wave记录累计的OOV单元出现次数，
  不是127个唯一词。每wave异常词面32/32/28/31，包含跨wave重复。
- 已映射发音冲突为0，但phoneme_conflict_check_incomplete=true，不能在修复并
  重新核验全部316词之前宣称全部发音一致。PT并集295、缺3705、不可映射0、
  冲突0，primary_phonemes_ready=true。两语言均无跨wave共享stem。
- 样例Adrián为/a ð ɾ j ã n/；v0.2共享vocab有PT的ɐ̃、ĩ、õ等，但无ã，
  tokenizer按NFD匹配时得到a及独立未识别U+0303。因此不是一般NFC/NFD编码
  转换就能解决，不能给西语随意映射到PT的ɐ̃或扩大已有Head输出类别。
- 已核对training/spanish_mfa_repair.py的repair_spanish_pronunciation及历史
  西语G2P诊断：原训练字典已使用语言限定的去鼻化标记规则。后续新ES热词表
  应复用该函数rules=()（无代理拼写glide规则），保存原IPA/处理后IPA和变更计数；
  必须对所有ES发音使用同一训练口径，而非仅改报OOV的59词。PT不调用此清理，
  保留PT鼻元音；不改共享tokenizer/vocab、词面拼写、训练标签或现有输出。
- 本地仅对回传的5个不同发音样例调用已有函数，全部转换后OOV=0；例如Adrián
  变为/a ð ɾ j a n/。这验证了示例兼容性，不能冒充对工作区全316词的检查，也
  不证明音频实际发音正确。本轮没有新增算法代码或创建4000词表。

剩余信息仅在用户已有report里：neighbors结构与旧4000表库存。无需再次扫描或
更新工作区代码。在容器内项目目录只读提取以下字段并回传即可：

```bash
R=outputs/wave_keyword_inventory_v1_b704685dc1/report.json
jq '{
  es_neighbors: .json_files["wave1/es/phonetic_neighbors_phoneme"].root_fields,
  pt_neighbors: .json_files["wave1/pt/phonetic_neighbors_phoneme"].root_fields,
  old_4000_tables
}' "$R"
```

也可直接返回完整report.json与sha256.txt。得到实际字段后继续4000表构建，构建
时仍要逐wave校验schema，不根据wave1盲猜所有文件；主词全部保留、冲突阻断、
补词规范化去重、来源/seed/变更审计及SHA固定。16份Top5/Top7交付仍待下一阶段。
本轮仅新增此结果记录；已有函数的5项实际样例验证及git diff --check通过。

## 0.129 2026-09-23 wave盘点v2：OOV逐词继续，分离词面并集与音素可用性

0.128用户实测摘要已独立提交2a36189。本轮只修复evaluation/wave_inventory.py的
盘点行为和测试，不修改vocab、IPA转换规则、源热词/近音词文件或检索门控。

- 旧版遇首个OOV即中止一份primary文件，使ES显示并集1。v2对缺失音素、OOV或空
  音素逐词记录并继续扫描：目标词面仍进入规范化并集，不因不可映射而被删除。
- complete_primary_files现表示结构正确且词面扫描完成的文件数，不表示每个音素
  已可用；counts_are_partial只针对词面扫描。primary_phonemes_ready单独要求4份
  完整扫描、无任何不可映射项及无已映射发音冲突，不能混为通过。
- 每语言报告unmappable_normalized_word_count、最多5词所有wave原音素示例、
  oov_units_top20；每文件列phoneme_issue_count及最多10个原音素/OOV/Unicode编码
  和Unicode名称示例。不把tokenizer返回的部分token序列用于发音一致性比较：
  不可映射项token_ids=null，冲突仅对完整可映射序列比较；若有未映射项，明确
  phoneme_conflict_check_incomplete=true，0个已知冲突不等于已经排除全部冲突。
- schema_version=2，仍tables_created=false。不会根据名字猜测西语异常符号，
  不做全局音素合并，也不运行G2P/Encoder/检索；旧盘点目录保持原样。

容器外项目目录拉取工作分支（最终SHA见交付消息）：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内重跑同一短命令，不使用GPU：

```bash
python -B scripts/inspect_wave_keyword_inputs.py
```

自动创建新的outputs/wave_keyword_inventory_v1_<随机后缀>；路径前缀沿用，实际
JSON schema为2，不复写6c459d81a1。无resume。请返回新return_files指定的完整
report.json与sha256.txt；终端摘要省略neighbors结构，不能只据摘要实现补词解析。
可在新目录sha256sum -c sha256.txt校验。得到字段、异常原音素及补充库库存后，
再确定最小兼容处理和两份4000表；当前两份表与16份推理交付仍未生成。

7项定向测试通过，新增回归覆盖首词OOV不截断后续词、完整并集独立计数、部分
token序列不冒充完整发音、Unicode证据；全量370 passed/23 skipped。新增/修改
Ruff、定向strict Mypy、git diff --check通过。全仓仍既有5处E501和3处unused-ignore。

## 0.128 2026-09-23 wave盘点回传：800条音文对应、PT并集295，ES被音素映射阻断

用户回传outputs/wave_keyword_inventory_v1_6c459d81a1的终端摘要，尚未回传该目录
完整report.json/sha256.txt。8组wave×language均100条音频/100条转写，无源内ID
对应错误，共800条；这仅证明文件/转写标识对应，不证明音文或发音标注准确。

- PT complete_primary_files=4、counts_are_partial=false，规范化并集295词，距离
  4000尚缺3705；未发现可映射音素序列冲突，四wave没有共享stem。
- ES complete_primary_files=0、counts_are_partial=true，终端显示union=1是旧盘点
  在首个OOV处提前停止产生的部分计数，不是实际西语并集。wave1/2首先遇到Adrián，
  wave3/4首先遇到Aero Club Orán；这也不证明只有这两个异常词或四处OOV。
- 当前终端没有给出原始IPA、具体OOV单元、近音词JSON结构及旧4000库存在性；
  因此不能判断是符号格式还是发音标签差异，不能直接改全局音素规则、丢词或换
  vocab来绕过错误。需要完整结构/异常证据才能继续两份正式4000词表。
- tables_created=false、model_loaded=false；目前未生成4000表、未跑GPU检索，
  与480h特征缓存任务保持独立。下一步修正只读盘点为逐词继续扫描，统计完整词面
  并集并单列所有不可映射词、原始音素及Unicode信息，不更改源词表和原音素值。

## 0.127 2026-09-23 wave1–4西/葡4000热词与16份下游交付：输入盘点入口

用户在480h缓存运行期间新增独立外部测试任务，最终使用另一张H200卡。宿主机仓库
wave1、wave2、wave3、wave4各有es/pt，容器内仓库前缀为/host_home/star/q00933266/
qwen3-asr-hotword。每组wav/、transcripts.txt、LANG_keyword_bias_phoneme.json及
LANG_phonetic_neighbors_phoneme.json；不解包同级RAR，不改这些原文件。

用户授权两步：第一步每语言取四wave目标热词并集，再由近音词和旧4000词表补到
恰好4000个规范化词面，PT/ES各固定一份，供四个同语种wave共用；第二步按旧配置
测试Top5/Top7，生成4wave×2语言×2K=16份Context Learning输入。不同wave即使有
同名音频也必须分开交付，不合并为会覆盖stem的全局JSON。

已核对0.76、0.78–0.79、0.82和external_keyword_retrieval/topk_replay源码：

- 旧Context Learning完整音频路径为冻结Qwen encoder + 原三语temporal2x Head +
  Anchor检索；不运行Qwen文本decoder，不属于暂停中的新机器纯推理移植。
- 固定参数threshold0.75、Top5/7、posterior weight0.25/minimum0.5、edit ratio上限
  0.35、margin0、minimum_phonemes1、Anchor ngrams2/3/4、per-entry24、offset1、
  start radius2、shortlist64、保存raw排名深度20。不能与旧4k流式0.86门控混淆。
- 下游schema保持音频stem -> [{"word": ..., "phoneme": ...}]；空召回保留[]。
  Top7优先从同一次已存排名精确重放，先验证排名充分，避免重复Encoder工作。
- 老入口硬编码pt-BR/Portuguese及external_pt ID，Top7出口来源映射仅支持MLS/
  Delivery。后续须最小范围显式支持es及wave来源，不能把西语伪装成葡语直接运行。
  本轮尚未修改检索核心/语言入口或生成任何16份结果，也不使用未完成的新480h Head。
- 后续召回至少单列各wave原始目标词与transcript匹配的真值口径，另外报告4000词表
  的整体结果/误触发，不能让新增普通填充词提高召回分母后冒充原目标热词结果。

### 本轮交付：CPU输入结构与并集盘点

本地没有新wave文件内容，仅有用户目录拓扑，尤其neighbors JSON结构尚未知。
新增scripts/inspect_wave_keyword_inputs.py、evaluation/wave_inventory.py，先读取
工作区实际结构，避免猜测字段、遗漏目标热词或静默挑一个冲突发音。

- 读取8份primary和8份neighbor JSON，记录SHA、大小、有限深度的字段/样例；重复
  JSON键报告问题，避免Python默认覆盖。未知primary schema报告需审查，不猜字段。
- 对已知keyword_sets/keyword_phonemes结构，盘点所有声明set的并集（不默认选
  空baseline），用现有匹配规范化去重，逐词检查IPA能否映射v0.2。报告完整文件数、
  每语言并集数、距离4000缺额、同规范化词面不同token序列的冲突和有限示例。
  任何某文件部分失败均标counts_are_partial，不将部分计数当完整并集。
- wav/递归数WAV/FLAC文件，仅文件系统元数据；复用旧transcripts解析器，报告
  源内stem重复、音频/文本缺配、各wave同名stem规模。跨wave同名只记录，后续分开
  输出；不改ID。转写仅有前两条截断示例，不在本步运行召回。
- 查找已记录旧库outputs/en_es_pt_streaming_e2e_4k_formal100_v1/capacity_LANG/
  representative/size_4000/hotwords.jsonl，记录存在性、SHA、条数、两条样例。
  找不到时不把库存当零或另找不明语言词表；回传后再明确补词来源。
- 原始输入只读，输出新outputs/wave_keyword_inventory_v1_<随机后缀>/report.json
  与sha256.txt；不读音频字节、不加载模型、不占GPU、不碰训练/cache输出。
  status=inventory_completed仅表示盘点完成；tables_created=false是预期状态，
  不能称两份4000表已经生成。近音词抽样配比待实际库存明确后固定并记录来源与seed，
  不根据本次测试预测/错误来挑补充词或调门控。

工作分支codex/g2p-coverage-scan；容器外项目根目录：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录只需一条，不选GPU：

```bash
python -B scripts/inspect_wave_keyword_inputs.py
```

回传终端return_files指向的report.json与sha256.txt两个小文件。终端仅打印数量与
问题，完整结构样例在report内，因此需要report而非只有终端摘要。不上传wav、RAR、
完整neighbors文件或模型。可在新目录内sha256sum -c sha256.txt核验。无resume，
重复运行创建新目录；显式--output-dir拒绝已有目录，原所有outputs保持。

6项定向测试通过，覆盖两语四wave并集、两类释放文件结构、发音冲突/OOV/未知schema、
音文缺配、重复JSON键、有限preview、原始输入不变；全量369 passed/23 skipped。
新增Ruff/strict Mypy及CLI help通过，git diff --check通过；全仓仍既有5处E501、
3处unused-ignore，无新增。后续需用户回传结构才能完成第一步4000词表正式发布。

## 0.126 2026-09-23 三语480h固定训练配置与H200 smoke/cache入口（待执行）

用户同意进入训练配置及特征缓存准备。本次新增configs/480h_training.workzone.json、
configs/480h_cache.workzone.yaml、training/multilingual_480_run.py及
scripts/run_multilingual_480h.py。交付分支codex/g2p-coverage-scan。本地只运行
合成测试、CLI help和show-training；没有下载/加载完整Qwen或运行H200实验。

### 固定训练方案

| 项目 | 本轮固定值 |
|---|---|
| train | 0.125 v2 combined，958,519条，三语各480h |
| train SHA | eae271e209a79ee6aff64356eb75a7e7b5f94c7069bd9e857cd8de00cb740c82 |
| validation | 历史en_es_pt_balanced_validation_4h_v1，8,101条 |
| validation SHA | d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499 |
| vocab | v0.2，SHA 0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4 |
| backbone/tap | 冻结Qwen3-ASR-1.7B，thinker.audio_tower.ln_post，1024维bf16 |
| Head | 随机初始化temporal_upsample，hidden512/kernel5/dropout0.1/time2 |
| train batch / LR | 256 / 0.0003，与原三语基线相同 |
| weight decay / clip | 0.0001 / 5 |
| 采样 | all_samples_once；每epoch每条一次，不补抽、不删除recovery |
| seed | 20260825，与原基线相同 |
| checkpoint | 三语等权Macro PER，tie-break为worst-group PER再validation loss |
| early stop | Macro PER，min delta 0.001（0.1个百分点），patience6，minimum5epochs |
| LR scheduler | 保持既有validation loss驱动，patience2/factor0.5/floor1e-5 |
| 预算 | 后续先5epoch pilot，再按相同参数resume至总计最多30epoch；本入口不执行 |

保留原固定validation是为了与9.662%等历史值直接比较，不宣称它已成为公平跨语言
clean validation。原完整validation与高密度视图仍保留，可作后续诊断，不混入本轮
checkpoint选择。西语仍使用此前确认的拉美来源。等epoch的480h比150h约3.2倍音频
曝光，因此这轮是扩量效果实验，不是严格等optimizer预算的单一数据多样性因果实验。
保持结构/标签/优化参数固定；扩量不保证PER必降，回传后仍按PT PER及S/D/I、英西
退化情况判断。smoke Head绝不用于正式Head初始化。

### 准备与输入检查

- 输入train、validation、vocab固定SHA；验证修复目录report/指纹/清单的sha256.txt。
  扫描manifest的ID/路径、split、language bucket、2x时间因子、条数和估计帧数，
  train/validation不重叠。固定旧validation的每条音频必须在0.125指纹的validation
  保护集合中；不重扫153GB音频、不读取sealed test manifest/标签/音频。
- 读取旧150h validation cache_config，要求model路径/dtype/tap、validation/vocab
  SHA一致；当前Qwen config.json及weight index SHA须与旧缓存记录一致。这里只
  哈希模型元数据，不重新哈希所有权重文件。新模型配置使用已确认H200模型路径，
  local_files_only=true、bfloat16、cuda:0，不依赖可能缺失的workzone.local.yaml。
- 将冻结顺序中每个language×release的前16条train、前8条validation存为smoke
  清单（最多96/48条）；包含original/recovery，缺某层不伪造。它只用于链路检查，
  不是统计评测或新基线，绝不按模型错误挑选。
- run_plan.json保存完整配置、输入SHA、smoke清单SHA、估计基础帧数/存储需求。
  帧估计不作为真实帧数；真实音频解码和CTC可行性由原cache pipeline严格核验。
- 固定新根目录outputs/multilingual_480h_run_v1；首次prepare拒绝任何已有目录。
  smoke/cache首次调用自动prepare；已有目录必须拥有匹配的run_plan和全部输入SHA。
  改配置/输入不会误续旧目录；可用--output-dir显式选新的根目录，两阶段须相同。

### 短命令：先少量链路检查，再全量缓存

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录，3仅为物理GPU示例，须换成已分配且有足够空闲显存的H200：

```bash
python -B scripts/run_multilingual_480h.py smoke --gpu 3
```

该步骤会加载完整Qwen提取少量smoke缓存，释放Qwen后随机初始化CTC Head并训练
1epoch。max96train/48val不会被冒充为完整训练。沿用batch8提取、512samples/shard，
真实Qwen只在工作区运行，建议该卡至少40GiB空闲；本地不运行。完成应返回
status=smoke_completed及有限loss/PER；无需要求小样本1epoch达到任何质量阈值。

smoke正常完成后，可直接执行全量缓存，无需等新代码：

```bash
python -B scripts/run_multilingual_480h.py cache --gpu 3
```

全量缓存严格要求smoke report已完成。提取958,519train和8,101validation；新目录
cache/train、cache/validation，单GPU/单进程顺序分片。重新提取validation约1.2GB
以保持本轮统一配置，不覆盖/搬移旧cache。预计train+val约140GB量级，实际以
run_plan的基础帧估计和最终报告为准；缓存前检查空闲空间达到估计×1.25+10GiB，
续跑扣除新目录已写入文件大小。没有磁盘/吞吐实测，不承诺完成时间。

两个命令重复执行会校验输入身份和已完成shard SHA后续跑，不删除旧产物。
smoke Head中断时恢复training_state_latest.pt；完成后不重新训练smoke Head。
缓存沿用现有文件锁和分片一致性检查，另加任务级锁防止两个入口并发。
若prepare因中断留下无run_plan的目录，保留现场并选新--output-dir，不能删目录
或跳过身份门槛。解码/实际CTC长度失败时报告原ID，不自动删样本或改标签。

### 验收与下一阶段

回传两次终端最后JSON，含run_plan_sha256和固定train/validation SHA。若文件返回，
只需run_plan.json、smoke_head/report.json、cache/feature_cache_report.json；不上传
完整清单/缓存/模型。全量期望status=cache_completed，train sample_count958519，
validation sample_count8101、两split status=pass，test_set_used=false。
formal_training_started必须仍为false。

`python -B scripts/run_multilingual_480h.py show-training`仅打印后续5epoch pilot和
30epoch resume的参数数组，不执行。正式训练将在缓存验收后单独交付短命令；不将
smoke Head续为正式Head，不跳过cache SHA核验、不读取test。未来正式输出head/
与smoke_head/分离。当前无需运行任何长训练命令。

本轮10项定向测试通过：身份/模型元数据变化阻断、未保护validation阻断、跨split
路径交集、层覆盖、原输入不变、显式GPU、smoke恢复、全量前置/空间门槛、命令参数。
包含子进程运行期间输入变化的结束复验。全量363 passed/23 skipped（本地无Torch）；
新增Ruff、定向strict Mypy、CLI help、
show-training、git diff --check通过。全仓Ruff仍5处既有E501，Mypy仍3处既有
unused-ignore；未改相关旧文件及用户未跟踪产物。H200运行结果尚未产生。

## 0.125 2026-09-23 三语480h修复完成、正式训练清单发布（用户返回）

用户执行0.124修复入口，返回status=completed、training_manifest_ready=true。
输出目录outputs/multilingual_480h_training_v2_a48f58e452。交付代码提交为
9dc0b27cbbb9b31b433c3de04ec66ee97b78ecfe；回传未单独包含工作区git HEAD，
不据此宣称已经再次核验运行时Git版本。本地粘贴附件SHA为
b174ad885f6dcf25247052e4863e838b46bc980a1e221d5652a33b6afb4f105b。

本轮解析回传JSON，核对排除ID摘要与configs/480h_repair.workzone.json一致，
逐维度条数/小时加总、三语合计、文件扫描数与保护集合数量全部自洽。以下manifest
SHA来自H200返回；本地没有这些大清单，未宣称独立重新读取并哈希了远程文件。

| 语言 | 最终条数 | 最终小时 | recovery小时 | recovery时长占比 |
|---|---:|---:|---:|---:|
| EN | 340,379 | 480.000095655 | 0.021795642 | 0.004541% |
| ES | 296,399 | 480.001822655 | 39.785051580 | 8.288521% |
| PT | 321,741 | 480.000408127 | 133.649949007 | 27.843716% |
| 合计 | 958,519 | 1440.002326437 | — | — |

- 按既定排除摘要9f81aa34b54234f5c9401796aeb1500dbfc47726b80b4a1f765a997f9e451d1e
  排除154条/0.201999236h；EN2条、PT152条，具体原因见0.123。
- EN补1条/0.001483385h（5.340187秒）；PT补159条/0.199667170h
  （11.980030分钟），补样字节扫描分别1/159个，均无拒绝。958513-154+160=958519。
  小时略超480是保留完整音频的正常结果，没有重复抽样凑小时。
- PT source小时为Noah500h 284.214870161、finance 111.838333630、MLS65.152634966、
  Common Voice14.752036037、FLEURS4.042533333；保留多来源和原自然密度分布。
  recovery27.843716%与修复前27.843469%近乎一致，不是固定强行抽27%。
- ES全部296399条保留。新ES文件SHA变化来自三语新版本元数据；原冻结ES引用仍为
  145c16e2019c4966c450557a150333eea19bcafec37f5eca36ee6e4d8815b300。
- 最终重新扫描1,000,489文件/153,693,381,203字节；唯一train文件SHA958,519，
  等于训练行数。conflict_groups_by_reason={}，train_holdout_path_overlaps=0，
  combined重复ID/路径均0。保护集合validation21,029、test20,941，与旧扫描一致。
  958519+21029+20941=1000489；retained_audio_rehashed=true。
- 英/西/葡原validation/test引用及SHA保持0.124以前版本不变。test字节仅用于身份
  保护；test_manifest_content_read=false、model_loaded=false、training_started=false。

正式清单身份（均在上述v2目录）：

```text
full_ctc_train_en.jsonl cb85a548f41f1b21e682d09a90264829296aae01381a60aa603bbcf66ba67a5b
full_ctc_train_es.jsonl 3f3abb4c53f511dd48b8b52ad59954480bbdcdf2cdfb62ec1ceb5aaa500d0777
full_ctc_train_pt.jsonl 987e650842f2dabcbd82abd1cbe3659a5005a195ee3fa6d7df77383de450d25a
full_ctc_train.jsonl eae271e209a79ee6aff64356eb75a7e7b5f94c7069bd9e857cd8de00cb740c82
```

数据清单准备阶段完成；不要再次运行repair或覆盖v1/v2。下一阶段先固定缓存和
训练配置，绑定本节combined SHA，保持Qwen3-ASR-1.7B冻结、现有temporal-2x Head
和音素定义，明确每epoch唯一音频遍历、预算/停止规则、checkpoint选择的validation
视图，再给用户独立H200缓存及训练命令。继续保留旧固定验证视图用于历史PER比较；
完整池的三语validation时长不同，不能直接将汇总micro PER当成三语等权指标。
本轮没有实施该阶段、创建缓存或启动训练，不把旧150h缓存当作新480h缓存使用。

结论边界：文件字节去重和保护集合隔离已通过，不证明重编码/重叠片段全部排除，
不保证跨来源speaker隔离或每条音素标签正确，也不能据此解释历史葡语PER的因果。
本次仅更新结果文档并核对回传算术/身份引用，git diff --check通过；没有源代码
变更，不重复运行上一轮已记录的353项回归测试。

## 0.124 2026-09-23 三语480h冲突修复与补齐入口（待H200执行）

0.123实测结果已单独提交ba66f04。原回传附件SHA为
9a303159b5d4c410ecf2429f61ac3a75177cc2bc5e57e95df5153988965d5ab9。
新增scripts/repair_multilingual_480h_training.py、training/multilingual_480_repair.py
和configs/480h_repair.workzone.json；交付分支codex/g2p-coverage-scan。

本次仅准备新训练清单，尚未在H200运行。固定Qwen模型、音素标签、Head结构和
原validation/test，不做训练、特征提取、G2P或任何模型推理。

### 修复规则与放行条件

1. 将0.123回传的report、指纹、三个pending清单SHA及154条排除ID集合SHA写入
   配置；先重算诊断并逐项核对。不接受换目录/改清单后沿用旧结论。
2. 新清单中排除154条：104条标签/时长不一致副本整组隔离、37条一致的多余副本、
   13条与heldout字节相同的train副本。不是删除原音频，也不是断言104条全为错标。
   原输出目录、原计划和原清单保持不变。ES没有排除项，全部原选择保持。
3. EN/PT仅从原train池补足各480h，绑定68820743.../47214e4b...两个完整SHA。
   只允许被移除行所属的source、release、时长桶、density桶、CTC ratio桶联合层；
   因此当前EN只补Swift、PT只补Noah500h。按各层移除时长占比分配实际缺额，
   层内以seed=20260923和ID稳定排序，优先填最大剩余配额。保留整条音频，因此
   总时长可略超480h；分层不足时只在其他被移除层中补，并记录配额/实际时长，
   不扩展到新的层。所有可用层均不足则停止，不重复抽样凑数。
4. 排除所有旧选择ID/路径，包括本次被隔离行。每个补入文件读取字节SHA，避开
   上次全部train、隔离副本、heldout指纹以及本次已接受补样，防止换路径补回。
   source清单重新核验标签、CTC可行性及摘要；文本/音素值不改。
5. 补齐后对完整修正版train及保护集合重新做文件字节核验，保留train文件SHA须
   与上次一致，补入文件须与刚核验的SHA一致。文件在读取前后的size/mtime须稳定。
   无训练内容重复、无train/heldout内容交集、无validation/test保护内容交集，
   且唯一train SHA数等于train条数，才能发布正式manifest。此步需要再次读取约
   153GB音频文件，耗时与原冻结的字节扫描相近；不使用GPU、不解码完整波形。
6. 输入元数据结束前再验SHA；按累计小时交错输出combined，并输出三个语言的
   manifest。仅更新dataset_version等版本字段，不改变保留样本的文本和音素。
   新版本为en-es-pt-480h-content-dedup-v2。

仍有的边界：SHA只能识别相同文件字节，不能识别重编码/重叠录音；不作标签正确率
或跨来源speaker隔离认证。test音频只读取字节用于身份保护，不读取test manifest
文本/音素、不评测。原heldout引用继续在report中保留，不修改验证或测试集合。

### 工作区执行与返回

容器外项目目录：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/repair_multilingual_480h_training.py
```

默认8线程，--workers可设1..32。自动新建
outputs/multilingual_480h_training_v2_<随机后缀>；拒绝已有目录，不支持resume。
失败保留FAILED.txt和部分产物，不删除覆盖；修正原因后再运行会用新目录。
不要同时重复运行，也不需要重跑旧freeze。先输出身份/选样进度，最终扫描每5120
个文件报告进度。磁盘需容纳新的清单和指纹（数GB），不复制音频。

成功时终端返回聚合JSON，需status=completed、training_manifest_ready=true、
三语各不少于480h、audio_identity_audit.conflict_groups_by_reason为空。
仅返回终端JSON即可；若使用文件，则返回新目录report.json和sha256.txt。
用户可将R设为终端打印的新输出目录，再核验：

```bash
(cd "$R" && sha256sum -c sha256.txt)
```

保留excluded_rows.jsonl、replacement_ids.jsonl和完整report内strata以供复查；
无需上传这些逐行文件、音频或模型。最终字节检查发现新冲突时返回FAILED.txt和
audio_identity_audit.json；不能把失败目录中的pending清单用于训练。

新增6项测试覆盖排除/同层补样/原输出不变、字节重复候选拒绝和不足、保留音频被
修改、pending篡改、排除集合身份变化及已有输出拒绝；结合旧诊断7项定向测试
13 passed。全量353 passed/23 skipped；本轮Ruff、strict Mypy、CLI help和
git diff --check通过。全仓检查仍有既有5处E501和3处unused-ignore，未修改相关
旧文件或用户未跟踪文件。这里只验证本地合成fixture，真实修复与小时数待H200回传。

## 0.123 2026-09-23 冲突全量诊断：拟移除154条，英缺5秒/葡缺11.96分钟

用户回传audit_multilingual_480h_conflicts.py，status=completed，files_written=false；
此时仍只是拟处理方案，没有删除数据、重写清单或开始训练。96个独立冲突SHA组，
85个train重复组与12个train/heldout组中有1组重叠；影响189条训练记录，分为PT185条
和EN4条，ES不受影响。不能再将全部冲突归为葡语或把组数当作待移除条数。

| 拟处理动作 | 组数 | 移除条数 | 小时 |
|---|---:|---:|---:|
| 同音频但训练记录标签/时长不一致，整组隔离 | 49 | 104（PT） | 0.149702778 |
| 一致副本每组保留一个，移除多余副本 | 35 | 37（PT35、EN2） | 0.035631944 |
| 与heldout相同的所有train副本移除 | 12 | 13（PT） | 0.016664514 |
| 合计 | 96 | 154 | 0.201999236 |

标签判据为原文、音素序列、语言的联合签名；前几个例子显示training_label_mismatch
为true，但不区分原文差异、标点差异还是实际音素差异，因此不能称104条已证实转写
错误，也不能据此量化葡语历史PER有多少来自标签。整组隔离是保守的数据选择措施。

所有PT冲突属于noah_500h，EN属于swift_us_english。train/validation交集4组/4条/
0.005254167h；train/test交集8组/9条/0.011410347h。这里没有读取test文本或音素
标签，也未做测试评测。原validation/test清单继续保留，不为消除泄漏而移动留出行。

拟移除：PT152条/0.199921458h（original101条/0.127526319h、recovery51条/
0.072395139h）；EN2条/0.002077778h，均original。拟处理后：

- EN340,378条/479.998612269h，距离480h为0.001387731h≈4.996秒；
- ES296,399条/480.001822655h，不动；
- PT321,582条/479.800740957h，距离480h为0.199259043h≈11.9555分钟；
- 合计958,359种唯一训练文件字节内容，尚待实际应用排除和补入后核验。

回传身份：

```text
report.json dbfbb605019d2bed7f6a64478ae7b1ab0c7fbf6312af7b7783370bc0047b015e
audio_fingerprints.jsonl a9d22b75f6b9ce8e3cceacca6cb4562a599912d670a0e9b36c4396a75a6161b1
en.pending.jsonl bb2737c0266711fabaef19a2636868ba26f2893fa8b2f529dd8902b94fbadd59
es.pending.jsonl 8486e688b9e4663109e6a6748da06d7e96eeef38e4e19386c54d0859859f71df
pt.pending.jsonl 32dda41bb57a25c3432ccf74ee86388111cbbe0a74dfe113abfe727d025e43eb
proposed_removed_ids_sha256 9f81aa34b54234f5c9401796aeb1500dbfc47726b80b4a1f765a997f9e451d1e
```

下一步新版本train应用上述排除，从原train池按被移除的source/release/时长/density/
ratio层补齐EN/PT；ES不重新采样。补入必须避开原已选ID/路径以及全部已知训练、
隔离和heldout文件SHA，不能把被隔离音频的另一副本补回来。既有outputs原封不动。

## 0.122 2026-09-23 复用指纹的只读冲突诊断入口（待H200执行）

0.121结果已独立提交0e979a3；用户粘贴附件本地SHA为
346e6a78da415e3df18f6d1d1358e17db6d81d9b665b3b22f24a4239483e2a2e。
新增scripts/audit_multilingual_480h_conflicts.py、training/audio_conflict_audit.py，
默认配置configs/480h_conflict_audit.workzone.json绑定ae7eae83de冲突目录。

此步只读已有report、audio_fingerprints和en/es/pt.pending.jsonl；分别对照原目录
sha256.txt验证身份，并在结束前复验。不会再次读取153.69GB音频，不运行模型，
不写任何文件，不修改/删除清单或音频，也不打开validation/test文本或音素标签。
fingerprints仍代表0.121扫描时的文件内容，不宣称重新校验了此刻的音频字节。

1. 从完整指纹重算总文件数、字节数、唯一训练SHA数和全部冲突组数，要求与冻结
   report完全一致；按SHA合并组，因此两类冲突重叠不会重复计入待处理训练行。
2. 关联所有冲突train路径到三个pending清单，核验split/language及完整条数小时；
   缺失关联阻断。报告影响范围和拟剔除范围的language/source/release条数小时。
3. train/validation与train/test分别报告组数及涉及train条数小时；同组同时关联
   val和test单独列出。保护路径可能是未释放源行组成的超集，不据此断言每个冲突
   都已进入某次模型评测。当前仍不能据Top10判断所有来源或历史Head的数据污染。
4. 只提出以下处理方案，不实际执行：
   - 含heldout副本：保留heldout，提出移除该SHA组全部train副本；
   - 纯train重复且language/text/phoneme_token_ids完全相同、时长一致：按ID/路径
     固定排序提出保留第一个、移除其余副本；
   - 纯train重复但上述标签或时长存在差异：提出全组隔离，不猜哪个标签正确；
   - 仅heldout内部冲突：单独标记需处理，不能靠删除train解决。
5. 原文完全相同是保守判据；标点/大小写不同也会标为差异，而非宣称转写已错。
   原文及音素序列不打印，Top-N仅返回必要ID/路径/来源/小时和差异布尔值。
6. 输出拟剔除后的每语种条数小时、达到480h所缺小时、拟剔除ID集合SHA和每类最多
   3组例子（每组最多3个train/heldout例子）。任何补入音频以后仍须与保留train和
   全部heldout做内容去重，不能凭不同路径或剩余库存足够直接放行。

工作分支codex/g2p-coverage-scan。容器外项目目录：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/audit_multilingual_480h_conflicts.py
```

返回终端JSON即可。此次无新输出目录、无resume、无GPU或音频读取；可重复只读
运行。不要重跑freeze来获得同样的阻断。status=completed仅指诊断完成，最终
training_ready仍false。待回传影响小时/标签差异后，在新版本计划中处理train，
按原source/release/density层补足，并保持原validation/test不改。

新增7项pytest验证含原因重叠去重计数、同音频文本/音素/时长差异、完整来源关联、
只读输入不变、损坏指纹/摘要/计划阻断；定向通过。全量347 passed/23 skipped；
新增Ruff/strict Mypy、CLI help和git diff --check通过。全仓仍5处既有E501和
3处既有unused-ignore，无新增问题。未触碰用户未跟踪文件或已有outputs。

## 0.121 2026-09-23 三语480h冻结被音频字节冲突阻断（用户返回）

工作区outputs/multilingual_480h_training_v1_ae7eae83de完成全部文件指纹扫描，
status=blocked_audio_conflicts、training_manifest_ready=false、artifacts={}。
这是数据身份检查触发的预期阻断，不是训练失败，也不能记录成三语正式清单已完成。
保持全部pending产物；禁止跳过冲突门槛、直接训练或覆盖/删除该目录。

- 所选EN/ES/PT仍为340,380/296,399/321,734条，合计958,513条，小时与0.119一致。
- checked_files=1,000,483，bytes_read=153,693,478,469；训练958,513加保护集合
  validation 21,029、test 20,941，恰等于文件总数，核对通过。
- unique_train_file_hashes=958,420，比训练记录少93，即存在93个多余文件字节副本。
  duplicate_train_file_bytes=85是冲突SHA组数，不是重复记录数。
- train_holdout_file_overlap=12亦为SHA组数，不可直接当成12条待删除训练样本。
  两类组可能重叠，不能直接将85+12当作独立问题总数。
- train_holdout_path_overlaps=0；这说明路径/ID不重叠仍会漏掉不同路径下的同一
  文件内容。此前“无路径重叠”的结论不等于内容无重叠，应以本次字节证据补充。
- Top10展示路径均在葡语Noah500h目录内，包括100201/100227、200021/200025。
  这不足以断言全部冲突都来自PT、同一个speaker或转写错误，也不能由此判断历史
  Head训练是否使用了这些具体副本，需要逐条身份核对才可下结论。
- EN保护=7,187/7,086，PT=10,899/10,958，ES保守超集=2,943/2,897；所有原heldout
  引用未改动。未出现validation_test_protected_file_overlap原因。

下一步只读复用该目录audio_fingerprints.jsonl和三个pending训练清单，统计全部
冲突来源、语言、条数小时、train与validation/test分别相交的规模，比较训练副本
之间的原文字段和音素标签是否一致。不再重扫153GB音频，不读sealed test标签，
不根据模型预测筛选。原则为保留heldout，从新train剔除泄漏副本；纯train重复且
文本/音素/语言/时长一致时才可提出保留一个副本；标注冲突提出全组隔离待核验。
本轮尚未执行剔除或补抽，缺多少唯一小时和哪些语言需要补齐仍待统计。

## 0.120 2026-09-23 固定三语480h计划的正式清单入口（待H200执行）

新增training/multilingual_480_freeze.py、scripts/freeze_multilingual_480h_training.py、
configs/multilingual_480h_freeze.workzone.json。默认绑定0.119计划目录及英葡两个
proposed_ids的SHA；0.119实测结果已独立提交c21539f。交付分支codex/g2p-coverage-scan。

### 输入、身份核验与边界

- 校验plan目录report/config/ID文件SHA，拒绝FAILED计划、非完成状态、目标小时
  不一致、输入改变；ID摘要必须匹配用户回传的216d18d...和791e3f3...。重新汇总
  全部ID的各维度条数/小时并与完整report一致，回填时逐条比对原train元数据。
  不重新采样，不调整source/recovery/density，不改文本或音素标签。
- 西语整份引用0.116正式清单，核验SHA与条数小时。原文件不写入；为了统一训练
  格式，在新三语输出的西语副本中仅增补experiment、dataset_version和language
  bucket等元数据。所有ES行均保留，未重新随机排序/选样。
- 英语与旧西语留出保护复用0.115的source TSV+speaker assignments，核验各输入
  SHA，得到包含未释放行的保守保护超集。不会借此声称Noah跨来源speaker隔离。
- 葡语原pool是existing_stable_split_hash，不能套用speaker切分。读取split_config
  所绑定SHA的ready/review来源文件，只使用身份、时长、split_hash、问题/长度
  元数据，复用原ready/recovery释放规则与96/2/2边界重建留出身份。逐来源逐split
  的条数小时必须与旧summary一致。源JSON被解析，其中含有文本/音素字段，但本步
  不使用这些字段核验发音、评测或调参；sealed test Manifest本身不打开。
- 三语原validation清单只用于核对路径是否全部在各自保护注册表中，并核验SHA/
  条数。test保护数量不得小于旧summary。本轮不评测模型、不重新划分或更改heldout。
  report中的heldout_references是原完整池的引用，不自动指定未来checkpoint选择
  用哪个validation视图；原正式三语评测视图也不在本步重建。
- 读取958,513条固定train及保护集合音频的完整文件字节SHA，默认8线程、256文件
  一批、每5120文件输出进度；这会重新读ES音频，以完成三语之间的去重。test保护
  音频亦仅为身份检查读字节，明确test_audio_bytes_read_for_identity_only=true。
  不解码波形、不加载Qwen/MFA/外部ASR、不使用GPU、不开始训练。
- 同一文件读取前后size/mtime须稳定；ES Noah已有文件SHA须匹配。训练ID/路径
  重复、train/heldout路径相交直接停止；训练字节重复、train/heldout字节重复、
  validation/test保护字节相交均使结果blocked_audio_conflicts，不自动删除/补抽。
- 全部输入SHA在音频检查结束后再次校验。成功才输出per-language train和按累计
  小时交错的combined train；字段与现有full-ctc-v1 feature/training parser兼容。
  无重复采样；新dataset_version=en-es-pt-480h-fixed-plan-v1。

限度：文件SHA不能识别重编码/重叠片段；葡语清单缺speaker，不作跨来源说话人隔离
保证。保护超集若包含重复的未释放行亦会保守阻断。身份检查不认证标签/完整音频
解码质量；训练特征缓存与模型训练仍属后续步骤。

### 工作区执行

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/freeze_multilingual_480h_training.py
```

这是全量文件读取任务，比前一步只读Manifest慢。不要并发重复运行；可用--workers
调整1..32线程。自动新建outputs/multilingual_480h_training_v1_<随机后缀>，拒绝
任何已有目录，无resume；失败保留产物，下次新目录，不删除覆盖旧outputs。
前置输入/保护验证失败时可能尚未建目录；建目录后的异常会写FAILED.txt，含该
标记或无完成report的目录不能用于特征或训练，即使已留下一部分清单文件。

成功输出full_ctc_train.jsonl、full_ctc_train_en/es/pt.jsonl、report.json、
input_config.json、input_identities.json、audio_fingerprints.jsonl、sha256.txt。
冲突结果保留各语言pending清单，不发布正式train，退出码2。成功只代表所述正式
train清单和身份检查通过，training_started仍false；预计958,513条/1440.003175h，
必须由用户H200实际回传确认，不能拿本地测试冒充完成。

R设为实际新目录，验证本次产物（不读sealed test Manifest）：

```bash
(cd "$R" && sha256sum -c sha256.txt)
```

返回终端JSON和sha256.txt或小report.json即可，不传音频/大Manifest/fingerprints。
如blocked，只需聚合原因与自动Top10例子；先处理冲突，不直接跑训练。通过后再
安排训练配置、固定评测引用与新三语Encoder特征提取。

本地验证：新增9项测试含端到端合成freeze、真实训练Manifest解析器兼容、跨语言
字节重复、validation/test字节冲突、计划/来源身份变化、PT恢复规则与分区统计、
输出保护；相关29项pytest通过。全量340 passed/23 skipped；新增Ruff/strict Mypy、
CLI help、git diff --check通过。全仓仍为5处既有E501和3处既有unused-ignore，
位置与0.118一致，未改用户文件或其他outputs。本地测试未下载或加载任何模型。

## 0.119 2026-09-23 英葡480h分层计划完成（用户返回）

回传附件SHA256=b5b9f5a5c04fd5231b7ec01fca7189d3327bb3f1894c9d12e527b65f1270f739。
首次命令缺少脚本，后续已成功执行；不把首次旧代码报错视为选数失败。
目录outputs/en_pt_480h_plan_v1_ac14317963，status=plan_completed，
training_ready=false；尚未生成最终训练清单、读取音频或运行模型。

| 语言 | 所选条数 | 小时 | original-ready小时 | recovery小时/占比 |
|---|---:|---:|---:|---:|
| en | 340,380 | 480.000690047 | 479.978894406 | 0.021795642 / 0.004541% |
| pt | 321,734 | 480.000662416 | 346.351826257 | 133.648836159 / 27.843469% |

PT原783h池recovery占27.844598%，所选与原池相差-0.001129个百分点，属于按
原池构成自然得到的比例，不是强制设置27%。PT来源选样：CV=14.752036h、
FLEURS=4.042533h、MLS=65.152635h、Noah500h=284.215124h、金融=111.838334h。
各维度条数和小时加总与总数一致。按小时比例比较，PT单bin最大绝对偏差：来源
0.001904pp、release 0.001129pp、时长0.002515pp、density 0.001092pp、ratio
0.001129pp。EN对应最大偏差均小于0.000423pp。紧凑回传不含联合stratum明细，
不能据此声称逐联合层quota已独立复核；冻结入口仍需核验完整report和选中ID。

EN所选保留全部1,560名已知speaker、缺失speaker=0；PT全部321,734条在此Manifest
缺失speaker字段，不能声称speaker-disjoint，也不等于所有原始来源都没有speaker
元数据。暂不据此修改划分或改选配比。

```text
proposed_ids_en.jsonl 216d18d24fe4b6d3520def5a1a25840ab7e57491ad6b528163939cc1620b593e
proposed_ids_pt.jsonl 791e3f3a8a43c90fc2a8de21723e09d3252e9a4aa41be8b1adf54538a7942d95
```

这些为H200程序回传身份，本地未取得ID文件独立重算；已核对ES引用SHA仍为0.116
版本，EN/PT heldout引用仍为0.117版本。加上既定ES=296,399条/480.001822655h，
计划总计958,513条/1440.003175118h。总量不是模型效果，也不是音素准确率证明。
下一步固定这些ID做机械音频身份/留出保护检查；不重抽、不调整密度、不开始训练。

## 0.118 2026-09-23 英葡各480h分层ID计划工具（待H200执行）

新增scripts/prepare_en_pt_480h_plan.py、training/en_pt_480_plan.py及
configs/en_pt_480h_plan.workzone.json。输入绑定0.117英葡train/validation/test
Manifest SHA与train条数/小时，另绑定0.116西语正式train的SHA/条数/小时。
交付分支codex/g2p-coverage-scan；0.117实测库存记录为独立结果提交d669f33。

本步生成待核对ID计划，不生成可直接训练的Manifest，不修改核心算法：

- 英语只从已释放US Swift train中选择；葡语只从已释放五来源train中选择。
  校验summary pass及test sealed/unused、train实际SHA、语言/来源、Temporal 2x、
  唯一ID/解析路径、逐来源和总体条数小时；任意异常阻断，不静默跳过。
- 复算label_length、相邻重复音素对应CTC minimum、effective frames及ratio，
  检查original-ready的1x可行性和recovery既有2x/0.90规则。沿用已有音素标签。
- 复用西语计划的stratified_select：source×release×时长×density×ratio联合分层，
  按可用小时比例分配quota；层内SHA(seed,ID)确定顺序，补足剩余quota至480h。
  seed=20260923。全音频不切割，最后一条可产生少量超时，不重复采样凑小时。
- density=L/(2*T_est)，CTC ratio=(L+adjacent_repeats)/(2*T_est)，分别报告，不能混用。
  时长bin边界3/6/10/20秒，density .2/.3/.4/.5/.6，ratio .5/.75/.9，上界包含。
  这是Manifest估计帧口径，不宣称等于未来真实Encoder cache帧数。
- 按现有PT来源小时计算的约数目标：Noah500h=284.22h、金融=111.84h、MLS=65.14h、
  CV=14.75h、FLEURS=4.04h。实际值受整条音频和分层quota影响，以回传为准。
  不固定recovery为27%，不按validation PER挑选，不强行匹配三语密度；按比例保留
  是控制域分布变化的起点，不是质量最优保证。英语约保留98.44%的现有train小时。
- 西语不重抽、不重写，仅核验已冻结SHA及全部ID/路径与新选英葡的交集；其正式
  清单仍由0.116路径直接引用。跨语言选中ID/解析路径重复一律阻断。
- 报告选前/选后各维度条数与小时、逐层quota偏差、已知来源内speaker数量/Top5小时
  和缺失speaker条数。不杜撰speaker，也不宣称跨来源speaker-disjoint。
- 不打开validation/test Manifest，不读音频字节，不推理，不下载，不训练。
  从原train派生，保留原split；正式冻结前仍需完成所选音频身份/heldout保护检查，
  本步不重新认证音文/G2P标签准确性，不宣称已完成内容去重。

### 工作区执行与回传

容器外项目目录更新（Git在容器外执行）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/prepare_en_pt_480h_plan.py
```

仅CPU/Manifest读取，约扫描87万英葡候选行和29.6万固定西语行，不需要GPU。
自动新建outputs/en_pt_480h_plan_v1_<随机后缀>，拒绝任何已存在目录（包括空目录）。
没有resume；失败保留当前目录/FAILED.txt，重试使用新目录，不删除已有outputs。
成功输出proposed_ids_en.jsonl、proposed_ids_pt.jsonl、config.json、report.json、
sha256.txt；不会输出full_ctc_train.jsonl或训练缓存。输入SHA在结束前再次核验。
仅status=plan_completed且无FAILED.txt才是完整计划；training_ready仍为false。

终端打印剔除逐层明细后的紧凑报告；完整细分在report.json。返回终端JSON或小文件
report.json和sha256.txt即可，不要上传ID大清单/音频/模型。R设为实际输出目录后：

```bash
(cd "$R" && sha256sum -c sha256.txt)
```

回传后核对实际来源比例、density/recovery变化、speaker缺失和quota偏差，再固化
计划并做所选音频身份检查/生成三语正式清单，然后才安排特征提取和训练命令。

本地验证：新13项+复用西语8项定向pytest通过；全量331 passed/23 skipped；新增
Ruff与strict Mypy、CLI help、git diff --check通过。全仓Ruff仍5处已知E501
（含个人PPT脚本2处），全仓Mypy仍3处既有unused-ignore，均与0.115一致、未触碰。
这些为本地合成fixture验证，未执行H200真实选数，不能写成新英葡480h已完成。

## 0.117 2026-09-23 英葡完整train库存及实际SHA回传一致

用户按0.116返回两个split_summary的精简字段和sha256sum结果。两份实际train
文件SHA均与各自summary一致；该结论来自H200回传，本地未读取H200完整Manifest。

| 语言/来源 | train条数 | train小时 |
|---|---:|---:|
| en / swift_us_english | 345,820 | 487.628442229 |
| pt / 全部 | 525,189 | 783.223636631 |
| pt / common_voice | 21,028 | 24.074124 |
| pt / fleurs | 1,890 | 6.593617 |
| pt / mls | 25,085 | 106.295530 |
| pt / noah_500h | 343,286 | 463.766467 |
| pt / noah_finance_200h | 133,900 | 182.493898 |

```text
en train 6882074371edfdc94a5ca66ff4ba487c0f965c5f7527b0026740af9b952f9dc8
pt train 47214e4bec638198fd0e00c929f113d8ad35dedfd13a904f1190f903f68b3fd4
```

对应pool为outputs/en_us_swift_temporal2x_v1和outputs/pt_combined_temporal2x_v1。
英语超480h约7.628442h，葡语超480h约303.223637h；现有已释放train库存数量上足够。
尚不能把该库存确认写成新480h已冻结或新音频内容去重已完成。

原heldout引用不变：en validation=7,187条/10.165238235556h，test=7,086条/
10.165610728611h；pt validation=10,899条/16.239130696944h，test=10,958条/
16.232128349444h。回传summary包含test聚合字段，不等于读取test答案或进行评测。

后续基于各自train池做source×release×时长×density×CTC ratio按小时比例分层，
固定seed；不预设葡语recovery比例、不按PER筛选、不统一三语density。此比例是
尽量维持来源/压力构成的扩量起点，未经实验不能宣称最优。西语0.116正式清单整份
引用。保持现有validation/test，不增加外部ASR、不训练、不改Head算法。

## 0.116 2026-09-23 西语480h正式清单冻结成功（用户返回）

用户返回freeze_es_480h_training.py结果：status=completed，
training_manifest_ready=true；正式输出为
`outputs/es_480h_training_v1_00d26e5c2e/full_ctc_train.jsonl`。
回传Manifest SHA256：

```text
145c16e2019c4966c450557a150333eea19bcafec37f5eca36ee6e4d8815b300
```

本地核对回传条数、来源/分组小时及保护集合文件总数相加一致。该SHA来自工作区
程序回传，本地未取得或重算完整Manifest，也未收到最终目录sha256.txt独立回传。
不再将西语480h写成“只有候选计划”；正式清单已按0.115的门槛完成。

| 分组 | 条数 | 小时 |
|---|---:|---:|
| 全部train | 296,399 | 480.001822655 |
| original-ready | 263,195 | 440.216771076 |
| temporal-2x recovery | 33,204 | 39.785051580 |

recovery占小时8.288521%，与0.114计划一致。旧train约181.868358h、Noah约
298.133465h，7个来源/批次的条数时长与计划一致；无冲突触发剔除或补抽。

文件身份检查：

- checked_files=302,239，bytes_read=41,669,411,266；
- unique_train_file_hashes=296,399，恰等于训练条数；
- conflict_groups_by_reason={}、conflict_examples=[]；
- train_holdout_path_overlaps=0；
- 保护集合validation=2,943路径、test=2,897路径；
- 296,399+2,943+2,897=302,239，分区计数一致。

保护集合是source/speaker元数据导出的保守超集，不能将2,943/2,897误写成实际
validation/test样本条数。原西语validation仍2,871条/4.368367448611h，原test仍
2,814条/4.139065h（历史summary），引用未改变：

```text
validation SHA256 6d1d77ad4ffb26c40d48995ff5902fe98dc2a5a65de05f87fdc6787f040cca77
test SHA256 508f23cfa89533e6855d3545a706189a13208be8c83b0ed1fcae7a965f98e919
```

model_loaded=false、training_started=false、test_manifest_content_read=false、
test_evaluation_performed=false；test_audio_bytes_read_for_identity_only=true。
不把文件身份扫描称作测试模型评测。字节去重不覆盖重编码/重叠片段，Noah未知
speaker不作跨来源speaker-disjoint保证；也未新增发音准确率或完整解码质量认证。
Encoder cache和训练仍未开始；已就绪的是本版西语数据清单。

### 下一步：确认英、葡现有train库存与来源，再派生各480h

西语冻结版本应完整保留，不再交给旧三语随机采样器重新抽一次；旧采样器会重新
排序ES且mandatory全保留会因480.0018>480而拒绝。后续三语汇合要直接引用这份
西语清单。历史en US Swift train=487.628442h、pt train=783.223637h，尚未为本版
派生480h，不应把原完整池或旧150h特征缓存当作新480h已准备好。

用户在H200容器项目根目录读取最新现有摘要和实际train文件SHA（只读，无GPU）：

```bash
jq '{file:input_filename,status,split_records,split_audio_hours,
  train_sources:(.corpus_metrics|map_values(.split_train)),manifest_sha256}' \
  outputs/en_us_swift_temporal2x_v1/split_summary.json \
  outputs/pt_combined_temporal2x_v1/split_summary.json
sha256sum outputs/en_us_swift_temporal2x_v1/full_ctc_train.jsonl \
  outputs/pt_combined_temporal2x_v1/full_ctc_train.jsonl
```

返回两个小JSON和两行SHA。先按实际现有source train小时设计分层选择；不强制葡语
27% recovery，不删除全部高密度样本，不重复凑小时，不改变原验证集来选训练样本。
未新增CV/MLS语料，不再要求寻找西语来源；本轮仅结果型文档提交，git diff --check
通过。全部已有outputs、个人未跟踪文件保持不变。

## 0.115 2026-09-23 固定480h计划的文件身份隔离与正式清单入口

新增spanish_480_freeze.py、scripts/freeze_es_480h_training.py及7项测试。
默认输入为0.114的`outputs/es_480h_plan_v1_7022ece6f0`，不重新选样、不调整比例。
0.114用户粘贴附件本地SHA为
`3b665018be1f8e90803d528f639f9af915f11d0037490d96a8a237631e1236c4`。

### 身份检查与输出门槛

1. 核验plan report/config/proposed_ids清单SHA及plan记录的所有输入SHA，拒绝FAILED
   计划、输入变化、重复选择、条数/时长不一致。读取原train及Noah ready/review
   回填标签，逐条复核ID、路径、时长、密度、ratio和release可行性，先写pending清单。
2. 用旧pool已校验的split_config、speaker assignment和其中绑定SHA的source TSV
   建立validation/test保护路径集合。只使用来源表的audio/speaker/source_split列，
   不使用转写做判断；同一heldout speaker的未释放源行亦纳入保护，形成保守超集。
   验证所有实际validation路径在保护集合内、test保护数量不小于旧summary记录。
   不打开sealed test manifest、不解析测试音素或使用测试答案，不做测试评测。
3. 默认8线程、256文件一批，读取所有所选train及保护集合音频的完整文件字节计算
   SHA，约每5120文件打印进度；因此**会读取test保护音频的字节用于机械身份核验**，
   明确报告test_audio_bytes_read_for_identity_only=true。不是ASR/CTC推理或测试
   调参。逐文件检查读取期间size/mtime稳定；新Noah必须与staging文件SHA一致。
4. 任意train文件字节重复、train/heldout重复、保护集合validation/test之间字节
   重复均阻断。已知路径交集亦阻断。不自动删副本、不改heldout、不补抽样凑时长。
5. 所有身份检查完成且无冲突时，才把pending改名为full_ctc_train.jsonl，生成
   split_summary和report，记录最终Manifest SHA。保持480.001822655h固定计划。
   旧train标签/文本不改，仅统一recovery元数据别名并保存original_release_source；
   Noah使用管线兼容language=es并保留source_language=es-419、原source_id、batch、
   文件SHA和空speaker_id。统一Temporal 2x，不修改Head或特征提取算法。
6. validation/test不复制不重分，只在新summary引用原路径、SHA、条数和小时。
   成功仅表示本轮正式数据清单及所述文件身份检查通过，不代表已生成全量Encoder
   cache或训练已开始。最终train已通过现有三语pool读取器的本地格式兼容测试。

局限明确保留：字节SHA不识别重编码副本或同录音重叠片段；Noah speaker身份未知，
不能保证跨来源speaker-disjoint。metadata保护超集可能因尚未释放的原始样本冲突
而保守阻断，需根据返回证据处理，不悄悄放行。不认证每个G2P发音正确，也不新增
音频解码质量评测。测试集的既有引用保持封存，test_set_used=false表示未作模型
评测/调参，不能误读成从未读取任何test相关音频字节。

### 工作区运行

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录：

```bash
python -B scripts/freeze_es_480h_training.py
```

这是约480h所选音频加heldout超集的存储读取任务，无GPU、无下载、不复制音频。
默认写全新`outputs/es_480h_training_v1_<随机后缀>`，任何已有输出路径均拒绝。
可用--workers调整1..32并发，--plan-dir指定其他经过审查的计划。无resume；失败
保留所有当前产物，下一次使用新目录，不能删除/覆盖已有outputs。

成功输出full_ctc_train.jsonl、split_summary.json、audio_fingerprints.jsonl、
report.json、sha256.txt。冲突则report.status=blocked_audio_conflicts、
training_manifest_ready=false，保留train.pending.jsonl，不产生正式train或pass
summary。其他异常写FAILED.txt；带该标记或无完成report的目录不可用于训练。

验证（R取实际输出目录）：

```bash
(cd "$R" && sha256sum -c sha256.txt)
```

仅回传终端JSON和sha256.txt；不要传音频、fingerprints大清单或训练Manifest。
若冲突，返回聚合及Top10例子即可，之后根据证据修正/重选。本地测试不是H200实测，
当前仍待用户执行，不能记录为480h正式train已冻结。

验证：7项定向pytest通过；全量318 passed/23 skipped；新增文件Ruff、strict Mypy、
CLI help和git diff --check通过。全仓仍有5处已知E501及3处已知unused-ignore，均在
本轮未改文件，与0.113相同。未修改个人未跟踪文件或已有outputs。

## 0.114 2026-09-23 西语480h候选计划完成（用户返回）

用户返回prepare_es_480h_plan.py结果，目录
`outputs/es_480h_plan_v1_7022ece6f0`，status=plan_completed、training_ready=false。
本地解析回传JSON，核对所有source/release/duration/density/ratio维度的条数和小时
加总一致。验证集SHA仍为6d1d77ad4ffb26c40d48995ff5902fe98dc2a5a65de05f87fdc6787f040cca77，
封存test引用SHA为508f23cfa89533e6855d3545a706189a13208be8c83b0ed1fcae7a965f98e919；
这些是引用，不是本轮读取test或重新评测的结果。

| 组成 | 条数 | 小时 |
|---|---:|---:|
| 保留旧train | 125,379 | 181.868357963 |
| 新选Noah | 171,020 | 298.133464692 |
| 合计 | 296,399 | 480.001822655 |
| 其中original-ready | 263,195 | 440.216771076 |
| 其中temporal-2x recovery | 33,204 | 39.785051580 |

超过目标约6.562秒，来自保留最后一条完整音频，不是重复凑时长。
recovery约占8.29%小时；旧train中为9.227010479h，新选Noah中为30.558041101h。
配比由来源和压力分层自然产生，并未要求27% recovery。

Noah四批次选样小时：APY170801048=109.708238121，APY161101034_R=56.362903377，
APY161101034_G=2.122424702，APY181231012=129.939898493。原SLR61/CV Rioplatense/
拉美CV辅助train全部保留，不把480h全替换为手机采集域。

组合密度L/(2*T_est)的小时分布：<=0.2为6.948353h，(0.2,0.3]为91.605049h，
(0.3,0.4]为219.088801h，(0.4,0.5]为126.433033h，(0.5,0.6]为31.269245h，
>0.6为4.657341h。CTC ratio包含相邻重复音素，不能与density混用；其(0.75,0.90]
仅124条/0.099889h。密度不是标签准确率，本次也没有PER结果。

下一步对固定计划进行文件内容去重与heldout身份保护；通过后才能生成最终训练
Manifest。Noah的speaker身份仍未知，不作speaker-disjoint承诺。不得因为小时已够
就跳过report.pending，也不得用其他源或重复样本自动补任何后续剔除量。
本节独立结果提交，仅文档更新；git diff --check通过。

## 0.113 2026-09-23 西语480h固定候选ID与分层配额计划入口

0.112旧train/validation范围复核通过后，新增plan-only入口，不调用要求已验证speaker
的旧切分器，不伪造Noah的speaker_id。新增config、spanish_480_plan.py、CLI及8项测试。

规则：

1. 固定0.112旧summary/train/validation SHA、0.110最终词典与vocab SHA、Noah源JSON
   身份。核验staging文件清单、旧train/validation数量时长及summary记录SHA；新清单
   记录各输入SHA并检查结束时未变化，作为本次计划的可复现身份。
2. 逐条以规范化绝对audio_path回连candidates旁表，验证文本/时长一致、完整分区，
   保留原source_id、批次、文件SHA、directory_group_hint及真实或空speaker_id。
   拒绝旁表重复路径/ID/文件SHA，拒绝新清单重复/ID碰撞及与旧train/validation路径交集。
3. 原ready必须无issues且1x可行；仅ctc_length_infeasible的review在2x ratio<=0.90时
   入候选。重算标签长度+相邻重复需求，验证token ID在1..89范围，不使用部分缺词
   标签。其他review不入选但保留原产物；不更改原文、词典、模型或release规则。
4. 旧train全部保留，Noah补足480h差额。Noah按batch×release×时长×ref/frame×CTC
   ratio联合分层，以每层可用小时占比确定配额，seed=20260923，层内按SHA(seed,ID)
   排序无放回选样。先不超过各层配额，剩余差额逐条补给小时缺额最大的层，直至目标。
   不裁切音频，数学上总量达到目标且超量小于最后一条音频时长（浮点误差容差1e-8秒）。
5. 分箱为上界包含：时长3/6/10/20秒，density=L/(2*T_est)边界0.2/0.3/0.4/0.5/0.6，
   ratio=(L+重复)/(2*T_est)边界0.5/0.75/0.9；这里是metadata估计帧，非实际缓存帧。
   报告完整联合分层及边际小时分布，不人为固定27% recovery，不读取模型预测选样。

生成proposed_ids.jsonl、report.json、config.json、sha256.txt，**不生成带训练标签的
full_ctc_train.jsonl**。报告status=plan_completed、training_ready=false。未读取任何
音频、模型或sealed test manifest；旧validation和test只保留原有身份，validation内容
仅供路径/ID机械核对。Noah已有文件SHA内部唯一，但新旧池跨路径文件内容去重尚未做，
sealed holdout的完整身份保护尚待补齐；speaker-disjoint亦不能因G目录而宣称通过。
这些都是report.pending明示的后续条件，计划不能直接供训练使用。旧train如有后续
内容重复/污染问题须重新处理并重选，不能因本阶段保留就免检。

### H200交付

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录（无需GPU/新依赖/下载）：

```bash
python -B scripts/prepare_es_480h_plan.py
```

默认读取configs/es_480h_plan.workzone.json，写全新
`outputs/es_480h_plan_v1_<随机后缀>`；显式--output-dir也拒绝任何已有路径。无resume，
新一次运行使用新目录；输入验证不通过则不创建目录，写出期间失败保留部分文件并
写FAILED.txt，禁止当成功计划使用。既有outputs、Manifest与用户文件不修改。

验证（R为程序实际打印的计划目录）：

```bash
(cd "$R" && sha256sum -c sha256.txt)
```

返回终端JSON及计划sha256.txt即可；终端省略大的联合分层和输入身份列表，但完整
内容保存在report.json。需要详细分布时仅回传该小JSON，不传原始Manifest或音频。
本地mock不是H200结果，实际选择条数、四批次配额、整体recovery比例待用户回传。

本地验证：定向8 passed；全量311 passed/23 skipped；新增Python文件Ruff/Mypy
strict、CLI help及git diff --check通过。全仓Ruff仍为0.101所述5处既有E501（包括
用户PPT临时脚本2处），全包Mypy仍3处既有unused-ignore；本轮未触碰这些文件。

## 0.112 2026-09-23 旧西语train/validation拉美范围复核通过（用户返回）

用户返回audit_es_480h_capacity.py结果，scope_issue_count=0。
train=125,379条/181.868357963056h，validation=2,871条/4.368367448611h。
SLR61/Rioplatense沿用已批准来源范围，CV辅助池逐条回连明确拉美元数据；这不是
声学口音分类或每条转写准确率证明。

| 来源 | train小时 | validation小时 |
|---|---:|---:|
| SLR61 | 7.532729 | 0.239668 |
| CV Rioplatense v26 | 14.899401 | 0.437847 |
| 明确拉美CV辅助池 | 159.436228 | 3.690852 |

回传身份：

```text
split_summary dfaaddc82ab8d35fb7aaf18ab818ad39523e6a17ad26aaf8fa7db9fdc7776040
speaker_assignments a027a59c4cd76351507c297e6e7ec8a9bc99bd530887aaab196339f205b6a510
train 617ae3723527dcdfd4c04ea7799bb68c7cfd13cb4cdd8386758d5aa31a9faa25
validation 6d1d77ad4ffb26c40d48995ff5902fe98dc2a5a65de05f87fdc6787f040cca77
CV inventory 77b3e3c65fc2bbe5d9d1af2c6fb45c7539de4e3e7d2c4c228226a384448d390e
MLS summary 6624aace27fe771b4e96b232e527b4bec00350b95ecb3ccbd6061f446c75a636
```

工具未读取test manifest内容、音频或模型，无文件写入。其insufficient_raw_capacity
指旧来源：181.868358h现有train+73.405812h未完成标签的CV原始候选=255.274170h。
该工具未计Noah新增496.274765h；不能将224.725830h旧缺口误报成当前仍缺数据。
新旧候选合计约678.143123h，当前计划保留旧train，从Noah补298.131642h。
新增CV 73.405812h及无拉美依据的MLS不纳入本轮，避免无必要扩大来源准备范围。

下一阶段量化batch/时长/密度分层选择和实际recovery比例，保留旧验证集身份。
现有speaker切分器要求speaker_id，不得伪造Noah speaker以强行复用；先生成明确
标为plan-only的固定候选ID清单和配置统计，后续补齐跨池内容去重/heldout保护后
才生成最终训练Manifest。不把候选计划宣称480h训练已交付。
本节为独立结果提交，仅文档修改，git diff --check通过。

## 0.111 2026-09-23 Noah逐句标签/CTC容量通过：496.274765h候选（用户返回）

用户返回0.110的full-manifest和temporal2x审计输出。粘贴附件
`da4250a7-a40e-49ea-967d-ff75f30a921c/pasted-text.txt`本地SHA256为
`43583347bbd81302bc057fee7c69dbbe7a5d51b6d19b0919eca47b17e9cace05`。
本地解析两个JSON并核对样本分区和小时加总；这是回传附件身份，不是工作区原始
summary的SHA。源staging及最终词典输出的SHA检查均由用户回传为全部OK。

### 实测结果与结论

`outputs/es_noah_mobile_full_manifest_v1`：58个分片，510.95685秒，未resume，
286,821条音频metadata全部可读，总499.999970062778h，全部split=unsplit。
`outputs/es_noah_mobile_temporal2x_audit_v1`：耗时3.27007秒，factor=2、q上限0.90。

| 互斥分组 | 条数 | 小时 | 本轮处理 |
|---|---:|---:|---|
| original-ready | 241,948 | 445.400484 | 可进入后续隔离/选样 |
| 纯时间问题，2x且q<=0.90 | 42,702 | 50.874281 | 可恢复候选 |
| 2x可行但0.90<q<=1 | 10 | 0.007336 | 暂缓，保留 |
| 2x仍不可行 | 119 | 0.080818 | 待检查，保留 |
| 存在其他问题 | 2,042 | 3.637051 | 待检查，保留 |
| 合计 | 286,821 | 499.999970 | 不删原始记录 |

ready+review=241,948+44,873=286,821，review子分组合计一致。原始issue_counts为
ctc_length_infeasible=42,876、dictionary_missing=2,124、standalone_h=131；这些是
issue计数，可能同句多词/多原因，不能把2,124直接当作独立缺词句子数，更不能与
其他issue简单相加当坏样本数。实际other-issue分组2,042条/3.637051h。

按既定规则可用候选=284,650条/496.274765h，其中recovery占小时10.251233%。
这是新Noah候选池的观察比例，不是三语或最终西语的固定配额。保持现有temporal2x
结构及q规则，不因有足够original-ready就删掉全部recovery。

与历史旧西语train 181.868358h相加，隔离/交叉去重之前的账面容量为678.143123h，
高于480h目标198.143123h。保留旧多来源train时，只需从Noah新增298.131642h；
当前无需用户再寻找西语语料。尚未创建最终480h，也未证明跨池唯一或speaker-disjoint。
不把候选数当训练结果，不声称本次已改善PER或核实了每个音素的真实发音。

### 下一步从容量检查转向480h选集准备

计划优先保留已合格旧train以维持来源多样性，再从Noah补足298.131642h；以四个
批次、时长、ref/frame与CTC ratio分层，纳入合法recovery，不按模型预测选样。
实际配额在交叉去重/隔离后量化；Noah的directory_group_hint不是已验证speaker。
既有严格拉美validation/test保持独立，不为凑train小时重分；未来若另建Noah域
validation，先隔离再选train，使用新版本清单。

当前还没收到0.101的旧西语train/validation范围复核结果。因此下一条短命令复用
已有工具确认旧池身份和明确拉美元数据，不重新要求用户证明新Noah口音，也不再
寻找MLS/CV来填容量。本工具只读，不加载模型/音频，不读取sealed test内容。
容器内项目根目录运行，无需为本轮文档更新pull：

```bash
python -B scripts/audit_es_480h_capacity.py
```

返回终端JSON即可。该工具早于Noah新增池：它报告的容量/shortfall仅限旧来源，
不会计入刚确认的496.274765h；若报告旧来源不足，不代表当前总容量不足。本轮
关注existing train/validation身份、existing_latam_scope和scope问题，而非旧缺口。
范围问题若被发现先明确报告，不自动移除或改写旧数据。现有outputs保持不变。

取得结果后，下一实现阶段应保留候选旁表的来源/文件SHA/原ID，构建新旧池身份
交叉检查和480h选择工具；使用新的输出目录，报告实际来源、recovery、密度、去重
和留出集保护结果。不能把旧speaker切分器直接套给speaker_id为空的Noah。
H200执行仍由用户完成，训练继续等待完整数据和配置就绪。

本次仅结果型HANDOFF更新；本地已核对回传JSON计数、比例及小时分区，git diff
--check通过；未修改代码、核心算法、outputs或用户未跟踪文件。

## 0.110 2026-09-23 Noah代理G2P与最终词典审计结果（用户返回）

回传附件`7cc747c8-3378-4c86-8666-bb745954f9d8/pasted-text.txt` SHA256为
`9897a482054be5d33854cb96064a816e7d3aafc87e2e56a99044d0878ea4e28d`；这是
本地粘贴附件身份，不冒充工作区report身份。用户返回repair_plan的6项SHA检查
全部OK，代理G2P耗时152.946秒，finalize和audit均执行完成。

最终词典为
`outputs/es_noah_mobile_mfa_repaired_v1/final/noah/noah_spanish_latin_america_repaired.v1.dict`。

```text
final dictionary SHA256 0a23386028849772672d661fd9fa685a0f89fad7c041a9dcc6d46c41146e3269
proxy dictionary SHA256 0ab0f8c851f6385700c566cf1193cbdde9f502c17319722176b628be5e5d7116
repair plan SHA256 79d92ca89d7b28fcfe9aa8ddb8fe9f4a72d8e41a144669d0e97de70108fd13a9
vocab SHA256 0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4
```

base_dictionary/words/word_counts的回传SHA与0.109一致。
最终68,930个词、68,930条单发音；missing=659，duplicate=0，extra=0，写入词典
的phone OOV=0、weighted OOV=0，CTC输出90类。词类型覆盖99.0530113%，词次
覆盖99.9440301%；未解决2,136词次，占0.0559699%。status=pass仅表示执行成功，
training_labels_ready=false仍正确，不能称整批标签已经齐全或发音已人工核验。

本次mfa_proxy成功10,474词/194,063词次，较计划少128词/299词次，故未解决从
531增加到659。已输出词典OOV为0不代表所有原始发音无OOV；finalize会隔离失败项。
沿用旧规则删除26,702个combining tilde、还原728个enye代理glide；这些是处理计数，
不是新增发音质量证据，不扩展为葡语或全局规则。

返回Top20中包括g(207)、h(134)、barça(57)、abad-lima(34)、resultados-futbol(29)、
guinea-bis(20)、são(14)、schäuble(11)及多种连字符词。大多detail=no_safe_proxy；
guinea-bisáu(14)为missing_proxy_pronunciation。尚无全量原因分布，不能说全部是
连字符或坏标注；不删除单字母、不自动拆词、不改写外来专名来凑100%覆盖。

### 下一步：逐句候选Manifest与Temporal-2x容量审计

现阶段无需强行解决所有659词才能检查容量。复用full-manifest builder，将每条
源记录保留在ready或needs_review：存在任何缺词的整句保持review，禁止删除缺词后
将部分标签作为完整训练答案。此为全池候选清单，不是最终train；split=unsplit。
这取代早期“全词表missing=0才允许任何Manifest扫描”的过强限制，但不放宽最终
入选句子的完整标签要求。

保持既有西语allow-exact-dictionary-connectors：仅词典提供唯一、无OOV的完整
发音时允许连接词；不会修复缺失的连字符词。先记录原1x可行性，再独立计算2x
纯时间压力恢复，沿用q=(L+相邻重复数)/(2*T_est)<=0.90的候选阈值；这不是要求
固定recovery占比，也不是ref phoneme/frame密度。所有非时间问题继续阻止恢复。
q>0.90但<=1的困难项仅计为deferred，不删除。时长/帧来自音频metadata估计，
不能冒充之后Qwen特征缓存的精确有效帧数。

此扫描重新读取音频header，不重新哈希全部音频、不复制音频、不加载Qwen。
旧builder仅保留source.tsv行号/路径，批次、文件SHA、原始source_id和分组提示
仍在不可变candidates.jsonl；后续选样必须按唯一audio_path回连并验证一对一，
不得把Gxxxx当已验证speaker或丢掉这份来源旁表。本轮不生成最终480h，不检查
现有holdout交集，也不读取sealed test，容量不是最终train承诺。

用户在容器内项目根目录运行以下块，无需为本次文档更新pull：

```bash
(
set -e
R=outputs/es_noah_mobile_source_v1_4bbc7edd5d
F=outputs/es_noah_mobile_mfa_repaired_v1/final
B=outputs/es_noah_mobile_full_manifest_v1
A=outputs/es_noah_mobile_temporal2x_audit_v1
V=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
(cd "$R" && sha256sum -c sha256.txt)
(cd "$F" && sha256sum -c sha256.txt)
test ! -e "$B"
test ! -e "$A"
python -B scripts/build_full_training_manifest.py \
  --tsv "$R/source.tsv" --audio-root /host_home \
  --dictionary "$F/noah/noah_spanish_latin_america_repaired.v1.dict" \
  --vocab "$V" --output-dir "$B" --language es-419 \
  --dataset noah_es_mobile --id-prefix noah_es_mobile_row \
  --split unsplit --allow-exact-dictionary-connectors --workers 16
python -B scripts/audit_temporal2x_recovery.py \
  --corpus "noah=$B" --output-dir "$A" \
  --time-upsampling-factor 2 --release-max-effective-ratio 0.90
)
```

返回两个终端summary JSON及SHA检查是否全部OK；不传Manifest或音频。重点核对
source_records=ready+review=286821，总时长约499.999970h，以及original_ready、
recommended recovery、other-issue blocked和deferred的数量/小时分区。issue计数
可重叠，不能简单相加当作坏样本数。完整分组结果保存在A/noah.json；需要时只取
其issue_analysis.reason_totals等小聚合，不索取大文件。

首次必须使用不存在的B/A目录。builder支持相同输入SHA/config的分片resume：若
中断，重新设置相同变量后仅重跑上面的builder命令；它会验证配置和已完成分片，
不修改输入或别的outputs。完成后在A尚不存在时执行audit。若audit已完成，读取
结果，不覆盖重跑；改变标签/配置须新建版本目录。

本次只更新结果和下一步命令，未改代码或算法；git diff --check通过，未触碰用户
未跟踪文件。主线仍为准备480h唯一、严格拉美西语train及独立validation。

## 0.109 2026-09-23 Noah独立G2P完成及残余修复计划（用户返回）

用户执行0.108命令完成。源staging的sha256.txt列出的11项文件全部OK，包含
report、inventory、candidates、source.tsv和词表。此为用户返回的工作区校验记录，
不声称本地持有并重新计算这些原始文件。

```text
G2P模型 SHA256 58e0743cf364d8aa2cc8b419e7c95dd9127908268af670f305722c601f726b2a
words SHA256 6eb90c4e058f04f0dd1d52a072420e72cba9efadca14c4454d64e5be86bdf1a2
word_counts SHA256 c2237dfd169d204a1fafe19157a94758e2dd59033343bd0a22a583c50b96d3f7
noah_raw.dict SHA256 20875b3e68debb6ea140227187df5c917f4dc5935bc0cb7593d9167536d9eeed
```

模型为现有spanish_latin_america_mfa.zip，top-1、16 jobs，日志耗时926.608秒。
原始词典位于`outputs/es_noah_mobile_mfa_v1/noah_raw.dict`，新计划位于其同级
`repair_plan`目录。词表/词频SHA与0.108返回一致。

| 分类 | 不同词数 | 词次 |
|---|---:|---:|
| base_exact | 56,226 | 3,349,855 |
| base_proxy | 2,230 | 270,285 |
| mfa_proxy | 10,602 | 194,362 |
| unresolved | 531 | 1,837 |
| 合计 | 69,589 | 3,816,339 |

去重代理词10,522个。计划词次覆盖99.9518649%，尚未解决占0.0481351%。
相比旧CV词典计划，unresolved从21,088降到531，支持旧词典覆盖范围是之前大量
unresolved的主要原因；这不是音频/文本标注修正或真实发音准确率提高的证据。
仍为training_labels_ready=false；代理尚未跑、最终词典尚未做vocab审计，不能把
计划覆盖当最终覆盖或小时保留率。现有531词会被finalize保留为unresolved，代理
G2P失败或phone OOV还可能增加该数，不能期待本轮自动降到0。

终端进度行停在82%、分母69,588，但MFA之后返回Done且prepare成功；进度刷新
不作为完整性证据，真实覆盖以词表集合、逐词计划及下一轮审计为准，不假定少1词
无害或凭进度条重跑全库。

### 下一步命令（不改代码，不需为此重新pull）

沿用原西语代理/phone cleanup/vocab规则。用户在H200容器项目根目录执行；仅G2P，
不运行Qwen，不训练，不下载。新输出根F必须不存在，失败保留文件；中断后使用
新的v2目录重跑此小阶段，无resume，不覆盖原词典或计划。

```bash
(
set -e
R=outputs/es_noah_mobile_source_v1_4bbc7edd5d
G=outputs/es_noah_mobile_mfa_v1
F=outputs/es_noah_mobile_mfa_repaired_v1
M=models/mfa/g2p/spanish_latin_america_mfa.zip
test -f "$M"
(cd "$G/repair_plan" && sha256sum -c sha256.txt)
test ! -e "$F"
mkdir "$F"
conda run --no-capture-output -n aligner mfa g2p \
  --num_jobs 16 --num_pronunciations 1 \
  "$G/repair_plan/proxy_words.txt" "$M" "$F/proxy.dict"
python -B scripts/finalize_spanish_mfa_repairs.py \
  --corpus "noah=$R/wordlist" --dictionary "noah=$G/noah_raw.dict" \
  --repair-root "$G/repair_plan" --proxy-dictionary "$F/proxy.dict" \
  --output-dir "$F/final"
sort -k2,2nr "$F/final/noah/unresolved_words.tsv" | head -20
)
```

返回终端finalize/audit汇总JSON、最后20行高频未解决词（字段word、corpus_count、
proxy、rules、detail）及校验是否全OK；不传大词典/音频。新目录内保存输入SHA、
词典SHA、逐词缺失原因和audit报告。status=pass仅表示程序完成，仍需检查
training_labels_ready和missing/OOV。后续先区分残余词原因并统计影响样本/小时，
不删掉句内缺词来伪造完整标签，不直接宣称整批500h可训练。
主线仍为标签准备、CTC可用小时、cross-split隔离/去重，再选择西语480h train。

本次仅结果型HANDOFF提交；git diff --check通过，既有代码、outputs、个人未跟踪
文件均未修改。本地与origin交付分支检查一致后更新并推送。

## 0.108 2026-09-22 Noah旧词典增量计划结果与适用范围纠正（用户返回）

用户返回`outputs/es_noah_mobile_g2p_plan_v1`的prepare结果。69,589个词、
3,816,339个词次，与0.107源词表规模一致；status=pass仅表示计划生成成功，
training_labels_ready=false。

| 计划分类 | 不同词数 | 词次 |
|---|---:|---:|
| base_exact | 35,669 | 3,273,619 |
| base_proxy | 1,920 | 227,616 |
| mfa_proxy | 10,912 | 237,031 |
| unresolved | 21,088 | 78,073 |

去重后proxy_words_for_mfa=10,826。planned_corpus_token_coverage=97.9542436%，
unresolved词次占2.0457564%。这是词次的计划覆盖，不是已通过音素审计的覆盖，
也不是可训练音频小时比例或人工标注准确率。不同原词可能共享代理，所以10,912
与10,826不矛盾。

用户报告的输入SHA：

```text
base_dictionary 341fee8513d745abef2dc47dcd071bceb6151e01e633c0de696e83643db972f6
word_counts c2237dfd169d204a1fafe19157a94758e2dd59033343bd0a22a583c50b96d3f7
words 6eb90c4e058f04f0dd1d52a072420e72cba9efadca14c4454d64e5be86bdf1a2
```

这些是回传JSON记录身份；仍未收到上一阶段report文件SHA/校验清单执行结果，
不声称已在本地检查H200产物字节。

代码复核发现0.107的复用方式不完整：spanish_g2p_proxy在没有acute/ñ/gü代理规则
时返回no_safe_proxy。因此一个旧CV词典不含的普通新词也可能进入unresolved。
这个脚本原本面向已经对本语料运行MFA之后的残余拼写问题，不是通用跨语料缺词
补全器。现有汇总未按detail统计，不能断言21,088词全部属于这一原因，更不能据此
删除相关音频或称为错误标注。保留原计划，停止把它当作完整增量待跑词表。

### 下一步：本语料原始词表先跑MFA，再准备既有修复流程

使用已在H200存在的spanish_latin_america_mfa.zip和既有top-1 G2P流程，直接处理
Noah完整69,589词表；这会重新生成部分已有词的发音，但不需新合并器，也不会遗漏
普通新词。保留同一模型、文本规范化和后续修复策略，不更改核心算法。
无Qwen加载/训练，无下载，无音频重扫描；仅用户在H200执行MFA G2P。

容器内项目根目录，以下整体小块在子shell中执行，任一步失败即停止；输出根G
必须不存在，旧产物不覆盖。若目录已存在/中断，保留该目录，改用v2等新目录，
不提供未验证的resume。现有脚本无需为本次文档提交重新pull。

```bash
(
set -e
R=outputs/es_noah_mobile_source_v1_4bbc7edd5d
M=models/mfa/g2p/spanish_latin_america_mfa.zip
G=outputs/es_noah_mobile_mfa_v1
test -f "$M"
(cd "$R" && sha256sum -c sha256.txt)
test ! -e "$G"
mkdir "$G"
sha256sum "$M" "$R/wordlist/words.txt"
conda run --no-capture-output -n aligner mfa g2p \
  --num_jobs 16 --num_pronunciations 1 \
  "$R/wordlist/words.txt" "$M" "$G/noah_raw.dict"
python -B scripts/prepare_spanish_mfa_repairs.py \
  --corpus "noah=$R/wordlist" \
  --dictionary "noah=$G/noah_raw.dict" \
  --output-dir "$G/repair_plan"
)
```

用户只需返回最终prepare JSON、打印的模型/词表SHA及SHA检查是否全部OK；若失败，
返回错误尾部。模型缺失则停止，另行准备资产，不在此命令内下载。
随后根据该独立词典的残余问题决定proxy增量，再finalize和vocab审计。
完成逐样本CTC可行性、交叉去重及holdout隔离后，才能判断新增298.13h是否满足。
本次仅结果型文档提交，git diff --check通过；无代码/outputs修改。

## 0.107 2026-09-22 Noah新增西语全量库存实测约500小时（用户返回）

用户执行`python -B scripts/prepare_noah_es_source.py`完成，返回终端report JSON。
实际输出为`outputs/es_noah_mobile_source_v1_4bbc7edd5d`，status=completed，
scope=source_inventory_and_wordlist_not_training_manifest。源JSON SHA仍为
`3a57177178368b1eae0ce0be659071e955edcde56a1227c830192a0adc90b93f`。
这是用户返回的工作区实测；本地核对计数、小时数加总一致，但尚未收到工作区
report.json的独立SHA或sha256.txt，不能声称本地已核验原始产物字节身份。

| 批次 | 候选条数 | 实测小时 |
|---|---:|---:|
| APY161101034_R | 45,018 | 94.404404 |
| APY170801048 | 79,586 | 183.450922 |
| APY161101034_G | 3,013 | 3.547695 |
| APY181231012 | 159,204 | 218.596950 |
| 合计 | 286,821 | 499.999970 |

全部音频metadata可读，采样率均16kHz；286,821个不同文件SHA，无同文件文本冲突，
review_records=0。所有记录有文本，共3,816,339个word token、69,589个unique word，
数字fragment=0。source.tsv及wordlist已生成，不必重新运行全量音频扫描。

结论：新来源的原始候选时长确实约500h，而不是把目录标称小时相加。按现有严格
拉美西语train 181.868358h，距480h仍需新增298.131642h；本候选池最终保留约
59.6263%即可满足数量要求，但这不是最终可训练时长承诺。

仍未完成音素标签、CTC可行性、split分配、与既有holdout的交叉去重。
音频检查为header和文件字节读取，不是完整解码或标注准确率认证；文件SHA去重
不覆盖重编码和同录音重叠片段。speaker身份仍未确认，Gxxxx只作分组提示。
保持training_ready=false，不启动训练，不修改原有验证集或outputs。

### 下一步：复用既有西语词典，生成增量G2P计划

复用已有`prepare_spanish_mfa_repairs.py`，保持旧西语文本/代理规则不变。
先使用旧Common Voice Spanish原始MFA词典，统计本次词表的exact/proxy覆盖和
需要补跑的proxy词数；原始词典可能有既知U+0303问题，最终必须经过既有finalize
及vocab audit，不能把prepare报告的计划覆盖当作最终音素质量通过。
本步不运行MFA、不下载模型、不读取音频或test，也不新增代码。

容器外同步（分支`codex/g2p-coverage-scan`）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内项目根目录，分段执行以下短命令；SHA检查失败时不要继续：

```bash
R=outputs/es_noah_mobile_source_v1_4bbc7edd5d
(cd "$R" && sha256sum -c sha256.txt)
sha256sum "$R/report.json"
```

```bash
D=outputs/es_candidate_train_sources_v1/common_voice/mfa_g2p
P=outputs/es_noah_mobile_g2p_plan_v1
test ! -e "$P" && python -B scripts/prepare_spanish_mfa_repairs.py \
  --corpus "noah=$R/wordlist" \
  --dictionary "noah=$D/common_voice_spanish_latin_america_mfa.dict" \
  --output-dir "$P"
```

新目录P保留prepare_summary.json、proxy_words.txt、逐词repair_plan及SHA清单。
无resume；如果P已存在，不覆盖也不删除，改用新版本目录再运行。
返回终端prepare JSON、report.json SHA及SHA核验是否全部OK即可，无需传大文件。
下一轮按proxy词数决定增量MFA命令，使用现有
`models/mfa/g2p/spanish_latin_america_mfa.zip`；若资产缺失则单独处理，脚本不下载。
完成标签覆盖、CTC压力及隔离检查后才统计最终新增train小时并选择480h。

本次仅结果型HANDOFF更新；git diff --check通过，未修改算法、代码或outputs。

## 0.106 2026-09-22 Noah全量音频库存、文件去重与G2P词表准备入口

0.105抽检通过后，开始实际准备源语料。新增阶段只创建音频/文本候选库存和词表，
不创建最终480h train，不分配validation/test，不执行MFA G2P或任何模型训练。

新增`noah_source_preparation.py`、`scripts/prepare_noah_es_source.py`及5项单元测试。
复用已固定源配置、JSON流式解析器、明确路径映射和既有西语文本词表提取逻辑。
默认8个CPU线程，分批256条并行，最多32线程；不把28万条任务同时塞入线程队列。
实际会读取全部音频文件字节计算SHA，I/O量显著大于200条metadata抽检；每5120条
向stderr打印进度。无需GPU/新模型/新依赖，不复制音频文件。

处理流程与语义：

1. 先核验源JSON固定SHA，再创建新目录；处理结束再核对源SHA，防止中途变化。
2. 对每条检查单音频、Spanish前缀、非空文本及assistant/response一致，保留源行号、
   原始response/音频路径、明确映射后路径、批次、原始身份字段和directory_group_hint。
   音频header读取frames/rate/duration，读取文件字节算SHA并检查大小/mtime未中途变更。
3. inventory保留每条源记录；读取/结构问题写明issue。对相同文件SHA，若文本完全
   相同，只保留源顺序第一条；若任一转写不同，则全部副本隔离，包括最先出现的那条，
   不靠后出现的转写覆盖前一条。精确字节去重不等于已识别重编码或同录音重叠片段。
4. 第二遍从完整inventory划分candidates/needs_review，强制两者数量相加等于源记录数。
   candidates写canonical source.tsv，保留source_id/source_batch等关联字段。
   speaker_id留空，source_split=unsplit；Gxxxx只作提示，不编造speaker-disjoint。
   es-419依据用户确认的整份来源范围，不声称用声学分类器重新验证。
5. 复用prepare_mfa_wordlist生成全量词表、词频、数字片段；保留重音/ñ，原文保存在
   inventory。只删除明确ASR格式前缀并裁剪外围空白，不改写转写答案。
6. 报告按批次列出实测可读小时、去重候选小时、隔离原因和词表规模；candidate_hours
   是进入G2P/CTC筛选前的候选量，不是新增可训练小时。training_ready=false。

固定schema字段及原始路径以源JSON SHA+行号关联；不直接调用会丢失关联字段/覆盖
既有输出的旧Swift转换器。当前没有检查与既有train/validation/test的内容重叠，
没有读取sealed test；最终划分选样前必须补齐交叉去重/holdout隔离。
音频检查是header和完整文件字节读取，不等同完整波形解码或人工标注准确率审核。

### 输出及中断行为

默认全新目录`outputs/es_noah_mobile_source_v1_<随机后缀>`，终端先打印实际路径。
也可显式--output-dir，但任何已存在目录（包括空目录）均拒绝。

```text
input_config.json
inventory.jsonl              所有源行及检查结果/文件SHA
candidates.jsonl              唯一、无同文件文本冲突的候选
needs_review.jsonl            无效、重复或冲突记录，不删除
source.tsv                   保留关联字段的音频/文本源表
wordlist/words.txt
wordlist/word_counts.tsv
wordlist/character_counts.tsv
wordlist/fragments_with_digits.tsv
wordlist/summary.json
report.json
sha256.txt
```

不支持resume；中断/异常保留已有部分文件，不宣称完成。普通异常另写failed report；
重新运行默认命令会创建新目录，旧目录不删除。成功报告status=completed仅说明源准备
结束；无候选时no_candidates退出非零。不自动训练、不覆盖旧数据或既有词典。

### H200运行与回传

容器外：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内现有qwen3-asr-hotword环境：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/prepare_noah_es_source.py
```

核验命令（R取程序打印的本次新目录，不使用latest目录猜测）：

```bash
(cd "$R" && sha256sum -c sha256.txt)
cat "$R/report.json"
```

只返回终端report JSON及本次sha256.txt；不传inventory/candidates大文件、音频或模型。
下一步根据实际候选小时、review规模和words规模准备复用西语词典/增量MFA及CTC标签，
经G2P/时间可行性与holdout隔离后的新增train须达到约298.13h，才能补现有181.87h到480h。
本阶段未在本地或工作区实际处理新全量音频；真正小时数待用户返回后另作结果提交。

验证：定向（准备+检查）16 passed；全量303 passed/23 skipped；新增3个Python文件
Ruff、严格Mypy、CLI help及git diff --check通过。全仓Ruff仍5处既有E501、全包Mypy
仍3处既有unused-ignore，详见0.101；不修改个人未跟踪文件或无关代码。

## 0.105 2026-09-22 Noah全量结构及200条音频抽检通过（用户返回）

用户返回`inspect_noah_es_source.py`的终端JSON，status=inspection_completed。
源JSON 151,743,643 bytes，脚本报告SHA校验通过，值仍为
`3a57177178368b1eae0ce0be659071e955edcde56a1227c830192a0adc90b93f`。
本地已解析回传，核对批次计数合计、总条数及抽样合计一致；回传附件SHA为
`e4fd515c9129f751d7c9177209ad9581cd659d071282cafa6d6ae34ad452d685`，
这是粘贴附件身份，不冒充工作区原始report文件SHA。

全量286,821条/286,821个精确唯一原始音频路径，全部language=Spanish，四个预期
结构字段每条均有，issue_counts为空：未发现多音频、空转写、assistant/response
不一致、精确路径重复或同路径转写冲突。

| 批次 | 源记录数 | 抽样/可读 |
|---|---:|---:|
| APY161101034_R（目录标称193h） | 45,018 | 50/50 |
| APY170801048（目录标称338h） | 79,586 | 50/50 |
| APY161101034_G（目录标称8.1h） | 3,013 | 50/50 |
| APY181231012（目录标称419.5h） | 159,204 | 50/50 |

完整文件实际有4批次，而首6KB预览只露出2批次。因此本次实际抽检200条，而不是
此前按2批次预计的100条。200条全部按用户确认的前缀映射读取成功，均为16kHz，
抽样合计约20.552分钟。不能把目录小时数相加，不能用这200条直接推算全库500h。

286,821条均没有speaker_id/speaker/country/accent/dialect/split元数据字段。
拉美和人工审核来源依据继续按用户对提供方数据的确认记录；不从新闻题材或目录
名称反推口音，不重新要求已给过的确认。Gxxxx路径仅作未验证分组线索，不宣称
说话人ID，尤其不跨批次合并同名G编号。

阶段结论：可以进入全量音频metadata/文件内容SHA、去重与词表准备；还不能宣称
500h全部可训练，尚未完成音素覆盖/CTC可行性、跨集去重或speaker划分。
本节单独结果提交，下一节记录新工具。未修改已有outputs，git diff --check通过。

## 0.104 2026-09-22 Noah源流式结构检查与分批次音频抽检入口

目的：在已接受用户人工审核/拉美范围/容器路径确认的基础上，机械核验源数据契约，
为后续全量时长和G2P准备提供依据。本节为代码交付，未产生工作区实测结果。

新增：

- `configs/noah_es_mobile_source.workzone.json`：源路径、用户回传SHA、明确音频
  前缀重写、seed=20260922、每批次抽50条，以及用户确认的来源信息；
- `src/qwen_hotword/training/noah_source_inspection.py`：标准库增量解析顶层数组，
  避免将145M单行JSON整体json.load；保留唯一音频路径/摘要用于重复检查；
- `scripts/inspect_noah_es_source.py`：一个短命令，只读输入，终端输出紧凑JSON；
- `tests/test_noah_source_inspection.py`：11项测试覆盖多字节跨块、截断/尾随逗号等
  非法JSON、明确路径映射、确定性分批次抽样、重复转写冲突、多音频及SHA拒绝。

检查范围：

1. 全文件SHA必须为0.103用户返回值；流式确认数组对象完整解析，包括结尾无多余内容。
2. 全量统计字段、语言、来源目录记录数及speaker/country/accent/dialect/split是否存在。
3. 检查每条是否恰好一个绝对音频路径、一个assistant消息、response与assistant完全
   一致、language=Spanish且转写前缀符合既有形式；不静默把多音频/多轮对话降为一条。
4. 精确路径去重并统计同路径不同response；标记问题，不删改数据、不输出可训练TSV。
5. 按noah_esZ00后的批次目录分组，seed+音频路径SHA优先级无放回抽各50条；
   若完整源只有两个预览批次则约100条，不使用文件开头代表全池。最多允许30批次，
   超过时停止确认分组。样本只读音频metadata，统计可读数、采样率、时长范围/总和，
   每组返回3个代表路径/截断文本及少量失败样例。
6. 用户确认的路径映射为显式重写：命中旧前缀就使用/host_home路径，不在两个不同
   文件间猜测或优先使用/home原路径。JSON本身/home_91路径不受重写影响。

报告`provider_provenance`保留用户确认，
`latin_american_scope_independently_verified=false`仅说明没有独立声学口音鉴定，
不否定用户确认、不作为阻断条件。`inspection_completed`表示检查过程及已查字段通过，
不是全库训练就绪；`inspection_has_flags`表示有结构/重复/音频提示，逐项解释后处理。
全库真实时长仍为null，training_ready=false。不用抽样时长推算500h，不进行G2P、
外部ASR、训练、内容重复检测或holdout关联；这些属于下一阶段全量准备。

### 工作区执行

容器外拉取`codex/g2p-coverage-scan`：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/inspect_noah_es_source.py
```

返回终端JSON即可，不传源JSON或音频。无需GPU、新依赖或模型。无输出目录，
无resume状态，重跑只读，不覆盖任何outputs。后续如发现格式/路径提示，先依据
报告修复适配，再进入完整音频metadata库存、保留speaker/split的源TSV与G2P准备。
新生成数据必须使用新目录，不能盲用旧转换器覆盖旧TSV或丢失speaker等信息。

本地定向11 passed；全量298 passed/23 skipped；新增3个Python文件Ruff、严格
Mypy、CLI help及git diff --check通过。全仓Ruff仍5个既有E501，全包Mypy仍3个
既有unused-ignore（位置见0.101），未修改无关文件。真实H200结果待用户返回。

## 0.103 2026-09-22 Noah西语源结构预览返回及用户确认（结果记录）

用户在容器内返回源文件ls/sha256sum/head结果：文件显示145M，路径可访问，
SHA256为`3a57177178368b1eae0ce0be659071e955edcde56a1227c830192a0adc90b93f`。
这是用户回传的文件身份，尚未在本地读取完整源文件；下一步脚本将核对该固定SHA。

开头为单行顶层JSON数组；预览记录具有messages、audios、response、language。
每条预览audios为一个绝对WAV路径，response与assistant消息可见内容一致，
格式`language Spanish<asr_text>转写`，language=Spanish。尚未全量确认这一契约。
音频目录出现两个批次：

- `APY161101034_R_227小时西班牙语手机采集语音数据_193小时`；
- `APY170801048_338小时西班牙语手机采集语音数据_338小时`。

目录名小时数不能相加当成JSON实际唯一时长；文本题材也不能用来判定口音。
预览未显示speaker/country/accent/split字段。`G0239/session01`与文件名组件
暂时只能作为路径线索，未证实speaker语义，尤其不可因跨批次同名就合并speaker。

用户随后明确确认：两批均属于拉美西语；容器内音频路径把/home改为/host_home，
映射后可访问。连同此前“文本经过人工审核”，作为用户确认的来源信息接受并记录，
不再把口音范围或挂载方案作为待确认阻塞项。无需外部ASR或声学口音分类器。
后续按明确前缀`/home/z00841352/27A/data/`到
`/host_home/z00841352/27A/data/`执行，不修改JSON源文件，也不重写其/home_91路径。

阶段结论：格式可以适配现有CTC准备管线；已确认来源范围及路径策略，尚未完成
全量结构/重复检查、音频抽检、实际小时数和G2P/CTC可用量核算，尚未加入训练。
本节为独立结果型记录；下一节另记检查工具交付。git diff --check通过，无产物覆盖。

## 0.102 2026-09-22 新增Noah拉美西语500h线索：先探查JSON结构

用户提供工作区路径：

```text
/home_91/c00809914/qwen3-asr-eagle/500h_latn_es_mobile_noah.json
```

提供方称约500h拉美西语，JSON结构未知。用户补充确认文本经过人工审核，JSON
只有一行且太大，无法在本地编辑器打开；无需用户打开或上传完整文件。人工审核
作为用户提供的来源信息记录，不冒充本地独立审核结果。当前只记录为新候选来源，
尚未读取实际JSON/音频，也未确认唯一条数、真实时长、具体口音元数据或可用比例。
本地机器不能直接读取这个工作区路径。用户授权检查并准备加入新的480h西语train，
但不能因此跳过数据准备直接开始训练。继续结构检查；不能仅凭文件名latn或
language=Spanish认证拉美口音。

### 当前判断和容量条件

现有已处理西语train约181.868358h，目标480h，需要新增至少298.131642h合格唯一
train。若新源实际为500h，则约59.63%最终能进入train就可补齐（指扣除重复、
holdout、坏音频、标签/CTC不合格后的比例）；尚无证据表明已达到这个比例。
计划保留现有训练覆盖，从新源补足，不将新500h全部直接拼成新的训练目标。
原英语487.628442h和葡语783.223637h可派生480h的判断不变，无固定27% recovery。

### 第一步：最短只读结构预览

在能访问/home_91的工作区环境运行，容器不可见时先在宿主机运行。此次不需要
Git更新、GPU或模型；不假设/home_91会自动挂载到/host_home，若不可见只返回错误。

```bash
P=/home_91/c00809914/qwen3-asr-eagle/500h_latn_es_mobile_noah.json
ls -lh "$P"
sha256sum "$P"
head -c 6000 "$P"
```

SHA256顺序读取文件但不解析到内存；预览最多6000字节，可在记录中间截断，截断
不等于JSON损坏。此预览仅用于确定顶层类型、字段、文本/音频路径写法，不是随机
质量抽样或全量JSON合法性证明。返回文件大小、SHA和这段预览，无需上传完整JSON。
首次预览若未露出记录结构，再按实际格式提供有界结构检查，不先假定Swift格式。

### 收到结构后已授权继续的数据准备顺序

1. 确认JSON/JSONL/嵌套容器、音频路径或分段offset、人工reference字段、speaker/
   session、语言/国家、既有split字段。多个音频或多轮对话不能静默只取第一条。
2. 少量音频路径试读、时长/采样率/声道和文本样例检查；路径映射须按实际挂载，
   不猜测根目录。保留原始标注与来源字段，不把模型response未经核实当人工reference。
3. 全量流式库存及音频metadata审核，报实际唯一小时、缺文件、空文本、重复ID/路径；
   对跨库改名副本及同录音重叠片段另做内容/片段去重，排除既有holdout及其speaker。
   在全量索引后固定seed分层抽约100条，覆盖时长/文本特征/来源元数据，不能用
   文件开头样本代表全池，也不能仅凭自动抽检宣布标签全对。
4. 延续现有拉美西语G2P/词表和Temporal2×可行性规则，按需增量生成词典与完整
   ready/review分区；recovery按规则保留，无人为固定占比。当前不引入外部ASR。
5. 新来源若有speaker/session，保留原split并按身份隔离；缺失时明确speaker隔离
   局限，不声称有保证。现有拉美validation及其ID不变，必要时另加新来源诊断
   validation（同样限拉美，先划分后选train），不替换历史基准或挪用test。
6. 最终确认>=298.131642h新增可用train后，与已有合格train组成480h唯一集合，
   输出source/release/密度/时长/说话人统计、固定ID和SHA；所有派生产物新目录，
   不覆盖旧outputs。随后才准备特征缓存和训练命令，仍由用户在H200执行。

已查看现有`convert_swift_json_to_tsv.py`及`swift_json.py`：只支持顶层list，取
首个audios和response/assistant文本，输出仅audio/text且有覆盖既有目标的能力。
因此当前不盲用该转换器，不能丢speaker/split/offset或多音频信息。具体适配待真实
结构返回后再做最小实现并加测试，提交/推送后给用户短命令。

本阶段只有docs更新，git diff --check通过；未新增代码、未生成数据、未重新运行
模型或无关测试。状态为等待工作区结构预览，不能记录成500h已验收或480h已完成。

## 0.101 2026-09-22 目标改为每语种480h；西语严格拉美容量核算入口

### 用户决定与当前状态

用户确认：各语种目标改为480小时唯一train，先利用现有资源；西语train和validation
必须继续在拉美范围。现有Common Voice/MLS若不足，明确报告缺口，由用户补找语料。
用户询问旧训练配比不代表要求复制旧配比：撤销0.100中葡语72.35/27.65及来源等比
扩容作为默认配额。葡语可从783.223637h管线可用train中选480h，保留合法recovery，
但不规定必须27%；实际组成在选样后量化。不得把recovery全部删掉再重复小池补时长。

英语并非最多480h：历史可用train为487.628442h，可选480h，余约7.628442h。
葡语可用train余量约303.223637h。这里“可训练”指既有音频/标签Manifest通过管线，
不意味着全部480h的Encoder特征缓存已经建好；已知三语缓存只对应此前各150h。
新子集要核验身份、准备缺少的特征，并保留Temporal2×兼容。不会自动认证标签全对。
当前仅启动西语容量/范围核算，未创建480h训练集、未抽英葡数据、未训练或下载模型。

### 为什么先核算

现有西语管线可用train=181.868358h，距480h=298.131642h。
原始CV总516.125434h不全是拉美；历史明确拉美部分约230h，且大量已包含在现有
train中。América central规则已补正，本轮用现有classifier重算而不复用旧tier字段。
MLS原始917.684176h现有库存没有明确拉美方言依据，当前准入小时为0；并不声称
其每条都是半岛口音。历史证据预示总量不足，具体新增候选与最小缺口等H200返回。
不通过混入地区未知、混合或半岛数据凑480h，不把同一CV clip跨版本重复计时。

### 已交付的只读工具

- `scripts/audit_es_480h_capacity.py`：默认target=480，默认既有两个输出根，可用参数覆盖。
- `src/qwen_hotword/training/spanish_capacity.py`：校验指定资产SHA，只读train、
  validation、speaker assignment元数据、CV inventory与MLS summary。
- `tests/test_spanish_capacity.py`：6项测试覆盖去重计时、holdout说话人、未知/混合/
  半岛排除、中美洲重分类、validation范围问题阻断、SHA篡改、重复clip拒绝以及
  容量充足也不得宣称label-ready；夹具中sealed test文件不存在，证明不读取其内容。

输入：

```text
outputs/es_candidate_inventory_v1/
  common_voice_inventory.tsv, mls_summary.json, sha256.txt
outputs/es_combined_temporal2x_v1/
  split_summary.json, full_ctc_train.jsonl, full_ctc_validation.jsonl,
  speaker_split_assignments.tsv, sha256.txt
```

算法：先复核当前train/validation的数量、时长、SHA和speaker split，再把CV辅助
记录按唯一clip关联回已有元数据，复核明确拉美tier、es locale、文本关联、音频
可读状态、时长和speaker；SLR61/Rioplatense核心沿用已验收的语料来源范围，报告
明确标记该证据层次，不冒充逐音频口音识别。发现范围问题则blocked，不自动删除。

剩余CV候选必须为官方train、明确拉美元数据、音频/文本关联通过、有speaker、
不与核心clip重复、不属于既有validation/test speaker；当前train/validation中的
CV clip单独计数，不能再次算新增。使用speaker assignment中的test身份防泄漏，
不读sealed test Manifest、文字、音频或模型结果。未设置新的speaker时长截断，
报告top5新增speaker小时数；所得为未做额外cap/标签过滤前的乐观容量上界。

输出区分：

- `existing_pool`：现有管线可用train/validation时长；
- `existing_latam_scope`与`scope_issue_count`：两者拉美范围依据及疑点；
- `additional_raw_candidates`：扣除既有记录和holdout后的新增原始候选；
- `optimistic_total_train_hours_before_label_filtering`：现有train+新增候选上界；
- `minimum_extra_hours_needed_even_if_all_candidates_pass`：即使新增全通过仍缺多少；
- `gap_from_existing_ready_train_hours`：距现成可用train的实际缺口。

`insufficient_raw_capacity`表示核算成功但容量不足（CLI正常退出）；
`labels_not_yet_verified`表示原始上界够，仍需标签/内容重复审核，不代表可以训练；
`blocked_existing_scope_audit`表示当前范围核验有问题，停止后续准备。
`training_dataset_created=false`、`files_written=false`始终明确写出。
该工具不统计两个候选库以外尚未释放的review数据，不能将局部容量界说成全球库存。

### 工作区命令与回传

分支`codex/g2p-coverage-scan`。容器外拉取：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内，现有qwen3-asr-hotword环境：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/audit_es_480h_capacity.py
```

只返回终端JSON与commit身份即可；不用GPU、不下载模型、不重跑MFA/音频审计。
无输出目录、无resume状态，重跑是只读复核；已有outputs完全不变。
收到结果后另作结果型HANDOFF提交，明确足够/不足及需用户新增的最低合格train时长。
若不足，不生成虚假的480h Manifest；若足够，才进入新增候选词表/G2P/Manifest准备。

### 本地验证（不等于H200真实容量）

定向pytest 12 passed；全量287 passed/23 skipped。新增3文件Ruff、严格Mypy、
CLI help、git diff --check通过。全仓Ruff仍有5个既有E501（scan_g2p_coverage.py三处、
个人未跟踪PPT脚本两处）；MYPYPATH=src mypy仍有3个既有unused-ignore，分别在
ctc_overfit.py:260、sharded_ctc.py:989、unfrozen_encoder_ctc.py:813；不修改无关文件。

## 0.100 2026-09-22 每语种500小时扩容：库存核对与配置草案（尚未准备/训练）

### 目的与阶段状态

用户将主线调整为准备英语/西语/葡语各500小时唯一训练音频，以降低葡语CTC PER。
顺序为先讨论配置，再准备语料，再准备训练。本阶段只核对历史库存与提出配置，
未启动训练、提取特征、下载语料或运行外部ASR；未修改核心算法和已有outputs。
0.99所述“下一步读混淆排行”保留为历史建议，不再作为本阶段的前置阻塞项。

数据量增加可能改善覆盖和泛化，但不保证PER单调下降，也不能消除参考标签系统误差。
Qwen官方报告附录MLS的1.7B WER为en 4.58/es 4.63/pt 7.71；FLEURS为
3.35/3.36/3.92，支持部分测试上葡语较弱，但不是本项目CTC PER的下限或因果解释。
Common Voice为en 7.39/es 4.65/pt 7.10，因此不能泛化成葡语在所有集合都弱于英语。
来源：https://arxiv.org/html/2601.21337v2 。

### 已核对的历史库存及缺口

读取0.9、0.28、0.32–0.38、0.47、0.62、0.84、0.98–0.99和相关构建器。
本地附件`selection_summary.json` SHA256为
`2040c3cabc9298aa159231dce6fb37db280afba6b2fc0ab81476155b5387f5ec`；
其combined manifest SHA与历史固定`def68274...df7e2`一致。
附件声明与文档互相印证，未声称重新扫描H200全量训练文件或验证当前原始音频。
Git起点为`f475d3f5285ff035e82d57f92f23bc00354a61f8`，与origin一致；
无tracked修改，已有个人未跟踪文件和暂停的Whisper清单均不动。

| 语言 | 历史已建train条数 | train唯一小时 | 达到500h至少需新增合格train小时 |
|---|---:|---:|---:|
| en（US Swift） | 345,820 | 487.628442 | 12.371558 |
| es（拉美范围） | 125,379 | 181.868358 | 318.131642 |
| pt（五来源） | 525,189 | 783.223637 | 0；容量多283.223637h |

三者上一版实际都只选了150h。不能把英语含validation/test的507.959291h算为
可用train500h。葡语783h包含original与recovery，是管线可用性证据，不是音素
准确率认证。缺口是最终合格、去重、排除holdout后的时长，不是原始采购/下载时长。
若新增原始音频最终入train的利用率为80%，英语/西语预算分别约15.47/397.66h；
利用率未知，该预算仅作情景估算。

西语还有已审计但未全部转为训练池的CV原始516.125434h及MLS 917.684176h。
CV中已排核心重复后明确Rioplatense 2.110041h、其他明确拉美228.542930h；
140.596660h地区未知、119.774463h半岛口音。旧分类还存在América central补正，
最终新增可用量需从既有inventory重新汇总，不能把这些原始数字与181.87h相加。
继续拉美限制是当前建议；是否允许泛西语已向用户询问，未擅自引入未知/半岛/MLS。

### 上一版真实配比（全部按音频时长）

| 语言 | original-ready小时/占比 | recovery小时/占比 |
|---|---|---|
| en | 149.991207 / 99.99319% | 0.010216 / 0.00681% |
| es | 142.614453 / 95.07534% | 7.387058 / 4.92466% |
| pt | 108.522236 / 72.34799% | 41.478101 / 27.65201% |

葡语recovery按条数为30.42916%，不能与27.65201%的小时占比混用。
失败候选只剩108.522h唯一葡语再重复抽样，PER由9.662%升至10.285%；
这支持不重演“删recovery并重复较小池”，但不是最佳比例或recovery独立收益的证明。
recovery是原本时间长度不够、2×后可行的真实音频，不是重复扩增，也不是坏标注标签。

葡语上一版来源及按同配比扩到500h的草案：

| 来源 | 旧150h内小时 | 占比 | 新500h内目标小时（四舍五入） |
|---|---:|---:|---:|
| Noah500口语 | 88.717421 | 59.1448% | 295.72 |
| Noah金融 | 35.067288 | 23.3781% | 116.89 |
| MLS | 20.426056 | 13.6173% | 68.09 |
| Common Voice | 4.548638 | 3.0324% | 15.16 |
| FLEURS | 1.240933 | 0.8273% | 4.14 |

### 建议配置，待讨论与库存验收后冻结

- 总量：en/es/pt各500h唯一train，1:1:1按音频小时；正常跨epoch再次遍历允许，
  禁止在同一版Manifest重复音频凑500h。每epoch每条一次，保留原loss/采样语义，
  同时报条数、帧数、音素数，不能声称等小时等于等梯度贡献。
- 葡语original/recovery目标361.74/138.26h，即72.35/27.65；来源按上表，
  在source×release层内保持旧train的密度/时长分布。比例为基线复用，不声称最优。
  建议source/release边际比例偏差各不超过1个百分点，最终逐层容量核实后冻结。
- 优先保留旧150h为新500h的子集；若发现确定质量问题应隔离并记录，不能强保错误。
  有隔离时要明确比较同时包含数据清理变化，必要时补相同清理的150h对照。
- 英语保留US范围；不为了凑恢复比例增加recovery。西语旧来源为拉美辅助85.0454%、
  Rioplatense9.9328%、SLR61 5.0218%。后两者当时已全量纳入train，不能简单按比例
  放大成49.66/25.11h；需要新的唯一语料。保留已可用核心22.432130h；新来源配额
  待口音范围与库存确定，暂不硬造比例。西语旧recovery 4.92%仅作参考，不强迫新域
  恰好同占比，更不能将葡语27.65%复制给其他语种。
- 音素密度rho=L/T2；CTC可行压力q=(L+相邻重复token数)/T2，两者分开统计。
  T2=2×Encoder实际帧数；筛选初期可用估计值，但cache后必须实际复验。
  保留2×；建议所有入选样本q<=0.90，其他标签/音频问题仍阻塞。
  该阈值沿用既有recovery安全规则，不是标签准确性的判据。
- 不把葡语训练密度强制匹配西语0.347，也不把validation三分位边界用于训练配额。
  先对旧150h和完整train统计固定rho桶[0,.25),[.25,.35),[.35,.45),[.45,.55),
  [.55,.70),[.70,.90]的小时/条数，以及q、时长、reference长度P10/P50/P90/P95。
  葡语扩容后每个rho桶小时份额建议距旧train不超过3个百分点（工程容差，非最优值）；
  缺少旧train直方图，当前不能捏造各桶配额。时长桶<2/2–5/5–10/10–20/>=20秒。
- 去重优先ID/规范音频路径，新增跨语料再做音频内容/片段重复检查；有speaker ID
  则保留原split并限制集中度，报告top-speaker时长。葡语旧150h全部缺speaker ID，
  优先补可关联的CV/MLS，Noah不能宣称speaker-disjoint。不动既有validation/test。
- 同现有Qwen冻结Encoder、ln_post、Temporal2× Head(h512,k5,dropout0.1)、词表/G2P。
  建议新Head随机初始化而非失败候选续训；初始batch256、lr3e-4、weight decay1e-4、
  clip5、seed20260825，Macro PER选checkpoint及现有早停口径先保持。
  先计划1epoch流程检查，后续预算单独确认；500h为每epoch数据量不是训练墙钟时长。
  同epoch的500h运行有约3.33倍数据曝光，报告step/小时/耗时，不能把全部收益称为
  单纯数据多样性因果收益；需要归因时再补等step对照。
- 固定Legacy三语validation和葡西密度匹配视图，不通过换验证集制造提升。
  建议事前目标：葡语Legacy及匹配视图均相对降至少10%（约<=8.696%/7.387%），
  英/西各不退化超过0.2个百分点，同时检查recovery/source的S/D/I；这些是拟议
  成功门槛而非效果承诺。最终热词/ASR效果仍另测。

### 立即下一步

先确定西语口音范围，并只读刷新三个既有split summary（不打开test内容）：

```bash
for p in en_us_swift es_combined pt_combined; do
  jq '{output_dir,status,split_records,split_audio_hours,corpus_metrics}' \
    "outputs/${p}_temporal2x_v1/split_summary.json"
done
```

这里只是当前summary快照，不替代后续train文件SHA与完整分层容量核验。
在用户接受数据配置方向后再交付只读source×release×density库存扫描及具体扩容工具，
不让用户复制长脚本。本阶段仅docs变更，git diff --check；无训练/数据成果可汇报。

## 0.99 2026-09-21 葡语诊断阶段总览与交接规则（当前主线）

用户要求：HANDOFF必须同时记录每个阶段的目的、实际做法、对应结果、解释边界及
下一步，不能只追加最终数字。以下总览串联0.83–0.98；各节的命令、身份SHA、
错误过程和详细结果继续保留。后续新增阶段也沿用这一记录方式，明确区分
“方案/代码已就绪”“H200已执行”“结果已返回”“证据仍待补齐”，不把计划当成果。

当前问题是**葡语CTC PER偏高**。CTC PER、热词检出和最终ASR WER是三个不同层次；
当前只诊断CTC。最近完成的是密度匹配子集评测，尚未完成16小时全池标注质量认证。

| 阶段与目的 | 实际做法 | 对应结果 | 当前判断/状态及详情 |
|---|---|---|---|
| 拆分错误与模型差异 | 同一2,662条葡语上比较原三语/旧葡语Head，按source、release、密度分层 | 原三语9.662%，旧葡语9.032%；原三语original-ready 8.437%，recovery 11.827% | 高密度与漏音素相关；低压力仍有较高PER，非单一瓶颈。见0.83 |
| 去掉recovery的训练消融（历史已完成） | 从原150h中留下69,918条/108.522h original-ready，再重复抽样补足曝光 | 候选旧集10.285%，比原三语更差；主要退化为deletion | 不是全池精选150h唯一数据；候选失败，不替换基线。做法见0.84，复测见0.98 |
| 核验候选池和原始文字 | 保留10,899条/16.239h候选；用户确认并检查训练ID/路径隔离，核对CV/FLEURS原始元数据 | CV414条中406精确一致、8仅格式差；FLEURS42条中41精确一致、1仅格式差 | 456条未发现转换阶段实质词变更；不是音频或发音正确性认证。不再重复训练重叠检查。见0.85–0.88 |
| 探索MFA作为声学检查工具 | 五来源各20条、共100条分层抽样；模型下载改为容器外单独进行，容器内本地对齐 | 98 aligned、1 incomplete、1 missing；63条OOV、61条未知词/音素提示 | 对齐能运行，但工具覆盖不足；不把98%当标签准确率，不扩成全池质量认证。见0.89–0.92 |
| 外部ASR交叉检查提议 | 曾准备Whisper下载清单，用户要求暂停 | 未下载模型、未运行推理，仅有未提交清单 | 保持暂停；未实施全量独立ASR审计，不把外部转写当标准答案。见0.92 |
| 控制葡西音素密度 | 从16h葡语池中按估计ref/frame做确定性、不重复的一对一分布匹配，不用预测/PER选样 | 选2,631条/4.003h，对照西语2,631条/4.001h；葡语2570 original-ready+61 recovery，与旧集重叠646条 | 固定诊断子集，不是clean集；只匹配密度，来源/句长等未全面控制。见0.93–0.94 |
| 解决帧数口径不一致 | 评测被严格相等检查拦住后，只读核验旧cache；保留IDs/标签，用实际帧数复验 | 8,101条无缺失、标签序列全部一致；781条估计值与实际值相差±2有效帧 | 估计与实际长度差不等于错标；改用实际长度，不静默放行标签变化。见0.95–0.97 |
| 固定子集上评测现有三个Head | 用已有冻结Qwen Encoder提取新子集特征；三个Head同集评测并重评Legacy，原三语另评西语 | 实际密度KS=0.0110通过；新葡语三语8.207%、旧单语7.618%、失败候选8.601%；西语3.910% | 密度匹配后差距仍大；主要下降为deletion，不能作纯密度因果归因或标签认证。见0.98 |

### 最近完成实验的核心解释

- 原三语葡语PER从9.661777%到8.207343%，下降1.454433个百分点；
  deletion从4.204526%到2.849603%，是净下降的主要组成。
- 密度接近后，葡语仍比西语3.909750%高4.297593点，约2.10倍；
  新葡语original-ready单独仍为8.124465%，不能把剩余差距都归因于recovery。
- 旧葡语Head的优势主要集中Noah500；MLS/金融反而略差于三语，且训练数据/时长
  不同。因此没有完成“只改变多语共享结构”的受控实验。
- 原三语Head保持基线。Legacy full与新密度匹配子集并存；不能拿新8.207%覆盖
  历史9.662%，更不能说“不改模型就把同一任务提升了1.455点”。
- 标签与缓存一致只验证数据身份；标签符合真实发音、source/domain贡献、G2P发音
  歧义、训练覆盖不足的独立影响仍待核实。150h唯一训练重建仍是候选，不预设重训。

### 当前固定资产与下一步

- 原16h候选：`outputs/pt_combined_temporal2x_v1/full_ctc_validation.jsonl`。
- 旧Legacy：`outputs/en_es_pt_balanced_validation_4h_v1/full_ctc_validation.jsonl`中的pt。
- 新子集：`outputs/pt_es_density_match_v1/full_ctc_validation.jsonl`，
  Manifest/ID SHA见0.94；估计密度的规则/配对不改，实际密度另记在评测报告。
- 已完成评测：`outputs/pt_es_density_match_v1/evaluation_oja96zhl/`，
  实际密度、三个Head的PER及分层指标见0.98；来源为用户粘贴报告，尚未返回
  本次sha256.txt/inputs.json，不声称完整产物文件哈希已在本地复核。
- 下一步先读取既有音素替换/删除/插入排行，确定剩余错误是否集中在少数音素；
  该步不需要GPU、不重新推理。尚未收到排行结果，不能提前断言是合理发音变体。

容器内项目目录只读命令：

```bash
R=outputs/pt_es_density_match_v1/evaluation_oja96zhl
jq '{top_substitutions,top_deletions,top_insertions}' \
  "$R/multilingual_baseline_matched_pt.json"
```

后续约束：保留全部outputs和原标签；新实验使用新目录；不读sealed test；
外部ASR、全池质量认证、纯推理入口移植、150h重建及新训练均没有在本阶段继续执行。
先基于返回证据确定下一项单因素实验，不同时修改数据、发音规则和Head结构。

本次仅补充阶段总览和交接规则，未改代码/数据/模型。`git diff --check`通过；
不重复运行与文档变更无关的模型或测试。结果型记录与代码交付继续分别提交。

## 0.98 2026-09-21 西语密度匹配葡语子集：PER结果已返回

用户返回`outputs/pt_es_density_match_v1/evaluation_oja96zhl/report.json`的粘贴内容，
`status=completed`，未返回本次sha256.txt/inputs.json；不能声称完成原始产物文件SHA复核。
已核对报告中原三语/旧葡语Head SHA与固定配置一致，候选SHA为
`f3f2f41ce0cea2df86a1fafcac044f1c12ae51d73a9aebeadf678dbd35c9d22d`。
逐视图验证S+D+I=errors、hyp=ref-D+I、PER=errors/ref，source/release计数逐项
求和与总体一致。新cache 2,631条、6分片、187,919原始帧、131,492参考音素；
有效帧375,838与评测一致。提取时间字段约56.995秒，仅为提取统计，不代表端到端耗时。

实际密度匹配通过：KS=0.0110224（阈值0.05），排序分位数差P95=0.00275215
（阈值0.025），最大差0.00896057（阈值0.10）。葡/西分别2,631条，
4.003024568/4.000865769小时；逐样本实际密度均值0.34542970/0.34677622。
总reference/总frame为0.34986351/0.34682664。两侧没有缺失或token标签不一致；
帧数审计中的mismatch仅指估计与实际偏差，已使用实际长度验收，不再阻断。

| Head | 旧葡语2662 PER | 匹配葡语2631 PER |
|---|---:|---:|
| 原三语baseline | 9.661777% | 8.207343% |
| 旧葡语专用 | 9.032353% | 7.617954% |
| 失败去recovery重复抽样候选 | 10.285078% | 8.600523% |

同次原三语Head西语PER=3.909750%。旧集结果复现既有历史数值。
原三语Head新葡语相对旧集下降1.454433个百分点（相对15.0535%），但仍比西语
高4.297593个百分点，为西语的2.0992倍。新葡语original_ready PER=8.124465%，
西语original_ready=3.873282%；剩余差距不只是新集合61条recovery造成。

| 原三语Head错误率（计数/reference） | 旧葡语 | 匹配葡语 | 西语 |
|---|---:|---:|---:|
| S | 3.975533% | 3.719618% | 1.682755% |
| D | 4.204526% | 2.849603% | 1.396394% |
| I | 1.481717% | 1.638122% | 0.830601% |

deletion下降1.354923个百分点，占PER净下降约93.16%；这只是算术错误率分解，
不是“时间压力造成93%的错误”的因果估计。S降低0.255915点、I增加0.156405点。
匹配葡语与西语剩余差距中，S差约2.037点，比D差约1.453点大，因此没有证据
把后续动作只聚焦于时间分辨率。预测/reference长度比从0.972772到0.987885。

同一新子集上旧葡语Head优于三语0.589389点，失败候选比三语差0.393180点。
旧葡语Head优势有明显域差异：Noah500从三语7.589831%到旧单语6.374095%，
但MLS为三语9.265238%/旧单语9.599751%，金融为三语8.872688%/旧单语9.155776%。
Noah500少856个错误，所有来源合计少775个错误，说明其他域部分抵消其优势。
结合旧Head使用313.8916小时Noah500训练，不可将差距纯粹解释为多语共享干扰。
候选仍是失败消融，基线保持原三语Head。

本轮结论：密度匹配视图的PER降低且主要表现为漏音素减少，但来源、句长等也发生
变化；没有控制所有混杂因素，不能量化密度的独立因果贡献，不能证明原标签错或
新标签正确。该视图暂作固定诊断对照，与Legacy并存，不替换历史9.662%基准。
最终ASR WER/热词召回均未测量，不能由本轮PER变化外推产品改善。

下一步建议先读取本次已有`multilingual_baseline_matched_pt.json`中的
`top_substitutions`、`top_deletions`、`top_insertions`，定位剩余错误的集中程度；
无需重跑模型。不直接合并音素、不引入外部ASR、不启动训练。错误排行只指示候选
问题，不能独自证明某发音变体或参考标签错误。

## 0.97 2026-09-21 使用实际帧数复验固定子集后继续PER评测

根据0.96，evaluate不再要求Manifest估计帧数与实际cache帧数逐条相等。
样本缺失、token序列变化和非Temporal 2x仍严格阻断。现有build代码、配置和已冻结
子集SHA均不变；只在内存中的诊断副本使用实际有效帧数，不改源Manifest和参考音素。

- 原西语和旧葡语使用已验证的原cache实际帧数；新葡语子集仍在新目录提取冻结特征。
- 新旧两侧统计实际帧数后，重新计算经验CDF KS、排序分位数配对绝对差P95及最大差；
  阈值仍为0.05、0.025、0.10，不因结果改变。帧数变化可能改变近邻顺序，故此处比较
  两侧排序后的经验分位数，不沿用旧pairs.json中的原始配对ID计算差值。
- 实际密度通过后才执行0.93的7项Head/view评测；失败则返回
  `insufficient_actual_density_match`并退出1，保留新特征/报告，不运行Head评测。
- 报告新增`actual_density`和`frame_audit`；新输出中另有
  `actual_density_report.json`、`frame_audit.json`、`actual_frames.json`（三视图逐ID实际帧数）。
  SHA清单包含这些JSON；原始selection报告仍准确表示估计帧数的选样依据。

交付分支`codex/g2p-coverage-scan`，提交标题
`Validate matched density using actual cached frame lengths`，SHA见交付回复。
容器外项目目录执行：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

容器内执行（GPU6为用户上次选择，仍须确保该卡可用）：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
CUDA_VISIBLE_DEVICES=6 python -B scripts/run_pt_density_match.py evaluate
```

**不要重新build**。旧selection和失败产物全部保留；每次evaluate创建全新
`outputs/pt_es_density_match_v1/evaluation_<随机>/`。本轮仍无resume；若失败，
先返回错误，不删除任何输出。只用已有Qwen模型，不下载、不运行外部ASR、不训练。
GPU/磁盘要求与0.93相同。原cache核验仍为只读。

完成或实际密度未通过时，返回本次`report.json`和`sha256.txt`即可，报告已包含
实际密度摘要；无需上传actual_frames、缓存或模型。验证：

```bash
R=outputs/pt_es_density_match_v1/evaluation_实际后缀
(cd "$R" && sha256sum -c sha256.txt)
```

本地：定向14 passed；全量281 passed / 23 skipped；修改文件Ruff、模块Mypy、
`git diff --check`通过。全仓仍只有5处旧E501与3处旧unused-ignore。
测试验证±2帧使用实际值且不改输入、同长度不同标签/缺失样本仍阻断、实际密度
不通过时不执行任何Head评测，以及通过时三个Head仍共用冻结ID集合。
本地未运行真实Qwen/Head；实际密度是否通过及PER等待H200结果。

## 0.96 2026-09-21 旧cache核对结果：标签一致，仅估计帧数±2

用户返回`audit-cache`：8,101/8,101条已检查、缺失0、标签序列不一致0。
帧数差异：en 214、pt 220、es 347，共781条；实际有效帧数减清单估计值为
-2的337条、+2的444条。Temporal 2x下相当于±1 Encoder帧。
cache fingerprint为`391743f70f95d684efbf9676acc3eb48b3d0ad417b84276cbbe121c76aa66aa7`，
Manifest SHA仍为`d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499`。
该结果证明清单与缓存标签身份一致，不证明音素标签符合实际声音。
帧数估计与实际不同，不应继续要求严格相等；具体取整/预处理环节未由本报告定位。

下一步保留0.94冻结的2,631条葡语及全部原始标签，以实际cache帧数重新计算葡/西
密度并复验原有KS/P95/最大差阈值，通过后才运行现有三个Head；不重新挑样本、
不根据PER调整集合，不修改Manifest中的估计值或任何既有outputs。

## 0.95 2026-09-21 只读核对旧validation实际帧数与标签

0.94错误仍保持阻断，不放宽帧数/标签校验，不改选样算法或已生成子集。
新增`audit-cache`入口，读取旧三语validation cache全部8,101条（以实际身份校验为准），
按语言分别统计帧数不一致/标签不一致、实际减估计帧数的分布、缺失样本及最多8条示例。
同长度但不同token序列也会检测；正常evaluate报错同时给出这些详细诊断。

分支`codex/g2p-coverage-scan`，提交标题`Explain density audit frame and label mismatches`。
先在容器外项目目录拉取并核对交付SHA：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

再在容器内主环境、项目目录执行：

```bash
python -B scripts/run_pt_density_match.py audit-cache
```

不需要GPU，不加载Qwen/CTC Head，不提特征，不改任何outputs或模型；不需resume。
只返回终端小JSON，不返回缓存。SHA逐分片验证后，报告包含cache fingerprint、
manifest SHA、`model_loaded=false`、`files_written=false`。
`status=mismatch`退出码1表示查到差异，不表示扫描失败或标签已判错。
既有selection与SHA保留，暂不重复build/evaluate；先根据差异决定后续帧数口径。

本地：定向12 passed；全量279 passed / 23 skipped；修改文件Ruff和模块Mypy通过，
全仓仍为5处旧E501及3处旧unused-ignore。`git diff --check`通过。
增加帧数/标签分离统计、同长度标签变化、缺失样本及两语种差异的合成测试；
没有在本地读取H200缓存或产生新的PER结果。

## 0.94 2026-09-21 密度子集已匹配，旧缓存帧数/标签检查中止

用户返回build：`status=matched`。葡语新子集2,631条/4.003024568小时，西语
参照2,631条/4.000865769小时；逐样本密度均值分别0.3460739563和0.3460722586，
KS=0.0022805017，配对差P95=0.0003324468、最大差0.0021008403。
选中葡语original_ready=2,570、recovery=61，保留五来源，与旧2,662条重叠646条。
音频时长和参考长度分布未匹配，不能声称全面公平或标签已认证。

选中Manifest SHA `9fffa94ed45dd0913fd335f5da3c74c1816bc602c029d0cdf8ebedf8b90571a7`；
ID SHA `0c2687ce96bb412462a3c0cc42e98d1b1fc382b572caed5aed6463b3b9eccf43`。
以上来自用户粘贴输出，未取得H200源文件字节作本地重算。
随后GPU6 evaluate通过旧validation cache 16/16分片校验，但在样本
`noah_pt_finance_200h_row_1363`报`Actual feature length/labels differ`。
程序在创建evaluation目录及加载Qwen前停止；尚无新特征或PER结果。
旧错误信息混合帧数和标签检查，暂时不能判定哪项不一致，不能据此判为标签错误。
下一步只读核对全部旧validation缓存的实际帧数及标签，先不重建子集、不放宽保护检查。

## 0.93 2026-09-21 葡语按西语密度分布匹配的固定子集评测

用户明确要求先实际做一版：从现有16小时葡语validation中，选音素密度分布接近
西语的子集，评测现有Head。暂停外部ASR审计；不训练、不修订标签、不改核心模型。
本节是条件分布评测，不是clean validation认证，也不是完整跨语言公平评测。

交付分支`codex/g2p-coverage-scan`；代码提交标题
`Add Spanish-density-matched Portuguese validation experiment`，SHA以交付回复为准。
运行配置`configs/pt_es_density_match.workzone.json`固定输入/Checkpoint路径、已知SHA及阈值。

**容器外拉代码**：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

**容器内先选子集（CPU，不读取音频、不加载模型）**：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
python -B scripts/run_pt_density_match.py build
```

固定来源：

- 葡语候选：`outputs/pt_combined_temporal2x_v1/full_ctc_validation.jsonl`，
  10,899条/16.239131小时，SHA
  `196d6e760dbfd626caf566ad333afd999ce6d2770f562add372f657ce9500524`。
- 西语参照：`outputs/en_es_pt_balanced_validation_4h_v1/full_ctc_validation.jsonl`
  中`balanced_language_bucket=es`的全部样本，历史数量2,631条/4.000866小时；
  三语清单SHA `d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499`。
- 旧葡语对照：同一三语清单中的全部pt（历史2,662条），保留9.662%的原始比较口径。

预先固定的选样规则：

1. density=`label_length / effective_ctc_input_length`；帧数必须为
   `estimated_ctc_input_length*2`，标签长度核对保存的`phoneme_token_ids`。
   它不是包含相邻重复音素开销的minimum CTC ratio，不混用两种比值。
2. 葡语全池及西语参照按密度排序，同密度以seed=20260921与ID的SHA排序。
   动态规划求一维有序、不重复的一对一匹配，最小化绝对密度差总和。
   保留每一条西语参照，选择同样数量的葡语，不删参照尾部、不重复抽样，
   不读取预测/PER，不排除recovery，不预选低错误来源。
3. 本版按样本数匹配密度分布。总时长、句长、来源、口音和token加权分布未匹配；
   报告提供source样本/小时、release数量、时长/参考长度分位数、固定0.025宽密度直方图、
   密度分位数、逐样本均值及总reference/总frame，便于检查组成变化。
4. 通过条件：经验CDF KS距离≤0.05、配对绝对密度差P95≤0.025、最大差≤0.10。
   阈值是本版预设的匹配容差，不是统计显著性或标签质量标准。
   若不满足，输出`insufficient_density_match`并退出1；评测入口拒绝运行。
   不事后根据PER放宽规则，不把失败匹配冒充“接近西语”。
5. 子集保留原记录/原音素标签，保存固定IDs、配对密度和SHA256。新目录固定为
   `outputs/pt_es_density_match_v1/`；build拒绝已有目录，绝不覆盖。

选样通过（`status=matched`）后，**容器内H200评测**：

```bash
# 3仅为示例；改为实际分配给你的物理GPU编号
CUDA_VISIBLE_DEVICES=3 python -B scripts/run_pt_density_match.py evaluate
```

不需要新下载。使用现有`/glusterfs_103/models/Qwen3-ASR-1.7B`和已有90类IPA词表；
由于16小时池中新的选中音频未必包含在旧三语cache里，本版显式为选中葡语重新
缓存冻结Encoder的`ln_post`特征，仅处理该validation子集，不读train/test。
完整Qwen加载/Encoder推理只由用户在H200执行，不生成最终转写，不进行优化器更新。
一张H200，建议至少30–40GiB可用显存，并为新特征预留10GiB磁盘；实际用量见报告。
不声称本地模拟测试测得H200耗时。

评测前验证：原三语Head与旧葡语Head的已知SHA，三个Head的词表、权重结构和
Temporal 2x；全部参考cache分片SHA；新特征模型的config/index SHA与原参考cache
身份及dtype一致。候选Head记录实际SHA，只标记`failed_recovery_ablation`，不替换基线。
选样文件SHA、代码SHA、配置均一致后才运行；实际缓存帧数/标签必须与Manifest完全
一致，否则停止，不能用估计密度冒充实际匹配。

矩阵：

| Head | 新匹配葡语 | 旧2662条葡语 | 原2631条西语 |
|---|---|---|---|
| 原三语baseline | 评测 | 同次重评 | 同次重评 |
| 旧葡语专用 | 评测 | 同次重评 | 不作为跨语参照 |
| 失败的去recovery候选 | 评测 | 同次重评 | 不作为基线 |

每项包含micro PER、S/D/I计数、reference数、预测/参考长度比、blank比例，及source/
release分层。S/D/I百分比为各计数/reference数。新子集和旧集可能有重叠，重叠数已报告。
若新PER降低，只能说明该密度匹配视图更易识别；来源/句长等也可能随之改变，
不能把下降全部归因于时间压力，不能证明原标签错误或新标签正确。

每次evaluate创建`outputs/pt_es_density_match_v1/evaluation_<随机>/`，已有outputs
和缓存只读；新特征在该新目录内。**不提供resume**：失败目录保留，重跑evaluate
创建新目录，固定选中ID不变。不删除缓存分片、不覆盖旧评测；本轮不用旧缓存续写模式。
需要重建selection时，用`--config`提供另一个output_dir，不修改已有配置/产物来续跑。

返回小文件：

1. `outputs/pt_es_density_match_v1/selection_report.json`及同目录`sha256.txt`；
2. 本次终端打印的`evaluation_<随机>/report.json`及同目录`sha256.txt`。

不要返回feature tensors、模型或完整outputs。验证命令：

```bash
(cd outputs/pt_es_density_match_v1 && sha256sum -c sha256.txt)
# 用终端打印的实际路径替换随机后缀
R=outputs/pt_es_density_match_v1/evaluation_实际后缀
(cd "$R" && sha256sum -c sha256.txt)
```

本地验证：定向11 passed；全量278 passed / 23 skipped；新增文件Ruff通过；
Mypy新增模块无报错，全仓仍有0.85记录的3处旧unused-ignore；全仓Ruff仍为5处旧E501。
`git diff --check`和CLI help通过。合成10,899候选/2,631参照完成匹配并验证2,631唯一ID；
小样本最优解与穷举一致，测试覆盖预测无关/顺序无关、身份篡改、帧数不符、
未匹配尾部停止、拒绝覆盖以及模拟三个Head共用同一子集/保留旧对照。
本地未加载Qwen或任何真实Checkpoint，不是H200 PER结果。

## 0.92 2026-09-21 MFA100条返回与当前诊断边界

用户返回`outputs/pt_mfa_pilot_v1_880hc9dt`的终端报告及SHA清单，代码提交
`113e0806791ec49853ac58cefcd4214b2d2f0ec5`。报告中4个代码文件SHA已与本地文件
逐一核对一致；validation仍为`196d6e760dbfd626caf566ad333afd999ce6d2770f562add372f657ce9500524`。
结果：100条中98 aligned、1 incomplete_alignment、1 missing_alignment。
词典OOV提示63条、未知词/音素61条、贴音频边界29条、长音素20条，计数可重叠。
这是工具覆盖与疑点提示，不是标注错误率；不能认定98条正确、另外2条脏数据。

报告提供的`report.json` SHA为
`60ac9511b4d6bc8cfce964c38a32634acbb24be075998f90cf7137492cc73ff3`，
选中ID SHA为`4a572cf8770a02096eccd6ead8b3a71ae02d0faece12d1bd9a8561d6da562d94`；
附件为终端粘贴，未取得原始报告字节，不能声称本地复算报告SHA通过。
MFA小试验不扩成全量标签认证。外部Whisper审计路线由用户暂停，只有本地未提交的
`configs/pt_quality_whisper_assets.json`下载清单；未下载模型、未运行外部ASR，
该文件不进入后续密度实验提交。

最新用户决定：从既有16小时葡语validation中选音素密度分布接近西语的子集，
先评测现有Head。该任务是密度控制的评测视图，不能替代音频/标签质量审核。

## 0.91 2026-09-20 MFA模型单独下载，容器内只读本地模型

本节取代0.89的自动下载运行方式。用户要求Git和模型下载在容器外进行，
容器内只运行已准备的MFA声学检查。原100条选样、阈值、参考标签和对齐参数不变。
未重训、未运行Qwen、未将对齐成功解释为标注正确。

交付分支`codex/g2p-coverage-scan`，实现提交标题
`Separate MFA asset download from container pilot`；上一独立结果提交`c84a778`
记录用户返回的证书失败。Git SHA以交付回复及以下命令核对。

**容器外**（Python 3.10+及curl即可，不需要Conda/MFA）：

```bash
cd /home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python3 -B scripts/download_pt_mfa_assets.py
```

下载器只访问官方GitHub两个固定release资产URL，不访问GitHub API列表：

- `acoustic-portuguese_mfa-v2.0.0a/portuguese_mfa.zip`，91,604,181字节；
- `dictionary-portuguese_brazil_mfa-v2.0.0a/portuguese_brazil_mfa.dict`，1,583,153字节。

两者在2026-09-20通过官方release API核对名称、URL及大小；API未提供digest。
下载使用系统curl的HTTPS验证，不设置`-k`/`--insecure`，只允许HTTPS及HTTPS重定向。
记录下载SHA、固定版本、大小及URL到`assets.json`，声学ZIP检查CRC，词典检查UTF-8。
SHA是本次下载/传输完整性记录，不冒充官方签名。若宿主机也报证书错误，停止并
回传错误，需配置管理员提供的可信CA或换可信下载环境，不关闭TLS校验。

完整资产放在忽略目录`models/mfa/pt_pilot_v2_0_0a/`，不提交模型。
成功后重复运行仅校验现有资产，不重下或覆盖；下载失败保留唯一的
`models/mfa/pt_pilot_v2_0_0a_download_<随机>/`临时目录，下次重跑新建临时目录。
若最终目录已存在但缺文件/哈希不符，拒绝覆盖；用`--output-dir NEW_PATH`另行下载，
并给容器入口传`--assets-dir`对应共享路径。无断点续传；源/既有outputs保持不动。

**回到容器内**（共享挂载目录，与上述宿主机仓库对应）：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda run --no-capture-output -n aligner python -B scripts/run_pt_mfa_pilot.py
```

入口先核验本地`assets.json`、固定版本/来源/大小及实际SHA，再调用MFA。
缺文件立即提示单独下载，不自动下载，不回退到旧G2P模型；Git缺失、报错或超时
不会阻断声学检查，报告`git_commit=null`并继续记录代码文件SHA。
MFA以本地资产绝对路径对齐，状态和临时文件仍隔离到本次新输出目录。
每次`outputs/pt_mfa_pilot_v1_<随机>/`，不resume、不覆盖；失败重跑仍选相同ID。

仅返回本次`report.json`和`sha256.txt`（脚本末尾给出完整路径）。若单独下载失败，
只返回终端错误，不继续容器内步骤。验证命令：

```bash
# 容器外：重复下载命令应只返回 verified_existing，验证两份资产SHA
python3 -B scripts/download_pt_mfa_assets.py
# 容器内：用实际本次路径替换后缀
RUN_DIR=outputs/pt_mfa_pilot_v1_实际后缀
(cd "$RUN_DIR" && sha256sum -c sha256.txt)
```

报告应有`asset_policy=local_verified_only_no_download`、两个资产SHA及
`asset_receipt_sha256`；完成时各状态覆盖100条，仍不意味着100条标注正确。

本地：定向16 passed；全量267 passed / 23 skipped；修改文件Ruff、模块Mypy、
`git diff --check`通过。全仓Ruff的5处E501和Mypy的3处unused-ignore均为0.85
已确认旧问题。测试模拟curl/MFA，不在本地下载模型或运行真实对齐；覆盖下载失败
保留文件、重跑只校验、损坏/缺失资产阻止调用、容器无Git及输入文件保持不变。

## 0.90 2026-09-20 MFA首次小试验下载证书失败（用户返回）

用户终端返回`outputs/pt_mfa_pilot_v1_gt0iz6k7`，`status=failed`、
`counts_by_status={}`。`mfa --help`已通过；随后`mfa model download acoustic`
访问`api.github.com`的release列表时，Requests报`CERTIFICATE_VERIFY_FAILED`，
具体为证书链包含不受信任的自签名证书。此时尚未准备音频或执行对齐，不能据此
判断任何样本质量或PER；也不能由此判定MFA运行依赖或Python版本不兼容。
可能涉及容器/Conda信任库与工作区网关证书不一致，但根因尚未取得证书链核实。

本记录依据用户终端输出；尚未返回report.json/sha256.txt或工作区Git SHA，
不能声称已本地重算H200产物哈希。保留失败目录及日志，不清理、不覆盖。
用户明确要求模型单独下载，且容器内Git有问题，Git拉取必须在容器外完成。
下一步拆分模型下载和本地资产对齐，不关闭TLS证书验证。

## 0.89 2026-09-20 葡语100条MFA声学小试验入口

方向：先检查音频、文字及发音参考的可靠性，再冻结验证集、比较三个现有Head。
本轮是检查工具的小试验，不生成clean标签、不重训、不运行Qwen，也不计算PER。
已完成的CV/FLEURS原始文字核对见0.88。用户已授权本轮准备及H200执行。

新增`configs/pt_mfa_pilot.workzone.json`、`scripts/run_pt_mfa_pilot.py`、
`training/pt_mfa_pilot.py`及`training/pt_mfa_runner.py`和定向测试。
工作分支`codex/g2p-coverage-scan`，代码提交标题
`Add stratified Portuguese MFA acoustic pilot`。H200仓库目录执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
conda run --no-capture-output -n aligner python -B scripts/run_pt_mfa_pilot.py
```

确认SHA与交付回复一致；若拉取失败不要reset/清理。只需CPU与网络下载MFA资产，
不需要GPU，不下载Qwen模型。本地只运行合成/模拟单元测试，真实命令仅由用户在H200执行。

固定设计：

- 候选仍为10,899条/约16.24小时的既有validation，严格核对SHA
  `196d6e760dbfd626caf566ad333afd999ce6d2770f562add372f657ce9500524`；不读train/test。
- 五来源各20条，总100条。每个source×release内部按`label_length/effective_ctc_input_length`
  排序（同值按ID），切样本数三分位；对非空release×density层确定性轮流分配配额，
  小层取尽后余额分配给其他层。用seed=20260920与ID的SHA排序选取；保留original/recovery。
- 样本选择不读取预测或PER，也不因后续OOV、解码失败或对齐失败更换样本。
  所有层输出总体数/抽样数/入选比例和密度范围；这是探索性过抽样，不是总体错标率估计。
- `selected_ids.txt`、`selection.json`及其SHA固定选中身份。每次重跑仍选择同一批ID。
- 保留原文本；`.lab`只折叠空白，不删除词/数字/标点，不重写音素参考。
  ffmpeg仅转为16kHz单声道PCM16副本，不裁剪；记录原音频及副本SHA。

模型与运行：

- 在全新隔离`MFA_ROOT_DIR`运行`mfa --help`作实际依赖预检；不根据kalpy元数据null
  擅自安装/升级依赖。预检失败返回报告，停止下载/对齐。
- 固定官方`portuguese_mfa` acoustic和`portuguese_brazil_mfa` dictionary，版本均为
  `v2.0.0a`。官方release已确认文件分别为`portuguese_mfa.zip`（91,604,181字节）和
  `portuguese_brazil_mfa.dict`（1,583,153字节）；下载后记录实际SHA。未在本地下载模型。
- 通过MFA 3.4.0的`mfa model download ... --version v2.0.0a`准备资产，然后
  `mfa align`，JSON导出、4个CPU jobs、`--single_speaker`关闭speaker adaptation；
  未知speaker不伪装成可靠说话人分组。不调用train/adapt/train_dictionary。
- MFA负责声学模型/词典兼容性校验，失败即停止，不合并或强行映射音素。
  不额外生成OOV发音；预检词典未覆盖词及MFA未知音素分别记为工具覆盖问题。

输出和恢复：

- 每次自动创建`outputs/pt_mfa_pilot_v1_<随机后缀>/`，末尾打印绝对路径。
  需要指定路径可加`--output-dir NEW_PATH`，已有路径一律拒绝。
- 模型下载、MFA全局状态、scratch、对齐临时目录、音频副本和日志都位于本次新目录。
  不覆盖既有outputs、模型或MFA全局缓存；子命令使用`--no_clean --no_final_clean --no_overwrite`。
- 不提供resume：失败/中断保留目录，再运行会新建目录，选中ID不变。不要删旧目录。
  每个外部步骤最多1小时，超时停止并报告；网络速度和环境差异可能影响耗时。
- 正常失败报告保存`error`及最多3000字符日志尾部。若用户直接中断，可能只有部分产物，
  仍保留它们，重新运行新目录；不把未完成输出当作结果。

返回与验收：仅回传`report.json`和`sha256.txt`，不传模型、100条音频或完整outputs。
末尾`return_files`给出两者完整路径。工作区可验证：

```bash
# RUN_DIR替换成脚本打印的本次输出目录
RUN_DIR=outputs/pt_mfa_pilot_v1_实际后缀
(cd "$RUN_DIR" && sha256sum -c sha256.txt)
```

完整逐条诊断在`sample_diagnostics.json`，报告只返回来源/分层汇总及每来源最多2条疑点ID。
正常验收检查100个选中ID、五来源各20、SHA通过、各状态计数覆盖全部100条；
`status=completed`仅表示流水线完成，不表示全部对齐成功或标签准确。
优先看准备失败/未对齐比例、词典覆盖、异常是否集中某来源/密度层；工具失败先处理工具，
不能反过来把难对齐样本删成“干净验证集”。

预先冻结的疑点提示：非静音音素≤10.1ms占比≥25%、单音素≥300ms、对齐语音贴近
音频首尾20ms、未知音素/词、词典未覆盖、数字、RMS<0.001、PCM饱和样本占比>1%、
时长与Manifest差>0.1s。它们只是查看线索，正常快速语音/长元音/边界也可能触发。
不把MFA输出当成独立核验的发音标签，不把失败率当成错标率，不据此计算PER改善。

已知检查器偏差：官方葡语模型包含CV、MLS训练来源，主要面向低噪声朗读；
Noah口语可能域外，CV/MLS也可能与检查器训练数据重叠；巴葡词典不保证适合所有pt口音。
参考：[官方声学模型](https://mfa-models.readthedocs.io/en/latest/acoustic/Portuguese/Portuguese%20MFA%20acoustic%20model%20v2_0_0a.html)、
[MFA 3.4对齐参数](https://montreal-forced-aligner.readthedocs.io/en/v3.4.0/user_guide/workflows/alignment.html)、
[模型下载参数](https://montreal-forced-aligner.readthedocs.io/en/v3.4.0/user_guide/models/index.html)。

本地验证：定向pytest 10 passed；全量pytest 261 passed / 23 skipped；新增代码Ruff、
Mypy及`git diff --check`通过。全仓Ruff仍为0.85记录的5处既有E501，Mypy仍为3处
既有unused-ignore，未修改用户PPT文件或相关旧代码。模拟下载、音频转换和对齐用于
验证命令编排/报表与拒绝覆盖，不是真实MFA或H200质量结果。

## 0.88 2026-09-20 CV/FLEURS核对与MFA环境返回

用户返回0.87的两份终端JSON，validation/CV元数据SHA与0.86一致，未另返回H200 Git SHA。
CV共414条：406条文字完全一致，另外8条全部只涉及ASCII双引号、大小写或空白，
未发现实质词序列变更；其中375条精确一致且无反对票、7条格式差异且无反对票，
31条精确一致但有反对票、1条格式差异且有反对票。
FLEURS共42条：41条精确一致，1条也仅是引号格式差异，原始TSV SHA为
`f6fc535628e96a141353c936def89cac59c6bfdb23a7d38e1b508d8479e59e00`。
以上456条未发现转换阶段漏词/改词证据，但不证明原始转写或音素参考符合实际声音，
也不能推广到整个10,899条候选池。

`aligner`环境返回MFA 3.4.0、pynini 2.1.7，mfa/sox/ffmpeg可执行文件存在；
kalpy包元数据为null，不能直接据此判定运行依赖缺失，后续实际MFA预检验证。
已搜索位置仅找到既有`models/mfa/g2p/portuguese_brazil_mfa.zip`，SHA为
`11e7f72ecccbae7905fa064017ac1a1fead9b6ebfdde4e23e506c28edfbcbcc8`，
尚未找到葡语声学模型或词典。不把现有G2P模型当成声学模型使用。

用户确认下一步实施约100条分层随机MFA声学小试验：H200执行，不重训、不改标签，
先验证检查工具的可用性和疑点分布，再决定是否扩展到整批候选及冻结新验证集。

## 0.87 2026-09-20 补齐CV/FLEURS转写核对及MFA只读环境盘点

0.86返回的3条CV差异是引号变化。不能据此把全部8条判为格式变化或错标。
本轮只扩展0.85入口，不筛选验证集、不修订标签、不运行声学模型：

- 保留原始文字匹配计数；对全部差异额外分类，最多返回8条示例；
- 单独识别ASCII双引号/大小写/空白差异，并复用现有G2P词提取器比较词序列和
  数字片段；不会删除重音、漏词或数字差异以制造“一致”；
- 增加CV文字匹配类别与反对票分组的交叉计数。票数是审核信息，不是准确率；
- FLEURS已确认`path/transcription`两列格式，逐条关联候选的完整解析路径与
  TSV所在目录下`train/<filename>`，拒绝仅按basename关联、重复元数据和列数异常；
- MFA盘点仅查询当前Python环境的安装版本、可执行文件位置、已有葡语资产路径/SHA。
  不导入MFA、不调用MFA CLI、不下载或加载模型、不初始化数据库、不读取语料。

H200工作区`codex/g2p-coverage-scan`执行以下短命令（第二条在现有主环境运行，
第三条在已知`aligner`环境中运行）：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
python -B scripts/audit_pt_validation_metadata.py
conda run -n aligner python -B scripts/audit_pt_validation_metadata.py --mfa-inventory
```

用`git rev-parse HEAD`核对交付回复的远端SHA；本轮代码提交标题为
`Extend Portuguese metadata audit and inspect MFA assets`。拉取失败或身份不符先停止，
不清理/reset工作区。两个命令均只打印JSON，不创建输出目录、不覆盖outputs，
不需要resume；中断后重跑即可。只回传两份终端JSON，若命令失败则回传简短错误。

默认元数据配置和validation SHA与0.85相同。新版报告`schema_version=2`，新增
`cv_difference_kinds`、`cv_text_vote_groups`、`fleurs`及FLEURS原始TSV SHA。
正常运行仍为`status=completed`，不代表标签质量认证；两来源checks分别应求和到
对应候选数。`same_g2p_words...`不等于已核验保存的音素ID或真实发音。

MFA盘点的`scope=mfa_package_and_asset_inventory_only`；版本为null表示当前环境未发现
对应包元数据，模型列表为空只表示未在已搜索位置找到。只搜索仓库`models/mfa`及
`MFA_ROOT_DIR`（默认`~/Documents/MFA`）下`pretrained_models`的acoustic/dictionary/g2p，
最多列出20个名称以portug开头的资产；不声称穷尽其他自定义路径。
MFA根目录依据[官方配置说明](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/configuration/index.html)。
盘点有文件不代表可运行或音素集兼容，需根据实际返回决定下一条声学检查命令。

不重复训练重叠检查，不触碰sealed test；原有三套Head和历史validation保持不变。

本地验证：定向pytest 15 passed；全量pytest 251 passed / 23 skipped；修改文件Ruff、
修改模块Mypy及`git diff --check`通过。全仓Ruff/Mypy仍只有0.85记录的既有报错，
未改动相关文件。本地没有读取H200音频或运行MFA，合成测试不代表标注质量结果。

## 0.86 2026-09-20 H200葡语原始转写元数据核对返回

用户返回0.85入口的终端JSON：`status=completed`，validation记录10,899，
候选清单SHA仍为`196d6e760dbfd626caf566ad333afd999ce6d2770f562add372f657ce9500524`。
CV原始`validated.tsv`共161,047行，SHA为
`63424f8c1f5709af9f690330519387deaab5002da154e99424dc47677250e2db`。
本次依据用户粘贴的报告核验字段和计数，未在本地重新读取H200文件；未另回传H200 Git SHA。

- 414条CV候选全部唯一关联到原始审核记录，406条文字精确一致，8条存在文字差异；
- 返回的3条差异均是原文双引号在当前文本中消失，尚不能外推其余5条，不能称为8条错标；
- 审核票数11个分组共414条，其中382条down=0、32条down>0；
  反对票只作为审核分歧证据，不表示确定标注错误；
- locale全部为`pt`，有210个非空说话人ID；332条未记录口音，不能把全部414条称为已确认巴葡；
- FLEURS `train.tsv`已确认两列`path, transcription`，音频路径为文件名；
  只读了前两行，42条候选尚未完成逐条关联；
- `model_loaded=false`、`test_set_used=false`、`files_written=false`。

结论仅限文本来源和审核信息：没有确认发音标签正确，也没有证明剩余PER均是模型错误。
下一步核对全部差异的词序列、FLEURS原始文字，并只读检查MFA声学模型准备情况。
保持原有16.24小时候选池及历史validation，不重训、不改标签、不重复训练重叠检查。

## 0.85 2026-09-18 葡语validation原始转写元数据只读核对入口

当前主线是核实葡语validation的标注依据；暂停重训、人工发音盲审和纯推理入口移植。
训练隔离检查已由用户完成，本入口不重复读取训练清单，不读取sealed test。
不能把`original_ready`、原始文字一致或MFA对齐成功当作逐音素标签准确的证明。

用户无法方便地向H200粘贴长Python命令，因此将现有检查封装为：

- `scripts/audit_pt_validation_metadata.py`：默认加载工作区配置，一条短命令运行；
- `configs/pt_validation_metadata.workzone.json`：保存已确认的候选清单SHA和原始数据路径；
- `src/qwen_hotword/training/pt_validation_metadata.py`：只读核对Common Voice原始
  `validated.tsv`，预览FLEURS **train** TSV前两行的格式，不加载任何模型；
- `tests/test_pt_validation_metadata.py`：覆盖超长字段、未闭合文字引号、列数异常、
  重复元数据、路径精确关联、文字差异、SHA失败和CLI运行。

修复此前内联命令的`csv.Error: field larger than field limit (131072)`：Common Voice
按物理行及制表符解析，保留文字中的引号；严格验证每行列数。格式不符合预期即报错，
不跳过、不修补原始行。该模式下序列化差异也可能造成`text_difference`，必须查看
示例后解释，不能自动认定为错标。核对不使用CTC预测、不筛选新集合、不重写参考标签。

交付分支为`codex/g2p-coverage-scan`。H200仓库目录执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git log -1 --format='%H %s'
python -B scripts/audit_pt_validation_metadata.py
```

拉取后的SHA应与本次交付回复提供的远端SHA一致，提交标题为
`Add read-only Portuguese validation metadata audit`。若拉取失败或SHA不符，先停止，
不要reset、覆盖或清理工作区。

默认配置使用`outputs/pt_combined_temporal2x_v1/full_ctc_validation.jsonl`，其SHA必须为
`196d6e760dbfd626caf566ad333afd999ce6d2770f562add372f657ce9500524`；不匹配直接退出。
配置中的相对路径相对仓库根目录解析；其他环境可通过`--config PATH`提供同结构JSON。

输出目录政策：本入口不创建输出目录，不修改任何outputs，只向终端打印紧凑JSON。
`-B`避免新增Python字节码文件。不需要`--resume`；中断后直接重复同一命令。
无需下载模型、申请GPU或传输完整语料。仅回传打印的JSON；报错则回传简短错误。

验收：命令退出码为0，JSON中`status=completed`、`model_loaded=false`、
`test_set_used=false`、`files_written=false`，`cv_checks`各项总数等于`cv_candidates`。
报告包含候选清单和CV原始元数据SHA、文字匹配分类、审核票数/口音分布、已知说话人数、
最多3条差异示例及FLEURS前2行格式。示例字段最多240字符并标记截断，审核票数最多20组。

`completed`仅表示检查运行完成；缺失/重复元数据、空文字、文字差异均作为事实报告，
不自动改为“数据干净/不干净”。FLEURS当前仅看格式，尚未完成原始转写关联。
本阶段不运行MFA，也不核实实际发音。收到结果后再确定FLEURS逐条关联和独立声学检查。
本地合成测试只验证程序行为，不能当作H200质量结果。

本地验证：定向pytest 8 passed；全量pytest 244 passed / 23 skipped；新增文件Ruff、
新增模块Mypy和`git diff --check`通过。全仓`ruff check .`另报用户未跟踪PPT脚本的
2处E501（未修改）及既有`scan_g2p_coverage.py`的3处E501；`mypy src`另报既有
`ctc_overfit.py`、`sharded_ctc.py`、`unfrozen_encoder_ctc.py`的3处unused-ignore。
在原始`1e28ad1`独立导出副本复跑，确认仓库内上述Ruff/Mypy报错已存在，未因本次新增。

## 0.84 2026-09-16 葡语original-ready同曝光CTC Head消融训练

0.83同集诊断已在H200完成并通过：2,662条葡语validation上，当前三语
Head PER为9.6618%，旧葡语专用Head为9.0324%，绝对差0.6294点。三语Head的
`original_ready`/`temporal_2x_recovery` PER分别为8.4366%/11.8273%，旧葡语Head
分别为7.8537%/11.1156%。高标签密度三分位在两个Head上均显著更差，因此下一步
先验证recovery训练样本是否会拖累Head，不修改Head结构。

当前平衡train中葡语为100,499条/150.000337小时：

```text
original_ready:       69,918条 / 108.522236小时
temporal_2x_recovery: 30,581条 /  41.478101小时
```

本次是低成本因果消融，不冒充为新的150小时唯一样本数据集：

- 英语、西语每条仍每epoch恰好一次；
- 葡语30,581条`temporal_2x_recovery`全部从训练采样排除；
- 从69,918条已缓存的葡语`original_ready`中，每epoch按`source_corpus`分层
  确定性补抽30,581次，保持各corpus的记录曝光数与基线一致；
- 总训练样本曝光仍为310,257条/epoch，不因删除recovery减少optimizer step；
- 每epoch重抽子集随epoch变化，但由seed和来源名确定，resume可复现；
- 复用现有43.3 GB的三语train feature cache，不加载Qwen Encoder、不重建cache、
  不读sealed test；
- validation保持完整8,101条和en/es/pt Macro选模，不删除困难验证样本。

新采样plan严格验证train Manifest与cache ID完全一致，并将Manifest SHA、cache
fingerprint、包含ID SHA、各来源pool/补抽数和plan fingerprint写入
`train_sampling_plan.json`。该fingerprint已绑定training state；改变采样后不会误resume。

工作区拉取并确认本节交付SHA：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

首次训练必须使用全新目录：

```bash
GPU_ID=3
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1
BASELINE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1
ABLATION_ROOT=outputs/en_es_pt_balanced_150h_pt_original_resample_ctc_formal_macro_v1

test -f "$BASELINE_ROOT/ctc_head_best.pt"
test ! -e "$ABLATION_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_full_ctc.py \
  --train-cache "$CACHE_ROOT/train" \
  --validation-cache "$CACHE_ROOT/validation" \
  --train-manifest "$TRAIN_ROOT/full_ctc_train.jsonl" \
  --validation-manifest "$VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab "$VOCAB" \
  --output-dir "$ABLATION_ROOT" \
  --device cuda:0 \
  --epochs 30 \
  --minimum-epochs 5 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.001 \
  --early-stopping-metric validation_macro_per \
  --checkpoint-selection-metric validation_macro_per \
  --validation-group-column balanced_language_bucket \
  --expected-validation-groups en,es,pt \
  --train-batch-size 256 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --max-gradient-norm 5 \
  --scheduler-patience 2 \
  --scheduler-factor 0.5 \
  --minimum-learning-rate 0.00001 \
  --seed 20260825 \
  --log-every-shards 25 \
  --head-type temporal_upsample \
  --head-hidden-dimension 512 \
  --head-kernel-size 5 \
  --head-dropout 0.1 \
  --head-time-upsampling-factor 2 \
  --train-sampling-policy pt_original_ready_equal_record_exposure
```

若中断，保留原目录，删除上面的`test ! -e` 验收行，在完全相同命令末尾加
`--resume`。不改seed、数据、采样policy或超参强续原目录。

完成后先做紧凑验收：

```bash
jq '{status,policy,cache_sample_count,included_unique_sample_count,
  excluded_portuguese_recovery_sample_count,
  portuguese_original_unique_sample_count,
  portuguese_original_oversample_pool_count,
  additional_portuguese_original_draws_per_epoch,epoch_sample_count,
  language_input,portuguese_release_input,portuguese_source_release_input,
  identity,fingerprint,test_set_used}' "$ABLATION_ROOT/train_sampling_plan.json"

jq '{status,train_sampling_policy,train_sampling_plan_fingerprint,
  train_samples_per_epoch,best_epoch,best_validation_phoneme_error_rate,
  best_validation_macro_phoneme_error_rate,best_validation_by_group,
  early_stopped,test_set_used,cache_sha256_verified}' "$ABLATION_ROOT/report.json"

jq -c '{epoch,learning_rate,train_per:.train.phoneme_error_rate,
  validation_per:.validation.phoneme_error_rate,
  macro_per:.validation_macro_phoneme_error_rate,
  by_group:(.validation_by_group|with_entries(.value=.value.phoneme_error_rate)),
  epoch_seconds}' "$ABLATION_ROOT/metrics.jsonl"

sha256sum "$ABLATION_ROOT/ctc_head_best.pt" \
  "$ABLATION_ROOT/report.json" \
  "$ABLATION_ROOT/metrics.jsonl" \
  "$ABLATION_ROOT/train_sampling_plan.json"
```

再将新三语Head放到完全相同的葡语2,662条validation上，与旧葡语专用Head
做同集分层诊断。上轮当前三语Head的诊断作为已冻结baseline，不覆盖：

```bash
PT_HEAD=outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt
PT_DIAG_ROOT=outputs/en_es_pt_balanced_ctc_pt_original_resample_stratified_diagnostics_v1

test ! -e "$PT_DIAG_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/diagnose_portuguese_ctc.py \
  --validation-cache "$CACHE_ROOT/validation" \
  --validation-manifest "$VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab "$VOCAB" \
  --multilingual-checkpoint "$ABLATION_ROOT/ctc_head_best.pt" \
  --portuguese-checkpoint "$PT_HEAD" \
  --output-dir "$PT_DIAG_ROOT" \
  --device cuda:0 \
  --batch-size 256

(cd "$PT_DIAG_ROOT" && sha256sum -c sha256.txt)
```

请回传以下小文件，不需要checkpoint、training state、feature cache或音频：

```text
en_es_pt_balanced_150h_pt_original_resample_ctc_formal_macro_v1/
  train_sampling_plan.json
  report.json
  metrics.jsonl
en_es_pt_balanced_ctc_pt_original_resample_stratified_diagnostics_v1/
  portuguese_ctc_diagnostics.json
  sha256.txt
```

只有当新Head在未改动的完整validation上降低葡语PER，且英/西PER没有不可接受退化，
才构建真正的“150小时唯一original-ready葡语”新Manifest，用未选中的同corpus
original-ready样本替换重抽；本轮不重建数据或Encoder cache。

## 0.83 2026-09-16 葡语CTC同集分层诊断与新旧Head对比

当前最高优先级已从端到端formal100切回CTC Head本身：最终三语Head在同一平衡
validation上的English/Spanish/Portuguese PER分别为4.736%/3.910%/9.662%。葡语错误中
deletion为4.205%，且reference phoneme/input frame约0.435，明显高于英语0.207和西语
0.347。本轮只做只读归因，不训练、不重建feature cache、不加载Qwen Encoder、不读取
sealed test，也不修改端到端、Anchor、Prompt或4k主线。

新增：

- `scripts/diagnose_portuguese_ctc.py`：从现有三语validation feature cache严格筛选
  `balanced_language_bucket=pt`，在完全相同的hidden states和音素标签上依次评估最终三语
  Head与旧葡语专用Head；
- `src/qwen_hotword/training/portuguese_ctc_diagnostics.py`：同时按`source_corpus`、
  `release_source`、二者交叉及每条样本`reference_tokens/effective_ctc_input_frames`稳定
  三分位统计PER、substitution、deletion、insertion、预测/参考长度比和blank比例；
- 逐样本配对比较记录三语Head相对葡语Head更好/更差/持平的数量及错误总差，并只保存
  最多50条高PER或退化样本用于音频文本边界/G2P复核；
- 两个checkpoint都必须匹配同一90类词表、1024维输入和Temporal 2×契约，否则在创建
  结果结论前明确失败，不把不同标签空间或时间轴冒充公平对比。

工作区拉取并确认本节交付SHA：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

只运行现有cache上的两个小型Head。输出目录必须全新；本工具不支持resume，失败时
回传错误，不删除或覆盖任何旧outputs：

```bash
GPU_ID=3
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
CACHE=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1/validation
MULTILINGUAL_HEAD=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
PORTUGUESE_HEAD=outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt
OUTPUT=outputs/en_es_pt_balanced_ctc_pt_stratified_diagnostics_v1

test -f "$VOCAB"
test -f "$VALIDATION_ROOT/full_ctc_validation.jsonl"
test -f "$CACHE/cache_summary.json"
test -f "$MULTILINGUAL_HEAD"
test -f "$PORTUGUESE_HEAD"
test ! -e "$OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/diagnose_portuguese_ctc.py \
  --validation-cache "$CACHE" \
  --validation-manifest "$VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab "$VOCAB" \
  --multilingual-checkpoint "$MULTILINGUAL_HEAD" \
  --portuguese-checkpoint "$PORTUGUESE_HEAD" \
  --output-dir "$OUTPUT" \
  --device cuda:0 \
  --batch-size 256
```

完成后只需执行以下紧凑验收：

```bash
(cd "$OUTPUT" && sha256sum -c sha256.txt)

jq 'def m: {
  samples: .sample_count,
  ref_ph: .reference_tokens,
  frames: .input_frames,
  per: .phoneme_error_rate,
  sub: (.substitutions / .reference_tokens),
  del: (.deletions / .reference_tokens),
  ins: (.insertions / .reference_tokens),
  hyp_ref: .hypothesis_reference_length_ratio
}; {
  status,
  selection,
  pressure_stratification,
  checkpoint_sha256: (.inputs.checkpoints | with_entries(.value |= .sha256)),
  checkpoints: (.checkpoints | with_entries(.value |= {
    overall: (.validation | m),
    by_dimension: (.validation_by_dimension |
      with_entries(.value |= with_entries(.value |= m)))
  })),
  comparison
}' "$OUTPUT/portuguese_ctc_diagnostics.json"

wc -l "$OUTPUT/top_error_samples.jsonl"
```

请回传`portuguese_ctc_diagnostics.json`和`sha256.txt`两个小文件；只有需要人工复核具体
错配样本时再回传`top_error_samples.jsonl`，不需要checkpoint、feature cache或音频。
结果判定顺序固定为：先看release/source差异，再看压力桶单调性，最后看旧葡语Head是否
在各桶一致优于三语Head。只有证据指向共享训练干扰时才讨论葡语增采样/语言条件化Head；
若两个Head都在同一来源或高压力样本上退化，则优先审计标签、G2P与音频文本边界，不盲目
增加同类语料时长。

## 0.82 2026-09-15 Delivery in/out 165词Top-5与Top-7重跑

本轮使用新的Delivery专用热词文件，只运行Delivery 2,071条WAV，
不读取或重跑MLS：

```text
/host_home/star/q00933266/qwen3-asr-hotword/pt_keyword_bias_phoneme-Delivery_20260706-inout.json
```

工区静态审计确认：`keyword_sets.baseline`为空，必须选择`all_keywords`；
`all_keywords`共165个词，去重后仍为165个，165个词均有同名MFA音素映射。
是否全部可映射到当前90类CTC词表由下述CPU预检严格验证；如有OOV则在加载
完整模型前中止，不静默丢弃热词。

拉取已支持Delivery-only Top-7精确重放的分支，最终SHA以本轮交付消息为准：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

设置本轮输入与全新输出目录：

```bash
MODEL=/glusterfs_103/models/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
KEYWORDS_INOUT=/host_home/star/q00933266/qwen3-asr-hotword/pt_keyword_bias_phoneme-Delivery_20260706-inout.json
DELIVERY_AUDIO=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/wav-total
DELIVERY_TEXT=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/transcripts.txt
DELIVERY_D5_ROOT=outputs/pt_external_keyword_retrieval_delivery_inout_top5_v1
DELIVERY_D7_ROOT=outputs/pt_external_keyword_retrieval_delivery_inout_top7_v1

test -f "$MODEL/config.json"
test -f "$CTC_CHECKPOINT"
test -f "$VOCAB"
test -f "$KEYWORDS_INOUT"
test -d "$DELIVERY_AUDIO"
test -f "$DELIVERY_TEXT"
test ! -e "$DELIVERY_D5_ROOT"
test ! -e "$DELIVERY_D7_ROOT"
```

Top-5 CPU预检，该命令不加载模型：

```bash
python scripts/run_external_keyword_retrieval.py --model "$MODEL" --ctc-checkpoint "$CTC_CHECKPOINT" --vocab "$VOCAB" --keyword-bias "$KEYWORDS_INOUT" --keyword-set all_keywords --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" --output-dir "$DELIVERY_D5_ROOT" --device cuda:0 --dtype bfloat16 --audit-only

(cd "$DELIVERY_D5_ROOT" && sha256sum -c sha256.txt)
jq '{status, keyword_set, keyword_count, vocabulary_size, oov_keyword_count,
  keywords_below_four_phonemes}' "$DELIVERY_D5_ROOT/keyword_audit.json"
jq '{status, sample_count, sources}' "$DELIVERY_D5_ROOT/dataset_audit.json"
```

必须确认`keyword_set=all_keywords`、`keyword_count=165`、`oov_keyword_count=0`，
并且只有Delivery 2,071条数据。预检不通过时停止，不启动GPU。

预检通过后在同一目录加`--resume`运行Top-5；物理GPU 3仍映射为逻辑
`cuda:0`，如GPU 3不空闲可只替换`GPU_ID`：

```bash
GPU_ID=3
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_external_keyword_retrieval.py --model "$MODEL" --ctc-checkpoint "$CTC_CHECKPOINT" --vocab "$VOCAB" --keyword-bias "$KEYWORDS_INOUT" --keyword-set all_keywords --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" --output-dir "$DELIVERY_D5_ROOT" --device cuda:0 --dtype bfloat16 --resume
```

中断时原样重跑最后一条命令，不删除`sample_shards`。Top-5完成后先校验源运行，
再在CPU上对已保存的Anchor排名精确重放Top-7；不重读音频、不重跑Encoder或
CTC Head。Top-5和Top-7共享threshold 0.75、posterior minimum 0.5、maximum
edit ratio 0.35，唯一改变为Top-K：

```bash
(cd "$DELIVERY_D5_ROOT" && sha256sum -c sha256.txt)

python scripts/replay_external_keyword_top7.py --source-run "$DELIVERY_D5_ROOT" --vocab "$VOCAB" --keyword-bias "$KEYWORDS_INOUT" --keyword-set all_keywords --output-dir "$DELIVERY_D7_ROOT"

(cd "$DELIVERY_D7_ROOT" && sha256sum -c sha256.txt)
```

Top-5源结果已经是只含Delivery的目标schema。为了与Top-7交付文件同名，
不覆盖源文件，只新增一个内容完全相同的显式交付副本：

```bash
test ! -e "$DELIVERY_D5_ROOT/delivery_retrieval_output.json"
cp "$DELIVERY_D5_ROOT/retrieval_output.json" "$DELIVERY_D5_ROOT/delivery_retrieval_output.json"
cmp "$DELIVERY_D5_ROOT/retrieval_output.json" "$DELIVERY_D5_ROOT/delivery_retrieval_output.json"
sha256sum "$DELIVERY_D5_ROOT/delivery_retrieval_output.json" > "$DELIVERY_D5_ROOT/delivery_retrieval_output.sha256"

jq 'keys | length' "$DELIVERY_D5_ROOT/delivery_retrieval_output.json"
jq 'keys | length' "$DELIVERY_D7_ROOT/delivery_retrieval_output.json"
wc -l "$DELIVERY_D7_ROOT/top7_replay_details.jsonl"
cat "$DELIVERY_D7_ROOT/topk_comparison.md"
```

三个数量必须都是2,071。最终交给Context Learning的两个新文件为：

```text
Top-5:
outputs/pt_external_keyword_retrieval_delivery_inout_top5_v1/delivery_retrieval_output.json

Top-7:
outputs/pt_external_keyword_retrieval_delivery_inout_top7_v1/delivery_retrieval_output.json
```

两者都保持每个音频stem映射到`[{"word": ..., "phoneme": ...}]`的原交付schema。
请回传CPU预检摘要、两个SHA校验、三个数量以及`topk_comparison.md`；不需要打包
音频、checkpoint、sample shards或完整详情。得到实测结果后再以独立结果提交
更新本节。

## 0.81 2026-09-14 Delivery专用热词表重跑与单来源Top-7交付

前一轮MLS和Delivery共用了`pt_keyword_bias_phoneme.json`的`hard_k266`，因此
Delivery统计中的174/179=97.21%最终Recall只覆盖这266个活动热词；不在
`hard_k266`中的Delivery词没有进入Anchor索引，不可能被返回。`shortlist_size=64`
是每条音频从完整活动热词表中取得的Anchor精排上限，不是全局只加载64个词。

新的Delivery词表位于工区：

```text
/host_home/star/q00933266/qwen3-asr-hotword/pt_keyword_bias_phoneme_sd.json
```

已回传的静态结构审计显示：`keyword_sets.all_keywords`有362个不同词面，
`keyword_phonemes`也有362个键。后续审计已确认唯一的键不一致是集合词面`"Dra. "`
末尾多一个空格，而对应音素键是`"Dra."`；音素并未缺失。
`baseline`为空集合。新集合与旧`hard_k266`精确字符串重合9个，并集为619个。
本轮必须使用`all_keywords`。保留原始文件不变，另行生成Delivery专用派生词表，
只对该精确词面去除末尾空格；不删除热词、不猜测或修改MFA音素。

代码将`scripts/replay_external_keyword_top7.py`的已验证CPU精确重放扩展为接受
MLS/Delivery的任意非空已知子集。新的Delivery-only源运行不再因缺少MLS而拒绝，
只生成`delivery_retrieval_output.json`；旧的MLS+Delivery双来源行为、指标及两个
独立交付文件保持不变。

### 0.81.1 拉取与新词表阻塞审计

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

NEW_KEYWORDS=/host_home/star/q00933266/qwen3-asr-hotword/pt_keyword_bias_phoneme_sd.json

jq '
  . as $root |
  {
    missing_phoneme_words: [
      $root.keyword_sets.all_keywords[]
      | select(($root.keyword_phonemes[.] // "") == "")
    ],
    extra_phoneme_keys: [
      ($root.keyword_phonemes | keys[]) as $key
      | select(($root.keyword_sets.all_keywords | index($key)) == null)
      | $key
    ]
  }
' "$NEW_KEYWORDS"
```

回传结果已精确确认`missing_phoneme_words=["Dra. "]`、
`extra_phoneme_keys=["Dra."]`。生成不覆盖原文件的派生Delivery词表：

```bash
DELIVERY_KEYWORD_ROOT=outputs/pt_external_keyword_retrieval_delivery_sd_keyword_table_v1
DELIVERY_KEYWORDS="$DELIVERY_KEYWORD_ROOT/pt_keyword_bias_phoneme_delivery.json"

test ! -e "$DELIVERY_KEYWORD_ROOT"
mkdir -p "$DELIVERY_KEYWORD_ROOT"

jq '
  .keyword_sets.all_keywords |= map(
    if . == "Dra. " then "Dra." else . end
  )
' "$NEW_KEYWORDS" > "$DELIVERY_KEYWORDS"

jq '
  . as $root |
  {
    keyword_count: ($root.keyword_sets.all_keywords | length),
    unique_keyword_count: ($root.keyword_sets.all_keywords | unique | length),
    phoneme_mapping_count: ($root.keyword_phonemes | length),
    missing_phoneme_words: [
      $root.keyword_sets.all_keywords[]
      | select(($root.keyword_phonemes[.] // "") == "")
    ]
  }
' "$DELIVERY_KEYWORDS"

sha256sum "$NEW_KEYWORDS" "$DELIVERY_KEYWORDS"
```

派生审计必须显示362/362/362且`missing_phoneme_words=[]`。

### 0.81.2 Delivery-only CPU预检

只有在362/362词都具备音素映射后才执行。使用新目录，不覆盖旧的
MLS/Delivery共用词表结果：

```bash
MODEL=/glusterfs_103/models/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
NEW_KEYWORDS=/host_home/star/q00933266/qwen3-asr-hotword/pt_keyword_bias_phoneme_sd.json
DELIVERY_KEYWORDS=outputs/pt_external_keyword_retrieval_delivery_sd_keyword_table_v1/pt_keyword_bias_phoneme_delivery.json
DELIVERY_AUDIO=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/wav-total
DELIVERY_TEXT=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/transcripts.txt
DELIVERY_OUTPUT=outputs/pt_external_keyword_retrieval_delivery_sd_v1
DELIVERY_TOP7_OUTPUT=outputs/pt_external_keyword_retrieval_delivery_sd_top7_replay_v1

test -f "$MODEL/config.json"
test -f "$CTC_CHECKPOINT"
test -f "$VOCAB"
test -f "$NEW_KEYWORDS"
test -f "$DELIVERY_KEYWORDS"
test -d "$DELIVERY_AUDIO"
test -f "$DELIVERY_TEXT"
test ! -e "$DELIVERY_OUTPUT"
test ! -e "$DELIVERY_TOP7_OUTPUT"

python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$DELIVERY_KEYWORDS" \
  --keyword-set all_keywords \
  --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" \
  --output-dir "$DELIVERY_OUTPUT" \
  --audit-only

(cd "$DELIVERY_OUTPUT" && sha256sum -c sha256.txt)
jq '{status, keyword_set, keyword_count, vocabulary_size, oov_keyword_count,
  keywords_below_four_phonemes}' "$DELIVERY_OUTPUT/keyword_audit.json"
jq '{status, sample_count, sources}' "$DELIVERY_OUTPUT/dataset_audit.json"
```

预检必须显示`keyword_set=all_keywords`、`keyword_count=362`、`oov_keyword_count=0`、
Delivery 2,071条音频/转写一一匹配且没有MLS来源。任一不满足时停止，不启动GPU。

### 0.81.3 H200完整检索、resume及Top-7交付

预检通过后在同一D5目录上加`--resume`。该命令只读Delivery，不加载MLS：

```bash
GPU_ID=3
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$DELIVERY_KEYWORDS" \
  --keyword-set all_keywords \
  --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" \
  --output-dir "$DELIVERY_OUTPUT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
```

中断时原样重跑上述命令，保留`sample_shards`。D5完成并校验后，CPU精确重放
Top-7；共享参数为threshold 0.75、posterior minimum 0.5、maximum edit ratio 0.35，
唯一改变为Top-K 5到7：

```bash
(cd "$DELIVERY_OUTPUT" && sha256sum -c sha256.txt)

python scripts/replay_external_keyword_top7.py \
  --source-run "$DELIVERY_OUTPUT" \
  --vocab "$VOCAB" \
  --keyword-bias "$DELIVERY_KEYWORDS" \
  --keyword-set all_keywords \
  --output-dir "$DELIVERY_TOP7_OUTPUT"

(cd "$DELIVERY_TOP7_OUTPUT" && sha256sum -c sha256.txt)
wc -l "$DELIVERY_TOP7_OUTPUT/top7_replay_details.jsonl"
jq 'keys | length' "$DELIVERY_TOP7_OUTPUT/delivery_retrieval_output.json"
cat "$DELIVERY_TOP7_OUTPUT/topk_comparison.md"
```

预期逐条详情和最终交付JSON都覆盖2,071条Delivery音频。交给Context Learning
的文件只是：

```text
outputs/pt_external_keyword_retrieval_delivery_sd_top7_replay_v1/delivery_retrieval_output.json
```

schema与旧交付完全相同：每个音频stem映射到0至7个`{"word": ..., "phoneme": ...}`
对象。请回传SHA校验、两个行数/键数和`topk_comparison.md`终端输出；不需要打包音频、
checkpoint、sample shards或完整详情。

本轮代码只改动Top-7交付工具的来源子集支持，不修改CTC、Anchor排名、门控、
三语训练、4k评测或旧输出。本地定向13项测试和全仓254项pytest通过；Ruff、
format、相关模块strict Mypy（skip第三方imports）和`git diff --check`通过。不带
`--follow-imports=skip`的Mypy仍会因当前NumPy stub的Python 3.12 `type`语句与项目
Python 3.10解析目标冲突而在第三方依赖解析阶段中止，不是本轮类型错误。

## 0.80 2026-09-11 当前两条推理入口的迁移与使用说明

新增`docs/INFERENCE_USAGE.md`，集中记录当前代码仓已经实现的两条推理路径、准确能力边界、
必需输入、运行命令、resume约束、输出文件、时延口径及新机器迁移清单。本节仅补文档，
不修改模型、CTC、Anchor、门控、Prompt、Qwen解码或任何既有输出。

两条现有入口明确区分如下：

1. `scripts/run_external_keyword_retrieval.py`是完整WAV/FLAC到CTC Anchor热词列表的入口，
   当前验证配置为266词、threshold 0.75、posterior minimum 0.5、Top-5。它不初始化vLLM、
   不运行Qwen文本decoder；transcript只用于检索后指标统计。完成的D5源运行可以通过
   `scripts/replay_external_keyword_top7.py`在CPU上精确重放Top-7。
2. `scripts/run_streaming_rag_evaluation.py`是CTC + Anchor + Prompt + Qwen的2秒流式端到端
   入口，但当前仍是formal评测型CLI：需要validation Manifest、cases、offline选择、CTC
   report和families等身份资产，不接受任意单条`--audio`。

讨论中的`scripts/transcribe_with_anchor_rag.py`尚未实现，不能作为当前迁移命令。未来纯
推理入口应在复用现有`streaming_backends.py`、`streaming_core.py`、Anchor和Prompt模块的
基础上剥离评测依赖，支持单音频/目录及完整音频/2秒流式模式；该工作不在本次文档变更中。

三语CTC Head使用同一个checkpoint；当前端到端评测仍按语种分别运行，并向Qwen传
`English`、`Spanish`或`Portuguese`。Qwen官方接口具备`language=None`自动识别，但本仓库
CLI尚未实现`--language auto`到`None`的映射，也没有完成自动识别后的热词子库与Prompt
切换；因此文档不把`--language auto`列为已支持功能。

工区拉取文档交付：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

最终提交SHA以本轮Git交付消息为准。完整命令见`docs/INFERENCE_USAGE.md`：路径一先用
`--audit-only`创建新输出目录，正式运行必须对同一目录加`--resume`；路径二只在配置与输入
身份完全相同时使用`--resume`。模型、三语CTC checkpoint、数据、Manifest和`outputs/`
通常不在Git，迁移时必须单独复制并核验。需要回传排查时优先提供`run_config.json`、摘要、
端到端运行的`latency_summary.json`和`sha256.txt`；完整音频检索的时延位于
`evaluation_summary.json`。不回传模型、音频、feature cache或完整sample shards。

## 0.79 2026-09-10 外部测试集D5/D7 Top-K隔离结果

0.78的CPU精确重放已在工区完成。重放基于0.76--0.77保存的逐条Anchor原始Top-20，
没有重新加载模型、读取音频或执行Qwen decoder；共享门控固定为
threshold=0.75、minimum posterior=0.5、maximum edit ratio=0.35、
posterior weight=0.25、margin=0，唯一变化是Top-K从5增加到7。

完整性核验通过：

~~~text
run_config.json:                 OK
top7_replay_details.jsonl:       OK
topk_comparison.json:            OK
topk_comparison.md:              OK
mls_retrieval_output.json:       OK
delivery_retrieval_output.json:  OK
README.md:                       OK

top7_replay_details.jsonl:       2,942 rows
mls_retrieval_output.json:       871 keys
delivery_retrieval_output.json:  2,071 keys
~~~

结果表：

| 来源 | 配置 | Raw Recall | 最终检索Recall | 最终检索Precision | 选中热词 | 纯负样本FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 总计 | D5 Top-5 | 754/879 = 85.78% | 705/879 = 80.20% | 705/2,231 = 31.60% | 2,231 | 864/2,361 = 36.59% |
| 总计 | D7 Top-7 | 777/879 = 88.40% | 712/879 = 81.00% | 712/2,243 = 31.74% | 2,243 | 864/2,361 = 36.59% |
| MLS | D5 Top-5 | 578/700 = 82.57% | 531/700 = 75.86% | 531/1,271 = 41.78% | 1,271 | 250/465 = 53.76% |
| MLS | D7 Top-7 | 601/700 = 85.86% | 538/700 = 76.86% | 538/1,283 = 41.93% | 1,283 | 250/465 = 53.76% |
| Delivery | D5 Top-5 | 176/179 = 98.32% | 174/179 = 97.21% | 174/960 = 18.12% | 960 | 614/1,896 = 32.38% |
| Delivery | D7 Top-7 | 176/179 = 98.32% | 174/179 = 97.21% | 174/960 = 18.12% | 960 | 614/1,896 = 32.38% |

D7相对D5的精确增量：

- 总计Raw命中增加23个，Raw Recall提升2.62个百分点；
- 最终正确检索增加7个，最终检索Recall提升0.80个百分点；
- 总选中项增加12个，其中7个正确、5个错误，新增项本身Precision为58.33%；
- 总体最终Precision从31.60%微升到31.74%，纯负样本FPR完全不变；
- 所有增量均来自MLS：MLS Raw Recall提升3.29个百分点，最终Recall提升1.00个百分点，
  Precision从41.78%微升到41.93%；
- Delivery的Raw、门控输出、Recall、Precision和纯负样本FPR全部不变，说明该来源在
  当前分数与门控下没有可由第6/7槽位释放的候选。

结论：Top-7在当前固定门控下对汇总Recall、Precision和纯负样本FPR均无回归，
可以作为本次Context Learning交付版本；它改善了MLS，但总体只新增12个输出项，
其中仍包含5个错误项，说明Top-K不是当前召回缺口的主要
限制。剩余问题仍主要位于MLS的上游排名与门控覆盖，以及两个来源都较高的错误候选量；
尤其Delivery虽然Recall达到97.21%，最终Precision只有18.12%，不能因Recall高而忽略
Prompt噪声风险。纯负样本FPR不变只说明Top-7没有新增“空真值样本是否出现任意输出”的
回归，不代表正样本内的错误候选没有增加；本轮新增的5个错误项发生在已有输出的样本中。

本节的“最终检索Recall”仍是完整音频到CTC Anchor门控列表的Recall，不是Qwen最终转写
Recall。时延沿用源D5实测值，CPU重放没有重新测量Top-7的Encoder/Anchor时延，因此本节
不声称Top-7具有一套独立端到端时延数据。

本次Top-7下游交付文件位于：

~~~text
outputs/pt_external_keyword_retrieval_mls_delivery_top7_replay_v1/
  mls_retrieval_output.json
  delivery_retrieval_output.json
~~~

二者已按来源拆分，schema保持音频stem映射到0至7个word/phoneme对象，可直接交给
Context Learning下游。完整逐条详情和comparison文件留在工区审计，不需要对外传递。

## 0.78 2026-09-09 外部测试集Top-7精确重放及MLS/Delivery独立交付

0.76--0.77的完整运行已为2,942条外部音频逐条保存Anchor原始Top-20排名。
本轮不再加载Qwen模型、CTC Head或音频，而是在CPU上使用同一排名精确重放
threshold=0.75 / minimum posterior=0.5 / Top-7，与原Top-5隔离对比Top-K影响。

新增：

- src/qwen_hotword/inference/external_keyword_topk_replay.py：校验源运行SHA、词表和门控，
  逐条复现已存Top-5，再重放Top-7并统计总计/MLS/Delivery的raw Recall、
  最终检索Recall、Precision和纯负样本FPR。若任一条的已存排名不足以证明精确Top-7，
  在创建输出目录前失败，不会将截断排名冒充完整结果。
- scripts/replay_external_keyword_top7.py：独立CPU入口，只接受已完成的固定D5源运行，
  不覆盖现有产物。
- tests/test_external_keyword_topk_replay.py：覆盖D5逐条复现、D7新增第6/7候选、
  排名截断拒绝、源结果不一致拒绝和两个独立交付JSON。

新输出目录包含：

~~~text
topk_comparison.json              机器可读的总计/分来源D5对D7指标及差值
topk_comparison.md                可直接放入总结的D5/D7表格
top7_replay_details.jsonl         逐条重放审计，共2,942行
mls_retrieval_output.json         MLS独立Top-7交付，预期871个key
delivery_retrieval_output.json    Delivery独立Top-7交付，预期2,071个key
run_config.json / README.md / sha256.txt
~~~

两个交付JSON保持Context Learning要求的原schema：音频stem为key，value为0至7个
word/phoneme对象，未召回时保留空数组。topk_comparison.md的Top-5行从已保存详情重算，
并与逐条已存选择绑定；Top-7行只改top_k=5到7。raw口径对应各自的raw recall@5和
raw recall@7。

时延不重新测量：表格JSON保留源D5模型运行中实测的纯检索和音频到结果时延，并明确标记
为共享的源运行观测值。CPU重放只对保存排名做常数级截断，不冒充Top-7的Encoder/Anchor
时延重测。

工区拉取本节最终交付SHA后执行：

~~~bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
KEYWORDS=pt_keyword_bias_phoneme.json
EXTERNAL_OUTPUT=outputs/pt_external_keyword_retrieval_mls_delivery_v1
TOP7_OUTPUT=outputs/pt_external_keyword_retrieval_mls_delivery_top7_replay_v1

test -f "$EXTERNAL_OUTPUT/run_config.json"
test -f "$EXTERNAL_OUTPUT/retrieval_details.jsonl"
test -f "$EXTERNAL_OUTPUT/evaluation_summary.json"
test -f "$EXTERNAL_OUTPUT/sha256.txt"
test -f "$VOCAB"
test -f "$KEYWORDS"
test ! -e "$TOP7_OUTPUT"

python scripts/replay_external_keyword_top7.py --source-run "$EXTERNAL_OUTPUT" --vocab "$VOCAB" --keyword-bias "$KEYWORDS" --keyword-set hard_k266 --output-dir "$TOP7_OUTPUT"
~~~

此命令不需要CUDA_VISIBLE_DEVICES，不读音频文件，不会修改原D5目录。完成后校验并紧凑
回传：

~~~bash
(cd "$TOP7_OUTPUT" && sha256sum -c sha256.txt)
wc -l "$TOP7_OUTPUT/top7_replay_details.jsonl"
jq 'keys | length' "$TOP7_OUTPUT/mls_retrieval_output.json"
jq 'keys | length' "$TOP7_OUTPUT/delivery_retrieval_output.json"
cat "$TOP7_OUTPUT/topk_comparison.md"
~~~

请回传上述终端输出。需要传给Context Learning下游的文件只是
mls_retrieval_output.json和delivery_retrieval_output.json；不需要传逐条详情、
原始音频、checkpoint或sample shards。收到结果后，再将实测D5/D7表格记入HANDOFF的
独立结果提交。

本轮本地验证：新增4项重放测试、相关入口12项测试以及全仓
230 passed / 23 skipped；新增文件Ruff、format、strict Mypy（skip imports）、
compileall、CLI help和git diff --check均通过。全仓Ruff仍只有既有
scripts/scan_g2p_coverage.py的3个长行以及用户未跟踪PPT临时目录的2个长行；
全包Mypy仍只有3处既有unused-ignore（ctc_overfit.py、sharded_ctc.py和
unfrozen_encoder_ctc.py）。本轮未修改这些无关文件。

## 0.77 2026-09-09 外部测试集WAV/FLAC混合格式修正

0.76首次CPU预检确认实际格式与先前口头信息不同：MLS来源是FLAC，Delivery
`wav-total`来源是WAV。旧入口只发现`.flac`，因此在创建输出目录前正确中止并报告
`source contains no FLAC files`；没有生成或覆盖任何产物。

修正后每个来源递归发现`.flac`和`.wav`（大小写不敏感），仍按文件stem与转写严格
一一匹配。`dataset_audit.json`为每个来源新增`audio_format_counts`，应分别显示MLS只有
FLAC、Delivery只有WAV；混合格式也会被如实记录。两种格式统一通过SoundFile读取，只有
运行时在内存中转mono/16 kHz/float32，绝不改写源文件。运行、门控、Anchor、输出、resume
及指标口径均不变。

拉取本节提交并核对最终SHA后，原样重跑0.76的`--audit-only`命令。由于上次错误发生在
`_prepare_output`之前，`$EXTERNAL_OUTPUT`应仍不存在，不需要也不得删除目录。预检输出中
重点确认：

```text
mls_portuguese.audio_format_counts:       {flac: >0, wav: 0}
delivery_20260706_ptbr.audio_format_counts:{flac: 0, wav: >0}
```

## 0.76 2026-09-08 葡语外部WAV/FLAC测试集266词完整音频Anchor检索入口

本轮为后续Context Learning新增一条完全隔离的外部测试入口，不修改既有4k容量、
三语训练、formal100 streaming或Prompt/Qwen解码逻辑。用户已确认本轮按“完整音频离线
检索”执行：每个WAV/FLAC文件整体经过Qwen3-ASR冻结audio encoder和三语Temporal 2x CTC
Head，再用Anchor索引对266个MFA热词做Top-64 shortlist、音素重排和固定门控，最终为
每个音频文件输出0至5个`word/phoneme`对象。本轮不运行Qwen LLM decoder；因此报告中的
`final_retrieval_recall`严格指audio到最终检索列表的召回率，不冒充最终ASR文本Recall。

新增文件：

- `src/qwen_hotword/inference/external_keyword_retrieval.py`
  - 读取多个`NAME=AUDIO_DIR,TRANSCRIPTS`来源，递归发现WAV/FLAC并按文件stem匹配转写。
  - 严格拒绝缺失转写、无音频转写、源内重复stem和跨来源重复stem，避免下游JSON覆盖。
  - 审计`pt_keyword_bias_phoneme.json`指定set中的全部词面和MFA音素，并要求100%映射到
    当前90类CTC词表。
  - 只使用转写做检索后的Unicode/case归一化完整连续短语真值匹配；转写不进入候选生成、
    排序或门控。
  - 原生读取WAV/FLAC，转为mono/16 kHz/float32；源音频不做离线格式转换。
  - 每条样本原子写入独立shard，`--resume`逐条复用；配置或输入SHA变化时拒绝resume。
  - 保留原始Top-20、greedy CTC音素、分数/edit ratio/posterior和门控结果，后续可先做
    case分析而不重跑Encoder。
  - 输出下游精简JSON、逐条详情、失败case、分来源/总计Recall与Precision、分阶段时延和
    SHA256清单。
- `scripts/run_external_keyword_retrieval.py`：CPU预检及H200完整运行CLI。
- `tests/test_external_keyword_retrieval.py`：词表、OOV、WAV/FLAC与transcript关联、ID冲突、
  下游schema和指标口径测试。

固定检索参数：

```text
all active hotwords:             hard_k266中的全部266词
retrieval backend:               anchor_guided
Anchor ngrams / per entry:       2,3,4 / 24
offset tolerance / start radius: 1 / 2
shortlist:                       64
threshold / Top-K:               0.75 / 5
posterior weight / minimum:      0.25 / 0.5
maximum edit ratio / margin:     0.35 / 0
saved raw rank depth:            20
```

词表本地静态审计确认附件中的`hard_k266`共有266个不同词面，266/266有MFA音素，
266/266可映射到当前90类CTC词表，0个OOV。`lua`和`elan`分别只有3个音素；为兑现“全部
266词参与检索”，本入口显式使用`minimum_phonemes=1`，不会沿用旧评测默认4而静默跳过
这两个词。该选择写入`run_config.json`和最终摘要。

### 容器可写挂载

`/home_91`当前未挂载到运行容器，不能给已启动容器动态增加bind mount。先在宿主机确认
现有容器启动参数和mount；重建容器时保留原GPU、共享内存、Conda和仓库mount参数，仅新增：

```bash
--mount type=bind,src=/home_91,dst=/home_91
```

进入新容器后先确认两个目录可见且保留宿主权限；本评测流程本身仍不改写源数据：

```bash
test -d /home_91/z00816262/data/2026data/27A/data/MLS_MultiLingual_LibriSpeech/mls_portuguese/test/audio
test -f /home_91/z00816262/data/2026data/27A/data/MLS_MultiLingual_LibriSpeech/mls_portuguese/test/transcripts.txt
test -d /home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/wav-total
test -f /home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/transcripts.txt
findmnt -T /home_91 -o TARGET,SOURCE,OPTIONS
```

### 拉取与CPU预检

先拉取交付分支；最终提交SHA以本轮Git交付消息为准，工作区必须用`git rev-parse HEAD`
逐字核对后再运行：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

MODEL=/glusterfs_103/models/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
KEYWORDS=pt_keyword_bias_phoneme.json
EXTERNAL_OUTPUT=outputs/pt_external_keyword_retrieval_mls_delivery_v1

MLS_AUDIO=/home_91/z00816262/data/2026data/27A/data/MLS_MultiLingual_LibriSpeech/mls_portuguese/test/audio
MLS_TEXT=/home_91/z00816262/data/2026data/27A/data/MLS_MultiLingual_LibriSpeech/mls_portuguese/test/transcripts.txt
DELIVERY_AUDIO=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/wav-total
DELIVERY_TEXT=/home_91/z00816262/data/2026data/27A/data/Delivery_20260706/Delivery_20260706_PT-BR/transcripts.txt

test -f "$MODEL/config.json"
test -f "$CTC_CHECKPOINT"
test -f "$VOCAB"
test -f "$KEYWORDS"
test ! -e "$EXTERNAL_OUTPUT"

python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$KEYWORDS" \
  --keyword-set hard_k266 \
  --source "mls_portuguese=$MLS_AUDIO,$MLS_TEXT" \
  --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" \
  --output-dir "$EXTERNAL_OUTPUT" \
  --audit-only

(cd "$EXTERNAL_OUTPUT" && sha256sum -c sha256.txt)
jq '{status, sample_count, sources, external_test_set_used,
  transcripts_used_for_candidate_generation}' "$EXTERNAL_OUTPUT/dataset_audit.json"
jq '{status, keyword_set, keyword_count, vocabulary_size, oov_keyword_count,
  keywords_below_four_phonemes}' "$EXTERNAL_OUTPUT/keyword_audit.json"
```

预检必须显示两个来源音频/转写逐条匹配、跨来源ID重复为0、`keyword_count=266`和
`oov_keyword_count=0`。任一不满足时停止，把CLI错误及两个转写文件各前3行的脱敏格式
发回；不要手工改ID、转写或源文件，也不要删除输出目录。

### H200完整检索与续跑

预检通过后在同一目录加`--resume`运行。下面只加载Transformers/Qwen audio encoder和
CTC Head，不初始化vLLM、不执行Qwen文本生成：

```bash
GPU_ID=3
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_external_keyword_retrieval.py \
  --model "$MODEL" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --vocab "$VOCAB" \
  --keyword-bias "$KEYWORDS" \
  --keyword-set hard_k266 \
  --source "mls_portuguese=$MLS_AUDIO,$MLS_TEXT" \
  --source "delivery_20260706_ptbr=$DELIVERY_AUDIO,$DELIVERY_TEXT" \
  --output-dir "$EXTERNAL_OUTPUT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume
```

中断时原样重跑最后一条命令；不要删除`sample_shards`，也不要使用新输出目录绕过身份
校验。完成后`retrieval_output.json`为后续要求的精简格式，所有音频stem均作为key，
未召回则值为空数组。验证与紧凑回传：

```bash
(cd "$EXTERNAL_OUTPUT" && sha256sum -c sha256.txt)

jq '{status, evaluation_scope, gate, retrieval_backend, overall, by_source}' \
  "$EXTERNAL_OUTPUT/evaluation_summary.json"

jq 'to_entries[:3]' "$EXTERNAL_OUTPUT/retrieval_output.json"
wc -l \
  "$EXTERNAL_OUTPUT/dataset_manifest.jsonl" \
  "$EXTERNAL_OUTPUT/retrieval_details.jsonl" \
  "$EXTERNAL_OUTPUT/failure_cases.jsonl"
```

本轮需要回传上述终端输出即可；若文件传输恢复，再附
`evaluation_summary.json`、`dataset_audit.json`、`keyword_audit.json`、`sha256.txt`
四个小文件。不要回传音频、checkpoint、完整sample shards或完整Top-20详情。

结果解读固定报告：各来源和总计`raw_recall_at_5`、`final_retrieval_recall`、
`final_retrieval_precision`、纯负样本FPR；以及纯检索、Encoder+Head检索、音频到结果的
P50/P95/P99/max和整体RTF。纯检索口径仍是greedy decode + Anchor + shortlist rerank/gate，
不含Processor/Encoder/Head，可与旧50 ms工程指标并列，但本次是整条音频而非2秒streaming
step，不作直接因果回归。

本轮本地验证：附件266词真实静态审计通过；新增8项测试和全仓249项pytest通过；新增
3个文件Ruff、format、compileall、CLI help、核心模块strict Mypy（skip第三方imports）及
`git diff --check`通过。全仓Ruff仍只有`scan_g2p_coverage.py`的3个既有长行和用户未跟踪
PPT临时目录的2个长行；按项目Python 3.10目标执行全包Mypy时，当前环境安装的NumPy stub
使用Python 3.12 `type`语句而在依赖解析阶段中止，这不是本轮模块类型错误。本机没有
`/home_91`数据或三语checkpoint，未加载Qwen模型、未生成真实检索结果，也未修改或纳入
用户PPT/图片文件。

## 0.75 2026-09-03 英西葡4k端到端分语种时延结果

0.74三语端到端运行自带的`streaming_{en,es,pt}/latency_summary.json`已完成
紧凑核验。这里的纯检索仍严格指每个streaming step中的CTC greedy decode +
Anchor query + Top-64 shortlist重排，不含CTC Processor、Encoder和Head；因此可与
既有`P95 < 50 ms`工程目标比较。

| 语种 | step数 | 纯检索P50 | P95 | P99 | max | 超50 ms | Detector P95 | 完整step P95 | D样本均值 | D RTF P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 334 | 14.21 ms | 22.71 ms | 27.77 ms | 32.62 ms | 0/334 = 0% | 90.25 ms | 140.56 ms | 294.50 ms | 0.046 / 0.102 |
| Spanish | 333 | 33.23 ms | 48.62 ms | 55.41 ms | 63.20 ms | 14/333 = 4.20% | 104.43 ms | 168.68 ms | 386.28 ms | 0.062 / 0.084 |
| Portuguese | 350 | 27.53 ms | 42.44 ms | 51.74 ms | 61.94 ms | 5/350 = 1.43% | 101.71 ms | 185.40 ms | 420.27 ms | 0.064 / 0.102 |

三个语种的纯检索P95均继续满足50 ms目标；英语余量很大，葡语仍有约7.6 ms
余量，西语仅有约1.4 ms余量。合计1,017个D step中有19个超过50 ms，约1.87%；
西语和葡语P99均超过50 ms。因此正式表述应是“逐语种P95验收通过，但西语尾部
接近边界”，不能表述为所有step都在50 ms内。

旧三语Head在旧葡语formal100 D5 conservative上的纯检索P95约37.32 ms；本轮
English 22.71 ms更快，Portuguese 42.44 ms略慢，Spanish 48.62 ms明显更接近边界。
由于语种、样本音频长度、step数和独立运行状态不同，这个横向差值只能作为工程观测，
不能声明checkpoint或语言本身造成时延因果变化。

Detector P95为90.25--104.43 ms，完整step P95为140.56--185.40 ms；它们分别加入
Encoder/Head以及Prompt/Qwen流式解码，不能混入50 ms纯检索口径。D组样本RTF P95
最高约0.102，三个语种均远低于1，整体仍满足实时处理。E比C更快仍不可解释为
Oracle Prompt加速，因为E绕过Detector且受同进程运行顺序和预热影响。

当前不需要因这组结果重跑端到端或立即改算法。后续若进入时延优化，应先只读分析
西语14个和葡语5个超50 ms step，按累计音频长度、Anchor postings、候选数以及
decode/Anchor/rerank分阶段归因；在完成归因前不调整50 ms验收定义。

## 0.74 2026-09-03 英西葡4k formal100三语端到端结果

0.69三语C/D5/E端到端流程在0.70--0.73入口修正后完成。最终轻量汇总为：

```text
outputs/en_es_pt_streaming_e2e_4k_formal100_v1/
  summary/multilingual_e2e_summary.json
```

汇总`status=pass`、`test_set_used=false`；英/西/葡各100条、合计300条，
C/D/E在各语种内使用相同样本，跨语种sample ID overlap为0。三语共用CTC
checkpoint SHA256：

```text
bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0
```

D固定为`threshold=0.86 / posterior=0 / Top-5`，其余门控参数为
`maximum_edit_ratio=0.35 / posterior_weight=0.25 / margin=0`。三语运行的
SHA256 manifest均已由汇总器验证。

三语micro热词计数和语言等权macro文本指标：

| 组 | Prompt Recall | 正确Prompt采用率 | 错误Prompt落地率 | 最终热词Recall | 最终热词Precision | 样本命中率macro | WER macro | CER macro | 纯负样本幻觉率macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C no-RAG | 0/516 = 0% | N/A | N/A | 480/516 = 93.02% | N/A | 86.67% | 5.47% | 2.06% | 0% |
| D D5 | 439/516 = 85.08% | 430/439 = 97.95% | 102/416 = 24.52% | 489/516 = 94.77% | 489/(489+102) = 82.74% | 90.00% | 5.04% | 1.92% | 0% |
| E Oracle | 516/516 = 100% | 497/516 = 96.32% | N/A | 497/516 = 96.32% | 100% | 92.92% | 4.80% | 1.87% | 0% |

D相对C增加9个最终热词命中，Recall绝对提升1.744个百分点；C到Oracle共有
17个可见增益，因此D恢复9/17约52.9%，距Oracle仍差8个命中、1.550个百分点。
D相对C的WER macro下降0.422个百分点，CER macro下降0.146个百分点。

分语种D5关键结果：

| 语种 | C/D/E最终命中 | D Prompt Recall | D正确Prompt采用率 | D错误Prompt落地率 | D最终Recall | D最终Precision | D WER | D CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 165/167/171 | 151/172 = 87.79% | 149/151 = 98.68% | 33/84 = 39.29% | 97.09% | 83.50% | 2.57% | 1.03% |
| Spanish | 165/170/171 | 156/172 = 90.70% | 156/156 = 100% | 33/180 = 18.33% | 98.84% | 83.74% | 2.65% | 0.78% |
| Portuguese | 150/152/155 | 132/172 = 76.74% | 125/132 = 94.70% | 36/152 = 23.68% | 88.37% | 80.85% | 9.92% | 3.94% |

结论：

- D5在三个语种都提高最终热词Recall，且纯负样本幻觉率保持0；
- Spanish最接近Oracle；Portuguese是当前三语短板，Prompt Recall、最终Recall、
  Precision和WER均最弱，且D相对C仅增加2个命中；
- 正确热词一旦进Prompt，三语micro采用率为97.95%，主要缺口首先是检索未注入
  `77/516`个期望热词；
- 同时416个错误注入中有102个写入最终文本。0%纯负样本幻觉与24.52%错误
  Prompt落地并不矛盾：后者主要发生在有正确热词的正样本，继续支持同族、嵌套、
  近邻冲突是Precision主因的判断；
- D与C的`mean_inference_seconds`差约68 ms/sample，但它是整条样本推理均值，
  不能替代或重新定义纯检索P95<50 ms口径。

本轮与旧葡语formal100使用不同的三语validation子集、单语资产和共享三语CTC
checkpoint，不能把葡语88.37%直接与旧葡语91.86%作同测试集回归结论。

下一步优先做只读case归因：按语种把D相对C新增命中、D仍缺失而E命中的8项、
77项Prompt未覆盖，以及102项错误落地按family/nested/near-neighbor分类。先从
Portuguese未覆盖和English高错误落地率入手；归因前不再扫描门控参数或重跑模型。

## 0.73 2026-09-03 三语streaming汇总字段兼容修正

英、西、葡三套C/D/E streaming运行完成后，最终轻量汇总在读取单语
`summary.json`时中止：

```text
summarize_multilingual_streaming_e2e.py: error:
quality field wrong_prompt_injected_hotwords is not an integer
```

单语结果没有损坏，也不需重新运行模型。真实streaming schema在顶层保留历史
字段`wrong_injected_hotwords`，并在`prompt_causal_metrics`内提供口径更明确的
`wrong_prompt_injected_hotwords`；三语汇总器的测试夹具却模拟了一个实际不存在
的同名顶层字段，因此实跑时读到`null`。

修正后的汇总器优先读取`prompt_causal_metrics.wrong_prompt_injected_hotwords`，
并兼容顶层`wrong_injected_hotwords`。测试夹具已改为真实单语schema。拉取本节
提交后，确认`$TRI_E2E_ROOT/summary`不存在，再只重跑0.69第七步汇总命令；
不得重跑英/西/葡streaming，也不得修改其SHA绑定产物。

## 0.72 2026-09-03 Anchor索引支持跨标签同音热词

0.71修正后，三语CTC步骤通过；最终streaming循环的英语、西语已完成，葡语
在模型加载后、构建4k Anchor索引时中止：

```text
run_streaming_rag_evaluation.py: error:
duplicate exact token sequence: sim_v3_hw_ptbr_0057
```

`pynvml`和`temperature`信息均为非阻塞警告。实际原因是葡语资产内不同语言
标签下的热词在90类CTC词表投影后具有相同token序列。registry按
`(language, token_ids)`允许这类同音词，但旧`IntegerAhoCorasick`和
`PhonemeAnchorIndex`又要求token序列全局唯一，两个组件契约不一致。

修正后，同一音素序列可关联多个不同hotword ID；Anchor query仍按case的
`active_hotword_ids`筛选，并把处于活动集合内的同音项分别作为候选返回。
重复hotword ID和空token序列仍会拒绝。本修正不改变CTC分数、门控参数、
已有英/西sample shard、字典、assets、capacity或offline产物。

拉取本节提交后，不删除任何输出，直接原样重跑0.69第六步完整
`for CODE in en es pt`循环并保留`--resume`：英语和西语会复用已完成shard，
葡语会从`streaming_pt`现有运行状态继续。若该目录只写入了同一配置的
`run_config.json`而没有sample shard，也属于安全resume状态。

## 0.71 2026-09-03 英语热词表Unicode规范化兼容修正

0.70修正后的三语词典导出和v3 assets均顺利完成，但第三步在循环的
第一个语种英语、加载热词表第12行时中止：

```text
V3 CTC EVALUATION FAILED: ValueError:
hotword row 12 phoneme tokens do not match token IDs
```

这不是GPU、feature cache、CTC Head或词典发音错误。共享v0.2词表的token 54
是NFC形式的`ç`，英语借词中该音素经`tokenize_ipa_to_vocab`后在资产中写成
语义完全等价的NFD `c + U+0327`。其token ID仍正确为54，但旧registry要求
`phoneme_tokens`必须和vocab token逐字节相同，因而误拒绝。

修正后：

- tokenizer对外返回`vocab.tokens[token_id]`的规范字符串，后续新资产不再写出
  NFD/NFC字符差异；
- registry允许Unicode规范等价的旧`phoneme_tokens`，但加载后立即换成
  vocab中的规范token；
- 任何真正音素或token ID不匹配仍会失败，没有放宽声学/检索契约。

已生成的`dictionary_en/es/pt`和`assets_en/es/pt`不需重建、不得删除或修改；
它们的SHA256保持不变。本次失败发生在`score_multi_nested_cases`创建
`ctc_en`输出之前。拉取修正提交后，直接原样重跑0.69第三步的
`for CODE in en es pt` CTC循环；不要回到第一/第二步。`pynvml` FutureWarning
仍为无关非阻塞警告。

## 0.70 2026-09-03 三语词典导出入口修正

0.69首次工区执行在英语第一行正确中止：

```text
missing word_pronunciations at .../full_ctc_train_en.jsonl:1
```

原因是早期formal split和Temporal 2×派生Manifest为训练效率只保留了整句
`phoneme_token_ids`，没有保留full-manifest中的逐词`word_pronunciations`。整句音素
没有词边界，不能安全反推单词发音。失败发生在创建`dictionary_en`之前，
所以不存在需要删除的半成品；用户已创建的空`TRI_E2E_ROOT`应保留。

修正后的词典导出器直接读取原始full-manifest `build_config.json`所绑定的
MFA词典：英语1份、西语3份、葡语1份。它会对输入词典重新计算SHA256，
跨西语来源同词多发音时按“在源词典中出现次数最多，平局用字典序”生成
唯一评测发音，同时把全部分歧写入`ambiguous_words.tsv`。这一步不运行MFA/
eSpeak G2P，不修改任何训练Manifest或旧词典，不读test。

工区拉取修正提交后，保留当前shell里已设置的变量和空总目录，直接
执行下面三条：

```bash
python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/en_us_swift_full_manifest_v1/build_config.json \
  --language en \
  --output-dir "$TRI_E2E_ROOT/dictionary_en"

python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/es_ar_full_manifests_v1/slr61/build_config.json \
  --build-config outputs/es_ar_full_manifests_v1/common_voice_rioplatense_v26/build_config.json \
  --build-config outputs/es_latam_cv_auxiliary_170h_full_manifest_v1/build_config.json \
  --language es \
  --output-dir "$TRI_E2E_ROOT/dictionary_es"

python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/noah_pt_full_500h/build_config.json \
  --language pt \
  --output-dir "$TRI_E2E_ROOT/dictionary_pt"

for CODE in en es pt; do
  (cd "$TRI_E2E_ROOT/dictionary_$CODE" && sha256sum -c sha256.txt) || break
done

jq -s 'map({status, language, source_dictionary_count,
  source_dictionary_pronunciations, unique_words, ambiguous_words,
  dictionary_sha256, test_set_used})' \
  "$TRI_E2E_ROOT"/dictionary_*/summary.json
```

三组都是`status=pass`、`test_set_used=false`且SHA通过后，不用先回传，直接
继续0.69第二步起的assets/CTC/offline/4k/streaming/summary流程。任一
`build_config.json`缺失、内部`dictionary.path`不存在或导出失败时停止，
只回传报错和该config的`jq '{dictionary}'`；不要手工猜词典或删目录。

## 0.69 2026-09-03 英西葡4k formal100三语端到端C/D5/E入口

0.68已将葡语门控收口为D5 `threshold=0.86 / posterior=0 / Top-5`；
两个D7均不保留。本轮不再扫描葡语threshold、posterior或Top-K，不训练，
不读sealed test，不改检索/提示词算法。目标是用三语平衡validation给英、西、
葡各构造一套独立formal100/4k资产，在完全相同的三语CTC Head和D5参数下
只跑：

```text
C: streaming no-RAG
D: streaming 4k Anchor RAG, 0.86 / posterior 0 / Top-5
E: streaming Oracle prompt
```

每语种固定100条（formal100的肯定/否定样本选择规则不变），最终同时保留
分语种指标和三语汇总。热词数、Prompt Recall和最终Recall/Precision用三语
micro计数；WER/CER用三个语种等权macro，避免文本长度不同让某一语种主导。

新增实现：

- `scripts/export_manifest_mfa_dictionary.py`从已有train/validation Manifest的
  `word_pronunciations` 导出MFA格式词典；它不运行G2P，不读test，并记录
  跨来源发音分歧。同词多发音时只为评测资产确定性选择Manifest中频次最高者，
  并用字典序打破平局；不回写训练Manifest。
- `build_hotword_capacity_assets` 现显式支持English/Spanish/Portuguese，
  保持原葡语默认调用兼容。
- `scripts/evaluate_multi_nested_hotwords.py` 增加可选
  `--cache-source-manifest`，允许从由三语合并Manifest建立的validation cache中评分
  单语种子集；单语种Manifest仍作为case/report身份。
- `scripts/summarize_multilingual_streaming_e2e.py` 会校验三个子运行的
  `sha256.txt`、样本一致性、C/D/E组、D5参数和共享checkpoint SHA，然后输出
  `multilingual_e2e_summary.json`。

工区先拉取并确认最终提交SHA：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

以下目录全部是新产物，不覆盖任何旧outputs。首次开始时执行初始检查；
中断后不要重新执行`test ! -e "$TRI_E2E_ROOT"`，只从尚未生成的子阶段继续。
已完成的普通子阶段不重跑；最后streaming阶段使用`--resume`粒度续跑，
不删除sample shards。

```bash
GPU_ID=3
MODEL=/glusterfs_103/models/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt
TRI_E2E_ROOT=outputs/en_es_pt_streaming_e2e_4k_formal100_v1

test -f "$MODEL/config.json"
test -f "$CTC_CHECKPOINT"
test -f "$BALANCED_TRAIN_ROOT/full_ctc_train_en.jsonl"
test -f "$BALANCED_TRAIN_ROOT/full_ctc_train_es.jsonl"
test -f "$BALANCED_TRAIN_ROOT/full_ctc_train_pt.jsonl"
test -f "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl"
test -f "$BALANCED_VALIDATION_ROOT/full_ctc_validation_en.jsonl"
test -f "$BALANCED_VALIDATION_ROOT/full_ctc_validation_es.jsonl"
test -f "$BALANCED_VALIDATION_ROOT/full_ctc_validation_pt.jsonl"
test -f "$FEATURE_CACHE_ROOT/validation/cache_summary.json"
test ! -e "$TRI_E2E_ROOT"
mkdir -p "$TRI_E2E_ROOT"

declare -A LANGUAGE=( [en]=English [es]=Spanish [pt]=Portuguese )
declare -A TRAIN=(
  [en]="$BALANCED_TRAIN_ROOT/full_ctc_train_en.jsonl"
  [es]="$BALANCED_TRAIN_ROOT/full_ctc_train_es.jsonl"
  [pt]="$BALANCED_TRAIN_ROOT/full_ctc_train_pt.jsonl"
)
declare -A VALIDATION=(
  [en]="$BALANCED_VALIDATION_ROOT/full_ctc_validation_en.jsonl"
  [es]="$BALANCED_VALIDATION_ROOT/full_ctc_validation_es.jsonl"
  [pt]="$BALANCED_VALIDATION_ROOT/full_ctc_validation_pt.jsonl"
)
declare -A PROMPT=(
  [en]='The following words may appear in the audio and are spelling references only. Use them only if they are actually spoken; do not force them into the transcription: {hotwords}'
  [es]='Las siguientes palabras pueden aparecer en el audio y solo sirven como referencia ortográfica. Úsalas únicamente si realmente se pronuncian; no las incluyas a la fuerza en la transcripción: {hotwords}'
  [pt]='As palavras a seguir podem aparecer no áudio e servem apenas como referência de grafia. Use-as somente se forem realmente faladas; não as inclua à força na transcrição: {hotwords}'
)
```

第一步只从原始full-manifest绑定的MFA词典导出三份评测词典，
无需GPU：

```bash
python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/en_us_swift_full_manifest_v1/build_config.json \
  --language en \
  --output-dir "$TRI_E2E_ROOT/dictionary_en"

python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/es_ar_full_manifests_v1/slr61/build_config.json \
  --build-config outputs/es_ar_full_manifests_v1/common_voice_rioplatense_v26/build_config.json \
  --build-config outputs/es_latam_cv_auxiliary_170h_full_manifest_v1/build_config.json \
  --language es \
  --output-dir "$TRI_E2E_ROOT/dictionary_es"

python scripts/export_manifest_mfa_dictionary.py \
  --build-config outputs/noah_pt_full_500h/build_config.json \
  --language pt \
  --output-dir "$TRI_E2E_ROOT/dictionary_pt"

for CODE in en es pt; do
  (cd "$TRI_E2E_ROOT/dictionary_$CODE" && sha256sum -c sha256.txt) || break
done
```

第二步构建三套单语种v3自然validation资产，无需GPU。任一语种
`status != pass`或nested critical group少于10时停止，不降低formal口径：

```bash
for CODE in en es pt; do
  python scripts/build_multi_nested_hotword_eval.py \
    --validation-manifest "${VALIDATION[$CODE]}" \
    --dictionary "$TRI_E2E_ROOT/dictionary_$CODE/manifest_mfa_dictionary.dict" \
    --vocab "$VOCAB" \
    --output-dir "$TRI_E2E_ROOT/assets_$CODE" \
    --seed 20260804 || break
done

jq -s 'map({status, validation_records, candidate_hotwords, selected_hotwords,
  nested_families, total_cases, actual_case_counts, conclusion_scope})' \
  "$TRI_E2E_ROOT"/assets_*/asset_summary_v3.json
```

第三步在现有三语validation feature cache上只跑共享CTC Head，不加载完整
Qwen；三个单语种Manifest只是合并cache的子集：

```bash
for CODE in en es pt; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_multi_nested_hotwords.py \
    --validation-cache "$FEATURE_CACHE_ROOT/validation" \
    --cache-source-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
    --validation-manifest "${VALIDATION[$CODE]}" \
    --dictionary "$TRI_E2E_ROOT/dictionary_$CODE/manifest_mfa_dictionary.dict" \
    --vocab "$VOCAB" \
    --checkpoint "$CTC_CHECKPOINT" \
    --hotwords "$TRI_E2E_ROOT/assets_$CODE/multi_nested_hotwords_v3.jsonl" \
    --families "$TRI_E2E_ROOT/assets_$CODE/hotword_families_v3.jsonl" \
    --cases "$TRI_E2E_ROOT/assets_$CODE/multi_nested_cases_v3.jsonl" \
    --asset-summary "$TRI_E2E_ROOT/assets_$CODE/asset_summary_v3.json" \
    --output-dir "$TRI_E2E_ROOT/ctc_$CODE" \
    --device cuda:0 \
    --batch-size 128 || break

  python scripts/rebuild_multi_nested_hotword_report.py \
    --vocab "$VOCAB" \
    --hotwords "$TRI_E2E_ROOT/assets_$CODE/multi_nested_hotwords_v3.jsonl" \
    --families "$TRI_E2E_ROOT/assets_$CODE/hotword_families_v3.jsonl" \
    --cases "$TRI_E2E_ROOT/assets_$CODE/multi_nested_cases_v3.jsonl" \
    --case-scores "$TRI_E2E_ROOT/ctc_$CODE/hotword_case_scores_v3.jsonl" \
    --base-report "$TRI_E2E_ROOT/ctc_$CODE/multi_nested_evaluation_report_v3.json" \
    --output "$TRI_E2E_ROOT/ctc_$CODE/multi_nested_evaluation_report_v3_corrected.json" || break

  (cd "$TRI_E2E_ROOT/ctc_$CODE" && sha256sum \
    hotword_case_scores_v3.jsonl \
    multi_nested_evaluation_report_v3.json \
    multi_nested_evaluation_report_v3_corrected.json > sha256.txt) || break
done
```

第四步为每语种只生成同一formal100选择及A/B/E离线控制报告，后面的
streaming只借用它的样本选择和Oracle定义。这一步会按语种顺序各加载一次
Qwen3-ASR-1.7B：

```bash
for CODE in en es pt; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_multi_nested_prompt_eval.py \
    --model "$MODEL" \
    --validation-manifest "${VALIDATION[$CODE]}" \
    --vocab "$VOCAB" \
    --hotwords "$TRI_E2E_ROOT/assets_$CODE/multi_nested_hotwords_v3.jsonl" \
    --families "$TRI_E2E_ROOT/assets_$CODE/hotword_families_v3.jsonl" \
    --cases "$TRI_E2E_ROOT/assets_$CODE/multi_nested_cases_v3.jsonl" \
    --ctc-case-scores "$TRI_E2E_ROOT/ctc_$CODE/hotword_case_scores_v3.jsonl" \
    --ctc-report "$TRI_E2E_ROOT/ctc_$CODE/multi_nested_evaluation_report_v3_corrected.json" \
    --output-dir "$TRI_E2E_ROOT/offline_$CODE" \
    --selection-profile formal100 \
    --retrieval-mode operating \
    --language "${LANGUAGE[$CODE]}" \
    --prompt-template "${PROMPT[$CODE]}" \
    --dtype bfloat16 \
    --device cuda:0 || break
done
```

第五步只构建100/4k两档单语种容量资产，不再生成500/1k/2k/5k/10k，
无需GPU：

```bash
for CODE in en es pt; do
  python scripts/build_hotword_capacity_assets.py \
    --training-manifest "${TRAIN[$CODE]}" \
    --dictionary "$TRI_E2E_ROOT/dictionary_$CODE/manifest_mfa_dictionary.dict" \
    --language "${LANGUAGE[$CODE]}" \
    --vocab "$VOCAB" \
    --base-hotwords "$TRI_E2E_ROOT/assets_$CODE/multi_nested_hotwords_v3.jsonl" \
    --base-cases "$TRI_E2E_ROOT/assets_$CODE/multi_nested_cases_v3.jsonl" \
    --selection "$TRI_E2E_ROOT/offline_$CODE/sample_selection.json" \
    --sizes 100,4000 \
    --output-dir "$TRI_E2E_ROOT/capacity_$CODE" || break
done
```

第六步是最终三语端到端，只跑C/D/E。每语种是一个100条×3组的独立
streaming运行，`--resume`只复用配置完全相同的sample shard。若显存不足，
先释放同卡其他进程或换`GPU_ID`，不改`gpu-memory-utilization`后强行复用原目录：

```bash
for CODE in en es pt; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_streaming_rag_evaluation.py \
    --model "$MODEL" \
    --validation-manifest "${VALIDATION[$CODE]}" \
    --vocab "$VOCAB" \
    --hotwords "$TRI_E2E_ROOT/capacity_$CODE/representative/size_4000/hotwords.jsonl" \
    --cases "$TRI_E2E_ROOT/capacity_$CODE/representative/size_4000/cases.jsonl" \
    --hotword-families "$TRI_E2E_ROOT/assets_$CODE/hotword_families_v3.jsonl" \
    --ctc-report "$TRI_E2E_ROOT/ctc_$CODE/multi_nested_evaluation_report_v3_corrected.json" \
    --offline-rag-dir "$TRI_E2E_ROOT/offline_$CODE" \
    --offline-format multi_nested_v3 \
    --offline-control-mode selection_only \
    --ctc-checkpoint "$CTC_CHECKPOINT" \
    --output-dir "$TRI_E2E_ROOT/streaming_$CODE" \
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
    --language "${LANGUAGE[$CODE]}" \
    --prompt-template "${PROMPT[$CODE]}" \
    --dtype bfloat16 \
    --device cuda:0 \
    --gpu-memory-utilization 0.18 \
    --resume || break
done
```

全部三语streaming通过后做CPU汇总：

```bash
python scripts/summarize_multilingual_streaming_e2e.py \
  --language-run en="$TRI_E2E_ROOT/streaming_en" \
  --language-run es="$TRI_E2E_ROOT/streaming_es" \
  --language-run pt="$TRI_E2E_ROOT/streaming_pt" \
  --output-dir "$TRI_E2E_ROOT/summary"
```

不需要打包或上传任何文件。只需运行下列校验和一个精简查询，将终端
输出粘贴回来：

```bash
for CODE in en es pt; do
  (cd "$TRI_E2E_ROOT/dictionary_$CODE" && sha256sum -c sha256.txt) || break
  (cd "$TRI_E2E_ROOT/ctc_$CODE" && sha256sum -c sha256.txt) || break
  (cd "$TRI_E2E_ROOT/offline_$CODE" && sha256sum -c sha256.txt) || break
  (cd "$TRI_E2E_ROOT/capacity_$CODE" && sha256sum -c sha256.txt) || break
  (cd "$TRI_E2E_ROOT/streaming_$CODE" && sha256sum -c sha256.txt) || break
done
(cd "$TRI_E2E_ROOT/summary" && sha256sum -c sha256.txt)

jq '{
  status,
  ctc_checkpoint_sha256,
  total_sample_count,
  identity_checks,
  per_language: (.per_language | with_entries(.value |= {
    sample_count,
    C: (.groups.C | {expected_hotwords, final_hotword_recall, wer, cer}),
    D: (.groups.D | {
      expected_hotwords,
      correct_prompt_injected_hotwords,
      prompt_hotword_recall,
      correct_prompt_adopted_hotwords,
      correct_prompt_adoption_rate,
      wrong_prompt_injected_hotwords,
      wrong_prompt_written_hotwords,
      wrong_prompt_landing_rate,
      final_hotword_recall,
      final_hotword_precision,
      negative_hotword_hallucination_rate,
      wer,
      cer
    }),
    E: (.groups.E | {expected_hotwords, final_hotword_recall,
      final_hotword_precision, wer, cer}),
    d_minus_c,
    e_minus_d
  })),
  aggregate
}' "$TRI_E2E_ROOT/summary/multilingual_e2e_summary.json"

jq -s 'map({language, source_dictionary_count, unique_words,
  ambiguous_words, dictionary_sha256})' \
  "$TRI_E2E_ROOT"/dictionary_*/summary.json
```

返回后只分析三类核心问题：各语种D5的Prompt Recall和最终Recall/Precision；
D相对C是否改善热词且未显著伤害WER/CER；E与D的差距是检索瓶颈还是
Qwen采用瓶颈。不追加D7、threshold sweep、容量阶梯或新训练试验。

本地没有加载Qwen/CTC模型，没有生成真实数据或评测outputs，没有修改旧算法、
训练代码或用户PPT文件。

## 0.68 2026-09-03 三语CTC Head葡语4k D-only标定候选端到端结果

工区已完成0.67的两个D7候选，`status=pass`、100条formal样本，全部内部身份
检查通过：checkpoint与既有D5相同，SHA256为
`bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0`；完整排名CTC
report绑定该checkpoint；标定、offline selection、非gate子运行配置和D样本均一致。
本次回传了汇总中记录的artifact SHA，但没有单独粘贴3组`sha256sum -c`的终端行；
不因此阻塞后续三语工程验证，但outputs仍只能视为工区产物。

| 葡语formal100配置 | 正确Prompt注入/总词 | Prompt Recall | 正确Prompt采用 | 错Prompt落字 | 最终Recall | 最终Precision | WER / CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D5 `0.86/p0/Top-5` | 138/172 | 80.23% | 130/138 = 94.20% | 33 | 158/172 = 91.86% | 82.72% | 8.032% / 3.377% |
| D7 `0.86/p0/Top-7` | 138/172 | 80.23% | 130/138 = 94.20% | 33 | 158/172 = 91.86% | 82.72% | 8.032% / 3.377% |
| D7 `0.83/p0/Top-7` | 145/172 | 84.30% | 135/145 = 93.10% | 33 | 156/172 = 90.70% | 82.54% | 8.233% / 3.517% |

`0.86/Top-7`相对D5只新增2个错误注入，正确注入、采用、最终文本、WER/CER
全部不变，所以纯CTC的+1真词并没有在formal100端到端中产生可见价值。

`0.83/Top-7`相对D5多注入7个正确词，Prompt Recall提高4.07个百分点，但也多注入
36个错误词。错词落字绝对数仍为33，所谓landing rate从27.5%降到21.15%只是因为
分母变大，不是错词更少。Qwen对正确Prompt只多采用5个，最终正确热词反而少2个，
Recall下降1.16个百分点，Precision下降0.18个百分点，WER/CER分别变差0.20/0.14
个百分点。负样本热词幻觉率三组均为0。

因此葡语门控标定正式收口：当前三语Head在葡语4k上保留D5
`0.86 / posterior 0 / Top-5`；不选任何一个本轮D7。不再做葡语threshold/Top-K
扫描或重跑C/E。下一个主任务是为英语和西语构建与葡语formal100同口径的4k端到端
资产，然后在英/西/葡三语分别比较C no-RAG、D5和E Oracle，不从葡语结果外推英西语。

## 0.67 2026-09-03 三语CTC Head葡语4k formal100标定候选D-only端到端入口

0.66的完整排名标定已经把下一步缩小为两个精确D7候选。本轮新增独立套件，
不修改0.57/0.61的六组formal100套件，不重跑C no-RAG或E Oracle，不改算法、
checkpoint、样本或既有outputs：

- D5基线：直接读取既有三语Head套件的`conservative`，`0.86 / posterior 0 / Top-5`；
- Top-K隔离候选：`0.86 / posterior 0 / Top-7`，只改Top-K；
- Recall候选：`0.83 / posterior 0 / Top-7`，同时改threshold和Top-K。

新脚本`scripts/run_streaming_calibrated_gate_suite.py`在加载模型前必须通过以下预检：
既有D5 suite及子目录SHA256有效，标定必须是完整排名、0个非精确点且两个
recommended candidate与上述参数精确一致，标定、当前完整排名CTC报告、新运行
和既有D5必须绑定同一个三语checkpoint。运行后还会验证子运行除gate、Git和GPU
分配外的配置相同，共用同一offline selection report和完全相同的formal100 D样本。
完整排名扩展会改变CTC report文件SHA，但不改checkpoint；套件单独记录新report SHA
和checkpoint绑定，不把这个已知的证据扩展误判为模型更换。

在工区拉取`codex/g2p-coverage-scan`的本轮提交并确认HEAD：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

使用新目录顺序跑两个D profile。工具一次只起一个子进程，前一个正常退出后才
起下一个；`--resume`只复用配置完全相同且已完成的profile/sample shard：

```bash
GPU_ID=3
MODEL=/glusterfs_103/models/Qwen3-ASR-1.7B
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
PT_ROOT=outputs/noah_pt_full_training_v1
CAP_ROOT="$PT_ROOT/hotword_capacity_eval_v1"
V3_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested"
V3_FULL_RANK_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_full_rank_v1"
V3_FULL_CALIBRATION_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_calibration_full_rank_v1"

ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE100_ROOT="$V3_ROOT/prompt_multi_nested_formal100_top5_v1"
BASELINE_SUITE_ROOT="$CAP_ROOT/streaming_gate_suite_4k_formal100_multilingual_ctc_v1"
CALIBRATED_SUITE_ROOT="$CAP_ROOT/streaming_calibrated_gate_suite_4k_formal100_multilingual_ctc_v1"
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt

test -f "$BASELINE_SUITE_ROOT/suite_summary.json"
test -f "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3_corrected.json"
test -f "$V3_FULL_CALIBRATION_ROOT/candidate_summary.json"
test -f "$CTC_CHECKPOINT"
test ! -e "$CALIBRATED_SUITE_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_streaming_calibrated_gate_suite.py \
  --model "$MODEL" \
  --validation-manifest "$PT_ROOT/full_ctc_validation.jsonl" \
  --vocab "$VOCAB" \
  --hotwords "$ASSET_4K_ROOT/representative/size_4000/hotwords.jsonl" \
  --cases "$ASSET_4K_ROOT/representative/size_4000/cases.jsonl" \
  --hotword-families "$V3_ROOT/hotword_families_v3.jsonl" \
  --ctc-report "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --offline-rag-dir "$OFFLINE100_ROOT" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --baseline-suite "$BASELINE_SUITE_ROOT" \
  --calibration-summary "$V3_FULL_CALIBRATION_ROOT/candidate_summary.json" \
  --output-dir "$CALIBRATED_SUITE_ROOT" \
  --language Portuguese \
  --gpu-memory-utilization 0.18 \
  --resume
```

若两个profile都未开始，首次也可保留`--resume`；新目录不存在时它不会改变运行
语义。若因显存不足失败，先释放同一张卡的其他进程后用原命令resume；不要改
`gpu-memory-utilization`后强行复用原目录，也不要删除已完成的sample shards。

运行完不需要打包或传文件。只运行以下命令并粘贴精简输出：

```bash
(cd "$CALIBRATED_SUITE_ROOT" && sha256sum -c sha256.txt)
(cd "$CALIBRATED_SUITE_ROOT/precision_guarded_top7" && sha256sum -c sha256.txt)
(cd "$CALIBRATED_SUITE_ROOT/f1_with_fpr_guard_top7" && sha256sum -c sha256.txt)

jq '{
  status,
  sample_count,
  identity_checks,
  baseline: {
    gate: .baseline.gate,
    quality: (.baseline.quality | {
      expected_hotwords,
      correct_prompt_injected_hotwords,
      prompt_hotword_recall,
      correct_prompt_adopted_hotwords,
      correct_prompt_adoption_rate,
      wrong_prompt_written_hotwords,
      wrong_prompt_landing_rate,
      final_hotword_recall,
      final_hotword_precision,
      negative_hotword_hallucination_rate,
      wer,
      cer
    })
  },
  candidates: (.candidates | with_entries(.value |= {
    gate,
    quality: (.quality | {
      expected_hotwords,
      correct_prompt_injected_hotwords,
      prompt_hotword_recall,
      correct_prompt_adopted_hotwords,
      correct_prompt_adoption_rate,
      wrong_prompt_written_hotwords,
      wrong_prompt_landing_rate,
      final_hotword_recall,
      final_hotword_precision,
      negative_hotword_hallucination_rate,
      wer,
      cer
    }),
    delta_from_d5_baseline
  })),
  comparisons,
  case_comparison
}' "$CALIBRATED_SUITE_ROOT/calibrated_gate_summary.json"

wc -l "$CALIBRATED_SUITE_ROOT/calibrated_gate_cases.jsonl"
```

这次不设预定通过线。结果回来后只判断两件事：0.86/Top-7的纯CTC小幅改善
是否能转化为Prompt/最终文本改善且不引入Precision或幻觉回退；0.83/Top-7新增的22个
纯CTC真词有多少进入Prompt并被Qwen正确写出，以及为此付出多少错词落地和最终
Precision代价。根据这两个端到端结果再决定保留D5、换成哪个D7，或转入
family-aware冲突消解；不因纯CTC F1自动选择0.83。

本轮本地验证：新增预检、身份比较、指标汇总、case差异和resume共3项单测
通过，联合既有gate suite、checkpoint regression和v3标定共12项定向测试通过；
全量pytest收集231项并通过（平台可选项按既有标记skip）。本轮3个新文件Ruff、
format、compileall和新核心模块strict Mypy通过，`git diff --check`通过。全仓
Ruff仍只有3个旧G2P长行和2个用户未跟踪PPT临时脚本长行；`MYPYPATH=src mypy`
仍只有3个未修改训练模块的旧`unused-ignore`。本机未加载Qwen/CTC模型、未生成
真实评测产物，也未修改或纳入用户的PPT文件。

## 0.66 2026-09-03 葡语v3完整100名无截断标定结果

工作区已按0.65完成一次CTC-only完整排名导出和CPU标定。评分目录3个文件、标定目录5个
文件均通过`sha256sum -c`。本次回传未包含报告内的checkpoint SHA和各artifact SHA字符串，
但标定源点逐项复算与0.62三语Head结果完全一致；后续端到端入口仍必须由程序验证CTC报告
与checkpoint绑定，不能仅凭这一致性替代身份检查。

每条210个validation case均保存100个active hotword完整排名。Top-1/3/5/7、21个threshold
和4个posterior门槛共336点全部`replay_exact=true`，0个非精确点，Pareto前沿26点。完整
排名证明0.64的4个截断不确定点没有隐藏一个更好的D5配置，但Top-7出现一个严格改善点。

三语Head源点保持：`0.86 / posterior 0 / Top-5`，321/410真词、336个选择，Recall
78.2927%、Precision 95.5357%、F1 86.0590%、positive-case hit 92.7778%、负例FPR 0%。

precision-guarded推荐为`0.86 / posterior 0 / Top-7`：

```text
selected / true positive:      337 / 322
Recall / Precision / F1:       78.5366% / 95.5490% / 86.2115%
positive-case hit rate:        92.7778%
negative-case FPR:             0%

相对新Head Top-5源点:
  selected / true positive:    +1 / +1
  Recall:                      +0.2439 pp
  Precision:                   +0.0132 pp
  F1:                          +0.1525 pp
```

这不是大幅召回恢复，但它是在完整证据下仅改变Top-K并同时不损失Precision/FPR的严格改善，
可作为端到端安全对照。

FPR-guarded F1最佳点为`0.83 / posterior 0 / Top-7`：

```text
selected / true positive:      364 / 343
Recall / Precision / F1:       83.6585% / 94.2308% / 88.6305%
positive-case hit rate:        95.5556%
negative-case FPR:             0%

相对新Head 0.86/Top-5:
  selected / true positive:    +28 / +22
  Recall:                      +5.3659 pp
  Precision:                   -1.3049 pp
  F1:                          +2.5715 pp

相对旧葡语Head 0.86/Top-5:
  selected / true positive:    +9 / +2
  Recall:                      +0.4878 pp
  Precision:                   -1.8256 pp
  F1:                          -0.5198 pp
```

下一轮只跑这两个D7候选：0.86用于隔离Top-K的微小安全收益，0.83用于判断22个纯CTC新增
真词是否能转化为Prompt Recall和最终热词Recall。两者共享posterior weight 0.25、minimum
posterior 0、maximum edit ratio 0.35、margin 0和全部4k Anchor/流式/Qwen设置。不重跑C/E，
也不把不同H200进程的细小时延差解释为门控因果。

## 0.65 2026-09-03 葡语v3完整100名shortlist与无截断标定入口

0.64的截断Top-5标定已经证明：现有证据下没有同时保持新Head 95.54% Precision和0%
负例FPR的Recall增益；0.83点只能以约1.32个百分点Precision换取约5.12个百分点Recall。
为排除4个非精确点以及Top-5门控后补位不可见的问题，本轮只扩展v3评分产物的可审计
保存深度，不改变声学模型、CTC解码、热词打分、门控、指标或Prompt算法。

`scripts/evaluate_multi_nested_hotwords.py`新增`--saved-ranked-matches`，默认仍为5；只有本轮
新目录显式传100。每条case仍保留兼容字段`ranking_top5`，同时新增`ranked_matches`、
`ranked_matches_available`、`ranked_matches_complete`和`active_hotword_count`。评分器会要求
每条100词case实际产生100个候选，否则失败，不把不完整文件标成complete。报告新增保存
深度、complete标志和case-scores SHA256。旧Top-5 JSONL仍可原样读取；新加载器会验证完整
排名的前五项与`ranking_top5`严格一致、available数量一致以及complete/active数量一致。

0.63标定器同步改为动态识别保存深度。旧Top-5文件仍按逐点充分条件判断是否精确，完整
100名文件则所有门控点直接具备精确证据；Top-K上限由实际保存深度决定。本轮完整标定显式
加入Top-7，以对应formal100的D5/D7差异，但仍不预设必须达到的目标。

先拉取本轮提交并确认HEAD，然后在全新目录做一次CTC-only评分。只加载三语best CTC Head
和已有葡语validation feature cache，不加载Qwen3-ASR完整模型、不提取Encoder特征、不训练、
不读test，也不覆盖0.61的Top-5分数：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

GPU_ID=3
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
PT_ROOT=outputs/noah_pt_full_training_v1
PT_VALIDATION_CACHE="$PT_ROOT/features_ln_post_bf16/validation"
PT_VALIDATION_MANIFEST="$PT_ROOT/full_ctc_validation.jsonl"
PT_DICTIONARY=outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict

V3_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested"
V3_MULTILINGUAL_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_v1"
CTC_CHECKPOINT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1/ctc_head_best.pt

V3_FULL_RANK_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_full_rank_v1"
V3_FULL_CALIBRATION_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_calibration_full_rank_v1"

test -f "$CTC_CHECKPOINT"
test -f "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json"
test ! -e "$V3_FULL_RANK_ROOT"
test ! -e "$V3_FULL_CALIBRATION_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_multi_nested_hotwords.py \
  --validation-cache "$PT_VALIDATION_CACHE" \
  --validation-manifest "$PT_VALIDATION_MANIFEST" \
  --dictionary "$PT_DICTIONARY" \
  --vocab "$VOCAB" \
  --checkpoint "$CTC_CHECKPOINT" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --asset-summary "$V3_ROOT/asset_summary_v3.json" \
  --output-dir "$V3_FULL_RANK_ROOT" \
  --device cuda:0 \
  --batch-size 128 \
  --saved-ranked-matches 100

python scripts/rebuild_multi_nested_hotword_report.py \
  --vocab "$VOCAB" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --case-scores "$V3_FULL_RANK_ROOT/hotword_case_scores_v3.jsonl" \
  --base-report "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3.json" \
  --output "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3_corrected.json"

(cd "$V3_FULL_RANK_ROOT" && sha256sum \
  hotword_case_scores_v3.jsonl \
  multi_nested_evaluation_report_v3.json \
  multi_nested_evaluation_report_v3_corrected.json > sha256.txt)
```

随后CPU-only做完整排名标定。加入Top-7后为4 × 21 × 4 = 336点：

```bash
python scripts/calibrate_v3_operating_points.py \
  --vocab "$VOCAB" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --case-scores "$V3_FULL_RANK_ROOT/hotword_case_scores_v3.jsonl" \
  --candidate-report "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --reference-report "$V3_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --top-ks 1,3,5,7 \
  --output-dir "$V3_FULL_CALIBRATION_ROOT"
```

两个工具都拒绝覆盖已有结果且没有resume。若CTC评分中断，不要删除目录或猜测续跑；先把
错误和目录文件名回传。本轮无需发送任何文件，只运行以下精简命令并粘贴输出：

```bash
(cd "$V3_FULL_RANK_ROOT" && sha256sum -c sha256.txt)
(cd "$V3_FULL_CALIBRATION_ROOT" && sha256sum -c sha256.txt)

jq '{
  checkpoint_sha256,
  case_scores_sha256,
  scoring_config,
  overall: .metrics.overall.operating
}' "$V3_FULL_RANK_ROOT/multi_nested_evaluation_report_v3_corrected.json"

jq '{
  status,
  case_count,
  hotword_count,
  saved_rank_depth,
  ranked_matches_complete,
  sweep_point_count,
  exact_point_count,
  non_exact_point_count,
  pareto_point_count,
  guarded_recall_gain_candidate_available,
  candidate_baseline: .candidate_baseline.overall,
  reference_baseline: .reference_baseline.overall,
  recommended_candidates: [
    .recommended_candidates[] | {
      role,
      config,
      overall: .metrics.overall,
      delta_from_candidate_baseline,
      delta_from_reference_baseline
    }
  ],
  replay_limitation,
  next_action
}' "$V3_FULL_CALIBRATION_ROOT/candidate_summary.json"
```

预期身份是三语best checkpoint
`bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0`，评分配置必须显示
`saved_ranked_matches=100`和`ranked_matches_complete=true`；标定应有336/336精确点、0个
非精确点。收到精简输出后再决定是否选0.83、其他D5/D7点或保持0.86；仍不自动重跑C/E。

本轮本地验证：完整排名加载/兼容、序列化、Top-7标定和既有formal Prompt共14项定向测试
通过；全量pytest 228项通过；两个相关核心模块strict Mypy、相关文件Ruff与format、两个CLI
help和`git diff --check`通过。全仓既有3个G2P长行、2个用户未跟踪PPT临时脚本长行及6个
旧模块的11项Mypy问题未在本轮修改。没有在本机加载模型或伪造完整排名结果。

## 0.64 2026-09-03 三语CTC Head葡语v3截断Top-5标定结果

工作区已使用0.63提交的CPU-only标定器完成252个组合。输出目录五个文件均通过
`sha256sum -c sha256.txt`；本次回传的是终端摘要和校验结果，未传文件本体及每个
artifact的SHA字符串。运行确认210条v3 validation case、500个热词，未执行CTC推理、
Qwen推理或读取sealed test。

252个点中248个可以由保存的`ranking_top5`证明为精确重放，4个点因Top-5截断无法证明；
精确Pareto点18个。原三语Head源点再次复算一致：

```text
threshold / posterior / Top-K: 0.86 / 0 / 5
selected / true positive:      336 / 321
Recall / Precision / F1:       78.2927% / 95.5357% / 86.0590%
positive-case hit rate:        92.7778%
negative-case FPR:             0%
```

在“Precision不得低于新Head 0.86源点且负例FPR不得升高”的约束下，没有任何精确点提高
Recall；因此precision-guarded推荐仍是原0.86点，状态为
`exact_sweep_complete_no_guarded_recall_gain`。这说明单纯放宽threshold不能无损恢复新Head
相对旧葡语Head的约20个Operating真词缺口。

FPR保持0时的F1最佳点为`threshold=0.83 / posterior=0 / Top-5`：

```text
selected / true positive:      363 / 342
Recall / Precision / F1:       83.4146% / 94.2149% / 88.4864%
positive-case hit rate:        95.5556%
negative-case FPR:             0%

相对新Head 0.86:
  true positive:               +21
  Recall:                      +5.1220 pp
  Precision:                   -1.3208 pp
  F1:                          +2.4274 pp

相对旧葡语Head 0.86:
  true positive:               +1（342 vs 341）
  Recall:                      +0.2439 pp
  Precision:                   -1.8415 pp
  F1:                          -0.6639 pp
```

0.83点改善了各多词组Recall：nested-family-plus-two为85.0%、three-independent为80.0%、
two-independent为83.33%；nested short-only为95.0%，long-present long为81.67%，且
short-only误触发long仍为0%。但single-hotword Precision降到87.10%，是整体Precision代价
最明显的分组。该点适合作为有明确取舍的候选，不是“无损替代”结论。

本轮不直接重跑葡语formal100端到端。下一步先在新目录做一次CTC-only v3评分，把每条
case的完整100名排序保存下来，再用同一网格做完全无截断标定。若完整证据仍证明不存在
Precision/FPR受保护的Recall增益，则保留0.86作为保守点，并由用户明确决定是否愿意用
约1.32个百分点Precision换取0.83点的约5.12个百分点Recall；不应继续用阈值微调掩盖
三语Head的葡语校准差异。

## 0.63 2026-09-03 三语CTC Head葡语v3 Operating离线标定

0.62已经确认三语Head的葡语Forced Ranking Recall@5仍为389/410（94.88%），但沿用
旧葡语Head的`threshold=0.86 / posterior=0 / Top-5`时，Operating Recall只有
321/410（78.29%）。本轮只标定新Head的门控，不训练、不加载Qwen、不读取sealed test、
不改4k Anchor、三语训练、热词评分或Prompt算法，也不覆盖0.61/0.62已有产物。

新增独立CPU-only工具：

- `src/qwen_hotword/hotwords/v3_operating_calibration.py`
- `scripts/calibrate_v3_operating_points.py`
- `tests/test_v3_operating_calibration.py`

它读取新三语Head已经生成的葡语v3 `hotword_case_scores_v3.jsonl`，固定原排名的
`posterior_weight=0.25`、`maximum_edit_ratio=0.35`和`minimum_top1_margin=0`，只扫描
threshold、minimum posterior和Top-K。默认网格为Top-1/3/5、threshold
0.70至0.90（步长0.01）的21个点、posterior 0/0.25/0.5/0.75，共252点。没有预先规定必须
达到的质量阈值；输出Pareto前沿，并按“Precision不低于新Head 0.86基线且负例FPR不升高”
选择precision-guarded点，再补一个相同FPR约束下的F1最佳点，最多返回两个不同候选。

必须注意原v3保存契约的限制：`ranking_top5`只保留门控前五名，而原Operating实际从
100个激活热词完整排名中过门控后再截Top-5。因此Top-5文件不是天然完整的gate replay
shortlist，Top-7更完全没有证据。标定器会逐case证明每个点是否精确：只有已保存候选已
填满该点Top-K，或第五名分数已经低于新threshold，才能证明未保存名次不会改变选择。
原0.86源点直接用文件中保存的真实`operating_matches`复核。任何不能证明完整的点都会
标记`replay_exact=false`，只作诊断，不进入Pareto或候选推荐；请求Top-K大于5会直接失败。
如果现有Top-5没有可恢复Recall的精确点，结论应是先补一次CTC-only完整shortlist导出，
而不是拿截断估计去重跑完整端到端。

工作区拉取与运行命令如下。本次只读已有小文件并做CPU计算，不需要GPU，也不支持resume；
输出目录必须全新，存在时不得删除或覆盖：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
PT_ROOT=outputs/noah_pt_full_training_v1
V3_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested"
V3_MULTILINGUAL_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_v1"
V3_CALIBRATION_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_calibration_v1"

test -f "$V3_ROOT/multi_nested_hotwords_v3.jsonl"
test -f "$V3_ROOT/hotword_families_v3.jsonl"
test -f "$V3_ROOT/multi_nested_cases_v3.jsonl"
test -f "$V3_ROOT/multi_nested_evaluation_report_v3_corrected.json"
test -f "$V3_MULTILINGUAL_ROOT/hotword_case_scores_v3.jsonl"
test -f "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json"
test ! -e "$V3_CALIBRATION_ROOT"

python scripts/calibrate_v3_operating_points.py \
  --vocab "$VOCAB" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --case-scores "$V3_MULTILINGUAL_ROOT/hotword_case_scores_v3.jsonl" \
  --candidate-report "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --reference-report "$V3_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --output-dir "$V3_CALIBRATION_ROOT"
```

输入校验包括：candidate/reference均未读取test、checkpoint SHA格式有效、vocab/hotword/
families/cases SHA与实际文件一致，以及candidate报告的Operating计数和指标可由case scores
逐项复算。输出记录所有输入文件的SHA256、候选与参考checkpoint SHA、网格和“不执行CTC/
Qwen推理”标志。验收与回传：

```bash
(cd "$V3_CALIBRATION_ROOT" && sha256sum -c sha256.txt)

jq '{
  status,
  case_count,
  hotword_count,
  sweep_point_count,
  exact_point_count,
  non_exact_point_count,
  pareto_point_count,
  guarded_recall_gain_candidate_available,
  candidate_baseline,
  reference_baseline,
  recommended_candidates,
  replay_limitation,
  next_action
}' "$V3_CALIBRATION_ROOT/candidate_summary.json"

wc -l \
  "$V3_CALIBRATION_ROOT/operating_point_sweep.jsonl" \
  "$V3_CALIBRATION_ROOT/exact_pareto_frontier.jsonl"
```

请回传以下小文件，不需要重新打包checkpoint、feature cache、音频、旧v3目录或4k suite：

```text
simulated_hotword_eval_v3_multi_nested_multilingual_ctc_calibration_v1/
  calibration_config.json
  candidate_summary.json
  exact_pareto_frontier.jsonl
  operating_point_sweep.jsonl
  sha256.txt
```

收到后先核验SHA和checkpoint身份。如果存在precision/FPR受保护且Recall提高的精确候选，
只选1至2个进入下一轮葡语D组端到端；如果状态为
`exact_sweep_complete_no_guarded_recall_gain`或`full_shortlist_required`，先补完整shortlist
证据，不重跑C/E，也不立即重训三语Head。

本轮本地验证：新增标定器3项测试通过，连同v3评估和既有Anchor sweep共10项定向测试
通过；全量pytest 225项通过；新增模块strict Mypy、相关文件Ruff、CLI help、Ruff format和
`git diff --check`通过。全仓Ruff仍只有3个`scan_g2p_coverage.py`既有长行和2个未跟踪PPT
临时脚本长行；全包Mypy仍为6个未修改旧模块的11项既有错误。本机没有读取工作区v3产物，
没有生成真实候选，也没有加载CTC/Qwen模型。

## 0.62 2026-09-03 三语CTC Head葡语4k formal100替换回归结果

工作区已按0.61完成checkpoint审计、三语validation诊断、葡语v3 CTC评分、
4k formal100六组端到端回归和最后的CPU对比。对比器首次运行因将子目录
`README.md`错解为仓库根文件而产生假SHA mismatch；这是只读校验器路径解析问题，
不是工作区产物损坏。修复提交
`e1c658f5ff6de73da0ffba7b3b42db1e3c8dea79`后只重跑CPU汇总即通过，
没有重跑模型或覆盖旧suite。

三语最终best checkpoint为epoch 24，SHA256为
`bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0`，文件大小
3,360,421 bytes。训练从epoch 5 resume，请求35 epoch，在epoch 26早停；final比best的
Macro PER只高0.0083个百分点，可认为已进入平台。与0.47的5 epoch pilot相比：

| 指标 | 5 epoch pilot | 最终best epoch 24 | 改善 |
| --- | ---: | ---: | ---: |
| Mixed validation PER | 7.417% | 6.616% | -0.801 pp |
| Macro PER | 6.894% | 6.103% | -0.791 pp |
| English PER | 5.652% | 4.736% | -0.916 pp |
| Spanish PER | 4.275% | 3.910% | -0.365 pp |
| Portuguese PER | 10.755% | 9.662% | -1.093 pp |

最终诊断完整消费8,101条validation和370,931个参考音素，总错误24,539：
substitution/deletion/insertion分别11,013/9,500/4,026。葡语占参考音素约44.03%，
但产生15,780个错误，占总错误约64.31%，仍是明显最差组。葡语预测/参考长度比
0.9728，比0.46时的0.9412有改善，但删除仍有6,867个。CTC压力分桶PER为：
低压力5.647%、中压力11.068%、高压力24.473%（高压力仅15条）。

葡语v3纯CTC使用原自然validation的210条case、500词和0.86/Top-5固定口径，
新报告checkpoint SHA与三语best一致。与0.11旧葡语Head相比：

| 纯CTC v3指标 | 旧葡语Head | 新三语Head | 变化 |
| --- | ---: | ---: | ---: |
| Forced Ranking Recall@5 | 95.37% | 94.88% (389/410) | 约-0.49 pp |
| Operating Recall | 83.17% | 78.29% (321/410) | 约-4.88 pp |
| Operating Precision | 96.06% | 95.54% (321/336) | 约-0.52 pp |
| Positive-case hit rate | 未单独记录 | 92.78% | — |
| Negative-case FPR | 0% | 0% | 0 pp |
| All-3-Hit@5 | 80% | 80% | 0 pp |
| Slot crowding loss | 0% | 1.67% | +1.67 pp |

新Head的Forced Top-5只少找回2/410个真词，但固定0.86 Operating gate少通过约20个
真词；Precision和负例FPR几乎不变。这证明葡语回退主要是新Head的分数/置信度
标定与旧门槛不再匹配，不是Top-5排序能力崩坏，也不支持直接重训的结论。

4k formal100对比完整通过以下身份验证：两套checkpoint不同，两份CTC报告分别绑定
对应checkpoint，root/child配置除checkpoint、派生报告、Git和显存分配外一致，六组
样本选择完全一致，源SHA全部验证通过。C no-RAG和E Oracle质量逐字段相同，
因此D组差值可作为checkpoint替换结论。候选注入或预测变化的profile-case行共207条。

| 配置 | Prompt Recall 旧→新 | 最终Recall 旧→新 | 最终Precision 旧→新 | 新Head负样本幻觉率 |
| --- | ---: | ---: | ---: | ---: |
| D5 conservative | 84.30% → 80.23% | 91.28% → 91.86% | 80.93% → 82.72% | 0% |
| D7 balanced | 88.37% → 84.88% | 91.28% → 90.12% | 80.93% → 82.01% | 0% |
| D5 recall-first | 91.28% → 87.21% | 91.28% → 90.12% | 80.93% → 81.15% | 0% |
| D7 recall-first | 93.60% → 88.37% | 93.02% → 91.86% | 80.00% → 80.20% | 5% |

新Head四个D组的正确Prompt注入分别比旧Head少7/6/7/9个，与纯CTC Operating Recall
回退一致。最终文本因base ASR和Qwen Prompt采用而部分遮蔽了检索损失；D5 conservative
甚至最终Recall增加0.58 pp、Precision增加1.79 pp。但最高召回口径D7 recall-first仍从
160/172降到158/172，且出现1/20级别的纯负样本最终幻觉；它不再比D5 conservative
提供更高最终Recall，却只有80.20% Precision。

新suite四个D组纯检索P95均继续满足`<50 ms`：conservative 37.32 ms、balanced
38.00 ms、D5 recall-first 35.15 ms、D7 recall-first 37.57 ms。balanced和D7 recall-first的
P99分别约50.52/51.85 ms，但当前验收目标始终是P95，不将P99偷换成失败。两次
独立H200运行的latency不用于声称Head性能因果改善或退化。

本轮正式结论：

- 三语Macro训练成功收敛，英/西/葡PER全部比5 epoch继续下降；
- 新三语Head目前不能在不加限定的情况下宣称替代葡语专用Head，因为葡语Prompt
  Recall明确下降；
- 若必须直接部署当前统一三语Head，葡语暂选D5 conservative，而不是D7
  recall-first；
- 下一个最小任务应优先复用已有v3 case scores做新Head的葡语threshold/posterior/Top-K
  离线标定，目标是恢复Operating Recall且不牺牲Precision/FPR；
- 标定选出1至2个候选点后再决定是否重跑少量葡语D组端到端；不盲目重跑C/E或
  完整六组；
- 英语和西语目前只有validation PER，还没有与葡语formal100等价的端到端资产，
  不得从本次葡语结果外推两语端到端质量。

本轮结果汇总目录为
`outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1/streaming_checkpoint_regression_multilingual_ctc_v1`，
`checkpoint_regression_summary.json`、`checkpoint_regression_cases.jsonl`和`README.md`的SHA256已全部通过。
这些outputs继续只保留在H200工作区，不提交Git。本地只记录用户回传的经身份验证
汇总，没有伪造或重生成工作区结果。在用户确认下一任务前，不开始新的标定或英/西语实验。

## 0.61 2026-09-03 三语CTC Head定稿审计与葡语4k formal100替换回归

本轮启动已完成英/西/葡混合训练Head的端到端验证，但工作区30 epoch resume后的
最终结果尚未回传，因此本地不预设best epoch、Macro PER或分语种PER。新增的
`audit_multilingual_ctc_checkpoint.py`会只读检查`report.json`/`metrics.jsonl`的训练完成状态、
test未使用、cache SHA已验证、en/es/pt分组、Macro PER选择口径、best epoch数值一致性以及
Temporal-2x h512-k5/90类Head契约，然后把最佳checkpoint和训练状态的SHA256冻结到新目录。

葡语回归不能直接把新Head与旧
`multi_nested_evaluation_report_v3_corrected.json`混用：旧报告绑定旧葡语checkpoint SHA，
流式评测器会正确拒绝这种身份错配。因此固定验证顺序是：

1. 审计并冻结三语训练best checkpoint身份；
2. 在三语4小时validation cache上重做一次mixed + en/es/pt详细PER诊断；
3. 在旧葡语validation cache上复用原v3的210条cases/500词/family，仅替换Head，生成与
   新checkpoint绑定的CTC报告；
4. 复用原formal100的100条选择、4k热词资产、Anchor参数、2秒流式语义和六组门控，
   在新目录运行C/D5-conservative/D7-balanced/D5-recall/D7-recall/E；
5. 用新增`compare_streaming_checkpoint_suites.py`验证两套SHA、样本选择和运行契约，
   输出新Head减旧Head的核心质量差值及变化case。

对比器仅允许checkpoint、由checkpoint派生的CTC报告、Git提交身份和
`gpu_memory_utilization`不同；其他配置或样本变化会直接失败。历史D5 recall-first曾使用
0.20，其他组使用0.15，所以新工具不把两次独立H200运行的latency解释为Head因果差异；
纯检索P95仍在新suite内独立验收。C no-RAG和E Oracle不使用CTC检索，对比器会将
它们作为重跑稳定性控制；若C/E质量漂移，D组差值必须带限制解释。

工作区执行命令如下。所有新产物都使用新目录；首次运行前的`test ! -e`不得删除或
改成覆盖。`GPU_ID`可换成当时空闲物理卡，程序内仍为逻辑`cuda:0`：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

GPU_ID=3
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1
FORMAL_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1
CTC_CHECKPOINT="$FORMAL_ROOT/ctc_head_best.pt"
CHECKPOINT_AUDIT_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_checkpoint_audit_v1
MIXED_DIAG_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_final_diagnostics_v1

PT_ROOT=outputs/noah_pt_full_training_v1
PT_VALIDATION_CACHE="$PT_ROOT/features_ln_post_bf16/validation"
PT_VALIDATION_MANIFEST="$PT_ROOT/full_ctc_validation.jsonl"
PT_DICTIONARY=outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
V3_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested"
V3_MULTILINGUAL_ROOT="$PT_ROOT/simulated_hotword_eval_v3_multi_nested_multilingual_ctc_v1"

CAP_ROOT="$PT_ROOT/hotword_capacity_eval_v1"
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE100_ROOT="$V3_ROOT/prompt_multi_nested_formal100_top5_v1"
BASELINE_SUITE_ROOT="$CAP_ROOT/streaming_gate_suite_4k_formal100_v1"
CANDIDATE_SUITE_ROOT="$CAP_ROOT/streaming_gate_suite_4k_formal100_multilingual_ctc_v1"
REGRESSION_ROOT="$CAP_ROOT/streaming_checkpoint_regression_multilingual_ctc_v1"

test -f "$FORMAL_ROOT/report.json"
test -f "$FORMAL_ROOT/metrics.jsonl"
test -f "$CTC_CHECKPOINT"
test -d "$BASELINE_SUITE_ROOT"
test ! -e "$CHECKPOINT_AUDIT_ROOT"
test ! -e "$MIXED_DIAG_ROOT"
test ! -e "$V3_MULTILINGUAL_ROOT"
test ! -e "$CANDIDATE_SUITE_ROOT"
test ! -e "$REGRESSION_ROOT"
```

先做CPU-only的训练产物审计：

```bash
python scripts/audit_multilingual_ctc_checkpoint.py \
  --training-dir "$FORMAL_ROOT" \
  --output-dir "$CHECKPOINT_AUDIT_ROOT" \
  --expected-groups en,es,pt
```

再在已有三语validation feature cache上做详细分组诊断，这一步只加载CTC Head和已缓存特征，
不加载完整Qwen：

```bash
mkdir -p "$MIXED_DIAG_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/diagnose_frozen_ctc.py \
  --validation-cache "$FEATURE_CACHE_ROOT/validation" \
  --validation-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab "$VOCAB" \
  --checkpoint "$CTC_CHECKPOINT" \
  --output "$MIXED_DIAG_ROOT/grouped_validation_diagnostics.json" \
  --device cuda:0 \
  --batch-size 256 \
  --group-column balanced_language_bucket \
  --expected-groups en,es,pt
(cd "$MIXED_DIAG_ROOT" && \
  sha256sum grouped_validation_diagnostics.json > sha256.txt)
```

然后使用原葡语v3自然validation cases生成新checkpoint绑定的CTC中间报告。本步不重建
cases或热词，不读sealed test，不加载完整Qwen：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/evaluate_multi_nested_hotwords.py \
  --validation-cache "$PT_VALIDATION_CACHE" \
  --validation-manifest "$PT_VALIDATION_MANIFEST" \
  --dictionary "$PT_DICTIONARY" \
  --vocab "$VOCAB" \
  --checkpoint "$CTC_CHECKPOINT" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --asset-summary "$V3_ROOT/asset_summary_v3.json" \
  --output-dir "$V3_MULTILINGUAL_ROOT" \
  --device cuda:0 \
  --batch-size 128

python scripts/rebuild_multi_nested_hotword_report.py \
  --vocab "$VOCAB" \
  --hotwords "$V3_ROOT/multi_nested_hotwords_v3.jsonl" \
  --families "$V3_ROOT/hotword_families_v3.jsonl" \
  --cases "$V3_ROOT/multi_nested_cases_v3.jsonl" \
  --case-scores "$V3_MULTILINGUAL_ROOT/hotword_case_scores_v3.jsonl" \
  --base-report "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3.json" \
  --output "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json"

(cd "$V3_MULTILINGUAL_ROOT" && sha256sum \
  hotword_case_scores_v3.jsonl \
  multi_nested_evaluation_report_v3.json \
  multi_nested_evaluation_report_v3_corrected.json > sha256.txt)
```

最后在全新目录跑同资产六组formal100。`--resume`是为了中断后原命令续跑；它不会引用或
覆盖旧suite。H200空闲显存不足时只换`GPU_ID`，不提高`gpu-memory-utilization=0.15`：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/run_streaming_gate_profile_suite.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest "$PT_VALIDATION_MANIFEST" \
  --vocab "$VOCAB" \
  --hotwords "$ASSET_4K_ROOT/representative/size_4000/hotwords.jsonl" \
  --cases "$ASSET_4K_ROOT/representative/size_4000/cases.jsonl" \
  --hotword-families "$V3_ROOT/hotword_families_v3.jsonl" \
  --ctc-report "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --offline-rag-dir "$OFFLINE100_ROOT" \
  --ctc-checkpoint "$CTC_CHECKPOINT" \
  --output-dir "$CANDIDATE_SUITE_ROOT" \
  --language Portuguese \
  --gpu-memory-utilization 0.15 \
  --resume

python scripts/compare_streaming_checkpoint_suites.py \
  --baseline-suite "$BASELINE_SUITE_ROOT" \
  --candidate-suite "$CANDIDATE_SUITE_ROOT" \
  --output-dir "$REGRESSION_ROOT"
```

验收与回传命令：

```bash
(cd "$CHECKPOINT_AUDIT_ROOT" && sha256sum -c sha256.txt)
(cd "$MIXED_DIAG_ROOT" && sha256sum -c sha256.txt)
(cd "$V3_MULTILINGUAL_ROOT" && sha256sum -c sha256.txt)
(cd "$CANDIDATE_SUITE_ROOT" && sha256sum -c sha256.txt)
(cd "$REGRESSION_ROOT" && sha256sum -c sha256.txt)

jq '{best_checkpoint, training_progress, head_contract, best_validation, final_validation}' \
  "$CHECKPOINT_AUDIT_ROOT/checkpoint_audit.json"
jq '{validation_samples, expected_groups, checkpoints}' \
  "$MIXED_DIAG_ROOT/grouped_validation_diagnostics.json"
jq '{checkpoint_sha256, head_config, scoring_config, metrics, status}' \
  "$V3_MULTILINGUAL_ROOT/multi_nested_evaluation_report_v3_corrected.json"
jq '{sample_count, profiles}' "$CANDIDATE_SUITE_ROOT/suite_summary.json"
jq . "$REGRESSION_ROOT/checkpoint_regression_summary.json"
wc -l "$REGRESSION_ROOT/checkpoint_regression_cases.jsonl"
```

请回传以下小文件，不需要checkpoint、feature cache、sample shard、完整timeline或模型：

```text
en_es_pt_balanced_150h_temporal2x_ctc_checkpoint_audit_v1/
  checkpoint_audit.json
  sha256.txt
en_es_pt_balanced_150h_temporal2x_ctc_final_diagnostics_v1/
  grouped_validation_diagnostics.json
  sha256.txt
simulated_hotword_eval_v3_multi_nested_multilingual_ctc_v1/
  multi_nested_evaluation_report_v3.json
  multi_nested_evaluation_report_v3_corrected.json
  sha256.txt
streaming_gate_suite_4k_formal100_multilingual_ctc_v1/
  suite_config.json
  suite_summary.json
  topk_isolation_summary.json
  sha256.txt
streaming_checkpoint_regression_multilingual_ctc_v1/
  checkpoint_regression_summary.json
  checkpoint_regression_cases.jsonl
  sha256.txt
```

收到后先核验SHA和两个checkpoint身份，再分别回答：三语训练是否已收敛、葡语PER是否相比
旧单语Head退化、4k D组Prompt Recall/最终Recall/Precision如何变化，以及C/E控制是否稳定。
本轮本地只做mock/单元验证，没有加载Qwen3-ASR-1.7B、没有读sealed test、没有修改
训练/检索/提示词算法，也没有改动旧outputs。全量pytest 222项通过，本轮相关
27项通过；新增两个核心模块strict Mypy和本轮文件Ruff均通过，CLI help与
`git diff --check`通过。全仓Ruff仍有`scan_g2p_coverage.py`的3个原有E501，另有2个来自
未跟踪PPT临时目录，本轮均未修改。本机全包Mypy在项目Python 3.10解析目标下被
NumPy的3.12 stub语法阻断；切到3.12目标后显示的11个错误均位于6个未修改的旧模块。

## 0.60 2026-08-29 eSpeak-ng/MFA三语新热词G2P选型入口

新增完全独立的 eSpeak-ng/MFA 对比工具，用于判断新热词 G2P 是否可由 eSpeak-ng
替代 MFA。本轮不设硬性通过阈值，只收集英、西、葡各500词的转换成功、90类CTC词表
OOV、规范化音素序列一致度和系统性差异；不修改4k热词任务或三语增训代码，不训练
CTC、不加载Qwen、不读取validation/test，也不覆盖任何现有产物。

独立实现为`src/qwen_hotword/phonemes/espeak_mfa_comparison.py`、
`scripts/compare_espeak_mfa_g2p.py`和`tests/test_espeak_mfa_comparison.py`。完整设计、输入
身份、运行命令、输出说明和后续结果统一维护在`docs/ESPEAK_NG选型.md`。工具固定使用
`en-us`/`es-419`/`pt-br`，从三语150小时平衡train Manifest统计词频，只选具有唯一
且可映射MFA发音的候选；每语言按特殊拼写、长词、低/中/高频五层确定性选500词。

工作区拉取`codex/g2p-coverage-scan`最新远端提交并核对HEAD后，严格按
`docs/ESPEAK_NG选型.md`第4节运行。输出固定为新目录
`outputs/espeak_mfa_selection_v1`；目录已存在时拒绝覆盖，不支持resume。运行前确认
`phonemizer`和系统`espeak-ng`版本，不能换用其他后端。完成后在输出目录执行
`sha256sum -c sha256.txt`，并回传该目录九个小型文本文件；其中两个JSONL各1500行。
后续本地先核验Git、输入和输出SHA，再分析逐词混淆及人工复核样本，并把实测结论只补入
`docs/ESPEAK_NG选型.md`，不写入前两个任务的实现代码或产物。

本地验证：新增文件Ruff、严格Mypy、3项定向pytest、CLI help和`git diff --check`
通过；全量pytest为195 passed、23 skipped。全仓Ruff仍只有接管时已存在的
`scripts/scan_g2p_coverage.py`三项E501；`MYPYPATH=src mypy`仍只有三个既有训练模块的
unused-ignore，本轮新增模块为0项。由于本机未安装`phonemizer`和系统`espeak-ng`，没有
生成或宣称真实G2P质量结果；mock只验证分层、语言voice、指标、输出和防覆盖契约。

## 0.59 2026-08-28 D5 Recall-first Top-K隔离实测结果

新增`recall_first_top5`已在4k formal100完成：100条validation工程校准样本、172个
expected sample-hotword对、322个流式step，`status=pass`、`test_set_used=false`。
两级SHA256全部通过；根汇总和D5子运行关键文件身份如下：

```text
suite_config.json:                    0d62a45101e226e0823591942d5a6f2d360a5a7d041309f8e0f5ed8af85e1619
suite_summary.json:                   9622e8b7db304878a56cf1cc321e9c83e5278791cd0b7fcdbf178201c8f2a6ca
topk_isolation_summary.json:          4ac5f93be48bdaef31ffe9f4c46d2887273b02e5af54e364a35121d42c39f524
topk_isolation_cases.jsonl:           695a3def22bb048161dd15ef4339af66412a2281b71e43fa7b304bbc6fafcb9b
recall_first_top5/run_config.json:     7dcce3bd710597a1edc009382d36c1fdccec89c0b67f0cf5e1295a2bb6bf3cc5
recall_first_top5/summary.json:        6929784921009de5fc7504fc5a9130244a98f5b4f35e9e87794cbece126a8b75
recall_first_top5/latency_summary.json: fd1d934a740d01dda8ae8598f0104749a32eeb8e68e0b0f16e81e03797214d43
```

D5运行身份为Qwen3-ASR-1.7B、qwen-asr 0.0.6、Portuguese、bfloat16、逻辑
`cuda:0`、`gpu_memory_utilization=0.20`、2 s / 2 chunks / 5 tokenizer tokens，4k
Anchor Top-64、radius 2、2/3/4-gram、每词24 anchors、offset tolerance 1；门控只有
`threshold=0.75`、posterior weight 0.25、minimum posterior 0.5、margin 0、Top-5。
validation、词表、checkpoint、4k hotwords/cases、模型config/tokenizer config及离线
formal100选择均在`run_config.json`记录SHA256，和0.58要求的输入一致。

本次返回证据有两项身份限制，必须显式保留：D5 `run_config.json`中的`git_commit`为
`unknown`，没有嵌入实际HEAD；成功运行所用物理GPU编号也没有随返回日志记录。运行行为
包含仅在交付提交`1a1e74c`引入的六profile和专用Top-K汇总，可支持该代码来源的工程
判断，但不能把产物内Git字段写成已验证SHA。物理GPU未知不影响质量计数，但D5和旧D7
的跨进程时延差异只能作为实测分布，不能解释为Top-K本身的确定性加速或减速。

D5核心质量结果：

```text
Prompt热词召回:       157/172 = 91.28%
正确Prompt采用:       145/157 = 92.36%
错误Prompt落字:        37/461 = 8.03%
最终热词Recall:       157/172 = 91.28%
最终热词Precision:    157/(157+37) = 80.93%
样本热词命中率:        82.50%
负样本最终热词幻觉率:   0%
WER / CER:             8.37% / 3.70%
```

相对完全同门控的D7 recall-first，Top-5少注入4个正确热词，Prompt召回下降
2.33个百分点；最终少识别3个expected热词，Recall下降1.74个百分点。同时Top-5少注入
97个错误候选、少落字3个错误热词，最终Precision提高0.93个百分点，WER/CER分别改善
0.13/0.22个百分点。正确Prompt采用率下降1.43个百分点；58条case的注入候选或最终文本
在Top-5/Top-7之间发生变化。

纯检索P50/P95/P99/max为24.46/39.16/48.63/61.54 ms，P95继续满足50 ms目标；
3/322个step超过50 ms，不能因P95通过而隐藏。Detector P95为97.94 ms，完整step P95
为316.85 ms，样本推理均值556.19 ms，样本RTF中位数/P95为0.077/0.151。D5本轮纯
检索P95比旧D7高0.94 ms且出现3个deadline miss，再次说明不同进程的微小时延差不能
直接归因于Top-K。

Top-K隔离结论：第6至7名候选确实带来3/172个最终热词Recall，但也带来97个错误Prompt
候选和3个错误落字。D7 recall-first仍是4k召回上限点；D5 recall-first以1.74个百分点
最终Recall换取0.93个百分点Precision，仍只有80.93%，没有达到85%目标。它与D5
conservative最终Recall/Precision恰好相同，但比conservative多340个错误注入且最终仍
落字37个，因此不能替代conservative作为低噪声配置。后续仍应优先做family-aware同族/
嵌套冲突消解，而不是仅靠缩小Top-K或继续放宽全局threshold。

本轮还暴露两个产物可追溯性问题：`_git_commit()`在容器的Git safe-directory场景会
静默写入`unknown`；子运行`sha256.txt`写仓库根相对路径，导致进入子目录执行
`sha256sum -c`时错误重复拼接路径。后续代码修复这两点；当前已完成产物不得为修复格式
而重跑或改写，其子SHA应从仓库根目录验证。

本地验证：流式RAG/suite定向pytest 11项通过，全量pytest 215项通过；本轮相关文件
Ruff和严格Mypy通过，`git diff --check`通过。全仓Ruff仍只有接管时已存在的
`scripts/scan_g2p_coverage.py`三项E501，本轮未修改该文件。未加载完整模型、未读取test。

## 0.58 2026-08-28 D5 Recall-first Top-K隔离组与Prompt召回口径

4k formal100套件新增`recall_first_top5`组：`threshold=0.75`、
`minimum_posterior_confidence=0.5`、`Top-K=5`。它与现有`recall_first` Top-7共享
完全相同的Anchor、精排、score threshold、posterior和margin参数，只改变Top-K，
用于隔离第6至7名候选对Prompt注入、最终Recall/Precision、WER/CER和时延的影响。

Prompt因果汇总新增`prompt_hotword_recall`：曾进入Prompt的正确expected
sample-hotword对数 / expected sample-hotword总数。formal100分母固定为172；例如原
D5 conservative为145/172、D7 balanced为152/172、D7 recall-first为161/172。
`correct_prompt_adoption_rate`继续表示正确候选进入Prompt后最终被Qwen写出的比例。
`wrong_prompt_landing_rate`继续表示错误注入候选最终落字的比例；
`wrong_prompt_filter_rate`与它互为补数，JSON中保留兼容，但不再在核心表重复展示。

套件现在包含六个profile：C no-RAG、D5 conservative、D7 balanced、
D5 recall-first、D7 recall-first和E Oracle。对原五组formal100目录使用完全相同命令
增加`--resume`时，只允许这一项profile的加法升级；已有五个完整子目录保持只读并由根
汇总直接复用，新`recall_first_top5/`子目录只运行新增D5组。根`suite_summary.json`
从既有`sample_results.jsonl`补算Prompt热词召回，不要求旧子运行跨Git提交resume。
不能改变其他输入、运行参数或SHA身份。
新增D5结果已在0.59完成实测；0.57两张表已补入实际结果，没有用D7截断结果推测。

本次不是只交付本地实现。工作区应拉取`codex/g2p-coverage-scan`的最新远端提交，确认
HEAD后，在原formal100根目录做受控加法resume。脚本会验证原五组suite配置，只复用
已经`status=pass`且文件完整的旧子目录；不会覆盖旧sample shard，也不会重跑旧五组。
新增D5写入此前不存在的`recall_first_top5/`，然后重建根汇总：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
V3_ROOT=outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested
OFFLINE100_ROOT="$V3_ROOT/prompt_multi_nested_formal100_top5_v1"
SUITE_FORMAL_ROOT="$CAP_ROOT/streaming_gate_suite_4k_formal100_v1"

test -d "$SUITE_FORMAL_ROOT"
test ! -e "$SUITE_FORMAL_ROOT/recall_first_top5"

CUDA_VISIBLE_DEVICES=2 python scripts/run_streaming_gate_profile_suite.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords "$ASSET_4K_ROOT/representative/size_4000/hotwords.jsonl" \
  --cases "$ASSET_4K_ROOT/representative/size_4000/cases.jsonl" \
  --hotword-families "$V3_ROOT/hotword_families_v3.jsonl" \
  --ctc-report "$V3_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --offline-rag-dir "$OFFLINE100_ROOT" \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --output-dir "$SUITE_FORMAL_ROOT" \
  --language Portuguese \
  --gpu-memory-utilization 0.15 \
  --resume
```

若物理GPU 2不可用，只替换`CUDA_VISIBLE_DEVICES`的物理卡号；程序内设备仍保持默认
逻辑`cuda:0`，其他参数不得改变。进程中断后原命令原样重跑即可，已完成profile和sample
shard由`--resume`复用。完成后先执行：

```bash
(cd "$SUITE_FORMAL_ROOT" && sha256sum -c sha256.txt)
sha256sum -c "$SUITE_FORMAL_ROOT/recall_first_top5/sha256.txt"

jq . "$SUITE_FORMAL_ROOT/topk_isolation_summary.json"
wc -l "$SUITE_FORMAL_ROOT/topk_isolation_cases.jsonl"
```

脚本会自动生成两个专用小产物：`topk_isolation_summary.json`只保留两组门控、核心质量、
完整分阶段时延及Top-5减Top-7差值；`topk_isolation_cases.jsonl`只包含注入候选或最终
文本发生变化的case。请回传以下文件，不需要回传模型、checkpoint、Posterior tensor、
完整`sample_results.jsonl`或`chunk_timeline.jsonl`：

```text
streaming_gate_suite_4k_formal100_v1/
  suite_config.json
  suite_summary.json
  topk_isolation_summary.json
  topk_isolation_cases.jsonl
  sha256.txt
  recall_first_top5/
    run_config.json
    summary.json
    latency_summary.json
    failure_cases.jsonl
    sha256.txt
```

收到这些文件后，下一次本地工作固定为：先核验两级SHA256和`run_config.json`中的Git、
模型、checkpoint、输入资产及门控身份，再补0.57两张表的D5实测值，分析Top-K对Recall、
Precision、错误落字和P95的净影响，最后单独提交并推送结果记录。

本地验证：流式suite/RAG定向pytest 9项通过，全量pytest 213项通过；本轮相关文件
Ruff和严格Mypy通过，CLI help与`git diff --check`通过。全仓Ruff仍只有接管时已存在的
`scripts/scan_g2p_coverage.py`三项E501，本轮未修改该文件。未加载完整模型、未读取test。

## 0.57 2026-08-26 formal100 Prompt因果指标结果

现有formal100分片经`--resume`完成纯重汇总，C、D5 conservative、D7-balanced、
D7-recall和E使用同一批100条（80正/20负），没有重跑模型或改变推理结果。0.58新增的
D5 recall-first使用相同选择，只改变D7 recall-first的Top-K；实测结果记录于0.59。

六组共同使用官方流式语义`chunk_size_sec=2.0`、`unfixed_chunk_num=2`、
`unfixed_token_num=5`；正常RAG组共同使用4k热词库、`anchor_guided`检索、shortlist 64、
start radius 2、音素2/3/4-gram、每词24个anchor、offset tolerance 1、
`maximum_edit_ratio=0.35`、`posterior_weight=0.25`和`minimum_top1_margin=0`。推理使用
Portuguese和bfloat16。原五组使用物理GPU 2映射为逻辑`cuda:0`、
`gpu_memory_utilization=0.15`；新增D5在另一进程使用逻辑`cuda:0`和0.20，成功运行的
物理GPU编号未随返回日志记录，因此跨组时延差只作实测分布，不作Top-K因果解释。
各组差异参数如下：

| 组 | RAG/候选来源 | threshold | minimum posterior | Top-K |
| --- | --- | ---: | ---: | ---: |
| C no-RAG | 不注入热词 | 不适用 | 不适用 | 不适用 |
| D5 conservative | CTC + Anchor operating gate | 0.86 | 0.0 | 5 |
| D7 balanced | CTC + Anchor operating gate | 0.82 | 0.5 | 7 |
| D5 recall-first | CTC + Anchor operating gate | 0.75 | 0.5 | 5 |
| D7 recall-first | CTC + Anchor operating gate | 0.75 | 0.5 | 7 |
| E Oracle | 只注入样本expected热词，不运行CTC检索门控 | 不适用 | 不适用 | expected数量 |

核心质量指标：

| 组 | Prompt热词召回率 | 正确Prompt采用率 | 错误Prompt落字率 | 最终Recall | 最终Precision | WER / CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C no-RAG | 0/172 = 0% | 不适用 | 不适用 | 89.53% | 不适用 | 8.30% / 3.39% |
| D5 conservative | 145/172 = 84.30% | 136/145 = 93.79% | 37/121 = 30.58% | 91.28% | 80.93% | 8.84% / 3.78% |
| D7 balanced | 152/172 = 88.37% | 141/152 = 92.76% | 37/227 = 16.30% | 91.28% | 80.93% | 8.84% / 3.89% |
| D5 recall-first | 157/172 = 91.28% | 145/157 = 92.36% | 37/461 = 8.03% | 91.28% | 80.93% | 8.37% / 3.70% |
| D7 recall-first | 161/172 = 93.60% | 151/161 = 93.79% | 40/558 = 7.17% | 93.02% | 80.00% | 8.50% / 3.92% |
| E Oracle | 172/172 = 100% | 161/172 = 93.60% | 0个错误注入（不适用） | 93.60% | 100% | 8.10% / 3.36% |

formal100时延如下；纯检索列为322个流式step上的CTC greedy decode + Anchor query +
shortlist重排，不含CTC Encoder/Head，这一列才对应`P95 < 50 ms`目标。Detector列包含
Processor、CTC Encoder/Head、decode和检索；step列再包含Prompt刷新与同轮Qwen流式解码。
样本耗时不含音频文件读取，长尾值保留冷启动/编译影响：

| 组 | 纯检索 P50 / P95 / P99 / max | Detector P95 | 完整step P95 | 样本推理均值 | 样本RTF中位数 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C no-RAG | 不适用 | 不适用 | 175.60 ms | 291.88 ms | 0.039 / 0.093 |
| D5 conservative | 26.56 / 39.10 / 42.86 / 49.66 ms | 94.70 ms | 290.24 ms | 576.83 ms | 0.085 / 0.144 |
| D7 balanced | 25.06 / 36.35 / 41.36 / 47.50 ms | 99.98 ms | 302.73 ms | 585.73 ms | 0.082 / 0.150 |
| D5 recall-first | 24.46 / 39.16 / 48.63 / 61.54 ms | 97.94 ms | 316.85 ms | 556.19 ms | 0.077 / 0.151 |
| D7 recall-first | 25.32 / 38.23 / 40.43 / 43.90 ms | 93.70 ms | 306.27 ms | 538.22 ms | 0.081 / 0.142 |
| E Oracle | 不适用 | 不适用 | 78.27 ms | 149.18 ms | 0.025 / 0.045 |

C组完整step的P99/max为274.16/4171.32 ms；三个D组Detector max分别为
1128.61/1166.58/661.16 ms，完整step max分别为5339.26/5436.83/4747.31 ms。
这些冷启动长尾不能计入50 ms纯检索验收，也不能静默删除。E组绕过Detector且与C组存在
同进程运行顺序和预热差异，因此E比C快不能解释成Oracle Prompt本身具有加速作用。

正确Prompt一旦进入候选，四个D组最终采用率都约92%至94%，与Oracle的93.60%接近。
说明Qwen能够稳定利用正确Prompt；D7 Recall-first最终Recall只比Oracle低1/172，即
0.58个百分点。Top-K隔离确认第6至7名带来3个最终正确热词，同时带来97个额外错误
候选和3个额外错误落字。

Qwen也能过滤大部分错误Prompt，但过滤比例不能脱离候选基数解释。Recall-first虽然过滤
92.83%，却注入558个错误sample-hotword对，剩余40个落字；D5只过滤69.42%，但错误候选
总量只有121，最终同样约37个落字。因此此前约79%的`final_prompted_hotword_precision`
并不与92.83%的错误过滤率矛盾。

20条纯负样本中，D5/balanced/recall-first分别有5/9/18条收到错误候选，错误候选数为
9/25/88，但最终错误落字均为0，负样本最终幻觉率均为0%。主要污染发生在本身含热词的
正样本：Recall-first的40个错误落字全部集中在`nested_family_plus_two`的25个和
`nested_long_present`的15个。同族/嵌套冲突是下一步Precision优化的明确目标。

通用文本变化：C组WER/CER为8.30%/3.39%；D5 conservative为8.84%/3.78%，balanced为
8.84%/3.89%，D5 recall-first为8.37%/3.70%，D7 recall-first为8.50%/3.92%，Oracle
为8.10%/3.36%。D7 recall-first提供最高D组Recall；D5 recall-first相对D7的WER/CER
更好，但最终Recall低1.74个百分点，Precision只高0.93个百分点。balanced相对
conservative没有Recall或最终Precision收益，错误候选更多、CER更差，暂时淘汰。

正式结论：Qwen可以过滤错误Prompt，尤其能完全保护这20个纯负样本，但不能可靠消解
正样本中的同族/嵌套近邻。若Precision 85%暂不作为硬门槛，D7 recall-first作为容量
上限测试配置，D5 conservative作为保守对照；D5 recall-first只作为Top-K隔离证据，
不替代两者。当前4k结果仍不是达到85% Precision的最终产品配置。后续
容量测试可以先用纯离线Anchor向5k以上推进，同时单独导出40个错误落字做family-aware
冲突消解，避免继续通过全局降低threshold增加候选噪声。

## 0.56 2026-08-26 流式端到端Prompt因果指标补全

formal100已按同一选择完成C、D5 conservative、D7 balanced、D7 recall-first和E，
但旧汇总没有显式给出“正确Prompt采用率”，且`final_prompted_hotword_precision`容易被
误读为错误候选过滤率。现已在不改变推理结果的前提下补齐以下显式指标：

- `correct_prompt_adoption_rate`：曾注入的正确sample-hotword对中，最终文本出现该词的比例；
- `prompt_hotword_recall`：曾注入的正确sample-hotword对 / expected sample-hotword总数；
- `wrong_prompt_filter_rate`：曾注入的错误sample-hotword对中，最终文本未出现该词的比例；
- `wrong_prompt_landing_rate`：曾注入的错误sample-hotword对中，最终真正落字的比例；
- `final_hotword_recall`：最终正确的expected热词数 / expected热词总数；
- `final_hotword_precision`：最终正确expected热词数 /（最终正确expected热词数 + 最终落字的
  错误注入热词数）；
- 继续保留WER/CER、相对C组变化和`negative_hotword_hallucination_rate`。

计数单位固定为“每个样本内去重、跨chunk累计的sample-hotword对”，不会把同一个候选在
多个2秒step反复注入重复计数。汇总同时保存正确注入/采用、错误注入/过滤/落字的原始
计数和嵌套`prompt_causal_metrics`，方便复核分母。

旧formal100的错误候选最终过滤率可以由既有落字率反算：conservative约69.42%、
balanced约83.70%、recall-first约92.83%。此前约79%的
`final_prompted_hotword_precision`不是“只过滤21%的错误候选”，而是“所有最终落字的
Prompt候选中约21%为错误”；两者分母不同。

现有100条sample shard已经包含重汇总所需字段。工作区拉取后用原formal100套件命令加
`--resume`即可；所有sample shard存在时不会加载Qwen、CTC或重跑音频，只会重读分片并
重写summary/README/SHA256。重汇总后重点回传：

```bash
(cd "$SUITE_FORMAL_ROOT" && sha256sum -c sha256.txt)

jq '{
  sample_count,
  profiles: (.profiles | with_entries(.value |= {
    gate,
    quality: (.quality | {
      correct_prompt_injected_hotwords,
      expected_hotwords,
      prompt_hotword_recall,
      correct_prompt_adopted_hotwords,
      correct_prompt_adoption_rate,
      wrong_injected_hotwords,
      wrong_prompt_written_hotwords,
      wrong_prompt_landing_rate,
      final_hotword_recall,
      final_hotword_precision,
      wer,
      cer,
      negative_hotword_hallucination_rate
    })
  }))
}' "$SUITE_FORMAL_ROOT/suite_summary.json"
```

这一步只补全因果统计口径，不调整4k词库、Anchor shortlist、门控参数、Prompt模板或
流式2 s / 2 chunks / 5 tokenizer tokens语义。

## 0.55 2026-08-26 4k五组流式端到端formal100结果

`streaming_gate_suite_4k_formal100_v1`已完成100条工程校准样本，其中80条正样本、
20条无热词负样本；五组均`status=pass`，未读取封存test集。正式结果如下：

| 组 | Exact recall | 样本命中率 | 最终Prompt精确率 | 负样本最终幻觉率 | WER | CER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-RAG | 89.53% | 78.75% | 不适用 | 0% | 8.30% | 3.39% |
| conservative 0.86/Top-5 | 91.28% | 82.50% | 78.61% | 0% | 8.84% | 3.78% |
| balanced 0.82/Top-7 | 91.28% | 82.50% | 79.21% | 0% | 8.84% | 3.89% |
| recall-first 0.75/Top-7 | 93.02% | 85.00% | 79.06% | 0% | 8.50% | 3.92% |
| Oracle | 93.60% | 86.25% | 100% | 0% | 8.10% | 3.36% |

Recall-first为160/172，Oracle为161/172，两者只差1个热词；相对no-RAG，Recall-first
提升3.49个百分点，样本命中率提升6.25个百分点。说明4k Anchor shortlist和放宽后的
CTC门控已接近当前流式Qwen/Prompt机制的Oracle上限，继续单纯降低threshold的收益很小。
Oracle仍有11个`prompt_injected_but_decoder_failed`，确认剩余问题不能全部归因于检索。

Qwen确实有显著过滤作用，但不是无条件安全：

- conservative/balanced/recall-first分别注入121/227/558个错误候选，最终错误写穿比例
  为30.58%/16.30%/7.17%；对应最终Prompt真/假热词为136/37、141/37、151/40。
- 20条纯负样本中，错误候选注入样本率分别为25%/45%/90%，但三个组最终热词幻觉率
  都是0%。因此Qwen对纯负样本的错误Prompt过滤很好。
- 最终约79%的Prompt精确率主要被正样本中的`nested_family_plus_two`和
  `nested_long_present`拉低；Qwen会在已有正确热词语境中写入错误的同族或嵌套近邻。
  这类错误不能用“负样本幻觉率为0”掩盖。
- 三个正常RAG组的WER/CER均比no-RAG略差；Recall-first的WER只增加0.20个百分点，
  但CER增加0.53个百分点。Oracle反而略优于no-RAG，说明问题来自错误候选集合，而不是
  Prompt机制必然损害通用识别。

4,000词纯检索目标在322个流式step上继续稳定通过：

| 组 | P50 | P95 | P99 | max | 超50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | 26.56 ms | 39.10 ms | 42.86 ms | 49.66 ms | 0% |
| balanced | 25.06 ms | 36.35 ms | 41.36 ms | 47.50 ms | 0% |
| recall-first | 25.32 ms | 38.23 ms | 40.43 ms | 43.90 ms | 0% |

上述50 ms目标只针对CTC greedy decode + Anchor query + shortlist重排，不包含Encoder与
Head。包含CTC Encoder/Head后的detector P95约93.7至100.0 ms；再包含Prompt和Qwen的
step P95约290至306 ms。三个RAG组样本RTF中位数约0.081至0.085、P95约0.142至0.150，
仍满足实时处理；冷启动长尾继续单独保留，不能混同纯检索预算。

当前决策：4k已满足“纯检索P95 < 50 ms、端到端Recall > 90%”，但没有满足最终Prompt
Precision 85%。如果暂时不把85%作为硬门槛，Recall-first是召回上限诊断点，
conservative是错误注入更少的安全点；balanced没有提供召回收益，暂不推荐。下一阶段
不再做全局threshold放宽，而应先导出并分析37至40个最终假热词，重点设计显式的
同族/嵌套冲突消解；同时可用纯离线Anchor评测向5k以上做容量阶梯，只有前一档仍满足
P95 < 50 ms和Recall > 90%时才做昂贵端到端复验。

## 0.54 2026-08-26 4k五组流式端到端smoke20结果

热修复后，`streaming_gate_suite_4k_smoke20_v1`在物理GPU 2上完成，根汇总
`status=pass`，共20个样本、每组67个流式step，未读取封存test集。实际运行使用
`gpu_memory_utilization=0.15`。这20条恰好全部是正样本，因此本轮可以验证正样本召回、
错误Prompt候选是否写穿和计算时延，但`negative_hotword_hallucination_rate`及
`negative_wrong_hotword_injection_rate`均为null，不能把本轮的Prompt精确率1.0解释成
包含负样本的产品精确率。

最终质量：

| 组 | 热词exact recall | 样本命中率 | WER | CER | 错误注入候选 | 错误写穿率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-RAG | 91.67% | 75% | 6.56% | 3.02% | 0 | 不适用 |
| conservative 0.86/Top-5 | 95.00% | 85% | 5.63% | 2.38% | 20 | 0% |
| balanced 0.82/Top-7 | 95.00% | 85% | 5.63% | 2.81% | 39 | 0% |
| recall-first 0.75/Top-7 | 95.00% | 85% | 6.25% | 3.09% | 103 | 0% |
| Oracle | 95.00% | 85% | 6.25% | 2.88% | 0 | 不适用 |

三个正常RAG组均比no-RAG多3.33个百分点热词召回；但放宽到balanced或recall-first没有
超过conservative的最终召回。conservative同时拥有最少错误注入、最低CER和并列最低
WER，因此是当前smoke中的临时最优运行点。错误候选20/39/103个均未写入最终文本，
说明Qwen在这20个正样本上有明显Prompt过滤能力；正式100条包含20个负样本后，才能
判断这种过滤能否可靠抑制负样本幻觉。

Oracle也只有95% recall，3个失败均为`prompt_injected_but_decoder_failed`。因此本轮
剩余5%不是单纯放宽CTC门控即可解决：至少在这批样本中，流式解码、Prompt利用或
fixed/unfixed机制已经构成上限。正常RAG失败则为CTC从未检出或检出太晚已固定；
conservative为2/1，balanced和recall-first均为1/2。

4,000词纯检索（CTC greedy decode + Anchor + shortlist重排，不含Encoder/Head）全部满足
50 ms目标：

| 组 | P50 | P95 | P99 | max | 超50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | 25.36 ms | 36.87 ms | 39.20 ms | 39.22 ms | 0% |
| balanced | 24.87 ms | 38.34 ms | 39.99 ms | 40.26 ms | 0% |
| recall-first | 26.33 ms | 38.29 ms | 39.48 ms | 39.64 ms | 0% |

包含Encoder、Head、检索、Prompt和Qwen的step P95约378/370/411 ms；样本级平均推理
约0.857/0.891/0.877 s，RTF中位数约0.088/0.091/0.093，整体具备实时性。但三个独立
进程的首轮冷启动/编译高度疑似主导了长尾：step max达到5.35至5.68 s、P99达到
2.08至2.26 s；正式报告需同时
保留全量冷启动指标和稳态指标，不能用当前20条的长尾直接代表在线稳态。Oracle与
no-RAG在同一个子进程中顺序运行，Oracle均值更低也不能解释成Oracle天然更快。

`ctc_first_detect_latency_sec`、`first_correct_latency_sec`和
`stabilization_latency_sec`计数为0，不是程序漏跑，而是这套自然formal case没有人工或
强制对齐的热词声学起止时间；这些“相对热词声学结束”的延迟只能由边界覆盖集报告。

下一步先校验smoke根目录和四个子目录SHA256，然后以新目录跑满formal100。原命令仅需：

```bash
SUITE_FORMAL_ROOT="$CAP_ROOT/streaming_gate_suite_4k_formal100_v1"
test ! -e "$SUITE_FORMAL_ROOT"
```

将`--output-dir "$SUITE_SMOKE_ROOT"`改为`--output-dir "$SUITE_FORMAL_ROOT"`，删除
`--max-samples 20`，其余模型、资产、checkpoint、流式参数和三组门控必须完全不变。
formal100用于回答负样本幻觉、错误候选写穿及三个门控的最终取舍；在结果出来前不调整
阈值、Top-K或Prompt模板。

## 0.53 2026-08-26 4k流式套件case loader热修复

首次`streaming_gate_suite_4k_smoke20_v1`在模型加载前失败，报错
`must have 100 unique active hotwords`。原因是Step 6的selection-only路径仍调用旧v3
loader，而旧loader为原formal100资产写死了100个active hotwords。4k容量case本身
没有问题，GPU和`gpu_memory_utilization=0.15`也尚未进入执行。

修复后：严格v3路径默认仍要求恰100个；只有显式`selection_only`路径允许可变
active容量，但仍强制ID唯一、expected是active子集，case/sample/audio唯一，并与
formal100选择的sample、audio和参考文本逐条核对。已有失败输出不需删除：拉取
补丁后在原套件命令末尾加`--resume`即可继续。

## 0.52 2026-08-26 4k五组流式端到端套件与分阶段时延

Step 5门控扫描已收口，现进入Step 6端到端Prompt过滤诊断。新套件固定在同一
formal100选择上跑五个概念组：

1. `no_rag`：流式C组；
2. `conservative`：0.86 / edit 0.35 / posterior min 0 / Top-5；
3. `balanced`：0.82 / edit 0.35 / posterior min 0.5 / Top-7；
4. `recall_first`：0.75 / edit 0.35 / posterior min 0.5 / Top-7；
5. `oracle`：流式E组。

新增`anchor_guided`真实流式detector路径：当前累计音频经Qwen Encoder和固定
Temporal 2x CTC Head后，先做greedy phoneme decode，再用2/3/4-gram Anchor生成Top-64
shortlist，最后只在shortlist上运行原近似评分和Operating门控。Anchor/重排期间
暂停Python GC，退出detector后恢复；不改CTC checkpoint、Prompt模板或Qwen流式
2 s / 2 chunks / 5 tokenizer tokens语义。

每个chunk在`chunk_timeline.jsonl.compute_timing`记录：Processor、Encoder、CTC Head、
CTC decode、Anchor query、shortlist matching/sort/select、纯检索、Prompt build/refresh、
Qwen streaming/finish和整步wall clock。`latency_summary.json`输出P50/P90/P95/P99/max，
并单独记录纯检索超过50 ms比例。样本级`inference_seconds`是detector+Prompt+Qwen
的实测端到端wall clock，不包含从磁盘加载音频；`real_time_factor`同时保存。

最终质量新增`final_prompted_hotword_precision`和
`wrong_injected_write_through_rate`：前者表示最终真正写入文本的Prompt热词精确率，
后者表示错误注入候选有多少被Qwen照抄。这两项用于判断检索端低Precision是否
真会传导到最终ASR。

工作区先拉20条smoke：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
V3_ROOT=outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested
OFFLINE100_ROOT="$V3_ROOT/prompt_multi_nested_formal100_top5_v1"
SUITE_SMOKE_ROOT="$CAP_ROOT/streaming_gate_suite_4k_smoke20_v1"

test ! -e "$SUITE_SMOKE_ROOT"
CUDA_VISIBLE_DEVICES=4 python scripts/run_streaming_gate_profile_suite.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords "$ASSET_4K_ROOT/representative/size_4000/hotwords.jsonl" \
  --cases "$ASSET_4K_ROOT/representative/size_4000/cases.jsonl" \
  --hotword-families "$V3_ROOT/hotword_families_v3.jsonl" \
  --ctc-report "$V3_ROOT/multi_nested_evaluation_report_v3_corrected.json" \
  --offline-rag-dir "$OFFLINE100_ROOT" \
  --ctc-checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --output-dir "$SUITE_SMOKE_ROOT" \
  --max-samples 20 \
  --language Portuguese \
  --gpu-memory-utilization 0.50
```

中断后原命令末尾加`--resume`。检查：

```bash
(cd "$SUITE_SMOKE_ROOT" && sha256sum -c sha256.txt)
jq '{sample_count, profiles: (.profiles | with_entries(.value |= {
  gate,
  quality: (.quality | {
    hotword_exact_recall,
    sample_hotword_hit_rate,
    final_prompted_hotword_precision,
    wrong_injected_write_through_rate,
    negative_hotword_hallucination_rate,
    wer,
    cer,
    mean_inference_seconds
  }),
  compute: (.latency.compute | {
    sample_inference_seconds,
    sample_real_time_factor,
    retrieval_over_50ms_rate,
    chunk_metrics
  })
}))}' "$SUITE_SMOKE_ROOT/suite_summary.json"
```

smoke通过后改用新目录`streaming_gate_suite_4k_formal100_v1`并删除
`--max-samples 20`跑满100条。套件内部顺序运行`C,E`、conservative D、balanced D、
recall-first D；每个子目录都可独立resume，根目录最后生成`suite_summary.json`和
SHA256。这一轮是工程校准集诊断，不是封存test泛化结论。

## 0.51 2026-08-26 4k门控扫描结果与端到端诊断决策

离线formal100完整64候选门控扫描已完成，SHA256全部通过。检索性能稳定满足目标：
P50/P95/P99/max为26.73/34.69/38.37/44.44 ms，100条中没有查询超过50 ms。共扫描
3,168个门控组合，得到17个Recall/Precision/FPR Pareto点。

固定基线`0.86 / edit 0.35 / Top-5`为138/172，Recall 80.23%，Precision 55.42%，
负例FPR 25%。Recall-first推荐点为`threshold 0.75 / edit 0.35 / minimum posterior
0.5 / margin 0 / Top-7`：155/172，Recall 90.12%，Precision 26.59%，负例FPR 90%。
Top-10同门控可到158/172、Recall 91.86%，但Precision进一步降到22.60%，因此第一轮
端到端诊断优先Top-7，不把Top-10作为主推荐。

扫描没有任何点同时达到Recall 90%和Precision 85%。这说明单纯放宽门控能补回Recall，
但会带入大量错误候选；它是定位CTC门控与Qwen Prompt容错能力的诊断配置，不是可直接
上线的产品配置。建议端到端至少保留三条CTC路径：保守0.86/Top-5、平衡0.82/Top-7
（Recall 86.05%、Precision 42.65%、FPR 45%）和Recall-first 0.75/Top-7，再加Oracle。

本轮还修正了`best_by_top_k`在某个Top-K没有达到目标Recall时的回退排序：旧逻辑会继续
先最大化Precision，导致Top-5显示Recall仅22.09%的点；新逻辑在未达目标时先最大化
Recall，再比较Precision/FPR。总体Top-7推荐不受该报告问题影响。拉取修复后只需删除
并重跑很快的`GATE_SWEEP_ROOT`，不需要重跑`GATE_SOURCE_ROOT`、Qwen、CTC或Anchor。

```bash
git pull origin codex/g2p-coverage-scan

GATE_SWEEP_ROOT_V2="$CAP_ROOT/operating_gate_sweep_offline_4000_r2_v2"
test ! -e "$GATE_SWEEP_ROOT_V2"

python scripts/sweep_hotword_operating_points.py \
  --benchmark-dir "$GATE_SOURCE_ROOT" \
  --output-dir "$GATE_SWEEP_ROOT_V2" \
  --profile representative \
  --size 4000 \
  --window full_current \
  --shortlist-size 64 \
  --top-ks 5,7,10 \
  --thresholds 0,0.50,0.60,0.70,0.75,0.80,0.82,0.84,0.86,0.88,0.90 \
  --maximum-edit-ratios 0.35,0.40,0.45,0.50,0.60,1.0 \
  --minimum-posterior-confidences 0,0.25,0.50,0.75 \
  --minimum-top1-margins 0,0.01,0.02,0.05 \
  --selection-scope final \
  --target-recall 0.90 \
  --diagnostic-precision-target 0.85 \
  --deadline-ms 50
```

V2只修正未达标Top-K的展示与推荐，全部扫描点、总体Top-7推荐和性能数据应与V1一致。

## 0.50 2026-08-26 Step 5门控扫描与Recall-first端到端候选

4k/50 ms路线现处于Step 5后段。Anchor-guided shortlist 64已经满足性能目标，但固定
`threshold=0.86 / maximum_edit_ratio=0.35 / Operating Top-5`在离线formal100上只有
80.23% Operating recall。Operating上限扩大到7或10几乎不增加召回，说明主要损失来自
门控拒绝，而不是输出槽位不足。第一轮端到端RAG前，先做一次纯CPU门控重放；精确率
85%继续记录，但本轮不作为硬性阻塞条件。

新增：

- `benchmark_anchor_rerank_capacity.py --saved-ranked-matches`：可把完整shortlist的精排
  特征写入`query_results.jsonl`。默认仍保存20条以兼容旧行为；门控扫描必须保存64条，
  否则被拒绝候选之后可能由第21至64名补位，旧Top-20不能给出精确结果。
- `scripts/sweep_hotword_operating_points.py`：不重跑Qwen、CTC、Anchor或编辑距离，只对
  完整已排序shortlist精确重放threshold、maximum edit ratio、minimum posterior、
  minimum top-1 margin和Operating Top-K。
- 输出`source_baseline`、全部`sweep_results.jsonl`、Recall/Precision/FPR Pareto前沿和
  `recall_first`推荐点。推荐规则为：P95先不超过50 ms，Recall先达到90%，再最大化
  Precision、最小化负例FPR并偏好更小Top-K。85% Precision只输出可达性，不参与阻塞。
- Posterior weight本轮固定为0.25。它会改变候选排序，不能仅凭当前已排序结果安全重放；
  如门控扫描仍不够，再单独做重新精排的posterior-weight消融。

工作区拉取后先生成一份完整64候选的离线扫描源，使用新目录，不覆盖旧基线：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
GATE_SOURCE_ROOT="$CAP_ROOT/benchmark_anchor_guided_offline_4000_r2_gate_source_v1"
GATE_SWEEP_ROOT="$CAP_ROOT/operating_gate_sweep_offline_4000_r2_v1"

test ! -e "$GATE_SOURCE_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir "$GATE_SOURCE_ROOT" \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64 \
  --lookbacks full \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --gc-policy defer_during_retrieval_pass \
  --rerank-mode anchor_guided \
  --anchor-start-radius 2 \
  --saved-ranked-matches 64

test ! -e "$GATE_SWEEP_ROOT"
python scripts/sweep_hotword_operating_points.py \
  --benchmark-dir "$GATE_SOURCE_ROOT" \
  --output-dir "$GATE_SWEEP_ROOT" \
  --profile representative \
  --size 4000 \
  --window full_current \
  --shortlist-size 64 \
  --top-ks 5,7,10 \
  --thresholds 0,0.50,0.60,0.70,0.75,0.80,0.82,0.84,0.86,0.88,0.90 \
  --maximum-edit-ratios 0.35,0.40,0.45,0.50,0.60,1.0 \
  --minimum-posterior-confidences 0,0.25,0.50,0.75 \
  --minimum-top1-margins 0,0.01,0.02,0.05 \
  --selection-scope final \
  --target-recall 0.90 \
  --diagnostic-precision-target 0.85 \
  --deadline-ms 50
```

检查与回传：

```bash
(cd "$GATE_SOURCE_ROOT" && sha256sum -c sha256.txt)
(cd "$GATE_SWEEP_ROOT" && sha256sum -c sha256.txt)

cat "$GATE_SWEEP_ROOT/summary.json"
jq '{
  status,
  source_baseline: (.source_baseline | {config, final}),
  recall_first: (.recall_first | {config, final}),
  best_by_top_k: (.best_by_top_k | with_entries(
    .value |= {status, recall_first: (.recall_first | {config, final})}
  )),
  delta_from_source_baseline,
  strict_recall_and_precision_point_count
}' "$GATE_SWEEP_ROOT/recommended_config.json"

sed -n '1,20p' "$GATE_SWEEP_ROOT/pareto_frontier.jsonl"
```

拿到结果后冻结两个端到端配置：保守组继续用0.86/0.35/Top-5；Recall-first组使用扫描
推荐点。随后做离线/流式保守RAG、流式Recall-first RAG和Oracle对照。该扫描使用既有
formal100作为工程校准集，推荐点不是独立测试集上的最终产品阈值，不能把扫描后的同集
指标表述为泛化结论。

## 0.49 2026-08-26 Operating Top-5/7/10质量观察

4k Anchor引导精排已证明Top-64离线P95为34.68 ms、Top-128为45.84 ms，且相对
full-search没有质量损失。现有Raw Top-7/10 recall可观察正确词是否落在第6至10名，
但旧报告只对Top-5应用0.86/edit/posterior/margin门控，无法判断扩大Operating上限后
Recall和Precision的真实变化。

评测现同时输出Operating@5/@7/@10：三档复用完全相同的候选分数与门控，仅改变门控
通过后最多保留的观察数量。`--top-k 5`仍是实际运行和Prompt注入配置；Operating@7/10
只用于容量研究，不会改写线上Top-5基线。每档输出：

```text
operating_correct_at_{5,7,10}
operating_recall_at_{5,7,10}
operating_precision_at_{5,7,10}
operating_positive_case_hit_rate_at_{5,7,10}
negative_case_false_positive_rate_at_{5,7,10}
any_step_operating_*_at_{5,7,10}
```

旧`operating_ids`、`negative_case_false_positive_rate`和所有`*_at_5`字段保留兼容。
历史汇总器会读取新字段；旧输出只能填Top-5，Top-7/10保持`null`，不能由Raw结果推测。

工作区拉取后用新目录重跑现有guided benchmark：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
STREAM_REPLAY_ROOT="$CAP_ROOT/replay_streaming_posterior_smoke20_v2"
OFFLINE_OPK_ROOT="$CAP_ROOT/benchmark_anchor_guided_offline_4000_r2_operating_k_v1"
STREAM_OPK_ROOT="$CAP_ROOT/benchmark_anchor_guided_streaming_4000_r2_operating_k_v1"

COMMON_OPK_ARGS=(
  --assets-root "$ASSET_4K_ROOT"
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
  --profiles representative
  --sizes 4000
  --shortlist-sizes 64,128
  --ngram-sizes 2,3,4
  --anchors-per-entry 24
  --offset-tolerance 1
  --threshold 0.86
  --top-k 5
  --maximum-edit-ratio 0.35
  --posterior-weight 0.25
  --minimum-posterior-confidence 0
  --minimum-top1-margin 0
  --deadline-ms 50
  --gc-policy defer_during_retrieval_pass
  --rerank-mode anchor_guided
  --anchor-start-radius 2
)

test ! -e "$OFFLINE_OPK_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_OPK_ARGS[@]}" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full \
  --output-dir "$OFFLINE_OPK_ROOT"

test ! -e "$STREAM_OPK_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_OPK_ARGS[@]}" \
  --replay "$STREAM_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full,2,4,6 \
  --output-dir "$STREAM_OPK_ROOT"
```

返回以下短输出即可：

```bash
(cd "$OFFLINE_OPK_ROOT" && sha256sum -c sha256.txt)
(cd "$STREAM_OPK_ROOT" && sha256sum -c sha256.txt)

jq '.representative["4000"].full_current' \
  "$OFFLINE_OPK_ROOT/quality_summary.json"

jq '.representative["4000"]' \
  "$STREAM_OPK_ROOT/quality_summary.json"

cat "$OFFLINE_OPK_ROOT/performance_summary.json"
cat "$STREAM_OPK_ROOT/performance_summary.json"
```

这轮只补观察指标，不调阈值、edit ratio、posterior、Anchor、shortlist或Prompt。拿到
Operating@5/7/10后再画Recall/Precision变化，决定下一步是允许自适应上限7，还是先做
置信度校准。

## 0.48 2026-08-25 Step 4结果、流式指标修正与Anchor引导精排

4k Step 4的离线formal100和累计流式Posterior smoke20均已完成并通过SHA256。
本轮没有改0.86阈值、Operating Top-5、edit ratio 0.35、posterior weight 0.25、
margin、Prompt或CTC输出。结论是：Anchor候选阶段已经足够快，当前50 ms的主要瓶颈
明确转移到了shortlist上的全序列滑窗编辑距离精排。

离线formal100的full-current结果：

| shortlist | 候选Recall | Raw R@5/R@7/R@10 | Operating R@5 | Operating P@5 | 负例FPR | Anchor P95 | rerank P95 | 总P95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 95.35% | 86.63% / 90.70% / 93.02% | 80.23% | 55.42% | 25.0% | 30.88 ms | 90.68 ms | 123.45 ms |
| 128 | 97.67% | 86.05% / 90.12% / 93.60% | 80.23% | 55.42% | 25.0% | 24.45 ms | 133.09 ms | 157.87 ms |
| 256 | 97.67% | 86.05% / 90.12% / 92.44% | 80.23% | 55.42% | 25.0% | 33.80 ms | 413.03 ms | 446.07 ms |

扩大shortlist没有改善Operating结果；64甚至略高于128/256的Raw Top-5，同时总P95
最低。因此现阶段以64作为性能主线、128作为质量对照，不再运行256精排主线。
负例FPR 25%是现有门控在这100条数据上的结果，不是Anchor新增误报；本步骤不靠调整
阈值掩盖它，留给Step 5单独处理。

累计流式smoke20只有20个正例、60个expected hotword，没有负例，不能估计FPR。
其final step结果为：full-current下shortlist 64的Raw R@5/R@7/R@10为
83.33%/86.67%/90.00%，Operating R@5为78.33%，总P95为118.36 ms；recent-2s/
64的final Raw R@5/R@7/R@10为33.33%/35.00%/36.67%，Operating R@5为28.33%，
总P95为44.22 ms。recent-4s/64总P95为79.52 ms，recent-6s/64为68.48 ms。

recent窗口的final-only召回不能解释为真实流式漏检：热词可能在较早的因果chunk被检出，
随后滑出最近窗口，最终行自然不再包含它。评测现已新增每个case/hotword的
`any_step_*` Raw Top-5/7/10 recall/precision、Operating Top-5 recall/precision、
positive case hit与负例FPR；历史汇总器也会从旧`query_results.jsonl`直接恢复这些
指标，无需重跑旧full-search基线。

下一项等价方向是Anchor引导局部精排：保留完全相同的评分、排序和Operating门控，
但用Anchor的`best_offset`作为decoded-token起点估计，只搜索其前后2个token位置，
而不是让每个候选扫描完整累计CTC序列。该搜索域变化必须和Step 4 full-search逐项比较，
只有质量差异可接受且总检索P95小于50 ms才可进入主线。

工作区拉取后运行新目录；当前先比较shortlist 64/128、radius 2，不运行256：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
STREAM_REPLAY_ROOT="$CAP_ROOT/replay_streaming_posterior_smoke20_v2"
OFFLINE_FULL_ROOT="$CAP_ROOT/benchmark_anchor_rerank_offline_formal100_4000_v1"
STREAM_FULL_ROOT="$CAP_ROOT/benchmark_anchor_rerank_streaming_smoke20_4000_v1"
OFFLINE_GUIDED_ROOT="$CAP_ROOT/benchmark_anchor_guided_rerank_offline_formal100_4000_r2_v1"
STREAM_GUIDED_ROOT="$CAP_ROOT/benchmark_anchor_guided_rerank_streaming_smoke20_4000_r2_v1"
GUIDED_HISTORY_ROOT="$CAP_ROOT/optimization_history_4000_guided_r2_v1"

COMMON_GUIDED_ARGS=(
  --assets-root "$ASSET_4K_ROOT"
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
  --profiles representative
  --sizes 4000
  --shortlist-sizes 64,128
  --ngram-sizes 2,3,4
  --anchors-per-entry 24
  --offset-tolerance 1
  --threshold 0.86
  --top-k 5
  --maximum-edit-ratio 0.35
  --posterior-weight 0.25
  --minimum-posterior-confidence 0
  --minimum-top1-margin 0
  --deadline-ms 50
  --gc-policy defer_during_retrieval_pass
  --rerank-mode anchor_guided
  --anchor-start-radius 2
)

test ! -e "$OFFLINE_GUIDED_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_GUIDED_ARGS[@]}" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full \
  --output-dir "$OFFLINE_GUIDED_ROOT"

test ! -e "$STREAM_GUIDED_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_GUIDED_ARGS[@]}" \
  --replay "$STREAM_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full,2,4,6 \
  --output-dir "$STREAM_GUIDED_ROOT"
```

检查并生成可追溯对比：

```bash
(cd "$OFFLINE_GUIDED_ROOT" && sha256sum -c sha256.txt)
(cd "$STREAM_GUIDED_ROOT" && sha256sum -c sha256.txt)

cat "$OFFLINE_GUIDED_ROOT/quality_summary.json"
cat "$OFFLINE_GUIDED_ROOT/performance_summary.json"
cat "$STREAM_GUIDED_ROOT/quality_summary.json"
cat "$STREAM_GUIDED_ROOT/performance_summary.json"

test ! -e "$GUIDED_HISTORY_ROOT"
python scripts/summarize_hotword_capacity_history.py \
  --stage full_search_offline_v1="$OFFLINE_FULL_ROOT" \
  --stage anchor_guided_offline_r2_v1="$OFFLINE_GUIDED_ROOT" \
  --stage full_search_streaming_v1="$STREAM_FULL_ROOT" \
  --stage anchor_guided_streaming_r2_v1="$STREAM_GUIDED_ROOT" \
  --profiles representative \
  --sizes 4000 \
  --output-dir "$GUIDED_HISTORY_ROOT"

(cd "$GUIDED_HISTORY_ROOT" && sha256sum -c sha256.txt)
cat "$GUIDED_HISTORY_ROOT/optimization_history.tsv"
```

验收顺序固定：先比较candidate recall，再比较Raw Top-5/7/10和Operating Top-5，
流式recent窗口以`any_step_*`为主、final-only为稳定保留诊断，最后比较总检索P95/P99。
如果radius 2丢失明显，再测radius 4；若质量等价但P95仍超50 ms，才进一步做内存分配
和候选批处理优化，不调整阈值和Top-K。

## 0.47 2026-08-25 4k Anchor GC结论、Step 4精排实现与三语训练状态

4k/50 ms六步路线的Step 3现已完成。工作区GC A/B质量文件逐字节一致；默认GC组
Anchor P50/P95/P99/max为19.96/63.56/70.79/83.46 ms，7%查询超过50 ms。
7条慢查询全部发生generation 0/1/2集合，单次查询中GC耗时为43.07至53.30 ms，
而总查询耗时为59.54至83.46 ms。`defer_during_anchor_pass`组不改变任何候选质量，
P50/P95/P99/max降为14.68/19.52/21.31/22.33 ms，0%超过50 ms。由此可判定
Step 3算法本体已满足50 ms，先前P95失败由Python cyclic GC落入请求关键路径造成，
不是posting数或Anchor召回需要牺牲。生产方案不能在每个2秒请求后同步
`gc.collect()`；后续还需以持续流、RSS和显式安全点验证GC调度。

三语Macro CTC 5 epoch pilot同时已经完成，best=epoch 5：mixed validation
PER 7.417%，Macro PER 6.894%，en/es/pt分别5.652%/4.275%/10.755%，三语均较
epoch 1继续下降，葡语仍为最差组但没有停滞。用户已用完全相同参数启动同目录
`--epochs 30 --resume`继续训练；结果未返回前不改batch、lr、seed、Head、早停或
选择口径，test仍不读取。

Step 4已新增：

- `src/qwen_hotword/hotwords/anchor_rerank.py`：按当前累计CTC frame轴因果裁剪
  full/2/4/6秒窗口，只对Anchor Top-64/128/256运行现有近似音素评分器；
- `scripts/benchmark_anchor_rerank_capacity.py`：固定0.86、Operating Top-5、
  edit ratio 0.35、posterior weight 0.25，同时输出Raw Top-5/7/10；
- 输出分开记录Anchor、rerank和总检索P50/P90/P95/P99/max、候选召回、精排召回、
  precision、正例case hit、负例FPR与50 ms deadline miss；
- AC exact只作为候选证据记录，不绕过既有Operating门控；窗口只读取当前replay
  row，不保留TTL、不读取未来chunk，也不使用Oracle或离线结果。

拉取后先跑离线formal100的full-current，再跑已有累计流式Posterior smoke20的
full/2/4/6秒消融。两个输出目录都必须是新目录：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
STREAM_REPLAY_ROOT="$CAP_ROOT/replay_streaming_posterior_smoke20_v2"
OFFLINE_RERANK_ROOT="$CAP_ROOT/benchmark_anchor_rerank_offline_formal100_4000_v1"
STREAM_RERANK_ROOT="$CAP_ROOT/benchmark_anchor_rerank_streaming_smoke20_4000_v1"
RERANK_HISTORY_ROOT="$CAP_ROOT/optimization_history_4000_rerank_v1"

COMMON_RERANK_ARGS=(
  --assets-root "$ASSET_4K_ROOT"
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
  --profiles representative
  --sizes 4000
  --shortlist-sizes 64,128,256
  --ngram-sizes 2,3,4
  --anchors-per-entry 24
  --offset-tolerance 1
  --threshold 0.86
  --top-k 5
  --maximum-edit-ratio 0.35
  --posterior-weight 0.25
  --minimum-posterior-confidence 0
  --minimum-top1-margin 0
  --deadline-ms 50
  --gc-policy defer_during_retrieval_pass
)

test ! -e "$OFFLINE_RERANK_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_RERANK_ARGS[@]}" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full \
  --output-dir "$OFFLINE_RERANK_ROOT"

test ! -e "$STREAM_RERANK_ROOT"
python scripts/benchmark_anchor_rerank_capacity.py \
  "${COMMON_RERANK_ARGS[@]}" \
  --replay "$STREAM_REPLAY_ROOT/ctc_replay.jsonl" \
  --lookbacks full,2,4,6 \
  --output-dir "$STREAM_RERANK_ROOT"
```

完成后返回以下短输出；不要上传Posterior tensor：

```bash
(cd "$OFFLINE_RERANK_ROOT" && sha256sum -c sha256.txt)
(cd "$STREAM_RERANK_ROOT" && sha256sum -c sha256.txt)

cat "$OFFLINE_RERANK_ROOT/quality_summary.json"
cat "$OFFLINE_RERANK_ROOT/performance_summary.json"
cat "$STREAM_RERANK_ROOT/quality_summary.json"
cat "$STREAM_RERANK_ROOT/performance_summary.json"
cat "$STREAM_RERANK_ROOT/summary.json"

test ! -e "$RERANK_HISTORY_ROOT"
python scripts/summarize_hotword_capacity_history.py \
  --stage exact_ac_v1="$CAP_ROOT/benchmark_exact_ac_offline_formal100_4000_v1" \
  --stage anchor_v4_metrics="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v4_metrics" \
  --stage anchor_gc_deferred="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v5_gc_deferred" \
  --stage rerank_offline_v1="$OFFLINE_RERANK_ROOT" \
  --profiles representative \
  --sizes 4000 \
  --output-dir "$RERANK_HISTORY_ROOT"

(cd "$RERANK_HISTORY_ROOT" && sha256sum -c sha256.txt)
cat "$RERANK_HISTORY_ROOT/optimization_history.tsv"
```

历史汇总器会把rerank的每个`window × shortlist`独立成行，不能把12种变体混算。
本地容量定向pytest 10项、本轮Ruff和全量pytest均通过；严格Mypy对三个相关
source/script为0错误（全依赖递归仍会报告仓库已有的两个unused-ignore，与本轮
无关）。全仓Ruff仍有9个已有`UP038`，均位于本任务未修改文件。下一步根据上述
结果选择64/128/256中满足总检索P95<50 ms且质量最好的候选规模，然后进入Step 5
排序稳定性和误报控制；目前不提前调整阈值或Prompt。

## 0.46 2026-08-25 三语分组smoke结果与Macro PER正式Trainer

0.45的单遍分组诊断已在工作区完成，8,101条validation全部消费，
Manifest/cache ID和`en,es,pt`分组严格验证通过，test未使用。epoch 1
smoke checkpoint结果：

```text
mixed validation PER: 0.099946
Macro PER:            0.095021
English PER:          0.088879
Spanish PER:          0.056087
Portuguese PER:       0.140096
```

葡语占全部reference phonemes约44.03%，但贡献约61.72%错误。葡语的
22,881个错误中删除为11,722，预测/参考长度比为0.9412，当前是明显的
漏音素/输出偏短，而不是插入偏置。但这仅为1 epoch，不足以判定葡语数据或
G2P有结构性问题。CTC压力分桶PER为：最低压力8.75%、中压力15.82%、
高压力26.84%，因此正式pilot必须持续观察删除和葡语PER是否随epoch下降。

正式分片Trainer现支持：

- validation cache单遍同时统计mixed、en/es/pt、Macro和最差语种PER；
- 每组sample/error/reference合计必须与mixed指标严格一致；
- best checkpoint可按`Macro PER -> worst-group PER -> mixed validation loss`
  确定性选择；
- early stopping可使用`validation_macro_per`；
- `metrics.jsonl`每epoch写入分语种、Macro和最差语种；
- `report.json`写入initial/best/final分组指标、选择口径、分组SHA256及完整
  optimizer/scheduler/seed超参数；
- resume指纹绑定validation group mapping和checkpoint selection策略，换分组
  或换选择口径后不会误续训；
- 旧的无分组训练状态仍可按原参数resume。

下一步不直接跑30 epoch，而是在新目录从随机Head初始化跑5 epoch pilot。
batch/lr/Head与已通过的smoke保持为256 / 3e-4 / h512-k5-dropout0.1-
Temporal 2×；只将正式早停与checkpoint选择切到Macro口径。物理GPU 3在进程内
仍是逻辑`cuda:0`：

```bash
git pull origin codex/g2p-coverage-scan

GPU_ID=3
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1
FORMAL_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_formal_macro_v1

test ! -e "$FORMAL_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_full_ctc.py \
  --train-cache "$FEATURE_CACHE_ROOT/train" \
  --validation-cache "$FEATURE_CACHE_ROOT/validation" \
  --train-manifest "$BALANCED_TRAIN_ROOT/full_ctc_train.jsonl" \
  --validation-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir "$FORMAL_ROOT" \
  --device cuda:0 \
  --epochs 5 \
  --minimum-epochs 5 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 0.001 \
  --early-stopping-metric validation_macro_per \
  --checkpoint-selection-metric validation_macro_per \
  --validation-group-column balanced_language_bucket \
  --expected-validation-groups en,es,pt \
  --train-batch-size 256 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --max-gradient-norm 5 \
  --scheduler-patience 2 \
  --scheduler-factor 0.5 \
  --minimum-learning-rate 0.00001 \
  --seed 20260825 \
  --log-every-shards 25 \
  --head-type temporal_upsample \
  --head-hidden-dimension 512 \
  --head-kernel-size 5 \
  --head-dropout 0.1 \
  --head-time-upsampling-factor 2
```

跑完后先不加`--resume`，返回：

```bash
cat "$FORMAL_ROOT/report.json"

jq -c '{
  epoch,
  learning_rate,
  train_per: .train.phoneme_error_rate,
  validation_per: .validation.phoneme_error_rate,
  macro_per: .validation_macro_phoneme_error_rate,
  worst_group: .validation_worst_group,
  worst_group_per: .validation_worst_group_phoneme_error_rate,
  by_group: (.validation_by_group | with_entries(.value = .value.phoneme_error_rate)),
  epoch_seconds
}' "$FORMAL_ROOT/metrics.jsonl"
```

验收后若三语PER都继续下降，再在完全相同参数下只把`--epochs 5`改为
`--epochs 30`并增加`--resume`。不能改batch、lr、seed、early-stop、Head或分组参数，
否则resume指纹会正确拒绝。test仍不读取。

## 0.45 2026-08-25 三语Feature Cache完成、Temporal 2× smoke通过与分语种PER诊断

工作区的三语450小时Feature Cache已完成，根目录报告为
`status=pass`，且没有读取test集：

- train：310,257条、606个shard、21,106,335帧、43,344,916,678 bytes；
- validation：8,101条、16个shard、562,779帧、1,155,707,472 bytes；
- tap/dtype：`thinker.audio_tower.ln_post` / bfloat16；
- 来源train Manifest：
  `outputs/en_es_pt_balanced_150h_temporal2x_v2/full_ctc_train.jsonl`；
- 来源validation Manifest：
  `outputs/en_es_pt_balanced_validation_4h_v1/full_ctc_validation.jsonl`。

三语Temporal 2× CTC Head单epoch smoke亦已通过，输出为
`outputs/en_es_pt_balanced_150h_temporal2x_ctc_smoke1_v1`。结构为
hidden 512 / kernel 5 / dropout 0.1 / time factor 2 / 90 classes，可训参数
838,746。cache SHA256全部验证通过，1 epoch耗143.82秒：

```text
random initial validation loss/PER: 13.418263 / 1.437186
epoch 1 train loss/PER:             0.849622 / 0.207504
epoch 1 validation loss/PER:        0.395094 / 0.099946
```

train PER高于val PER不是当前泄漏证据：train统计累积了整个epoch内不断
更新的中间模型预测，validation只使用epoch末权重且dropout关闭。该smoke
只证明结构、cache、训练和checkpoint链路可用；还不应直接启动30 epoch正式训练，
先确认混合validation PER没有掩盖某个语种退化。

`scripts/diagnose_frozen_ctc.py`现可在同一次validation cache遍历中，按
Manifest的sample ID和`balanced_language_bucket`统计en/es/pt PER以及三语
Macro PER。它严格要求Manifest与cache ID完全一致，且分组恰好为
`en,es,pt`；缺失、多余或重复ID会直接失败。不重建ccache，不读取test。

工作区先运行：

```bash
git pull origin codex/g2p-coverage-scan

GPU_ID=3
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
SMOKE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_ctc_smoke1_v1

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/diagnose_frozen_ctc.py \
  --validation-cache "$FEATURE_CACHE_ROOT/validation" \
  --validation-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --checkpoint "$SMOKE_ROOT/ctc_head_best.pt" \
  --output "$SMOKE_ROOT/validation_multilingual_diagnostics.json" \
  --device cuda:0 \
  --batch-size 256 \
  --group-column balanced_language_bucket \
  --expected-groups en,es,pt

jq '{validation_samples, group_column, checkpoints: [.checkpoints[] | {
  checkpoint_path,
  validation_loss,
  validation_per: .validation.phoneme_error_rate,
  validation_macro_phoneme_error_rate,
  validation_by_group
}]}' "$SMOKE_ROOT/validation_multilingual_diagnostics.json"
```

该训练只训练小型CTC Head，不加载Qwen backbone，因此不会自然占满H200
80%显存。不应为了占显存人为分配无用tensor；拿到分语种结果后，再用新输出
目录做`batch_size=256/512/1024`的短基准，按samples/s、epoch耗时和实际peak GPU
memory选最高有效batch，不用80%显存占用率代替吞吐验收。

## 0.44 2026-08-24 4k Anchor GC尾延迟A/B（代码完成，待工作区实测）

4k Anchor v3/v4在严格两阶段计时后，P50/P90已稳定在20至30毫秒范围，
但P95仍约62毫秒，还有6%至7%查询超过50毫秒。本轮不修改任何质量逻辑，
只增加显式`normal`与`defer_during_anchor_pass` GC策略，用来判断Python cyclic
GC是否造成尾延迟。

新增记录包括：

- `gc_events.jsonl`：每次GC的generation、耗时、回收对象数和对应query；
- `gc_summary.json`：GC启用/恢复状态、query重叠率、超时query中是否发生GC、
  计时口径外的前/后显式GC耗时；
- `query_results.jsonl`和超时诊断行：每query的GC次数、耗时和generation；
- `summary.json`/`run_config.json`与优化历史：显式`gc_policy`。

defer策略只在连续Anchor查询计时阶段禁用自动GC；结束后先恢复调用者原
GC状态，再在50毫秒口径外执行显式`gc.collect()`。完整CPU参考扫描仍在
Anchor计时后运行。默认仍为normal，不改旧命令行为。Top-5/7/10、Top-64/
128/256候选、0.86门控、Prompt和所有参考评分均不变。

工作区拉取后，等Feature Cache结束且CPU/存储负载稳定，连续运行两组新目录：

```bash
git pull origin codex/g2p-coverage-scan

CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
GC_NORMAL_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v5_gc_normal"
GC_DEFER_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v5_gc_deferred"

test ! -e "$GC_NORMAL_ROOT"
test ! -e "$GC_DEFER_ROOT"

COMMON_ARGS=(
  --assets-root "$ASSET_4K_ROOT"
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl"
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json
  --profiles representative
  --sizes 4000
  --shortlist-sizes 64,128,256
  --ngram-sizes 2,3,4
  --anchors-per-entry 24
  --offset-tolerance 1
  --threshold 0.86
  --top-k 5
  --maximum-edit-ratio 0.35
  --posterior-weight 0.25
  --minimum-posterior-confidence 0
  --minimum-top1-margin 0
  --deadline-ms 50
)

python scripts/benchmark_anchor_hotword_capacity.py \
  "${COMMON_ARGS[@]}" \
  --gc-policy normal \
  --output-dir "$GC_NORMAL_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  "${COMMON_ARGS[@]}" \
  --gc-policy defer_during_anchor_pass \
  --output-dir "$GC_DEFER_ROOT"
```

完成后检查SHA和质量不变约束：

```bash
(cd "$GC_NORMAL_ROOT" && sha256sum -c sha256.txt)
(cd "$GC_DEFER_ROOT" && sha256sum -c sha256.txt)

cmp "$GC_NORMAL_ROOT/quality_summary.json" \
    "$GC_DEFER_ROOT/quality_summary.json"

cat "$GC_NORMAL_ROOT/performance_summary.json"
cat "$GC_NORMAL_ROOT/gc_summary.json"
cat "$GC_DEFER_ROOT/performance_summary.json"
cat "$GC_DEFER_ROOT/gc_summary.json"
```

`cmp`必须无输出并返回0。若normal的慢query与GC高度重叠，且defer将P95
稳定压到50毫秒内，才能将GC调度作为主因。若defer仍约62毫秒，下一步
应做CPU affinity/pyperf与分配峰值诊断，暂不做会降低候选召回的posting cap。

本地验证：全仓Ruff通过；GC/Anchor与历史定向pytest 7项通过；全量
pytest 201项通过；本轮3个source/script文件严格Mypy为0错误；CLI help和
`git diff --check`通过。全仓`src scripts` Mypy仍有8个旧文件的14项Torch/
第三方类型错误，本轮文件不在其中。真实4k A/B尚未在本地运行，test集未读取。

## 0.43 2026-08-24 三语Feature Cache Temporal 2×实际长度修复

三语450小时Feature Cache在train shard `000031`首次遇到
`slr61_es_row_22`时停止：真实Encoder base length为63，CTC minimum为66。该记录
不是坏音频或坏标签；它携带`ctc_time_upsampling_factor=2`，Temporal Head的有效长度
应为`63 * 2 = 126`，因此物理上满足`126 >= 66`。

根因是`extract_frozen_features()`仍沿用旧1×检查，只比较`input_length < minimum`；
`write_feature_shard()`、已有分片恢复校验和后续磁盘训练均已正确按
`input_length * ctc_time_upsampling_factor`检查。现在提取阶段统一调用
`validate_actual_encoder_ctc_length()`，失败信息同时记录base length、factor、effective
length和minimum。1×不可行样本仍严格拒绝，只有Manifest明确标记的2×样本按2×放行；
缓存内容仍保存原始base Encoder hidden states，Temporal上采样仍由训练Head完成。

已完成的train shard `000000`至`000030`不删除、不覆盖。失败发生在`000031`提取过程，
该分片尚未报告completed；原命令重跑会先逐个验证已完成分片的metadata、文件大小和
SHA256并显示`resumed`，随后从`000031`重新生成。不要删除输出目录、
`cache_index.json`或`.feature_cache.lock`；进程异常退出时flock已由context manager释放，
锁文件本身保留是正常现象。

工作区更新和只读检查：

```bash
git pull origin codex/g2p-coverage-scan

FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1

jq '{status, completed_shards, completed_samples, shard_count}' \
  "$FEATURE_CACHE_ROOT/train/cache_index.json"

find "$FEATURE_CACHE_ROOT/train/shards" -maxdepth 1 \
  -name 'shard-00003*' -type f -print | sort
```

然后使用**与首次运行完全相同**的model/config、train/validation Manifest、vocab、
`--samples-per-shard 512`和输出目录重跑。固定命令为：

```bash
GPU_ID=4
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/cache_full_training_features.py \
  --config configs/workzone.local.yaml \
  --train-manifest "$BALANCED_TRAIN_ROOT/full_ctc_train.jsonl" \
  --validation-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir "$FEATURE_CACHE_ROOT" \
  --encoder-batch-size 8 \
  --samples-per-shard 512
```

若首次使用的物理GPU不是4，只替换`GPU_ID`，其余参数不变。正常日志应先出现31条
`resumed train feature shard=...`，然后重新开始`cached train shard 000031`。完整任务
结束后再检查train/validation的`cache_summary.json`、根目录
`feature_cache_report.json`和全部分片SHA；当前仍不得读取test或开始CTC训练。

## 0.42 2026-08-24 4k Anchor v3结果与Top-5/7/10阶段台账

工作区`benchmark_anchor_offline_formal100_4000_v3`全部SHA256通过，严格两阶段计时
已生效，`timing_protocol=all_anchor_queries_before_full_scan_reference`。质量第三次与
v1/v2完全一致：Exact为132/172（76.74%），Anchor Top-64为164/172（95.35%），
Top-128/256均为168/172（97.67%），Top-256覆盖完整CPU Raw Top-5的99.2%，无
Anchor率为0，test未使用。

v3 Anchor查询P50/mean/P90/P95/P99/max约为18.86/21.17/30.07/62.73/73.32/
77.37 ms，6%查询超过50 ms。相对v1，P50下降约27.5%，P95下降约28.2%，deadline
miss从11%降到6%；但P95仍高于50 ms目标12.73 ms，因此尚未验收通过。P90仅30.07
ms且剩余6个慢查询不应在缺少证据时直接归因于posting数量；下一性能步骤先做显式
GC/调度尾延迟诊断，再决定是否做会改变质量的高DF posting cap。

为了逐步比较优化收益，容量评测现在统一记录：

- Raw Top-5/7/10 correct、recall、precision及正例case hit；
- Operating Top-5 correct、recall、precision及负例FPR；
- Top-7/10始终仅观察，不改变Operating Top-5或Prompt注入数量；
- `raw_precision_at_k`分母是所有final query实际返回的Raw候选数，包含负例；
- `operating_precision_at_5`分母是实际通过0.86等门控的候选数。

Anchor benchmark同时保存`anchor_top5/7/10_ids`和完整扫描参考的
`reference_raw_top5/7/10_ids`、`reference_operating_ids`。新增
`scripts/summarize_hotword_capacity_history.py`把全扫描、Exact AC和Anchor不同版本
归一化为`optimization_history.json/tsv`。旧结果缺少的指标写`null`，不得从Top-5
伪造Top-7/10。

下一轮先用相同算法生成仅增加指标的v4新目录；它不改变Anchor、阈值、Top-5或计时
协议：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
EXACT_4K_ROOT="$CAP_ROOT/benchmark_exact_ac_offline_formal100_4000_v1"
ANCHOR_4K_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v4_metrics"
HISTORY_ROOT="$CAP_ROOT/optimization_history_4000_v1"

test ! -e "$ANCHOR_4K_ROOT"
test ! -e "$HISTORY_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64,128,256 \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --output-dir "$ANCHOR_4K_ROOT"

(cd "$ANCHOR_4K_ROOT" && sha256sum -c sha256.txt)

python scripts/summarize_hotword_capacity_history.py \
  --stage exact_ac_v1="$EXACT_4K_ROOT" \
  --stage anchor_v1="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v1" \
  --stage anchor_v2="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v2" \
  --stage anchor_v3="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v3" \
  --stage anchor_v4_metrics="$ANCHOR_4K_ROOT" \
  --profiles representative \
  --sizes 4000 \
  --output-dir "$HISTORY_ROOT"

(cd "$HISTORY_ROOT" && sha256sum -c sha256.txt)
cat "$ANCHOR_4K_ROOT/quality_summary.json"
cat "$ANCHOR_4K_ROOT/performance_summary.json"
cat "$HISTORY_ROOT/optimization_history.json"
```

正式v4仍应避开Feature Cache或其他CPU高负载。v4的候选质量必须与v3一致；它的时延
只是同算法复测，不得被描述为新优化。拿到完整Top-5/7/10台账后，再进入显式
`normal`与`defer_during_anchor_pass` GC策略的尾延迟消融。

本地验证：全仓Ruff通过；容量定向pytest 15项通过；全量pytest 198项通过；本轮6个
source/script文件严格Mypy为0错误；CLI help与`git diff --check`通过。全仓
`src scripts`严格Mypy仍有9个既有文件的15项Torch/可选RapidFuzz类型错误，本轮文件
不在其中；包含tests的全仓严格Mypy还会暴露大量既有测试fixture类型问题，不纳入本轮
无关修复。

## 0.41 2026-08-24 4k Anchor v2诊断与两阶段计时v3

工作区`benchmark_anchor_offline_formal100_4000_v2`全部SHA256通过，质量与v1完全
一致：Top-64为164/172（95.35%），Top-128/256均为168/172（97.67%），Top-256
覆盖完整CPU Raw Top-5的99.2%，test未使用。4个expected miss明确为：

```text
sim_v3_hw_ptbr_0169  两个case
sim_v3_hw_ptbr_0436  一个case
sim_v3_hw_ptbr_0073  一个case
```

这4项均不在对应完整CPU Raw Top-5中；因此它们是更上游的声学/近似排序难例，并非
Top-256相对现有正式Top-5造成的新增损失。另4个Raw Top-5未覆盖项全部是非expected
capacity distractor，包括1个negative case；Top-256没有漏掉这些case里的正确热词。

v2 P50/P90/P95/P99/max约为24.07/39.23/82.21/90.51/119.21 ms，8%查询超过
50 ms，仍未通过。但慢查询与posting量不单调：2,686和3,615 posting的查询也约
72 ms，而6,000至7,700 posting的正常查询约20至26 ms。根因是benchmark v1/v2
在每个case内先执行3至8秒完整CPU参考扫描，再立即计时Anchor；参考扫描的GC、分配器
和调度抖动污染了下一次“纯Anchor”测量。

v3不改索引、候选或排序，唯一变化是严格两阶段：完成warmup和一次`gc.collect()`后，
先连续计时全部100次Anchor查询；所有Anchor计时结束后，再单独执行100次慢参考并
合并质量字段。`summary.json`和`run_config.json`显式记录：

```text
timing_protocol: all_anchor_queries_before_full_scan_reference
```

新输出不得覆盖v1/v2：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
ANCHOR_4K_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v3"

test ! -e "$ANCHOR_4K_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64,128,256 \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --output-dir "$ANCHOR_4K_ROOT"

(cd "$ANCHOR_4K_ROOT" && sha256sum -c sha256.txt)
cat "$ANCHOR_4K_ROOT/quality_summary.json"
cat "$ANCHOR_4K_ROOT/performance_summary.json"
cat "$ANCHOR_4K_ROOT/diagnostic_summary.json"
```

正式计时仍需避开Feature Cache或其他CPU高负载。只有v3隔离计时后P95仍超过50 ms，
才进入高DF posting cap等质量会变化的消融。

## 0.40 2026-08-24 4k Anchor v1结果与等价加速v2

工作区`benchmark_anchor_offline_formal100_4000_v1`全部SHA256通过，test未使用。
每个case仍严格是4,000个active热词；索引文件包含4,402个跨case唯一entry，查询时
按case active ID过滤。v1质量结果：

- AC exact为132/172，Recall 76.74%；
- Anchor Top-64为164/172，Recall 95.35%，正例case hit 97.50%；
- Top-128与Top-256均为168/172，Recall 97.67%，正例case hit 98.75%；
- 完整CPU Raw Top-5覆盖率在Top-64/128/256分别为91.6%/96.6%/99.2%；
- 100个查询均有Anchor候选，Top-64/128/256均打满。

质量方向成立，但v1未通过50 ms：Anchor查询P50/P90/P95/P99/max分别约为
26.02/79.23/87.36/94.83/95.76 ms，11%查询超过50 ms。全量CPU参考扫描P95约
6.49秒，但它只用于质量标签且没有混入Anchor时延。索引构建1.87秒，RSS增量约
126.4 MB；慢查询P95访问约9,317个posting。Top-256没有比Top-128额外找回4个
expected miss，因此不能靠继续扩大shortlist解决。

v1结果保留不覆盖。代码现做了保持候选排序语义的等价加速：offset聚合从“每个center
扫描全部offset”改为只查±1固定窗口，Top-256从全候选排序改为有界`heapq.nsmallest`，
并缓存active ID集合。新增`diagnostic_summary.json`与`diagnostic_cases.jsonl`，直接
列出Top-256的4个expected miss、Raw Top-5未覆盖项和超过50 ms的查询，不需要上传
完整大文件。本地4k合成探针P95由约2.20降至1.94 ms；仍只作实现检查。

正式v2必须使用新目录，且应等Feature Cache结束、CPU/存储负载稳定后计时：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
ANCHOR_4K_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v2"

test ! -e "$ANCHOR_4K_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64,128,256 \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --output-dir "$ANCHOR_4K_ROOT"

(cd "$ANCHOR_4K_ROOT" && sha256sum -c sha256.txt)
cat "$ANCHOR_4K_ROOT/quality_summary.json"
cat "$ANCHOR_4K_ROOT/performance_summary.json"
cat "$ANCHOR_4K_ROOT/diagnostic_summary.json"
cat "$ANCHOR_4K_ROOT/diagnostic_cases.jsonl"
```

v2首先要求候选质量与v1一致；若隔离负载后P95仍超过50 ms，再进入显式
high-document-frequency posting cap/Anchor稀有度消融，不能静默丢弃posting或改
0.86、Top-5及Prompt。

## 0.39 2026-08-24 三语Feature Cache运行中与4k Anchor候选器

工作区已按0.38固定命令启动三语Encoder Feature Cache：train输入为
`outputs/en_es_pt_balanced_150h_temporal2x_v2/full_ctc_train.jsonl`
（310,257条、450.003271小时），validation输入为
`outputs/en_es_pt_balanced_validation_4h_v1/full_ctc_validation.jsonl`
（8,101条、12.002467小时），输出为
`outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1`。该任务当前仅记录为
**运行中**，不能宣称完成；尚未开始CTC训练，也没有读取test。若中断，保留输出目录并
原样重跑0.38最后一条`cache_full_training_features.py`命令，由分片SHA256校验后续跑。

同时恢复0.25冻结的4,000词/50 ms任务，前两步不重跑。Step 3新增：

- `src/qwen_hotword/hotwords/anchor_index.py`：稀有度加权的音素2/3/4-gram位置
  Anchor倒排索引；每词默认最多24个Anchor，用±1位置偏移窗口聚合证据；
- `src/qwen_hotword/hotwords/anchor_capacity.py`：复用sealed CTC replay，对AC精确
  命中与Anchor候选取并集，生成确定且互相嵌套的Top-64/128/256 shortlist；
- `scripts/benchmark_anchor_hotword_capacity.py`：同时计算正确热词候选召回、完整
  CPU近似扫描Raw Top-5覆盖率、无Anchor率、候选数、索引构建内存及查询
  P50/P95/P99；
- `tests/test_anchor_capacity.py`：覆盖位置对齐、active-only、AC并集、嵌套确定性、
  慢参考不计入50 ms等约束。

本地验证：Anchor与容量定向pytest共13项通过，全量pytest共196项通过，全仓Ruff、
本轮严格Mypy、CLI help和`git diff --check`通过。全仓严格Mypy仍为6个既有文件的
11项错误，本轮两个新source模块为0项。合成4,000词/100-token实现级探针的查询
P95约2.2 ms；该数字只证明实现没有明显复杂度回退，不能替代下述工作区正式资产
结果。

该阶段不改变Operating Top-5、threshold 0.86、maximum edit ratio 0.35、Prompt、
TTL或候选保留策略。50 ms只计`anchor_index.query()`，全量近似评分仅作为质量参考，
单独写入`full_scan_reference_seconds`，不得混入口径。工作区首次只跑4k正式档：

```bash
CAP_ROOT=outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
ASSET_4K_ROOT="$CAP_ROOT/assets_with_4000_v2"
OFFLINE_REPLAY_ROOT="$CAP_ROOT/replay_offline_formal100_v1"
ANCHOR_4K_ROOT="$CAP_ROOT/benchmark_anchor_offline_formal100_4000_v1"

test ! -e "$ANCHOR_4K_ROOT"

python scripts/benchmark_anchor_hotword_capacity.py \
  --assets-root "$ASSET_4K_ROOT" \
  --replay "$OFFLINE_REPLAY_ROOT/ctc_replay.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --profiles representative \
  --sizes 4000 \
  --shortlist-sizes 64,128,256 \
  --ngram-sizes 2,3,4 \
  --anchors-per-entry 24 \
  --offset-tolerance 1 \
  --threshold 0.86 \
  --top-k 5 \
  --maximum-edit-ratio 0.35 \
  --posterior-weight 0.25 \
  --minimum-posterior-confidence 0 \
  --minimum-top1-margin 0 \
  --deadline-ms 50 \
  --output-dir "$ANCHOR_4K_ROOT"

(cd "$ANCHOR_4K_ROOT" && sha256sum -c sha256.txt)
cat "$ANCHOR_4K_ROOT/quality_summary.json"
cat "$ANCHOR_4K_ROOT/performance_summary.json"
cat "$ANCHOR_4K_ROOT/summary.json"
```

全量CPU参考扫描可能耗时数分钟，但不需要GPU。为了得到可信的50 ms CPU时延，避免
在同一主机CPU/存储正被Feature Cache高负载占用时做正式计时；可先让Feature Cache
完成，或在隔离CPU资源上运行。收到结果后先看Top-64/128/256的expected recall与
reference Raw Top-5覆盖率，再进入Step 4的小候选集近似精排；不能因为首次结果不佳
就调整0.86或扩大Prompt注入Top-K。

## 0.38 2026-08-24 三语4小时validation通过，待单H200 Feature Cache

工作区的`outputs/en_es_pt_balanced_validation_4h_v1`已通过全部SHA256
校验。合计8,101条、12.002467小时：

- 英语：2,808条、4.000688小时；
- 西语：2,631条、4.000866小时；
- 葡语：2,662条、4.000913小时。

三语时长差约0.81秒；与310,257条平衡train的ID/音频重叠均为0，
validation内部重复为0。每条记录具有`experiment=full-ctc-v1`、
`split=validation`和独立的validation dataset version。test未打开、未使用。

四个validation Manifest SHA256为：

```text
combined: d43d143f12cc4bbc9273476540640c43e1e46d095ce1e1893a6667ce4b044499
en:       1b1bbd87851010cd2dc821d186310a3c10b956ebfd3e6470b54f43942a58d27f
es:       c00b492472bcb1c0cce541fa2893b40be476c580e75f577c68c4fc2e34fa1273
pt:       06a467b418050ab100f9d8db1620c64bce8eef071397ca34f05fd7880d9a4dbf
```

下一阶段固定使用合并train和合并validation构建新的冻结Encoder Feature
Cache，不读取任何test。预计train/validation分别为606/16个512-sample
shard，按既有bfloat16 ln_post格式约需31 GiB最终存储；启动前建议确保
输出文件系统至少有45至50 GiB可用空间。运行使用单张H200，选中的
物理GPU通过`CUDA_VISIBLE_DEVICES`暴露为逻辑`cuda:0`。

工作区命令：

```bash
GPU_ID=4
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1
FEATURE_CACHE_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_feature_cache_v1

test -f configs/workzone.local.yaml
test ! -e "$FEATURE_CACHE_ROOT"
nvidia-smi -i "$GPU_ID"
df -h outputs

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/doctor.py \
  --config configs/workzone.local.yaml \
  --output outputs/en_es_pt_balanced_feature_cache_environment_v1.json

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/cache_full_training_features.py \
  --config configs/workzone.local.yaml \
  --train-manifest "$BALANCED_TRAIN_ROOT/full_ctc_train.jsonl" \
  --validation-manifest "$BALANCED_VALIDATION_ROOT/full_ctc_validation.jsonl" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir "$FEATURE_CACHE_ROOT" \
  --encoder-batch-size 8 \
  --samples-per-shard 512
```

若中断，保留目录并原样重跑最后一条命令；不再执行
`test ! -e "$FEATURE_CACHE_ROOT"`。构建器会逐分片校验已完成文件并续跑，损坏分片
会拒绝复用，不得跳过SHA256校验。完成后返回：

```bash
cat "$FEATURE_CACHE_ROOT/feature_cache_report.json"
cat "$FEATURE_CACHE_ROOT/train/cache_summary.json"
cat "$FEATURE_CACHE_ROOT/validation/cache_summary.json"
```

## 0.37 2026-08-24 三语150小时v2通过与4小时平衡validation入口

工作区的`outputs/en_es_pt_balanced_150h_temporal2x_v2`已通过全部SHA256
校验。v2与v1的选中结果完全一致：英/西/葡分别为106,440/
103,318/100,499条和150.001423/150.001511/150.000337小时，合计
310,257条、450.003271小时。重复ID/音频均为0，test未读取。

v2记录契约已生效：`experiment=full-ctc-v1`、
`dataset_version=en-es-pt-temporal2x-balanced-v2`，并保留
`source_dataset_version`和`balanced_language_bucket`。该train现可作为后续
Feature Cache的固定输入。四个Manifest SHA256为：

```text
combined: def68274fe9af3497d46c38161de40f772dcc1d0c23811a9f4da3f59de0df7e2
en:       1fb8b494a591cbcfd138efdc678f108de850e04eb99518d7fdc766a19a4e519b
es:       dca9ac1d30c4f4a9fde96a74a6a562c6bcf17bbdf045e2a9f95843ef2eeb214b
pt:       f35bd38aa4f60610383ecc676f2f6dafedd6a1d3cc3177a9c7d7c3d119f00e2e
```

三语来源validation分别约为10.165238/4.368367/16.239131小时。为保持
对称且不迫近西语上限，首个validation固定为每语言4小时，合计约12小时。
新增：

- `scripts/build_balanced_multilingual_validation.py`
- `build_balanced_multilingual_validation()` in
  `src/qwen_hotword/training/balanced_multilingual.py`

构建器显式读取三个validation Manifest并按稳定SHA256优先级选到至少4小时；
只从summary记录test身份，不打开test内容。它会校验来源validation的SHA/条数/
时长，扫描v2 train的ID和音频身份，并拒绝任何train-validation重叠。
输出同样补齐`full-ctc-v1`契约，可直接作为Feature Cache的validation输入。

工作区命令：

```bash
BALANCED_TRAIN_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2
BALANCED_VALIDATION_ROOT=outputs/en_es_pt_balanced_validation_4h_v1

test ! -e "$BALANCED_VALIDATION_ROOT"

python scripts/build_balanced_multilingual_validation.py \
  --language-pool en=outputs/en_us_swift_temporal2x_v1 \
  --language-pool es=outputs/es_combined_temporal2x_v1 \
  --language-pool pt=outputs/pt_combined_temporal2x_v1 \
  --training-root "$BALANCED_TRAIN_ROOT" \
  --target-hours 4 \
  --seed 20260824 \
  --output-dir "$BALANCED_VALIDATION_ROOT"

(cd "$BALANCED_VALIDATION_ROOT" && sha256sum -c sha256.txt)
cat "$BALANCED_VALIDATION_ROOT/selection_summary.json"
cat "$BALANCED_VALIDATION_ROOT/selection_config.json"
wc -l "$BALANCED_VALIDATION_ROOT"/full_ctc_validation*.jsonl
```

验收标准：每语言至少4小时且超出量不大于该语言最长选中clip；跨train
ID/音频重叠、validation内重复均为0；`test_set_used=false`、
`test_set_content_read=false`。返回SHA、summary/config和行数后，再启动单H200
Encoder Feature Cache。

本地验证：相关4个代码/测试文件Ruff与严格Mypy通过，定向pytest 3项
通过，全量pytest为193 passed，全仓Ruff与`git diff --check`通过。全仓
严格Mypy仍为23个既有文件的113项错误，本轮文件为0项。

## 0.36 2026-08-24 三语150小时v1选样通过与v2 Manifest契约修正

工作区已成功生成`outputs/en_es_pt_balanced_150h_temporal2x_v1`并通过
全部SHA256校验：英/西/葡分别为150.001423/150.001511/150.000337小时，
合计310,257条、450.003271小时，三语时长差约4.23秒。重复ID和音频均为0，
西语`slr61`和`common_voice_rioplatense_v26`全量train均已保留，封存
validation/test没有打开或用于选样。因此v1的选样ID、时长和来源组成
已冻结为正确审计基线，不得删除或覆盖。

进入Encoder feature cache前发现一个Manifest契约问题：
`cache_full_training_features.py`通过`load_experiment_records`强制要求每条
`experiment=full-ctc-v1`。葡语合并池原记录已有此字段，但英语和西语合并池
原记录没有。直接缓存v1会在加载英/西记录时失败。

修正版只对派生输出记录做契约标准化，不改变稳定哈希优先级、强制来源、
选中ID、条数或时长：

```text
experiment: full-ctc-v1
dataset_version: en-es-pt-temporal2x-balanced-v2
source_dataset_version: 保留输入记录的dataset_version
balanced_language_bucket: en / es / pt
```

新输出目录必须使用`outputs/en_es_pt_balanced_150h_temporal2x_v2`，不得覆盖v1：

```bash
BALANCED_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v2

test ! -e "$BALANCED_ROOT"

python scripts/build_balanced_multilingual_training.py \
  --language-pool en=outputs/en_us_swift_temporal2x_v1 \
  --language-pool es=outputs/es_combined_temporal2x_v1 \
  --language-pool pt=outputs/pt_combined_temporal2x_v1 \
  --include-all-source es=slr61 \
  --include-all-source es=common_voice_rioplatense_v26 \
  --target-hours 150 \
  --seed 20260824 \
  --output-dir "$BALANCED_ROOT"

(cd "$BALANCED_ROOT" && sha256sum -c sha256.txt)
cat "$BALANCED_ROOT/selection_summary.json"
cat "$BALANCED_ROOT/selection_config.json"
wc -l "$BALANCED_ROOT"/full_ctc_train*.jsonl
```

验收时v2的三语选中条数、时长、来源分布必须与v1一致，只允许文件
SHA256因增加上述元数据而改变。返回SHA校验和新summary/config后，再构建三语
训练validation与Encoder feature cache；不对v1进行缓存。

本地验证：相关文件Ruff与严格Mypy通过，定向pytest通过，全量pytest为
192 passed，全仓Ruff通过，`git diff --check`通过。全仓严格Mypy仍有
23个既有文件中的113项错误，本轮3个代码/测试文件为0项。

## 0.35 2026-08-24 英西葡Temporal 2× 150小时1:1:1派生入口

US-only英语池已在工作区完成且全部SHA256校验通过：360,093条、507.959291小时；
train/validation/test为345,820/7,187/7,086条和487.628442/10.165238/
10.165611小时。US-only范围内原ready 360,067条，释放Temporal 2× 26条；4条其他
问题继续隔离。AU/CN合计29,641条、41.663130小时被正确排除，其中包含21条原本
可由Temporal 2×恢复的非US记录。最终1,622名US speaker按1,560/31/31分配，
speaker/audio/ID跨split overlap均为0，test封存且未用于选择。

三个完整train池现均满足派生条件：英语487.628442小时、西语181.868358小时、
葡语783.223637小时，全部绑定Temporal 2×。新增只读派生器：

- `src/qwen_hotword/training/balanced_multilingual.py`
- `scripts/build_balanced_multilingual_training.py`
- `tests/test_balanced_multilingual.py`

派生器只打开三个`full_ctc_train.jsonl`和对应`split_summary.json`；validation/test
只从summary记录路径、SHA256、条数和小时，不打开内容。它验证输入train SHA256、
summary计数/小时、语言标签和所有记录的`ctc_time_upsampling_factor=2`。每种语言
按稳定SHA256优先级选到至少150小时；西语强制完整保留`slr61`与
`common_voice_rioplatense_v26`，再从明确拉美辅助池补足。英语和葡语在各自完整
train自然来源分布上稳定抽样。每种语言最多只超出一条录音时长。

输出包含三个单语种派生train和一个合并train。合并时每次从累计已输出音频时长
最少的语言取下一条，避免Manifest/feature shard形成长语言块；普通训练shuffle后
每epoch仍使用每种语言约150小时，总计约450小时。完整单语种池不修改，独立
validation/test不合并、不重分、不回流。

工作区运行：

```bash
BALANCED_ROOT=outputs/en_es_pt_balanced_150h_temporal2x_v1

test ! -e "$BALANCED_ROOT"

python scripts/build_balanced_multilingual_training.py \
  --language-pool en=outputs/en_us_swift_temporal2x_v1 \
  --language-pool es=outputs/es_combined_temporal2x_v1 \
  --language-pool pt=outputs/pt_combined_temporal2x_v1 \
  --include-all-source es=slr61 \
  --include-all-source es=common_voice_rioplatense_v26 \
  --target-hours 150 \
  --seed 20260824 \
  --output-dir "$BALANCED_ROOT"

(cd "$BALANCED_ROOT" && sha256sum -c sha256.txt)
cat "$BALANCED_ROOT/selection_summary.json"
wc -l \
  "$BALANCED_ROOT/full_ctc_train_en.jsonl" \
  "$BALANCED_ROOT/full_ctc_train_es.jsonl" \
  "$BALANCED_ROOT/full_ctc_train_pt.jsonl" \
  "$BALANCED_ROOT/full_ctc_train.jsonl"
```

通过标准：每种语言selected hours均不小于150且overshoot不超过该语言最大选中
录音；西语mandatory source均完整；合并约450小时；重复ID/音频均为0；
`test_set_used=false`、`test_set_content_read=false`。先返回SHA校验、
`selection_summary.json`和`wc -l`，通过后才为新train构建Encoder feature cache。

本地验证：相关7个文件Ruff和严格Mypy通过，定向pytest 7项通过，全量pytest为
170 passed、22 skipped，CLI help与`git diff --check`通过。仓库全量Ruff仍为既有
9个UP038，仓库全量严格Mypy仍为既有17个文件102项，本轮文件未新增错误。

## 0.34 2026-08-24 US-only英语Temporal 2×合并入口

Swift英语speaker审计已在工作区通过：389,738条全部与完整Manifest一一关联，
解析失败、Manifest缺失/额外、重复speaker+utterance均为0；候选speaker 1,800名，
每个speaker最多约0.868178小时。speaker前缀固定为4段，第三段全部为F/M，末尾
utterance段全部为数字，因此`WAV stem去掉末段`可用于speaker整体切分。391名
speaker跨多个shard，不能以shard作为切分边界。

审计同时发现输入并非纯美式：`US=360,043`、`us=54`，另有`AU=13,640`、
`CN=16,001`。用户已确认本轮只使用`US/us`；过滤不区分大小写，AU/CN保留在原始
TSV和完整Manifest中但不进入新的英语池，也不能继续把包含它们的全集称为纯
`en-US`。

为避免复制西语的split/Temporal规则，现将其底层扩展为通用speaker构建器，同时
保留原西语入口兼容性。新增英语专用封装：

- `src/qwen_hotword/training/english_combined.py`
- `scripts/build_english_us_temporal2x_training.py`
- `tests/test_english_combined.py`

并扩展`src/qwen_hotword/training/spanish_combined.py`：支持数据集版本、预期语言和
可选speaker首段过滤；西语默认参数与原结果不变。英语入口强制speaker审计pass、
四类结构错误为0、审计/库存记录一致和语言严格为`en-US`。它保留原ready，按既有
严格策略释放US-only范围内可恢复的Temporal 2×记录，排除其他review及AU/CN，
然后按完整speaker确定性建立96/2/2。输出再次验证speaker/audio/ID跨split overlap
均为0，test封存且不得用于调参。

工作区运行：

```bash
EN_MANIFEST=outputs/en_us_swift_full_manifest_v1
EN_SPEAKER_AUDIT=outputs/en_us_swift_speaker_inventory_v1
EN_COMBINED=outputs/en_us_swift_temporal2x_v1

test ! -e "$EN_COMBINED"

python scripts/build_english_us_temporal2x_training.py \
  --manifest-dir "$EN_MANIFEST" \
  --speaker-inventory "$EN_SPEAKER_AUDIT/speaker_inventory.tsv" \
  --speaker-audit-summary "$EN_SPEAKER_AUDIT/summary.json" \
  --output-dir "$EN_COMBINED" \
  --allowed-speaker-first-components US \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --speaker-split-seed 20260824

(cd "$EN_COMBINED" && sha256sum -c sha256.txt)
cat "$EN_COMBINED/split_summary.json"
wc -l \
  "$EN_COMBINED/full_ctc_train.jsonl" \
  "$EN_COMBINED/full_ctc_validation.jsonl" \
  "$EN_COMBINED/full_ctc_test.jsonl"
```

返回`split_summary.json`和SHA校验结果。通过后再从英语train确定性派生约150小时；
此时仍不读取英语test内容，也不把英西葡完整池直接拼接。

工作区实际构建已通过：US-only 360,093条、507.959291小时；train为345,820条、
487.628442小时，validation/test分别为7,187/7,086条和10.165238/10.165611小时；
26条US Temporal 2×记录释放，AU/CN 29,641条全部排除，4条其他问题继续隔离。
speaker/audio/ID跨split overlap均为0，test封存且未使用。

本地验证：相关文件Ruff与严格Mypy通过，英语/西语合并定向pytest 5项通过，
全量pytest通过，`git diff --check`通过。仓库全量Ruff仍为既有9个UP038，仓库
全量严格Mypy仍为既有17个文件102项；本次文件未新增错误，也未顺带修改无关问题。

## 0.33 2026-08-24 英语Temporal 2×审计与speaker库存入口

上一阶段的西语合并已在工作区完成且全部SHA256校验通过：最终131,064条、
190.375791小时；train/validation/test分别为125,379/2,871/2,814条和
181.868358/4.368367/4.139065小时。Temporal 2×释放7,740条；跨split的speaker、
audio和ID overlap均为0，Common Voice显式split保留125,329条，SLR61按speaker
分配为train/validation/test 42/1/1名。test已封存且未用于选择。

美式英语完整Manifest的Temporal 2×只读审计和SHA256校验已在工作区通过。原ready
为389,687条、549.584746小时；唯一问题为`ctc_length_infeasible`的47条、
0.037675小时均可在2×且有效ratio不超过0.90时释放。另4条、0.007907小时由其他
标签问题阻塞；没有高压力或2×后仍不可行记录。英语完整候选约549.622421小时，
容量足以支持最终150小时train份额。

英语不能在确认speaker边界前直接按文件哈希切分。现有样例WAV basename如
`US_101_F_8297_1051478.wav`明显含重复说话人前缀，但字段语义尚未被全量证明。
新增只读库存工具：

- `src/qwen_hotword/training/english_speaker_inventory.py`
- `scripts/audit_swift_english_speakers.py`
- `tests/test_english_speaker_inventory.py`

工具将WAV stem最后一个下划线段视为utterance ID，前缀作为候选speaker ID；全量
关联`source.tsv`与ready/review Manifest，统计前缀字段数、speaker数量、每speaker
clip/小时分布、跨shard speaker、数字utterance后缀、解析失败和重复speaker+utterance。
它不读取音频、不修改源TSV或Manifest。输出`speaker_inventory.tsv`包含
`audio/speaker_id/source_split=unsplit/duration_seconds`，只有summary status pass、
解析/关联/重复均为0且字段分布合理后，才可作为英语speaker-disjoint切分输入。

本地验证：新增文件Ruff和严格Mypy通过，定向pytest通过；全量pytest为
166 passed、22 skipped，`git diff --check`通过。仓库全量Ruff/Mypy仍只包含此前
无关文件中的既有问题，本次未扩大范围处理。

工作区拉取后运行：

```bash
EN_SOURCE=outputs/en_external_train_sources_v1/swift_us_english/source.tsv
EN_MANIFEST=outputs/en_us_swift_full_manifest_v1
EN_SPEAKER_AUDIT=outputs/en_us_swift_speaker_inventory_v1

test ! -e "$EN_SPEAKER_AUDIT"

python scripts/audit_swift_english_speakers.py \
  --source-tsv "$EN_SOURCE" \
  --manifest-dir "$EN_MANIFEST" \
  --output-dir "$EN_SPEAKER_AUDIT"

(cd "$EN_SPEAKER_AUDIT" && sha256sum -c sha256.txt)
cat "$EN_SPEAKER_AUDIT/summary.json"
sed -n '1,41p' "$EN_SPEAKER_AUDIT/parse_failures.tsv"
```

先返回summary和parse failures。通过后再建立英语speaker保留式Temporal 2×池；
当前不裁剪英语、西语或葡语完整池，也不提前生成三语训练Manifest。

## 0.32 2026-08-24 西语辅助Manifest与split保留式Temporal 2×合并入口

明确拉美Common Voice辅助池完整Manifest已在工作区完成：118,177条、
171.594529小时全部进入ready/review分区；ready为107,694条、157.550974小时，
review为10,483条、14.043555小时。按既有speaker-disjoint split统计，ready为：

- train：102,938条、150.534372小时；
- validation：2,406条、3.599035小时；
- test：2,350条、3.417567小时。

主要review原因是`ctc_length_infeasible=7,588`，其次为
`dictionary_missing=3,148`和`standalone_h=35`；issue计数可重叠。三套西语
Temporal 2×只读审计全部SHA256通过，按有效ratio `<=0.90`可释放7,740条、
9.465169小时；其中SLR61 117条/0.159028小时、Rioplatense 125条/0.177887小时、
拉美辅助池7,498条/9.128254小时。15条/0.009880小时高压力记录继续推迟，其他
issue记录不释放。三套原ready加推荐恢复合计约190.375790小时。

现有葡语`build_temporal2x_combined_training.py`会按`split_hash`重新分配所有记录，
不能用于西语，否则会破坏Common Voice既有官方/speaker-disjoint split。新增：

- `src/qwen_hotword/training/spanish_combined.py`
- `scripts/build_spanish_temporal2x_training.py`
- `tests/test_spanish_combined.py`

新构建器用绝对音频路径把Manifest与每套`source.tsv`一一关联；两套Common Voice
严格保留`source_split`；SLR61的`unsplit`记录按完整speaker确定性分配为96/2/2，
同一speaker不能跨split。跨语料speaker、audio或ID泄漏均直接失败。只纳入原ready
和唯一issue为`ctc_length_infeasible`、Temporal 2×有效ratio不超过0.90的记录。
测试输出只做机械封存，不用于选择、调参或训练。

工作区拉取后运行：

```bash
ES_AR_MANIFEST_ROOT=outputs/es_ar_full_manifests_v1
ES_AR_SOURCE_ROOT=outputs/es_ar_train_sources_v1
ES_AUX_MANIFEST_ROOT=outputs/es_latam_cv_auxiliary_170h_full_manifest_v1
ES_AUX_SOURCE_ROOT=outputs/es_latam_cv_auxiliary_170h_v1
ES_COMBINED_ROOT=outputs/es_combined_temporal2x_v1

test ! -e "$ES_COMBINED_ROOT"

python scripts/build_spanish_temporal2x_training.py \
  --corpus slr61="$ES_AR_MANIFEST_ROOT/slr61" \
  --corpus common_voice_rioplatense_v26="$ES_AR_MANIFEST_ROOT/common_voice_rioplatense_v26" \
  --corpus common_voice_latam_auxiliary="$ES_AUX_MANIFEST_ROOT" \
  --source-tsv slr61="$ES_AR_SOURCE_ROOT/slr61/source.tsv" \
  --source-tsv common_voice_rioplatense_v26="$ES_AR_SOURCE_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --source-tsv common_voice_latam_auxiliary="$ES_AUX_SOURCE_ROOT/source.tsv" \
  --output-dir "$ES_COMBINED_ROOT" \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --speaker-split-seed 20260824

(cd "$ES_COMBINED_ROOT" && sha256sum -c sha256.txt)
cat "$ES_COMBINED_ROOT/split_summary.json"
wc -l "$ES_COMBINED_ROOT"/full_ctc_*.jsonl
```

确认train大于150小时、三类cross-split overlap均为0、显式split保持不变且
`recovered_records=7,740`后，再从完整西语train池确定性派生150小时。三语最终
`1:1:1`仍按train音频小时/有效曝光定义；英语和葡语完整池不被截断或覆盖，
validation/test保持各语言独立封存。

本地验证：新增3个strict Mypy文件零错误；新增文件Ruff通过；西语合并及相关
selection/source/MFA/full-manifest定向pytest 22项通过；全量pytest 164项通过、
22项按依赖条件跳过；`git diff --check`通过。全仓Ruff仍有9个既有UP038，位于本轮
未修改的multi-nested/prompt/retrieved-RAG/temporal-recovery文件；全仓strict Mypy
仍有102个既有错误，分布于17个本轮未修改文件，本轮3个新增文件为0。

## 0.31 2026-08-24 英西葡1:1:1最终训练集约束与西语基线字典审计

用户重申最终目标是供后续训练的英语/西语/葡语`1:1:1`组合训练集。
在没有新口径前，`1:1:1`按**train音频小时/有效训练曝光**定义，不按样本条数；
三种语言clip平均时长不同，按条数1:1:1会破坏实际声学权重。第一版计划从三个
已封存的单语种池各派生约150小时train，总计约450小时。

英语约549.584746小时ready池和葡语Temporal 2×约815.69小时完整池仍全部保留；
平衡集是不覆盖原产物的新派生Manifest。`1:1:1`只作用于train；各语言validation/test
保持独立、speaker/audio防泄漏并继续封存，不为凑比例重分已封存test。当前仍先
完成西语单语种处理，不提前构建三语合并Manifest。

选中的西语171.594529小时辅助池已提取词表：118,177条文本、
1,176,517个word token、66,304个unique word，数字fragment为0。用早先的全量
Common Voice Spanish MFA字典做定向基线审计后：

- dictionary word coverage：80.734194%；missing words：12,774；
- corpus token coverage：88.758173%；
- words with OOV phones：9,010；OOV phone units：9,278；weighted units：49,475；
- `training_labels_ready=false`。

`oov_phone_counts.tsv`中的OOV单元显示为空白，与之前西语MFA审计一致，对应
`U+0303 COMBINING TILDE`。高频缺词主要是`más/está/además/años/nació`等带
西语重音或`ñ`的正常词，不应删除文本或改成无重音拼写。下一步复用
`prepare_spanish_mfa_repairs.py`只为该选中词表生成增量代理计划，再决定一次
增量MFA规模；不重跑全量Common Voice G2P。

## 0.30 2026-08-24 西语明确拉美CV辅助池选样通过

0.29修正后的170小时辅助池选样在工作区通过，全部SHA256校验通过：

- 输出：`outputs/es_latam_cv_auxiliary_170h_v1`；
- 118,177条、171.594529小时、1,523个speaker；
- train：113,108条、164.102775小时、1,483个speaker；
- validation：2,540条、3.819581小时、26个speaker；
- test：2,529条、3.672173小时、14个speaker；
- 三类speaker跨split重叠均为0；169个与核心集共有的speaker只出现在train；
- 所有1,352条额外Rioplatense（2.110041小时）全部保留；
- 选中其他明确拉美116,825条、169.484488小时；
- 2小时speaker上限后可用容量176.391071小时，选样后仍留4.796542小时
  同级quota余量；未使用未知、半岛或MLS。

`source.tsv`和`selected_inventory.tsv`均为118,178行（含header），与summary一致。
此阶段只冻结了音频/文本/speaker/split选择，尚未证明该子集MFA/CTC label ready。

下一步先提取选中子集词表，并用早先已生成的通用Common Voice Spanish完整
MFA字典做定向基线审计，暂不重跑MFA：

```bash
ES_AUX_ROOT=outputs/es_latam_cv_auxiliary_170h_v1
ES_CANDIDATE_ROOT=outputs/es_candidate_train_sources_v1
ES_AUX_G2P="$ES_AUX_ROOT/mfa_g2p"
ES_AUX_BASE_AUDIT="$ES_AUX_ROOT/mfa_audit_base_v1"
ES_CV_BASE_DICT="$ES_CANDIDATE_ROOT/common_voice/mfa_g2p/common_voice_spanish_latin_america_mfa.dict"
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json

test -f "$ES_CV_BASE_DICT"
test ! -e "$ES_AUX_G2P"
test ! -e "$ES_AUX_BASE_AUDIT"

python scripts/prepare_mfa_g2p.py \
  --tsv "$ES_AUX_ROOT/source.tsv" \
  --output-dir "$ES_AUX_G2P" \
  --text-column text \
  --max-records 0 \
  --minimum-word-count 1

python scripts/audit_mfa_dictionary.py \
  --words "$ES_AUX_G2P/words.txt" \
  --word-counts "$ES_AUX_G2P/word_counts.tsv" \
  --dictionary "$ES_CV_BASE_DICT" \
  --vocab "$VOCAB" \
  --output-dir "$ES_AUX_BASE_AUDIT"

cat "$ES_AUX_G2P/summary.json"
cat "$ES_AUX_BASE_AUDIT/summary.json"
cat "$ES_AUX_BASE_AUDIT/oov_phone_counts.tsv"
sed -n '1,81p' "$ES_AUX_BASE_AUDIT/missing_words.tsv"
```

先返回两个summary、phone OOV计数和前80个缺词。下一轮再只对选中词表生成
共享增量修复计划，不要重跑全部Common Voice的MFA G2P。

## 0.29 2026-08-24 西语170小时选样的speaker配额修正

工作区首次运行0.28命令在写出产物前停止，报错：

```text
insufficient eligible speaker-disjoint hours for splits: train, validation
```

这不表示230+小时明确拉美原始库存不足。原因是第一版同时使用了1小时
speaker硬上限和先哈希后分split：少数高产speaker的大量音频被上限截断，
剩余时长又因speaker哈希偶然性不能精确满足163.2/3.4/3.4小时。

选样器已修正为`deterministic_duration_balanced_by_speaker`：先以speaker为不可分割
单位，再按当前未满的时长比例确定性地补足96/2/2配额；同一speaker仍只能
进入一个split。核心train speaker仍强制进train，核心validation/test speaker仍整体
排除，全部额外Rioplatense仍优先保留。普通拉美speaker上限改为2小时；如该
上限后总容量仍不足，新报错会显式打印`available/target/cap`，不再只报split名。

拉取最新提交后重跑；失败运行没有生成正式输出，不用删除原数据：

```bash
git pull origin codex/g2p-coverage-scan

python scripts/select_spanish_auxiliary_pool.py \
  --source-tsv "$ES_CANDIDATE_ROOT/common_voice/source.tsv" \
  --inventory-tsv "$ES_INVENTORY_ROOT/common_voice_inventory.tsv" \
  --rioplatense-tsv "$ES_AR_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --output-dir "$ES_AUX_ROOT" \
  --target-hours 170 \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --maximum-latin-american-speaker-hours 2.0 \
  --seed 20260824
```

完成后仍先校验`sha256.txt`，并返回`summary.json`与两个TSV的`wc -l`。

## 0.28 2026-08-24 西语明确拉美CV 170小时辅助池选样

全量候选库存审计通过，所有579,031条音频可读且SHA256验证通过：

- 通用Common Voice Spanish：358,330条、516.125434小时、5,015个speaker；
- MLS Spanish：220,701条、917.684176小时、86个speaker；
- CV中与Rioplatense v26核心train clip重复：9,862条、15.268736小时，
  必须排除；没有核心validation/test clip重复；
- 排除clip重复后，额外明确Rioplatense为1,352条、2.110041小时，其他明确
  拉美为162,405条、228.542930小时；
- 地区未知CV 140.596660小时、半岛CV 119.774463小时和MLS暂时不用。

库存v1将`América central`的6,120条记录保守地放入other tier；这是
元数据标签规则漏项，不是音频问题。`classify_spanish_accent`已增加
`america central`标记，新选样器从inventory的原始`accent`重新分类，因此无需
重跑全量音频metadata审计。

新增：

- `src/qwen_hotword/training/spanish_selection.py`
- `scripts/select_spanish_auxiliary_pool.py`
- `tests/test_spanish_selection.py`

第一版不再把“约170小时release pool”解释为包含核心语料，而是单独选
170小时明确拉美CV辅助池，再加约23.696562小时Temporal 2×核心候选，
原始可释放规模约193.7小时。这为MFA缺词、CTC长度失效和96/2/2划分留出
约29%的余量；最终是否`train >= 150 h`仍必须以Manifest/Temporal 2×实际结果
为准。若不足，只从剩余明确拉美CV增补，不立即引入未知、半岛或MLS。

辅助池按speaker时长配额确定性划分为96/2/2；先全部保留额外Rioplatense记录，其他明确
拉美记录按稳定顺序补足。普通拉美speaker最多贡献2小时，避免少数大
speaker主导。与核心train重合的speaker只能进train；与核心validation/test
重合的speaker整体排除。原始CV官方holdout、非`es`、文本不一致、不可读
音频和核心clip重复仍不得入选。

工作区运行（仅读现有库存，不重跑音频审计）：

```bash
ES_CANDIDATE_ROOT=outputs/es_candidate_train_sources_v1
ES_AR_ROOT=outputs/es_ar_train_sources_v1
ES_INVENTORY_ROOT=outputs/es_candidate_inventory_v1
ES_AUX_ROOT=outputs/es_latam_cv_auxiliary_170h_v1

test ! -e "$ES_AUX_ROOT"

python scripts/select_spanish_auxiliary_pool.py \
  --source-tsv "$ES_CANDIDATE_ROOT/common_voice/source.tsv" \
  --inventory-tsv "$ES_INVENTORY_ROOT/common_voice_inventory.tsv" \
  --rioplatense-tsv "$ES_AR_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --output-dir "$ES_AUX_ROOT" \
  --target-hours 170 \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --maximum-latin-american-speaker-hours 2.0 \
  --seed 20260824

(cd "$ES_AUX_ROOT" && sha256sum -c sha256.txt)
cat "$ES_AUX_ROOT/summary.json"
wc -l "$ES_AUX_ROOT/source.tsv" "$ES_AUX_ROOT/selected_inventory.tsv"
```

先返回`summary.json`和`wc -l`。确认总时长、三个split均达标、speaker跨split
重叠为0、核心holdout speaker未进入后，下一轮才对该`source.tsv`提取词表，
复用现有Common Voice Spanish MFA字典做选中词表的共享增量修复和严格复审。

本地验证：全仓Ruff通过；全量pytest 183项通过；库存/选样新增模块、CLI与
测试的Mypy strict通过；CLI help和`git diff --check`通过。全仓strict Mypy仍有
113项既有类型问题，分布在23个本轮未修改文件，本轮相关6个文件为0。

## 0.27 2026-08-24 西语150小时扩容：候选库存与CV元数据关联

当前阿根廷/拉普拉塔核心两语料在Temporal 2×下可释放15,872条、23.696562小时，
不足最终`train_ready >= 150 h`目标。扩容顺序固定为：明确阿根廷/Rioplatense元数据
> 其他明确拉美Common Voice > 地区未知/混合Common Voice > 限量MLS。MLS通常更偏
西班牙西语，且当前候选G2P是拉美模型，因此MLS只作后备并建议不超过最终train小时的
10%至15%；在真实库存结果返回前不硬编码配额。

通用Swift转换后的MLS/Common Voice TSV只有`audio/text`，不能直接筛方言或官方
split。新增只读库存工具：

- `src/qwen_hotword/training/spanish_inventory.py`
- `scripts/audit_spanish_candidate_inventory.py`
- `tests/test_spanish_inventory.py`

工具并行读取全部音频metadata，输出真实记录数/小时、时长分布和采样率。MLS从标准
文件名恢复speaker ID，但统一标为`mls_source_unknown_likely_peninsular`，不能声明为
拉美或阿根廷。Common Voice用音频basename关联原始v25
`validated/train/dev/test.tsv`，恢复client、accent、locale和官方split；accent只分成
保守的元数据tier，不作为声学方言预测。缺元数据、locale非`es`、文本关联不一致、
官方validation/test和音频不可读记录一律进入显式exclude pool。

Rioplatense v26核心`source.tsv`的clip basename同时作为跨版本防泄漏集合。通用CV中
与核心train/validation/test任一clip重复的记录都标为`exclude_core_overlap`；尤其
validation/test重叠绝不能回流辅助训练。输出目录包含：

```text
summary.json
mls_summary.json
common_voice_summary.json
mls_inventory.tsv
common_voice_inventory.tsv
run_config.json
sha256.txt
```

工作区运行（CPU/存储任务，不需要GPU）：

```bash
ES_CANDIDATE_ROOT=outputs/es_candidate_train_sources_v1
ES_AR_ROOT=outputs/es_ar_train_sources_v1
ES_INVENTORY_ROOT=outputs/es_candidate_inventory_v1
CV_ES_ROOT=/host_home/z00841352/27A/data/Common_Voice_Scripted_Speech_25.0/cv-corpus-25.0-2026-03-09/es

test ! -e "$ES_INVENTORY_ROOT"

python scripts/audit_spanish_candidate_inventory.py \
  --mls-tsv "$ES_CANDIDATE_ROOT/mls/source.tsv" \
  --common-voice-tsv "$ES_CANDIDATE_ROOT/common_voice/source.tsv" \
  --common-voice-root "$CV_ES_ROOT" \
  --rioplatense-tsv "$ES_AR_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --output-dir "$ES_INVENTORY_ROOT" \
  --workers 16 \
  --progress-every 10000

(cd "$ES_INVENTORY_ROOT" && sha256sum -c sha256.txt)
cat "$ES_INVENTORY_ROOT/summary.json"
cat "$ES_INVENTORY_ROOT/mls_summary.json"
cat "$ES_INVENTORY_ROOT/common_voice_summary.json"
```

先返回三个JSON。下一轮根据`training_pool_hours`、`official_split_hours`、
`accent_tier_hours`和`core_overlap_hours`确定约170小时release pool，再运行通用
MLS/CV的共享增量MFA、完整Manifest与Temporal 2×审计。当前不直接合并、不读取核心
sealed test音频内容、不把metadata tier写成`es-AR`。

本地验证：全仓Ruff通过；新增库存定向pytest 3项通过；全量pytest 180项通过；新增
模块、CLI与测试Mypy strict通过；CLI help和`git diff --check`通过。对`src scripts
tests`强制运行全仓strict Mypy仍有113项既有类型问题，分布在Torch训练模块、旧测试
JSON对象断言和streaming mock协议等23个未修改文件，本轮三个新增文件为0。

## 0.26 2026-08-21 阿根廷/拉普拉塔西语修复审计与Manifest入口

共享增量修复已在H200完成。共享代理词3,264个，MFA模型、代理词表和代理字典
SHA256均已保存；prepare、finalize及全部输出SHA256校验通过。修复后结果：

- SLR61：3,486/3,487个词可用，missing 1、phone OOV 0、duplicate 0，token覆盖
  99.998087%。唯一未解析词为单次出现的`smart-phone`（`no_safe_proxy`）。
- Common Voice Rioplatense v26：18,520/18,786个词可用，missing 266、phone OOV 0、
  duplicate 0，token覆盖99.729417%。剩余原因是228个`no_safe_proxy`、32个
  `missing_proxy_pronunciation`和6个`unsupported_diaeresis_context`，共278个
  corpus token。样本以连字符复合词、葡语/其他外语专名和异常字符为主，不应为追求
  100%覆盖而强制生成低可信西语发音。

因此不再重跑或扩大MFA代理规则。完整Manifest必须保留所有源行：可解析行进入
`train_ready.jsonl`，包含上述低可信词的少量行进入`needs_review.jsonl`，不得静默
删除。`training_labels_ready=false`在这里表示词典并非100%覆盖，不能把整套词典称为
全量ready；它不再阻止使用保留式Manifest builder量化逐行ready/review影响。

Rioplatense的规范`source.tsv`同时含官方`source_split`，不能把整份文件标成一个固定
split。`build_full_training_manifest.py`和`full_manifest.py`新增显式
`--split-column`：逐行只接受`train/validation/test/unsplit`，与非默认`--split`
互斥，并在summary中记录`split=mixed`、`split_column`与`split_counts`。旧固定split
调用和旧resumable配置保持兼容。此阶段只构建和审计Manifest，不训练；Rioplatense
test在下一步按记录中的官方split单独导出并封存。

工作区使用新目录，分别构建，不合并：

```bash
ES_AR_ROOT=outputs/es_ar_train_sources_v1
ES_MANIFEST_ROOT=outputs/es_ar_full_manifests_v1
VOCAB=configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json

python scripts/build_full_training_manifest.py \
  --tsv "$ES_AR_ROOT/slr61/source.tsv" \
  --audio-root /host_home \
  --dictionary "$ES_AR_ROOT/repaired_mfa_v1/slr61/slr61_spanish_latin_america_repaired.v1.dict" \
  --vocab "$VOCAB" \
  --output-dir "$ES_MANIFEST_ROOT/slr61" \
  --language es \
  --dataset slr61_argentinian_spanish \
  --id-prefix slr61_es_row \
  --split unsplit \
  --allow-exact-dictionary-connectors \
  --shard-size 5000 \
  --workers 16

python scripts/build_full_training_manifest.py \
  --tsv "$ES_AR_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --audio-root /host_home \
  --dictionary "$ES_AR_ROOT/repaired_mfa_v1/common_voice_rioplatense_v26/common_voice_rioplatense_v26_spanish_latin_america_repaired.v1.dict" \
  --vocab "$VOCAB" \
  --output-dir "$ES_MANIFEST_ROOT/common_voice_rioplatense_v26" \
  --language es \
  --dataset common_voice_rioplatense_v26 \
  --id-prefix cv_rio_v26_es_row \
  --split-column source_split \
  --allow-exact-dictionary-connectors \
  --shard-size 5000 \
  --workers 16
```

下一轮先返回两套`summary.json`、各自`wc -l train_ready.jsonl needs_review.jsonl`，
以及Rioplatense summary中的`split_counts`。确认source=ready+review、missing audio为0、
SLR仍为unsplit且Rioplatense为9,903/266/224后，再实现SLR speaker-disjoint划分与
Rioplatense官方split封存；当前不合并两套语料，也不加入MLS/通用Common Voice。

## 0.25 2026-08-21 热词容量/50 ms任务暂停点与六步恢复路线

本任务目前暂时让位于西语数据处理，但以下路线已经确认，后续恢复时不要重新讨论或
从全词库RapidFuzz扫描重新开始。目标口径是：4,000个active葡语热词下，从已有CTC
输出到检索Top-K的纯检索P95不超过50毫秒；不含audio processor、Qwen Encoder、
CTC Head和LLM。产品决策仍使用Operating Top-5、threshold 0.86、maximum edit
ratio 0.35和既有posterior/margin门控；Raw Top-7/10只作扩容观察，不能增加Prompt
注入数量或改变正式通过标准。

固定六步如下：

1. **证据与慢基线冻结（已完成）**：封存formal100离线质量、smoke20累计流式时延、
   CTC前缀稳定性、rank displacement和负例碰撞证据。已确认2k全词库逐词近似匹配
   P95约2.29秒，瓶颈在CPU全扫描；累计CTC并非append-only，2/4/6秒lookback覆盖
   率约64.5%/87.1%/96.8%。
2. **可复用Posterior Replay（已完成并封存）**：H200的
   `replay_streaming_posterior_smoke20_v2`共20个case、67个step、17个tail flush，
   `argmax_preserving_float16`分片全部SHA256通过，
   `greedy_equivalence_mismatches=0`。后续候选器和精排必须复用这份因果资产，避免
   每次重新跑Qwen/CTC造成计时噪声。
3. **快速候选召回层（进行到诊断子步骤）**：已完成Aho-Corasick精确快路径与4k
   基线，工作区4k纯查询P95约0.258毫秒，证明50毫秒工程预算可达；但精确召回仅
   127/172（73.84%），正例case hit 69/80（86.25%），负例FPR 3/20（15%），不能
   单独上线。恢复入口是实现音素2/3/4-gram Anchor倒排索引，比较Top-64/128/256
   候选集；Top-128为速度消融、Top-256为主方案，必要时用更大候选或Posterior
   scorer作诊断/后备。候选层验收必须同时统计对CPU完整扫描Top-5的覆盖率、目标
   热词召回率、无Anchor率和候选集大小。
4. **小候选集近似精排与流式窗口消融（未开始）**：只对Anchor候选运行现有近似
   音素评分器，不再扫描4,000词；分别比较完整当前CTC序列以及2/4/6秒recent
   lookback。保留AC exact命中作为低延迟证据，但不能让未来音频或离线结果进入
   当前step。记录各阶段P50/P95/P99，并确认4k总检索P95仍低于50毫秒。
5. **排序稳定性与误报控制（未开始）**：在不改0.86、Top-5和Prompt模板的前提下，
   评估显式候选retention、family diversity/最长匹配和低信息短语控制；任何TTL、
   跨chunk保留或去重策略都必须成为显式配置并单独做消融。正式输出仍为Operating
   Top-5，同时保存Raw Top-7/10，重点复核`e essa`等跨词边界误触发及新增容量造成
   的回归。
6. **4k正式验收与上线建议（未开始）**：在同一formal100与累计流式replay上比较
   100/500/1k/2k/4k，报告Raw Top-5/7/10、Operating Top-5、正例case hit、负例
   FPR、检索P50/P95/P99、deadline miss、索引构建时间和RSS。通过条件至少包括4k
   纯检索P95小于50毫秒、质量相对冻结基线的变化可解释且没有静默使用sealed test；
   最终再决定默认Anchor规模、lookback和是否启用Posterior fallback。

因此，**下次恢复直接从Step 3的Anchor shortlist实现开始**，不是重跑Step 1/2，
也不是把AC exact通道误当成最终检索器。恢复前可补充一个短诊断JSON，集中输出45个
exact miss、3个负例误检和2个扩容回归的原因，以及Anchor Recall@64/128/256；该
诊断不需要上传音频、完整文本或Posterior张量。详细冻结结果与命令分别见本文件
0.20至0.23和`docs/HOTWORD_CAPACITY_EVAL.md`第7至9节。

## 0.24 2026-08-21 阿根廷西语共享增量MFA修复

SLR61与Common Voice Rioplatense的完整MFA G2P已经运行过，现有字典保持只读，
不重新处理全部3,487/18,786个输入词。当前阻塞是首轮字典分别仍缺656/3,445个词，
且已生成发音包含西语非音位性的`U+0303 COMBINING TILDE`。修复流程缩减为一个
跨语料共享增量批次：先复用原词或去acute后已存在的唯一发音，只把仍缺失的`ñ/ü`
等安全代理词去重写入一个`proxy_words.txt`；MFA只对这个小文件运行一次；最后分别
回映射两套原始词形并自动运行共享v0.2 CTC词表审计。

新增：

- `src/qwen_hotword/training/spanish_mfa_repair.py`
- `scripts/prepare_spanish_mfa_repairs.py`
- `scripts/finalize_spanish_mfa_repairs.py`
- `tests/test_spanish_mfa_repair.py`

代理规则只作用于G2P输入，不修改训练文本：acute accent只在代理中移除；`ñ`使用
探针较可靠的`ni`代理，并只在该规则下把代理输出的`ɲ j`还原为`ɲ`；`gü`使用`gw`
代理。最终西语候选发音只删除`U+0303`，不会改变葡语词典或共享词表。任何没有安全
代理、多发音、代理G2P失败、空发音或仍有phone OOV的词继续写入
`unresolved_words.tsv`，`training_labels_ready=false`，不得构建完整Manifest。

工作区使用两个新目录，不覆盖现有MFA字典或审计：

```bash
ES_AR_ROOT=outputs/es_ar_train_sources_v1
SLR_ROOT="$ES_AR_ROOT/slr61"
CV_RIO_ROOT="$ES_AR_ROOT/common_voice_rioplatense_v26"
SLR_DICT="$SLR_ROOT/mfa_g2p/slr61_spanish_latin_america_mfa.v1.dict"
CV_RIO_DICT="$CV_RIO_ROOT/mfa_g2p/common_voice_rioplatense_v26_spanish_latin_america_mfa.v1.dict"
ES_REPAIR_ROOT="$ES_AR_ROOT/shared_mfa_repair_v1"
ES_REPAIRED_ROOT="$ES_AR_ROOT/repaired_mfa_v1"

python scripts/prepare_spanish_mfa_repairs.py \
  --corpus slr61="$SLR_ROOT/mfa_g2p" \
  --dictionary slr61="$SLR_DICT" \
  --corpus common_voice_rioplatense_v26="$CV_RIO_ROOT/mfa_g2p" \
  --dictionary common_voice_rioplatense_v26="$CV_RIO_DICT" \
  --output-dir "$ES_REPAIR_ROOT"

conda run --no-capture-output -n aligner mfa g2p \
  --verbose \
  --num_jobs 16 \
  --num_pronunciations 1 \
  "$ES_REPAIR_ROOT/proxy_words.txt" \
  models/mfa/g2p/spanish_latin_america_mfa.zip \
  "$ES_REPAIR_ROOT/proxy_spanish_latin_america_mfa.v1.dict"

python scripts/finalize_spanish_mfa_repairs.py \
  --corpus slr61="$SLR_ROOT/mfa_g2p" \
  --dictionary slr61="$SLR_DICT" \
  --corpus common_voice_rioplatense_v26="$CV_RIO_ROOT/mfa_g2p" \
  --dictionary common_voice_rioplatense_v26="$CV_RIO_DICT" \
  --repair-root "$ES_REPAIR_ROOT" \
  --proxy-dictionary "$ES_REPAIR_ROOT/proxy_spanish_latin_america_mfa.v1.dict" \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir "$ES_REPAIRED_ROOT"
```

先返回共享`prepare_summary.json`、最终`summary.json`以及两套
`mfa_audit/summary.json`。只有两套审计的missing、OOV、duplicate均为0且
`training_labels_ready=true`后，才分别构建SLR61与Rioplatense完整Manifest；当前
仍不合并MLS/通用Common Voice，也不读取或重分Rioplatense官方test。

本地验证：全仓Ruff通过；西语修复/诊断/来源/phone normalization定向pytest 10项
通过；全量pytest 175项通过；新模块Mypy strict通过；两个CLI help和
`git diff --check`通过。全仓Mypy仍有12项既有Torch/可选RapidFuzz类型问题，均位于
本轮未修改模块，新模块为0。

## 0.23 2026-08-21 Posterior v2封存与4k/50 ms AC基线

H200已完成`replay_streaming_posterior_smoke20_v2`：20个case、67个replay row、
20个final row、17个tail flush、3个float16分片、90个class和7,552个有效时间帧。
量化为`argmax_preserving_float16`，只有1帧需要argmax保持修正，
`greedy_equivalence_mismatches=0`、总体及分片`status=pass`，`sha256.txt`
内全部8项均为OK，且没有读取sealed test。Step 2正式封存。

领导目标明确为4,000个active热词下“已有CTC decoded音素到检索Top-K”的纯检索
P95不超过50毫秒；不包含processor、Encoder、CTC Head或LLM。产品仍保持
Operating Top-5/threshold 0.86，Raw Top-7/10只作扩容后排名观察。

新增纯Python整数音素Aho-Corasick精确检索基线和独立CLI。它每个step重扫完整
当前greedy音素序列，不假设CTC前缀append-only；支持case active词库、最长匹配、
span置信度排序和去重。独立benchmark输出Exact availability、Exact Top-5/7/10、
正例case hit rate、负例FPR、匹配数、索引构建/内存和50毫秒deadline miss。

用用户上传的formal100/2k包做本地只读实测：表中2,402个唯一entry建成
14,916个节点，构建约12.28毫秒；查询平均/P95/P99/最大值约为
0.101/0.156/0.201/0.305毫秒，表明纯AC通道有充足时延余量。但当前CTC
纯exact只覆盖132/172期望词（76.74%），2k负例中3/20有精确误触发，
因此不能用AC单路替代近似检索。

容量资产默认档位新增4,000，但仍保留10k为最大采样池，以保持旧
100/500/1k/2k/5k/10k档确定性不变。工作区必须写新目录
`assets_with_4000_v2`，先比较旧新共有档SHA256，再运行
`benchmark_exact_hotword_capacity.py`的formal100 100/500/1k/2k/4k曲线。详细命令见
`docs/HOTWORD_CAPACITY_EVAL.md`第9节。AC通过后再实现Anchor shortlist；GPU Posterior
scorer保留为Anchor召回不足时的后备。

本轮任务文件Ruff、容量定向pytest、59个source模块Mypy strict、CLI help和
`git diff --check`均通过；全量pytest共收集171项并通过（依赖不可用项按既有规则
skip）。全仓Ruff仍只有9个既有UP038，位于未改动的multi-nested、prompt、retrieved
和temporal模块，本轮没有纳入无关机械修改。

## 0.22 2026-08-20 2k容量Step 1收口与Top-7/10观察口径

Step 1分析包`capacity_analysis_bundle_step1_v1.tar`已完整校验，共26个预期
文件，SHA256为
`17c60ef30b427d3550917d3f80aa038385c86613c250bd009692c3584b8a82df`。
本阶段不再需要补传文件。

formal100的100/500/1k/2k词Raw Recall@5分别为95.35%/94.19%/91.86%/
88.37%，Operating Recall@5为81.40%/81.40%/81.40%/80.23%，负例FPR为
0%/0%/5%/20%。从100扩到2k后，原本已通过Operating Top-5而新掉出的
目标只有`temos`和`pro pão ficar`，都是rank 5变rank 6；其余新增
displacement多为100词时已被0.86门控拒绝的目标。2k负例误检包括
`saque`、`dispositivos`和两次`e essa`；其中`e essa`暴露了通用低信息
短语及跨词边界音素匹配问题，不能用降低0.86阈值掩盖。

47个相邻流式step中31个改写旧CTC后缀，append-only rate仅34.04%，
17/20个case至少改写一次；尾部flush改写率13/17=76.47%。根据step位置
和累计音频时长的近似诊断（不是强制对齐时间戳），2/4/6秒lookback分别
覆盖约64.5%/87.1%/96.8%的历史修订。因此第4步必须比较2/4/6秒，
不能只做append-only或默认固定2秒精排。

稳态Encoder/CTC Head/decode的P95约为45.8/7.7/2.6毫秒，2k全扫描matching
P95约2.29秒，证明主瓶颈是逐词CPU精确匹配。第3步预定同时评估
GPU shortlist Top-128/256/512，主方案Top-256，Top-128为速度消融，Top-512为
召回诊断/后备；验收以“是否覆盖CPU完整扫描的精确Top-5”为主，不只看
期望热词是否进shortlist。

产品口径继续封存为Top-5/threshold 0.86，不改Prompt注入数。从现在开始
容量评测额外保存Raw Top-7和Top-10作为观察范围，用于分析增容后的
rank 6至10轻度掉队。`capacity_benchmark.py`新增逐query
`raw_top7_ids`/`raw_top10_ids`、对应命中数，以及汇总Raw correct/recall和
positive-case hit rate。Operating决策、FPR和容量通过标准仍只使用Top-5。

当前六步状态：Step 1已收口；Step 2的argmax-preserving float16代码已完成，
等待H200在新目录`replay_streaming_posterior_smoke20_v2`重跑；Step 3至6尚未开始。

## 0.21 2026-08-20 Posterior Replay float16近并列修复

第2步Posterior Replay首次H200 smoke完成20条、67个step和3个float16分片，
但严格验证在67个row中发现1个greedy token/span不等价，因而正确中止，
没有生成`summary.json`。pynvml弃用和generation temperature都是无关警告。

本地已用近并列frame稳定复现：float32中phone 2略大于phone 1，转float16后
两者变成相同数值，`argmax`因索引顺序改选phone 1。这是量化边界，不是CTC
模型或音频失败。

修复不放宽校验，而是将存储定义改为显式
`argmax_preserving_float16`：先记录float32每帧赢家，常规转float16，只对舍入后
赢家改变的帧将冲突值下调一个float16 ULP。Manifest新增总体和每分片
`argmax_correction_frames`、`maximum_abs_quantization_error`；索引也记录每个row的
修正数和误差。验证仍要求SHA256、shape/dtype、长度、有限值、logsumexp和
greedy token/span全部通过；若再失败，额外写入带case/chunk的
`posterior_validation_failure.json`。

失败的`replay_streaming_posterior_smoke20_v1`保留，不覆盖。修复后必须在
`replay_streaming_posterior_smoke20_v2`重跑。只有v2报告
`greedy_equivalence_mismatches=0`后，才继续第3步GPU Top-256候选器。
本地任务文件Ruff、定向pytest 8项、全量pytest 169项、57个source模块
Mypy strict、CLI help smoke和`git diff --check`均通过。

## 0.20 2026-08-19 2k葡语热词库：诊断冻结与Posterior Replay

用户将本阶段产品目标固定为2,000个在线葡语热词。已有RapidFuzz精确全扫描
基线表明，仅做增量编辑距离不足以达标：2k的Raw Recall@5从100词的95.35%
降到88.37%，Operating Recall@5为80.23%，负例FPR为20%；20条真实累计流式
replay上retrieval P95为2.893秒，CTC+retrieval P95为3.021秒。对照的100词
分别为109毫秒和514毫秒。

已确认的工程路线是：

```text
float16 frame-level CTC log posteriors
-> GPU packed CTC candidate recall over 2k
-> Top-128/256
-> recent-window exact rerank + explicit retention/family diversity
-> unchanged Operating Top-5
```

本轮只完成上述架构的输入证据层，没有实现GPU scorer，也没有改变CTC
checkpoint、评分公式、0.86阈值、Top-5、Prompt或模型：

- `capacity_benchmark.py`新增每阶段source时延分布，不再只保存合计值。
- 新增`ctc_prefix_stability.json`：按case衡量相邻累计chunk的最长公共前缀、
  旧后缀被改写token数、append-only率与case级不稳定率。
- 新增`rank_displacement_cases.jsonl`/`rank_displacement_summary.json`：逐例区分
  正确词被挤出Raw Top-5、Raw Top-5内但被Operating guard拦下，以及负例误触发。
- `capacity_replay.py`新增可选`--save-log-posteriors`：将每个因果累计step的
  `[T,90]` log-softmax以float16填充分片保存，JSONL只存索引与decoded对照。
- Posterior分片在写入后立即做SHA256、shape/dtype、有效长度、有限值、
  logsumexp归一化和greedy token/span 100%等价校验；任一不一致直接失败。
- CLI新增`--posterior-shard-size`，默认32个step一个分片。旧replay默认行为不变。

本地张量测试使用仓库`.conda`的Torch 2.10真实执行，已覆盖float16分片
round-trip、严格校验和greedy等价；基础Python无Torch时对应测试会正常skip。
完整H200命令和验收条件见`docs/HOTWORD_CAPACITY_EVAL.md`第7–8节。第一步先用
旧formal100离线replay生成2k排名挤出主诊断，再用旧smoke20流式replay生成跨chunk
前缀与时延诊断；第二步在新目录生成20条Posterior smoke。预期旧case
未变时仍是67个step/17个tail flush，必须看到
`greedy_equivalence_mismatches=0`后才能开始GPU Top-128/256 scorer。

本轮本地验证：任务文件Ruff通过；定向pytest 8项通过且已用仓库
`.conda`的Torch实跑；全量pytest 169项通过；57个source模块全量
Mypy strict通过；两个CLI `--help`和`git diff --check`通过。全仓库Ruff仍报
9个既有UP038，位于本轮未改动的multi-nested/prompt/retrieved/temporal文件；
本轮没有把它们纳入交付。

## 0.19 2026-08-18 葡语100至10k热词库容量评测（代码完成，待H200）

按照已确认的容量阶梯`100/500/1k/2k/5k/10k`新增葡语单语热词库容量评测。
10k只是当前精确扫描实现的压力上限，不预先宣称为上线容量；100k不在本轮范围。
固定复用Temporal 2× CTC Head、Top-5、threshold 0.86、maximum edit ratio 0.35、
posterior weight 0.25、minimum phones 4及既有v3 formal100，不训练、不调参、不读
sealed test。

新增实现：

- `src/qwen_hotword/hotwords/capacity_assets.py`：从Noah葡语train-only Manifest
  确定性采样真实1至4词连续n-gram，校验MFA/phone覆盖，构建Representative与
  Hard-negative两套严格嵌套active词库；支持读取既有formal100
  `sample_selection.json`，强制formal100并拒绝明确的forced选择，同时兼容尚无
  `retrieval_mode`字段的旧Operating formal100；以选择文件中的最长匹配真值覆盖
  原始多真值，防止扩容时悄悄改变case集合或真值口径。
- `src/qwen_hotword/hotwords/capacity_replay.py`：生成完整Validation feature-cache
  离线CTC replay，或真实累计0-2/0-4/...音频的流式CTC replay；后者分段记录
  processor、Encoder、CTC Head、posterior decode和GPU内存，尾部不足2秒保留。
- `src/qwen_hotword/hotwords/capacity_benchmark.py`：同一decoded replay回放各级词库，
  输出Raw Recall@1/3/5/10/20、Operating Recall@5、MRR/rank/margin、负例FPR、
  Top-5 churn、matching/sort/select延迟、P50/P90/P95/P99、2秒deadline、RSS、
  Python heap、GPU峰值及累计音频时长分桶；默认retrieval P95超过2秒后停止扩容。
- `score_decoded_hotwords`和`profile_decoded_hotwords`：与现有logits评分完全复用
  同一个matcher和门控逻辑，只把不可变decoded phoneme作为输入，避免每一级重复
  Encoder造成质量与计时噪声。
- CLI：`build_hotword_capacity_assets.py`、`build_hotword_capacity_replay.py`、
  `benchmark_hotword_capacity.py`。
- 测试：`tests/test_hotword_capacity.py`，覆盖logits/replay等价、严格嵌套active
  数量、train-only候选、Representative/Hard两profile、离线质量/性能输出、容量
  headroom建议及sealed-test拒绝。

输出根目录固定建议为：

```text
outputs/noah_pt_full_training_v1/hotword_capacity_eval_v1
```

完整工作区命令和判定口径见`docs/HOTWORD_CAPACITY_EVAL.md`。执行顺序必须是：CPU
资产构建、formal100离线replay、Representative离线benchmark、Hard-negative离线
benchmark、20条流式replay smoke、Representative流式benchmark；smoke通过后再做
formal100流式replay。最终上线建议只取Representative，Hard-negative只作压力诊断。

本地不加载真实H200模型或30.9GB缓存。全仓库Ruff通过；全量pytest 166项通过；
5个任务source模块Mypy strict通过；容量/距离定向pytest通过；3个CLI
`--help`和`git diff --check`通过。全仓库Mypy仍是6个既有Torch/transformers模块的
11项错误，本轮5个模块为0。发布前remote-parent检查仍需在收口阶段执行。

H200第一次Representative运行在`python_dynamic_programming`编辑距离后端上，
100词档Raw Recall@5为164/172（95.35%）、Operating Recall@5为140/172
（81.40%）、负例FPR为0；但retrieval P95/P99为2.070/2.515秒，耗时几乎全部在
matching，因而保护机制在100词档停止。`recommended_online_cap=0`只表示该慢速
参考实现不满足工程SLO，不代表CTC质量或产品容量为0。随后新增RapidFuzz等价距离
后端和随机序列一致性测试；下一轮必须写入新目录，并确认报告中的
`edit_distance_backend=rapidfuzz`后再解释扩容曲线。

## 0.18 2026-08-18 v3 Formal100 Operating/Forced Top-5 实测结论与词库上限计划

同一批v3 `formal100`（80正例、20负例、172个期望热词）已经完成
`0.86 / Operating Top-5`和无门控`Forced Top-5`的离线与2秒流式控制实验。
两轮都只读validation资产，没有读取sealed test；Forced结果保存在新的输出目录，
没有覆盖Operating基线。

正式对比如下（Recall均为最终文本的严格完整词/短语exact recall）：

| 推理组 | 策略 | 正确/期望 | Recall | WER | CER | 负样本错误注入 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 离线B | Operating Top-5，threshold 0.86 | 157/172 | 91.2791% | 8.2329% | 3.3610% | 0/20 |
| 离线B | Forced Raw Top-5，无门控 | 158/172 | 91.8605% | 8.1660% | 3.4078% | 20/20 |
| 流式D | Operating Top-5，threshold 0.86 | 160/172 | 93.0233% | 7.9652% | 3.3453% | 0/20 |
| 流式D | Forced Raw Top-5，无门控 | 159/172 | 92.4419% | 8.6345% | 4.0019% | 20/20 |
| 流式E | Oracle | 161/172 | 93.6047% | 8.0991% | 3.3610% | 0/20 |

Forced离线相对Operating只多正确1个热词（+0.5814个百分点），并把所有20条
负样本都注入了错误候选；共注入302个错误候选，最终严格写出1个，但相对baseline
没有新增严格错误热词。Forced流式相对Operating反而少正确1个热词（-0.5814个
百分点），WER增加0.6693个百分点（119个词错误变为129个，增加10个），CER增加
0.6566个百分点。严格热词幻觉率仍为0不代表错误Prompt无害：普通词错误已经明显
增加。

流式失败证据也支持保留门控：Forced使`ctc_never_detected`从5降为0，但
`ctc_detected_too_late_already_fixed`从3增至9，且20条负样本均成为
`wrong_hotword_injected`。因此无门控只是让更多低置信、随累计音频波动的候选进入
Prompt，并没有提高最终流式Recall。正式2秒流式基线继续固定Top-5、threshold 0.86、
maximum edit ratio 0.35、posterior weight 0.25和5-token rollback；Forced目录只作为
诊断对照封存。早期约99%的数字是另一资产上的CTC raw ranking recall，不是Qwen最终
端到端Recall，不能用来替代本次结论。

Forced离线报告位于：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/
  prompt_multi_nested_formal100_forced_top5_v1/multi_nested_prompt_report.json
```

下一阶段拟先用已经训练完成的葡语Temporal 2× CTC Head测单语热词库上限，分成两条
互不混淆的曲线：

1. 检索质量上限：保持同一批音频、CTC checkpoint、CTC decoded序列、Top-5和
   0.86门控不变，只扩大每条case的active词库，主指标为raw Recall@5，同时报告
   Operating Recall@5、目标热词rank/MRR、负样本FPR和跨chunk候选稳定性。
2. 工程性能上限：每2秒对当前累计音频重新检索，分别记录Encoder、CTC Head、
   posterior decode/CPU copy、hotword matching、Top-K排序和Prompt刷新耗时，并记录
   p50/p90/p95/p99、2秒deadline miss、进程RSS、GPU allocated/reserved/peak、词库
   加载时间及每词内存。

词库规模采用确定性嵌套超集，最终固定先测100、500、1k、2k、5k、10k；10k为本轮
压力上限，不测25k/50k/100k。若某一级retrieval p95已经超过2秒或资源达到预先约定
的上限，则记录首个失败点并停止更大规模。扩展词来自train-only葡语真实1至多词
n-gram，不读取sealed test；每个case
必须排除参考文本中存在但未标注为目标的词，防止把真实热词误算成干扰项。除频率分层
的代表性词库外，还要单列长度匹配、近音、包含/被包含关系等hard-negative词库，避免
随机干扰词高估Recall。

现有实现还不能直接产生可信的工程上限报告：`score_hotwords`会对每个active entry
逐一执行Python局部编辑距离窗口扫描，再对全部match排序；当前v3每条case固定只有
100个active hotwords。流式时间线只记录整条样本`inference_seconds`，没有上述分段
耗时或RSS/GPU峰值。下一轮应先增加独立benchmark资产构建器和分段计时器；质量实验
复用一次生成的离线/逐chunk CTC decoded序列，在所有词库规模上replay，避免重复运行
Encoder造成噪声。随后只在100、性能拐点和最大可接受规模上跑真实全链路流式D组确认。

## 0.17 2026-08-18 v3 Raw/Forced Top-5 无门控消融

正式100条 `0.86 / Operating Top-5` 离线与流式评测均已在H200完成并保持
`status=pass`。最终精确热词Recall A/B/C/D/E = 89.53%/91.28%/89.53%/
93.02%/93.60%；流式D比离线B高1.74个百分点，D与流式Oracle E仅差0.58个百分点。
自然100条没有人工确认或强制对齐时间戳，因此其Recall/WER/CER有效，但
`boundary_summary`为空，声学结束相对延迟为null；这不表示CTC没有检出。

为验证早期无阈值Ranking Top-5的记忆口径，新增显式
`--retrieval-mode forced_topk`。该模式不是把threshold简单设为0，而是离线直接读取
`ranking_top5`、流式每个累计音频step直接使用`ranked_matches[:5]`；0.86 score、
0.35 edit ratio、posterior confidence和margin门控均不参与候选选择。`minimum_phonemes=4`
与`posterior_weight=0.25`仍用于生成原始排名。样本仍为同一formal100，旧Operating
目录不覆盖，新报告显式记录`threshold=null`、`guards_applied=false`和候选来源。

必须先生成独立的离线Forced控制目录，再以它作为流式A/B导入和配置校验来源。
工作区命令见`docs/STREAMING_RAG_EVAL.md`“Raw/Forced Top-5消融”。该消融用于观察
Recall收益和错误注入/WER代价，不替代0.86部署基线，也不读取sealed test。

本地实际验证：Forced/流式定向pytest 24项通过；全量pytest通过；本轮文件Ruff
通过；54个source模块全量Mypy通过；两个CLI `--help`和`git diff --check`通过。
全仓库Ruff仍只有9个既有UP038，不在本轮改动范围。

## 0.16 2026-08-17 v3 Top-5 流式正式100条控制实验（H200 已完成）

首次 Top-3 50条流式smoke已在H200完成，A/B/C/D/E均为50条且
`status=pass`。该次使用v2 `stratified_100/retrieved_rag_v1`，最终精确热词
Recall A/B/C/D/E = 95%/96.25%/96.25%/96.25%/96.25%；D相对B的Recall
变化为0，WER/CER分别增加0.252/0.145个百分点。该smoke仅验证流式
链路，不是正式离线→流式结论；原目录不覆盖，作为参考封存。

用户确认正式控制实验应对齐更早的v3多词/组合/嵌套端到端Top-5，
而不是v2 Top-3。v3固定参数为0.86/Top-5/.35/.25/min posterior 0/
minimum phonemes 4/margin 0/Temporal 2×。旧v3端到端只有50条，因此新增
`formal100`选样profile：把原七组配额严格2倍扩展到80正例+20负例，
且原50条必为新100条的确定性子集。

正式流程分两步：先用`run_multi_nested_prompt_eval.py
--selection-profile formal100`在新目录生成这100条的离线A/B/Oracle；
再用`run_streaming_rag_evaluation.py --offline-format multi_nested_v3`读同一
`sample_selection.json`运行C/D/E。流式入口在加载模型前强制校验Top-5
参数、Prompt/语言/dtype/`max_new_tokens`/qwen-asr版本、输入SHA256、
CTC修正报告及checkpoint SHA256，任意不同立即失败。

同时修正两个smoke暴露的统计问题：注入前已正确不再产生负的
`chunks_from_injection_to_first_correct`，而是单独记录
`correct_before_first_injection`；只有人工或强制对齐证明热词位于尾块时才分类
`tail_flush_failure`。完整H200命令和新输出目录见`docs/STREAMING_RAG_EVAL.md`。

本地实际验证：流式/v3定向pytest 23项通过；全量pytest通过；
本轮文件Ruff通过；54个source模块全量Mypy通过；两个CLI `--help`通过；
`git diff --check`通过。全仓库Ruff在本地新版规则下仍报告9个既有UP038，
位于本轮未修改的multi-nested/retrieved/prompt/temporal历史代码，本轮不做
扩张性机械改写。

## 0.15 2026-08-17 流式端到端热词 RAG 评测（代码完成，待 H200 实跑）

本轮暂停西语数据处理，新增独立的 Qwen3-ASR-1.7B 流式端到端热词
评测，不改训练数据、模型权重、CTC checkpoint、Manifest 或既有离线结果。

已对照 Qwen 官方示例和当前上游源码确认：流式仅支持 vLLM，默认每个
2 秒 chunk 累计重放全部已收到音频；前 2 个 chunk 无文本 prefix，从第 3 个
chunk 开始回退最后 5 个 `processor.tokenizer` token；尾音通过
`finish_streaming_transcribe` 不补零处理。官方接口没有流中动态 context
setter，所以本评测把同轮候选注入明确标记为 experimental state refresh：通过
公开 initializer 构造临时 state，只刷新活跃 state 的 Prompt 元数据。安装版本
字段不兼容时立即失败，不会假装同轮生效或静默延迟。

新增：

- `src/qwen_hotword/inference/streaming_core.py`：2 秒调度、尾音 flush、tokenizer
  5-token rollback、fixed/unfixed 记录、因果候选、同轮 Prompt、逐 chunk
  时间线、延迟与失败分类。
- `src/qwen_hotword/inference/streaming_backends.py`：官方 vLLM streaming adapter，以及
  独立 Transformers Qwen Encoder + 封存 Temporal 2× CTC Head 的累计音频检测器。
- `src/qwen_hotword/inference/streaming_rag.py`：统一 A/B/C/D/E、单样本原子分片
  resume、Recall/WER/CER、边界、延迟、稳定性与失败汇总。
- `src/qwen_hotword/inference/streaming_boundary.py`：只接受强制对齐或人工确认
  时间戳，通过运行时前置静音生成不覆盖原音频的 2 秒相位变体。
- `scripts/run_streaming_rag_evaluation.py`
- `scripts/build_streaming_boundary_eval.py`
- `tests/test_streaming_core.py`
- `tests/test_streaming_boundary.py`
- `tests/test_streaming_rag.py`
- `docs/STREAMING_RAG_EVAL.md`

原始端到端集复用离线 `sample_selection.json` 和 A/B 预测；C/D/E 重新做真实
流式推理。D 每一步只读当前累计音频，没有候选 TTL/永久保留；E 的
Oracle 只来自该 case 的 expected IDs，不进入 D。每个 chunk 记录 CTC Top-K/
置信度、实际注入、Prompt 生效 chunk、fixed prefix、回退 token IDs/文本、
unfixed/完整 partial、热词状态和文本 diff。

无强制对齐的原始集不会伪造声学结束时间，因此该集的 Recall/WER/CER
有效，基于声学结束的 latency 保持 `null`。边界资产默认强制覆盖
chunk 中间、边界前、跨边界、边界后、尾音、多词短语、跨多 chunk 长热词、
多热词和负例；缺类别时拒绝标记为完整基线。

本地不加载完整 Qwen/vLLM。已完成 fake backend 和纯 CPU 单测，覆盖空/
短于/等于/长于 2 秒、尾音、前两 chunk 无 prefix、tokenizer 级 5-token
回退、多字节文本、当轮候选、Oracle 隔离、边界分桶、时间线失败归因和
可恢复分片。H200 尚未完成的项目是安装版本 API 实测、50条 smoke、100条
完整原始集和人工/强制对齐边界集。

本地实际验证：

```text
Ruff（全仓库）: pass
Pytest 定向:    pass, 17 tests
Pytest 全仓库: pass, 153 tests
Mypy 新增四个 source 模块: pass
Mypy 全仓库: 仍有 11 个既有 Torch/类型注解错误，新模块 0 个
CLI --help smoke: pass
git diff --check: pass
```

完整工作区命令、输入路径、边界 spec 格式、输出和恢复规则见
`docs/STREAMING_RAG_EVAL.md`。第一轮必须保留 2/2/5 原始基线，报告后再决定是否
测 1 秒 chunk 或扩大 unfixed token 数。

## 0.14 2026-08-17 阿根廷/拉普拉塔西语：原始结构复核与专用转换器

西语新增两套只读来源，数据根目录为：

```text
/host_home/star/q00933266/data/es_ar_sources_v1
```

SLR61解压目录共有5,919个WAV。三个阿根廷索引文件均为无表头两列TSV，不能用
`csv.DictReader`读取，否则每个文件会少算首行。原始索引规模为female 3,921、
male 1,818、`es-ar` weather 90，共5,829行；实测weather的90个`source_id`已经全部
包含在female索引中，因此去重后是5,739条唯一阿根廷语音。库存中的另外180个WAV
分别是这90条`es-ar` weather的重复副本和90条`es-es`天气语音。转换器必须逐一验证
重复weather的文本和WAV内容相同，并明确排除`es-es`。`extracted/line_index.tsv`是
male索引的重复副本，也不得再次纳入。

Common Voice Rioplatense v26包含train 9,903、dev 266、test 224，共10,393条，和
`clips/`下10,393个MP3一致。该语料的地域标签覆盖阿根廷、乌拉圭、巴拉圭和玻利维亚
东部，不能宣称为纯阿根廷西语。必须保留官方train/dev/test和`client_id`，并审计
跨split说话人重叠；官方test后续保持封存，不能重分到训练集。

新增两个专用转换入口：

```text
scripts/convert_slr61_argentinian_to_tsv.py
scripts/convert_common_voice_rioplatense_to_tsv.py
```

两者输出规范TSV时均保留`source_id`、`speaker_id`、`source_split`、`language`、
`dialect`和来源元数据。主语言仍写`es`；方言只作为来源元数据记录为`argentinian`
或`rioplatense`，不把Rioplatense误标为纯`es-AR`。当前只做转换、全量音频/库存及
split审计；审计结果确认前不运行MFA G2P或完整Manifest。

工区拉取后先记录Common Voice归档SHA256，再运行转换：

```bash
ES_AR_DATA_ROOT=/host_home/star/q00933266/data/es_ar_sources_v1
ES_AR_OUTPUT_ROOT=outputs/es_ar_train_sources_v1

sha256sum \
  "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/downloads/es-Rioplatense.tar.gz" \
  > "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/downloads/common_voice_rioplatense_v26_sha256.txt"

python scripts/convert_slr61_argentinian_to_tsv.py \
  --source-root "$ES_AR_DATA_ROOT/slr61_argentinian_spanish" \
  --output-tsv "$ES_AR_OUTPUT_ROOT/slr61/source.tsv" \
  --check-audio \
  --scan-audio-inventory

python scripts/convert_common_voice_rioplatense_to_tsv.py \
  --corpus-root "$ES_AR_DATA_ROOT/common_voice_rioplatense_v26/extracted/es-Rioplatense" \
  --output-tsv "$ES_AR_OUTPUT_ROOT/common_voice_rioplatense_v26/source.tsv" \
  --check-audio \
  --scan-audio-inventory
```

先回传`slr61/slr61_conversion_summary.json`和
`common_voice_rioplatense_v26/common_voice_conversion_summary.json`。SLR61应读取
5,829个原始索引行、精确写入5,739条唯一语音，逐一验证并排除90条相同的`es-ar`
weather副本，再排除90条`es-es`，缺失、内容不一致和其他未索引WAV均为0；Common
Voice应精确写入10,393条、三组split规模不变、缺失/重复/未引用MP3为0。若Common
Voice出现跨split说话人重叠，转换器会返回`warn`；先决定是否沿用官方split或重新做
说话人隔离，不能直接继续G2P和Manifest。

首轮SLR61和Rioplatense MFA字典审计发现大量带重音或`ñ`的输入词没有生成发音，
且phone OOV在终端显示为空白组合符。新增`diagnose_spanish_mfa_audit.py`，用于区分
只去acute accent即可映射、必须去全部组合符才可映射和仍不可恢复的缺词，并把不可见
OOV的Unicode码点写入JSON。诊断只读现有字典和审计TSV，不修改或修复字典：

```bash
python scripts/diagnose_spanish_mfa_audit.py \
  --audit-dir outputs/es_ar_train_sources_v1/slr61/mfa_audit_v1 \
  --audit-dir outputs/es_ar_train_sources_v1/common_voice_rioplatense_v26/mfa_audit_v1
```

每个输入目录会生成`spanish_diagnostics.json`。诊断结果确认前不要构建西语Manifest，
也不要全局删除组合波浪号；葡语鼻元音仍依赖这些组合符。

实际诊断确认phone OOV只有`U+0303 COMBINING TILDE`，来自西语G2P的非音位性
鼻化；该符号后续只在西语修复字典内处理。缺词主要来自MFA模型不接受acute accent、
`ñ`和`ü`输入。acute accent可以用去重音代理词重新G2P，但不能把`ñ`直接折叠成
`n`，否则会把`/ɲ/`错误变成`/n/`。已加入小型代理拼写探针：

```bash
conda run --no-capture-output -n aligner mfa g2p \
  --num_pronunciations 1 \
  configs/phonemes/spanish_latin_america_repair_probe.v1.txt \
  models/mfa/g2p/spanish_latin_america_mfa.zip \
  outputs/spanish_latin_america_repair_probe.v1.dict

cat outputs/spanish_latin_america_repair_probe.v1.dict
```

探针用于比较`ny`/`ni`对`ñ`以及`gw`对`gü`的输出；结果确认前不生成正式修复字典。

## 0.13 2026-08-14 美式英语独立处理：MFA复审通过，待完整Manifest

美式英语 Swift JSON 已在工区独立转换并完成全量音频审计，输出目录为
`outputs/en_external_train_sources_v1/swift_us_english`。原始 JSON 及音频路径为：

```text
Swift JSON:
/host_home/z00841352/27A/data/en/json/swift_en_美式英语.json

Audio prefix:
/host_home/z00841352/27A/data/en/untar_files/美式英语
```

转换和音频审计均通过：389,738条记录全部写入，语言字段全部为`English`，
389,738个WAV全部存在，空audio/text、缺失和重复音频均为0。词表提取共得到
2,940,208个word token和46,041个唯一词，数字片段为0。

候选English US MFA模型和第一版生成字典的SHA256为：

```text
english_us_mfa.zip:
9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02

words.txt:
15546c3ab1dc7136a732bd25524e0681c88a4bd2e484915db1d03b4526d4ecfe

swift_us_english_english_us_mfa.v1.dict:
2dd3c3e045eabaeac4666a4a99cf472bf7a4ef816adf35943a072bd2262c23e2
```

MFA为46,041个输入词生成46,039个唯一发音，无额外词和重复发音；只缺`h`
（corpus count 3）和`lx`（count 1）。初次v0.2共享CTC词表审计还发现590个
发音包含同一个OOV符号`ʷ`，corpus-weighted count为12,460。该符号是MFA附在
辅音后的唇化修饰符，例如`kʷ`、`tʷ`和`ɟʷ`；共享词表已有普通`w`，因此phone
归一化现将`ʷ`展开为`w`，得到`k w`、`t w`和`ɟ w`，无需重新运行MFA G2P。

工区拉取本提交后只需重新运行字典审计：

```bash
python scripts/audit_mfa_dictionary.py \
  --words outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/words.txt \
  --word-counts outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/word_counts.tsv \
  --dictionary outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/swift_us_english_english_us_mfa.v1.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/en_external_train_sources_v1/swift_us_english/mfa_audit_v2
```

工区复审确认phone OOV及其corpus-weighted count均归零；词型和token覆盖率分别为
99.995656%和99.999864%。`training_labels_ready`只因为`h`和`lx`两个缺词保持
false。这4个corpus token不得静默删除，完整Manifest阶段应将包含它们的记录写入
`needs_review`。

英语词表另有1,717个含撇号或连字符的唯一词，共17,671个corpus token；这些词在
第一版字典中全部有且只有一个精确发音，缺失和歧义均为0。完整Manifest构建器因此
增加显式`--allow-exact-dictionary-connectors`：启用后不再仅因连接符把已有唯一
精确发音的英语词送入review，但字典缺失、多发音和phone OOV仍按原规则review。
默认关闭，既有葡语和西语策略不变；该策略写入build config和summary，不能与默认
策略的旧shard混用。

英语完整Manifest命令：

```bash
python scripts/build_full_training_manifest.py \
  --tsv outputs/en_external_train_sources_v1/swift_us_english/source.tsv \
  --audio-root /host_home \
  --dictionary outputs/en_external_train_sources_v1/swift_us_english/mfa_g2p/swift_us_english_english_us_mfa.v1.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/en_us_swift_full_manifest_v1 \
  --language en-US \
  --dataset swift_us_english \
  --id-prefix swift_us_english_row \
  --split unsplit \
  --allow-exact-dictionary-connectors \
  --shard-size 5000 \
  --workers 16
```

该构建仍不与西语或葡语合并，也不生成正式96/2/2 split；先回传summary、ready/review
记录数与小时数，再做Temporal 2×可恢复审计。

## 0.12 2026-08-13 英西葡混训前置：西语独立处理第一阶段（代码完成，待工作区审计）

英西葡混训不直接拼接原始数据，先分别完成英语和西语的独立TSV、音频、G2P、
字典和Manifest审计。当前西语已知输入只有两份只读Swift JSON：

```text
MLS Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/mls/swift_librispeech_es.json

Common Voice Spanish:
/data/h00911716/code/ms-swift/self_test/datalist/es/cv/swift_cv_es.json

Candidate G2P model:
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/g2p/spanish_latin_america_mfa.zip
```

尚未生成或确认西语TSV、MFA字典、覆盖率报告、ready/review Manifest。MLS Spanish
通常偏西班牙来源，Common Voice `es` 可能混合多个地区；当前统一只标记为`es`，
不声明`es-AR`或纯拉美西语。Latin America MFA模型先作为候选标签器，其G2P成功率
不能代替方言适配结论。

Swift转换输出绝对容器音频路径。旧TSV审计默认把任何绝对路径判为失败，已新增
显式`--allow-absolute-audio`：只有传入该开关才允许绝对路径，仍会逐条检查文件
存在性；默认相对路径安全策略不变。原Swift JSON始终只读，所有产物进入新的
`outputs/es_external_train_sources_v1`，不覆盖葡语或原始文件。

工作区按顺序运行，先不要并行：

```bash
python scripts/convert_swift_json_to_tsv.py \
  --input /data/h00911716/code/ms-swift/self_test/datalist/es/mls/swift_librispeech_es.json \
  --output-tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --expected-language Spanish \
  --audio-prefix-rewrite /home_92=/host_home \
  --check-audio \
  --progress-every 10000

python scripts/convert_swift_json_to_tsv.py \
  --input /data/h00911716/code/ms-swift/self_test/datalist/es/cv/swift_cv_es.json \
  --output-tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --expected-language Spanish \
  --audio-prefix-rewrite /home_92=/host_home \
  --check-audio \
  --progress-every 10000

python scripts/audit_training_tsv.py \
  --tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --audio-root /host_home \
  --allow-absolute-audio \
  --max-records 0 \
  --sample-count 5 \
  --output outputs/es_external_train_sources_v1/mls/audio_audit_full.json

python scripts/audit_training_tsv.py \
  --tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --audio-root /host_home \
  --allow-absolute-audio \
  --max-records 0 \
  --sample-count 5 \
  --output outputs/es_external_train_sources_v1/common_voice/audio_audit_full.json

python scripts/prepare_mfa_g2p.py \
  --tsv outputs/es_external_train_sources_v1/mls/source.tsv \
  --output-dir outputs/es_external_train_sources_v1/mls/mfa_g2p \
  --text-column text \
  --max-records 0 \
  --minimum-word-count 1

python scripts/prepare_mfa_g2p.py \
  --tsv outputs/es_external_train_sources_v1/common_voice/source.tsv \
  --output-dir outputs/es_external_train_sources_v1/common_voice/mfa_g2p \
  --text-column text \
  --max-records 0 \
  --minimum-word-count 1
```

需要返回两套`swift_json_conversion_summary.json`、`audio_audit_full.json`和
`mfa_g2p/summary.json`。只有两套转换均无unexpected language、音频审计缺失为0，
才继续各自运行MFA G2P、字典覆盖/phone OOV审计及完整Manifest；若路径前缀不匹配，
先根据转换报告修正映射，不得跳过音频检查。下一步是西语第二阶段G2P与Manifest，
不是三语合并或训练。

## 0.11 2026-08-04 v3结果复核与50条多关键词 Prompt 端到端验证（代码完成，待工作区运行）

工作区 v3 CTC 专项资产和评分已完成：210条自然 Validation case、7组目标全部满足、
500个热词、80个嵌套 family，primary audio 全部互异，`status=pass`。固定
`threshold=0.86/top_k=5` 的主要实测结果为：总体 Ranking Recall@5 95.37%；
Operating Precision 96.06%、Recall 83.17%、负例FPR 0%；3个独立热词 Recall@5
93.33%、All-3-Hit@5 80%，未达到本轮95%/85%的工程参考目标；组合非嵌套词
Recall@5 96.40%，高于单词91.67%，当前没有“组合词更难”的证据；short-only长词
Forced Top-5误排20%，但Operating误触发0%；总体 slot crowding loss为0，不过有3条
局部case出现family成员占位并挤掉独立真词。

复核发现旧报告有两处仅影响指标汇总、不影响CTC逐case评分的口径问题：

1. Longest-match Operating Precision 错把同family的短词冗余命中计为false positive；
2. 嵌套short/long专项指标使用全局family成员集合，可能把case里的独立词误归入其他family。

已修正为逐case family口径，并新增CPU-only报告重建工具；它复用已有
`hotword_case_scores_v3.jsonl`，不加载Head、不读取特征、不重复推理，也不覆盖原报告。
原报告保留用于审计，修正版另存：

```bash
python scripts/rebuild_multi_nested_hotword_report.py \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_case_scores_v3.jsonl \
  --base-report outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3.json \
  --output outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3_corrected.json
```

新增固定50条 Qwen3-ASR-1.7B Prompt 端到端评估：40正例+10负例，分组固定为
3独立词10、nested family+2独立词10、nested long 8、2独立词6、nested short 3、
单热词3、负例10。模型只加载一次，依次运行全部Baseline、仅有Operating候选时的
Retrieved Prompt、40条正例Oracle Prompt；无候选case直接复用Baseline，不重复推理。
最终目标采用Longest-match，contained short单独统计为redundant family hit，不当作
错误候选；真正无关候选的注入、写入和新增幻觉另行统计。CTC配置必须仍为固定
0.86/Top-5/.35/.25，不允许在本命令内调参；不读取sealed test。

物理GPU 5运行：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_multi_nested_prompt_eval.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --ctc-case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_case_scores_v3.jsonl \
  --ctc-report outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_evaluation_report_v3_corrected.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/prompt_multi_nested_v1 \
  --device cuda:0 \
  --dtype bfloat16
```

新增/修改文件：

- `src/qwen_hotword/hotwords/multi_nested.py`
- `src/qwen_hotword/inference/multi_nested_prompt.py`
- `scripts/rebuild_multi_nested_hotword_report.py`
- `scripts/run_multi_nested_prompt_eval.py`
- `tests/test_multi_nested_hotwords.py`
- `tests/test_multi_nested_prompt.py`

本地实际验证：Ruff全仓库pass；Pytest全仓库128 tests pass；新模块Mypy pass；
两个CLI `--help` smoke pass；`git diff --check` pass。fake/mock覆盖固定分组选择、
audio-disjoint、模型单次加载、Baseline/Retrieved/Oracle、嵌套冗余与真正错误分离、
原子输出及防覆盖。

预计Prompt输出：`sample_selection.json`、`baseline_predictions.jsonl`、
`retrieved_predictions.jsonl`、`oracle_predictions.jsonl`、
`multi_nested_prompt_report.json`。需要返回修正后的CTC报告和Prompt报告；若报告发现
具体幻觉或异常提升，再按其中case ID返回对应prediction行。当前限制：这是固定50条
Validation小样本，不是生产评估；没有工作区Prompt真实结果前不得宣称Prompt有效。
下一步只检查50条结果，再决定是否进入threshold/top-k调优或扩大正式评估。

## 0.10 2026-08-04 多关键词与组合/嵌套关键词专项评估（代码完成，待工作区运行）

本轮只实现 Validation CTC 专项评估，不训练模型、不提取 Encoder 特征、不读取
sealed test，也不运行 Qwen Prompt 推理。复用固定资产：

```text
Validation manifest: outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl
Validation cache:    outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation
Temporal 2× Head:    outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt
MFA dictionary:      outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict
```

新增独立 v3 资产构建器，严格从自然 Validation 文本选择约210条 audio-disjoint
case：1/2/3个独立关键词、nested short-only、long-present、nested family+2个独立词
及负例。组合词必须是连续完整词组；独立词 span 不重叠、无包含关系；嵌套 family
同时保存 containment 和 longest-match Ground Truth。每条 case 固定100个 active
hotwords，并记录规范化文本、真实 word span、family、困难负例和选择理由。输出目录
非空时拒绝覆盖；关键嵌套组少于10条时保留最大自然子集并标记
`smoke_insufficient_data`，不伪造样本或降低标准。无 speaker ID，因此只声明
audio-disjoint。

评分只复用现有 Validation cache 和 Temporal 2× Head，固定参数不可搜索：

```text
top_k=5, threshold=0.86, maximum_edit_ratio=0.35
posterior_weight=0.25, minimum_posterior_confidence=0.0
minimum_phonemes=4, minimum_top1_margin=0.0
time_axis=temporal_upsample_2x_only
```

报告严格区分 Forced Ranking Top-5 与 threshold/edit/posterior guard 后最多Top-5的
Operating 结果。包含总体/分组 Micro Recall@1/3/5、Any/All-Hit、All-3-Hit、Mean
Hits、Raw Precision@5、Operating P/R/F1、正例命中率和负例FPR；按 hotword form、
音素长度及 form×length 分桶；嵌套专项包含 short-only长词误触发、双GT、family
槽位、redundant hit、其他独立词Recall、slot crowding loss及具体归因case。报告明确
说明3个真实词而固定返回5个候选时 Raw Precision@5 的理论上限60%只是计算口径，
不是模型准确率上限。

本轮文件：

- `src/qwen_hotword/hotwords/multi_nested.py`
- `scripts/build_multi_nested_hotword_eval.py`
- `scripts/evaluate_multi_nested_hotwords.py`
- `tests/test_multi_nested_hotwords.py`
- `docs/HANDOFF.md`

本地实际验证：

```text
Ruff（全仓库）: pass
Pytest（全仓库）: pass, 126 tests
Mypy（multi_nested新模块）: pass
两个CLI --help smoke: pass
git diff --check: pass
```

工作区先运行CPU资产构建：

```bash
python scripts/build_multi_nested_hotword_eval.py \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested
```

先检查 `asset_summary_v3.json` 的实际分组数；无论 formal 或 insufficient_data，均可
继续运行固定口径 GPU 评分以获得 smoke 数据。物理 GPU 5 暴露为逻辑 cuda:0：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_multi_nested_hotwords.py \
  --validation-cache outputs/noah_pt_full_training_v1/features_ln_post_bf16/validation \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --dictionary outputs/noah_pt_mfa_g2p/noah_pt_portuguese_brazil_mfa.dict \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --checkpoint outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/ctc_head_best.pt \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_hotwords_v3.jsonl \
  --families outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/hotword_families_v3.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/multi_nested_cases_v3.jsonl \
  --asset-summary outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested/asset_summary_v3.json \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v3_multi_nested \
  --device cuda:0 \
  --batch-size 128
```

预计生成：

```text
multi_nested_hotwords_v3.jsonl
hotword_families_v3.jsonl
multi_nested_cases_v3.jsonl
sample_selection_v3.json
asset_summary_v3.json
hotword_case_scores_v3.jsonl
multi_nested_evaluation_report_v3.json
```

需要返回 `asset_summary_v3.json` 和 `multi_nested_evaluation_report_v3.json`；若要
逐case核查 crowding 再返回 `hotword_case_scores_v3.jsonl`。当前限制是本机没有真实
Validation/cache/checkpoint，尚未产生或宣称任何真实效果结论。下一步只根据CTC专项
结果决定是否运行50条多热词/组合词 Prompt 端到端验证。

## 0.9 2026-08-04 Temporal 2× 五语料合并训练集（工作区已完成）

Temporal 2× 只读审计已在工作区完成，5套语料合计结果：

```text
原 ready:              380,289 / 588.552262 h
纯时间恢复 ratio<=0.9: 166,757 / 227.142632 h
合并候选总计:          547,046 / 815.694894 h
```

工作区构建已完成并通过：train 525,189条/783.223637小时，validation
10,899条/16.239131小时，sealed test 10,958条/16.232128小时；全局重复ID、重复
音频和跨split overlap均为0，`source_manifests_modified=false`、`test_set_used=false`、
`status=pass`。后续不再把该合并manifest写成“待构建”。

用户决定不再只释放 Noah 500h，而是把以下5套语料的原 ready 与安全时间恢复集
一次性组成新的独立训练数据版本，按已有稳定 `split_hash` 做96/2/2：

```text
Noah 金融 200h
Noah 原 500h
MLS Portuguese
Common Voice Portuguese
FLEURS Portuguese
```

新构建器严格纳入两类记录：

1. 原 `train_ready.jsonl` 的全部 ready 记录；
2. `needs_review.jsonl` 中 issue 集合恰好只有 `ctc_length_infeasible`、Temporal
   2× 后可行且 effective ratio `<=0.90` 的记录。

任何 dictionary/connector/digit/standalone-h/empty-target 等其他问题仍然阻塞；
`(0.90,1.00]` 高压力样本和2×后仍不可行样本不释放。构建器只读源文件，输出
目录非空时拒绝覆盖，写入临时文件后原子替换；全局拒绝重复ID和重复绝对音频
路径，并检查跨split ID/音频重叠为0。输出保留 `source_corpus`、原始 language
和 `release_source`，MLS/CV 的 `pt` 不会被伪改为 `pt-BR`。

所有新记录明确写入：

```text
dataset_version: temporal2x-combined-v1
ctc_time_upsampling_factor: 2
estimated_ctc_input_length: 原 Encoder CTC 帧数
effective_ctc_input_length: 原帧数 * 2
```

feature-cache/训练边界已同步支持这一合约：恢复样本在缓存校验时按2×有效时间轴
判断可行；新缓存记录 time factor，训练时若使用小于数据要求的 Head factor 会
拒绝启动。旧 manifest 未写该字段时默认1，既有旧缓存元数据缺字段也按1兼容。

本轮文件：

- `src/qwen_hotword/training/combined_training.py`
- `scripts/build_temporal2x_combined_training.py`
- `tests/test_combined_training.py`
- `src/qwen_hotword/training/ctc_overfit.py`
- `src/qwen_hotword/training/feature_cache.py`
- `src/qwen_hotword/training/sharded_ctc.py`
- `tests/test_ctc_overfit.py`
- `tests/test_feature_cache.py`
- `docs/data.md`

本地验证：

```text
Ruff（全仓库）: pass
Pytest 定向（combined/feature/loader/sharded CTC）: pass, 24 tests
Pytest 全仓库: pass
Mypy（combined_training 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

工作区不需要GPU，从项目根目录运行：

```bash
python scripts/build_temporal2x_combined_training.py \
  --corpus noah_finance_200h=outputs/noah_pt_finance_200h/full_manifest_v1 \
  --corpus noah_500h=outputs/noah_pt_full_500h \
  --corpus mls=outputs/pt_external_train_sources_v1/mls/full_manifest_v1 \
  --corpus common_voice=outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard \
  --corpus fleurs=outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard \
  --output-dir outputs/pt_combined_temporal2x_v1 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --train-fraction 0.96 \
  --validation-fraction 0.02 \
  --test-fraction 0.02 \
  --progress-every 50000
```

预计输出：

```text
outputs/pt_combined_temporal2x_v1/full_ctc_train.jsonl
outputs/pt_combined_temporal2x_v1/full_ctc_validation.jsonl
outputs/pt_combined_temporal2x_v1/full_ctc_test.jsonl
outputs/pt_combined_temporal2x_v1/split_config.json
outputs/pt_combined_temporal2x_v1/split_summary.json
```

首先返回 `split_summary.json`。通过标准：总数547,046、原ready 380,289、恢复
166,757、总时长约815.694894小时、三split均非空、duplicate和cross-split overlap
全部为0、`source_manifests_modified=false`、`test_set_used=false`、status pass。
test输出生成后立即封存，不参与特征缓存、选模或调参。

当前限制：这是按样本自然比例的首个全集版本，尚未做corpus sampling weight；
MLS仍保留Portuguese来源身份，不能宣称为纯巴葡。下一步只在summary通过后缓存
新train/validation的Encoder特征，test不读取。

## 0.8 2026-08-03 Temporal 2× 训练语料恢复审计（工作区已完成）

用户决定先重新审计旧 full manifest 的时间筛选，再决定第一批释放量。本轮只读，
不生成训练 manifest、不修改原 ready/review、不缓存特征、不训练模型。审计顺序：

```text
Noah 金融 200 小时
Noah 原 500 小时
MLS
Common Voice
FLEURS
```

Temporal Head 的实际 `output_lengths` 已确认严格为原 Encoder CTC 长度乘 2。
本轮 effective ratio 定义为：

```text
ctc_minimum_input_length / (estimated_ctc_input_length * 2)
```

为保护 Noah 500 小时已经封存的 test，工具不逐行读取任何
`train_ready.jsonl`；原 ready 记录数/小时只从原 `summary.json` 获取。记录级
扫描只读取从未进入正式 train/validation/test 切分的 `needs_review.jsonl`，报告
显式记录 `ready_manifest_content_read=false`、`sealed_test_content_read=false`。

Review 分类严格互斥：

1. issues 恰好只有 `ctc_length_infeasible`：纯时间问题；
2. 纯时间问题在 2× 后拆为可恢复与仍不可行；
3. 可恢复再拆为 effective ratio `<=0.90` 的首批建议集，以及 `(0.90,1.00]`
   的高压力延后集；
4. 只要包含任何其他 issue，即使 2× 时间可行也归入“其他问题阻塞”，不得进入
   第一批恢复。

每套 corpus 报告包含原 ready/review 记录数与小时、2× 总可恢复、首批建议、
高压力延后、仍不可行、其他 issue 阻塞、effective ratio 分桶、每种 issue
总量、精确 issue 组合和两两交集。所有分类同时记录数量、已知小时和缺 duration
数量；输入 summary/review 记录 SHA256，ready 只记路径和大小、不读取或计算
SHA256。输出目录非空时拒绝覆盖。

首次工作区运行在 Noah 金融 review 第 3,507 行发现合法的“只有
`estimated_ctc_input_length`、没有 `ctc_minimum_input_length`”记录。这是
`empty_ctc_target`/标签组装失败一类记录的正常 partial metadata，不是数据损坏。
审计器已修正：非时间问题的 partial length 计入 ratio unavailable 并继续保持
阻塞；纯 `ctc_length_infeasible` 候选仍强制要求两个长度字段完整。

本轮代码：

- `src/qwen_hotword/training/temporal_recovery.py`
- `scripts/audit_temporal2x_recovery.py`
- `tests/test_temporal_recovery.py`

本地验证：

```text
Ruff（全仓库）: pass
Pytest 定向: pass
Pytest 全仓库: pass
Mypy（temporal_recovery）: pass
CLI --help smoke: pass
git diff --check: pass
```

工作区无需 GPU，按用户指定优先级运行：

```bash
python scripts/audit_temporal2x_recovery.py \
  --corpus noah_finance_200h=outputs/noah_pt_finance_200h/full_manifest_v1 \
  --corpus noah_500h=outputs/noah_pt_full_500h \
  --corpus mls=outputs/pt_external_train_sources_v1/mls/full_manifest_v1 \
  --corpus common_voice=outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard \
  --corpus fleurs=outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard \
  --output-dir outputs/temporal2x_recovery_audit_v1 \
  --time-upsampling-factor 2 \
  --release-max-effective-ratio 0.90 \
  --progress-every 50000
```

预计输出：

```text
outputs/temporal2x_recovery_audit_v1/summary.json
outputs/temporal2x_recovery_audit_v1/noah_finance_200h.json
outputs/temporal2x_recovery_audit_v1/noah_500h.json
outputs/temporal2x_recovery_audit_v1/mls.json
outputs/temporal2x_recovery_audit_v1/common_voice.json
outputs/temporal2x_recovery_audit_v1/fleurs.json
```

审计结束后先检查六个小 JSON，再决定第一批释放量。倾向方案保持为“旧训练集 +
纯时间问题、2× 后 effective ratio <=0.90 的恢复集”；本轮不创建该合并版本。

## 0.7 2026-07-31 Retrieved RAG 端到端验证（代码完成，待工作区运行）

本轮把已经完成的两段链路真正接起来：

```text
预生成 validation CTC case scores
  -> 固定 threshold=0.86 / top-k=3 / margin=0
  -> 热词 Prompt
  -> Qwen3-ASR-1.7B 最终转写
  -> Baseline / Retrieved / Oracle 归因
```

只做流程 smoke，不搜索 threshold/top-k，不训练 Encoder/CTC Head，不读 sealed
test，不修改 Qwen 模型。CTC 候选除了 `score >= 0.86`，继续使用正式评分阶段的
`edit_ratio <= 0.35` 与 `minimum_posterior_confidence=0`；不能把阈值简化为只
过滤 score。

现有 500 条 score 文件按本轮固定 `top-k=3` 复算为 Precision 93.47%、
Recall 88.48%、负例 case FPR 2.8%。此前记录的 93.29% / 89.89% 是
`top-k=5` 口径；两者没有冲突，本轮只是先固定 top-k=3 跑通流程。

上一轮 Prompt smoke 已在工作区完成。40 条 validation case 的实际结果为：

```text
Baseline: 44/48, hotword recall 91.67%, positive case hit 30/30
Oracle:   44/48, hotword recall 91.67%, absolute gain 0
Negative Prompt: 0/10 错误热词写入，幻觉率 0
Model load count: 1
```

这证明 `Qwen3ASRModel.transcribe(..., context=prompt)` 接口、固定葡语模板和安全
控制可以运行，但未证明 Oracle Prompt 在当前常见词样本上有收益。Oracle 仍漏
`pra vocês`（2 次）、`pode pausar`、`então vamo`。因此本轮完成标准是链路与
归因正确，不以显著 Recall 提升作为通过条件。

### 实现

- `src/qwen_hotword/inference/retrieved_rag.py`
  - 严格校验 validation manifest、v2 case、hotword table 与 CTC score 一致；
  - 确定性选择 60 正例、40 负例；正例覆盖短/中/长热词和单/多热词；
  - 负例优先纳入全部 threshold 触发 case，再用固定 seed 补足，专门观察错误
    候选注入后的最终转写污染；
  - Baseline 跑全部 100 条；有候选时才跑 Retrieved Prompt，无候选直接复用
    Baseline；60 条正例另跑 Oracle；
  - 模型只加载一次，逐次打印 phase、累计调用、耗时、速度和 ETA；
  - 记录 CTC 检索 Precision/Recall/FPR、最终热词 Recall/case hit、检索漏召回、
    检索正确但 Decoder 未写出、相对 Baseline 的热词救回、错误候选写入、
    文本变化和简单 corpus WER；
  - 五个输出统一原子写入，非空目录拒绝覆盖，记录所有输入 SHA256。
- `scripts/run_retrieved_rag.py`
- `tests/test_retrieved_rag.py`

本地 fake inference 覆盖阈值与 edit guard、margin、确定性分层选样、CTC
误触发负例、模型单次加载、无 Prompt Baseline 复用、多热词 Prompt、三路
Recall、错误候选写入归因、WER、进度和防覆盖。

本地实际验证：

```text
Ruff（全仓库）: pass
Ruff format（本轮文件）: pass
Pytest 定向: pass
Pytest 全仓库: pass
Mypy（retrieved_rag 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

### 工作区运行

先拉取交付分支，然后从项目根目录在物理 GPU 5 运行：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_retrieved_rag.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotwords_v2.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotword_cases_v2.jsonl \
  --ctc-case-scores outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/scoring_temporal2x_v2/hotword_case_scores.jsonl \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1 \
  --threshold 0.86 \
  --top-k 3 \
  --minimum-top1-margin 0 \
  --device cuda:0
```

输出：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/sample_selection.json
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/baseline_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/retrieved_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/oracle_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/retrieved_rag_v1/retrieved_rag_report.json
```

需要优先返回 `retrieved_rag_report.json`；若要逐 case 归因，再返回另外四个小
文件。当前限制：v2 是 validation 模拟常见词，Baseline 很高；负例选样故意
富集 CTC 误触发，所以选中 40 条负例内部的误触发率不能当作无偏 FPR，报告另
保留全 500 case 的正式 FPR。CTC score 是上一阶段离线生成，本轮尚未实现在线
registry/reload。

下一步只检查工作区端到端结果；threshold/top-k 调优留到后续独立实验。

## 0.6 2026-07-30 三套葡语 Swift JSON（第一版独立处理完成）

Noah 金融 200 小时第一版 full manifest 已完成：

```text
Source:          142,985 / 195.3711 h
Training-ready:   86,614 / 119.5436 h
Needs review:     56,371
Status:           pass

ctc_length_infeasible: 53,699
dictionary_missing:     3,390
unresolved_connector:   3,377
standalone_h:               10
empty_ctc_target:            8
```

MFA corpus-token coverage 为 99.8380%，v0.2 phone OOV 为 0。缺词主要与连字符
重合。金融数据先保留这版结果，未来用 2× 时间轴与连接词解析恢复，不阻塞三套新
语料。

FLEURS、MLS、Common Voice 葡语 Swift JSON 已分别完成转换、词表、巴葡 MFA
候选 G2P、v0.2 audit 和完整 manifest。三者独立保存、全部作为 train 候选；
没有修改源 JSON/音频，没有建立 validation/test，也没有合并或缓存 Encoder
特征。

本轮更新 Swift JSON 转换器：

- 支持重复传入 `--audio-prefix-rewrite OLD=NEW`；
- 只匹配完整路径前缀边界；
- 在改写后执行 `--check-audio`；
- summary 记录 rewrite 配置、改写数量、缺失音频和耗时；
- 大文件加载前后打印状态，每 10,000 条打印转换速度和累计结果；
- 检测到语言不一致或缺失音频时 summary 标记 `warn`。

工作区转换与词表实际结果：

```text
Corpus          Records   Audio   Word tokens   Unique words   Digit fragments
FLEURS            2,793   WAV          60,947          7,743               847
MLS              37,533   FLAC      1,261,190         75,392                 0
Common Voice     22,923   MP3         154,407         26,593               845
```

三套均 `status=pass`，全部 63,249 条音频在路径改写后存在，无跳过或语言异常。
FLEURS/Common Voice 中的数字不能从 CTC 标签中静默丢失，因此 full manifest
新增 `unresolved_digit` review 原因；暂不自动决定年份、金额或序数的葡语读法。
该保护只影响后续新 manifest，不改动原始 JSON、TSV 或已有 Noah v1 输出。

MFA audit：

```text
Corpus          Token coverage   Missing words   Duplicate entries   Phone OOV
FLEURS                99.4192%             124                   1           0
MLS                    98.8174%           2,294                 305           0
Common Voice           99.2889%             802                  12           0
```

最终 full manifest：

```text
Corpus          Source records/h       Ready records/h       Review
FLEURS          2,793 / 10.1789 h       1,966 / 6.8475 h         827
MLS            37,533 / 160.9632 h     26,030 / 110.3132 h     11,503
Common Voice   22,923 / 26.4790 h      21,803 / 24.9761 h       1,120
Total          63,249 / 197.6211 h     49,799 / 142.1368 h     13,450
```

正式使用路径：

```text
FLEURS:
outputs/pt_external_train_sources_v1/fleurs/full_manifest_v2_digitguard

MLS:
outputs/pt_external_train_sources_v1/mls/full_manifest_v1

Common Voice:
outputs/pt_external_train_sources_v1/common_voice/full_manifest_v2_digitguard
```

FLEURS 的旧 `full_manifest_v1` 未启用数字保护，只保留历史对照，后续不得用于
训练。Common Voice v1 的 ready 数量虽与 v2 相同，正式引用也固定为 v2。

主要遗留问题：

1. FLEURS 的 839 个、Common Voice 的 350 个 `unresolved_digit` issue 在有
   上下文安全的数字读法规则前保留 review。
2. MLS 有 14,698 个 connector issue，导致大量记录受影响；旧拼写和复合词修复
   需新建恢复版本，不能覆盖 v1。
3. MLS 使用巴葡 MFA 仅证明 phone/vocab 技术兼容，不能证明所有说话人是 pt-BR；
   合并到巴葡训练前仍需元数据或跨说话人音频抽查。
4. 三套与 Noah 数据尚未做跨语料音频/文本去重，也未决定训练采样权重。

下一步只在用户决定合并策略后，构建 train-only 合并 manifest；在此之前不要
缓存特征或启动新 CTC 训练。

## 0.5 2026-07-29 Noah 200 小时巴葡金融数据（代码就绪，待首轮审计）

用户确认新增的 200 小时金融领域数据是巴西葡萄牙语，可复用 Noah 500 小时的
巴葡 MFA 和完整 CTC manifest 流程。该数据先独立处理、全部作为 train 候选，
暂不与旧 500 小时数据合并，也不建立新的 validation/test。

用户给出的宿主机 TSV 为：

```text
/home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv
```

容器内候选路径按现有挂载规则暂定为：

```text
TSV:
/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv

Audio root candidate:
/host_home/z00841352/27A/data/Noah_espt/noah_pt
```

现有 full manifest builder 原先写死
`dataset=noah_pt_full_500h`、`id=noah_pt_row_*` 和 `split=unsplit`，直接复用会
导致新旧语料身份错误和 ID 冲突。本轮已给
`scripts/build_full_training_manifest.py` 和
`src/qwen_hotword/training/full_manifest.py` 增加：

```text
--dataset
--id-prefix
--split
```

旧默认值和旧 500 小时 `build_config.json` 的 resume 兼容性保持不变。新数据
固定使用：

```text
dataset:   noah_pt_finance_200h
id prefix: noah_pt_finance_200h_row
language:  pt-BR
split:     train
```

第一步只运行 1,000 行只读审计，不直接开始 MFA 长任务：

```bash
python scripts/audit_training_tsv.py \
  --tsv "/host_home/z00841352/27A/data/Noah_espt/tsv/pt_tsv/200小时巴西葡萄牙语金融领域口语化语音数据.tsv" \
  --audio-root /host_home/z00841352/27A/data/Noah_espt/noah_pt \
  --max-records 1000 \
  --sample-count 5 \
  --output outputs/noah_pt_finance_200h/audit_first_1000.json
```

需要返回：

```text
outputs/noah_pt_finance_200h/audit_first_1000.json
```

通过标准：字段存在、1,000 行均有 audio/text、音频解析 1,000/1,000、缺失为
0、绝对 audio 值为 0。若失败，先根据 report 中样本纠正 audio root，不启动
G2P。

本地定向验证：

```text
Ruff: pass
Pytest（full manifest + G2P prep + MFA audit）: pass, 9 tests
CLI --help smoke: pass
git diff --check: pass
```

首轮审计通过后的下一步：全量审计 → 独立 word list → 巴葡 MFA G2P →
dictionary/vocab audit → 独立 full manifest。具体数据边界同步记录在
`docs/data.md`。

## 0.4 2026-07-29 Prompt Injection 最小验证（代码完成，待工作区运行）

本轮目标是关键词 RAG 的第一步，仅在 formal validation 上比较：

```text
Baseline:                40 条音频，不注入热词
Oracle Prompt:           30 条正例，注入该音频真实包含的热词
Negative Prompt Control: 10 条负例，各注入 1 个严格不在参考文本中的热词
```

不读取 CTC `hotword_case_scores`，不接 Retrieved RAG，不使用 sealed test，不训练
Encoder/CTC Head，也不修改 Qwen 模型结构。固定 seed `20260729` 确定性选样；
正例按 4–7、8–12、13+ 音素三档选择，并混合单热词和多热词 case。最终选择完整
写入 `sample_selection.json`。

### 已确认的官方 Prompt 接口

工作区固定目标仍是模型 `Qwen3-ASR-1.7B`；`qwen-asr==0.0.6` 是 Python
推理库版本，不是模型大小。已直接检查该库的真实接口：

```python
Qwen3ASRModel.transcribe(
    audio,
    context="",
    language=None,
    return_time_stamps=False,
)
```

实际参数名为 `context`。官方 `_build_messages` 将 `context` 放入 system
message，将音频放入 user message。本轮通过现有 `load_asr_model` 只加载一次
模型，固定 `language="Portuguese"`、`return_time_stamps=False`，不额外覆盖
beam、sampling 或其他 `generate` 参数；运行报告会记录 wrapper backend、
`max_new_tokens` 和 `max_inference_batch_size` 的实际值。

固定且唯一的葡萄牙语模板为：

```text
As palavras a seguir podem aparecer no áudio e servem apenas como referência de grafia. Use-as somente se forem realmente faladas; não as inclua à força na transcrição: {hotwords}
```

Baseline 传空 `context`。Oracle 和 Negative Control 均使用同一模板；热词只是
拼写参考，不描述为必须输出。

### 实现文件

- `src/qwen_hotword/inference/hotword_prompt.py`
  - NFKC、casefold、去标点、空格规范化和严格完整单词/连续词组匹配。
  - 不扩展单复数、口语 alias；`coisa` 不匹配 `coisas`，
    `relacionamento` 不匹配 `relacionamentos`。
- `src/qwen_hotword/inference/prompt_smoke.py`
  - validation-only 校验、确定性分层选样、三路推理、指标计算、逐条进度和
    原子输出；非空输出目录拒绝覆盖。
  - Baseline 只计算一次并复用于 Oracle/Negative 对照。
  - 记录模型与四个输入文件的路径、大小和 SHA256。
- `scripts/run_hotword_prompt_smoke.py`
- `tests/test_hotword_prompt_smoke.py`

### 本地实际验证

本地测试使用 fake inference，不加载 1.7B 权重：

```text
Ruff（全仓库）: pass
Ruff format（本轮文件）: pass
Pytest 定向: pass, 7 tests
Pytest 全仓库: pass, 108 tests
Mypy（两个 inference 新模块）: pass
CLI --help smoke: pass
git diff --check: pass
```

覆盖 Baseline 空 Prompt、Oracle 单/多热词 Prompt、负例错误 Prompt、空热词、
模型只加载一次、严格词匹配、Recall/幻觉率、确定性选样、进度开关结果一致、
防覆盖、非 validation split 拒绝和 sealed test 拒绝。

### 工作区运行命令

从项目根目录在物理 GPU 5 运行；它在进程内映射为逻辑 `cuda:0`：

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_hotword_prompt_smoke.py \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B \
  --validation-manifest outputs/noah_pt_full_training_v1/full_ctc_validation.jsonl \
  --vocab configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json \
  --hotwords outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotwords_v2.jsonl \
  --cases outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/stratified_hotword_cases_v2.jsonl \
  --output-dir outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1 \
  --device cuda:0
```

每完成一次推理会打印 phase、累计完成数、耗时、cases/s 和 ETA。总计 80 次：
40 次 Baseline、30 次 Oracle、10 次 Negative Control。

需要返回并检查以下五个小文件：

```text
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/sample_selection.json
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/baseline_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/oracle_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/negative_prompt_predictions.jsonl
outputs/noah_pt_full_training_v1/simulated_hotword_eval_v2_stratified_100/prompt_smoke_v1/prompt_smoke_report.json
```

当前限制：尚无工作区真实推理结果，因此不能宣称 Prompt 有效；这是 40 条
validation Oracle smoke，不是业务验收、Retrieved RAG 或完整误触发评估。

下一步：接入Retrieved RAG小规模评估。

## 0.3 2026-07-29 热词评分阶段收尾（当前状态）

v2 已在工作区完成：100 个热词、250 个正例、250 个负例，每条 case 激活完整
100 词 registry。原始排序结果为 Recall@1/3/5 =
67.98% / 96.63% / 99.44%，Top-1 正例 case 命中率 96.8%。

完整 `hotword_case_scores.jsonl` 复算确认，旧报告在 threshold=0.90 时的
Recall=43.26% 主要不是 Head 或阈值问题，而是
`minimum_top1_margin=0.03` 会在 Top-2 接近时清空整条多热词 case：

```text
旧策略（margin=0.03）: Precision 96.25%, Recall 43.26%, negative FPR 2.4%
关闭 margin，阈值 0.90: Precision 95.41%, Recall 87.64%, negative FPR 2.4%
关闭 margin，阈值 0.86: Precision 93.29%, Recall 89.89%, negative FPR 2.8%
```

margin 共误杀 75 个正例 case、158 个已经过阈值的正确热词；67/75 case 的
Top-1 和 Top-2 都是正确热词，且 margin 没有减少任何负例 case 误触发。因此
多热词默认 margin 已改为 0.0，仍保留 CLI 参数供单标签实验显式启用。默认
threshold sweep 补入 0.86。

评分现会打印 Head 加载、每个 feature shard 的累计 case、耗时、cases/s、ETA
以及输出路径，并在报告中记录 scoring/evaluation wall seconds。

6 个严格词面负例触发均有明确文本来源：`coisa` 匹配 `coisas` 3 次、
`relacionamento` 匹配 `relacionamentos` 2 次、`vamos` 匹配口语 `vamo`
1 次。它们主要是局部音素子串与数据标签口径不一致，并非随机声学误触发。
Top-5 仅漏 `design` 和 `ruim é` 各 1 次。

遗留问题（进入下一阶段时保留）：

1. 正式业务热词、人名/品牌、speaker-disjoint 与困难负例尚未提供；v2 只用于
   validation 开发，不是最终业务验收集。
2. 单复数和口语变体应通过 registry 显式 alias/pronunciation 管理，不能默认
   对所有品牌、人名开放子串匹配。
3. 0.86/0.90 都只是 validation 候选工作点，正式阈值需在未来业务集确认；已
   消耗的 sealed CTC test 不得用于热词调参。
4. 下一阶段按项目路线进入在线 hotword registry/reload 与 Qwen prompt
   injection，再评估最终转写热词命中率、普通词退化和部署延迟。

本轮代码：

- `src/qwen_hotword/hotwords/scoring.py`
- `src/qwen_hotword/hotwords/evaluation.py`
- `scripts/evaluate_hotword_scoring.py`
- `tests/test_hotword_scoring.py`

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy（scoring/evaluation，skip imports）: pass
Pytest 定向: pass, 10 tests
Pytest 全仓库: pass, 101 tests
CLI --help smoke: pass
git diff --check: pass
```

## 0.2 2026-07-27 v2 分层模拟热词与 Recall@K（已完成）

用户确认 v1 的 50 个 validation 模拟热词只完成了链路 smoke test，不能作为
正式热词评估集。v1 的主要偏差是全部热词仅出现一次、音素长度为 14–24，
缺少短词、中等长度词和较高频词。

本轮新增独立 v2，不覆盖 v1：

- `scripts/build_stratified_hotwords.py`
- `build_stratified_hotword_assets` in
  `src/qwen_hotword/hotwords/simulation.py`
- `evaluate_hotword_ranking` and length-bucket Recall@K in
  `src/qwen_hotword/hotwords/evaluation.py`
- `--ranking-ks` in `scripts/evaluate_hotword_scoring.py`

v2 固定构造 100 个热词：

```text
4–7 phonemes:   30
8–12 phonemes:  40
13–18 phonemes: 20
19–24 phonemes: 10
```

每个长度桶混合 occurrence=1、occurrence=2–5 和 occurrence>=6 的候选；不足时
从同长度桶其他频率候选补齐。可通过 `--exclude-hotwords` 排除 v1 的词面和完全
相同发音。v2 输出目录必须为空，任何已有结果都拒绝覆盖。

case 构造会先做 coverage selection，保证 100 个热词都至少进入一个正例 case；
默认生成 500 个 case。每条 case 激活完整 100 词 registry，因此 Recall@1、
Recall@3 和 Recall@5 是在全部 100 个候选上的真实排序指标，不是从较小随机
候选集计算。报告同时输出整体 Recall@K 和四个音素长度桶的 Recall@K。该排名
指标不加 score threshold；原 threshold sweep、Precision 和负例 FPR 报告继续
保留并与排序能力分开解释。

当前数据仍不能保证人名/品牌类别或 speaker-disjoint，因为 formal validation
manifest 没有实体类别和 speaker ID。v2 是普通难度的代表性开发集，不是最终
业务验收集；仍不读取已经消耗的 formal CTC test 集。

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy（本轮两个模块，skip imports）: pass
Pytest 定向: pass, 8 tests
Pytest 全仓库: pass, 99 tests
CLI --help smoke: pass
git diff --check: pass
```

工作区下一步：

1. 在新目录 `simulated_hotword_eval_v2_stratified_100` 构建 v2。
2. 人工查看 100 词表和 summary 的长度/频率分布。
3. 用固定 2x temporal best Head 评估全部 500 个 validation case。
4. 返回 `stratified_hotword_summary_v2.json` 和
   `hotword_scoring_report.json`，重点读取整体及分长度 Recall@1/3/5。

## 0.1 2026-07-25 sealed test PER 一次性评估（已完成）

用户已明确要求获取当前固定 CTC 模型的正式 test PER，因此允许首次打开此前
封存的 `full_ctc_test.jsonl`。此次评估必须保持以下冻结条件：

```text
checkpoint:
  outputs/noah_pt_full_training_v1/run_temporal_upsample_ctc_h512_k5_lr3e4_v1/
  ctc_head_best.pt

Head:
  temporal_upsample, hidden=512, kernel=5, dropout=0.1, time axis=2x

decode:
  greedy argmax CTC collapse, blank_id=0
```

不得根据 test PER 重新选择 checkpoint、调整 Head、修改解码策略或调参。后续若有
新模型版本，必须建立新的正式评估版本和新的独立测试协议，不能反复使用本次结果
进行开发。

本轮新增：

- `src/qwen_hotword/training/sealed_test.py`
  - 直接对 test 音频分块提取冻结的 `ln_post` 特征并立即评估。
  - 不写 test feature cache，不保留可用于反复调参的测试特征。
  - 只接受文件名为 `ctc_head_best.pt` 的 2x temporal checkpoint。
  - 记录 test PER、loss、substitution/deletion/insertion、预测/参考长度比、
    blank ratio、高频错误以及 manifest/vocab/model/checkpoint SHA256。
  - 报告已存在时拒绝覆盖。
- `scripts/evaluate_sealed_ctc_test.py`
  - 必须显式传 `--acknowledge-sealed-test-evaluation`。
  - 只接受 `experiment=full-ctc-v1, split=test` manifest。
  - 输出明确标记 `test_set_used=true`、`one_time_evaluation=true` 和
    `checkpoint_selection_or_tuning_permitted=false`。
- `tests/test_sealed_test.py`
  - 覆盖 2x temporal best checkpoint 的一次性流式评估。
  - 覆盖报告防覆盖和 latest checkpoint 拒绝。

本地实际验证：

```text
Ruff（本轮文件）: pass
Mypy sealed_test.py: pass
Pytest 定向: pass, 2 tests
Pytest 全仓库: pass, 97 tests
CLI --help smoke: pass
git diff --check: pass
```

工作区已于 2026-07-25 在 GPU 5 完成唯一一次正式评估：

```text
test samples:       4,860
test loss:          0.309711
test PER:           0.0677893 (6.7789%)
validation PER:     0.0676448 (6.7645%)
val-test gap:       0.0144 percentage points
sub/del/ins:        6,510 / 6,646 / 3,256
prediction/reference length ratio: 0.9860
blank frame ratio:  0.4038
status:             pass
```

checkpoint SHA256:
`abaadac43c40daf8e2eee339653c64bfafa44fd0267eb7930449bd8d927de774`。
test manifest SHA256:
`a00f111643d75a33884a73ab7e21f520e7dd4e744f56b09a41c51b20da10dedf`。
Test 与 validation 几乎一致，没有明显过拟合。该 test 已消耗，不得再用于当前
checkpoint 或解码策略的选择和调参。

## 0. 2026-07-22 最新状态（后续交接以本节为准）

### 当前目标与已确认决策

时间上采样冻结 Encoder CTC Head 已完成正式训练和 validation 诊断。
用户已确认：

- 部署时只使用 `time_upsampling_factor=2` 的新 Head 时间轴。
- 原始 1× 线性 Head 只作历史对照和研究分析。
- 暂不修改 CTC 压力分桶报告，不让该报告阻塞产品路径。
- 当前进入 phoneme-space hotword scoring 和误触发控制阶段。
- 领导尚未提供正式热词表，先从 formal validation 文本和 Noah MFA 词典构建
  可复现的 pt-BR 模拟热词表，用于打通评分链路。
- 本阶段仍不读取封存 test 集。模拟 validation 结果只用于开发和阈值初调，
  不得宣称为最终泛化结论。

### 工作区已完成的新 Head 实验

2026-07-21 工作区完整训练结果：

```text
run:              run_temporal_upsample_ctc_h512_k5_lr3e4_v1
head:             temporal_upsample, hidden=512, kernel=5, dropout=0.1, 2×
trainable params: 838,746
best epoch:       24
train loss/PER:   0.246010 / 0.066503
validation loss:  0.305222
validation PER:   0.067645
early stop:       true, validation_loss patience=6
test used:        false
status:           completed
```

与旧线性 Head 的 validation 诊断对比：

```text
linear best:    PER 0.293637, deletion 52,016, substitution 16,052,
                insertion 2,636, prediction/reference length 0.7949
temporal 2×: PER 0.067645, deletion 6,812,  substitution 6,557,
                insertion 2,919, prediction/reference length 0.9838
```

这证明新 Head 的时间对齐能力明显更强，并已达到进入热词评分阶段的标准。

### 本轮已实现（待工作区运行）

- `src/qwen_hotword/hotwords/registry.py`
  - 定义可序列化的热词条目。
  - 校验热词 ID、语种、词面、MFA 发音、phoneme token 与 token ID 一致性。
  - 拒绝 blank、越界 ID、重复 ID 和重复发音。
- `src/qwen_hotword/hotwords/simulation.py`
  - 只接受 `split=validation` 记录。
  - 从 1–2 词 validation 短语构造确定性 pt-BR 模拟热词表。
  - 用 Noah MFA 词典和当前 v0.2 词表生成精确 phoneme token IDs。
  - 生成 positive-confusable 和 negative 验证 case，每条 case 有自己的在线
    active hotword 集合。
- `src/qwen_hotword/hotwords/scoring.py`
  - 在 Head 有效时间轴上做 CTC greedy collapse。
  - 用局部音素编辑距离和 posterior confidence 对热词排序。
  - 支持 score threshold、最大 edit ratio、最低 posterior、top-k 和 top-1
    margin 歧义抑制。
- `src/qwen_hotword/hotwords/evaluation.py`
  - 强制 checkpoint 必须是 2× `TemporalUpsampleCtcHead`。
  - 只从 validation feature cache 取模拟 case，不读 test。
  - 输出 precision、recall、F1、positive case hit/top-1 accuracy、negative case
    false-positive rate 和阈值扫描。
  - 默认控制目标是 precision >= 0.90 且 negative-case FPR <= 0.03；未达标时
    会显式写 `meets_control_targets: false`。
- 新 CLI：
  - `scripts/build_simulated_hotwords.py`
  - `scripts/evaluate_hotword_scoring.py`
- 新测试：
  - `tests/test_hotword_scoring.py`
  - `tests/test_simulated_hotwords.py`

### 本轮本地测试

2026-07-22 实际结果：

```text
Ruff（本轮相关文件）: pass
Pytest 定向:                 pass, 6 tests
Pytest 全仓库:             pass, 91 tests
CLI --help smoke:              pass
git diff --check:              pass
Mypy src/qwen_hotword:         11 existing errors, 0 in new hotword modules
```

Mypy 的 11 个既有错误集中在 `qwen_backbone.py`、`ctc_head.py` 和四个训练器的
Torch 类型注解；本轮未扩大范围修改它们。

### 下一步（当前最高优先级）

1. 在工作区从 formal validation manifest 生成 50 个模拟热词和 200 个
   validation-only case。
2. 人工快速查看 `simulated_hotwords.jsonl` 的词面和发音是否合理。
3. 在 GPU 5 上用新 Head best checkpoint 运行 hotword threshold sweep。
4. 将 `hotword_scoring_report.json` 和必要的失败 case 发回分析。
5. 根据报告固定第一版阈值与误触发策略，再接入可在线 reload 的正式
   hotword registry 和 Qwen prompt injection。

### 工作树与保留修改

本轮开始时实际本地基线为 `main@33fe87a`，用户已确认继续在 `main`
工作。本地 `main` 与 GitHub 发布分支历史仍不同步；发布时应继续采用隔离
worktree，将本阶段独立 commit 移植到 `origin/codex/g2p-coverage-scan`，不应强推
本地 `main`。

以下是其他任务/用户的未提交修改，必须继续保留，不得纳入本阶段 commit：

- `docs/PHONEME_VOCAB.md`
- `docs/WORKZONE_RUNBOOK.md`
- `scripts/scan_g2p_coverage.py`
- `tests/test_g2p_coverage.py`
- `tests/test_phoneme_vocab.py`
- `configs/phonemes/en_es_ptbr_fr_id_precision_ipa_vocab.v0.3.json`
- `work/`

## 1. 上一阶段目标（历史记录）

在 Qwen3-ASR-1.7B Audio Encoder 保持冻结、复用既有 `ln_post` BF16 特征缓存的
前提下，将正式分片 CTC 训练链路从仅支持线性 Head 扩展为可选的时间上采样
Head，以验证更高时间分辨率和局部上下文能否降低验证集 PER，尤其是删除错误。

目标结构已实现：

```text
thinker.audio_tower.ln_post
-> LayerNorm
-> 确定性时间上采样（默认 2 倍 repeat_interleave）
-> 1x1 投影 + 轻量 depthwise 时序卷积 + 1x1 上下文投影
-> GELU + Dropout
-> Linear(hidden_dim, 90)
-> CTC Loss
```

## 2. 实际工作树与交接冲突

2026-07-21 恢复任务时的实际状态：

```text
branch: main
HEAD:   56f2c37 Add unfrozen encoder CTC trainer
remote: main...origin/main [ahead 28, behind 5]
```

上一版 HANDOFF 记录的 `unfrozen-encoder-ctc@d36ba23` 与实际工作树不一致。
`d36ba23` 上的以下诊断文件在当时的 `main` 中不存在：

- `scripts/diagnose_frozen_ctc.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `tests/test_ctc_diagnostics.py`

本轮没有在带未提交修改的情况下切换分支，而是以 `main@56f2c37` 为真实基线，
将必要诊断能力移植到当前工作树，并更新为支持新 Head。未切换、回退、删除
或覆盖其他任务的未提交修改。

## 3. 本轮已完成

- 完成 `TemporalUpsampleCtcHead` 的代码审查与收口：
  - 恢复 `LinearCtcHead` 的构造和输入维度校验。
  - 校验 Head 维度、奇数卷积核、dropout 和上采样倍率。
  - 在卷积前后按有效输出长度遮蔽 padding，避免卷积 bias 污染有效边界。
  - `compute_ctc` 恢复旧线性路径校验，并返回 Head 变换后的有效输入长度。
- 将新 Head 接入 `training/sharded_ctc.py`：
  - 通过 `build_ctc_head` 工厂构建线性或上采样 Head。
  - 训练、验证、贪心解码和 PER 统计统一使用 `CtcComputation.input_lengths`。
  - 训练状态和报告写入完整 `head_config`。
  - 新 Head 配置纳入 resume fingerprint；旧线性训练的 fingerprint 保持不变，
    可继续恢复。
- 更新 checkpoint 兼容性：
  - best/latest checkpoint 保留旧顶层字段，同时新增 `head_config`。
  - 可从新 checkpoint 重建时间上采样 Head。
  - 旧 `LinearCtcHead` checkpoint 仍可重建和加载。
  - Experiment A 的初始 Head 加载会显式拒绝结构不匹配的 checkpoint。
- 更新正式训练 CLI：
  - `--head-type {linear,temporal_upsample}`
  - `--head-hidden-dimension`
  - `--head-kernel-size`
  - `--head-dropout`
  - `--head-time-upsampling-factor`
  - 旧命令为保持向后兼容仍默认 `linear`；新实验必须显式传
    `--head-type temporal_upsample`。
- 恢复并升级验证诊断链路：
  - 按 checkpoint 元数据重建 Head。
  - loss、解码、blank ratio、预测/参考长度比和 CTC 压力分桶均使用上采样后
    长度。
  - 保留删除、插入、替换及高频 token 统计。
- 确认特征缓存无需重建：现有校验要求目标在原始 Encoder 帧长下也可行，
  对上采样 Head 是更严格但安全的超集。

## 4. 本轮修改文件

本阶段代码与测试：

- `src/qwen_hotword/modeling/ctc_head.py`
- `src/qwen_hotword/training/sharded_ctc.py`
- `src/qwen_hotword/training/ctc_overfit.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `scripts/train_full_ctc.py`
- `scripts/diagnose_frozen_ctc.py`
- `tests/test_ctc_head.py`
- `tests/test_ctc_diagnostics.py`
- `tests/test_sharded_ctc.py`
- `docs/HANDOFF.md`

工作树中仍存在以下其他任务的未提交修改，本轮未改动、未删除：

- `docs/PHONEME_VOCAB.md`
- `docs/WORKZONE_RUNBOOK.md`
- `scripts/scan_g2p_coverage.py`
- `tests/test_g2p_coverage.py`
- `tests/test_phoneme_vocab.py`
- `configs/phonemes/en_es_ptbr_fr_id_precision_ipa_vocab.v0.3.json`
- `work/`

## 5. 测试结果

2026-07-21 本地实际结果：

```text
Ruff（本轮相关文件）: pass
Mypy src/qwen_hotword:          pass, 33 source files
Pytest (.conda + torch 2.10):   pass, 86 tests
git diff --check:               pass
CLI --help smoke:               pass
```

新增/更新测试覆盖：

- 线性 Head 形状、长度和旧校验回归。
- 上采样 Head 输出 shape 与精确 2 倍有效长度。
- 改变 padding 区域数值不影响有效卷积输出。
- 上采样 Head 的实际 CPU CTC loss。
- 新 Head checkpoint round-trip 和旧线性 checkpoint 兼容。
- 线性与上采样两种 Head 的分片训练、checkpoint、optimizer/scheduler 状态
  和 epoch resume CPU smoke test。
- 诊断模块从两种 checkpoint 重建 Head，且输入帧统计使用有效输出长度。
- 全仓库回归测试。

基础 Python 环境没有 torch，其张量测试会 skip；最终验证使用仓库 `.conda`
环境的 torch 2.10.0，所有测试均真实执行。

## 6. 仍待完成的工作区实验

代码接入和本地完成标准已达到。下一个最高优先级是在 H200 工作区对真实特征
缓存做受控实验；本机没有该 30.9 GB 缓存，因此本轮未伪造“真实缓存已跑”
的结果。

建议步骤：

1. 在新输出目录中，用少量真实 train/validation shard 跑 1–2 epoch smoke test。
2. 确认 report 与 checkpoint 的 `head_config.head_type` 为 `temporal_upsample`，且 resume
   能从下一 epoch 继续。
3. 使用与线性基线相同的 train/validation 特征缓存完整训练，不读取封存
   test 集。
4. 对线性 best checkpoint 和上采样 best checkpoint 运行
   `scripts/diagnose_frozen_ctc.py`。
5. 比较验证 PER、删除/插入/替换、预测/参考长度比、blank frame ratio 和各 CTC
   压力分桶，再决定是否进入 Encoder adapter/LoRA。

正式命令需显式加入：

```bash
PYTHONPATH=src python scripts/train_full_ctc.py \
  ...现有 train/validation cache 与 manifest 参数... \
  --output-dir outputs/full_ctc_temporal_upsample_v1 \
  --head-type temporal_upsample \
  --head-hidden-dimension 512 \
  --head-kernel-size 5 \
  --head-dropout 0.1 \
  --head-time-upsampling-factor 2
```

不得复用旧线性训练输出目录，也不得用线性 checkpoint 作为新 Head 初始权重。

## 7. 已知风险与已确认决策

- 2026-07-21 用户已确认：本项目后续直接在 `main` 分支继续更新，不再为冻结
  Encoder + 时间上采样 CTC Head 阶段单独创建 `codex/` 分支。
- 当前工作树混有其他任务修改。分支归属虽已确认，但未明确提交范围前仍不应
  整体提交、rebase 或整理其他未提交修改。
- 上采样会提高 Head 中间 activation 和训练时间；需用 H200 smoke test 确认
  `train_batch_size=256` 是否仍合适，必要时只调低 Head 训练 batch size。
- 上采样能增加对齐路径，但不能自动解决 G2P/标签噪声或声学表征不足；必须以
  validation PER 和错误分解为准。
- 当前时间上采样 Head 参数量约取决于 hidden dimension，不再是 92,250 参数的线性
  Head；报告会记录实际可训参数。

## 8. 下一任务需读取的最小文件

1. `AGENTS.md`
2. `docs/HANDOFF.md`
3. `src/qwen_hotword/modeling/ctc_head.py`
4. `src/qwen_hotword/training/sharded_ctc.py`
5. `scripts/train_full_ctc.py`
6. `src/qwen_hotword/training/ctc_diagnostics.py`
7. `scripts/diagnose_frozen_ctc.py`
8. `tests/test_ctc_head.py`
9. `tests/test_sharded_ctc.py`
10. `tests/test_ctc_diagnostics.py`

开始下一步前应再执行 `git status --short --branch`。后续保留在 `main` 分支；
不需要再询问是否创建新的 `codex/` 分支。
