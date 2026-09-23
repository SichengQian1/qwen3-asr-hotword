# 工作交接记录

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
