# 新聊天交接prompt：英语热词检索根因诊断（2026-10-14）

请接管qwen3-asr-hotword，先完整读取AGENTS.md，再读docs/HANDOFF.md最新0.164和
0.163，以及0.159–0.162相关代码/取证。不要从头重建已完成实验。检查实际Git状态、
未跟踪文件及origin关系，不假设本地/工区一致；未知修改先报告，不覆盖用户文件。

仓库：https://github.com/SichengQian1/qwen3-asr-hotword
分支：codex/g2p-coverage-scan
本地：/Users/qsc/Documents/qwen3-ASR增训-latest
容器外：/home/star/q00933266/qwen3-asr-hotword
容器内：/host_home/star/q00933266/qwen3-asr-hotword
完整模型：/glusterfs_103/models/Qwen3-ASR-1.7B
本次文档更新前已核实远端SHA：b779a8e448cd3ba6b6ff920999100d9bb4b4d72e。
文档更新后的实际SHA请通过git/上一聊天最终回复核验，不把上述起点当最新提交。

当前主线是英语CTC热词检索漏召回与误匹配的根因诊断，尤其最新Ziyan10dB英语。
必须区分CTC PER、热词检索召回及最终ASR WER；此次Ziyan没有做最终LLM转写。
先分析为何发生及证据，再讨论机制和单因素改动；不要迎合“词表/90音素/噪声一定有错”。
每次看完数据，告诉我大概结论、马上做什么、当前大方向。命令要短，长逻辑写成脚本
提交推送让我pull；GPU保留参数，完整模型只由我在H200执行，不在本地加载或下载。
不要重训，不自动调门控/改音素/改核心检索，不覆盖、删除或重建任何已有outputs。

当前固定资产：
- JSON：/data/c00809914/audio/ms-swift-qwen3-asr/test/en-ziyan_2-5h_noise_10db_test.json
  SHA：32465f58a3c00d36c7d4d75b9b6c7200962789b9e8de7e4c340feec624cf20e1。
- 新资产：outputs/ziyan_english_assets_v1/（report.json、targets.json、english_raw.dict、
  keyword_table.json、dataset_manifest.jsonl、sha256.txt）。
- 检索：outputs/ziyan_english_retrieval_480h_v1/；逐样本文件为
  ziyan_en_noise10db/retrieval_details.jsonl；delivery下两份Top5/Top7。
- 当前Head：outputs/multilingual_480h_run_v1/head/ctc_head_best.pt，SHA
  320123fe7a6815d37e9ef4dc042b19a10521e6ce9c00d1a226e22c9e77ba4265。
- vocab：configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json（90类），SHA
  0f4939babf24b35ab8459273b13715b44423870c5d27ffe19f3bd3809e593af4。
- scripts/run_ziyan_english.py及evaluation/ziyan_assets.py、inference/ziyan_retrieval.py
  是当前实验入口；旧Anchor/评分/门控未改。Top64为发音变体，词面合并后Top5/7。
  score>=0.75、edit<=0.35、posterior>=0.5；
  score=(1-edit_ratio)*(0.75+0.25*posterior)，posterior不是词级存在概率。

已完成：从新参考冻结372个内容词并新跑English US MFA Top1；只替换原759个英语目标，
保留原2642填充词和ES/PT词条（我明确确认过），新表3606词面，不凑4000。
712音频实际0.826725小时，文件hash多出6条重复，未去重；413有新目标、299无新目标。
目标分母1100个不同词×音频。Top5命中639（58.09%）、raw680；Top7命中701（63.73%）、
raw762。输出Top5共1236/参考外578，Top7共1416/参考外696；完成2份交付。
固定Top7理想容量1007/1100=91.55%（22条音频超过7目标）；漏399对，不能全怪模型。
这些指标来自我返回的终端JSON，工区report及完整身份尚未以文件独立返回核验。

当前刚发现：28个旧填充词（非新目标）出现在新参考某处。Top5输出这些词42次/39音频，
Top7输出47次/43音频，但查询没有匹配对应音频参考，不能全算正确或全算错。
按SHA一致的原JSON及生产strict_phrase_match核对：ants只在spoken_languages_244一条
参考出现，却输出13次，至少12次参考外；conditions 1条/输出4次，contain 1条/输出4次，
drones 1条/输出3次，guarantees 1条/输出2次。尚无这些误例的CTC片段和听审。
总正确非目标输出最多19次，所以47次至少28次参考外，但“28次”与“28种词”不是一回事。
没有证明这些词挤出了哪些目标；移除交付词不等于从Top64索引删除词后重跑。

请先进行只读取证，不重新推理：
1. 核验资产/检索小report和身份，确认逐样本字段，给我最少短命令或小脚本返回摘要。
2. 将399个新集目标漏召回按未进候选、门控组合失败、合格但低于Top7分解；候选未保存
   不能直接判定无Anchor，必要时CPU用已保存CTC重放。检查重复音频及参考一致性。
3. 提取ants等误例及真实正确对照，包含参考、目标、词条发音/来源、score/edit/posterior、
   CTC匹配片段及同音频竞争目标。给我少量对应完整WAV提取命令让我听，不伪造音频时间。
4. 得到证据后再选一个受控修改；若排除28词，使用新表/新目录，同372目标/712音频/
   Head/参数做配对，说明参考条件化及候选数量变化，不直接归为泛化收益。

旧英语诊断已完成一层：干净6000条在审核联合表→4k表上目标Top7为3216→3215/4374，
CTC序列完全一致，扩表误匹配更多但不能解释全部英语低召回；漏项666未进候选、485门控、
8低于Top7。18例已返回并核验SHA，6个未进候选中5个无Anchor、1个Gentleman截断前排614；
这是旧集样本，不能套成Ziyan的分布。Night/Home/Good/Dance/Court/Great听审仍未完成，
固定音频ID和路径见HANDOFF0.164。

原759目标MFA审计已完成：当前vs新MFA 746一致/13未解决/0不同；当前vs训练词典696一致/
63未解决/0不同。官方MFA一致不代表真实发音正确，13个外语/重音项需另外审核。
不能说“没跑MFA就是全部标签错”或“重跑MFA能修复召回”。三语4k新dict脚本已交付，
成功产物尚未返回，不自动替换生产标签。

60条最终ASR排序/优先级探索已完成：B/C有效54条1条规范转写改变，目标不变；B/D有效46
无变化；E/F无规范变化；G/H只有20条有效排序样本无变化；重复B 3条无变化。目标A/B/C/D/G/H
均81/85，E/F79/85；仅小样本，不能宣布排序/优先级普遍无效，更不能拿检索召回等同WER。
本轮交接主线仍是检索诊断，不扩跑prompt实验。

葡语标签质量审计、310P完整4k检索时延等为独立未闭环旁线；新机器纯推理/Anchor移植暂停。
三语480h训练已完成，不继续训练。后续代码修改按AGENTS做Ruff、定向/全量pytest、Mypy、
git diff --check，重要阶段更新HANDOFF，提交推送当前分支并报告远端SHA。
HANDOFF只记节点成果/结果/未完成状态，不为每个小兼容修复新增流水记录。
