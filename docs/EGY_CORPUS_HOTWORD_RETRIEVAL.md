# 埃及阿语：真实语料词表与整段 Top5/Top7

## 本轮固定内容

用户于2026-11-23授权：结合实际语料选词，并交付可直接运行到Top5/Top7结果的工区命令。
这替代此前164条人工策划候选的审阅流程；保留旧文件，不把164条整体当成正式词表。
使用新EGY ×3 epoch29，Head SHA256
`c561799f91650b08b148f101d2d152c225d141be9ac3425eb90b9b9a5164b320`，
r15 merged Encoder权重SHA256
`7dd3d4da16d9fef24c38feaa649b4aaca467a2499e48fceb5968611d98aa5443`，
原167类音素表不变。仅整段检索，不加噪、不调用Qwen Decoder、不训练。

## 工区命令

在项目Conda环境、项目根目录执行，默认物理GPU4，单进程；需要其他卡只改`--gpu`。
先确认该卡可用。首次下载NER约436MB；占用GPU的阶段依次执行，不与Encoder同时驻留。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
python -u -B scripts/evaluate_egy_corpus_hotwords.py --gpu 4
```

默认输出：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/egy_ep29_corpus_hotwords_v1/`。
所有阶段有心跳，逐音频/逐候选显示真实数量。可用nohup后台运行：

```bash
mkdir -p outputs
nohup python -u -B scripts/evaluate_egy_corpus_hotwords.py --gpu 4 \
  > outputs/egy_ep29_corpus_hotwords_v1.log 2>&1 &
tail -f outputs/egy_ep29_corpus_hotwords_v1.log
```

同命令可恢复：校验身份后复用完成的逐条NER、发音、检索结果；未完成条目重做。
不可同时开两份指向相同输出目录的任务。GPU编号不影响恢复；更换数据、词表政策、
相关代码或模型必须新建`--output-dir`，不会复用身份不同的预测。

## 六个阶段与评测口径

1. **锁定模型和测试集。** 校验原始x2构建时封存的test清单，排除train/validation
   的ID、音频路径、音频SHA和可识别原始录音分组重合。独立抽样不读检索结果。
2. **先抽音频。** 在test里按时长open_source/gaoshan_train/coarse_train=8:1:1。
   用稳定哈希排序、完整音频取样，每组超额小于一条音频，不切音频凑秒数。
   根据此前test容量，gaoshan约0.365h是限制项，总目标约3.651h，以实际selection报告为准。
   未按热词数量选音频，也不要求每条必有热词。不把来源EGY标签宣称为人工方言鉴定。
3. **从实际参考文本提候选。** 固定版本CAMeLBERT NER取置信度≥0.9的PERS/LOC/ORG
   原文完整片段，保留例句和来源；拒绝跨词边界的半词。专业术语仅补充原草案中
   医疗、金融经济、科技、能源环境四类里确实在所选文本出现的词。
   新实体不限于原164条。每个来源训练文本固定抽300条提取同类干扰词，排除在任意
   所选评测参考文本中匹配到的词，最多加入与目标候选等量的干扰词。
4. **同前端生成发音。** 使用已安装CALIMA EGY模型和现有阿语修正规则，isolated模式，
   生成`ar_ipa:`音素ID。不使用测试句子的存储音素答案来替代热词发音。
   无分析、待审例外、非法ID、少于4音素等词保留为inactive，理由可查；目标不从主分母删除。
5. **冻表后推理。** 逐条原始音频，核实字节SHA和时长，完整音频经过r15 Encoder及×3 Head。
   Anchor shortlist64、offset tolerance1、start radius2；参考文本/目标ID从不进入检索函数。
   新的×3加载入口独立于旧的固定×2流式入口，不放宽旧入口约束。
6. **统一计分。** 真值为表中短语与参考文本的保守词边界匹配，去音标和延长符，
   保留alef/hamza/ya/ta区别；最长非重叠短语优先，同一词同一音频重复出现只算一对。
   不自动把前缀连接形式合并成同一实体，不调整阈值追求高分。

`all.operating.top5/top7`是主指标：score≥0.75、posterior≥0.5、edit ratio≤0.35后取前K。
`all.raw_ranked.top5/top7`是Anchor shortlist内未做上述置信过滤的排名对照，不等于全词表精确搜索。
报告同时含分来源结果、负样本误提示率、参考文本存在精度、目标密度、TopK容量上限、
发音可用目标覆盖率。嵌套短词若真实存在但不是独立目标，不算“参考文本不存在”的误报。

**结果名称应使用“语料条件化实体/术语检索评测”。** 词表结合了测试参考文本，因此能衡量
音频检索已知词表的能力，但不是未知业务词表的盲测。实体模型在MSA ANERcorp上训练，
用于EGY转写时分类可能有误；这是自动候选，不是阿语专家审定的实体或发音。
不能因某词难检索就删掉它重报同一轮结果。后续业务评测需独立冻结业务词表。

## 文件与回传

- `candidates.tsv`：可直接看阿文词、类型、目标/干扰角色、IPA、启用状态。
- `candidates.json`：实体原始提取依据、NER置信度、来源和例句。
- `table.json`：最终词表与逐词发音分析、失败原因。
- `selected_audio.json`、`selection.json`：音频清单及实际三来源时长。
- `retrieval_rows/`：每条音频预测、分数、解码、耗时、用于计分的目标ID。
- `report.json`：总Top5/Top7和分来源指标、模型/词表身份。
- `return_summary.zip`：只打包小报告，不包含模型、音频、逐条大结果。
- `return_part_*.txt`：同一小包的Base64分片，每片<32KiB，平台无法传zip时使用。

查看百分比：

```bash
jq '{status, candidate_summary, pronunciation_target_coverage,
  top5_percent:(.all.operating.top5.target_recall*100),
  top7_percent:(.all.operating.top7.target_recall*100),
  raw_top5_percent:(.all.raw_ranked.top5.target_recall*100),
  raw_top7_percent:(.all.raw_ranked.top7.target_recall*100)}' \
  outputs/egy_ep29_corpus_hotwords_v1/report.json
```

回传`return_summary.zip`即可，候选明细需要核对时再传`candidates.tsv`。

## 工具来源

[CAMeL-Lab官方模型卡](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa-ner)，
固定revision `54e2905e7c756883b00877cd48ed710a304af0d1`；权重和tokenizer配置逐文件固定SHA256。
模型卡说明其基于CAMeLBERT MSA，在ANERcorp训练。
安装/下载按AGENTS授权仅对本次HTTPS下载跳过证书验证，下载后先验SHA256；不改系统SSL设置。
阿语发音沿用项目`arabic_pronunciation_policy.json`及CALIMA EGY，不引入MSA发音来替代EGY。
