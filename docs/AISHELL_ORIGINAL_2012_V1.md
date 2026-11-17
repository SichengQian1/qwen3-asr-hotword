# AISHELL原始2012词版本：中英183类检索

用户决定按原始关键词口径重新测一版。本次恢复原始2012候选词、2135条音频、630种目标词，
计分分母为2782个“音频＋目标词”组合，不应用1054词版的白名单或19个逐句排除项。
普通词、子串匹配目标均保留。不会把“常宁”改标成“张常宁”，也不新增未标注目标。
这表示遵循原始文件的口径，不表示已确认原作者的标注意图或完整专名正确率。

输入文件固定SHA与上一版相同：`kw/kw_candidates.tsv`、`val/kw_subset/kw.jsonl`。
根目录 `/glusterfs_104/asr-data-Open-Source/zh/aishell`，音频使用 `wav/test/<utt_id>.wav`。
2012词按稳定词面顺序构建索引，630词均在候选库中；如标注不在库中，报错而非丢弃。
原始2135条有目标子集不是AISHELL全部7176条测试音频。

## 模型及检索条件

- Head仍为 `outputs/zh_en_183_noise70_dual_v2/head/ctc_head_best.pt`，epoch25。
- 额外固定权重SHA：`955e31483b54040c5fce9da6e9287f604545087cdddf15db434a2ce7557991d1`。
  不能仅凭目录相同接受后来替换的Head。
- 中英US/UK共183类音素表不变。中文MFA3.0.0、训练前端及Top1不变。
- 原始音频不加新噪声，冻结原始Qwen3-ASR-1.7B；无训练、无Decoder生成。
- Anchor候选64、局部起点半径2、音素片段2/3/4、每词24锚点、偏移容差1不变。
- 仍报告raw及gated Top5/7。raw是Anchor后、置信门限前；gated沿用score0.75、
  edit ratio0.35、posterior0.5，不根据本测试重新调参。
- 分母2782，同句同词重复提及计一次；另报630词宏平均召回。不是除以2012或2135。
- MFA转换必须2012词全部成功；有缺失/OOV则停止，不缩小候选库或计分分母。同音不同词面保留。

## 工区运行

在已有qwen3-asr-hotword环境运行，物理GPU5需可用且至少35GiB空闲：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -u -B scripts/evaluate_aishell_zh_en.py --gpu 5 --config configs/aishell_original_retrieval.workzone.json
```

默认配置仍是旧1054版，本次必须传上面的新config。
同一命令自动续跑本次目录，不加resume。重新构建2012词索引并运行2135条检索，
不使用1054词版的候选排序/结果缓存。旧输出完整保留；不要把output-dir指向旧目录。
共享评测模块本次升级后，旧运行记录的代码SHA不同，因此不要尝试修改其身份文件强行续跑；
旧结果可直接读取，若确需重新运行旧口径，需指定另一个新输出目录。

新输出根目录：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/aishell_zh_en183_original2012_top57_v1
```

实际生成热词表在新目录的 `phonemes/indexed_pronunciations.jsonl`；
`report.json`记录口径、计分、模型/词表/索引SHA与延迟，`retrieval_details.jsonl`记录逐条结果。
运行后应为active_surfaces2012、sample_count2135、target_words630、target_pairs2782、num_classes183，
scope为 `AISHELL_original_offline_full_audio_retrieval`。失败也会生成紧凑诊断包。
只需回传：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/aishell_zh_en183_original2012_top57_v1/return_bundle.zip
```

## 如何比较

旧版为1054词/1245音频/271目标词/1489组合，新版为2012词/2135音频/630目标词/2782组合。
二者同时改变候选库和测试分母；因此不能把总体召回差值只归因为候选库扩大。
本次先完整报告原始口径，不额外设计消融或更改参数。
沿用原标注不是音频听审；这个有目标子集不能给出总体无目标误报率，也没有最终ASR WER。

本地已用之前保留的哈希匹配原始文件复现2012/2135/630/2782，旧筛选仍为1054/1245/271/1489，
并确认19个原先排除的逐句对应全部恢复。实际H200结果待回传。
