# AISHELL：中英183类 Head 的1054词检索评测

本次只测原始音频整段的音素热词检索 Top5 / Top7，不训练、不加新噪声、不运行 Qwen Decoder。
训练时70%音频加10 dB噪声的政策保持不变，不等于测试音频也必须加噪。

## 固定数据与模型

| 项目 | 本次使用 |
|---|---|
| 候选词库 | 1054个已确认保留的词面，全部参与同一个检索索引 |
| 测试音频 | 1245条；使用 `wav/test/<utt_id>.wav`，逐条核对同名 `.txt` |
| 有效目标 | 271个不同词；1489个“音频＋目标词”组合，同句重复提及同一目标计一次 |
| 原始数据根目录 | `/glusterfs_104/asr-data-Open-Source/zh/aishell` |
| 数据筛选政策 | `configs/aishell_review_1054.v1.json`：1054词白名单＋19个明确排除的错误目标组合 |
| Head | `outputs/zh_en_183_noise70_dual_v2/head/ctc_head_best.pt` |
| 词表 | `configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json`，183类 |
| 词表SHA256 | `ef5a411afbdb238a400302da2d871fe64f8f94e8d065fdf045b8bf5f39a2b5cb` |
| Encoder | 原始 `/glusterfs_103/models/Qwen3-ASR-1.7B`，冻结 |
| 中文G2P | `mandarin_china_mfa` 3.0.0，和本轮训练相同的固定资产、Top1及归一化 |
| 英语G2P | 同一中英基础配置中的 `english_us_mfa` / `english_uk_mfa` 3.3.0；中文词面若有拉丁文字，沿用训练US路由 |

不使用旧90/100类 Head、英阿 Head或其索引。启动时验证中英训练完成标记、最佳权重/报告SHA、
词表ID顺序、183类Temporal 2x结构和训练时保存的Qwen文件身份。
训练目录尚未生成 `training_completed.json` 时停止，不会回退到旧模型。

## 工区运行

在已有 `qwen3-asr-hotword` 环境中，先确认物理GPU4是本次可用卡：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -u -B scripts/evaluate_aishell_zh_en.py --gpu 4
```

用已有MFA `aligner` 环境。固定中文前端会放入本次输出的私有目录，不修改Conda环境；
必要下载按工区政策跳过TLS校验，但强制核对固定SHA256。完整Qwen只从工区本地加载，不下载。
复用现有Qwen加载封装，当前仍会加载基础模型权重，但仅执行Encoder和CTC分支，不运行Decoder生成；
该脚本不是Encoder单独部署或显存优化实验。加载前检查所选卡至少有35 GiB空闲，不停止其他任务。

同一命令自动续跑，无需 `--resume`。已验证的热词发音批次和逐条检索结果直接复用，
重启会再核对音频/模型/配置身份。配置、模型、输入、相关代码或依赖版本发生变化时拒绝混用；
确需新实验时用 `--output-dir outputs/另一个新目录`。不更改历史训练缓存或原始数据。

流程是：核验模型 → 重建固定筛选 → 核验音频及训练/验证重复 → 热词MFA转写 → 1245条检索 → 汇总。
每个阶段提供进度。输入参考答案与目标词只用于选择固定测试集和最后计分，不传入Encoder、CTC或候选排序。

## 热词生成与检索

- 从原始 `kw/kw_candidates.tsv`、`val/kw_subset/kw.jsonl` 按固定SHA和审核配置重建。
  不从1245条参考文本补造目标，不只拿271个正例词建库。生成清单与发音只写入忽略的outputs。
- 中文使用与训练一致的 pypinyin 0.55.0（词典声调，不额外变调）＋jieba 0.42.1（HMM关闭）
  ＋固定MFA Top1，再映射至183类ID。数据集提供的拼音字段不用作替代标签。
- 遇到未知符号、缺失发音、OOV、空标签，保留失败原因并停止；不删字、保留部分发音或偷偷缩小1054词。
  同音而不同词面的热词分别保留，并正常竞争排名。
- 使用已有音频预处理、Encoder `ln_post`、Temporal 2x Head、CTC贪心解码、Anchor召回及局部编辑距离/后验重排。
  Anchor采用2/3/4音素片段、每词最多24锚点、偏移容差1、最多64个候选、局部起点半径2。
  完整配置见 `configs/aishell_retrieval.workzone.json`。

## 指标口径

主看 `raw_top5` / `raw_top7` 的 `recall`：正确召回的有效目标组合数 ÷ **1489**。
这是经过Anchor候选筛选和重排、尚未经过置信门限的TopK；不是遍历全部1054词所得的理想排名。
另报 `gated_top5` / `gated_top7`，使用既有门限：score≥0.75、edit ratio≤0.35、posterior≥0.5。
门限先过滤再分别取前5/7，因此返回数量可少于K。本次不利用测试集调参。

各口径都附逐词命中数、271词宏平均召回和整句目标全部命中数。主指标按出现组合计数，
不是“命中过的不同词数÷271”，也不是“命中的句数÷1245”。
纯检索延迟不含音频读取、Encoder和Head；逐条详情另外保留各阶段时延。
续跑保留原条目的实际计时，不把缓存读取计时当作推理提速。

局限：目标只是文本审核，未逐条听审；783个非目标候选是词面筛选保留，不能宣称全部为已认证负例。
测试集是有有效目标的子集，不能据此计算总体无热词误报率。另列“参考全文未出现的入选词”
和“文本出现但未标注/被排除的入选词”，不把后者直接当作误报。
检查的是本轮Head训练/验证音频的文件字节及标准化PCM重复，不证明说话人/近重复隔离或Qwen预训练未见过AISHELL。
本词库筛选参考过本测试文本，结果定位为固定基线，不能冒充全新业务盲测。

## 输出及回传

固定输出目录：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/aishell_zh_en183_1054_top57_v1
```

- `report.json`：召回、分母、逐词计分、延迟及模型身份。
- `phonemes/indexed_pronunciations.jsonl`：1054词的实际音素、token ID及来源。
- `g2p_report.json`、`phonemes/failures.jsonl`：发音覆盖率与失败明细。
- `audio/audio_manifest.jsonl`、`retrieval_details.jsonl`：固定音频、有效目标及逐条检索明细。
- `sample_shards/`：原子保存的逐条结果，支持断点续跑。
- `overlap_audit.json`：与本轮训练/验证的重复检查。
- `return_bundle.zip`：紧凑报告、资产身份、覆盖情况和最多20条漏召回示例；不包含权重或音频。

运行后只需返回：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/aishell_zh_en183_1054_top57_v1/return_bundle.zip
```

本地已验证筛选精确复现1054/1245/271/1489，并完成单元/相关回归检查；实际H200召回待运行，不能以本地测试替代。
