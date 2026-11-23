# 埃及阿语热词候选审阅（2026-11-23）

## 已确认范围

- 模型：新EGY ×3，最佳第29轮，Head SHA256
  `c561799f91650b08b148f101d2d152c225d141be9ac3425eb90b9b9a5164b320`。
- 配套r15 checkpoint-5100-merged Encoder；167类原音素表。
- 后续只做阿语原始音频、不加噪、整段Top5/Top7检索，不跑Decoder。
- 评测来源按音频时长 open-source:gaoshan:coarse=8:1:1，不是按句数。
- 用户先审阅候选词，确认后才允许生成正式检索词表并运行评测。

## 本次产物是什么

`configs/hotwords/egy_entities_terms.review.v1.json` 是164条**人工策划的待审候选**，
包含阿文、中文释义、分类、稳定ID；不是从工区语料实际提取出的实体，不宣称出现过，
也不是已通过发音核准的正式索引。草案没有为凑4000词加入功能词、碎片或普通动词。

类别：地名32、地标交通12、机构22、品牌组织14、人物10、医疗22、金融经济20、
科技20、能源环境12。中文释义辅助用户审阅，尚未由阿语专家核准。
10对嵌套关系单列，例如 القاهرة/جامعة القاهرة、ضغط الدم/ارتفاع ضغط الدم；
保留为候选不表示未来计分会把短词与长词同时当成目标。
包含少数需语境确认的指称，如“السادس من أكتوبر”（城市/日期）、“الدلتا”。

当前脚本**没有自动发现词表外的新实体**。它为这份候选补充冻结训练集中的语料证据。
如回传发现覆盖不足，应基于训练文本另扩候选，再审阅，不能根据测试检索成功与否挑词。

## 工区执行

分支 `codex/g2p-coverage-scan`，在项目根目录原Python环境执行，无需GPU或新依赖：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
python -u -B scripts/prepare_egy_hotword_candidates.py
```

脚本校验已确认的Head、词表、train清单SHA，以及完成报告中的epoch、Head形状和
Encoder权重身份；本阶段不重新读取数GB Encoder权重，仅核对已有报告中的身份。
随后仅扫描新×3目录下的 `train.jsonl`，跳过英语；不读validation/test清单、不读WAV、
不加载Qwen、不启动CAMeL/G2P、不运行检索。命令显示阶段及实际已处理/总行数。

默认每次建立新的 `outputs/egy_hotword_review_XXXXXXXX/review/`。
不支持把旧审阅结果覆盖后续跑；中断时保留旧目录，再执行同命令生成新目录。
也可指定不存在的 `--output-dir`。本地无语料时可用 `--preview` 仅生成草案，
状态明确为 `editorial_preview_only`，词频显示“待工区核实”，不会填虚构的0。

## 计数规则与输出

匹配执行NFC、去阿文附标及延长线，保留完整词边界和标点边界。
不把“مصر”当成“مصري”的子串命中，也不把“جامعة، القاهرة”计为“جامعة القاهرة”。
不合并alef/hamza、ya/alif-maqsura、ta-marbuta/ha等字母。支持显式别名，但当前草案未加别名。
所以精确出现数是保守文本证据，不等同实体全覆盖率。

前接و/ف/ب/ل等的可能形式，另列待审句数和例句，不擅自剥离后算成真实关键词。
每词记录训练句数、出现次数、分来源句数、最多两条精确匹配例句和一条前接形式例句。
词频不用于自动淘汰低频词；零出现词保留，并标明“不曾在当前保守匹配下观察到”，
可在用户审核后决定保留为干扰项、修改写法或删除。

输出：

- `候选词表.md`：便于阅读的阿文/中文分类表及嵌套关系。
- `candidates.tsv`：可填写“用户决定”的审阅表。
- `candidates.json`：候选、来源计数和原文例句。
- `nested_pairs.json`：10对嵌套关系；随候选变更重新计算。
- `report.json`：代码/数据身份、扫描量和未执行阶段。
- `return_review.zip`：上述五份小文件；不含音频、模型或训练清单。
- `return_part_*.txt`：同一ZIP的Base64分片，每份不超过32KiB，便于上传。

核验 `report.json.status=candidate_review_ready`、
`corpus_scan_completed=true`、`training_counts.rows=489330`，
`test_set_read=false`、`retrieval_started=false`、`phonemization_started=false`。
回传 `return_review.zip`；上传受限时回传全部 `return_part_*.txt`。

## 用户审阅之后

词表数量目前不设硬目标。语义保留/删除、别名和嵌套计分规则确认后，再使用与训练一致的
埃及发音前端核验候选能否生成完整167类音素ID，单列歧义或失败，不能静默删词。
词表及检索策略冻结后再打开独立测试数据，按时长8:1:1确定评测清单并测覆盖、目标密度、
负样本及Top5/Top7。禁止依据检索结果反选词表、排除失败正例或报告未经测量的召回率。
