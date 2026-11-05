# 中英阿共享音素表 v1.0：普通话、美式英语、埃及阿语

2026-11-05。此版本冻结发音单位及 ID，适合查阅、映射和覆盖核对。中文指大陆普通话，
英语沿用现有美式 MFA 约定；阿语以埃及口语的**开罗参考**为主，另保留现代标准阿语
（MSA）及历史 MFA 原始词典层。尚未取得目标语料，不能将开罗参考外推为所有埃及地区的实测覆盖。

| 文件 | 用途 |
| --- | --- |
| [音素表 JSON](../configs/phonemes/zh_en_ar_ipa_vocab.v1.0.json) | `tokens` 的下标是固定 ID；`languages` 给出各语种实际归属，`sources` 保存逐来源原始单位、版本和出处。 |
| [逐来源映射 TSV](../configs/phonemes/zh_en_ar_ipa_mapping.v1.0.tsv) | 由 `language`、`source`、`source_unit` 查 `shared_ipa_unit` 和 `token_id`。同一个单位在不同来源出现时保留多行。 |

共有 **248 类：246 个发音单位 + `<blank>` 0 和 `<unk>` 1**。三语各自有 142、78、95 个
发音单位；它们有交集，不能直接相加。其中 13 个三语共享、43 个恰好两语共享、190 个仅归属
一个语种。阿语的 95 个是三份来源层的并集；开罗来源层本身有 79 个单位，不能把全部 95 个
都称为开罗口语音素。部分来源采用较细的表音单位，因此类别数也不等于语言学上最小音位数。

## 阿语到底用什么

1. **开罗埃及阿语是主参考。** 用 Leddy-Cecere 和 Schroepfer 的
   [Egyptian Arabic](https://doi.org/10.4324/9780429025563-17) 第 3.1–3.2 节、表 17.2–17.3，
   录入 30 个辅音和 10 个元音单位，单独记录该辅音表的长辅音表示。保留 `eː/oː`、
   `a/ɑ`、`aː/ɑː`、`zˤ`、`lˤ`，以及书中列出的元音表面变体；借词中的 `p/v` 也有位置。
   再纳入 [Youssef 的开罗表音清单](https://doi.org/10.7557/12.2510)（包括 `rˤ`）及
   [腭化研究](https://doi.org/10.7557/12.3739)的 `tʲ/dʲ`、[开罗 r 的声学研究](https://pubmed.ncbi.nlm.nih.gov/23782008/)
   的闪音 `ɾ`。这些补充保留不同来源的观察，**不默认把表面变体当成必选音位，也不自动生成读法**。
   长辅音容量只按两份基础表各自明确的规则生成，不能据此宣称每个长音都已在业务录音出现。
2. **MSA 保留独立参考层。** 用[现代标准阿语辅音、元音表](https://pmc.ncbi.nlm.nih.gov/articles/PMC9932739/#tbl1)
   录入 28 个基础辅音和 `a/aː、i/iː、u/uː`，并按[叠辅音研究](https://journals.pan.pl/Content/82346/mainfile.pdf?handler=pdf)
   单列相应长辅音。`ː` 表示长度，附着在前面的完整单位上；例如 `dʒː` 和 `ʔː` 都各占一个 ID。
   这是标准读法及混用时的表示层，不替代埃及词的读法选择。
3. **历史 Arabic MFA v2.0.0 是兼容层。** 从 Montreal Corpus Tools 的
   [固定原始词典](https://raw.githubusercontent.com/MontrealCorpusTools/mfa-models/1279e7d03a289943b690b8fc11ce5d864577c4ff/dictionary/arabic/mfa/arabic_mfa.dict)
   读取 14,202 行、10,753 个不同键（含词典的 `<unk>`）。其 73 个发音单位与
   [MFA 官方清单](https://mfa-models.readthedocs.io/en/latest/dictionary/Arabic/Arabic%20MFA%20dictionary%20v2_0_0.html)
   一致，另有非发音符号 `spn`。这套旧词典不声明方言，也不是 G2P；保留其音值范围，
   不把其中每个词的旧读音当作埃及标注。MSA 比旧词典补入 `i u x hː lː xː ʔː ʕː`；
   开罗来源在这两个层之外再贡献 14 个阿语单位，其中 4 个已经在英语表里，因此全表增加 10 类。

为何不直接用 DialG2P 当音素表：我核对了 [QCRI 固定埃及文件](https://github.com/qcri/DialG2P/blob/a98e887c54621bbfed497509beea5b84ab37fc7c/Data/DialG2P_Egy.tsv)，
实际有 6,889 行，两列是 ID 和带标音文字，包含部分 `书写/口语` 对照；它不是发布的完整 IPA
符号清单、音频或可直接运行的 G2P 权重。可用来复核字形与实际读法的差异。其
[论文](https://aclanthology.org/2025.arabicnlp-main.38/)结果基于该研究的埃及推文数据，不能代替业务听审。

例如，[埃及 CODA 的开罗规则和例词](https://aclanthology.org/L12-1328.pdf)表明，开罗口语
`قلب`（心）可写成发音 `ʔ a l b`，MSA 参考读法是 `q a l b`；`ج` 在典型开罗读法中常为
`ɡ`，标准参考表仍有 `dʒ`。因此 `q` 与 `ʔ`、`ɡ` 与 `dʒ` 都分别保留。**同一字母可以有
不同读法；不能把字母全局替换，也不能只因符号已有就假设上游已选对。** 词面到发音的
埃及语字典、上下文补音及母语听审是另一个环节，不在此音素表内冒充完成。

## 三语怎么映射到一张表

普通话和英语分别固定 Montreal Corpus Tools 的 `mandarin_china_mfa`（G2P 元数据 3.0.0）
和 `english_us_mfa`（3.3.0）。**同时查 G2P 的声明和固定发音词典的实际发音列**，避免把模型
元数据等同于词典覆盖。两套 G2P 元数据未给有效 WER/PER，本次也没有执行 G2P 或听审。

| 语言 | 取单位的范围 | 本表处理 |
| --- | --- | --- |
| 普通话 | [G2P 元数据](https://huggingface.co/MontrealCorpusTools/mandarin_mfa/resolve/cfbee14002d514fcf411c16676877a96b7afe6ff/g2p/mandarin_china_mfa/meta.json) 137 个；[固定词典](https://huggingface.co/MontrealCorpusTools/mandarin_mfa/resolve/cfbee14002d514fcf411c16676877a96b7afe6ff/dictionary/mandarin_china_mfa.dict) 发音列去除 `spn` 后 139 个 | 两来源取并集得 142 个。词典增加 `m̩˧ n̩˥˩ n̩˨˩˦ ŋ̍˥˩ ŋ̍˨˩˦`；声调和成音节标记不拆不删。 |
| 美式英语 | [G2P 元数据](https://huggingface.co/MontrealCorpusTools/english_mfa/resolve/1009e3c6c689f10875146a5e4e5c2233f8eaf922/g2p/english_us_mfa/meta.json) 78 个；[固定词典](https://huggingface.co/MontrealCorpusTools/english_mfa/resolve/1009e3c6c689f10875146a5e4e5c2233f8eaf922/dictionary/english_us_mfa.dict) 去除 `spn` 后也是同 78 个 | 复用项目既有的标签约定，例如 `aj`、`dʒ` 各自一个单位；保留细音值 `i/iː`、`t/t̪` 等。 |
| 阿语 | 开罗参考、MSA 参考、历史 MFA 原始词典各自有独立来源行 | 音值、长短、叠辅音和咽化均保留；不开启跨方言读法替换。 |

此外核对了两个 G2P 的 `phones.sym`。不计 FST 的 `<eps>`，中文表为 134 个、英语为 69 个；
小于各自元数据声明。少出的项目没有从 v1.0 删除，因为它们仍有声明或词典来源。收录一个
单位不表示该 FST 一定能够产出它；字典中的 `spn` 和 FST 的 `<eps>` 都不是发音单位，
不能映射成 CTC blank，`<unk>` 也不能替代未知发音。英语原来源不提供词重音标签；外部的
重音、边界及其余未定义符号不静默删除，需明确定义后再接入。中文声调已随音值保留。

映射只接收**已经切分好的 IPA 单位**。按空白切分，应用现有少量整单位 IPA 别名
（`g→ɡ`、`t͡ʃ/t͜ʃ→tʃ`、`d͡ʒ/d͜ʒ→dʒ`），再按 Unicode NFD 等价用规范单位查 ID。
不能把这套规则套给拼音、Buckwalter、Arabizi 或其他转写系统：这些系统的同形符号可能
指不同声音。不同方案之间的音值分歧也不会靠 Unicode 处理合并。

| 示例 | v1.0 实际 ID | 为什么这样做 |
| --- | --- | --- |
| 三语的 `m` | `77` | 来源都使用同一个 IPA 单位，采用一个 ID；不要求三语的声学实现逐样本完全相同。 |
| `k`、`kʰ`、`kʷ` | `69`、`70`、`71` | 送气和唇化都保留。 |
| `a`、`aː`、`a˥`、`a˨˩˦` | `2`、`17`、`18`、`22` | 长度和声调都不能因基础符号相同而折叠。 |
| `q`、`ʔ` | `112`、`240` | 埃及与标准读法的两个单位都可表达。 |
| `ɡ`、`dʒ` | `204`、`32` | 有别名的是 IPA `g/ɡ` 的写法，`ɡ/dʒ` 是不同音值。 |
| `z`、`zˤ`、`zˤː` | `158`、`160`、`161` | 普通、咽化和咽化长辅音不合并。 |
| `r`、`rˤ`、`ɾ`、`ɹ`、`ɻ` | `114`、`116`、`218`、`216`、`217` | 发音不同，不能统称为一个 r。 |

表的 ID 独立于原英西葡资产。每条发音仍应带语言及方言/来源信息；ID 共享不会自动判定
所属语言。完整性指所选可追溯来源的发音单位保留，不等于每个单位都应该成为业务首选读法，
更不等于所有埃及地区、专名、借词和混说都已覆盖。细分越多，人工定义和复核成本越高；
后续应先看实际录音和使用场景，再决定是否改变映射，不能在 v1.0 内静默换义或重新排 ID。

## 只读核验

```bash
python -B -m pytest tests/test_zh_en_ar_vocab.py
sha256sum configs/phonemes/zh_en_ar_ipa_{vocab,mapping}.v1.0.*
```

核验检查来源的每个原始单位是否有记录、ID 和现有 IPA 读取逻辑是否一致、易混单位是否保持
独立、未知单位是否整条拒绝。只读取仓库内小文件；无 GPU、音频、G2P 调用、运行目录或续跑步骤。
