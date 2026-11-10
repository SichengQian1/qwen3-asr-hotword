# 英阿 MFA G2P 音素表 v1.0

英阿表采用 2026-11-10 核验的官方美式英语和 GlobalPhone 阿语模型，复用中英阿 v1.1
已核对的发音单位和来源映射。这里单独分配连续 CTC ID。

| 范围 | 固定模型 | 发音单位 | CTC ID |
| --- | --- | ---: | --- |
| 保留类 | CTC blank、预留 unknown | 2 | 0–1 |
| 美式英语 | [english_us_mfa 3.3.0](https://huggingface.co/MontrealCorpusTools/english_mfa/tree/1009e3c6c689f10875146a5e4e5c2233f8eaf922/g2p/english_us_mfa) | 70 | 2–71 |
| 标准阿语 | [arabic_g2p 2.0.0 官方历史包](https://github.com/MontrealCorpusTools/mfa-models/blob/fa849afcc4336ee0e9f0bb66224c38808e875f96/g2p/arabic_g2p.zip) | 42 | 72–113 |

总计 **114 类**。英语用 MFA IPA，阿语用大小写敏感的 GlobalPhone 单位，保留
`ar_gp:` 前缀。一个阿语输出单元对应一个 CTC 类。即便英、阿输出写法相同，
也各自有 ID，例如英语 `m` 与阿语 `ar_gp:m`。英文音素仍按 v1.1 约定规范化。

- [词表 JSON](../configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json) 的 `tokens` 下标是
  CTC ID。JSON 中也记录了两个固定模型的来源与 SHA256。
- [映射 TSV](../configs/phonemes/en_ar_mfa_g2p_mapping.v1.0.tsv) 用 `language` 和
  `source_unit` 查 `token_id`。FST 和外置符号表的编号留作溯源。源 epsilon 和
  unknown 没有目标 ID。

## 使用

英语运行时使用对应快照的 `graphemes.txt` 和 `phones.txt`，或直接用 FST 内嵌表。
同包英语 `.sym` 错位的排查见[中英阿 v1.1 说明](ZH_EN_AR_MFA_G2P_VOCAB_V1_1.md)。
阿语模型的输入是 GlobalPhone 罗马化文本，原始阿文先走独立的转写前端。

| 语言 | 模型输入 | G2P 实际输出 | 英阿 v1.0 CTC ID |
| --- | --- | --- | --- |
| 英语 | `cat` | `cʰ æ t` | `7 43 34` |
| 阿语 | `salam` | `s a l a m` | `105 81 95 81 97` |

已复查两套 FST 的全部输出弧，并用固定模型和 MFA 3.4.3 重写实现测试六个词。
开始训练时，按英阿表重新编码目标和热词索引，CTC Head 的输出维度为 114。
复用中英阿已有发音时，可以用保存的发音单位查新 ID。既有中英阿 ID 属于原表。
目标语料的原始阿文转写和词级质量需要按实际数据另行验证。
