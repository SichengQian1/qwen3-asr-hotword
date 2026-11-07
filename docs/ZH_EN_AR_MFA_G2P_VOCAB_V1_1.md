# 中英阿 MFA G2P 输出表 v1.1

本版面向同一套 MFA G2P 打标流程，使用 2026-11-08 核对的官方普通话、美式英语和
GlobalPhone 阿语模型。取模型 FST 输出弧上实际使用的单位。原 v1.0 另作埃及口语参考。

| 语言 | 官方模型及 G2P 版本 | 输入 | 发音单位 |
| --- | --- | --- | ---: |
| 普通话 | [mandarin_china_mfa 3.0.0](https://huggingface.co/MontrealCorpusTools/mandarin_mfa/tree/cfbee14002d514fcf411c16676877a96b7afe6ff/g2p/mandarin_china_mfa) | NFC 带调拼音 | 134 |
| 美式英语 | [english_us_mfa 3.3.0](https://huggingface.co/MontrealCorpusTools/english_mfa/tree/1009e3c6c689f10875146a5e4e5c2233f8eaf922/g2p/english_us_mfa) | 规范化英语词形 | 70 |
| 阿语 | [arabic_g2p 2.0.0](https://github.com/MontrealCorpusTools/mfa-models/blob/fa849afcc4336ee0e9f0bb66224c38808e875f96/g2p/arabic_g2p.zip)，官方历史包 | GlobalPhone 约定的罗马化文本 | 42 |

中英有 29 个共用 IPA 单位，阿语原生 GlobalPhone 单位单独编号。因此共有 **219 类**
（217 个发音单位和两个保留类）。`<blank>` 为 0，`<unk>` 为 1。FST 的 epsilon 和源
`<unk>` 不产生训练目标。阿语保留大小写与单元边界，例如 `ar_gp:q` 与 `ar_gp:Q` 各有
一个 ID，`ar_gp:alal` 是一个单位。这一版阿语沿用 GlobalPhone 的朗读标准语约定，原始
阿文输入需要独立的 GlobalPhone 转写前端。[MFA 历史模型说明](https://montreal-forced-aligner.readthedocs.io/en/v1.0/pretrained_models.html) · [GlobalPhone 论文](https://research-karlsruhe.de/pubs/ICASSP2013-Schultz_GlobalPhone.pdf)

- [模型身份和词表 JSON](../configs/phonemes/zh_en_ar_mfa_g2p_vocab.v1.1.json) 的 `tokens`
  下标为 CTC ID，`models` 记录各源文件 SHA256。
- [逐语种映射 TSV](../configs/phonemes/zh_en_ar_mfa_g2p_mapping.v1.1.tsv) 用
  `language` 和 `source_unit` 查 `shared_unit` 或 `token_id`。`fst_symbol_id` 是真实
  FST 标签编号，`external_symbol_id` 记录官方外置表编号。生产打标使用 `token_id`。

## 英语解码用内嵌符号表

固定英语快照中，外置 `graphemes.sym` 有 324 项与 FST 同编号处不匹配，
外置 `phones.sym` 有 31 项与 FST 同编号处不匹配，且缺少 FST ID 70 的 `m̩`。
同一快照提供的 **`graphemes.txt` 和 `phones.txt` 与内嵌表导出结果逐字节相同**。
用错两份 `.sym` 时，本地的 `cat` 变成 `cʰ æ n`，使用匹配表时输出 `cʰ æ t`。
既有英西葡 v0.3 固定的是 `.txt`，此项错位限于上述两份 `.sym`。
普通话内嵌和外置的输入、输出表逐项一致。阿语 FST 使用 UTF-8 输入和包内外置
`phones.sym` 解码。模型声明中额外的 3 个普通话、8 个英语音素在相应 FST 输出中没有
出现，因此它们留在 JSON 的来源差异记录中。

英语运行时采用官方 `.txt` 或内嵌符号表。如果手头只有英语 `.sym`，
可以将 G2P 目录复制到新位置，再从原 FST 导出两份匹配的符号表。下面的 `EN_G2P`
指固定快照的 G2P 子目录。
[MFA 3.4.3 G2P 读取代码](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/v3.4.3/montreal_forced_aligner/g2p/generator.py)

```bash
EN_LOCAL=$(mktemp -d)
cp -a "$EN_G2P"/. "$EN_LOCAL"/
fstsymbols --save_isymbols="$EN_LOCAL/graphemes.sym" \
  --save_osymbols="$EN_LOCAL/phones.sym" "$EN_G2P/model.fst" /dev/null
```

## 从 G2P 发音到 CTC 标签

按空白取完整的 G2P 输出单元，使用对应语言的 TSV 行映射。原始阿语输出如
`s a l a m`，映射后为 `ar_gp:s ar_gp:a ar_gp:l ar_gp:a ar_gp:m`。中英仍按现有 IPA
单位约定进入通用读取器。任何无法映射的原始符号整句进入复核。三种模型都有输入字符
约束，先核对输入字母表再生成发音，防止丢弃不认识的字符后还生成部分读音。
这份表提供新的统一 ID，训练标签及相应的热词索引需要用同一版表生成。

## 已做的验证

用 OpenFst 1.8.4/Pynini 2.1.7 扫三份实际 FST 的全部输出弧，134、70、42 个非 epsilon
输出均有有效 CTC ID。使用 MFA 3.4.3 的重写实现以正确输入运行了九个词。

| 语言 | 测试输入 | 实际 G2P 输出 | v1.1 ID |
| --- | --- | --- | --- |
| 普通话 | `nǐ` | `ɲ i˨˩˦` | `197 97` |
| 英语 | `cat` | `cʰ æ t` | `68 167 130` |
| 阿语 | `salam` | `s a l a m` | `43 19 33 19 35` |

这些是词级 G2P 输出和编码的验证。原始阿文转写到 GlobalPhone 输入及目标语料打标
仍待单独接入验证。本地只读资产验证命令如下。

```bash
python -B -m pytest tests/test_zh_en_ar_mfa_g2p_vocab.py
sha256sum configs/phonemes/zh_en_ar_mfa_g2p_{vocab,mapping}.v1.1.*
```
