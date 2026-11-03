# 三语音素热词 RAG：v0.3 / 100类 CTC Head 接入交接说明

日期：2026-11-03。面向已有旧版音素RAG推理流程的接手同事。

## 1. 交付范围先说明

**使用本轮已经生成的有效热词库时，核心运行资产是三个文件：新音素表、新热词发音
记录、新CTC Head检查点；连同本说明一起交付。** 接收方须已有相应Head实现和支持
v0.3标签规则的推理代码。这三个文件不是独立可运行的软件包，也不是三个路径替换
之后就能保证所有旧程序直接兼容。

- 接收方已有本仓库相应实现：成套加载三个文件，重建运行时索引，核对配置即可。
- 接收方只有旧90类程序或自行实现的服务：还需适配100类输出、音素单位、JSONL读取
  和索引重建；具体代码衔接见第5节，不需要重写整条RAG算法。
- 后续需要新增/修改热词：还需固定版本G2P资源和相同文字/音素处理工具，见第6节。
  它们是热词更新所需的构建工具，不要求每段音频推理都运行MFA。
- 如要完整交接原4000词面清单及未激活条目，建议另附`hotword_report.json`、
  `hotword_unresolved.jsonl`，以及业务侧原词面/ID清单。三件套中的JSONL只包含有效
  发音记录，不能从中还原全部未激活条目；这两份审计文件不是CTC前向的必需输入。

本次训练的是新的CTC分支，Qwen3-ASR-1.7B主模型仍冻结。流程仍为：

```text
音频 → 原Qwen音频Encoder → ln_post 1024维特征
     → 新100类CTC Head → 音素检索 → Top-k热词文字
     → 原Prompt注入逻辑 → 原Qwen LLM Decoder → 转写
```

本文件只说明交付与接入，不会启动训练、重标注、安装环境或替换生产资产。

## 2. 三个核心文件的准确路径

以下为H200容器内路径。搬到其他机器后可以改路径，文件内容及其配套关系需保留。
其中Head和热词记录身份来自已核验的工区结果，本地未加载其权重或直接读取完整索引；
音素表在本地也已校验SHA。接收方应对实际收到的文件重算SHA。

### ① 新音素表

```text
/host_home/star/q00933266/qwen3-asr-hotword/configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json
```

- JSON，`tokens`按下标定义ID，共100类：98个音素单位、`<blank>`和`<unk>`。
- `<blank>`为0、`<unk>`为1；不能重排、按字母重建ID或删除暂未使用的类别。
- 文件还定义了`normalization_policy.tokenization = mfa_units_v0.3`。

### ② 新热词发音记录

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/mfa_v03_noise70_scratch_v1/indexed_pronunciations.v03.jsonl
```

- JSONL，一行一条有效发音记录；不是一个大JSON对象，也不是已编译的内存倒排数组。
- 本轮原4000个词面中有3986个有效词面、3998条语言发音记录，14个词面未激活。
  语言发音条数与业务词面数不是同一指标；未激活条目没有用旧发音回退。
- 它已用与本轮训练相同的固定G2P Top1和v0.3规则生成，交接现有库时不必再跑一遍MFA。

### ③ 新CTC Head检查点

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/mfa_v03_noise70_scratch_v1/head/ctc_head_best.pt
```

- 这是最佳CTC Head检查点，不是完整Qwen模型，也不是310P可直接执行的OM文件。
- 使用其中`head_config`、`vocab_tokens`、`state_dict`，按结构重建后严格加载。
- 本轮为TemporalUpsample Head：输入1024、hidden512、kernel5、dropout0.1、时间倍率2、
  输出100，可训练参数843876。推理需`eval()`，关闭训练时Dropout行为。

三个文件应作为一个版本成套传递，不与旧版互相替换。

| 文件 | SHA256 |
|---|---|
| 音素表 | `59fbac2f4d7251056f63b30597a8ab4fd4ae7e2a1801a2c527e680e6fdaaec7f` |
| 热词发音JSONL | `b478451da20fc15d99fed4f3da99f4164cdc8d266e29d05f2b08c4de80d4a98d` |
| CTC Head PT | `a6eba051856dbe4c7ac793dd237c06b7f98cd951a76199f57ee3b84c57483229` |

工区只读校验命令，不加载模型：

```bash
sha256sum \
  /host_home/star/q00933266/qwen3-asr-hotword/configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json \
  /host_home/star/q00933266/qwen3-asr-hotword/outputs/mfa_v03_noise70_scratch_v1/indexed_pronunciations.v03.jsonl \
  /host_home/star/q00933266/qwen3-asr-hotword/outputs/mfa_v03_noise70_scratch_v1/head/ctc_head_best.pt
```

音素表内仍保留建表阶段的`foundation_candidate_requires_audio_review`和
`training_ready=false`元数据。后续已按项目授权完成训练和H200评测，完成状态见
`HANDOFF.md`第0.198节及工区`training_completed.json`；这些旧元数据不是运行时
验收结论。交接本轮固定文件时不要自行改字、重存或修改这些字段，否则SHA将变化。

## 3. 相比旧版，具体改变了什么

| 项目 | 旧90类版本 | 本轮100类版本 | 接入影响 |
|---|---|---|---|
| CTC词表 | v0.2，88个音素+两个特殊类 | v0.3，98个音素+两个特殊类 | 新Head、新外部表和后处理保持一致 |
| ID | 原90个位置 | 原90个位置保留，尾部追加10个单位 | 保留旧ID不代表旧发音序列仍适用 |
| Head | 对应旧标签的90类权重 | 新标签、随机初始化从头训练的100类权重 | 不复用旧Head、旧导出模型或90类输出缓冲区 |
| 训练音频条件 | 旧版/上一轮抗噪条件 | 本轮每语种约70%记录加10dB噪声 | 是训练条件，正常推理不用主动加噪 |
| 发音标签 | 旧G2P及旧处理策略 | 固定新版三语G2P Top1及v0.3规则 | 三语热词均重新生成，不能只更新葡语几项 |
| 音素切分 | 旧全局替换及字符串匹配分支 | 保留MFA以空白分隔的完整音素单位 | 保留唇化、鼻化，按单位查ID |
| 未知单位 | 旧处理流程 | v0.3出现未知单位则整条发音不接受 | 不丢字符、截取部分发音或混入旧标签冒充成功 |
| 热词索引 | 基于旧发音序列构建 | 基于新发音记录重建 | 重建N-gram、Anchor位置、倒排数据和相关缓存 |

追加的10个单位为：`ã`、`cʷ`、`kʷ`、`pʷ`、`tʷ`、`ɟʷ`、`ɡʷ`、`ɾʲ`、`ɾ̃`、`ʁ`。
本轮没有新增重音CTC类别；不能把元音上的鼻化符号当成可随意删除的装饰。

例如，`kʷ`作为一个完整音素单位映射到一个ID，不能按旧规则改成两个单位`k w`。
新版G2P也可能改变只涉及旧90类的词序列，因此“ID没有越界”不是兼容性证明。

## 4. 热词JSONL怎样使用

主要字段如下，具体以实际文件为准，不手工编造发音或token IDs：

| 字段 | 含义和处理方式 |
|---|---|
| `hotword_id` | 发音记录ID，本轮生成的ID可能与旧版不同；不能假设旧业务ID可直接沿用 |
| `language` | 发音语言，如`en`、`es`、`pt-BR`；需保留 |
| `surface` | 热词原文字面，用于最终提示；保留大小写、重音等拼写 |
| `normalized` | 词面归并键；用于将多语言发音候选归并到同一业务词面 |
| `words` | 本轮文本处理得到的词单元 |
| `pronunciation` | 按空白分隔的音素字符串，便于查看 |
| `phoneme_tokens` / `token_ids` | 真正的检索标签；必须逐项对应本轮音素表 |
| `source` | 本轮为`pinned_new_G2P_Top1` |
| `validation_occurrences` | 本轮生成记录为0，是元数据，不是候选置信度或不可用标志 |

运行时使用`token_ids`建立检索索引，检索到发音候选后按词面归并、排名和门控，最后
向Decoder发送选中的`surface`文字。若下游依赖旧业务ID，需要显式维护映射，不能
把新发音ID、业务词面ID和文件行号混为一个东西。

索引加载至少检查：ID为整数且不含blank/unk、非空且在100类范围内、ID与音素单位
逐项一致、发音记录ID唯一。Unicode只接受规范等价，不通过删除鼻化/重音等信息
强行匹配。保留有效同音不同词面的候选，仍需通过现有排序和门控处理歧义。

旧倒排索引及Anchor位置不能继续使用。即使沿用相同N-gram/局部编辑距离算法，也要
以新`token_ids`重新构建。三个资产作为一个版本切换，请求固定使用同一版本，并清理
旧候选/音素后验等关联缓存，避免一次请求中混用两套标签。

## 5. 接入代码必须同步的地方

### 5.1 Head加载、维度和有效长度

沿用同一Qwen3-ASR-1.7B和`thinker.audio_tower.ln_post`的1024维特征，不能改取投影后
的2048维表示喂给Head。输入为`[B,T,1024]`；本轮Head输出为`[B,2T,100]`，每个样本
有效长度为其Encoder有效长度的2倍，不能把padding帧当真实帧，也不能继续按T解码。

仓库中的加载方式可参考以下片段。它依赖现有项目代码和PyTorch，代码为交接示例，
本说明的编辑过程没有执行它，也没有在本地加载工区PT：

```python
from pathlib import Path
import torch

from qwen_hotword.modeling.ctc_head import build_ctc_head_from_checkpoint
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

root = Path("/host_home/star/q00933266/qwen3-asr-hotword")
vocab_path = root / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json"
head_path = root / "outputs/mfa_v03_noise70_scratch_v1/head/ctc_head_best.pt"
vocab = load_phoneme_vocab(vocab_path)
payload = torch.load(head_path, map_location="cpu", weights_only=True)
if tuple(payload.get("vocab_tokens", ())) != tuple(vocab.tokens):
    raise ValueError("Head与外部音素表不配套")
head = build_ctc_head_from_checkpoint(payload)
if head.num_classes != 100 or head.time_upsampling_factor != 2:
    raise ValueError("不是本轮100类Temporal 2x Head")
head.load_state_dict(payload["state_dict"], strict=True)
head.eval()
```

完整工程实现见：

- `src/qwen_hotword/modeling/ctc_head.py`：结构、checkpoint工厂和输出长度。
- `src/qwen_hotword/inference/streaming_backends.py`：成套vocab校验、严格加载及FP32 Head。
- `src/qwen_hotword/inference/external_keyword_retrieval.py`：使用有效2T长度解码和查询。

若公司侧使用310P静态模型，需要用新Head重新导出/编译并匹配100输出、动态或固定T
以及有效长度处理；一个PT文件不是已有旧OM模型的替换权重。硬件上Head精度和算子
实现另按部署环境验证，不能直接据H200结果宣称已适配310P。

### 5.2 音素规则与输入格式

- `src/qwen_hotword/phonemes/coverage.py`读取音素表的tokenization政策，v0.3走
  `_tokenize_mfa_units`：按空白单位、NFD规范等价和有限精确别名处理，未知单位整条拒收。
  自研C++/服务侧若复制了旧规则，需同步该语义；不是只换JSON中的token数组。
- `scripts/run_external_keyword_retrieval.py`当前默认音素表仍为v0.2，其`--keyword-bias`
  读取的是含`keyword_sets`/`keyword_phonemes`的JSON对象。**不能把本轮JSONL路径直接
  填进去当作接口兼容。**
- 通用`hotwords.registry.load_hotword_table()`对`validation_occurrences>0`及同语言
  重复发音还有旧约束，本轮JSONL的计数字段为0。应参考本轮
  `src/qwen_hotword/evaluation/mfa_round.py`的`assets()`读取与校验方式，为实际服务
  衔接相应数据接口；不要仅把0改为1或去掉同音词来绕过旧读取器。
- 本轮评测还将新发音ID映射回原业务词面，再合并多语言变体。若接手服务不保留旧
  评测资产，应从有效JSONL按`normalized`维护发音→词面的映射，并与业务ID清单关联，
  不能让同一词面的不同语言发音占满Top-k。

### 5.3 可以复用的部分

Qwen主模型、音频处理原则、CTC去重/blank处理、Anchor召回及局部重排算法、Prompt
构造和Decoder均可复用。LLM tokenizer及其文本词表不替换为100类音素表；向Decoder
注入的仍是文字。三件套替换也不要求顺带更换推理引擎或重装vLLM环境。

本轮沿用原分组门限、Anchor参数及Top7完成评测，没有证明旧门限在新分数分布下
最优。接手时先复现已测配置，不擅自把某一旧入口默认值当成所有业务的统一门限。

## 6. 后续新增热词，额外需要什么

仅使用已交付有效JSONL推理时不必运行MFA；需要线上/离线新增词时，必须沿用下列
训练时固定G2P模型，不能每次下载任意“最新版本”：

| 语言 | 方言/模型 | G2P模型版本 | 固定仓库revision |
|---|---|---|---|
| 英语 | `english_us_mfa` | 3.3.0 | `1009e3c6c689f10875146a5e4e5c2233f8eaf922` |
| 西语 | `spanish_latin_america_mfa` | 3.2.0 | `aba092613e8d8f074f381a09e610ecf64704ef20` |
| 葡语 | `portuguese_brazil_mfa` | 3.4.3 | `70c9d6fafa338a38269a6b770a8c7d9575d0b96f` |

版本/文件SHA配置：`configs/mfa_v03_foundation.workzone.json`。
已准备资源目录：

```text
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/hf_g2p_20261029_v03
```

模型版本不等于MFA软件版本；本轮MFA软件沿用3.4.0，不要求把软件版本改成3.4.3。
使用本轮文本处理、新G2P Top1、v0.3映射及相同失败政策，参考
`src/qwen_hotword/training/mfa_round_common.py`的`words()`和
`src/qwen_hotword/training/mfa_round_labels.py`的`generate()`/`top1_tokens()`/`build_hotwords()`。
`generate()`依赖原准备配置和输出目录，不是给生产服务直接调用的独立在线API；生产
新增词接口需将这些政策封装成可复用工具，不能只拷一个model.fst自行猜输入输出。

新词应完整生成或返回明确失败，不静默丢字、不拼接部分成功结果。更新热词库时刷新
相应索引并记录新版本；一般不因此重训CTC Head。旧90类资产保留给旧Head回滚，新旧
两套不能混用。

## 7. 接手后的最小核验与已知边界

1. 重算三个文件SHA，与第2节一致；核对Head内完整`vocab_tokens`与外部表完全相同。
2. 核对100输出、blank=0/unk=1、2倍有效时间长度及1024维输入；Head使用推理模式。
3. 按新JSONL检查音素/ID对应，保留语言与词面，重建索引；确认未加载旧持久化索引。
4. 用少量现有音频检查“Encoder→Head→候选文字→实际Prompt→Decoder”接通，保存
   版本、候选和转写供复核。接手方对其自身服务/硬件做这一步，不必先重训或再跑全量。
5. 若业务仍要求原4000词面全部有效，先看14个未激活词的报告与失败清单，明确修复或
   隔离；不将3986有效词面的文件宣称为4000词全部覆盖。

已完成的依据：本轮新标签/100类Head训练及6800基础音频检索、最终ASR见`HANDOFF.md`
第0.198节；H200上Transformers与原生vLLM0.23全量后端对照见0.204节。结果不意味着
三语所有指标都提高，也不证明公司310P部署、静态导出或单实例两阶段特征复用已验证。
10组热词命中集合变化的方向仍待核查，不能在交接中改写成已完成或无影响。

本说明依据已核验的源代码及工区记录；未在本地读取完整Qwen、工区PT或重新生成热词。
仓库分支为`codex/g2p-coverage-scan`，相应说明的推送提交见交付消息。
