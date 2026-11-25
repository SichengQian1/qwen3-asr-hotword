# 英语／埃及阿语音素标注：下游接入交接

日期：2026-11-25。代码分支：`codex/g2p-coverage-scan`。
核对代码基线：`00ec03589ac045c7f2073bb1ee8b53b29e8d25bb`。
本文记录**现用实现**，不包含尚未批准的发音规则调整。不需要重新训练或加载Qwen即可做文本音素标注。

## 1. 接入结论

| 语言 | 当前打标路径 | 输出如何进入英阿CTC词表 |
| --- | --- | --- |
| 英语 `en` | 文本规范化／分词 → 固定美式MFA G2P 3.3.0 → 官方Phonetisaurus Top1 → 项目音素单位规范化 | 英语单位不加语言前缀，保持原英语ID |
| 埃及阿语 `ar_egy` | 阿文分词 → CALIMA EGY r13词法分析、逐词MLE Top1 → `diac` → 项目埃及发音规则 | 每个IPA单位加 `ar_ipa:`，再查询固定ID |

**只有阿语不再走MFA。英语仍走MFA，但要使用本轮固定的3.3.0模型与转换代码，不能直接拿更早英西葡实验的词典或mmservice结果替换。**

阿语不使用GlobalPhone罗马化，不把阿文字母直接送入旧 `arabic_g2p.zip`。
本轮训练和验证选的是声明为埃及方言的来源，MSA不补样；`ar_msa`代码保留供旧实验使用，不是当前默认。

输出是**文本推导的音素序列**，没有音素起止时间，不是强制对齐结果，也不是从音频识别出的音素。
训练文本和检索热词必须使用同一版本前端、规则及ID表。下游若直接接收本项目生成的标签，优先使用已提供的ID数组，不要重新猜测切分。

## 2. 工区环境与路径

以下命令均在H200的**已有容器内**执行，而非宿主机；CPU即可，无需指定GPU。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
conda activate qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

宿主机对应仓库为 `/home/star/q00933266/qwen3-asr-hotword`，只用于查文件／Git时不要混淆容器路径。
运行下面示例前保持仓库根目录为当前目录。文档命令只生成新的下游文本结果，不启动训练、不覆盖已有标签。

| 用途 | 环境／目录 |
| --- | --- |
| 命令编排环境 | `qwen3-asr-hotword`；工区已使用Python 3.12 |
| 阿语真正执行环境 | `/host_home/star/q00933266/qwen3-asr-hotword/models/arabic_native_v1/runtime/bin/python`，CAMeL Tools **1.6.0** |
| 阿语数据根目录 | `/host_home/star/q00933266/qwen3-asr-hotword/models/arabic_native_v1/data` |
| 英语真正执行环境 | `conda run --no-capture-output -n aligner python -B ...`；配置认可MFA **3.4.0／3.4.3** 运行程序 |
| 英语G2P资源根目录 | `/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/hf_g2p_20261029_v03` |
| 英阿统一ID表 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/phonemes/en_ar_native_ipa_vocab.v1.1.json` |
| 阿语当前规则 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/arabic_pronunciation_policy.json` |
| 阿语环境／模型身份配置 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/arabic_native_frontend.workzone.json` |
| 英语模型身份／环境配置 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/mfa_v03_foundation.workzone.json` |

MFA运行程序版本与G2P模型版本是两回事：`aligner`里的3.4.0／3.4.3运行程序加载的是本轮固定的英语G2P **3.3.0** 模型。

### 2.1 阿语实际模型文件

相对于 `models/arabic_native_v1/data/`：

| 文件 | 本轮用途与固定数据包版本 |
| --- | --- |
| `morphology_db/calima-egy-r13/morphology.db` | 埃及词法库，0.2.0 |
| `disambig_mle/calima-egy-r13/model.json` | 埃及逐词MLE模型，0.1.0 |
| `catalogue.json` | CAMeL导入所需的本地目录信息；避免导入时意外尝试联网下载 |
| `morphology_db/calima-msa-r13/morphology.db` | 历史MSA分支，0.4.0；当前EGY标注不使用 |
| `disambig_mle/calima-msa-r13/model.json` | 历史MSA分支，0.2.5；当前EGY标注不使用 |

现有`verify_assets()`会按配置检查EGY和MSA全部固定文件及LICENSE，因此保留现有完整数据目录。
下游打包这些模型时一并保留LICENSE；配置记录CALIMA数据包许可证为GPL v2。

只有环境缺失时才运行已有安装入口：

```bash
python -B scripts/label_arabic_native.py setup
```

这里只执行`setup`。**不要运行该旧脚本的`label`来代替修正后的阿语前端**。
安装使用项目配置中的局部TLS校验绕过和固定SHA校验，不改全局系统证书设置。
已有环境正常时不需要重装，也不需要调用`camel_data`重新下载。

### 2.2 英语实际模型

```text
models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa/model.fst
models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa/meta.json
models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa/phones.txt
models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa/graphemes.txt
```

资源目录还包含README、LICENSE、dictionary及`verified.json`等校验材料；下文调用会校验配置要求的文件，不只需要一个FST。
固定来源：`MontrealCorpusTools/english_mfa`，revision `1009e3c6c689f10875146a5e4e5c2233f8eaf922`。
FST SHA256：`001bc21fe1934f3b1e86ae006aca9673c6be60de33698101396bec8545b3f7f3`。
这些是本轮固定身份，不表示要求下游自动追踪网上最新模型。

## 3. 统一音素表及映射约定

固定文件：`configs/phonemes/en_ar_native_ipa_vocab.v1.1.json`，**167类**。

```text
vocab SHA256:
b337255bf239fdec3b4bdb0770bf39941fee74246336774c7ece8741d4534bf2

当前 arabic_pronunciation_policy.json SHA256:
d1af333f28f354d196b0100e8b0308b889470bbbfc5f2c31d358460471b8bfaf

当前 arabic_native_frontend.workzone.json SHA256:
289c36e6f988cf147d7c1f335af9a8baaf57f932164cc9207128bf32942b6d1a
```

ID就是JSON中`tokens`数组的下标，从0开始：

- ID 0：`<blank>`，CTC空白，不加入文本标签。
- ID 1：`<unk>`，保留特殊类别；当前完整标签不靠它吞掉未知音素。
- ID 2–71：英语音素，共70个；连同两个特殊类别构成保留的72类英语前缀。
- ID 72–166：阿语音素，共95个，使用`ar_ipa:`命名空间。

例如英语`b`是ID **4**，阿语`ar_ipa:b`是ID **74**；符号看起来相同，训练类别仍不同。
阿语规则示例：若CALIMA给`بيت`的分析是`بَيت`，当前规则输出：

```json
{
  "phones": ["b", "eː", "t"],
  "phoneme_tokens": ["ar_ipa:b", "ar_ipa:eː", "ar_ipa:t"],
  "phoneme_token_ids": [74, 84, 116]
}
```

这是指定`diac`输入下的规则示例，不是对任意录音的人工发音鉴定。

注意：

1. 不重新排序、压缩或给两种语言的相同IPA合并ID。
2. `eː`、`tʃ`、带鼻化／送气等修饰的单位必须作为完整token；不要按Unicode字符拆分。
3. 阿语用前端返回的`ids`；英语用项目`tokenize_ipa_to_vocab()`输出。不要给阿语裸IPA调用英语的无前缀映射，否则会落到错误类别。
4. 句级标签按词序拼接，不加空格token、词界token或blank，也不先做CTC重复折叠。
5. 当前表含长辅音单位，例如`ar_ipa:bː`；不能随意改成两个短辅音ID。

## 4. 阿语如何标注

### 4.1 实际步骤及原因

1. **传入阿文正文**：去掉外层`language Arabic<asr_text>`标记，不传整个Swift JSON给前端。保留用于判断停顿的标点，不做把不同辅音合并的额外“清洗”。
2. **规范化和切词**：复用项目`input_words()`。数字、特定符号、控制字符或非阿文词进入review，不自动拼读数字或猜外文人名。
3. **词法分析与选解**：`MorphologyDB`加载EGY库，`Analyzer(..., backoff="NONE")`，`MLEDisambiguator(..., top=1)`。使用每个词排名第一的分析，接受`lex`／`spvar`来源。
4. **读取`diac`**：EGY库这条路径不提供可直接使用的CAPHI，因此需项目规则从带标音拼写转换到IPA。`diac`本身不等于已听审的真实发音。
5. **应用当前发音修正**：下表列出实际启用状态。
6. **映射到ID**：对每个IPA单位构造`ar_ipa:<单位>`，在167类表中查ID。未知单位不自动降级为近似音或`<unk>`。
7. **接入方检查review标志**：尤其检查`unreviewed_exception_words`，存在时不可只因函数返回`ids`就直接激活。

实现：`src/qwen_hotword/phonemes/arabic_pronunciation.py`中的`PronunciationFrontend`。
它继承旧分析器加载逻辑，但应用新的发音规则；不要直接使用旧`CalimaFrontend.label()`生成本轮新标签。

### 4.2 当前规则，不把待做项写成已完成

| 项目 | 当前行为 |
| --- | --- |
| `_`附着词连接符 | 启用：转换前跳过，不再因此整句拒收；不等于所有附着处省音均已建模 |
| 无显式保护标音的`ay/aw`结构 | 启用：转`eː/oː`；显式sukun、shadda或滑音上元音等保护条件保留 |
| 分析器把原文辅音改写为同源字母 | 启用：满足逐字对齐条件时恢复原文指定辅音；不强行对齐不同长度词形 |
| 原文保留的`ث/ذ/ظ` | 启用当前开罗约定`s/z/zˤ`；不是逐条音频方言识别 |
| `ج/ق`基础规则 | 当前采用`ɡ/ʔ`，属于现有EGY前端约定 |
| 词尾`aː/iː/uː` | 启用缩短为`a/i/u` |
| 词尾代词`h` | 仅明确`enc0=3ms_dobj/3ms_poss`且为`i/u+h`时去掉；不删除全部词尾h |
| 句中冠词连读`linked_article` | **关闭**；代码存在候选实现，但未进入当前标签 |
| MSA停顿词尾`msa_pause_endings` | **关闭**；当前纯EGY也不走MSA分支 |
| `ثورة、ده、عشان`词汇例外 | 仍是`candidate`；按surface与diac精确命中时标记待审，未自动批准 |
| 上下文消歧 | 仍是逐词MLE Top1；没有上下文神经模型纠正同形词 |

模式：训练句子用`sentence`，独立热词／短语用`isolated`；`linked`用于显式连读调用。
当前连读开关关闭，**模式名字不会自动去除词首喉塞音**。以后启用规则时，必须同步训练标签和热词索引，并重新评估旧CTC兼容性。

### 4.3 单条快速调用

已有环境就绪时，在容器仓库根目录：

```bash
python -B scripts/run_arabic_pronunciation_refresh.py query \
  --group ar_egy --mode isolated --text 'بيت'
```

句子：

```bash
python -B scripts/run_arabic_pronunciation_refresh.py query \
  --group ar_egy --mode sentence --text 'لأن دي مش حاجة سهلة يعني.'
```

外层命令自动转到`models/arabic_native_v1/runtime/bin/python`，设置`CAMELTOOLS_DATA`并检查资产。
终端可能包含阶段日志，**不能把整段stdout直接当一个纯JSON接口解析**。服务集成／批量生成优先使用下面Python接口。
`query`会返回待审信息，不会替下游自动拒收词汇例外；接入方必须检查它。

### 4.4 阿语批量接口示例

下面演示两条输入。实际接入时替换`requests`，前端只加载一次，逐条复用。
输出写到全新随机目录，终端显示校验日志及输出路径；review项不输出可直接使用的ID。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
PYTHONPATH=.:src models/arabic_native_v1/runtime/bin/python -B - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import tempfile
import subprocess
from pathlib import Path

root = Path.cwd()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg_path = root / "configs/arabic_native_frontend.workzone.json"
policy_path = root / "configs/arabic_pronunciation_policy.json"
vocab_path = root / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json"
cfg, policy, vocab = read(cfg_path), read(policy_path), read(vocab_path)
assert sha(vocab_path) == "b337255bf239fdec3b4bdb0770bf39941fee74246336774c7ece8741d4534bf2"
assert sha(policy_path) == "d1af333f28f354d196b0100e8b0308b889470bbbfc5f2c31d358460471b8bfaf"
assert importlib.metadata.version("camel-tools") == cfg["camel_tools_version"]
assets = root / cfg["assets"]
os.environ["CAMELTOOLS_DATA"] = str(assets)  # 必须在导入CAMeL之前设置。
assert (assets / "catalogue.json").is_file(), "本地catalogue缺失，请先运行setup"
from scripts.label_arabic_native import verify_assets
from qwen_hotword.phonemes.arabic_pronunciation import PronunciationFrontend
verify_assets(cfg)
frontend = PronunciationFrontend(assets, "ar_egy", vocab["tokens"], policy)
requests = [
    {"id": "term-001", "text": "بيت", "mode": "isolated"},
    {"id": "utterance-001", "text": "لأن دي مش حاجة سهلة يعني.", "mode": "sentence"},
]
(root / "outputs").mkdir(exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="downstream_ar_labels_", dir=root / "outputs"))
identity = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "camel_tools_version": importlib.metadata.version("camel-tools"),
            "vocab_sha256": sha(vocab_path),
            "policy_sha256": sha(policy_path), "frontend_config_sha256": sha(cfg_path)}
with (out / "labels.jsonl").open("x", encoding="utf-8") as target:
    for item in requests:
        result = dict(item, language="ar", dialect="EGY", identity=identity)
        try:
            label = frontend.label(item["text"], mode=item["mode"])
            if label["unreviewed_exception_words"]:
                raise ValueError("unreviewed_pronunciation_exception:" +
                                 ",".join(label["unreviewed_exception_words"]))
            ids = label["ids"]
            if not ids or any(not 72 <= i < len(vocab["tokens"]) for i in ids):
                raise ValueError("invalid_Arabic_ids")
            result.update(status="ready", label=label,
                          phoneme_tokens=[vocab["tokens"][i] for i in ids],
                          phoneme_token_ids=ids)
        except ValueError as error:
            result.update(status="review", reason=str(error))
        target.write(json.dumps(result, ensure_ascii=False) + "\n")
print(out / "labels.jsonl")
PY
```

`label`中的关键字段：

| 字段 | 含义 |
| --- | --- |
| `words`、`diacritized_words` | 分词及选中的diac |
| `phones`、`ids` | 原始IPA单位列表及固定CTC ID |
| `word_traces` | 每词分析、旧新读音、命中规则、候选例外、音素边界 |
| `word_traces[].phone_start/phone_end` | 音素数组的半开区间下标，不是时间戳 |
| `unreviewed_exception_words` | 当前命中但未批准的例外词，非空则隔离 |
| `context_disambiguation` | 当前为`word_based_MLE_Top1` |
| `human_pronunciation_review_completed` | 当前为false，表示没有人工发音验收 |

原始接口的`phones`不带前缀，下游统一token取`vocab["tokens"][id]`即可。
不要把字符串拼接后的IPA重新按字符拆成ID。

## 5. 英语如何标注与调用

### 5.1 与训练一致的处理

1. 使用`mfa_round_common.words()`：NFC、casefold、统一撇号／连字符、按字母及组合符号切词；词内合法撇号和连字符保留。
2. 数字、待读符号、非空白控制字符返回问题；不默认念出数字，不通过删除数字“修好”文本。
3. 按唯一词批量调用固定英语FST。调用的是MFA官方`PhonetisaurusRewriter`，`num_pronunciations=1`、严格字符表检查，不使用查不到就套旧词典的fallback。
4. 按完整MFA空格分隔单位映射，使用`mfa_units_v0.3`：NFD匹配组合符号，保留送气、鼻化、长短和复合单位；仅使用代码中明确的tie-bar别名及`g→ɡ`等规范化。
5. 每个词必须状态ok、恰有一个可用候选、无OOV，才按词序拼接ID。一词失败则整条待审，不返回残缺标签。

虽然英语训练音频包含埃及口音，**英语文本标签使用的仍是这套美式G2P约定**，不是另有埃及口音英语G2P模型。

### 5.2 现成英语词典缓存

已完成训练用缓存：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_ar_english_g2p_20261110_172826/english_g2p.json
```

结构为顶层`words`对象，键为规范化单词，值含`status`及`candidates`。
历史缓存记录的是`en_ar_mfa_g2p_vocab.v1.0.json`身份；它的前72类与当前v1.1完全一致，所以英语ID可以原样使用。
**这仅适用于英语前缀，不能复用旧表的`ar_gp:*`阿语ID。**

缓存未收录的新热词，按下面入口生成即可，不需要再次扫描训练语料或重新提Encoder。

### 5.3 新英语词／短语批量生成示例

在`qwen3-asr-hotword`环境中执行；编排函数会自动到`aligner`环境做G2P，不在主环境安装MFA。
此示例把worker的目标表明确覆盖为本轮英阿v1.1，避免误用基础配置默认的英西葡表。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
PYTHONPATH=.:src python -B - <<'PY'
import hashlib
import json
import tempfile
import subprocess
from pathlib import Path
from qwen_hotword.training.mfa_round_common import words
from qwen_hotword.training.mfa_round_labels import generate, top1_tokens
from qwen_hotword.phonemes.mfa_v03_assets import write_json

root = Path.cwd()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
vocab_path = root / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json"
vocab = read(vocab_path)
assert sha(vocab_path) == "b337255bf239fdec3b4bdb0770bf39941fee74246336774c7ece8741d4534bf2"
legacy = read(root / "configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json")
assert vocab["tokens"][:72] == legacy["tokens"][:72]
texts = ["hello world", "Cairo University"]
prepared = [(text, *words(text)) for text in texts]
wanted = {w for _, ws, issues in prepared if not issues for w in ws}
(root / "outputs").mkdir(exist_ok=True)
out = Path(tempfile.mkdtemp(prefix="downstream_en_labels_", dir=root / "outputs"))
foundation = read(root / "configs/mfa_v03_foundation.workzone.json")
foundation.update(vocab=str(vocab_path), vocab_sha256=sha(vocab_path))
worker_config = out / "english_foundation.json"
write_json(worker_config, foundation)
generated = generate(root, {
    "foundation_config": str(worker_config),
    "assets_dir": "models/mfa/hf_g2p_20261029_v03",
    "g2p_words_per_part": 2000, "g2p_workers": 1,
}, out, {"en": wanted})["en"] if wanted else {}
identity = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "model_revision": "1009e3c6c689f10875146a5e4e5c2233f8eaf922",
            "fst_sha256": "001bc21fe1934f3b1e86ae006aca9673c6be60de33698101396bec8545b3f7f3"}
results = []
for text, ws, issues in prepared:
    ids, missing = top1_tokens(ws, generated) if not issues else ([], [])
    reasons = issues + missing
    if not reasons and (not ids or any(not 2 <= i < 72 for i in ids)):
        reasons.append("invalid_English_ids")
    row = {"language": "en", "text": text, "words": ws, "identity": identity,
           "vocab_sha256": sha(vocab_path), "status": "review" if reasons else "ready"}
    if reasons:
        row["reasons"] = reasons
    else:
        row.update(phoneme_tokens=[vocab["tokens"][i] for i in ids], phoneme_token_ids=ids)
    results.append(row)
write_json(out / "labels.json", results)
print(out / "labels.json")
PY
```

每词的原始MFA IPA、score、tokens和ID在输出目录`g2p/en/part-*/predictions.jsonl`中。
`score`是G2P输出的模型分数，不是“正确发音概率”，不能当成音频置信度。
上面是文本批量标注，不运行音频识别，不改既有训练集。

## 6. 下游应保存的最小结果

每条至少保存：

- 业务ID、原始词面／正文、language、阿语dialect、调用mode。
- `phoneme_tokens`、`phoneme_token_ids`；失败时status=review及原因，不激活残缺ID。
- vocab SHA256；阿语另存policy SHA256、CAMeL版本及资产身份；英语另存模型revision／FST SHA256。
- 发音代码版本（`git rev-parse HEAD`）；阿语保留`word_traces`便于后续追溯修正。

词表新增、发音表修改与模型重训是不同动作：按**现有**约定给新词生成ID可更新热词索引；
改变现有音素ID或发音规则则可能与已训练Head不一致，不能只改下游一侧。
当前已用于检索的英阿Head为纯埃及语料×3第29轮，关联配置是
`configs/egy_hotword_review.workzone.json`；模型不是本步骤的依赖，但选择CTC时必须匹配上述167类表。

## 7. 训练语料与文本接口的边界

训练流水线还会做源结构检查、两份文本一致性判断、音频头读取、去重、保护集隔离、时长配额与CTC长度检查。
这些不是G2P函数本身的功能。正文生成标签不表示音频一定合格，也不直接产生可训练manifest。

当前纯埃及流程：

```text
原始Swift JSON/JSONL
 → 来源盘点与正文准入（保存assistant、response；一致候选保留assistant原正字法）
 → PronunciationFrontend(... ar_egy ...).label(text, mode="sentence")
 → 未审核例外／无分析／OOV等进入review
 → 音频与CTC长度检查、去重和分组划分
 → 英阿各约392.51h，阿语按时长open-source:gaoshan:coarse = 8:1:1
 → train.jsonl / validation.jsonl / test.jsonl
```

英语对应原埃及口音英语＋美式补样池，复用本轮英语G2P标签。最终清单路径：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_egy_480_ctc_v1/train.jsonl
/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_egy_480_ctc_v1/validation.jsonl
/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_egy_480_ctc_v1/test.jsonl
```

目录名带480是上限配置；该轮实际约392.51h/语种，不能按目录名写成已使用480h。
×3使用同一批冻结数据。这里只提供核对位置，下游接文本前端不需要读取这些大清单。

## 8. 已知不足与交接核验

- 当前阿语仍为自动标注实验，没有完成高频词例外词典、上下文选音和人工逐音素验收。
- `linked_article`仍关闭；`ثورة、ده、عشان`对应未批准分析会进入review。不要为增加覆盖率直接忽略标志。
- 原文属于EGY来源不等于逐条音频经方言鉴定；文本规范化一致也不等于音文完全一致。
- 函数返回成功只说明该版本规则与ID映射可用，不等于已听审正确。
- 不输出词级时间戳，不根据CTC预测反向修改参考标签。

交接时先核对：两种环境可用、固定资产SHA通过、167类词表SHA一致；
用上面小样确认英语ID在2–71、阿语在72–166，未知输入进入review；再处理业务词表。
本文命令依据现有源码核对；本地检查代码片段语法、ID示例和规则测试，未代替工区完整模型执行。

## 9. 代码导航

| 功能 | 仓库文件 |
| --- | --- |
| 阿语现用规则／统一查询类 | `src/qwen_hotword/phonemes/arabic_pronunciation.py` |
| CALIMA加载、阿文分词、基础转换 | `src/qwen_hotword/phonemes/arabic_native.py` |
| 阿语安装和固定资产校验 | `scripts/label_arabic_native.py`（仅setup／校验复用） |
| 阿语查询命令 | `scripts/run_arabic_pronunciation_refresh.py query` |
| 本轮纯EGY训练标注调用 | `scripts/run_en_egy_dynamic.py`中的`label_job` |
| 本轮热词标注调用 | `scripts/evaluate_egy_corpus_hotwords.py`中的`phonemize` |
| 英语规范化／分词 | `src/qwen_hotword/training/mfa_round_common.py`、`training/g2p_prep.py` |
| 英语Top1批量编排 | `src/qwen_hotword/training/mfa_round_labels.py`中的`generate`／`top1_tokens` |
| MFA官方重写器调用 | `src/qwen_hotword/phonemes/mfa_v03_runtime.py` |
| 英语音素单位到ID | `src/qwen_hotword/phonemes/coverage.py`中的`_tokenize_mfa_units` |

原分析建议保存在`docs/ARABIC_CTC_PER_ANALYSIS_2026-11-12.md`；
本文件说明当前生效方式，不把分析文档中的全部建议当作已启用规则。
