# 本轮中英音素表与工区文本转音素用法

本文对应**普通话＋美式英语＋英式英语，共享183类音素表**。这里的“生成音素”是把文字转成发音序列及token ID，不是合成语音。

## 1. 需要交付的音素文件

工区仓库根目录：

```text
/host_home/star/q00933266/qwen3-asr-hotword
```

以下路径均相对该根目录：

| 文件 | 用途 |
|---|---|
| `configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json` | 正式音素表；`tokens`数组下标即CTC类别ID |
| `configs/phonemes/zh_en_mfa_g2p_mapping.v1.0.tsv` | 普通话／美音／英音各自的MFA输出单位到共享token及ID的对照 |
| `configs/zh_en_foundation.workzone.json` | 固定模型版本、路径和文件SHA256 |

音素表为181个发音单位＋`<blank>`（ID 0）＋`<unk>`（ID 1），共183类。
音素表SHA256：

```text
ef5a411afbdb238a400302da2d871fe64f8f94e8d065fdf045b8bf5f39a2b5cb
```

保留声调、长音、送气、唇化、腭化和成音节标记；一个多字符IPA单位仍是一个token，不能逐字符拆分。
相同规范单位共享ID；英式相对“中＋美音”新增 `fʷ vʷ əw ɛː ɜ ɜː`。
三份固定G2P模型没有独立重音符号输出，本轮没有额外删除或补造重音。
`mapping.tsv`中的`fst_symbol_id`是MFA内部编号，**不能直接作为CTC的token ID**，要使用`token_id`列。

## 2. 配套MFA模型

| 路由标记 | 范围 | G2P模型版本 | 仓库内模型目录 |
|---|---|---|---|
| `zh` | 大陆普通话 | `mandarin_china_mfa` 3.0.0 | `models/mfa/hf_g2p_20261113_zh_en/zh/g2p/mandarin_china_mfa` |
| `us` | 美式英语 | `english_us_mfa` 3.3.0 | `models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa` |
| `uk` | 英式英语 | `english_uk_mfa` 3.3.0 | `models/mfa/hf_g2p_20261113_zh_en/uk/g2p/english_uk_mfa` |

上述目录需保留配套的 `model.fst`、`meta.json`、`phones.txt`、`graphemes.txt` 等已校验文件。
不要自行替换为下载页面的另一套符号编号；本项目中文、英式符号表已按对应FST内嵌编号整理。
中文和英式资产已随仓库交付；美式资产复用工区此前部署的版本，转移到其它机器时也要一起提供。
**单独拿到音素表JSON，不能完成G2P转换。**

工区MFA在已有Conda环境 `aligner` 中运行；项目支持已审查的MFA运行时3.4.0／3.4.3。
这是软件版本，与上表G2P模型的3.0.0／3.3.0是两回事。以下方法只检查和调用现有环境，不升级它。

## 3. 文字怎么变成这张表的音素

```text
中文：汉字文本 → 固定中文前端 → 带调拼音词 → 中文MFA Top1 → 共享音素token → ID
英语：英文文本 → 分词及规范化 → US或UK MFA Top1 → 共享音素token → ID
```

- 中文复用训练前端：`pypinyin==0.55.0`处理连续中文片段的读音，`jieba==0.42.1`分词且`HMM=False`；保留词典声调，不额外执行变调规则。**中文MFA的直接输入是带调拼音，不是汉字。**
- `group=zh`的中英混合文本中，拉丁字母词按`us`转换。纯英语按调用方指定的`us`或`uk`路由，不自动判断口音。
- 每个词只取G2P Top1，再按原顺序拼接；训练标签不插入词边界或blank。需要保留分词信息时保存下方示例的`units`字段。
- 数字、待确定读法的多字母全大写缩写、未知字符、G2P失败或音素OOV，整条输出`rejected`，不输出部分成功的ID序列，也不自动用`<unk>`顶替。
- 多音字、专名读音受上下文影响；生成成功只表示映射完整，不等于人工确认发音正确。使用时保持与训练同一套前端和映射。

对应实现：`src/qwen_hotword/training/zh_en_labels.py`（前端与MFA配置）、
`src/qwen_hotword/phonemes/mfa_v03_runtime.py`（G2P）、
`src/qwen_hotword/phonemes/coverage.py`（音素到共享ID）。

## 4. 工区可直接运行的示例

在原 `qwen3-asr-hotword` 环境、仓库根目录执行。只修改`inputs`中的文字和路由即可；
每次生成独立结果目录，不接触训练输出，不加载Qwen，不需要GPU。
中文前端的两个固定小型包如尚未准备，会下载到本次输出的私有目录并核验SHA，不修改原Python环境。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
PYTHONPATH=src python -B - <<'PY'
from pathlib import Path
from tempfile import mkdtemp

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import run_worker, write_json
from qwen_hotword.training.zh_en_common import read, rows, save_rows
from qwen_hotword.training.zh_en_foundation import verify_assets
from qwen_hotword.training.zh_en_labels import (
    frontend, install_frontend, make_worker,
)

root = Path.cwd()
cfg = read(root / "configs/zh_en_noise70_dual.workzone.json")
foundation = read(root / cfg["foundation_config"])
verify_assets(root, foundation)
(root / "outputs").mkdir(exist_ok=True)
out = Path(mkdtemp(prefix="zh_en_phonemes_", dir=root / "outputs"))
install_frontend(out, cfg["frontend_packages"])
worker, assets, prefix = make_worker(root, out, foundation, cfg)
vocab = load_phoneme_vocab(root / foundation["vocab"])

# group只能填写zh、us、uk；同一英文可分别生成美式、英式两个候选。
inputs = [
    {"id": "zh_1", "group": "zh", "text": "北京医院"},
    {"id": "us_1", "group": "us", "text": "Central hospital"},
    {"id": "uk_1", "group": "uk", "text": "Central hospital"},
]
if len({x["id"] for x in inputs}) != len(inputs):
    raise ValueError("input id必须唯一")
prepared, unique = [], set()
for item in inputs:
    if item["group"] not in {"zh", "us", "uk"}:
        raise ValueError("group必须是zh、us或uk")
    try:
        units = frontend(item["text"], item["group"])
        unique.update((u["language"], u["word"]) for u in units)
        prepared.append({**item, "units": units, "errors": []})
    except ValueError as error:
        prepared.append({**item, "units": [], "errors": [str(error)]})

request, prediction = out / "request.jsonl", out / "predictions.jsonl"
save_rows(request, ({"language": lang, "word": word} for lang, word in sorted(unique)))
lookup = {}
if unique:
    run_worker(root, worker, assets, prefix, request, prediction)
    lookup = {(r["language"], r["word"]): r for r in rows(prediction)}

results = []
for item in prepared:
    ids = []
    for unit in item["units"]:
        result = lookup[(unit["language"], unit["word"])]
        candidates = result["candidates"]
        if result["status"] != "ok" or not candidates or not candidates[0]["usable"]:
            item["errors"].append(f'{unit["word"]}: {result["status"]}')
        else:
            ids.extend(candidates[0]["token_ids"])
    if item["errors"]:
        ids = []  # 任一词失败，不能留下部分发音冒充成功。
    results.append({
        **item,
        "status": "rejected" if item["errors"] else "ok",
        "phoneme_tokens": [vocab.tokens[i] for i in ids],
        "phoneme_token_ids": ids,
        "vocab_sha256": foundation["vocab_sha256"],
    })
save_rows(out / "phonemes.jsonl", results)
write_json(out / "input.json", inputs)
print("结果文件：", out / "phonemes.jsonl")
print("逐词原始MFA输出：", prediction if unique else "无可转换词")
PY
```

`phonemes.jsonl`每行对应一个输入，包含原文、路由、前端分词、共享音素和ID；只接收`status=ok`的完整结果。
`predictions.jsonl`保存逐词MFA原始IPA、候选分数、token映射及失败原因，便于定位具体哪个词没有转换成功。
需要兼容美音和英音的英文热词，分别用`us`、`uk`生成并保留两个发音候选，**不能把两种读音首尾拼成一个发音**。
