# 三语 MFA G2P 使用说明

## 当前使用的模型

v0.3 使用以下固定的 G2P 模型。工区 `aligner` 环境使用 MFA 软件 **3.4.0**。

| 语言及输入代码 | G2P 模型 | 模型版本 | 固定资源地址 |
|---|---|---|---|
| 美式英语 `en` | `english_us_mfa` | **3.3.0** | [英语模型快照](https://huggingface.co/MontrealCorpusTools/english_mfa/tree/1009e3c6c689f10875146a5e4e5c2233f8eaf922/g2p/english_us_mfa) |
| 拉美西语 `es` | `spanish_latin_america_mfa` | **3.2.0** | [西语模型快照](https://huggingface.co/MontrealCorpusTools/spanish_mfa/tree/aba092613e8d8f074f381a09e610ecf64704ef20/g2p/spanish_latin_america_mfa) |
| 巴西葡语 `pt` | `portuguese_brazil_mfa` | **3.4.3** | [葡语模型快照](https://huggingface.co/MontrealCorpusTools/portuguese_mfa/tree/70c9d6fafa338a38269a6b770a8c7d9575d0b96f/g2p/portuguese_brazil_mfa) |

## 工区资源地址

模型及其符号表按 `en`、`es`、`pt` 存在以下目录。各文件的版本和 SHA 记录在
`configs/mfa_v03_foundation.workzone.json`。

```text
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/hf_g2p_20261029_v03
```

## 生成新词发音

在工区项目环境运行下面的 CPU 命令。将示例词换成业务词，每行填一个语言代码和
单词。入口使用 Top1 发音，并按 v0.3 音素表生成 token ID。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
mkdir -p "$PWD/outputs"
MFA_G2P_RUN_DIR=$(mktemp -d "$PWD/outputs/mfa_v03_words_XXXXXX")
cat > "$MFA_G2P_RUN_DIR/request.jsonl" <<'JSONL'
{"language":"en","word":"cat"}
{"language":"es","word":"casa"}
{"language":"pt","word":"carro"}
JSONL

python -B - "$MFA_G2P_RUN_DIR/g2p_top1.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("configs/mfa_v03_foundation.workzone.json").read_text(encoding="utf-8"))
config["g2p_num_pronunciations"] = 1
Path(sys.argv[1]).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
PY

MFA_ROOT_DIR="$MFA_G2P_RUN_DIR/mfa_work" conda run --no-capture-output -n aligner \
  python -B scripts/prepare_mfa_v03.py worker \
  --config "$MFA_G2P_RUN_DIR/g2p_top1.json" \
  --assets-dir "$PWD/models/mfa/hf_g2p_20261029_v03" \
  --request "$MFA_G2P_RUN_DIR/request.jsonl" \
  --worker-output "$MFA_G2P_RUN_DIR/predictions.jsonl"

cat "$MFA_G2P_RUN_DIR/predictions.jsonl"
cat "$MFA_G2P_RUN_DIR/predictions.jsonl.report.json"
```

结果中的 `candidates[0]` 包含 `ipa`、`tokens`、`token_ids` 和 `oov_units`。入库条件
为 `status=ok`、恰好一个候选、`usable=true` 且 `oov_units` 为空。输入语言码用 `en`、
`es`、`pt`，入库时巴葡发音记录的语言标签为 `pt-BR`。

多词热词用 `src/qwen_hotword/training/mfa_round_common.py` 的 `words()` 拆词并处理
返回的问题。逐词生成后按词序拼接 ID，任一词失败时将整条发音转入待处理列表。
