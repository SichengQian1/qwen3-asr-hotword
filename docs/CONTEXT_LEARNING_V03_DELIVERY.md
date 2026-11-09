# 三语 Context Learning 交付与 MFA G2P 用法

交付版本为 v0.3 / new70。适用美式英语、拉美西语和巴西葡语。

## 6800 条交付

沿用上次 same-step 的带噪回归集。英语三国各 2000 条，西语和葡语各四个 wave、每个
100 条。用本轮 100 类 Head、v0.3 音素表和新版热词发音索引重新计算 26,354 个累计
音频前缀。每条音频按 2 秒、4 秒到实际末尾切点输出，Top5 和 Top7 共用同一排序。
数据集键、音频 stem 和门限取自上次交付的封存配置。

在 H200 容器的项目环境运行，GPU 参数填当次分配的物理卡号。以下命令以 GPU4 为例。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_multilingual_stream.py \
  --config configs/multilingual_stream_new70.workzone.json \
  --effect same-step --gpu 4
```

运行入口先校验旧样本清单、带噪音频 SHA、三件套和时序，再加载 Encoder 与 CTC。
中断后用同一源码、配置、设备和输出目录，加 `--resume` 续跑。可在运行命令上加
`--audit-only` 单独执行 CPU 预检。看到终端 `status=completed` 后交付下列目录。

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/multilingual_stream_contexts_v03_new70_noise10db_v1/full_same-step/delivery/
```

| 文件 | 用途 |
|---|---|
| `stream_contexts.json` | `RAG_top5_stream` / `RAG_top7_stream` → 数据集键 → 音频 stem → 按时间顺序的候选数组。候选为 `word`、`phoneme`、`score`，空列表表示该步没有通过门限的词。 |
| `durations.json` | 数据集键及音频 stem 对应的秒数。 |
| `audio_manifest.json` | 对应的音频路径、SHA、时长和前缀计数。 |

`phoneme` 取该时间点胜出的新版发音，`word` 供下游构造文字提示。同一词面的多种
发音合并后占一个候选位置。每个时间点使用当步累计音频重新检索。

供方运行报告和 SHA 封存留在 `full_same-step` 根目录。终端摘要或 `summary_bundle.zip`
可用于回传核验。实际交给 context learning 的是 `delivery/` 三份 JSON。

## 这轮模型和 G2P 的变化

本轮冻结 Qwen Encoder，用新标签从头训练了 100 类 Temporal 2× CTC Head。
v0.3 保留原 90 个 ID，追加 10 个音素单位。新版按 MFA 空白分隔的单位映射音素，
保留鼻化和唇化，例如 `kʷ` 映射到一个 ID。相同业务词面使用固定三语 G2P Top1
重新生成发音和 token IDs，运行时据此重建 Anchor 与倒排索引。

原 4000 词面中，本轮 JSONL 有 3986 个有效词面和 3998 条语言发音记录。评测分母
沿用旧目标，未激活的目标词计作漏检。三件套路径、SHA、加载和静态模型导出要求见
[新版 CTC 推理接入说明](CTC_V03_INFERENCE_HANDOFF.md)。

| 语言 | 方言 G2P | 本轮模型版本 | 固定 revision |
|---|---|---|---|
| 英语 | `english_us_mfa` | **3.3.0** | `1009e3c6c689f10875146a5e4e5c2233f8eaf922` |
| 西语 | `spanish_latin_america_mfa` | **3.2.0** | `aba092613e8d8f074f381a09e610ecf64704ef20` |
| 葡语 | `portuguese_brazil_mfa` | **3.4.3** | `70c9d6fafa338a38269a6b770a8c7d9575d0b96f` |

表中版本来自 G2P 的 `meta.json`。训练时实际使用的 `aligner` 环境中，MFA 软件为
**3.4.0**。项目 worker 的版本检查允许 MFA 软件 3.4.0 和 3.4.3。固定模型和文件 SHA
记录在 `configs/mfa_v03_foundation.workzone.json`。

## 用新版 MFA 给新词生成发音

已有热词库直接读取交付的发音 JSONL。新增词走下面的 CPU G2P 入口。接手环境需要
仓库代码、v0.3 音素表、已有 `aligner` 环境和配套三语资源目录。资源包括 FST、符号
表、模型元数据及配置固定的其他文件。工区已准备好下列目录。

```text
/host_home/star/q00933266/qwen3-asr-hotword/models/mfa/hf_g2p_20261029_v03
```

以下命令建立一个新的工作目录，输入每行一个语言和单词。把示例词换成业务需要的词。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
mkdir -p "$PWD/outputs"
JOB=$(mktemp -d "$PWD/outputs/mfa_v03_words_XXXXXX")
cat > "$JOB/request.jsonl" <<'JSONL'
{"language":"en","word":"cat"}
{"language":"es","word":"casa"}
{"language":"pt","word":"carro"}
JSONL

python -B - "$JOB/g2p_top1.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path("configs/mfa_v03_foundation.workzone.json").read_text(encoding="utf-8"))
config["g2p_num_pronunciations"] = 1
Path(sys.argv[1]).write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
PY

MFA_ROOT_DIR="$JOB/mfa_work" conda run --no-capture-output -n aligner \
  python -B scripts/prepare_mfa_v03.py worker \
  --config "$JOB/g2p_top1.json" \
  --assets-dir "$PWD/models/mfa/hf_g2p_20261029_v03" \
  --request "$JOB/request.jsonl" \
  --worker-output "$JOB/predictions.jsonl"

cat "$JOB/predictions.jsonl"
cat "$JOB/predictions.jsonl.report.json"
```

结果中的 `candidates[0]` 包含 `ipa`、`tokens`、`token_ids` 和 `oov_units`。入库条件
为 `status=ok`、恰好一个候选、`usable=true` 且 `oov_units` 为空。输入语言码用 `en`、
`es`、`pt`，入库时巴葡发音记录的语言标签为 `pt-BR`。

多词热词先用 `src/qwen_hotword/training/mfa_round_common.py` 的 `words()` 拆词并处理
报告的数字、符号或控制字符问题。逐词生成后按词序拼接 ID。任一词失败则整条发音
转入待处理列表。保留业务 `surface`，生成新版发音记录，再刷新相应检索索引。

本次代码交付后的实际 6800 条前缀运行由 H200 工区执行。本地测试覆盖输入身份、时序、
新版音素输出、格式校验和恢复。实际检索指标及下游消费情况以工区结果为准。
