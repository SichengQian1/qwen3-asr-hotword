# 中英音素 CTC / RAG：下游接入与 delivery 接口

本文面向接手推理或 context learning 的工程师与开发助手。**处理新音频时，输入是本地 Qwen3-ASR-1.7B 基座、一份 WAV 清单和本运行包；输出是每条音频在各时间点可用的热词提示。** 如果已经拿到某次评测的 `delivery.zip`，直接从第 4 节读取，无需加载模型。

## 1. 交付版本与职责

| 项目 | 本版约定 |
|---|---|
| 基座 | 下游自备完整的 Qwen3-ASR-1.7B 本地快照，包含权重分片、索引、音频处理器、tokenizer 等文件。 |
| CTC 分支 | 中英 480h 版，固定 epoch 25 的 Temporal 2× Head；取冻结 Encoder 的 `thinker.audio_tower.ln_post`，每帧 1,024 维，输出 183 类。 |
| 音素表与热词表 | `assets/vocab.json` 的数组下标是 CTC ID（0 为 blank，1 为 unk）。`assets/joint2411.jsonl` 有 2,411 个词面、2,412 条发音。两种语言均检索整张联合表。 |
| 运行接口 | `run_delivery.py` 是文件批处理 CLI：读取完整 WAV，按真实接收顺序模拟逐步检索，整批结束后写一个 ZIP。它不提供麦克风、WebSocket 或逐步推送服务。 |
| 下游职责 | 选择 Top5 或 Top7，将**本步**词面作为 Decoder 的参考上下文；音频解码、文本输出、流式回改和转写评分由下游负责。 |

运行包 `zh_en_rag_kit.zip` 是可复用程序。结果包 `delivery.zip` 是一批指定音频的候选时间线。结果包不内嵌 WAV；跨机器转写时，提供同一份音频，按 `audio_sha256` 核对文件内容。

## 2. 从解压到第一条真实音频

环境：Linux、Python 3.12、支持 BF16 的 NVIDIA CUDA GPU；本项目工区使用 H200。推理使用 PyTorch 和 qwen-asr，本入口会加载基座，再取 Encoder 特征。按基座加载规划显存。无需 vLLM、MFA、训练语料或训练缓存。

在终端执行下列步骤。将 `MODEL` 和 `WAV` 改为**接收方机器上的实际绝对路径**；GPU 编号使用分配给本任务的物理卡号。

```bash
sha256sum zh_en_rag_kit.zip   # 与交付方提供的 SHA256 比对
unzip zh_en_rag_kit.zip
cd zh_en_rag
MODEL=/actual/path/Qwen3-ASR-1.7B
WAV=/actual/path/example.wav
```

在独立环境或现有 Qwen 环境中，先安装与 `requirements.txt` 版本匹配的 CUDA 版 PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
python -m pip check
python run_delivery.py --model "$MODEL" --check
```

`--check` 比对包内文件和基座登记文件的 SHA256，检查音素 ID、词数及读音数。看到 `Assets verified. No GPU inference was run.` 表示资产校验通过。下一步用真实音频验收 GPU 和完整流程：

```bash
python - "$WAV" > input.jsonl <<'PY'
import json, sys
print(json.dumps({"sample_id": "demo01", "language": "zh", "group": "demo",
                  "audio_path": sys.argv[1]}, ensure_ascii=False))
PY
CUDA_VISIBLE_DEVICES=0 python -u run_delivery.py \
  --model "$MODEL" --manifest input.jsonl --output demo_delivery.zip
```

程序先检查音频清单，再加载模型并建立索引；结束时校验时间线、写 ZIP、打印输出绝对路径。一次成功运行可以产生全空的候选时间线：这表示没有词通过门控。切换到多条音频只需在输入清单中追加记录。

## 3. 输入契约：一行 JSON 对应一条音频

| 字段 | 规则 |
|---|---|
| `sample_id` | 必填非空字符串。在同一 `group + language` 内唯一，建议使用稳定的业务录音 ID。 |
| `language` | 必填，取 `zh` 或 `en`。它设置该音频的处理语言；不自动判断语种，也不把检索范围缩成中文表或英文表。 |
| `audio_path` | 必填，指向可读、非空的 WAV。相对路径以 **JSONL 文件所在目录** 为基准。 |
| `group` | 可选非空字符串，默认 `custom`，不得含 `@@`。输出数据集键为 `group@@language`。 |

清单使用 UTF-8，不需要转写参考或目标热词。多声道按声道平均成单声道；非 16 kHz 音频在每一步仅将**当时已收到的前缀**重采样至 16 kHz。原始采样率、帧数、时长和文件哈希保存在输出清单中。

## 4. 输出契约：一个 ZIP、三个文件

| ZIP 内路径 | 键与内容 |
|---|---|
| `delivery/stream_contexts.json` | `[mode][group@@language][sample_id][step]` → 本步按分数降序排列的候选列表。模式名精确为 `RAG_top5_stream` 和 `RAG_top7_stream`。 |
| `delivery/durations.json` | `[group@@language][sample_id]` → 原音频时长，单位秒。 |
| `delivery/audio_manifest.json` | `[group@@language][sample_id]` → 录音信息，含 `audio_path`、`audio_sha256`、`frames`、`sample_rate`、`channels`、`duration_seconds` 和 `step_count`。 |

每项候选恰有三个字段：`word` 是展示和提示用的原词面；`phoneme` 是本次匹配到的读音字符串；`score` 是 0 到 1 的检索分数，不是 ASR 转写正确的概率。每步允许 0～K 项。同一步的 Top5 等于同一步 Top7 的前五项；两种模式任选其一，不拼接。

**时间规则（same-step）：** 步号从 0 开始。若时长为 D 秒，第 i 步对应音频从开头到 `min(2 × (i + 1), D)` 秒，步数为 `ceil(D / 2)`。例如一条 **4.3 秒**录音：

| step | 检索使用的音频 | Decoder 可使用的候选 |
|---:|---|---|
| 0 | 0～2.0 秒 | 只读第 0 步；列表为空就使用基础提示。 |
| 1 | 0～4.0 秒 | 用第 1 步替换第 0 步。上一轮出现、本轮消失的词不继续提示。 |
| 2 | 0～4.3 秒 | 用第 2 步做句尾提示；即使最后只增加 0.3 秒，也有这一条记录。 |

整 4 秒恰好两步；短于 2 秒也有一个句尾步。边界是**候选可用时间**，不是该词的起止时间，也不是推理耗时。以下代码直接读取第 2 节生成的包，可作为 Decoder 接入点：

```python
import json
from zipfile import ZipFile

mode, key, uid = "RAG_top7_stream", "demo@@zh", "demo01"
with ZipFile("demo_delivery.zip") as z:
    def load(name):
        return json.loads(z.read(f"delivery/{name}.json"))
    contexts = load("stream_contexts")
    meta = load("audio_manifest")[key][uid]
    duration = load("durations")[key][uid]

steps = contexts[mode][key][uid]
assert len(steps) == meta["step_count"]
for i, items in enumerate(steps):
    end_s = min(2 * (i + 1), duration)
    words = [item["word"] for item in items]
    print(uid, end_s, words)
    # Decoder 接口接收截至 end_s 的音频，以及本步的 words。
    # words 非空时按当前顺序作为参考词；为空时省略热词提示。
```

Decoder 以音频为依据，不把候选强制写进文本。按整句做离线对照时可显式选取最后一步；按流式协议回放时逐步读取，禁止用末步候选回填前面的音频。

## 5. 检索过程与运行边界

1. **加载。** 用 `assets/ctc_head_best.pt` 内的结构和权重恢复 Head，并核对词表顺序。用已保存的发音和 token ID 在内存中建立倒排索引；固定表推理不会重新生成发音。
2. **听音。** 每个边界都把从句首到当前边界的音频前缀送入 Encoder，Head 产生音素预测，再作 CTC 解码。该入口逐条、逐前缀计算；它不复用上一步的 Encoder 特征。长录音的计算量随累计前缀增长。
3. **找词。** 用 2／3／4 个音素的片段匹配倒排索引，先筛最多 64 条发音候选，再做局部重排，将同一词面的多个读音合并。随后按固定门控筛选：分数 ≥ 0.75、编辑比 ≤ 0.35、后验置信度 ≥ 0.50；第一与第二名的最低分差为 0。
4. **导出。** 保存通过门控的两种 Top-K 时间线，不填满空位，不导出门控前原始排名。程序不计算召回率、PER 或 WER；召回评测需另备目标标注，WER 需另有 Decoder 输出和参考转写。
5. **部署与扩展。** CLI 加载一次模型处理一份清单；未提供跨任务常驻服务或逐条断点续跑。长清单按批拆分，给每批独立输出名。固定词表、音素表和 Head 按版本配套；需要新增词时另走制表流程（普通话 MFA 3.0.0、美式／英式 MFA 3.3.0 及对应映射），重新发布资产和校验信息。

## 6. 验收与排查

| 现象 | 处理 |
|---|---|
| `missing/changed pinned input` | 按报错路径区分包内资产与基座文件。前者重新解压原包，后者切换到配套的完整快照；用原始 SHA 做校验。 |
| `duplicate …`、音频找不到 | 检查同一个数据集键内的 ID、JSONL 所在目录、音频挂载和读权限；同一 ID 可以在不同组单独出现。 |
| `provide --manifest and a fresh --output path` | 填入输入清单，并为本次 ZIP 选择尚不存在的路径。原结果会保留。 |
| CUDA 启动或显存报错 | 在当前环境检查 CUDA 版 torch、物理卡分配、BF16 支持和空闲显存；`CUDA_VISIBLE_DEVICES` 内首张卡对应程序的 `cuda:0`。 |
| ZIP 能读但对不上音频 | 用数据集键和 sample_id 联合定位，核对音频哈希与时长；跨机器时把音频路径映射至本机。不要按文件顺序配对。 |
| 最后一步有词，前面为空 | 逐步打印候选验证其出现时间。更长的前缀提供了更多发音证据；回放时保持原来的空列表。 |

交接验收记录建议留四项：运行包 SHA256、包内 `manifest.json` 的 `revision` 和 `source_run_sha256`、实际使用的音频哈希、一条录音的逐步候选。对比两套后端时，同一音频、同一时间边界和同一模式配对。
