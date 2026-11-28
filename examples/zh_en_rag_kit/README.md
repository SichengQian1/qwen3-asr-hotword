# 中英音素 CTC / RAG 下游运行包

下游开发与 AI 接入请先读 [接口与运行边界交接说明](DOWNSTREAM_HANDOFF.md)，内含时间步示例和可运行的 delivery 读取代码。

固定版：Qwen3-ASR-1.7B + 中英 480h 版 epoch 25 的 Temporal 2× CTC Head，183 类，联合表 2,411 词面 / 2,412 个读音。输入一条或多条 WAV，输出每个 2 秒音频前缀（含最后不足 2 秒的句尾）的门控后 Top5 和 Top7。中英文音频都查同一张中英联合表。交付边界到 context learning 的输入为止；不跑 LLM Decoder，不输出转写文本。

## 首次运行

1. 解压后进入 `zh_en_rag/`。建议 Linux、Python 3.12、支持 BF16 的 NVIDIA GPU（本项目在 H200 环境）；本包不是 Ascend/vLLM 镜像。准备你们自己的完整 Qwen3-ASR-1.7B 本地快照，包括音频处理器及 tokenizer。推理入口会加载基座模型，不是只占 Head 的显存。
2. 优先复用能跑 Qwen 的 Python 环境。在新环境先按本机驱动和 CUDA 配置从相应 PyTorch CUDA 源安装与 requirements.txt 匹配的 torch，然后运行 `python -m pip install -r requirements.txt`。离线环境需自行准备这些依赖及传递依赖的 wheel，包内不含 Python 环境。固定表已有读音，推理不用安装 MFA。
3. 先校验模型与资产（从当前目录运行）：

```bash
python run_delivery.py --model /actual/path/Qwen3-ASR-1.7B --check
```

4. 写一个 UTF-8 的 `input.jsonl`，每行一条；无需参考转写、目标词或数据集标签。audio_path 可为绝对路径或相对清单的路径，language 是 zh 或 en，group 可选。同组同语言内 sample_id 必须唯一：

```json
{"sample_id":"demo01","group":"demo","language":"zh","audio_path":"./demo.wav"}
```

换成自己有的 WAV 和物理卡号；解压目录无需是原仓库：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_delivery.py --model /actual/path/Qwen3-ASR-1.7B --manifest input.jsonl --output my_delivery.zip
```

5. 看到 `status: pass` 并打印 ZIP 绝对路径后可交付。每次选择不存在的输出路径；此轻量入口未做逐音频断点续跑，长清单可按批拆分。验收时用一条真实录音核对逐步候选；若有旧 delivery 和同一 WAV，可逐步比对词面、排名与空列表。没有命中也可正常产出；退出成功不代表一定召回。

## 下游读取约定

- `delivery/stream_contexts.json`：`模式 → group@@language → sample_id → 按时间排列的候选列表`。模式是 RAG_top5_stream、RAG_top7_stream。每项包含 word、phoneme、score；第 i 步仅使用截至该步的音频，同一步可用。空列表就不提示；不补齐 K，也不把最后一步回填之前。不是每词的时间戳。
- `delivery/durations.json` 与 `delivery/audio_manifest.json`：相同的数据集键、ID 和时长，清单附音频路径与哈希。ZIP 不含 WAV。进一步接 Decoder 时，读取本步 word 做参考词，下游自行实现提示注入与转写。
- `assets/ctc_head_best.pt`、`assets/vocab.json`、`assets/joint2411.jsonl` 必须配套。词表内的 token IDs 对应 183 类的顺序；运行时建 Anchor 倒排索引，再局部重排、合并多发音、门控。代码固定 Anchor 64、分数 0.75、最大编辑比 0.35、最低后验 0.5。
- 本包固定词表，不提供新增词的 G2P 工具。后续自建词表需另交普通话 MFA 3.0.0、美/英式 MFA 3.3.0、映射与规范化工具；不能直接编辑封版资产继续沿用本版校验。
