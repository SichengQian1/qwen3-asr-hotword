# 昇腾 910 系列中英音素检索测试包：环境、启动与 H200 对照

**交付对象：离线检索，不是 1036 的服务补丁。** 输入本地 Qwen3-ASR-1.7B 原始快照和 WAV 清单；输出逐 2 秒前缀及句尾的门控 Top5/Top7，供 context learning 使用。不启动 vLLM、ACL Graph、投机解码、WebSocket 或 LLM，不修改生产服务，不加载第二份 Decoder。H200 对照入口也只加载音频塔。两端共享同一份上游 Qwen 0.0.6 音频模块源码（附 LICENSE），使用显式 eager attention；代码不导入同名的定制 qwen_asr 服务包。

## 1. 先分配芯片，再配测试环境

报告描述的是 Atlas 800I A3、SOC ascend910_9391、A3 镜像；“910B4”作为交接称呼，实际机型以测试机 npu-smi、驱动和 SOC 为准。报告中的宿主 NPU4/Chip0、设备节点 davinci8、线上容器逻辑 0 三者不是同一种编号。快照时该芯片约 54.6 GB/64 GB 在用，其余芯片也有服务。**先请管理员分配独占的空闲芯片或维护窗口；在独立测试容器只暴露这颗芯片。不要在 1036 容器里安装依赖、抢占芯片或停服务。** 若实际是另一台 910B4，使用那台机器匹配的基础镜像和驱动，不照搬 A3 镜像。脚本使用容器内单颗可见 NPU，启动时打印实际芯片名及可用显存。

以报告中的已配套镜像为基础：Python 3.12、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post4、CANN 9.1.0。这里 +cpu 是底座构建标识，算子通过 torch_npu 在昇腾执行。不要安装 cu12 或 H200 的 requirements。保留镜像中配套的驱动挂载、CANN 环境与 NPU 工具；测试容器不用启动服务入口或加载其补丁。其他硬件组合按[昇腾 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)选型。

包内音频塔按训练前端的 transformers 4.57.6 工作；报告中的 5.5.4 留给生产服务。下面命令在**独立测试容器**执行。先由管理员按镜像方式加载 CANN 环境，再在可写目录创建测试 venv（继承镜像自带的 torch 和 torch_npu）。将变量改为当地位置；WHEELHOUSE 使用 Python 3.12 / Linux aarch64 轮包及其传递依赖。支持内网包源时也可用内部索引替代离线安装参数。

```bash
npu-smi info
python3 -m venv --system-site-packages ./venv-rag
source ./venv-rag/bin/activate
WHEELHOUSE=/actual/path/to/aarch64-wheelhouse
printf '%s\n' 'torch==2.10.0+cpu' 'torch-npu==2.10.0.post4' 'numpy==1.26.4' > ascend-constraints.txt
python -m pip install --no-index --find-links "$WHEELHOUSE" -c ascend-constraints.txt \
  'transformers==4.57.6' 'tokenizers==0.22.2' 'huggingface-hub>=0.34,<1' \
  'safetensors==0.8.0' 'librosa==0.11.0' 'soundfile==0.14.0' 'PyYAML>=6,<7' 'rapidfuzz==3.14.5'
python -c "import torch,torch_npu,transformers; print(torch.__version__,torch_npu.__version__,transformers.__version__,torch.npu.is_available(),torch.npu.device_count())"
```

预期最后三项为 4.57.6、True、1。镜像可能同时装有依赖 5.x 的 vLLM，两种 Transformers 放在不同环境里，不在测试 venv 启动该服务。测试进程只读取本地快照（报告中是 /models/Qwen3-ASR-1.7B）。无需在下游运行 MFA、下载 Qwen 权重或安装上游 qwen-asr 服务栈。

## 2. 用同一个包准备两端输入

将**同一份** zh_en_ascend2696.zip 复制到两台机器，不要分别重新打包。核对交付方给出的 SHA256，解压到各自的空目录：

```bash
sha256sum zh_en_ascend2696.zip
unzip zh_en_ascend2696.zip
cd zh_en_ascend
python run_ascend.py --model /models/Qwen3-ASR-1.7B --check
```

此检查仅校验包内资产、基座哈希及音素 ID。基座须使用训练时的同一快照；服务以 FP16 加载原始权重不改变源文件哈希，重保存或量化过的快照则是不同资产。

选一条含明确热词、长于 2 秒的真实录音，将原 WAV 同字节复制到两端，不重新编码。以下示例将这条音频命名为 demo.wav，并放到 input.jsonl 同目录。建议再加一条无目标录音及一条英文录音，避免“双方都空”被解释成有效召回。清单不传参考答案：

```bash
printf '%s\n' '{"sample_id":"demo01","group":"paired","language":"zh","audio_path":"demo.wav"}' > input.jsonl
sha256sum demo.wav
```

## 3. 在两端分别跑同表、同精度推理

**H200：** 在原项目环境中，从这次解压的 zh_en_ascend 目录运行；将卡号改成分配的空闲 H200。不要用旧 2,411 词 delivery 当基准。

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_ascend.py --backend cuda --dtype bfloat16 \
  --model /glusterfs_103/models/Qwen3-ASR-1.7B --manifest input.jsonl --output h200.zip
```

**昇腾：** 在独占测试芯片的容器和上述 venv 中运行。如果容器映射与报告相同，内部选择逻辑 0；不要把宿主 NPU ID 4 填作容器设备号。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -u run_ascend.py --backend npu --dtype bfloat16 \
  --model /models/Qwen3-ASR-1.7B --manifest input.jsonl --output ascend.zip
```

先测 BF16 对 BF16，Head 两端均为 FP32。需要测线上 FP16 条件时，两条命令同时改成 float16 并使用新的输出名。每次推理产出 ZIP 和紧邻的 `.zip.run.json`（包、词表和结果哈希、后端、设备、精度）。保留这两个文件。当前入口顺序处理清单，未提供断点续算，不覆盖已有结果。短语音优先；每步重算累计前缀，长录音的时间和显存不能按单个 2 秒块估算。

## 4. 逐前缀对比 Top5 和 Top7

把两端的四个文件汇集到同一目录，保留文件名；在解压的包内运行（示例假设四个文件就在当前目录）：

```bash
python compare_delivery.py h200.zip ascend.zip --out paired_diff.json
```

工具先比对同一包、词表、精度、输出哈希、样本 ID、原 WAV 哈希和帧数，再逐时间步比。4.3 秒音频按 2.0、4.0、4.3 秒比较，整 4 秒只有两步；Top5 和 Top7 独立列出。

| 指标 | 解释 |
|---|---|
| steps / ordered_word_matches | 总步数 / 候选词及其顺序完全相同的步数。 |
| same_word_sets / both_empty | 同一批词（不论排序）的步数 / 两端均空的步数。 |
| max_abs_score_delta_common_word | 同一个时间步、两端共同候选词的最大分数绝对差。shared_word_comparisons 为比较次数，零次时差值为 null。 |
| paired_steps | 所有时间步的结束时间和两端的实际候选词、读音、分数，可直接并排查看同一句。 |
| differences | 词面、顺序或读音发生变化的时间步索引，在 paired_steps 中查看该步详情。 |

这张表衡量**后端一致性**，不是相对人工目标的召回率。只看 ZIP 哈希也不合适：音频路径可不同，分数可能有数值差。出现分歧先看首个不同前缀、相同词的分数是否在 0.75 附近，并考虑编辑比、后验门控和候选排序；仅从 delivery 不能归因到某个算子。交回 paired_diff.json、两份回执和两端运行日志。若报不支持的算子，保留完整首个 traceback、实际芯片名和 CANN/torch_npu 版本，不要直接改分数门限。

## 5. 词表与下游接口

以 2,411 词联合表为底，保留旧行和多读音顺序；304 次业务列举按大小写和空白规范化后为 293 个词面，复用已有 8 词，新加 285 个。结果为 **2,696 个词面、2,697 条发音**。影视标题内部中文逗号、“！”、“·”与原词面保留；目录标题和说明文字不入表。“下一首”和“下1首”等同音不同词面均保留，仍分别参与竞争。`assets/word_audit.json` 记录原表哈希、实际增量、复用、重复及读法。MFA 使用本项目中文训练前端（普通话 3.0.0，混排英文走美式 3.3.0）。数字 2 读“二”、2026 读“二零二六”、1.5 读“一点五”、TV 读字母，AgentOS 用 “agent oh ess” 作为当前业务读法；这是索引读法，不改展示词。口语“历史、我的、返回”和短命令加入后也可能增加负句提示；留一批含这些短词的负例给后续质量评测。

ZIP 仍是 delivery/{durations,audio_manifest,stream_contexts}.json。stream_contexts 按模式 RAG_top5_stream / RAG_top7_stream → group@@language → sample_id → 步列表，候选字段 word/phoneme/score。空列表表示本步不提示。索引及重排仍在 CPU，不调用生产服务的强制中文逻辑。此包验证到 context learning 的输入；接生产服务的图捕获、动态 Prompt 与 rollback 是另一项集成。
