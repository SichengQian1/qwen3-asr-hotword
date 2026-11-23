# 埃及阿语：动态 Prompt 流式端到端模拟

## 工区运行

分支：`codex/g2p-coverage-scan`。在原 `qwen3-asr-hotword` 环境执行：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
python -u -B scripts/run_egy_streaming_e2e.py --gpu 4
```

默认物理 GPU 4，子进程内为 cuda:0。先自动进行20条工程联调（覆盖三个来源及正负例），
三组都完成后自动跑全量，不再需要手动确认。20条仅检查执行链路，不用于选择门限或报告
总体质量；全量包含这20条，复用已经完成的结果。`--check` 只检查输入与环境。

入口配置：`configs/egy_streaming_e2e.workzone.json`。默认输出：
`outputs/egy_ep29_streaming_e2e_mem15/`，不会覆盖整段检索或训练结果。

首次运行在Decoder启动前因空闲30.27 GiB小于50%预算69.91 GiB而退出，
之后曾用15%固定预算恢复。用户确认目前慢在三组ASR，现已改为**动态预算＋三组并批**。
目录继续沿用`mem15`历史名称，实际预算以`Decoder memory`日志为准。

若任务仍在执行，先在原终端按一次Ctrl+C并等待退出，再拉取代码执行上方命令。
本次不改配置文件或原contract；仅允许已审阅旧runner SHA `9c274bf1…543f5f3`
升级为资源/调度实现，记录`memory_runtime_upgrade.json`。已完成的前缀和ASR组合保留，
未完成组合重做。其他模型、数据、词表或算法代码变化仍拒绝混用。

## 固定输入

- 新 EGY ×3，第29轮 Head：`outputs/en_egy_480_up3_standalone_v1/head/ctc_head_best.pt`。
- r15 `checkpoint-5100-merged` 同时用于特征提取和 ASR 解码；由已冻结 Encoder plan 获取路径。
- 原整段评测 `outputs/egy_ep29_corpus_hotwords_v1/selected_audio.json` 的2078条原始音频，
  按时长 open-source/gaoshan/coarse 8:1:1；不另抽样、不加噪。
- 原 `table.json` 共383条、379条可检索。词表 SHA：
  `088052f81ae19f23373f0745225f717cae36f75e9925b9099b891207a8d217a0`。
- 580个目标词—音频对，包含发音不可用条目的目标，不缩小分母。
- 词表已经根据本批语料构造，因此这是语料条件化词表评测，不是未知业务词表盲测。

源 test 清单重新验哈希并重建确定性选样；再次检查训练/验证的音频和可识别录音隔离。
运行前后检查 Head、模型权重、词表、配置和相关代码身份。

## 三组对照和实际执行

1. `no_rag`：官方流式状态机，不注入热词。
2. `top5`：当前累计音频产生的 Operating Top5 候选，刷新当前步 prompt。
3. `top7`：同一份候选中的 Operating Top7，其他设置相同。

每2秒送入新音频；不足2秒的尾段也处理。CTC 在每个边界只读取截至该时刻的累计音频。
使用既有 Anchor64、score≥0.75、edit_ratio≤0.35、posterior≥0.5；不足5/7条不硬凑。
不把整段检索结果直接用于第一个时间点，也不把原文、正确目标提供给 Decoder。

前缀检索进程先运行并退出，再启动 Decoder，因此不会同时在同一GPU加载两套大模型。
Top5/Top7共享前缀结果；回放严格要求当前音频长度等于缓存边界，不能提前使用后续候选。
这复现了因果输入和动态 prompt 的语义，但不是并发线上服务或实时延迟测量。

官方 `qwen-asr==0.0.6` 管理分块、历史输出及回退窗口（2个chunk、5个token）。
每步通过官方 initializer 生成新的 prompt 元数据，仅刷新 `prompt_raw/context/force_language`，
保留音频和转写状态；最终生成请求的 prompt、token摘要、音频长度也会记录和核验。

复用原有 `.venv/qwen_asr_vllm023_cu129/bin/python` 中 native vLLM 0.23.0。
官方状态机留在主环境，vLLM位于独立子进程，避免旧 qwen-asr 注册覆盖新版原生实现。
不安装/升级任何包，不下载模型。缺环境时在全量GPU工作开始前报告明确路径。
Decoder 使用 r15、FP16、temperature=0、seed=20261123、max_new_tokens=512、
max_model_len=8192、eager、关闭prefix cache；模型、采样、Prompt及评分规则保持一致。
同一音频最多三个独立官方状态并行推进。请求队列最多等待20ms收集最多三个请求，
一次调用同一vLLM引擎的generate批量解码（max_num_seqs=3）；不加载三份模型。
每个状态拥有独立prompt、音频累计和历史输出，结果按请求顺序及prompt摘要/长度核验。
只有一组或两组未完成时也正常运行，无须等第三组，不会重跑已完成组合。

默认`--decoder-memory auto`：每次Decoder启动时读取逻辑cuda:0的实际空闲/总显存，
预算为 `free - max(2 GiB, 10% × free)`，向下换算为vLLM占整卡比例。
上次30.27 GiB空闲时约申请27.24 GiB；这不是运行中持续扩容，其他任务仍可能改变空闲量。
显存、有效比例、物理卡和并发数记录在`decoder_launches/*.json`，汇入最终报告。
`--decoder-memory fixed`可显式恢复配置中的15%分配；并批仍保留。

启动日志`Streaming resume: completed=...; pending=...`显示实际待处理组合；
`Native decoder batch=... requests=3 elapsed=...`证明请求已并批，并记录该批耗时。
各请求的`batch_id/batch_size/batch_seconds`保存在实际请求证据里，不能把三个重复批次耗时
相加当作墙钟耗时。加速幅度需要H200实测，不能宣称一定三倍。
并批可能产生浮点差异，不保证逐字节复现单序列结果；复用的旧结果与新增批次通过
`legacy_cells_without_launch_id`及每条`decoder_launch_id`区分，模型与评测口径不变。
遇到输出被token上限截断会停止，不把截断转写算作正常完成。

## 输出与指标

- `report.json`：全量三组和分来源指标。
- `smoke_report.json`：20条执行链路检查结果，不能当总体指标。
- `asr/{no_rag,top5,top7}/*.json`：逐音频结果。`payload` 中含原文、最终转写、
  每步 `timeline`（累计时刻、候选、prompt、部分转写）及实际 Decoder 请求。
- `prefixes/*.json`：因果音频前缀、检索结果和计算时间，不含特征张量。
- `contract.json`、`runtime_main.json`、`runtime_decoder.json`：数据/模型/代码/环境身份。
- `return_summary.zip`：小报告、身份及本次升级审阅记录；无需传音频、模型或全部时间线。

主结果是最终文本的热词召回率（正确目标词—音频对 / 580），不是候选集合召回率。
同时报告 corpus WER/CER、正确注入目标在最终文本中的出现比例、错误候选落入文本的比例、
负样本出现原文未记载词表词的比例，以及相对于无RAG的逐目标新增/丢失数量。
正确注入与最终出现的共现不能直接证明因果采纳；配对新增/丢失有助于判断实际增益。
原文未记载并不等价于人工听音确认未说，尤其阿文拼写存在差异，须保留这一口径。

WER/CER采用既有 Arabic normalization：NFC、大小写归一、去短元音/叠音符/延长符、
标点转空格；保留不同alef、ya、ta等字母区别。汇总编辑距离除以汇总参考长度，
不是逐句WER的平均。热词匹配与原整段评测保持同一口径：最长不重叠目标，重复提及只算一次。
1647条负样本也参与WER/CER与误报统计，不只评估431条正样本。

## 续跑与核验

中断后重复原命令；完整且身份匹配的逐音频前缀/逐模式ASR结果复用，未完成单元重做。
缓存有内容校验，不接受损坏或不同输入的旧结果。改变配置、词表、模型或相关代码时
指定新的 `--output-dir`；本次已审阅的资源/调度升级例外，可保持原目录续跑。
一个输出目录只能由一个任务占用；不删除已有成果。

```bash
jq '{status,samples,by_mode}' outputs/egy_ep29_streaming_e2e_mem15/report.json
```

应看到 `status=completed`、2078条，三组 `target_pairs` 均为580。
回传 `outputs/egy_ep29_streaming_e2e_mem15/return_summary.zip` 即可。
本地无H200和工区模型，只能验证执行逻辑；实际端到端指标由工区运行产生。
