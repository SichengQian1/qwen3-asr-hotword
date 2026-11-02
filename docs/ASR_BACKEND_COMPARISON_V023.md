# H200 小测试：Transformers 与原生 vLLM 0.23.0 的最终转写对照

2026-11-02。目标是直接回答“相同音频、Prompt、精度与生成设置下，两套推理路径的
转写是否一致”。不重训、不运行新CTC、不替换生产环境或历史outputs。**当前交付为
工区运行代码；本地只做了无模型测试/静态检查，没有安装CUDA包、加载Qwen或得到H200结果。**

## 一条命令做什么

在工区原来的Python3.12环境、仓库根目录执行：

```bash
python -B scripts/compare_asr_backends.py all --gpu 5
```

5是此前用户授权的物理卡号；需要换卡时填写另一张已分配的空闲卡。它会自动：

1. 检查Linux x86_64/Python3.12/H200、驱动及该卡占用。占用超过1GiB则停止，不结束
   其他任务。此阈值为本次独占小测试的工程设置，不表示低于此值就构成使用授权。
2. 在`.venv/qwen_asr_vllm023_cu129`创建独立环境并安装固定依赖，原Conda环境保留。
   首次安装需要网络下载依赖，建议预留25GiB磁盘；不下载Qwen权重。
3. 执行一个很小的真实CUDA/vLLM RMSNorm算子检查；它只筛除基础运行时问题，不证明
   所有注意力或JIT算子已可用。失败会留下小报告，不更改宿主驱动。
4. 从最新`outputs/mfa_v03_noise70_scratch_v1/retrieval/cells`读取已保存new70结果，
   英/西/葡×原始/10dB各取5条≤30秒的音频，共30条。按来源文件SHA顺序选取，
   不看转写成功与否；文件SHA和规范化波形SHA精确去重，不宣称说话人独立。
5. 每条做无热词、相同已保存Top7两种Prompt，两套后端各60次转写，共120次。
   两套模型分进程先后运行，不同时驻留；不会导入上一份Ascend两阶段样例。
6. 汇总完全一致、格式差异、词语差异、生成token差异、小样本参考WER与热词命中变化，
   自动生成`comparison.tsv`和`return_bundle.zip`。

## 固定版本与两个容易踩的点

| 组件 | 本次测试 |
|---|---|
| vLLM | 官方`0.23.0+cu129`发布wheel，URL及SHA固定在配置中 |
| Torch / torchaudio / torchvision | `2.11.0+cu129` / `2.11.0+cu129` / `0.26.0+cu129` |
| Transformers / qwen-asr | `4.57.6` / `0.0.6`，两套后端共用此独立环境 |
| 模型 | `/glusterfs_103/models/Qwen3-ASR-1.7B`原有完整快照，元数据SHA写入输出 |
| 精度及请求 | 两边FP16、greedy、batch1、eager、输出最多512 token、上下文4096 |

`+cu129`表示CUDA12.9构建，主版本仍为0.23.0。H200 CUDA后端和310P Ascend后端即使
主版本相同，也不是相同硬件/算子组合；这个测试不能保证310P结果逐字相同。

**不要使用`qwen-asr[vllm]`安装本次环境**：0.0.6的该extra固定vLLM0.14.0。本入口
安装的是不带extra的`qwen-asr==0.0.6`及明确的官方vLLM0.23.0 wheel。
[已发布包元数据](https://pypi.org/pypi/qwen-asr/0.0.6/json)、
[vLLM0.23.0依赖](https://pypi.org/pypi/vllm/0.23.0/json)。

旧`qwen_asr`模块导入时会尝试注册其携带的旧vLLM实现。本次准备/Transformers进程仅
在导入期间阻止这个可选旧vLLM子模块（由上游try/except处理），避免注册表交叉污染；
不修改安装源码。**原生vLLM进程不导入qwen_asr**，并核对实际模型模块必须是
`vllm.model_executor.models.qwen3_asr`。没有用旧包装器冒充新版本。

CUDA12.9 GA对应驱动575.51.03；CUDA12.x还存在最低525.60.13起的次版本兼容路径，
不能把“驱动低于575”一概当作不能运行。入口按次版本兼容下限初筛并记录模式，
实际是否满足所需算子以GPU检查和后续真实推理为准；旧驱动可能因PTX/JIT等限制失败。
这时回传报错，不自动安装驱动或换成不匹配的Torch。
[NVIDIA CUDA12.9说明](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)、
[次版本兼容边界](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 如何保证在比同一件事

- 不复用过去BF16的ASR文本当本次FP16结果，两边都新跑。
- 音频只统一读取/转单声道/必要时重采样一次，保存float32小波形；两边都使用同一
  波形。共同预先补零到160采样点整数倍（最多约9.94ms），避免两套Processor的
  边界补齐政策造成额外帧差异；这是本次共同测试输入，不声称完全复现历史输入。
- Prompt由原处理器一次序列化，保留语言后缀，注入相同Top7原始词面。参考转写只
  用于事后评分，不进入模型。无候选时保留空提示，记录数量，不人为挑成功样本。
- 两边核对实际展开后的完整输入token IDs。不同则停止，写`input_mismatch.json`，
  不把输入不一致统计成后端转写差异。对齐EOS和greedy约束；遇到尚未映射的额外
  生成限制或输出截断也停止，不隐藏问题。
- 记录包版本、模型元数据、关键安装源码SHA、每条波形与来源文件SHA。没有重算
  所有大型模型shard的SHA；同一只读模型目录是本轮假设。

一致性分两层：原文完全一致；忽略NFC等价、大小写、标点与空白后词语一致。
**不会删除西葡重音、鼻化拼写、ñ或数字来制造一致。**即使是greedy，也不预设两套
浮点/算子实现必须逐字相同。词语不同时结合参考及专名命中判断，不自动认定vLLM错。

## 输出、恢复和查看

默认配置：`configs/asr_backend_comparison_v023.workzone.json`。
默认输出：`outputs/asr_backend_comparison_v023_v1`。

| 文件 | 含义 |
|---|---|
| `report.json` / `comparison.md` | 总体及各语言/条件/Prompt的对照汇总 |
| `comparison.tsv` | 每条参考、两边转写、词面、音频路径及差异类别 |
| `details.json` | 逐条词错误数及注入词的命中差异 |
| `requests.json` | 固定30音频/60请求、候选及共同Prompt；不是训练标签 |
| `environment.json` / `device.json` / `gpu_probe.json` | 环境、驱动模式与真实基础算子检查 |
| `transformers/runtime.json` / `vllm/runtime.json` | 本次推理运行时及关键安装源码身份 |
| `failure_*.json`、`logs/` | 某个阶段失败原因与完整日志 |
| `return_bundle.zip` | 小结果包：JSON/TSV/MD和日志尾部，无权重/大特征/波形 |

重复**同一条all命令**，会校验同一实验身份、复用已安装的固定环境和成功的逐条结果；
不会重选样本或重跑成功项，无需`--resume`。完全完成后再次运行只核验并重打包，不
加载Qwen。换源码、参数或目标GPU需新输出目录；如确需更换依赖，另设新的环境路径，
不覆盖已完成结果。一个环境只用于本次串行测试，不要并行启动多份安装。

只查看进度，不安装、不加载模型：

```bash
python -B scripts/compare_asr_backends.py status
```

运行后只需回传：

```text
outputs/asr_backend_comparison_v023_v1/return_bundle.zip
```

这是小规模完整ASR后端对照，不是CTC PER、总体WER、310P适配、两阶段Encoder缓存
或显存/时延基准测试。若有词语差异，再针对这些条目复测/定位，无需先扩大量。
