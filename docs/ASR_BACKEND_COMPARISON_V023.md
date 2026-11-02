# H200 小测试：Transformers 与原生 vLLM 0.23.0 的最终转写对照

2026-11-02。目标是直接回答“相同音频、Prompt、精度与生成设置下，两套推理路径的
转写是否一致”。不重训、不运行新CTC、不替换生产环境或历史outputs。**当前交付为
工区运行代码；本地只做了无模型测试/静态检查，没有安装CUDA包、加载Qwen或得到H200结果。**

## 一条命令做什么

在工区原来的Python3.12环境、仓库根目录执行：

```bash
python -B scripts/compare_asr_backends.py all --gpu 4
```

4是用户本次明确指定的物理卡号，允许与该卡现有任务共享。它会自动：

1. 检查Linux x86_64/Python3.12/H200、驱动及实际剩余显存。已有占用不再自动拒绝；
   按下述预算检查可用空间，不结束其他任务。各GPU阶段开始前重新检查一次。
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

## 共享4卡的显存预算

旧版把`memory.used > 1024 MiB`作为拒绝条件，用户在GPU4已占用98055 MiB时被拦截。
该次停在首次预检查，尚未安装环境或执行任何转写；它不是模型显存不足的实测证据。
**本次改为检查`nvidia-smi`实际报告的`memory.free`，并设置以下工程预算：**

| 项目 | 当前设置 | 含义 |
|---|---|---|
| vLLM执行器预算 | 总显存×0.15，约21GiB/140GiB卡 | 包含模型与可用KV空间的预算，按总量而非剩余量计算 |
| Transformers缓存分配器上限 | 24GiB | 在加载模型前设置PyTorch分配器上限，不限制其他进程 |
| 空闲余量 | 8GiB | 在较大的单后端预算之外保留，不主动占用 |
| 启动要求 | `max(总显存×0.15, 24GiB)+8GiB` | 约32GiB实际剩余空间；两后端串行，不把两者预算相加 |

如果该H200仍为约140GiB，总占用98055MiB对应约44GiB剩余，预计能通过；用户日志
没提供总量/实际free，不能据此断言已通过。入口会打印实际free、各预算和检查结果。
这些是本轮小测试的资源设置，**不是已测峰值**，也不能保证运行中其他任务扩容后仍
有足够空间。PyTorch上限仅约束其缓存分配器；CUDA上下文/库的直接分配等不都在其
范围内，vLLM预算也不能视为所有启动瞬态的硬隔离。显存余量用于降低冲突风险，
不保证共享计算没有性能影响。本次只比较转写，不把共享卡耗时作为后端性能结论。
[vLLM0.23预算说明](https://docs.vllm.ai/en/v0.23.0/configuration/engine_args/#--gpu-memory-utilization)、
[PyTorch2.11分配器上限](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.set_per_process_memory_fraction.html)。

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

## 安装时提示自签名证书链怎么办

2026-11-02用户在GPU4得到142875MiB free/143771MiB total，显存检查通过；随后创建
venv成功，pip下载GitHub vLLM wheel时提示`self-signed certificate in certificate chain`。
此时尚未加载模型或开始转写。它是HTTPS信任问题，不能当作vLLM依赖/模型不兼容；
是否由企业代理CA引起，未取得工区证书链前仍是推测。之前MFA下载也有相似记录。

新增独立小入口，不改已绑定实验身份的四个原运行源码/配置文件，继续同一v2目录：

```bash
python -B scripts/check_asr_install_tls.py --run --gpu 4
```

先检查环境变量指定的CA、Linux系统bundle、当前Conda/Python bundle（仅已有文件）。
逐个使用严格HTTPS HEAD检查官方GitHub固定wheel及其重定向、PyPI索引/实际wheel
地址、PyTorch CUDA索引，打印每项结果，不下载大型wheel或占用GPU。某一bundle全部
通过后，只在测试子进程内设置`PIP_CERT`等CA变量，自动接着运行原`all --gpu 4`。
不改系统证书、pip配置、原Conda依赖或已有contract；失败日志保留，不新建实验目录。
依赖版本不变，原venv继续使用，安装成功的固定包可由pip复用；后续仍要以实际pip
安装/包检查为准，HEAD检查不能证明所有传递依赖的下载路径可访问。

若全都`certificate_verification_failed`，需要工区管理员提供可信CA的**PEM bundle**
（包括需要的公司根证书/公共根），将实际路径传入同一入口；代码无法凭空补出该证书：

```text
python -B scripts/check_asr_install_tls.py --ca-bundle /实际路径/company-ca-bundle.pem --run --gpu 4
```

`http_error_403`、超时或HEAD不受支持等是其他可访问性问题，要按输出区分，不能都
继续称作证书错误。可省略`--run --gpu 4`只做CPU/网络检查。此入口不关闭验证，也不
自动信任从报错连接拿到的证书。[pip官方CA配置说明](https://pip.pypa.io/en/stable/topics/https-certificates/)
明确支持`PIP_CERT`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`；PyPI wheel检查地址来自
[qwen-asr0.0.6官方元数据](https://pypi.org/pypi/qwen-asr/0.0.6/json)。

## 输出、恢复和查看

默认配置：`configs/asr_backend_comparison_v023.workzone.json`。
默认输出：`outputs/asr_backend_comparison_v023_shared_v2`。
旧`outputs/asr_backend_comparison_v023_v1`失败记录保留。由于源码与预算改变，使用新
目录绑定新实验身份，不绕过旧contract校验；旧次尚未开始推理，没有转写需要重跑。
依赖版本和独立环境路径不变，若固定环境已成功安装且身份一致可复用，不强制重装。

| 文件 | 含义 |
|---|---|
| `report.json` / `comparison.md` | 总体及各语言/条件/Prompt的对照汇总 |
| `comparison.tsv` | 每条参考、两边转写、词面、音频路径及差异类别 |
| `details.json` | 逐条词错误数及注入词的命中差异 |
| `requests.json` | 固定30音频/60请求、候选及共同Prompt；不是训练标签 |
| `environment.json` / `device.json` / `gpu_probe.json` | 环境、驱动模式与真实基础算子检查 |
| `memory_checks/*.json` | 各GPU阶段开始前的剩余显存和预算快照；变化不改变设备身份 |
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
outputs/asr_backend_comparison_v023_shared_v2/return_bundle.zip
```

这是小规模完整ASR后端对照，不是CTC PER、总体WER、310P适配、两阶段Encoder缓存
或显存/时延基准测试。若有词语差异，再针对这些条目复测/定位，无需先扩大量。

## 2026-11-02：用户指定扩大到完整6800条回归集

小样本已在H200完成：60对中57对原文相同，1对仅格式不同，2对有词语不同；30对带
热词结果的规范化词序列均相同。它不能证明全量行为，因此按用户新要求扩大范围。
本节是新一轮交付，**全量实测尚未执行**。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
VLLM_ALLOW_INSECURE_SERIALIZATION=1 python -B scripts/compare_asr_backends.py all --full-eval --gpu 4
```

- 英语6000条、西语400条、葡语400条是**基础音频条数**；热词库仍为上轮共同4k词面
  对应的new70实际Top7。每条基础音频包含original/noise10db两种条件，共13600份音频；
  每份分别无热词/相同Top7，两套后端各27200次，共54400次。无候选时保留空提示，
  不因无热词、错词或转写结果而筛选。
- 新配置`configs/asr_backend_comparison_v023_full.workzone.json`，新输出
  `outputs/asr_backend_comparison_v023_full6800_v1`，不覆盖小样本、训练或旧ASR输出。
  原小样本依然可用`status`只读查看；本次涉及运行源码变更，不能用旧目录再次启动all
  绕过旧contract。新全量任务从自己的目录续跑。
- **只复用已安装成功的独立venv**`.venv/qwen_asr_vllm023_cu129`，要求已有环境身份和
  ready记录，缺失即停止；不会调用pip安装或下载模型。仍核验精确版本/安装源码。
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`延续已成功小测试的本地可信`apply_model`
  函数RPC设置，只作用于该命令及子进程，不是SSL设置，不更改全局环境。
- 使用相同Qwen3-ASR-1.7B快照、FP16、greedy、batch1/TP1、种子、EOS、512输出token、
  4096上下文和显存预算。Transformers与vLLM串行运行；不切换注意力实现，不把本轮
  偷换成新数值设置。`enforce_eager=True`不代表vLLM使用Transformers eager attention。
- 复用最新100类Head/new70索引对应的已保存检索结果，不重跑Encoder+CTC检索，不重训、
  不重做MFA映射。两边完整ASR都新跑；历史BF16转写不作为本轮FP16结果。
- CPU先审计全部new70记录，核对Head/vocab/index SHA、六组计数和原始/带噪sample身份
  及参考文本配对，随后逐音频验证源SHA、规范化/共同补零、保存共同波形与两种Prompt。
  **不沿用小样本的30秒筛选，不去掉重复内容记录**；按身份完整保留并报告重复数，
  不把重复记录当作新增独立证据，也不宣称speaker-disjoint。
- CPU准备按音频保存带自校验的独立小cell，可断点复用；提前核对Processor展开后的
  输入长度。上下文超限、无效音频、错配、后端输入token不同或输出截断时明确停止，
  不悄悄删样本/截句凑齐数量。之后可根据错误条目决定是否另开扩大长度预算的版本，
  不能把失败后的不完整集合当完整评测。
- 推理每100条进度、静默阶段每10秒活动提示。同一命令中断后重跑，无需`--resume`，
  校验来源/源码/参数/包身份后只继续未完成项。已完成全量任务重复命令仅验证/打包。
  CPU准备和推理均无需一次把所有音频/声学特征装进内存；共同float32波形占磁盘约
  每小时0.23GB，13600份若平均10秒约8.7GB（估算，实际见preparation_summary）。

只读进度：

```bash
python -B scripts/compare_asr_backends.py status --full-eval
```

`full_preparation_audio_conditions`最终应为13600，`saved_calls`两边分别27200。
`report.json`有总体、逐语言、逐Prompt和语言×条件×Prompt的完全一致/格式/词语差异、
参考WER与注入词命中差异；`target_hits_differ`仅涉及参考支持的注入候选，不是完整
业务热词召回率或全体词都正确的证明。本集是已查看过的回归集，不是新增盲测。

回传新输出目录的`return_bundle.zip`即可。全量模式自动限制为**小于100000字节**，
包含汇总、身份、日志尾部和最多24条分层轮取的差异预览；超限时缩短预览/省略日志。
完整`comparison.tsv`、`details.json`、`requests.json`及逐条cell留在工区，未打进小包；
预览不是所有差异。没有模型权重、完整波形或大型特征。需要具体条目时再按request_id
定位，不能依据预览比例估计差异率。

本地63项CPU检查通过，含模拟Processor的准备/续跑、长音频与重复保留、全量身份拒收、
已就绪环境约束及100KB打包；静态检查不代表已在H200执行这轮54400次推理。此轮也
不能验证310P、两阶段Encoder缓存、CTC输出一致性或后端显存/性能收益。
