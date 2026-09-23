# Python模块开发与开发者测试

| 项目 | 内容 |
|---|---|
| 对应能力项 | 编程语言应用；开发者测试；调试与定位 |
| 重点模块 | 文本与音素处理、分片缓存及训练控制、多语热词检索与导出 |
| 记录日期 | 2026-09-24 |
| 证据范围 | 可定位源码、实际测试用例、静态检查结果和问题回归记录 |

## 1. 开发范围与职责划分

项目使用Python实现多语音素热词数据链路和CTC模型分支，采用“命令行入口、业务模块、配置、测试”分离的组织方式。

| 层次 | 职责 | 代表文件 |
|---|---|---|
| 命令行入口 | 参数解析、配置读取、调用模块、打印结果 | [run_multilingual_480h.py](../scripts/run_multilingual_480h.py)、[run_wave_keyword_retrieval.py](../scripts/run_wave_keyword_retrieval.py) |
| 数据处理 | 规范化、词表、标签、音频与文本元数据处理 | [g2p_prep.py](../src/qwen_hotword/training/g2p_prep.py)、[full_manifest.py](../src/qwen_hotword/training/full_manifest.py) |
| 模型与训练 | Head定义、缓存、CTC训练及恢复 | [ctc_head.py](../src/qwen_hotword/modeling/ctc_head.py)、[feature_cache.py](../src/qwen_hotword/training/feature_cache.py)、[sharded_ctc.py](../src/qwen_hotword/training/sharded_ctc.py) |
| 热词功能 | 词表构建、索引、检索与接口导出 | [wave_keywords.py](../src/qwen_hotword/evaluation/wave_keywords.py)、[wave_retrieval.py](../src/qwen_hotword/inference/wave_retrieval.py) |
| 配置与测试 | 参数集中定义、正常及异常行为验证 | [训练配置](../configs/480h_training.workzone.json)、[tests目录](../tests/) |

报告以三个代表性模块说明实现质量，不以代码数量、运行次数或测试数量代替业务正确性。个人职责归属应结合实际分工确认；本文不作超出技术记录的独立贡献或正式鉴定声明。

## 2. 实现案例一：文本规范化与音素标签装配

### 2.1 接口和数据结构

`normalize_training_text(text: str) -> str`负责Unicode、大小写、空白和连接符规范化；`extract_word_tokens(text: str) -> list[str]`提取词；`digit_fragments(text: str) -> list[str]`单独记录数字片段。职责分开，避免在一个函数里同时清洗、猜测读音和修改训练样本。

以下为实际实现节选：

```python
def normalize_training_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    for source, target in CONNECTOR_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()
```

词频采用`Counter`累计，路径使用`Path`，结果采用dataclass和结构化字典。标签装配记录每个词的发音、token ID、OOV单元和处理状态；出现缺词或歧义时保留原因，而不是用空标签冒充成功。

### 2.2 规范与异常处理

1. 入口检查文件、列名和参数范围，异常明确包含问题对象。
2. Unicode重音字母按语言数据保留，不使用统一ASCII清洗破坏发音。
3. 数字和连接符的处理有显式策略，不依赖隐式猜测。
4. 合法记录和待复核记录具有明确去向，支持统计源记录是否完整保留。
5. 元数据TSV采用其已确认格式的专用解析策略，遇到列数错误停止，避免静默错位。

### 2.3 对应开发者测试

实际断言示例，来自[文本测试](../tests/test_g2p_prep.py)：

```python
assert normalize_training_text("  VOCÊ  d’água — NÃO  ") == "você d'água - não"
assert extract_word_tokens("Você d'água, não? bem-vindo!") == [
    "você", "d'água", "não", "bem-vindo",
]
assert digit_fragments("Temos 4 modelos e vídeo em 8K.") == ["4", "8k."]
```

这些断言检查外部行为：既验证应变换的字符，也验证不应丢失的发音信息。更多用例见[清单构建测试](../tests/test_full_manifest.py)和[元数据测试](../tests/test_pt_validation_metadata.py)，包括缺词、数字、超短音频、超长字段及错误列数。

## 3. 实现案例二：分片缓存与训练状态管理

### 3.1 模块接口

| 接口 | 职责 |
|---|---|
| `cache_feature_split(...)` | 按split和分片参数提取特征、记录配置与完成度 |
| `validate_feature_shard(...)` | 校验分片内容和元数据约束 |
| `load_disk_feature_cache(...)` | 加载完整缓存描述，检查split、来源、词表及分片一致性 |
| `train_sharded_ctc_head(...)` | 按片训练和评估，维护Head、优化器、调度器及指标 |
| `run_training_stage(...)` | 检查阶段前置条件，显式启动pilot或后续续训 |

通过分片读取控制内存占用，避免将约140GB缓存一次性装入内存。`CtcComputation`等结构化结果显式返回logits、loss及有效长度，减少调用方猜测张量含义。

### 3.2 变长序列正确性

Head在时间扩展和上下文计算中使用真实长度屏蔽padding。其核心掩码逻辑为：

```python
steps = torch.arange(values.shape[1], device=values.device)
mask = steps.unsqueeze(0) < lengths.to(values.device).unsqueeze(1)
return values * mask.unsqueeze(-1).to(values.dtype)
```

对应测试不是只检查张量形状，而是把同一有效序列的padding分别设为0和1000，断言有效位置输出一致。这能发现“形状正确但padding污染卷积上下文”的问题。见[CTC Head测试](../tests/test_ctc_head.py)。

### 3.3 恢复与产物保护

恢复前检查缓存、词表、训练参数和Head结构是否匹配。状态包含已完成epoch、优化器、调度器、历史指标和最佳模型信息；恢复从最近完整epoch继续。已存在的正式目录不被首次命令覆盖，缺少状态文件时不将恢复请求降级成随机重训。

阶段入口要求先完成缓存，再运行完整数据5轮训练；正式续训要求已完成该前置阶段。新正式Head和少量链路检查Head分离，防止误用初始化。

测试通过子进程替身检查实际命令参数、GPU选择、前置条件和恢复开关；真实训练质量另以模型运行结果验证。测试代码：[缓存](../tests/test_feature_cache.py)、[分片训练](../tests/test_sharded_ctc.py)、[阶段控制](../tests/test_multilingual_480_run.py)。

## 4. 实现案例三：多语词表与下游接口

### 4.1 兼容性和输入边界

多语接口允许已支持的语言名称和代码别名，拒绝未知或不匹配语言。对词面、音素和token ID分别检查，避免“语言名称正确”掩盖内容不兼容。

词表构建区分必留目标和可选补词：必留项有问题时阻断；可选项发音冲突时隔离并记录，合法词不足时失败。不同词面的相同音素序列可以保留，不在接口层擅自改变任务定义。

### 4.2 可测试性

检索执行器支持注入detector factory和retrieve function。测试夹具能构造明确排名、空候选和边界数据，不需要完整模型即可检查编排和输出。

下游JSON示意如下，内容仅用于说明接口：

```json
{
  "sample_001": [{"word": "示例词", "phoneme": "示例音素"}],
  "sample_002": []
}
```

不同wave可以具有相同stem，但各组文件分开输出；单文件内ID不得丢失。Top5和Top7分别导出，字段严格为`word`和`phoneme`。

### 4.3 测试设计

| 场景 | 断言重点 |
|---|---|
| 8组输入 | 导出16份文件，各组ID完整 |
| 空召回 | 保留ID及空列表 |
| 候选排名 | Top5结果可复现，Top7使用充分保存的排序重放 |
| 指标分母 | 原始目标与填充词指标分开，正确普通补词不被自动记成错误 |
| 输入变化 | 恢复拒绝与原输入不一致的任务 |
| 中断恢复 | 仅计算缺失样本，完成项不重复执行 |
| 最后一组异常 | 全组预检即阻断，不先加载模型再发现结构错误 |

对应测试：[词表测试](../tests/test_wave_keywords.py)、[检索与导出测试](../tests/test_wave_retrieval.py)、[TopK重放测试](../tests/test_external_keyword_topk_replay.py)。

## 5. 静态检查与开发者测试结果

2026-09-24复核以下命令对应的结果：

```bash
python -m pytest -p no:cacheprovider -rs
ruff check --no-cache src scripts tests
mypy src
```

| 检查项 | 结果 | 解释 |
|---|---|---|
| pytest全量回归 | 434 passed，23 skipped，0 failed | 23项因缺少Torch跳过，不能算数值回归通过 |
| Ruff，范围为src/scripts/tests | 3处E501 | 均位于`scan_g2p_coverage.py`的长行，未通过全范围静态检查 |
| Mypy，范围为src | 3处unused-ignore | 位于`ctc_overfit.py`、`sharded_ctc.py`、`unfrozen_encoder_ctc.py`，需要清理冗余类型忽略 |
| 代码覆盖率 | 未提供数值报告 | 不用通过数量推导行覆盖率或分支覆盖率 |

项目在[pyproject.toml](../pyproject.toml)中启用Ruff的E/F/I/UP/B/SIM规则和Mypy strict模式。已有脚本入口E402例外及可选第三方依赖类型设置需要按具体理由维护；本次文档整理未通过增加忽略规则来隐藏告警。

这些证据能够说明已具备编码约束和回归机制，但不足以宣称已获得正式Clean Code鉴定、满足未定义缺陷率门槛，或不存在任何发布质量事件。

## 6. 持续测试配置与覆盖风险

现有[持续测试工作流](../.github/workflows/ci.yml)定义Python 3.12任务，安装开发与CI依赖，执行pytest、Ruff和Mypy。由此可以确认仓库具有持续测试配置；本报告未附本次流水线运行记录，不能声明所有流水线均成功。

依赖配置包含pytest-cov，但工作流未设置`--cov`、分支覆盖采集或覆盖率门槛。因此“存在测试”和“达到覆盖率要求”是两个不同结论。

| 风险 | 当前证据 | 建议补充 |
|---|---|---|
| 缺少Torch导致数值用例跳过 | 23项跳过记录 | 在有Torch的任务中复跑，保存成功与跳过明细 |
| 行/分支覆盖率未知 | 无覆盖率报告 | 使用`pytest --cov=qwen_hotword --cov-branch --cov-report=term-missing`采集后分析 |
| 持续测试是否实际执行成功未知 | 有工作流配置，无本次执行记录 | 保留每次任务结果和失败定位信息 |
| 未测试的存储故障或并发条件 | 已有任务锁及部分损坏/恢复测试 | 增加写入中断、磁盘不足和并发争用用例 |
| 新语料格式与语言别名变化 | 已有典型兼容回归 | 引入脱敏真实结构夹具，覆盖新增生产端格式 |

上述建议不作为本次已经新增或完成的测试声明。本报告只汇总现有代码与验证，不修改主训练逻辑或测试策略。

## 7. 调试与定位案例

### 7.1 超长文本与引号解析

通过构造不闭合引号和14万字符字段，将大文件中难以直接浏览的问题缩小为可复现夹具。检查文件格式约定后，采用专用逐行解析及列数约束，并用同名不同路径场景排除错误关联。该方法把“某批语料偶发失败”转为确定性的输入契约问题。

证据：[元数据实现](../src/qwen_hotword/training/pt_validation_metadata.py)及`test_literal_quotes_long_fields_and_conservative_join`等回归用例。

### 7.2 语言别名和旧词表兼容

先比较主词表、近音词表和历史补充词表的实际字段，而不是依次放宽所有校验。修复合法别名和规范化表达，同时补充错误语言、空语言、混合标签和不足词数的失败断言，避免“修复兼容性”变成“接受错误数据”。

证据：[词表实现](../src/qwen_hotword/evaluation/wave_keywords.py)及`test_mixed_old_and_neighbor_language_aliases`、`test_old_languages_all_audited_without_accepting_foreign_or_unknown_tags`等用例。

### 7.3 训练恢复的配置漂移

使用模拟子进程构造缓存数量错误、配置变化和运行期间输入变化，验证程序在不满足条件时不会开始或错误宣告完成。恢复要求状态文件存在，正式阶段要求完成5轮前置训练，避免错误续接或误初始化。

证据：[阶段控制实现](../src/qwen_hotword/training/multilingual_480_run.py)及`test_training_preflight_blocks_invalid_cache_or_plan`、`test_training_rechecks_inputs_after_execution`等用例。

上述案例属于项目开发与集成调试记录，不冒充生产网络事故处理经历。

## 8. 代码材料范围与结论

本报告引用的核心材料位于`src/`、`scripts/`、`tests/`、`configs/`，依赖与检查规则位于`pyproject.toml`，持续测试配置位于工作流文件。模型权重、音频、特征缓存和生成数据不属于源码示例；本文关键结论和测试结果已直接列出，无需读取大型产物即可理解。

实现体现了接口分层、语言数据处理、显式错误策略、分片资源管理、可注入测试依赖和针对性回归。现有测试支持已覆盖场景的正确性；尚待补齐数值回归、覆盖率量化及静态告警清理，作为进一步质量评审依据。
