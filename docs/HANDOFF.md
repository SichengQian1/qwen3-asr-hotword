# 工作交接记录

## 1. 当前目标

在 Qwen3-ASR-1.7B Audio Encoder 保持冻结、复用既有 `ln_post` BF16 特征缓存的
前提下，将正式分片 CTC 训练链路从仅支持线性 Head 扩展为可选的时间上采样
Head，以验证更高时间分辨率和局部上下文能否降低验证集 PER，尤其是删除错误。

目标结构已实现：

```text
thinker.audio_tower.ln_post
-> LayerNorm
-> 确定性时间上采样（默认 2 倍 repeat_interleave）
-> 1x1 投影 + 轻量 depthwise 时序卷积 + 1x1 上下文投影
-> GELU + Dropout
-> Linear(hidden_dim, 90)
-> CTC Loss
```

## 2. 实际工作树与交接冲突

2026-07-21 恢复任务时的实际状态：

```text
branch: main
HEAD:   56f2c37 Add unfrozen encoder CTC trainer
remote: main...origin/main [ahead 28, behind 5]
```

上一版 HANDOFF 记录的 `unfrozen-encoder-ctc@d36ba23` 与实际工作树不一致。
`d36ba23` 上的以下诊断文件在当时的 `main` 中不存在：

- `scripts/diagnose_frozen_ctc.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `tests/test_ctc_diagnostics.py`

本轮没有在带未提交修改的情况下切换分支，而是以 `main@56f2c37` 为真实基线，
将必要诊断能力移植到当前工作树，并更新为支持新 Head。未切换、回退、删除
或覆盖其他任务的未提交修改。

## 3. 本轮已完成

- 完成 `TemporalUpsampleCtcHead` 的代码审查与收口：
  - 恢复 `LinearCtcHead` 的构造和输入维度校验。
  - 校验 Head 维度、奇数卷积核、dropout 和上采样倍率。
  - 在卷积前后按有效输出长度遮蔽 padding，避免卷积 bias 污染有效边界。
  - `compute_ctc` 恢复旧线性路径校验，并返回 Head 变换后的有效输入长度。
- 将新 Head 接入 `training/sharded_ctc.py`：
  - 通过 `build_ctc_head` 工厂构建线性或上采样 Head。
  - 训练、验证、贪心解码和 PER 统计统一使用 `CtcComputation.input_lengths`。
  - 训练状态和报告写入完整 `head_config`。
  - 新 Head 配置纳入 resume fingerprint；旧线性训练的 fingerprint 保持不变，
    可继续恢复。
- 更新 checkpoint 兼容性：
  - best/latest checkpoint 保留旧顶层字段，同时新增 `head_config`。
  - 可从新 checkpoint 重建时间上采样 Head。
  - 旧 `LinearCtcHead` checkpoint 仍可重建和加载。
  - Experiment A 的初始 Head 加载会显式拒绝结构不匹配的 checkpoint。
- 更新正式训练 CLI：
  - `--head-type {linear,temporal_upsample}`
  - `--head-hidden-dimension`
  - `--head-kernel-size`
  - `--head-dropout`
  - `--head-time-upsampling-factor`
  - 旧命令为保持向后兼容仍默认 `linear`；新实验必须显式传
    `--head-type temporal_upsample`。
- 恢复并升级验证诊断链路：
  - 按 checkpoint 元数据重建 Head。
  - loss、解码、blank ratio、预测/参考长度比和 CTC 压力分桶均使用上采样后
    长度。
  - 保留删除、插入、替换及高频 token 统计。
- 确认特征缓存无需重建：现有校验要求目标在原始 Encoder 帧长下也可行，
  对上采样 Head 是更严格但安全的超集。

## 4. 本轮修改文件

本阶段代码与测试：

- `src/qwen_hotword/modeling/ctc_head.py`
- `src/qwen_hotword/training/sharded_ctc.py`
- `src/qwen_hotword/training/ctc_overfit.py`
- `src/qwen_hotword/training/ctc_diagnostics.py`
- `src/qwen_hotword/training/edit_distance.py`
- `scripts/train_full_ctc.py`
- `scripts/diagnose_frozen_ctc.py`
- `tests/test_ctc_head.py`
- `tests/test_ctc_diagnostics.py`
- `tests/test_sharded_ctc.py`
- `docs/HANDOFF.md`

工作树中仍存在以下其他任务的未提交修改，本轮未改动、未删除：

- `docs/PHONEME_VOCAB.md`
- `docs/WORKZONE_RUNBOOK.md`
- `scripts/scan_g2p_coverage.py`
- `tests/test_g2p_coverage.py`
- `tests/test_phoneme_vocab.py`
- `configs/phonemes/en_es_ptbr_fr_id_precision_ipa_vocab.v0.3.json`
- `work/`

## 5. 测试结果

2026-07-21 本地实际结果：

```text
Ruff（本轮相关文件）: pass
Mypy src/qwen_hotword:          pass, 33 source files
Pytest (.conda + torch 2.10):   pass, 86 tests
git diff --check:               pass
CLI --help smoke:               pass
```

新增/更新测试覆盖：

- 线性 Head 形状、长度和旧校验回归。
- 上采样 Head 输出 shape 与精确 2 倍有效长度。
- 改变 padding 区域数值不影响有效卷积输出。
- 上采样 Head 的实际 CPU CTC loss。
- 新 Head checkpoint round-trip 和旧线性 checkpoint 兼容。
- 线性与上采样两种 Head 的分片训练、checkpoint、optimizer/scheduler 状态
  和 epoch resume CPU smoke test。
- 诊断模块从两种 checkpoint 重建 Head，且输入帧统计使用有效输出长度。
- 全仓库回归测试。

基础 Python 环境没有 torch，其张量测试会 skip；最终验证使用仓库 `.conda`
环境的 torch 2.10.0，所有测试均真实执行。

## 6. 仍待完成的工作区实验

代码接入和本地完成标准已达到。下一个最高优先级是在 H200 工作区对真实特征
缓存做受控实验；本机没有该 30.9 GB 缓存，因此本轮未伪造“真实缓存已跑”
的结果。

建议步骤：

1. 在新输出目录中，用少量真实 train/validation shard 跑 1–2 epoch smoke test。
2. 确认 report 与 checkpoint 的 `head_config.head_type` 为 `temporal_upsample`，且 resume
   能从下一 epoch 继续。
3. 使用与线性基线相同的 train/validation 特征缓存完整训练，不读取封存
   test 集。
4. 对线性 best checkpoint 和上采样 best checkpoint 运行
   `scripts/diagnose_frozen_ctc.py`。
5. 比较验证 PER、删除/插入/替换、预测/参考长度比、blank frame ratio 和各 CTC
   压力分桶，再决定是否进入 Encoder adapter/LoRA。

正式命令需显式加入：

```bash
PYTHONPATH=src python scripts/train_full_ctc.py \
  ...现有 train/validation cache 与 manifest 参数... \
  --output-dir outputs/full_ctc_temporal_upsample_v1 \
  --head-type temporal_upsample \
  --head-hidden-dimension 512 \
  --head-kernel-size 5 \
  --head-dropout 0.1 \
  --head-time-upsampling-factor 2
```

不得复用旧线性训练输出目录，也不得用线性 checkpoint 作为新 Head 初始权重。

## 7. 已知风险与已确认决策

- 2026-07-21 用户最终确认继续以原工区流程发布 CTC 训练更新：代码接到
  `codex/g2p-coverage-scan`，工区使用 `git pull origin codex/g2p-coverage-scan`。
- GitHub `origin/main` 与 CTC 训练历史从 `d7907d3` 后分叉；本轮不合并或强制
  覆盖远端 `main`。
- 开发工作树仍可包含其他任务修改，但每次发布必须显式限制提交范围，
  不应整体提交、rebase 或整理无关未提交修改。
- 上采样会提高 Head 中间 activation 和训练时间；需用 H200 smoke test 确认
  `train_batch_size=256` 是否仍合适，必要时只调低 Head 训练 batch size。
- 上采样能增加对齐路径，但不能自动解决 G2P/标签噪声或声学表征不足；必须以
  validation PER 和错误分解为准。
- 当前时间上采样 Head 参数量约取决于 hidden dimension，不再是 92,250 参数的线性
  Head；报告会记录实际可训参数。

## 8. 下一任务需读取的最小文件

1. `AGENTS.md`
2. `docs/HANDOFF.md`
3. `src/qwen_hotword/modeling/ctc_head.py`
4. `src/qwen_hotword/training/sharded_ctc.py`
5. `scripts/train_full_ctc.py`
6. `src/qwen_hotword/training/ctc_diagnostics.py`
7. `scripts/diagnose_frozen_ctc.py`
8. `tests/test_ctc_head.py`
9. `tests/test_sharded_ctc.py`
10. `tests/test_ctc_diagnostics.py`

开始下一步前应再执行 `git status --short --branch`。工区训练以
`origin/codex/g2p-coverage-scan` 为发布基线，不从 `origin/main` 拉取本阶段 CTC 更新。
