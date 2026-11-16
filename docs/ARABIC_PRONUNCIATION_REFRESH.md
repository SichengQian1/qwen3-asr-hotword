# 阿语发音修正、重新划分与英阿训练

2026-11-16；分支 `codex/g2p-coverage-scan`。目标是降低真实的阿语音素识别错误，
同时检查英语是否退化。本文交付的是可执行流程；新的工区标签、训练 PER 和母语者审核尚未返回。

## 六步及当前边界

1. **固定参照与音素表**：继续使用 r15 checkpoint-5100 merged Encoder、167 类
   `configs/phonemes/en_ar_native_ipa_vocab.v1.1.json`。ID、英语前缀、历史标签和输出不改。
   音素表 SHA256 为 `b337255bf239fdec3b4bdb0770bf39941fee74246336774c7ece8741d4534bf2`。
2. **修正发音前端**：新增 `arabic_pronunciation.py`，以固定 CAMeL/CALIMA 分析为输入，
   逐词保留原分析、原读音、新读音、规则、上下文边界和例外审核状态。
3. **全量重标与质量复核**：重放原始阿语审计清单；输出可标注/待复核两路，
   高频词、差异明细、随机听审、规则听审和剩余失败原因。不根据一个 ASR 的 CER 自动判原文错误。
4. **重新冻结数据**：英语清单字节不变；阿语按来源分层、按原录音分组划分、候选字节去重，
   生成训练、验证及封存 test。审核确认的文本错误可通过独立排除清单加入。
5. **复用特征并重新训练**：同一 r15 Encoder 的相同音频复用特征，在新分片写入新标签；
   仅新音频提取 Encoder。重新随机初始化 Head，沿用原 40 轮/学习率/早停配置。
6. **评价效果**：分英语、埃及阿语、MSA、来源给出 PER、错误类型与折叠诊断；
   另在旧验证音频可重标子集上，用相同新参考对比旧新 Head。
   母语者核对的小集合和热词召回仍需后续实测；自动标签上的低 PER 不等于真实发音已正确。

旧 r15 参照是英语 PER 4.7172%、阿语 17.5986%、两语言 macro PER 11.1579%。
新主验证集和标签均有变化，不能直接用两个主报告的 PER 相减作为修复收益。
配对报告使用同一参考，但仍是自动参考，旧模型原划分也未证明原录音隔离。

## 发音约定与尚需核对的部分

配置：`configs/arabic_pronunciation_policy.json`。训练和下游查询共用同一个前端。
EGY 使用埃及 CALIMA 的 `diac`；MSA 使用 MSA CALIMA 的 `caphi`。两者映射到 `ar_ipa:*` ID。

| 规则 | 本次默认行为 | 保护条件 |
| --- | --- | --- |
| EGY `_` | 按词内连接符处理 | 其他未知字符仍显式报待复核 |
| MSA 词尾 i_y / u_w | 合为 iː / uː | h_i_y_a、i_y_y 等非目标形式不改 |
| EGY ay/aw | 无显式标音的目标结构转 eː/oː | sukun、shadda、滑音上显式元音保护；有词汇例外 |
| EGY 拼写变体辅音 | 能逐字对齐时恢复转写原辅音 | 长度不同或有其他差异不强行对齐 |
| EGY 齿间音 | 保留 ث/ذ/ظ 的默认开罗约定 s/z/zˤ | 不宣称覆盖全部埃及说话人和借词 |
| EGY 词尾长元音 | aː/iː/uː 缩短 | 独立开关，保留原值供比较 |
| EGY 代词 h | 明确 3ms 后缀、i/u+h 时去 h | 不全局删除词尾 h；ده 等走词汇审核 |
| 冠词连读 | **默认关闭，导出候选差异** | 已实现句中/停顿、前词元音条件，不删除任意喉塞音 |
| MSA 停顿词尾 | **默认关闭，导出候选差异** | 已实现 cas/词性/构造式/附着代词/ة 的条件，不全局去元音 |

前三个候选词汇例外为 ثورة、ده、عشان；未经核对不自动覆盖。包含对应未批准分析的句子
在冻结时隔离。批准例外须记录 `group/surface/diac/phones/status/evidence/reviewer`；
按词和分析精确匹配，不把一个同形词的所有语境强制改成同一读法。
当前仍是 **word-based MLE Top1**，并未声称解决上下文消歧。无分析、未知 CAPHI 等也没有被隐式猜测填充。

规则依据包括用户提供的 [修复分析](ARABIC_CTC_PER_ANALYSIS_2026-11-12.md)、
[CODA 拼写说明](https://camel-guidelines.readthedocs.io/en/latest/orthography/)、
[方言发音说明](https://camel-guidelines.readthedocs.io/en/latest/phonology/) 和
[CAMeL 分析字段](https://camel-tools.readthedocs.io/en/stable/reference/camel_morphology_features.html)。
拼写规范不能直接当作逐条录音的发音真值。

## 工区执行

在原 `qwen3-asr-hotword` 环境、仓库根目录执行；原隔离的 CAMeL runtime 和下载资产继续使用。
建议 tmux。CPU 标注不占 GPU，可与旧 ×3 对照同时进行。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -u -B scripts/run_arabic_pronunciation_refresh.py label --workers 32
```

标注入口限制 1–64 个进程，默认 16，底层数值库每进程 1 线程；32 并不是所有容器都适合，
须有相应 CPU 配额和内存。输出每个分片的真实进度、已完成行数、耗时和心跳。
不读 WAV，不加载 Qwen。中断后：

```bash
python -u -B scripts/run_arabic_pronunciation_refresh.py label --workers 32 --resume
```

完成后先查看 `outputs/arabic_pronunciation_repaired/report.json` 和 `review_delivery/`：

- `ar_egy_top300.json`、`ar_msa_top300.json`：实际语料高频词、每种读音、词次、原分析和完整音频路径。
- `listening.json`：每组最多 50 随机＋50 改动样本，再补规则代表；可能重合。
  填写文本/音素是否对齐和审核者，不伪造词级时间。
- `word_differences.jsonl`：完整逐词差异，留工区，不需要整份上传。
- `blocking_issues.json`：待复核首个原因频数；每句首错统计不等于所有词错误统计。

重导出不会覆盖已编辑审核文件。质量排除清单每行需有 `source`、`row`、`reason`、`reviewer`，
在运行配置 `quality_exclusions` 指向它。没有该清单时不自动删掉高 CER 音频。
修改发音规则或批准词汇例外后，须另指定新的 labels/output 目录，不能混进已封存身份的续跑。

冻结与训练入口已经接通；确认标签摘要及需要的审核修改后执行：

```bash
python -u -B scripts/run_arabic_pronunciation_refresh.py freeze
python -u -B scripts/run_arabic_pronunciation_refresh.py run --gpu 4 --factor 2
```

冻结中断加 `--resume`；run 同命令自动复用已完成分片、从已保存完整 epoch 恢复。
输入/策略/代码身份改变会拒绝续跑。默认新目录：`outputs/en_ar_pronunciation_repaired`。
run 等物理 GPU 4 无计算进程且满足显存条件后启动，不终止其他任务。已运行的旧 ×3 不会被此流程修改。
×3 的结果尚未返回，故新标签对照默认仍 ×2；不能在同一训练目录中间换倍数。

原 ×3 如仍停在 epoch 0，待物理 GPU 4 分配空闲后可单独恢复：

```bash
python -u -B scripts/run_en_ar_merged.py retrain-head --gpu 4 \
  --source-run outputs/en_ar_r15_5100_ctc_v1 \
  --head-time-upsampling-factor 3 \
  --output-dir outputs/en_ar_r15_5100_head_up3_v1 --resume
```

这条旧命令不具备本次新入口的自动空卡排队逻辑，不能与其他 GPU 4 作业同时手工启动。

## 数据与缓存细节

- 阿语目标训练 480h。尽量保留合格 EGY 训练池；MSA 四来源按等时长分配，某来源不足则重分配。
- EGY/MSA 各留 2h validation 和 2h test；每组四来源各 0.5h，按完整条目略超目标。
- 先按原始录音 hash 划分，再在各来源选择时长。EGY 视频、MGB2 录音、广播节目标识分组；
  标贝使用说话人标识。未知命名隔离，不宣称全部来源说话人隔离。
- 旧阿语验证录音不进入新训练；已导出听审/高频词例子的录音不进入封存 test。
  英语训练/验证音频字节、旧阿语验证字节受保护。
- 对选中候选及有限预取完整读文件计算 SHA；16 个线程，SQLite 逐文件持久化，
  续跑复用大小/mtime 未变的已完成指纹。相同字节去重、冲突标签整组隔离并补足配额。
  这不是重编码音频的 PCM 去重，不把文件 SHA 不同当成声学一定不同。
- SHA 校验 r15 模型和原特征分片，相同 ID/路径才复用隐藏状态；新阿语缓存重写标签，原缓存不动。
  英语直接引用旧 r15 缓存，新增阿语音频才提取 Encoder。需另留约一份阿语特征的存储空间；启动前检查。
- test 不进入训练或特征提取。人工标准小集与最终热词评测不能用自动报告冒充。

## 下游一致性接口

```bash
python -B scripts/run_arabic_pronunciation_refresh.py query \
  --group ar_egy --mode isolated --text 'البيت'
```

`sentence` 保留文本停顿边界；`isolated` 是独立词/短语；`linked` 表示首词也处于连读。
下游须携带 group、mode、policy 身份，读取输出的候选例外标志。默认未启用连读规则时三者不会
凭模式名自动删喉塞音。此入口提供统一转换，尚未擅自替换现有生产检索词表。

## 回传与验收

首次标注回传：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/arabic_pronunciation_repaired/return_summary.zip`。

训练后回传：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/en_ar_pronunciation_repaired/return_summary.zip`。
含标签/准备/Head 摘要、分组验证汇总、配对结果、特征复用统计；不含模型、音频或全量逐词明细。
详细逐样本结果在 `validation_diagnostics.json`，Head 在 `head_up2/`。

验收要同时看阿语分组 PER、英语 PER、折叠口径和听审结果。
`human_pronunciation_review_completed=false` 是当前真实状态；不会因为程序跑通自动变为 true。
是否降低真实 PER 要等工区结果及发音核对；不能只通过删音素、换分母或筛掉困难例子宣称提升。

## 本地验证范围

定向测试 44 passed / 3 skipped；新增代码 Ruff、Mypy 通过，CLI 帮助可运行。
全量 pytest 已尝试：本机 Python 3.14 缺 Torch，5 个旧测试文件收集失败；排除这 5 个后，
1041 passed / 67 skipped / 14 failed / 2 errors，余下失败均报告缺 Torch 或 soundfile。
全仓 Ruff 17 项、Mypy 11 项为本次未修改文件中的既有/环境问题，没有为通过检查覆盖他人改动。
本地没有 CAMeL 实资产、r15 权重、GPU 训练或实际音频质量结论。
