# 中英183类：补中文480小时、双H200特征提取及Head训练

本轮范围：数据与标签准备 → 冻结Qwen3-ASR-1.7B提特征 → 随机初始化Head训练 → 验证PER。
不做热词表、检索、delivery或最终ASR。英阿任务保持独立。

## 固定配置

| 项目 | 本轮设置 |
|---|---|
| 入口 | `scripts/run_zh_en_dual.py` |
| 配置 | `configs/zh_en_noise70_dual.workzone.json` |
| 原结果，只读复用 | `outputs/zh_en_183_noise70_scratch_v1` |
| 新输出 | `outputs/zh_en_183_noise70_dual_v2` |
| 训练规模 | 中文480h；英语480h，其中美音336h、英音144h |
| 时长检查 | 三组均须达到目标；按完整音频取样，可能略超目标；不足即停，不复制样本补小时 |
| 原中文源 | `/glusterfs_103/asr-data_imedia_zh-en/zh/ASRCHINESE0025-noah_carrec230-000500H_shards` |
| 新增中文源 | `/glusterfs_103/asr-data_imedia_zh-en/zh/ASRCHINESE0007-KingASR059-000620H_shards` |
| MFA模型 | 普通话 `mandarin_china_mfa` 3.0.0；美音/英音分别 `english_us_mfa` / `english_uk_mfa` 3.3.0 |
| 音素表 | `configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json`，183类，ID不变 |
| CPU | 并行上限64；受CPU配额/可用RAM约束；底层OMP/MKL/OpenBLAS等线程为1 |
| GPU | 用户分配的物理4、5卡；不自动选择或停止其它任务 |
| 噪声 | 各训练来源70%记录替换为10dB混音，30%原声；不是额外复制70%样本 |
| Head | 2倍时间上采样temporal Head，隐藏512、卷积核5、dropout0.1；随机初始化 |
| 训练 | 冻结Encoder；batch256、初始lr3e-4、最多40epoch；4卡训练，5卡提特征后释放 |

验证集沿用来源划分，目标中文10h、美音5h、英音5h；可用下限分别5/2.5/2.5h，报告不足目标的小时。
每条验证音频有original和noise10db配对版本。选模使用中/英×干净/带噪四组macro PER，
并报告中/美/英×干净/带噪六组PER。test封存，不生成其训练标签、不提特征、不参与选模。
PER针对自动音素标签，不是人工发音正确率，也不等于热词召回或ASR WER。

## 旧成果如何复用

1. 固定旧contract SHA：`2824ccdba79972f85b812c28be6c4eea64988c5a52056944c5e84abe0a1c3073`。
   核验旧代码、配置、MFA基础资产、每个已完成标签分片的SHA/行数，以及全量来源计数。
2. 旧三来源的音频审计、文本前端和标签不重算。新目录保存标签视图，旧目录任何文件不改。
3. 只读取新增中文tar音频并计算内容指纹；与旧PCM指纹合并去重。
   新数据若连到旧验证/测试音频，相关新文件族继承该划分；连到不兼容旧划分则隔离。
   转写冲突也隔离，旧记录的ID、标签和split保留，额外问题写在新视图中。
4. 在新增G2P之前先检查各组可用音频小时；完成新标签后再检查完整标签小时。
5. 新文本先做同版中文前端；唯一词优先复用旧G2P结果，包括旧失败状态。
   仅缺失词进入固定模型的Top1 G2P；失败记录单列，不丢字符、不拿部分发音冒充完整成功。
6. 优先保留原可用中文训练记录，再从新增源补足480h。按固定seed取样及分配噪声。

已做的是精确16k单声道PCM去重；近重复、说话人身份尚未验证。缺少speaker/session元数据时
文件名族只是隔离代理，不能宣称真正说话人独立。新增源须为未压缩tar里的同名txt/wav配对，
脚本不会将620H目录名当作真实时长；实际形式或时长不符时保留诊断并停止。

## 双卡比较及正式提取

不是“中文固定4卡、英语固定5卡”。两卡从同一个分片队列领取任务，较长分片优先，完成一片
再领取下一片。SQLite事务保证同一分片只被一个工作进程领取。

- 比较每卡1、2、4进程，即两卡合计2、4、8个独立模型实例。每个进程只看见分配卡的`cuda:0`。
- 同一批训练音频按中/美/英、原声/带噪、≤4s/4–10s/10–30s分层固定抽取，最多288条；
  每16条一个测速任务。测速不使用验证集或test，不进行小样本训练。
- 各档独立加载并预热，所有进程就绪后统一开始计时；进行两轮，顺序1→2→4、4→2→1。
- 比较两轮“处理音频秒数/实际墙钟秒数”的中位数。模型加载/预热单列，不计入该吞吐；
  音频读取、混音、推理、缓存写入及核验均计入。不会把旧缓存命中当作新推理速度。
- 两轮都完成的档位才可获选，速度相同优先少进程。输出每档耗时、吞吐、显存峰值和失败原因。
  这是该环境固定样本上的选择，不保证所有超长音频或变化后的存储负载下仍最优。
- 先按每进程12GiB＋每卡8GiB余量检查空闲显存（规划值，不是实测占用）。显存不足/OOM档位
  排除；数据、标签或身份错误直接停止，不能以换档掩盖。
- 正式任务若显存不足，从选中档位依次减到2/1；核验并保留已完成分片，只补未完成部分。
  每卡1进程仍不可运行则停止，不抢占其它任务。
- 获胜轮已生成的音频特征会复用到正式缓存；其它测速轮保留记录。正式分片仍按512条组织。

Encoder仍保持官方逐音频计算路径，每个进程有一个有界读取预取线程；同卡多进程提供并发。
未改成padding后一次forward多条音频，避免未经验证改变浮点计算/长度行为。
模型实例仍可能包含完整Qwen权重，因此多进程会增加显存占用；吞吐提高多少以工区结果为准。

Head训练保持单卡及原batch/学习率，不增加DDP；模型很小，不能仅凭显存空闲推断双卡训练更快。
这样本轮只改变数据规模和提取吞吐，不同时引入分布式训练行为差异。

## 运行与恢复

分支固定 `codex/g2p-coverage-scan`。在原 `qwen3-asr-hotword` 环境执行：

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/run_zh_en_dual.py all --gpus 4 5
```

建议在已有tmux会话中执行，断开本地连接后任务仍可继续。不要同时再开旧的`run_zh_en_training.py all`。
旧标注目录必须保留；无需新安装vLLM、不下载Qwen模型、不修改aligner环境。

中断后使用**相同命令**，不另加`--resume`：

- 音频审计分片、中文前端、G2P和标签核验身份后跳过；元数据仍可能重新扫描、校验和聚合。
- prepare完整完成后直接核验其产物，不重新跑G2P。
- 完成的测速轮可复用，中途未完成的一轮重新计时，不能把残缺计时当成公平比较。
- 正式缓存逐分片复用，训练恢复最近完成epoch的模型/优化器状态；未保存epoch需重跑。
- 不会覆盖旧v1目录；新目录绑定代码/配置/来源身份。身份不一致时停止，不能删校验绕过。
- 检测到上次GPU工作进程可能仍在运行时拒绝重复启动；异常强杀或更换主机后，先确认旧任务已退出。

查看保存进度（不加载模型）：

```bash
python -B scripts/run_zh_en_dual.py status
```

实际GPU子进程日志：`outputs/zh_en_183_noise70_dual_v2/gpu_runs/<轮次>/worker-*.log`。
训练日志：`outputs/zh_en_183_noise70_dual_v2/logs/train.log`；epoch记录：`head/metrics.jsonl`。
日志中的`status`只是最近保存的状态；`status`命令不是实时进程存活证明。

结果：

- `head/ctc_head_best.pt`、`head/ctc_head_latest.pt`、`head/training_state_latest.pt`
- `training_completed.json`、`validation_per.json`
- `label_report.json`（来源可用量/失败原因/实际小时/噪声比例/音素频次）
- `encoder_autotune.json`（测速与选择）、`production_workers.json`（正式档位/回退）
- `return_bundle.zip`（小型报告、epoch指标和SHA清单，不含音频、张量或模型）

结束或失败后返回：
`/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en_183_noise70_dual_v2/return_bundle.zip`。
尚未在工区运行本轮；本地逻辑/模拟进程测试不能替代H200速度、完整数据兼容性和PER结果。
