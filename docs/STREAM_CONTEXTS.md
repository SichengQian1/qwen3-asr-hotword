# 英语流式 RAG 时间线交付 v1

这是供方2026-10-19文件契约的RAG生产端。它读取因果音频前缀，产出供下游流式ASR
回放使用的JSON；不启动vLLM、不输出ASR转写，也不是实时网络接口或实时性能验收。
源码、配置、Head、词表和输入身份写入run_config，原有整句结果不改。

## 固定资产与检索

配置为configs/english_stream_contexts.workzone.json，引用已交付的
configs/english_best_noise10db_4k.workzone.json。继续使用完成训练的第27轮best、原90类
vocab、完整三语4,000词面/4,012发音索引，以及原Anchor Top64、0.75/0.35/0.5门控。
只增加因果前缀调度及交付层，score仍是原匹配得分，不是词级存在概率。

原干净4k封存plan用于核验音频ID、参考和词表；参考文本只用于资产核验，不进入检索。
不需要先完成新的整句带噪推理。带噪音频仍在：

```text
/host_home/star/两波验证集-可直接推理/2026-10-09-带噪10dB/<en_chile|en_colombia|en_mexico>/en/wav
```

准备先读取全部6,000条的音频头，固定ID集合和实际时长；smoke只对选中的20条完整hash。
选样固定seed、覆盖三国和实际可用的时长桶，不复制/补长音频来凑边界，也不假定已获得
词级跨边界位置。full才读取全部6,000条音频字节。干净音频不重新读取，保留原重复记录。

每个前缀读取当前截止点之前的新帧、合并到当前前缀，再做单声道转换/必要重采样，
最后进入Encoder/CTC。不会先提取整句Encoder特征再切片。原始采样帧控制边界，最后
不足2秒的尾段保留；整数2秒音频不额外增加finish槽。每步独立重检当前累计音频，不做
跨步永久并集，不引入VAD、TTL、门控调参或新的热词标签。

## 输出契约与两处供方歧义

交付标准对象：mode → 数据集key → 音频stem → 各边界候选数组。

| 数据集 | key |
|---|---|
| en_chile | encl@@en |
| en_colombia | encol@@en |
| en_mexico | enmx@@en |

每个候选为原样的word、获胜phoneme和score。一次检索产生完整的合格排序词面列表，
然后分别截成最多5/7个，写入RAG_top5_stream、RAG_top7_stream；空结果为[]。
两个mode共用每步检索，不重复跑Encoder。供方手册要求下游截K，但其原校验器把>K
作为ERROR；本版按校验器兼容方式交付，完整合格列表仍保存在prefix_shards。
Vocabulary字符串及orig/join_phon/nospace变化留给下游，不改变原词和音素。

时序必须显式选择，没有默认值：

| effect | 第0步 | 第1步 | 末步 |
|---|---|---|---|
| same-step | 0～min(T,2)s检索 | 0～min(T,4)s检索 | 当前完整音频前缀检索 |
| next-step | [] | 使用上一步前缀检索 | 使用倒数第二个前缀检索 |

same-step对应手册第3节，需要当步ASR等待RAG；next-step对应第6节的一块延迟。
两种都只交ceil(T/2)个元素；next-step不额外增加尾部槽，最后生成的前缀结果不被
消费，短于2秒的整句只有空context。不得混用两种时序指标。实际主口径需要下游确认。
本版不实现基于ASR部分转写的反馈召回，也不声称它与音频CTC路线相同。

供方原校验器和样例作为测试夹具原样保存在tests/fixtures，来源SHA记录于
stream_interface_v1_provenance.json。校验器以.py.txt保存，避免把第三方格式纳入
项目Python风格修复；运行时由Python显式执行。其durations实际要求key → utt → 秒，
不是手册一处写出的扁平utt → 秒。本版还检查有限得分、排序、去重、两个mode一致性
和实际输入截止点；仅供方脚本通过不等同于因果性或真实时延证明。

## 工区运行

在容器内项目根目录，使用原Qwen环境。先拉取codex/g2p-coverage-scan并比对交付SHA。
4换成已分配的物理GPU。建议首轮明确用same-step验证前缀与接口；这不代表下游已
确认其调度。如果下游要求第6节，改用next-step，输出目录会自动分开。

```bash
python -B scripts/run_stream_contexts.py smoke --effect same-step --gpu 4
```

阶段、真实样本/前缀计数、耗时和10秒活动提示持续显示。可选--audit-only、不带GPU，
只做准备核验；正常命令本身已经核验，不必重复跑。中断时保持同一Git/配置/资产：

```bash
python -B scripts/run_stream_contexts.py smoke --effect same-step --gpu 4 --resume
```

默认新目录outputs/english_stream_contexts_ep27_noise10db_v1/smoke_same-step。
已完成前缀通过摘要/PCM输入/音频身份检查后复用，完整成功后--resume不再加载模型。
输出目录拒绝覆盖和嵌套输入。独占锁位于输出旁的orchestration目录，进程退出释放
OS锁，不通过删锁绕过活进程。0.170/0.171源码和输出不改；旧任务恢复仍须使用它们
封存的Git版本，不强行放宽旧身份。

| 产物 | 用途 |
|---|---|
| delivery/stream_contexts.json | 给下游的两个mode时间线 |
| delivery/durations.json | 按数据集分组的真实时长 |
| delivery/audio_manifest.json | 本次音频集合、原路径、帧数和SHA |
| causality_audit.jsonl | 每个消费步对应的检索步及输入截止点 |
| prefix_shards/ | 完整合格候选、CTC/匹配细节、前缀PCM摘要和耗时 |
| report.json / run_config.json | 契约验证、固定身份、选择范围和真实前缀统计 |
| return_bundle.zip | 上述交付文件和小报告，不含权重、WAV或全部前缀原件 |

smoke只覆盖选定20条，**下游必须按audio_manifest选择这20条**，不能拿整个2,000条
源目录去要求smoke JSON全覆盖。full才覆盖整个6,000条。完成smoke后将return_bundle.zip
回传核验，再让下游检查实际context采用和部分/最终转写，确认时序后运行全量：

```bash
python -B scripts/run_stream_contexts.py full --effect same-step --gpu 4 \
  --smoke-run outputs/english_stream_contexts_ep27_noise10db_v1/smoke_same-step
```

full要求成功且同Head、配置、代码、音频和effect的smoke；不自动从smoke启动full。
默认输出full_same-step，恢复也加--resume并保持--smoke-run一致。不要混用不同
effect的smoke。report的prefix_compute_seconds不含ASR解码或网络等待，不能当作
端到端实时延迟。最终WER、目标词命中、参考外词落字及真正采用时序由下游日志评估。
