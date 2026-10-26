# 新版MMService G2P三语全量复测（2026-10-26）

## 本轮比较对象

用户提供新版`mmservice_g2p_cpp(1).zip`，包SHA256为
`2408faf751ac07e4ff83f3d2081a5eabe1a354a36944b968132e523dfb583ff1`。
8份源码/头文件/Makefile/README的SHA封存在`configs/mmservice_g2p_v2.workzone.json`。
用户将工区`mm/`替换为新版；测试先复制到全新构建目录，不覆盖旧独立构建或旧结果。

沿用上轮实际训练清单和原词典：EN/ES/PT各约480小时，共881,749句。
语言内唯一词分别12,844 / 92,715 / 101,940，共207,499个语言×词输入；
来源×词标签对分别12,844 / 144,499 / 161,686。保留同词不同来源的原标签。
完整逐句重建再次核对原训练ID，再比较新版输出的IPA和90类ID、按词频加权差异及整句差异。
参考是原生产MFA标签，不重新跑官方MFA，也不把标签一致率称为真实发音准确率。

新版代码增加`--lang`；必须显式传入`en/es/pt`。西语启用推理前`ñ→ni、gü→gw`、去尖重音，
以及发音去鼻化和代理滑音合并。新版对所有ES词预先做代理；旧项目仅缺词才做代理，
因此可能修复旧差异，也可能改变旧`base_exact`词。测试原样运行新版，不先替同事改规则。
额外ES清洗结果仍单列；判断新版服务行为以明确`--lang`后的native结果为主。

## 工区命令

在容器仓库根目录使用既有`aligner`。先将压缩包中的8个文件直接放到`mm/`，确保
`mm/mmservice_g2p_main.cpp`、`mm/Makefile`、`mm/README.md`存在，再执行：

```bash
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
```

新目录编译（与上次相同的x86/OpenFST方式，覆盖Makefile默认ABI编译参数）：

```bash
B=$(mktemp -d outputs/mm_g2p_v2_x86_XXXXXXXX)
cp mm/*.cpp mm/*.h mm/Makefile mm/README.md "$B/"
conda run --no-capture-output -n aligner bash -c 'make -C "$1" OPENFST="$CONDA_PREFIX" CXX=g++ CXXFLAGS="-std=c++17 -O2" -j2' _ "$B"
```

编译成功后运行三语全量比较：

```bash
conda run --no-capture-output -n aligner python -B scripts/compare_mmservice_g2p_training.py \
  --release-config configs/mmservice_g2p_v2.workzone.json \
  --binary "$B/mmservice_g2p"
```

`B`是当前终端变量；换终端后将其重新设为刚打印/实际生成的构建目录。测试核验8份源码SHA、
ELF x86_64和执行权限，记录新二进制SHA，逐语种调用`--lang en/es/pt`。完整G2P二进制需在工区编译，
本地检查不冒充工区FST推理。程序未自动证明二进制与源码的编译关系，按上述命令从审核副本编译。

## 结果及回传

默认自动创建`outputs/mmservice_g2p_training_<随机后缀>`；可用`--output-dir`指定全新目录。
无resume，拒绝覆盖或与输入目录嵌套。失败时保留目录及日志，再试使用新目录。
输出包括：

- `report.json`：新版本及源码身份、二进制SHA、实际语言参数、三语全部统计。
- `words.jsonl`：全部来源×词标签对，含原标签、新版native和额外ES清洗标签/ID。
- `samples.jsonl`：全部训练句的标签核对和比较状态。
- `delivery/en_differences.jsonl`与`.tsv`：英语所有native不一致/失败/OOV条目。
- `delivery/es_differences.jsonl`与`.tsv`：西语所有native不一致/失败/OOV条目。
- `delivery/pt_differences.jsonl`与`.tsv`：巴葡所有native不一致/失败/OOV条目。
- `delivery/README.md`：同事交接说明。无差异时JSONL为空，TSV仅有表头。
- `return_summary.zip`及每份≤32KiB的`return_part_NNN.txt`：小报告/示例及大明细身份。

把小ZIP或全部编号TXT（二选一）及终端摘要返回即可；完整`delivery`留在工区，可直接交同事。
大明细不会塞进小回传包；`report.delivery.files`记录全部交付文件SHA/大小，供核验。

本轮不用GPU、不读取WAV、不提Encoder、不重训CTC或改生产词典/检索。
有命名阶段、实际完成数和10秒活动提示，`--timeout`默认7200秒/语种C++进程。
上一轮巴葡C++全量约39.6分钟，仅作为历史耗时参考，不保证新版时长。
完成后比较新版native各类差异数，并检查旧9个失败词及其余ES差异是否消失、是否新增差异。

## 本地验证边界

本地使用合成词典/模拟FST进程验证参数、身份、统计和交付完整性，不加载三语模型。
另对附件原C++字符串模块实际编译小探针，验证ES去鼻化、EN/PT保留鼻化、分词及显式滑音合并；
该探针不包含FST推理。实际全量结果必须以工区新报告为准。

相关G2P测试60 passed，全仓863 passed（133.69秒）；改动实现/CLI的Ruff、严格Mypy、
Python3.10语法、CLI help及`git diff --check`通过。
