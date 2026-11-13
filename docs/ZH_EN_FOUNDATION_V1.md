# 中英项目第一步：183类音素表、MFA资源及语料格式检查

2026-11-13 用户确认中文＋美式＋英式的 G2P 输出表。使用原始
Qwen3-ASR-1.7B，后续冻结 Encoder、从头训练新 Head。本次只交付资源和格式检查，
不启动标注、提特征或训练。英阿实验继续使用其独立资产。

## 已确定的资源

| 分支 | G2P版本 | 实际输出单位 | 工区仓库内目录 |
| --- | --- | ---: | --- |
| 大陆普通话 | mandarin_china_mfa 3.0.0 | 134 | models/mfa/hf_g2p_20261113_zh_en/zh/g2p/mandarin_china_mfa |
| 美式英语 | english_us_mfa 3.3.0 | 70 | models/mfa/hf_g2p_20261029_v03/en/g2p/english_us_mfa |
| 英式英语 | english_uk_mfa 3.3.0 | 72 | models/mfa/hf_g2p_20261113_zh_en/uk/g2p/english_uk_mfa |

中文、英式 FST 和运行元数据按用户要求直接纳入 Git，pull 后即可访问文件；
美式沿用此前工区已部署的固定版本，不重复保存。每个新 FST 约39MB，均小于
GitHub单文件限制。这里仅包含小型 MFA 模型，不包含 Qwen、语料、特征缓存。
资源附带上游模型卡、CC BY 4.0 许可和来源说明。

中文固定官方 revision `cfbee14002d514fcf411c16676877a96b7afe6ff`；
英语固定 `1009e3c6c689f10875146a5e4e5c2233f8eaf922`。
`model.fst` 和 `meta.json` 保持上游字节；新包的两份 `.txt` 和两份 `.sym`
均从 FST 内嵌符号表导出。官方 UK `.sym` 的编号与实际 FST 不一致，不能原样用于
解码；所交付的 `.sym` 已按内嵌表导出并记录新 SHA。美式继续使用既有 `.txt`。

- 词表：`configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json`。
- 逐来源映射：`configs/phonemes/zh_en_mfa_g2p_mapping.v1.0.tsv`。
- 工区路径和资源 SHA：`configs/zh_en_foundation.workzone.json`。

共有181个发音单位，加 blank/unk 后183类。中文和美式共享29个单位；英式仅新增
`fʷ vʷ əw ɛː ɜ ɜː`，放在原中美177类之后。保留声调、长音、送气、唇化、腭化及
成音节标记；不按字符拆开多字符IPA。三份FST没有独立重音输出，本次没有删除重音。
该表用于G2P输出标签，不直接兼容官方词典的全部扩展变体；如果改变为词典优先策略，
需要重新审定单位覆盖。中文多音字、轻声和变调前端尚未确定。

## 用户提供的语料

```text
美式JSON：/host_home/z00841352/27A/data/en/json/swift_en_美式英语.json
英式JSON：/host_home/z00841352/27A/data/en/json/swift_en_英国.json
中文根目录：/glusterfs_103/asr-data_imedia_zh-en/zh/ASRCHINESE0025-noah_carrec230-000500H_shards
基模：/glusterfs_103/models/Qwen3-ASR-1.7B
```

本机不能读取这些工区语料。中文目录名称中的shards/500H不是格式或有效时长证明。
历史美式JSON曾包含US/AU/CN标签；不能把整个文件名直接当成逐条美式口音认证。
本次先保留原字段和来源样例，后续选样再确定明确的口音范围。
中英总时长、美英训练比例、噪声比例及新的训练/验证划分尚未冻结。

## 工区运行一次

分支 `codex/g2p-coverage-scan`，在现有 Qwen Python 环境执行；无需GPU和新的依赖安装。

```bash
cd /host_home/star/q00933266/qwen3-asr-hotword
git pull --ff-only origin codex/g2p-coverage-scan
git rev-parse HEAD
python -B scripts/inspect_zh_en_foundation.py
```

检查资源 SHA 和183类映射；每份英语JSON最多读取8行；中文目录最多观察10,000个文件，
选每种类型至多2个、合计至多12个文件检查。tar按流读取至多64个成员，限制跳过的
累计声明大小；只查看少量文本，不解包、不加载音频，不读取完整语料到内存。
若存在同名音频和文本，会报告配对样例；发现配对只说明布局线索，不认定训练格式已通过。
已知test/dev/validation命名路径不读取文本；尚不能证明未标记来源的split身份。
未知二进制、挂载缺失、JSON结构不匹配都记录到报告，不猜测格式或静默丢条目。

默认输出每次创建新的 `outputs/zh_en_foundation_probe_<随机ID>`，不覆盖历史输出。
这是短时只读检查，没有训练断点；中断后重新执行同一命令会生成新目录。
指定 `--output-dir` 时目录必须不存在。仅检查资源可加 `--assets-only`。

返回终端给出的 `return_bundle.zip`，内含 `report.json` 与本次配置。
`status=inspection_completed` 只表示检查程序完成，需同时查看 `assets.status`、
各输入的error/schema_parse_error、实际样例与split来源。它不代表标注或训练完成。

取得报告后，先据实际中文分片结构建立音频—文本读取契约，再落地汉字到带调拼音、
三分支G2P、完整183类标签和拒收清单。大规模标注、去重划分、提特征及训练均另起
独立输出。当前没有新中英训练结果。

## 本地验证

真实固定资产SHA、三分支输出表、全部单位的NFC/NFD映射核验通过；
边界检查覆盖未知单位整条拒收、tar不解包及不跟随链接、已知heldout跳过、
JSON限量读取、挂载缺失和未知格式保留。相邻音素表回归共21项通过。
只做CPU资产和小型合成目录验证，没有声称读取了工区语料或运行完整Qwen。
