# 英文热词分组交付说明｜中英 480 小时模型 · Context Learning

这份说明随 `grouped_delivery.zip` 交给下游转写和评测同事。五个包用于回答一个问题：**面对同一套 2,351 词中英候选库，常规英文、跨语言专名、品牌、导航名称和特殊读法，各自能给转写提供多少帮助？**

每包给出一批英文音频的工区路径，以及原系统在每个时间点实际送出的热词提示。下游可用同一段音频分别跑无提示、Top5 提示和 Top7 提示，再用该组的逐句目标评分。本轮使用中英 183 类音素模型、480 小时版 CTC Head（epoch 25）。文中召回率衡量检索候选是否包含目标；最终转写效果需用下游生成的文本计算。

## 1. 五个包分别测什么

| 外包内目录 | 想测的问题 | 目标热词举例 | 实测词数 | 目标句词对 | 末次门控后 Top7 词面召回 |
|---|---|---|---:|---:|---:|
| `regular/` 常规英文 | 英文表达、常规英文写法的名称，在智慧家庭句子中能否被找回？ | `call back`、`Philips Hue`、`RedNote` | 13 | 59 | 74.58% |
| `cross/` 跨语言专名 | 英语句子里出现转写人名、地名或含外语名称的品牌时，能否找回指定写法或同一实体？ | `Hongyadong`、`Shen Teng`、`Louis Vuitton` | 73 | 355 | 21.97% |
| `review/` 边界词及其他品牌 | 读法或分类边界需要单列的品牌、缩写、人名及其他表达，表现如何？ | `Ecovacs`、`Midea`、`PM2.5`、`Paul Atreides` | 21 | 98 | 21.43% |
| `navigation/` 导航实体 | 英文合成导航句子中的站点、机场、道路、片区、景点等能否被找回？ | `Al-Malaz 503`、`KAFD`、`Riyadh railway station` | 525 | 50,306 | 22.98% |
| `multi/` 数字／缩写及列明多读法 | 把跨域的数字读法、英文缩写、指定多读法条目放在一起看，效果如何？ | `Ne Zha 2`、`FTTR`、`KAFD`、`Al-Malaz 503` | 284 | 27,665 | 22.92% |

**“目标句词对”是召回的分母。** 同一条录音标有两个不同的目标词，贡献两个句词对；同一个词在同句重复，计一个。前四组按词面互斥，共 632 个实测目标词、50,818 个句词对。第五组是叠加标签。按句统计时，即使前四组也可能共用一条音频。各包实际句数看外包根目录 `summary.json` 的 `utterances` 字段。

## 2. 怎样从原始大包划出这五组

1. **先固定原系统的候选。** 检索时每条音频都面对同一张 2,351 词联合表。导出直接读取已完成运行的缓存，包括每个时间边界的词、发音、得分、排序以及空列表。分组阶段只挑选音频和评分目标。
2. **找到每句话要考哪些词。** 智慧家庭采用元数据中的 `keywords`，英文正例来自 `val_v1`（同词新句，282 句）和 `val_v2`（留出词，200 句）。英文导航采用源索引每句的 `entities[].surface`，按源音频路径 MD5 的前 16 位关联最终 WAV，只计属于导航 583 词清单且在对应录音有标注的目标。25,839 条导航录音实际覆盖其中 525 词。英文智慧家庭的 180 条无目标负句保留在原始交付，供误提示评测。
3. **先按来源，再按词分组。** 智慧家庭 107 个词逐项指定一个主组：13 个 `regular`、73 个 `cross`、21 个 `review`。导航清单统一归 `navigation`。例如 `Hongyadong` 是地名，但它来自智慧家庭，主组是 `cross`；`Kingdom Arena` 虽然是英文写法，来源是导航清单，主组就是 `navigation`。这套规则描述本轮业务场景和词面，发音人是否用美式口音需另行确认。
4. **额外打上 `multi` 标签。** 智慧家庭固定四词：`FTTR`、`PM2.5`、`Ne Zha 2`、`ORVIBO`。导航按原始词面筛选：含数字，或含一个由至少两个大写英文字母组成的完整词段（实现为 `\d|\b[A-Z]{2,}\b`）。实测命中 280 个导航词，加上上述四词，共 284 个。该标签标示需要观察的读法现象；单个词到底念字母还是连读，仍以音频为准。
5. **按目标分包，按音频去重。** 一句含某组任一目标，就把这句完整时间线放进该包，在该包的评分文件中只列属于本组的目标。同一句可进多个包，在每个包内只出现一次。候选中即使出现别组词也原样保留。跨包汇总句数时按原始 `key + sample_id` 去重；相同推理设置下，同模式的转写也可按这个身份复用。

**实际例子。** 导航录音 `c07a78326d26b000` 的句子提到了 “Al-Malaz five hundred and three” 和 “Al-Abbas Al Ghanwai four hundred and four”。逐句标注写作 `Al-Malaz 503` 和 `Al-Abbas Al Ghanwai 404`。两个词都属于 `navigation`，也都触发 `multi`。这条录音会在两个包里各出现一次，每包各贡献两个目标句词对。下游只需为同一模式生成一次转写，即可分别计入两组。

## 3. 哪些词算常规、跨语言和边界词

智慧家庭采用逐词固定清单。读这一节就能查全三个主组。品牌所属国家、词是否用拉丁字母写、库里是否有中文别名，都不足以单独决定其主组。例如品牌英文名 `RedNote` 在常规组，`Douyin` 在跨语言组；`Ecovacs` 在边界组。三者都可以合法地出现在英语句子里。

### 常规英文（13 词）

- `answer for me`、`call back`、`ClassIn`、`DingTalk`、`Dune: Part Two`、`Fliggy`、`Hive Box`、`MatePad`
- `Matthew McConaughey`、`Philips Hue`、`RedNote`、`Roborock`、`the Bund`

### 跨语言专名（73 词）

- `An Xin`、`Ao Bing`、`Cainiao Station`、`Chen Jing`、`Chiang Mai`、`Da Nang`、`Da Shi`、`Daocheng Yading`
- `Denis Villeneuve`、`Dianping`、`Douban`、`Douyin`、`Du Leying`、`Duoduo`、`Ele.me`、`Frant Gwo`
- `Gao Qiqiang`、`Gaotu`、`Guilin`、`Gulangyu`、`Haidilao`、`Hermès`、`Hongyadong`、`Hu Ge`
- `Huanghe Road`、`Huangshan`、`Jeju Island`、`Jia Ling`、`Jiuzhaigou`、`Kuaishou`、`Lei Jiayin`、`Li Xin`
- `Lijiang`、`Liu Peiqiang`、`Liu Rui`、`Louis Vuitton`、`Meituan Delivery`、`Mr. Bao`、`Nannan`、`Ne Zha 2`
- `Phuket`、`Qianqian`、`Qin Hui`、`Qingdao`、`Qunar`、`Sanya`、`Senso-ji`、`Sentosa`
- `Shen Gongbao`、`Shen Teng`、`Shi Qiang`、`Sun Jun`、`Taiyi Zhenren`、`Timothée Chalamet`、`Tu Hengyu`、`Wang Jun`
- `Wang Miao`、`Wong Kar-wai`、`Wu Jing`、`Xi'an`、`Xiamen`、`Xishuangbanna`、`Xu Jizhou`、`Xueersi`
- `Ye Wenjie`、`Yu Hewei`、`Yuanfudao`、`Zhang Luyi`、`Zhang Songwen`、`Zhang Yi`、`Zhang Yimou`、`Zhangjiajie`
- `Zuoyebang`

### 边界词及其他品牌（21 词）

- `Arc'teryx`、`Bilibili`、`Chani`、`Chimelong`、`Cillian Murphy`、`Dreame`、`Ecovacs`、`FTTR`
- `Genshin Impact`、`Gree`、`Haier`、`Klook`、`Mi Home`、`Midea`、`ORVIBO`、`Paul Atreides`
- `PM2.5`、`qipao`、`Seewo`、`Tuya`、`Zendaya`

## 4. 工区文件从哪里拿

**给下游传 `grouped_delivery.zip`。** 工区容器仓库根目录是 `/host_home/star/q00933266/qwen3-asr-hotword`。导出程序每次创建带唯一后缀的目录，完整路径形如：

```text
/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en480_group_delivery_<本次后缀>/grouped_delivery.zip
```

以导出终端最后一行「请交给下游」为准。找不到那行时，可在工区执行以下只读命令列出已完成的交付：

```bash
find /host_home/star/q00933266/qwen3-asr-hotword/outputs -maxdepth 2 -type f -name grouped_delivery.zip -print
```

| 分组 | 总包内的独立交付路径 | 工区原始音频 |
|---|---|---|
| 常规英文 | `regular/delivery.zip` | `/data/z00816262/tts_zh_en_v1/wav_16k/` |
| 跨语言专名 | `cross/delivery.zip` | `/data/z00816262/tts_zh_en_v1/wav_16k/` |
| 边界词及其他品牌 | `review/delivery.zip` | `/data/z00816262/tts_zh_en_v1/wav_16k/` |
| 导航实体 | `navigation/delivery.zip` | `/data/z00816262/en_tts_final_zh_en_v2/` |
| 数字／缩写及列明多读法 | `multi/delivery.zip` | 同时涉及上述两个目录；逐句取本包 `audio_manifest.json` 的 `audio_path` |

五个目录位于总压缩包内；WAV 位于工区共享盘。职责不同的几个源文件如下：

| 用途 | 工区容器完整路径 |
|---|---|
| 2,351 词联合检索库，含词面、音素、ID 和来源 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/zh_en_joint2351.v1.jsonl` |
| 智慧家庭 107 词的主组、认可别名、附加标签 | `/host_home/star/q00933266/qwen3-asr-hotword/configs/zh_en_english_eval_groups.v1.tsv` |
| 智慧家庭逐句标注、口语化文本及 WAV 路径 | `/data/z00816262/tts_zh_en_v1/meta_16k.jsonl` |
| 英文导航逐句实体和源音频路径索引 | `/host_home/star/zh_en_work/en_tts_index_step25.jsonl` |
| 原 11 组完整交付（包括负例） | `/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en183_480_joint2351_eleven_v1/delivery.zip` |
| 本轮已生成的分组指标、逐词命中数 | `/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en480_english_groups_ke7442yc/groups.tsv`、同目录 `words.tsv` |
| 后十名原音听审包 | `/host_home/star/q00933266/qwen3-asr-hotword/outputs/zh_en480_english_groups_ke7442yc/listening.zip` |

## 5. 下游怎样接入和评分

以 `cross/` 为例。先解开外层压缩包，再解开 `cross/delivery.zip`。保留旁边的 `cross/evaluation_targets.jsonl` 供评分。

| 文件位置 | 下游使用方法 |
|---|---|
| 内包 `delivery/audio_manifest.json` | 用原数据集 key 和 utterance ID 定位音频并读取 `audio_path`。同一包可含多个原 key，如 `en-smarthome-val_v1@@en` 和 `en-smarthome-val_v2@@en`。 |
| 内包 `delivery/durations.json` | 读取同一 key、ID 对应的时长（秒），对齐 2 秒边界；末尾不足 2 秒也有一个边界。 |
| 内包 `delivery/stream_contexts.json` | 按 `模式 → 原 key → ID → 时间边界` 读取。模式为 `RAG_top5_stream`、`RAG_top7_stream`。每个边界用截至当时的累计音频检索，当步生效；候选包含 `word`、`phoneme`、`score`。遇到空列表就使用空热词提示。 |
| 外包 `cross/evaluation_targets.jsonl` | 每行用 `key`、`sample_id` 与模型输出对齐。只在离线评分时读取 `targets[].surface` 和 `accepted_aliases`。将这些答案注入 prompt 会使评测失真。其他四组沿用同样的目录结构。 |
| 外包根目录 `summary.json` | 核对每组词数、目标句词对数、句数、时间边界数和内包 SHA256。 |

**词面和实体分开评分。** 对英文目标 `Hongyadong`，严格词面命中要求候选中出现 `Hongyadong`（比较时忽略大小写并归一空白）。实体口径还接受本表登记的 `洪崖洞`。两种算法用同一组音频、相同的 TopK 和分母；未登记别名的词只接受本词。本轮该词有 3 个句词对，两种口径在末次 Top5、Top7 的门控前后都是 0/3。下游最终是否写对地名，仍要单看转写结果。

## 附录：导航和附加组的完整实测词面

下面按原词面的拼写列出 525 个实际出现在逐句评分目标中的导航词。分为两张清单，避免同一长词表重复展示。需要知道某个词属于哪些**录音**时，以各组 `evaluation_targets.jsonl` 为准。

### 同时进入 navigation 和 multi（280 词）

- `Abbas Bin Firnas 202`、`Abdulaziz Al Olayan 402`、`Abdullah Al Ameri 101`、`Abdullah Bin Abbas 101`、`Abdullah Bin Al Zubayr 406`
- `Abdullah Bin Qays 301`、`Abdullah Bin Rawaha 201`、`Abdullah Bin Shihaiween 204`、`Abdullatif Bin Mubarak 402`、`Abdulqadir Al Jazairi 301`
- `Abdulrahman Ad Dakhel 201`、`Abdulrahman Al Ghafqi 301`、`Abdulrahman Bin Owf 202`、`Abi Al Aswad Al Do'Li 106`、`Abi Ayoub Ansari 301`
- `Abi Bakr Alsiddiq 214`、`Abi Hafs bin Shaheen 101`、`Abi Jafar Al Mansoor 301`、`Abi Omama Al Harethi 401`、`Abi Salma 201`
- `Abi Tammam 102`、`Ad Dar 402`、`Ad-Dirah 605`、`Al A'Sha 401`、`Al Ahmadi Port 101`
- `Al Ahsa' 207`、`Al Batha 201`、`Al Dharan 201`、`Al Farazdaq 101`、`Al Faryan 205`
- `Al Hadida 401`、`Al Haeer 101`、`Al Hamasa 401`、`Al Haweya 301`、`Al Hawta 101`
- `Al Imam 201`、`Al Jamea 303`、`Al Jarir Valley 201`、`Al Johari 402`、`Al Khadraa 101`
- `Al Khafji 201`、`Al Khalifah Al Mamoon 301`、`Al Khalil bin Ahmed 205`、`Al Kharj 101`、`Al Kharj 24 A`
- `Al Khatib Al Baghdadi 401`、`Al Maamoura 201`、`Al Mahad 102`、`Al Majrada Valley 401`、`Al Mansuriyah 101`
- `Al Nadwah 301`、`Al Nafisi 401`、`Al Oraija 404`、`Al Qadesiya 101`、`Al Qaraa 301`
- `Al Qiyas 201`、`Al Remal 403`、`Al Shafa 415`、`Al Sharae' 401`、`Al Shati 203`
- `Al Taqwa 301`、`Al Thumeiri 201`、`Al Urubah 404`、`Al Waraq 101`、`Al Washm 401`
- `Al Zewawi 402`、`Al-'Anouz 301`、`Al-'Aziziyah 107`、`Al-Abbas Al Ghanwai 404`、`Al-Aflaj 402`
- `Al-Anwar 201`、`Al-Aziziyah 501`、`Al-Badeah 603`、`Al-Bahar Al-Mutawasat 204`、`Al-Basala 301`
- `Al-Biqa' 202`、`Al-Ezdihar 604`、`Al-Fakhiriyah 601`、`Al-Falah 601`、`Al-Hada 604`
- `Al-Hallah 301`、`Al-Hamra 503`、`Al-Hashemi-II`、`Al-Iskan 505`、`Al-Jazeerah 507`
- `Al-Khaleej 616`、`Al-Khalidiah 22 B`、`Al-Khbar 101`、`Al-Maizaliyah 601`、`Al-Malaz 503`
- `Al-Mansoorah 202`、`Al-Mansurah 20 A`、`Al-Mansurah 604`、`Al-Maseef 602`、`Al-Mathar 501`
- `Al-Muanisiyah 504`、`Al-Mughera Bin Shoubah 101`、`Al-Muhammadiyah 501`、`Al-Muraba 610`、`Al-Murooj 603`
- `Al-Muthanna Bin Haritha 501`、`Al-Nuzha 602`、`Al-Qalah 101`、`Al-Qalm 301`、`Al-Qitar 101`
- `Al-Qubbah 101`、`Al-Quds 606`、`Al-Wisham 604`、`Al-Wizarat 501`、`Al-Yamamah 18 A`
- `Al-Yaqout 301`、`Al-Yarmuk 501`、`Al-Yasmeen 501`、`Albaraa Bin A'Zib 401`、`Ali Ibn Abi Talib 17 A`
- `Amr Bin Al Aas 307`、`An-Nahdah 501`、`An-Nasar 402`、`An-Nasiriyah 601`、`Anas Bin Malik 405`
- `Antakia 302`、`Aous Bin Awf 402`、`Ar Radadi 301`、`Ar-Raed 501`、`Ar-Rafeah 502`
- `Ar-Rawabi 501`、`Ar-Rawdah 504`、`Ar-Rayan 501`、`Ar-Rimayah 502`、`Arfat 412`
- `As-Salam 501`、`As-Sulaimanyah 509`、`As-Suwaidi Al-Gharbi 02 A`、`As-Suwaidi Al-Gharbi 601`、`Ash Shabab 101`
- `Ash-Sharafiyah 604`、`Ash-Shifa 603`、`Ash-Shomaisi 602`、`Asma Bint Abi Bakr 105`、`Assir 202`
- `At Tahliyah 401`、`At Termithi 103`、`At-Tuyur 202`、`Ath Thumamah 310`、`Ayesha Bint Abi Bakr 114`
- `Az Zahrawi 404`、`Az-Zahra 501`、`Badiuzaman Al Hamadhani 101`、`Bilal Bin Rabah 418`、`Bishr Bin Safwan 403`
- `Dabas Bin Rashed 301`、`Dahrat Al-Badeah 501`、`Dahrat Laban 602`、`Dhafar 201`、`Dirab 10 A`
- `Dirab 413`、`East An-Naseem 504`、`Eastern Ring 204`、`Esmail Sabri 201`、`Fatima Al Zahraa 201`
- `Fatimah Valley 102`、`GPYW Indoor Stadium`、`Hafsa Bint Omar 413`、`Hail 206`、`Hajr Valley 401`
- `Hamad Bin Salama 201`、`Hamza Bin Abdulmuttalib 03 A`、`Haroon Rasheed 208`、`Hassan Bin Al Nouman 206`、`Hassan Bin Thabit 205`
- `Hisham Bin Abdulmalik 301`、`Hudaifa Bin Al Yaman 401`、`Ibn Al Ameed 301`、`Ibn Al Muqaffa 201`、`Ibn Al Talaa 201`
- `Ibn Taymeeyah 304`、`Ibn Tulun 101`、`Ibn-'Amir 303`、`Ibn-Hajar 302`、`Ibn-Hazam 301`
- `Ibnul-Haitham 101`、`Imam Abi Hanifah 304`、`Imam Malik 101`、`Imam Muslim 203`、`Imam Shafiee 101`
- `Irqah 609`、`Ishbiliyah 602`、`Istanboul 101`、`Jaber Bin Abdullah 201`、`Jarir 406`
- `Jeddah 302`、`Jeham 401`、`Jubail 101`、`KAFD`、`KAFD (Riyadh Metro)`
- `KAFD Grand Mosque`、`KAFD World Trade Center`、`Khadijah Bint Khuwailid 301`、`Khalid Bin Al Walid 02 A`、`Khalid Bin Al Walid 214`
- `Khalid Bin Yazid 301`、`Khurais 301`、`King Abdulaziz 03 A`、`King Abdulaziz 215`、`King Abdullah 310`
- `King Fahd 201`、`King Salman 301`、`KSU 503`、`Laban 501`、`Ma'Ahad Al Idarah 301`
- `Madina Munwarah 414`、`Makkah Al Mukarramah 303`、`Mali 301`、`Manfouha 301`、`Manfuha 502`
- `Marat 402`、`Middle Al-Uraija 609`、`Mohammed Al Anbari 202`、`Mohammed Al Barqi 302`、`Mohammed Bin Ayman 301`
- `Mohammed Bin Hindi 301`、`Mohammed Bin Sh'Ail 301`、`Musa Bin Nusair 406`、`Nahawand 301`、`Nasr Bin Sayyar 201`
- `Nasseriya 303`、`New Industrial City 501`、`North Al-Mathar 502`、`Northern Ring Road (Route 65)`、`Olaya 218`
- `Omar Bin Abdulaziz 301`、`Omar Bin Al Khatab 302`、`Orwa Bin Al Zubayr 401`、`Qotaiba Bin Muslim 406`、`Qurtubah 501`
- `Radwa 301`、`Radwan 401`、`Rahwan 201`、`Rajia 101`、`Raya 301`
- `Riyadh TV Tower`、`Rodh Al Jawa 401`、`Sadir 405`、`Saeed Bin Zaid 101`、`Safwan Bin Omaya 101`
- `Sahara Al Maghrabiyah 101`、`Salahuddin Al Ayubi 108`、`Salahuddin Al Ayubi 11 A`、`Salam 101`、`Saleet Bin Qeis 401`
- `Salman Al Farsi 207`、`SAMA Money Museum`、`Sari 101`、`Saybawiyyah 201`、`Shagra 401`
- `Shiba Bin Nassah 101`、`Siraj Bin Abdulmalik 301`、`Southern Ring 401`、`Southern Ring Road (Route 50)`、`STC`
- `STC C`、`Sufian Bin Owf 301`、`Sultanah 206`、`Suwaidi Al Am 410`、`Taif 301`
- `Takhassusi 220`、`Taybah 602`、`Tima 101`、`Turki Bin Madhi 201`、`Umm Al Hamam 208`
- `Unayzah 203`、`Utayqah 17 A`、`Utayqah 616`、`West Al-Uraija 602`、`West An-Naseem 501`
- `Western Ring 203`、`Yanbu 301`、`Zainab Al Jirjaniyah 301`、`Zubair Bin Al-'Awam 410`、`Zuhair Bin Qays 402`

### 只进入 navigation（245 词）

- `'Otayyigah`、`Abu Bakr Al Siddiq Road`、`Abu Dhabi Square A`、`Ad Dar Al Baida`、`Ad Difa'`
- `Ad-Dirah`、`Airport Road`、`Al Ahmadi Governorate`、`Al Arouba Road`、`Al Asimah Governorate`
- `Al Farwaniyah Governorate`、`Al Fazari Station`、`Al Hamra Tower`、`Al Iskan`、`Al Jahra`
- `Al Jahra Governorate`、`Al Khalidiya Park`、`Al Kindi Station`、`Al Maigliah Market Center`、`Al Majdoul Tower`
- `Al Malqa Road`、`Al Manakh Road`、`Al Masmak Palace`、`Al Nakheel Tower (Riyadh)`、`Al Oud cemetery`
- `Al Rajhi Grand Mosque`、`Al Salam Palace`、`Al Shaheed Park`、`Al Suwaidi Park`、`Al Thumairi Gate`
- `Al Tijaria Tower`、`Al Yamamah Palace`、`Al-'Oud`、`Al-Andalus`、`Al-Aziziyah`
- `Al-Badi'ah`、`Al-Ghadir`、`Al-Ghannamiyah`、`Al-Hamra`、`Al-Hayir`
- `Al-Kharj Road`、`Al-Khozama`、`Al-Ma'athar`、`Al-Ma'izliyyah`、`Al-Malaz`
- `Al-Manakh`、`Al-Manar`、`Al-Mansorah`、`Al-Mansurah`、`Al-Mansuriyya`
- `Al-Margab`、`Al-Marwah`、`Al-Masani'`、`Al-Masif`、`Al-Mugharrazat`
- `Al-Muhammadiyya`、`Al-Murooj`、`Al-Nadheem`、`Al-Nafil`、`Al-Nahdhah`
- `Al-Nakheel`、`Al-Naseem (West)`、`Al-Nasriyyah`、`Al-Olaya`、`Al-Olayya`
- `Al-Qadisiyah`、`Al-Qayrawan`、`Al-Qibli Mosque (Riyadh)`、`Al-Qouds`、`Al-Ra'id`
- `Al-Rabwah`、`Al-Rahmaniyya`、`Al-Rayyan`、`Al-Rimayah`、`Al-Sa'adah`
- `Al-Sahafa`、`Al-Seif Palace`、`Al-Shabab Club Stadium`、`Al-Shifa`、`Al-Sulaymaniyyah`
- `Al-Suwaidi`、`Al-Suwaidi (West)`、`Al-Urayja`、`Al-Urayja Al-Wusta`、`Al-Watan Park`
- `Al-Wurood`、`Al-Yamamah`、`Alfaisal University`、`Alfaisaliah Road`、`Ali Ibn Abi Talib D`
- `Alrajhi Bank Tower`、`Arab Open University`、`As-Salam`、`Ash-Shumaisi`、`Assima Mall`
- `Bait Al-Othman Museum`、`Bayan Palace`、`Boulevard City`、`Burj Al Anoud`、`Burj al-Malaz`
- `Burj Mubarak Al Kabir`、`Burj Rafal`、`Central Bank of Kuwait`、`Central Blood Bank`、`City of 'Ar'ar`
- `Dahrat Al-Badi'ah`、`Dakhnah Gate`、`Dar al Athar al Islamiyyah`、`Deera Square`、`Deerab`
- `Dharat Laban`、`Dharat Nemar`、`Digital City (Riyadh)`、`Eastern Ring Road`、`Ein Heet Cave`
- `Eleyshah`、`Gharnatah`、`Grand Mosque of Kuwait`、`Hamad Tower`、`Hawalli Governorate`
- `Hazm`、`Holy Family Co-Cathedral, Kuwait`、`Imam Abdulrahman bin Faisal Road`、`Imam Baqir Mosque`、`Imam Hussein Mosque (Kuwait)`
- `Imam Mohammad Ibn Saud Islamic University`、`Imam Turki bin Abdullah Grand Mosque`、`Imam Turki bin Abdullah Road`、`InterContinental Kuwait Downtown`、`Irqah`
- `Irqah Hospital`、`Jabrah`、`Jarir`、`Khalid Bin Alwaleed Road A`、`King Abdulaziz International Airport`
- `King Abdulaziz Medical City`、`King Abdulaziz Public Library`、`King Abdulaziz Racetrack`、`King Abdulaziz Road`、`King Abdullah bin Abdulaziz International Airport`
- `King Abdullah Financial District`、`King Abdullah Medical Road`、`King Abdullah Park`、`King Faisal Road`、`King Faisal Specialist Hospital and Research Centre`
- `King Khalid Airport Mosque`、`King Khalid Grand Mosque`、`King Khalid International Airport`、`King Khalid Medical Road`、`King Khalid Royal Reserve`
- `King Salman Oasis`、`King Saud Medical City`、`King Saud University`、`King Saud University Stadium`、`Kingdom Arena`
- `Kingdom Centre`、`Kuwait International Airport`、`Kuwait National Assembly Building`、`Kuwait National Cultural District`、`Kuwait National Museum`
- `Kuwait Science and Natural History Museum`、`Kuwait Scientific Center`、`Liberation Tower (Kuwait)`、`Makkah Road (Mecca Rd)`、`Manfuha`
- `Manfuha Al-Jadidah`、`Ministry of Interior building`、`Miqal`、`Mohammed Abdo Arena`、`Mubarak Al-Kabeer Governorate`
- `Murabba Palace`、`Museum of Modern Art (Kuwait)`、`Naila Art Gallery`、`Nasiriyah Gate`、`National Museum C`
- `National Museum of Saudi Arabia`、`National Museum Park`、`Nemar`、`Neom Bay Airport`、`Northern Ring Road`
- `Olaya Towers`、`Prince Abdul Mohsin Bin Abdulaziz International Airport`、`Prince Faisal bin Fahd Sports City Stadium`、`Prince Mohammad bin Abdulaziz International Airport`、`Prince Mohammed bin Abdulaziz Rd`
- `Prince Sattam bin Abdulaziz Road`、`Prince Sultan bin Abdulaziz Airport`、`Prince Sultan University`、`Prince Turki bin Abdul Aziz Stadium`、`Public Investment Fund Tower`
- `Qaṣr al-Badi'a`、`Qaṣr al-Shamsīyya`、`Qiddiya Road`、`Qortubah`、`Quincy House, Riyadh`
- `Ritz-Carlton, Riyadh`、`Riyadh city fortifications`、`Riyadh Dry Port`、`Riyadh International Convention and Exhibition Center`、`Riyadh North Station`
- `Riyadh Park`、`Riyadh railway station`、`Riyadh Valley Co`、`Riyadh Water Tower`、`Riyadh Zoo`
- `Riyadh-Al Kharj Highway`、`Riyadh-Dammam Highway`、`Riyadh-Makkah Expressway`、`Roshn Front`、`Royal Saudi Air Force Museum`
- `Sabah Al Ahmad Sea City`、`Sadu House`、`Safat Clocktower`、`Sahara Mall`、`Salam Park`
- `Shalaga cemetery`、`Sheikh Abdullah Al-Salem Cultural Centre`、`Sheikh Jaber Al-Ahmad Cultural Centre`、`Sheikh Muhammad bin Ibrahim Mosque`、`Shubra`
- `Sinaiyah al-Qadimah`、`Souk Sharq`、`Souq Al-Mubarakiya`、`Souq al-Thumairi`、`Souq al-Zal`
- `Southern Ring Road`、`Square Capital Tower`、`State of Kuwait`、`Sultanah`、`Swaigah Trade Center`
- `Syah`、`Tahlia Street (Prince Mohammed Rd)`、`Taif International Airport`、`Tareq Rajab Museum`、`Taybah`
- `Transportation Center`、`Tuwaiq`、`Tuwaiq Palace`、`Umm Al-Hamam (West)`、`Umm Sulaim`
- `Umm Tina (well)`、`Uraydh`、`Villas in the Sky`、`Wadi Laban Road`、`Wahat al-Nakheel`

附加组的完整实测范围为附录第一张清单的 280 词，加上智慧家庭的 `FTTR`、`PM2.5`、`Ne Zha 2`、`ORVIBO`。
