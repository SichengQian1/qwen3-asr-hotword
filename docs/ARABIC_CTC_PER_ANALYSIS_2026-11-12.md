# 阿语 CTC PER 0.17 成因分析：标注链路与数据质量排查

日期：2026-11-12。核对的源码：分支 `codex/g2p-coverage-scan`，提交 `c6a02b5db89b66c9bb8df206092667cd9e934528`。
配套文档：[阿语打标与 Manifest 制作交接](ARABIC_NATIVE_LABELING_HANDOFF_2026-11-12.md)、[HANDOFF.md](HANDOFF.md) 0.216。

## 1. 结论摘要

PER 0.17 指 r15 Encoder 下阿语验证集的 17.60%（33,454/190,095），同一个 Head 上英语是 4.72%。核对仓库代码后，结论是：误差以阿语特有的问题为主，没有让整批标签作废的单点 bug，但有两处确定的代码缺陷。

- **两处代码缺陷（仓库代码实测）**
    1. EGY 规则不认 CALIMA 输出里的附着词界 `_`，含 `_` 的整句会被判为 `unsupported_Arabic_letter` 拒收。用真实埃及口语文本按仓库流程跑，通过词法分析的句子里有 27% 因此被丢，主要是 ما…ش 否定句和与格附着形式。
    2. MSA 分支把 CALIMA 的词尾长元音写法 `i_y`/`u_w` 映射成了 `i j`/`u w`：في 变成 `f i j`，الذي 变成 `ʔ a lː a ð i j`。这是 v2 y 修复的副作用，且落在最高频的词上。
- **标签约定与口语系统性不符（代码与 CALIMA 实测）**
    - MSA：格尾、tanwin、ة 的 t、句中 hamzat wasl，合计约占标签单位的 19%。MLE 选的格尾对照标准形式只对 64.9%。
    - EGY：eː/oː 被标成 a j / a w，ث/ذ/ظ 标成 θ/ð/ðˤ，词尾 ه 标成 h，句中冠词一律标成 ʔ i，合计约占 9%。
- **MGB-2 的转写不是逐字的（官方文献）**：代码确认是按文件顺序填满的，所以补样训练集和 MSA 验证集大概率都来自 MGB-2。实际来源分布待工区确认。
- **模型侧不能排除**
    - 英语 4.72% 看上去排除了 Head 的问题，但英语验证集约每秒 3.9 个目标（由缓存字节数推算），阿语是 13.2 个。
    - 阿语每个 80ms 编码帧平均要出 1.06 个音素，而 Head 只是把每帧复制两份，再过一个 k=5 的逐通道卷积。
    - 把 Encoder 从基模换成 r15，阿语 PER 降了 4.8 个点，说明表征也在起作用。
    - 训练在线 PER 为 14.41%（英阿按音素混算），按音素量估算阿语训练 PER 约 18%，与验证集相当。这说明问题不是过拟合，而是连训练集都拟合不上（推断）。

下一步分三件事。第一，立即修掉两处代码缺陷（第 6 节 P0）。第二，用现有验证解码做 D0，按组、来源、语速拆分 PER；并做 D3，检查文本是否与录音一致。第三，用已有缓存把 Head 上采样改成 ×3 重训一次（D8，约 3.5 小时），确认模型侧占多少。

## 2. 先校准：0.17 衡量的是什么

0.17 是模型输出与**自动标签**之间的编辑距离，不是与真实读音之间的距离。它由三部分叠加而成，单看总数判断不了是哪一环出了问题，必须拆开。

| 误差来源 | 机制 | 能否被模型学会 | 在 PER 上的表现 |
| --- | --- | --- | --- |
| 模型误差 | 声学难分、帧率限制、Head 容量 | — | 集中在短元音、长短对立等难分类别 |
| 标签约定误差 | 前端给的音素和实际读法系统性不同（格尾、词首 hamzat wasl、开罗 eː/oː 等） | 部分能：同一书写词的标签固定，带上下文的编码器能“背”下一部分 | 集中在词尾、词首、元音位置；逐句 PER 分布整体右移 |
| 文本-音频不符（数据 WER） | 转写漏词、改写、切分截断 | 不能：对模型是随机的 | 逐句 PER 出现长尾，少数句子 PER > 0.5 |

**第二类误差的隐患不只在 PER 上**：模型若学会了把开罗的 [eː] 输出成 `a j`、把句中不发音的 hamza 也输出，PER 会好看一些，但音素类别的含义已被污染。热词检索用的查询音素如果来自不同路径（英语前端、人工拼写、另一套规则），就会对不上。

### 需要的参照

- **英语基线（HANDOFF 0.216 已记录）**：同一个 Head、同一轮训练下，英语验证 PER 为 4.72%（r15）和 5.94%（基模），阿语为 17.60% 和 22.36%。差距确实是阿语特有的。
- **但英语基线覆盖不了音素密度**：英语验证缓存 385,107,681 字节，特征为 bf16 [T,1024]，折合约 18.8 万个编码帧、约 4.2h。参考单位 58,742 个，即每秒约 3.9 个目标；阿语是 13.2 个。也就是说，英语每个编码帧平均对应 0.31 个音素，阿语是 1.06 个。
- **Head 的时间分辨率（代码已核对）**：`TemporalUpsampleCtcHead` 先把每个编码帧 `repeat_interleave` 成两份，再经过 1×1 投影、k=5 的逐通道卷积和线性输出，共 87.8 万参数。同一帧复制出的两步要输出不同音素，只能靠这个卷积。阿语平均每帧就要出一个音素，语速快时会碰到这个上限；英语基本碰不到。
- **Encoder 也在起作用**：换 Encoder 后阿语降了 4.8 个点、英语降了 1.2 个点。AuT 编码器预训练以中英为主（[Qwen3-ASR 技术报告](https://arxiv.org/pdf/2601.21337)）。HANDOFF 也注明，r15 是否见过这批验证音频未知。
- **训练集也拟合不上（推断）**：训练在线 PER 14.41% 是英阿按音素微平均（`sharded_ctc.py` 已核对）。英语约 800 万个目标、阿语约 2,270 万个，假设英语训练 PER 约 4.7%，反推阿语训练 PER 约 18%，与验证集相当。这是拟合下限，不是泛化差距：要么标签里有学不会的东西，要么模型容量不够。
- **公开参照**：TIMIT 人工音素标注的朗读英语，wav2vec 2.0 LARGE 测试 PER 8.3%（[wav2vec 2.0](https://arxiv.org/pdf/2006.11477) 表 3）。
- **要按组报 PER**：EGY 和 MSA 两个验证集分开报。两组的标签体系和语料来源都不同，合在一起的 17.60% 会把问题平均掉。

## 3. 实测：用固定版本 CALIMA 和仓库前端复现

实测确认：两组标签都不是“读音”，而是书面形态。EGY 的 diac 是 CODA 拼写，MSA 的 CAPHI 是全屈折、逐词、不看上下文的书面体。在此之外还有两处代码缺陷：EGY 不认 `_`，MSA 词尾 `i_y`/`u_w` 变成假辅音。

**复现环境**：camel-tools 1.6.0，四个数据包版本与交接文档相同（EGY 词法库 0.2.0、MSA 词法库 0.4.0、EGY MLE 0.1.0、MSA MLE 0.2.5），解包后文件 SHA 与官方 catalogue 一致。前端代码取自 c6a02b5，`src/qwen_hotword/phonemes/arabic_native.py` 的 SHA 为 `686394f8…`，与交接文档中“当前正确后端”一致；词表 SHA 也一致。下表“标签”一列是直接运行仓库里的 `egyptian_units()` 和 `caphi_units()` 得到的。对交接文档 §6 的探针句，diac 和 IPA 与文档逐词相同。

### 埃及组：CODA 拼写被当成了读音

- **EGY 词法库没有 CAPHI**（测过：所有分析的 `caphi` 字段为空）。EGY 只能走 diac 加规则，“改用原生 CAPHI”不可行。
- CODA 用 fatha 加无标记的 ي/و 表示开罗话的 eː/oː，用 MSA 同源字母写 ث/ذ/ظ（[CODA* 指南](https://camel-guidelines.readthedocs.io/en/latest/orthography/)）。更麻烦的是，转写者按读音写的 ت/د，会被词法分析的拼写变体匹配改回 ث/ذ。

| 转写 | CALIMA-EGY Top1 diac（实测） | 仓库规则输出的标签（实测） | 开罗实际读音 | 问题 |
| --- | --- | --- | --- | --- |
| بيت | بَيت | b a j t | beːt | eː 被拆成 a j |
| يوم | يَوم | j a w m | joːm | oː 被拆成 a w |
| ليه / إيه | لَيه / أَيه | l a j h / ʔ a j h | leːh / ʔeːh | 同上，极高频 |
| البيت（句中） | البَيت | ʔ i l b a j t | (fi) lbeːt | 句中冠词也标 ʔ i |
| عشان | عَشَأْن | ʕ a ʃ a ʔ n | ʕaʃaːn | 拼写变体引入 ʔ，长元音变短 |
| ياخد | ياخُذ | j aː x u ð | jaːxud | 转写者写的 د 被改回 ذ |
| أكتر | أَكْثَر | ʔ a k θ a r | ʔaktar | ت 被改回 ث |
| ثورة / ظهرك | ثَورَة / ظَهْرَك | θ a w r a / ðˤ a h r a k | sawra / zˤahrak | 开罗话没有齿间音 |
| ده / شفته | دَه / شُفتُه | d a h / ʃ u f t u h | da / ʃuftu | 词尾 ه 不发音 |
| يعني / في | يَعْنِي / فِي | j a ʕ n iː / f iː | jaʕni / fi | 开罗话词尾长元音读短 |
| مبقاش | ما_بَقاش | 报错 `unsupported_Arabic_letter:_`，整句拒收 | mabʔaːʃ | 否定句被系统性丢弃 |

用公开的埃及口语文本（[EDC](https://github.com/TaghreedT/EDC)，Facebook 语料）随机抽 4,000 句，共 66,734 个可分析的词，结果如下。文体与厂商转写不同，只作量级参考。

- **`_` 导致整句拒收（代码确认）**：30.6% 的句子至少有一个词的 diac 含 `_`，即否定 ما_…ش、与格 لـ 等附着词界。`_` 不在 `egyptian_units()` 的字母表里，会抛出 `unsupported_Arabic_letter:_`。按仓库流程完整跑这 3,941 句：25.1% 因 `no_analysis` 被丢（工区 EGY 是 26%，量级吻合），通过词法分析的句子里又有 27.1% 因 `_` 被丢。工区 EGY 的 8,208 条 `unsupported_Arabic_letter` 约占通过分析句子的 16%，应当大部分就是它，可用附录 A 的 grep 确认。
- **句中冠词**：规则对词首 ا 一律输出 ʔ 加元音（ال 前用 i），不区分是否在句首。保留下来的句子里，9.6% 的词是句中冠词，占标签单位的 3.5%。
- 0.46% 的词（310 个）被拼写变体改写了辅音：ت→ث 134、د→ذ 103、س→ث 37、ض→ظ 22、ز→ذ 14。
- 也见到整词选错：حضن（拥抱）被分析成 حَأَظُنّ（我将认为），الفضه 被分析成 أَلْفِظه。

### 补样组：全屈折书面体，与广播口语不符

在 [UD Arabic-PADT](https://github.com/UniversalDependencies/UD_Arabic-PADT) 测试集上实测：680 句新闻、共 21,337 个词，该树库带人工元音化的标准形式。

- **格尾选错三分之一**：10,229 个名词/形容词中，MLE Top1 的词尾格位与标准形式一致的只有 64.9%。最常见的错误是偏向属格 -i：本应 -a 却选 -i 774 次，本应 -u 却选 -i 646 次。就算说话人完整读格尾，标签也有三分之一不对；而广播口语常用停顿形，根本不读格尾。
- **每个 ال 词都以 `2_a` 开头**：句中的 hamzat wasl 在连读里不发 ʔ。في المؤتمر 的标签是 `f_i_y 2_a_l_m_u_2_t_a_m_a_r_i`，实际读 [fi lmuʔtamar]。
- **词尾长元音的假辅音**：CALIMA-MSA 把词尾 ـِي 写成 `i_y`（في `f_i_y`、الذي `…dh_i_y`、يعني `y_a_3_n_i_y`），词尾 ـُو 写成 `u_w`（أبو `2_a_b_u_w`），词内长元音则是 `ii`（الرئيس `2_ii_s`）。仓库 `caphi_units()` 实测：في 输出 `f i j`，الذي 输出 `ʔ a lː a ð i j`，أبو 输出 `ʔ a b u w`。也就是说，v2 的 y 修复把这些假 y 变成了假 j。
- **高频词整词选错**：بين（在…之间）的 Top1 是 بَيِّنَ（bayyina，“明显的”），标签 `b_a_y_y_i_n_a`，实际读 [bajna]。分析器给出的候选概率相同，MLE 选了错的那个。

举一句新闻体来看量级。قال الرئيس في المؤتمر 的标签共 27 个单位：q aː l a | ʔ a rː a ʔ iː s i | f i j | ʔ a l m u ʔ t a m a r i。典型播音读作 [q aː l a rː a ʔ iː s f i l m u ʔ t a m a r]，共 20 个。两者编辑距离为 7，占标签的 26%。

## 4. 成因清单与估算贡献

17.60% 最可能是几类问题叠加的结果，而不是某一个环节坏了。补样组的书面体标签暴露面最大，约占标签单位的 19%；MGB-2 的非逐字转写可能是最大的单项；埃及组的读音偏差约占 9%。还有一项阿语独有的模型侧压力：阿语的音素密度是英语的 3.4 倍。

“暴露”是标签里处在易错位置的单位占比，不等于 PER 贡献。标签对同一书写词固定不变，模型会背下一部分，例如学会在 ال 前输出 ʔ a。下表的贡献区间按“暴露 × 口语中不符的比例 × 没背下的比例”估算。第三项未知，所以只给区间，要靠第 5 节的诊断确认。

| # | 成因 | 组 | 实测暴露（占标签单位） | 对 PER 的可能贡献 | 证据强度 | 确认方法 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 格尾、tanwin、ة 的 t：全屈折，且三分之一选错 | MSA | 10.5%（格元音 6.1%、tanwin 2.3%、ة-t 2.1%） | 3–7 个点 | 暴露测过；贡献推断 | 折叠 PER（F4）；词尾错误占比 |
| 2 | 句中 hamzat wasl 标成 ʔ a | MSA | 7.3% | 0–4 个点 | 暴露测过；贡献推断 | 删除错误里 ʔ 的占比；F3 |
| 3 | 词尾 `i_y` / `u_w` 变成假 j / w（代码缺陷） | MSA | 0.9% | ≤ 1 个点 | 代码实测 | j 的插入率 |
| 4 | MGB-2 非逐字转写、对齐插值、未标注的外语配音 | MSA | 未测；官方对齐中 37% 的词没有锚点匹配 | 可能是最大单项 | 先验；按文件顺序填充已由代码确认 | D3 |
| 5 | 方言语段被按 MSA 标音 | MSA | MGB-2 约 30% 是方言 | 1–3 个点 | 先验 | 按节目分组报 PER |
| 6 | eː/oː 被拆成 a j / a w | EGY | 1.7%（涉及 4.7% 的词） | 1–2 个点 | 代码实测 | j/w 插入率 |
| 7 | 句中冠词标成 ʔ i | EGY | 3.5% | 0–2 个点 | 代码实测 | ʔ 删除占比 |
| 8 | 词尾长元音，开罗话读短 | EGY | 2.4%（涉及 12.7% 的词） | 0.5–2 个点 | 标签为代码实测；读音为先验 | i/iː、u/uː、a/aː 混淆 |
| 9 | ث/ذ/ظ 标成 θ/ð/ðˤ、词尾 ه 标成 h、عشان 等 | EGY | 约 1.4% | 0.5–1 个点 | 代码实测 | 混淆矩阵 |
| 10 | `_` 导致整句拒收（代码缺陷） | EGY | 不进标签；可分析句子中约 16–27% 被丢 | 不直接进 PER；EGY 训练集缺否定句 | 代码实测 | review 的 grep |
| 11 | 模型侧：阿语每帧约 1.06 个音素，Head 仅复制帧加 k=5 卷积；换 Encoder 降 4.8 点 | 两组 | 阿语每秒 13.2 个目标，英语约 3.9 个 | 未知，可能有数个点 | 代码与训练日志已核对 | D8：PER 随语速变化的曲线；×3 上采样重训 |
| 12 | 评测口径 | 两组 | — | 只改变读数 | 代码已核对：阿语是音素微平均，选模用英阿宏平均 | 分 EGY / MSA 报 |

暴露数字的来源：MSA 来自 PADT 测试集（159,352 个标签单位），EGY 来自 EDC 抽样（约 36 万个单位，分母是近似值）。工区自己的标签可以用附录 A 的 `label_exposure.py` 精确重算。

第 4 项的依据来自 [MGB-2 官方论文](https://ar5iv.labs.arxiv.org/html/1609.05625)：

- 转写不是逐字的，可能包含改写、去重复和概括。
- 没匹配上的词用线性插值定时，所以段边界可能切在别人的话上。
- 英语、法语配音没有在转写中标出来。
- 除去约 300h 短片外，其余 1,200h 未做进一步过滤。

工区这份 MGB2 JSON 只有 290,477 段、928.97h，而官方是超过 55 万段、1,200h。说明它经过了二次切分或筛选，标准未知，需要问数据提供方。

填充顺序已由代码确认。`scripts/run_en_ar_training.py` 的 `freeze()` 按 inventory 顺序逐条遍历：EGY 在前，补样池按 ABN→MGB2→biaobei→LDC 顺序。训练达到 480h、验证达到各 2h 以后就不再收录。只要 MGB-2 的就绪量够补约 370h，biaobei 和 LDC 就一条都进不了训练。MSA 验证集也来自 ABN 和 MGB-2 文件开头。这样一来，MSA 的 PER 几乎完全是在转写质量最差的那个来源上测的。实际分布用附录 A 的来源分布命令确认。

第 11 项的细节见第 2 节「需要的参照」。英语 4.72% 说明 Head 在低密度下没问题，但证明不了它在阿语的密度下也没问题。训练器 `scripts/train_full_ctc.py` 已有 `--head-time-upsampling-factor` 参数；英阿编排脚本只需透传这个参数，就能在现有缓存上重训，不必重新提特征。

## 5. 诊断方案

九项检查里，D5（前端代码）和 D1（英语基线）已借仓库完成。剩下的里，D0、D3、D6 都是只读的，一天内能做完；D8 需要约 3.5 小时 GPU。脚本见附录 A，已用合成数据和公开语料跑通。

1. **D5 前端代码核对（已完成）**：用仓库的 `egyptian_units()` 和 `caphi_units()` 实际跑了探针，结果见第 3 节表格。剩下一件事：用附录 A 的 grep 统计 EGY 的 8,208 条 `unsupported_Arabic_letter` 里 `_` 占多少。
2. **D0 拆分 PER（用现有验证解码结果）**：按组、按来源文件报微平均 PER，给出宏平均、替换/删除/插入占比、逐句 PER 分位数和前 20 种错误对。带上时长字段还会按语速分桶。脚本 `per_breakdown.py`。
3. **D1 英语基线（已取得）**：HANDOFF 0.216 记录英语为 4.72%。不足之处是英语音素密度只有阿语的三分之一左右，由 D8 补。
4. **D6 标签暴露统计**：在工区真实的 `labels.jsonl` 上，只统计进入 `train.jsonl` 的行，重算第 4 节的暴露占比。脚本 `label_exposure.py`，只读文本。
5. **D3 文本一致性**：用基座 Qwen3-ASR-1.7B（不加热词）解码 1,456 条验证集，再从每个训练来源各抽 300 条。规范化后算 CER，并和逐句音素 PER 对照。脚本 `text_vs_asr.py`，约需 1 卡时。
6. **D2 折叠 PER**：对 hyp 和 ref 同时做三种折叠后重算，分别是 F1 合并元音长短、F2 合并双辅音、F3 去掉 ʔ。`per_breakdown.py` 已包含。词尾折叠（F4，去掉词尾短元音和 tanwin 的 n）需要词边界，可以对验证集定点重放前端拿到。
7. **D4 干净参照集**：从 biaobei（朗读）和 LDC 中抽约 1h 未进入训练的句子测 PER。先用来源分布命令确认这两个来源不在 train 中。
8. **D8 模型侧确认**：先看 D0 的语速分桶，高语速桶 PER 明显更高就是分辨率瓶颈的信号。再用现有缓存把 Head 上采样改成 ×3 重训一次，按 r15 的记录约需 3.5 小时。阿语降、英语不变，就能量出模型侧的份额。
9. **人工抽检（约 3 小时母语者时间）**：每组 100 条，其中 50 条取逐句 PER 最高的尾部、50 条随机。标注两件事：文本是否与录音一致；另抽 20 条逐音素核对标签。这是得到“标签真实错误率”的唯一办法。

### 怎么判读

| 观察到 | 说明 | 下一步 |
| --- | --- | --- |
| D8 中 ×3 上采样让阿语明显下降，或高语速桶 PER 明显更高 | Head 的时间分辨率是阿语的一个主因 | 采用更高倍数，并可考虑加大 Head |
| MSA 显著高于 EGY | 问题在补样组的标签或数据 | 看下面两行区分是哪一个 |
| 逐句 PER > 0.4 的句子超过 5%，集中在 MGB2；D3 中 PER 随 CER 陡升 | 文本与录音不符是主因 | 按 CER 或 CTC loss 过滤 MGB2，并重建验证集 |
| 前几种错误对是 ʔ 删除、词尾元音删除、i 后插入 j；F3/F4 折叠后降幅大 | 标签约定是主因 | 按第 6 节改 MSA 标签规则 |
| F1/F2 折叠后降幅大 | 长短对立：可能是 EGY 词尾长元音约定，也可能是分辨率 | 对照 D8；按第 6 节改 EGY 规则 |
| 干净参照集的 PER 明显低于 MGB2 验证集 | 数据质量是主因，而且参照集还是跨域的 | 换数据来源，按来源分层抽样 |

## 6. 修复建议与优先级

有三件事不用等诊断就能做：两处代码缺陷已经确认，直接修；×3 上采样重训只是改一个参数，成本低。其余按诊断结果排序。任何改动都会改变标签或划分，按交接文档 §14 用新的实验身份和新输出目录，保留 v2。行号指 c6a02b5 版本的 `src/qwen_hotword/phonemes/arabic_native.py`。

| 优先级 | 动作 | 代码位置 | 针对成因 | 成本 | 前提 |
| --- | --- | --- | --- | --- | --- |
| P0 | 把 `_` 当作词内连接跳过，和 tatweel 一样处理，不再整句拒收。ما_ 读短 ma、附着处元音省略可以之后细化 | `egyptian_units()` 第 107 行，跳过 `ـ` 的分支 | #10 | 一行代码，EGY 重标 | 无 |
| P0 | 映射前把词尾 `i_y` 改为 `ii`、`u_w` 改为 `uu`。后面还有元音时不动，如 هي `h_i_y_a` 仍为 hija | `caphi_units()` 第 95 行 `split` 之后 | #3 | 两行代码。标签里没存 CAPHI，需要重放 MSA 前端，可参照 v2 修复的并行入口 | 无 |
| P1 | Head 上采样改成 ×3，在现有缓存上重训 | `train_full_ctc.py --head-time-upsampling-factor`，需要英阿编排透传 | #11 | 约 3.5 小时 GPU | 无 |
| P1 | 高频词例外词典：两组各取前 300 个高频词，由母语者写出口语读音，优先于规则 | `CalimaFrontend.label()` 取 diac 之后 | #1 #2 #6–#9，以及 بين 这类整词选错 | 母语者约 1 天 | 前 300 词覆盖 EDC 词次的 42.8%、PADT 的 35.4%（实测） |
| P1 | 句中冠词不标 ʔ；前一词以元音结尾时，连冠词元音一起去掉。两组都改 | `label()` 的逐词循环，第 207–217 行，需要传入词位置 | #2 #7 | 小 | D0 显示 ʔ 删除占大头 |
| P1 | MSA 改用播音口语形式：去掉名词的格元音和 tanwin，ة 读 a | `label()` 中用分析的 `cas` 字段处理 | #1 | 中；热词查询端必须同步改 | D0/D2 显示词尾错误占大头 |
| P1 | MGB2 过滤：用 D3 的 CER，或已训 Head 的逐句 CTC loss，去掉最差的 10–20%；去掉外语配音段 | 新的筛选步骤 | #4 | 中 | D3 显示 PER 随 CER 陡升 |
| P1 | 补样池按来源分层抽样，不再按文件顺序填满；验证集按节目或文件整体留出，四个来源各有代表；另设独立 test | `scripts/run_en_ar_training.py` 的 `freeze()` 第 254–341 行 | #4 #12 | 小 | 来源分布命令的结果 |
| P2 | EGY：无 sukun 的 ـَي/ـَو → eː/oː（ثورة 这类读 aw 的词交给例外词典）；词尾长元音读短；词尾代词 ـُه/ـِه 和 ده、كده 去掉 h | `egyptian_units()` 第 134–142 行 | #6 #8 #9 | 小 | 无 |
| P2 | 拼写变体改写的辅音以转写者原字母为准；转写里保留的 ث/ذ/ظ 按开罗默认读 s/z/zˤ | `CONSONANTS` 第 12–55 行，并在 `label()` 中对照原词 | #9 | 小 | 无 |

两点约束：

- **训练标签和热词查询必须用同一套约定**。只改一边，PER 会变好，召回却可能变差。
- **每次只改一类，在同一验证集上对比**。但标签变了以后，新旧 PER 不再可比：要么在折叠口径下比，要么用人工核对过的小测试集比，或者直接看热词召回。

## 7. 交给 Codex 的执行方案

分两轮做。第一轮只读诊断，再加一次 ×3 重训，不改标签；工区跑一次，回传 4 个小文件。第二轮按第一轮的结果，一次性重标、重新冻结、重训，避免为每条修复各重训一次。可以直接粘贴的 Codex 指令见附录 B。

分工：Codex 在本地实现并推送；工区只需按 HANDOFF 里的命令跑，回传小文件；由 Claude 对照下面的门槛判读结果，并定稿第二轮的规格。

### 第一轮：不改标签

| WP | 内容 | 复用的现有代码 | 工区资源 | 回传 |
| --- | --- | --- | --- | --- |
| WP1 | 标签与数据统计：review 原因明细（不支持的字符、无分析的高频词、不支持的 CAPHI 单位）、train/validation 行的标签暴露、来源分布与行号范围 | 新脚本，暴露规则照搬附录 A | CPU，几十分钟 | `labels_report.json` |
| WP2 | 验证集逐样本诊断：按 `source_group` 和 `source` 分组 PER、CTC 压力分桶、Top 删除/插入/替换、逐样本 hyp/ref、F1–F3 折叠 PER、逐句 PER 分位数；另导出听审抽样表 | 扩展 `diagnose_frozen_ctc.py` / `ctc_diagnostics.py`，它们已有分组、压力分桶和 Top 错误 | GPU 4，分钟级 | `val_diagnostics.json`；逐样本文件和听审表留工区 |
| WP3 | 文本一致性：基座 Qwen3-ASR-1.7B 解码阿语验证集 1,456 条，加上每个训练来源 300 条，算 CER/WER，并与 WP2 的逐样本 PER 关联 | `casablanca_wer.load_qwen` | GPU 4，约 1 小时 | `text_agreement` 汇总 JSON |
| WP4 | Head ×3 上采样对照重训：只换倍数，按引用复用 r15 的英阿缓存 | `run_en_ar_merged.py` 新增 `retrain-head`；`train_sharded_ctc_head` 已有该参数 | GPU 4，约 4 小时 | `head/report.json` |

WP1 和 WP2 互不依赖，先跑；WP3、WP4 在同一张卡上排队。听审表交母语者，对应第 5 节第 9 项。

### 判定门槛（先定好，再看结果）

| 假设 | 支持 | 否定 | 支持时第二轮做什么 |
| --- | --- | --- | --- |
| `_` 是 EGY 不支持字符的来源 | `_` 占 `unsupported_Arabic_letter` 的 80% 以上 | 低于 50%：列出其他字符再查 | P0 修 `_`（无论结果如何都修） |
| 补样只来自 ABN 和 MGB-2 | train 中 biaobei、LDC 条数为 0 | 两者都占相当比例 | 按来源分层抽样 |
| Head 时间分辨率不够 | ×3 后阿语 PER 降 1.0 个点以上，英语变化不超过 0.3 个点 | 阿语降不到 0.5 个点 | 采用 ×3，可再试 ×4 |
| 文本与录音不符 | MSA 逐句 PER > 0.4 的句子占 5% 以上，且 CER > 0.3 桶的平均 PER 至少是 CER < 0.1 桶的 2 倍 | 长尾低于 2%，且 PER 与 CER 的相关系数低于 0.2 | 按 CER 或 CTC loss 过滤 MGB-2 |
| MSA 句中 hamzat wasl | F3 折叠后 MSA PER 降 1.0 个点以上，且 ʔ 删除进入前 5 种错误 | 降不到 0.3 个点 | 句中冠词不标 ʔ |
| 词尾与长短约定 | MSA 前 10 种错误里至少 3 种是词尾元音、tanwin 的 n 或 i 后插入 j；EGY F1 折叠后降 2 个点以上 | 都不满足 | 启用 MSA 停顿形和 EGY 词尾短元音规则 |
| 以上全部否定 | — | — | 标签和数据不是主因，转查 Encoder 表征与 Head 容量 |

### 第二轮：一次性重标、冻结、重训

1. **前端 v3**：P0 两项必做；P1/P2 规则按门槛结果启用。每条规则对应规则配置文件里的一个开关，写进标注契约身份。单测用第 3 节表格里的词，期望输出见附录 B。
2. **热词端同步**：热词查询调用同一个前端。孤立热词按“句中”形式生成（不带 ʔ），和训练标签保持同一套约定。
3. **全量重标**：写到新目录，如 `outputs/arabic_native_labels_v3`。给 `label_arabic_native.py` 加 `--workers`，复用修复脚本的 spawn 调度。
4. **重新冻结**：补样池按来源分层；验证集按节目或原始录音整体留出，四个来源各有代表；另留一份封存的 test。门槛支持时，再按 CER 过滤 MGB-2。
5. **训练**：用第一轮选定的上采样倍数。
6. **评测**：标签变了，旧 PER 不再可比。比较三样东西：折叠口径的 PER、母语者核对过的小测试集（来自听审表）上的 PER、热词召回。

## 8. 证据范围

本文核对了仓库代码和 HANDOFF 记录，但没有接触工区的标签、音频、模型和解码结果，也没有听音。涉及“PER 贡献多少”和“实际来源分布”的结论，都要等第 5 节的检查确认。

| 强度 | 内容 |
| --- | --- |
| 测过：仓库代码（c6a02b5） | `_` 触发 `unsupported_Arabic_letter`；`f_i_y`→`f i j`；ـَي/ـَو→a j/a w；ث/ذ/ظ→θ/ð/ðˤ；词尾 ه→h；词首 ا→ʔ 加元音；Head 结构；训练 PER 为英阿微平均；`freeze()` 按文件顺序填充 |
| 测过：本地固定版本 CALIMA | EGY 库无 CAPHI；EGY diac 的写法、辅音改写、`_` 出现率；MSA 格尾一致率 64.9%、ال 词的 `2_a`、بين 选错；两组的暴露占比；前 300 词覆盖率 |
| 记录：HANDOFF 0.216 | 英语 4.72% / 阿语 17.60%（r15），基模 5.94% / 22.36%；训练在线 PER 14.41%；英语验证缓存字节数 |
| 先验：已打开的文献 | MGB-2 非逐字转写、37% 的词无锚点匹配、外语配音未标、约 30% 是方言；AuT 编码器 12.5Hz、预训练以中英为主；TIMIT PER 8.3%；CODA 拼写规则 |
| 推断：未核实 | 8,208 条中 `_` 的占比；补样训练集只含 ABN 和 MGB-2 前段；英语验证约 4.2h 及每秒 3.9 个目标（由字节数换算）；阿语训练 PER 约 18%；各成因的 PER 贡献区间 |

测量语料的局限：PADT 是书面新闻，EDC 是 Facebook 文本，都不是语音转写。工区实际标签的暴露占比请用 D6 重算。

### 来源

- [CODA*: Conventional Orthography for Dialectal Arabic（CAMeL 指南）](https://camel-guidelines.readthedocs.io/en/latest/orthography/)
- [The MGB-2 Challenge: Arabic Multi-Dialect Broadcast Media Recognition](https://ar5iv.labs.arxiv.org/html/1609.05625)
- [Qwen3-ASR Technical Report](https://arxiv.org/pdf/2601.21337)
- [wav2vec 2.0（TIMIT PER，表 3）](https://arxiv.org/pdf/2006.11477)
- [UD Arabic-PADT 树库](https://github.com/UniversalDependencies/UD_Arabic-PADT)
- [EDC 埃及方言语料](https://github.com/TaghreedT/EDC)
- [CAMeL Tools 形态特征说明](https://camel-tools.readthedocs.io/en/stable/reference/camel_morphology_features.html)

## 附录 A：诊断脚本

以下脚本都是只读的，不改动任何产物，已用合成数据和公开语料（PADT、EDC）跑通。`per_breakdown.py` 与 `text_vs_asr.py` 需要先把验证集解码结果导出成脚本注释里说明的 JSONL。脚本为一次性诊断用，没有按仓库 `src/` 的 lint 规范整理，因此只放在本文中，不作为仓库代码提交。

### D5 / 来源分布：两条 shell 命令

```bash
# EGY 的 unsupported_Arabic_letter 到底是哪个字符（验证“就是 _”的推断）
cat outputs/arabic_native_labels_v1/parts/ar_egy/*/review.jsonl \
 | jq -r 'select(.label_issue|startswith("unsupported_Arabic_letter"))|.label_issue' \
 | sort | uniq -c | sort -rn | head -20

# 训练/验证集按来源文件的条数、小时和行号范围（验证“补样只用到 ABN + MGB2 前段”）
A=outputs/en_ar_native_ctc_v1
cat "$A/train.jsonl" "$A/validation.jsonl" \
| jq -r '[.split,.source_group,(.source|split("/")[-1]),.duration_seconds,.source_row]|@tsv' \
| awk -F'\t' '{k=$1"\t"$2"\t"$3;n[k]++;h[k]+=$4;if(!(k in lo)||$5<lo[k])lo[k]=$5;if($5>hi[k])hi[k]=$5}
  END{for(k in n)printf "%s\t%d条\t%.1fh\trow %d-%d\n",k,n[k],h[k]/3600,lo[k],hi[k]}' | sort
```

### D5 前端探针（已在 c6a02b5 上跑过，结果见第 3 节）

```python
# 把本地 CALIMA 实测的 diac / CAPHI 直接喂给项目转换函数，不需要加载 CALIMA。
import sys
sys.path.insert(0, "src")
from qwen_hotword.phonemes.arabic_native import caphi_units, egyptian_units

EGY = {  # diac（CALIMA-EGY 实测输出） : 开罗实际读音
    "البَيت": "ilbeːt", "بَيت": "beːt", "لَيه": "leːh", "يَوم": "joːm", "دَه": "da",
    "شُفتُه": "ʃuftu", "عَشَأْن": "ʕaʃaːn", "ياخُذ": "jaːxud", "أَكْثَر": "ʔaktar",
    "ما_بَقاش": "mabʔaːʃ", "يَعْنِي": "jaʕni",
}
MSA = {  # CAPHI（CALIMA-MSA 实测输出） : 期望
    "f_i_y": "f iː", "2_a_l_l_a_dh_i_y": "… ð iː", "2_a_b_u_w": "ʔ a b uː",
    "2_a_l_7_u_k_uu_m_a_t_i": "句中常读 l ħ u k uː m a", "b_a_y_y_i_n_a": "“之间”应为 b a j n a",
}
for d, want in EGY.items():
    try:
        print("EGY", d, " ".join(egyptian_units(d)), "| 实际", want)
    except ValueError as e:
        print("EGY", d, "ERR", e, "| 实际", want)
for c, want in MSA.items():
    try:
        print("MSA", c, " ".join(caphi_units(c)), "| 期望", want)
    except ValueError as e:
        print("MSA", c, "ERR", e, "| 期望", want)
```

### D6 标签暴露统计：label_exposure.py

用法：`python label_exposure.py outputs/arabic_native_labels_caphi_y_v2/labels.jsonl --manifest outputs/en_ar_native_ctc_v1/train.jsonl`

```python
"""D6: count fragile label positions in labels.jsonl (reads text only, no WAV).
usage: python label_exposure.py outputs/arabic_native_labels_caphi_y_v2/labels.jsonl [--manifest train.jsonl]"""
import sys, json, re, collections, argparse
p = argparse.ArgumentParser(); p.add_argument('labels'); p.add_argument('--manifest')
a = p.parse_args()
keep = None
if a.manifest:  # restrict to rows that actually entered the manifest
    keep = {(r['source'], r['source_row']) for r in map(json.loads, open(a.manifest, encoding='utf8'))}
SV = '\u064E\u064F\u0650'; TAN = '\u064B\u064C\u064D'
RULES = {
 'ar_msa': {
  'wasla_mid (ʔ a)':  (lambda w, d, i: i > 0 and w.startswith('ال'), 2),
  'final_short_vowel': (lambda w, d, i: re.search('[' + SV + ']$', d.replace('\u0651', '')) is not None, 1),
  'final_tanwin':      (lambda w, d, i: re.search('([' + TAN + ']|\u064Bا|\u064Bى)$', d) is not None, 2),
  'taa_marbuta_t':     (lambda w, d, i: re.search('ة[' + SV + TAN + ']$', d) is not None, 1),
  'final_i_y (fake j)': (lambda w, d, i: d.endswith('\u0650ي') and len(w) > 1, 1),
  'final_u_w (fake w)': (lambda w, d, i: d.endswith('\u064Fو') and len(w) > 1, 1),
 },
 'ar_egy': {
  'fatha+y/w (e:/o:)': (lambda w, d, i: len(re.findall('\u064E[يو](?![\u064B-\u0652اىيو])(?=.)', d)) > 0, 2),
  'final_long_vowel':  (lambda w, d, i: re.search('(\u0650ي|\u064Fو|\u064Eا|\u064Eى)$', d) is not None and len(w) > 1, 1),
  'th/dh/zh letters':  (lambda w, d, i: any(c in d for c in 'ثذظ'), 1),
  'respelled t/d->th/dh': (lambda w, d, i: any(x in w and y in d and y not in w for x, y in (('ت','ث'),('د','ذ'),('س','ث'),('ض','ظ'),('ز','ذ'),('ز','ظ'))), 1),
  'final_vowel+h':     (lambda w, d, i: re.search('[' + SV + ']ه$', d) is not None and w not in ('لله', 'الله'), 1),
  'clitic_underscore': (lambda w, d, i: '_' in d, 1),
  'ashaan':            (lambda w, d, i: w in ('عشان', 'علشان'), 2),
 }}
tot = collections.Counter(); hit = collections.Counter(); unit = collections.Counter(); utt_hit = collections.Counter()
for line in open(a.labels, encoding='utf8'):
    r = json.loads(line); g = r.get('group')
    if g not in RULES or (keep is not None and (r['source'], r['row']) not in keep): continue
    tot[g, 'utts'] += 1; tot[g, 'words'] += len(r['words']); tot[g, 'units'] += len(r['ids'])
    seen = set()
    for i, (w, d) in enumerate(zip(r['words'], r['diacritized_words'])):
        for k, (f, u) in RULES[g].items():
            if f(w, d, i): hit[g, k] += 1; unit[g, k] += u; seen.add(k)
    for k in seen: utt_hit[g, k] += 1
for g in RULES:
    if not tot[g, 'utts']: continue
    print(f"== {g}: {tot[g,'utts']} utts, {tot[g,'words']} words, {tot[g,'units']} label units")
    s = 0
    for k in RULES[g]:
        s += unit[g, k] if k not in ('respelled t/d->th/dh', 'clitic_underscore') else 0
        print(f"  {k:24s} words {hit[g,k]/tot[g,'words']*100:5.1f}%  utts {utt_hit[g,k]/tot[g,'utts']*100:5.1f}%  ~units {unit[g,k]/tot[g,'units']*100:5.2f}%")
    print(f"  approx. fragile units total ~{s/tot[g,'units']*100:.1f}% (respelled/underscore excluded from sum)")
```

### D0 / D2 / D8 PER 拆分、折叠与语速分桶：per_breakdown.py

```python
"""D0/D2/D7: PER breakdown from a validation decode.
input JSONL, one line per utterance: {"utt":..., "group":"ar_egy|ar_msa", "source":..., "ref":["ʔ","a",...], "hyp":[...], "duration": seconds (optional)}
(ref/hyp are phone strings without the 'ar_ipa:' prefix; produce it from the existing eval loop)."""
import sys, json, collections
def align(r, h):
    n, m = len(r), len(h); D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(r[i-1] != h[j-1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1]+(r[i-1] != h[j-1]):
            ops.append(('C' if r[i-1] == h[j-1] else 'S', r[i-1], h[j-1])); i -= 1; j -= 1
        elif i and D[i][j] == D[i-1][j]+1: ops.append(('D', r[i-1], None)); i -= 1
        else: ops.append(('I', None, h[j-1])); j -= 1
    return D[n][m], ops[::-1]
VOW = {'a','i','u','e','o'}
def is_vowel(p): return p.rstrip('ː') in VOW
FOLDS = {
 'F1 vowel length merged': lambda s: [p.rstrip('ː') if is_vowel(p) else p for p in s],
 'F2 gemination merged':   lambda s: [p if is_vowel(p) else p.rstrip('ː') for p in s],
 'F3 glottal stop removed': lambda s: [p for p in s if p != 'ʔ'],
}
FOLDS['F1+F2+F3'] = lambda s: FOLDS['F3 glottal stop removed'](FOLDS['F2 gemination merged'](FOLDS['F1 vowel length merged'](s)))
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf8')]
def micro(rs, f=lambda s: s):
    e = n = 0
    for r in rs: e += align(f(r['ref']), f(r['hyp']))[0]; n += len(f(r['ref']))
    return e / max(n, 1)
for key in ('group', 'source'):
    print(f'== micro PER by {key}')
    by = collections.defaultdict(list)
    for r in rows: by[r[key]].append(r)
    for k, rs in sorted(by.items()): print(f'  {k[-60:]:60s} n={len(rs):5d}  PER={micro(rs):.3f}')
for g in sorted({r['group'] for r in rows}):
    rs = [r for r in rows if r['group'] == g]; ops = collections.Counter(); pairs = collections.Counter(); per = []
    for r in rs:
        e, al = align(r['ref'], r['hyp']); per.append(e / max(len(r['ref']), 1))
        for o, a, b in al:
            if o != 'C': ops[o] += 1; pairs[(o, a, b)] += 1
    N = sum(len(r['ref']) for r in rs); per.sort()
    print(f'\n== {g}: utts={len(rs)} ref_units={N} micro={micro(rs):.3f} macro={sum(per)/len(per):.3f}')
    print('  S/D/I share of ref units:', {o: round(c/N, 3) for o, c in ops.items()})
    q = lambda x: per[min(len(per)-1, int(x*len(per)))]
    print(f'  per-utt PER p50={q(.5):.2f} p90={q(.9):.2f} p99={q(.99):.2f}  share>0.4={sum(p>0.4 for p in per)/len(per):.3f}  share>0.7={sum(p>0.7 for p in per)/len(per):.3f}')
    tail = [r for r, p in zip(rs, [align(r['ref'], r['hyp'])[0]/max(len(r['ref']),1) for r in rs]) if p <= 0.4]
    print(f'  PER after dropping utts with PER>0.4: {micro(tail):.3f}  (gap vs micro = share carried by the long tail)')
    for name, f in FOLDS.items(): print(f'  {name:26s} PER={micro(rs, f):.3f}')
    if all('duration' in r for r in rs):  # optional field: seconds; PER vs target density (head resolution check)
        for lo, hi in ((0, 9), (9, 12), (12, 15), (15, 99)):
            b = [r for r in rs if lo <= len(r['ref']) / max(r['duration'], 1e-6) < hi]
            if b: print(f'  ref phones/s {lo:2d}-{hi:2d}: n={len(b):5d}  PER={micro(b):.3f}')
    print('  top error pairs:', [(f'{o}:{a}->{b}', c) for (o, a, b), c in pairs.most_common(20)])
```

### D3 文本与基座 ASR 的一致性：text_vs_asr.py

```python
"""D3: is the reference text what was said? Compare it with base Qwen3-ASR output (no hotwords).
input JSONL: {"utt":..., "group":..., "source":..., "ref_text":..., "asr_text":..., "per": <this utt's phone PER>}"""
import sys, json, re, collections, statistics
def norm(t):
    t = re.sub('[ً-ْٰـ]', '', t)
    t = re.sub('[أإآٱ]', 'ا', t).replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي')
    t = re.sub(r'[^\w\s]', ' ', t)
    return ' '.join(t.split())
def ed(a, b):
    d = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        p = d[:]; d[0] = i
        for j, y in enumerate(b, 1): d[j] = min(p[j]+1, d[j-1]+1, p[j-1]+(x != y))
    return d[-1]
rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf8')]
for r in rows:
    a, b = norm(r['ref_text']).replace(' ', ''), norm(r['asr_text']).replace(' ', '')
    r['cer'] = ed(a, b) / max(len(a), 1)
    w1, w2 = norm(r['ref_text']).split(), norm(r['asr_text']).split()
    r['wer'] = ed(w1, w2) / max(len(w1), 1)
by = collections.defaultdict(list)
for r in rows: by[(r['group'], r['source'].split('/')[-1])].append(r)
print('group/source  n  median_CER  WER  share_CER>0.3  meanPER')
for k, rs in sorted(by.items()):
    print(k, len(rs), round(statistics.median(r['cer'] for r in rs), 3), round(sum(r['wer'] for r in rs)/len(rs), 3),
          round(sum(r['cer'] > 0.3 for r in rs)/len(rs), 3), round(sum(r['per'] for r in rs)/len(rs), 3))
print('\nphone PER by text-agreement bucket (if PER rises steeply with CER, text/audio mismatch is a main driver):')
for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 9)):
    rs = [r for r in rows if lo <= r['cer'] < hi]
    if rs: print(f'  CER {lo:.1f}-{hi:.1f}: n={len(rs):5d} ({len(rs)/len(rows)*100:4.1f}%)  mean PER={sum(r["per"] for r in rs)/len(rs):.3f}')
if len(rows) > 2:
    print('pearson(CER, PER) =', round(statistics.correlation([r['cer'] for r in rows], [r['per'] for r in rows]), 3))
```

## 附录 B：Codex 指令

第一轮指令可以直接粘贴给 Codex。第二轮等第一轮的回传文件到了以后，由 Claude 按第 7 节的判定门槛定稿；下面先给出骨架和单测期望输出。

### 第一轮：只读诊断 + Head ×3 对照重训

```text
你在仓库 qwen3-asr-hotword 工作，分支 codex/g2p-coverage-scan，开始前 git pull --ff-only。

先读这些，再动手：
- AGENTS.md（交付规则、进度显示、输出目录规则）
- docs/ARABIC_CTC_PER_ANALYSIS_2026-11-12.md（本任务依据：第 5 节诊断、第 7 节执行方案、附录 A 脚本）
- docs/ARABIC_NATIVE_LABELING_HANDOFF_2026-11-12.md
- docs/HANDOFF.md 的 0.214–0.216

任务目标：测量当前阿语标签与数据的状况，并做一次只改 Head 上采样倍数的对照训练。本轮不改任何标签。

硬性规则：
1. 不过度设计、不过度建设、不过度测试。只做 WP1–WP4 写明的最小实现。规格外的问题写进交付说明的“发现的问题”，不要自行实现。
2. 先复用现有代码，不新写平行实现：
   - scripts/diagnose_frozen_ctc.py 与 src/qwen_hotword/training/ctc_diagnostics.py 已有分组 PER、sample_groupings、CTC pressure 分桶、Top 删除/插入/替换和逐样本指标；
   - src/qwen_hotword/evaluation/casablanca_wer.py 已有 load_qwen；
   - scripts/run_en_ar_merged.py、run_en_ar_training.train_stage、sharded_ctc.train_sharded_ctc_head（已有 head_time_upsampling_factor 参数）。
3. 以下目录只读，绝不写入：outputs/arabic_native_labels_v1、outputs/arabic_native_labels_caphi_y_v2、outputs/en_ar_native_ctc_v1、outputs/en_ar_r15_5100_ctc_v1。所有输出写新目录；目录已存在就报错退出，--resume 的场景除外。缓存、Head 和清单路径从源运行的 plan.json / cache_sources.json 读，不要硬编码。
4. 不读 test；不改 src/qwen_hotword/phonemes/arabic_native.py。
5. 本地只用小的合成数据验证机制，不得声称工区结果。
6. 长任务按 AGENTS.md 显示阶段名、已完成/总数、耗时和周期活动提示。
7. 交付按 AGENTS.md：ruff、mypy、相关单测、git diff --check 全部通过；docs/HANDOFF.md 新增 0.217 节点，写明分支、pull 命令、四个 WP 的工区命令、resume 行为、输出目录、需要回传的小文件；提交并推送 codex/g2p-coverage-scan，回报远端 commit SHA。

WP1 标签与数据只读统计（CPU）
新脚本 scripts/diagnose_arabic_labels.py，子命令 stats --output-dir outputs/arabic_per_diag_v1/labels：
a. review 原因明细：读两份标签目录各分片的 review.jsonl（EGY 用 v1，ar_msa 用 v2，按各自 report.json / contract.json 定位分片），按 label_issue 冒号后的值计数。输出 unsupported_Arabic_letter 的字符分布、no_analysis 的前 200 个词及次数、unsupported_CAPHI 的单位分布。
b. 标签暴露：取 v2 聚合 labels.jsonl 中进入 en_ar_native_ctc_v1 的 train.jsonl / validation.jsonl 的行（用 source+row 关联 source+source_row），按附录 A label_exposure.py 的规则分 split、分组计数。规则逐条照搬，不增不删。
c. 来源分布：train / validation 按 source_group 和 source 文件统计条数、小时、source_row 最小/最大值，并附该文件在 inventory 中的总行数。
输出 labels_report.json，控制在 200 KB 以内。

WP2 验证集逐样本诊断（GPU 4，分钟级）
扩展 scripts/diagnose_frozen_ctc.py：
a. 若 CLI 尚未暴露 sample_groupings，增加 --grouping NAME=MANIFEST_FIELD，可重复；本轮用 source_group=source_group 和 source=source。
b. --per-sample-output PATH：每个样本一行，含 sample_id、各分组值、duration_seconds、参考与假设的音素标签串、编辑数、参考长度、CTC pressure 分桶。
c. 报告新增两项。一是按分组的逐样本 PER 分位数（p50/p90/p99），以及 PER>0.4、PER>0.7 的占比。二是 F1/F2/F3 折叠后的 PER，定义同附录 A per_breakdown.py：F1 合并元音长短，F2 合并双辅音，F3 删除 ʔ。只作用于 ar_ipa: 单位，英语单位不动。
用 outputs/en_ar_r15_5100_ctc_v1 的最佳 Head 跑其验证缓存。输出 outputs/arabic_per_diag_v1/val_diagnostics.json 和 val_per_sample.jsonl。
另加 diagnose_arabic_labels.py review-sheet --per-sample PATH：每组 100 条，50 条取逐样本 PER 最高，50 条按 audio_sha256 排序取前 50。导出 CSV：id、组、source、source_row、audio_path、时长、原文、diacritized_words、参考音素、假设音素、PER。CSV 留工区，不回传。

WP3 文本一致性（GPU 4，约 1 小时）
新脚本 scripts/check_arabic_text_agreement.py：
- 模型：基座 /glusterfs_103/models/Qwen3-ASR-1.7B，用 casablanca_wer.load_qwen，language="Arabic"，不加热词。
- 输入：en_ar_native_ctc_v1/validation.jsonl 的全部阿语行，加 train.jsonl 中每个 source 文件按 audio_sha256 排序的前 300 行。
- 文本规范化：照搬附录 A text_vs_asr.py 的 norm。
- 输出逐句 JSONL：id、split、组、source、参考文本、ASR 文本、CER、WER。中断后跳过已写的 id 续跑。
- 输出汇总 JSON：每组、每个 source 的 CER 中位数、平均 WER、CER>0.3 占比。传入 --per-sample（WP2 输出）时，再给验证集各 CER 分桶的平均 PER，以及 PER 与 CER 的 Pearson 相关系数。
- 输出目录 outputs/arabic_per_diag_v1/text_agreement。

WP4 Head ×3 上采样对照重训（GPU 4，约 4 小时）
scripts/run_en_ar_merged.py 新增动作 retrain-head：
- 参数 --source-run outputs/en_ar_r15_5100_ctc_v1、--head-time-upsampling-factor（默认 2）、--output-dir（新目录）、--resume。
- 校验源运行 report 为 completed，且缓存 SHA 与 plan 一致；然后按引用复用其英阿缓存，不复制、不重提。
- 除上采样倍数外，所有超参数、seed、早停和选模规则与 r15 运行完全一致。
- 清单里的 ctc_time_upsampling_factor=2 是可行性下限：倍数 ≥ 2 放行，< 2 拒绝。
- report 记录倍数、源运行身份，以及与源运行相同的指标字段。
工区命令用 --head-time-upsampling-factor 3 --output-dir outputs/en_ar_r15_5100_head_up3_v1。

本地验收：
- 每个 WP 用合成小样本跑通；覆盖三种情况：输出目录已存在时报错、只读目录未被写入、WP4 倍数 < 2 被拒。
- 不做大规模或性能测试。

交付说明须包含：远端 commit SHA；四个 WP 的工区命令与预计资源；需要回传的文件（labels_report.json、val_diagnostics.json、text_agreement 汇总 JSON、up3 运行的 head/report.json）；“发现的问题”清单。
```

### 第二轮：骨架（第一轮结果回来后定稿）

第二轮沿用第一轮的硬性规则 1–7，并将第 4 条改为“只允许按下表修改 arabic_native.py”。工作包如下：

1. **WP5 前端 v3**：P0 两项必做。其余规则各对应新增配置文件 `configs/arabic_native_rules.v3.json` 里的一个开关，只启用判定门槛支持的那些。配置文件的 SHA 写进标注契约。热词端调用同一入口，孤立热词按句中形式生成。
2. **WP6 全量重标**：写到 `outputs/arabic_native_labels_v3`。`label_arabic_native.py` 增加 `--workers`，复用 `repair_arabic_native_labels.py` 的 spawn 调度。
3. **WP7 重新冻结**：补样池按来源分层；验证集按节目或原始录音整体留出（取不到标识时，按音频父目录分组）；另留封存的 test；门槛支持时按 WP3 的 CER 过滤 MGB-2。
4. **WP8 训练与评测**：用第一轮选定的上采样倍数。报告折叠口径 PER，以及母语者核对小测试集上的 PER。

#### WP5 单测期望输出

前五行是 P0，不论门槛结果如何都要通过；其余行只在对应规则启用时生效。“当前输出”一列是 c6a02b5 上实测的结果；P0 的期望输出已用模拟修复验证。

| 输入（diac 或 CAPHI） | 当前输出 | v3 期望 | 规则 |
| --- | --- | --- | --- |
| ما_بَقاش | 报错 `unsupported_Arabic_letter:_` | m aː b a ʔ aː ʃ（不再报错） | P0 `_` |
| `f_i_y` | f i j | f iː | P0 词尾 i_y |
| `2_a_l_l_a_dh_i_y` | ʔ a lː a ð i j | ʔ a lː a ð iː | P0 词尾 i_y |
| `2_a_b_u_w` | ʔ a b u w | ʔ a b uː | P0 词尾 u_w |
| `h_i_y_a` | h i j a | h i j a（不变） | P0 回归 |
| بَيت / يَوم | b a j t / j a w m | b eː t / j oː m | EGY eː/oː |
| دَه / شُفتُه | d a h / ʃ u f t u h | d a / ʃ u f t u | EGY 词尾 h |
| يَعْنِي | j a ʕ n iː | j a ʕ n i | EGY 词尾长元音读短 |
| البَيت（句中，前词以元音结尾） | ʔ i l b a j t | l b eː t | 句中冠词，加 eː |
| `2_a_l_7_u_k_uu_m_a_t_i`（句中，名词，cas=g） | ʔ a l ħ u k uː m a t i | l ħ u k uː m a | 句中冠词，加 MSA 停顿形 |
| ياخُذ（转写为 ياخد） | j aː x u ð | j aː x u d | 转写者原字母优先 |
| أَكْثَر（转写为 أكتر） | ʔ a k θ a r | ʔ a k t a r | 转写者原字母优先 |
