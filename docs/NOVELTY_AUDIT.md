# 第五篇选题近邻与可证伪性核查

检索日期：2026-11-17（Asia/Shanghai）。状态：有界初步检索；不是系统综述，不构成新颖性证明。本文不含本研究实测结果。

## 当前判断

值得优先执行低成本现象检验，但尚不足以立项训练新模型。“睡眠分期抗噪”“按通道质量门控”“用缺失/噪声增强训练”“谱正则控制 Lipschitz 常数”均已有直接近邻。可争取的区别是**仅干预局部采样，以仍然逐点未修改的其他时间片段为结果对象，报告净额外错误及距离衰减，并分离预处理、跨通道聚合和时间上下文路径**。本次检索未定位到完整采用这一组合估计量与机制拆解的直接论文；这只是检索边界内的缺口，不可改写为“首次”。

## 实际可用的本地相关指导

已实际读取：`C:/Users/zhu/.codex/skills/scientific-brainstorming/SKILL.md`（1.4）、`hypothesis-generation/SKILL.md`（2.4）。应用于候选假设、竞争解释、预定结果对象和停止条件，未使用其可选外部付费服务。实际目录中还找到 scientific-critical-thinking、experimental-design、literature-review、citation-management、research-lookup、statistical-analysis、statistical-power、scientific-visualization、matplotlib、neurokit2 等。`nature-academic-search` 与 `nature-ref-verifier` 不在已枚举的本地目录中，未加载。

程序性指导出处：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. [arXiv 当前记录](https://arxiv.org/abs/2609.00065)，本次读取记录为 v2；不将其当作睡眠分期科学证据。

## 必查近邻与边界

| 工作及一手入口 | 已核对内容 | 对第五篇的约束 |
|---|---|---|
| REST, WWW 2020, DOI [10.1145/3366423.3380241](https://doi.org/10.1145/3366423.3380241)；[作者全文](https://arxiv.org/html/2001.11363)；[官方代码](https://github.com/duggalrahul/REST) | 单通道睡眠分期，Sleep-EDF/SHHS；对抗训练、谱正则、稀疏压缩；按不同噪声强度评估宏 F1。 | 谱约束抗噪本身不能称创新；须在相同编码器、上下文、预算下纳入其思想的公平基线。其总体噪声分类评价不同于未修改片段的配对净错误。 |
| DSF, NeuroImage 2022, DOI [10.1016/j.neuroimage.2022.118994](https://doi.org/10.1016/j.neuroimage.2022.118994)；[作者预印本](https://arxiv.org/abs/2105.12916)；[出版全文](https://www.sciencedirect.com/science/article/pii/S1053811922001239) | 动态空间滤波学习忽略损坏通道，与通道损坏增强结合；包含病理检测和睡眠分期；比较 Autoreject、特征模型等。 | “坏通道会伤害 EEG 分类、注意力可抑制坏通道”已有。必须把跨通道防护与跨时间防护分开；匹配噪声增强不可缺。 |
| CoRe-Sleep, TNSRE 2024, DOI [10.1109/TNSRE.2024.3354388](https://doi.org/10.1109/TNSRE.2024.3354388)；[作者全文](https://arxiv.org/html/2304.06485) | EEG/EOG 协调表征，兼容缺失和噪声模态；内层 epoch 与外层 21 epochs（10.5 min）Transformer；SHHS-1，70/30 参与者训练/测试划分，训练另留100人验证。 | noisy/missing modality segments 和多模态稳健融合不是新问题。已核对全文的噪声分析主要比较噪声患者/区域与模态组合，不是本题所需的距离分层未修改区净额外错误。 |
| AnySleep, npj Digital Medicine 2026, DOI [10.1038/s41746-026-03406-7](https://doi.org/10.1038/s41746-026-03406-7)；[正式全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13722680/)；[代码](https://github.com/dslaborg/AnySleep) | 正式论文已公开；全记录推断；通过变通道配置评估其通道无关能力。 | 全局依赖是候选机制，不是已证实缺陷。应固定用户指定提交与 checkpoint；模型对通道配置稳健不能推出对局部损坏稳健，反之亦然。 |
| G-CMTF Net, Symmetry 2026, DOI [10.3390/sym18020316](https://doi.org/10.3390/sym18020316)；[出版全文](https://www.mdpi.com/2073-8994/18/2/316) | EEG 主流、辅助 EOG/EMG 的 reliability-aware gated fusion 与长程编码。论文明确承认学到的门反映任务重要性，非外部验证信号质量；受控 EEG 损坏仍列未来工作。 | 质量/可靠性门控为必比简单方法；不应把 attention/gate 数值直接解释为质量真值，不能借其边界夸大本题新颖性。 |
| ProtoSleepNet, [全文入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060499/) | 时间硬注意力 masking、通道 dropout、可接不同 epoch/sequence encoder；讨论 epoch encoder 跨时间泄漏会破坏严格时间定位。 | 上下文屏蔽/局部定位已有非常接近的表述，后续必须完整逐节比对其设计和估计量。PMC 后续请求触发验证码，本轮仅完成可访问摘要与方法段核查。 |
| MaskSleepNet, 2023, [PubMed](https://pubmed.ncbi.nlm.nih.gov/37028323/) | masking + CNN + SE + 多头注意力，处理输入模态差异。 | 屏蔽输入/模态并保留时间信息不是新方法；本轮仅摘要核查，全文待查。 |
| The Masking Impact of Intra-artifacts..., TNSRE 2022, DOI [10.1109/TNSRE.2022.3173994](https://doi.org/10.1109/TNSRE.2022.3173994)；[作者/出版PDF入口](https://ieeexplore.ieee.org/ielx7/7333/4359219/09771421.pdf) | 比较 EEG 内源眼动/肌电/心电成分的正负影响及消除策略，涉及 Sleep-EDF、MASS。 | 不能泛称“伪迹对睡眠模型的影响无人研究”；EOG 含生理分期信息，不能把所有眼动等同外部干扰。 |
| Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning, [作者记录](https://arxiv.org/abs/2511.06785) | 通过 masking/prompt 减少每 epoch 采集信号，关注缺失信号下分类。 | 与本题目标不同，但 mask-aware 模型需在后续候选基线搜集中考虑；本轮摘要级。 |

## 全局统计污染与序列稳定性依据

- [scikit-learn RobustScaler 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)说明中位数/IQR 相较均值/方差对离群值稳健。这支持对照选择，**不证明整夜 IQR 已被短干扰显著污染**。数学推断：若使用 `z(t)=(x(t)-m(x))/s(x)`，只要重算后的 m/s 改变，原始未修改点的 z 也可能改变；是否达到分类损害必须实测。少数极值未必移动分位数，整夜短暂干扰也可能被平均充分稀释。
- [Stable Recurrent Models, Miller & Hardt, ICLR 2019](https://arxiv.org/abs/1805.10369)研究稳定循环模型与有限前馈近似；[Lipschitz Recurrent Neural Networks, Erichson et al., ICLR 2021](https://arxiv.org/abs/2006.12070)给出稳定条件并评估输入/参数扰动稳健性。它们不能直接证明 AnySleep 有害传播，也不能证明限制全局谱范数就限制某一时间位置的远端错误。
- 方法论推断：整体 Lipschitz 范数控制约束总体输出变化；本题还需要跨位置影响、距离衰减和分类 margin。小幅 logit 改动在低 margin epoch 仍可翻转类别，翻转也可能纠错。新方法若只套全局谱范数会很难区别于 REST。
- [Widmann et al., 2015, DOI 10.1016/j.jneumeth.2014.08.002](https://pubmed.ncbi.nlm.nih.gov/25128257/)讨论滤波的失真/伪迹、响应与报告；[作者全文](https://home.uni-leipzig.de/biocog/eprints/widmann_a2015jneuroscimeth250_34.pdf)明确讨论不应跨信号不连续边界滤波。须保存重采样/滤波后差异支持区域，分开“原始采样未变”和“模型输入未变”。

## 训练重叠和外部验证边界

AnySleep 正式论文 Table 2 报告 SEDF-SC 训练/验证/测试分别为115/15/23条记录，SEDF-ST为30/6/8；因此 Sleep-EDF 是其训练数据来源之一，不能称来源独立的外部验证。应在固定 checkpoint 所对应的已提交划分中逐条匹配参与者，避免同人不同夜泄漏。论文列 CAP 25条记录为 hold-out 高频分析（16 healthy、9 insomnia），没有用于该论文神经网络训练；**checkpoint-specific 对应关系仍须代码审计确认**。CAP 已用于作者的模型评价和下游分析，最多称“未用于该模型训练的第二来源复核”，不能笼统称完全未见/未经选择的独立外部验证。选择其他 checkpoint、YASA 或以后训练模型时应各自建立重叠表。

REST 训练/评估使用 Sleep-EDF 与 SHHS；CoRe-Sleep 使用 SHHS-1；DSF 公共 PC18/TUAB 与私有 Muse 部分并不等于任一 checkpoint 无重叠。若模型来源未给具体训练清单，状态应为 unknown。

## 建议预先冻结的识别与分析规则

1. 估计对象：对预定局部干预 I，同一模型在原始输入与仅 I 采样改变后的输出，比较每个**未修改**、标签有效 epoch 的 `1[p_corrupt != y] - 1[p_original != y]`。同时保存 correct→wrong、wrong→correct、wrong→different-wrong 和完整 logit。不能以翻转率代替净损害。
2. 原始可含天然伪迹，命名用 original/untouched，不用绝对 clean；oracle 冻结名称可写 original-input oracle，避免误导。
3. 用几何时刻或固定种子预先抽样位置，不观察错误选点。空操作和重复推断应为数值一致/预定容差；先做同一整夜重复，后做扰动。
4. 距离定义为 epoch 时间区间到干预区间的最短距离。建议 `[0,30s)`, `[30,60s)`, `[60,120s)`, `[120,300s)`, `[300,600s)`, `>=600s`；主要邻近结果为30–300s。跨精确模型卷积/滤波支持的远端另列，不能以随意5分钟取代精确支持。若支持范围比整夜大，应明确没有可用远端集合。
5. 冻结原始预处理统计量/各 attention 层权重并与重算做2×2诊断；这些是 oracle。固定全部全局项后超出有限局部支持的输入不变输出若不等，应先排查隐藏全局项、数值非确定性和数据处理差异，不能直接归因上下文。
6. 单通道使 channel softmax 恒定为1，但不屏蔽归一化、卷积、残余全局聚合或时间上下文；消失/保留现象皆须限定路径。
7. 同人先聚合夜、位置、种子；参与者作为重采样/置信区间单位，第二来源分开报告。10–20人属于开发性探索，不能把多位置当增加独立样本量。

## 实用门槛和停止/调整建议（建议值，非既有事实）

可在看配对结果前冻结：邻近30–300s区域参与者平均净额外错误至少 **1个百分点** 作为进入方法阶段的候选实用大小；同时报告每次干预多导致的 epoch 数及损害尾部分布。用 **0.5个百分点** 作简单方法剩余损害的候选容忍界。这些数值是研究设计选择，不是公认临床阈值，应在开发报告中保留理由与敏感性结果。

- 若现象仅见于不现实超大振幅、标签错位、不同长度推断或数值不确定性：先修复/缩小结论，不能立新方法。
- 若邻近损害稳定可见、方向跨参与者和第二来源复核且具有实用大小：继续机制拆解。首轮小样本不要求 p<0.05 作单独闸门；若区间宽到无法区分0与实用大小，应判“不确定”，按预定开发规模补足。
- 若稳健归一化、检测坏段后按连续段重启上下文、质量屏蔽门控或匹配扰动增强使剩余损害低于容忍界，且保留原始性能：建议停止新架构路线，可转成可复现诊断/工程基线研究。
- 若达到预定开发规模仍未发现可复核实用损害，或远端仅是极小 logit 波动、纠错与致错相抵：明确报告负结果；停止当前“限制传播”方法立项，不追加按结果选点的对抗实验挽救选题。
- 只有现象和简单基线闸门均通过，才研究“局部证据＋受约束上下文残差”；对照须相同局部编码器、相同上下文与预算，并包含无约束残差、质量门控、匹配增强、REST式谱正则。有效性重点为保留上下文收益的同时减小未修改区净损害，不能只展示整体准确率提升。

## 检索边界与待查

本轮查询覆盖三个指定DOI、AnySleep正式论文/仓库、`sleep staging local corruption propagation`, `temporal context noise neighboring epochs`, `quality gating EEG EOG`, `context masking`, `global normalization outliers`, `stable recurrent models` 等。检索源为网页学术索引，再打开作者/出版/官方文档一手入口。未穷尽 PubMed/IEEE/ACM 逐库引文网络；ProtoSleepNet部分全文受验证码限制；MaskSleepNet、MASS属摘要级。未运行或复现这些近邻代码。未定位到可信的直接“整夜 IQR 短伪迹导致远端净错误”实测证据；这同时意味着本题现象仍可能不存在。
