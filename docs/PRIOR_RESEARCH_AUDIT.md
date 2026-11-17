# 前四篇论文与第五篇边界审计

核查日期：2026-11-17（北京时间）。这是对当时默认分支的只读来源审计，未运行前四篇实验，未访问本地不可见的云端 /workspace。

## 固定读取身份与范围

| 仓库 | 当前默认分支 | 读取提交 | AGENTS.md |
|---|---|---|---|
| cross-subject-mi-eeg | main | c4ea1f4ee1d078bcb015922f8092f05e263f2399 | 根路径404；完整递归树22929项、truncated=false，无任何AGENTS.md |
| few-shot-mi-eeg | main | da2a3acee24746981bbcdd779ab88132da6e24c4 | 唯一根AGENTS已读 |
| research3 | research/paper3-protocol-v2-w0 | c58c6b55206df0e44c56a89e0c9fd0e9d031046a | 唯一根AGENTS已读 |
| research4 | main | 167d1d2b197cb2735a3fcb3c0110a382d7043ca6 | 唯一根AGENTS已读 |

四库README、论文入口及适用AGENTS已核查。论文2当前MAIN全文、论文3实验协议与METHOD_SPEC全文、论文3第2/7/9轮入口、论文4paper/current/README与manifest及PAPER_RESULTS_POLICY已读取。来源URL、Git blob SHA、所读取文本保存在本地 work/prior_sources.json。论文4manifest仅保存身份与文件数，不复刻大型来源清单。

优先使用GitHub connector。Paper1递归树两次返回transport closed，随后通过本机直连GitHub REST获得完整树；根AGENTS 404与完整树检索一致。首次few-shot仓库元信息读取网络失败，后续README/提交/完整树/文件读取成功，不把首次失败混写为仓库不存在。

## 论文1：源队列构成，不能当第五篇的新贡献

当前公开入口为[2026-10-29 IJBET整理](https://github.com/jackzhu119/cross-subject-mi-eeg/blob/c4ea1f4ee1d078bcb015922f8092f05e263f2399/research_runs/PAPER1_IJBET_20261029/README_zh.md)。完整当前匿名稿及编辑材料在仓库外另行交付；本次没有声称已读取该完整新稿。

主线是跨被试MI的源参与者配置与弱类别覆盖。当前入口明确将eligible-v3研究与学校source-risk-v1目标函数研究分别报告：两项完成研究用相同85名外部参与者，不能合并为独立外部复现。尚未完成的composition-v2 630项计划未写入论文。README所列源配置收益是前作结果，本次不重算、不迁入第五篇结果。

第五篇应避开的重复：增加源参与者规模、弱类风险目标、泛称跨被试EEG稳健性。可继承的规范是代码/输入/结果身份冻结、完整参与者差异和等预算对照。

## 论文2：少标签重校准的组件分解

当前稿件为[Disentangling spatial-filter and classifier contributions to limited-label motor-imagery EEG recalibration](https://github.com/jackzhu119/few-shot-mi-eeg/blob/da2a3acee24746981bbcdd779ab88132da6e24c4/manuscript/bmt_component_revision_20261029/MAIN.md)。现有结果比较同一当前会话标签用于分类器重校准、空间滤波重学、support-only EA；以参与者为统计单位。Kumar与BNCI来自两套采集，六个分析层不是六个独立数据集。BNCI中原先空间重学增量接近零、EA增益为正，表明“空间更新普适有效”并不成立。

AGENTS要求离线解码、模型表征、人类学习和行为分开；任何synthetic输出保留synthetic=true与software_validation_only；变换仅从允许的训练/支持数据拟合，保存真实版本、种子与hash。其中旧云端环境路径及开发分支是该库历史执行指令，不将其套用为第五篇本机环境事实。

第五篇应避开的重复：用预测纠错/新引错分解本身宣称创新，或把空间适应/少样本标签预算当作新主题。可继承共同分母的错误转移与参与者配对统计。第五篇真正待检验的对象是固定模型、固定其余原始采样时，局部输入干预对其他时间位置的净影响。

## 论文3：短前缀、几何约束与可靠性已进入前作范围

[实验协议](https://github.com/jackzhu119/research3/blob/c58c6b55206df0e44c56a89e0c9fd0e9d031046a/protocol/EXPERIMENT_PROTOCOL.md)与[实施规格](https://github.com/jackzhu119/research3/blob/c58c6b55206df0e44c56a89e0c9fd0e9d031046a/protocol/METHOD_SPEC.md)明确20个真实经过的校准试次前缀、完整未来run查询、源/元训练/元验证/内部留出/外测隔离。可逆且受限的矩阵指数、元学习低维适应方向、时空Lie、PCT新增错误控制、可靠性门控、支持信息约束与源预测锚定均已在前作方案/各轮范围内。

当前README有较新状态汇总，而第9轮独立说明仍写“未提交”；不能把过时方案状态当当前运行状态，也不能由任务完成数推出效果。本次未读取第3篇原始预测重算效果，不做“第三篇方法有效/无效”的新结论。现有README/协议中的方法与状态只是研究边界证据。

第五篇应避开的重复：泛称“受约束”“可靠性门控”“源残差”就是创新。需明确区分：

- 第3篇：目标会话少量标签驱动的模型/输入适应，约束空间变化和更新后的错误。
- 第5篇：冻结参数模型中局部原始信号受损，量化未修改时间位置的配对净额外错误，拆分记录级统计、通道聚合、时间上下文路径。
- 若后续提出约束上下文残差，必须以相同局部编码器/上下文范围/训练预算的无约束版本及简单隔离基线作对照；不能把前作约束思想的换任务当充分新颖性。

AGENTS的双通道结果交付规则是前作后续轮次的运行政策：紧凑论文数据先交付、完整科学输出独立持久化；摘要、科学完成和完整归档分开。第五篇同样应及时保存本机代码、配对预测、日志及哈希，但本次不对学校任务做任何操作。

## 论文4：严格遵守当前白名单与撤回规则

[当前结果入口](https://github.com/jackzhu119/research4/blob/167d1d2b197cb2735a3fcb3c0110a382d7043ca6/paper/current/README.md)、[manifest](https://github.com/jackzhu119/research4/blob/167d1d2b197cb2735a3fcb3c0110a382d7043ca6/paper/current/manifest.json)和[PAPER_RESULTS_POLICY](https://github.com/jackzhu119/research4/blob/167d1d2b197cb2735a3fcb3c0110a382d7043ca6/PAPER_RESULTS_POLICY.md)已读。仅使用2026-11-08修正种类/产地后、manifest明确列入的证据。

2026-11-12撤回的错误枣仁/酸枣仁混种标签模型、旧预测重贴标签分析、混合论文汇总不得从旧提交/备份/生成器恢复成论文结果。本次没有读取任何已撤回成绩。原855谱和当前18表1672谱仍是有效输入，删除的是错标派生结果。corrected 1305、V4 330与B1—5有效正负结果均保留，legacy/OLD命名不自动等于撤回对象。

独立实物与采集条件属于用户声明，不能升格为植物学鉴定或新批次证明；B1—5是既有曝光实物上的内部探索，E05仍为NOT_RUN_NEEDS_NEW_BATCH_DATA。元学习未完成最终分类成绩和量子计算不能混入当前分类入口。没有在此审计运行构建/验证命令，因此不宣称已重新校验所有论文4文件/原谱hash。

与第五篇的内容重复较少，主要继承的是标签身份、撤回政策、负结果保留和独立性边界。

## 第五篇应冻结的定位

建议把第五篇暂定为“局部输入干预的非局部错误代价及其机制”，而不是预先确立一个新网络。原始采样未改的位置仅称“未修改”，不称天然干净；预测翻转、净新增错误和被纠正错误分别保存。影响范围必须区分滤波/卷积精确支持与记录级依赖，oracle干净统计/通道权重冻结只做诊断。

此前云端规划没有本机可见证据。本地旧research5位于 C:/Users/zhu/Documents/Codex/2026-11-17/new-chat-4/work/research5，父任务已发现空Git仓库与两个未跟踪文件并保留；当前交付库 outputs/research5由父任务建立，本审计不改动已有pilot或日志。研究5初始内容与连接/执行验收由父任务单独记录。

这份审计不能证明第五篇新颖，也不能证明AnySleep存在实用大小的远端误判。近邻文献核查与真实配对实验仍是继续条件。

