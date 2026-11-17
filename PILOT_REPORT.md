# 首轮离线验证报告

日期：2026-11-17（Asia/Shanghai）。当前结论：**连接、阶段A和单晚配对/oracle计算完成；首例未发现总体净损害。旧环境有YASA依赖版本冲突，正在独立兼容环境复核并设置Drive持久化开发队列，不进入新方法训练。**

## 已运行

本机官方浏览器桥：实际发现、调用连接工具，收到动态工具变更，添加/执行/回读测试单元，stdout为COLAB_LOCAL_OK。官方远程MCP也在免费CPU独立session通过同一测试。原生桌面热加载兼容性边界见CONNECTION_STATUS.md。

固定AnySleep提交88e13e31b3fa02482b650a844f6665f2cbbdb82c与run1 checkpoint；下载来源及SHA256见manifests/downloads.json。源码逐文件校验Git blob，未修改官方模型和预处理实现。

完整SC4001E0记录：3个EEG/EOG导联，100Hz，7,950,000采样/通道，79,500秒（22.083小时），2,650个30秒有效标签。MNE输入为伏特，AnySleep按官方128Hz重采样、全局20×IQR裁剪和RobustScaler处理。没有裁剪睡眠窗口，也没有分块。

两管线均完成原始推断与重复推断：

| 管线 | 准确率 | 五类宏F1 | Kappa | 重复预测翻转 | 最大输出差 |
|---|---:|---:|---:|---:|---:|
| AnySleep run1 | 0.946038 | 0.828286 | 0.870223 | 0 | logit 0 |
| YASA 0.8.1 | 0.923774 | 0.731065 | 0.812602 | 0 | 概率 0 |

这些是**一个开发参与者整份长记录**的工程复现结果；长时间Wake占比高，不是睡眠窗口论文基准，不用于模型优劣或泛化结论。AnySleep单次推断约30.97/31.51秒，推断后RSS约1.64/1.68GB，这不是峰值内存。整个阶段A约111.78秒。CPU环境没有GPU显存指标。

按官方公开配置链，SC4001E0映射sedf-sc/001，同参与者两晚001/002在test；Sleep-EDF仍是AnySleep开发来源数据库。YASA官方训练数据库说明未列Sleep-EDF/CAP，逐身份去重未核验。详见docs/TRAINING_OVERLAP.md。发布checkpoint顶层仅观察到state_dict，不能由此认证完整训练配置和日志。

## 失败与修复

最初stage-a运行因MNE read_annotations返回orig_time=None而停止，保存FAILED状态和回溯。直接比对PSG/hypnogram两份EDF头日期时间，均为24.04.8916.13.00；仅在严格解析且相同的情况下显式采用offset=0，保存证据后stage-a-v2通过。旧失败记录没有删除。

依赖安装最初将pandas升级到3.1.1，与Colab自身固定版本不兼容；随后恢复2.2.3的做法虽然完成实际推断，但不满足YASA0.8.1要求pandas>=2.3的元数据约束。第一次GitHub任务也在这个冲突处失败，失败日志artifact已成功保存。已改为独立科学环境pandas3.1.1，避免修改Colab notebook内核依赖，必须复核后再解释YASA数字。YASA权重含旧版sklearn LabelEncoder，当前sklearn1.6.1加载给出InconsistentVersionWarning；实际执行和重复一致性通过，但不保证跨版本序列化普遍兼容。

## 单晚配对与机制结果（已执行，兼容环境复核前为初步结果）

预定EEG Fpz-Cz、原始采样[3,975,000,3,978,000)，30秒，原始记录中点；其余采样逐点不变断言通过。空操作通过。三个实质干预均在AnySleep重采样后改变3,864个采样点（比原始30秒区间向外延伸），局部拓扑支持为30个输出epoch。

| 管线/干预 | 未修改区原对后错 | 原错后对 | 预测翻转 | 净额外错误（2649 epoch分母） |
|---|---:|---:|---:|---:|
| AnySleep flat/clip/noise（各自） | 0 | 0 | 0 | 0 |
| YASA flat | 3 | 4 | 8 | −0.03775个百分点 |
| YASA clip | 1 | 1 | 2 | 0 |
| YASA noise | 1 | 4 | 6 | −0.11325个百分点 |

AnySleep固定统计量/全部通道权重/两者的oracle变体也没有未修改区类别翻转。精确局部支持外的2620个epoch仍有极小logit变化，flat/clip/noise最大分别约0.00570/0.00190/0.00521；两种全局路径同时固定时支持外logit差为0。这支持本例存在全局**数值依赖**，没有证明分类净损害。

主邻近窗口30–300秒共18个epoch：AnySleep各干预0致错0纠错；YASA flat/noise各0致错1纠错，clip0/0。单人单位置不能泛化，负净错误也不能称干扰有益。YASA尚无经认证的精确局部支持，300s+不叫真正远端。

原始日志保留了fixedboth的oracle_statistics字段错误false；实际代码同时固定了统计量和权重，后续已修正标注，不重写旧日志。结果、逐epoch表、logit/概率和距离图已下载本地并上传Google Drive。

## 后台存储与正在准备 / 待运行

- 用户最终指定Colab计算、Google Drive持久保存。目标文件夹为[research5](https://drive.google.com/drive/folders/14k_7KUZu535k0oKM7SDWisLK7HWvifBO)。原始数据/模型与阶段A包、阶段B结果包已上传；Colab挂载Drive后写入并回读COLAB_DRIVE_OK，且connector独立查到同folder内的文件。
- GitHub保留代码；原Actions方案首个任务安装失败后，按用户选择已改为手动备用，不作为当前计算路径。后台Drive驱动使用独立环境、所有数据/模型/结果/日志直接写Drive、阶段状态/校验与可恢复队列。
- **已实际启动** `colab-drive-20261117-v2`：2026-11-17 17:20:40（Asia/Shanghai），代码身份1b7b69e9a79ca03e1be5，上传包SHA256为62c7991a2a6c600e162656af591126264b057cbbedcb84b1a0a537da20538c07。Drive独立回读状态为RUNNING/stage-a，且环境记录确认独立prefix、继承runtime PyTorch、pandas3.1.1、YASA0.8.1，科学依赖闭包检查通过。[实际状态目录](https://drive.google.com/drive/folders/1i-q-8btlTDvBx9b7M4SqL5BU4Q-B6nvs)。这是启动证据，不是声称后续全部任务完成。
- 2026-11-17 17:23:44，Drive独立回读变为 **RUNNING/stage-b**：兼容环境的阶段A已通过，驱动已自动衔接阶段B，无需本机下一条指令。阶段B之后的开发队列仍以自身状态为准，未预报为已完成。
- Drive初版v1因Colab系统Python的ensurepip不可用而失败，原日志与状态保留。v2已实际通过`venv --without-pip`与base pip `--python`管理目标环境的启动测试，并进入模型计算。真实执行中pandas提示继承的可选bottleneck1.4.2低于其加速推荐版本；记录该警告，不将其当已消失。
- 单晚兼容环境A/B通过后，预定10人×3个时间位置的开发队列可自动继续，4小时有限预算；参与者为统计单位，不根据错误选择位置。若Colab回收，不能自动保证重新申请或重新授权；已完成阶段保留，下一次运行按版本/校验继续。
- CAP、小规模多参与者复核、简单基线、匹配扰动增强、任何新方法训练均未完成。

## 当前研究建议

继续完成兼容环境复核和预定开发规模，暂不训练新架构。REST已有谱约束抗噪，DSF/CoRe-Sleep和质量门控/context masking均有直接近邻。AnySleep全局依赖已有本例数值证据，但实用净损害仍可能为零。固定全局路径后的精确局部支持可超过5分钟，300s+不能直接称远端。

只有未修改区净损害可复核、达到预定实用大小、且简单方法不能充分解决，才进入局部证据＋受约束上下文残差方法阶段。否则停止或调整方向，公开保留负结果。
