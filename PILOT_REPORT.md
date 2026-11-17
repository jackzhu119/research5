# 首轮离线验证报告

日期：2026-11-17（Asia/Shanghai）。当前结论：**连接与阶段A复现通过；尚未据此证明局部干扰有未修改区净损害，不进入新方法训练。**

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

依赖安装最初将pandas升级到3.1.1，与Colab自身固定版本不兼容；已恢复并固定2.2.3，补充jedi。YASA权重含旧版sklearn LabelEncoder，当前sklearn1.6.1加载给出InconsistentVersionWarning；实际执行和重复一致性通过，但这不保证跨版本序列化普遍兼容。运行日志完整保留，后续云运行单独记录环境。

## 正在运行 / 待运行

- 首轮SC400单位置30秒的noop/flat/clip/noise配对、13层权重与统计量oracle2×2诊断已提交Colab CPU执行；完成前不报告损害数字。
- 按用户新要求，推送可独立运行的GitHub后台工作流；只有真正出现queued/in_progress运行记录后，才可确认本机关机不影响它。后台结果与失败日志以Actions artifact保存。
- CAP、小规模多参与者复核、简单基线、匹配扰动增强、任何新方法训练均未完成。

## 当前研究建议

继续完成低成本配对现象检验，暂不训练新架构。REST已有谱约束抗噪，DSF/CoRe-Sleep和质量门控/context masking均有直接近邻。AnySleep全局依赖已由源码核实，但实用净损害仍可能为零。固定全局路径后的精确局部支持可超过5分钟，300s+不能直接称远端。

只有未修改区净损害可复核、达到预定实用大小、且简单方法不能充分解决，才进入局部证据＋受约束上下文残差方法阶段。否则停止或调整方向，公开保留负结果。
