# 首轮数据与预训练重叠审计

核查日期：2026-11-17。范围：AnySleep固定提交、官方数据转换代码、YASA v0.8.1源码和官方训练说明。本文件是来源审计，不是重新训练、成员推断或完整训练日志复验。

## 结论与使用口径

| 模型 | Sleep-EDF SC400 / SC4001E0 | CAP |
|---|---|---|
| AnySleep models/anysleep-run1.pth | 按固定提交公开的训练配置，SC4001E0映射sedf-sc/001，位于test；同参与者第二夜002也在test。不是该公开配置的train或valid。Sleep-EDF整体仍为模型开发使用的同来源数据库，不能称独立跨数据库外部验证。 | 本轮只核对split内没有CAP键及官方README将examples/cap归为external入口；尚未核对完整训练日志，不能据此宣称绝对未接触。 |
| YASA v0.8.1默认分类器 | 官方公开训练数据库说明不含Sleep-EDF，故可标“官方声明的训练数据库之外”；没有逐身份和原始记录哈希匹配证明。 | 官方公开训练数据库说明不含CAP，具有相同限制。 |

SC400是原始文件名前缀口径；官方转换后的参与者ID是00，记录ID001（第一夜）与002（第二夜）。**不要误把SC4001E0匹配到split中的401**；401对应另一条SC4401…记录。

第一晚实验仍是软件复现与现象探索。若之后扩到Sleep-EDF其他参与者，必须逐人读取固定split并将其训练/验证/测试角色记录进manifest；不能因第一名恰在test就将整个Sleep-EDF称未见数据。

## AnySleep：checkpoint → 配置 → 记录身份 → split

固定提交：88e13e31b3fa02482b650a844f6665f2cbbdb82c。

1. [config/exp002/run.md，第4–6行](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/config/exp002/run.md#L4-L6)将exp002/exp002a训练命令明确关联到anysleep-run1/2/3.pth。其Git blob为d1378f8fb119a6682af8152bdc29adcaecdd5883。
2. [exp002a.yaml，第3–7行](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/config/exp002/exp002a.yaml#L3-L7)导入anysleep_config与usleep_split.yaml。配置blob分别为98b782e139feba0505fe923d6aa09aef62d87c6a和55237b55c2666ebc39598b2002215dd2c37cf85d。
3. [AnySleepTrainDataset，第78行](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/base/data/anysleep_train_dataset.py#L78)调用train模式；[BaseDataset，第85–90行](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/base/data/base_dataset.py#L85-L90)分别读取datasets.test/valid/train，因此split YAML不只是注释。
4. 官方[Sleep-Datasets/config/dataset/sedf-sc.yaml，第10–22行](https://github.com/dslaborg/Sleep-Datasets/blob/7eb09c6a1da5c9b1c04f8bcb2f5ccf51fd39c34e/config/dataset/sedf-sc.yaml#L10-L22)指向sleep-edfx/1.0.0的sleep-cassette EDF；PSG与hypnogram用SC4([0-9]{3}).*提取记录ID，用前两位提取参与者。因此SC4001E0 → 001 → subject00。转换配置blob为24c38f8866c521b71f94ba0495c13d1bb31c6609；该转换仓库本次读取提交为7eb09c6a1da5c9b1c04f8bcb2f5ccf51fd39c34e，不冒称这是训练当年的历史转换提交。
5. [usleep_split.yaml，第19691–19693行](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/config/usleep_split.yaml#L19691-L19693)将sedf-sc的001与002列入test；同文件train/sedf-sc第9338–9453行和valid/sedf-sc第18478–18493行都没有001/002。split blob为e5f87799446ed99023060401ffd480e444514854。

公开checkpoint对象：models/anysleep-run1.pth，Git blob 8e545c4af53a7466b0b9df26a3d554ced47ac12d，Git树声明12,763,675字节。实际下载SHA-256和加载结果须以主实验运行manifest为准，本审计不把Git blob SHA当文件SHA-256。

上述证据支持“公开配置中的留出测试参与者”。它尚未独立证明发布checkpoint严格由该配置和这份split产生；完整Hydra运行配置/日志与权重内部metadata尚未在此审计核验。官方README将详细日志移至Zenodo，因此不能把代码声明升级成完整训练链签名证明。

## YASA 0.8.1：默认权重与训练数据库说明

版本固定为[v0.8.1 / f3308ccd712d1889c67b28cc0e956034713496ae](https://github.com/raphaelvallat/yasa/tree/f3308ccd712d1889c67b28cc0e956034713496ae)。

- [该标签staging.py，第103–109行](https://github.com/raphaelvallat/yasa/blob/f3308ccd712d1889c67b28cc0e956034713496ae/src/yasa/staging.py#L103-L109)说明默认分类器以NSRR约3000夜训练/验证，指向作者yasa_classifier训练仓库。源码blob为6aa21ccae114557845c7e691bd744aab4305fe3e。
- [官方训练仓库README，第7行](https://github.com/raphaelvallat/yasa_classifier/blob/42d8300a3f1f5b57333c74a0c815710bafbe00c2/README.md#L7)列明CCSHS、CFS、CHAT、HomePAP、MESA、MrOS、SHHS；名单未包含Sleep-EDF或CAP。训练仓库本次读取提交42d8300a3f1f5b57333c74a0c815710bafbe00c2，README blob为8bf3504bd8bd3ac3f937b3a5632851422463c317。
- [默认模型选择，第428–447行](https://github.com/raphaelvallat/yasa/blob/f3308ccd712d1889c67b28cc0e956034713496ae/src/yasa/staging.py#L428-L447)按EEG/EOG/EMG/人口信息组合选择精确前缀的最高版本权重。v0.8.1源码树的8份默认分类器文件都为lgb_0.5.0。若首轮使用EEG+EOG、无EMG与人口信息，预期文件为clf_eeg+eog_lgb_0.5.0.joblib（Git blob d86cdfe3f37adbcff25356fbdd4b965386852a91）；最终必须保存实际载入路径与SHA-256。
- [YASA 0.8.1在线API说明](https://yasa-sleep.org/generated/yasa.SleepStaging.html)与标签源码训练说明一致。期刊全文站点本次返回Client Challenge，未将其当已读取全文证据。

“公开训练名单不含两库”是数据库层面的证据，不是逐参与者身份去重结果。尚未重新执行作者训练notebook，也未追溯每个joblib的逐夜训练记录。自定义path_to_model或后续换权重必须重新审计，不能沿用这份结论。

## 对首轮报告的具体要求

- AnySleep首晚写“官方公开split中的Sleep-EDF留出测试参与者，用于复现/机制pilot”；不写独立外部数据库验证。
- YASA写“独立算法管线；官方声明的训练数据库未列Sleep-EDF/CAP”，同时保留导联适用性、实际权重和逐记录去重未核验限制。
- 本轮对SC4001E0的扰动位置和幅度如果用于调试/设定扩展规则，该晚即为开发暴露数据；后续不能再包装为冻结方法的独立确认。
- 不因AnySleep是公开预训练模型就假定Sleep-EDF或CAP全部未见；扩大下载前先冻结参与者角色与权重身份。

