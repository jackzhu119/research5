# 2026-11-19 Colab 回传快照

下列文件来自 Google Drive 的实际运行回传。采集时各文件的时间、字节数和 SHA-256 见 [snapshot.json](snapshot.json)。快照保存当时已回传的统计表及队列状态。队列会继续向 Drive 写入，GitHub 文件对应这个时间点的采集。

| 队列 | 快照时的进度（北京时间） | 结果 |
| --- | --- | --- |
| v3 Sleep-EDF | L4，09:47 结束，21/21 个任务、22/22 名固定测试受试者 | [主比较](v3-training/head_report.md) · [逐人差值](v3-training/head_subject_contrasts.csv) · [逐人分臂风险](v3-training/head_subject_risk.csv) · [划分](v3-training/sleep_cohort.json) |
| v3 CAP | T4，09:55 结束，14/14 条记录（10 条正常、4 条病理） | [主结果](v3-cap/summary.json) · [逐记录结果](v3-cap/participant_metrics.json) · [组级汇总](v3-cap/group_metrics.json) · [纳排](v3-cap/cohort.json) |
| v2 `pinfix` 重跑 | 20/20 名扩展队列、20/20 名机制分析与 20 名的小模型实验均已完成。10:53 的最新状态为 CAP 验证运行中，已回传首条记录 n1。 | [扩展队列](v2-pinfix/group_metrics_v2.csv) · [整夜机制分析](v2-pinfix/stage_c_v2_summary.json) · [小模型](v2-pinfix/small_model_v2_summary.json) · [CAP 部分回传](v2-pinfix/cap_v2_summary.json) |

v2 `pinfix` 使用旧开发受试者重跑。v3 Sleep-EDF 头部实验使用固定的同来源留出受试者。v3 CAP 用 AnySleep 的固定权重，按正常及病理分别汇总。CAP 的 `cohort.json` 取自锁定清单，保留纳排、导联和比较设置。表里的受试者与记录标识来自公开数据库。 [Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) · [CAP Sleep Database](https://physionet.org/content/capslpdb/1.0.0/)

两个 v3 队列的 Colab 启动包 SHA-256 一致，启动包及各文件的签名已核对。下面可再次核对归档字节，并在装有 NumPy 的环境中用逐受试者表复算 `D = C + ΔS`、主比较的 95% bootstrap 区间和未注入整夜 macro-F1 差值。复算以采集后的汇总表为输入，无须下载 PSG 或再次训练。

```bash
cd evidence/drive-2026-11-19
sha256sum -c SHA256SUMS
python verify_primary.py
```

[论文主线、文献与下一轮验证](../../docs/ANALYSIS_PIVOT_2026-11-19.md)
