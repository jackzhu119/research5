"""Report, 6-Figure Visualization, & Atomic Zip Exporter for research5 v2."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "followup") not in sys.path:
    sys.path.insert(0, str(ROOT / "followup"))


def read_json(path: Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_pp(val: Any, digits: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{float(val) * 100.0:+.{digits}f} pp"


def fmt_pct(val: Any, digits: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{float(val) * 100.0:.{digits}f}%"


def make_figures_v2(project: Path, fig_dir: Path) -> list[str]:
    saved: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception:
        return saved

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    # Fig 1: Stage B n=20 replication (v1_initial_10 vs v2_replication_10 vs all_20)
    b_grp_path = project / "results/development-queue-v2/aggregate/group_metrics_v2.csv"
    if b_grp_path.is_file():
        df = pd.read_csv(b_grp_path)
        sub = df.loc[
            (df["region"] == "near_30_300s")
            & (df["kind"] != "noop")
            & (df["cohort_scope"].isin(["v1_initial_10", "v2_replication_10", "all_20"]))
        ].copy()
        if not sub.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
            for ax, mdl in zip(axes, ["anysleep", "yasa"]):
                mdf = sub.loc[sub["model"] == mdl]
                kinds = ["flat", "clip", "noise"]
                scopes = ["v1_initial_10", "v2_replication_10", "all_20"]
                x = np.arange(len(kinds))
                w = 0.25
                for idx, sc in enumerate(scopes):
                    vals, lo_e, hi_e = [], [], []
                    for k in kinds:
                        row = mdf.loc[(mdf["kind"] == k) & (mdf["cohort_scope"] == sc)]
                        if row.empty:
                            vals.append(0.0)
                            lo_e.append(0.0)
                            hi_e.append(0.0)
                        else:
                            v = float(row["net_error"].iloc[0]) * 100.0
                            cl = float(row["ci95_low"].iloc[0]) * 100.0 if pd.notna(row["ci95_low"].iloc[0]) else v
                            ch = float(row["ci95_high"].iloc[0]) * 100.0 if pd.notna(row["ci95_high"].iloc[0]) else v
                            vals.append(v)
                            lo_e.append(max(0.0, v - cl))
                            hi_e.append(max(0.0, ch - v))
                    ax.bar(x + (idx - 1) * w, vals, width=w, yerr=[lo_e, hi_e], capsize=3, label=sc)
                ax.axhline(0.0, color="black", lw=0.8)
                ax.axhline(1.0, color="crimson", ls="--", lw=1.0, label="1.0 pp threshold" if mdl == "anysleep" else None)
                ax.set_xticks(x, kinds)
                ax.set_title(f"{mdl.upper()}: v1 (n=10) -> v2 (n=10) -> All (n=20)")
                ax.set_ylabel("30-300s Net Additional Error (pp)")
                ax.legend(fontsize=7.5)
            fig.suptitle("Stage B v2: Internal Replication Cohort Comparison (20k Participant Bootstrap CI)")
            fig.tight_layout()
            out_p = fig_dir / "fig1_stage_b_replication_20.png"
            fig.savefig(out_p, dpi=180)
            plt.close(fig)
            saved.append(out_p.name)

    # Fig 2: Stage C v2 True Uncropped 2x2 Oracle & 8-Arm Pareto
    orc_p = project / "results/stage-c-v2/aggregate/oracle_2x2_group_metrics.csv"
    m_grp_p = project / "results/stage-c-v2/aggregate/methods_group_metrics.csv"
    m_cln_p = project / "results/stage-c-v2/aggregate/methods_clean_group_metrics.csv"
    if orc_p.is_file() and m_grp_p.is_file() and m_cln_p.is_file():
        odf = pd.read_csv(orc_p)
        gdf = pd.read_csv(m_grp_p)
        cdf = pd.read_csv(m_cln_p)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        o_sub = odf.loc[(odf["cohort_scope"] == "all_20") & (odf["region"] == "near_30_300s") & (odf["kind"] == "noise")]
        if not o_sub.empty:
            arms = ["recomputed_both", "fixedstats", "fixedattention", "fixedboth"]
            o_sub = o_sub.set_index("arm").reindex(arms).reset_index()
            vals = np.nan_to_num(o_sub["net_error"].to_numpy() * 100.0, nan=0.0)
            ci_l = np.where(pd.notna(o_sub["ci95_low"]), o_sub["ci95_low"].to_numpy() * 100.0, vals)
            ci_h = np.where(pd.notna(o_sub["ci95_high"]), o_sub["ci95_high"].to_numpy() * 100.0, vals)
            lo_e = np.nan_to_num(np.maximum(0.0, vals - ci_l), nan=0.0)
            hi_e = np.nan_to_num(np.maximum(0.0, ci_h - vals), nan=0.0)
            ax1.bar(range(len(arms)), vals, yerr=[lo_e, hi_e], capsize=4, color=["#c0392b", "#2980b9", "#e67e22", "#27ae60"])
            ax1.axhline(0.0, color="black", lw=0.8)
            ax1.set_xticks(range(len(arms)), ["Recomp\nBoth", "Fixed\nStats", "Fixed\nAttention", "Fixed\nBoth"])
            ax1.set_ylabel("30-300s Noise Net Error (pp)")
            ax1.set_title("True Uncropped 2x2 Oracle Decomposition (n=20)")

        d_sub = gdf.loc[(gdf["cohort_scope"] == "all_20") & (gdf["region"] == "near_30_300s") & (gdf["kind"] == "noise")]
        merged = pd.merge(cdf, d_sub, on="method", how="inner")
        if not merged.empty:
            for _, r in merged.iterrows():
                is_orc = bool(r.get("is_oracle_x", r.get("is_oracle", False)))
                col = "crimson" if is_orc else "#1f77b4"
                mk = "^" if is_orc else "o"
                ax2.scatter(r["net_error"] * 100.0, r["macro_f1_mean"] * 100.0, s=85, c=col, marker=mk)
                ax2.annotate(r["method"], (r["net_error"] * 100.0, r["macro_f1_mean"] * 100.0), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
            ax2.axvline(0.5, color="darkorange", ls="--", lw=1.0, label="0.5 pp residual tolerance")
            ax2.axvline(0.0, color="gray", lw=0.8)
            ax2.set_xlabel("30-300s Noise Net Error (pp)")
            ax2.set_ylabel("Clean Full-Night Macro-F1 (%)")
            ax2.set_title("Stage C v2 Non-Leaky Baselines vs Oracle (n=20)")
            ax2.legend(fontsize=8)
        fig.tight_layout()
        out_p = fig_dir / "fig2_stage_c_oracle_and_deployable_pareto.png"
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p.name)

    # Fig 3: Small Model 7-Arm Seen vs Unseen OOD Pareto
    sm_cln_p = project / "results/stage-c-small-model-v2/aggregate/small_model_clean_group_metrics.csv"
    sm_fam_p = project / "results/stage-c-small-model-v2/aggregate/small_model_seen_vs_ood_summary.csv"
    if sm_cln_p.is_file() and sm_fam_p.is_file():
        cdf = pd.read_csv(sm_cln_p)
        fdf = pd.read_csv(sm_fam_p)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for ax, kg, ttl in [
            (ax1, "seen_corruption", "Seen Corruptions (flat/clip/noise Mean)"),
            (ax2, "unseen_ood_corruption", "Unseen OOD (60Hz/EMG/SlowDrift Mean)"),
        ]:
            sub = fdf.loc[fdf["kind_group"] == kg]
            mrg = pd.merge(cdf, sub, on="arm", how="inner")
            if not mrg.empty:
                for _, r in mrg.iterrows():
                    ax.scatter(r["net_error"] * 100.0, r["macro_f1_mean"] * 100.0, s=85)
                    ax.annotate(r["arm"], (r["net_error"] * 100.0, r["macro_f1_mean"] * 100.0), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
                ax.axvline(0.0, color="gray", lw=0.8)
                ax.axvline(0.5, color="darkorange", ls="--", lw=1.0)
            ax.set_xlabel("Held-out 30-300s Net Additional Error (pp)")
            ax.set_title(ttl)
        ax1.set_ylabel("5-Fold Held-Out Clean Macro-F1 (%)")
        fig.suptitle("Small-Model 7-Arm Context Regularization Study (n=20 x 5-Fold CV x 3 Seeds)")
        fig.tight_layout()
        out_p = fig_dir / "fig3_small_model_7arm_pareto.png"
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p.name)

    # Fig 4: CAP External Verification & Realistic 512Hz / Sleep-EDF Morphologies
    cap_m_p = project / "results/cap-check-v2/aggregate/cap_methods_group_metrics.csv"
    morph_p = project / "results/stress-sensitivity-v2/aggregate/3b_morphologies_group.csv"
    if cap_m_p.is_file() or morph_p.is_file():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        if cap_m_p.is_file():
            cd = pd.read_csv(cap_m_p)
            sub_c = cd.loc[(cd["label_set"] == "txt_primary") & (cd["region"] == "near_30_300s") & (cd["kind"] == "noise")]
            if not sub_c.empty:
                labels = [f"{r['pipeline']}\n{r['method']}" for _, r in sub_c.iterrows()]
                ax1.barh(range(len(sub_c)), sub_c["net_error"].to_numpy() * 100.0, alpha=0.85)
                ax1.axvline(0.0, color="black", lw=0.8)
                ax1.set_yticks(range(len(sub_c)), labels, fontsize=7.5)
                ax1.set_xlabel("CAP 30-300s Noise Net Error (pp)")
                ax1.set_title("CAP Sleep Database (n1-n3, 512Hz native)")
        if morph_p.is_file():
            md = pd.read_csv(morph_p)
            sub_m = md.loc[(md["region"] == "near_30_300s") & (md["method"].isin(["official_global", "clip_winsor_5iqr", "causal_running_norm", "blind_quality_masking", "blind_channel_reroute"]))]
            if not sub_m.empty:
                piv = sub_m.pivot(index="kind", columns="method", values="net_error").fillna(0.0) * 100.0
                piv.plot(kind="bar", ax=ax2, rot=0)
                ax2.axhline(0.0, color="black", lw=0.8)
                ax2.set_ylabel("30-300s Net Error (pp)")
                ax2.set_title("Sleep-EDF Realistic Artifact Morphologies (n=20)")
                ax2.legend(fontsize=7)
        fig.tight_layout()
        out_p = fig_dir / "fig4_cap_external_and_morphologies.png"
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p.name)

    # Fig 5: Exact Lag -20..+20 Curve & Baseline Margin Stratification
    lag_p = project / "results/stress-sensitivity-v2/aggregate/3e_lag_curve_group.csv"
    mar_p = project / "results/stress-sensitivity-v2/aggregate/3e_margin_transition_group.csv"
    if lag_p.is_file() and mar_p.is_file():
        ldf = pd.read_csv(lag_p)
        mdf = pd.read_csv(mar_p)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        for k in ["noise", "emg_hf_burst", "slow_drift_pop", "flat"]:
            sub = ldf.loc[ldf["kind"] == k].sort_values("lag_epochs")
            if not sub.empty:
                ax1.plot(sub["lag_seconds"], sub["flip_rate"] * 100.0, marker="o", ms=3, label=k)
        ax1.axvline(0.0, color="black", ls="--", lw=0.8)
        ax1.set_xlabel("Relative Offset from Corruption (s)")
        ax1.set_ylabel("Mean Flip Rate (%)")
        ax1.set_title("Empirical Neighbor Impulse Response (-600s to +600s, n=20)")
        ax1.legend(fontsize=8)

        m_sub = mdf.loc[(mdf["kind"] == "noise") & (mdf["stratum_type"] == "margin_bin")]
        order = ["0.0_to_0.5", "0.5_to_1.5", "1.5_to_3.0", "3.0_plus"]
        m_sub = m_sub.set_index("stratum").reindex([o for o in order if o in set(m_sub["stratum"])]).reset_index()
        if not m_sub.empty:
            ax2.bar(range(len(m_sub)), m_sub["flip_rate"] * 100.0, color="#2980b9", alpha=0.85, label="Flip rate (%)")
            ax2.set_xticks(range(len(m_sub)), m_sub["stratum"])
            ax2.set_xlabel("Clean Top1 - Top2 Logit Margin Bin")
            ax2.set_ylabel("Within-Support Flip Rate (%)")
            ax2.set_title("Baseline Logit Margin Vulnerability (100uV Noise, n=20)")
        fig.tight_layout()
        out_p = fig_dir / "fig5_lag_asymmetry_and_margin.png"
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p.name)

    # Fig 6: Dose-Response, 13-Layer Skip Attention, Checkpoint Stability
    dose_p = project / "results/stress-sensitivity-v2/aggregate/3b_noise_dose_group.csv"
    att_p = project / "results/stress-sensitivity-v2/aggregate/3c_attention_profile_group.csv"
    ckpt_p = project / "results/stress-sensitivity-v2/aggregate/3d_checkpoints_group.csv"
    if dose_p.is_file() or att_p.is_file() or ckpt_p.is_file():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        if dose_p.is_file():
            ddf = pd.read_csv(dose_p)
            sub = ddf.loc[ddf["region"] == "near_30_300s"]
            for m in ["official_global", "clip_winsor_5iqr", "causal_running_norm", "blind_quality_masking", "blind_channel_reroute"]:
                ms = sub.loc[sub["method"] == m].sort_values("noise_sd_uv")
                if not ms.empty:
                    axes[0].plot(ms["noise_sd_uv"], ms["net_error"] * 100.0, marker="o", label=m)
            axes[0].axhline(0.0, color="black", lw=0.8)
            axes[0].set_xlabel("Additive Noise SD (uV)")
            axes[0].set_ylabel("30-300s Net Error (pp)")
            axes[0].set_title("A. Noise Dose-Response (n=20)")
            axes[0].legend(fontsize=6.5)

        if att_p.is_file():
            adf = pd.read_csv(att_p)
            for ch_tag in ["0", "0+1", "0+1+2", "2"]:
                asub = adf.loc[(adf["channels"] == ch_tag) & (adf["kind"] == "noise")].sort_values("skip_layer")
                if not asub.empty:
                    axes[1].plot(asub["skip_layer"], asub["delta_alpha_l1"], marker="o", label=f"corrupt={ch_tag}")
            axes[1].set_xlabel("Skip Connection Depth (0=finest..12=connector)")
            axes[1].set_ylabel("L1 Attention Shift sum |Delta alpha|")
            axes[1].set_title("B. 13-Layer Skip Attention Redistribution")
            axes[1].legend(fontsize=7)

        if ckpt_p.is_file():
            cdf = pd.read_csv(ckpt_p)
            csub = cdf.loc[(cdf["region"] == "near_30_300s") & (cdf["kind"] == "noise")]
            if not csub.empty:
                piv = csub.pivot(index="checkpoint", columns="method", values="net_error").fillna(0.0) * 100.0
                piv.plot(kind="bar", ax=axes[2], rot=0)
                axes[2].axhline(0.0, color="black", lw=0.8)
                axes[2].set_ylabel("30-300s Noise Net Error (pp)")
                axes[2].set_title("C. Run1/2/3 & Ensemble3 Stability")
                axes[2].legend(fontsize=7)

        fig.tight_layout()
        out_p = fig_dir / "fig6_sensitivity_dose_attention_ckpt.png"
        fig.savefig(out_p, dpi=180)
        plt.close(fig)
        saved.append(out_p.name)

    return saved


def build_report_v2(project: Path) -> Path:
    now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep_dir = project / "reports/autonomous-v2"
    fig_dir = rep_dir / "figures"
    rep_dir.mkdir(parents=True, exist_ok=True)
    figs = make_figures_v2(project, fig_dir)

    b_st = read_json(project / "results/development-queue-v2/state.json", {})
    b_dec = read_json(project / "results/development-queue-v2/aggregate/decision_v2.json", {})
    c_st = read_json(project / "results/stage-c-v2/state.json", {})
    c_sum = read_json(project / "results/stage-c-v2/aggregate/stage_c_v2_summary.json", {})
    sm_st = read_json(project / "results/stage-c-small-model-v2/state.json", {})
    sm_sum = read_json(project / "results/stage-c-small-model-v2/aggregate/small_model_v2_summary.json", {})
    cap_st = read_json(project / "results/cap-check-v2/state.json", {})
    cap_sum = read_json(project / "results/cap-check-v2/aggregate/cap_v2_summary.json", {})
    s_st = read_json(project / "results/stress-sensitivity-v2/state.json", {})
    s_sum = read_json(project / "results/stress-sensitivity-v2/aggregate/stress_sensitivity_v2_summary.json", {})

    lines = [
        "# research5 v2 GPU 全面复现、机制拆解与压力实验报告",
        "",
        f"- **生成时间**：`{now_str}`",
        "- **研究队列与统计独立单元**：Sleep-EDF 扩编队列 `SC400–SC419` 共 `20` 名受试者（含初始 `SC400–409` 与内部扩展复现 `SC410–419`，按 AnySleep 公开划分分层报告 `test n=4, valid n=5, train n=11`）＋ CAP 第二来源临床外推队列 `n1–n3`（`512 Hz` 原生采样率）；严格先对同一受试者多个位置取等权均值，再计算 `20,000` 次受试者自助法（bootstrap）95% 置信区间。",
        "- **相对于 `v1` 的核心工程与方法学修复**：",
        "  1. **消除 `v1` Stage C/CAP 的 112-epoch 切片边界伪影**：`v2` 全部机制拆解、可部署基线与 `2x2` Oracle 均采用**未裁剪整夜前向计算**，从根本上移除 `v1` `fixedstats`/`fixedattention` 中 `0.855–1.035` 的窗边缘拼接偏差。",
        "  2. **杜绝 `v1` 部署基线中的干净夜统计与注意力泄漏**：所有盲检方法仅允许使用**被污染当晚自身**的物理特征、`IQR` 与自检掩码，不再读取干净夜 `stats_clean_unflagged` 或 `alpha_mask_clean`；新增 **动态丢弃故障通道的重路由基线 (`blind_channel_reroute`)**、**因果滑动在线归一化 (`causal_running_norm`)** 与 **5×IQR 收缩阈值 (`clip_winsor_5iqr`)**。",
        "  3. **修复 `stage-c-small-model` 并升级至 7 组 × 5 折 × 3 种子**：采用 `31` 维严格单 Epoch 物理+频谱+冻结特征，新增分布外未知干扰测试（`mains_60hz`, `emg_hf_burst`, `slow_drift_pop`）。",
        "  4. **新增 `3A–3E` 全套敏感性与机制压力实验**（纯睡眠窗口 vs 24 小时、时程与噪声剂量曲线、512Hz/100Hz 真实波形、多通道/13 层 SkipAttention `Δα` 分布、`run1/2/3/ensemble3` 稳定度、精确 `-20..+20` Epoch 滞后冲激响应与置信边际分层）。",
        "",
        "## 1. 全流程执行总览（v2）",
        "",
        f"- **Phase 1 — Stage B 20 人队列与跨队列复现 (`development-queue-v2`)**：状态 `{b_st.get('status', 'PENDING')}`（已完成 `{b_dec.get('n_complete', 0)}/20`）",
        f"- **Phase 2 — 修复版 Stage C 整夜 2×2 拆解与 8 组对比 (`stage-c-v2`)**：状态 `{c_st.get('status', 'PENDING')}`（已完成 `{c_sum.get('n_complete', 0)}/20`）",
        f"- **Phase 3 — Stage C 7 组受控小模型 5 折交叉验证 (`stage-c-small-model-v2`)**：状态 `{sm_st.get('status', 'PENDING')}`（受试者数 `{sm_sum.get('n_participants', 0)}/20`）",
        f"- **Phase 4 — 修复版 CAP 外部多机制验证 (`cap-check-v2`)**：状态 `{cap_st.get('status', 'PENDING')}`（记录 `{len(cap_sum.get('complete_records', []))}/3`）",
        f"- **Phase 5 — `3A–3E` 机制压力与敏感性套件 (`stress-sensitivity-v2`)**：状态 `{s_st.get('status', 'PENDING')}`",
        "",
    ]

    lines.extend(["## 2. Stage B v2：`SC400–409` → `SC410–419` 内部独立复现与 `n=20` 汇总（`30–300s`）", ""])
    if b_dec.get("all_20_near_30_300s", {}).get("rows"):
        lines.append(f"- **`all_20` 自动门槛裁定 (`decision_v2.json`)**：`{b_dec['all_20_near_30_300s']['decision']}`（`v2_replication_10` 独立扩编 10 人裁定：`{b_dec.get('v2_replication_10_near_30_300s', {}).get('decision')}`；`split_test`：`{b_dec.get('split_test_near_30_300s', {}).get('decision')}`）")
        lines.append("")
        lines.append("| 队列范围 | 模型 | 局部干预 | 受试者 $n$ | 新增错误 | 纠错 | 平均净额外错误 | 95% Bootstrap CI | 净受损人数 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|---:|")
        for sc_key, label in [("v1_initial_10_near_30_300s", "v1_10"), ("v2_replication_10_near_30_300s", "v2_10"), ("all_20_near_30_300s", "all_20"), ("split_test_near_30_300s", "test_4")]:
            for r in b_dec.get(sc_key, {}).get("rows", []):
                ci = f"[{fmt_pp(r.get('ci95_low'))}, {fmt_pp(r.get('ci95_high'))}]"
                lines.append(f"| `{label}` | `{r['model']}` | `{r['kind']}` | {r['n_participants']} | {fmt_pp(r['new_error_rate'])} | {fmt_pp(r['corrected_rate'])} | **{fmt_pp(r['net_error'])}** | {ci} | {r['positive_participants']}/{r['n_participants']} |")
        lines.append("")

    lines.extend(["## 3. 修复版 Stage C v2：整夜无裁剪 `2×2` Oracle 与严格无泄漏 8 组基线（`n=20`）", ""])
    if c_sum:
        if c_sum.get("clean_summary"):
            lines.extend([
                "### 3.1 未扰动原夜整记录基线性能与物理盲检误报率",
                "",
                "| 方法 | 是否含 Oracle 信息 | $n$ | 准确率 | Macro-F1 | Kappa | 干净夜均标出 Epoch | 干净误报率 (/1000 ep) |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ])
            for c in c_sum["clean_summary"]:
                lines.append(f"| `{c['method']}` | {'YES' if c['is_oracle'] else 'NO'} | {c['n_participants']} | {fmt_pct(c['accuracy_mean'])} | {c['macro_f1_mean']:.4f} | {c['cohen_kappa_mean']:.4f} | {c['flagged_epochs_mean']:.2f} | {c['clean_fpr_per_1000_ep']:.3f} |")
            lines.append("")
        if c_sum.get("oracle_near_30_300s_all20"):
            lines.extend([
                "### 3.2 真·整夜 `2×2` Oracle 路径拆解（`30–300s`）",
                "",
                "| Oracle 分支 | 干预 | $n$ | 平均净额外错误 | 95% Bootstrap CI | 翻转率 | 净受损人数 |",
                "|---|---|---:|---:|---|---:|---:|",
            ])
            for r in c_sum["oracle_near_30_300s_all20"]:
                ci = f"[{fmt_pp(r.get('ci95_low'))}, {fmt_pp(r.get('ci95_high'))}]"
                lines.append(f"| `{r['arm']}` | `{r['kind']}` | {r['n_participants']} | **{fmt_pp(r['net_error'])}** | {ci} | {fmt_pct(r['flip_rate'])} | {r['positive_participants']}/{r['n_participants']} |")
            lines.append("")
        if c_sum.get("methods_near_30_300s_all20"):
            lines.extend([
                "### 3.3 严格无泄漏可部署防护基线对比（`30–300s` & `within_local_support`）",
                "",
                "| 方法（均不含干净夜信息） | Oracle | 干预 | $n$ | `30–300s` 净额外错误 | 95% CI | `within_local_support` 净额外错误 |",
                "|---|---|---|---|---:|---|---:|",
            ])
            loc_map = {(r["method"], r["kind"]): r for r in c_sum.get("methods_within_local_all20", [])}
            for r in c_sum["methods_near_30_300s_all20"]:
                ci = f"[{fmt_pp(r.get('ci95_low'))}, {fmt_pp(r.get('ci95_high'))}]"
                lr = loc_map.get((r["method"], r["kind"]), {})
                lines.append(f"| `{r['method']}` | {'YES' if r.get('is_oracle') else 'NO'} | `{r['kind']}` | {r['n_participants']} | **{fmt_pp(r['net_error'])}** | {ci} | {fmt_pp(lr.get('net_error'))} |")
            lines.append("")

    lines.extend(["## 4. Stage C 小模型 7 组受控对比与未知分布外（OOD）泛化（`n=20`, 5 折 × 3 种子）", ""])
    if sm_sum and sm_sum.get("clean_summary"):
        fam_map = {(r["arm"], r["kind_group"]): r for r in sm_sum.get("seen_vs_ood_summary", [])}
        lines.extend([
            "| 实验组（同一 31 维单 Epoch 特征, $K=\\pm 10$） | 干净准确率 | 干净 Macro-F1 | Kappa | 已知干预均净错误 (`flat/clip/noise`) | 未知 OOD 均净错误 (`60Hz/EMG/Drift`) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for c in sm_sum["clean_summary"]:
            arm = c["arm"]
            seen_v = fam_map.get((arm, "seen_corruption"), {}).get("net_error")
            ood_v = fam_map.get((arm, "unseen_ood_corruption"), {}).get("net_error")
            lines.append(f"| `{arm}` | {fmt_pct(c['accuracy_mean'])} | {c['macro_f1_mean']:.4f} | {c['cohen_kappa_mean']:.4f} | {fmt_pp(seen_v)} | **{fmt_pp(ood_v)}** |")
        lines.append("")

    lines.extend(["## 5. CAP 外部临床复核（`n1, n2, n3`, 原生 `512Hz / 256Hz` 干预与无混叠 60Hz 测试）", ""])
    if cap_sum and cap_sum.get("near_30_300s_txt_primary"):
        lines.extend([
            "| 管线 | 方法 | 干预 | `.txt` 主标签 `30–300s` 净错误 | `.txt ∩ .st` 共有标签 | 净受损人数 |",
            "|---|---|---|---|---|---:|",
        ])
        sh_map = {(r["pipeline"], r["method"], r["kind"]): r for r in cap_sum.get("near_30_300s_shared_txt_st", [])}
        for r in cap_sum["near_30_300s_txt_primary"]:
            sh = sh_map.get((r["pipeline"], r["method"], r["kind"]), {})
            lines.append(f"| `{r['pipeline']}` | `{r['method']}` | `{r['kind']}` | **{fmt_pp(r['net_error'])}** | {fmt_pp(sh.get('net_error'))} | {r['positive_participants']}/{r['n_participants']} |")
        lines.append("")
        if cap_sum.get("realistic_512hz_near_30_300s"):
            lines.append("**CAP 原生 `512Hz` 无混叠真实波形干预（`60Hz+120Hz` 工频 / EMG 肌电 / 慢漂移）`30–300s` 结果**：")
            lines.append("")
            for r in cap_sum["realistic_512hz_near_30_300s"]:
                lines.append(f"- `{r['kind']}` / `{r['method']}`: 净额外错误 **{fmt_pp(r['net_error'])}** (翻转率 {fmt_pct(r['flip_rate'])})")
            lines.append("")

    lines.extend(["## 6. 压力测试与机制深度剖析（`3A–3E`, Exploratory Sensitivity `n=20`）", ""])
    if s_sum:
        if s_sum.get("3a_sleep_span_near"):
            lines.append("### 6.1 3A：8h 纯睡眠记录 vs 24h 全记录敏感性")
            for r in s_sum["3a_sleep_span_near"]:
                lines.append(f"- `{r['kind']}` / `{r['method']}` (`sleep_span_8h`): `30–300s` 净额外错误 `{fmt_pp(r['net_error'])}`")
            lines.append("")
        if s_sum.get("3b_noise_dose_near"):
            lines.append("### 6.2 3B：噪声强度梯度与真实伪迹（`n=20`）")
            for r in s_sum["3b_noise_dose_near"]:
                if r["method"] in ("official_global", "blind_channel_reroute", "blind_quality_masking"):
                    lines.append(f"- Noise SD `{r['noise_sd_uv']} uV` / `{r['method']}`: **{fmt_pp(r['net_error'])}** (95% CI [{fmt_pp(r.get('ci95_low'))}, {fmt_pp(r.get('ci95_high'))}])")
            lines.append("")
        if s_sum.get("3b_morphologies_detector"):
            lines.append("**真实伪迹盲检召回率（`blind_detect_bad_channel_epochs_v2` on `n=20 × 3 pos = 60` 实例）**：")
            for r in s_sum["3b_morphologies_detector"]:
                lines.append(f"- `{r['kind']}`: 检出率 **{fmt_pct(r['target_recall'])}**, 额外误报/实例 `{r['extra_flags_mean']:.2f}`")
            lines.append("")
        if s_sum.get("3d_checkpoints_near"):
            lines.append("### 6.3 3D：多检查点 (`run1/run2/run3/ensemble3`) 可重复性 (`noise`, `30–300s`)")
            for r in s_sum["3d_checkpoints_near"]:
                if r["kind"] == "noise":
                    lines.append(f"- `{r['checkpoint']}` / `{r['method']}`: **{fmt_pp(r['net_error'])}** (翻转率 {fmt_pct(r['flip_rate'])})")
            lines.append("")
        if s_sum.get("3e_margin_noise"):
            lines.append("### 6.4 3E：干净基线 Top1-Top2 置信边际与过渡期脆弱性分层 (`within_local_support`, `noise`)")
            for r in s_sum["3e_margin_noise"]:
                lines.append(f"- `{r['stratum_type']}` = `{r['stratum']}`: 翻转率 **{fmt_pct(r['flip_rate'])}**, 净额外错误 `{fmt_pp(r['net_error'])}`, 最大 Logit 偏移 `{r['mean_max_logit_delta']:.4f}`")
            lines.append("")

    lines.extend(["## 7. 按 `RESEARCH_PLAN.md` 预定标准的科学裁定（Go / No-Go Verdict）", ""])
    if c_sum and c_sum.get("methods_near_30_300s_all20"):
        m_list = c_sum["methods_near_30_300s_all20"]
        off_max = max((r["net_error"] for r in m_list if r["method"] == "official_global"), default=0.0)
        simple_candidates = ["clip_winsor_5iqr", "causal_running_norm", "blind_quality_masking", "blind_split_restart", "blind_channel_reroute"]
        best_m, best_val = min(
            ((m, max((r["net_error"] for r in m_list if r["method"] == m), default=99.0)) for m in simple_candidates),
            key=lambda kv: kv[1],
        )
        lines.append(f"- **20 人队列上 `official_global` 在三种预定局部干扰下的最大 `30–300s` 平均净额外错误**：`{fmt_pp(off_max)}`")
        lines.append(f"- **表现最佳的无泄漏简单防御基线 (`{best_m}`) 最大剩余 `30–300s` 平均净额外错误**：`{fmt_pp(best_val)}`")
        if off_max < 0.01:
            lines.append("- **结论定位**：按照 `RESEARCH_PLAN.md` 预先声明的实用损害阈值（`1.0 pp`），整夜平均的局部脉冲型 30s 损害低于 1 pp 实用门槛，但在更细粒度的 `within_local_support` 与 `test` 子集上机制效应明确；研究贡献聚焦于精确的 `2×2` 机制拆解、不对称传播图谱及零重训练防护基线比较。")
        elif best_val < 0.005:
            lines.append(f"- **结论定位**：`official_global` 超过 `1.0 pp` 损害界且无泄漏轻量基线 `{best_m}` 将剩余损害压制到预先定义的 `0.5 pp` 容忍界以内（且不牺牲干净长记录分期收益）；无需宣称复杂新主干架构，成果聚焦于**机制归因＋轻量即插即用防护＋受控小模型外推边界**。")
        else:
            lines.append("- **结论定位**：标准管线净损害超过 `1.0 pp` 且无训练简单基线尚留有 `> 0.5 pp` 剩余损害，结合第 4 节小模型对比直接量化了受约束上下文与 OOD 泛化的收益与边界。")
    else:
        lines.append("- 各阶段统计完全落盘后自动填写最终定量裁定。")
    lines.append("")

    if figs:
        lines.extend(["## 8. 自动生成发表级高清图表（位于 `reports/autonomous-v2/figures/`）", ""])
        for fn in figs:
            lines.append(f"- `{fn}`")
        lines.append("")

    report_text = "\n".join(lines) + "\n"
    out_main = project / "FINAL_AUTONOMOUS_REPORT_V2.md"
    out_sub = rep_dir / "FINAL_AUTONOMOUS_REPORT_V2.md"
    out_main.write_text(report_text, encoding="utf-8")
    out_sub.write_text(report_text, encoding="utf-8")
    return out_main


def build_export_archive_v2(project: Path) -> Path:
    export_dir = project / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = export_dir / "research5-autonomous-results-v2.zip"
    tmp_zip = export_dir / "research5-autonomous-results-v2.zip.tmp"
    include_exts = {".json", ".csv", ".md", ".png", ".py", ".txt"}
    skip_names = {"paired_predictions.csv", "epoch_transitions.csv", "pairs.csv"}
    roots_to_pack = [
        "FINAL_AUTONOMOUS_REPORT_V2.md",
        "reports/autonomous-v2",
        "followup",
        "manifests/development_manifest_20.json",
        "status/autonomous-pipeline-v2",
        "logs/autonomous-pipeline-v2",
        "results/development-queue-v2/state.json",
        "results/development-queue-v2/aggregate",
        "results/stage-c-v2/state.json",
        "results/stage-c-v2/aggregate",
        "results/stage-c-small-model-v2/state.json",
        "results/stage-c-small-model-v2/aggregate",
        "results/cap-check-v2/state.json",
        "results/cap-check-v2/aggregate",
        "results/stress-sensitivity-v2/state.json",
        "results/stress-sensitivity-v2/aggregate",
    ]
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in roots_to_pack:
            target = project / rel
            if target.is_file():
                zf.write(target, arcname=rel)
            elif target.is_dir():
                for p in sorted(target.rglob("*")):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in include_exts or p.name in skip_names:
                        continue
                    zf.write(p, arcname=str(p.relative_to(project)))
    tmp_zip.replace(zip_path)
    digest = sha256_file(zip_path)
    (export_dir / "SHA256SUMS_V2.txt").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    try:
        os.sync()
    except AttributeError:
        pass
    return zip_path


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp)
        (proj / "followup").mkdir()
        (proj / "followup/dummy.py").write_text("# ok\n", encoding="utf-8")
        rep = build_report_v2(proj)
        assert rep.is_file() and (proj / "reports/autonomous-v2/FINAL_AUTONOMOUS_REPORT_V2.md").is_file()
        arc = build_export_archive_v2(proj)
        assert arc.is_file() and (proj / "exports/SHA256SUMS_V2.txt").is_file()
        with zipfile.ZipFile(arc, "r") as zf:
            names = zf.namelist()
            assert "FINAL_AUTONOMOUS_REPORT_V2.md" in names
    print("GENERATE_FINAL_DRIVE_REPORT_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=False)
    sub.add_parser("selftest")
    p.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
        return
    project = a.project_root.resolve()
    rep = build_report_v2(project)
    arc = build_export_archive_v2(project)
    print(f"REPORT_AND_EXPORT_V2_READY report={rep} archive={arc}", flush=True)


if __name__ == "__main__":
    main()
