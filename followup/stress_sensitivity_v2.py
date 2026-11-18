"""Stress & Mechanistic Sensitivity Suite v2 (Experiments 3A-3E over SC400-SC419).

Sub-experiments (all predeclared in manifests/development_manifest_20.json):
- 3A: Recording-span sensitivity (~8h sleep span vs 24h full night, n=20).
- 3B: Duration sweep (10/30/60/120/300s), Noise dose-response (25-500 uV x 3 seeds),
      and Realistic artifact morphologies (mains_60hz, emg_hf_burst, slow_drift_pop) + detector recall.
- 3C: Multi-channel corruption ([0], [0,1], [0,1,2], [2]) + 13-layer SkipAttention delta-alpha profile.
- 3D: Multi-checkpoint stability (run1, run2, run3, 3-checkpoint logit ensemble).
- 3E: Exact epoch lag curve (k=-20..+20) + baseline logit-margin / transition stratification.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "followup") not in sys.path:
    sys.path.insert(0, str(ROOT / "followup"))

from common_v2 import (
    apply_fixed_stats,
    blind_detect_bad_channel_epochs_v2,
    build_masked_input,
    causal_running_robust_norm,
    clip_winsor_norm,
    compute_clean_metrics,
    compute_region_metrics_extended,
    configure_torch_determinism,
    download_verified,
    ensure_checkpoint,
    extract_baseline_margin_diagnostics,
    infer_anysleep_channel_reroute_full,
    infer_anysleep_full,
    infer_anysleep_split_restart_full,
    load_sleep_edf_record,
    local_window_robust_norm,
    participant_bootstrap_ci,
    perturb_extended,
    read_json,
    resample_channels_128hz,
    resolve_device,
    robust_norm_channelwise,
    sha256_file,
    sync_filesystem,
    utc_now,
    write_json,
)
from manifest_v2 import build_manifest_20
from stage_b_expansion_v2 import STAGE_NAMES, chosen_epoch_for_fraction
from support import anysleep_local_support

STAGE_ID = "stress-sensitivity-v2"


def compute_sleep_span_bounds(truth: np.ndarray, margin_minutes: int = 30) -> tuple[int, int]:
    """Predeclared rule: [first_sleep - margin, last_sleep + margin] clamped to valid recording."""
    n_epochs = len(truth)
    margin_ep = int(margin_minutes) * 2
    sleep_idx = np.flatnonzero(np.isin(truth, [1, 2, 3, 4]))
    if len(sleep_idx) == 0:
        return 0, n_epochs
    lo = max(0, int(sleep_idx[0]) - margin_ep)
    hi = min(n_epochs, int(sleep_idx[-1]) + margin_ep + 1)
    if hi - lo < 10:
        return 0, n_epochs
    return lo, hi


def eval_deployable_subset(
    model,
    x: np.ndarray,
    raw_128: np.ndarray,
    sf: float,
    truth: np.ndarray,
    clean_preds: dict[str, np.ndarray],
    corrupt_channels: list[int],
    start: int,
    end: int,
    kind: str,
    seed: int,
    clip_volts: float,
    noise_sd_volts: float,
    dev: str,
    methods: list[str],
    meta: dict[str, Any],
) -> tuple[list[dict], dict[str, Any]]:
    """Run requested deployable methods on a single perturbation instance and return region rows."""
    n_epochs = len(truth)
    usable_128 = n_epochs * 3840
    y = perturb_extended(
        x,
        corrupt_channels,
        start,
        end,
        kind,
        sf=sf,
        seed=seed,
        clip_volts=clip_volts,
        noise_sd_volts=noise_sd_volts,
    )
    res1 = resample_channels_128hz(y, sf, raw_resampled_ref=raw_128, changed_channels=corrupt_channels)
    changed = np.any(raw_128 != res1, axis=0)
    local_support = anysleep_local_support(changed)

    rows: list[dict] = []
    extra: dict[str, Any] = {"local_support": local_support, "res1": res1}
    bad_ce = None
    z_mask = None

    for m in methods:
        if m == "official_global":
            z1, _ = robust_norm_channelwise(res1)
            l1, alpha1 = infer_anysleep_full(model, z1, dev)
            pred = l1[:n_epochs].argmax(1).astype(np.int16)
            extra["off_logits"] = l1[:n_epochs]
            extra["alpha1"] = alpha1
            extra["z1"] = z1
        elif m == "clip_winsor_5iqr":
            zw, _ = clip_winsor_norm(res1, 5.0)
            pred = infer_anysleep_full(model, zw, dev)[0][:n_epochs].argmax(1).astype(np.int16)
        elif m == "local_window_norm_32ep":
            zl = local_window_robust_norm(res1, 32, 3840)
            pred = infer_anysleep_full(model, zl, dev)[0][:n_epochs].argmax(1).astype(np.int16)
        elif m == "causal_running_norm":
            zc = causal_running_robust_norm(res1, 60, epoch_samples=3840)
            pred = infer_anysleep_full(model, zc, dev)[0][:n_epochs].argmax(1).astype(np.int16)
        else:
            if bad_ce is None:
                bad_ce = blind_detect_bad_channel_epochs_v2(
                    res1[:, :usable_128], raw_native=y, raw_sf=sf, epoch_samples=3840
                )
                extra["bad_ce"] = bad_ce
                z_mask, _, _ = build_masked_input(res1, bad_ce, epoch_samples=3840)
            if m == "blind_quality_masking":
                pred = infer_anysleep_full(model, z_mask, dev)[0][:n_epochs].argmax(1).astype(np.int16)
            elif m == "blind_split_restart":
                pred = infer_anysleep_split_restart_full(model, z_mask, np.any(bad_ce, axis=0), dev, 3840)[:n_epochs].argmax(1).astype(np.int16)
            elif m == "blind_channel_reroute":
                pred = infer_anysleep_channel_reroute_full(model, z_mask, bad_ce, dev, 3840)[:n_epochs].argmax(1).astype(np.int16)
            else:
                raise ValueError(f"Unknown method: {m}")

        for r in compute_region_metrics_extended(truth, clean_preds[m], pred, sf, start, end, local_support):
            rows.append({**meta, "method": m, **r})

    return rows, extra


def clean_baselines_dict(model, x: np.ndarray, raw_128: np.ndarray, sf: float, n_epochs: int, dev: str, methods: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    usable_128 = n_epochs * 3840
    z0, stats0 = robust_norm_channelwise(raw_128)
    base_logits, alpha0 = infer_anysleep_full(model, z0, dev)
    base_logits = base_logits[:n_epochs]
    out: dict[str, np.ndarray] = {"official_global": base_logits.argmax(1).astype(np.int16)}
    if "clip_winsor_5iqr" in methods:
        out["clip_winsor_5iqr"] = infer_anysleep_full(model, clip_winsor_norm(raw_128, 5.0)[0], dev)[0][:n_epochs].argmax(1).astype(np.int16)
    if "local_window_norm_32ep" in methods:
        out["local_window_norm_32ep"] = infer_anysleep_full(model, local_window_robust_norm(raw_128, 32, 3840), dev)[0][:n_epochs].argmax(1).astype(np.int16)
    if "causal_running_norm" in methods:
        out["causal_running_norm"] = infer_anysleep_full(model, causal_running_robust_norm(raw_128, 60, epoch_samples=3840), dev)[0][:n_epochs].argmax(1).astype(np.int16)
    if any(m.startswith("blind_") for m in methods):
        bad_ce = blind_detect_bad_channel_epochs_v2(raw_128[:, :usable_128], raw_native=x, raw_sf=sf, epoch_samples=3840)
        z_mask, _, _ = build_masked_input(raw_128, bad_ce, epoch_samples=3840)
        if "blind_quality_masking" in methods:
            out["blind_quality_masking"] = infer_anysleep_full(model, z_mask, dev)[0][:n_epochs].argmax(1).astype(np.int16)
        if "blind_split_restart" in methods:
            out["blind_split_restart"] = infer_anysleep_split_restart_full(model, z_mask, np.any(bad_ce, axis=0), dev, 3840)[:n_epochs].argmax(1).astype(np.int16)
        if "blind_channel_reroute" in methods:
            out["blind_channel_reroute"] = infer_anysleep_channel_reroute_full(model, z_mask, bad_ce, dev, 3840)[:n_epochs].argmax(1).astype(np.int16)
    return out, {"base_logits": base_logits, "stats0": stats0, "alpha0": alpha0}


def sub_is_complete(sub_dir: Path) -> bool:
    st = read_json(sub_dir / "status.json", {})
    receipt = st.get("sha256_receipt", {})
    return (
        st.get("status") == "COMPLETE"
        and bool(receipt)
        and all((sub_dir / fn).is_file() and sha256_file(sub_dir / fn) == hx for fn, hx in receipt.items())
    )


def finish_sub(sub_dir: Path, payload_map: dict[str, Any], elapsed_s: float) -> None:
    sub_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for fn, obj in payload_map.items():
        if fn.endswith(".csv"):
            import pandas as pd
            pd.DataFrame(obj).to_csv(sub_dir / fn, index=False)
        else:
            write_json(sub_dir / fn, obj)
        receipt[fn] = sha256_file(sub_dir / fn)
    write_json(
        sub_dir / "status.json",
        {"status": "COMPLETE", "seconds": round(elapsed_s, 2), "completed_utc": utc_now(), "sha256_receipt": receipt},
    )
    sync_filesystem()


def summarize_paired_csv(rows: list[dict], group_cols: list[str], agg_path: Path, seed: int, reps: int) -> list[dict]:
    import pandas as pd

    if not rows:
        return []
    df = pd.DataFrame(rows)
    part = df.groupby(["participant"] + group_cols, as_index=False).agg(
        new_error_rate=("new_error_rate", "mean"),
        corrected_rate=("corrected_rate", "mean"),
        net_error=("net_error", "mean"),
        flip_rate=("flip_rate", "mean"),
    )
    grp = []
    for keys, sub in part.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = sub.sort_values("participant")["net_error"].to_numpy()
        lo, hi = participant_bootstrap_ci(vals, seed=seed, replicates=reps)
        grp.append({
            **dict(zip(group_cols, keys)),
            "n_participants": int(len(vals)),
            "net_error": float(vals.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "new_error_rate": float(sub["new_error_rate"].mean()),
            "corrected_rate": float(sub["corrected_rate"].mean()),
            "flip_rate": float(sub["flip_rate"].mean()),
            "positive_participants": int((vals > 0).sum()),
        })
    pd.DataFrame(grp).to_csv(agg_path, index=False)
    return grp


def run_stress_sensitivity_v2(args) -> None:
    from anysleep_no_hydra import AnySleep

    project = Path(args.project_root).resolve()
    plan = read_json(args.manifest) or build_manifest_20()
    base_cfg = plan["base_config"]
    sens_cfg = plan["sensitivity_predeclarations"]
    seed_bs = int(plan["analysis"]["bootstrap_seed"])
    reps_bs = int(plan["analysis"]["bootstrap_replicates"])

    out_dir = project / "results" / STAGE_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"
    state = read_json(state_file, {"stage_id": STAGE_ID, "status": "RUNNING", "subexperiments": {}, "started_utc": utc_now()})
    state["status"] = "RUNNING"
    write_json(state_file, state)

    dev = resolve_device(args.device)
    configure_torch_determinism(dev, seed=int(base_cfg["noise_seed"]), strict=True)
    models = {
        ck: AnySleep(path=str(ensure_checkpoint(project, ck)), sleep_stage_frequency=1).eval().to(dev)
        for ck in ("run1", "run2", "run3")
    }
    model = models["run1"]

    # Ensure all 20 verified files are ready cleanly once
    for rec in plan["participants"]:
        download_verified(rec["psg_url"], project / "data" / rec["psg"], expected_sha256=rec["psg_sha256"])
        download_verified(rec["hypnogram_url"], project / "data" / rec["hypnogram"], expected_sha256=rec["hypnogram_sha256"])

    # ------------------------------------------------------------------
    # Sub-experiment 3A: Sleep-Only Span (~8h) vs Full 24h Cassette (n=20)
    # ------------------------------------------------------------------
    sub_3a = out_dir / "subexperiments/3a_sleep_span"
    if not sub_is_complete(sub_3a):
        print("[Stress-v2 3A] Running ~8h sleep-span vs 24h recording sensitivity (n=20)...", flush=True)
        t0 = time.monotonic()
        methods_3a = ["official_global", "local_window_norm_32ep", "causal_running_norm", "blind_quality_masking", "blind_channel_reroute"]
        clean_3a, paired_3a = [], []
        for rec in plan["participants"]:
            p_id = rec["participant"]
            _, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
            lo_ep, hi_ep = compute_sleep_span_bounds(truth, int(sens_cfg["sleep_span_window_margin_minutes"]))
            x_sp = x[:, int(lo_ep * 30 * sf) : int(hi_ep * 30 * sf)].copy()
            truth_sp = truth[lo_ep:hi_ep].copy()
            raw_128_sp = resample_channels_128hz(x_sp, sf)
            c_preds, _ = clean_baselines_dict(model, x_sp, raw_128_sp, sf, len(truth_sp), dev, methods_3a)
            for m in methods_3a:
                cm = compute_clean_metrics(truth_sp, c_preds[m])
                cm.update({"participant": p_id, "span": "sleep_span_8h", "span_epochs": len(truth_sp), "full_epochs": len(truth), "method": m})
                clean_3a.append(cm)
            for frac in plan["positions"]:
                ch_ep = chosen_epoch_for_fraction(truth_sp, frac)
                s_sp, e_sp = int(ch_ep * 30 * sf), int((ch_ep + 1) * 30 * sf)
                for kind in ("flat", "clip", "noise"):
                    rows, _ = eval_deployable_subset(
                        model, x_sp, raw_128_sp, sf, truth_sp, c_preds, base_cfg["corrupt_channels"],
                        s_sp, e_sp, kind, int(base_cfg["noise_seed"]), float(base_cfg["clip_volts"]),
                        float(base_cfg["noise_sd_volts"]), dev, methods_3a,
                        {"participant": p_id, "span": "sleep_span_8h", "position": frac, "kind": kind},
                    )
                    paired_3a.extend(rows)
            del x, x_sp, raw_128_sp
            gc.collect()
        finish_sub(sub_3a, {"clean_3a.csv": clean_3a, "paired_3a.csv": paired_3a}, time.monotonic() - t0)
    else:
        print("[Stress-v2 3A] Verified & skipped.", flush=True)

    # ------------------------------------------------------------------
    # Sub-experiment 3B: Duration Sweep, Noise Dose x3 seeds, Morphologies
    # ------------------------------------------------------------------
    sub_3b = out_dir / "subexperiments/3b_dose_morphologies"
    if not sub_is_complete(sub_3b):
        print("[Stress-v2 3B] Running duration, noise dose, & realistic morphology sweeps (n=20)...", flush=True)
        t0 = time.monotonic()
        methods_dose = ["official_global", "clip_winsor_5iqr", "causal_running_norm", "blind_quality_masking", "blind_channel_reroute"]
        methods_morph = ["official_global", "clip_winsor_5iqr", "local_window_norm_32ep", "causal_running_norm", "blind_quality_masking", "blind_split_restart", "blind_channel_reroute"]
        dur_rows, dose_rows, morph_rows, morph_det_rows, morph_oracle_rows = [], [], [], [], []
        for rec in plan["participants"]:
            p_id = rec["participant"]
            _, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
            n_ep = len(truth)
            raw_128 = resample_channels_128hz(x, sf)
            c_preds, base_aux = clean_baselines_dict(model, x, raw_128, sf, n_ep, dev, methods_morph)

            # 1) Duration sweep at pos=0.50 (10, 30, 60, 120, 300s)
            ch50 = chosen_epoch_for_fraction(truth, 0.5)
            s50 = int(ch50 * 30 * sf)
            usable_raw = int(n_ep * 30 * sf)
            for dur_s in sens_cfg["duration_sweep_seconds"]:
                e_dur = min(usable_raw, s50 + int(dur_s * sf))
                for kind in ("flat", "noise"):
                    rows, _ = eval_deployable_subset(
                        model, x, raw_128, sf, truth, c_preds, base_cfg["corrupt_channels"],
                        s50, e_dur, kind, int(base_cfg["noise_seed"]), float(base_cfg["clip_volts"]),
                        float(base_cfg["noise_sd_volts"]), dev, methods_dose,
                        {"participant": p_id, "duration_s": int(dur_s), "kind": kind},
                    )
                    dur_rows.extend(rows)

            # 2) Noise amplitude dose (25, 50, 100, 200, 500 uV x 3 seeds) at pos=0.50 (30s)
            e50 = s50 + int(30 * sf)
            for amp_v in sens_cfg["noise_sd_sweep_volts"]:
                for nseed in sens_cfg["noise_sweep_seeds"]:
                    rows, _ = eval_deployable_subset(
                        model, x, raw_128, sf, truth, c_preds, base_cfg["corrupt_channels"],
                        s50, e50, "noise", int(nseed), float(base_cfg["clip_volts"]),
                        float(amp_v), dev, methods_dose,
                        {"participant": p_id, "noise_sd_uv": round(amp_v * 1e6, 1), "noise_seed": int(nseed)},
                    )
                    dose_rows.extend(rows)

            # 3) Realistic artifact morphologies across 3 positions + detector recall + 2x2 oracle
            bad_clean = blind_detect_bad_channel_epochs_v2(raw_128[:, : n_ep * 3840], raw_native=x, raw_sf=sf, epoch_samples=3840)
            for frac in plan["positions"]:
                ch_f = chosen_epoch_for_fraction(truth, frac)
                sf_i, ef_i = int(ch_f * 30 * sf), int((ch_f + 1) * 30 * sf)
                for m_kind in sens_cfg["morphologies"]:
                    rows, extra = eval_deployable_subset(
                        model, x, raw_128, sf, truth, c_preds, base_cfg["corrupt_channels"],
                        sf_i, ef_i, m_kind, int(base_cfg["noise_seed"]), float(base_cfg["clip_volts"]),
                        float(base_cfg["noise_sd_volts"]), dev, methods_morph,
                        {"participant": p_id, "position": frac, "kind": m_kind},
                    )
                    morph_rows.extend(rows)
                    bad_p = extra["bad_ce"]
                    t_det = bool(bad_p[0, ch_f])
                    ex_f = int(np.sum(bad_p & ~bad_clean)) - (1 if t_det and not bad_clean[0, ch_f] else 0)
                    morph_det_rows.append({"participant": p_id, "position": frac, "kind": m_kind, "target_detected": t_det, "extra_flags_vs_clean": int(max(0, ex_f))})

                    # Full-night uncropped 2x2 oracle on realistic morphologies
                    z1 = extra["z1"]
                    alpha1 = extra["alpha1"]
                    fz = apply_fixed_stats(extra["res1"], base_aux["stats0"])
                    sup = extra["local_support"]
                    for arm_n, l_mat in [
                        ("recomputed_both", extra["off_logits"]),
                        ("fixedstats", infer_anysleep_full(model, fz, dev, fixed_alpha=alpha1)[0][:n_ep]),
                        ("fixedattention", infer_anysleep_full(model, z1, dev, fixed_alpha=base_aux["alpha0"])[0][:n_ep]),
                        ("fixedboth", infer_anysleep_full(model, fz, dev, fixed_alpha=base_aux["alpha0"])[0][:n_ep]),
                    ]:
                        ap = l_mat.argmax(1).astype(np.int16)
                        for r in compute_region_metrics_extended(truth, c_preds["official_global"], ap, sf, sf_i, ef_i, sup):
                            morph_oracle_rows.append({"participant": p_id, "position": frac, "kind": m_kind, "arm": arm_n, **r})
            del x, raw_128
            gc.collect()
        finish_sub(
            sub_3b,
            {
                "duration_sweep.csv": dur_rows,
                "noise_dose_sweep.csv": dose_rows,
                "morphologies_deployable.csv": morph_rows,
                "morphologies_detector.csv": morph_det_rows,
                "morphologies_oracle_2x2.csv": morph_oracle_rows,
            },
            time.monotonic() - t0,
        )
    else:
        print("[Stress-v2 3B] Verified & skipped.", flush=True)

    # ------------------------------------------------------------------
    # Sub-experiment 3C: Multi-channel sets & 13-layer Skip Attention
    # ------------------------------------------------------------------
    sub_3c = out_dir / "subexperiments/3c_multichannel_attention"
    if not sub_is_complete(sub_3c):
        print("[Stress-v2 3C] Running multi-channel & 13-layer SkipAttention delta-alpha (n=20)...", flush=True)
        t0 = time.monotonic()
        methods_3c = ["official_global", "blind_quality_masking", "blind_channel_reroute"]
        mc_rows, att_rows = [], []
        for rec in plan["participants"]:
            p_id = rec["participant"]
            _, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
            n_ep = len(truth)
            raw_128 = resample_channels_128hz(x, sf)
            c_preds, base_aux = clean_baselines_dict(model, x, raw_128, sf, n_ep, dev, methods_3c)
            alpha0 = base_aux["alpha0"]
            for frac in plan["positions"]:
                ch_f = chosen_epoch_for_fraction(truth, frac)
                sf_i, ef_i = int(ch_f * 30 * sf), int((ch_f + 1) * 30 * sf)
                for ch_set in sens_cfg["multi_channel_sets"]:
                    ch_tag = "+".join(str(c) for c in ch_set)
                    for kind in ("noise", "flat"):
                        rows, extra = eval_deployable_subset(
                            model, x, raw_128, sf, truth, c_preds, list(ch_set),
                            sf_i, ef_i, kind, int(base_cfg["noise_seed"]), float(base_cfg["clip_volts"]),
                            float(base_cfg["noise_sd_volts"]), dev, methods_3c,
                            {"participant": p_id, "position": frac, "channels": ch_tag, "kind": kind},
                        )
                        mc_rows.extend(rows)
                        if frac == 0.5:
                            alpha1 = extra["alpha1"]
                            for layer_idx in range(len(alpha0)):
                                a0 = np.asarray(alpha0[layer_idx]).reshape(-1)
                                a1 = np.asarray(alpha1[layer_idx]).reshape(-1)
                                da = a1 - a0
                                att_rows.append({
                                    "participant": p_id,
                                    "channels": ch_tag,
                                    "kind": kind,
                                    "skip_layer": layer_idx,
                                    "alpha_clean_ch0": float(a0[0]),
                                    "alpha_clean_ch1": float(a0[1]),
                                    "alpha_clean_ch2": float(a0[2]),
                                    "alpha_pert_ch0": float(a1[0]),
                                    "alpha_pert_ch1": float(a1[1]),
                                    "alpha_pert_ch2": float(a1[2]),
                                    "delta_alpha_ch0": float(da[0]),
                                    "delta_alpha_ch1": float(da[1]),
                                    "delta_alpha_ch2": float(da[2]),
                                    "delta_alpha_l1": float(np.sum(np.abs(da))),
                                })
            del x, raw_128
            gc.collect()
        finish_sub(sub_3c, {"multichannel.csv": mc_rows, "attention_profile.csv": att_rows}, time.monotonic() - t0)
    else:
        print("[Stress-v2 3C] Verified & skipped.", flush=True)

    # ------------------------------------------------------------------
    # Sub-experiment 3D: Multi-checkpoint Stability (run1/2/3/ensemble3)
    # ------------------------------------------------------------------
    sub_3d = out_dir / "subexperiments/3d_multi_checkpoint"
    if not sub_is_complete(sub_3d):
        print("[Stress-v2 3D] Running run1/run2/run3/ensemble3 stability check (n=20)...", flush=True)
        t0 = time.monotonic()
        clean_3d, paired_3d = [], []
        for rec in plan["participants"]:
            p_id = rec["participant"]
            _, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
            n_ep = len(truth)
            raw_128 = resample_channels_128hz(x, sf)
            z0, _ = robust_norm_channelwise(raw_128)
            bad_ce_c = blind_detect_bad_channel_epochs_v2(raw_128[:, : n_ep * 3840], raw_native=x, raw_sf=sf, epoch_samples=3840)
            zm_c, _, _ = build_masked_input(raw_128, bad_ce_c, epoch_samples=3840)

            base_log_ckpt = {ck: infer_anysleep_full(models[ck], z0, dev)[0][:n_ep] for ck in ("run1", "run2", "run3")}
            base_log_ckpt["ensemble3"] = (base_log_ckpt["run1"] + base_log_ckpt["run2"] + base_log_ckpt["run3"]) / 3.0
            base_pred_off = {ck: l.argmax(1).astype(np.int16) for ck, l in base_log_ckpt.items()}

            # Clean reroute predictions per checkpoint + true ensemble3 logit mean
            rer_logs_c = {ck: infer_anysleep_channel_reroute_full(models[ck], zm_c, bad_ce_c, dev, 3840)[:n_ep] for ck in ("run1", "run2", "run3")}
            rer_logs_c["ensemble3"] = (rer_logs_c["run1"] + rer_logs_c["run2"] + rer_logs_c["run3"]) / 3.0
            clean_reroute = {ck: l.argmax(1).astype(np.int16) for ck, l in rer_logs_c.items()}

            for ck in ("run1", "run2", "run3", "ensemble3"):
                cm = compute_clean_metrics(truth, base_pred_off[ck])
                cm.update({"participant": p_id, "checkpoint": ck, "method": "official_global"})
                clean_3d.append(cm)
                cm_r = compute_clean_metrics(truth, clean_reroute[ck])
                cm_r.update({"participant": p_id, "checkpoint": ck, "method": "blind_channel_reroute"})
                clean_3d.append(cm_r)

            for frac in plan["positions"]:
                ch_f = chosen_epoch_for_fraction(truth, frac)
                sf_i, ef_i = int(ch_f * 30 * sf), int((ch_f + 1) * 30 * sf)
                for kind in ("flat", "clip", "noise"):
                    y = perturb_extended(x, base_cfg["corrupt_channels"], sf_i, ef_i, kind, sf=sf, seed=int(base_cfg["noise_seed"]), clip_volts=float(base_cfg["clip_volts"]), noise_sd_volts=float(base_cfg["noise_sd_volts"]))
                    res1 = resample_channels_128hz(y, sf, raw_resampled_ref=raw_128, changed_channels=base_cfg["corrupt_channels"])
                    changed = np.any(raw_128 != res1, axis=0)
                    sup = anysleep_local_support(changed)
                    z1, _ = robust_norm_channelwise(res1)
                    bad_p = blind_detect_bad_channel_epochs_v2(res1[:, : n_ep * 3840], raw_native=y, raw_sf=sf, epoch_samples=3840)
                    zmp, _, _ = build_masked_input(res1, bad_p, epoch_samples=3840)

                    p_logs = {ck: infer_anysleep_full(models[ck], z1, dev)[0][:n_ep] for ck in ("run1", "run2", "run3")}
                    p_logs["ensemble3"] = (p_logs["run1"] + p_logs["run2"] + p_logs["run3"]) / 3.0
                    rer_logs_p = {ck: infer_anysleep_channel_reroute_full(models[ck], zmp, bad_p, dev, 3840)[:n_ep] for ck in ("run1", "run2", "run3")}
                    rer_logs_p["ensemble3"] = (rer_logs_p["run1"] + rer_logs_p["run2"] + rer_logs_p["run3"]) / 3.0
                    for ck in ("run1", "run2", "run3", "ensemble3"):
                        pp = p_logs[ck].argmax(1).astype(np.int16)
                        for r in compute_region_metrics_extended(truth, base_pred_off[ck], pp, sf, sf_i, ef_i, sup):
                            paired_3d.append({"participant": p_id, "position": frac, "checkpoint": ck, "method": "official_global", "kind": kind, **r})
                        rp = rer_logs_p[ck].argmax(1).astype(np.int16)
                        for r in compute_region_metrics_extended(truth, clean_reroute[ck], rp, sf, sf_i, ef_i, sup):
                            paired_3d.append({"participant": p_id, "position": frac, "checkpoint": ck, "method": "blind_channel_reroute", "kind": kind, **r})
            del x, raw_128, z0
            gc.collect()
        finish_sub(sub_3d, {"clean_checkpoints.csv": clean_3d, "paired_checkpoints.csv": paired_3d}, time.monotonic() - t0)
    else:
        print("[Stress-v2 3D] Verified & skipped.", flush=True)

    # ------------------------------------------------------------------
    # Sub-experiment 3E: Exact Lag -20..+20 & Logit Margin / Transition
    # ------------------------------------------------------------------
    sub_3e = out_dir / "subexperiments/3e_lag_margin"
    if not sub_is_complete(sub_3e):
        print("[Stress-v2 3E] Running lag -20..+20 impulse curve & baseline margin stratification (n=20)...", flush=True)
        t0 = time.monotonic()
        lag_rows, margin_rows = [], []
        margin_bins = [("0.0_to_0.5", 0.0, 0.5), ("0.5_to_1.5", 0.5, 1.5), ("1.5_to_3.0", 1.5, 3.0), ("3.0_plus", 3.0, float("inf"))]

        for rec in plan["participants"]:
            p_id = rec["participant"]
            _, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
            n_ep = len(truth)
            raw_128 = resample_channels_128hz(x, sf)
            z0, _ = robust_norm_channelwise(raw_128)
            base_logits, _ = infer_anysleep_full(model, z0, dev)
            base_logits = base_logits[:n_ep]
            is_trans = np.zeros(n_ep, dtype=bool)
            for idx_t in range(n_ep):
                if truth[idx_t] < 0:
                    continue
                if (idx_t > 0 and truth[idx_t - 1] >= 0 and truth[idx_t - 1] != truth[idx_t]) or (
                    idx_t + 1 < n_ep and truth[idx_t + 1] >= 0 and truth[idx_t + 1] != truth[idx_t]
                ):
                    is_trans[idx_t] = True

            for frac in plan["positions"]:
                ch_f = chosen_epoch_for_fraction(truth, frac)
                sf_i, ef_i = int(ch_f * 30 * sf), int((ch_f + 1) * 30 * sf)
                for kind in ("flat", "clip", "noise", "emg_hf_burst", "slow_drift_pop"):
                    y = perturb_extended(x, base_cfg["corrupt_channels"], sf_i, ef_i, kind, sf=sf, seed=int(base_cfg["noise_seed"]), clip_volts=float(base_cfg["clip_volts"]), noise_sd_volts=float(base_cfg["noise_sd_volts"]))
                    res1 = resample_channels_128hz(y, sf, raw_resampled_ref=raw_128, changed_channels=base_cfg["corrupt_channels"])
                    changed = np.any(raw_128 != res1, axis=0)
                    local_sup = anysleep_local_support(changed)
                    z1, _ = robust_norm_channelwise(res1)
                    pert_logits = infer_anysleep_full(model, z1, dev)[0][:n_ep]
                    diag = extract_baseline_margin_diagnostics(base_logits, pert_logits, truth)

                    # Exact lag curve k in [-20..+20], k != 0
                    for k in range(-20, 21):
                        if k == 0:
                            continue
                        e_idx = ch_f + k
                        if 0 <= e_idx < n_ep and truth[e_idx] >= 0:
                            lag_rows.append({
                                "participant": p_id,
                                "position": frac,
                                "kind": kind,
                                "lag_epochs": k,
                                "lag_seconds": k * 30,
                                "flipped": int(diag["flipped"][e_idx]),
                                "new_error": int(diag["new_error"][e_idx]),
                                "corrected": int(diag["corrected"][e_idx]),
                                "net_error": float(diag["net_error_vec"][e_idx]),
                                "max_logit_delta": float(diag["max_logit_delta"][e_idx]),
                            })

                    # Stratify within_local_support_untouched by margin bin + transition status
                    loc_unt = local_sup & (np.arange(n_ep) != ch_f) & (truth >= 0)
                    for b_label, lo_m, hi_m in margin_bins:
                        bm = loc_unt & (diag["margin"] >= lo_m) & (diag["margin"] < hi_m)
                        if np.any(bm):
                            margin_rows.append({
                                "participant": p_id,
                                "position": frac,
                                "kind": kind,
                                "stratum_type": "margin_bin",
                                "stratum": b_label,
                                "epochs": int(bm.sum()),
                                "flip_rate": float(diag["flipped"][bm].mean()),
                                "new_error_rate": float(diag["new_error"][bm].mean()),
                                "corrected_rate": float(diag["corrected"][bm].mean()),
                                "net_error": float(diag["net_error_vec"][bm].mean()),
                                "mean_max_logit_delta": float(diag["max_logit_delta"][bm].mean()),
                            })
                    for tr_lbl, tm in [("transition_boundary", loc_unt & is_trans), ("stable_interior", loc_unt & ~is_trans)]:
                        if np.any(tm):
                            margin_rows.append({
                                "participant": p_id,
                                "position": frac,
                                "kind": kind,
                                "stratum_type": "transition",
                                "stratum": tr_lbl,
                                "epochs": int(tm.sum()),
                                "flip_rate": float(diag["flipped"][tm].mean()),
                                "new_error_rate": float(diag["new_error"][tm].mean()),
                                "corrected_rate": float(diag["corrected"][tm].mean()),
                                "net_error": float(diag["net_error_vec"][tm].mean()),
                                "mean_max_logit_delta": float(diag["max_logit_delta"][tm].mean()),
                            })
            del x, raw_128, z0
            gc.collect()
        finish_sub(sub_3e, {"lag_curve_raw.csv": lag_rows, "margin_transition_raw.csv": margin_rows}, time.monotonic() - t0)
    else:
        print("[Stress-v2 3E] Verified & skipped.", flush=True)

    # ------------------------------------------------------------------
    # Aggregate all 3A-3E tables for reporting & plotting
    # ------------------------------------------------------------------
    import pandas as pd

    summary: dict[str, Any] = {"updated_utc": utc_now(), "n_participants": len(plan["participants"])}

    if (sub_3a / "paired_3a.csv").is_file():
        pd.read_csv(sub_3a / "clean_3a.csv").groupby(["span", "method"], as_index=False).agg(
            n_participants=("participant", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            span_epochs_mean=("span_epochs", "mean"),
        ).to_csv(agg_dir / "3a_sleep_span_clean_summary.csv", index=False)
        rows_p = pd.read_csv(sub_3a / "paired_3a.csv").to_dict(orient="records")
        g3a = summarize_paired_csv(rows_p, ["span", "method", "kind", "region"], agg_dir / "3a_sleep_span_group_metrics.csv", seed_bs, reps_bs)
        summary["3a_sleep_span_near"] = [r for r in g3a if r["region"] == "near_30_300s"]

    if (sub_3b / "duration_sweep.csv").is_file():
        gdur = summarize_paired_csv(pd.read_csv(sub_3b / "duration_sweep.csv").to_dict(orient="records"), ["duration_s", "kind", "method", "region"], agg_dir / "3b_duration_sweep_group.csv", seed_bs, reps_bs)
        gdose = summarize_paired_csv(pd.read_csv(sub_3b / "noise_dose_sweep.csv").to_dict(orient="records"), ["noise_sd_uv", "method", "region"], agg_dir / "3b_noise_dose_group.csv", seed_bs, reps_bs)
        gmorph = summarize_paired_csv(pd.read_csv(sub_3b / "morphologies_deployable.csv").to_dict(orient="records"), ["kind", "method", "region"], agg_dir / "3b_morphologies_group.csv", seed_bs, reps_bs)
        gorc = summarize_paired_csv(pd.read_csv(sub_3b / "morphologies_oracle_2x2.csv").to_dict(orient="records"), ["arm", "kind", "region"], agg_dir / "3b_morphologies_oracle_2x2.csv", seed_bs, reps_bs)
        dmorph = pd.read_csv(sub_3b / "morphologies_detector.csv").groupby("kind", as_index=False).agg(
            instances=("target_detected", "size"),
            target_recall=("target_detected", "mean"),
            extra_flags_mean=("extra_flags_vs_clean", "mean"),
        )
        dmorph.to_csv(agg_dir / "3b_morphologies_detector_summary.csv", index=False)
        summary["3b_duration_near"] = [r for r in gdur if r["region"] == "near_30_300s"]
        summary["3b_noise_dose_near"] = [r for r in gdose if r["region"] == "near_30_300s"]
        summary["3b_morphologies_near"] = [r for r in gmorph if r["region"] == "near_30_300s"]
        summary["3b_morphologies_detector"] = dmorph.to_dict(orient="records")

    if (sub_3c / "multichannel.csv").is_file():
        gmc = summarize_paired_csv(pd.read_csv(sub_3c / "multichannel.csv").to_dict(orient="records"), ["channels", "kind", "method", "region"], agg_dir / "3c_multichannel_group.csv", seed_bs, reps_bs)
        att_df = pd.read_csv(sub_3c / "attention_profile.csv")
        att_grp = att_df.groupby(["channels", "kind", "skip_layer"], as_index=False).agg(
            n_participants=("participant", "nunique"),
            delta_alpha_ch0=("delta_alpha_ch0", "mean"),
            delta_alpha_ch1=("delta_alpha_ch1", "mean"),
            delta_alpha_ch2=("delta_alpha_ch2", "mean"),
            delta_alpha_l1=("delta_alpha_l1", "mean"),
        )
        att_grp.to_csv(agg_dir / "3c_attention_profile_group.csv", index=False)
        summary["3c_multichannel_near"] = [r for r in gmc if r["region"] == "near_30_300s"]

    if (sub_3d / "paired_checkpoints.csv").is_file():
        pd.read_csv(sub_3d / "clean_checkpoints.csv").groupby(["checkpoint", "method"], as_index=False).agg(
            n_participants=("participant", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
        ).to_csv(agg_dir / "3d_checkpoints_clean_group.csv", index=False)
        gck = summarize_paired_csv(pd.read_csv(sub_3d / "paired_checkpoints.csv").to_dict(orient="records"), ["checkpoint", "method", "kind", "region"], agg_dir / "3d_checkpoints_group.csv", seed_bs, reps_bs)
        summary["3d_checkpoints_near"] = [r for r in gck if r["region"] == "near_30_300s"]

    if (sub_3e / "lag_curve_raw.csv").is_file():
        ldf = pd.read_csv(sub_3e / "lag_curve_raw.csv")
        l_part = ldf.groupby(["participant", "kind", "lag_epochs", "lag_seconds"], as_index=False).agg(
            flip_rate=("flipped", "mean"),
            new_error_rate=("new_error", "mean"),
            corrected_rate=("corrected", "mean"),
            net_error=("net_error", "mean"),
            mean_max_logit_delta=("max_logit_delta", "mean"),
        )
        l_grp = l_part.groupby(["kind", "lag_epochs", "lag_seconds"], as_index=False).agg(
            n_participants=("participant", "nunique"),
            flip_rate=("flip_rate", "mean"),
            new_error_rate=("new_error_rate", "mean"),
            corrected_rate=("corrected_rate", "mean"),
            net_error=("net_error", "mean"),
            mean_max_logit_delta=("mean_max_logit_delta", "mean"),
        )
        l_grp.to_csv(agg_dir / "3e_lag_curve_group.csv", index=False)

        mdf = pd.read_csv(sub_3e / "margin_transition_raw.csv")
        m_part = mdf.groupby(["participant", "kind", "stratum_type", "stratum"], as_index=False).agg(
            epochs=("epochs", "sum"),
            flip_rate=("flip_rate", "mean"),
            new_error_rate=("new_error_rate", "mean"),
            corrected_rate=("corrected_rate", "mean"),
            net_error=("net_error", "mean"),
            mean_max_logit_delta=("mean_max_logit_delta", "mean"),
        )
        m_grp = m_part.groupby(["kind", "stratum_type", "stratum"], as_index=False).agg(
            n_participants=("participant", "nunique"),
            flip_rate=("flip_rate", "mean"),
            net_error=("net_error", "mean"),
            mean_max_logit_delta=("mean_max_logit_delta", "mean"),
        )
        m_grp.to_csv(agg_dir / "3e_margin_transition_group.csv", index=False)
        summary["3e_margin_noise"] = m_grp.loc[m_grp["kind"] == "noise"].to_dict(orient="records")

    write_json(agg_dir / "stress_sensitivity_v2_summary.json", summary)
    state["status"] = "COMPLETE"
    state["completed_utc"] = utc_now()
    write_json(state_file, state)
    sync_filesystem()
    print("STRESS_SENSITIVITY_V2_COMPLETE", flush=True)


def selftest() -> None:
    # Verify sleep-span trimming math and margin diagnostics on synthetic timeline
    truth = np.array([-1] * 20 + [0] * 120 + [2] * 400 + [4] * 100 + [0] * 200, dtype=np.int16)
    lo, hi = compute_sleep_span_bounds(truth, margin_minutes=30)
    # First sleep is index 140 -> lo = 140 - 60 = 80; last sleep is index 639 -> hi = 639 + 61 = 700
    assert lo == 80 and hi == 700
    rng = np.random.default_rng(20261117)
    bl = rng.normal(0.0, 1.0, size=(50, 5))
    pl = bl.copy()
    bl[10] = [0.1, 2.0, 1.9, 0.0, 0.0]  # low margin = 0.1
    pl[10] = [0.1, 1.8, 2.1, 0.0, 0.0]  # flips 1 -> 2
    t = np.ones(50, dtype=np.int16)
    d = extract_baseline_margin_diagnostics(bl, pl, t)
    assert abs(float(d["margin"][10]) - 0.1) < 1e-6 and bool(d["flipped"][10]) and bool(d["new_error"][10])
    print("STRESS_SENSITIVITY_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    r.add_argument("--manifest", type=Path, default=ROOT / "manifests/development_manifest_20.json")
    r.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
    else:
        run_stress_sensitivity_v2(a)


if __name__ == "__main__":
    main()
