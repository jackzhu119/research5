"""Stage C v2: Repaired Full-Night Uncropped 2x2 Oracle Decomposition & Non-Leaky Baselines.

Fixes every flaw identified in stage-c-v1:
1) Zero 112-epoch crop splicing: every oracle arm (recomputed_both, fixedstats, fixedattention,
   fixedboth) executes full-night uncropped forward passes, directly verifying finite
   structural-support invariance (max |d_logit| beyond_support <= 1e-3 without overwriting).
2) Zero clean-night leakage: every blind deployable baseline runs on the perturbed recording
   using ONLY parameters derived from that recording (never clean stats or clean attention).
3) Compares 8 arms over all 20 participants (SC400-SC419 x 3 positions) with full 13-region
   granularity plus blind-detector real-EEG precision/recall telemetry.
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
from stage_b_expansion_v2 import ALL_KINDS, REAL_KINDS, chosen_epoch_for_fraction
from support import anysleep_local_support

STAGE_ID = "stage-c-v2"

DEPLOYABLE_METHODS = [
    ("official_global", False),
    ("clip_winsor_5iqr", False),
    ("local_window_norm_32ep", False),
    ("causal_running_norm", False),
    ("blind_quality_masking", False),
    ("blind_split_restart", False),
    ("blind_channel_reroute", False),
    ("oracle_masking_upper_bound", True),
]


def _pure_csv_or_df(rows: list[dict], path: Path) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def aggregate_stage_c_v2(out_dir: Path, plan: dict) -> dict[str, Any]:
    import pandas as pd

    part_dirs = sorted(p for p in (out_dir / "participants").glob("SC*") if (p / "status.json").is_file())
    oracle_rows, method_rows, clean_rows, logit_rows, det_rows = [], [], [], [], []
    complete_ids = []
    for pdir in part_dirs:
        st = read_json(pdir / "status.json", {})
        if st.get("status") != "COMPLETE":
            continue
        complete_ids.append(st["participant"])
        oracle_rows.extend(read_json(pdir / "oracle_2x2_positions.json", []))
        method_rows.extend(read_json(pdir / "methods_positions.json", []))
        clean_rows.extend(read_json(pdir / "methods_clean_metrics.json", []))
        logit_rows.extend(read_json(pdir / "oracle_logit_diagnostics.json", []))
        det_rows.extend(read_json(pdir / "detector_audit.json", []))

    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    seed = int(plan["analysis"]["bootstrap_seed"])
    reps = int(plan["analysis"]["bootstrap_replicates"])
    expected_pos = len(plan["positions"])

    def summarize_paired(rows: list[dict], group_cols: list[str], prefix: str) -> list[dict]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df.to_csv(agg_dir / f"{prefix}_position_metrics.csv", index=False)
        extra_keep = ["is_oracle"] if "is_oracle" in df.columns and "is_oracle" not in group_cols else []
        part_group_cols = ["participant", "cohort_wave", "anysleep_published_split"] + extra_keep + group_cols
        part = df.groupby(part_group_cols, as_index=False).agg(
            positions=("position", "nunique"),
            epochs_sum=("epochs", "sum"),
            new_errors_sum=("new_errors", "sum"),
            corrected_errors_sum=("corrected_errors", "sum"),
            flips_sum=("flips", "sum"),
            new_error_rate=("new_error_rate", "mean"),
            corrected_rate=("corrected_rate", "mean"),
            net_error=("net_error", "mean"),
            flip_rate=("flip_rate", "mean"),
        )
        part["complete_three_positions"] = part["positions"] == expected_pos
        part.to_csv(agg_dir / f"{prefix}_participant_metrics.csv", index=False)

        complete_df = part.loc[part["complete_three_positions"]].copy()
        summary = []
        scopes = [
            ("all_20", complete_df),
            ("v1_initial_10", complete_df.loc[complete_df["cohort_wave"] == "v1_initial_10"]),
            ("v2_replication_10", complete_df.loc[complete_df["cohort_wave"] == "v2_replication_10"]),
            ("split_test", complete_df.loc[complete_df["anysleep_published_split"] == "test"]),
        ]
        for scope_name, sc_df in scopes:
            if sc_df.empty:
                continue
            for keys, sub in sc_df.groupby(extra_keep + group_cols):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                sub = sub.sort_values("participant")
                vals = sub["net_error"].to_numpy()
                lo, hi = participant_bootstrap_ci(vals, seed=seed, replicates=reps)
                row = {"cohort_scope": scope_name, **dict(zip(extra_keep + group_cols, keys))}
                row.update({
                    "n_participants": int(len(vals)),
                    "net_error": float(vals.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "new_error_rate": float(sub["new_error_rate"].mean()),
                    "corrected_rate": float(sub["corrected_rate"].mean()),
                    "flip_rate": float(sub["flip_rate"].mean()),
                    "positive_participants": int((vals > 0).sum()),
                    "negative_participants": int((vals < 0).sum()),
                })
                summary.append(row)
        pd.DataFrame(summary).to_csv(agg_dir / f"{prefix}_group_metrics.csv", index=False)
        return summary

    oracle_summary = summarize_paired(oracle_rows, ["arm", "kind", "region"], "oracle_2x2")
    method_summary = summarize_paired(method_rows, ["method", "kind", "region"], "methods")

    clean_summary = []
    if clean_rows:
        cdf = pd.DataFrame(clean_rows)
        cdf.to_csv(agg_dir / "methods_clean_participant_metrics.csv", index=False)
        for (method, is_orc), sub in cdf.groupby(["method", "is_oracle"]):
            clean_summary.append({
                "method": method,
                "is_oracle": bool(is_orc),
                "n_participants": int(len(sub)),
                "accuracy_mean": float(sub["accuracy"].mean()),
                "macro_f1_mean": float(sub["macro_f1"].mean()),
                "cohen_kappa_mean": float(sub["cohen_kappa"].mean()),
                "flagged_epochs_mean": float(sub["flagged_epochs"].mean()),
                "clean_fpr_per_1000_ep": float(1000.0 * sub["flagged_epochs"].sum() / max(1, sub["scored_epochs"].sum())),
            })
        pd.DataFrame(clean_summary).to_csv(agg_dir / "methods_clean_group_metrics.csv", index=False)

    if logit_rows:
        ldf = pd.DataFrame(logit_rows)
        ldf.to_csv(agg_dir / "oracle_logit_diagnostics.csv", index=False)
        l_summary = ldf.groupby(["arm", "kind"], as_index=False).agg(
            n_instances=("max_abs_logit_delta_beyond_support", "size"),
            max_delta_beyond_support=("max_abs_logit_delta_beyond_support", "max"),
            mean_delta_beyond_support=("max_abs_logit_delta_beyond_support", "mean"),
            max_delta_all=("max_abs_logit_delta_all", "max"),
        )
        l_summary.to_csv(agg_dir / "oracle_logit_summary.csv", index=False)

    det_summary = []
    if det_rows:
        ddf = pd.DataFrame(det_rows)
        ddf.to_csv(agg_dir / "detector_audit_positions.csv", index=False)
        det_summary = ddf.groupby("kind", as_index=False).agg(
            instances=("target_detected", "size"),
            target_recall=("target_detected", "mean"),
            extra_flags_mean=("extra_flags_vs_clean", "mean"),
        ).to_dict(orient="records")
        pd.DataFrame(det_summary).to_csv(agg_dir / "detector_audit_summary.csv", index=False)

    summary = {
        "updated_utc": utc_now(),
        "complete_participants": sorted(complete_ids),
        "n_complete": len(complete_ids),
        "clean_summary": clean_summary,
        "detector_audit_summary": det_summary,
        "oracle_near_30_300s_all20": [
            r for r in oracle_summary if r.get("cohort_scope") == "all_20" and r.get("region") == "near_30_300s"
        ],
        "methods_near_30_300s_all20": [
            r for r in method_summary
            if r.get("cohort_scope") == "all_20" and r.get("region") == "near_30_300s" and r.get("kind") != "noop"
        ],
        "methods_within_local_all20": [
            r for r in method_summary
            if r.get("cohort_scope") == "all_20" and r.get("region") == "within_local_support_untouched" and r.get("kind") != "noop"
        ],
    }
    write_json(agg_dir / "stage_c_v2_summary.json", summary)
    return summary


def run_stage_c_v2(args) -> None:
    from anysleep_no_hydra import AnySleep

    project = Path(args.project_root).resolve()
    plan = read_json(args.manifest) or build_manifest_20()
    base_cfg = plan["base_config"]
    out_dir = project / "results" / STAGE_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"
    state = read_json(state_file, {"stage_id": STAGE_ID, "status": "RUNNING", "participants": {}, "started_utc": utc_now()})
    state["status"] = "RUNNING"
    write_json(state_file, state)

    dev = resolve_device(args.device)
    configure_torch_determinism(dev, seed=int(base_cfg["noise_seed"]), strict=True)
    ckpt = ensure_checkpoint(project, "run1")
    model = AnySleep(path=str(ckpt), sleep_stage_frequency=1).eval().to(dev)
    epoch_samples = 3840

    for rec in plan["participants"]:
        p_id = rec["participant"]
        pdir = out_dir / "participants" / p_id
        prev = read_json(pdir / "status.json", {})
        receipt = prev.get("sha256_receipt", {})
        if (
            prev.get("status") == "COMPLETE"
            and receipt
            and all((pdir / fn).is_file() and sha256_file(pdir / fn) == hx for fn, hx in receipt.items())
        ):
            print(f"[StageC-v2] Verified & skipped completed {p_id}", flush=True)
            continue

        t0 = time.monotonic()
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"[StageC-v2] Running complete uncropped full-night Stage C on {p_id} ({dev})...", flush=True)
        download_verified(rec["psg_url"], project / "data" / rec["psg"], expected_sha256=rec["psg_sha256"])
        download_verified(rec["hypnogram_url"], project / "data" / rec["hypnogram"], expected_sha256=rec["hypnogram_sha256"])

        raw_obj, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
        del raw_obj
        n_epochs = len(truth)
        usable_128 = n_epochs * epoch_samples
        raw_128 = resample_channels_128hz(x, sf)

        # Clean baseline predictions per method (strictly computed on clean recording ONLY)
        z0, stats0 = robust_norm_channelwise(raw_128)
        base_logits, alpha0 = infer_anysleep_full(model, z0, dev)
        rep_logits, _ = infer_anysleep_full(model, z0, dev)
        if not np.allclose(base_logits, rep_logits, rtol=1e-6, atol=1e-7) or not np.array_equal(base_logits.argmax(1), rep_logits.argmax(1)):
            raise RuntimeError(f"Baseline repeatability failed for {p_id}")
        base_logits = base_logits[:n_epochs]
        base_pred_off = base_logits.argmax(axis=1).astype(np.int16)
        del z0, rep_logits

        z_w5_clean, _ = clip_winsor_norm(raw_128, clip_iqr_mult=5.0)
        logits_w5_clean, _ = infer_anysleep_full(model, z_w5_clean, dev)
        del z_w5_clean

        z_loc_clean = local_window_robust_norm(raw_128, window_epochs=32, epoch_samples=epoch_samples)
        logits_loc_clean, _ = infer_anysleep_full(model, z_loc_clean, dev)
        del z_loc_clean

        z_cau_clean = causal_running_robust_norm(raw_128, trailing_epochs=60, epoch_samples=epoch_samples)
        logits_cau_clean, _ = infer_anysleep_full(model, z_cau_clean, dev)
        del z_cau_clean

        bad_ce_clean = blind_detect_bad_channel_epochs_v2(
            raw_128[:, :usable_128], raw_native=x, raw_sf=sf, epoch_samples=epoch_samples
        )
        bad_any_clean = np.any(bad_ce_clean, axis=0)
        z_mask_clean, _, _ = build_masked_input(raw_128, bad_ce_clean, epoch_samples=epoch_samples, guard_samples=32)
        logits_mask_clean, _ = infer_anysleep_full(model, z_mask_clean, dev)
        logits_split_clean = infer_anysleep_split_restart_full(model, z_mask_clean, bad_any_clean, dev, epoch_samples=epoch_samples)
        logits_reroute_clean = infer_anysleep_channel_reroute_full(model, z_mask_clean, bad_ce_clean, dev, epoch_samples=epoch_samples)
        del z_mask_clean
        gc.collect()

        clean_preds = {
            "official_global": base_pred_off,
            "clip_winsor_5iqr": logits_w5_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            "local_window_norm_32ep": logits_loc_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            "causal_running_norm": logits_cau_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            "blind_quality_masking": logits_mask_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            "blind_split_restart": logits_split_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            "blind_channel_reroute": logits_reroute_clean[:n_epochs].argmax(axis=1).astype(np.int16),
            # Unperturbed night has zero known injected corruptions -> identical to official_global clean baseline
            "oracle_masking_upper_bound": base_pred_off,
        }

        clean_rows = []
        for method, is_orc in DEPLOYABLE_METHODS:
            cm = compute_clean_metrics(truth, clean_preds[method])
            is_blind_det = method in ("blind_quality_masking", "blind_split_restart", "blind_channel_reroute")
            cm.update({
                "participant": p_id,
                "record": rec["record"],
                "cohort_wave": rec["cohort_wave"],
                "anysleep_published_split": rec["anysleep_published_split"],
                "method": method,
                "is_oracle": is_orc,
                "flagged_epochs": int(bad_any_clean.sum()) if is_blind_det else 0,
                "flagged_channel_epochs": int(bad_ce_clean.sum()) if is_blind_det else 0,
            })
            clean_rows.append(cm)

        oracle_rows, method_rows, logit_rows, det_rows = [], [], [], []
        zeros_sup = np.zeros(n_epochs, dtype=bool)
        for frac in plan["positions"]:
            chosen = chosen_epoch_for_fraction(truth, frac)
            start = int(chosen * 30 * sf)
            end = start + int(base_cfg["duration_seconds"] * sf)

            # Explicit noop zero-flip control check for all 8 arms
            y_noop = perturb_extended(x, base_cfg["corrupt_channels"], start, end, "noop", sf=sf)
            if not np.array_equal(x, y_noop):
                raise AssertionError(f"Noop modified raw EEG on {p_id}")
            for method, is_orc in DEPLOYABLE_METHODS:
                for r in compute_region_metrics_extended(truth, clean_preds[method], clean_preds[method], sf, start, end, zeros_sup):
                    method_rows.append({
                        "participant": p_id,
                        "record": rec["record"],
                        "cohort_wave": rec["cohort_wave"],
                        "anysleep_published_split": rec["anysleep_published_split"],
                        "position": frac,
                        "method": method,
                        "is_oracle": is_orc,
                        "kind": "noop",
                        **r,
                    })

            for kind in REAL_KINDS:
                y = perturb_extended(
                    x,
                    base_cfg["corrupt_channels"],
                    start,
                    end,
                    kind,
                    sf=sf,
                    seed=int(base_cfg["noise_seed"]),
                    clip_volts=float(base_cfg["clip_volts"]),
                    noise_sd_volts=float(base_cfg["noise_sd_volts"]),
                )
                res1 = resample_channels_128hz(y, sf, raw_resampled_ref=raw_128, changed_channels=base_cfg["corrupt_channels"])
                changed = np.any(raw_128 != res1, axis=0)
                local_support = anysleep_local_support(changed)
                beyond_mask = ~local_support

                # --- True Uncropped Full-Night 2x2 Oracle Decomposition ---
                z1, _ = robust_norm_channelwise(res1)
                recomp_l, alpha1 = infer_anysleep_full(model, z1, dev)
                fz = apply_fixed_stats(res1, stats0)
                fstats_l, _ = infer_anysleep_full(model, fz, dev, fixed_alpha=alpha1)
                fatt_l, _ = infer_anysleep_full(model, z1, dev, fixed_alpha=alpha0)
                fboth_l, _ = infer_anysleep_full(model, fz, dev, fixed_alpha=alpha0)
                recomp_l = recomp_l[:n_epochs]
                fstats_l = fstats_l[:n_epochs]
                fatt_l = fatt_l[:n_epochs]
                fboth_l = fboth_l[:n_epochs]

                if np.any(beyond_mask):
                    fb_beyond = float(np.max(np.abs(fboth_l[beyond_mask] - base_logits[beyond_mask])))
                    if fb_beyond > 1e-3:
                        raise AssertionError(
                            f"True uncropped fixedboth leaked beyond structural support ({fb_beyond:.6f} > 1e-3) on {p_id} p={frac} {kind}"
                        )

                arms = [
                    ("recomputed_both", recomp_l),
                    ("fixedstats", fstats_l),
                    ("fixedattention", fatt_l),
                    ("fixedboth", fboth_l),
                ]
                for arm_name, arm_l in arms:
                    arm_pred = arm_l.argmax(axis=1).astype(np.int16)
                    logit_rows.append({
                        "participant": p_id,
                        "position": frac,
                        "kind": kind,
                        "arm": arm_name,
                        "max_abs_logit_delta_all": float(np.max(np.abs(arm_l - base_logits))),
                        "max_abs_logit_delta_beyond_support": float(np.max(np.abs(arm_l[beyond_mask] - base_logits[beyond_mask]))) if np.any(beyond_mask) else 0.0,
                        "local_support_epochs": int(local_support.sum()),
                    })
                    for r in compute_region_metrics_extended(truth, base_pred_off, arm_pred, sf, start, end, local_support):
                        oracle_rows.append({
                            "participant": p_id,
                            "record": rec["record"],
                            "cohort_wave": rec["cohort_wave"],
                            "anysleep_published_split": rec["anysleep_published_split"],
                            "position": frac,
                            "arm": arm_name,
                            "kind": kind,
                            **r,
                        })
                del z1, fz, recomp_l, fstats_l, fatt_l, fboth_l

                # --- Non-Leaky Deployable Baselines + Oracle Upper Bound ---
                z_w5_p, _ = clip_winsor_norm(res1, clip_iqr_mult=5.0)
                w5_p = infer_anysleep_full(model, z_w5_p, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
                del z_w5_p

                z_loc_p = local_window_robust_norm(res1, window_epochs=32, epoch_samples=epoch_samples)
                loc_p = infer_anysleep_full(model, z_loc_p, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
                del z_loc_p

                z_cau_p = causal_running_robust_norm(res1, trailing_epochs=60, epoch_samples=epoch_samples)
                cau_p = infer_anysleep_full(model, z_cau_p, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
                del z_cau_p

                # Strictly non-leaky blind detection using ONLY perturbed night y / res1!
                bad_ce_pert = blind_detect_bad_channel_epochs_v2(
                    res1[:, :usable_128], raw_native=y, raw_sf=sf, epoch_samples=epoch_samples
                )
                bad_any_pert = np.any(bad_ce_pert, axis=0)
                target_det = bool(bad_ce_pert[0, chosen])
                extra_flags = int(np.sum(bad_ce_pert & ~bad_ce_clean)) - (1 if target_det and not bad_ce_clean[0, chosen] else 0)
                det_rows.append({
                    "participant": p_id,
                    "position": frac,
                    "kind": kind,
                    "chosen_epoch": chosen,
                    "target_detected": target_det,
                    "total_flagged_ch_ep": int(bad_ce_pert.sum()),
                    "extra_flags_vs_clean": int(max(0, extra_flags)),
                })

                z_mask_p, _, _ = build_masked_input(res1, bad_ce_pert, epoch_samples=epoch_samples, guard_samples=32)
                mask_p = infer_anysleep_full(model, z_mask_p, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
                split_p = infer_anysleep_split_restart_full(model, z_mask_p, bad_any_pert, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(axis=1).astype(np.int16)
                reroute_p = infer_anysleep_channel_reroute_full(model, z_mask_p, bad_ce_pert, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(axis=1).astype(np.int16)
                del z_mask_p

                # Oracle masking upper bound (strictly tagged is_oracle=True)
                oracle_ce = np.zeros_like(bad_ce_pert)
                oracle_ce[0, chosen] = True
                z_orc_p, _, _ = build_masked_input(res1, oracle_ce, epoch_samples=epoch_samples, guard_samples=32)
                orc_p = infer_anysleep_full(model, z_orc_p, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
                del z_orc_p, res1, y

                pert_map = {
                    "official_global": [r for r in oracle_rows if r["position"] == frac and r["kind"] == kind and r["arm"] == "recomputed_both"],
                    "clip_winsor_5iqr": w5_p,
                    "local_window_norm_32ep": loc_p,
                    "causal_running_norm": cau_p,
                    "blind_quality_masking": mask_p,
                    "blind_split_restart": split_p,
                    "blind_channel_reroute": reroute_p,
                    "oracle_masking_upper_bound": orc_p,
                }

                for method, is_orc in DEPLOYABLE_METHODS:
                    if method == "official_global":
                        reg_rows = [
                            {k: v for k, v in rr.items() if k not in ("participant", "record", "cohort_wave", "anysleep_published_split", "position", "arm", "kind")}
                            for rr in pert_map["official_global"]
                        ]
                    else:
                        reg_rows = compute_region_metrics_extended(truth, clean_preds[method], pert_map[method], sf, start, end, local_support)
                    for r in reg_rows:
                        method_rows.append({
                            "participant": p_id,
                            "record": rec["record"],
                            "cohort_wave": rec["cohort_wave"],
                            "anysleep_published_split": rec["anysleep_published_split"],
                            "position": frac,
                            "method": method,
                            "is_oracle": is_orc,
                            "kind": kind,
                            **r,
                        })
                gc.collect()

        del x, raw_128
        gc.collect()

        files_to_write = {
            "oracle_2x2_positions.json": oracle_rows,
            "methods_positions.json": method_rows,
            "methods_clean_metrics.json": clean_rows,
            "oracle_logit_diagnostics.json": logit_rows,
            "detector_audit.json": det_rows,
        }
        receipt_out = {}
        for fn, payload in files_to_write.items():
            write_json(pdir / fn, payload)
            receipt_out[fn] = sha256_file(pdir / fn)
        p_status = {
            "status": "COMPLETE",
            "participant": p_id,
            "seconds": round(time.monotonic() - t0, 2),
            "completed_utc": utc_now(),
            "sha256_receipt": receipt_out,
        }
        write_json(pdir / "status.json", p_status)
        state["participants"][p_id] = p_status
        state["updated_utc"] = utc_now()
        write_json(state_file, state)
        aggregate_stage_c_v2(out_dir, plan)
        sync_filesystem()
        print(f"[StageC-v2] Completed {p_id} in {p_status['seconds']:.1f}s", flush=True)

    summary = aggregate_stage_c_v2(out_dir, plan)
    state["status"] = "COMPLETE"
    state["completed_utc"] = utc_now()
    write_json(state_file, state)
    sync_filesystem()
    print("STAGE_C_V2_COMPLETE", summary.get("n_complete"), flush=True)


def selftest() -> None:
    rng = np.random.default_rng(20261117)
    n_epochs, ep_raw, ep_128 = 64, 3000, 3840
    raw = rng.normal(0.0, 2e-5, size=(3, n_epochs * ep_raw)).astype(np.float32)
    kernel = np.ones(7, dtype=np.float32) / 7.0
    for c in range(3):
        raw[c] = np.convolve(raw[c], kernel, mode="same")
    idx_128 = np.clip((np.arange(n_epochs * ep_128) * (3000.0 / 3840.0)).astype(int), 0, raw.shape[1] - 1)
    res_clean = raw[:, idx_128].copy()

    y = perturb_extended(raw, [0], 30 * ep_raw, 31 * ep_raw, "noise", sf=100.0, seed=20261117, noise_sd_volts=1e-4)
    flags = blind_detect_bad_channel_epochs_v2(y[:, idx_128], raw_native=y, raw_sf=100.0, epoch_samples=ep_128)
    assert bool(flags[0, 30])
    z_mask, stats, vmask = build_masked_input(y[:, idx_128], flags, epoch_samples=ep_128)
    assert z_mask.shape == res_clean.shape and np.all(z_mask[:, 30 * ep_128 : 31 * ep_128][0] == 0.0)
    # Non-leaky stats check: masked sample exclusion keeps unflagged mean/scale stable even under huge spike
    z_huge = y[:, idx_128].copy()
    z_huge[0, 30 * ep_128 : 31 * ep_128] = 5.0
    _, stats_huge, _ = build_masked_input(z_huge, flags, epoch_samples=ep_128)
    np.testing.assert_allclose(stats_huge["scale"], stats["scale"], rtol=1e-5)
    assert len(DEPLOYABLE_METHODS) == 8
    print("STAGE_C_V2_SELFTEST_OK")


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
        run_stage_c_v2(a)


if __name__ == "__main__":
    main()
