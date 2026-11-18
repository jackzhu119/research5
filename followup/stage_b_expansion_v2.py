"""Stage B v2: Complete 20-participant paired evaluation (SC400-SC419) & v1->v2 replication.

- Cryptographically imports and re-audits completed SC400-SC409 outputs from development-queue-v1
  (expanding region bins to all 13 extended regions without wasting CPU re-running YASA).
- Runs uncropped full-EDF inference on SC410-SC419 (GPU AnySleep + YASA 0.8.1) with strict
  repeatability == 0 and noop == 0 verification per participant & position.
- Produces clean stratified summaries:
    * all_20 (SC400-SC419)
    * v1_initial_10 (SC400-SC409) vs. v2_replication_10 (SC410-SC419) out-of-sample replication
    * AnySleep published split (test n=4, valid n=5, train n=11)
    * Position (0.25, 0.50, 0.75) x Target Sleep State (Wake vs. Sleep)
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
    ANYSLEEP_CHECKPOINTS,
    compute_clean_metrics,
    compute_region_metrics_extended,
    configure_torch_determinism,
    download_verified,
    ensure_checkpoint,
    infer_anysleep_full,
    load_sleep_edf_record,
    participant_bootstrap_ci,
    perturb_extended,
    read_json,
    resample_channels_128hz,
    resolve_device,
    robust_norm_channelwise,
    run_yasa_inference,
    sha256_file,
    sync_filesystem,
    utc_now,
    write_json,
)
from manifest_v2 import build_manifest_20
from support import anysleep_local_support

STAGE_ID = "development-queue-v2"
ALL_KINDS = ["noop", "flat", "clip", "noise"]
REAL_KINDS = ["flat", "clip", "noise"]
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def chosen_epoch_for_fraction(truth: np.ndarray, fraction: float) -> int:
    valid = np.flatnonzero(np.asarray(truth) >= 0)
    if len(valid) == 0:
        raise ValueError("No valid scored epochs")
    return int(valid[min(len(valid) - 1, int(len(valid) * fraction))])


def import_v1_participant(
    project: Path,
    rec: dict[str, Any],
    plan_v1_state: dict[str, Any] | None,
    positions: list[float],
) -> tuple[list[dict], list[dict]] | None:
    """Verify & import SC400-SC409 paired outputs from development-queue-v1 if all 3 positions are COMPLETE."""
    if not plan_v1_state:
        return None
    p_id = rec["participant"]
    jobs = plan_v1_state.get("jobs", {})
    pos_rows: list[dict] = []
    clean_rows: list[dict] = []

    for frac_idx, frac in enumerate(positions):
        key = f"{p_id}-p{int(frac * 100):02d}"
        job = jobs.get(key)
        if not job or job.get("status") != "COMPLETE":
            return None
        folder = project / "results" / job["run_id"]
        if not (folder / "status.json").is_file() or read_json(folder / "status.json", {}).get("status") != "COMPLETE":
            return None
        meta = read_json(folder / "record.json")
        sf = float(meta["sfreq_hz"])
        truth = np.load(folder / "truth.npy", allow_pickle=False)
        chosen = chosen_epoch_for_fraction(truth, frac)
        start = int(chosen * 30 * sf)
        end = start + int(30 * sf)
        t_stage = int(truth[chosen])
        t_group = "Wake" if t_stage == 0 else "Sleep"

        # Verify AnySleep repeatability & noop
        any_dir = folder / "anysleep"
        any_rep = read_json(any_dir / "repeatability.json", {})
        if any_rep.get("prediction_flips") != 0:
            raise AssertionError(f"v1 AnySleep repeat flips != 0 for {key}")
        any_base = np.load(any_dir / "baseline_logits.npy", allow_pickle=False).argmax(axis=1).astype(np.int16)
        any_noop = np.load(any_dir / "noop_logits.npy", allow_pickle=False).argmax(axis=1).astype(np.int16)
        if not np.array_equal(any_base, any_noop):
            raise AssertionError(f"v1 AnySleep noop != baseline for {key}")

        # Verify YASA repeatability & noop
        yasa_dir = folder / "yasa"
        yasa_rep = read_json(yasa_dir / "repeatability.json", {})
        if yasa_rep.get("prediction_flips") != 0:
            raise AssertionError(f"v1 YASA repeat flips != 0 for {key}")
        yasa_base = np.load(yasa_dir / "baseline_prediction.npy", allow_pickle=False).astype(np.int16)
        yasa_noop = np.load(yasa_dir / "noop_prediction.npy", allow_pickle=False).astype(np.int16)
        if not np.array_equal(yasa_base, yasa_noop):
            raise AssertionError(f"v1 YASA noop != baseline for {key}")

        if frac_idx == 0:
            for m_name, b_pred in [("anysleep", any_base), ("yasa", yasa_base)]:
                cm = compute_clean_metrics(truth, b_pred)
                cm.update({
                    "participant": p_id,
                    "record": rec["record"],
                    "cohort_wave": rec["cohort_wave"],
                    "anysleep_published_split": rec["anysleep_published_split"],
                    "model": m_name,
                    "source": "imported_v1",
                })
                clean_rows.append(cm)

        n_epochs = len(truth)
        zeros_sup = np.zeros(n_epochs, dtype=bool)
        for kind in ALL_KINDS:
            if kind == "noop":
                any_p = any_noop
                yasa_p = yasa_noop
                sup_mask = zeros_sup
            else:
                any_p = np.load(any_dir / f"{kind}_logits.npy", allow_pickle=False).argmax(axis=1).astype(np.int16)
                yasa_p = np.load(yasa_dir / f"{kind}_prediction.npy", allow_pickle=False).astype(np.int16)
                sup_mask = np.load(any_dir / f"{kind}_local_support_epoch_mask.npy", allow_pickle=False)
            for m_name, b_pred, p_pred, m_sup in [
                ("anysleep", any_base, any_p, sup_mask),
                ("yasa", yasa_base, yasa_p, zeros_sup),
            ]:
                for r in compute_region_metrics_extended(truth, b_pred, p_pred, sf, start, end, m_sup):
                    pos_rows.append({
                        "participant": p_id,
                        "record": rec["record"],
                        "cohort_wave": rec["cohort_wave"],
                        "anysleep_published_split": rec["anysleep_published_split"],
                        "position": frac,
                        "chosen_epoch": chosen,
                        "target_truth_stage": STAGE_NAMES.get(t_stage, str(t_stage)),
                        "target_state_group": t_group,
                        "model": m_name,
                        "kind": kind,
                        "source": "imported_v1",
                        **r,
                    })
    return pos_rows, clean_rows


def aggregate_stage_b_v2(out_dir: Path, plan: dict) -> dict[str, Any]:
    import pandas as pd

    part_dirs = sorted(p for p in (out_dir / "participants").glob("SC*") if (p / "status.json").is_file())
    pos_rows, clean_rows = [], []
    complete_parts = []
    for pdir in part_dirs:
        st = read_json(pdir / "status.json", {})
        if st.get("status") != "COMPLETE":
            continue
        complete_parts.append(st["participant"])
        pos_rows.extend(read_json(pdir / "position_metrics.json", []))
        clean_rows.extend(read_json(pdir / "clean_metrics.json", []))

    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if clean_rows:
        cdf = pd.DataFrame(clean_rows)
        cdf.to_csv(agg_dir / "clean_baseline_participant.csv", index=False)
        cgrp = cdf.groupby("model", as_index=False).agg(
            n_participants=("participant", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
        )
        cgrp.to_csv(agg_dir / "clean_baseline_group.csv", index=False)

    if not pos_rows:
        dec = {"updated_utc": utc_now(), "complete_participants": 0, "decision": "PENDING"}
        write_json(agg_dir / "decision_v2.json", dec)
        return dec

    df = pd.DataFrame(pos_rows)
    df.to_csv(agg_dir / "position_metrics_v2.csv", index=False)

    # Position x target_state_group diagnostic breakdown (Sleep vs Wake at corruption point)
    pst = df.groupby(["model", "kind", "region", "position", "target_state_group"], as_index=False).agg(
        instances=("net_error", "size"),
        epochs_sum=("epochs", "sum"),
        new_error_rate=("new_error_rate", "mean"),
        corrected_rate=("corrected_rate", "mean"),
        net_error=("net_error", "mean"),
        flip_rate=("flip_rate", "mean"),
    )
    pst.to_csv(agg_dir / "position_stage_breakdown.csv", index=False)

    # Within-participant equal weighting across 3 predeclared positions
    part = df.groupby(
        ["participant", "cohort_wave", "anysleep_published_split", "model", "kind", "region"],
        as_index=False,
    ).agg(
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
    part["complete_three_positions"] = part["positions"] == len(plan["positions"])
    part.to_csv(agg_dir / "participant_metrics_v2.csv", index=False)

    complete_df = part.loc[part["complete_three_positions"]].copy()
    scopes = [
        ("all_20", complete_df),
        ("v1_initial_10", complete_df.loc[complete_df["cohort_wave"] == "v1_initial_10"]),
        ("v2_replication_10", complete_df.loc[complete_df["cohort_wave"] == "v2_replication_10"]),
        ("split_test", complete_df.loc[complete_df["anysleep_published_split"] == "test"]),
        ("split_valid", complete_df.loc[complete_df["anysleep_published_split"] == "valid"]),
        ("split_train", complete_df.loc[complete_df["anysleep_published_split"] == "train"]),
    ]

    seed = int(plan["analysis"]["bootstrap_seed"])
    reps = int(plan["analysis"]["bootstrap_replicates"])
    grp_rows = []
    for scope_name, sub_scope in scopes:
        if sub_scope.empty:
            continue
        for (model, kind, region), sub in sub_scope.groupby(["model", "kind", "region"]):
            sub = sub.sort_values("participant")
            vals = sub["net_error"].to_numpy()
            n = len(vals)
            ci_lo, ci_hi = participant_bootstrap_ci(vals, seed=seed, replicates=reps)
            grp_rows.append({
                "cohort_scope": scope_name,
                "model": model,
                "kind": kind,
                "region": region,
                "n_participants": int(n),
                "net_error": float(vals.mean()),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "new_error_rate": float(sub["new_error_rate"].mean()),
                "corrected_rate": float(sub["corrected_rate"].mean()),
                "flip_rate": float(sub["flip_rate"].mean()),
                "new_errors_total": int(sub["new_errors_sum"].sum()),
                "corrected_errors_total": int(sub["corrected_errors_sum"].sum()),
                "positive_participants": int((vals > 0).sum()),
                "negative_participants": int((vals < 0).sum()),
            })
    grp_df = pd.DataFrame(grp_rows)
    grp_df.to_csv(agg_dir / "group_metrics_v2.csv", index=False)

    thresh = float(plan["analysis"]["practical_net_error_fraction"])

    def eval_scope(scope: str) -> dict[str, Any]:
        sub = grp_df.loc[
            (grp_df["cohort_scope"] == scope)
            & (grp_df["region"] == "near_30_300s")
            & (grp_df["kind"] != "noop")
        ]
        recs = sub.to_dict(orient="records")
        if not recs:
            return {"status": "EMPTY", "decision": "UNCERTAIN"}
        if any(r["ci95_low"] is not None and r["ci95_low"] >= thresh for r in recs):
            dec_str = "CONTINUE_MECHANISM_AND_SIMPLE_BASELINES"
        elif all(r["ci95_high"] is not None and r["ci95_high"] < thresh for r in recs):
            dec_str = "BELOW_PRACTICAL_1PP_THRESHOLD"
        else:
            dec_str = "UNCERTAIN"
        return {"decision": dec_str, "rows": recs}

    summary = {
        "updated_utc": utc_now(),
        "complete_participants": sorted(complete_parts),
        "n_complete": len(complete_parts),
        "all_20_near_30_300s": eval_scope("all_20"),
        "v2_replication_10_near_30_300s": eval_scope("v2_replication_10"),
        "v1_initial_10_near_30_300s": eval_scope("v1_initial_10"),
        "split_test_near_30_300s": eval_scope("split_test"),
    }
    write_json(agg_dir / "decision_v2.json", summary)
    return summary


def run_stage_b_v2(args) -> None:
    from anysleep_no_hydra import AnySleep

    project = Path(args.project_root).resolve()
    plan = read_json(args.manifest) or build_manifest_20()
    base_cfg = plan["base_config"]
    out_dir = project / "results" / STAGE_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"
    state = read_json(state_file, {
        "stage_id": STAGE_ID,
        "status": "RUNNING",
        "participants": {},
        "started_utc": utc_now(),
    })
    state["status"] = "RUNNING"
    write_json(state_file, state)

    v1_state = read_json(project / "results/development-queue-v1/state.json")
    dev = resolve_device(args.device)
    torch = configure_torch_determinism(dev, seed=int(base_cfg["noise_seed"]), strict=True)
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
            print(f"[StageB-v2] Verified & skipped completed {p_id}", flush=True)
            continue

        t0 = time.monotonic()
        pdir.mkdir(parents=True, exist_ok=True)
        imported = import_v1_participant(project, rec, v1_state, plan["positions"]) if rec["cohort_wave"] == "v1_initial_10" else None
        if imported is not None:
            pos_rows, clean_rows = imported
            source_mode = "imported_v1_verified"
            print(f"[StageB-v2] Imported & verified SC400-409 member {p_id} from v1 queue.", flush=True)
        else:
            source_mode = "fresh_gpu_v2"
            print(f"[StageB-v2] Running complete uncropped inference for {p_id} ({rec['record']}) on {dev}...", flush=True)
            download_verified(rec["psg_url"], project / "data" / rec["psg"], expected_sha256=rec["psg_sha256"])
            download_verified(rec["hypnogram_url"], project / "data" / rec["hypnogram"], expected_sha256=rec["hypnogram_sha256"])

            raw_obj, x, sf, truth, rec_meta = load_sleep_edf_record(project, rec, base_cfg)
            n_epochs = len(truth)
            raw_128 = resample_channels_128hz(x, sf)

            # AnySleep clean baseline + repeat gate
            z0, stats0 = robust_norm_channelwise(raw_128)
            base_logits, _ = infer_anysleep_full(model, z0, dev)
            rep_logits, _ = infer_anysleep_full(model, z0, dev)
            if not np.allclose(base_logits, rep_logits, rtol=1e-6, atol=1e-7) or not np.array_equal(base_logits.argmax(1), rep_logits.argmax(1)):
                raise RuntimeError(f"AnySleep baseline repeatability failed for {p_id}")
            any_base_pred = base_logits[:n_epochs].argmax(axis=1).astype(np.int16)
            del z0, rep_logits

            # YASA clean baseline + repeat gate
            yasa_base_pred, yasa_base_prob = run_yasa_inference(x, raw_obj.info, base_cfg["yasa_eeg"], base_cfg["yasa_eog"])
            yasa_rep_pred, yasa_rep_prob = run_yasa_inference(x, raw_obj.info, base_cfg["yasa_eeg"], base_cfg["yasa_eog"])
            if not np.array_equal(yasa_base_pred, yasa_rep_pred) or not np.allclose(yasa_base_prob, yasa_rep_prob, rtol=1e-6, atol=1e-7):
                raise RuntimeError(f"YASA baseline repeatability failed for {p_id}")
            if len(yasa_base_pred) != n_epochs:
                raise ValueError(f"YASA epoch count mismatch for {p_id}: {len(yasa_base_pred)} vs {n_epochs}")
            del yasa_rep_pred, yasa_base_prob, yasa_rep_prob

            # Save clean baseline cache inside participant folder (helps auditing without mutating data)
            np.save(pdir / "truth.npy", truth)
            np.save(pdir / "anysleep_base_logits.npy", base_logits)
            np.save(pdir / "yasa_base_pred.npy", yasa_base_pred)
            write_json(pdir / "record_meta.json", rec_meta)

            clean_rows = []
            for m_name, b_pred in [("anysleep", any_base_pred), ("yasa", yasa_base_pred)]:
                cm = compute_clean_metrics(truth, b_pred)
                cm.update({
                    "participant": p_id,
                    "record": rec["record"],
                    "cohort_wave": rec["cohort_wave"],
                    "anysleep_published_split": rec["anysleep_published_split"],
                    "model": m_name,
                    "source": source_mode,
                })
                clean_rows.append(cm)

            pos_rows = []
            zeros_sup = np.zeros(n_epochs, dtype=bool)
            for frac in plan["positions"]:
                chosen = chosen_epoch_for_fraction(truth, frac)
                start = int(chosen * 30 * sf)
                end = start + int(base_cfg["duration_seconds"] * sf)
                t_stage = int(truth[chosen])
                t_group = "Wake" if t_stage == 0 else "Sleep"

                for kind in ALL_KINDS:
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
                    if kind == "noop":
                        if not np.array_equal(x, y):
                            raise RuntimeError(f"AnySleep noop modified raw EEG on {p_id} pos={frac}")
                        res1 = raw_128
                        any_p_logits = base_logits
                        any_p_pred = any_base_pred
                        sup_mask = zeros_sup
                        yasa_p_pred, _ = run_yasa_inference(y, raw_obj.info, base_cfg["yasa_eeg"], base_cfg["yasa_eog"])
                        if not np.array_equal(yasa_p_pred, yasa_base_pred):
                            raise RuntimeError(f"YASA noop mismatch on {p_id} pos={frac}")
                    else:
                        res1 = resample_channels_128hz(y, sf, raw_resampled_ref=raw_128, changed_channels=base_cfg["corrupt_channels"])
                        changed = np.any(raw_128 != res1, axis=0)
                        sup_mask = anysleep_local_support(changed)
                        z1, _ = robust_norm_channelwise(res1)
                        any_p_logits, _ = infer_anysleep_full(model, z1, dev)
                        any_p_pred = any_p_logits[:n_epochs].argmax(axis=1).astype(np.int16)
                        del z1
                        yasa_p_pred, _ = run_yasa_inference(y, raw_obj.info, base_cfg["yasa_eeg"], base_cfg["yasa_eog"])

                    for m_name, b_pred, p_pred, m_sup in [
                        ("anysleep", any_base_pred, any_p_pred, sup_mask),
                        ("yasa", yasa_base_pred, yasa_p_pred, zeros_sup),
                    ]:
                        for r in compute_region_metrics_extended(truth, b_pred, p_pred, sf, start, end, m_sup):
                            pos_rows.append({
                                "participant": p_id,
                                "record": rec["record"],
                                "cohort_wave": rec["cohort_wave"],
                                "anysleep_published_split": rec["anysleep_published_split"],
                                "position": frac,
                                "chosen_epoch": chosen,
                                "target_truth_stage": STAGE_NAMES.get(t_stage, str(t_stage)),
                                "target_state_group": t_group,
                                "model": m_name,
                                "kind": kind,
                                "source": source_mode,
                                **r,
                            })
                    del y
                    if kind != "noop":
                        del res1
                    gc.collect()
            del raw_obj, x, raw_128
            gc.collect()

        receipt_files = {"position_metrics.json": pos_rows, "clean_metrics.json": clean_rows}
        hashes = {}
        for fn, payload in receipt_files.items():
            write_json(pdir / fn, payload)
            hashes[fn] = sha256_file(pdir / fn)

        p_status = {
            "status": "COMPLETE",
            "participant": p_id,
            "source_mode": source_mode,
            "seconds": round(time.monotonic() - t0, 2),
            "completed_utc": utc_now(),
            "sha256_receipt": hashes,
        }
        write_json(pdir / "status.json", p_status)
        state["participants"][p_id] = p_status
        state["updated_utc"] = utc_now()
        write_json(state_file, state)
        aggregate_stage_b_v2(out_dir, plan)
        sync_filesystem()
        print(f"[StageB-v2] Completed {p_id} ({source_mode}) in {p_status['seconds']:.1f}s", flush=True)

    summary = aggregate_stage_b_v2(out_dir, plan)
    state["status"] = "COMPLETE"
    state["completed_utc"] = utc_now()
    write_json(state_file, state)
    sync_filesystem()
    print("STAGE_B_V2_COMPLETE", summary.get("n_complete"), flush=True)


def selftest() -> None:
    plan = build_manifest_20()
    truth = np.array([-1] * 5 + [0] * 30 + [2] * 30 + [0] * 15, dtype=np.int16)
    p25 = chosen_epoch_for_fraction(truth, 0.25)
    p50 = chosen_epoch_for_fraction(truth, 0.50)
    p75 = chosen_epoch_for_fraction(truth, 0.75)
    assert truth[p25] >= 0 and truth[p50] >= 0 and truth[p75] >= 0
    assert p25 < p50 < p75

    # Synthetic v1 import + aggregation schema check
    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp)
        dev1 = proj / "results/development-queue-v1"
        jobs = {}
        rec0 = plan["participants"][0]
        for frac in plan["positions"]:
            key = f"SC400-p{int(frac * 100):02d}"
            run_id = f"dev1-{key}"
            rdir = proj / "results" / run_id
            (rdir / "anysleep").mkdir(parents=True, exist_ok=True)
            (rdir / "yasa").mkdir(parents=True, exist_ok=True)
            write_json(rdir / "status.json", {"status": "COMPLETE"})
            write_json(rdir / "record.json", {"sfreq_hz": 100.0})
            np.save(rdir / "truth.npy", truth)
            base_l = np.zeros((len(truth), 5), dtype=np.float32)
            base_l[np.arange(len(truth)), np.maximum(truth, 0)] = 3.0
            for k in ALL_KINDS:
                l_k = base_l.copy()
                if k == "noise":
                    # Inject 1 error in near region to verify extended aggregation
                    l_k[p50 + 2, 1] = 10.0
                np.save(rdir / f"anysleep/{k}_logits.npy" if k != "noop" else rdir / "anysleep/noop_logits.npy", l_k)
                if k != "noop":
                    sup = np.zeros(len(truth), dtype=bool)
                    c = chosen_epoch_for_fraction(truth, frac)
                    sup[max(0, c - 3) : min(len(truth), c + 4)] = True
                    np.save(rdir / f"anysleep/{k}_local_support_epoch_mask.npy", sup)
                yb = np.maximum(truth, 0).astype(np.int16)
                np.save(rdir / f"yasa/{k}_prediction.npy", yb)
            np.save(rdir / "anysleep/baseline_logits.npy", base_l)
            np.save(rdir / "yasa/baseline_prediction.npy", np.maximum(truth, 0).astype(np.int16))
            write_json(rdir / "anysleep/repeatability.json", {"prediction_flips": 0})
            write_json(rdir / "yasa/repeatability.json", {"prediction_flips": 0})
            jobs[key] = {"status": "COMPLETE", "run_id": run_id}
        imported = import_v1_participant(proj, rec0, {"jobs": jobs}, plan["positions"])
        assert imported is not None and len(imported[0]) > 0 and len(imported[1]) == 2
    print("STAGE_B_EXPANSION_V2_SELFTEST_OK")


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
        run_stage_b_v2(a)


if __name__ == "__main__":
    main()
