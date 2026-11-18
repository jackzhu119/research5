"""CAP Sleep Database v2 External Verification (n1, n2, n3).

Upgrades over v1:
1) True uncropped full-night GPU/CPU inference (no 112-epoch splicing).
2) Full 2x2 oracle decomposition (fixedstats / fixedattention / fixedboth) + beyond-support assert.
3) Strictly non-leaky 8-arm comparison with heterogeneous-rate (512Hz/256Hz) physics blind detector.
4) Unaliased 512Hz realistic perturbation stress test (60Hz+120Hz mains, EMG burst, slow drift/pop).
5) Multi-checkpoint (run1/2/3) & multi-channel (C4-A1, F4-C4, ROC-LOC, dual-EEG) sensitivity on CAP.
"""
from __future__ import annotations

import argparse
import csv
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
    local_window_robust_norm,
    perturb_extended,
    read_json,
    resolve_device,
    robust_norm_channelwise,
    sha256_file,
    sync_filesystem,
    utc_now,
    write_json,
)
from manifest_v2 import build_manifest_20
from stage_b_expansion_v2 import chosen_epoch_for_fraction
from support import anysleep_local_support

CAP_BASE_URL = "https://physionet.org/files/capslpdb/1.0.0/"
CAP_STAGE_ID = "cap-check-v2"
CAP_CHANNELS = ["C4-A1", "F4-C4", "ROC-LOC"]
CAP_RECORDS = {
    "n1": {
        "expected_edf_sha256": "25d4e62a7a90d8245c71c7f168e34756238bb6ac7b1d43dcc75818d0331d43f4",
        "expected_first_offset_s": 210,
        "expected_scored_txt": 1140,
        "expected_full_epochs": 1154,
        "txt_only_offsets_s": [5250, 6300, 11820, 16350, 17280, 21360, 23190, 27150],
    },
    "n2": {
        "expected_edf_sha256": "2bea9302fb7a50a10558cddcb9a6666da55799e051b90fc5f4cd661943b68782",
        "expected_first_offset_s": 11220,
        "expected_scored_txt": 999,
        "expected_full_epochs": 1470,
        "txt_only_offsets_s": [20220, 22710],
    },
    "n3": {
        "expected_edf_sha256": "d3c8e3cace8c2ed6c020ddfdd658ab344fcd0eceeee6b6f12a3d06747cd12ac2",
        "expected_first_offset_s": 3030,
        "expected_scored_txt": 999,
        "expected_full_epochs": 1102,
        "txt_only_offsets_s": [11490, 16260],
    },
}
STAGE_MAP = {"SLEEP-S0": 0, "SLEEP-S1": 1, "SLEEP-S2": 2, "SLEEP-S3": 3, "SLEEP-S4": 3, "SLEEP-REM": 4}

CAP_METHODS = [
    ("official_global", False),
    ("clip_winsor_5iqr", False),
    ("local_window_norm_32ep", False),
    ("causal_running_norm", False),
    ("blind_quality_masking", False),
    ("blind_split_restart", False),
    ("blind_channel_reroute", False),
    ("oracle_masking_upper_bound", True),
]


def parse_hms_seconds(hms: str) -> int:
    sep = ":" if ":" in hms else "."
    parts = [int(p) for p in hms.strip().split(sep)]
    if len(parts) != 3:
        raise ValueError(f"Malformed clock string: {hms!r}")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_cap_txt_labels(txt_path: Path, edf_clock_hms: str, total_duration_s: int, spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    edf_start_s = parse_hms_seconds(edf_clock_hms)
    n_epochs = total_duration_s // 30
    truth_txt = np.full(n_epochs, -1, dtype=np.int16)
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if "Sleep Stage" in line and "Event" in line and "Duration" in line), None)
    if header_idx is None:
        raise ValueError(f"Could not locate TSV header in {txt_path}")
    reader = csv.DictReader(lines[header_idx:], delimiter="\t")
    time_col = next(c for c in reader.fieldnames if "Time" in c)
    event_col = next(c for c in reader.fieldnames if c.strip() == "Event")
    dur_col = next(c for c in reader.fieldnames if "Duration" in c)

    prev_rel_s = None
    day_offset = 0
    prev_clock = None
    first_offset_s = None
    stage_events = 0
    for row in reader:
        ev = (row.get(event_col) or "").strip()
        if ev not in STAGE_MAP:
            continue
        dur = int(float((row.get(dur_col) or "0").strip()))
        if dur != 30:
            raise ValueError(f"Unexpected non-30s stage event ({dur}s) in {txt_path}: {row}")
        clk = parse_hms_seconds((row.get(time_col) or "").strip())
        if prev_clock is not None and clk < prev_clock:
            day_offset += 86400
        elif prev_clock is None and clk < edf_start_s:
            day_offset += 86400
        prev_clock = clk
        rel_s = clk + day_offset - edf_start_s
        if rel_s < 0 or rel_s + 30 > total_duration_s:
            raise ValueError(f"Stage event outside EDF duration: rel_s={rel_s}, total={total_duration_s}")
        if rel_s % 30 != 0:
            raise ValueError(f"Stage onset not on 30s boundary relative to EDF start: rel_s={rel_s}")
        if prev_rel_s is not None and rel_s <= prev_rel_s:
            raise ValueError(f"Non-increasing stage timestamps: {prev_rel_s} -> {rel_s}")
        prev_rel_s = rel_s
        if first_offset_s is None:
            first_offset_s = rel_s
        ep = rel_s // 30
        if truth_txt[ep] != -1:
            raise ValueError(f"Duplicate stage assignment at epoch {ep}")
        truth_txt[ep] = STAGE_MAP[ev]
        stage_events += 1

    if first_offset_s != spec["expected_first_offset_s"]:
        raise AssertionError(f"First stage offset {first_offset_s} != audited {spec['expected_first_offset_s']}")
    if n_epochs != spec["expected_full_epochs"]:
        raise AssertionError(f"Full epoch count {n_epochs} != audited {spec['expected_full_epochs']}")
    if stage_events != spec["expected_scored_txt"]:
        raise AssertionError(f"Scored .txt count {stage_events} != audited {spec['expected_scored_txt']}")

    truth_shared = truth_txt.copy()
    for off_s in spec["txt_only_offsets_s"]:
        ep = off_s // 30
        if truth_shared[ep] == -1:
            raise AssertionError(f"Audited txt-only offset {off_s}s was already -1")
        truth_shared[ep] = -1

    meta = {
        "edf_start_clock": edf_clock_hms,
        "first_stage_offset_seconds": first_offset_s,
        "full_30s_epochs": n_epochs,
        "tail_unaligned_seconds": total_duration_s - n_epochs * 30,
        "scored_epochs_txt": int((truth_txt >= 0).sum()),
        "scored_epochs_shared_txt_st": int((truth_shared >= 0).sum()),
        "unscored_epochs_txt": int((truth_txt < 0).sum()),
    }
    return truth_txt, truth_shared, meta


def load_cap_native_channels(edf_path: Path, channels: list[str]) -> tuple[list[np.ndarray], list[float], str, int]:
    import mne

    with edf_path.open("rb") as f:
        hdr = f.read(256)
    edf_clock = hdr[176:184].decode("ascii")
    n_records = int(hdr[236:244].decode("ascii").strip())
    rec_dur = float(hdr[244:252].decode("ascii").strip())
    total_duration_s = int(round(n_records * rec_dur))
    native_signals, native_sfs = [], []
    for ch in channels:
        raw_ch = mne.io.read_raw_edf(edf_path, include=[ch], preload=True, verbose="error")
        if raw_ch.ch_names != [ch]:
            raise ValueError(f"Channel {ch} missing in {edf_path}; got {raw_ch.ch_names}")
        sig = raw_ch.get_data()[0].astype(np.float32)
        sf = float(raw_ch.info["sfreq"])
        del raw_ch
        gc.collect()
        if not np.isfinite(sig).all():
            raise ValueError(f"Non-finite samples in {edf_path}:{ch}")
        native_signals.append(sig)
        native_sfs.append(sf)
    return native_signals, native_sfs, edf_clock, total_duration_s


def resample_channels_to_128hz(signals: list[np.ndarray], sfs: list[float], n_epochs: int) -> np.ndarray:
    from scipy.signal import resample_poly

    target_len = n_epochs * 3840
    rows = []
    for sig, sf in zip(signals, sfs):
        sf_int = int(round(sf))
        usable_native = n_epochs * 30 * sf_int
        seg = sig[:usable_native]
        z = (resample_poly(seg, 128, sf_int) if sf_int != 128 else seg.copy()).astype(np.float32)
        if len(z) != target_len:
            raise ValueError(f"Resampled length {len(z)} != {target_len}")
        rows.append(z)
    return np.stack(rows, axis=0)


def detect_cap_bad_channel_epochs(
    z_128: np.ndarray,
    native_signals: list[np.ndarray],
    native_sfs: list[float],
    epoch_samples: int = 3840,
) -> np.ndarray:
    """Apply physics-grounded blind detector per channel with its own native sampling rate."""
    flags = []
    for c in range(z_128.shape[0]):
        n_ep = z_128.shape[1] // epoch_samples
        sf_int = int(round(native_sfs[c]))
        nat_len = n_ep * 30 * sf_int
        nat = native_signals[c][:nat_len][None, :]
        fc = blind_detect_bad_channel_epochs_v2(
            z_128[c : c + 1], raw_native=nat, raw_sf=float(sf_int), epoch_samples=epoch_samples
        )
        flags.append(fc[0])
    return np.stack(flags, axis=0)


def perturb_and_resample_cap(
    z0_128: np.ndarray,
    signals: list[np.ndarray],
    sfs: list[float],
    n_epochs: int,
    channels_to_corrupt: list[int],
    chosen_ep: int,
    kind: str,
    seed: int = 20261117,
    clip_volts: float = 2e-5,
    noise_sd_volts: float = 1e-4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Perturb native CAP channel(s) strictly at [chosen_ep] before 128Hz resampling."""
    from scipy.signal import resample_poly

    z1_128 = z0_128.copy()
    nat_list = list(signals)
    for c in channels_to_corrupt:
        sf_c = int(round(sfs[c]))
        usable = n_epochs * 30 * sf_c
        seg = signals[c][:usable][None, :].astype(np.float64)
        s0 = chosen_ep * 30 * sf_c
        e0 = (chosen_ep + 1) * 30 * sf_c
        y_c = perturb_extended(
            seg,
            [0],
            s0,
            e0,
            kind,
            sf=float(sf_c),
            seed=seed,
            clip_volts=clip_volts,
            noise_sd_volts=noise_sd_volts,
        )[0].astype(np.float32)
        nat_list[c] = y_c
        res_c = (resample_poly(y_c, 128, sf_c) if sf_c != 128 else y_c.copy()).astype(np.float32)
        z1_128[c] = res_c[: n_epochs * 3840]
    return z1_128, nat_list


def run_yasa_cap(z_128_c0_c2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mne
    import yasa

    info = mne.create_info(ch_names=["C4-A1", "ROC-LOC"], sfreq=128.0, ch_types=["eeg", "eog"])
    raw = mne.io.RawArray(z_128_c0_c2.astype(np.float64), info, verbose=False)
    sls = yasa.SleepStaging(raw, eeg_name="C4-A1", eog_name="ROC-LOC")
    hyp = sls.predict()
    if hasattr(hyp, "hypno"):
        labels = np.asarray(hyp.hypno).astype(str)
        proba = hyp.proba.to_numpy()
    else:
        labels = np.asarray(hyp).astype(str)
        proba = sls.predict_proba().to_numpy()
    del raw, sls
    lmap = {"WAKE": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
    pred = np.asarray([lmap[s.upper()] for s in labels], dtype=np.int16)
    return pred, np.asarray(proba, dtype=np.float64)


def aggregate_cap_v2(out_dir: Path) -> dict[str, Any]:
    import pandas as pd

    all_clean, all_paired, all_oracle, all_real, all_sens = [], [], [], [], []
    complete_recs = []
    for rdir in sorted((out_dir / "records").glob("n*")):
        st = read_json(rdir / "status.json", {})
        if st.get("status") != "COMPLETE":
            continue
        complete_recs.append(rdir.name)
        all_clean.extend(read_json(rdir / "clean_metrics.json", []))
        all_paired.extend(read_json(rdir / "paired_positions.json", []))
        all_oracle.extend(read_json(rdir / "oracle_2x2_positions.json", []))
        all_real.extend(read_json(rdir / "realistic_512hz_positions.json", []))
        all_sens.extend(read_json(rdir / "sensitivity_ckpt_chan.json", []))

    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    if all_clean:
        cdf = pd.DataFrame(all_clean)
        cdf.to_csv(agg_dir / "cap_clean_metrics.csv", index=False)
        cdf.groupby(["pipeline", "method"], as_index=False).agg(
            records=("record", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            cohen_kappa_mean=("cohen_kappa", "mean"),
            flagged_epochs_mean=("flagged_epochs", "mean"),
        ).to_csv(agg_dir / "cap_clean_group_metrics.csv", index=False)

    def summarize_df(rows: list[dict], cols: list[str], fname: str) -> list[dict]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        df.to_csv(agg_dir / f"{fname}_position_metrics.csv", index=False)
        part = df.groupby(["participant"] + cols, as_index=False).agg(
            positions=("position", "nunique"),
            new_error_rate=("new_error_rate", "mean"),
            corrected_rate=("corrected_rate", "mean"),
            net_error=("net_error", "mean"),
            flip_rate=("flip_rate", "mean"),
        )
        part.to_csv(agg_dir / f"{fname}_participant_metrics.csv", index=False)
        grp = part.groupby(cols, as_index=False).agg(
            n_participants=("participant", "nunique"),
            net_error=("net_error", "mean"),
            new_error_rate=("new_error_rate", "mean"),
            corrected_rate=("corrected_rate", "mean"),
            flip_rate=("flip_rate", "mean"),
            positive_participants=("net_error", lambda s: int((s > 0).sum())),
        )
        grp.to_csv(agg_dir / f"{fname}_group_metrics.csv", index=False)
        return grp.to_dict(orient="records")

    p_grp = summarize_df(all_paired, ["label_set", "pipeline", "method", "is_oracle", "kind", "region"], "cap_methods")
    o_grp = summarize_df(all_oracle, ["arm", "kind", "region"], "cap_oracle_2x2")
    r_grp = summarize_df(all_real, ["method", "kind", "region"], "cap_realistic_512hz")
    s_grp = summarize_df(all_sens, ["sweep_type", "setting", "method", "region"], "cap_sensitivity")

    summary = {
        "updated_utc": utc_now(),
        "complete_records": sorted(complete_recs),
        "near_30_300s_txt_primary": [r for r in p_grp if r["label_set"] == "txt_primary" and r["region"] == "near_30_300s" and r["kind"] != "noop"],
        "near_30_300s_shared_txt_st": [r for r in p_grp if r["label_set"] == "shared_txt_st" and r["region"] == "near_30_300s" and r["kind"] != "noop"],
        "oracle_near_30_300s": [r for r in o_grp if r["region"] == "near_30_300s"],
        "realistic_512hz_near_30_300s": [r for r in r_grp if r["region"] == "near_30_300s"],
        "sensitivity_near_30_300s": [r for r in s_grp if r["region"] == "near_30_300s"],
    }
    write_json(agg_dir / "cap_v2_summary.json", summary)
    return summary


def run_cap_v2(args) -> None:
    from anysleep_no_hydra import AnySleep

    project = Path(args.project_root).resolve()
    plan = read_json(args.manifest) or build_manifest_20()
    base_cfg = plan["base_config"]
    cap_data = Path(args.cap_cache_dir) if args.cap_cache_dir else (Path("/content/cap-ephemeral-data") if Path("/content").is_dir() else project / "_tmp_cap")
    out_dir = project / "results" / CAP_STAGE_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"
    state = read_json(state_file, {"stage_id": CAP_STAGE_ID, "status": "RUNNING", "records": {}, "started_utc": utc_now()})
    state["status"] = "RUNNING"
    write_json(state_file, state)

    cap_data.mkdir(parents=True, exist_ok=True)
    sums_path = cap_data / "SHA256SUMS.txt"
    download_verified(CAP_BASE_URL + "SHA256SUMS.txt", sums_path)
    official_sums = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            official_sums[Path(parts[1]).name] = parts[0]

    dev = resolve_device(args.device)
    configure_torch_determinism(dev, seed=int(base_cfg["noise_seed"]), strict=True)
    models = {
        ck: AnySleep(path=str(ensure_checkpoint(project, ck)), sleep_stage_frequency=1).eval().to(dev)
        for ck in ("run1", "run2", "run3")
    }
    model = models["run1"]
    epoch_samples = 3840

    selected = [r.strip() for r in args.records.split(",") if r.strip()]
    for rec_name in selected:
        spec = CAP_RECORDS[rec_name]
        rdir = out_dir / "records" / rec_name
        prev = read_json(rdir / "status.json", {})
        receipt = prev.get("sha256_receipt", {})
        if (
            prev.get("status") == "COMPLETE"
            and receipt
            and all((rdir / k).is_file() and sha256_file(rdir / k) == v for k, v in receipt.items())
        ):
            print(f"[CAP-v2] Verified & skipped completed record {rec_name}", flush=True)
            continue

        print(f"[CAP-v2] Running complete uncropped CAP analysis for {rec_name} ({dev})...", flush=True)
        rdir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        txt_file = cap_data / f"{rec_name}.txt"
        edf_file = cap_data / f"{rec_name}.edf"
        txt_sha = download_verified(CAP_BASE_URL + f"{rec_name}.txt", txt_file, official_sums.get(f"{rec_name}.txt"))
        if official_sums.get(f"{rec_name}.edf") and official_sums[f"{rec_name}.edf"] != spec["expected_edf_sha256"]:
            raise ValueError(f"SHA256SUMS mismatch for {rec_name}.edf")
        try:
            edf_sha = download_verified(CAP_BASE_URL + f"{rec_name}.edf", edf_file, spec["expected_edf_sha256"])
            signals, sfs, edf_clock, total_s = load_cap_native_channels(edf_file, CAP_CHANNELS)
        finally:
            edf_file.unlink(missing_ok=True)

        truth_txt, truth_shared, align_meta = parse_cap_txt_labels(txt_file, edf_clock, total_s, spec)
        n_epochs = len(truth_txt)
        sf_c0 = float(sfs[0])
        z0_128 = resample_channels_to_128hz(signals, sfs, n_epochs)

        # Clean baselines per method (no leakage!)
        z0_norm, stats0 = robust_norm_channelwise(z0_128)
        base_logits, alpha0 = infer_anysleep_full(model, z0_norm, dev)
        rep_logits, _ = infer_anysleep_full(model, z0_norm, dev)
        np.testing.assert_allclose(base_logits, rep_logits, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(base_logits.argmax(1), rep_logits.argmax(1))
        base_logits = base_logits[:n_epochs]
        base_pred_off = base_logits.argmax(axis=1).astype(np.int16)
        del rep_logits

        yasa_base, yasa_prob = run_yasa_cap(z0_128[[0, 2]])
        yasa_rep, yasa_prob_rep = run_yasa_cap(z0_128[[0, 2]])
        np.testing.assert_array_equal(yasa_base, yasa_rep)
        np.testing.assert_allclose(yasa_prob, yasa_prob_rep, rtol=1e-6, atol=1e-7)
        del yasa_rep, yasa_prob, yasa_prob_rep

        z_w5_c, _ = clip_winsor_norm(z0_128, 5.0)
        pred_w5_c = infer_anysleep_full(model, z_w5_c, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
        del z_w5_c

        z_loc_c = local_window_robust_norm(z0_128, 32, epoch_samples)
        pred_loc_c = infer_anysleep_full(model, z_loc_c, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
        del z_loc_c

        z_cau_c = causal_running_robust_norm(z0_128, 60, epoch_samples=epoch_samples)
        pred_cau_c = infer_anysleep_full(model, z_cau_c, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
        del z_cau_c

        bad_ce_c = detect_cap_bad_channel_epochs(z0_128, signals, sfs, epoch_samples=epoch_samples)
        bad_any_c = np.any(bad_ce_c, axis=0)
        z_mask_c, _, _ = build_masked_input(z0_128, bad_ce_c, epoch_samples=epoch_samples)
        pred_mask_c = infer_anysleep_full(model, z_mask_c, dev)[0][:n_epochs].argmax(axis=1).astype(np.int16)
        pred_split_c = infer_anysleep_split_restart_full(model, z_mask_c, bad_any_c, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(axis=1).astype(np.int16)
        pred_reroute_c = infer_anysleep_channel_reroute_full(model, z_mask_c, bad_ce_c, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(axis=1).astype(np.int16)
        del z_mask_c

        clean_preds = {
            "official_global": base_pred_off,
            "clip_winsor_5iqr": pred_w5_c,
            "local_window_norm_32ep": pred_loc_c,
            "causal_running_norm": pred_cau_c,
            "blind_quality_masking": pred_mask_c,
            "blind_split_restart": pred_split_c,
            "blind_channel_reroute": pred_reroute_c,
            "oracle_masking_upper_bound": base_pred_off,
        }
        clean_rows = []
        for method, is_orc in CAP_METHODS:
            cm = compute_clean_metrics(truth_txt, clean_preds[method])
            is_det = method in ("blind_quality_masking", "blind_split_restart", "blind_channel_reroute")
            cm.update({"record": rec_name, "pipeline": "anysleep", "method": method, "is_oracle": is_orc, "flagged_epochs": int(bad_any_c.sum()) if is_det else 0})
            clean_rows.append(cm)
        ym = compute_clean_metrics(truth_txt, yasa_base)
        ym.update({"record": rec_name, "pipeline": "yasa", "method": "official_global", "is_oracle": False, "flagged_epochs": 0})
        clean_rows.append(ym)

        # Clean baseline predictions for checkpoints run2/run3
        ckpt_clean_preds = {"run1": base_pred_off}
        for ck in ("run2", "run3"):
            ck_l, _ = infer_anysleep_full(models[ck], z0_norm, dev)
            ckpt_clean_preds[ck] = ck_l[:n_epochs].argmax(axis=1).astype(np.int16)
        del z0_norm

        paired_rows, oracle_rows, real_rows, sens_rows = [], [], [], []
        zeros_sup = np.zeros(n_epochs, dtype=bool)
        for frac in plan["positions"]:
            chosen_ep = chosen_epoch_for_fraction(truth_txt, frac)
            start_c0 = int(chosen_ep * 30 * sf_c0)
            end_c0 = int((chosen_ep + 1) * 30 * sf_c0)

            for label_set, truth_vec in [("txt_primary", truth_txt), ("shared_txt_st", truth_shared)]:
                for method, is_orc in CAP_METHODS:
                    for r in compute_region_metrics_extended(truth_vec, clean_preds[method], clean_preds[method], sf_c0, start_c0, end_c0, zeros_sup):
                        paired_rows.append({"participant": rec_name, "record": rec_name, "label_set": label_set, "position": frac, "pipeline": "anysleep", "method": method, "is_oracle": is_orc, "kind": "noop", **r})
                for r in compute_region_metrics_extended(truth_vec, yasa_base, yasa_base, sf_c0, start_c0, end_c0, zeros_sup):
                    paired_rows.append({"participant": rec_name, "record": rec_name, "label_set": label_set, "position": frac, "pipeline": "yasa", "method": "official_global", "is_oracle": False, "kind": "noop", **r})

            for kind in ("flat", "clip", "noise"):
                z1_128, nat_pert = perturb_and_resample_cap(
                    z0_128, signals, sfs, n_epochs, [0], chosen_ep, kind,
                    seed=int(base_cfg["noise_seed"]),
                    clip_volts=float(base_cfg["clip_volts"]),
                    noise_sd_volts=float(base_cfg["noise_sd_volts"]),
                )
                changed = np.any(z0_128 != z1_128, axis=0)
                local_support = anysleep_local_support(changed)
                beyond_mask = ~local_support

                # True full-night 2x2 oracle
                z1_norm, _ = robust_norm_channelwise(z1_128)
                recomp_l, alpha1 = infer_anysleep_full(model, z1_norm, dev)
                fz = apply_fixed_stats(z1_128, stats0)
                fstats_l, _ = infer_anysleep_full(model, fz, dev, fixed_alpha=alpha1)
                fatt_l, _ = infer_anysleep_full(model, z1_norm, dev, fixed_alpha=alpha0)
                fboth_l, _ = infer_anysleep_full(model, fz, dev, fixed_alpha=alpha0)
                recomp_l, fstats_l, fatt_l, fboth_l = recomp_l[:n_epochs], fstats_l[:n_epochs], fatt_l[:n_epochs], fboth_l[:n_epochs]
                if np.any(beyond_mask):
                    fb_d = float(np.max(np.abs(fboth_l[beyond_mask] - base_logits[beyond_mask])))
                    if fb_d > 1e-3:
                        raise AssertionError(f"CAP fixedboth leaked beyond support ({fb_d}) on {rec_name} {kind}")
                for arm_name, al in [("recomputed_both", recomp_l), ("fixedstats", fstats_l), ("fixedattention", fatt_l), ("fixedboth", fboth_l)]:
                    ap = al.argmax(axis=1).astype(np.int16)
                    for r in compute_region_metrics_extended(truth_txt, base_pred_off, ap, sf_c0, start_c0, end_c0, local_support):
                        oracle_rows.append({"participant": rec_name, "record": rec_name, "position": frac, "arm": arm_name, "kind": kind, **r})

                # Multi-checkpoint on noise
                if kind == "noise":
                    for ck in ("run1", "run2", "run3"):
                        cl = recomp_l if ck == "run1" else infer_anysleep_full(models[ck], z1_norm, dev)[0][:n_epochs]
                        cp = cl.argmax(axis=1).astype(np.int16)
                        for r in compute_region_metrics_extended(truth_txt, ckpt_clean_preds[ck], cp, sf_c0, start_c0, end_c0, local_support):
                            sens_rows.append({"participant": rec_name, "position": frac, "sweep_type": "checkpoint", "setting": ck, "method": "official_global", **r})
                del z1_norm, fz

                # Deployable methods (strictly non-leaky)
                w5_p = infer_anysleep_full(model, clip_winsor_norm(z1_128, 5.0)[0], dev)[0][:n_epochs].argmax(1).astype(np.int16)
                loc_p = infer_anysleep_full(model, local_window_robust_norm(z1_128, 32, epoch_samples), dev)[0][:n_epochs].argmax(1).astype(np.int16)
                cau_p = infer_anysleep_full(model, causal_running_robust_norm(z1_128, 60, epoch_samples=epoch_samples), dev)[0][:n_epochs].argmax(1).astype(np.int16)

                bad_ce_p = detect_cap_bad_channel_epochs(z1_128, nat_pert, sfs, epoch_samples=epoch_samples)
                bad_any_p = np.any(bad_ce_p, axis=0)
                z_mask_p, _, _ = build_masked_input(z1_128, bad_ce_p, epoch_samples=epoch_samples)
                mask_p = infer_anysleep_full(model, z_mask_p, dev)[0][:n_epochs].argmax(1).astype(np.int16)
                split_p = infer_anysleep_split_restart_full(model, z_mask_p, bad_any_p, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(1).astype(np.int16)
                reroute_p = infer_anysleep_channel_reroute_full(model, z_mask_p, bad_ce_p, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(1).astype(np.int16)
                del z_mask_p

                orc_ce = np.zeros_like(bad_ce_p)
                orc_ce[0, chosen_ep] = True
                z_orc_p, _, _ = build_masked_input(z1_128, orc_ce, epoch_samples=epoch_samples)
                orc_p = infer_anysleep_full(model, z_orc_p, dev)[0][:n_epochs].argmax(1).astype(np.int16)
                del z_orc_p

                yasa_p, _ = run_yasa_cap(z1_128[[0, 2]])
                pert_map = {
                    "official_global": recomp_l.argmax(1).astype(np.int16),
                    "clip_winsor_5iqr": w5_p,
                    "local_window_norm_32ep": loc_p,
                    "causal_running_norm": cau_p,
                    "blind_quality_masking": mask_p,
                    "blind_split_restart": split_p,
                    "blind_channel_reroute": reroute_p,
                    "oracle_masking_upper_bound": orc_p,
                }
                for label_set, truth_vec in [("txt_primary", truth_txt), ("shared_txt_st", truth_shared)]:
                    for method, is_orc in CAP_METHODS:
                        for r in compute_region_metrics_extended(truth_vec, clean_preds[method], pert_map[method], sf_c0, start_c0, end_c0, local_support):
                            paired_rows.append({"participant": rec_name, "record": rec_name, "label_set": label_set, "position": frac, "pipeline": "anysleep", "method": method, "is_oracle": is_orc, "kind": kind, **r})
                    for r in compute_region_metrics_extended(truth_vec, yasa_base, yasa_p, sf_c0, start_c0, end_c0, zeros_sup):
                        paired_rows.append({"participant": rec_name, "record": rec_name, "label_set": label_set, "position": frac, "pipeline": "yasa", "method": "official_global", "is_oracle": False, "kind": kind, **r})
                del z1_128, nat_pert
                gc.collect()

            # Unaliased 512Hz realistic morphologies on CAP C4-A1
            for r_kind in ("mains_60hz", "emg_hf_burst", "slow_drift_pop"):
                z1_128, nat_pert = perturb_and_resample_cap(
                    z0_128, signals, sfs, n_epochs, [0], chosen_ep, r_kind,
                    seed=int(base_cfg["noise_seed"]),
                    clip_volts=float(base_cfg["clip_volts"]),
                    noise_sd_volts=float(base_cfg["noise_sd_volts"]),
                )
                changed = np.any(z0_128 != z1_128, axis=0)
                local_support = anysleep_local_support(changed)
                off_p = infer_anysleep_full(model, robust_norm_channelwise(z1_128)[0], dev)[0][:n_epochs].argmax(1).astype(np.int16)
                w5_p = infer_anysleep_full(model, clip_winsor_norm(z1_128, 5.0)[0], dev)[0][:n_epochs].argmax(1).astype(np.int16)
                cau_p = infer_anysleep_full(model, causal_running_robust_norm(z1_128, 60, epoch_samples=epoch_samples), dev)[0][:n_epochs].argmax(1).astype(np.int16)
                bad_ce_r = detect_cap_bad_channel_epochs(z1_128, nat_pert, sfs, epoch_samples=epoch_samples)
                z_m_r, _, _ = build_masked_input(z1_128, bad_ce_r, epoch_samples=epoch_samples)
                mask_p = infer_anysleep_full(model, z_m_r, dev)[0][:n_epochs].argmax(1).astype(np.int16)
                reroute_p = infer_anysleep_channel_reroute_full(model, z_m_r, bad_ce_r, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(1).astype(np.int16)
                for m_name, mp in [
                    ("official_global", off_p),
                    ("clip_winsor_5iqr", w5_p),
                    ("causal_running_norm", cau_p),
                    ("blind_quality_masking", mask_p),
                    ("blind_channel_reroute", reroute_p),
                ]:
                    for r in compute_region_metrics_extended(truth_txt, clean_preds[m_name], mp, sf_c0, start_c0, end_c0, local_support):
                        real_rows.append({"participant": rec_name, "position": frac, "method": m_name, "kind": r_kind, **r})
                del z1_128, nat_pert, z_m_r

            # Channel sensitivity on noise at C4-A1 [0], F4-C4 [1], ROC-LOC [2], dual-EEG [0,1]
            for ch_tag, ch_list in [("C4-A1", [0]), ("F4-C4", [1]), ("ROC-LOC", [2]), ("C4A1+F4C4", [0, 1])]:
                z1_128, nat_pert = perturb_and_resample_cap(
                    z0_128, signals, sfs, n_epochs, ch_list, chosen_ep, "noise",
                    seed=int(base_cfg["noise_seed"]),
                    clip_volts=float(base_cfg["clip_volts"]),
                    noise_sd_volts=float(base_cfg["noise_sd_volts"]),
                )
                changed = np.any(z0_128 != z1_128, axis=0)
                local_support = anysleep_local_support(changed)
                off_p = infer_anysleep_full(model, robust_norm_channelwise(z1_128)[0], dev)[0][:n_epochs].argmax(1).astype(np.int16)
                bad_ce_s = detect_cap_bad_channel_epochs(z1_128, nat_pert, sfs, epoch_samples=epoch_samples)
                z_m_s, _, _ = build_masked_input(z1_128, bad_ce_s, epoch_samples=epoch_samples)
                reroute_p = infer_anysleep_channel_reroute_full(model, z_m_s, bad_ce_s, dev, epoch_samples=epoch_samples)[:n_epochs].argmax(1).astype(np.int16)
                for m_name, mp in [("official_global", off_p), ("blind_channel_reroute", reroute_p)]:
                    for r in compute_region_metrics_extended(truth_txt, clean_preds[m_name], mp, sf_c0, start_c0, end_c0, local_support):
                        sens_rows.append({"participant": rec_name, "position": frac, "sweep_type": "channels", "setting": ch_tag, "method": m_name, **r})
                del z1_128, nat_pert, z_m_s

        del signals, z0_128
        gc.collect()

        meta_out = {"record": rec_name, "channels": CAP_CHANNELS, "native_sfreqs_hz": sfs, "edf_sha256": edf_sha, "txt_sha256": txt_sha, **align_meta}
        files_to_write = {
            "alignment_and_hashes.json": meta_out,
            "clean_metrics.json": clean_rows,
            "paired_positions.json": paired_rows,
            "oracle_2x2_positions.json": oracle_rows,
            "realistic_512hz_positions.json": real_rows,
            "sensitivity_ckpt_chan.json": sens_rows,
        }
        receipt = {}
        for fn, payload in files_to_write.items():
            write_json(rdir / fn, payload)
            receipt[fn] = sha256_file(rdir / fn)
        r_status = {"status": "COMPLETE", "record": rec_name, "seconds": round(time.monotonic() - t0, 2), "completed_utc": utc_now(), "sha256_receipt": receipt}
        write_json(rdir / "status.json", r_status)
        state["records"][rec_name] = r_status
        write_json(state_file, state)
        aggregate_cap_v2(out_dir)
        sync_filesystem()
        print(f"[CAP-v2] Completed {rec_name} in {r_status['seconds']:.1f}s", flush=True)

    summary = aggregate_cap_v2(out_dir)
    state["status"] = "COMPLETE"
    state["completed_utc"] = utc_now()
    write_json(state_file, state)
    sync_filesystem()
    print("CAP_VERIFICATION_V2_COMPLETE", summary.get("complete_records"), flush=True)


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "fake_n1.txt"
        content = "\n".join([
            "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation",
            "W\tSupine\t22:09:33\tSLEEP-S0\t30\tEEG",
            "S2\tSupine\t22:10:03\tMCAP-A1\t5\tEEG",
            "S2\tSupine\t22:11:03\tSLEEP-S2\t30\tEEG",
            "S4\tSupine\t00:01:03\tSLEEP-S4\t30\tEEG",
        ])
        p.write_text(content, encoding="utf-8")
        spec = {"expected_first_offset_s": 210, "expected_scored_txt": 3, "expected_full_epochs": 300, "txt_only_offsets_s": [300]}
        t_txt, t_sh, meta = parse_cap_txt_labels(p, "22.06.03", 300 * 30, spec)
        assert t_txt[7] == 0 and t_txt[8] == -1 and t_txt[10] == 2 and meta["scored_epochs_txt"] == 3
        assert t_sh[10] == -1 and meta["scored_epochs_shared_txt_st"] == 2

    # Heterogeneous native rate detector test (512Hz, 512Hz, 256Hz)
    rng = np.random.default_rng(20261117)
    n_ep = 40
    sfs = [512.0, 512.0, 256.0]
    nat = [rng.normal(0.0, 2e-5, size=n_ep * 30 * int(sf)).astype(np.float32) for sf in sfs]
    for c in range(3):
        nat[c] = np.convolve(nat[c], np.ones(15, dtype=np.float32) / 15.0, mode="same")
    # Inject clip at epoch 20 on ch 0 (512Hz)
    s0, e0 = 20 * 30 * 512, 21 * 30 * 512
    nat[0][s0:e0] = np.clip(nat[0][s0:e0], -2e-6, 2e-6)
    z128 = np.stack([nat[c][:: int(round(sfs[c] / 128.0))][: n_ep * 3840] for c in range(3)], axis=0)
    fl = detect_cap_bad_channel_epochs(z128, nat, sfs, epoch_samples=3840)
    assert bool(fl[0, 20]) and not np.any(fl[1:])
    print("CAP_VERIFICATION_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    r.add_argument("--manifest", type=Path, default=ROOT / "manifests/development_manifest_20.json")
    r.add_argument("--records", default="n1,n2,n3")
    r.add_argument("--device", default="auto")
    r.add_argument("--cap-cache-dir", default="")
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
    else:
        run_cap_v2(a)


if __name__ == "__main__":
    main()
