"""Frozen 20-participant development manifest (SC400-SC419) & SHA256 data/model preparer."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "followup") not in sys.path:
    sys.path.insert(0, str(ROOT / "followup"))

from common_v2 import (
    ANYSLEEP_CHECKPOINTS,
    ANYSLEEP_COMMIT,
    PHYSIONET_SC400_SC419,
    SLEEP_EDF_BASE_URL,
    download_verified,
    ensure_checkpoint,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)


def build_manifest_20(base_config_path: Path | None = None) -> dict:
    cfg_path = base_config_path or (ROOT / "pilot/config_smoke.json")
    base_cfg = read_json(cfg_path)
    if not base_cfg:
        raise FileNotFoundError(f"Missing base config: {cfg_path}")

    participants = []
    for p_id, rec, psg, psg_sha, hyp, hyp_sha, split in PHYSIONET_SC400_SC419:
        mapped_id = rec[3:6]
        participants.append({
            "participant": p_id,
            "availability": "AVAILABLE",
            "cohort_wave": "v1_initial_10" if int(p_id[2:]) < 410 else "v2_replication_10",
            "night_rule": "first night SC4xx1xx; literal PhysioNet sleep-edfx/1.0.0 SHA256SUMS match",
            "record": rec,
            "psg": psg,
            "psg_sha256": psg_sha,
            "psg_url": SLEEP_EDF_BASE_URL + psg,
            "hypnogram": hyp,
            "hypnogram_sha256": hyp_sha,
            "hypnogram_url": SLEEP_EDF_BASE_URL + hyp,
            "anysleep_record_id": mapped_id,
            "anysleep_subject_id": mapped_id[:2],
            "anysleep_published_split": split,
        })

    return {
        "schema": "research5-development-queue-v2",
        "queue_id": "development-queue-v2",
        "frozen_utc": "2026-11-18T00:00:00+00:00",
        "scope": "Bounded 20-participant development & internal replication cohort (SC400-SC419, first night each). Sleep-EDF is a development source in AnySleep; stratify explicitly by published train/valid/test split.",
        "anysleep_commit": ANYSLEEP_COMMIT,
        "checkpoints": ANYSLEEP_CHECKPOINTS,
        "base_config": base_cfg,
        "participants": participants,
        "positions": [0.25, 0.5, 0.75],
        "position_rule": "floor(q * valid-scored-epoch count), sorted by time, q in .25/.5/.75; independent of model predictions",
        "duration_seconds": 30,
        "sensitivity_predeclarations": {
            "sleep_span_window_margin_minutes": 30,
            "duration_sweep_seconds": [10, 30, 60, 120, 300],
            "noise_sd_sweep_volts": [2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4],
            "noise_sweep_seeds": [20261117, 20261118, 20261119],
            "morphologies": ["mains_60hz", "emg_hf_burst", "slow_drift_pop"],
            "multi_channel_sets": [[0], [0, 1], [0, 1, 2], [2]],
            "checkpoints": ["run1", "run2", "run3"],
        },
        "analysis": {
            "primary_region": "near_30_300s",
            "practical_net_error_fraction": 0.01,
            "simple_baseline_tolerance_fraction": 0.005,
            "bootstrap_seed": 20261117,
            "bootstrap_replicates": 20000,
        },
    }


def validate_manifest_20(plan: dict) -> None:
    if plan.get("schema") != "research5-development-queue-v2":
        raise ValueError("Invalid manifest_v2 schema")
    parts = plan.get("participants", [])
    expected_ids = [f"SC{i}" for i in range(400, 420)]
    if [p["participant"] for p in parts] != expected_ids:
        raise ValueError("Participant list must be exact ordered SC400-SC419")
    splits = [p["anysleep_published_split"] for p in parts]
    if splits.count("test") != 4 or splits.count("valid") != 5 or splits.count("train") != 11:
        raise ValueError(f"Unexpected AnySleep split counts in SC400-SC419: {splits}")


def prepare_project_assets(
    project: Path,
    plan: dict,
    participants: list[str] | None = None,
    all_checkpoints: bool = True,
) -> list[dict]:
    validate_manifest_20(plan)
    project = Path(project).resolve()
    data_dir = project / "data"
    downloads = []

    target_set = set(participants) if participants else None
    for row in plan["participants"]:
        if target_set is not None and row["participant"] not in target_set:
            continue
        for kind_key, url_key, fname_key, sha_key in [
            ("psg", "psg_url", "psg", "psg_sha256"),
            ("hypnogram", "hypnogram_url", "hypnogram", "hypnogram_sha256"),
        ]:
            dest = data_dir / row[fname_key]
            info = download_verified(row[url_key], dest, expected_sha256=row[sha_key])
            downloads.append({
                "asset_type": kind_key,
                "participant": row["participant"],
                **info,
            })

    ckpt_keys = ["run1", "run2", "run3"] if all_checkpoints else ["run1"]
    for ck in ckpt_keys:
        p = ensure_checkpoint(project, ck)
        downloads.append({
            "asset_type": "checkpoint",
            "run_key": ck,
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    write_json(project / "manifests/v2_downloads.json", {
        "updated_utc": utc_now(),
        "assets": downloads,
    })
    return downloads


def selftest() -> None:
    plan = build_manifest_20()
    validate_manifest_20(plan)
    assert len(plan["participants"]) == 20
    for row in plan["participants"]:
        assert len(row["psg_sha256"]) == 64
        assert len(row["hypnogram_sha256"]) == 64
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "m20.json"
        write_json(out, plan)
        loaded = read_json(out)
        assert loaded == plan
    print("MANIFEST_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    f = sub.add_parser("freeze")
    f.add_argument("--output", type=Path, default=ROOT / "manifests/development_manifest_20.json")
    prep = sub.add_parser("prepare")
    prep.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    prep.add_argument("--manifest", type=Path, default=ROOT / "manifests/development_manifest_20.json")
    prep.add_argument("--participants", default="")
    prep.add_argument("--run1-only", action="store_true")
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
    elif a.command == "freeze":
        plan = build_manifest_20()
        validate_manifest_20(plan)
        write_json(a.output, plan)
        print(f"FROZEN_MANIFEST_20: {a.output} ({len(plan['participants'])} participants)", flush=True)
    else:
        plan = read_json(a.manifest) or build_manifest_20()
        parts = [x.strip() for x in a.participants.split(",") if x.strip()] or None
        rows = prepare_project_assets(a.project_root, plan, participants=parts, all_checkpoints=not a.run1_only)
        print(f"PREPARED_ASSETS: {len(rows)} verified files", flush=True)


if __name__ == "__main__":
    main()
