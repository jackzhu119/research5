"""Recompute the reported v3 primary contrasts from the exported participant tables."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap(values, seed, draws=20_000, omit_constant=False):
    data = np.asarray(values, dtype=float)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Bootstrap requires finite participant contrasts")
    estimate = float(data.mean())
    if data.size < 2 or (omit_constant and np.ptp(data) == 0):
        return estimate, None
    indices = np.random.default_rng(seed).integers(data.size, size=(draws, data.size))
    return estimate, np.quantile(data[indices].mean(axis=1), [0.025, 0.975])


def verify(label, values, expected, interval, seed, draws=20_000, omit_constant=False):
    estimate, ci = bootstrap(values, seed, draws, omit_constant)
    np.testing.assert_allclose(estimate, expected, rtol=0, atol=1e-12, err_msg=label)
    if interval is None:
        if ci is not None:
            raise ValueError(f"Expected a missing interval for {label}")
    else:
        np.testing.assert_allclose(ci, interval, rtol=0, atol=1e-12, err_msg=label)
    display_ci = "CI unavailable" if ci is None else f"95% CI [{100 * ci[0]:+.3f}, {100 * ci[1]:+.3f}] pp"
    print(f"{label}: n={len(values)}, {100 * estimate:+.3f} pp, {display_ci}")


def verify_sleep_edf():
    summary = read_json("v3-training/head_summary.json")
    cohort = read_json("v3-training/sleep_cohort.json")
    weights = read_json("v3-training/weights_frozen_before_test.json")
    if summary["state"] != "complete" or summary["completed_jobs"] != summary["expected_jobs"]:
        raise ValueError("Incomplete head experiment")
    if len(weights) != summary["expected_jobs"] or summary["missing_jobs"] or summary["coverage_issues"]:
        raise ValueError("Missing weights, jobs, or participant coverage")

    contrasts = read_csv("v3-training/head_subject_contrasts.csv")
    clean = read_csv("v3-training/head_subject_clean.csv")
    clean_rows = [row for row in clean if row["split"] == "heldout"]
    clean_by_arm = {(row["participant"], row["arm"]): float(row["macro_f1"]) for row in clean_rows}
    if len(clean_rows) != len(clean_by_arm):
        raise ValueError("Repeated held-out participant/arm")

    heldout = sorted(item["participant"] for item in cohort["participants"] if item["role"] == "test")
    seed = summary["bootstrap"]["seed"]
    draws = summary["bootstrap"]["replicates"]
    for region, section in ((summary["primary_region"], "primary"), ("affected", "affected")):
        rows = sorted(
            (row for row in contrasts if (row["split"], row["kind"], row["region"])
             == ("heldout", summary["primary_kind"], region)),
            key=lambda row: row["participant"],
        )
        if [row["participant"] for row in rows] != heldout or len(rows) != summary["expected_heldout_subjects"]:
            raise ValueError(f"Participant coverage mismatch in {region}")
        np.testing.assert_allclose(
            [float(row["D"]) for row in rows],
            [float(row["C"]) + float(row["delta_S"]) for row in rows], rtol=0, atol=1e-12,
        )
        for metric, stats in summary[section].items():
            verify(f"Sleep-EDF {region} {metric}", [float(row[metric]) for row in rows],
                   stats["estimate"], stats["ci95"], seed, draws, omit_constant=True)

    values = [clean_by_arm[subject, summary["candidate"]] - clean_by_arm[subject, summary["comparator"]]
              for subject in heldout]
    stats = summary["clean_macro_f1_difference"]
    verify("Sleep-EDF original full-record macro-F1", values, stats["estimate"], stats["ci95"],
           seed, draws, omit_constant=True)


def verify_cap():
    summary = read_json("v3-cap/summary.json")
    participants = read_json("v3-cap/participant_metrics.json")
    groups = read_json("v3-cap/group_metrics.json")
    if summary["partial_aggregation"] or set(summary["complete_records"]) != set(summary["planned_records"]):
        raise ValueError("Incomplete CAP experiment")
    np.testing.assert_allclose(
        [row["corrupt_risk_vs_official"] for row in participants],
        [row["clean_cost_vs_official"] + row["own_clean_spillover"] - row["official_spillover"]
         for row in participants], rtol=0, atol=1e-12,
    )
    for group in groups:
        if (group["checkpoint"], group["kind"], group["region"], group["montage_stratum"], group["analysis_role"]) != (
            "run1", "noise", "near_30_300s", "all_montages", "primary"
        ):
            continue
        cohort, method = group["cohort"], group["method"]
        rows = sorted(
            (row for row in participants if all(row[field] == group[field] for field in
             ("cohort", "checkpoint", "kind", "method", "region", "analysis_role"))),
            key=lambda row: row["record"],
        )
        if len({row["record"] for row in rows}) != len(rows) or len(rows) != group["n_records"]:
            raise ValueError(f"CAP participant coverage mismatch for {cohort} {method}")
        metric = "corrupt_risk_vs_official"
        verify(f"CAP {cohort} {method} D", [row[metric] for row in rows],
               group[metric], group[metric + "_ci95"], seed=20261117)


if __name__ == "__main__":
    verify_sleep_edf()
    verify_cap()
