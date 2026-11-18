"""Stage C Small-Model v2: 7-Arm Controlled Context, Blind Masking & Unseen-OOD Study.

Enhancements over broken v1:
1) Fixes all shape/broadcast issues and replaces simple 14-dim stats with a 31-dim
   strictly epoch-local representation combining frozen AnySleep single-epoch trunk
   features (11 dims) + physical/quality descriptors (15 dims) + Hann spectral
   bandpower ratios (5 dims) with zero cross-epoch leakage.
2) Tests 7 matched arms (same frozen features, same K=+/-10 epoch +/-300s receptive
   field, hidden=64) across 5-fold participant CV over all 20 participants (SC400-419)
   x 3 training seeds:
     - local_only (K=0 lower-bound neighbor vulnerability reference)
     - unconstrained_context (clean-trained sequence head)
     - matched_pert_aug (trained with seen fold flat/clip/noise donor corruptions)
     - rest_spectral_reg (spectral-norm + stability regularization + aug)
     - blind_mask_then_context (clean-trained, zero-masks anomalous feature epochs)
     - quality_gated_context (clean-trained differentiable gate + spectral norm + cap)
     - constrained_residual (matched aug + spectral norm + quality gate + neighbor-invariance loss)
3) Evaluates both SEEN corruptions (flat, clip, noise) and UNSEEN OOD corruptions
   (mains_60hz, emg_hf_burst, slow_drift_pop) under noop==0 and K-support invariance.
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
    compute_clean_metrics,
    compute_region_metrics_extended,
    configure_torch_determinism,
    download_verified,
    ensure_checkpoint,
    load_sleep_edf_record,
    participant_bootstrap_ci,
    perturb_extended,
    read_json,
    resample_channels_128hz,
    resolve_device,
    sha256_file,
    sync_filesystem,
    utc_now,
    write_json,
)
from manifest_v2 import build_manifest_20
from stage_b_expansion_v2 import chosen_epoch_for_fraction

EXP_ID = "stage-c-small-model-v2"
SEEN_KINDS = ["flat", "clip", "noise"]
OOD_KINDS = ["mains_60hz", "emg_hf_burst", "slow_drift_pop"]
ALL_TEST_KINDS = ["noop"] + SEEN_KINDS + OOD_KINDS

ARMS = [
    "local_only",
    "unconstrained_context",
    "matched_pert_aug",
    "rest_spectral_reg",
    "blind_mask_then_context",
    "quality_gated_context",
    "constrained_residual",
]


def compute_epoch_local_physical_features(raw_128_epochs: np.ndarray) -> np.ndarray:
    """Compute strictly epoch-local physical + spectral descriptors on (E, 3840, C).

    Zero cross-epoch statistics or interactions: modifying epoch e leaves all e' != e identical.
    Returns (E, 5*C + 5) = (E, 20) float32 for C=3.
    """
    ep = np.asarray(raw_128_epochs, dtype=np.float64)
    n_ep, ep_len, n_ch = ep.shape
    med = np.median(ep, axis=1, keepdims=True)
    centered = ep - med
    rms = np.sqrt(np.mean(centered**2, axis=1))  # (E, C)
    diff_rms = np.sqrt(np.mean(np.diff(ep, axis=1) ** 2, axis=1))  # (E, C)
    abs_ep = np.abs(ep)
    ep_max = np.max(abs_ep, axis=1)  # (E, C)

    near_rail = abs_ep >= np.maximum(0.99 * ep_max[:, None, :], 1e-9)
    raw_diff = np.abs(np.diff(ep, axis=1))
    rail_flat_frac = np.mean((raw_diff <= 1e-9) & near_rail[:, 1:, :] & near_rail[:, :-1, :], axis=1)
    flat_flag = (diff_rms <= 2e-7).astype(np.float64)
    crest = np.minimum(ep_max / np.maximum(rms, 1e-12), 20.0)

    log_rms = np.log1p(rms * 1e6)
    log_diff = np.log1p(diff_rms * 1e6)

    # Spectral relative bandpower over EEG channels (ch 0..min(2, C)-1)
    eeg_c = centered[:, :, : min(2, n_ch)]
    win = np.hanning(ep_len)[None, :, None]
    fft_vals = np.fft.rfft(eeg_c * win, axis=1)
    psd = np.mean(np.abs(fft_vals) ** 2, axis=2)  # (E, F)
    freqs = np.fft.rfftfreq(ep_len, d=1.0 / 128.0)
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_pow = np.maximum(np.sum(psd[:, total_mask], axis=1, keepdims=True), 1e-24)
    bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 45.0)]
    band_feats = []
    for lo_f, hi_f in bands:
        bmask = (freqs >= lo_f) & (freqs < hi_f) if hi_f < 45.0 else (freqs >= lo_f) & (freqs <= hi_f)
        band_feats.append(np.sum(psd[:, bmask], axis=1, keepdims=True) / total_pow)
    spectral = np.concatenate(band_feats, axis=1)  # (E, 5)

    return np.concatenate(
        [log_rms, log_diff, rail_flat_frac, flat_flag, crest, spectral], axis=1
    ).astype(np.float32)


def epoch_local_anysleep_normalize(raw_128_epochs: np.ndarray) -> np.ndarray:
    """Apply AnySleep 20*IQR clip + RobustScaler strictly inside each 30s epoch separately."""
    ep = np.asarray(raw_128_epochs, dtype=np.float64)
    pct = np.percentile(ep, [75.0, 25.0], axis=1)  # (2, E, C)
    iqr = (pct[0] - pct[1])[:, None, :]
    safe_iqr = np.where(iqr > 1e-9, iqr, 2e-5)
    thresh = safe_iqr * 20.0
    clipped = np.clip(ep, -thresh, thresh)
    c_pct = np.percentile(clipped, [75.0, 25.0], axis=1)  # (2, E, C)
    c_scale = (c_pct[0] - c_pct[1])[:, None, :]
    c_scale = np.where(c_scale > 1e-12, c_scale, safe_iqr)
    c_med = np.median(clipped, axis=1, keepdims=True)
    return ((clipped - c_med) / c_scale).astype(np.float32)


def extract_strictly_local_epoch_features_v2(
    model,
    x_128hz: np.ndarray,
    n_epochs: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 31-dim strictly epoch-local features (11 AnySleep trunk + 20 physical/spectral)."""
    import torch

    epoch_samples = 3840
    ch = x_128hz.shape[0]
    usable = x_128hz[:, : n_epochs * epoch_samples].reshape(ch, n_epochs, epoch_samples)
    ep_raw = np.transpose(usable, (1, 2, 0))  # (E, 3840, C)
    ep_norm = epoch_local_anysleep_normalize(ep_raw)
    phys_feats = compute_epoch_local_physical_features(ep_raw)

    feats = []
    sc_layers = model.segment_classifier.segment_classifier
    with torch.inference_mode():
        for i in range(0, n_epochs, batch_size):
            b_tensor = torch.from_numpy(np.ascontiguousarray(ep_norm[i : i + batch_size])).to(device)
            bsz, _, n_ch = b_tensor.shape
            h = b_tensor.transpose(1, 2).reshape(bsz * n_ch, 1, -1)
            x_res = []
            for idx, enc in enumerate(model.encoders):
                x_r, h = enc(h)
                x_r = x_r.reshape(bsz, n_ch, *x_r.shape[1:])
                x_res.append(model.skip_connections[idx](x_r))
            h = model.connector(h).reshape(bsz, n_ch, -1, h.shape[-1])
            h = model.skip_connections[-1](h)
            for dec in model.decoders:
                h = dec(h, x_res.pop())
            h_pen = sc_layers[2](sc_layers[1](sc_layers[0](h))).squeeze(-1)  # (bsz, 6)
            h_log = sc_layers[5](sc_layers[4](sc_layers[3](h_pen.unsqueeze(-1)))).squeeze(-1)  # (bsz, 5)
            feats.append(torch.cat([h_pen, h_log], dim=1).detach().cpu().numpy())
    trunk_feat = np.concatenate(feats, axis=0)
    return np.concatenate([trunk_feat, phys_feats], axis=1).astype(np.float32)


def build_sequence_head(d_in: int, arm: str, context_radius: int = 10, hidden: int = 64):
    import torch
    from torch import nn
    from torch.nn.utils import spectral_norm

    kernel = 2 * context_radius + 1

    class SequenceHeadV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = arm
            self.local_head = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ELU(),
                nn.Linear(hidden, 5),
            )
            if arm != "local_only":
                use_sn = arm in ("rest_spectral_reg", "quality_gated_context", "constrained_residual")
                maybe_sn = spectral_norm if use_sn else (lambda m: m)
                self.in_proj = maybe_sn(nn.Linear(d_in, hidden))
                self.ctx_conv1 = maybe_sn(nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=context_radius))
                self.ctx_conv2 = maybe_sn(nn.Conv1d(hidden, 5, kernel_size=1))
                if arm in ("quality_gated_context", "constrained_residual"):
                    self.gate_thr = nn.Parameter(torch.tensor(0.5))
                    self.raw_cap = nn.Parameter(torch.tensor(-0.2))

        def forward(self, u: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
            loc_logits = self.local_head(u)
            if self.arm == "local_only":
                return loc_logits

            dev = torch.sqrt(torch.mean(u**2, dim=-1, keepdim=True) + 1e-6)
            if self.arm == "blind_mask_then_context":
                with torch.no_grad():
                    gate = (dev <= 2.8).to(u.dtype)
                u_ctx = gate * u
            elif self.arm in ("quality_gated_context", "constrained_residual"):
                thr = 1.6 + 2.4 * torch.sigmoid(self.gate_thr)
                gate = torch.sigmoid(4.0 * (thr - dev))
                u_ctx = gate * u
            else:
                u_ctx = u

            h = torch.nn.functional.elu(self.in_proj(u_ctx)).transpose(1, 2)
            if valid_mask is not None:
                h = h * valid_mask.unsqueeze(1)
            c1 = torch.nn.functional.elu(self.ctx_conv1(h))
            if valid_mask is not None:
                c1 = c1 * valid_mask.unsqueeze(1)
            ctx_delta = self.ctx_conv2(c1).transpose(1, 2)
            if valid_mask is not None:
                ctx_delta = ctx_delta * valid_mask.unsqueeze(-1)
            if self.arm in ("quality_gated_context", "constrained_residual"):
                cap = 0.35 + 1.15 * torch.sigmoid(self.raw_cap)
                ctx_delta = cap * torch.tanh(ctx_delta / cap)
            return loc_logits + ctx_delta

    return SequenceHeadV2()


def corrupt_sequence_batch_for_aug(
    u_batch: np.ndarray,
    y_batch: np.ndarray,
    valid_batch: np.ndarray,
    corrupt_bank: list[np.ndarray],
    rng: np.random.Generator,
    aug_prob: float = 0.75,
    context_radius: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject seen-family isolated corruptions on training fold sequences and return neighbor-invariance mask."""
    out = u_batch.copy()
    bsz, t_max, d_in = out.shape
    corrupt_mask = np.zeros((bsz, t_max), dtype=bool)
    neighbor_mask = np.zeros((bsz, t_max), dtype=bool)

    for b in range(bsz):
        if rng.random() > aug_prob:
            continue
        valid_idx = np.flatnonzero(valid_batch[b] & (y_batch[b] >= 0))
        if len(valid_idx) == 0:
            continue
        n_inj = max(1, min(4, len(valid_idx) // 500 + 1))
        chosen = rng.choice(valid_idx, size=min(n_inj, len(valid_idx)), replace=False)
        for ep in chosen:
            ep_i = int(ep)
            if corrupt_bank:
                donor = corrupt_bank[int(rng.integers(len(corrupt_bank)))]
                out[b, ep_i] = donor
            else:
                out[b, ep_i] = out[b, ep_i] * rng.uniform(0.0, 4.0) + rng.normal(0.0, 3.0, size=d_in).astype(np.float32)
            corrupt_mask[b, ep_i] = True
            lo = max(0, ep_i - context_radius)
            hi = min(t_max, ep_i + context_radius + 1)
            neighbor_mask[b, lo:hi] = True
    neighbor_mask = neighbor_mask & (~corrupt_mask) & valid_batch & (y_batch >= 0)
    return out, corrupt_mask, neighbor_mask


def make_cv_folds(participants: list[str], n_folds: int = 5) -> list[list[str]]:
    parts = sorted(participants)
    return [parts[i::n_folds] for i in range(n_folds) if parts[i::n_folds]]


def train_and_eval_arms_v2(
    clean_feats: dict[str, np.ndarray],
    truths: dict[str, np.ndarray],
    pert_feats: dict[tuple[str, float, str], np.ndarray],
    starts_ends: dict[tuple[str, float], tuple[int, int, float]],
    splits: dict[str, str],
    device: str = "cpu",
    train_seeds: list[int] | None = None,
    epochs: int = 75,
    context_radius: int = 10,
    positions: list[float] | None = None,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch = configure_torch_determinism(device, seed=20261117, strict=True)
    participants = sorted(clean_feats.keys())
    folds = make_cv_folds(participants, n_folds=min(5, max(2, len(participants) // 2)))
    d_in = next(iter(clean_feats.values())).shape[1]
    seeds = train_seeds or [20261117, 20261118, 20261119]
    pos_list = positions or [0.25, 0.5, 0.75]

    clean_eval_rows: list[dict] = []
    paired_eval_rows: list[dict] = []

    for fold_idx, val_parts in enumerate(folds):
        train_parts = [p for p in participants if p not in val_parts]
        cat_train = np.concatenate([clean_feats[p] for p in train_parts], axis=0)
        f_mean = cat_train.mean(axis=0, keepdims=True)
        # Floor std at 0.05 so rare binary indicators (flat_flag) cannot explode gradients on clean/OOD inputs
        f_std = np.maximum(cat_train.std(axis=0, keepdims=True), 0.05)

        norm_clean = {p: ((clean_feats[p] - f_mean) / f_std).astype(np.float32) for p in participants}
        norm_pert = {k: ((v - f_mean) / f_std).astype(np.float32) for k, v in pert_feats.items()}

        # Build SEEN corruption donor bank strictly from current training fold ONLY
        corrupt_bank: list[np.ndarray] = []
        for (p, pos, kind), u_p in norm_pert.items():
            if p in train_parts and kind in SEEN_KINDS:
                s_samp, _, sf = starts_ends[p, pos]
                ep_idx = int(s_samp // int(30 * sf))
                if 0 <= ep_idx < len(u_p):
                    corrupt_bank.append(u_p[ep_idx].copy())

        # Pack padded training batch once per fold (speedup ~16x on GPU)
        bsz = len(train_parts)
        t_max = max(len(norm_clean[p]) for p in train_parts)
        u_tr_np = np.zeros((bsz, t_max, d_in), dtype=np.float32)
        y_tr_np = np.full((bsz, t_max), -1, dtype=np.int64)
        m_tr_np = np.zeros((bsz, t_max), dtype=bool)
        for bi, p in enumerate(train_parts):
            ln = len(norm_clean[p])
            u_tr_np[bi, :ln] = norm_clean[p]
            y_tr_np[bi, :ln] = truths[p].astype(np.int64)
            m_tr_np[bi, :ln] = True

        u_tr_clean_t = torch.from_numpy(u_tr_np).to(device)
        y_tr_t = torch.from_numpy(y_tr_np).to(device)
        m_tr_t = torch.from_numpy(m_tr_np.astype(np.float32)).to(device)

        for t_seed in seeds:
            for arm_idx, arm in enumerate(ARMS):
                run_seed = int(t_seed + fold_idx * 100 + arm_idx)
                torch.manual_seed(run_seed)
                if device.startswith("cuda"):
                    torch.cuda.manual_seed_all(run_seed)
                rng = np.random.default_rng(run_seed)

                head = build_sequence_head(d_in, arm, context_radius=context_radius, hidden=64).to(device)
                opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
                loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

                head.train()
                for _ in range(epochs):
                    opt.zero_grad()
                    if arm in ("matched_pert_aug", "rest_spectral_reg", "constrained_residual"):
                        u_aug_np, _, n_mask_np = corrupt_sequence_batch_for_aug(
                            u_tr_np, y_tr_np, m_tr_np, corrupt_bank, rng, aug_prob=0.75, context_radius=context_radius
                        )
                        u_in = torch.from_numpy(u_aug_np).to(device)
                    else:
                        n_mask_np = None
                        u_in = u_tr_clean_t

                    if arm == "rest_spectral_reg":
                        jitter = u_in + 0.15 * torch.randn_like(u_in)
                        u_both = torch.cat([u_in, jitter], dim=0)
                        m_both = torch.cat([m_tr_t, m_tr_t], dim=0)
                        out_both = head(u_both, valid_mask=m_both)
                        logits, jitter_logits = out_both[:bsz], out_both[bsz:]
                        loss = loss_fn(logits.reshape(-1, 5), y_tr_t.reshape(-1))
                        loss = loss + 0.25 * torch.mean((jitter_logits - logits) ** 2)
                    elif arm == "constrained_residual" and n_mask_np is not None and np.any(n_mask_np):
                        u_both = torch.cat([u_in, u_tr_clean_t], dim=0)
                        m_both = torch.cat([m_tr_t, m_tr_t], dim=0)
                        out_both = head(u_both, valid_mask=m_both)
                        logits, clean_logits = out_both[:bsz], out_both[bsz:].detach()
                        loss = loss_fn(logits.reshape(-1, 5), y_tr_t.reshape(-1))
                        nm_t = torch.from_numpy(n_mask_np).to(device)
                        loss = loss + 0.60 * torch.mean((logits[nm_t] - clean_logits[nm_t]) ** 2)
                    else:
                        logits = head(u_in, valid_mask=m_tr_t)
                        loss = loss_fn(logits.reshape(-1, 5), y_tr_t.reshape(-1))

                    loss.backward()
                    opt.step()

                head.eval()
                with torch.inference_mode():
                    for p in val_parts:
                        u_val = torch.from_numpy(norm_clean[p]).unsqueeze(0).to(device)
                        base_logits = head(u_val).squeeze(0).cpu().numpy()
                        base_pred = base_logits.argmax(axis=1).astype(np.int16)
                        cm = compute_clean_metrics(truths[p], base_pred)
                        cm.update({
                            "participant": p,
                            "fold": fold_idx,
                            "train_seed": t_seed,
                            "arm": arm,
                            "anysleep_published_split": splits[p],
                        })
                        clean_eval_rows.append(cm)

                        n_ep = len(truths[p])
                        for pos in pos_list:
                            s_samp, e_samp, sf = starts_ends[p, pos]
                            for kind in ALL_TEST_KINDS:
                                if kind == "noop":
                                    u_p_np = norm_clean[p]
                                elif (p, pos, kind) in norm_pert:
                                    u_p_np = norm_pert[p, pos, kind]
                                else:
                                    continue

                                changed_u = np.any(np.abs(u_p_np - norm_clean[p]) > 1e-6, axis=1)
                                loc_sup = np.zeros(n_ep, dtype=bool)
                                if arm == "local_only":
                                    loc_sup = changed_u.copy()
                                else:
                                    for idx_c in np.flatnonzero(changed_u):
                                        lo = max(0, int(idx_c) - context_radius)
                                        hi = min(n_ep, int(idx_c) + context_radius + 1)
                                        loc_sup[lo:hi] = True

                                pert_logits = head(torch.from_numpy(u_p_np).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                                pert_pred = pert_logits.argmax(axis=1).astype(np.int16)

                                if kind == "noop" and not np.array_equal(base_pred, pert_pred):
                                    raise AssertionError(f"Noop mismatch in {arm} (seed={t_seed}) on {p}")

                                outside = ~loc_sup
                                if np.any(outside):
                                    max_out = float(np.max(np.abs(pert_logits[outside] - base_logits[outside])))
                                    if max_out > 1e-5:
                                        raise AssertionError(f"Context leakage beyond K={context_radius} in {arm}: {max_out}")

                                kind_group = "noop" if kind == "noop" else ("seen_corruption" if kind in SEEN_KINDS else "unseen_ood_corruption")
                                for r in compute_region_metrics_extended(truths[p], base_pred, pert_pred, sf, s_samp, e_samp, loc_sup):
                                    paired_eval_rows.append({
                                        "participant": p,
                                        "fold": fold_idx,
                                        "train_seed": t_seed,
                                        "arm": arm,
                                        "position": pos,
                                        "kind": kind,
                                        "kind_group": kind_group,
                                        "anysleep_published_split": splits[p],
                                        **r,
                                    })
        del u_tr_clean_t, y_tr_t, m_tr_t
        gc.collect()

    return {"clean_rows": clean_eval_rows, "paired_rows": paired_eval_rows}


def aggregate_small_model_v2(out_dir: Path, clean_rows: list[dict], paired_rows: list[dict], plan: dict) -> dict[str, Any]:
    import pandas as pd

    agg_dir = out_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    seed = int(plan["analysis"]["bootstrap_seed"])
    reps = int(plan["analysis"]["bootstrap_replicates"])

    cdf = pd.DataFrame(clean_rows)
    cdf.to_csv(agg_dir / "small_model_clean_seed_metrics.csv", index=False)
    clean_part = cdf.groupby(["participant", "arm", "anysleep_published_split"], as_index=False).agg(
        seeds=("train_seed", "nunique"),
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        cohen_kappa=("cohen_kappa", "mean"),
    )
    clean_part.to_csv(agg_dir / "small_model_clean_participant_metrics.csv", index=False)

    clean_grp = clean_part.groupby("arm", as_index=False).agg(
        n_participants=("participant", "nunique"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        cohen_kappa_mean=("cohen_kappa", "mean"),
    )
    clean_grp.to_csv(agg_dir / "small_model_clean_group_metrics.csv", index=False)

    pdf = pd.DataFrame(paired_rows)
    pdf.to_csv(agg_dir / "small_model_position_seed_metrics.csv", index=False)

    # First average across 3 training seeds -> then across 3 positions -> participant-level unit!
    part_seed = pdf.groupby(
        ["participant", "anysleep_published_split", "arm", "kind", "kind_group", "region", "position"],
        as_index=False,
    ).agg(
        new_error_rate=("new_error_rate", "mean"),
        corrected_rate=("corrected_rate", "mean"),
        net_error=("net_error", "mean"),
        flip_rate=("flip_rate", "mean"),
    )
    part_p = part_seed.groupby(
        ["participant", "anysleep_published_split", "arm", "kind", "kind_group", "region"],
        as_index=False,
    ).agg(
        positions=("position", "nunique"),
        new_error_rate=("new_error_rate", "mean"),
        corrected_rate=("corrected_rate", "mean"),
        net_error=("net_error", "mean"),
        flip_rate=("flip_rate", "mean"),
    )
    part_p.to_csv(agg_dir / "small_model_participant_metrics.csv", index=False)

    grp_rows = []
    for (arm, kind, kind_grp, region), sub in part_p.groupby(["arm", "kind", "kind_group", "region"]):
        vals = sub.sort_values("participant")["net_error"].to_numpy()
        lo, hi = participant_bootstrap_ci(vals, seed=seed, replicates=reps)
        grp_rows.append({
            "arm": arm,
            "kind": kind,
            "kind_group": kind_grp,
            "region": region,
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
    grp_df = pd.DataFrame(grp_rows)
    grp_df.to_csv(agg_dir / "small_model_group_metrics.csv", index=False)

    # Also summarize seen vs unseen OOD grand averages per arm
    fam_part = (
        part_p.loc[(part_p["region"] == "near_30_300s") & (part_p["kind"] != "noop")]
        .groupby(["participant", "arm", "kind_group"], as_index=False)
        .agg(net_error=("net_error", "mean"), flip_rate=("flip_rate", "mean"))
    )
    fam_rows = []
    for (arm, kg), sub in fam_part.groupby(["arm", "kind_group"]):
        vals = sub.sort_values("participant")["net_error"].to_numpy()
        lo, hi = participant_bootstrap_ci(vals, seed=seed, replicates=reps)
        fam_rows.append({
            "arm": arm,
            "kind_group": kg,
            "n_participants": int(len(vals)),
            "net_error": float(vals.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "flip_rate": float(sub["flip_rate"].mean()),
        })
    pd.DataFrame(fam_rows).to_csv(agg_dir / "small_model_seen_vs_ood_summary.csv", index=False)

    summary = {
        "updated_utc": utc_now(),
        "n_participants": len(clean_part["participant"].unique()),
        "arms": ARMS,
        "clean_summary": clean_grp.to_dict(orient="records"),
        "seen_vs_ood_summary": fam_rows,
        "near_30_300s": [r for r in grp_rows if r["region"] == "near_30_300s" and r["kind"] != "noop"],
    }
    write_json(agg_dir / "small_model_v2_summary.json", summary)
    return summary


def run_experiment_v2(args) -> None:
    from anysleep_no_hydra import AnySleep

    project = Path(args.project_root).resolve()
    plan = read_json(args.manifest) or build_manifest_20()
    base_cfg = plan["base_config"]
    out_dir = project / "results" / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"

    dev = resolve_device(args.device)
    configure_torch_determinism(dev, seed=int(base_cfg["noise_seed"]), strict=True)
    ckpt = ensure_checkpoint(project, "run1")
    model = AnySleep(path=str(ckpt), sleep_stage_frequency=1).eval().to(dev)

    cache_dir = out_dir / "local_features_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_feats, truths, pert_feats, starts_ends, splits = {}, {}, {}, {}, {}

    expected_keys = ["clean", "truth"] + [
        f"p{int(pos * 100):02d}_{k}" for pos in plan["positions"] for k in (SEEN_KINDS + OOD_KINDS)
    ]

    for rec in plan["participants"]:
        p = rec["participant"]
        splits[p] = rec["anysleep_published_split"]
        p_cache = cache_dir / f"{p}.npz"
        download_verified(rec["psg_url"], project / "data" / rec["psg"], expected_sha256=rec["psg_sha256"])
        download_verified(rec["hypnogram_url"], project / "data" / rec["hypnogram"], expected_sha256=rec["hypnogram_sha256"])

        if p_cache.is_file():
            try:
                loaded = np.load(p_cache, allow_pickle=False)
                if all(k in loaded.files for k in expected_keys):
                    truth = loaded["truth"].astype(np.int16)
                    truths[p] = truth
                    clean_feats[p] = loaded["clean"].astype(np.float32)
                    sf_val = float(loaded["sf"][0]) if "sf" in loaded.files else 100.0
                    for pos in plan["positions"]:
                        ch_ep = chosen_epoch_for_fraction(truth, pos)
                        starts_ends[p, pos] = (int(ch_ep * 30 * sf_val), int((ch_ep + 1) * 30 * sf_val), sf_val)
                        for k in SEEN_KINDS + OOD_KINDS:
                            pert_feats[p, pos, k] = loaded[f"p{int(pos * 100):02d}_{k}"].astype(np.float32)
                    print(f"[SmallModel-v2] Loaded cached 31-dim local features for {p}", flush=True)
                    continue
            except Exception as exc:
                print(f"[SmallModel-v2] Cache invalidated for {p} ({exc}); rebuilding...", flush=True)

        print(f"[SmallModel-v2] Extracting 31-dim strictly-local features for {p} on {dev}...", flush=True)
        raw_obj, x, sf, truth, _ = load_sleep_edf_record(project, rec, base_cfg)
        del raw_obj
        n_ep = len(truth)
        truths[p] = truth
        raw_128 = resample_channels_128hz(x, sf)
        u_clean = extract_strictly_local_epoch_features_v2(model, raw_128, n_ep, dev)
        clean_feats[p] = u_clean
        arrays_to_save: dict[str, np.ndarray] = {"clean": u_clean, "truth": truth, "sf": np.array([sf], dtype=np.float64)}

        for pos in plan["positions"]:
            ep_idx = chosen_epoch_for_fraction(truth, pos)
            s_samp = int(ep_idx * 30 * sf)
            e_samp = s_samp + int(base_cfg["duration_seconds"] * sf)
            starts_ends[p, pos] = (s_samp, e_samp, sf)

            # Extract over a 5-epoch native window [ep-2 : ep+3], keep middle 3 [ep-1 : ep+2]
            w_lo = max(0, ep_idx - 2)
            w_hi = min(n_ep, ep_idx + 3)
            s_lo = int(w_lo * 30 * sf)
            s_hi = int(w_hi * 30 * sf)
            x_sub = x[:, s_lo:s_hi].copy()
            lo = max(0, ep_idx - 1)
            hi = min(n_ep, ep_idx + 2)

            for kind in SEEN_KINDS + OOD_KINDS:
                y_sub = perturb_extended(
                    x_sub,
                    base_cfg["corrupt_channels"],
                    s_samp - s_lo,
                    e_samp - s_lo,
                    kind,
                    sf=sf,
                    seed=int(base_cfg["noise_seed"]),
                    clip_volts=float(base_cfg["clip_volts"]),
                    noise_sd_volts=float(base_cfg["noise_sd_volts"]),
                )
                z1_sub = resample_channels_128hz(y_sub, sf)
                sub_128 = z1_sub[:, (lo - w_lo) * 3840 : (hi - w_lo) * 3840]
                u_p = u_clean.copy()
                u_p[lo:hi] = extract_strictly_local_epoch_features_v2(model, sub_128, hi - lo, dev)
                pert_feats[p, pos, kind] = u_p
                arrays_to_save[f"p{int(pos * 100):02d}_{kind}"] = u_p
        del x, raw_128
        gc.collect()
        tmp_npz = cache_dir / f"{p}.tmp.npz"
        np.savez_compressed(tmp_npz, **arrays_to_save)
        tmp_npz.replace(p_cache)

    res = train_and_eval_arms_v2(
        clean_feats,
        truths,
        pert_feats,
        starts_ends,
        splits,
        device=dev,
        train_seeds=[20261117, 20261118, 20261119],
        epochs=args.epochs,
        context_radius=10,
        positions=list(plan["positions"]),
    )
    summary = aggregate_small_model_v2(out_dir, res["clean_rows"], res["paired_rows"], plan)
    write_json(
        state_file,
        {
            "stage_id": EXP_ID,
            "status": "COMPLETE",
            "n_participants": len(clean_feats),
            "feature_dim": int(next(iter(clean_feats.values())).shape[1]),
            "completed_utc": utc_now(),
        },
    )
    sync_filesystem()
    print("STAGE_C_SMALL_MODEL_V2_COMPLETE", summary.get("n_participants"), flush=True)


def selftest() -> None:
    rng = np.random.default_rng(20261117)
    n_test_ep, epoch_samples, ch = 12, 3840, 3
    ep_raw = rng.normal(0.0, 2e-5, size=(n_test_ep, epoch_samples, ch)).astype(np.float64)
    ep_norm = epoch_local_anysleep_normalize(ep_raw)
    phys = compute_epoch_local_physical_features(ep_raw)
    assert ep_norm.shape == (n_test_ep, epoch_samples, ch) and np.isfinite(ep_norm).all()
    assert phys.shape == (n_test_ep, 20) and np.isfinite(phys).all()
    # Strictly epoch-local invariance check: perturbing epoch 5 must leave all epochs != 5 bit-identical
    ep_mut = ep_raw.copy()
    ep_mut[5, :, 0] += 5e-4
    phys_mut = compute_epoch_local_physical_features(ep_mut)
    ep_norm_mut = epoch_local_anysleep_normalize(ep_mut)
    assert np.array_equal(phys[:5], phys_mut[:5]) and np.array_equal(phys[6:], phys_mut[6:])
    assert np.array_equal(ep_norm[:5], ep_norm_mut[:5]) and np.array_equal(ep_norm[6:], ep_norm_mut[6:])
    assert len(ARMS) == 7
    print("STAGE_C_SMALL_MODEL_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    r.add_argument("--manifest", type=Path, default=ROOT / "manifests/development_manifest_20.json")
    r.add_argument("--epochs", type=int, default=75)
    r.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
    else:
        run_experiment_v2(a)


if __name__ == "__main__":
    main()
