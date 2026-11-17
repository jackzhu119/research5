"""Utilities for a paired local-corruption pilot; no model is bundled.

The evaluation expects one row per scored epoch AND perturbation instance. Input
columns: participant, record, perturbation_id, epoch_start, epoch_end,
corrupt_start, corrupt_end (all in original samples; end-exclusive), truth,
baseline_pred, perturbed_pred. See README.md before using on actual EEG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def perturb(x, channels, start, end, kind, seed=0, clip=None, noise_sd=None):
    """Change selected channels only within [start,end), in native signal units."""
    x = np.asarray(x)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError('x must be floating-point [channels, samples]')
    if not (0 <= start < end <= x.shape[1]):
        raise ValueError('invalid half-open corruption interval')
    if not channels or len(set(channels)) != len(channels) or min(channels) < 0 or max(channels) >= x.shape[0]:
        raise ValueError('invalid channels')
    y = x.copy()
    if kind == 'noop':
        pass
    elif kind == 'flat':
        y[channels, start:end] = 0
    elif kind == 'clip':
        if clip is None or not np.isfinite(clip) or clip <= 0:
            raise ValueError('clip must be finite and positive, in input units')
        y[channels, start:end] = np.clip(x[channels, start:end], -clip, clip)
    elif kind == 'noise':
        if noise_sd is None or not np.isfinite(noise_sd) or noise_sd <= 0:
            raise ValueError('noise_sd must be finite and positive, in input units')
        rng = np.random.default_rng(seed)
        y[channels, start:end] += rng.normal(0, noise_sd, (len(channels), end - start))
    else:
        raise ValueError('unsupported kind')
    keep = np.ones(x.shape, dtype=bool)
    keep[channels, start:end] = False
    if not np.array_equal(y[keep], x[keep], equal_nan=True):
        raise AssertionError('samples outside targeted channels/interval changed')
    return y


def analyze(path, outdir, sfreq, guard_seconds=0):
    """Compute transitions per instance then average instances within participant.

    Guard is a separately labelled filtering-uncertainty margin; 0 does not
    prove a filter's support is zero. Distances use epoch-edge to artifact-edge.
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols = {'participant', 'record', 'perturbation_id', 'epoch_start', 'epoch_end',
            'corrupt_start', 'corrupt_end', 'truth', 'baseline_pred', 'perturbed_pred'}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    if sfreq <= 0 or guard_seconds < 0:
        raise ValueError('sfreq must be positive and guard_seconds nonnegative')
    if df.duplicated(['participant', 'record', 'perturbation_id', 'epoch_start']).any():
        raise ValueError('duplicate epoch for perturbation instance')
    for col in ['epoch_start', 'epoch_end', 'corrupt_start', 'corrupt_end']:
        vals = pd.to_numeric(df[col], errors='raise')
        if vals.isna().any() or (vals != np.floor(vals)).any():
            raise ValueError(f'{col} must contain integer sample indices')
        df[col] = vals.astype('int64')
    if ((df.epoch_start < 0) | (df.corrupt_start < 0) | (df.epoch_end <= df.epoch_start) | (df.corrupt_end <= df.corrupt_start)).any():
        raise ValueError('invalid sample intervals')
    for col in ['truth', 'baseline_pred', 'perturbed_pred']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    valid_truth = df.truth.isin([0, 1, 2, 3, 4])
    if not df.loc[valid_truth, ['baseline_pred', 'perturbed_pred']].isin([0, 1, 2, 3, 4]).all().all():
        raise ValueError('predictions on scored epochs must use stages 0..4')
    overlap = (df.epoch_start < df.corrupt_end) & (df.epoch_end > df.corrupt_start)
    distance = np.maximum(np.maximum(df.corrupt_start - df.epoch_end, df.epoch_start - df.corrupt_end), 0) / sfreq
    df['distance_s'] = distance
    df['region'] = pd.cut(distance, bins=[-1, 30, 60, 120, 300, float('inf')],
                          labels=['0-30s', '30-60s', '60-120s', '120-300s', '300s+'], right=False).astype(object)
    # Exact zero distance is in 0-30s; right=False makes each upper bound exclusive.
    df.loc[overlap, 'region'] = 'corrupted'
    df.loc[~overlap & (distance < guard_seconds), 'region'] = 'filter_guard'
    df['scored'] = valid_truth
    df['new_error'] = valid_truth & (df.baseline_pred == df.truth) & (df.perturbed_pred != df.truth)
    df['corrected'] = valid_truth & (df.baseline_pred != df.truth) & (df.perturbed_pred == df.truth)
    df['flipped'] = valid_truth & (df.baseline_pred != df.perturbed_pred)
    df['net_error'] = df.new_error.astype(int) - df.corrected.astype(int)
    keys = ['participant', 'record', 'perturbation_id', 'region']
    scored = df.loc[valid_truth]
    instance = scored.groupby(keys, observed=True, as_index=False).agg(
        epochs=('truth', 'size'), new_error=('new_error', 'mean'), corrected=('corrected', 'mean'),
        flip_rate=('flipped', 'mean'), net_error=('net_error', 'mean'))
    # Equal weighting of perturbation instances per participant, regardless of
    # epoch count. For a multi-night protocol, predefine weighting separately.
    participant = instance.groupby(['participant', 'region'], observed=True, as_index=False).agg(
        instances=('net_error', 'size'), epochs=('epochs', 'sum'), new_error=('new_error', 'mean'),
        corrected=('corrected', 'mean'), flip_rate=('flip_rate', 'mean'), net_error=('net_error', 'mean'))
    group = participant.groupby('region', observed=True, as_index=False).agg(
        participants=('participant', 'size'), new_error=('new_error', 'mean'),
        corrected=('corrected', 'mean'), flip_rate=('flip_rate', 'mean'), net_error=('net_error', 'mean'))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'epoch_transitions.csv', index=False)
    instance.to_csv(out / 'per_instance.csv', index=False)
    participant.to_csv(out / 'per_participant.csv', index=False)
    group.to_csv(out / 'summary.csv', index=False)
    (out / 'analysis_metadata.json').write_text(json.dumps({'sampling_frequency_hz': sfreq,
        'guard_seconds': guard_seconds, 'guard_note': 'must be justified by filter support',
        'distance_definition': 'gap between half-open epoch and perturbation intervals, before guard',
        'bin_edges_seconds': [0, 30, 60, 120, 300], 'valid_truth': [0, 1, 2, 3, 4],
        'aggregation': 'equal perturbation instances per participant; equal participants in group'}, indent=2), encoding='utf-8')
    return group


def selftest():
    import tempfile
    import pandas as pd
    x = np.arange(30, dtype=float).reshape(3, 10)
    assert np.array_equal(perturb(x, [0], 2, 5, 'noop'), x)
    for kind, kw in [('flat', {}), ('clip', {'clip': 2}), ('noise', {'noise_sd': 1})]:
        y = perturb(x, [0], 2, 5, kind, **kw)
        assert np.array_equal(x[1:], y[1:]) and np.array_equal(x[0, :2], y[0, :2]) and np.array_equal(x[0, 5:], y[0, 5:])
    assert np.array_equal(perturb(x, [0], 2, 5, 'noise', seed=7, noise_sd=1), perturb(x, [0], 2, 5, 'noise', seed=7, noise_sd=1))
    with tempfile.TemporaryDirectory() as tmp:
        rows = [dict(participant='P1', record='R1', perturbation_id='A', epoch_start=30, epoch_end=60,
                     corrupt_start=0, corrupt_end=10, truth=0, baseline_pred=0, perturbed_pred=1),
                dict(participant='P1', record='R1', perturbation_id='A', epoch_start=60, epoch_end=90,
                     corrupt_start=0, corrupt_end=10, truth=0, baseline_pred=1, perturbed_pred=0)]
        path = Path(tmp) / 'fake.csv'
        pd.DataFrame(rows).to_csv(path, index=False)
        result = analyze(path, Path(tmp) / 'out', 1)
        assert result.loc[result.region == '0-30s', 'net_error'].iloc[0] == 1
        assert result.loc[result.region == '30-60s', 'net_error'].iloc[0] == -1
    print('SYNTHETIC_SELFTEST_OK (no real EEG or model inference)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selftest')
    a = sub.add_parser('analyze')
    a.add_argument('--pairs', required=True)
    a.add_argument('--outdir', required=True)
    a.add_argument('--sfreq', type=float, required=True)
    a.add_argument('--guard-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest()
    else:
        print(analyze(args.pairs, args.outdir, args.sfreq, args.guard_seconds).to_string(index=False))
