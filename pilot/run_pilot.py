"""One-record development pilot. No training, no silent cropping or chunking.

Run from the repository root. `prepare` downloads only the two named EDF files
and one pinned checkpoint. `run` writes results incrementally and marks failures.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/anysleep/examples'))
from paired_tools import perturb, analyze
from support import anysleep_local_support


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def digest(path, algo='sha256', git_blob=False):
    h = hashlib.new(algo)
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    import requests
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with tmp.open('wb') as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
    tmp.replace(dest)


def prepare(cfg):
    source = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
    files = [(source + cfg['psg'], ROOT / 'data' / cfg['psg']),
             (source + cfg['hypnogram'], ROOT / 'data' / cfg['hypnogram']),
             (f"https://raw.githubusercontent.com/dslaborg/AnySleep/{cfg['anysleep_commit']}/models/{cfg['checkpoint']}",
              ROOT / 'models' / cfg['checkpoint'])]
    manifest = []
    for url, dest in files:
        print('Downloading/checking', dest.name, flush=True)
        download(url, dest)
        row = {'url': url, 'path': str(dest.relative_to(ROOT)), 'bytes': dest.stat().st_size,
               'sha256': digest(dest), 'git_blob_sha1': digest(dest, 'sha1', True)}
        if dest.suffix == '.pth':
            if row['git_blob_sha1'] != cfg['checkpoint_git_blob']:
                raise ValueError('Checkpoint differs from the pinned GitHub blob')
        manifest.append(row)
        write_json(ROOT / 'manifests/downloads.json', manifest)
    # Do not invent a provider digest when this version does not expose one.
    for name in ['SHA256SUMS.txt', 'SHA256SUMS']:
        try:
            download('https://physionet.org/files/sleep-edfx/1.0.0/' + name,
                     ROOT / 'manifests' / ('physionet-' + name))
            break
        except Exception as exc:
            print('Optional provider digest list unavailable:', type(exc).__name__, flush=True)
    print('PREPARE_COMPLETE', flush=True)


def environment(out):
    import torch
    import psutil
    versions = {}
    for key in ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'mne', 'yasa', 'lightgbm', 'antropy', 'psutil']:
        try:
            versions[key] = md.version(key)
        except md.PackageNotFoundError:
            versions[key] = None
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpu.append({'index': i, 'name': torch.cuda.get_device_name(i), 'free_bytes': free, 'total_bytes': total})
    write_json(out / 'environment.json', {'python': sys.version, 'platform': platform.platform(),
               'versions': versions, 'gpu': gpu, 'cuda': torch.version.cuda,
               'ram_available_bytes': psutil.virtual_memory().available, 'pid': os.getpid()})
    (out / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    provenance = json.loads((ROOT / 'manifests/anysleep_sources.json').read_text())
    for row in provenance['files']:
        path = ROOT / 'vendor/anysleep' / row['path']
        if digest(path, 'sha1', True) != row['expected_git_blob_sha1']:
            raise ValueError(f'Vendored source changed: {row["path"]}')
    write_json(out / 'source_manifest.json', provenance)


def baseline_scores(out, truth, pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
    keep = truth >= 0
    write_json(out / 'baseline_scores.json', {'scored_epochs': int(keep.sum()),
        'accuracy': accuracy_score(truth[keep], pred[keep]),
        'macro_f1_labels_0_to_4': f1_score(truth[keep], pred[keep], labels=[0, 1, 2, 3, 4], average='macro', zero_division=0),
        'cohen_kappa': cohen_kappa_score(truth[keep], pred[keep]),
        'confusion_matrix_true_rows_predicted_columns_0_to_4': confusion_matrix(truth[keep], pred[keep], labels=[0, 1, 2, 3, 4]).tolist(),
        'interpretation': 'one development record; data/model overlap not ruled out'})


def read_record(cfg, out):
    import mne
    raw = mne.io.read_raw_edf(ROOT / 'data' / cfg['psg'], include=cfg['channels'], preload=True, verbose='warning')
    raw.reorder_channels(cfg['channels'])
    raw.set_channel_types({c: ('eog' if c.startswith('EOG') else 'eeg') for c in cfg['channels']}, verbose=False)
    sf = float(raw.info['sfreq'])
    if sf != round(sf):
        raise ValueError('Noninteger sample rate requires a separately audited alignment')
    x = raw.get_data()  # MNE uses volts for EEG/EOG.
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite input; define an explicit handling protocol before inference')
    ann = mne.read_annotations(ROOT / 'data' / cfg['hypnogram'])
    header_origins = {}
    for role, filename in [('psg', cfg['psg']), ('hypnogram', cfg['hypnogram'])]:
        with (ROOT / 'data' / filename).open('rb') as file:
            value = file.read(256)[168:184].decode('ascii')
        if not re.fullmatch(r'\d{2}\.\d{2}\.\d{2}\d{2}\.\d{2}\.\d{2}', value):
            raise ValueError(f'Missing or malformed {role} EDF date/time header')
        header_origins[role] = {'literal_header_bytes_168_to_183': value,
            'parsed_calendar_local_without_timezone': datetime.datetime.strptime(value, '%d.%m.%y%H.%M.%S').isoformat()}
    if ann.orig_time is not None and raw.info['meas_date'] is not None:
        offset = (ann.orig_time - raw.info['meas_date']).total_seconds()
        origin_method = 'MNE absolute annotation origin minus PSG measurement date'
    elif header_origins['psg']['literal_header_bytes_168_to_183'] == header_origins['hypnogram']['literal_header_bytes_168_to_183']:
        # read_annotations may drop the EDF origin. Matching explicit EDF date/time
        # headers establish the same origin; no calendar/timezone offset is guessed.
        offset = 0.0
        origin_method = 'identical explicitly parsed PSG/hypnogram EDF date/time headers; MNE annotation origin absent'
    else:
        raise ValueError('MNE origin absent and EDF header origins differ; stop for explicit alignment audit')
    n_epochs = int(x.shape[1] // (30 * sf))
    truth = np.full(n_epochs, -1, dtype=np.int16)
    for onset, duration, desc in zip(ann.onset + offset, ann.duration, ann.description):
        if desc not in cfg['label_map']:
            continue
        # Only fully contained epochs receive a scored label.
        first = max(0, int(math.ceil((onset - 1e-7) / 30)))
        last = min(n_epochs, int(math.floor((onset + duration + 1e-7) / 30)))
        if first < last:
            if (truth[first:last] >= 0).any():
                raise ValueError('Overlapping stage annotations')
            truth[first:last] = cfg['label_map'][desc]
    if not (truth >= 0).any():
        raise ValueError('No aligned scored epochs')
    valid = np.flatnonzero(truth >= 0)
    chosen = int(valid[min(len(valid) - 1, int(len(valid) * cfg['position_fraction']))])
    start = int(chosen * 30 * sf)
    end = start + int(cfg['duration_seconds'] * sf)
    if end > x.shape[1]:
        raise ValueError('Predeclared interval extends beyond EDF')
    write_json(out / 'record.json', {'record': cfg['record'], 'participant': cfg['participant'],
        'sfreq_hz': sf, 'samples': x.shape[1], 'duration_seconds': x.shape[1] / sf,
        'channels': raw.ch_names, 'units': 'V from MNE; YASA internally converts to microvolts',
        'peak_absolute_volts': np.max(np.abs(x), axis=1).tolist(),
        'channel_iqr_volts': np.subtract(*np.percentile(x, [75, 25], axis=1)).tolist(),
        'annotation_offset_seconds': offset, 'psg_meas_date': raw.info['meas_date'],
        'annotation_orig_time': ann.orig_time, 'origin_method': origin_method, 'edf_header_origin_evidence': header_origins,
        'scored_epochs': int((truth >= 0).sum()),
        'unscored_epochs': int((truth < 0).sum()), 'tail_samples_not_scored': x.shape[1] - int(n_epochs * 30 * sf),
        'corrupt_interval_samples': [start, end], 'position_rule': cfg['position_rule'],
        'raw_unchanged_assertion': 'checked on every generated intervention before preprocessing'})
    np.save(out / 'truth.npy', truth)
    return raw, x, sf, truth, start, end


def preprocess(x, sf, fixed=None):
    from scipy.signal import resample_poly
    from sklearn.preprocessing import RobustScaler
    z = resample_poly(x, 128, int(sf), axis=1) if sf != 128 else x.copy()
    if fixed is None:
        threshold = np.subtract(*np.percentile(z, [75, 25], axis=1)) * 20
        clipped = np.clip(z, -threshold[:, None], threshold[:, None])
        scaler = RobustScaler().fit(clipped.T)
        fixed = {'threshold': threshold, 'center': scaler.center_, 'scale': scaler.scale_}
    else:
        clipped = np.clip(z, -fixed['threshold'][:, None], fixed['threshold'][:, None])
    return ((clipped - fixed['center'][:, None]) / fixed['scale'][:, None]), fixed, z


class AttentionProbe:
    def __init__(self, model):
        self.values, self.fixed, self.handles = {}, None, []
        for i, block in enumerate(model.skip_connections):
            def hook(module, args, output, i=i):
                if self.fixed is not None:
                    self.values[i] = self.fixed[i].copy()
                    return output.new_tensor(self.fixed[i])
                self.values[i] = output.detach().cpu().numpy().copy()
            self.handles.append(block.softmax.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_anysleep(cfg, out, x, sf, truth, start, end, device, pilot, diagnostics):
    import torch
    import psutil
    from anysleep_no_hydra import AnySleep
    from predict_edf_file_logits_plain import preprocess as official_preprocess
    checkpoint = ROOT / 'models' / cfg['checkpoint']
    if digest(checkpoint, 'sha1', True) != cfg['checkpoint_git_blob']:
        raise ValueError('Checkpoint differs from the pinned Git blob at inference time')
    write_json(out / 'checkpoint.json', {'path': str(checkpoint), 'sha256': digest(checkpoint),
        'git_blob_sha1': cfg['checkpoint_git_blob'], 'source_commit': cfg['anysleep_commit']})
    torch.manual_seed(cfg['noise_seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = AnySleep(path=str(ROOT / 'models' / cfg['checkpoint']), sleep_stage_frequency=1).eval().to(device)
    probe = AttentionProbe(model)
    z0, stats0, raw_resampled = preprocess(x, sf)
    # First prove the independently instrumented preprocessing matches the pinned example.
    reference = official_preprocess(dict(zip(cfg['channels'], x.copy())), int(sf), 128)
    np.testing.assert_allclose(z0, np.asarray([reference[c] for c in cfg['channels']]), rtol=1e-7, atol=1e-8)
    del reference
    np.savez(out / 'baseline_preprocess_stats.npz', **stats0)
    run_log = []

    def infer(z, name, fixed=None):
        probe.fixed = fixed
        probe.values = {}
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(z.T)).float().unsqueeze(0).to(device)
            logits = model(tensor).detach().cpu().numpy()[0]
        if len(logits) != len(truth):
            raise ValueError(f'epoch count mismatch: logits={len(logits)}, labels={len(truth)}')
        np.save(out / f'{name}_logits.npy', logits)
        np.savez(out / f'{name}_attention.npz', **{f'layer_{i}': v for i, v in probe.values.items()})
        run_log.append({'name': name, 'seconds': time.monotonic() - t,
            'rss_after_bytes': psutil.Process().memory_info().rss,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.startswith('cuda') else None,
            'oracle_statistics': ('fixedstats' in name or 'fixedboth' in name), 'oracle_attention': fixed is not None,
            'complete_edf': True, 'chunks': 1})
        write_json(out / 'inference_runs.json', run_log)
        return logits, {i: v.copy() for i, v in probe.values.items()}

    try:
        base, alpha0 = infer(z0, 'baseline')
        baseline_scores(out, truth, base.argmax(1))
        repeat, _ = infer(z0, 'repeat')
        write_json(out / 'repeatability.json', {'max_absolute_logit_delta': float(np.max(np.abs(base - repeat))),
                   'prediction_flips': int((base.argmax(1) != repeat.argmax(1)).sum()),
                   'allclose_rtol_1e-6_atol_1e-7': bool(np.allclose(base, repeat, rtol=1e-6, atol=1e-7))})
        if not np.allclose(base, repeat, rtol=1e-6, atol=1e-7) or (base.argmax(1) != repeat.argmax(1)).any():
            raise RuntimeError('Baseline inference is not repeatable; stop before perturbation interpretation')
        if not pilot:
            return
        pairs = []
        # Direct local FIR footprint, before recomputing any global statistics.
        for kind in cfg['kinds']:
            y = perturb(x, cfg['corrupt_channels'], start, end, kind, seed=cfg['noise_seed'],
                        clip=cfg['clip_volts'], noise_sd=cfg['noise_sd_volts'])
            if kind == 'noop' and not np.array_equal(x, y):
                raise AssertionError('No-op altered samples')
            z1, stats1, resampled1 = preprocess(y, sf)
            changed = np.any(raw_resampled != resampled1, axis=0)
            np.save(out / f'{kind}_resample_changed_mask.npy', changed)
            local = anysleep_local_support(changed)
            resample_epochs = changed[:len(truth) * 3840].reshape(len(truth), 3840).any(axis=1)
            np.save(out / f'{kind}_local_support_epoch_mask.npy', local)
            which = np.flatnonzero(changed)
            write_json(out / f'{kind}_filter_footprint.json', {'changed_resampled_samples': int(changed.sum()),
                'first_changed_sample_128hz': int(which[0]) if len(which) else None,
                'last_changed_sample_128hz_inclusive': int(which[-1]) if len(which) else None,
                'local_support_epochs': int(local.sum()),
                'support_definition': 'exact Boolean structural propagation for the pinned topology, eval BN, fixed stats/attention; not empirical effects',
                'global_stats_recomputed_in_primary': True})
            np.savez(out / f'{kind}_preprocess_stats.npz', **stats1)
            result, _ = infer(z1, kind)
            if kind == 'noop' and (not np.array_equal(result.argmax(1), base.argmax(1)) or not np.allclose(base, result, rtol=1e-6, atol=1e-7)):
                raise RuntimeError('No-op failed')
            variants = [(kind, result)]
            if diagnostics and kind != 'noop':
                fixed_z, _, _ = preprocess(y, sf, stats0)
                for label, z, alpha in [('fixedstats', fixed_z, None), ('fixedattention', z1, alpha0),
                                        ('fixedboth', fixed_z, alpha0)]:
                    p, _ = infer(z, kind + '_' + label, alpha)
                    variants.append((kind + '_' + label, p))
            for label, result in variants:
                for epoch in range(len(truth)):
                    pairs.append(dict(participant=cfg['participant'], record=cfg['record'],
                        perturbation_id=label, epoch_start=int(epoch * 30 * sf), epoch_end=int((epoch + 1) * 30 * sf),
                        corrupt_start=start, corrupt_end=end, truth=int(truth[epoch]),
                        baseline_pred=int(base[epoch].argmax()), perturbed_pred=int(result[epoch].argmax()),
                        beyond_local_support=bool(not local[epoch]), oracle=('fixed' in label)))
                    pairs[-1]['direct_resample_touched'] = bool(resample_epochs[epoch])
            save_analysis(pairs, out, sf)
    finally:
        probe.close()


def save_analysis(pairs, out, sf):
    import pandas as pd
    from matplotlib import pyplot as plt
    df = pd.DataFrame(pairs)
    df.to_csv(out / 'paired_predictions.csv', index=False)
    # Keep interventions separate; the inherited utility aggregates all IDs together.
    summaries = []
    for name, part in df.groupby('perturbation_id', sort=False):
        folder = out / 'analysis' / name
        folder.mkdir(parents=True, exist_ok=True)
        part.to_csv(folder / 'pairs.csv', index=False)
        group = analyze(folder / 'pairs.csv', folder, sf, guard_seconds=0)
        group['intervention'] = name
        summaries.append(group)
        trans = pd.read_csv(folder / 'epoch_transitions.csv')
        scored = trans.loc[trans.scored & (trans.region != 'corrupted')]
        rows = []
        for region, p in [('all_unmodified', scored), ('beyond_exact_local_support', scored.loc[scored.beyond_local_support])]:
            rows.append({'region': region, 'scored_epochs': len(p), 'new_errors': int(p.new_error.sum()),
                         'corrected_errors': int(p.corrected.sum()), 'prediction_flips': int(p.flipped.sum()),
                         'net_additional_error': float(p.net_error.mean()) if len(p) else None})
        write_json(folder / 'unmodified_summary.json', rows)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / 'distance_summary.csv', index=False)
    order = ['corrupted', '0-30s', '30-60s', '60-120s', '120-300s', '300s+']
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, p in summary.groupby('intervention', sort=False):
        ax.plot(range(len(order)), [float(p.loc[p.region == r, 'net_error'].iloc[0] * 100) if (p.region == r).any() else np.nan for r in order], marker='o', label=name)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel('Net additional error (percentage points)')
    ax.set_xlabel('Gap between epoch edge and perturbed interval edge')
    ax.set_title('One participant / one predeclared position; descriptive only')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'distance_impact.png', dpi=180)
    plt.close(fig)


def run_yasa(cfg, out, raw, x, sf, truth, start, end, pilot):
    import mne
    import pandas as pd
    import yasa
    import psutil
    modeldir = Path(yasa.__file__).parent / 'classifiers'
    assets = [{'path': str(p), 'sha256': digest(p)} for p in modeldir.rglob('*') if p.is_file()]
    write_json(out / 'classifier_assets.json', assets)
    runs = []

    def infer(signal, label):
        t = time.monotonic()
        # Do not feed AnySleep-normalized or prefiltered data to YASA.
        r = mne.io.RawArray(signal, raw.info.copy(), verbose=False)
        sls = yasa.SleepStaging(r, eeg_name=cfg['yasa_eeg'], eog_name=cfg['yasa_eog'])
        hyp = sls.predict()
        if hasattr(hyp, 'hypno'):
            labels = np.asarray(hyp.hypno).astype(str)
            proba = hyp.proba.copy()
        else:
            labels = np.asarray(hyp).astype(str)
            proba = sls.predict_proba()
        label_map = {'WAKE': 0, 'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'R': 4}
        pred = np.asarray([label_map[s.upper()] for s in labels], dtype=np.int16)
        if len(pred) != len(truth):
            raise ValueError('YASA/label epoch mismatch')
        np.save(out / f'{label}_prediction.npy', pred)
        proba.to_csv(out / f'{label}_probabilities.csv', index=False)
        sls.get_features().to_csv(out / f'{label}_features.csv', index=False)
        runs.append({'name': label, 'seconds': time.monotonic() - t, 'rss_after_bytes': psutil.Process().memory_info().rss})
        write_json(out / 'inference_runs.json', runs)
        return pred, proba

    baseline, prob = infer(x, 'baseline')
    baseline_scores(out, truth, baseline)
    repeat, prob2 = infer(x, 'repeat')
    write_json(out / 'repeatability.json', {'prediction_flips': int((baseline != repeat).sum()),
        'max_absolute_probability_delta': float(np.max(np.abs(prob.to_numpy() - prob2.to_numpy())))})
    if not np.array_equal(baseline, repeat) or not np.allclose(prob, prob2, rtol=1e-6, atol=1e-7):
        raise RuntimeError('YASA repeatability failed')
    if pilot:
        pairs = []
        for kind in cfg['kinds']:
            y = perturb(x, cfg['corrupt_channels'], start, end, kind, seed=cfg['noise_seed'],
                        clip=cfg['clip_volts'], noise_sd=cfg['noise_sd_volts'])
            result, _ = infer(y, kind)
            if kind == 'noop' and not np.array_equal(result, baseline):
                raise RuntimeError('YASA no-op failed')
            for epoch in range(len(truth)):
                pairs.append(dict(participant=cfg['participant'], record=cfg['record'], perturbation_id=kind,
                    epoch_start=int(epoch * 30 * sf), epoch_end=int((epoch + 1) * 30 * sf), corrupt_start=start,
                    corrupt_end=end, truth=int(truth[epoch]), baseline_pred=int(baseline[epoch]),
                    perturbed_pred=int(result[epoch]), beyond_local_support=False, oracle=False))
            save_analysis(pairs, out, sf)
    # YASA filtering/feature context differs. Never call an unverified fixed distance 'far'.
    write_json(out / 'support_note.json', {'exact_yasa_support_audited': False,
        'beyond_local_support_column': 'all False until YASA FIR and rolling-feature support audited',
        'known_global_path': 'robust normalization of rolling features; no finite total support',
        'input': 'native volt samples; internal conversion/filtering only',
        'montage_limitation': 'Fpz-Cz differs in polarity/reference from preferred central C4-Fpz; report applicability separately'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--config', type=Path, default=ROOT / 'pilot/config_smoke.json')
    p.add_argument('--model', choices=['anysleep', 'yasa', 'both'], default='both')
    p.add_argument('--phase', choices=['A', 'B'], default='A')
    p.add_argument('--diagnostics', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--run-id', default='smoke-v1')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text(encoding='utf-8'))
    if a.command == 'prepare':
        prepare(cfg)
        return
    out = ROOT / 'results' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    if os.environ.get('RESEARCH5_SOURCE_IDENTITY_JSON'):
        write_json(out / 'pipeline_source_identity.json', json.loads(os.environ['RESEARCH5_SOURCE_IDENTITY_JSON']))
    write_json(out / 'config.json', cfg)
    write_json(out / 'status.json', {'status': 'RUNNING', 'phase': a.phase, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    try:
        environment(out)
        raw, x, sf, truth, start, end = read_record(cfg, out)
        for name in (['anysleep', 'yasa'] if a.model == 'both' else [a.model]):
            folder = out / name
            folder.mkdir()
            print('Starting', name, 'complete EDF; no chunking', flush=True)
            if name == 'anysleep':
                run_anysleep(cfg, folder, x, sf, truth, start, end, a.device, a.phase == 'B', a.diagnostics)
            else:
                run_yasa(cfg, folder, raw, x, sf, truth, start, end, a.phase == 'B')
        write_json(out / 'status.json', {'status': 'COMPLETE', 'phase': a.phase, 'inference_only': True,
                   'participants': 1, 'interpretation': 'engineering/development smoke; no population claim'})
        print('PILOT_RUN_COMPLETE', flush=True)
    except BaseException as exc:
        write_json(out / 'status.json', {'status': 'FAILED', 'error_class': type(exc).__name__,
                   'message': str(exc), 'traceback': traceback.format_exc(),
                   'no_silent_chunk_fallback': True})
        raise


if __name__ == '__main__':
    main()
