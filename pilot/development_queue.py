"""Frozen ten-participant development queue; Colab compute, Drive persistence.

`plan` reads only the official PhysioNet directory (no EDF/model downloads).
`run` requires compatible completed single-night A/B stages, then executes the
existing unchunked pilot for both pipelines, three positions, no oracle arms.
No training, keepalive, participant replacement or outcome-dependent positions.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
from html.parser import HTMLParser
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
QUEUE_ID = 'development-queue-v1'
KINDS = ['noop', 'flat', 'clip', 'noise']
COMPAT_KEYS = ['anysleep_commit', 'checkpoint', 'checkpoint_git_blob', 'channels',
               'yasa_eeg', 'yasa_eog', 'corrupt_channels', 'duration_seconds',
               'clip_volts', 'noise_sd_volts', 'noise_seed', 'kinds', 'label_map']
PACKAGES = ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'mne', 'yasa',
            'lightgbm', 'antropy', 'psutil']


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def source_identity():
    """Stdlib-only identity also used by the Drive driver for A/B receipts."""
    paths = ['pilot/run_pilot.py', 'pilot/paired_tools.py', 'pilot/support.py',
             'manifests/anysleep_sources.json', 'requirements-cloud.txt']
    provenance = read_json(ROOT / 'manifests/anysleep_sources.json')
    paths += ['vendor/anysleep/' + row['path'] for row in provenance['files']]
    return {'algorithm': 'sha256', 'anysleep_commit': provenance['commit'],
            'files': {p: sha256(ROOT / p) for p in sorted(paths)}}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for key, value in attrs:
                if key == 'href' and value:
                    self.names.add(unquote(Path(urlparse(value).path).name))


def make_plan(path):
    with urlopen(DIRECTORY, timeout=60) as response:
        payload = response.read()
        if urlparse(response.url).hostname != 'physionet.org':
            raise ValueError('Official directory redirected to an unapproved host')
    links = Links()
    links.feed(payload.decode('utf-8'))
    base = read_json(ROOT / 'pilot/config_smoke.json')
    rows = []
    for number in range(400, 410):
        participant = f'SC{number}'
        # Match an observed first-night filename; do not invent the E/F suffix.
        psg = sorted(n for n in links.names if re.fullmatch(participant + r'1[A-Z0-9]{2}-PSG\.edf', n))
        hyp = sorted(n for n in links.names if re.fullmatch(participant + r'1[A-Z0-9]{2}-Hypnogram\.edf', n))
        row = {'participant': participant, 'night_rule': 'first night, literal directory match',
               'psg_candidates': psg, 'hypnogram_candidates': hyp}
        if len(psg) != 1 or len(hyp) != 1:
            row.update(availability='UNAVAILABLE_OR_AMBIGUOUS',
                       exclusion_reason='Need exactly one literal first-night PSG and hypnogram; no replacement')
        else:
            record = psg[0].removesuffix('-PSG.edf')
            mapped_id = record[3:6]
            # These roles were read from AnySleep 88e13e3/usleep_split.yaml.
            roles = {'00': 'test', '01': 'train', '02': 'test', '03': 'train',
                     '04': 'train', '05': 'train', '06': 'train', '07': 'train',
                     '08': 'valid', '09': 'valid'}
            row.update(availability='AVAILABLE', record=record, psg=psg[0], hypnogram=hyp[0],
                       psg_url=DIRECTORY + psg[0], hypnogram_url=DIRECTORY + hyp[0],
                       anysleep_record_id=mapped_id, anysleep_subject_id=mapped_id[:2],
                       anysleep_published_split=roles[mapped_id[:2]])
        rows.append(row)
    plan = {'schema': 'research5-development-queue-v1', 'queue_id': QUEUE_ID,
            'frozen_utc': now(), 'status': 'PLANNED_NOT_EXECUTED',
            'directory_evidence': {'url': DIRECTORY, 'fetched_utc': now(),
                                   'html_sha256': hashlib.sha256(payload).hexdigest(),
                                   'edf_or_checkpoint_downloaded_by_plan': False},
            'scope': 'Development on exposed/public data; no independent external-validation claim',
            'protocol_deviation': 'First bounded expansion is SC400-SC409 (10), no ID substitution; original plan allowed 20 candidates.',
            'base_config': base, 'source_identity': source_identity(), 'participants': rows,
            'positions': [0.25, 0.5, 0.75], 'duration_seconds': 30,
            'models': ['anysleep', 'yasa'], 'phase': 'B', 'oracle_diagnostics': False,
            'budget': {'cumulative_wall_seconds': 14400, 'keepalive': False,
                       'timeout_action': 'terminate current subprocess and preserve partial outputs'},
            'analysis': {'primary_region': '30 <= epoch-edge gap seconds < 300, scored and raw unmodified',
                         'other_region': 'all scored raw-unmodified epochs',
                         'kinds_separate': KINDS, 'participant_aggregation': 'equal mean of all three completed positions',
                         'group_aggregation': 'equal participants, complete 3-position participants only',
                         'bootstrap_seed': 20261117, 'bootstrap_replicates': 20000,
                         'practical_net_error_fraction': 0.01,
                         'minimum_complete_participants': 10,
                         'stage': 'development, descriptive intervals, no multiplicity-adjusted confirmation',
                         'new_method_training_allowed': False}}
    path = Path(path)
    if path.exists() and read_json(path) != plan:
        raise FileExistsError('Manifest already exists. Freeze a new named version instead of overwriting it.')
    write_json(path, plan)
    print(json.dumps({'status': 'PLAN_FROZEN', 'manifest': str(path),
                      'available': sum(r['availability'] == 'AVAILABLE' for r in rows),
                      'planned_positions': len(rows) * 3, 'no_data_downloaded': True}, indent=2))


def validate_plan(plan):
    if plan['schema'] != 'research5-development-queue-v1' or plan['queue_id'] != QUEUE_ID:
        raise ValueError('Unknown queue identity')
    if [r['participant'] for r in plan['participants']] != [f'SC{i}' for i in range(400, 410)]:
        raise ValueError('Participant list differs from frozen ten-person scope')
    if plan['positions'] != [0.25, 0.5, 0.75] or plan['duration_seconds'] != 30:
        raise ValueError('Position/duration rule changed')
    if plan['base_config']['kinds'] != KINDS or plan['oracle_diagnostics']:
        raise ValueError('Controls must remain and oracle arms are excluded')
    if plan['source_identity'] != source_identity():
        raise ValueError('Pilot/vendor/dependency source identity changed; freeze a new compatible manifest')


def assert_drive_layout():
    if not sys.platform.startswith('linux') or not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Queue execution requires Colab Linux with user-mounted MyDrive')
    drive = Path('/content/drive/MyDrive').resolve()
    for folder in ['data', 'models', 'results', 'logs', 'manifests']:
        resolved = (ROOT / folder).resolve()
        if not resolved.is_relative_to(drive):
            raise RuntimeError(f'{folder} is not persisted in mounted Drive: {resolved}')


def config_for(plan, record, fraction):
    cfg = dict(plan['base_config'])
    cfg.update({key: record[key] for key in ['participant', 'record', 'psg', 'hypnogram']})
    cfg.update(protocol_version=QUEUE_ID, duration_seconds=30, position_fraction=fraction,
               position_rule='floor(q * valid-scored-epoch count), sorted by time, q in .25/.5/.75; never predictions',
               development_only=True, anysleep_published_split=record['anysleep_published_split'])
    return cfg


def check_run(folder, cfg, phase, require_receipt=True):
    """Re-read predictions: failed repeat/noop checks cannot be waived by status."""
    import numpy as np
    import pandas as pd
    status = read_json(folder / 'status.json')
    if status.get('status') != 'COMPLETE' or status.get('phase') != phase:
        raise ValueError(f'Prerequisite/output not complete for phase {phase}: {folder}')
    actual = read_json(folder / 'config.json')
    for key in COMPAT_KEYS + ['participant', 'record']:
        if actual[key] != cfg[key]:
            raise ValueError(f'Incompatible run configuration {key}: {folder}')
    if require_receipt and read_json(folder / 'pipeline_source_identity.json') != source_identity():
        raise ValueError('Missing or changed pipeline source receipt')
    for model in ['anysleep', 'yasa']:
        out = folder / model
        repeat = read_json(out / 'repeatability.json')
        if repeat['prediction_flips'] != 0:
            raise AssertionError('Repeated inference changed predictions')
        if model == 'anysleep':
            base = np.load(out / 'baseline_logits.npy', allow_pickle=False)
            again = np.load(out / 'repeat_logits.npy', allow_pickle=False)
            np.testing.assert_allclose(base, again, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(base.argmax(1), again.argmax(1))
            if phase == 'B':
                no_op = np.load(out / 'noop_logits.npy', allow_pickle=False)
                np.testing.assert_allclose(base, no_op, rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(base.argmax(1), no_op.argmax(1))
        else:
            base = np.load(out / 'baseline_prediction.npy', allow_pickle=False)
            again = np.load(out / 'repeat_prediction.npy', allow_pickle=False)
            np.testing.assert_array_equal(base, again)
            probs = pd.read_csv(out / 'baseline_probabilities.csv')
            np.testing.assert_allclose(probs, pd.read_csv(out / 'repeat_probabilities.csv'), rtol=1e-6, atol=1e-7)
            if phase == 'B':
                np.testing.assert_array_equal(base, np.load(out / 'noop_prediction.npy', allow_pickle=False))
                np.testing.assert_allclose(probs, pd.read_csv(out / 'noop_probabilities.csv'), rtol=1e-6, atol=1e-7)
        if phase == 'B':
            paired = pd.read_csv(out / 'paired_predictions.csv')
            if not set(KINDS).issubset(set(paired.perturbation_id)):
                raise ValueError('Missing mandatory perturbation/control outputs')
            noop = paired.loc[paired.perturbation_id == 'noop']
            if (noop.baseline_pred != noop.perturbed_pred).any():
                raise AssertionError('No-op paired predictions changed')


def prerequisite_gate(plan, a_id, b_id, device):
    cfg = plan['base_config']
    a, b = ROOT / 'results' / a_id, ROOT / 'results' / b_id
    check_run(a, cfg, 'A')
    check_run(b, cfg, 'B')
    ea, eb = read_json(a / 'environment.json'), read_json(b / 'environment.json')
    versions = {name: md.version(name) for name in PACKAGES}
    for env in [ea, eb]:
        if env['versions'] != versions or env['python'] != sys.version:
            raise ValueError('A/B and queue Python/package environments differ; repeat compatible A/B first')
    if ea.get('cuda') != eb.get('cuda'):
        raise ValueError('A and B CUDA build differs')
    import torch
    gpu_identity = [{'name': torch.cuda.get_device_name(i),
                     'total_bytes': torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())]
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable; no silent CPU fallback')
    for folder in [a, b]:
        runs = read_json(folder / 'anysleep/inference_runs.json')
        used_cuda = runs[0]['cuda_peak_allocated_bytes'] is not None
        if used_cuda != device.startswith('cuda'):
            raise ValueError('A/B inference device type differs from requested queue device')
    for env in [ea, eb]:
        observed = [{'name': g['name'], 'total_bytes': g['total_bytes']} for g in env['gpu']]
        if observed != gpu_identity:
            raise ValueError('A/B and current GPU identities differ; revalidate compatible A/B')
    return {'passed_utc': now(), 'stage_a': a_id, 'stage_b': b_id,
            'source_identity': source_identity(), 'versions': versions,
            'device': device, 'gpu_identity': gpu_identity,
            'no_op_and_repeated_predictions_reloaded_and_asserted': True}


def stop_process(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=8)


def command(args, log, remaining, heartbeat):
    env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg',
               RESEARCH5_SOURCE_IDENTITY_JSON=json.dumps(source_identity()))
    with log.open('w', encoding='utf-8') as out:
        proc = subprocess.Popen([str(x) for x in args], cwd=ROOT, env=env,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while proc.poll() is None:
                left = remaining - (time.monotonic() - started)
                if left <= 0:
                    raise subprocess.TimeoutExpired(args, remaining)
                try:
                    proc.wait(timeout=min(10, left))
                except subprocess.TimeoutExpired:
                    heartbeat()
        except BaseException:
            stop_process(proc)
            raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args)


def file_receipt(folder):
    paths = ['status.json', 'config.json', 'environment.json', 'pipeline_source_identity.json', 'truth.npy']
    for model in ['anysleep', 'yasa']:
        paths.extend(f'{model}/{name}' for name in ['paired_predictions.csv', 'repeatability.json'])
        if model == 'anysleep':
            paths += [f'{model}/{name}_logits.npy' for name in ['baseline', 'repeat', 'noop', 'flat', 'clip', 'noise']]
        else:
            paths += [f'{model}/{name}_{suffix}' for name in ['baseline', 'repeat', 'noop', 'flat', 'clip', 'noise']
                      for suffix in ['prediction.npy', 'probabilities.csv']]
    return {p: sha256(folder / p) for p in paths}


def aggregate(plan, state, out):
    import numpy as np
    import pandas as pd
    positions = []
    for job in state['jobs'].values():
        if job['status'] != 'COMPLETE':
            continue
        folder = ROOT / 'results' / job['run_id']
        sf = read_json(folder / 'record.json')['sfreq_hz']
        for model in ['anysleep', 'yasa']:
            data = pd.read_csv(folder / model / 'paired_predictions.csv')
            if data.duplicated(['perturbation_id', 'epoch_start']).any():
                raise ValueError('Duplicate paired epochs')
            distance = np.maximum(np.maximum(data.corrupt_start - data.epoch_end,
                                              data.epoch_start - data.corrupt_end), 0) / sf
            untouched = (data.epoch_end <= data.corrupt_start) | (data.epoch_start >= data.corrupt_end)
            scored = data.truth.isin([0, 1, 2, 3, 4])
            for kind in KINDS:
                for region, mask in [('all_untouched', untouched),
                                     ('near_30_300s', untouched & (distance >= 30) & (distance < 300))]:
                    p = data.loc[scored & mask & (data.perturbation_id == kind)]
                    if p.empty:
                        raise ValueError('No valid epochs in a frozen analysis region')
                    new = (p.baseline_pred == p.truth) & (p.perturbed_pred != p.truth)
                    corrected = (p.baseline_pred != p.truth) & (p.perturbed_pred == p.truth)
                    positions.append(dict(participant=job['participant'], position=job['position'],
                                          model=model, kind=kind, region=region, epochs=len(p),
                                          new_errors=int(new.sum()), corrected_errors=int(corrected.sum()),
                                          new_error_rate=float(new.mean()), corrected_rate=float(corrected.mean()),
                                          net_error=float(new.mean() - corrected.mean()),
                                          flip_rate=float((p.baseline_pred != p.perturbed_pred).mean())))
    out.mkdir(parents=True, exist_ok=True)
    if not positions:
        write_json(out / 'decision.json', {'decision': 'UNCERTAIN', 'reason': 'No complete paired jobs',
                                          'new_method_training_allowed': False})
        return
    frame = pd.DataFrame(positions)
    frame.to_csv(out / 'position_metrics.csv', index=False)
    keys = ['participant', 'model', 'kind', 'region']
    participants = frame.groupby(keys, as_index=False).agg(
        positions=('position', 'nunique'), total_epochs_descriptive=('epochs', 'sum'),
        new_error_rate=('new_error_rate', 'mean'), corrected_rate=('corrected_rate', 'mean'),
        net_error=('net_error', 'mean'), flip_rate=('flip_rate', 'mean'))
    participants['complete_three_positions'] = participants.positions == len(plan['positions'])
    participants.to_csv(out / 'participant_metrics.csv', index=False)
    rows = []
    analysis = plan['analysis']
    for (model, kind, region), data in participants.loc[participants.complete_three_positions].groupby(['model', 'kind', 'region']):
        data = data.sort_values('participant')
        values = data.net_error.to_numpy()
        n = len(values)
        low = high = None
        if n >= 2:
            rng = np.random.default_rng(analysis['bootstrap_seed'])
            boots = values[rng.integers(n, size=(analysis['bootstrap_replicates'], n))].mean(axis=1)
            low, high = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
        rows.append({'model': model, 'kind': kind, 'region': region, 'n_participants': n,
                     'net_error': float(values.mean()), 'ci95_low': low, 'ci95_high': high,
                     'new_error_rate': float(data.new_error_rate.mean()),
                     'corrected_rate': float(data.corrected_rate.mean()),
                     'positive_participants': int((values > 0).sum()),
                     'interpretation': 'development participant bootstrap; positions and epochs are not n'})
    group = pd.DataFrame(rows)
    group.to_csv(out / 'group_metrics.csv', index=False)
    primary = [r for r in rows if r['region'] == 'near_30_300s' and r['kind'] != 'noop']
    target = analysis['minimum_complete_participants']
    threshold = analysis['practical_net_error_fraction']
    complete = len(primary) == 6 and all(r['n_participants'] >= target for r in primary)
    if not complete:
        decision, reason = 'UNCERTAIN', 'Fewer than 10 complete three-position participants; retain failures and do not replace IDs'
    elif any(r['ci95_low'] is not None and r['ci95_low'] >= threshold for r in primary):
        decision, reason = 'CONTINUE_MECHANISM_AND_SIMPLE_BASELINES', 'At least one predeclared pipeline/perturbation shows practical near-region harm; cross-source and simple-baseline checks remain'
    elif all(r['ci95_high'] is not None and r['ci95_high'] < threshold for r in primary):
        decision, reason = 'STOP_CURRENT_NEW_ARCHITECTURE_ROUTE', 'All primary development intervals lie below the predeclared 1pp practical size; retain negative results'
    else:
        decision, reason = 'UNCERTAIN', 'Intervals do not distinguish negligible from practical harm consistently'
    write_json(out / 'decision.json', {'decision': decision, 'reason': reason, 'threshold_fraction': threshold,
                                      'primary_comparisons': primary, 'new_method_training_allowed': False,
                                      'single_night_zero_is_not_a_population_stop_rule': True,
                                      'simple_baselines_evaluated': False, 'second_source_verified': False,
                                      'multiplicity': 'exploratory development; no confirmatory familywise claim'})


def run_queue(args):
    plan = read_json(args.manifest)
    validate_plan(plan)
    assert_drive_layout()
    gate = prerequisite_gate(plan, args.stage_a_run_id, args.stage_b_run_id, args.device)
    out = ROOT / 'results' / plan['queue_id']
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'state.json'
    identity = {'manifest_sha256': sha256(args.manifest), 'source_identity': source_identity(),
                'queue_source_sha256': sha256(__file__), 'environment_versions': gate['versions'],
                'device': gate['device'], 'gpu_identity': gate['gpu_identity']}
    state = read_json(state_path) if state_path.exists() else {
        'schema': 'research5-development-state-v1', 'identity': identity, 'jobs': {},
        'created_utc': now(), 'elapsed_seconds': 0, 'status': 'READY'}
    if state['identity'] != identity:
        raise ValueError('Resume identity mismatch: manifest/source/environment changed; preserve old queue')
    for path, expected in state.get('input_sha256', {}).items():
        if sha256(ROOT / path) != expected:
            raise ValueError('Previously frozen data/model changed: ' + path)
    # A local advisory lock prevents overlapping controllers in this runtime.
    import fcntl
    lock_name = '/tmp/research5-' + canonical(str(out.resolve()))[:20] + '.lock'
    lock = open(lock_name, 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = min(14400, plan['budget']['cumulative_wall_seconds'])
    invocation_limit = min(14400, args.max_hours * 3600)
    if invocation_limit <= 0:
        raise ValueError('max-hours must be positive and at most the remaining fixed budget')
    begun = time.monotonic()
    if state['status'] == 'RUNNING':
        # A lost runtime can leave <=10 seconds after its last saved heartbeat.
        # Charge that full interval conservatively; do not count powered-off time.
        state['elapsed_seconds'] += 10
    previous_elapsed = state['elapsed_seconds']

    def persist():
        state['elapsed_seconds'] = previous_elapsed + time.monotonic() - begun
        state['updated_utc'] = now()
        write_json(state_path, state)

    def remaining():
        elapsed = time.monotonic() - begun
        return min(budget - previous_elapsed - elapsed, invocation_limit - elapsed)

    write_json(out / 'manifest.json', plan)
    write_json(out / 'prerequisite_gate.json', gate)
    state.update(status='RUNNING', host=socket.gethostname(), pid=os.getpid())
    persist()
    try:
        for record in plan['participants']:
            for fraction in plan['positions']:
                key = f"{record['participant']}-p{int(fraction * 100):02d}"
                cfg = config_for(plan, record, fraction) if record['availability'] == 'AVAILABLE' else None
                old = state['jobs'].get(key)
                if old and old['status'] == 'COMPLETE':
                    folder = ROOT / 'results' / old['run_id']
                    if file_receipt(folder) != old['file_receipt']:
                        raise ValueError('Completed results changed during resume: ' + key)
                    check_run(folder, cfg, 'B')
                    continue
                if old and old['status'] in ['FAILED', 'SOURCE_UNAVAILABLE']:
                    continue  # Preserve failures; no automatic replacement or silent repeated trials.
                if remaining() <= 0:
                    state['status'] = 'BUDGET_REACHED'
                    return
                if cfg is None:
                    state['jobs'][key] = {'participant': record['participant'], 'position': fraction,
                                          'status': 'SOURCE_UNAVAILABLE', 'reason': record['exclusion_reason']}
                    persist()
                    continue
                attempts = (old or {}).get('attempts', [])
                attempt = len(attempts) + 1
                run_id = f'{QUEUE_ID}-{key}-attempt{attempt}'
                while (ROOT / 'results' / run_id).exists():
                    attempt += 1
                    run_id = f'{QUEUE_ID}-{key}-attempt{attempt}'
                job = {'participant': record['participant'], 'position': fraction, 'status': 'RUNNING',
                       'run_id': run_id, 'started_utc': now(), 'attempts': attempts + [run_id]}
                state['jobs'][key] = job
                config = out / 'configs' / (key + '.json')
                write_json(config, cfg)
                job['config_sha256'] = sha256(config)
                persist()
                try:
                    for mode in ['prepare', 'run']:
                        cmd = [sys.executable, ROOT / 'pilot/run_pilot.py', mode, '--config', config]
                        if mode == 'run':
                            cmd += ['--model', 'both', '--phase', 'B', '--device', args.device, '--run-id', run_id]
                        command(cmd, out / f'{key}-attempt{attempt}-{mode}.log', remaining(), persist)
                        if source_identity() != plan['source_identity']:
                            raise RuntimeError('Pipeline source changed during queue execution')
                        if mode == 'prepare':
                            downloads = read_json(ROOT / 'manifests/downloads.json')
                            inputs = state.setdefault('input_sha256', {})
                            for item in downloads:
                                path, observed = item['path'], item['sha256']
                                if path in inputs and inputs[path] != observed:
                                    raise RuntimeError('Data/model changed since its first queue use: ' + path)
                                inputs[path] = observed
                            job['download_receipt'] = downloads
                            write_json(out / 'sources' / (key + '.json'), downloads)
                        persist()
                    folder = ROOT / 'results' / run_id
                    check_run(folder, cfg, 'B')
                    job.update(status='COMPLETE', completed_utc=now(), file_receipt=file_receipt(folder))
                except subprocess.TimeoutExpired:
                    job.update(status='INTERRUPTED_BUDGET', error='Fixed wall-clock budget reached; partial output preserved')
                    state['status'] = 'BUDGET_REACHED'
                    persist()
                    return
                except Exception as exc:
                    job.update(status='FAILED', error_class=type(exc).__name__, error=str(exc),
                               traceback=traceback.format_exc(), failed_utc=now())
                persist()
                aggregate(plan, state, out / 'aggregate')
        state['status'] = 'COMPLETE_WITH_FAILURES' if any(j['status'] != 'COMPLETE' for j in state['jobs'].values()) else 'COMPLETE'
    finally:
        if state['status'] == 'RUNNING':
            state['status'] = 'INTERRUPTED'
        persist()
        aggregate(plan, state, out / 'aggregate')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(json.dumps({'status': state['status'], 'elapsed_seconds': state['elapsed_seconds'],
                          'state': str(state_path), 'keepalive': False, 'new_training': False}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    plan = subs.add_parser('plan')
    plan.add_argument('--manifest', type=Path, default=ROOT / 'pilot/development_manifest.json')
    run = subs.add_parser('run')
    run.add_argument('--manifest', type=Path, default=ROOT / 'pilot/development_manifest.json')
    run.add_argument('--stage-a-run-id', required=True)
    run.add_argument('--stage-b-run-id', required=True)
    run.add_argument('--device', default='cuda')
    run.add_argument('--max-hours', type=float, default=4)
    args = parser.parse_args()
    if args.command == 'plan':
        make_plan(args.manifest)
    else:
        run_queue(args)


if __name__ == '__main__':
    main()
