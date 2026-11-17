"""Independent CPU runner: engineering gate -> paired pilot -> oracle diagnostics.

No local browser, OAuth token, Colab session or running personal computer is
required. This file never trains or expands participants. A scientific go/no-go
decision remains a later, explicit evidence review.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def run_key():
    identity = os.environ.get('GITHUB_RUN_ID', time.strftime('%Y%m%dT%H%M%S', time.gmtime()))
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    return f'gha-{identity}-{attempt}'


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name == 'posix':
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=10)


def run_command(args, label, folder, deadline):
    """Stream output while sampling process-tree resident memory once per second."""
    import psutil
    logfile = folder / f'{label}.log'
    print('Starting:', ' '.join(args), flush=True)
    start = time.monotonic()
    proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=(os.name == 'posix'))
    peak_sampled = 0
    timed_out = False

    def pump():
        with logfile.open('w', encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        with (folder / 'resource_samples.jsonl').open('a', encoding='utf-8') as samples:
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    timed_out = True
                    stop_process(proc)
                    break
                try:
                    process = psutil.Process(proc.pid)
                    tree = [process] + process.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                    peak_sampled = max(peak_sampled, rss)
                    samples.write(json.dumps({'label': label, 'elapsed_seconds': time.monotonic() - start,
                        'process_tree_rss_bytes': rss, 'available_ram_bytes': psutil.virtual_memory().available}) + '\n')
                    samples.flush()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        stop_process(proc)
        thread.join(timeout=10)
        row = {'command': args, 'exit_code': proc.returncode, 'seconds': time.monotonic() - start,
               'timeout': timed_out, 'peak_sampled_process_tree_rss_bytes': peak_sampled,
               'sample_interval_seconds': 1, 'log': str(logfile.relative_to(ROOT))}
        write_json(folder / f'{label}_command.json', row)
    if timed_out:
        raise TimeoutError('Internal run budget exhausted; stopped before the outer Actions timeout to preserve artifacts')
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, args)
    return row


def stage_a_gate(folder):
    status = read_json(folder / 'status.json', {})
    if status.get('status') != 'COMPLETE':
        raise RuntimeError('Stage A did not complete')
    evidence = {}
    for model in ['anysleep', 'yasa']:
        scores = read_json(folder / model / 'baseline_scores.json', {})
        repeat = read_json(folder / model / 'repeatability.json', {})
        if scores.get('scored_epochs', 0) <= 0 or repeat.get('prediction_flips') != 0:
            raise RuntimeError(f'Stage-A gate failed for {model}: missing scored epochs or repeat inconsistency')
        if model == 'anysleep' and not repeat.get('allclose_rtol_1e-6_atol_1e-7', False):
            raise RuntimeError('AnySleep repeatability tolerance failed')
        evidence[model] = {'scores': scores, 'repeatability': repeat}
    return evidence


def summary(key, append_to_job=True):
    folder = ROOT / 'results' / key
    folder.mkdir(parents=True, exist_ok=True)
    state = read_json(folder / 'controller_status.json', {'status': 'NOT_STARTED'})
    lines = ['# First offline EEG pilot', '', f"Controller status: **{state.get('status')}**", '',
             f"Scientific revision: `{os.environ.get('GITHUB_SHA', 'local')}`", '',
             'One predeclared development participant; complete EDF inference, no chunking, no training.', '',
             'Stage B runs only after stage A completes with repeated predictions unchanged. Oracle diagnostics freeze original statistics and/or all 13 original channel-weight blocks; these are not deployable methods.', '']
    if os.environ.get('SETUP_OUTCOME'):
        lines.append(f"Dependency setup: {os.environ['SETUP_OUTCOME']}; pilot step: {os.environ.get('PILOT_OUTCOME', 'unknown')}.")
    for suffix in ['a', 'b']:
        phase = ROOT / 'results' / f'{key}-{suffix}'
        phase_state = read_json(phase / 'status.json', {})
        lines.extend(['', f"## Stage {suffix.upper()}: {phase_state.get('status', 'NOT_RUN')}", ''])
        if phase_state.get('message'):
            lines.append(str(phase_state['message']))
        for model in ['anysleep', 'yasa']:
            score = read_json(phase / model / 'baseline_scores.json')
            if score:
                lines.append(f"- {model}: {score['scored_epochs']} scored epochs; baseline accuracy {score['accuracy']:.4f}; descriptive development result.")
    if state.get('message'):
        lines.extend(['', f"Failure/blocker: {state['message']}"])
    lines.extend(['', 'Completed and partial predictions, statuses, logs, resolved dependency versions and distance plots are in the run artifact. Download before the 90-day retention expires. Artifacts are not a permanent repository result archive.', '',
                  'No automatic participant expansion or new-method training follows this run. Net harm, simple baselines, overlap and second-source checks still determine whether the topic should continue.'])
    content = '\n'.join(lines) + '\n'
    (folder / 'CLOUD_RUN_SUMMARY.md').write_text(content, encoding='utf-8')
    if append_to_job and os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write(content)
    print(content, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout-minutes', type=float, default=80)
    parser.add_argument('--summary-only', action='store_true')
    a = parser.parse_args()
    key = run_key()
    if a.summary_only:
        summary(key)
        return
    import psutil
    folder = ROOT / 'results' / key
    folder.mkdir(parents=True, exist_ok=False)
    state = {'status': 'RUNNING', 'stage': 'preflight', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'revision': os.environ.get('GITHUB_SHA'), 'repository': os.environ.get('GITHUB_REPOSITORY'),
        'runner': os.environ.get('RUNNER_NAME'), 'platform': platform.platform(),
        'cpus_observed': os.cpu_count(), 'ram_total_bytes': psutil.virtual_memory().total,
        'ram_available_bytes': psutil.virtual_memory().available, 'training': False,
        'complete_edf': True, 'local_computer_required': False, 'commands': []}
    write_json(folder / 'controller_status.json', state)
    deadline = time.monotonic() + a.timeout_minutes * 60
    try:
        if a.timeout_minutes <= 0:
            raise ValueError('Run budget must be positive')
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            event = read_json(Path(os.environ['GITHUB_EVENT_PATH']), {})
            if event.get('repository', {}).get('private') is not False:
                raise RuntimeError('This unattended workflow is restricted to standard runners for public repositories')
        if psutil.virtual_memory().available < 10 * 1024**3:
            raise RuntimeError('Less than 10 GiB available RAM for full-EDF inference; refusing a changed/chunked protocol')
        commands = [('prepare', [sys.executable, 'pilot/run_pilot.py', 'prepare']),
                    ('stage-a', [sys.executable, 'pilot/run_pilot.py', 'run', '--model', 'both', '--phase', 'A', '--device', 'cpu', '--run-id', key + '-a']),
                    ('stage-b', [sys.executable, 'pilot/run_pilot.py', 'run', '--model', 'both', '--phase', 'B', '--diagnostics', '--device', 'cpu', '--run-id', key + '-b'])]
        for label, command in commands:
            state['stage'] = label
            write_json(folder / 'controller_status.json', state)
            state['commands'].append(run_command(command, label, folder, deadline))
            if label == 'stage-a':
                state['stage_a_gate'] = stage_a_gate(ROOT / 'results' / (key + '-a'))
                write_json(folder / 'stage_a_gate.json', state['stage_a_gate'])
        state['status'] = 'COMPLETE'
        state['interpretation'] = 'engineering pilot complete; no automatic scientific go decision'
    except BaseException as exc:
        state.update(status='FAILED', message=str(exc), error_class=type(exc).__name__, traceback=traceback.format_exc())
        raise
    finally:
        state['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(folder / 'controller_status.json', state)
        summary(key, append_to_job=False)


if __name__ == '__main__':
    main()
