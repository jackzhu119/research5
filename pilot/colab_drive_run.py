"""Run the fixed pilot in Colab with data, models, results and logs on Drive.

The notebook kernel retains its packages; a separate ephemeral virtual environment
inherits CPU/GPU PyTorch and installs the scientific dependencies locally. This
does not keep Colab alive or bypass runtime limits. Reconnect to a normal runtime
and rerun to resume only completed, version-checked stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_files(source):
    required = ['pilot', 'vendor', 'requirements-cloud.txt', 'manifests/anysleep_sources.json']
    for name in required:
        if not (source / name).exists():
            raise FileNotFoundError(f'Incomplete uploaded code bundle: {name}')
    files = []
    for directory in ['pilot', 'vendor', 'manifests']:
        files.extend(p for p in (source / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc', '.ipynb']
                     and p.name != 'downloads.json' and not p.name.startswith('physionet-'))
    files.extend(source.glob('requirements-*.txt'))
    return sorted(set(files))


def stop_child(proc):
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


class DriveRun:
    def __init__(self, project, run_id, logdir, state):
        self.project, self.run_id, self.logdir, self.state = project, run_id, logdir, state
        self.status_file = project / 'status' / run_id / 'controller_status.json'
        self.heartbeat_file = project / 'status' / run_id / 'heartbeat.json'

    def heartbeat(self):
        now = time.time()
        self.state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))
        write_json(self.status_file, self.state)
        write_json(self.heartbeat_file, {'unix_time': now, 'utc': self.state['updated_utc'],
            'status': self.state['status'], 'stage': self.state['stage'], 'host': socket.gethostname(),
            'pid': os.getpid(), 'note': 'Progress recording only; no Colab keepalive'})

    def command(self, command, name, cwd, extra_env=None):
        self.state['stage'] = name
        self.heartbeat()
        logfile = self.logdir / f'{name}.log'
        print('Starting:', ' '.join(map(str, command)), flush=True)
        start = time.monotonic()
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
        if extra_env:
            env.update(extra_env)
        env.setdefault('OMP_NUM_THREADS', str(min(4, os.cpu_count() or 1)))
        env.setdefault('MKL_NUM_THREADS', env['OMP_NUM_THREADS'])
        proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=(os.name == 'posix'))

        def pump():
            with logfile.open('w', encoding='utf-8') as log:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        try:
            while proc.poll() is None:
                self.heartbeat()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_child(proc)
            thread.join(timeout=10)
            write_json(self.logdir / f'{name}_command.json', {'command': list(map(str, command)),
                'exit_code': proc.returncode, 'seconds': time.monotonic() - start, 'log': str(logfile)})
            self.heartbeat()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)


def prepare_workspace(source, project, files, code_id):
    workspace = Path('/content/research5-exec') / code_id
    snapshot = project / 'code_snapshots' / code_id
    for file in files:
        relative = file.relative_to(source)
        for destination in [workspace / relative, snapshot / relative]:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) == sha256(file):
                continue
            shutil.copyfile(file, destination)
    manifest_dir = project / 'manifests' / code_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for file in (workspace / 'manifests').rglob('*'):
        if file.is_file():
            dest = manifest_dir / file.relative_to(workspace / 'manifests')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha256(dest) != sha256(file):
                raise RuntimeError(f'Persisted source manifest changed: {dest}')
            if not dest.exists():
                shutil.copyfile(file, dest)
    # Links live in /content. Drive itself does not need to support symlinks.
    for name, target in [('data', project / 'data'), ('models', project / 'models'),
                         ('results', project / 'results'), ('logs', project / 'logs'), ('manifests', manifest_dir)]:
        target.mkdir(parents=True, exist_ok=True)
        link = workspace / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f'Unexpected persistent-directory link: {link}')
        elif link.exists():
            if name != 'manifests':
                raise RuntimeError(f'Unexpected local persistence directory: {link}')
            shutil.rmtree(link)  # exact generated workspace manifests already copied above
            link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    return workspace


def verify_downloads(workspace):
    manifest = read_json(workspace / 'manifests/downloads.json', [])
    for row in manifest:
        path = workspace / row['path']
        if not path.is_file() or sha256(path) != row['sha256']:
            raise RuntimeError(f'Data/model changed since recorded download: {path}')


def verify_completed(folder):
    if read_json(folder / 'status.json', {}).get('status') != 'COMPLETE':
        return False
    manifest = read_json(folder / 'completion_manifest.json')
    if not manifest:
        return False
    for relative, checksum in manifest.items():
        path = folder / relative
        if not path.is_file() or sha256(path) != checksum:
            raise RuntimeError(f'Completed stage output changed: {path}')
    return True


def mark_completed(folder):
    if read_json(folder / 'status.json', {}).get('status') != 'COMPLETE':
        raise RuntimeError(f'Stage did not complete: {folder}')
    hashes = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*')
              if p.is_file() and p.name != 'completion_manifest.json'}
    write_json(folder / 'completion_manifest.json', hashes)


ENV_PROBE = '''
import importlib.metadata as md, json, pathlib, torch
from packaging.requirements import Requirement
from packaging.version import Version
root = pathlib.Path.cwd()
seeds = [Requirement(x.strip()).name for x in (root / "requirements-cloud.txt").read_text().splitlines()
         if x.strip() and not x.lstrip().startswith("#")] + ["torch"]
pending, seen, failures = list(seeds), set(), []
while pending:
    name = pending.pop()
    canonical = name.lower().replace("_", "-")
    if canonical in seen: continue
    seen.add(canonical)
    for value in md.requires(name) or []:
        req = Requirement(value)
        if req.marker and not req.marker.evaluate({"extra": ""}): continue
        try: version = md.version(req.name)
        except md.PackageNotFoundError:
            failures.append(str(req) + " missing"); continue
        if req.specifier and Version(version) not in req.specifier:
            failures.append(str(req) + " installed=" + version)
        pending.append(req.name)
if failures: raise RuntimeError("Scientific dependency closure mismatch: " + repr(failures))
keys = ["torch", "numpy", "scipy", "pandas", "scikit-learn", "mne", "yasa", "lightgbm", "antropy", "numba", "joblib"]
info = {"versions": {n: md.version(n) for n in keys}, "python": __import__("sys").version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependency_check": "scientific dependency closure only; unrelated inherited Colab UI packages not imported"}
print("RESEARCH5_ENV_JSON=" + json.dumps(info))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path('/content/drive/MyDrive/research5'))
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--run-id')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--development-queue', action='store_true', help='After compatible A/B gates, run the separately frozen 10-person development queue; no new architecture training')
    parser.add_argument('--recover-stale-lock', action='store_true', help='Only use after the previous runtime has stopped; heartbeat must be >10 minutes old')
    a = parser.parse_args()
    drive = Path('/content/drive')
    mydrive = drive / 'MyDrive'
    project, source = a.project_root.resolve(), a.source_root.resolve()
    if not drive.is_mount() or not mydrive.is_dir():
        raise RuntimeError('Mount Google Drive in the notebook first; /content/drive must be an actual mount')
    if not project.is_dir() or not project.is_relative_to(mydrive.resolve()):
        raise RuntimeError('The existing research5 project directory must be inside the mounted MyDrive')
    files = source_files(source)
    file_hashes = {str(p.relative_to(source)): sha256(p) for p in files}
    code_id = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()[:20]
    run_id = a.run_id or 'colab-drive-' + code_id
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
        raise ValueError('run-id must contain only letters, digits, hyphen or underscore')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    logdir = project / 'logs' / run_id / stamp
    logdir.mkdir(parents=True, exist_ok=False)
    state = {'status': 'RUNNING', 'stage': 'initializing', 'run_id': run_id, 'code_id': code_id,
        'source_root': str(source), 'project_root': str(project), 'started_utc': stamp,
        'host': socket.gethostname(), 'training': False, 'automatic_expansion': False,
        'frozen_development_queue_requested': a.development_queue,
        'resume_unit': 'whole completed phase; partial phases retained and rerun', 'logs': str(logdir)}
    runner = DriveRun(project, run_id, logdir, state)
    lock = project / 'status' / run_id / 'run.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        heartbeat = read_json(runner.heartbeat_file, {})
        stale = time.time() - max(heartbeat.get('unix_time', 0), lock.stat().st_mtime) > 600
        if not a.recover_stale_lock or not stale:
            raise RuntimeError('Another run or interrupted run holds the Drive lock. Stop the old runtime before explicit stale-lock recovery.')
        lock.rename(lock.with_name('run.lock.abandoned-' + stamp))
    lock.mkdir()
    write_json(lock / 'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'started_utc': stamp})
    runner.heartbeat()
    try:
        workspace = prepare_workspace(source, project, files, code_id)
        venv = Path('/content/research5-venvs') / code_id
        python = venv / 'bin/python'
        if not python.is_file():
            runner.command([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], 'create-environment', workspace)
        runner.command([python, '-m', 'pip', 'install', '-r', workspace / 'requirements-cloud.txt'], 'install-scientific-environment', workspace)
        runner.command([python, '-c', ENV_PROBE], 'validate-scientific-environment', workspace)
        log = (logdir / 'validate-scientific-environment.log').read_text(encoding='utf-8')
        env = json.loads(next(line.split('=', 1)[1] for line in log.splitlines() if line.startswith('RESEARCH5_ENV_JSON=')))
        device = 'cuda' if a.device == 'auto' and env['cuda_available'] else ('cpu' if a.device == 'auto' else a.device)
        if device == 'cuda' and not env['cuda_available']:
            raise RuntimeError('Requested CUDA is unavailable; select an available runtime without buying extra compute')
        signature = {'code_id': code_id, 'source_hashes': file_hashes, 'config_sha256': sha256(workspace / 'pilot/config_smoke.json'),
            'scientific_versions': env['versions'], 'python_major_minor': '.'.join(sys.version.split()[0].split('.')[:2]),
            'device': device, 'gpu': env['gpu'] if device == 'cuda' else None}
        signature_file = project / 'status' / run_id / 'run_signature.json'
        old = read_json(signature_file)
        if old is not None and old != signature:
            raise RuntimeError('Code/config/environment differs from this run; use a new run-id instead of mixing stages')
        write_json(signature_file, signature)
        write_json(project / 'status' / run_id / 'environment.json', env)
        if a.setup_only:
            state.update(status='SETUP_COMPLETE', stage='ready_for_pilot')
            return
        verify_downloads(workspace)
        runner.command([python, 'pilot/run_pilot.py', 'prepare'], 'prepare-one-record', workspace)
        verify_downloads(workspace)
        identity_env = None
        if (workspace / 'pilot/development_queue.py').is_file():
            identity_code = "import sys,json;sys.path.insert(0,'pilot');from development_queue import source_identity;print('RESEARCH5_IDENTITY_JSON='+json.dumps(source_identity(),sort_keys=True))"
            runner.command([python, '-c', identity_code], 'capture-pipeline-identity', workspace)
            identity_log = (logdir / 'capture-pipeline-identity.log').read_text(encoding='utf-8')
            identity_env = next(line.split('=', 1)[1] for line in identity_log.splitlines() if line.startswith('RESEARCH5_IDENTITY_JSON='))
        if a.development_queue and identity_env is None:
            raise RuntimeError('Development queue source/identity function is missing from this uploaded bundle')
        from cloud_run import stage_a_gate
        for phase in ['A', 'B']:
            stage_id = run_id + '-' + phase.lower()
            folder = project / 'results' / stage_id
            if verify_completed(folder):
                state['stage'] = 'resumed-completed-' + phase.lower()
                runner.heartbeat()
                print('Verified and skipped completed phase', phase, flush=True)
            else:
                if folder.exists():
                    folder.rename(folder.with_name(stage_id + '-partial-' + stamp))
                command = [python, 'pilot/run_pilot.py', 'run', '--model', 'both', '--phase', phase,
                           '--device', device, '--run-id', stage_id]
                if phase == 'B':
                    command.append('--diagnostics')
                runner.command(command, 'stage-' + phase.lower(), workspace,
                    {'RESEARCH5_SOURCE_IDENTITY_JSON': identity_env} if identity_env is not None else None)
                mark_completed(folder)
            if phase == 'A':
                evidence = stage_a_gate(folder)
                write_json(project / 'status' / run_id / 'stage_a_gate.json', evidence)
        state.update(status='PILOT_COMPLETE_REVIEW_REQUIRED', stage='complete',
            result_folders=[str(project / 'results' / (run_id + '-a')), str(project / 'results' / (run_id + '-b'))],
            next_step='Review net harm and simple baselines; no automatic training or participant expansion')
        if a.development_queue:
            runner.command([python, 'pilot/development_queue.py', 'run', '--manifest',
                workspace / 'pilot/development_manifest.json', '--stage-a-run-id', run_id + '-a',
                '--stage-b-run-id', run_id + '-b', '--device', device, '--max-hours', '4'],
                'frozen-development-queue', workspace)
            queue_folder = project / 'results/development-queue-v1'
            state.update(status='DEVELOPMENT_QUEUE_RETURNED_REVIEW_REQUIRED', stage='development-queue-returned',
                development_queue_state=read_json(queue_folder / 'state.json', {}),
                development_decision=read_json(queue_folder / 'aggregate/decision.json', {}),
                next_step='Inspect saved queue state and participant metrics; no new-architecture training')
    except BaseException as exc:
        state.update(status='FAILED', message=str(exc), error_class=type(exc).__name__, traceback=traceback.format_exc())
        raise
    finally:
        runner.heartbeat()
        report = project / 'reports' / run_id / 'RUN_SUMMARY.md'
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('\n'.join(['# Colab / Drive execution status', '',
            f"Status: {state['status']}", f"Stage: {state['stage']}",
            f"Code/config identity: {code_id}", f"Logs: {logdir}",
            f"Stage A: {project / 'results' / (run_id + '-a')}",
            f"Stage B: {project / 'results' / (run_id + '-b')}",
            f"Frozen development queue requested: {a.development_queue}",
            f"Queue state: {project / 'results/development-queue-v1/state.json'}",
            f"Message: {state.get('message', state.get('next_step', 'See controller status'))}", '',
            'No new architecture was trained. A stale RUNNING heartbeat is not a successful result.',
            'Completed stages are version/config/checksum verified on resume; partial phases are preserved.',
            'This driver does not prevent normal Colab disconnection or runtime reclamation.', '']), encoding='utf-8')
        if lock.exists():
            shutil.rmtree(lock)  # only this run's generated lock, never data or outputs


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received runtime signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
