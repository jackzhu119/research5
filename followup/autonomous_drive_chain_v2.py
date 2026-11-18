"""Self-bootstrapping foreground Colab Pro GPU orchestrator for research5 v2.

Designed for unattended execution (Colab Pro Background Execution):
1) Bootstraps an isolated scientific venv inheriting Colab's CUDA PyTorch.
2) Runs strict GPU preflight determinism probe (AnySleep + 7-arm SequenceHead backward),
   all module selftests, and SHA256/SHA1 verification of SC400-SC419 + run1/2/3.
3) Runs Phases 1-5 sequentially with per-participant/subexperiment atomic resume receipts:
     - Phase 1: Stage B v2 (n=20, imports verified SC400-409 + evaluates SC410-419)
     - Phase 2: Stage C v2 (true uncropped 2x2 oracle + 8 non-leaky arms, n=20)
     - Phase 3: Stage C Small-Model v2 (7 arms x 5-fold CV x 3 seeds + unseen OOD, n=20)
     - Phase 4: CAP v2 External Verification (n1-n3, 512Hz native + unaliased morphologies)
     - Phase 5: Stress & Sensitivity v2 (3A sleep span, 3B dose/morphologies,
                3C multi-channel/attention, 3D multi-checkpoint, 3E lag/margin)
4) Refreshes FINAL_AUTONOMOUS_REPORT_V2.md, 6 PNG figures, and zip archive after every phase.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "followup") not in sys.path:
    sys.path.insert(0, str(ROOT / "followup"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


def sync_fs() -> None:
    try:
        os.sync()
    except AttributeError:
        pass


class OrchestratorV2:
    def __init__(self, project: Path, exec_root: Path, venv_python: Path):
        self.project = project
        self.exec_root = exec_root
        self.python = venv_python
        self.status_dir = project / "status/autonomous-pipeline-v2"
        self.log_dir = project / "logs/autonomous-pipeline-v2"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.main_log = self.log_dir / "pipeline.log"
        self.status_file = self.status_dir / "status.json"
        self.heartbeat_file = self.status_dir / "heartbeat.json"
        self.state: dict[str, Any] = {
            "pipeline_id": "autonomous-pipeline-v2",
            "status": "RUNNING",
            "phase": "initializing",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_utc": utc_now(),
            "completed_phases": [],
            "note": "Foreground Colab Pro GPU->Drive pipeline; results stream directly to Drive after every participant/phase.",
        }

    def log(self, msg: str) -> None:
        line = f"[{utc_now()}] {msg}\n"
        with self.main_log.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        print(line, end="", flush=True)

    def heartbeat(self, **extra: Any) -> None:
        self.state.update(extra)
        now = time.time()
        self.state["updated_utc"] = utc_now()
        self.state["unix_time"] = now
        write_json(self.status_file, self.state)
        write_json(
            self.heartbeat_file,
            {
                "unix_time": now,
                "utc": self.state["updated_utc"],
                "status": self.state["status"],
                "phase": self.state["phase"],
                "pid": os.getpid(),
                "host": socket.gethostname(),
            },
        )

    def run_step(self, name: str, cmd: list[Any], cwd: Path | None = None, echo_stdout: bool = False) -> None:
        self.heartbeat(phase=name)
        step_log = self.log_dir / f"{name}.log"
        self.log(f"Starting phase {name}: {' '.join(map(str, cmd))}")
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 1)))
        env.setdefault("MKL_NUM_THREADS", env["OMP_NUM_THREADS"])
        t0 = time.monotonic()
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd or self.exec_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )

        def pump() -> None:
            with step_log.open("a", encoding="utf-8") as lf:
                assert proc.stdout is not None
                for line in proc.stdout:
                    lf.write(line)
                    lf.flush()
                    if echo_stdout or any(
                        tag in line
                        for tag in ("[StageB-v2]", "[StageC-v2]", "[SmallModel-v2]", "[CAP-v2]", "[Stress-v2", "COMPLETE", "OK", "PREFLIGHT", "ERROR", "Traceback")
                    ):
                        print(f"  {line}", end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                self.heartbeat(phase=name, active_child_pid=proc.pid, phase_elapsed_s=round(time.monotonic() - t0, 1))
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=10)
            reader.join(timeout=10)

        if proc.returncode != 0:
            raise RuntimeError(f"Phase {name} exited with code {proc.returncode}; inspect {step_log}")

        self.state["completed_phases"].append(
            {"phase": name, "seconds": round(time.monotonic() - t0, 1), "completed_utc": utc_now()}
        )
        self.heartbeat(phase=f"{name}-complete", active_child_pid=None)
        sync_fs()
        self.log(f"Completed phase {name} in {time.monotonic() - t0:.1f}s")

    def refresh_report(self, tag: str) -> None:
        self.run_step(
            f"report-export-{tag}",
            [self.python, self.exec_root / "followup/generate_final_drive_report_v2.py", "--project-root", self.project],
            cwd=self.exec_root,
        )


def sync_code_assets(src_followup: Path, project: Path, exec_root: Path) -> None:
    """Ensure Drive/followup and exec_root/followup + manifest_20 stay synchronized."""
    for dst_dir in (project / "followup", exec_root / "followup"):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_followup.glob("*.py")):
            target = dst_dir / p.name
            if target.resolve() != p.resolve():
                shutil.copyfile(p, target)

    manifest_src = src_followup.parent / "manifests/development_manifest_20.json"
    if manifest_src.is_file():
        for m_dst in (project / "manifests/development_manifest_20.json", exec_root / "manifests/development_manifest_20.json"):
            m_dst.parent.mkdir(parents=True, exist_ok=True)
            if m_dst.resolve() != manifest_src.resolve():
                shutil.copyfile(manifest_src, m_dst)


def bootstrap_venv(exec_root: Path, venv_dir: Path, logger) -> Path:
    """Create isolated venv inheriting runtime CUDA PyTorch and install scientific stack cleanly."""
    py_exe = venv_dir / "bin/python"
    if not py_exe.exists():
        logger(f"Creating isolated scientific venv at {venv_dir} (--system-site-packages)...")
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv_dir)],
            check=True,
        )

    probe_cmd = [
        str(py_exe),
        "-c",
        "import torch, numpy, scipy, pandas, sklearn, mne, yasa, matplotlib, requests, psutil; _ = torch.zeros(2).numpy()",
    ]
    if subprocess.run(probe_cmd, capture_output=True).returncode == 0:
        logger("Scientific venv imports already satisfied (inheriting runtime CUDA PyTorch).")
        return py_exe

    req_file = exec_root / "requirements-cloud.txt"
    fallback = ["mne", "yasa", "scipy", "pandas", "scikit-learn", "matplotlib", "lightgbm", "antropy", "requests", "psutil"]
    if req_file.is_file():
        logger("Installing scientific stack via requirements-cloud.txt...")
        r1 = subprocess.run(
            [sys.executable, "-m", "pip", "--python", str(py_exe), "install", "-q", "-r", str(req_file)],
            text=True,
            capture_output=True,
        )
        if r1.returncode != 0 or subprocess.run(probe_cmd, capture_output=True).returncode != 0:
            logger(f"requirements-cloud.txt conflicted with runtime PyTorch/NumPy; restoring clean numpy<2.5 + scientific stack: {r1.stderr[-300:]}")
            subprocess.run([sys.executable, "-m", "pip", "--python", str(py_exe), "install", "-q", "numpy<2.3", *fallback], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "--python", str(py_exe), "install", "-q", *fallback], check=True)

    subprocess.run(probe_cmd, check=True)
    return py_exe


def acquire_drive_lock(project: Path, auto_recover: bool) -> Path:
    lock_dir = project / "status/autonomous-pipeline-v2/run.lock"
    hb_file = project / "status/autonomous-pipeline-v2/heartbeat.json"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    if lock_dir.exists():
        hb = read_json(hb_file, {})
        age_s = time.time() - max(float(hb.get("unix_time", 0.0)), lock_dir.stat().st_mtime)
        if auto_recover or age_s > 180:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            lock_dir.rename(lock_dir.with_name(f"run.lock.abandoned-{stamp}"))
        else:
            raise RuntimeError(f"Active Drive lock at {lock_dir} (heartbeat age {age_s:.1f}s)")
    lock_dir.mkdir(parents=True, exist_ok=True)
    write_json(lock_dir / "owner.json", {"host": socket.gethostname(), "pid": os.getpid(), "started_utc": utc_now()})
    return lock_dir


def gpu_and_sequencehead_preflight_code(device: str, require_gpu: bool) -> str:
    return f"""
import json, sys, pathlib
sys.path.insert(0, 'followup')
import torch, numpy, scipy, pandas, sklearn, mne, yasa, matplotlib
from common_v2 import gpu_preflight_probe, configure_torch_determinism, resolve_device
from stage_c_small_model_v2 import ARMS, build_sequence_head
dev = resolve_device({device!r})
if {require_gpu!r} and not dev.startswith('cuda'):
    raise RuntimeError('GPU runtime required (--require-gpu), but CUDA is unavailable. In Colab: Runtime -> Change runtime type -> L4/A100 GPU!')
info = gpu_preflight_probe(dev)
configure_torch_determinism(dev, seed=20261117, strict=True)
x_syn = torch.randn(2, 48, 31, device=dev)
y_syn = torch.randint(0, 5, (2, 48), device=dev)
m_syn = torch.ones(2, 48, device=dev)
for arm in ARMS:
    head = build_sequence_head(31, arm, context_radius=10, hidden=64).to(dev)
    head.train()
    if arm == 'rest_spectral_reg':
        out_b = head(torch.cat([x_syn, x_syn + 0.15 * torch.randn_like(x_syn)], dim=0), valid_mask=torch.cat([m_syn, m_syn], dim=0))
        out, jit_o = out_b[:2], out_b[2:]
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 5), y_syn.reshape(-1)) + 0.25 * torch.mean((jit_o - out) ** 2)
    elif arm == 'constrained_residual':
        out_b = head(torch.cat([x_syn, x_syn], dim=0), valid_mask=torch.cat([m_syn, m_syn], dim=0))
        out, cln_o = out_b[:2], out_b[2:].detach()
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 5), y_syn.reshape(-1)) + 0.60 * torch.mean((out - cln_o) ** 2)
    else:
        out = head(x_syn, valid_mask=m_syn)
        loss = torch.nn.functional.cross_entropy(out.reshape(-1, 5), y_syn.reshape(-1))
    loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(loss).all()
info['sequence_head_cuda_backward_arms'] = len(ARMS)
print('PREFLIGHT_GPU_JSON=' + json.dumps(info))
"""


def run_chain_v2(args) -> None:
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    local_lock = open("/tmp/research5-autonomous-pipeline-v2.lock", "a")
    try:
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ALREADY_RUNNING: local lock for autonomous-pipeline-v2 is held.", flush=True)
        return

    project = Path(args.project_root).resolve()
    exec_root = Path(args.exec_root).resolve() if args.exec_root else ROOT
    if not project.is_dir():
        raise RuntimeError(f"Project root directory missing: {project}")

    sync_code_assets(Path(__file__).resolve().parent, project, exec_root)
    manifest_path = exec_root / "manifests/development_manifest_20.json"
    venv_py = Path(args.venv_python) if args.venv_python else bootstrap_venv(
        exec_root, Path(args.venv_dir), lambda m: print(f"[Bootstrap] {m}", flush=True)
    )

    drive_lock = acquire_drive_lock(project, auto_recover=not args.no_auto_recover_lock)
    orch = OrchestratorV2(project, exec_root, venv_py)
    orch.heartbeat(device_requested=args.device, require_gpu=args.require_gpu)
    orch.log("Autonomous Colab Pro GPU->Drive v2 pipeline started.")

    try:
        # Phase 0A: Fast unit selftests on every v2 module
        selftest_code = (
            "import subprocess, sys\n"
            "cmds = [['followup/common_v2.py'], ['followup/manifest_v2.py','selftest'], "
            "['followup/stage_b_expansion_v2.py','selftest'], ['followup/stage_c_v2.py','selftest'], "
            "['followup/stage_c_small_model_v2.py','selftest'], ['followup/cap_verification_v2.py','selftest'], "
            "['followup/stress_sensitivity_v2.py','selftest'], ['followup/generate_final_drive_report_v2.py','selftest']]\n"
            "[subprocess.run([sys.executable] + c, check=True) for c in cmds]\n"
            "print('ALL_V2_SELFTESTS_OK')"
        )
        orch.run_step("preflight-unit-selftests", [venv_py, "-c", selftest_code], echo_stdout=True)

        # Phase 0B: GPU deterministic preflight probe (AnySleep + 7 SequenceHeads)
        if not args.skip_gpu_probe:
            orch.run_step(
                "preflight-gpu-determinism",
                [venv_py, "-c", gpu_and_sequencehead_preflight_code(args.device, args.require_gpu)],
                echo_stdout=True,
            )
            probe_log = (orch.log_dir / "preflight-gpu-determinism.log").read_text(encoding="utf-8")
            for ln in probe_log.splitlines():
                if ln.startswith("PREFLIGHT_GPU_JSON="):
                    write_json(orch.status_dir / "environment.json", json.loads(ln.split("=", 1)[1]))

        # Phase 0C: Freeze manifest & SHA256-verify all 20 Sleep-EDF files + run1/2/3
        orch.run_step("freeze-and-prepare-20", [venv_py, exec_root / "followup/manifest_v2.py", "freeze", "--output", manifest_path])
        orch.run_step("verify-all-assets", [venv_py, exec_root / "followup/manifest_v2.py", "prepare", "--project-root", project, "--manifest", manifest_path])

        # Phase 1: Stage B v2 (n=20)
        b_st = read_json(project / "results/development-queue-v2/state.json", {})
        if b_st.get("status") == "COMPLETE" and len(b_st.get("participants", {})) >= 20:
            orch.log("Stage B v2 already 20/20 COMPLETE on Drive; skipping.")
        else:
            orch.run_step("stage-b-expansion-v2", [venv_py, exec_root / "followup/stage_b_expansion_v2.py", "run", "--project-root", project, "--manifest", manifest_path, "--device", args.device], echo_stdout=True)
            orch.refresh_report("after-stage-b-v2")

        # Phase 2: Stage C v2 (Repaired uncropped 2x2 oracle + 8 arms, n=20)
        c_st = read_json(project / "results/stage-c-v2/state.json", {})
        if c_st.get("status") == "COMPLETE" and len(c_st.get("participants", {})) >= 20:
            orch.log("Stage C v2 already 20/20 COMPLETE on Drive; skipping.")
        else:
            orch.run_step("stage-c-v2", [venv_py, exec_root / "followup/stage_c_v2.py", "run", "--project-root", project, "--manifest", manifest_path, "--device", args.device], echo_stdout=True)
            orch.refresh_report("after-stage-c-v2")

        # Phase 3: Stage C Small-Model v2 (7 arms x 5-fold CV x 3 seeds + unseen OOD)
        sm_st = read_json(project / "results/stage-c-small-model-v2/state.json", {})
        if sm_st.get("status") == "COMPLETE":
            orch.log("Stage C Small-Model v2 already COMPLETE on Drive; skipping.")
        else:
            orch.run_step("stage-c-small-model-v2", [venv_py, exec_root / "followup/stage_c_small_model_v2.py", "run", "--project-root", project, "--manifest", manifest_path, "--epochs", str(args.small_model_epochs), "--device", args.device], echo_stdout=True)
            orch.refresh_report("after-small-model-v2")

        # Phase 4: CAP v2 External Verification (n1, n2, n3)
        cap_st = read_json(project / "results/cap-check-v2/state.json", {})
        if cap_st.get("status") == "COMPLETE" and len(cap_st.get("records", {})) >= 3:
            orch.log("CAP v2 already 3/3 COMPLETE on Drive; skipping.")
        else:
            orch.run_step("cap-verification-v2", [venv_py, exec_root / "followup/cap_verification_v2.py", "run", "--project-root", project, "--manifest", manifest_path, "--records", "n1,n2,n3", "--device", args.device], echo_stdout=True)
            orch.refresh_report("after-cap-v2")

        # Phase 5: Stress & Sensitivity 3A-3E (n=20)
        s_st = read_json(project / "results/stress-sensitivity-v2/state.json", {})
        if s_st.get("status") == "COMPLETE":
            orch.log("Stress & Sensitivity v2 already COMPLETE on Drive; skipping.")
        else:
            orch.run_step("stress-sensitivity-v2", [venv_py, exec_root / "followup/stress_sensitivity_v2.py", "run", "--project-root", project, "--manifest", manifest_path, "--device", args.device], echo_stdout=True)

        # Final comprehensive report, 6 figures & archive export
        orch.refresh_report("final")
        orch.heartbeat(status="ALL_STAGES_COMPLETE", phase="complete", completed_utc=utc_now())
        sync_fs()
        orch.log("ALL_STAGES_COMPLETE_V2: Results, 6 PNG figures, FINAL_AUTONOMOUS_REPORT_V2.md, and research5-autonomous-results-v2.zip saved cleanly on Google Drive.")
    except BaseException as exc:
        orch.heartbeat(status="FAILED", error_class=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        orch.log(f"PIPELINE_V2 FAILED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        try:
            orch.refresh_report("partial-on-error")
        except Exception:
            pass
        raise
    finally:
        try:
            shutil.rmtree(drive_lock, ignore_errors=True)
            fcntl.flock(local_lock, fcntl.LOCK_UN)
            local_lock.close()
        except Exception:
            pass
        sync_fs()


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root_tmp = Path(tmp)
        proj = root_tmp / "proj"
        ex = root_tmp / "exec"
        proj.mkdir()
        ex.mkdir()
        sync_code_assets(Path(__file__).resolve().parent, proj, ex)
        assert (proj / "followup/autonomous_drive_chain_v2.py").is_file()
        assert (ex / "followup/common_v2.py").is_file()
        lock = acquire_drive_lock(proj, auto_recover=False)
        assert lock.is_dir()
        lock2 = acquire_drive_lock(proj, auto_recover=True)
        assert lock2.is_dir()
    print("AUTONOMOUS_DRIVE_CHAIN_V2_SELFTEST_OK")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=False)
    sub.add_parser("selftest")
    p.add_argument("--project-root", type=Path, default=Path("/content/drive/MyDrive/research5"))
    p.add_argument("--exec-root", type=Path, default=ROOT)
    p.add_argument("--venv-dir", type=Path, default=Path("/content/research5-venvs/v2"))
    p.add_argument("--venv-python", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--small-model-epochs", type=int, default=75)
    p.add_argument("--skip-gpu-probe", action="store_true")
    p.add_argument("--no-auto-recover-lock", action="store_true")
    a = p.parse_args()
    if a.command == "selftest":
        selftest()
    else:
        run_chain_v2(a)


if __name__ == "__main__":
    main()
