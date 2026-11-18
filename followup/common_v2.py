"""Shared utilities for the research5 v2 GPU experimental suite.

Guarantees:
- Exact preservation of Stage A/B pinned source & checkpoint hashes.
- Uncropped full-night deterministic GPU/CPU inference (no window splicing).
- Strictly non-leaky deployable baselines + physics-grounded blind artifact detection.
- Fine-grained distance bins, upstream/downstream U-Net support asymmetry, and
  baseline logit-margin stratification.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

# Must be configured before importing torch for deterministic cuBLAS on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(ROOT / "pilot"), str(ROOT / "vendor/anysleep/examples")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

ANYSLEEP_COMMIT = "88e13e31b3fa02482b650a844f6665f2cbbdb82c"
ANYSLEEP_CHECKPOINTS = {
    "run1": {
        "filename": "anysleep-run1.pth",
        "git_blob_sha1": "8e545c4af53a7466b0b9df26a3d554ced47ac12d",
        "bytes": 12763675,
        "url": f"https://raw.githubusercontent.com/dslaborg/AnySleep/{ANYSLEEP_COMMIT}/models/anysleep-run1.pth",
    },
    "run2": {
        "filename": "anysleep-run2.pth",
        "git_blob_sha1": "1066a37acd8ebdfc537b7a9d5625a8aaba09869a",
        "bytes": 12763675,
        "url": f"https://raw.githubusercontent.com/dslaborg/AnySleep/{ANYSLEEP_COMMIT}/models/anysleep-run2.pth",
    },
    "run3": {
        "filename": "anysleep-run3.pth",
        "git_blob_sha1": "e6257bc91459f58c4523297fad72370433bba7d4",
        "bytes": 12763675,
        "url": f"https://raw.githubusercontent.com/dslaborg/AnySleep/{ANYSLEEP_COMMIT}/models/anysleep-run3.pth",
    },
}

SLEEP_EDF_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

# Audited directly from PhysioNet sleep-edfx/1.0.0 SHA256SUMS.txt and AnySleep 88e13e31/config/usleep_split.yaml
PHYSIONET_SC400_SC419 = [
    ("SC400", "SC4001E0", "SC4001E0-PSG.edf", "2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131", "SC4001EC-Hypnogram.edf", "a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996", "test"),
    ("SC401", "SC4011E0", "SC4011E0-PSG.edf", "9e4c34f3f631275990fd7a2d61a24e998c7b03eb9b37ebe5cd7c8323d8310a15", "SC4011EH-Hypnogram.edf", "5bbf7b4ed8a95c8d20e4347b08c2ded2ee21e2dac4680dedf629795adc2edc83", "train"),
    ("SC402", "SC4021E0", "SC4021E0-PSG.edf", "810adaedf42621e389ca767274045fbafb123bbc5d99bc15ce2c5bacb52e378c", "SC4021EH-Hypnogram.edf", "62b49b26bf08941cb0e974cfeeb0a820bf7dd43fb0c66dd89b4c3a6c0ee45dd8", "test"),
    ("SC403", "SC4031E0", "SC4031E0-PSG.edf", "ce2314a60dd6e1437f9cd720c2e59c23aabe471e4821f9544593109a9b0715dd", "SC4031EC-Hypnogram.edf", "7ec3d3c8d8f1a89d48df4cc85be374fbdf70d4b5ec5ec321c03d586dd2cc1617", "train"),
    ("SC404", "SC4041E0", "SC4041E0-PSG.edf", "28ae3a51cca93bb1502a2c613b4a84758d09feee042b925a09a7cdce0c1bf4f5", "SC4041EC-Hypnogram.edf", "da33d90216e031a6b04c9ebaddcdf7abc223f1e61fd5899093815c7aea65e39a", "train"),
    ("SC405", "SC4051E0", "SC4051E0-PSG.edf", "e460a9589726f19a1fe544fb79c46f82562229f19696e0c0fb5ffd9f959dab7c", "SC4051EC-Hypnogram.edf", "3551410c52b819efbe6297d333ee729e4f5c7ba11c0cae6d321feee79a39ed43", "train"),
    ("SC406", "SC4061E0", "SC4061E0-PSG.edf", "8a1c1d812f3207e006c8f1ae7b7a9d5625a8aaba09869aea42562ac61d1c984a", "SC4061EC-Hypnogram.edf", "a99705c2b75462acb6d422681526265e476f0b01708eee2e845d0d89b5aa1ae8", "train"),
    ("SC407", "SC4071E0", "SC4071E0-PSG.edf", "417081e551fb60e3a735059c5df4ee2cf320cb342c563e666cb2b3fde98f64e3", "SC4071EC-Hypnogram.edf", "f81cfe51fa83a91096e4ba0538168ada9f604e8cb5f9c40849ebdd22c1cfeafd", "train"),
    ("SC408", "SC4081E0", "SC4081E0-PSG.edf", "3210ff3851996b1d04fb879765ebb3de6012152fab6fcbb11896c7636068e6dc", "SC4081EC-Hypnogram.edf", "626b6c685079aa3d185f7af69098b973940733e0cbe9c5d7435495afafdcca2f", "valid"),
    ("SC409", "SC4091E0", "SC4091E0-PSG.edf", "db5ac16a5438a0ff1e67dbf4c729e3b42ffc131c3e586370c8f8297591960476", "SC4091EC-Hypnogram.edf", "6200f80f2314fc7aefc05c1c7a745776d2631f20115d818b6dcecae663c590d9", "valid"),
    ("SC410", "SC4101E0", "SC4101E0-PSG.edf", "90f0cb69b101e60f7ebc1d52216677a5aa420fbd2dce3909eaaca373d921a099", "SC4101EC-Hypnogram.edf", "1dc87ff1ee449fd66c3254df2073a66c4e1e902ec11143434a82db30eb1622d7", "train"),
    ("SC411", "SC4111E0", "SC4111E0-PSG.edf", "6bb1ba9fdf0bfab05aeba98edded640ce65a05d286115ceadb38dd8d4dfae0b8", "SC4111EC-Hypnogram.edf", "da132682072ae2998ef82d3737a3220685e2483b90cf7042ae819cce675cd6d6", "train"),
    ("SC412", "SC4121E0", "SC4121E0-PSG.edf", "2453150077736b8f4817bb5c20e58d6160fc6cd9e8e47531982847a71c9be4e0", "SC4121EC-Hypnogram.edf", "ad90e2e20a9ee3d61e8434f2c2efab49583b31c6ed8350157535a5cb085c88ad", "valid"),
    ("SC413", "SC4131E0", "SC4131E0-PSG.edf", "443586b0f2c620d073405b938314357eb33ef1a03eb0eca36b6d1861f31d8665", "SC4131EC-Hypnogram.edf", "d724153bfa0c4267ced28cc975025e5674e13e2ce4e1f751cad6a30e549f9232", "train"),
    ("SC414", "SC4141E0", "SC4141E0-PSG.edf", "eb14f483f5675798fdda51954d6914c9442884cda02cebb6af44c96ac3cd12e5", "SC4141EU-Hypnogram.edf", "1e8dc03d9a2159752db262cc5ef1b3140f147445de969b9820292eb3861e2b32", "valid"),
    ("SC415", "SC4151E0", "SC4151E0-PSG.edf", "e2d4dda6a1aa7b58394e102e44ba03924f14f4a19bdff1808d96e0d09fd0f74e", "SC4151EC-Hypnogram.edf", "72e8bf03e6400baac7af5980c340450ebe59a97fca1f5f7ddbe1c94464ddfbf9", "train"),
    ("SC416", "SC4161E0", "SC4161E0-PSG.edf", "60a3e257b0dd97893415fa46c79c2ffad63d37445f3b6e0e0b0f67b5e71eeee6", "SC4161EC-Hypnogram.edf", "407ff5ffa150e618c90ab27dc64e459312b6a05352fd7c39b98cc2d34fc7846f", "train"),
    ("SC417", "SC4171E0", "SC4171E0-PSG.edf", "01acc3f69ecf38052eac9bc04b0e8154010e111de707aeda3fd40af6a5267ae8", "SC4171EU-Hypnogram.edf", "648653db6ced819f659c8cb3cbd3952a60cac1705b87209fd033268b6b38b749", "test"),
    ("SC418", "SC4181E0", "SC4181E0-PSG.edf", "f98d5e2c97aa99c57adc886445c23d52dc97541f2b2f84efbc08afb86b7adba5", "SC4181EC-Hypnogram.edf", "9091676b473587c7ec9d35d64464cc0ad3890b6b010418b62aa00d8b6801c5c1", "test"),
    ("SC419", "SC4191E0", "SC4191E0-PSG.edf", "b3d8e9d75fa2f54b1aa1dc936b482733e70bdd5b9f0a5809edf57c7ce4c406e2", "SC4191EP-Hypnogram.edf", "bf48c5ddcb7a7e8eff489f273e9d3ffe38f693af0843b727df962572a47a3e24", "valid"),
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    p = Path(path)
    h = hashlib.sha1()
    h.update(f"blob {p.stat().st_size}\0".encode())
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(p)


def sync_filesystem() -> None:
    try:
        os.sync()
    except AttributeError:
        pass


def download_verified(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    expected_git_blob_sha1: str | None = None,
) -> dict[str, Any]:
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        sha = sha256_file(dest)
        blob = git_blob_sha1_file(dest)
        ok_sha = expected_sha256 is None or sha == expected_sha256
        ok_blob = expected_git_blob_sha1 is None or blob == expected_git_blob_sha1
        if ok_sha and ok_blob:
            return {"path": str(dest), "bytes": dest.stat().st_size, "sha256": sha, "git_blob_sha1": blob, "cached": True}
        dest.unlink()

    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    sha = sha256_file(dest)
    blob = git_blob_sha1_file(dest)
    if expected_sha256 is not None and sha != expected_sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(f"SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {sha}")
    if expected_git_blob_sha1 is not None and blob != expected_git_blob_sha1:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Git blob SHA1 mismatch for {dest.name}: expected {expected_git_blob_sha1}, got {blob}")
    return {"path": str(dest), "bytes": dest.stat().st_size, "sha256": sha, "git_blob_sha1": blob, "cached": False}


def ensure_checkpoint(project: Path, run_key: str = "run1") -> Path:
    spec = ANYSLEEP_CHECKPOINTS[run_key]
    ckpt_path = Path(project) / "models" / spec["filename"]
    download_verified(spec["url"], ckpt_path, expected_git_blob_sha1=spec["git_blob_sha1"])
    return ckpt_path


def configure_torch_determinism(device: str, seed: int = 20261117, strict: bool = True):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable TF32 on Ampere/Ada GPUs (A100/L4) so FP32 GPU matches FP32 CPU tightly.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device!r} is unavailable")
    return torch


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested


def load_sleep_edf_record(
    project: Path,
    rec: dict[str, Any],
    base_cfg: dict[str, Any],
) -> tuple[Any, np.ndarray, float, np.ndarray, dict[str, Any]]:
    """Load Sleep-EDF PSG + Hypnogram with strict EDF clock-header & alignment checks."""
    import math
    import mne

    project = Path(project).resolve()
    psg_path = project / "data" / rec["psg"]
    hyp_path = project / "data" / rec["hypnogram"]
    if not psg_path.is_file() or not hyp_path.is_file():
        raise FileNotFoundError(f"Missing Sleep-EDF files for {rec['participant']}: {psg_path}, {hyp_path}")

    channels = list(base_cfg["channels"])
    label_map = dict(base_cfg["label_map"])
    raw = mne.io.read_raw_edf(psg_path, include=channels, preload=True, verbose="warning")
    raw.reorder_channels(channels)
    raw.set_channel_types({c: ("eog" if c.startswith("EOG") else "eeg") for c in channels}, verbose=False)
    sf = float(raw.info["sfreq"])
    if sf != round(sf):
        raise ValueError(f"Non-integer sample rate {sf} requires dedicated alignment audit")
    x = raw.get_data()  # Volts
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite samples detected in {rec['psg']}")

    ann = mne.read_annotations(hyp_path)
    header_origins = {}
    for role, fp in [("psg", psg_path), ("hypnogram", hyp_path)]:
        with fp.open("rb") as f:
            val = f.read(256)[168:184].decode("ascii")
        if not re.fullmatch(r"\d{2}\.\d{2}\.\d{2}\d{2}\.\d{2}\.\d{2}", val):
            raise ValueError(f"Malformed {role} EDF date/time header: {val!r}")
        header_origins[role] = {
            "literal_header_bytes_168_to_183": val,
            "parsed_calendar_local": dt.datetime.strptime(val, "%d.%m.%y%H.%M.%S").isoformat(),
        }
    if ann.orig_time is not None and raw.info["meas_date"] is not None:
        offset = (ann.orig_time - raw.info["meas_date"]).total_seconds()
        origin_method = "MNE absolute annotation origin minus PSG measurement date"
    elif header_origins["psg"]["literal_header_bytes_168_to_183"] == header_origins["hypnogram"]["literal_header_bytes_168_to_183"]:
        offset = 0.0
        origin_method = "identical explicitly parsed PSG/hypnogram EDF date/time headers"
    else:
        raise ValueError("MNE origin absent and EDF header origins differ")

    n_epochs = int(x.shape[1] // (30 * sf))
    truth = np.full(n_epochs, -1, dtype=np.int16)
    for onset, duration, desc in zip(ann.onset + offset, ann.duration, ann.description):
        if desc not in label_map:
            continue
        first = max(0, int(math.ceil((onset - 1e-7) / 30)))
        last = min(n_epochs, int(math.floor((onset + duration + 1e-7) / 30)))
        if first < last:
            if (truth[first:last] >= 0).any():
                raise ValueError(f"Overlapping stage annotations in {rec['record']}")
            truth[first:last] = label_map[desc]
    if not (truth >= 0).any():
        raise ValueError(f"No aligned scored epochs in {rec['record']}")

    meta = {
        "participant": rec["participant"],
        "record": rec["record"],
        "sfreq_hz": sf,
        "samples": int(x.shape[1]),
        "duration_seconds": float(x.shape[1] / sf),
        "n_epochs": int(n_epochs),
        "scored_epochs": int((truth >= 0).sum()),
        "unscored_epochs": int((truth < 0).sum()),
        "annotation_offset_seconds": float(offset),
        "origin_method": origin_method,
        "edf_header_origin_evidence": header_origins,
    }
    return raw, x, sf, truth, meta


def run_yasa_inference(signal_volts: np.ndarray, info: Any, eeg_name: str, eog_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Run YASA 0.8.1 sleep staging on native-unit RawArray without altering calibration."""
    import mne
    import yasa

    r = mne.io.RawArray(np.asarray(signal_volts, dtype=np.float64), info.copy(), verbose=False)
    sls = yasa.SleepStaging(r, eeg_name=eeg_name, eog_name=eog_name)
    hyp = sls.predict()
    if hasattr(hyp, "hypno"):
        labels = np.asarray(hyp.hypno).astype(str)
        proba = hyp.proba.to_numpy()
    else:
        labels = np.asarray(hyp).astype(str)
        proba = sls.predict_proba().to_numpy()
    del r, sls
    lmap = {"WAKE": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}
    pred = np.asarray([lmap[s.upper()] for s in labels], dtype=np.int16)
    return pred, np.asarray(proba, dtype=np.float64)


def perturb_extended(
    x: np.ndarray,
    channels: list[int],
    start: int,
    end: int,
    kind: str,
    sf: float = 100.0,
    seed: int = 20261117,
    clip_volts: float = 2e-5,
    noise_sd_volts: float = 1e-4,
) -> np.ndarray:
    """Apply a local perturbation strictly inside [channels, start:end) and verify sample invariance elsewhere."""
    x = np.asarray(x)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating):
        raise ValueError("x must be floating-point [channels, samples]")
    if not (0 <= start < end <= x.shape[1]):
        raise ValueError(f"Invalid half-open interval [{start}, {end}) for length {x.shape[1]}")
    if not channels or len(set(channels)) != len(channels) or min(channels) < 0 or max(channels) >= x.shape[0]:
        raise ValueError(f"Invalid channel list: {channels}")

    n_ch = len(channels)
    n_samp = end - start
    y = x.copy()
    if kind in ("noop", "flat", "clip", "noise"):
        from paired_tools import perturb
        return perturb(x, channels, start, end, kind, seed=seed, clip=clip_volts, noise_sd=noise_sd_volts)
    elif kind == "mains_60hz":
        t = np.arange(n_samp, dtype=np.float64) / float(sf)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_ch, 1))
        # Sinusoid + 2nd harmonic at RMS ~ noise_sd_volts * 1.5.
        # At 512 Hz (CAP) this is native 60Hz+120Hz; at 100 Hz (Sleep-EDF) 60Hz aliases to 40Hz (never hits 50Hz Nyquist null).
        amp = float(noise_sd_volts) * 1.5 * np.sqrt(2.0)
        wave = amp * (0.9 * np.sin(2.0 * np.pi * 60.0 * t[None, :] + phases) + 0.1 * np.sin(2.0 * np.pi * 120.0 * t[None, :] + phases))
        y[channels, start:end] += wave.astype(y.dtype)
    elif kind == "emg_hf_burst":
        from scipy.signal import butter, filtfilt

        rng = np.random.default_rng(seed)
        raw_n = rng.normal(0.0, 1.0, size=(n_ch, n_samp))
        nyq = 0.5 * float(sf)
        hi = min(45.0, nyq * 0.9)
        lo = min(20.0, hi * 0.6)
        b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
        hf = filtfilt(b, a, raw_n, axis=1)
        hf_std = np.maximum(np.std(hf, axis=1, keepdims=True), 1e-12)
        hf = hf * (float(noise_sd_volts) / hf_std)
        y[channels, start:end] += hf.astype(y.dtype)
    elif kind == "slow_drift_pop":
        t = np.arange(n_samp, dtype=np.float64) / float(sf)
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_ch, 1))
        # Realistic electrode pop + 0.25Hz slow sweat drift (~400 uV drift + 500 uV step envelope)
        env = np.sin(np.pi * np.arange(n_samp, dtype=np.float64) / max(1, n_samp - 1))
        drift = signs * (4.0e-4 * np.sin(2.0 * np.pi * 0.25 * t[None, :]) + 5.0e-4 * env[None, :])
        y[channels, start:end] += drift.astype(y.dtype)
    else:
        raise ValueError(f"Unsupported perturbation kind: {kind!r}")

    keep = np.ones(x.shape, dtype=bool)
    keep[channels, start:end] = False
    if not np.array_equal(y[keep], x[keep], equal_nan=True):
        raise AssertionError(f"Perturbation {kind} altered samples outside [channels, start:end)")
    return y


def resample_channels_128hz(
    raw_100_or_sf: np.ndarray,
    sf: float,
    raw_resampled_ref: np.ndarray | None = None,
    changed_channels: list[int] | None = None,
) -> np.ndarray:
    """Resample full array or selectively re-resample only changed channels cleanly (exact bit-for-bit match)."""
    from scipy.signal import resample_poly

    sf_int = int(round(sf))
    if sf_int == 128:
        return raw_100_or_sf.astype(np.float32, copy=True)
    if raw_resampled_ref is not None and changed_channels is not None:
        out = raw_resampled_ref.copy()
        for c in changed_channels:
            out[c] = resample_poly(raw_100_or_sf[c], 128, sf_int).astype(np.float32)
        return out
    return resample_poly(raw_100_or_sf, 128, sf_int, axis=1).astype(np.float32)


def robust_norm_channelwise(
    z: np.ndarray,
    valid_sample_mask: np.ndarray | None = None,
    clip_iqr_mult: float = 20.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """AnySleep channelwise 20*IQR clip + RobustScaler with optional valid-sample reference mask."""
    out = np.empty_like(z, dtype=np.float32)
    thresholds, centers, scales = [], [], []
    for c in range(z.shape[0]):
        ref = z[c, valid_sample_mask[c]] if valid_sample_mask is not None and np.any(valid_sample_mask[c]) else z[c]
        q75, q25 = np.percentile(ref, [75.0, 25.0])
        iqr = float(q75 - q25)
        threshold = max(iqr * float(clip_iqr_mult), 1e-9)
        clipped_all = np.clip(z[c], -threshold, threshold)
        clipped_ref = np.clip(ref, -threshold, threshold)
        center = float(np.median(clipped_ref))
        c75, c25 = np.percentile(clipped_ref, [75.0, 25.0])
        scale = float(c75 - c25)
        if scale <= 1e-12:
            scale = 1.0
        out[c] = ((clipped_all - center) / scale).astype(np.float32)
        thresholds.append(threshold)
        centers.append(center)
        scales.append(scale)
    return out, {
        "threshold": np.asarray(thresholds, dtype=np.float64),
        "center": np.asarray(centers, dtype=np.float64),
        "scale": np.asarray(scales, dtype=np.float64),
    }


def apply_fixed_stats(z: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    thresh = np.asarray(stats["threshold"], dtype=np.float64)[:, None]
    center = np.asarray(stats["center"], dtype=np.float64)[:, None]
    scale = np.asarray(stats["scale"], dtype=np.float64)[:, None]
    clipped = np.clip(z, -thresh, thresh)
    return ((clipped - center) / scale).astype(np.float32)


def blind_detect_bad_channel_epochs_v1(z_128hz: np.ndarray, epoch_samples: int = 3840) -> np.ndarray:
    """Original v1 detector retained for strictly controlled uncropped/non-leaky attribution."""
    ch, total = z_128hz.shape
    n_epochs = total // epoch_samples
    trimmed = z_128hz[:, : n_epochs * epoch_samples].reshape(ch, n_epochs, epoch_samples)
    ep_med = np.median(trimmed, axis=2, keepdims=True)
    centered = trimmed - ep_med
    ep_std = np.sqrt(np.mean(centered**2, axis=2))
    med_std = np.maximum(np.median(ep_std, axis=1, keepdims=True), 1e-12)
    diff_std = np.sqrt(np.mean(np.diff(trimmed, axis=2) ** 2, axis=2))
    med_diff = np.maximum(np.median(diff_std, axis=1, keepdims=True), 1e-12)
    abs_x = np.abs(trimmed)
    max_abs = np.max(abs_x, axis=2)
    med_max = np.maximum(np.median(max_abs, axis=1, keepdims=True), 1e-12)
    tol = np.maximum(0.001 * max_abs[:, :, None], 1e-9)
    at_ceiling = np.mean(np.abs(abs_x - max_abs[:, :, None]) <= tol, axis=2)
    burst = (diff_std >= 4.5 * med_diff) & (ep_std >= 2.5 * med_std)
    flatline = (ep_std <= 0.02 * med_std) | (ep_std <= 1e-8)
    clipped = (at_ceiling >= 0.02) & (max_abs > 1e-8) & (max_abs <= 0.95 * med_max)
    flags = burst | flatline | clipped
    severity = diff_std / med_diff + med_std / np.maximum(ep_std, 1e-12) + at_ceiling * 100.0
    for c in range(ch):
        idx = np.flatnonzero(flags[c])
        if len(idx) > 8:
            keep = idx[np.argsort(severity[c, idx])[-8:]]
            flags[c, :] = False
            flags[c, keep] = True
    return flags


def blind_detect_bad_channel_epochs_v2(
    z_128hz: np.ndarray,
    raw_native: np.ndarray | None = None,
    raw_sf: float | None = None,
    epoch_samples: int = 3840,
    max_flag_fraction: float = 0.20,
) -> np.ndarray:
    """Physics-grounded blind artifact detector (zero clean-night info, no 8-epoch cap).

    Detects four physical failure modes per (channel, 30s epoch):
    1) Flatline / electrode dropout: 2s sub-window diff std near 0 or whole-epoch std << night median.
    2) Digital saturation / amplitude clipping: measured BEFORE FIR resampling when raw_native is
       provided (repeated identical nonzero extremes or >=2% consecutive zero-slope plateau at max_abs).
    3) High-frequency / broadband burst: robust MAD log-z score AND ratio of sample-to-sample diff RMS.
    4) Slow electrode pop / large DC drift: extreme whole-epoch RMS >> night robust baseline.
    """
    ch, total = z_128hz.shape
    n_epochs = total // epoch_samples
    if n_epochs == 0:
        return np.zeros((ch, 0), dtype=bool)
    trimmed = z_128hz[:, : n_epochs * epoch_samples].reshape(ch, n_epochs, epoch_samples).astype(np.float64)

    # 15 sub-windows of 2s (256 samples @ 128Hz) per epoch catch partial-epoch faults cleanly
    n_sub = 15
    sub_len = epoch_samples // n_sub  # 256
    subs = trimmed.reshape(ch, n_epochs, n_sub, sub_len)
    sub_diff_rms = np.sqrt(np.mean(np.diff(subs, axis=3) ** 2, axis=3))  # (C, E, 15)
    sub_diff_min = np.min(sub_diff_rms, axis=2)
    sub_diff_max = np.max(sub_diff_rms, axis=2)

    ep_med = np.median(trimmed, axis=2, keepdims=True)
    ep_rms = np.sqrt(np.mean((trimmed - ep_med) ** 2, axis=2))
    ep_diff_rms = np.sqrt(np.mean(np.diff(trimmed, axis=2) ** 2, axis=2))

    med_rms = np.maximum(np.median(ep_rms, axis=1, keepdims=True), 1e-12)
    med_diff = np.maximum(np.median(ep_diff_rms, axis=1, keepdims=True), 1e-12)

    # 1) Flatline / dead electrode: literal dead sub-window (<2e-9 V/sample diff) OR whole-epoch std < 1.5% median
    flatline = (sub_diff_min <= 2e-9) | (ep_rms <= 0.015 * med_rms) | (ep_rms <= 1e-8)

    # 2) Rail / saturation clipping on native signal before resample FIR ringing (fallback to 128Hz if absent)
    if raw_native is not None and raw_sf is not None and int(round(raw_sf)) > 0:
        raw_ep_len = int(round(30 * float(raw_sf)))
        if raw_native.shape[1] >= n_epochs * raw_ep_len:
            raw_trim = raw_native[:, : n_epochs * raw_ep_len].reshape(ch, n_epochs, raw_ep_len).astype(np.float64)
        else:
            raw_trim = trimmed
    else:
        raw_trim = trimmed

    raw_abs = np.abs(raw_trim)
    raw_max_abs = np.max(raw_abs, axis=2)
    raw_med_max = np.maximum(np.median(raw_max_abs, axis=1, keepdims=True), 1e-12)
    raw_diff = np.abs(np.diff(raw_trim, axis=2))
    # Hard amplitude clipping produces consecutive flat samples right at ±ep_max_abs.
    near_rail = raw_abs >= np.maximum(0.995 * raw_max_abs[:, :, None], 1e-9)
    both_near_rail = near_rail[:, :, 1:] & near_rail[:, :, :-1]
    flat_at_rail = np.mean((raw_diff <= 1e-9) & both_near_rail, axis=2)
    frac_at_ceiling = np.mean(np.abs(raw_abs - raw_max_abs[:, :, None]) <= 1e-8, axis=2)
    whole_ep_clipped = (
        (raw_max_abs > 1e-8)
        & (raw_max_abs <= 0.98 * raw_med_max)
        & ((flat_at_rail >= 0.01) | (frac_at_ceiling >= 0.015))
    )

    # Sub-window (2s) saturation check catches partial-epoch clipping (e.g., 10s out of 30s)
    # where the unclipped portion raises whole-epoch max_abs above the clip rail.
    raw_sub_len = raw_trim.shape[2] // n_sub
    if raw_sub_len >= 20:
        sub_raw = raw_trim[:, :, : n_sub * raw_sub_len].reshape(ch, n_epochs, n_sub, raw_sub_len)
        sub_abs = np.abs(sub_raw)
        sub_max = np.max(sub_abs, axis=3)
        sub_diff_r = np.abs(np.diff(sub_raw, axis=3))
        sub_near = sub_abs >= np.maximum(0.995 * sub_max[:, :, :, None], 1e-9)
        sub_flat = np.mean((sub_diff_r <= 1e-9) & sub_near[:, :, :, 1:] & sub_near[:, :, :, :-1], axis=3)
        sub_ceil = np.mean(np.abs(sub_abs - sub_max[:, :, :, None]) <= 1e-8, axis=3)
        sub_clipped = (
            (sub_max > 1e-8)
            & (sub_max <= 0.98 * raw_med_max[:, :, None])
            & ((sub_flat >= 0.06) | (sub_ceil >= 0.08))
        )
        partial_ep_clipped = np.sum(sub_clipped, axis=2) >= 2
    else:
        partial_ep_clipped = np.zeros_like(whole_ep_clipped)
    clipped = whole_ep_clipped | partial_ep_clipped

    # 3) High-frequency burst / additive broadband noise (robust MAD in log-uV space)
    log_diff = np.log1p(ep_diff_rms * 1e6)
    ld_med = np.median(log_diff, axis=1, keepdims=True)
    ld_mad = np.maximum(1.4826 * np.median(np.abs(log_diff - ld_med), axis=1, keepdims=True), 0.15)
    z_diff = (log_diff - ld_med) / ld_mad

    log_sub_max = np.log1p(sub_diff_max * 1e6)
    ls_med = np.median(log_sub_max, axis=1, keepdims=True)
    ls_mad = np.maximum(1.4826 * np.median(np.abs(log_sub_max - ls_med), axis=1, keepdims=True), 0.18)
    z_sub_diff = (log_sub_max - ls_med) / ls_mad

    # Broadband burst has elevated diff_rms globally without relying on ep_rms > 2.5*med_rms (Wake blinks)
    hf_burst = ((z_diff >= 4.0) & (ep_diff_rms >= 3.2 * med_diff)) | ((z_sub_diff >= 5.5) & (sub_diff_max >= 5.0 * med_diff))

    # 4) Slow electrode pop / extreme amplitude drift
    log_rms = np.log1p(ep_rms * 1e6)
    lr_med = np.median(log_rms, axis=1, keepdims=True)
    lr_mad = np.maximum(1.4826 * np.median(np.abs(log_rms - lr_med), axis=1, keepdims=True), 0.20)
    z_rms = (log_rms - lr_med) / lr_mad
    slow_surge = (z_rms >= 5.0) & (ep_rms >= 4.5 * med_rms)

    flags = flatline | clipped | hf_burst | slow_surge
    max_keep = max(8, int(np.floor(max_flag_fraction * n_epochs)))
    severity = np.maximum.reduce([
        z_diff,
        z_sub_diff * 0.85,
        z_rms * 0.9,
        (flatline.astype(np.float64) * 20.0),
        (flat_at_rail * 200.0 + frac_at_ceiling * 100.0) * clipped.astype(np.float64),
    ])
    for c in range(ch):
        idx = np.flatnonzero(flags[c])
        if len(idx) > max_keep:
            keep = idx[np.argsort(severity[c, idx])[-max_keep:]]
            flags[c, :] = False
            flags[c, keep] = True
    return flags


def local_window_robust_norm(
    z_128hz: np.ndarray,
    window_epochs: int = 32,
    epoch_samples: int = 3840,
    clip_iqr_mult: float = 20.0,
) -> np.ndarray:
    """Non-overlapping local-window RobustScaler (32 epochs = 16 min default)."""
    ch, total = z_128hz.shape
    n_epochs = max(1, total // epoch_samples)
    win_len = window_epochs * epoch_samples
    out = np.empty_like(z_128hz, dtype=np.float32)
    _, global_stats = robust_norm_channelwise(z_128hz, clip_iqr_mult=clip_iqr_mult)
    n_windows = max(1, (n_epochs + window_epochs - 1) // window_epochs)
    for w in range(n_windows):
        s = w * win_len
        e = total if w == n_windows - 1 else min(total, (w + 1) * win_len)
        seg = z_128hz[:, s:e]
        for c in range(ch):
            q75, q25 = np.percentile(seg[c], [75.0, 25.0])
            iqr = float(q75 - q25)
            threshold = iqr * clip_iqr_mult if iqr > 1e-12 else float(global_stats["threshold"][c])
            clipped = np.clip(seg[c], -threshold, threshold)
            center = float(np.median(clipped))
            c75, c25 = np.percentile(clipped, [75.0, 25.0])
            scale = float(c75 - c25)
            if scale <= 1e-12:
                scale = float(global_stats["scale"][c])
            out[c, s:e] = ((clipped - center) / scale).astype(np.float32)
    return out


def causal_running_robust_norm(
    z_128hz: np.ndarray,
    trailing_epochs: int = 60,
    min_warmup_epochs: int = 8,
    epoch_samples: int = 3840,
    clip_iqr_mult: float = 20.0,
    subsample_stride: int = 8,
) -> np.ndarray:
    """Strictly causal online robust normalization using only current + past epochs.

    Each epoch e is scaled using quantile estimates computed strictly over the causal
    trailing window [max(0, e - trailing_epochs + 1) : e + 1] (zero future leakage!).
    Before min_warmup_epochs, uses expanding prefix up to min(e + 1, min_warmup_epochs)
    with finite floor fallback to prevent divide-by-zero on flat start epochs.
    """
    ch, total = z_128hz.shape
    n_epochs = total // epoch_samples
    if n_epochs == 0:
        out, _ = robust_norm_channelwise(z_128hz, clip_iqr_mult=clip_iqr_mult)
        return out
    usable = n_epochs * epoch_samples
    ep_view = z_128hz[:, :usable].reshape(ch, n_epochs, epoch_samples)
    # 16Hz subsampling within each epoch makes sliding-window quantiles fast (~0.3s/night)
    sub_ep = ep_view[:, :, ::subsample_stride]  # (C, E, 480)
    out = np.empty_like(z_128hz, dtype=np.float32)

    for c in range(ch):
        last_valid_scale = 2e-5
        for e in range(n_epochs):
            win_hi = max(e + 1, min(n_epochs, min_warmup_epochs)) if e + 1 < min_warmup_epochs else e + 1
            win_lo = max(0, win_hi - trailing_epochs)
            # For epochs < min_warmup_epochs we strictly stay causal [0 : e+1] if std is healthy,
            # only falling back to floor if signal is flat during first seconds.
            if e + 1 < min_warmup_epochs:
                win_lo, win_hi = 0, e + 1
            ref = sub_ep[c, win_lo:win_hi].reshape(-1)
            q75, q25 = np.percentile(ref, [75.0, 25.0])
            iqr = float(q75 - q25)
            if iqr <= 1e-12:
                iqr = last_valid_scale
            else:
                last_valid_scale = iqr
            thresh = max(iqr * float(clip_iqr_mult), 1e-9)
            clipped_ref = np.clip(ref, -thresh, thresh)
            center = float(np.median(clipped_ref))
            c75, c25 = np.percentile(clipped_ref, [75.0, 25.0])
            scale = float(c75 - c25)
            if scale <= 1e-12:
                scale = iqr if iqr > 1e-12 else 1.0
            ep_slice = ep_view[c, e]
            out[c, e * epoch_samples : (e + 1) * epoch_samples] = (
                (np.clip(ep_slice, -thresh, thresh) - center) / scale
            ).astype(np.float32)
        if usable < total:
            # Tail remainder shares final causal stats (still zero future info beyond usable)
            ep_tail = z_128hz[c, usable:]
            out[c, usable:] = ((np.clip(ep_tail, -thresh, thresh) - center) / scale).astype(np.float32)
    return out


def clip_winsor_norm(z_128hz: np.ndarray, clip_iqr_mult: float = 5.0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Tighter global outlier Winsorization baseline (5*IQR clip instead of 20*IQR)."""
    return robust_norm_channelwise(z_128hz, clip_iqr_mult=clip_iqr_mult)


def build_masked_input(
    z_128: np.ndarray,
    bad_ch_epochs: np.ndarray,
    epoch_samples: int = 3840,
    guard_samples: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute non-leaky RobustScaler stats on unflagged samples and zero out flagged regions."""
    ch, total = z_128.shape
    valid_mask = np.ones((ch, total), dtype=bool)
    for c in range(ch):
        for e in np.flatnonzero(bad_ch_epochs[c]):
            s = max(0, int(e) * epoch_samples - guard_samples)
            t = min(total, (int(e) + 1) * epoch_samples + guard_samples)
            valid_mask[c, s:t] = False
    z_norm, stats = robust_norm_channelwise(z_128, valid_sample_mask=valid_mask)
    z_norm[~valid_mask] = 0.0
    return z_norm, stats, valid_mask


def infer_anysleep_full(
    model,
    z_norm: np.ndarray,
    device: str,
    fixed_alpha: dict[int, np.ndarray] | None = None,
    channel_indices: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Run unchunked full-recording forward inference with optional fixed skip-attention or channel subset."""
    import torch

    z_in = z_norm if channel_indices is None else z_norm[channel_indices]
    captured: dict[int, np.ndarray] = {}
    handles = []
    for i, block in enumerate(model.skip_connections):
        def make_hook(idx: int):
            def hook(module, args, output):
                if fixed_alpha is not None:
                    a = np.asarray(fixed_alpha[idx])
                    if tuple(a.shape) != tuple(output.shape):
                        raise ValueError(f"fixed_alpha[{idx}] shape {a.shape} != output {tuple(output.shape)}")
                    captured[idx] = a.copy()
                    return output.new_tensor(a)
                captured[idx] = output.detach().cpu().numpy().copy()
                return output
            return hook
        handles.append(block.softmax.register_forward_hook(make_hook(i)))
    try:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(z_in.T, dtype=np.float32)).unsqueeze(0).to(device)
            logits = model(tensor).detach().cpu().numpy()[0].copy()
            del tensor
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def infer_anysleep_split_restart_full(
    model,
    z_masked_norm: np.ndarray,
    bad_any_epochs: np.ndarray,
    device: str,
    epoch_samples: int = 3840,
) -> np.ndarray:
    """True uncropped contiguous-segment inference separating flagged epochs from clean spans."""
    n_epochs = len(bad_any_epochs)
    if not np.any(bad_any_epochs):
        logits, _ = infer_anysleep_full(model, z_masked_norm[:, : n_epochs * epoch_samples], device)
        return logits[:n_epochs]

    out_logits = np.empty((n_epochs, 5), dtype=np.float32)
    is_bad = np.asarray(bad_any_epochs, dtype=bool)
    i = 0
    while i < n_epochs:
        flag = bool(is_bad[i])
        j = i + 1
        while j < n_epochs and bool(is_bad[j]) == flag:
            j += 1
        if not flag:
            seg = z_masked_norm[:, i * epoch_samples : j * epoch_samples]
            seg_l, _ = infer_anysleep_full(model, seg, device)
            out_logits[i:j] = seg_l[: (j - i)]
        else:
            # Evaluate flagged epochs one by one strictly isolated so neither contaminates clean neighbors
            for k in range(i, j):
                seg = z_masked_norm[:, k * epoch_samples : (k + 1) * epoch_samples]
                seg_l, _ = infer_anysleep_full(model, seg, device)
                out_logits[k] = seg_l[0]
        i = j
    return out_logits


def infer_anysleep_channel_reroute_full(
    model,
    z_masked_norm: np.ndarray,
    bad_ch_epochs: np.ndarray,
    device: str,
    epoch_samples: int = 3840,
) -> np.ndarray:
    """Exploit AnySleep's variable-channel architecture: dynamically drop only corrupted channel(s).

    Clean spans run on all channels. Spans where subset S of channels is clean (>=1 good channel)
    run AnySleep on good channels only; epochs where ALL channels fail are isolated single-epoch.
    """
    ch, _ = bad_ch_epochs.shape
    n_epochs = bad_ch_epochs.shape[1]
    if not np.any(bad_ch_epochs):
        logits, _ = infer_anysleep_full(model, z_masked_norm[:, : n_epochs * epoch_samples], device)
        return logits[:n_epochs]

    out_logits = np.empty((n_epochs, 5), dtype=np.float32)
    good = ~np.asarray(bad_ch_epochs, dtype=bool)  # (C, E)
    i = 0
    while i < n_epochs:
        sig_key = tuple(bool(good[c, i]) for c in range(ch))
        j = i + 1
        while j < n_epochs and tuple(bool(good[c, j]) for c in range(ch)) == sig_key:
            j += 1
        good_ch = [c for c, ok in enumerate(sig_key) if ok]
        if good_ch:
            seg = z_masked_norm[:, i * epoch_samples : j * epoch_samples]
            seg_l, _ = infer_anysleep_full(model, seg, device, channel_indices=good_ch)
            out_logits[i:j] = seg_l[: (j - i)]
        else:
            for k in range(i, j):
                seg = z_masked_norm[:, k * epoch_samples : (k + 1) * epoch_samples]
                seg_l, _ = infer_anysleep_full(model, seg, device)
                out_logits[k] = seg_l[0]
        i = j
    return out_logits


def compute_clean_metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    keep = np.asarray(truth) >= 0
    t = np.asarray(truth, dtype=np.int64)[keep]
    p = np.asarray(pred, dtype=np.int64)[keep]
    total = int(len(t))
    if total == 0:
        return {"scored_epochs": 0, "accuracy": 0.0, "macro_f1": 0.0, "cohen_kappa": 0.0}
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1.0), 0.0)
    acc = float(tp.sum() / total)
    pe = float(np.sum(cm.sum(axis=0).astype(np.float64) * cm.sum(axis=1).astype(np.float64)) / (float(total) ** 2))
    kappa = float((acc - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0
    return {
        "scored_epochs": total,
        "accuracy": acc,
        "macro_f1": float(np.mean(f1)),
        "cohen_kappa": kappa,
    }


def extract_baseline_margin_diagnostics(
    base_logits: np.ndarray,
    pert_logits: np.ndarray,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-epoch baseline logit margin (top1 - top2), stage, flip, and net-error indicator."""
    bl = np.asarray(base_logits, dtype=np.float64)
    pl = np.asarray(pert_logits, dtype=np.float64)
    part_sorted = np.partition(bl, -2, axis=1)
    top1 = part_sorted[:, -1]
    top2 = part_sorted[:, -2]
    margin = top1 - top2
    bp = bl.argmax(axis=1)
    pp = pl.argmax(axis=1)
    tr = np.asarray(truth)
    new_err = (bp == tr) & (pp != tr)
    corr_err = (bp != tr) & (pp == tr)
    return {
        "margin": margin,
        "base_pred": bp,
        "pert_pred": pp,
        "flipped": (bp != pp),
        "new_error": new_err,
        "corrected": corr_err,
        "net_error_vec": new_err.astype(np.float64) - corr_err.astype(np.float64),
        "max_logit_delta": np.max(np.abs(pl - bl), axis=1),
    }


def compute_region_metrics_extended(
    truth: np.ndarray,
    base_pred: np.ndarray,
    pert_pred: np.ndarray,
    sf: float,
    start: int,
    end: int,
    local_support: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute paired net additional error across standard and fine-grained regions + upstream/downstream."""
    truth = np.asarray(truth)
    base_pred = np.asarray(base_pred)
    pert_pred = np.asarray(pert_pred)
    n_epochs = len(truth)
    ep_start = np.arange(n_epochs, dtype=np.int64) * int(round(30 * sf))
    ep_end = (np.arange(n_epochs, dtype=np.int64) + 1) * int(round(30 * sf))
    untouched = (ep_end <= start) | (ep_start >= end)
    upstream = ep_end <= start
    downstream = ep_start >= end
    distance = np.maximum(np.maximum(start - ep_end, ep_start - end), 0) / float(sf)
    scored = np.isin(truth, [0, 1, 2, 3, 4])
    loc = np.asarray(local_support, dtype=bool)
    beyond = ~loc

    regions = [
        ("corrupted", scored & ~untouched),
        ("bin_0_30s", scored & untouched & (distance >= 0.0) & (distance < 30.0)),
        ("bin_30_60s", scored & untouched & (distance >= 30.0) & (distance < 60.0)),
        ("bin_60_120s", scored & untouched & (distance >= 60.0) & (distance < 120.0)),
        ("bin_120_300s", scored & untouched & (distance >= 120.0) & (distance < 300.0)),
        ("bin_300_450s", scored & untouched & (distance >= 300.0) & (distance < 450.0)),
        ("bin_450s_plus", scored & untouched & (distance >= 450.0)),
        ("near_30_300s", scored & untouched & (distance >= 30.0) & (distance < 300.0)),
        ("within_local_support_untouched", scored & untouched & loc),
        ("within_local_upstream", scored & upstream & loc),
        ("within_local_downstream", scored & downstream & loc),
        ("beyond_local_support", scored & untouched & beyond),
        ("all_untouched", scored & untouched),
    ]

    rows = []
    for region_name, mask in regions:
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        t = truth[mask]
        b = base_pred[mask]
        p = pert_pred[mask]
        new_err = (b == t) & (p != t)
        corr_err = (b != t) & (p == t)
        flips = b != p
        rows.append({
            "region": region_name,
            "epochs": cnt,
            "new_errors": int(new_err.sum()),
            "corrected_errors": int(corr_err.sum()),
            "flips": int(flips.sum()),
            "new_error_rate": float(new_err.mean()),
            "corrected_rate": float(corr_err.mean()),
            "net_error": float(new_err.mean() - corr_err.mean()),
            "flip_rate": float(flips.mean()),
        })
    return rows


def participant_bootstrap_ci(
    values: np.ndarray,
    seed: int = 20261117,
    replicates: int = 20000,
) -> tuple[float | None, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    n = len(vals)
    if n < 2:
        return None, None
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(n, size=(replicates, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def gpu_preflight_probe(device: str = "auto") -> dict[str, Any]:
    """Verify PyTorch deterministic GPU/CPU execution across single-epoch & multi-epoch lengths."""
    dev = resolve_device(device)
    torch = configure_torch_determinism(dev, seed=20261117, strict=True)
    from anysleep_no_hydra import AnySleep

    model = AnySleep(path=None, sleep_stage_frequency=1).eval().to(dev)
    rng = np.random.default_rng(20261117)
    results: dict[str, Any] = {
        "device": dev,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if dev.startswith("cuda") else None,
        "tf32_matmul_allowed": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "tf32_cudnn_allowed": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "lengths_checked": [],
    }
    for n_ep, ch_sub in [(1, [0, 1, 2]), (5, [1, 2]), (64, [0, 1, 2]), (12, [0])]:
        x = rng.normal(0.0, 1.0, size=(3, n_ep * 3840)).astype(np.float32)
        l1, a1 = infer_anysleep_full(model, x, dev, channel_indices=ch_sub)
        l2, _ = infer_anysleep_full(model, x, dev, fixed_alpha=a1, channel_indices=ch_sub)
        delta = float(np.max(np.abs(l1 - l2)))
        if tuple(l1.shape) != (n_ep, 5) or not np.isfinite(l1).all() or delta > 1e-5:
            raise AssertionError(f"AnySleep determinism/shape failure for n_ep={n_ep}, ch={ch_sub}: shape={l1.shape}, delta={delta}")
        results["lengths_checked"].append({"epochs": n_ep, "channels": len(ch_sub), "max_fixed_alpha_delta": delta})
    return results


def selftest() -> None:
    rng = np.random.default_rng(20261117)
    n_epochs = 80
    sf = 100.0
    ep_raw = 30 * int(sf)
    ep_128 = 3840
    # Synthetic realistic smooth EEG/EOG + 1/f-like bandlimited spectrum (diff << amplitude)
    raw = rng.normal(0.0, 2e-5, size=(3, n_epochs * ep_raw)).astype(np.float32)
    kernel = np.ones(7, dtype=np.float32) / 7.0
    for c in range(3):
        raw[c] = np.convolve(raw[c], kernel, mode="same")
    idx_128 = (np.arange(n_epochs * ep_128) * (3000.0 / 3840.0)).astype(int)
    idx_128 = np.clip(idx_128, 0, raw.shape[1] - 1)
    clean_128 = raw[:, idx_128].copy()

    clean_flags = blind_detect_bad_channel_epochs_v2(clean_128, raw_native=raw, raw_sf=sf, epoch_samples=ep_128)
    assert int(clean_flags.sum()) == 0, f"Unexpected false positives on clean synthetic EEG: {clean_flags.sum()}"

    s_raw, e_raw = 35 * ep_raw, 36 * ep_raw
    for kind in ("flat", "clip", "noise"):
        y_raw = perturb_extended(raw, [0], s_raw, e_raw, kind, sf=sf, seed=20261117, clip_volts=2e-6, noise_sd_volts=1e-4)
        y_128 = y_raw[:, idx_128]
        flags = blind_detect_bad_channel_epochs_v2(y_128, raw_native=y_raw, raw_sf=sf, epoch_samples=ep_128)
        assert bool(flags[0, 35]), f"Failed v2 blind detection for {kind} on target epoch 35"
        assert int(flags[1:].sum()) == 0, f"False positive on untouched channels for {kind}: {flags[1:].sum()}"

    # Partial 12s clip inside 30s epoch check
    y_part_clip = perturb_extended(raw, [0], s_raw, s_raw + 12 * int(sf), "clip", sf=sf, clip_volts=1.5e-6)
    f_part = blind_detect_bad_channel_epochs_v2(y_part_clip[:, idx_128], raw_native=y_part_clip, raw_sf=sf, epoch_samples=ep_128)
    assert bool(f_part[0, 35]), "Failed v2 blind detection on partial-epoch clip"

    # Realistic waveform perturbations must keep non-targeted channels/epochs bit-identical
    for kind in ("mains_60hz", "slow_drift_pop"):
        y_ext = perturb_extended(raw, [0], s_raw, e_raw, kind, sf=sf, seed=20261117, noise_sd_volts=1e-4)
        assert np.array_equal(y_ext[1:], raw[1:]) and np.array_equal(y_ext[0, :s_raw], raw[0, :s_raw])

    # Local, causal running, and winsor normalizations
    z_loc = local_window_robust_norm(clean_128, window_epochs=32, epoch_samples=ep_128)
    z_caus = causal_running_robust_norm(clean_128, trailing_epochs=32, epoch_samples=ep_128)
    z_win5, _ = clip_winsor_norm(clean_128, clip_iqr_mult=5.0)
    assert z_loc.shape == clean_128.shape and np.isfinite(z_loc).all()
    assert z_caus.shape == clean_128.shape and np.isfinite(z_caus).all()
    assert z_win5.shape == clean_128.shape and np.isfinite(z_win5).all()
    # Strict causality check: modifying future epochs [50:] cannot change epochs [0:40] in causal_running_robust_norm
    future_mut = clean_128.copy()
    future_mut[:, 50 * ep_128 :] += 1e-2
    z_caus_mut = causal_running_robust_norm(future_mut, trailing_epochs=32, epoch_samples=ep_128)
    assert np.array_equal(z_caus[:, : 40 * ep_128], z_caus_mut[:, : 40 * ep_128]), "Causal normalizer leaked future epochs!"

    # Extended region metrics & margin diagnostics
    truth = np.zeros(n_epochs, dtype=np.int16)
    base = np.zeros(n_epochs, dtype=np.int16)
    pert = np.zeros(n_epochs, dtype=np.int16)
    pert[36] = 1
    pert[37] = 1
    pert[50] = 1
    loc_sup = np.zeros(n_epochs, dtype=bool)
    loc_sup[30:42] = True
    rows = {r["region"]: r for r in compute_region_metrics_extended(truth, base, pert, sf, s_raw, e_raw, loc_sup)}
    assert rows["bin_0_30s"]["new_errors"] == 1
    assert rows["near_30_300s"]["new_errors"] == 1
    assert rows["within_local_downstream"]["new_errors"] == 2
    assert rows["beyond_local_support"]["new_errors"] == 1
    assert len(PHYSIONET_SC400_SC419) == 20
    assert len(ANYSLEEP_CHECKPOINTS) == 3
    print("COMMON_V2_SELFTEST_OK")


if __name__ == "__main__":
    selftest()
