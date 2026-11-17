# Source and applicability audit

Audit date: 2026-11-17. These are source-level observations. They do not establish
new errors on unchanged EEG or a practical weakness in either model.

## AnySleep

Pinned source: [dslaborg/AnySleep at
88e13e31b3fa02482b650a844f6665f2cbbdb82c](https://github.com/dslaborg/AnySleep/tree/88e13e31b3fa02482b650a844f6665f2cbbdb82c).
The recursive tree at that commit contained no AGENTS.md. Only the necessary
standalone source, official preprocessing example, main architecture source and
MIT license are vendored; no untracked upstream file is assumed available.
`manifests/anysleep_sources.json` contains SHA256 and original Git blob identities.
All four vendored files match the original Git blobs byte-for-byte.

The official [plain EDF
example](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/examples/predict_edf_file_logits_plain.py)
reads EEG/EOG as MNE volt-valued data, uses `resample_poly` to 128 Hz, clips each
channel symmetrically at 20 times its whole-record IQR, then fits RobustScaler
(median and IQR) on each complete clipped channel. The median is not the center
of the preceding clipping interval. The example forwards the complete recording
in one batch. Its model is `models/anysleep-run1.pth`, not an early/late-fusion
checkpoint. The audit verified that this checkpoint is 12,763,675 bytes with
Git blob SHA1 `8e545c4af53a7466b0b9df26a3d554ced47ac12d`; preparation downloads it
from the pinned commit and verifies this identity before loading with
`torch.load(weights_only=True)`.

The [main
architecture](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/base/model/anysleep.py)
and its [standalone
implementation](https://github.com/dslaborg/AnySleep/blob/88e13e31b3fa02482b650a844f6665f2cbbdb82c/examples/anysleep_no_hydra.py)
use time-mean features before an MLP and softmax across input channels. This
happens at every encoder skip and at the connector (13 sites for depth=12).
Those coefficients are reused across the entire feature sequence. A localized
input change may therefore influence remote features by changing global
preprocessing statistics or these global coefficients, in addition to the
convolutional local path. Short changes may be diluted by the whole-record
average and robust quantiles. Source dependence alone says nothing about the
size or sign of classification harm.

All BatchNorm layers must be in eval mode to use their stored statistics. Training
mode would introduce additional dependencies and is outside this pilot. One
channel makes the channel softmax exactly 1, removing this specific pathway;
global preprocessing and convolutional context remain.

## Exact local structural support

The pinned depth=12 topology has 9-sample same convolutions in each encoder,
left padding for odd lengths before factor-2 max pooling, a 9-sample connector,
nearest-neighbor upsampling, a 2-sample same convolution, the official asymmetric
crop, skip concatenation and a 9-sample convolution in each decoder. The final
head uses pointwise convolutions and 3,840-sample average pooling per 30 s epoch.
`pilot/support.py` propagates a Boolean dependency through those exact operations.
It assumes fixed clipping/scaling and attention, plus eval-mode BatchNorm. It is
the theoretical structural support; zero weights or inactive nonlinearities can
make actual effects smaller.

For a purely synthetic 24-hour input-length mask with a central 30 s interval,
the topology reaches output epochs 1426 through 1453 inclusive: the epoch-edge
extent is 420 s before and 390 s after that interval. This is an executed
structural calculation, not a real EEG prediction or accuracy result. Actual
length/position and FIR footprint are recomputed for each intervention. It
follows that neither the proposed 0.5–5 min bins nor `300s+` alone defines a
remote effect outside the local convolutional support.

The AnySleep resampling operation has finite FIR support. The script records
the exact changed sample mask immediately after resampling each raw pair, before
global statistics are fitted. It then propagates that mask through the local
topology and saves the output mask. Numeric effects outside raw modification can
therefore be distinguished from direct resampling spread. With recomputed global
statistics or attention the *total* theoretical support is global.

## YASA independent pipeline

The audited current version is [YASA 0.8.1
SleepStaging](https://yasa-sleep.org/generated/yasa.SleepStaging.html), with its
[tagged implementation](https://github.com/raphaelvallat/yasa/blob/v0.8.1/src/yasa/staging.py).
It takes original MNE Raw data, performs its own processing and conversion to
microvolts, and expects sampling above 80 Hz (internally 100 Hz). The user should
not z-score or prefilter the input. Our EDF channels are explicitly marked EEG
and EOG and enter YASA in volts; AnySleep-scaled data never enters YASA.

YASA prefers a central electrode referenced to mastoid or Fpz. Sleep-EDF's
Fpz-Cz is a different derivation and polarity relative to the recommended central
C4-Fpz configuration. The independent pipeline can be run as an applicability
check, but its performance and the meaning of induced artifacts need separate
assessment. No inversion or re-reference is silently introduced in this pilot.

YASA uses a 0.4–30 Hz FIR filtering call, per-epoch features, a 15-epoch centered
triangular rolling mean and 4-epoch past rolling mean. It robustly scales these
smoothed features over the complete record using 5th/95th percentiles. Therefore
YASA also has both local context and a global statistic dependency. It is
independent of AnySleep's neural architecture; it is not a context-free control.
The exact YASA FIR/rolling structural support has not yet been certified by the
harness. Its `beyond_local_support` field is deliberately false and its far
analysis empty pending that audit, as recorded in `support_note.json`.

YASA's default classifier was trained using National Sleep Research Resource
data according to the [official publication](https://doi.org/10.7554/eLife.70092).
Every installed bundled classifier file is hashed at execution. Dataset-level
provenance does not replace a participant-level overlap check. AnySleep was
trained on multiple public datasets, including Sleep-EDF according to its
paper/config sources; this pilot is an engineering and mechanism development
set, not independent external validation. CAP should only be called a second
source until the exact checkpoint's participant overlap audit is complete.

## Data and labels

The only planned initial download is
[Sleep-EDF Expanded version 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/),
files `sleep-cassette/SC4001E0-PSG.edf` and
`sleep-cassette/SC4001EC-Hypnogram.edf`. The participant key is SC400 (one night
SC4001E0); all epochs/positions from that person remain one statistical unit.
No CAP data are downloaded by the smoke configuration. The second-source target
is [CAP version 1.0.0](https://physionet.org/content/capslpdb/1.0.0/).

The label mapping is Wake=0, N1=1, N2=2, N3=3, REM=4; R&K stage 4 merges into N3.
Movement/unknown/gaps and partly covered epochs are unscored. Measurement and
annotation absolute origins are checked. If MNE omits the annotation origin, the
two explicitly parsed EDF date/time headers must match exactly before assigning
zero offset; their literal values are saved. A differing/malformed origin or
incompatible epoch count is a hard failure. The entire EDF is forwarded to the models. If the EDF
contains prolonged waking time, that remains part of the preprocessing/inference
unit and must be reported when interpreting the smoke result.

## Remaining audit limits

No source-only statement is a finding of net harm. Full-run validation still
requires actual baseline repeatability, no-op, finite corruptions, raw-pair
invariance, resource records and label alignment on data. One participant yields
descriptive transitions, not a confidence interval for participants. Stronger
claims require predeclared development expansion, second-source replication,
simple mitigation baselines, training-overlap review and participant-level
uncertainty. The implementation does not train a new model.
