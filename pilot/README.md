# First offline pilot

This directory is an executable development harness, not evidence that the
scientific phenomenon exists. The existing `paired_tools.py` was preserved.

The main script uses the complete named PSG EDF as the inference unit, including
any extended wake recording. No time crop, sliding inference or chunking is used.
One source participant, one position and one duration cannot justify inference
about a population or a go decision for a new method.

Use an isolated Python environment with the appropriate PyTorch build installed.
YASA 0.8.1 requires pandas >=2.3. In Colab, do not force its base environment back
to pandas 2.2.3 to satisfy Colab's own package constraint: use a separate virtual
environment/kernel, or the independent GitHub Actions environment. The prepared
notebook's dependency cell must target that isolated environment before reuse.
An earlier Colab run executed with pandas 2.2.3 despite this metadata conflict;
its numerical results require confirmation in the compatible cloud environment.

From the isolated environment:

```text
python -m pip install -r requirements-pilot.txt
python pilot/run_pilot.py prepare
python pilot/run_pilot.py run --model both --phase A --device cuda --run-id stage-a
python pilot/run_pilot.py run --model both --phase B --device cuda --run-id stage-b
```

Use `--device cpu` on a CPU runtime. Each run directory must be new. Failures are
saved to `results/<run-id>/status.json` and raised; out-of-memory is not handled by
silently changing the scientific inference unit. Long inference runs retain all
official global channel attention, while the oracle mechanism diagnostic below
explicitly intervenes on those computations:

```text
python pilot/run_pilot.py run --model anysleep --phase B --diagnostics --device cuda --run-id stage-c-oracle
```

`--diagnostics` adds fixed original clipping/scaling statistics, fixed original
attention coefficients in all 13 blocks simultaneously, and both interventions.
These use inaccessible original clean-input quantities and are diagnostic
oracles. They are not deployable baselines. Per-layer effective attention is
saved for every inference. Selective single-layer freezing, single-channel
experiments and simple mitigation baselines remain to be run later.

The JSON config fixes source, participant, channels, corruption position rule,
30 s duration, 20 microvolt clipping, additive 100 microvolt white Gaussian noise,
and RNG seed before predictions. It selects the middle *scored* epoch by ordinal
index using reference annotations, never model correctness. This conditional
sampling must be retained when expanding the protocol. The unchanged raw values
outside the chosen channel and interval are checked exactly before filtering.

AnySleep labels: 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM. R&K stage 4 merges into N3.
Unknown, movement, annotation gaps and partly covered epochs are unscored (-1).
Absolute EDF annotation origin is compared with PSG measurement time. If MNE
omits that origin, both EDF date/time headers are read and strictly parsed;
identical nonmissing headers permit zero offset with literal evidence saved.
Differing or malformed headers cause failure.

YASA receives the original volt-valued MNE input, with correct EEG/EOG types,
without AnySleep preprocessing. It internally converts to microvolts. Fpz-Cz is
a nonpreferred derivation relative to the recommended central lead; results are
an applicability check. YASA also has temporal smoothing and global feature
normalization, so it is an independent pipeline, not a no-context control.

Every completed inference saves probabilities/logits, effective attention (for
AnySleep), timings and resident memory after inference; CUDA runs additionally
record peak allocated/reserved device memory. Baseline scores, repeat consistency,
local raw-resampling footprint, paired transitions, unmodified-region harm,
fixed distance bins and a distance plot are written incrementally. Host peak RAM
is not sampled continuously and must not be called peak RAM.

The structural AnySleep support includes convolution, odd left padding, pooling,
upsampling, cropping and final epoch pooling, under eval BatchNorm and fixed
global quantities. It is a Boolean dependency support rather than an observed
effective receptive field. Far analysis excludes every structurally reachable
output epoch after direct resampling. The generic `300s+` bin is not itself a
far-support category. No exact local-support claim is currently made for YASA.

Setup bounds are in `requirements-pilot.txt`. Each actual execution exports
`pip-freeze.txt` as its resolved environment. Vendored AnySleep source and its
Git blobs are pinned and checked at runtime. The downloaded checkpoint must
match its pinned Git blob and all downloaded files get SHA256 checksums.

Pending after smoke validation: freeze expanded participants/positions/durations,
audit pretraining overlap, use participant-level summaries/uncertainty, CAP
loader and replication, robust-normalization alternatives, segment exclusion,
quality gates and matched augmentation. The harness does not silently treat
these pending scientific comparisons as completed.
