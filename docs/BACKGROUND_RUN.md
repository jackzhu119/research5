# Unattended offline run

The `First offline EEG pilot` GitHub Actions workflow is triggered by a push to
`main` or by **Actions → First offline EEG pilot → Run workflow**. Once GitHub
accepts the run, the personal computer can disconnect or shut down: execution
uses a GitHub-hosted machine and neither Colab browser nor local OAuth is needed.

This workflow performs inference only. It downloads the two predeclared
Sleep-EDF files and one pinned AnySleep checkpoint, runs both pipelines on the
complete EDF, checks repeated predictions, then runs paired no-op/flat/clipping/
noise cases. AnySleep additionally runs the oracle fixed-statistic,
fixed-channel-weight and combined diagnostics; all 13 channel-weight blocks are
frozen simultaneously in the relevant diagnostic, not claimed as 13 independent
experiments. No participant expansion or new-method training is automatic.

The job is restricted to public repositories before runner allocation and uses
only standard `ubuntu-latest`. The [official runner
reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
checked 2026-11-17, lists 4 CPUs and 16 GB for public repositories and states
standard public-repository compute is free. This configuration does not select
larger runners, enable billing or buy any plan. Private forks are skipped. Actual
CPU and available RAM are recorded; a machine with less than 10 GiB free RAM
fails instead of changing the full-record protocol.

The outer job timeout is 100 minutes. The controller uses an 80-minute internal
budget after installation, allowing time to save failure status and upload
partial results. An exceptionally slow installation or infrastructure failure
can still interrupt cleanup; preservation after an externally terminated runner
cannot be guaranteed. The controller samples process-tree RAM once per second,
so its reported maximum is a sampled resident-memory maximum, not an exact
instantaneous hardware peak.

Each run uploads an artifact named `offline-pilot-<run-id>-<attempt>`, whether the
pilot succeeds or fails. It includes completed/partial result arrays, paired
tables, logs, environment freezes, statuses, protocol and provenance. The
configured retention is 90 days, subject to repository policy. Download the
artifact before expiry and preserve it in the local research5 results archive;
an Actions artifact is not permanent storage. The Actions run summary reports
stage statuses and only numerical outputs actually written by executed code.

`requirements-cloud.txt` fixes the scientific packages, including pandas 3.1.1,
in a separate environment satisfying YASA 0.8.1's pandas >=2.3 requirement.
The earlier Colab execution with pandas 2.2.3 had a dependency-metadata conflict
and needs compatible-environment confirmation. CPU PyTorch 2.14.1 is installed from the official
PyTorch CPU index. The installed resolution is saved again per execution.
Actions are pinned to verified upstream v7 commit identities as of 2026-11-17.

If a run fails, inspect its summary, `controller_status.json`, command logs and
the last stage's `status.json`. Do not count a partial stage as passed. If stage A
fails, stage B does not start. A successful smoke run is engineering validation,
not evidence that unmodified-epoch harm has practical magnitude or that a new
model should be trained.
