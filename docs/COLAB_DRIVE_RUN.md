# Colab compute, Google Drive persistence

Use `pilot/COLAB_DRIVE_PILOT.ipynb` after uploading the latest complete code bundle
to the existing `MyDrive/research5/bundles/` folder. The notebook mounts Drive
through Google's ordinary authorization and runs `pilot/colab_drive_run.py`.
The project folder must already exist and `/content/drive` must be an actual
mount; the driver fails otherwise. No token or older callback URL is reused.

All downloaded EDF data, model checkpoints, predictions, reports, logs,
environment records, status and source snapshots are written under
`MyDrive/research5`. Only the execution workspace and virtual environment are
ephemeral `/content` files. Links to the persistent directories are created in
the local execution workspace, so Drive does not need to support symlinks.
Only the one predeclared Sleep-EDF record and one checkpoint are downloaded.

The driver creates an isolated virtual environment with `--system-site-packages`
to reuse the runtime's installed CPU/GPU PyTorch, then installs the fixed
scientific packages there, including pandas 3.1.1 required by YASA 0.8.1. This
does not replace pandas in the notebook kernel. It validates the dependency
closure of the scientific pipeline; inherited Colab UI packages are not imported
into that subprocess and their unrelated base-environment constraints are not
used to downgrade the scientific packages. Actual versions are saved to Drive.

First use may run setup separately:

```text
python pilot/colab_drive_run.py --project-root /content/drive/MyDrive/research5 --setup-only
python pilot/colab_drive_run.py --project-root /content/drive/MyDrive/research5
```

The normal sequence is stage A for both pipelines, an explicit repeatability
gate, then stage B with both pipelines and the AnySleep oracle diagnostic.
All 13 channel-weight blocks are fixed simultaneously in the appropriate
diagnostic. No new method is trained and no automatic expansion to 20 people
follows. The explicitly authorized `--development-queue` option continues only
after compatible A/B gates into the separately frozen 10-person SC400–SC409
development queue, three predeclared 30-second positions each. Its manifest and
source receipts are checked by `development_queue.py`; it is not an adaptive
search for damaging positions. That queue has a four-hour computation budget.
A completed pilot or development queue still requires scientific review of net harm,
simple baselines, training overlap and a second source.

`status/<run-id>/controller_status.json` records stage boundaries and failure
details; `heartbeat.json` records progress about every 10 seconds while an
actual subprocess is running. Logs flush every output line directly to Drive.
This is ordinary job monitoring, not an idle-avoidance mechanism. There is no
browser keepalive, runtime-limit bypass or automatic purchase of compute.

On a normal restart, the driver checks code/config/scientific-version/device
signatures, downloaded file checksums and completed-stage output checksums.
Only a fully completed phase with its completion manifest is skipped. A partial
phase is preserved with a timestamped `-partial-` name and rerun as a whole;
there is no resume inside an interrupted forward pass or corruption loop.
Different code/config/environments must use a new run identity. CPU and GPU
stages are never silently combined in one resumed run.

A run lock prevents concurrent notebooks from writing the same result paths.
After an abrupt runtime loss, first ensure the old runtime is stopped. Only
then use `--recover-stale-lock`; it also requires a heartbeat/lock older than
10 minutes and preserves the abandoned lock. Stale heartbeat alone is not
proof that another runtime has stopped.

Closing the personal computer does not stop code already executing on Colab's
server, but Colab can still disconnect, reclaim or time-limit that runtime.
Execution continuity is not guaranteed. Files already flushed to mounted Drive
are the recovery basis. A hard termination or Drive outage may prevent the last
status write; compare the heartbeat time and actual stage output instead of
treating a stale RUNNING status as success.

GitHub Actions is now manual-only and is not this computation path. The source
repository remains useful for code provenance, while this run's data/model/
result persistence is Google Drive as requested.
