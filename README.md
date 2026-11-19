# research5: Local corruption in EEG sleep staging

This repository tests whether a local change to EEG/EOG causes net additional sleep-staging errors in other, unchanged epochs. It does not assume that such harm exists.

## Current results

The [2026-11-19 Drive snapshot](evidence/drive-2026-11-19/README.md) includes completed v3 Sleep-EDF and CAP results, per-participant tables, and the v2 rerun status at capture. The [paper direction review](docs/ANALYSIS_PIVOT_2026-11-19.md) evaluates the failed method comparison and proposes a fixed external validation of error propagation and mitigation costs.

## Initial pilot

The first pilot uses pinned AnySleep source and checkpoint plus YASA on one predeclared Sleep-EDF record. Whole-record inference, no-op and repeatability controls precede paired corruption and oracle mechanism diagnostics. No new model training is enabled until practical harm and simple-baseline gates are satisfied.

See RESEARCH_PLAN.md, CONNECTION_STATUS.md, PILOT_REPORT.md, pilot/README.md and docs/ for scope and provenance. The selected workflow runs in Colab and saves data, checkpoints, outputs and logs directly to Google Drive. See docs/COLAB_DRIVE_RUN.md. GitHub Actions remains a manual-only backup. Normal Colab limits still apply; saved state can be resumed but uninterrupted execution is not guaranteed.
