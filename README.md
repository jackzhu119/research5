# research5: Local corruption in EEG sleep staging

This repository tests whether a local change to EEG/EOG causes net additional sleep-staging errors in other, unchanged epochs. It does not assume that such harm exists.

The first pilot uses pinned AnySleep source and checkpoint plus YASA on one predeclared Sleep-EDF record. Whole-record inference, no-op and repeatability controls precede paired corruption and oracle mechanism diagnostics. No new model training is enabled until practical harm and simple-baseline gates are satisfied.

See RESEARCH_PLAN.md, CONNECTION_STATUS.md, PILOT_REPORT.md, pilot/README.md and docs/ for scope and provenance. The cloud workflow saves outputs and failure logs as GitHub Actions artifacts; it does not depend on a local computer remaining online.
