# Participant and source policy

The first smoke configuration contains only SC400 night SC4001E0. It is an
engineering/development record. No holdout exists yet, and no number from this
record is a population estimate. Earlier model training may have included this
participant or other Sleep-EDF participants; use the exact pretrained-model
training manifest before calling any later result independent external validation.

Do not download the full dataset to populate this list. After the minimal
reproduction passes, choose approximately 10–20 development participants by
deterministic identifier rules independent of predictions; keep every night and
position from each person in one split. Freeze that list and perturbation
expansion before the next outcome inspection. Record all exclusions with reasons.

The CAP source is planned but has no selected or executed participant yet.
Selection requires checking channels, label availability, sampling/time origin
and exact checkpoint overlap. No CAP row is invented here. A source-level
holdout statement is weaker than an audited participant-level training manifest.

Summaries should first aggregate repeated positions/seeds/nights within each
participant using predeclared weighting, then summarize participants. New-error
and corrected-error counts are both required; net additional error is their
difference divided by the same eligible unmodified-epoch denominator. Flips
between two wrong labels are not new errors. The smoke harness separates
interventions so no-op and corruptions are never averaged together.
