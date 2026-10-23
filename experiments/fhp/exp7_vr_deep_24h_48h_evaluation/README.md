# Experiment 7 — VR-Deep duration and strategic evaluation

Evaluate the frozen **Experiment 6 24h and 48h policies**, seeds 0, 1 and 2.
These are earlier/later checkpoints of the same trajectories, not Experiment
5's separate 24h run. No training or checkpoint selection occurs.

The final **UCV-Escher Experiment 38 48h** models are included **only in the
strategic audit** on the same positions. **SD-CFR is excluded.**

## Prespecified analysis

One Batch job has two independent tasks, each on an n2-standard-16
(16 vCPU, 64 GB), running concurrently when quota allows.

| Task | Evaluation |
| --- | --- |
| Gameplay | All nine 48h-vs-24h seed pairs, 50,000 duplicate pairs per cell; five published rule agents, 10,000 pairs per policy/opponent; LBR, 1,000 pairs per policy, 4,096 preflop rollouts |
| Strategic | All six VR policies and three final UCV policies; five rule opponents plus three fixed UCV Exp9 anchors; full remaining-round response calculations on the frozen board library |

Gameplay totals **756,000 duplicate pairs / 1,512,000 hands**, including 6,000
LBR pairs. Direct and rule matches use 5,000-pair shards; LBR uses 10-pair
shards. Seat swapping reduces deal variance. LBR payoff is the responder's:
lower is better for the evaluated policy.

The main gameplay endpoint is mean 48h-versus-24h payoff across all nine cells.
Also report the three same-trajectory predecessor matchups. The nine cells are
not nine independently trained agents: bootstrap resampling shares the same
three trajectory weights at both horizons. Conditional deal intervals are
reported separately from trajectory-bootstrap intervals; neither is presented
as a combined population guarantee. Common-opponent comparisons first aggregate
within each seed and then use the three paired 48-minus-24 differences.
Three trajectories provide limited uncertainty resolution. Supporting contrasts
are exploratory, without familywise significance or an automatic winner claim.

### Strategic positions and interpretation

- **Development:** the existing 64 board classes, all six textures, with
  ordinary opponent ranges plus synthetic range tilts -2 and +2.
- **Assessment:** 32 newly frozen board classes selected without model scores,
  ordinary ranges only. Excludes the recorded earlier strategic library,
  reference-deal boards and rare-state panels. All 13 trips-board classes have
  already appeared in those diagnostics, so trips remain development-only.
  The new assessment covers the other five feasible textures.
- Both partitions enumerate the same supported public histories and private
  hands for every candidate, through the shared evaluation suite. No additional
  policy-generated reference-deal layer is mixed into the primary metric.
- The primary strategic endpoint is **remaining-round root response gap**:
  BB foregone per supported audited root against the fixed opponent.
  Opponents are equally weighted within training seed. Smaller is better.
  VR's horizon comparison is paired; VR-versus-UCV uses independently resampled
  training seeds, not integer-matched seeds.
- Detailed reports retain hand-strength and pot-odds groups, severe local gaps,
  final-call failures and concrete board/private-card/betting-history examples.
  Selected largest-error examples illustrate behaviour but are post-hoc,
  conditional diagnostics, not estimates of typical full-game loss.

Do not pool partitions: their texture weights differ. Assessment positions are
new relative to *recorded diagnostics*, not guaranteed unseen during training
or proof against all earlier model-selection effects. The older library's
reserved boards remain unused. No tuning or selection occurs on either panel.

LBR, root response gaps and local action gaps are **not exact exploitability**.
A lower remaining-round gap against a fixed panel does not prove proximity to
a Nash equilibrium. Do not combine unlike metrics into a single strength score.
This broader audit complements, but does not numerically reproduce, the
targeted rare-state diagnostics in UCV Experiment 39.

## Sources and reproducibility

The sources.json file freezes exact SHA256, horizon, seed, nodes and iterations:

- VR: vr6-ray48-20261003-221425.
- UCV candidates: exp38-rich48-20261019-230016, the same final models
  checked for the UCV Experiment 40 comparison.
- Fixed anchors: exp9-cache24-20261001-132550.

Only small playable policies and provenance are downloaded, **not** replay or
training-state archives. Native loaders verify checkpoint and training metadata.
The shared suite and native Python packages are bundled from the local sibling
checkouts with file hashes and Git commit provenance. No model files or
credentials are bundled. Native families are loaded in isolated processes.
Dependencies retain the repo's pins, including OpenSpiel 1.6.3.

The position_library.json file freezes positions and historical exclusions.
Inputs, evaluator sources and cache identities must agree on resume. Recovery
uses the original uploaded bundle, not today's local checkout.

## Launch from the VR-Deep repository root

The sibling evaluation-suite and UCV repositories must be available locally.
Override paths with EVALUATION_SUITE_ROOT and UCV_REPO_ROOT if necessary.
The service account needs read access to **both** result buckets, write access
to the output bucket, and normal Batch/Logging permissions. Source existence
is checked before paid VMs start; native verification occurs before scoring.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp_vr_deep/fhp-vr-deep-experiments

export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-fhp-vr-deep-results"
export VR_BUCKET="$BUCKET"
export UCV_BUCKET="gs://clever-overview-399515-fhp-escher-results"
export SA_EMAIL="fhp-vr-deep-runner@${PROJECT_ID}.iam.gserviceaccount.com"
export RUN_ID="vr7-eval2448-$(date -u +%Y%m%d-%H%M%S)"

bash gcp/run_exp7_duration_evaluation.sh run
```

This initiates the complete experiment once. Each task first runs a small
correctness smoke, then its full evaluation automatically. Smoke is not
performance evidence and never opens assessment positions.
The job continues after the laptop disconnects.

Preparation without cloud access:

```bash
bash gcp/run_exp7_duration_evaluation.sh dry-run
```

Optional smoke-only job (distinct run ID):

```bash
export RUN_ID="vr7-eval2448-smoke-$(date -u +%Y%m%d-%H%M%S)"
bash gcp/run_exp7_duration_evaluation.sh smoke-only
```

The full job permits up to 48h per task, including setup and uploads; this is
a safety ceiling, **not a runtime prediction or training budget**. Gameplay
has a 40h scoring allowance; the strategic task allows 20h per partition.
There is no projected-cost qualification gate rejecting a long but valid run.
Completed work is saved every five minutes and on exit. Actual runtime depends
on the richer UCV encoder and the broad strategic analysis; inspect smoke
timings before relying on an earlier estimate.

## Status, recovery and results

```bash
bash gcp/run_exp7_duration_evaluation.sh status
```

If interrupted, keep the **original** RUN_ID and execution environment:

```bash
bash gcp/run_exp7_duration_evaluation.sh resume
```

This launches a new attempt using the original uploaded code and configuration.
Verified match shards, exported board tables and analysed boards are reused;
a completed branch is skipped. Do not alter cloud inputs or caches. Up to five
minutes/in-flight work may need repeating. Resume refuses an active job in the
same namespace. An incomplete task never publishes a full success marker.
Completion markers are uploaded only after their underlying files.

Download lightweight analytical results, retaining detailed position examples:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='(^|/)(task_results|tables|workers|board_results)/|[.]lock$' \
  "$BUCKET/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Main outputs under analysis:

- matches/full/summary.json, analysis_summary.md and duration_comparison.png.
- strategic/full/summary.json and report.html.
- strategic/full/{development,assessment}/analysis/: detailed HTML, JSON,
  graphics and diagnostic examples.
- Both branches' full/SUCCESS.json markers must exist for completion.
  Smoke markers do not qualify.
- Each branch includes resource snapshots, native-validation logs, bootstrap
  logs, failure summaries and a source manifest.

Retain full cloud caches for recovery even when excluding them from local
analytical-results downloads.
