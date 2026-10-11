# Final 48-hour SD-CFR / UCV-ESCHER / VR-Deep evaluation

Read-only evaluation of the saved 48-hour cohorts. No training, refitting,
policy conversion, archive thinning or node matching.

| Algorithm | Source run | Frozen training commit | Deployment |
|---|---|---|---|
| SD-CFR Exp6 | `sdcfr6-48h-20261002-161544` | `bb689252d6b322adb3de6102d095a49c3ed87250` | full uniform historical trajectory mixture |
| UCV-ESCHER Exp16 | `exp16-feat48-20261004-182051` | `e66d4da515eb212e5026a965ac5c39c86144c901` | saved average-policy network |
| VR-Deep Exp6 | `vr6-ray48-20261003-221425` | `6b95a3ecd191f02bd6972c72a3bf8514483ae02b` | saved average-policy network |

The canonical FHP game, source identities, policy hashes, UCV Exp10-to-Exp16
lineage and all SD archive chunks are checked before evaluation.

**Protocol v2 removes LBR entirely from this experiment**, including smoke,
pilots, scoring and reports. The new strategic-position audit is a separate
component, not a replacement exploitability estimate. Other experiments'
LBR implementations are unchanged.

## Primary: head-to-head and rule-agent matches

Three co-primary comparisons: SD-CFR minus UCV-ESCHER, SD-CFR minus VR-Deep,
and UCV-ESCHER minus VR-Deep. Each has all nine training-seed cross-products
and 100,000 duplicate deal pairs per cell: **2.7 million direct pairs**.
Deals replay with seats swapped; corresponding comparison cells share
chance/action streams. Positive values favour the left algorithm.

All nine policies also play the same five published rule agents with 10,000
pairs per policy-agent cell: **450,000 additional pairs**.

Direct comparisons retain pointwise 95% crossed-seed bootstrap intervals,
Bonferroni-compatible 98.33% intervals for the three-endpoint family and
conditional Monte Carlo SEs. Bootstrap draws independently resample left/right
training seeds, then average the 3×3 matrix. Rule summaries use training seeds
as observations. Nodes and active seconds are descriptive only.

The smoke now contains **72 tasks**, two pairs each, covering every direct and
rule-agent cell; it has **no LBR tasks**. Finite-statistic validation stays strict.
Main scoring requires a fresh timing pilot whose twofold-margin estimate fits
22 hours. Its Batch cap is 26 hours; standalone smoke has a two-hour cap.
Caps are safety limits, not forecasts. Smoke results are not strength evidence.

## Diagnostic: strategic-position audit

Uses the shared `fhp-evaluation-suite` strategic audit at the exact
`DEFAULT_SUITE_REF` pinned in the Batch launcher. The SD audit adapter remains pinned to
`b670d99968d49f33744eed4a76640057e43c15b8` for batched full-mixture queries.
This changes neither trained policies nor the primary match loader. Native
families run in separate processes; vendored packages cannot shadow the suite.

| Stage | Candidates (48h only) | Core boards | Reference pairs per opponent | Batch cap |
|---|---|---|---|---|
| Strategic pilot | Seed 0 of each algorithm | 8 development | 8 | 6 hours |
| Strategic development | All three seeds of each algorithm | 64 development | 64 | 24 hours |

The pilot tests correctness and feasibility on **real full-size SD archives**,
not a reduced-model approximation. SD cost is still uncertain. Development
requires a successful pilot with identical pinned code and explicit user review;
it never launches automatically. Audit jobs are independent of the match jobs.

The library seed (`20261006`), panel and reference policy match the existing
Exp16 strategic audit:

- Five corrected published rule agents plus the three frozen Exp9 24h policies
  from `exp9-cache24-20261001-132550`, with pinned checkpoint hashes.
- Exp9 seed 0 generates the same reference-position distribution for every candidate.
- Both seats, all seven legal preflop entry histories, reachable final-round
  decisions and all 1,176 private hands per board; exact compatible-hand and
  remaining-round integration, including blockers and ties.
- Tilt 0 is primary; -2/+2 are separately labelled synthetic range sensitivity checks.
- **All 32 assessment boards stay locked**, including in reference sampling.
  The launcher provides no assessment mode; development is diagnostic evidence.

The detailed shared report covers provable errors such as folding an unbeatable
hand at a terminal call/fold decision, range-conditioned errors, local decision
gaps, remaining-round response gaps, pot odds, hand/board/history categories and
concrete high-gap positions. It does not assume that strong hands must always
raise or weak hands must always fold.

The cross-algorithm report focuses on **remaining-round root response gap**:
foregone value against a fixed opponent relative to an optimal remaining
continuation. Opponents receive equal weights **within each seed first**, followed
by seed means and descriptive SEs. Matching seed integers across algorithms are
not paired experiments. Common board/root support is checked before comparison.
No position-level pseudo-replication or combined winner score is used.

Audit values are **BB per audited decision/root**, not mbb per whole-game hand.
They are neither LBR nor whole-game exploitability, and cannot establish
equilibrium convergence. The reference sample is supplementary and small.
Exact-value, native/batched parity, hidden-card privacy, card-order and
suit-invariance checks must pass. SD uses **all** historical networks with exact
own-reach weights; it cannot substitute a sampled archive member.

## Resources, provenance and recovery

Every component uses one `n2-standard-16` CPU VM, a 200 GB disk and no automatic
Batch retries. The revised strategic executor uses **four concurrent board
workers, each with four inference threads** (16 inference threads in total).
Each worker receives one policy/board task and retains every SD-CFR historical
network. It has a two-hour timeout and a 13,000 MiB process-tree RSS cap; the
sum of concurrent limits is 52,000 MiB, leaving coordinator/OS headroom in the
62,000 MiB allocation. Numerical analysis runs separately with four threads
and a 52,000 MiB cap.

An invocation has a **4.5-hour pilot / 21-hour development** budget shared by
export and numerical analysis. Analysis is additionally capped at three hours
in pilot. A new wave is not started when the remaining budget is shorter than
the slowest completed task of a comparable policy family plus 20% (bounded by
the per-task cap). Six-/24-hour
Batch caps remain unchanged, including setup/upload headroom. These are bounds,
not completion forecasts; completion may require multiple explicitly requested
resume jobs. No job launches another job.

### October 2026 timeout repair

The old `fhp-threeway-sp-20261010-215643` pilot stopped after three hours in
SD-CFR export: three of **16 boards** completed (eight core plus eight
reference-deal boards), taking roughly 41--59 minutes each. Completed boards
passed their checks; peak worker RSS was about 737 MiB. This was a per-policy
runtime limit, not a numerical or memory failure.

The new execution contract is `fhp_threeway_48h_strategic_v2_board_exports`.
SD-CFR board tasks run first, so the most expensive workload is measured first.
`export_progress.json` and `export_segments/attempt_*/progress.json` record
completed/remaining counts, per-task times, first-wave and later-wave throughput,
and memory bounds. The first wave is a real full-archive throughput benchmark,
**not** a controlled serial-speedup estimate. For explicit allocation comparisons,
use fresh pilot namespaces with `--strategic-export-workers 1`, `2`, or `4`;
default `4`. Worker count is frozen on resume and must match the reviewed pilot
for development. Timings do not change the board panel, opponents, numerical
checks or scoring, and do not establish playing strength.

Completed board caches are checksum-verified and skipped on resume. Expiration
of the invocation budget records `deferred_time_budget`, uploads progress and
exits cleanly **without** final analysis `SUCCESS.json`. Thus Batch `SUCCEEDED`
can mean a completed segment, not a completed audit. An individual board
timeout, memory failure, invalid output or numerical exception remains a failure;
it is not silently accepted or automatically retried. Worker logs are now in
`workers/<policy>/<board>/attempt_*/`; analysis retains `workers/analysis/attempt_*/`.

Partial outputs upload every five minutes. Data uploads precede board-cache
markers, followed by final success markers. A missing final marker means
**incomplete**, even if Batch succeeds after a main timing deferral.
Audit failures retain worker diagnostics and completed board caches.

Resume requires identical code, inputs, runtime and execution settings. The
audit hashes every SD chunk. After a byte-identical download on a new VM it
restores only local staging-file timestamps from the original manifest, keeping
the shared cache identity stable. Cloud training artifacts are never modified.

**Use fresh namespaces for this revision.** Do not resume
`fhp-threeway-48h-20261007-231524` or
`fhp-threeway-smoke-20261008-002547` with the new protocol. In particular,
**do not resume `fhp-threeway-sp-20261010-215643` with the revised executor**.
Its partial outputs are preserved; changed evaluator fingerprints must not be
relabelled as the new experiment. The launcher rejects a legacy or changed
strategic resume contract before starting another VM. Subsequent segments of a
new-protocol run can reuse its own verified caches with identical code/settings.

## Launch

From this repository root, commit and push the new evaluator before cloud use:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-vr-deep-results"
export SD_BUCKET="gs://${PROJECT_ID}-fhp-deep-cfr-results"
export UCV_BUCKET="gs://${PROJECT_ID}-fhp-escher-results"
export VR_BUCKET="gs://${PROJECT_ID}-fhp-vr-deep-results"
export SA_EMAIL="fhp-vr-deep-runner@${PROJECT_ID}.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
```

The runner needs source-bucket reads (including Exp9 in the UCV bucket), VR
output-bucket writes, and normal Batch/logging permissions. No IAM bindings are
changed automatically. Native loader and suite refs are frozen defaults.

### Primary match evaluation

```bash
export RUN_ID="fhp-threeway-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_threeway_48h_evaluation.sh check-sources --evaluation main
bash gcp/run_threeway_48h_evaluation.sh smoke-only --evaluation main
bash gcp/run_threeway_48h_evaluation.sh status

# After smoke succeeds, use a fresh production namespace.
export RUN_ID="fhp-threeway-48h-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_threeway_48h_evaluation.sh run --evaluation main
```

### Strategic pilot (separate job)

```bash
export PILOT_RUN_ID="fhp-threeway-sp-$(date -u '+%Y%m%d-%H%M%S')"
export RUN_ID="$PILOT_RUN_ID"
bash gcp/run_threeway_48h_evaluation.sh check-sources --evaluation strategic-pilot
bash gcp/run_threeway_48h_evaluation.sh run --evaluation strategic-pilot
bash gcp/run_threeway_48h_evaluation.sh status --evaluation strategic-pilot
```

Inspect both Batch and audit completion. If `STATUS.json` reports
`deferred_time_budget`, wait for the current Batch segment to stop, then retain
the **same new-protocol RUN_ID, REPO_REF and worker setting**:

```bash
gcloud storage cat "$BUCKET/$RUN_ID/analysis/strategic-pilot/STATUS.json"
bash gcp/run_threeway_48h_evaluation.sh resume --evaluation strategic-pilot
```

Review failed board logs before resuming a genuine failure. Changing worker
count, source, code or other frozen settings requires a fresh namespace.

Review runtime, export checks, coverage and concrete decision cases. If full SD
export is not feasible, do not launch development or silently drop SD-CFR:
retain the primary comparison and revisit the audit budget.

### Strategic development (only after pilot review)

Keep all code refs identical to the successful pilot. The review flag is your
explicit acknowledgement, not permission inferred by a job.

```bash
export RUN_ID="fhp-threeway-sd-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_threeway_48h_evaluation.sh run \
  --evaluation strategic-development \
  --pilot-run-id "$PILOT_RUN_ID" --pilot-reviewed
```

Prepare any request locally without cloud calls by replacing `run` with
`dry-run --output /tmp/fhp-threeway-job.json`. After an interrupted attempt
stops, use `resume` with the original run ID, component, code refs and (for
development) pilot flags. Never run two writers for the same namespace.

## Reports and completion

Primary outputs: `$BUCKET/$RUN_ID/analysis/main/`, including
`analysis_summary.md`, `summary.json`, `matchups.csv`,
`direct_comparison_summary.csv`, `rule_agent_summary.csv`,
`final_48h_head_to_head.png`, `timing_pilot.json`, `STATUS.json`, `SUCCESS.json`.

Audit outputs: `$BUCKET/$RUN_ID/analysis/strategic-pilot/analysis/` or
`.../strategic-development/analysis/`:

- `threeway_report.html` / `threeway_summary.json`: cross-algorithm root-gap comparison;
- `report.html` / `summary.json`: shared-suite detailed positions and diagnostics;
- `runtime_summary.json`: worker timings and peak memory;
- `SUCCESS.json`: checksummed completion, explicitly excluding LBR/assessment.

The stage root retains `contract.json`, `manifest.json`, `STATUS.json`, the
shared `status.json`, export timing/progress segments, worker logs and resumable
tables/board results. Only final checksummed `analysis/SUCCESS.json` confirms
completion; `deferred_time_budget` is incomplete. A complete
primary run does not imply that the separate strategic audit is complete.

Compact download excluding large caches and individual match shards:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='.*(task_results|tables|board_results|workers)/.*' \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

Unit tests and control policies do not establish trained-model runtime or
playing strength. Cloud execution and pilot review remain separate steps.
