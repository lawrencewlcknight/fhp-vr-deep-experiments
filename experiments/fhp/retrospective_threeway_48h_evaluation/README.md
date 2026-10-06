# Final 48-hour SD-CFR / UCV-ESCHER / VR-Deep evaluation

This is a read-only retrospective evaluation of the strongest saved 48-hour
policy cohort from each completed FHP training programme:

| Algorithm | Source run | Frozen training commit | Deployment |
|---|---|---|---|
| SD-CFR Exp6 | `sdcfr6-48h-20261002-161544` | `bb689252d6b322adb3de6102d095a49c3ed87250` | full uniform historical trajectory mixture |
| UCV-ESCHER Exp16 | `exp16-feat48-20261004-182051` | `e66d4da515eb212e5026a965ac5c39c86144c901` | saved average-policy network |
| VR-Deep Exp6 | `vr6-ray48-20261003-221425` | `6b95a3ecd191f02bd6972c72a3bf8514483ae02b` | saved average-policy network |

No training, refitting, policy conversion, archive thinning or node matching is
performed. UCV Exp16's continuation from Exp10 is validated. The canonical FHP
game dictionary, source manifests, policy hashes and all SD archive chunks are
checked before scoring.

## Prespecified evaluation

The three co-primary endpoints are final 48-hour head-to-head values:

1. SD-CFR minus UCV-ESCHER;
2. SD-CFR minus VR-Deep;
3. UCV-ESCHER minus VR-Deep.

Each endpoint contains all nine independent training-seed cross-products and
100,000 duplicate deal pairs per cell: 900,000 pairs per endpoint and 2.7
million direct pairs overall. Positive values favour the named left algorithm.
The same chance/action streams are reused across corresponding cells of the
three comparisons.

The main stage also evaluates all nine policies against the same five published
rule agents with 10,000 duplicate pairs per policy-agent cell (450,000 pairs).
A separate stage evaluates Local Best Response with 1,000 duplicate pairs per
policy and 4,096 pre-flop rollouts (9,000 pairs). LBR is a lower bound only. For
LBR, SD-CFR uses the exact own-reach behavioural mixture required by
counterfactual hand queries; it never exposes a sampled hidden archive member.

The three direct comparisons receive:

- pointwise 95% crossed training-seed bootstrap intervals;
- Bonferroni-compatible 98.33% family intervals for the three-endpoint family;
- conditional duplicate-play Monte Carlo standard errors.

Bootstrap draws independently resample the three left and three right training
seeds, then average the resulting 3×3 matrix. Rule-agent and LBR summaries use
the three training seeds as the observations. Nodes and active seconds are
reported descriptively; they are not used for matching or normalization.

## Cloud resources and safety gates

One `n2-standard-16` VM with a 200 GB balanced disk downloads only deployable
policies, immutable SD archive chunks and validation metadata. It runs:

1. a real-policy smoke covering every direct cell, rule-agent cell and LBR
   target;
2. fresh timing pilots on the actual VM;
3. main scoring only if its twofold-margin projection fits a 22-hour cap;
4. LBR only after the main `SUCCESS.json` is durable, and only if its independent
   twofold-margin projection fits a 10-hour cap.

The Batch wall limit is 36 hours, retries are disabled, results are written
atomically, and partial shards are uploaded every five minutes. `resume` checks
the complete scientific identity before reusing any shard. A timing deferral is
recorded in `STATUS.json` and must not be interpreted as an evaluation result.

## Permissions

The runner needs `roles/storage.objectViewer` on the SD-CFR and UCV-ESCHER
source buckets and `roles/storage.objectAdmin` on the VR-Deep output bucket.
It also needs the normal Batch agent and logging permissions documented in
[`docs/GCP_BATCH_EXPERIMENTS.md`](../../../docs/GCP_BATCH_EXPERIMENTS.md).

## Launch

From this repository root, commit and push the evaluator first, then set:

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

The exact source run IDs and loader refs are frozen defaults. Check all source
objects without submitting a VM:

```bash
export RUN_ID="fhp-threeway-check"
bash gcp/run_threeway_48h_evaluation.sh check-sources
```

Run a standalone smoke in a disposable namespace:

```bash
export RUN_ID="fhp-threeway-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_threeway_48h_evaluation.sh smoke-only
bash gcp/run_threeway_48h_evaluation.sh status
```

After it succeeds, submit the production evaluation with a fresh ID:

```bash
export RUN_ID="fhp-threeway-48h-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_threeway_48h_evaluation.sh run
bash gcp/run_threeway_48h_evaluation.sh status
```

Resume an interrupted or safely deferred namespace with unchanged code and
sources:

```bash
export RUN_ID="the-existing-run-id"
export REPO_REF="the-original-full-evaluator-commit"
bash gcp/run_threeway_48h_evaluation.sh resume
```

Prepare and inspect the exact Batch request without cloud calls:

```bash
python gcp/threeway_48h_evaluation_batch.py dry-run \
  --run-id fhp-threeway-dry-run \
  --repo-ref "$REPO_REF" \
  --output /tmp/fhp-threeway-job.json
```

## Outputs

Main results are under `$BUCKET/$RUN_ID/analysis/main/`:

- `analysis_summary.md` — concise primary comparison table;
- `summary.json` — complete interpretation and caveats;
- `matchups.csv` — 27 direct cells and 45 rule-agent cells;
- `direct_comparison_summary.csv` — co-primary estimates and intervals;
- `rule_agent_summary.csv` — three-seed rule-agent summaries;
- `final_48h_head_to_head.png` — the three 3×3 matrices;
- `timing_pilot.json`, `STATUS.json`, `SUCCESS.json` and resumable task shards.

LBR outputs, when the independent gate passes, are under
`$BUCKET/$RUN_ID/analysis/lbr/`. Download the compact analysis without task
shards:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  --exclude='.*task_results/.*' \
  "$BUCKET/$RUN_ID/analysis" \
  "cloud_outputs/$RUN_ID/analysis"
```

Inspect `main/STATUS.json` and `lbr/STATUS.json` before drawing conclusions.
