# Experiment 1: Leduc configuration transfer

Experiment 1 trains **VR-DeepDCFR+** and **VR-DeepPDCFR+** on the canonical FHP
game for paired seeds `0`, `1`, and `2`.

## Training contract

The shared training dictionary is the paper/Table 2 Leduc configuration used in
the Leduc ESCHER-architecture repository. The only changed field is
`game_name: leduc_poker -> FHP`:

- replay capacities: 1,000,000 for advantage, average policy, and baseline;
- learning rate: `1e-3`;
- 10,000 traversals per player and outer iteration;
- 750 advantage, 5,000 policy, and 10,000 baseline fitting steps;
- batch size 2,048;
- three hidden layers of 64 units;
- history-value baseline enabled;
- cumulative advantage networks are not reinitialized;
- one policy fit/diagnostic checkpoint per completed outer iteration.

Variant-specific parameters are:

| Algorithm | Alpha | Gamma | Immediate-regret reinitialization |
|---|---:|---:|---|
| VR-DeepDCFR+ | 2.0 | 2.0 | Not applicable |
| VR-DeepPDCFR+ | 2.3 | 2.0 | Enabled |

The training clock excludes policy-checkpoint fitting and snapshot writes. Each
seed saves reloadable policies after 6 and 12 effective training hours and
stops after the 12-hour checkpoint. The 100-iteration setting remains the
Leduc safety cap; it is not used as the primary horizon.

## Analysis contract

Exact tabular exploitability is not attempted for FHP. At each matched 6-hour
and 12-hour checkpoint, the runner instead plays 100,000 deal pairs with the
algorithms in both seat assignments. It reports VR-DeepDCFR+ minus
VR-DeepPDCFR+ payoff in chips and milli-big-blinds per hand, with per-evaluation
standard errors and 95% confidence intervals. Cross-seed aggregates follow the
mean/standard-error conventions of the Leduc repository.

Evaluation games are never included in `nodes_touched` or effective training
time.

## Local production run

The default command runs all six 12-hour workers sequentially and then performs
the paired evaluation:

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run
```

Run only one algorithm or seed with:

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run \
  --algorithms vr_deep_dcfr_plus \
  --seeds 0
```

## Local smoke test

This executes both variants, two policy checkpoints, snapshot reload, sampled
head-to-head evaluation, aggregation, and plots with tiny settings. Its policy
results are not scientifically meaningful.

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run \
  --smoke \
  --seeds 0 \
  --head-to-head-deals 4 \
  --output-root outputs/smoke_tests
```

## GCP Batch production run

Configure:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-vr-deep-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/fhp-poker-vr-deep-experiments.git"
```

The six-worker sequential run contains 72 effective training hours. Policy
fitting, snapshotting, evaluation, installation, and VM variation add wall-time
overhead, so the command uses a conservative 120-hour timeout:

```bash
JOB_NAME="fhp-vr-deep-exp1-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_leduc_config_transfer.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 432000 8000 32000 100 pd-balanced
```

## Split-job execution and recovery

The two algorithms can be trained concurrently in separate 60-hour Batch jobs:

```bash
JOB_DCFR="fhp-vr-deep-exp1-dcfr-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_DCFR" \
  "python -m experiments.fhp.exp1_leduc_config_transfer.run \
    --algorithms vr_deep_dcfr_plus \
    --output-root outputs/cloud/$JOB_DCFR" \
  n2-standard-8 216000 8000 32000 100 pd-balanced

JOB_PDCFR="fhp-vr-deep-exp1-pdcfr-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_PDCFR" \
  "python -m experiments.fhp.exp1_leduc_config_transfer.run \
    --algorithms vr_deep_pdcfr_plus \
    --output-root outputs/cloud/$JOB_PDCFR" \
  n2-standard-8 216000 8000 32000 100 pd-balanced
```

After downloading both output trees, combine them and perform the head-to-head
evaluation locally:

```bash
python -m experiments.fhp.exp1_leduc_config_transfer.run \
  --aggregate-run-dir cloud_outputs/DCFR_RUN \
  --aggregate-run-dir cloud_outputs/PDCFR_RUN \
  --output-root outputs/exp1_aggregated
```

`--aggregate-run-dir` discovers relocatable `worker_runs/*/result.json` files,
rejects duplicate algorithm/seed pairs, and regenerates all tables and plots.

## GCP smoke test

```bash
JOB_NAME="fhp-vr-deep-exp1-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.exp1_leduc_config_transfer.run \
    --smoke --seeds 0 --head-to-head-deals 4 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100 pd-balanced
```

Monitor and download with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

The combined run directory contains:

- `experiment_metadata.json` and `run_status.json`;
- `seed_summary.csv`, `checkpoint_curves.csv`, and `checkpoint_manifest.csv`;
- `aggregate_summary.json` and `checkpoint_aggregate.json`;
- `head_to_head_pairs.csv` and `head_to_head_summary.json`;
- `nodes_by_training_time.png` and `head_to_head_by_checkpoint.png`;
- worker inputs, logs, results, manifests, failures, and reloadable snapshots.
