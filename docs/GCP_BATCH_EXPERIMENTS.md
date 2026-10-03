# Running the FHP VR-Deep experiments on Google Cloud Batch

For **Experiment 5 (eight Ray traversal workers, unchanged central fitting)**,
see [the synchronous collection guide](../experiments/fhp/exp5_vr_deep_ray8/README.md).
It retains Experiment 3's three seeds, `n2-standard-16` VMs, 24 active hours and
6/12/18/24-hour policies. The eight-thread central learner is unchanged; eight
one-thread actors share the existing total traversal budget on each seed VM.
With the environment below configured and the implementation committed/pushed:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr5-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp5_vr_deep_ray8.sh smoke-only
bash gcp/run_exp5_vr_deep_ray8.sh status

# After smoke succeeds, choose a fresh namespace.
export RUN_ID="vr5-ray8-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp5_vr_deep_ray8.sh run
bash gcp/run_exp5_vr_deep_ray8.sh status
```

The full workflow includes a real eight-actor smoke/equivalence gate before
launching the three training VMs. It installs `requirements-ray.txt` only for
this experiment, needs 48 N2 vCPUs plus the controller, and performs no policy
evaluation. Ray startup/merge/worker timings and whole-job resource diagnostics
are retained alongside the existing training outputs.

For **Experiment 4 (short computation-thread profiling on n2-standard-16)**,
see [the controlled profiling guide](../experiments/fhp/exp4_vr_deep_thread_profile/README.md).
It compares 1/2/4/8/16 threads sequentially, with three repeats, identical
prefit states and minibatches, and trace-controlled full-iteration timings.
No policy evaluation or 24-hour training jobs are launched. With the environment
below configured and the new implementation committed and pushed:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr4-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp4_vr_deep_thread_profile.sh smoke-only
bash gcp/run_exp4_vr_deep_thread_profile.sh status

# After a successful smoke, choose a fresh namespace.
export RUN_ID="vr4-threads-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp4_vr_deep_thread_profile.sh run
bash gcp/run_exp4_vr_deep_thread_profile.sh status
```

The full job includes its own smoke gate, needs only 16 N2 vCPUs, and has a
five-hour profiling budget within a six-hour Batch cap. Partial outputs and
resource diagnostics are retained on failure; no automatic retry is enabled.
Main results are under `$BUCKET/$RUN_ID/profile/analysis`.

For **Experiment 3 (Experiment 2 on n2-standard-16, 24 active hours)**, see
[the VM-only experiment guide](../experiments/fhp/exp3_vr_deep_lossless_n2_standard16/README.md).
With the environment variables below set and the new commit pushed:

```bash
export REPO_REF="$(git rev-parse HEAD)"
# Optional standalone cloud smoke; this never submits full training.
export RUN_ID="vr3-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh smoke-only
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh status

# After the smoke succeeds, use a NEW namespace for the full workflow.
export RUN_ID="vr3-vm16-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh run
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh status
```

The full workflow always runs its own cloud smoke before three parallel seeds.
It requires 48 N2 vCPUs for training, plus the controller. Eight fitting threads
and all Experiment 2 learning settings stay unchanged; only training/smoke
VMs are larger. No poker-performance evaluation jobs are launched.

For **Experiment 2 (UCV Exp.2 encoder, otherwise unchanged VR-Deep baseline)**,
use the [dedicated experiment guide](../experiments/fhp/exp2_vr_deep_lossless_24h/README.md)
and `bash gcp/run_exp2_vr_deep_lossless_24h.sh run`. It uses the same service
account, resources and cloud-smoke/three-worker/aggregation structure below.

For the **new Experiment 1 (VR-DeepPDCFR+, three seeds, 24 active hours)**, use
[`run_exp1_vr_deep_pdcfr_24h.sh` and its experiment guide](../experiments/fhp/exp1_vr_deep_pdcfr_24h/README.md).
That guide includes the additional controller IAM permissions and the
smoke -> three parallel training VMs -> aggregation workflow. The one-time
project/bucket/account setup below remains applicable. The later generic
submission examples below refer to the **archived** two-algorithm study.

This guide gives the complete command-line workflow for running this repository
on Google Cloud Batch. It covers:

1. configuring Google Cloud locally;
2. creating a Cloud Storage bucket and Batch service account;
3. checking the repository and submission helper;
4. running and verifying a smoke test;
5. running Archived Experiment 1 as one job or two concurrent algorithm jobs;
6. monitoring, diagnosing, downloading, and aggregating results;
7. choosing resources and cleaning up.

The Batch helper creates a temporary VM, clones this GitHub repository, creates
an isolated Python 3.11 environment, installs the project, starts independent
resource monitoring, runs the selected command, uploads `outputs/` to Cloud
Storage, and exits. Batch manages the VM lifecycle; there is no persistent VM
to stop after a completed job.

---

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed locally;
- permission to enable APIs, create service accounts and IAM bindings, submit
  Batch jobs, and create or use a Cloud Storage bucket;
- this repository pushed to GitHub at
  `https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git`;
- sufficient Batch CPU quota in the selected region.

The included helper clones the repository with HTTPS. A public repository works
without further configuration. For a private repository, replace the clone
method with a deploy key, short-lived GitHub token, or pre-built private
container image; do not commit credentials to this repository.

Before every cloud run, make sure the required local commits are on the remote:

```bash
git status --short
git log --oneline origin/main..main
git push origin main
```

The second command should print nothing after the push.

---

## 2. One-time local Google Cloud setup

Authenticate and select the project:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"
```

Use `europe-west1` unless there is a project-specific reason to choose another
region:

```bash
export REGION="europe-west1"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

---

## 3. Create a Cloud Storage bucket

Create a regional bucket for working outputs:

```bash
export BUCKET_NAME="${PROJECT_ID}-fhp-vr-deep-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

If the bucket already exists, do not recreate it. Verify access instead:

```bash
gcloud storage buckets describe "$BUCKET"
```

Keep the bucket in the same region as the Batch job where practical.

---

## 4. Create a Batch service account

Create a dedicated service account:

```bash
export SA_NAME="fhp-vr-deep-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="FHP VR-Deep experiment runner" \
  --project="$PROJECT_ID"
```

Allow it to write logs and report Batch agent status:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"
```

Allow it to write job artifacts to the results bucket:

```bash
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Allow your user account to run jobs as this service account and read logs:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

Your account must also be allowed to submit Batch jobs. If it is not already
covered by project-level permissions, ask the project administrator for an
appropriate Batch role such as `roles/batch.jobsEditor`.

---

## 5. Variables required in each new terminal

Set these before invoking the submission helper:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-fhp-vr-deep-results"
export SA_EMAIL="fhp-vr-deep-runner@${PROJECT_ID}.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/fhp-vr-deep-experiments.git"

gcloud config set project "$PROJECT_ID"
```

Check them before submission:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
echo "$REPO_URL"
```

`REPO_URL` is optional when using the public repository above because that URL
is already the helper's default. Keeping it explicit makes the submitted job
easier to audit.

---

## 6. Check the included Batch helper

Run all commands in this guide from the repository root:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/fhp_vr_deep/fhp-vr-deep-experiments
```

The repository already contains:

- `gcp/submit_batch_experiment.sh` — creates and submits the Batch job;
- `gcp/read_batch_task_logs.sh` — reads logs scoped to one exact Batch job;
- `fhp_vr_deep/batch_diagnostics.py` — records resource snapshots and diagnoses
  memory, timeout, disk, installation, and experiment failures.

Check the scripts are executable and syntactically valid:

```bash
chmod +x gcp/submit_batch_experiment.sh gcp/read_batch_task_logs.sh
bash -n gcp/submit_batch_experiment.sh
bash -n gcp/read_batch_task_logs.sh
```

`bash -n` should return silently.

The submission helper takes eight positional arguments:

```text
JOB_NAME
PYTHON_EXPERIMENT_COMMAND
MACHINE_TYPE
MAX_RUN_SECONDS
CPU_MILLI
MEMORY_MIB
BOOT_DISK_SIZE_GB
BOOT_DISK_TYPE
```

Its Archived Experiment 1 defaults are:

```text
n2-standard-8 432000 8000 32000 100 pd-balanced
```

The generated job uses `maxRetryCount: 0`. A failed scientific run is therefore
not silently retried with the same seed and output identity.

---

## 7. Submit the GCP smoke test first

The smoke test exercises both VR-Deep variants, both time-checkpoint events,
snapshot save/reload, sampled seat-swapped play, aggregation, plots, resource
monitoring, and Cloud Storage upload. Its policy results are meaningless.

```bash
JOB_NAME="fhp-vr-deep-archieved-exp1-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
    --smoke \
    --seeds 0 \
    --head-to-head-deals 4 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 \
  21600 \
  4000 \
  16000 \
  100 \
  pd-balanced
```

The 21,600-second limit gives installation and diagnostics generous headroom;
the reduced experiment itself should finish much sooner.

Immediately confirm that Batch accepted the job:

```bash
gcloud batch jobs list --location "$REGION"
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
```

Do not submit the production run until the smoke job reaches `SUCCEEDED`, its
outputs are present in Cloud Storage, and the downloaded `run_status.json`
reports two completed workers and no failures.

---

## 8. Monitor a running job

List jobs:

```bash
gcloud batch jobs list --location "$REGION"
```

Describe the current job:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
```

Common states are `QUEUED`, `SCHEDULED`, `RUNNING`, `SUCCEEDED`, and `FAILED`.
After completion, `status.runDuration` reports time spent in the running state.

Read task logs with the repository helper:

```bash
./gcp/read_batch_task_logs.sh "$JOB_NAME"
```

Filter the exact job's logs for likely failures:

```bash
./gcp/read_batch_task_logs.sh \
  "$JOB_NAME" \
  ERROR Traceback Killed "out of memory" "No space left" SIGTERM
```

The helper resolves the Batch UID and scopes the Cloud Logging query to that
UID before applying text filters, preventing logs from neighbouring jobs from
being mistaken for this run.

---

## 9. Verify and download smoke outputs

List uploaded objects:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

Download the complete job output:

```bash
mkdir -p "cloud_outputs/$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

Locate the experiment directory and inspect its status:

```bash
find "cloud_outputs/$JOB_NAME" -name run_status.json -print
find "cloud_outputs/$JOB_NAME" -name summary.json -print
find "cloud_outputs/$JOB_NAME" -name failure.json -print
```

A successful smoke output contains at least:

- `run_status.json` with `completed_workers: 2` and an empty failure list;
- two worker directories under `worker_runs/`;
- four checkpoint snapshots in total;
- `seed_summary.csv`, `checkpoint_curves.csv`, and
  `checkpoint_manifest.csv`;
- `head_to_head_pairs.csv` and `head_to_head_summary.json`;
- `nodes_by_training_time.png` and `head_to_head_by_checkpoint.png`;
- `batch_run.log`, `resource_snapshots.jsonl`, `batch_diagnostics.json`, and
  `batch_status.json` in the job output root.

---

## 10. Run the full Archived Experiment 1 as one job

The default command trains both algorithms for seeds `0`, `1`, and `2`. Each of
the six workers runs to the 12-hour effective-training checkpoint, giving 72
effective training hours in total. Average-policy fitting, snapshotting,
installation, evaluation, and VM variation add wall-clock overhead. Use the
120-hour safety cap:

```bash
JOB_NAME="fhp-vr-deep-archieved-exp1-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 \
  432000 \
  8000 \
  32000 \
  100 \
  pd-balanced
```

This is the simplest workflow and automatically produces the cross-algorithm
head-to-head results after training. It also has the largest single-job failure
surface because all six workers run sequentially.

---

## 11. Recommended split-job production workflow

To reduce elapsed time and isolate failures, run the two algorithms
concurrently. Each job trains three sequential 12-hour seeds and uses a
60-hour cap.

Submit VR-DeepDCFR+:

```bash
JOB_DCFR="fhp-vr-deep-archieved-exp1-dcfr-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_DCFR" \
  "python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
    --algorithms vr_deep_dcfr_plus \
    --output-root outputs/cloud/$JOB_DCFR" \
  n2-standard-8 \
  216000 \
  8000 \
  32000 \
  100 \
  pd-balanced
```

Submit VR-DeepPDCFR+:

```bash
JOB_PDCFR="fhp-vr-deep-archieved-exp1-pdcfr-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_PDCFR" \
  "python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
    --algorithms vr_deep_pdcfr_plus \
    --output-root outputs/cloud/$JOB_PDCFR" \
  n2-standard-8 \
  216000 \
  8000 \
  32000 \
  100 \
  pd-balanced
```

Monitor them independently:

```bash
gcloud batch jobs describe "$JOB_DCFR" --location "$REGION"
gcloud batch jobs describe "$JOB_PDCFR" --location "$REGION"
```

Algorithm-only runs intentionally cannot perform the cross-algorithm matchup.
That analysis is generated after both output trees are downloaded.

---

## 12. Download and aggregate split jobs

Download both jobs:

```bash
mkdir -p "cloud_outputs/$JOB_DCFR" "cloud_outputs/$JOB_PDCFR"

gcloud storage cp --recursive \
  "$BUCKET/$JOB_DCFR/outputs" \
  "cloud_outputs/$JOB_DCFR/"

gcloud storage cp --recursive \
  "$BUCKET/$JOB_PDCFR/outputs" \
  "cloud_outputs/$JOB_PDCFR/"
```

Find the precise experiment run directories. Each argument passed to
`--aggregate-run-dir` must be the directory that directly contains
`worker_runs/`:

```bash
find "cloud_outputs/$JOB_DCFR" -path '*/worker_runs/*/result.json' -print
find "cloud_outputs/$JOB_PDCFR" -path '*/worker_runs/*/result.json' -print
```

Set the two directories based on that output, for example:

```bash
export DCFR_RUN_DIR="cloud_outputs/$JOB_DCFR/outputs/cloud/$JOB_DCFR/archieved_exp1_RUN_DIR"
export PDCFR_RUN_DIR="cloud_outputs/$JOB_PDCFR/outputs/cloud/$JOB_PDCFR/archieved_exp1_RUN_DIR"
```

Aggregate and run the default 100,000-deal-pair evaluations:

```bash
python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
  --aggregate-run-dir "$DCFR_RUN_DIR" \
  --aggregate-run-dir "$PDCFR_RUN_DIR" \
  --output-root outputs/archieved_exp1_aggregated
```

The aggregator:

- discovers relocatable `worker_runs/*/result.json` artifacts;
- rejects duplicate algorithm/seed pairs;
- validates snapshot type, version, seed, algorithm, checkpoint, and exact FHP
  game parameters;
- performs matched 6-hour and 12-hour seat-swapped evaluation;
- regenerates all CSV, JSON, and plot artifacts.

For a quick aggregation wiring test, add `--head-to-head-deals 4`. Do not use
that reduced result for scientific reporting.

---

## 13. Recover a missing algorithm or seed

The runner accepts algorithm and seed subsets. If only one worker is missing,
submit just that pair rather than repeating completed work:

```bash
JOB_RECOVERY="fhp-vr-deep-archieved-exp1-recovery-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_RECOVERY" \
  "python -m experiments.fhp.archieved_exp1_leduc_config_transfer.run \
    --algorithms vr_deep_pdcfr_plus \
    --seeds 2 \
    --output-root outputs/cloud/$JOB_RECOVERY" \
  n2-standard-8 \
  64800 \
  8000 \
  32000 \
  100 \
  pd-balanced
```

The 64,800-second cap is 18 hours for one 12-hour effective-training worker.
Download the recovery job and pass its precise experiment directory alongside
the earlier directories to `--aggregate-run-dir`.

Do not combine duplicate algorithm/seed outputs. The aggregator rejects them so
that an accidental rerun cannot silently replace the originally selected seed.

---

## 14. Understand resource and disk arguments

The final six helper arguments are:

```text
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB BOOT_DISK_TYPE
```

Useful N2 combinations are:

| Machine type | CPU milli | Memory MiB | Approximate resources |
|---|---:|---:|---|
| `n2-standard-2` | `2000` | `8000` | 2 vCPUs, 8 GiB |
| `n2-standard-4` | `4000` | `16000` | 4 vCPUs, 16 GiB |
| `n2-standard-8` | `8000` | `32000` | 8 vCPUs, 32 GiB |

The task request must fit the selected machine type. Archived Experiment 1's approved
production reference is `n2-standard-8` with a 100 GiB `pd-balanced` boot disk.
The million-entry replay buffers make the smaller smoke VM unsuitable as an
untested production default.

C4-family VMs do not support Persistent Disk. The helper rejects a C4 machine
combined with a `pd-*` disk before submission. If deliberately testing C4, use
an appropriate Hyperdisk type such as `hyperdisk-balanced` and verify current
regional support and quotas first.

---

## 15. Runtime limits

`MAX_RUN_SECONDS` is a hard Batch safety cap:

| Seconds | Duration | Intended use here |
|---:|---:|---|
| `21600` | 6 hours | smoke-test headroom |
| `64800` | 18 hours | one-seed recovery |
| `216000` | 60 hours | three-seed algorithm job |
| `345600` | 96 hours | shorter all-worker cap, not recommended initially |
| `432000` | 120 hours | full six-worker job |

The solver's 6-hour and 12-hour checkpoints use **effective training time**,
which excludes average-policy fitting and checkpoint serialization. Batch uses
wall-clock time. A 12-hour solver horizon therefore needs more than 12 hours of
Batch duration.

If Batch reaches `maxRunDuration`, it terminates the task. Keep enough headroom
for the final policy fit, snapshot verification, head-to-head evaluation, and
Cloud Storage upload.

---

## 16. Diagnose a failed job

Start with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
```

Then list or download whatever cleanup artifacts were uploaded:

```bash
gcloud storage ls --recursive "$BUCKET/$JOB_NAME/"
```

The job-level diagnostics are:

- `batch_run.log` — bootstrap, installation, experiment, and cleanup output;
- `resource_snapshots.jsonl` — 15-second CPU, memory, process, cgroup, and disk
  snapshots;
- `batch_diagnostics.json` — summarized resource and failure diagnosis;
- `batch_status.json` — exit codes, diagnosis, timestamp, and bucket destination.

Worker-level failures contain `failure.json`, and the parent run records
`failed_runs.json` and `run_status.json` when possible.

Typical evidence:

| Evidence | Likely cause | Response |
|---|---|---|
| `exit code 137`, `Killed`, cgroup OOM event | memory exhaustion | use more memory or inspect buffer allocation |
| `No space left on device` | boot disk full | increase `BOOT_DISK_SIZE_GB` |
| `maxRunDuration` or SIGTERM near cap | timeout | increase the wall-clock cap or recover the missing seed |
| repository clone failure | wrong/private `REPO_URL` | verify URL and authentication |
| dependency resolution/download failure | network or package issue | inspect bootstrap log and pinned requirements |
| experiment traceback | code/configuration failure | inspect worker log and `failure.json` |
| Cloud Storage permission error | missing bucket role | recheck `roles/storage.objectAdmin` |

Do not interpret a partial worker directory as a completed scientific run. A
valid production worker has `stop_reason: training_time_budget`, two checkpoint
snapshots, `summary.json`, and `result.json`.

---

## 17. Inspect performance and right-size later jobs

Use `resource_snapshots.jsonl`, `batch_diagnostics.json`, worker summaries, and
Batch `status.runDuration` rather than guessing.

A VM may be oversized if peak memory is far below 32 GiB, CPU utilization is
persistently low, and a smaller controlled run has equivalent throughput. It
may be undersized if memory approaches the cgroup limit, swap pressure appears,
workers are killed, or training-node throughput collapses.

Keep the first production experiment on the approved `n2-standard-8` reference
VM. Treat later VM changes as systems experiments and record them in the run
manifest rather than mixing them into an algorithm comparison unnoticed.

---

## 18. Clean up

Batch VMs are temporary and should terminate when jobs finish or fail. Delete a
completed or failed Batch job record when it is no longer needed:

```bash
gcloud batch jobs delete JOB_NAME --location "$REGION" --quiet
```

Inspect bucket contents before deleting anything:

```bash
gcloud storage ls --recursive "$BUCKET/"
```

Delete one job's uploaded folder only when its outputs are safely retained
elsewhere:

```bash
gcloud storage rm --recursive "$BUCKET/JOB_NAME/"
```

Do not delete the entire results bucket unless every required checkpoint,
manifest, log, table, and plot has been preserved.

---

## 19. Why the helper uses isolated Python environments

The Batch script installs `uv`, gives the Cloud SDK a Python 3.10 runtime, and
creates a separate Python 3.11 environment for PyTorch, OpenSpiel, and this
repository:

```text
/tmp/fhp-vr-deep-venv
```

Keeping these runtimes separate prevents experiment dependencies from breaking
the Google Cloud CLI used during final artifact upload. The environment is
temporary and disappears with the Batch VM.

---

## 20. Official Google Cloud references

- [Batch quickstart and required IAM roles](https://docs.cloud.google.com/batch/docs/create-run-example-job)
- [`gcloud batch jobs submit` reference](https://docs.cloud.google.com/sdk/gcloud/reference/batch/jobs/submit)
- [Batch roles and permissions](https://docs.cloud.google.com/iam/docs/roles-permissions/batch)
- [`gcloud storage buckets add-iam-policy-binding` reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/buckets/add-iam-policy-binding)
- [`gcloud storage cp` reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp)
