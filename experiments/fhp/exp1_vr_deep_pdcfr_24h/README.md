# Experiment 1: selected VR-DeepPDCFR+ on FHP, 24 active hours

This is the new active Experiment 1. The former two-algorithm study remains
unchanged under `experiments/fhp/archieved_exp1_leduc_config_transfer`.

## Research question and frozen configuration

Does the retained Leduc VR-DeepPDCFR+ configuration transfer to canonical FHP?
This is a baseline, not a new architecture or hyperparameter search. The
optimised implementation is used with a cloud numerical-equivalence gate.
VR-DeepDCFR+ is not an arm of this experiment.

| Setting | Value |
|---|---|
| Independent training seeds | 0, 1, 2 |
| Budget per seed | 24 active training hours |
| Training VM | One `n2-standard-8` per seed, all three in parallel |
| Resource request | 8 vCPUs, 30,000 MiB; 200 GiB balanced boot disk |
| Compute | CPU, 8 Torch/OMP/MKL threads; serial traversal within each seed |
| Hidden layers | 64, 64, 64 for each network |
| Learning rate | Adam, 0.001 |
| Traversals | 10,000 per player per outer iteration |
| Cumulative / immediate regret updates | 750 / 750 per player per iteration |
| Critic fitting | 10,000 configured steps; retained inclusive loop executes 10,001 |
| Average-policy fitting | Reset network and optimiser; 5,000 updates per checkpoint |
| Policy objective | Iteration-weighted probability MSE, unchanged from Leduc |
| Minibatch size | 2,048 for every fit |
| Replay capacities | 1 million per-player regret, 1 million policy, 1 million critic |
| Regret weighting / exploration | alpha=2.3, gamma=2.0, epsilon=0.6 |
| Reset rules | Warm cumulative weights/Adam; reset immediate network/Adam; existing critic reset |
| Input | Raw OpenSpiel information-state tensor (190); critic history is both players' tensors (380) |
| Representation changes | None: no canonical-card or curated feature additions |
| Checkpoint targets | 6, 12, 18, 24 active hours |

The inherited algorithm's replay-reset and ring-buffer semantics are retained;
these are capacity limits, not a change to which historical samples are kept.
Float32 replay and fit-local frozen-prediction caches are implementation
optimisations. The refreshing critic TD target is not frozen across refreshes.

## Clock, checkpoints and comparability

Each checkpoint is captured at the first **completed outer iteration** crossing
its threshold, after both players' traversal, regret fitting and critic fitting.
The final threshold stops training. Actual active time, wall time, iteration,
node count and threshold overshoot are recorded. The one-million-iteration
safety cap is not the intended stopping rule; hitting it fails completion checks.

Average-policy fitting runs only at these four checkpoints. Fitting, policy
persistence, reload validation and uploads are excluded from the active clock,
and checkpoint operations preserve Python/NumPy/Torch training RNG states.
Solver initialisation and environment setup are outside the active clock too.
Consequently this takes **more than 24 elapsed hours**; each training task has a
36-hour wall-time ceiling including setup and checkpoint work. The controller
has a 72-hour ceiling. These ceilings are failure safeguards, not convergence
criteria or runtime guarantees.

The seed labels, FHP game, machine class, 24-hour budget and four checkpoint
targets align with UCV-ESCHER Experiment 1. Node counts remain algorithm-specific;
equal node counts are not equivalent floating-point work. Excluding output-policy
fitting permits training-clock comparisons, not equal end-to-end deployment-cost
claims. Retain both active and wall time when comparing methods.

**No exact exploitability is computed.** These outputs establish throughput and
provide playable policies, not a claim of equilibrium convergence. Quality
evaluation is deferred to the existing shared `fhp-evaluation-suite`, through
`fhp_vr_deep.evaluation_adapter`; it is not duplicated in this experiment or run
inside the training clock. The snapshots support the same rule-agent, LBR,
trained-exploiter and head-to-head evaluators. A subsequent evaluation should
match the UCV comparison's evaluator version, deal budgets, seed handling and
units. Neither UCV nor SD-CFR outputs are required to launch this training run.

## Cloud workflow

`controller -> smoke/equivalence -> train (3 VMs) -> aggregate`

The cloud smoke tests scheduling, artifact reloads and contract safeguards, then
compares the optimised solver with the pinned pre-optimisation implementation
over seeds 0, 1 and 2 at the production 2,048 batch size and eight threads.
This verifies parameters, optimisers, sampled replay, RNG and non-timing results
on a short fixed-work run; it is not a proof for every future trajectory.
Any gate failure prevents the expensive training stage. The complete Git
history must be available for the reference commit.

The three workers have `taskCountPerNode=1`; they are three VMs, **24 N2 vCPUs**
in total, plus a small E2 controller. No distributed traversal or distributed
fitting within a seed is introduced. On-demand VMs are used, with automatic
task retries disabled because there are no resumable full training states.

After the implementation has been committed and pushed, from this repository:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr1-24h-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp1_vr_deep_pdcfr_24h.sh run
```

The terminal must already have `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`
set for this repository, typically `fhp-vr-deep-runner@PROJECT.iam.gserviceaccount.com`.
`BUCKET` accepts either a bare bucket name or `gs://bucket`. Avoid stale
`REPO_REF` values carried over from another experiment. The wrapper checks that
the selected commit actually contains the new files. A successful submission
runs remotely; the laptop can disconnect.

The existing setup in `docs/GCP_BATCH_EXPERIMENTS.md` covers Batch agent,
logging, storage and submitter permissions. This controller additionally needs
the following one-time permissions to submit its children (run only if needed,
using an account authorised to administer IAM):

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/batch.jobsEditor"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
```

Submission checks service-account existence, bucket access and a fresh output
namespace. Remote child-job permission errors fail visibly; the launcher never
creates accounts or changes IAM automatically.

```bash
bash gcp/run_exp1_vr_deep_pdcfr_24h.sh status
```

For inspection without cloud calls or submission:

```bash
bash gcp/run_exp1_vr_deep_pdcfr_24h.sh dry-run
```

A local smoke is optional; the cloud smoke always runs first. With the project's
dependencies installed in Python 3.11:

```bash
bash gcp/run_exp1_vr_deep_pdcfr_24h.sh smoke-local
```

## Outputs and recovery

Storage layout beneath `$BUCKET/$RUN_ID`:

```text
smoke/                          Small smoke and fixed-work equivalence outputs
workers/task_000_..._seed_0/     (also task_001 seed 1, task_002 seed 2)
  checkpoints/*.pt              Four playable average-policy snapshots
  checkpoint_manifest.json/csv  Relative paths, SHA-256 hashes and timing
  checkpoint_rows.json/csv      Losses, nodes, active/wall time, overshoot
  training_progress.jsonl       Per-iteration training/resource diagnostics
  run_manifest.json            Frozen config, commit, runtime and provenance
  summary.json                 Timing, throughput and replay occupancy
  SUCCESS.json                 Complete worker inventory, uploaded last
analysis/                       Three-seed JSON/CSV summaries and nodes/time plot
diagnostics/                    Independent memory/CPU monitor per seed
```

All 12 playable policy files are retained. **No full training states, optimiser
states or replay reservoirs are persisted**, including at 24 hours. These policy
snapshots enable subsequent evaluation but cannot resume learner training. This
keeps artifacts far smaller than the earlier multi-GB replay archives; actual
file sizes are recorded in the snapshot manifests.

The aggregator requires all three correctly identified, completed workers,
checks config and commit consistency, verifies every artifact hash and reloads
all policies. It reports seed means and standard errors; smoke data cannot be
aggregated as production evidence. Analysis is safe to rerun after downloading.

To download only the analysis (not policies):

```bash
RESULTS_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive "$RESULTS_BUCKET/$RUN_ID/analysis" "outputs/$RUN_ID/analysis"
```

To retain the playable policies and all worker metadata too:

```bash
gcloud storage rsync --recursive "$RESULTS_BUCKET/$RUN_ID/workers" "outputs/$RUN_ID/workers"
```

If only aggregation fails, it can be rerun on GCP without retraining:

```bash
bash gcp/run_exp1_vr_deep_pdcfr_24h.sh aggregate-only
```

A failed training seed requires an explicit recovery decision; no automatic
24-hour restart or overwrite is performed. Existing playable checkpoints remain
available for evaluation even if the worker never reaches its final checkpoint.
