# Experiment 3: Experiment 2 on n2-standard-16

## Question and controlled comparison

Does a larger VM allow the unchanged VR-DeepPDCFR+ learner to complete more
training in 24 active hours? This is a **VM-only comparison** against active
VR-Deep Experiment 2 at commit `247b9a3cced860b601f3d6b6d43a36d4a54d06d5`.
The matching UCV-ESCHER hardware study is **Experiment 6**, not its Experiment 3
(which increased network width).

Only experiment identifiers and the training/smoke VM allocation change.
The exact Experiment 2 solver is reused: no new parallel traversal, altered
update rules, larger networks, larger replay or additional features.

| Setting | Experiment 3 |
|---|---|
| Algorithm | VR-DeepPDCFR+, unchanged from Experiment 2 |
| Training seeds | 0, 1, 2, each starting from scratch |
| Training machine | One on-demand `n2-standard-16` per seed, three concurrently |
| Batch allocation | 16 vCPUs, 62,000 MiB requested on the 64 GiB VM |
| Boot disk | Unchanged 200 GiB pd-balanced |
| Within-seed execution | Serial traversal; **eight** Torch/OMP/MKL/OpenBLAS threads |
| Inputs | Unchanged lossless suit-canonical encoder: 183 policy/regret, 263 critic |
| Networks | Unchanged flat 64-64-64 MLPs |
| Budget | **24 active training hours per seed** |
| Checkpoints | **6, 12, 18 and 24 active hours** |
| Retention | Twelve playable policies, diagnostics and metadata; no full training states |
| Performance evaluation | **None; deferred to a separate evaluation experiment** |

The full configuration is inherited unchanged, including 10,000 traversals per
player/iteration; 750 cumulative and immediate regret updates; 10,000 configured
critic steps (the inherited loop executes 10,001); 5,000 reset average-policy
updates per checkpoint; 2,048 minibatches; one-million-entry replay capacities;
Adam 0.001; alpha=2.3, gamma=2.0 and epsilon=0.6. Policy fitting retains
iteration-weighted probability MSE. Tests check configuration equality with
Experiment 2 and verify that both experiments use the same solver factory.

Keeping eight fitting threads is intentional: increasing threads would introduce
a second change. Extra CPUs need not produce a speed-up, especially during
serial traversal. More nodes or lower training loss do not establish a stronger
poker policy; that requires later evaluation of the retained checkpoints.

## Clock and outputs

Both players finish the outer iteration crossing each threshold before the
checkpoint is fitted. Training stops at the completed iteration crossing 24h.
Policy fitting, saving, reload checks and uploads remain outside the active
clock and preserve training RNG state. Setup and checkpoint work are billable
overhead, so this is **not a hard 24-hour VM lifetime**. Training tasks retain
the 36-hour wall-time ceiling, with no automatic retries; the controller retains
its 72-hour ceiling. Timeouts are safeguards, not additional training budgets.

No exact exploitability, rule-agent, LBR, exploiter or head-to-head evaluation is
implemented or launched. Average-policy fitting produces playable snapshots;
reload/distribution checks are artifact validation, not playing-strength tests.
The existing diagnostic record may contain empty evaluation placeholders.

Each worker retains nodes, iterations, active/wall times, threshold overshoot,
per-phase fitting/collection times, losses, replay occupancy and peak RSS.
Independent resource monitoring retains CPU counters, load, memory/cgroup OOM
events, disk use and exit/failure diagnostics, with the updated memory request.
Aggregation produces JSON/CSV training summaries and a nodes-versus-time plot,
not a performance ranking. Compare per-phase throughput and recorded runtime
versions as well as nodes when comparing with historical Experiment 2 outputs.

The output layout is the same as Experiment 2:

```text
$BUCKET/$RUN_ID/
  smoke/
  workers/task_000_lossless_vr_deep_pdcfr_plus_n2_16_seed_0/  # also seeds 1, 2
    checkpoints/*.pt
    checkpoint_manifest.json/csv
    checkpoint_rows.json/csv
    run_manifest.json
    training_progress.jsonl
    summary.json
    SUCCESS.json
  diagnostics/
  analysis/
```

The run manifest records the larger VM contract and unchanged thread/learner
settings. Aggregation rejects mismatched identities, contracts, missing seeds
and corrupt policy artifacts. Policies remain compatible with the native
encoded-policy loader and `fhp_vr_deep.evaluation_adapter` for later evaluation.

## GCP Batch instructions

Run from the **VR-Deep repository**, with `PROJECT_ID`, `REGION`, `BUCKET` and
the VR-Deep `SA_EMAIL` already set. Use the existing IAM setup in
[the Batch guide](../../../docs/GCP_BATCH_EXPERIMENTS.md). Commit and push the
implementation first; launchers reject older commits missing Experiment 3.

### Optional local smoke (no cloud charge)

```bash
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh smoke-local
```

This uses the selected Python environment (`PYTHON` may override `python3`),
tiny replay/fit budgets, one seed and one thread. It exercises all four
checkpoint hooks and aggregation, not production-duration performance.

### Standalone cloud smoke (no full training submitted)

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr3-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh smoke-only
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh status
```

Cloud smoke uses one `n2-standard-16`, runs regression/equivalence tests at the
existing thread settings, fits tiny encoded checkpoints and reloads them. It
does not measure production-capacity peak memory or a 24-hour trajectory.
Wait for the smoke job to succeed before choosing to launch the full run.

### Full three-seed run

Use a **new RUN_ID**, distinct from the standalone smoke namespace:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr3-vm16-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh run
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh status
```

The remote workflow is `controller -> cloud smoke -> train (three VMs) ->
aggregate`. The full workflow runs its own smoke gate even if a standalone
smoke was previously run. Your laptop can disconnect after submission.
Three workers require **48 available regional N2 vCPUs**, plus the small E2
controller. Aggregation remains on `n2-standard-8`; neither auxiliary stage
changes the training comparison. Workers exit after their artifacts upload,
and Batch manages VM cleanup. No permissions or quotas are changed automatically.

To inspect job configurations without submitting anything:

```bash
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh dry-run
```

### Download or recover aggregation

With `RUN_ID` still identifying the completed full run:

```bash
VR_RESULTS_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "outputs/$RUN_ID/analysis" "outputs/$RUN_ID/workers"
gcloud storage rsync --recursive "$VR_RESULTS_BUCKET/$RUN_ID/analysis" "outputs/$RUN_ID/analysis"
# Optional: download all playable policies and worker metadata too.
gcloud storage rsync --recursive "$VR_RESULTS_BUCKET/$RUN_ID/workers" "outputs/$RUN_ID/workers"
```

If training succeeded but aggregation failed, retain the original `RUN_ID`
and pinned `REPO_REF`, then run:

```bash
bash gcp/run_exp3_vr_deep_lossless_n2_standard16.sh aggregate-only
```

This repeats analysis only, never training. Failed/interrupted training cannot
resume from the policy-only snapshots; a rerun requires a new RUN_ID. Existing
cloud prefixes and local worker directories are never silently overwritten.

## Regression checks

In the repository's Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

The cloud smoke runs the Experiment 3, Experiment 2, Experiment 1 and efficiency
tests using its Python 3.11 environment before any full training is submitted.
