# Experiment 2: UCV Experiment 2 input representation for VR-DeepPDCFR+

Does the lossless suit-canonical representation improve VR-DeepPDCFR+ on FHP?
This is an **encoder-only transfer**, relative to active VR-Deep Experiment 1
at commit `af3b3b47af54217d0fb22499d9aaf31730a3a7ce`. It does not transfer the
UCV learner, branched networks, larger hidden layers, critic ensemble, grouped
policy fitting or cross-entropy loss.

## Frozen configuration

| Setting | Experiment 2 |
|---|---|
| Seeds / budget | 0, 1, 2; 24 active training hours each |
| Machines | Three separate `n2-standard-8` VMs, concurrently |
| Within-seed execution | Serial traversals, 8 Torch/OMP/MKL threads |
| Regret and policy input | Exact UCV Exp.2 player encoder, 183 values (formerly 190) |
| Critic input | Exact UCV Exp.2 full-state encoder, 263 values (formerly 380) |
| Hidden layers | Unchanged flat 64–64–64 MLPs in every network |
| Traversals | 10,000 per player per outer iteration |
| Regret / immediate-regret updates | 750 / 750 per player per iteration |
| Critic updates | 10,000 configured; inherited inclusive loop executes 10,001 |
| Average-policy fitting | 5,000 reset updates at each checkpoint; iteration-weighted probability MSE |
| Batch sizes | 2,048 throughout |
| Replay capacities | 1m per-player regret, 1m average policy, 1m critic |
| Adam learning rate / weights / exploration | 0.001; alpha=2.3, gamma=2.0; epsilon=0.6 |
| Checkpoint targets | 6, 12, 18, 24 active hours |
| Retention | All 12 playable policies, analysis and metadata; no full training states |

All training hyperparameters, target refreshes, fitting/reset rules, RNG
isolation, machine resources and checkpoint semantics come from Experiment 1.
Changing the input dimension necessarily changes the input-layer parameter
count; this is not a parameter-count-matched architecture study. The hidden
architecture remains fixed. Learned outputs are expected to differ: this is
a representation experiment, not an implementation-equivalence claim.

### What the representation contains

Exact cards and exact decision-state betting history remain distinguishable,
apart from a global renaming of suits. The representation adds rank and suit
counts, pair/suitedness flags, and a five-card hand-category indicator. It uses
neither strength buckets nor sampled equity estimates. See
[encoder provenance](../../../docs/EXP2_ENCODER_PROVENANCE.md).

Player inputs depend only on that player's cards, the public board and public
betting history. The critic retains privileged access to both private hands,
as it already did in Experiment 1. Its suit canonicalisation is separate from
the player's: **the next player's policy input must not be sliced from the
critic vector**. Critic replay therefore stores the separately encoded
183-value next-player input. This storage change is necessary for correctness;
it is not an extra feature or change to the TD target.

The encoder itself is vendored, so training requires no UCV repository or UCV
outputs. Its bounded 4,096-entry caches reuse encodings within traversal work.
The float32 replay and fit-local frozen-prediction optimisations remain enabled.

## Timing and interpretation

Checkpoint fitting happens at the first completed outer iteration crossing
each threshold, after both players have updated. Fitting, saving, reload
validation and uploads are outside the active clock. All overhead and threshold
overshoot are reported. Completion therefore takes longer than 24 elapsed
hours. Each training task retains Experiment 1's 36-hour wall-time ceiling.

The intended comparison is Experiment 2 versus Experiment 1 at the same active
time, reporting nodes, phase times, wall time and memory as well as subsequent
policy quality. The encoder may improve generalisation while costing additional
feature-computation time; improvement is a hypothesis, not a promised result.
The previous UCV representation result changed the network architecture too,
so it does not isolate this feature transfer's effect on VR-Deep.

**No exact exploitability or cross-algorithm quality evaluation is run here.**
The saved policies support subsequent common-deal rule-agent, LBR, exploiter
and head-to-head evaluation through the shared evaluation suite. Use
`fhp_vr_deep.evaluation_adapter.load_policy_for_evaluation` or
`evaluate_checkpoint` for these version-3 snapshots: the suite's existing
generic `fhp-evaluate ... SNAPSHOT` loader assumes raw OpenSpiel inputs and
must not be used for this encoder without its own adapter update. The VR
adapter loads both raw and encoded policies, so evaluation remains comparable.

## Launch and checks

The workflow is `controller -> cloud smoke -> three training VMs -> aggregate`.
The smoke gate verifies pinned UCV reference vectors, suit symmetry,
private-information boundaries, encoded cached/uncached fitting, checkpoint
scheduling and reloads. It also retains Experiment 1's short raw-solver
equivalence check against the pinned pre-optimisation implementation. These
are regression tests, not a proof of long-run numerical equivalence.

After committing and pushing this implementation, with `PROJECT_ID`, `REGION`,
`BUCKET` and the VR-Deep `SA_EMAIL` already set:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr2-encode-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp2_vr_deep_lossless_24h.sh run
```

This requires 24 available N2 vCPUs for three eight-vCPU training VMs, plus the
small E2 controller. Use the same IAM setup as
[Experiment 1](../exp1_vr_deep_pdcfr_24h/README.md). The wrapper rejects a stale
commit missing the new experiment. It does not create accounts, change IAM or
overwrite an existing output namespace. No automatic training retries occur;
lightweight policy snapshots are not resumable learner states.

```bash
bash gcp/run_exp2_vr_deep_lossless_24h.sh dry-run
bash gcp/run_exp2_vr_deep_lossless_24h.sh status
# Optional local test; the cloud smoke always runs first:
bash gcp/run_exp2_vr_deep_lossless_24h.sh smoke-local
```

## Outputs

Experiment 1's output contract is retained under this run's dedicated prefix:

```text
smoke/                           Smoke and equivalence results
workers/task_000_lossless_vr_deep_pdcfr_plus_seed_0/  (also seeds 1 and 2)
  checkpoints/*.pt               Four playable version-3 policies per seed
  checkpoint_manifest.json/csv   Relative paths, hashes and sizes
  checkpoint_rows.json/csv       Nodes, times, losses and threshold overshoot
  training_progress.jsonl        Per-iteration fitting/resource diagnostics
  run_manifest.json              Frozen configuration and encoder provenance
  summary.json                   Throughput, input sizes, model/replay sizes
  SUCCESS.json                   Verified inventory, uploaded last
analysis/                        Three-seed summaries and nodes/time plot
diagnostics/                     CPU/memory monitor
```

Aggregation verifies all three workers, encoder identity, frozen configuration,
source commit consistency, artifact hashes, completed time budgets and playable
reloads. It cannot silently combine raw Experiment 1 policies with encoded
Experiment 2 policies. It does not infer policy quality from losses or nodes.

```bash
RESULTS_BUCKET="gs://${BUCKET#gs://}"
mkdir -p "outputs/$RUN_ID/analysis" "outputs/$RUN_ID/workers"
gcloud storage rsync --recursive "$RESULTS_BUCKET/$RUN_ID/analysis" "outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive "$RESULTS_BUCKET/$RUN_ID/workers" "outputs/$RUN_ID/workers"
# If only aggregation fails, no retraining is needed:
bash gcp/run_exp2_vr_deep_lossless_24h.sh aggregate-only
```
