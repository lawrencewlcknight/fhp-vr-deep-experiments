# Experiment 5: eight Ray traversal workers, unchanged central fitting

This experiment compares **synchronous traversal parallelism** with Experiment
3. It is not a distributed learner: all regret, critic and average-policy
fitting stays in one central process with the existing eight computation
threads and unchanged update rules.

## Fixed control and training schedule

Use the Experiment 3 configuration at `96c749410fff8ab89cda29ef28974124cb155d98`:

- Three independent seeds, 0/1/2, each on its own `n2-standard-16` Batch VM.
  Thus there are eight traversal actors **per seed**, not eight shared across
  the three seeds. Full training requires 48 N2 vCPUs, plus the controller.
- 24 active training hours per seed, playable policy checkpoints at 6, 12, 18
  and 24 hours, stopping after the first completed outer iteration crossing
  the final threshold. The inherited 36-hour worker safety limit is unchanged.
- The same lossless FHP encoder, 64–64–64 networks, 2,048-row minibatches,
  one-million-entry replay capacities, learning rates and fitting counts.
- **10,000 traversals per player per iteration in total**, split into 1,250
  for each of eight actors. The traversal budget is not multiplied by eight.
- No initial/intermediate exploitability, benchmark-opponent, LBR or head-to-head
  evaluation. Policy quality will be evaluated separately. Checkpoint policies
  use the existing loader and are not full resumable training states.

## Collection and update order

For each outer iteration:

1. Broadcast the current regret/immediate-regret networks and the current
   **online critic** to eight persistent actors. The critic's TD target is not
   used in place of the online critic during traversal.
2. Each actor collects its share of player 0's trajectories using the existing
   OpenSpiel `dfs`, control variates and target construction, without fitting.
3. Wait for every actor. Validate counts, snapshot version, replay columns,
   shapes and finite values; merge in ascending actor ID, never arrival order.
4. Run the existing central player-0 regret fit and critic fit, unchanged.
5. Broadcast the updated networks, collect player 1's trajectories in parallel,
   merge, then run the existing central player-1 fits.
6. At the established iteration boundary, emit diagnostics and take any due
   time checkpoint, including the unchanged central average-policy fit.

Actors are idle during fitting. There is no overlap between collection and
learning, no policy lag, no concurrent learner updates and no change to target
cache, optimizer, reset or synchronization rules. The implementation overrides
collection; the traversal and fit methods are inherited from Experiment 2/3.

Each actor is a distinct process with one Ray CPU reservation and one
Torch/OpenMP/MKL/OpenBLAS computation thread. The central learner retains eight
threads. This follows the persistent actor pattern already used for
UCV-ESCHER, adapted to VR-Deep's single online critic and ordered player updates.
Ray is pinned to **2.51.2**, matching the UCV-ESCHER repository. Ray actor
resources and restart/retry settings follow its
[actor API](https://docs.ray.io/en/latest/ray-core/actors.html) and
[actor fault-tolerance contract](https://docs.ray.io/en/latest/ray-core/fault_tolerance/actors.html).

## Replay and randomness safeguards

Workers return **all** collected regret, policy and critic records. They do not
maintain or subsample independent million-row replay buffers. Their bounded
append-only staging uses the game's maximum path length times the assigned
traversal count, and raises an error rather than dropping or overwriting data.
Only populated float32/int64 arrays are transferred, including separately
encoded 183-value next-player information for the 263-value critic input.

The driver alone applies the original reservoir insertion rule and critic
circular-buffer retention, in deterministic worker order. Initial reservoir
fills and ring insertion use bulk copies; overflow reservoir draws retain the
existing row-by-row sampling rule. Workers' local data are not treated as
equally weighted reservoirs, which would introduce an avoidable sampling bias.

Each collection uses a deterministic random stream keyed by run seed, worker
ID, iteration and player. Infrastructure startup and actor collection cannot
consume the central learner's post-initialization RNG state. This makes worker
scheduling irrelevant to the sampling order. It does **not** produce the same
trajectory sequence as Experiment 3's single global stream; central replay
arrival order also differs. One-thread worker inference can introduce small
floating-point differences from the sequential eight-thread collector.

Accordingly, this is an algorithm-preserving execution comparison, not a claim
of bitwise equivalence between full sequential and parallel learning runs.
Tests establish exact worker-vs-existing-DFS equivalence for matched networks
and random streams, and exact replay insertion equivalence for matched rows.
Compare aggregate throughput and phase timings across the same three seed
labels; do not treat those labels as identical sampled trajectories.

## Resources, stopping and diagnostics

Each seed uses the same allocation as Experiment 3: 16,000 CPU milli, 62,000
MiB task memory and a 200 GiB `pd-balanced` boot disk. Ray owns a local runtime
with eight actor CPU slots. Its object store is explicitly limited to 2 GiB
(256 MiB for smoke), rather than taking a default fraction of VM memory.

Actor readiness has a five-minute timeout after runtime initialization; each
collection barrier has a 30-minute timeout. Actor restarts, task retries and
Batch retries are disabled. Missing/failed actors abort the run rather than
silently reduce the traversal budget. Runtime startup, collection failures,
fitting failures and normal completion clean up the owned Ray runtime. The
checkpoint runner retains previous valid checkpoint uploads on failure.

The existing independent CPU/memory/cgroup-OOM monitor runs on each training
VM. Driver tracebacks and Ray errors are available in Cloud Logging and
`failure.json`. `parallel_runtime.json` records the actual eight actor PIDs,
thread configuration, Ray version, staging bounds, object-store allocation
and startup duration, and is included in the verified success inventory.

The active clock includes snapshot creation, transfer/waiting, traversal,
response validation and replay merging. Actor startup is excluded, just as
solver construction is excluded from the control's training clock; it is
reported separately. Checkpoint fitting, writing, validation and uploading
remain excluded, exactly as in Experiment 3.

Use these outputs to assess the benefit:

- `summary.json`: total training nodes, completed iterations, nodes per active
  second, central phase timings and driver peak RSS.
- `training_progress.jsonl` and checkpoint rows: cumulative
  `parallel_snapshot_seconds`, `parallel_dispatch_seconds`,
  `parallel_wait_seconds`, `parallel_validation_seconds`,
  `parallel_merge_seconds`, worker collection/CPU/snapshot times, imbalance,
  returned regret/policy/critic row counts, peak response bytes and peak
  individual actor RSS.
- `analysis/training_progress.csv` and `seed_summary.json`: consolidated data
  for comparison with Experiment 3. The usual nodes-versus-active-hours plot
  is retained. Checkpoint manifests provide the policies for later evaluation.

Worker CPU seconds are summed work, not elapsed wall time; per-actor peak RSS
is not simultaneous whole-job peak memory. Use independent resource snapshots
for the latter. A faster collector does not imply an eightfold end-to-end
speedup: central fitting and communication can dominate. Policy quality per
hour requires the separate evaluation framework.

## Local validation

Use Python 3.11 in the project's environment and install the optional Ray
requirements (Experiments 1–4 still need only their original requirements):

```bash
python -m pip install -r requirements-ray.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_vr_deep_parallel.py tests/test_exp5_vr_deep_ray8.py

# Requires permission to create local processes and communication sockets.
FHP_RUN_RAY_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_exp5_vr_deep_ray8.py -k real_eight_actor
FHP_RUN_RAY_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_exp5_vr_deep_ray8.py -k real_ray_production_minibatch
bash gcp/run_exp5_vr_deep_ray8.sh smoke-local
```

The cloud smoke gate enables the real eight-actor test; it cannot pass by
substituting the inline test transport. It also runs the existing central
fitter equivalence benchmark, a bounded 2,048-row-minibatch Ray iteration test,
then a complete eight-actor checkpoint smoke.
Smoke assigns two traversals to every actor for each player and verifies all
four policy files can be reloaded before any full seed job is submitted.

## GCP Batch smoke and full run

From this repository, with `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` set.
Commit and push the implementation first: Batch clones the pinned remote SHA.

```bash
export REPO_REF="$(git rev-parse HEAD)"

# Optional standalone cloud smoke; does not submit full training.
export RUN_ID="vr5-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp5_vr_deep_ray8.sh smoke-only
bash gcp/run_exp5_vr_deep_ray8.sh status

# After smoke succeeds, use a NEW namespace for the full workflow.
export RUN_ID="vr5-ray8-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp5_vr_deep_ray8.sh run
bash gcp/run_exp5_vr_deep_ray8.sh status
```

The full workflow automatically performs its own smoke gate, launches three
independent seed VMs, and aggregates only after all three succeed. The laptop
may disconnect after submission. Training and Ray processes stop at completion;
Batch manages VM shutdown. No evaluation jobs are submitted. Use `dry-run` to
inspect the job definitions without submitting.

Download analysis after success:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Seed artifacts and policies remain under `$BUCKET/$RUN_ID/workers/`; resource
diagnostics are under `$BUCKET/$RUN_ID/diagnostics/`. Never reuse a run namespace
or restart a failed seed from a policy-only checkpoint as though it were a
complete optimizer/replay state.
