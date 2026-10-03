# Experiment 6: 48-hour Ray training with a resumable final state

This is a duration-only replication of Experiment 5 at
`d7001548801a58578bcfe458f3d29eb00fadbc22`. It starts from scratch, using seeds
0/1/2 on three independent `n2-standard-16` VMs. Each seed has eight persistent
single-threaded Ray traversal actors and the unchanged eight-thread central
learner. Networks remain 64–64–64, minibatches 2,048, replay capacities one
million, and traversals 10,000 **total per player**, not per actor. All encoder,
cache, critic refresh, fitting and sampling settings remain Experiment 5's.

There is no exploitability, head-to-head or benchmark-opponent evaluation.
Use the saved policies in a separate evaluation, particularly this run's own
24-hour versus 48-hour policies. Longer training alone does not establish an
improvement in policy quality or convergence.

## Clock and artifacts

- Train for **48 cumulative active hours** per seed.
- Save playable policies at **6, 12, 18, 24, 30, 36, 42 and 48 hours**.
- Stop at the first completed outer iteration crossing the final threshold.
  Report actual active time and overshoot; do not stop halfway through a fit.
- Active time includes collection, transfer/validation/merge and central
  regret/critic fitting. It excludes startup, checkpoint-policy fitting,
  persistence, validation, uploads and any downtime between continuations.
- Save **one full training state at the final boundary**, after checkpoint
  RNG restoration. Earlier policy snapshots are NOT resumable training states.
- Verify the archive, upload it, then publish the worker SUCCESS marker last.
  Any save/upload failure fails the job; it must not masquerade as success.

Each worker's `training_state/` directory contains:

- `learner.pt`: all network/target/best-network and optimizer states, Python /
  NumPy / Torch RNG, iteration/node/traversal counters, diagnostic history and
  cumulative phase/parallel counters.
- Separate `.npy` replay columns, containing only populated rows in physical
  buffer order. The manifest retains capacities, reservoir seen counts and
  circular write positions. Saving uses array views and restore streams mmap
  chunks into preallocated buffers rather than duplicating the entire replay.
- `manifest.json`: file sizes and SHA-256 hashes, game/encoder/training config,
  seed, code/runtime/thread contracts, actual active time, completed target and
  continuation provenance.

Actor processes are recreated, not pickled. Their phase-keyed seed scheme uses
the restored run seed, actor ID, iteration and player. Temporary inference and
feature caches are rebuilt; no learning replay is reset. Resume checks require
the source code commit, dependencies, platform architecture and learner threads.
Numerical restart equivalence is tested on the validated CPU runtime, not
promised across different CPU hardware or BLAS implementations.

Full states are several GB per seed. They are independent of policy files and
must be downloaded as an entire directory. Only load trusted archives: integrity
hashes do not make arbitrary Python/PyTorch pickle files safe.

## GCP Batch smoke and full run

Set `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL` as for Experiment 5. The full
workflow uses 48 N2 vCPUs concurrently plus a small controller. Commit and push
this implementation first; Batch clones the pinned remote SHA.

```bash
export REPO_REF="$(git rev-parse HEAD)"
unset EXP6_RESUME_RUN_ID EXP6_START_HOURS EXP6_TARGET_HOURS

# Optional standalone cloud smoke: no full training is submitted.
export RUN_ID="vr6-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp6_vr_deep_ray8_48h.sh smoke-only
bash gcp/run_exp6_vr_deep_ray8_48h.sh status

# Full workflow; uses its own smoke gate before launching the three seeds.
export RUN_ID="vr6-ray48-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp6_vr_deep_ray8_48h.sh run
bash gcp/run_exp6_vr_deep_ray8_48h.sh status
```

The cloud gate tests eight real actors, all eight compressed-time policies,
full-state reload and continuation equivalence (including production-sized
minibatches). It also retains Experiment 5's tests and central-fitter equivalence
benchmark. Smoke checkpoints are flagged and cannot be resumed as production.

Worker wall-time safety limit: **72 hours**. Controller limit: **96 hours**.
The training target is still 48 active hours. Retain the existing memory/OOM,
CPU, process, Ray worker and phase diagnostics. No automatic worker or Batch
retries are enabled; there is no mid-run recovery archive. Ray shuts down on
normal completion or failure and Batch manages the VM lifecycle.

Use `dry-run` to inspect configurations without cloud submission. `smoke-local`
is a functional check; it does not establish GCP throughput.

## Continue all three seeds, for example from 48 to 72 hours

Use a **fresh RUN_ID**, the original source code SHA and the same bucket. The
source run must be a completed Experiment 6 run with verified final states for
all three seeds. Experiment 5's policy-only files cannot be used here.

```bash
export REPO_REF="<full commit SHA recorded in the source run>"
export EXP6_RESUME_RUN_ID="<completed Experiment 6 RUN_ID>"
export EXP6_START_HOURS=48
export EXP6_TARGET_HOURS=72
export RUN_ID="vr6-continue-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp6_vr_deep_ray8_48h.sh run
bash gcp/run_exp6_vr_deep_ray8_48h.sh status
```

This still gates on cloud smoke. Each worker downloads its own source state,
validates its hashes and compatibility, restores it, and continues to **72 total
active hours**, not 72 additional hours. New policies are at 54/60/66/72 hours
and a new final resumable state is saved at 72. Metadata links the source run
and manifest hash; old outputs are never overwritten. The original run retains
its earlier policies. Analysis in each continuation contains that segment's
policy checkpoints and cumulative counters, with its source recorded explicitly.

Later continuations use the previous completed target as `EXP6_START_HOURS`.
Use six-hour boundaries and at most 48 additional active hours per submission.
The loader rejects changed configurations/runtimes rather than silently starting
fresh or accepting incomplete replay. Cloud continuation installs the source
archive's Python 3.11 patch version; scientific dependencies remain pinned in
the source commit's requirements. For local continuation, reproduce that source
environment before resuming.

Local continuation of one seed uses the same verified archive:

```bash
python -m experiments.fhp.exp6_vr_deep_ray8_48h.run resume \
  --output-root /absolute/path/to/new-continuation-output \
  --task-index 0 --threads 8 \
  --resume-from /absolute/path/to/source-worker/training_state \
  --resume-run-id vr6-original-run --start-hours 48 --target-hours 72
```

The saved archive must match the local runtime/platform. Cloud Linux archives
are not claimed to resume equivalently under macOS; resume them on matching VMs.

## Download results

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"

# Optional: all seed policies, diagnostics metadata and full training states.
mkdir -p "cloud_outputs/$RUN_ID/workers"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/workers" "cloud_outputs/$RUN_ID/workers"
```

Resource-monitor outputs remain under `$BUCKET/$RUN_ID/diagnostics/`.

## Tests

```bash
python -m pip install -r requirements-ray.txt -r requirements-dev.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_exp6_vr_deep_ray8_48h.py
FHP_RUN_RAY_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_exp6_vr_deep_ray8_48h.py
```

The opt-in tests require local process and socket access. Restart tests compare
all network parameters, optimizer tensors, replay contents/counters and learning
RNG against uninterrupted training, including reservoir overflow, circular
wraparound, checkpoint RNG restoration and critic target refresh boundaries.
