# VR-Deep Experiments 1–3: retrospective evaluation

Evaluation only. No training, policy refitting, replay downloads or exact
exploitability. The shared `fhp-evaluation-suite` owns scoring; VR-Deep's native
versioned policy loader handles raw Experiment 1 and encoded Experiments 2–3.

## Frozen sources and questions

The source bucket is the existing VR bucket:
`gs://clever-overview-399515-fhp-vr-deep-results`.

| Experiment | Source run | Main question |
|---|---|---|
| 1 | `vr1-24h-20261003-013356` | Raw-input baseline |
| 2 | `vr2-encode-20261003-021829` | Does the lossless suit-canonical encoding improve strength? |
| 3 | `vr3-vm16-20261003-162134` | Does the larger VM improve strength at equal active training time? |

Seeds **0, 1, 2**, with completed-iteration checkpoint crossings at **6, 12, 18,
24 hours**: 36 policies. `sources.json` pins the downloaded successful analyses'
inventory and summary hashes. Every policy is checked against the frozen hash,
native game/encoder metadata, training configuration, seed, clock and worker
success inventory before scoring. Mixed training commits within a source are
rejected; commits may legitimately differ between experiments.

Primary endpoints are final 24h Exp2–Exp1 and Exp3–Exp2 head-to-head. Exp3–Exp1,
earlier checkpoints and temporal comparisons are supporting analyses. Matching
seed labels is not the same as identical random training trajectories. Equal
active hours are not equal billed compute cost.

## Protocol and LBR cost control

| Evaluation | Coverage | Duplicate pairs per matchup |
|---|---|---:|
| Five published rule agents | All 36 policies | 10,000 |
| Cross-experiment play | All three experiment pairs, same training seed, all four times | 50,000 |
| Temporal play | All six later-versus-earlier pairs within each experiment/seed | 50,000 |
| LBR, conditional on timing gate | **Only nine final 24h policies** | 1,000 |

The main stage has **270 matches / 6,300,000 duplicate pairs**. LBR adds **9,000
pairs**, in ten-pair recovery shards. Each pair plays both seats with the same
chance/action seeds. Policy actions are sampled, not argmax. We reuse the UCV
retrospective split chance/action seed arrays, base seed 20260922, opponent
ordering and rule/direct/temporal/LBR seed offsets. LBR uses 4,096 preflop
rollouts and the existing exact flop equity and local action scorer.

One `n2-standard-16` CPU VM runs 16 independent scoring processes, each with one
Torch/BLAS thread. No Ray or GPU is needed. The workflow is:

1. Download the frozen policy/metadata allowlist and validate all 36 policies.
2. Run real-checkpoint cloud smoke through raw/encoded loaders and all scoring
   modes. Its tiny budgets are execution checks, never strength evidence.
3. Profile the main workload on final policies (50 pairs per representative
   match), then run the main comparison only if projected time fits its cap.
4. Save and upload **`main/SUCCESS.json` before starting LBR**.
5. Time LBR in **three independent ten-pair batches on each final policy**:
   270 pilot pairs total, using the full 4,096-rollout setting. Pilot deals use
   separate seeds and are not included in strength estimates.
6. Run full final-policy LBR only if the forecast fits. Otherwise record
   `deferred_pilot_timeout` or `deferred_cost_gate`; do not invent zero LBR scores,
   weaken the rollout protocol, or discard the successful main comparison.

Both timing pilots have a **15-minute hard cap**. Forecasts use the slowest
observed rate per workload class (per target for LBR), a twofold margin,
parallel-load/serial-tail allowance and fixed overhead. The main stage has a
22-hour cap; LBR has a 10-hour cap including its pilot. The Batch safety ceiling
is 36 elapsed hours including setup, smoke, finalisation and upload. It exits as
soon as work finishes. These are limits, **not runtime promises**. A passing
pilot cannot guarantee later workloads' runtime; hard caps remain active.

LBR is a lower-bound diagnostic, not exact exploitability. SD-CFR's expensive
all-history reconstruction is not used: each VR query uses one saved average
policy network. No earlier-checkpoint LBR is enabled in this first comparison.

## Running from the VR-Deep repository root

Keep the updated shared suite in its sibling directory, or set
`EVALUATION_SUITE_ROOT` to its checkout. The launcher uploads a source-only bundle
of both repos, records both Git commits and dirty flags, and pins its SHA-256.
It does not clone a moving remote branch. Commit/push both repos before research
runs for durable provenance; edits do not need to be public for the VM to load
the bundle. No credentials, checkpoints, `.git`, or existing results are bundled.

With `PROJECT_ID`, `REGION`, `BUCKET` (the **VR** bucket) and `SA_EMAIL` set:

```bash
# No cloud calls; inspect outputs/batch/$RUN_ID/prepare-*/job.json.
export RUN_ID="vr-eval123-plan-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh dry-run

# Standalone cloud smoke; same sources and n2-standard-16 hardware.
export RUN_ID="vr-eval123-smoke-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh smoke-only
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh status

# After smoke succeeds, use a new namespace. Full run repeats its own smoke.
export RUN_ID="vr-eval123-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh run
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh status
```

The laptop can disconnect after submission. No jobs are submitted by a local
test or dry-run. The job rejects occupied namespaces and refuses automatic
retries. To recover an interrupted run, keep its RUN_ID and both checkouts at
the exact same revisions/content, then run:

```bash
bash gcp/run_retrospective_exp1_exp2_exp3_evaluation.sh resume
```

This submits a new job name under the same output prefix, refuses active jobs,
checks the original bundle hash, restores task caches, and skips verified
complete stages. Changing dependencies, policies, budgets or evaluator identity
invalidates recovery. A killed worker fails fast; completed tasks are retained.
Periodic uploads occur every five minutes; abrupt termination may lose the most
recent unsynced tasks. Resume after a cost deferral is not a budget override:
unchanged timings/budgets will defer again and require an explicit design change.

## Outputs and interpretation

`$BUCKET/$RUN_ID/analysis/` contains `main/` and `lbr/` as separate outcomes.
Inspect each **`STATUS.json`**, not just Batch's SUCCEEDED state: a safe cost
deferral is a completed workflow but **not a completed LBR evaluation**.
Each fully scored stage has a hash-verified `SUCCESS.json`; smoke has its own
marker and cannot satisfy a production stage.

Tables retain within-match uncertainty, seed-level results, across-seed
Student-t 95% intervals and paired within-seed differences. Three training
seeds—not millions of hands or LBR shards—are the inferential units. Intervals
are exploratory and not multiplicity-adjusted. Figures include the final
head-to-head matrix, rule/direct time and node curves, temporal heatmaps and
final LBR comparisons if completed. Node plots do not claim matched-node budgets.

Provenance, cost-gate measurements, bootstrap logs, periodic CPU/RSS/cgroup
snapshots, exit/failure diagnostics and final kernel/disk/memory logs accompany
the scores. A final always-run uploader preserves outputs on ordinary failure.
An abrupt whole-VM loss still depends on the latest periodic upload.

```bash
export BUCKET_ROOT="gs://${BUCKET#gs://}"
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive --exclude='.*task_results/.*' \
  "${BUCKET_ROOT%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

## Local tests

With the shared suite importable and the existing development dependencies:

```bash
python -m pytest -q tests/test_retrospective_exp1_exp2_exp3_evaluation.py
```

Shared orchestration/deal/statistical tests live in
`fhp-evaluation-suite/tests/test_cohort.py`. Synthetic policies test compatibility
and process execution only; they are not production performance measurements.
