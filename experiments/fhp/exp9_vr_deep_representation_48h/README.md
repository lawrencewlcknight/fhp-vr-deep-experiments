# Experiment 9: VR-DeepPDCFR+, rich versus original at 48 hours

This repository owns the VR-DeepPDCFR+ representation study: Experiment 8 rich-AE policies versus Experiment 6 original policies. It reuses completed training runs; no new training is required. The experiment was originally registered as UCV evaluation 43 and has moved here.

`config.json` fixes this algorithm's six source policies, metadata checksums, training seeds and evaluation budget. `run.py` checks that contract and invokes the shared evaluator. Scoring remains a pinned dependency at UCV commit `b6eabbeeeca6d8ae8f07f383a5296492c2cd2376`; there is one implementation of the statistical and gameplay protocol, not separate scoring copies in the two algorithm repositories.

The native experiment number is **9**. For continuity, the shared evaluator's protocol, result `experiment_id=43`, Batch workload `fhp-exp43` and RNG streams remain unchanged. New Batch jobs additionally carry `owner_algorithm=vr` and `owner_experiment=9`; `experiment_contract.json` and `experiment_repository_commit.txt` at the output root record the owning definition and revision. This numbering distinction is intentional and is not a change to the scientific comparison. Already launched jobs and their artifacts remain valid.

Original source: `vr6-ray48-20261003-221425`. Rich source: `vr8-rich48-20261101-234215`.

## Prespecified protocol

- Three training seeds per cohort: 0, 1 and 2; all nine cross-seed pairings.
- Each pairing receives 100,000 seat-swapped duplicate deal pairs: 200,000 hands.
- Each experiment totals 900,000 duplicate pairs / **1.8 million hands**. Both together: 3.6 million hands.
- Twenty resumable shards of 5,000 duplicate pairs per cell: 180 production shards per experiment.
- Fresh deterministic RNG base seeds 642048 (SD) and 643048 (VR), with separate smoke, profiling and production streams. Duplicate deals use the existing split deal/action stream layout.
- Automatic smoke (2 pairs per cell) and timing profile (128 per cell) precede production. They do not enter reported strength estimates or select a policy. Timing forecasts are informational, not a performance stopping rule.
- SD-CFR uses **every historical strategy** in each native 48-hour archive, uniformly sampled once per player per hand. No last-network substitution, averaged-network approximation or archive thinning.
- VR-Deep uses the native saved 48-hour average-policy network. Neither algorithm is refitted or tuned.

The native Experiment 41 loader revisions support both original and rich encoders; both sides use those same evaluation implementations. Training revisions remain separately recorded. Original player inputs contain 183 features; rich inputs contain 231. Source metadata bytes, configurations, checkpoint boundaries and checkpoint payloads are verified. All SD archive chunks are checksum-verified, with complete iteration coverage required. Metadata validation runs before the cloud VM downloads the policy archives.

`config.json` freezes this experiment's six source workers; the shared dependency additionally pins both experiments. `training_cost_context.json` records their exact encoder, node count, iteration count, clock and runtime information. Within each algorithm the manifests differ in training configuration only by encoder selection. Python versions differ (3.11.17 original versus 3.11.18 rich); VR also records different Linux/platform versions. Training code revisions differ. Consequently this is a retrospective fixed-budget comparison, not a perfectly isolated causal rerun. Rich features can change training throughput; node counts describe that consequence and do not normalise winnings.

## Analysis

The primary outcome is the equally weighted mean over the 3×3 matrix, in mbb/hand. Reports include all nine cell means, seat-specific outcomes, per-cell deal uncertainty and node counts.

Training-seed sensitivity uses 10,000 independent row/column bootstrap draws. Reports provide 95% pointwise intervals and 97.5% intervals for the prespecified family of these two comparisons (Bonferroni adjustment). The nine cells share six trained policies: they are **not nine independent training replications**. Equal seed numbers are not treated as paired training randomness. Bootstrap intervals with only three seeds per cohort remain exploratory.

Conditional deal Monte Carlo uncertainty is reported separately, using duplicate-pair dispersion and independent cell streams. It includes sampled policy/action randomness conditional on the trained policies. It is not combined with the seed intervals into a joint confidence claim. Opponent-specific head-to-head improvement does not establish reduced exploitability or universal strength.

Incomplete, duplicated or corrupted production shards cannot produce a final aggregate. Resume requires identical policies, implementation, dependency versions, protocol and streams; worker count may change. Completed shards are reused. A successful smoke is explicitly marked as having no strength evidence.

## Launch from this repository

After committing and pushing Experiment 9 in **this** repository:

```bash
export PROJECT_ID=clever-overview-399515
export REGION=europe-west1
export BUCKET=gs://clever-overview-399515-fhp-escher-results
export SA_EMAIL=fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com
export REPO_REF='<full pushed commit of this VR-DeepPDCFR+ repository containing Experiment 9>'
export RUN_ID="vr9-repr48-$(date -u '+%Y%m%d-%H%M%S')"
export EXP9_SMOKE_ONLY=0

bash gcp/run_exp9_vr_deep_representation_48h.sh dry-run
bash gcp/run_exp9_vr_deep_representation_48h.sh run
```

`REPO_REF` now refers to this algorithm's repository, **not the UCV repository**. The shared dependency revision is fixed in `config.json`; do not supply the old UCV commit as `REPO_REF`. The output bucket above preserves the existing evaluation destination and runner configuration; repository ownership does not require a different bucket or new IAM grants. `BUCKET` can be changed to another bucket accessible to the chosen runner.

The launcher exports its builder and experiment contract from `REPO_REF`. It reads the shared builder/preflight from the pinned dependency revision, using the sibling UCV checkout when available (`EVALUATION_REPO` overrides its location), or cloning a temporary checkout otherwise. No dirty sibling files are copied. The cloud worker checks out this repository at `REPO_REF`, records its contract, and calls its Experiment 9 entry point. Native model loaders retain their original pinned revisions.

Each job uses one `n2-standard-16` VM, 200 GB disk and 16 single-thread evaluation workers. It includes source validation, a smoke test and profiling before 1.8 million scored hands. A 24-hour Batch ceiling bounds each invocation; it is not a runtime forecast. Periodic uploads preserve completed shards.

Supported actions: `dry-run`, `preflight`, `smoke-cloud`, `run`, `resume`, `status`. A separate smoke job needs a distinct RUN_ID. Preflight verifies source metadata and explicit runner bucket grants; it changes neither IAM nor source artifacts. Completed production requires `analysis/SUCCESS.json` with `smoke=false` and 900,000 duplicate pairs.

```bash
bash gcp/run_exp9_vr_deep_representation_48h.sh status
# After an interrupted job has stopped, keep the same RUN_ID and BUCKET:
bash gcp/run_exp9_vr_deep_representation_48h.sh resume
```

Resuming an old `exp43-...` job is supported with the same RUN_ID, BUCKET, source buckets and compatible environment. The pinned shared evaluator preserves its original resume contract. Do not start a duplicate evaluation merely because the experiment moved repositories.

## Download analysis only

```bash
bash gcp/download_exp9_representation_analysis.sh
```

This reads `BUCKET` and `RUN_ID`; its default destination is this repository's `cloud_outputs/$RUN_ID/analysis`. Set `OUTPUT_DIR` to override it. The pinned helper verifies completion and checksums and retrieves only ten analysis artifacts plus the receipt. Model objects, SD history archives and scoring shards are excluded. Launch provenance remains at the cloud output root.

## Local scoring

Arrange sources under `INPUT/{original,rich}/vr/workers/<native-worker-name>/`. Only this algorithm's sources are needed. SD requires all historical chunks; VR requires the saved 48-hour average policy. Sources and output must be disjoint.

```bash
python3 -m experiments.fhp.exp9_vr_deep_representation_48h.run \
  --ucv-repo /path/to/fhp-ucv-escher-experiments \
  --vr-repo /path/to/fhp-vr-deep-experiments \
  --sd-repo /path/to/fhp-deep-cfr-experiments \
  --source-root INPUT --output OUTPUT --workers 16
```

The UCV source tree must match the pinned shared evaluator and supporting libraries. The wrapper verifies their bytes before invoking scoring in a fresh process, avoiding collisions between the repositories' Python packages. Append `--smoke-only` for a loading/gameplay check without strength evidence.
