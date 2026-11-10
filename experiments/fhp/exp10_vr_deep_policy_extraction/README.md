# Experiment 10: can policy extraction explain part of the FHP performance gap?

This is a fixed-replay intervention on **VR-Deep FHP Experiment 8**. It asks
whether changing only the procedure that converts VR-Deep's stored strategies
into its deployed policy improves performance against the fixed **UCV FHP
Experiment 38** cohort. No self-play trajectories, critics, or regret networks
are trained. This experiment addresses the FHP comparison; it does not make a
Leduc mechanism claim.

The three Experiment 8 full-state archives retain their policy reservoirs.
Their final 48-hour policies, reservoir arrays and metadata are pinned by SHA256
in `source_pins.json`, as are the three Experiment 38 opponent policies. The
48-hour boundary is the completed iteration checkpoint, not an interpolated
policy. All inputs use the same existing rich AE representation.

## Fixed intervention and controls

For each of the three source training seeds, fit each procedure twice with
prespecified fitting seeds. Both fits start from fresh weights and Adam state.

| Policy | Network | Loss and sampling | Adam rate | Updates |
|---|---|---|---:|---:|
| Archived VR | Saved final policy | No refit | — | 0 |
| Native refit control | Flat 64/64/64 MLP | Uniform replay rows; weighted MSE | .001 | 5,000 |
| UCV extraction applied to VR replay | Card/context branches of width 64; 192/192 trunk | Grouped soft-target CE with mass correction | .003 | 20,000 |

Batch size is 2,048; all fits use the original VR iteration weights
`w_i = (2 t_i / T)^2`. For each exact encoded information set, the UCV arm
constructs the weighted mean target. Uniform sampling of groups is corrected by
`group_mass * number_of_groups / number_of_rows`. Thus grouping preserves the
weighted row cross-entropy objective, rather than changing the weighting of
frequent information sets. Legal-action masks must agree inside a group.

The UCV network, loss, learning rate, update count and reset convention match
Experiment 38's extraction package; the data and weighting remain VR-Deep's.
This is **12 short policy fits**, not twelve new 48-hour training runs. The
native refit detects improvement caused by ordinary refitting variability.
Different architectures cannot share initial weights; fitting repetitions use
prespecified seeds, and no fit is selected based on its results.

## Measurements and inference

1. **Replay fidelity:** weighted KL, total variation, and MSE against the
   empirical weighted mean policy, evaluated on the entire fixed reservoir.
   These are descriptive in-sample fidelity diagnostics, not held-out error,
   an equilibrium reference, or exact full-game exploitability.
2. **Primary gameplay contrast:** adapted VR payoff minus archived VR payoff
   against the same fixed UCV opponent. All nine VR/UCV training-seed cells are
   evaluated for both fit repetitions, with 100,000 seat-swapped duplicate
   pairs per cell per repetition per policy.
3. **Control contrast against the same opponents:** native-refit VR is also
   evaluated against UCV on exactly those deal/action streams. Report adapted
   minus native-refit payoff and native-refit minus archived payoff. This avoids
   relying on direct head-to-head transitivity to interpret the control.
4. **Secondary direct matches:** adapted versus archived VR and adapted versus
   native-refit VR, within each source seed and fitting repetition, again using
   100,000 duplicate pairs each.

The workload is **6.6 million duplicate pairs / 13.2 million hands**, divided
into 480 resumable tasks of 5,000 deals (each task scores two or three matches).
Chance and action RNGs are separate; paired contrasts use common streams, and
their variance is computed from the per-deal differences. One big blind is 100
chips; results are reported in milli-big-blinds per hand.

Average the two fitting repetitions within each training-seed cell first.
For common-opponent contrasts report a 95% crossed bootstrap over VR and UCV
training seeds, plus conditional Monte Carlo standard errors. There are only
**three independently trained models per cohort**, not six. The bootstrap is
exploratory with n=3; secondary intervals are descriptive and do not constitute
a collection of multiplicity-adjusted confirmatory tests. Direct matches use
seed-level paired summaries. No stopping or checkpoint selection uses outcomes.

Interpretation:

- Better fidelity **and** improved payoff over both archived and native-refit
  controls support a contribution from the extraction package on these fixed
  VR trajectories and opponents.
- Better fidelity without better payoff shows that improved replay fitting did
  not translate into a gameplay benefit in this comparison.
- Similar improvements from both refits weaken attribution to the UCV package.
- A null result does not establish that extraction is irrelevant: inspect the
  effect intervals and achieved fidelity before drawing conclusions.

Even a positive result isolates the **package**, not grouping, loss, capacity,
learning rate or fitting budget individually. Follow-up component ablations
would be needed to distinguish those causes. It says nothing directly about
which upstream critic/regret difference explains any remaining gap. Do not
calculate a fraction of an old head-to-head gap using unmatched evaluations.

## Local verification and execution

Run from the VR-Deep repository root:

```bash
python3 -m unittest discover -s tests -p test_exp10_policy_extraction.py -v
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run smoke \
  --output /tmp/vr10-smoke --workers 2
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run benchmark \
  --output /tmp/vr10-timing --benchmark-steps 100
```

Smoke fixtures are explicitly synthetic and cannot be aggregated as production
results. Production requires downloading **only ~2.86 GB of policy replay**
plus small checkpoints and manifests, not full learners or critic replay:

```bash
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run fetch \
  --input cloud_outputs/vr10-inputs
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run fit \
  --input cloud_outputs/vr10-inputs --output cloud_outputs/vr10-diagnosis --workers 3
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run profile-matches \
  --input cloud_outputs/vr10-inputs --output cloud_outputs/vr10-diagnosis --workers 8
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run evaluate \
  --input cloud_outputs/vr10-inputs --output cloud_outputs/vr10-diagnosis --workers 8
python3 -m experiments.fhp.exp10_vr_deep_policy_extraction.run aggregate \
  --output cloud_outputs/vr10-diagnosis
```

Completed fits and match shards are checksum-checked before reuse. Partial fits
restart; completed match shards resume. A changed implementation, contract or
runtime requires a new output directory. Aggregation refuses missing or corrupt
shards. Source hashes, configuration, per-fit diagnostics, match moments and
analysis receipts are retained. Inputs are never modified.

## GCP launch and analysis-only download

The repository-owned launcher builds one `n2-standard-16` Batch job: three fit
workers followed by eight scoring workers, each with one PyTorch thread. The
VM has 62,000 MiB allocated memory and a 150 GB boot disk. No GPU is needed.
Its main runnable is capped at 23 hours within a 24-hour task limit. Outputs
sync every five minutes and at normal exit; completed work can be resumed.

After committing and pushing this implementation:

```bash
export PROJECT_ID=clever-overview-399515
export REGION=europe-west1
export BUCKET=gs://clever-overview-399515-fhp-vr-deep-results
export SA_EMAIL=leduc-dream-runner@clever-overview-399515.iam.gserviceaccount.com
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="vr10-extract-$(date -u '+%Y%m%d-%H%M%S')"
bash gcp/run_exp10_policy_extraction.sh dry-run
bash gcp/run_exp10_policy_extraction.sh run
```

The service account needs read access to both pinned source buckets and write
access to the result bucket. The launcher does not change IAM. `profile` in
place of `run` performs refits and the short on-VM scoring profile, then stops;
use `resume` with the same variables to finish. `resume` also recovers an
interrupted full job without selecting a new configuration. Keep the same
pushed commit. Do not launch concurrent jobs into the same output prefix.

After completion:

```bash
python3 gcp/download_exp10_policy_extraction_analysis.py \
  --source "$BUCKET/$RUN_ID" --output "cloud_outputs/$RUN_ID"
```

This downloads the verified analysis, experiment manifest and per-seed fit
diagnostics. It does **not** download policy models, replay arrays or full
training states. Start with `analysis/report.md`, `summary.json`,
`fidelity_by_seed.csv` and `match_cells.csv`. A successful profile-only job has
no final analysis completion marker.

## Runtime estimate and validation evidence

Plan for **4–8 hours on the specified GCP VM**, including setup, replay grouping,
refitting and evaluation. This is a planning estimate, not a measured full run.
The on-VM `analysis/match_timing.json` gives a fresh scoring-only projection;
it excludes setup and fitting and does not stop the run based on performance.

On the local machine (2026-11-10, CPU, one PyTorch thread), a 100-update timing
fixture with batch size 2,048 measured approximately 2.50 ms/update for the
native arm and 4.86 ms/update for the UCV arm. The projected twelve fits total
about 11 CPU-minutes on that fixture, excluding million-row memory access,
grouping and diagnostics. A separate real-checkpoint test measured 512
duplicate pairs in 1.78 seconds. These local timings are not GCP benchmarks.
Earlier FHP cloud evaluations imply substantially slower per-worker scoring;
the planning range allows for that difference. Evaluation should dominate.

Validation includes weighted grouping objective/gradient equivalence, one-step
agreement with VR's existing MSE trainer, duplicate-seat cancellation,
paired-variance arithmetic, deterministic task coverage, incomplete/corrupt
result rejection, and Batch shell syntax. The synthetic pipeline was run through
fitting, saved-policy reload, matches and aggregation. Separately, the pinned
real policy loaders were checked; UCV action probabilities matched its existing
loader on 82 decisions, and grouped targets/weights matched its existing
trainer. Production replay fitting and the full cloud job have not been run.
