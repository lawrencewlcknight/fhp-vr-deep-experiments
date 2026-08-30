# Analysis conventions

FHP experiments use the same analysis vocabulary and artifact shapes as the
Leduc ESCHER-architecture experiments wherever the underlying quantity remains
computable.

## Required training artifacts

Each production run must write:

- `run_manifest.json` with experiment identity, Git commit, seed, algorithm,
  exact game definition, configuration, configuration digest, and provenance;
- `checkpoint_curves.csv` and `.json` with checkpoint index, iteration,
  episodes, training nodes, training time, wall-clock time, replay occupancy,
  and all available losses;
- reloadable average-policy snapshots plus SHA-256 hashes;
- `seed_summary.csv`, `aggregate_summary.json`, and `summary.json`;
- a failure artifact when a worker does not complete.

Aggregates use paired seeds and report count, mean, sample standard deviation,
standard error, minimum, maximum, and 95% confidence intervals. Curves show
individual seeds faintly and the cross-seed mean with uncertainty, matching the
Leduc presentation convention.

## Policy-quality evaluation

Do not report training loss as policy strength and do not label a sampled FHP
metric as exact exploitability. Exact whole-tree Leduc evaluation is replaced
by reproducible sampled, seat-swapped head-to-head evaluation. Record:

- both policy hashes and manifests;
- deal/evaluation seeds and number of deals;
- payoff by seat and seat-averaged payoff in game units and milli-big-blinds;
- standard error and a 95% confidence interval;
- paired differences between the two algorithms at matched checkpoints.

Nodes touched exclude evaluation trajectories. Effective training time excludes
policy fitting, snapshot serialization, and evaluation. Checkpoint wall-clock
time includes policy fitting up to that checkpoint but is sampled immediately
before snapshot serialization; `total_worker_wall_clock_seconds` includes the
complete worker and artifact-validation path. These distinct clocks must be
retained so algorithmic progress and systems cost are not conflated.
