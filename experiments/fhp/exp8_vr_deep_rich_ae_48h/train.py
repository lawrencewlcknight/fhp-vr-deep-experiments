"""Fresh rich-AE runs through the established eight-actor timed learner."""
from pathlib import Path
from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as native
from experiments.fhp.exp6_vr_deep_ray8_48h.train import make_solver
from experiments.fhp.exp6_vr_deep_ray8_48h.training_state import save_state
from fhp_vr_deep.io_utils import sha256_file
from . import config

def run_worker(output_root, task_index, *, smoke=False, remote_uri=None, threads=config.THREADS):
    created = []
    def factory(seed, training_config):
        solver = make_solver(seed, training_config, smoke=smoke)
        created.append(solver)
        if solver.feature_encoder.metadata() != config.contract(smoke)["feature_encoder"]:
            raise ValueError("Learner encoder differs from experiment contract")
        worker = Path(output_root) / "workers" / config.task_name(task_index)
        native.write_json(worker / "parallel_runtime.json", solver.parallel_runtime)
        return solver
    def final_state(solver, worker, final, manifest):
        directory = worker / "training_state"
        save_state(directory, solver, training_config=manifest["training_config"],
            seed=manifest["seed"], target_hours=48, elapsed_seconds=final["training_elapsed_seconds"],
            is_smoke=smoke, experiment=config)
        return dict(final_training_state="training_state/manifest.json",
            final_training_state_sha256=sha256_file(directory / "manifest.json"),
            final_training_state_bytes=sum(p.stat().st_size for p in directory.iterdir()),
            continuation_source=None, final_training_state_verified=True)
    try:
        return native.run_worker(output_root, task_index, smoke=smoke, remote_uri=remote_uri,
            threads=threads, experiment=config, solver_factory=factory, after_training=final_state)
    finally:
        for solver in created:
            solver.close()
