"""Versioned, atomic training archives. Load only this project's trusted outputs.

Replay columns are separate .npy files: saving uses populated views, and restore
streams memory-mapped columns into preallocated buffers without a second replay
copy. Ray handles, uninitialised tails and temporary prediction caches are never
serialized. Checksums provide integrity, not authentication of untrusted pickle.
"""

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import math
import platform

import numpy as np
import torch

from experiments.fhp.exp1_vr_deep_pdcfr_24h.train import ROOT, write_json
from fhp_vr_deep.game import serialisable_game_definition
from fhp_vr_deep.io_utils import canonical_sha256, read_json, repository_commit, sha256_file
from . import config

SCHEMA = "fhp_vr_ray_training_state_v1"
MODELS = ("model", "target_model", "imm_model", "best_model")
OPTIMIZERS = ("optimizer", "imm_optimizer")
COUNTERS = ("num_iteration", "episode", "nodes_touched", "_last_average_policy_loss")
CHUNK_ROWS = 8192


def trainers(solver):
    return dict(regret_0=solver.regret_trainers[0], regret_1=solver.regret_trainers[1],
                critic=solver.q_value_trainer, policy=solver.ave_policy_trainer)


def runtime_contract():
    return dict(python=platform.python_version(), system=platform.system(), machine=platform.machine(),
                torch=str(torch.__version__), numpy=np.__version__, ray=version("ray"),
                open_spiel=version("open_spiel"), scipy=version("scipy"))


def learner_state(solver):
    return dict(
        trainers={name: {key: deepcopy(getattr(trainer, key).state_dict())
                        for key in (*MODELS, *OPTIMIZERS) if hasattr(trainer, key)}
                  for name, trainer in trainers(solver).items()},
        model_modes={name: {key: getattr(trainer, key).training for key in MODELS if hasattr(trainer, key)}
                     for name, trainer in trainers(solver).items()},
        rng=solver._capture_rng_state(), counters={key: getattr(solver, key) for key in COUNTERS},
        phase_seconds=deepcopy(solver.phase_seconds), parallel_totals=deepcopy(solver.parallel_totals),
        logger_pending=deepcopy(solver.logger._pending), logger_history=deepcopy(solver.logger.history),
        checkpoint_rows=deepcopy(solver.checkpoint_rows))


def save_state(directory, solver, *, training_config, seed, target_hours, elapsed_seconds,
               is_smoke=False, parent=None):
    if (not getattr(solver, "safe_to_snapshot", False)
            or solver._active_checkpoint_started_at is not None):
        raise ValueError("Save a full state only after a completed iteration and checkpoint RNG restoration")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("Invalid completed active time")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    temporary = directory.with_name(directory.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    payload = learner_state(solver)
    torch.save(payload, temporary / "learner.pt")
    buffers = {}
    for name, trainer in trainers(solver).items():
        buf = trainer.buffer
        size = min(len(buf), buf.buffer_size)
        arrays = {}
        for key, value in vars(buf).items():
            if isinstance(value, np.ndarray):
                filename = f"{name}__{key}.npy"
                np.save(temporary / filename, value[:size], allow_pickle=False)
                arrays[key] = dict(file=filename, shape=list(value[:size].shape), dtype=value.dtype.str)
        buffers[name] = dict(capacity=buf.buffer_size, populated_rows=size, arrays=arrays,
                             counters={key: getattr(buf, key) for key in ("cur_id", "size") if hasattr(buf, key)})
    metadata = dict(schema=SCHEMA, algorithm_id=config.ALGORITHM_ID, seed=seed, is_smoke=is_smoke,
                    training_config=training_config, training_config_sha256=canonical_sha256(training_config),
                    feature_encoder=solver.feature_encoder.metadata(), game=serialisable_game_definition(),
                    repository_commit=repository_commit(ROOT), baseline_commit=config.BASELINE_COMMIT,
                    runtime=runtime_contract(), threads=torch.get_num_threads(), worker_count=8,
                    traversal_rng=config.baseline.contract()["traversal_rng"],
                    boundary="completed_outer_iteration_after_policy_checkpoint_rng_restoration",
                    completed_target_hours=target_hours, active_seconds=elapsed_seconds,
                    iteration=solver.num_iteration, episode=solver.episode, nodes_touched=solver.nodes_touched,
                    buffers=buffers, parent=parent,
                    files={p.name: dict(sha256=sha256_file(p), size_bytes=p.stat().st_size)
                           for p in temporary.iterdir() if p.is_file()})
    write_json(temporary / "manifest.json", metadata)
    # Verify all serialized bytes and replay shapes before publishing the directory.
    verify_state(temporary)
    temporary.replace(directory)
    return metadata


def verify_state(directory):
    directory = Path(directory)
    meta = read_json(directory / "manifest.json")
    if (meta["schema"] != SCHEMA or meta["algorithm_id"] != config.ALGORITHM_ID
            or meta["boundary"] != "completed_outer_iteration_after_policy_checkpoint_rng_restoration"
            or meta["worker_count"] != 8 or meta["seed"] not in config.SEEDS
            or meta["training_config_sha256"] != canonical_sha256(meta["training_config"])
            or not math.isfinite(meta["active_seconds"]) or meta["active_seconds"] <= 0
            or meta["iteration"] < 1 or meta["nodes_touched"] < meta["episode"]
            or meta["episode"] != meta["iteration"] * 2 * meta["training_config"]["num_traversals"]):
        raise ValueError("Invalid training-state contract/counters")
    if set(meta["buffers"]) != {"regret_0", "regret_1", "critic", "policy"}:
        raise ValueError("Incomplete replay inventory")
    required = {"learner.pt"}
    for buf in meta["buffers"].values():
        required.update(spec["file"] for spec in buf["arrays"].values())
    if set(meta["files"]) != required:
        raise ValueError("Incomplete state-file inventory")
    for name, record in meta["files"].items():
        path = directory / name
        if Path(name).name != name or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe training-state path")
        if not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Missing/corrupt training state: {name}")
    for name, buf in meta["buffers"].items():
        size, capacity, counters = buf["populated_rows"], buf["capacity"], buf["counters"]
        if size < 0 or size > capacity or counters["cur_id"] < 0:
            raise ValueError("Invalid replay counters")
        if name == "critic":
            if (set(counters) != {"cur_id", "size"} or counters["size"] != size
                    or counters["cur_id"] >= capacity or (size < capacity and counters["cur_id"] != size)):
                raise ValueError("Invalid circular replay counters")
        elif set(counters) != {"cur_id"} or min(counters["cur_id"], capacity) != size:
            raise ValueError("Invalid reservoir counter")
        for spec in buf["arrays"].values():
            value = np.load(directory / spec["file"], mmap_mode="r", allow_pickle=False)
            if list(value.shape) != spec["shape"] or len(value) != size or value.dtype.str != spec["dtype"]:
                raise ValueError("Invalid saved replay shape/dtype")
            for start in range(0, size, CHUNK_ROWS):
                if not np.isfinite(value[start:start + CHUNK_ROWS]).all():
                    raise ValueError("Nonfinite saved replay")
            del value
    return meta


def restore_state(directory, solver, *, training_config, seed, start_hours, is_smoke=False):
    directory = Path(directory)
    meta = verify_state(directory)
    checks = dict(training_config=training_config, seed=seed, is_smoke=is_smoke,
                  completed_target_hours=start_hours, feature_encoder=solver.feature_encoder.metadata(),
                  game=serialisable_game_definition(), threads=torch.get_num_threads(),
                  repository_commit=repository_commit(ROOT), runtime=runtime_contract(),
                  baseline_commit=config.BASELINE_COMMIT,
                  traversal_rng=config.baseline.contract()["traversal_rng"])
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise ValueError(f"Resume incompatibility: {key}; use the source commit/configuration/runtime")
    # Own, integrity-checked archives only; this payload includes Python/NumPy RNG.
    payload = torch.load(directory / "learner.pt", map_location="cpu", weights_only=False)
    if set(payload["trainers"]) != set(trainers(solver)):
        raise ValueError("Missing trainer state")
    for name, trainer in trainers(solver).items():
        states = payload["trainers"][name]
        expected = {key for key in (*MODELS, *OPTIMIZERS) if hasattr(trainer, key)}
        if name == "critic":
            expected.add("best_model")
        if set(states) != expected:
            raise ValueError(f"Incomplete model/optimizer state: {name}")
        for key, state in states.items():
            if not hasattr(trainer, key):
                setattr(trainer, key, trainer.init_model())
            getattr(trainer, key).load_state_dict(state)
            if key in MODELS:
                getattr(trainer, key).train(payload["model_modes"][name][key])
        buf, saved = trainer.buffer, meta["buffers"][name]
        arrays = {key: value for key, value in vars(buf).items() if isinstance(value, np.ndarray)}
        if buf.buffer_size != saved["capacity"] or set(arrays) != set(saved["arrays"]):
            raise ValueError("Incompatible replay schema")
        for key, value in arrays.items():
            source = np.load(directory / saved["arrays"][key]["file"], mmap_mode="r", allow_pickle=False)
            if value.shape[1:] != source.shape[1:] or value.dtype != source.dtype:
                raise ValueError("Incompatible replay dimensions/dtype")
            for start in range(0, len(source), CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, len(source))
                value[start:stop] = source[start:stop]
            del source
        for key, value in saved["counters"].items():
            setattr(buf, key, value)
    for key in COUNTERS:
        setattr(solver, key, payload["counters"][key])
    if (solver.num_iteration != meta["iteration"] or solver.episode != meta["episode"]
            or solver.nodes_touched != meta["nodes_touched"]):
        raise ValueError("Learner and manifest counters disagree")
    solver.phase_seconds = payload["phase_seconds"]
    solver.parallel_totals = payload["parallel_totals"]
    if (solver.parallel_totals["collections"] != 2 * solver.num_iteration
            or solver.parallel_totals["traversals"] != solver.episode):
        raise ValueError("Invalid parallel continuation counters")
    solver.logger._pending, solver.logger.history = payload["logger_pending"], payload["logger_history"]
    solver.checkpoint_rows = payload["checkpoint_rows"]
    solver.elapsed_before_resume = meta["active_seconds"]
    solver.safe_to_snapshot = True
    solver._stop_requested = False
    solver._last_iteration_seconds = None
    solver.feature_encoder._information_cache.clear()
    solver.feature_encoder._full_state_cache.clear()
    # Restore LAST: model/optimizer/actor construction must not advance learning RNG.
    solver._restore_rng_state(payload["rng"])
    return meta
