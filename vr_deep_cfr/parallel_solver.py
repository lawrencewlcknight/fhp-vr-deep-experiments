"""Synchronous Ray collection for VR-Deep; all fitting remains in the driver.

The OpenSpiel traversal and target construction are inherited verbatim. Actors
return every experience row; only the driver applies reservoir/circular replay
retention. No worker-local reservoir subsampling, stale-policy fitting or async
updates are allowed.
"""

from copy import deepcopy
import os
import time

import numpy as np
import torch

from experiments.fhp.exp1_vr_deep_pdcfr_24h import train as timed
from experiments.fhp.exp2_vr_deep_lossless_24h.train import TimedEncodedVRDeepPDCFRPlus, make_solver as sequential_solver
from fhp_vr_deep.io_utils import peak_rss_mib
from .solver import ReservoirBuffer, CircularBuffer, set_seed

RAY_VERSION = "2.51.2"
WORKER_COUNT = 8
OBJECT_STORE_BYTES = 2 * 1024**3
SMOKE_OBJECT_STORE_BYTES = 256 * 1024**2
STARTUP_TIMEOUT_SECONDS = 300
COLLECTION_TIMEOUT_SECONDS = 1800
WORKER_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
WORKER_ENV.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")


def partition(total, workers=WORKER_COUNT):
    if total < 0 or workers < 1:
        raise ValueError("Invalid traversal budget or worker count")
    quotient, remainder = divmod(int(total), int(workers))
    return [quotient + int(i < remainder) for i in range(workers)]


def collection_seed(seed, worker_id, iteration, player):
    if seed < 0 or worker_id < 0 or iteration < 1 or player not in (0, 1):
        raise ValueError("Invalid collection identity")
    return int(np.random.SeedSequence([int(seed), 5102026, int(worker_id), int(iteration), int(player)])
               .generate_state(1, dtype=np.uint32)[0])


def inference_snapshot(solver, player):
    def state(model):
        return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}

    return dict(token=(int(solver.num_iteration), int(player)),
                regrets=[dict(model=state(t.model), imm_model=state(t.imm_model)) for t in solver.regret_trainers],
                # dfs calls the ONLINE/best fitted critic, not its TD target network.
                critic=state(solver.q_value_trainer.model))


class StagingReservoir(ReservoirBuffer):
    """Append-only, bounded staging; never perform worker-local reservoir draws."""

    def add(self, *args):
        if self.cur_id >= self.buffer_size:
            raise RuntimeError("Traversal staging capacity exceeded; refusing to drop rows")
        super().add(*args)


class StagingCircular(CircularBuffer):
    def add(self, *args):
        if self.size >= self.buffer_size:
            raise RuntimeError("Critic staging capacity exceeded; refusing to overwrite rows")
        super().add(*args)


def packed(buffer):
    size = min(len(buffer), buffer.buffer_size)
    return {key: value[:size].copy() for key, value in vars(buffer).items() if isinstance(value, np.ndarray)}


def payload_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(payload_bytes(v) for v in value.values())
    return 0


class TraversalWorker:
    """Persistent single-threaded inference process; never invokes fitting."""

    def __init__(self, training_config, seed, worker_id, max_traversals):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.seed, self.worker_id = int(seed), int(worker_id)
        self.max_traversals = int(max_traversals)
        small = deepcopy(training_config)
        # Construct the same inference networks without allocating full replay
        # eight times. The small initial buffers are immediately replaced below.
        for name in ("advantage", "ave_policy", "baseline"):
            small[f"{name}_buffer_size"] = 1
        self.solver = sequential_solver(seed, small)
        self.solver.logger._verbose = False
        bound = max(1, self.max_traversals * int(self.solver.game.max_game_length()))
        for trainer in (*self.solver.regret_trainers, self.solver.ave_policy_trainer):
            trainer.buffer = StagingReservoir(bound, trainer.input_size, trainer.output_size)
        critic = self.solver.q_value_trainer
        critic.buffer = StagingCircular(bound, critic.input_size, critic.state_size, critic.output_size,
                                        derive_next_state=False)
        self.row_bound = bound

    def ping(self):
        return dict(worker_id=self.worker_id, pid=os.getpid(), torch_threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(), staging_row_bound=self.row_bound,
                    thread_environment={key: os.environ.get(key) for key in WORKER_ENV})

    def collect(self, count, player, iteration, snapshot):
        started = time.perf_counter()
        if count < 1 or count > self.max_traversals or tuple(snapshot["token"]) != (iteration, player):
            raise ValueError("Invalid traversal request or stale inference snapshot")
        if torch.get_num_threads() != 1:
            raise RuntimeError("Traversal worker must have one computation thread")
        solver = self.solver
        for trainer, states in zip(solver.regret_trainers, snapshot["regrets"]):
            for name, state in states.items():
                getattr(trainer, name).load_state_dict({key: torch.from_numpy(value.copy()) for key, value in state.items()})
        solver.q_value_trainer.model.load_state_dict(
            {key: torch.from_numpy(value.copy()) for key, value in snapshot["critic"].items()})
        for trainer in (*solver.regret_trainers, solver.ave_policy_trainer, solver.q_value_trainer):
            trainer.buffer.reset()
        solver.num_iteration = int(iteration)
        solver.num_traversals = int(count)
        solver.episode = solver.nodes_touched = 0
        # Explicit phase-keyed streams make worker scheduling/reuse irrelevant.
        phase_seed = collection_seed(self.seed, self.worker_id, iteration, player)
        set_seed(phase_seed)
        load_seconds = time.perf_counter() - started
        cpu_start, collect_start = time.process_time(), time.perf_counter()
        solver.collect_training_data(player)  # The unchanged sequential OpenSpiel dfs.
        collect_seconds, cpu_seconds = time.perf_counter() - collect_start, time.process_time() - cpu_start
        result = dict(worker_id=self.worker_id, token=(iteration, player), seed=phase_seed,
                      traversals=solver.episode, nodes=solver.nodes_touched,
                      regret=packed(solver.regret_trainers[player].buffer),
                      policy=packed(solver.ave_policy_trainer.buffer), critic=packed(solver.q_value_trainer.buffer),
                      collection_seconds=collect_seconds, cpu_seconds=cpu_seconds,
                      snapshot_load_seconds=load_seconds, peak_rss_mib=peak_rss_mib())
        result["total_seconds"] = time.perf_counter() - started
        result["payload_bytes"] = payload_bytes(result)
        return result


def validate_rows(buffer, rows):
    schema = {key: value for key, value in vars(buffer).items() if isinstance(value, np.ndarray)}
    if set(schema) != set(rows):
        raise ValueError("Wrong replay columns from traversal worker")
    sizes = set()
    for key, template in schema.items():
        array = rows[key]
        if (not isinstance(array, np.ndarray) or array.ndim != template.ndim or
                array.shape[1:] != template.shape[1:] or array.dtype != template.dtype):
            raise ValueError(f"Wrong replay shape/dtype: {key}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite traversal data: {key}")
        sizes.add(len(array))
    if len(sizes) != 1:
        raise ValueError("Inconsistent replay row counts")
    return sizes.pop()


def append_reservoir(buffer, rows):
    """Bulk initial fill, then exactly the existing reservoir insertion rule."""
    count = len(rows["infostate_buf"])
    fill = min(count, max(0, buffer.buffer_size - buffer.cur_id))
    for key, array in rows.items():
        getattr(buffer, key)[buffer.cur_id:buffer.cur_id + fill] = array[:fill]
    buffer.cur_id += fill
    for i in range(fill, count):
        buffer.add(rows["infostate_buf"][i], rows["q_value_buf"][i],
                   rows["q_value_mask_buf"][i], rows["iteration_buf"][i])


def append_circular(buffer, rows):
    """Preserve the exact ring order and write index, including wraparound."""
    count, offset = len(rows["history_buf"]), 0
    while offset < count:
        size = min(count - offset, buffer.buffer_size - buffer.cur_id)
        for key, array in rows.items():
            getattr(buffer, key)[buffer.cur_id:buffer.cur_id + size] = array[offset:offset + size]
        offset += size
        buffer.cur_id = (buffer.cur_id + size) % buffer.buffer_size
    buffer.size = min(buffer.size + count, buffer.buffer_size)


class RayTraversalPool:
    def __init__(self, training_config, seed, *, object_store_bytes=OBJECT_STORE_BYTES):
        import ray
        self.ray, self.workers, self.owns_runtime = ray, [], False
        if ray.__version__ != RAY_VERSION:
            raise RuntimeError(f"Experiment 5 requires ray=={RAY_VERSION}, found {ray.__version__}")
        if ray.is_initialized():
            raise RuntimeError("Use a fresh process: Experiment 5 must own its local Ray runtime")
        started = time.perf_counter()
        try:
            self.owns_runtime = True
            # Never attach to another cluster via RAY_ADDRESS or allocate Ray's
            # default fraction of the machine's memory to the object store.
            context = ray.init(address="local", num_cpus=WORKER_COUNT, include_dashboard=False,
                               object_store_memory=object_store_bytes, log_to_driver=True)
            self.session_dir = context.address_info.get("session_dir")
            remote = ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0,
                                runtime_env={"env_vars": WORKER_ENV})(TraversalWorker)
            maximum = max(partition(training_config["num_traversals"]))
            self.workers = [remote.remote(training_config, seed, i, maximum) for i in range(WORKER_COUNT)]
            health = ray.get([worker.ping.remote() for worker in self.workers], timeout=STARTUP_TIMEOUT_SECONDS)
            if (len({row["pid"] for row in health}) != WORKER_COUNT or
                    any(row["worker_id"] != i or row["torch_threads"] != 1 or row["interop_threads"] != 1
                        for i, row in enumerate(health)) or
                    any(any(row["thread_environment"][key] != value for key, value in WORKER_ENV.items()) for row in health)):
                raise RuntimeError("Incorrect Ray actor allocation or thread configuration")
            self.metadata = dict(backend="ray", ray_version=ray.__version__, workers=health,
                                 object_store_bytes=object_store_bytes, startup_seconds=time.perf_counter() - started,
                                 session_dir=self.session_dir, max_restarts=0, max_task_retries=0)
        except BaseException:
            self.close()
            raise

    def collect(self, counts, player, iteration, snapshot):
        start = time.perf_counter()
        snapshot_ref = self.ray.put(snapshot)
        references = [worker.collect.remote(count, player, iteration, snapshot_ref)
                      for worker, count in zip(self.workers, counts) if count]
        sync_seconds = time.perf_counter() - start
        wait_start = time.perf_counter()
        # One bounded barrier. ray.get preserves submission order, not finish
        # order, so replay retention is independent of worker scheduling.
        results = self.ray.get(references, timeout=COLLECTION_TIMEOUT_SECONDS)
        return results, dict(dispatch_seconds=sync_seconds, wait_seconds=time.perf_counter() - wait_start)

    def close(self):
        if self.ray is None:
            return
        for worker in self.workers:
            try:
                self.ray.kill(worker, no_restart=True)
            except Exception:
                pass
        self.workers = []
        if self.owns_runtime:
            self.ray.shutdown()
        self.ray = None


class ParallelVRDeepPDCFRPlus(TimedEncodedVRDeepPDCFRPlus):
    """Override collection only; player order, dfs and every fitter are inherited."""

    def __init__(self, *, collector_config, run_seed, pool_factory=RayTraversalPool,
                 object_store_bytes=OBJECT_STORE_BYTES, **kwargs):
        super().__init__(**kwargs)
        self.pool = None
        self.parallel_totals = dict(collections=0, traversals=0, snapshot_seconds=0., dispatch_seconds=0.,
                                    wait_seconds=0., validation_seconds=0., merge_seconds=0.,
                                    worker_collection_seconds=0., worker_cpu_seconds=0., worker_snapshot_seconds=0.,
                                    imbalance_seconds=0., peak_payload_bytes=0, peak_worker_rss_mib=0.,
                                    regret_rows=0, policy_rows=0, critic_rows=0)
        self.run_seed = int(run_seed)
        state = self._capture_rng_state()
        try:
            self.pool = pool_factory(collector_config, run_seed, object_store_bytes=object_store_bytes)
            self.parallel_runtime = deepcopy(self.pool.metadata)
        finally:
            # Ray startup, imports and port selection are not learner draws.
            self._restore_rng_state(state)

    def collect_training_data(self, player):
        return self._timed("collection", self._collect_parallel, player)

    def _collect_parallel(self, player):
        self.regret_trainers[player].reset_buffer()
        counts = partition(self.num_traversals)
        started = time.perf_counter()
        snapshot = inference_snapshot(self, player)
        self.parallel_totals["snapshot_seconds"] += time.perf_counter() - started
        state = self._capture_rng_state()
        try:
            results, timing = self.pool.collect(counts, player, self.num_iteration, snapshot)
        finally:
            # Only central replay insertion below may advance the learner RNG.
            self._restore_rng_state(state)
        validation_started = time.perf_counter()
        expected_ids = [i for i, count in enumerate(counts) if count]
        if [r["worker_id"] for r in results] != expected_ids:
            raise ValueError("Missing, reordered or duplicate traversal workers")
        row_counts = []
        for result in results:
            i = result["worker_id"]
            if (tuple(result["token"]) != (self.num_iteration, player) or result["traversals"] != counts[i]
                    or result["seed"] != collection_seed(self.run_seed, i, self.num_iteration, player)
                    or result["nodes"] < counts[i]):
                raise ValueError("Wrong traversal budget, stream or snapshot version")
            returned = {}
            for trainer, key in ((self.regret_trainers[player], "regret"),
                                 (self.ave_policy_trainer, "policy"), (self.q_value_trainer, "critic")):
                rows = validate_rows(trainer.buffer, result[key])
                if rows > counts[i] * self.game.max_game_length():
                    raise ValueError("Traversal response exceeds its bounded row budget")
                returned[key] = rows
            # Each visited decision supplies one critic row and either a
            # traverser-regret or opponent-policy row. A hand has a decision.
            if (returned["critic"] < counts[i] or returned["critic"] != returned["regret"] + returned["policy"]
                    or payload_bytes(result) != result["payload_bytes"]):
                raise ValueError("Incomplete experience rows or incorrect payload accounting")
            row_counts.append(returned)
        self.parallel_totals["validation_seconds"] += time.perf_counter() - validation_started
        merge_start = time.perf_counter()
        for result in results:
            append_reservoir(self.regret_trainers[player].buffer, result["regret"])
            append_reservoir(self.ave_policy_trainer.buffer, result["policy"])
            append_circular(self.q_value_trainer.buffer, result["critic"])
        self.parallel_totals["merge_seconds"] += time.perf_counter() - merge_start
        self.nodes_touched += sum(r["nodes"] for r in results)
        self.episode += sum(r["traversals"] for r in results)
        totals = self.parallel_totals
        totals["collections"] += 1
        totals["traversals"] += self.num_traversals
        for key in ("regret", "policy", "critic"):
            totals[key + "_rows"] += sum(row[key] for row in row_counts)
        for key, value in timing.items():
            totals[key] += value
        for key, field in (("worker_collection_seconds", "collection_seconds"), ("worker_cpu_seconds", "cpu_seconds"),
                           ("worker_snapshot_seconds", "snapshot_load_seconds")):
            totals[key] += sum(r[field] for r in results)
        totals["imbalance_seconds"] += max(r["total_seconds"] for r in results) - min(r["total_seconds"] for r in results)
        totals["peak_payload_bytes"] = max(totals["peak_payload_bytes"], sum(r["payload_bytes"] for r in results))
        totals["peak_worker_rss_mib"] = max(totals["peak_worker_rss_mib"], max(r["peak_rss_mib"] for r in results))
        self._record_parallel_metrics()

    def _record_parallel_metrics(self):
        for key, value in self.parallel_totals.items():
            self.logger.record("parallel_" + key, value)
        self.logger.record("parallel_workers", WORKER_COUNT)
        self.logger.record("parallel_startup_seconds", self.parallel_runtime["startup_seconds"])

    def evaluate(self, **kwargs):
        # The inherited FHP method emits diagnostics only. Re-record counters
        # because several time thresholds can be crossed by one long iteration.
        self._record_parallel_metrics()
        return super().evaluate(**kwargs)

    def solve(self, *args, **kwargs):
        try:
            return super().solve(*args, **kwargs)
        finally:
            self.close()

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None


def make_solver(seed, training_config, *, smoke=False, pool_factory=RayTraversalPool):
    def construct(**kwargs):
        return ParallelVRDeepPDCFRPlus(collector_config=training_config, run_seed=seed,
                                      pool_factory=pool_factory,
                                      object_store_bytes=SMOKE_OBJECT_STORE_BYTES if smoke else OBJECT_STORE_BYTES,
                                      **kwargs)
    return timed.make_solver(seed, training_config, solver_class=construct)
