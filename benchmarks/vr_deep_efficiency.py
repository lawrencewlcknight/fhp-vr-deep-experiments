"""Compare the optimised implementation with a pinned, unmodified Git version.

Run with ``python -m benchmarks.vr_deep_efficiency`` from a full checkout.
Loads reference source in memory, without checking out files or changing HEAD.
Only timings are allowed to differ; validation raises on numerical differences.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import types

import numpy as np
import torch

from vr_deep_cfr import variants
from vr_deep_cfr.logger import Logger


REFERENCE_COMMIT = "4b5d6c8d32126dd34fff887a12d0b9fa88096136"
ROOT = Path(__file__).resolve().parents[1]


def load_reference():
    name = "_vr_deep_efficiency_reference"
    if name + ".variants" in sys.modules:
        return sys.modules[name + ".variants"]
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
    for filename in ("logger", "solver", "variants"):
        source = subprocess.run(
            ["git", "show", f"{REFERENCE_COMMIT}:vr_deep_cfr/{filename}.py"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        module = types.ModuleType(name + "." + filename)
        module.__package__ = name
        sys.modules[module.__name__] = module
        exec(compile(source, f"{REFERENCE_COMMIT}/{filename}.py", "exec"), module.__dict__)
    return sys.modules[name + ".variants"]


def run_fixed_work(module, *, seed, iterations=3, traversals=256, capacity=4096,
                   batch=64, regret_steps=16, critic_steps=53, policy_steps=16,
                   layers=3, width=64, predictive=True, cache=True):
    cls = module.VRDeepPDCFRPlus if predictive else module.VRDeepDCFRPlus
    extra = dict(reinitialize_imm_regret_networks=True) if predictive else {}
    started = time.perf_counter()
    solver = cls(
        game_name="FHP", num_episodes=iterations * traversals * 2,
        num_traversals=traversals, advantage_buffer_size=capacity,
        ave_policy_buffer_size=capacity, baseline_buffer_size=capacity,
        advantage_batch_size=batch, ave_policy_batch_size=batch, baseline_batch_size=batch,
        advantage_network_train_steps=regret_steps, baseline_network_train_steps=critic_steps,
        ave_policy_network_train_steps=policy_steps, num_layers=layers, num_hiddens=width,
        learning_rate=0.001, alpha=2.3 if predictive else 2.0, gamma=2.0,
        evaluation_frequency=1, reinitialize_advantage_networks=False, use_baseline=True,
        device="cpu", seed=seed, logger=Logger(verbose=False), **extra,
    )
    initialization_seconds = time.perf_counter() - started
    trainers = [*solver.regret_trainers, solver.q_value_trainer, solver.ave_policy_trainer]
    for trainer in trainers:
        trainer.cache_frozen_predictions = cache
    phase_seconds = {}
    for name in ("collect_training_data", "train_regret", "train_baseline", "train_average_policy"):
        original = getattr(solver, name)
        def measured(*args, _name=name, _fn=original, **kwargs):
            start = time.perf_counter()
            result = _fn(*args, **kwargs)
            phase_seconds[_name] = phase_seconds.get(_name, 0.0) + time.perf_counter() - start
            return result
        setattr(solver, name, measured)
    started = time.perf_counter()
    rows = solver.solve()
    training_seconds = time.perf_counter() - started
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    # Compare everything consumed by fitting, not float64 backing storage that
    # was always converted to float32. Full-buffer reads consume no RNG.
    state = []
    array_bytes = 0
    for trainer in trainers:
        entry = {}
        for name in ("model", "target_model", "imm_model", "best_model", "optimizer", "imm_optimizer"):
            if hasattr(trainer, name):
                entry[name] = copy.deepcopy(getattr(trainer, name).state_dict())
        entry["replay"] = tuple(t.clone() for t in trainer.buffer.sample(-1))
        entry["seen_or_write_index"] = trainer.buffer.cur_id
        entry["size"] = len(trainer.buffer)
        state.append(entry)
        array_bytes += sum(v.nbytes for v in vars(trainer.buffer).values() if isinstance(v, np.ndarray))
    observations = [
        {k: v for k, v in row.items() if not k.endswith("seconds")}
        for row in rows
    ]
    return dict(
        state=state, rng=rng, observations=observations,
        nodes=solver.nodes_touched, training_seconds=training_seconds,
        initialization_seconds=initialization_seconds, phase_seconds=phase_seconds,
        replay_array_bytes=array_bytes,
    )


def assert_exact(a, b, path="state"):
    """Recursive bit-for-bit numeric comparison, with useful failure paths."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path}: tensors differ; max abs {(a-b).abs().max().item()}"
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_exact(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (left, right) in enumerate(zip(a, b)):
            assert_exact(left, right, f"{path}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def compare(reference, candidate):
    for key in ("state", "rng", "observations", "nodes"):
        assert_exact(reference[key], candidate[key], key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--traversals", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--regret-steps", type=int, default=16)
    parser.add_argument("--critic-steps", type=int, default=53)
    parser.add_argument("--policy-steps", type=int, default=16)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reference_module = load_reference()
    results = []
    settings = dict(iterations=args.iterations, traversals=args.traversals, capacity=args.capacity,
                    batch=args.batch, regret_steps=args.regret_steps, critic_steps=args.critic_steps,
                    policy_steps=args.policy_steps)
    for index, seed in enumerate(args.seeds):
        runs = {}
        # Alternate order to reduce one-sided warm-up/timing effects.
        for label in (("reference", "optimised") if index % 2 == 0 else ("optimised", "reference")):
            runs[label] = run_fixed_work(
                reference_module if label == "reference" else variants,
                seed=seed, cache=not args.no_cache, **settings,
            )
        compare(runs["reference"], runs["optimised"])
        result = {"seed": seed, "exact": True, "nodes": runs["reference"]["nodes"]}
        for label, run in runs.items():
            result[label] = {k: v for k, v in run.items() if k not in ("state", "rng", "observations", "nodes")}
        result["speedup"] = runs["reference"]["training_seconds"] / runs["optimised"]["training_seconds"]
        results.append(result)
        print(json.dumps(result), flush=True)
    print(json.dumps({"reference_commit": REFERENCE_COMMIT, "all_exact": True,
                      "median_speedup": float(np.median([r["speedup"] for r in results]))}))


if __name__ == "__main__":
    main()
