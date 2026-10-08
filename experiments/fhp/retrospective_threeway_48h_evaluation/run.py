"""Prespecified final-policy league for the three completed 48-hour FHP runs.

This module never trains or transforms a policy.  It validates and evaluates
the 48-hour checkpoints from SD-CFR Experiment 6, UCV-ESCHER Experiment 16 and
VR-Deep Experiment 6.  Every direct comparison uses all nine cross-seed cells.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
from importlib.metadata import version
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

SEEDS = (0, 1, 2)
ALGORITHMS = ("sd", "ucv", "vr")
COMPARISONS = (
    ("sd_vs_ucv", "sd", "ucv"),
    ("sd_vs_vr", "sd", "vr"),
    ("ucv_vs_vr", "ucv", "vr"),
)
LABELS = {"sd": "SD-CFR Exp6", "ucv": "UCV-ESCHER Exp16", "vr": "VR-Deep Exp6"}
RUN_IDS = {"sd": "sdcfr6-48h-20261002-161544",
           "ucv": "exp16-feat48-20261004-182051",
           "ucv_source10": "exp10-features-20261001-161740",
           "vr": "vr6-ray48-20261003-221425"}
SOURCE_COMMITS = {"sd": "bb689252d6b322adb3de6102d095a49c3ed87250",
                  "ucv": "e66d4da515eb212e5026a965ac5c39c86144c901",
                  "vr": "6b95a3ecd191f02bd6972c72a3bf8514483ae02b"}
SOURCE_NAMES = {
    "sd": ("exp6_sd_cfr_parallel_48h", "parallel_structured_uniform_sd_cfr_48h"),
    "ucv": ("exp10_fhp_hand_board_features", "hand_board_cached_parallel_ucv_escher"),
    "vr": ("exp6_fhp_vr_deep_ray8_48h", "lossless_vr_deep_pdcfr_plus_ray8_48h"),
}
CONFIG_SHA256 = {"sd": "4e7993aa32d755d9b3f69140c257b3ffc2ee03a48187aa43dd3be760172fc576",
                 "ucv": "065734572e18f4597dd8f494ea39230c1f193b9d68aa775138c4a8a3e39eddad",
                 "vr": "dfbbfc38f8cede42aa27844994af2c40f639f46ef42f29f23efd8ba629a0746f"}
PROTOCOL = "fhp_threeway_final_48h_cross_seed_v1"
DIRECT_PAIRS = 100_000
RULE_PAIRS = 10_000
LBR_PAIRS = 1_000
SHARD_PAIRS = 5_000
LBR_SHARD_PAIRS = 10
LBR_ROLLOUTS = 4096
BASE_SEED = 601_048
FAMILY_CONFIDENCE = 1.0 - 0.05 / len(COMPARISONS)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()
                if key not in {"path", "repository_roots"}}
    if isinstance(value, (tuple, list)):
        return [portable(item) for item in value]
    return value


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes root: {relative}")
    return path


def canonical_config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def setup_repositories(sd_repo, ucv_repo):
    sd_repo, ucv_repo = Path(sd_repo).resolve(), Path(ucv_repo).resolve()
    for repo, file in ((sd_repo, "deep_cfr_poker/sd_cfr_disk.py"),
                       (sd_repo, "fhp_evaluation/duplicate.py"),
                       (ucv_repo, "fhp_escher/checkpointing.py")):
        if not (repo / file).is_file():
            raise ValueError(f"Required source package is missing: {repo / file}")
    # The VR checkout stays first so UCV's loader uses the audited compatible
    # VR MLP primitives instead of importing a second vr_deep_cfr package.
    for repo in (sd_repo, ucv_repo):
        if str(repo) not in sys.path:
            sys.path.append(str(repo))
    import deep_cfr_poker.sd_cfr_disk as sd_disk
    import fhp_escher.checkpointing as ucv_checkpointing
    import fhp_evaluation.duplicate as duplicate
    if Path(sd_disk.__file__).resolve() != sd_repo / "deep_cfr_poker/sd_cfr_disk.py":
        raise ValueError("A different SD-CFR loader was imported")
    if Path(ucv_checkpointing.__file__).resolve() != ucv_repo / "fhp_escher/checkpointing.py":
        raise ValueError("A different UCV loader was imported")
    if Path(duplicate.__file__).resolve() != sd_repo / "fhp_evaluation/duplicate.py":
        raise ValueError("Use the pinned evaluator with historical-mixture episode hooks")


def implementation_digest(sd_repo, ucv_repo):
    root = Path(__file__).resolve().parents[3]
    files = {}
    for prefix, repo, directories in (
        ("vr", root, ("fhp_vr_deep", "vr_deep_cfr", "experiments/fhp/retrospective_threeway_48h_evaluation")),
        ("sd", Path(sd_repo), ("deep_cfr_poker", "fhp_evaluation")),
        ("ucv", Path(ucv_repo), ("fhp_escher",)),
    ):
        for directory in directories:
            base = repo / directory
            for path in sorted(base.rglob("*.py")):
                files[f"{prefix}/{path.relative_to(repo)}"] = sha256(path)
    return digest(files)


def _worker(root, algorithm, seed):
    identifier = SOURCE_NAMES[algorithm][1]
    return Path(root) / "workers" / f"task_{seed:03d}_{identifier}_seed_{seed}"


def _validate_common(manifest, algorithm, seed):
    from fhp_vr_deep.game import FHP_GAME_PARAMETERS
    expected_name, expected_id = SOURCE_NAMES[algorithm]
    if (manifest.get("repository_commit") != SOURCE_COMMITS[algorithm]
            or manifest.get("experiment_name") != expected_name
            or manifest.get("algorithm_id") != expected_id
            or int(manifest.get("seed", -1)) != seed
            or manifest.get("game", {}).get("parameters") != dict(FHP_GAME_PARAMETERS)
            or manifest.get("reference_vm", {}).get("machine_type") != "n2-standard-16"):
        raise ValueError(f"Unexpected {algorithm} source identity/game/VM for seed {seed}")
    config = manifest.get("config", manifest.get("training_config"))
    if canonical_config_hash(config) != CONFIG_SHA256[algorithm]:
        raise ValueError(f"Unexpected {algorithm} training configuration")


def _final_row(rows, algorithm):
    if len(rows) != 8 or [int(row["checkpoint_target_hours"]) for row in rows] != list(range(6, 49, 6)):
        raise ValueError(f"{algorithm}: expected the complete six-hour checkpoint schedule")
    row = rows[-1]
    if (int(row["checkpoint_target_hours"]) != 48
            or float(row["actual_training_elapsed_seconds"]) < 48 * 3600
            or int(row["nodes_touched"]) <= 0):
        raise ValueError(f"{algorithm}: invalid final checkpoint boundary")
    return row


def _validate_ucv_lineage(root, source10_root, seed, final_record):
    worker = _worker(root, "ucv", seed)
    source = _worker(source10_root, "ucv", seed)
    lineage = read_json(worker / "continuation_source.json")
    source_manifest = read_json(source / "run_manifest.json")
    source_rows = read_json(source / "checkpoint_manifest.json")
    source_success = read_json(source / "SUCCESS.json")
    if (source_manifest.get("repository_commit") != SOURCE_COMMITS["ucv"]
            or source_manifest.get("experiment_name") != "exp10_fhp_hand_board_features"
            or int(source_manifest.get("seed", -1)) != seed):
        raise ValueError("Invalid UCV Experiment 10 lineage source")
    source_final = source_rows[-1]
    expected = dict(seed=seed, source_total_hours=24, total_hours=48,
                    source_commit=SOURCE_COMMITS["ucv"],
                    source_state_path=source_final.get("training_state_path"),
                    source_state_sha256=source_final.get("training_state_sha256"),
                    source_summary_sha256=source_success.get("summary_sha256"))
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid UCV continuation lineage for seed {seed}")
    if not lineage.get("source_worker", "").endswith(f"/{RUN_IDS['ucv_source10']}/workers/{source.name}"):
        raise ValueError("UCV Experiment 16 used a different source cohort")
    final_record.update(continuation_source_sha256=sha256(worker / "continuation_source.json"),
                        source10_run_manifest_sha256=sha256(source / "run_manifest.json"),
                        source10_checkpoint_manifest_sha256=sha256(source / "checkpoint_manifest.json"),
                        source10_success_sha256=sha256(source / "SUCCESS.json"))


def validate_sources(source_root, sd_repo, ucv_repo):
    """Return exactly nine verified final-policy records."""
    setup_repositories(sd_repo, ucv_repo)
    from deep_cfr_poker.game import load_fhp_game
    from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader
    from fhp_escher.checkpointing import LoadedFHPPolicy
    from fhp_vr_deep.evaluation_adapter import load_policy_for_evaluation
    game = load_fhp_game()
    roots = {name: Path(source_root) / name for name in ("sd", "ucv", "vr")}
    source10 = Path(source_root) / "ucv_source10"
    records, verified_chunks = [], {}
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            worker = _worker(roots[algorithm], algorithm, seed)
            contained(roots[algorithm], worker.relative_to(roots[algorithm]))
            manifest = read_json(worker / "run_manifest.json")
            success = read_json(worker / "SUCCESS.json")
            _validate_common(manifest, algorithm, seed)
            if ((worker / "FAILURE.json").exists()
                    or bool(manifest.get("smoke", manifest.get("is_smoke", False)))):
                raise ValueError("Failed or smoke source worker")
            checkpoint_rows = read_json(worker / "checkpoint_manifest.json")
            row = _final_row(checkpoint_rows, algorithm)
            path = contained(worker, row["path"])
            if sha256(path) != row["sha256"] or path.stat().st_size != int(row.get("size_bytes", path.stat().st_size)):
                raise ValueError(f"Final {algorithm} checkpoint hash/size mismatch")
            if algorithm == "sd":
                if (success.get("checkpoints") != 8
                        or success.get("seed") != seed
                        or success.get("completed_iterations") != row["outer_iteration"]
                        or success.get("final_nodes") != row["nodes_touched"]
                        or manifest.get("strategy_weighting") != "uniform"
                        or manifest.get("parallel_execution", {}).get("workers") != 8):
                    raise ValueError("Unexpected SD-CFR completion/deployment contract")
                archive = read_json(path)
                for chunk in archive["chunks"]:
                    chunk_path = contained(path.parent, chunk["path"])
                    if chunk_path not in verified_chunks:
                        verified_chunks[chunk_path] = sha256(chunk_path)
                    if verified_chunks[chunk_path] != chunk["sha256"]:
                        raise ValueError("SD-CFR archive chunk hash mismatch")
                reader = DiskArchiveReader(path, game, verify=False)
                metadata = archive.get("metadata", {})
                if (reader.count != int(row["outer_iteration"])
                        or canonical_config_hash(metadata.get("solver_config")) != CONFIG_SHA256["sd"]
                        or metadata.get("feature_encoder") != manifest.get("feature_encoder")
                        or metadata.get("parallel_execution") != manifest.get("parallel_execution")):
                    raise ValueError("SD-CFR archive provenance/iteration mismatch")
            elif algorithm == "ucv":
                runtime = read_json(worker / "runtime_manifest.json")
                if (success.get("status") != "complete"
                        or success.get("seed") != seed or success.get("checkpoints") != checkpoint_rows
                        or runtime.get("frozen_critic_target_cache") is not True
                        or runtime.get("traversal_execution") != "ray_parallel"):
                    raise ValueError("Unexpected UCV execution/completion contract")
                loaded = LoadedFHPPolicy(game, path)
                payload = loaded.checkpoint
                if (payload.get("seed") != seed or payload.get("algorithm_id") != SOURCE_NAMES["ucv"][1]
                        or payload.get("experiment_name") != SOURCE_NAMES["ucv"][0]
                        or payload.get("checkpoint_target_seconds") != 48*3600
                        or payload.get("nodes_touched") != row["nodes_touched"]
                        or payload.get("outer_iteration") != row["outer_iteration"]
                        or payload.get("feature_encoder") != manifest.get("feature_encoder")
                        or canonical_config_hash(payload.get("training_config")) != CONFIG_SHA256["ucv"]):
                    raise ValueError("UCV checkpoint payload mismatch")
            else:
                for name in ("run_manifest.json", "checkpoint_manifest.json", "summary.json"):
                    if success.get("files", {}).get(name) != sha256(worker / name):
                        raise ValueError("VR-Deep worker metadata differs from SUCCESS")
                summary = read_json(worker / "summary.json")
                if (success.get("experiment_name") != SOURCE_NAMES["vr"][0]
                        or success.get("seed") != seed
                        or success.get("training_config_sha256") != CONFIG_SHA256["vr"]
                        or success.get("files", {}).get(row["path"]) != row["sha256"]
                        or summary.get("final_policy_sha256") != row["sha256"]
                        or summary.get("final_policy_snapshot") != row["path"]
                        or manifest.get("parallel_traversal_workers") != 8
                        or manifest.get("start_hours") != 0 or manifest.get("target_hours") != 48):
                    raise ValueError("Unexpected VR-Deep completion/execution contract")
                _, loaded = load_policy_for_evaluation(path)
                payload = loaded.metadata
                if (payload.get("seed") != seed or payload.get("algorithm_id") != SOURCE_NAMES["vr"][1]
                        or payload.get("nodes_touched") != row["nodes_touched"]
                        or payload.get("iteration") != row["iteration"]
                        or payload.get("feature_encoder") != manifest.get("feature_encoder")
                        or canonical_config_hash(payload.get("authors_parameterisation")) != CONFIG_SHA256["vr"]
                        or payload.get("checkpoint", {}).get("checkpoint_id") != "time_48h"
                        or payload.get("checkpoint", {}).get("training_elapsed_seconds")
                           != row["training_elapsed_seconds"]):
                    raise ValueError("VR-Deep checkpoint payload mismatch")
            record = dict(id=f"{algorithm}_s{seed}_48h", algorithm=algorithm,
                          label=LABELS[algorithm], seed=seed, training_hours=48,
                          active_seconds=float(row["actual_training_elapsed_seconds"]),
                          nodes_touched=int(row["nodes_touched"]),
                          outer_iteration=int(row.get("outer_iteration", row.get("iteration"))),
                          path=str(path), sha256=row["sha256"], source_run=RUN_IDS[algorithm],
                          source_commit=SOURCE_COMMITS[algorithm],
                          run_manifest_sha256=sha256(worker / "run_manifest.json"),
                          checkpoint_manifest_sha256=sha256(worker / "checkpoint_manifest.json"),
                          continuation_source_sha256="", source10_run_manifest_sha256="",
                          source10_checkpoint_manifest_sha256="", source10_success_sha256="")
            if algorithm == "ucv":
                _validate_ucv_lineage(roots[algorithm], source10, seed, record)
            records.append(record)
    if sorted((row["algorithm"], row["seed"]) for row in records) != list(itertools.product(ALGORITHMS, SEEDS)):
        raise ValueError("Expected three distinct seeds for all three algorithms")
    return records


def _seed(stage, *parts):
    codes = {"smoke": 1, "profile": 2, "production": 3}
    return int(np.random.SeedSequence([BASE_SEED, codes[stage], *map(int, parts)]).generate_state(1)[0])


def build_tasks(records, stage, implementation):
    if stage not in {"smoke", "profile", "production", "lbr_smoke", "lbr_profile", "lbr"}:
        raise ValueError(stage)
    index = {(row["algorithm"], row["seed"]): row for row in records}
    tasks = []
    if stage.startswith("lbr"):
        # Even smoke shards need two independent pairs for finite sample
        # variance and uncertainty fields in the strictly serialized result.
        total = LBR_PAIRS if stage == "lbr" else (10 if stage == "lbr_profile" else 2)
        shard_size = LBR_SHARD_PAIRS if stage == "lbr" else total
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for training_seed in SEEDS:
                cell = f"lbr_{algorithm}_s{training_seed}"
                for shard, start in enumerate(range(0, total, shard_size)):
                    tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell, kind="lbr",
                        a=index[algorithm, training_seed], opponent="local_best_response",
                        num_deals=min(shard_size, total-start), evaluation_seed=_seed(
                            "production" if stage == "lbr" else stage.removeprefix("lbr_"),
                            900, algorithm_index, training_seed, shard),
                        lbr_seed=BASE_SEED + 800_000,
                        rollouts=16 if stage == "lbr_smoke" else LBR_ROLLOUTS, stage=stage,
                        protocol=PROTOCOL, implementation=implementation))
        return tasks
    direct_total = DIRECT_PAIRS if stage == "production" else (128 if stage == "profile" else 2)
    rule_total = RULE_PAIRS if stage == "production" else (128 if stage == "profile" else 2)
    from fhp_evaluation.rule_agents import PUBLISHED_AGENT_NAMES
    for comparison_index, (comparison_id, left, right) in enumerate(COMPARISONS):
        for left_seed, right_seed in itertools.product(SEEDS, repeat=2):
            cell = f"{comparison_id}_{left_seed}_{right_seed}"
            shard_size = SHARD_PAIRS if stage == "production" else direct_total
            for shard, start in enumerate(range(0, direct_total, shard_size)):
                tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell, kind="direct",
                    comparison_id=comparison_id, a=index[left, left_seed], b=index[right, right_seed],
                    num_deals=min(shard_size, direct_total-start),
                    # Common random numbers across the three algorithm pairings.
                    evaluation_seed=_seed(stage, 100, left_seed, right_seed, shard), stage=stage,
                    protocol=PROTOCOL, implementation=implementation))
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        for training_seed in SEEDS:
            for opponent_index, opponent in enumerate(PUBLISHED_AGENT_NAMES):
                cell = f"rule_{algorithm}_s{training_seed}_{opponent}"
                shard_size = SHARD_PAIRS if stage == "production" else rule_total
                for shard, start in enumerate(range(0, rule_total, shard_size)):
                    tasks.append(dict(task_id=f"{cell}_{shard:03d}", cell_id=cell, kind="rule",
                        a=index[algorithm, training_seed], opponent=opponent,
                        num_deals=min(shard_size, rule_total-start),
                        evaluation_seed=_seed(stage, 500, opponent_index, training_seed, shard),
                        stage=stage, protocol=PROTOCOL, implementation=implementation))
    return tasks


_POLICIES = OrderedDict()


def initialise_worker(sd_repo, ucv_repo):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setup_repositories(sd_repo, ucv_repo)


def loaded_policy(record, game, *, behavioural=False):
    key = record["algorithm"], record["sha256"], behavioural
    if key not in _POLICIES:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("Checkpoint changed after validation")
        if record["algorithm"] == "sd":
            from deep_cfr_poker.sd_cfr_disk import DiskArchiveReader, DiskBehaviouralPolicy, DiskSampledPolicy
            reader = DiskArchiveReader(record["path"], game, verify=False)
            value = DiskBehaviouralPolicy(reader, game) if behavioural else DiskSampledPolicy(reader)
        elif record["algorithm"] == "ucv":
            from fhp_escher.checkpointing import LoadedFHPPolicy
            value = LoadedFHPPolicy(game, record["path"])
        else:
            from vr_deep_cfr.policy_snapshots import LoadedVRPolicy
            value = LoadedVRPolicy(game, record["path"])
        _POLICIES[key] = value
    _POLICIES.move_to_end(key)
    while len(_POLICIES) > 4:
        _POLICIES.popitem(last=False)
    return _POLICIES[key]


def execute_task(task):
    if task["num_deals"] < 2:
        raise ValueError("Evaluation shards require at least two duplicate-deal pairs")
    from deep_cfr_poker.game import load_fhp_game
    from fhp_evaluation.duplicate import evaluate_duplicate_match
    from fhp_evaluation.lbr import LBRConfig, LocalBestResponsePolicy
    from fhp_evaluation.rule_agents import published_rule_agents
    started = time.perf_counter()
    game = load_fhp_game()
    target = loaded_policy(task["a"], game, behavioural=task["kind"] == "lbr")
    if task["kind"] == "direct":
        left, right = target, loaded_policy(task["b"], game)
        left_name, right_name = task["a"]["label"], task["b"]["label"]
    elif task["kind"] == "rule":
        left, right = target, published_rule_agents(game)[task["opponent"]]
        left_name, right_name = task["a"]["label"], task["opponent"]
    elif task["kind"] == "lbr":
        left = LocalBestResponsePolicy(game, target, config=LBRConfig(
            preflop_rollout_samples=task["rollouts"], seed=task["lbr_seed"]))
        right, left_name, right_name = target, "local_best_response", task["a"]["label"]
    else:
        raise ValueError(task["kind"])
    result = evaluate_duplicate_match(game, left, right, num_deals=task["num_deals"],
        seed=task["evaluation_seed"], seed_layout="split", policy_a_name=left_name,
        policy_b_name=right_name).to_dict()
    row = dict(task=portable(task), result=result, elapsed_seconds=time.perf_counter()-started)
    row["result_sha256"] = digest(row)
    return row


def validate_result(row, task):
    body = {key: value for key, value in row.items() if key != "result_sha256"}
    if (digest(body) != row.get("result_sha256") or row.get("task") != portable(task)
            or row["result"].get("num_deal_pairs") != task["num_deals"]
            or row["result"].get("num_games") != 2 * task["num_deals"]):
        raise ValueError("Corrupt or incompatible cached match shard")
    for key in ("mean_chips_per_hand", "std_chips_per_pair", "mean_mbb_per_hand"):
        if not math.isfinite(row["result"][key]):
            raise ValueError("Non-finite evaluation result")


def run_tasks(tasks, output, *, workers, sd_repo, ucv_repo, deadline=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending, results = [], []
    for task in tasks:
        path = output / (task["task_id"] + ".json")
        if path.exists():
            row = read_json(path)
            validate_result(row, task)
            results.append(row)
        else:
            pending.append(task)
    if not pending:
        return results
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialise_worker,
                               initargs=(str(sd_repo), str(ucv_repo)))
    aborted = False
    try:
        iterator, futures = iter(pending), {}
        def submit_one():
            task = next(iterator, None)
            if task is not None:
                futures[pool.submit(execute_task, task)] = task
        for _ in range(workers):
            submit_one()
        while futures:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Evaluation budget reached; completed shards retained for resume")
            wait_seconds = 5 if deadline is None else max(.001, min(5, deadline-time.monotonic()))
            done, _ = wait(futures, timeout=wait_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                row = future.result()
                validate_result(row, task)
                write_json(output / (task["task_id"] + ".json"), row)
                results.append(row)
                if deadline is None or time.monotonic() < deadline:
                    submit_one()
                print(f"{task['stage']}: {len(results)}/{len(tasks)} shards complete", flush=True)
    except BaseException:
        aborted = True
        for process in tuple(pool._processes.values()):
            if process.is_alive():
                process.terminate()
        raise
    finally:
        pool.shutdown(wait=not aborted, cancel_futures=True)
    if len(results) != len(tasks):
        raise TimeoutError("Evaluation budget reached; completed shards retained for resume")
    return results


def cost_estimate(probes, tasks, workers):
    rates = defaultdict(list)
    def key(task):
        return (task["kind"], task["a"]["algorithm"], task.get("b", {}).get("algorithm"),
                task.get("opponent"))
    for row in probes:
        rates[key(row["task"])].append(row["elapsed_seconds"] / row["task"]["num_deals"])
    costs = []
    for task in tasks:
        if not rates[key(task)]:
            raise ValueError(f"Unprofiled task class: {key(task)}")
        costs.append(max(rates[key(task)]) * task["num_deals"])
    seconds = 2.0 * (sum(costs) / workers + max(costs, default=0)) + 600
    return dict(predicted_hours_with_2x_margin=seconds/3600, workers=workers,
                remaining_duplicate_pairs=sum(task["num_deals"] for task in tasks),
                timing_is_estimate_not_guarantee=True)


def pool_shards(rows):
    n = sum(row["result"]["num_deal_pairs"] for row in rows)
    mean = sum(row["result"]["num_deal_pairs"] * row["result"]["mean_chips_per_hand"] for row in rows) / n
    ss = sum((row["result"]["num_deal_pairs"]-1) * row["result"]["std_chips_per_pair"]**2
             + row["result"]["num_deal_pairs"] * (row["result"]["mean_chips_per_hand"]-mean)**2
             for row in rows)
    se = math.sqrt(ss/(n-1)/n) * 10
    return dict(duplicate_pairs=n, hands=2*n, mean_mbb_per_hand=mean*10,
                mc_se_mbb_per_hand=se, mc_ci95_low=mean*10-1.96*se,
                mc_ci95_high=mean*10+1.96*se,
                policy_a_player0_mean_mbb=10*sum(row["result"]["num_deal_pairs"] * row["result"]["policy_a_player0_mean_chips"] for row in rows)/n,
                policy_a_player1_mean_mbb=10*sum(row["result"]["num_deal_pairs"] * row["result"]["policy_a_player1_mean_chips"] for row in rows)/n)


def cluster_interval(matrix, *, confidence=.95, draws=10_000, seed=871_223):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Bootstrap requires a complete 3x3 cross-seed matrix")
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 3, (draws, 3))
    right = rng.integers(0, 3, (draws, 3))
    values = matrix[left[:, :, None], right[:, None, :]].mean(axis=(1, 2))
    tail = (1-confidence)/2
    return tuple(float(value) for value in np.quantile(values, (tail, 1-tail)))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    # Preserve the first-seen column order while accepting audit fields that
    # apply only to some policy families (for example UCV continuation
    # lineage). csv.DictWriter otherwise infers a partial schema from row zero
    # and rejects later records with additional provenance fields.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _student_summary(values):
    from fhp_evaluation.statistics import sample_summary
    result = sample_summary(values)
    return {f"seed_{key}": value for key, value in result.items()}


def report_main(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell_id in sorted({row["task"]["cell_id"] for row in results}):
        selected = [row for row in results if row["task"]["cell_id"] == cell_id]
        task = selected[0]["task"]
        pooled = pool_shards(selected)
        if task["kind"] == "direct":
            cells.append(dict(cell_id=cell_id, kind="direct", comparison_id=task["comparison_id"],
                algorithm=task["a"]["algorithm"], opponent=task["b"]["algorithm"],
                seed=task["a"]["seed"], opponent_seed=task["b"]["seed"],
                nodes_touched=task["a"]["nodes_touched"], opponent_nodes=task["b"]["nodes_touched"], **pooled))
        else:
            cells.append(dict(cell_id=cell_id, kind="rule", comparison_id="",
                algorithm=task["a"]["algorithm"], opponent=task["opponent"], seed=task["a"]["seed"],
                opponent_seed="", nodes_touched=task["a"]["nodes_touched"], opponent_nodes="", **pooled))
    direct_summaries, matrices = [], {}
    for index, (comparison_id, left, right) in enumerate(COMPARISONS):
        selected = [row for row in cells if row["comparison_id"] == comparison_id]
        if sorted((row["seed"], row["opponent_seed"]) for row in selected) != list(itertools.product(SEEDS, repeat=2)):
            raise ValueError("Incomplete cross-seed matrix; refusing aggregate")
        if any(row["duplicate_pairs"] != DIRECT_PAIRS for row in selected):
            raise ValueError("Incomplete direct-play deal budget")
        matrix = np.empty((3, 3))
        for row in selected:
            matrix[row["seed"], row["opponent_seed"]] = row["mean_mbb_per_hand"]
        matrices[comparison_id] = matrix
        lo, hi = cluster_interval(matrix, seed=871_223+index)
        family_lo, family_hi = cluster_interval(matrix, confidence=FAMILY_CONFIDENCE, seed=881_223+index)
        direct_summaries.append(dict(comparison_id=comparison_id, left_algorithm=left, right_algorithm=right,
            positive_favours=left, mean_mbb_per_hand=float(matrix.mean()), cluster_ci95_low=lo,
            cluster_ci95_high=hi, family_ci98_33_low=family_lo, family_ci98_33_high=family_hi,
            positive_cells=int((matrix > 0).sum()), cross_seed_cells=9,
            conditional_mc_se_mbb=math.sqrt(sum(row["mc_se_mbb_per_hand"]**2 for row in selected))/9,
            left_nodes_mean=float(np.mean([row["nodes_touched"] for row in selected])),
            right_nodes_mean=float(np.mean([row["opponent_nodes"] for row in selected])),
            duplicate_pairs=sum(row["duplicate_pairs"] for row in selected)))
    rule_summaries = []
    for algorithm in ALGORITHMS:
        opponents = sorted({row["opponent"] for row in cells if row["kind"] == "rule"})
        for opponent in opponents:
            selected = sorted((row for row in cells if row["kind"] == "rule"
                               and row["algorithm"] == algorithm and row["opponent"] == opponent),
                              key=lambda row: row["seed"])
            if [row["seed"] for row in selected] != list(SEEDS) or any(row["duplicate_pairs"] != RULE_PAIRS for row in selected):
                raise ValueError("Incomplete rule-agent seed panel")
            rule_summaries.append(dict(algorithm=algorithm, opponent=opponent,
                nodes_mean=float(np.mean([row["nodes_touched"] for row in selected])),
                duplicate_pairs=sum(row["duplicate_pairs"] for row in selected),
                **_student_summary([row["mean_mbb_per_hand"] for row in selected])))
    write_csv(output / "matchups.csv", cells)
    write_csv(output / "direct_comparison_summary.csv", direct_summaries)
    write_csv(output / "rule_agent_summary.csv", rule_summaries)
    summary = dict(protocol=PROTOCOL, status="complete", training_checkpoint_hours=48,
        direct_comparisons=direct_summaries, rule_agent_results=rule_summaries,
        units="mbb/hand", family_confidence=FAMILY_CONFIDENCE,
        inference=("10,000 independent row/column training-seed bootstrap draws. The three co-primary "
                   "direct comparisons receive Bonferroni-compatible 98.33% intervals; pointwise 95% "
                   "intervals and conditional Monte Carlo SEs are also reported."),
        caveats=["Head-to-head value is not exploitability or proof of convergence.",
                 "Only three historical training seeds per algorithm are available.",
                 "Active-time accounting differs by algorithm; this is the prespecified 48-hour endpoint.",
                 "Node counts are descriptive only and are not used for matching or normalization."])
    write_json(output / "summary.json", summary)
    plots(matrices, direct_summaries, output)
    lines = ["# Final 48-hour three-way FHP evaluation", "",
             "Positive values favour the algorithm on the left. Values are mbb/hand.", "",
             "| Comparison | Mean | 95% cluster CI | Family-compatible 98.33% CI |",
             "|---|---:|---:|---:|"]
    for row in direct_summaries:
        lines.append(f"| {row['left_algorithm']} vs {row['right_algorithm']} | {row['mean_mbb_per_hand']:.3f} | "
                     f"[{row['cluster_ci95_low']:.3f}, {row['cluster_ci95_high']:.3f}] | "
                     f"[{row['family_ci98_33_low']:.3f}, {row['family_ci98_33_high']:.3f}] |")
    lines += ["", "Node counts are reported descriptively in `direct_comparison_summary.csv`; no node-matched endpoint is included.", ""]
    (output / "analysis_summary.md").write_text("\n".join(lines))
    return summary


def report_lbr(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cells = []
    for cell_id in sorted({row["task"]["cell_id"] for row in results}):
        selected = [row for row in results if row["task"]["cell_id"] == cell_id]
        task = selected[0]["task"]
        pooled = pool_shards(selected)
        if pooled["duplicate_pairs"] != LBR_PAIRS:
            raise ValueError("Incomplete LBR deal budget")
        cells.append(dict(cell_id=cell_id, algorithm=task["a"]["algorithm"], seed=task["a"]["seed"],
                          nodes_touched=task["a"]["nodes_touched"], rollouts=task["rollouts"], **pooled))
    summaries = []
    for algorithm in ALGORITHMS:
        selected = sorted((row for row in cells if row["algorithm"] == algorithm), key=lambda row: row["seed"])
        if [row["seed"] for row in selected] != list(SEEDS):
            raise ValueError("Incomplete LBR training-seed panel")
        summaries.append(dict(algorithm=algorithm, positive_favours="local_best_response",
            nodes_mean=float(np.mean([row["nodes_touched"] for row in selected])),
            duplicate_pairs=sum(row["duplicate_pairs"] for row in selected),
            **_student_summary([row["mean_mbb_per_hand"] for row in selected])))
    write_csv(output / "lbr_by_seed.csv", cells)
    write_csv(output / "lbr_summary.csv", summaries)
    summary = dict(protocol=PROTOCOL, status="complete", lower_bound_only=True,
                   preflop_rollouts=LBR_ROLLOUTS, results=summaries, units="mbb/hand",
                   positive_favours="local_best_response")
    write_json(output / "summary.json", summary)
    return summary


def plots(matrices, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    bound = max(max(float(np.abs(matrix).max()) for matrix in matrices.values()), .001)
    for axis, (comparison_id, left, right) in zip(axes, COMPARISONS):
        matrix = matrices[comparison_id]
        image = axis.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:.2f}", ha="center", va="center",
                      bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        axis.set(xticks=SEEDS, yticks=SEEDS, xlabel=f"{right} seed", ylabel=f"{left} seed",
                 title=f"{left} minus {right}")
    fig.colorbar(image, ax=axes, label="mbb/hand", shrink=.8)
    fig.suptitle("Final 48-hour cross-seed head-to-head matrices")
    fig.subplots_adjust(left=.06, right=.92, bottom=.14, top=.82, wspace=.35)
    fig.savefig(output / "final_48h_head_to_head.png", dpi=180)
    plt.close(fig)


def _complete_stage(output, identity):
    success = output / "SUCCESS.json"
    if not success.exists():
        return False
    saved = read_json(success)
    if saved.get("identity_sha256") != digest(identity):
        raise ValueError("Completed stage belongs to a changed evaluation contract")
    for name, expected in saved.get("files", {}).items():
        if sha256(contained(output, name)) != expected:
            raise ValueError("Completed stage output is corrupt")
    return True


def run_stage(args, records, implementation, identity):
    output = args.output / args.stage
    output.mkdir(parents=True, exist_ok=True)
    if _complete_stage(output, identity):
        write_json(output / "STATUS.json", dict(status="complete", reused=True, stage=args.stage))
        return
    if args.stage == "smoke":
        tasks = build_tasks(records, "smoke", implementation) + build_tasks(records, "lbr_smoke", implementation)
        results = run_tasks(tasks, output / "tasks", workers=args.workers,
                            sd_repo=args.sd_repo, ucv_repo=args.ucv_repo,
                            deadline=time.monotonic() + args.max_hours*3600)
        write_json(output / "SUCCESS.json", dict(smoke=True, tasks=len(results), strength_evidence=False,
                                                   identity_sha256=digest(identity), files={}))
        write_json(output / "STATUS.json", dict(status="complete", stage="smoke"))
        return
    production_stage = "production" if args.stage == "main" else "lbr"
    profile_stage = "profile" if args.stage == "main" else "lbr_profile"
    tasks = build_tasks(records, production_stage, implementation)
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    write_json(output / "STATUS.json", dict(status="profiling", stage=args.stage))
    try:
        probes = run_tasks(build_tasks(records, profile_stage, implementation),
                           output / "profile" / str(time.time_ns()), workers=args.workers,
                           sd_repo=args.sd_repo, ucv_repo=args.ucv_repo,
                           deadline=min(deadline, time.monotonic() + (1800 if args.stage == "lbr" else 900)))
    except TimeoutError:
        write_json(output / "STATUS.json", dict(status="deferred_pilot_timeout", stage=args.stage,
            message="Full scoring was not started; completed earlier stages are unaffected."))
        return
    remaining = []
    for task in tasks:
        cached = output / "task_results" / (task["task_id"] + ".json")
        if cached.exists():
            validate_result(read_json(cached), task)
        else:
            remaining.append(task)
    estimate = cost_estimate(probes, remaining, args.workers)
    estimate.update(limit_hours=args.max_hours,
                    passed=estimate["predicted_hours_with_2x_margin"] + (time.monotonic()-started)/3600 <= args.max_hours)
    write_json(output / "timing_pilot.json", estimate)
    if not estimate["passed"]:
        write_json(output / "STATUS.json", dict(status="deferred_cost_gate", stage=args.stage, **estimate))
        return
    write_json(output / "STATUS.json", dict(status="scoring", stage=args.stage))
    results = run_tasks(tasks, output / "task_results", workers=args.workers,
                        sd_repo=args.sd_repo, ucv_repo=args.ucv_repo, deadline=deadline)
    summary = report_main(results, output) if args.stage == "main" else report_lbr(results, output)
    files = {str(path.relative_to(output)): sha256(path) for path in output.iterdir()
             if path.is_file() and path.name not in {"STATUS.json", "SUCCESS.json"}}
    write_json(output / "SUCCESS.json", dict(status="complete", protocol=PROTOCOL,
        duplicate_pairs=sum(task["num_deals"] for task in tasks), tasks=len(tasks),
        summary_sha256=digest(summary), identity_sha256=digest(identity), files=files))
    write_json(output / "STATUS.json", dict(status="complete", stage=args.stage))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "main", "lbr"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sd-repo", type=Path, required=True)
    parser.add_argument("--ucv-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-hours", type=float, default=22)
    parser.add_argument("--source-uri", action="append", default=[])
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16 or not 0 < args.max_hours <= 36:
        parser.error("Use 1--16 workers and a positive safety cap up to 36 hours")
    setup_repositories(args.sd_repo, args.ucv_repo)
    import torch
    torch.set_num_threads(1)
    records = validate_sources(args.source_root, args.sd_repo, args.ucv_repo)
    implementation = implementation_digest(args.sd_repo, args.ucv_repo)
    environment = dict(python=platform.python_version(),
                       dependencies={name: version(name) for name in ("torch", "numpy", "open_spiel", "scipy", "matplotlib")})
    identity = dict(protocol=PROTOCOL, implementation_sha256=implementation,
                    sources=portable(records), source_uris=sorted(args.source_uri), environment=environment,
                    direct_pairs_per_cross_seed_cell=DIRECT_PAIRS, rule_pairs_per_policy_agent=RULE_PAIRS,
                    lbr_pairs_per_policy=LBR_PAIRS, lbr_preflop_rollouts=LBR_ROLLOUTS,
                    seed_layout="split", comparisons=COMPARISONS,
                    inference=dict(bootstrap_draws=10_000, family_confidence=FAMILY_CONFIDENCE),
                    policy_representations=dict(sd_direct="full_uniform_historical_trajectory_mixture",
                        sd_lbr="exact_own_reach_behavioural_mixture",
                        ucv="saved_deployed_average_policy_network",
                        vr="saved_deployed_average_policy_network"))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "evaluation_manifest.json"
    contract = dict(identity=identity, workers=args.workers)
    if manifest.exists() and read_json(manifest) != contract:
        raise ValueError("Output already belongs to a different evaluation contract")
    write_json(manifest, contract)
    write_csv(args.output / "checkpoint_index.csv", portable(records))
    run_stage(args, records, implementation, identity)


if __name__ == "__main__":
    main()
