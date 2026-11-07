"""Run this repository's frozen experiment using the shared evaluator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def command(ucv_repo, remaining):
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    if (config["experiment_id"] != 9 or config["training_seeds"] != [0,1,2]
            or config["pairs_per_cell"] != 100000 or config["shard_pairs"] != 5000
            or config["family_confidence"] != .975
            or config["base_seed"] != {"sd":642048,"vr":643048}[config["algorithm"]]):
        raise ValueError("Experiment contract differs from the frozen protocol")
    ucv_repo = Path(ucv_repo).resolve()
    if "--algorithm" in remaining or any(v.startswith("--algorithm=") for v in remaining):
        raise ValueError("The experiment fixes its algorithm")
    # Check the actual shared source tree against the pinned revision; HEAD may
    # advance for unrelated documentation/experiment changes without affecting it.
    for directory in ("experiments/fhp/representation_head_to_head",
                      "experiments/fhp/exp41_fhp_matched_representation_48h",
                      "experiments/fhp/exp40_fhp_threeway_48h",
                      "experiments/fhp/exp38_fhp_strength_critic2500_48h", "fhp_escher"):
        files = subprocess.check_output(["git", "-C", str(ucv_repo), "ls-tree", "-r", "--name-only",
            config["shared_evaluator_ref"], "--", directory], text=True).splitlines()
        if not files:
            raise ValueError("Missing pinned shared evaluator")
        for name in files:
            if Path(name).suffix not in (".py", ".json"):
                continue
            expected = subprocess.check_output(["git", "-C", str(ucv_repo), "show",
                config["shared_evaluator_ref"]+":"+name])
            if (ucv_repo/name).read_bytes() != expected:
                raise ValueError(f"Shared evaluator differs from the pinned revision: {name}")
    pins = json.loads((ucv_repo/"experiments/fhp/representation_head_to_head/source_pins.json").read_text())
    if pins[config["algorithm"]] != config["source_pins"]:
        raise ValueError("Experiment sources differ from pinned evaluator")
    return [sys.executable, "-m", "experiments.fhp.representation_head_to_head.run",
            *remaining, "--algorithm", config["algorithm"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--ucv-repo", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    subprocess.run(command(args.ucv_repo, remaining), cwd=args.ucv_repo, check=True)


if __name__ == "__main__":
    main()
