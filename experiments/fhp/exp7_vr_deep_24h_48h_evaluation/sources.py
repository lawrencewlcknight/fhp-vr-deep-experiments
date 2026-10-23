"""Hash-pinned playable policies. Never download replay or restart archives."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "sources.json").read_text())


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(root):
        raise ValueError("Source path escapes its input directory")
    return path


def records(root, component):
    if component not in ("matches", "strategic"):
        raise ValueError("Unknown component")
    result = []
    for row in PINS["records"]:
        if component == "matches" and row["family"] != "vr_deep":
            continue
        path = contained(root, row["source_run"] + "/" + row["relative"])
        if sha256(path) != row["sha256"] or ("size_bytes" in row and path.stat().st_size != row["size_bytes"]):
            raise ValueError(f"Changed policy: {row['id']}")
        result.append(dict(row, path=str(path)))
    expected = 6 if component == "matches" else 12
    if len(result) != expected or len({r["id"] for r in result}) != expected:
        raise ValueError("Incomplete or duplicate frozen cohort")
    return result


def download(root, vr_bucket, ucv_bucket, component):
    def fetch(uri, path, expected=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        if expected and path.exists() and sha256(path) == expected:
            return
        temp = path.with_suffix(path.suffix + ".download")
        subprocess.run(["gcloud", "storage", "cp", uri, str(temp)], check=True, timeout=300)
        if expected and sha256(temp) != expected:
            raise ValueError(f"Downloaded source checksum mismatch: {uri}")
        temp.replace(path)
    downloaded = set()
    for row in PINS["records"]:
        if component == "matches" and row["family"] != "vr_deep":
            continue
        bucket = vr_bucket if row["family"] == "vr_deep" else ucv_bucket
        relative = row["source_run"] + "/" + row["relative"]
        fetch(bucket.rstrip("/") + "/" + relative, contained(root, relative), row["sha256"])
        if row["cohort"] == "anchor":
            continue
        worker = str(Path(relative).parent.parent)
        for name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json"):
            key = worker + "/" + name
            if key not in downloaded:
                fetch(bucket.rstrip("/") + "/" + key, contained(root, key), row.get("metadata_sha256", {}).get(name))
                downloaded.add(key)
    return records(root, component)


def validate_native(rows, family):
    """Invoked in isolated processes so the two native experiments packages cannot collide."""
    if family == "vr_deep":
        from fhp_vr_deep.evaluation_adapter import load_policy_for_evaluation
        from experiments.fhp.exp6_vr_deep_ray8_48h.config import contract
        expected = contract(False)
    elif family == "ucv":
        from fhp_escher.checkpointing import LoadedFHPPolicy
        from fhp_escher.game import load_fhp_game
    else:
        raise ValueError("Only VR and UCV are allowed; no SD dependency")
    validated = []
    for row in rows:
        if row["family"] != family:
            continue
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Checkpoint changed before deserialisation")
        if family == "vr_deep":
            _, policy = load_policy_for_evaluation(path)
            p = policy.metadata
            if (p["seed"] != row["seed"] or p["nodes_touched"] != row["nodes"]
                    or p["iteration"] != row["iteration"]
                    or p["algorithm_id"] != expected["algorithm_id"]
                    or p["authors_parameterisation"] != expected["training_config"]
                    or p.get("feature_encoder") != expected.get("feature_encoder")
                    or p["checkpoint"]["checkpoint_id"] != f"time_{row['hours']:02d}h"
                    or p["checkpoint"]["training_elapsed_seconds"] != row["active_seconds"]):
                raise ValueError(f"Wrong VR payload: {row['id']}")
        else:
            p = LoadedFHPPolicy(load_fhp_game(), path).checkpoint
            if p["seed"] != row["seed"] or p.get("checkpoint_target_seconds") != row["hours"] * 3600:
                raise ValueError(f"Wrong UCV payload: {row['id']}")
            if row["cohort"] != "anchor" and (p["nodes_touched"] != row["nodes"]
                    or p["outer_iteration"] != row["iteration"]
                    or p["algorithm_id"] != "strategic_ae_critic2500_ucv_escher"):
                raise ValueError("Wrong final UCV configuration")
        if row["cohort"] != "anchor":
            worker = path.parent.parent
            manifest = read(worker / "run_manifest.json")
            if manifest["seed"] != row["seed"] or manifest["repository_commit"] != row["source_commit"]:
                raise ValueError("Wrong native training provenance")
            if row["active_seconds"] < row["hours"] * 3600:
                raise ValueError("Checkpoint predates its training horizon")
            for name, expected_hash in row.get("metadata_sha256", {}).items():
                if name in ("run_manifest.json", "checkpoint_manifest.json", "SUCCESS.json") and sha256(worker / name) != expected_hash:
                    raise ValueError("Changed pinned UCV metadata")
        validated.append(row["id"])
    if not validated:
        raise ValueError("No native policies validated")
    return dict(status="succeeded", family=family, policies=validated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "validate-native"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component", choices=("matches", "strategic"), required=True)
    parser.add_argument("--vr-bucket")
    parser.add_argument("--ucv-bucket")
    parser.add_argument("--family", choices=("vr_deep", "ucv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = download(args.root, args.vr_bucket, args.ucv_bucket, args.component) if args.action == "download" else records(args.root, args.component)
    write(args.output, rows if args.action == "download" else validate_native(rows, args.family))


if __name__ == "__main__":
    main()
