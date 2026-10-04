#!/usr/bin/env python3
"""Source-bundled, single-VM VR evaluation; no local training or credentials bundled."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = "experiments/fhp/retrospective_exp1_exp2_exp3_evaluation"
SOURCES = json.loads((ROOT / MODULE_DIR / "sources.json").read_text())


def cloud(*command):
    try:
        return subprocess.run(["gcloud", *command], check=True, text=True,
                              capture_output=True, timeout=300).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr or error.stdout or "gcloud command failed") from error


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def bundle(suite, destination):
    files = {}
    for prefix, root, packages in (("native", ROOT, ("fhp_vr_deep", "vr_deep_cfr", "experiments")),
                                    ("evaluator", suite, ("fhp_evaluation",))):
        for package in packages:
            base = root / package
            if base.is_symlink() or not (base / "__init__.py").is_file():
                raise ValueError(f"Missing source package {base}")
            for path in sorted(base.rglob("*.py")):
                if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
                    raise ValueError("Symlinked source refused")
                files[f"{prefix}/{path.relative_to(root)}"] = path.read_bytes()
    for name in (f"{MODULE_DIR}/sources.json", "gcp/vr_eval123_worker.sh",
                 "gcp/retrospective_exp1_exp2_exp3_evaluation_batch.py", "requirements.txt"):
        files[f"native/{name}"] = (ROOT / name).read_bytes()
    if len(files) > 2048 or sum(map(len, files.values())) > 16 * 1024**2:
        raise ValueError("Unexpectedly large source bundle")
    provenance = dict(native_commit=git(ROOT, "rev-parse", "HEAD"),
                      native_dirty=bool(git(ROOT, "status", "--porcelain")),
                      suite_commit=git(suite, "rev-parse", "HEAD"),
                      suite_dirty=bool(git(suite, "status", "--porcelain")),
                      files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    files["source_manifest.json"] = json.dumps(provenance, sort_keys=True).encode()
    with destination.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(destination.read_bytes()).hexdigest(), provenance


def validate(args):
    for value, regex, label in ((args.run_id, r"vr-eval123-[a-z0-9][a-z0-9-]{0,27}", "RUN_ID"),
                               (args.project, r"[a-z][a-z0-9-]{4,61}[a-z0-9]", "PROJECT_ID"),
                               (args.region, r"[a-z]+-[a-z]+[0-9]+", "REGION"),
                               (args.service_account, r"[a-z0-9-]+@(?:[a-z0-9-]+\.iam|developer)\.gserviceaccount\.com", "SA_EMAIL")):
        if not re.fullmatch(regex, value or ""):
            raise ValueError(f"Invalid or missing {label}")
    args.bucket = "gs://" + (args.bucket or "").removeprefix("gs://").rstrip("/")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", args.bucket):
        raise ValueError("BUCKET must be a bare bucket or gs://bucket without a prefix")
    if not (args.suite / "fhp_evaluation/cohort.py").is_file():
        raise ValueError("Set EVALUATION_SUITE_ROOT to the updated fhp-evaluation-suite checkout")


def build_job(args, checksum, *, resume=False, smoke=False):
    destination = f"{args.bucket}/{args.run_id}"
    env = dict(VR_EVAL_DESTINATION=destination, VR_EVAL_BUCKET=args.bucket,
               VR_EVAL_BUNDLE_SHA256=checksum, VR_EVAL_RESUME=str(int(resume)),
               VR_EVAL_SMOKE=str(int(smoke)), PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               MPLCONFIGDIR="/tmp/vr-eval-mpl")
    finalize = """#!/usr/bin/env bash
set -Eeuo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
WORK=/workspace/vr-eval123
mkdir -p "$WORK/output"
dmesg -T > "$WORK/output/kernel.log" 2>&1 || true
df -h > "$WORK/output/disk_final.txt" || true
free -m > "$WORK/output/memory_final.txt" || true
exec 9>"$WORK/upload.lock"
flock -w 180 9
timeout 360 gcloud storage rsync --recursive \
  --exclude='.*[.]tmp$|(^|/)SUCCESS[.]json$' "$WORK/output" "$VR_EVAL_DESTINATION/analysis"
for marker in "$WORK/output"/{smoke,main,lbr}/SUCCESS.json; do
  [[ -f "$marker" ]] || continue
  timeout 30 gcloud storage cp "$marker" "$VR_EVAL_DESTINATION/analysis/${marker#"$WORK/output/"}"
done
"""
    cap = 2 * 3600 if smoke else 36 * 3600
    return dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1,
        taskSpec=dict(computeResource=dict(cpuMilli=16000, memoryMib=62000), maxRetryCount=0,
                      maxRunDuration=f"{cap}s", environment=dict(variables=env), runnables=[
                          dict(script=dict(text=(ROOT / "gcp/vr_eval123_worker.sh").read_text()), timeout=f"{cap-1200}s"),
                          dict(alwaysRun=True, timeout="900s", script=dict(text=finalize))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account), instances=[dict(policy=dict(
            machineType="n2-standard-16", provisioningModel="STANDARD", bootDisk=dict(sizeGb=80, type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="vr-eval123", run_id=args.run_id, stage="smoke" if smoke else "evaluation"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "run", "smoke-only", "resume", "status"))
    for name, env in (("project", "PROJECT_ID"), ("region", "REGION"), ("bucket", "BUCKET"),
                      ("service-account", "SA_EMAIL"), ("run-id", "RUN_ID")):
        parser.add_argument("--" + name, default=os.environ.get(env))
    parser.add_argument("--suite", type=Path, default=Path(os.environ.get("EVALUATION_SUITE_ROOT", ROOT.parents[1] / "fhp-evaluation-suite")))
    args = parser.parse_args(argv)
    validate(args)
    query = ["batch", "jobs", "list", "--project", args.project, "--location", args.region,
             "--filter", f"labels.run_id={args.run_id}", "--format=json"]
    if args.action == "status":
        print(cloud(*query))
        return
    output = ROOT / "outputs/batch" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="prepare-", dir=output))
    checksum, provenance = bundle(args.suite.resolve(), attempt / "source.tar.gz")
    destination = f"{args.bucket}/{args.run_id}"
    smoke = args.action == "smoke-only"
    request = dict(bundle_sha256=checksum, sources=SOURCES, smoke=smoke, provenance=provenance)
    if args.action != "dry-run":
        cloud("iam", "service-accounts", "describe", args.service_account, "--project", args.project)
        jobs = json.loads(cloud(*query))
        if any(job.get("status", {}).get("state") not in ("SUCCEEDED", "FAILED") for job in jobs):
            raise ValueError("An active job already owns this output namespace")
        if args.action == "resume":
            saved = json.loads(cloud("storage", "cat", f"{destination}/inputs/request.json"))
            if saved["bundle_sha256"] != checksum or saved["smoke"]:
                raise ValueError("Resume requires unchanged source bundle and a full-run namespace")
        else:
            objects = json.loads(cloud("storage", "objects", "list", f"{destination}/**",
                                       "--exhaustive", "--limit=1", "--format=json"))
            if not isinstance(objects, list) or objects or jobs:
                raise ValueError("RUN_ID is occupied; use a fresh ID or resume")
            for source in SOURCES.values():
                cloud("storage", "objects", "describe", f"{args.bucket}/{source['run_id']}/analysis/SUCCESS.json")
    job = build_job(args, checksum, resume=args.action == "resume", smoke=smoke)
    for name, value in (("job.json", job), ("request.json", request)):
        (attempt / name).write_text(json.dumps(value, indent=2) + "\n")
    print(f"Prepared {attempt / 'job.json'}")
    if args.action == "dry-run":
        print("No cloud calls or jobs submitted.")
        return
    if args.action != "resume":
        for name in ("source.tar.gz", "request.json", "job.json"):
            cloud("storage", "cp", "--if-generation-match=0", str(attempt / name), f"{destination}/inputs/{name}")
    # Concurrent resume clients observing the same job set submit the SAME name;
    # Batch's atomic create prevents two VMs writing the same task cache.
    prior_jobs = sorted(job["name"] for job in jobs)
    suffix = "-retry-" + hashlib.sha256(json.dumps(prior_jobs).encode()).hexdigest()[:8] if args.action == "resume" else ""
    name = args.run_id + suffix
    print(cloud("batch", "jobs", "submit", name, "--project", args.project,
                "--location", args.region, "--config", str(attempt / "job.json")))
    print(f"Submitted {name}; laptop may disconnect. Results: {destination}/analysis/")


if __name__ == "__main__":
    main()
