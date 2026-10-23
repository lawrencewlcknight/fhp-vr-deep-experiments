#!/usr/bin/env python3
"""One Batch submission, two parallel CPU tasks; immutable-source recovery."""
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
MODULE_DIR = "experiments/fhp/exp7_vr_deep_24h_48h_evaluation"
PINS = json.loads((ROOT/MODULE_DIR/"sources.json").read_text())
PROTOCOL = "vr_exp7_cloud_v1"


def cloud(*args):
    result = subprocess.run(["gcloud",*args],text=True,capture_output=True,timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def git(root,*args):
    return subprocess.run(["git","-C",str(root),*args],check=True,text=True,
                          capture_output=True).stdout.strip()


def bundle(suite, ucv, destination):
    files, provenance = {}, {}
    for prefix,root,packages in (
        ("native",ROOT,("fhp_vr_deep","vr_deep_cfr","experiments")),
        ("evaluator",suite,("fhp_evaluation",)),
        ("ucv",ucv,("fhp_escher","vr_deep_cfr","unbiased_escher","adaptive_escher","experiments"))):
        provenance[prefix] = dict(commit=git(root,"rev-parse","HEAD"))
        for package in packages:
            base = root/package
            if not (base/"__init__.py").is_file():
                raise ValueError(f"Missing source package {base}")
            for path in sorted(base.rglob("*.py")):
                if any(p.is_symlink() for p in (path,*path.parents) if p != root):
                    raise ValueError("Symlinked source refused")
                files[f"{prefix}/{path.relative_to(root)}"] = path.read_bytes()
    for name in (f"{MODULE_DIR}/sources.json",f"{MODULE_DIR}/position_library.json",
                 "requirements.txt","gcp/vr_exp7_worker.sh","gcp/vr_exp7_sync.sh"):
        files[f"native/{name}"] = (ROOT/name).read_bytes()
    if len(files)>4096 or sum(map(len,files.values()))>32*1024**2:
        raise ValueError("Unexpected source bundle size")
    provenance["files"] = {k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    files["source_manifest.json"] = json.dumps(provenance,sort_keys=True).encode()
    with destination.open("xb") as raw, gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode="w") as archive:
            for name,data in sorted(files.items()):
                entry=tarfile.TarInfo(name)
                entry.size,entry.mtime,entry.mode=len(data),0,0o644
                archive.addfile(entry,io.BytesIO(data))
    return hashlib.sha256(destination.read_bytes()).hexdigest(),provenance


def validate(args):
    for value,regex,label in (
        (args.run_id,r"vr7-eval2448-[a-z0-9][a-z0-9-]{0,30}","RUN_ID"),
        (args.project,r"[a-z][a-z0-9-]{4,61}[a-z0-9]","PROJECT_ID"),
        (args.region,r"[a-z]+-[a-z]+[0-9]+","REGION"),
        (args.service_account,r"[a-z0-9-]+@(?:[a-z0-9-]+\.iam|developer)\.gserviceaccount\.com","SA_EMAIL")):
        if not re.fullmatch(regex,value or ""):
            raise ValueError(f"Invalid or missing {label}")
    for name in ("bucket","vr_bucket","ucv_bucket"):
        value="gs://"+(getattr(args,name) or "").removeprefix("gs://").rstrip("/")
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]",value):
            raise ValueError(f"{name} must identify a bucket without a prefix")
        setattr(args,name,value)


def build_job(args,checksum,smoke=False):
    cap = 4*3600 if smoke else 48*3600
    env = dict(VR7_DEST=f"{args.bucket}/{args.run_id}",VR7_VR_BUCKET=args.vr_bucket,
        VR7_UCV_BUCKET=args.ucv_bucket,VR7_SHA256=checksum,VR7_RESUME="0",
        VR7_SMOKE=str(int(smoke)),PYTHONUNBUFFERED="1",PYTHONFAULTHANDLER="1",
        PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",
        MPLCONFIGDIR="/tmp/vr7-mpl")
    finalize = """#!/usr/bin/env bash
set -Eeuo pipefail
case "${BATCH_TASK_INDEX:?}" in 0) COMPONENT=matches;; 1) COMPONENT=strategic;; *) exit 2;; esac
WORK=/workspace/vr-exp7
OUT="$WORK/output"
mkdir -p "$OUT"
dmesg -T > "$OUT/kernel.log" 2>&1 || true
df -h > "$OUT/disk_final.txt" || true
free -m > "$OUT/memory_final.txt" || true
export WORK OUT COMPONENT
if [[ -f "$WORK/source/native/gcp/vr_exp7_sync.sh" ]]; then
  bash "$WORK/source/native/gcp/vr_exp7_sync.sh"
else
  timeout 600 gcloud storage rsync --recursive "$OUT" "$VR7_DEST/analysis/$COMPONENT"
fi
"""
    return dict(taskGroups=[dict(taskCount=2,parallelism=2,taskCountPerNode=1,
        taskSpec=dict(computeResource=dict(cpuMilli=16000,memoryMib=62000),
            maxRetryCount=0,maxRunDuration=f"{cap}s",environment=dict(variables=env),runnables=[
                dict(script=dict(text=(ROOT/"gcp/vr_exp7_worker.sh").read_text()),timeout=f"{cap-1200}s"),
                dict(alwaysRun=True,timeout="900s",script=dict(text=finalize))]))],
        allocationPolicy=dict(serviceAccount=dict(email=args.service_account),
            instances=[dict(policy=dict(machineType="n2-standard-16",provisioningModel="STANDARD",
                                       bootDisk=dict(sizeGb=100,type="pd-balanced")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=dict(workload="vr-exp7",run_id=args.run_id,stage="smoke" if smoke else "evaluation"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("dry-run","run","smoke-only","resume","status"))
    for name,env in (("project","PROJECT_ID"),("region","REGION"),("bucket","BUCKET"),
                     ("service-account","SA_EMAIL"),("run-id","RUN_ID")):
        parser.add_argument("--"+name,default=os.environ.get(env))
    parser.add_argument("--vr-bucket",default=os.environ.get("VR_BUCKET","gs://clever-overview-399515-fhp-vr-deep-results"))
    parser.add_argument("--ucv-bucket",default=os.environ.get("UCV_BUCKET","gs://clever-overview-399515-fhp-escher-results"))
    parser.add_argument("--suite",type=Path,default=Path(os.environ.get("EVALUATION_SUITE_ROOT",ROOT.parents[1]/"fhp-evaluation-suite")))
    parser.add_argument("--ucv",type=Path,default=Path(os.environ.get("UCV_REPO_ROOT",ROOT.parents[1]/"fhp_ucv_escher/fhp-ucv-escher-experiments")))
    args=parser.parse_args(argv)
    validate(args)
    query=["batch","jobs","list","--project",args.project,"--location",args.region,
           "--filter",f"labels.run_id={args.run_id}","--format=json"]
    if args.action=="status":
        print(cloud(*query));return
    destination=f"{args.bucket}/{args.run_id}"
    output=ROOT/"outputs/batch"/args.run_id
    output.mkdir(parents=True,exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix="prepare-",dir=output))
    jobs=[]
    if args.action!="dry-run":
        cloud("iam","service-accounts","describe",args.service_account,"--project",args.project)
        jobs=json.loads(cloud(*query))
        if any(j.get("status",{}).get("state") not in ("SUCCEEDED","FAILED") for j in jobs):
            raise ValueError("An active job already owns this output namespace")
    if args.action=="resume":
        # Resume from the originally uploaded code, even after local commits/edits.
        request=json.loads(cloud("storage","cat",f"{destination}/inputs/request.json"))
        if request["protocol"]!=PROTOCOL or request["smoke"]:
            raise ValueError("Resume requires an Experiment 7 full-run namespace")
        job=json.loads(cloud("storage","cat",f"{destination}/inputs/job.json"))
        if hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()!=request["job_sha256"]:
            raise ValueError("Original job configuration changed")
        spec=job["taskGroups"][0]["taskSpec"]
        spec["environment"]["variables"]["VR7_RESUME"]="1"
        # Account/region/project must be the original execution context.
        if request["execution"]!=dict(project=args.project,region=args.region,service_account=args.service_account):
            raise ValueError("Resume in the original project/region/service account")
    else:
        checksum,provenance=bundle(args.suite.resolve(),args.ucv.resolve(),attempt/"source.tar.gz")
        smoke=args.action=="smoke-only"
        job=build_job(args,checksum,smoke)
        request=dict(protocol=PROTOCOL,bundle_sha256=checksum,smoke=smoke,sources=PINS,
            provenance=provenance,job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
            execution=dict(project=args.project,region=args.region,service_account=args.service_account))
        if args.action!="dry-run":
            objects=json.loads(cloud("storage","objects","list",f"{destination}/**","--exhaustive","--limit=1","--format=json"))
            if not isinstance(objects,list) or objects or jobs:
                raise ValueError("Occupied RUN_ID: use a new ID or resume")
            # Small checkpoint existence checks, before any paid VM starts.
            for r in PINS["records"]:
                bucket=args.vr_bucket if r["family"]=="vr_deep" else args.ucv_bucket
                cloud("storage","objects","describe",f"{bucket}/{r['source_run']}/{r['relative']}")
    for name,value in (("job.json",job),("request.json",request)):
        (attempt/name).write_text(json.dumps(value,indent=2)+"\n")
    print(f"Prepared {attempt/'job.json'}",flush=True)
    if args.action=="dry-run":
        print("No cloud calls or jobs submitted.");return
    if args.action!="resume":
        for name in ("source.tar.gz","request.json","job.json"):
            cloud("storage","cp","--if-generation-match=0",str(attempt/name),f"{destination}/inputs/{name}")
    suffix="-r"+hashlib.sha256(json.dumps(sorted(j["name"] for j in jobs)).encode()).hexdigest()[:8] if args.action=="resume" else ""
    name=args.run_id+suffix
    print(cloud("batch","jobs","submit",name,"--project",args.project,"--location",args.region,"--config",str(attempt/"job.json")))
    print(f"Submitted {name}. Two tasks: matches and strategic. Results: {destination}/analysis/")


if __name__=="__main__":
    main()
