"""Repository ownership, pinned scoring equivalence and offline launcher checks."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
UCV = ROOT.parents[1]/"fhp_ucv_escher/fhp-ucv-escher-experiments"
PACKAGE = "exp9_vr_deep_representation_48h"
CONTRACT = ROOT/"experiments/fhp"/PACKAGE/"config.json"
CONFIG = json.loads(CONTRACT.read_text())


def load(path):
    spec=importlib.util.spec_from_file_location("owner_"+path.stem,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path):
    batch=load(ROOT/"gcp/exp9_representation_batch.py")
    shared=tmp_path/"shared_builder.py"
    shared.write_bytes(subprocess.check_output(["git","-C",str(UCV),"show",
        CONFIG["shared_evaluator_ref"]+":gcp/representation_head_to_head_batch.py"]))
    args=batch.parser().parse_args(["--algorithm",CONFIG["algorithm"],"--contract",str(CONTRACT),
        "--shared-builder",str(shared),"--repo-ref","a"*40,"--run-id","owner9-test", "--bucket","gs://test-output",
        "--service-account","runner@test.iam.gserviceaccount.com","--output",str(tmp_path/"job.json")])
    return batch,args


@pytest.mark.parametrize("smoke,resume",[(False,False),(True,False),(False,True)])
def test_owner_commit_and_shared_commit_are_separate(prepared,smoke,resume):
    batch,args=prepared;args.smoke_only=smoke;args.resume=resume
    job=batch.build_job(args)
    assert job["labels"]["owner_experiment"]=="9"
    assert job["labels"]["owner_algorithm"]==CONFIG["algorithm"]
    assert job["labels"]["workload"]==f"fhp-exp{CONFIG['legacy_evaluation_id']}"
    spec=job["taskGroups"][0]["taskSpec"]
    assert spec["maxRunDuration"]=="86400s"
    for runnable in spec["runnables"]:
        subprocess.run(["bash","-n"],input=runnable["script"]["text"],text=True,check=True)
    script=spec["runnables"][0]["script"]["text"]
    assert f'git -C "$REPOSITORY" checkout --detach {CONFIG["shared_evaluator_ref"]}' in script
    assert f'git -C "$OWNER_REPOSITORY" checkout --detach {args.repo_ref}' in script
    assert f'python "$OWNER_REPOSITORY/experiments/fhp/{PACKAGE}/run.py" --ucv-repo "$REPOSITORY"' in script
    assert "experiment_contract.json" in script and "experiment_repository_commit.txt" in script
    assert script.count("--algorithm "+CONFIG["algorithm"])==1  # metadata validator only
    assert ("--smoke-only" in script)==smoke
    assert "training_states/" not in script
    assert all(pin["run_id"] in script for pin in CONFIG["source_pins"].values())


def test_same_source_uris_and_full_policy_downloads(prepared):
    batch,args=prepared
    _,shared=batch.shared(args)
    assert batch.source_uris(args)==shared.source_uris(args)
    assert dict(batch.metadata_uris(args))==dict(shared.metadata_uris(args))
    script=batch.build_job(args)["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert shared.policy_download_script(args) in script
    pinned=json.loads(subprocess.check_output(["git","-C",str(UCV),"show",
        CONFIG["shared_evaluator_ref"]+":experiments/fhp/representation_head_to_head/source_pins.json"]))
    assert CONFIG["source_pins"]==pinned[CONFIG["algorithm"]]


@pytest.mark.parametrize("mutation",["algorithm","budget","source","builder"])
def test_reject_drift(prepared,tmp_path,mutation):
    batch,args=prepared
    config=json.loads(CONTRACT.read_text())
    if mutation=="algorithm":args.algorithm="vr" if args.algorithm=="sd" else "sd"
    elif mutation=="budget":config["pairs_per_cell"]=100
    elif mutation=="source":config["source_pins"]["rich"]["run_id"]="different"
    elif mutation=="builder":args.shared_builder.write_text(args.shared_builder.read_text()+"\n")
    args.contract=tmp_path/"contract.json";args.contract.write_text(json.dumps(config))
    with pytest.raises(ValueError):batch.build_job(args)


def test_local_wrapper_preserves_fixed_algorithm_and_package_isolation():
    wrapper=load(ROOT/"experiments/fhp"/PACKAGE/"run.py")
    command=wrapper.command(UCV,["--source-root","input","--output","output","--alg","wrong"])
    assert command[-2:]==["--algorithm",CONFIG["algorithm"]]
    with pytest.raises(ValueError):wrapper.command(UCV,["--algorithm=ucv"])
    result=subprocess.run([sys.executable,str(ROOT/"experiments/fhp"/PACKAGE/"run.py"),
        "--ucv-repo",str(UCV),"--help"],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert "--source-root" in result.stdout


def test_local_wrapper_rejects_modified_shared_scoring(tmp_path):
    wrapper=load(ROOT/"experiments/fhp"/PACKAGE/"run.py")
    # A missing tree cannot silently fall back to another installed package.
    with pytest.raises((ValueError,subprocess.CalledProcessError,FileNotFoundError)):
        wrapper.command(tmp_path,[])


def test_launcher_dry_run_from_owner_commit(tmp_path):
    checkout=tmp_path/"owner";checkout.mkdir()
    for relative in (f"experiments/fhp/{PACKAGE}/config.json",f"experiments/fhp/{PACKAGE}/run.py",
                     "gcp/exp9_representation_batch.py","gcp/run_exp9_vr_deep_representation_48h.sh"):
        destination=checkout/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,destination)
    subprocess.run(["git","init","-q",str(checkout)],check=True)
    subprocess.run(["git","-C",str(checkout),"add","."],check=True)
    subprocess.run(["git","-C",str(checkout),"-c","user.name=Test","-c","user.email=test@example.invalid",
        "-c","commit.gpgsign=false","commit","-qm","fixture"],check=True)
    ref=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    env=dict(os.environ,PROJECT_ID="test-project",REGION="europe-west1",BUCKET="gs://test-output",
        SA_EMAIL="runner@test.iam.gserviceaccount.com",RUN_ID="owner9-test",REPO_REF=ref,
        EVALUATION_REPO=str(UCV),EXP9_SMOKE_ONLY="0")
    # Any cloud invocation during dry-run is an error, even if credentials exist.
    bin_dir=tmp_path/"bin";bin_dir.mkdir()
    cloud=bin_dir/"gcloud";cloud.write_text("#!/bin/sh\nexit 99\n");cloud.chmod(0o755)
    env["PATH"]=str(bin_dir)+os.pathsep+env["PATH"]
    result=subprocess.run(["bash",str(checkout/"gcp/run_exp9_vr_deep_representation_48h.sh"),"dry-run"],
        env=env,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert "no cloud job submitted" in result.stdout
    path=Path(result.stdout.split("Prepared ",1)[1].split(";",1)[0])
    job=json.loads(path.read_text())
    script=job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert f'git -C "$OWNER_REPOSITORY" checkout --detach {ref}' in script


def test_download_launcher_syntax():
    subprocess.run(["bash","-n",str(ROOT/"gcp/download_exp9_representation_analysis.sh")],check=True)
