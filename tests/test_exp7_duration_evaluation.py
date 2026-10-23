import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.fhp.exp7_vr_deep_24h_48h_evaluation import matches, sources, strategic
from fhp_evaluation.statistics import sample_summary
from fhp_evaluation.strategic_audit.library import canonical_board, validate_library
from fhp_evaluation.cohort import digest, execute_tasks, portable, write_json

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
SUITE = WORKSPACE/"fhp-evaluation-suite"
UCV = WORKSPACE/"fhp_ucv_escher/fhp-ucv-escher-experiments"
spec = importlib.util.spec_from_file_location("vr7_batch",ROOT/"gcp/exp7_duration_evaluation_batch.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)
VR = [r for r in sources.PINS["records"] if r["cohort"]=="vr"]


def test_frozen_task_count_and_scope():
    tasks = matches.build_tasks(VR)
    assert len(tasks)==750
    assert sum(t["deals"] for t in tasks)==756000
    assert {t["a"]["family"] for t in tasks}=={"vr_deep"}
    assert len({t["id"] for t in tasks})==len(tasks)
    assert sum(t["kind"]=="lbr" for t in tasks)==600
    assert {t["rollouts"] for t in tasks if t["kind"]=="lbr"}=={4096}
    assert {t["a"]["hours"] for t in tasks if t["kind"]=="direct"}=={48}
    assert {t["b"]["hours"] for t in tasks if t["kind"]=="direct"}=={24}
    with pytest.raises(ValueError):
        matches.build_tasks(VR[:-1])


def test_paired_rule_and_independent_smoke_streams():
    tasks = matches.build_tasks(VR)
    for seed in range(3):
        streams = [{(t["opponent"],t["shard"]):t["evaluation_seed"] for t in tasks
                    if t["kind"]=="rule" and t["a"]["seed"]==seed and t["a"]["hours"]==h}
                   for h in (24,48)]
        assert streams[0]==streams[1]
    smoke = matches.build_tasks(VR,smoke=True)
    assert len(smoke)==45 and sum(t["deals"] for t in smoke)==90
    assert {t["evaluation_seed"] for t in smoke}.isdisjoint(t["evaluation_seed"] for t in tasks)


def test_exact_shard_pooling():
    data = np.random.default_rng(8).normal(size=133)
    pooled = matches.pool_summaries([sample_summary(d) for d in np.array_split(data,9)])
    exact = sample_summary(data)
    for key in ("n","mean","std","se"):
        assert pooled[key]==pytest.approx(exact[key])


def test_seed_not_match_or_shard_inference():
    results=[]
    for task in matches.build_tasks(VR,smoke=True):
        value=task["a"]["seed"]+task["a"]["hours"]/24
        results.append(dict(task=portable(task),summary=sample_summary([value-1,value+1]),elapsed_seconds=1))
    summary=matches.summarise(results,smoke=True)
    for kind in ("rule","lbr"):
        delta=summary["paired_48_minus_24_changes"][kind]
        assert delta["n"]==3
        assert delta["paired_seed_changes"]==[1,1,1]
    assert summary["strength_evidence"] is False
    cells=[dict(seed=i,opponent_seed=j,mean=100 if i==j else 0,se=1) for i in range(3) for j in range(3)]
    result=matches.paired_bootstrap(cells,draws=1000)
    assert result["same_trajectory"]["mean_mbb_per_hand"]==100
    assert result["same_trajectory"]["trajectory_bootstrap_ci95"]==pytest.approx([100,100])
    assert result["all_cross_seed"]["mean_mbb_per_hand"]==pytest.approx(100/3)
    assert result["all_cross_seed"]["conditional_deal_se"]==pytest.approx(1/3)


def test_new_assessment_excludes_historical_board_classes():
    lib=strategic.library()
    validate_library(lib)
    assert lib["partition_counts"]==dict(development=64,assessment=32)
    old={canonical_board(b) for b in lib["historical_exclusions"]}
    new={canonical_board(r["board"]) for r in lib["boards"] if r["partition"]=="assessment"}
    assert len(new)==32 and new.isdisjoint(old)
    assert not lib["reference_deals"]
    assert not any(r["texture"]=="trips_3suit" for r in lib["boards"] if r["partition"]=="assessment")
    smoke=strategic.library(True)
    validate_library(smoke)
    assert len(smoke["boards"])==2
    assert {r["partition"] for r in smoke["boards"]}=={"development"}


@pytest.mark.parametrize("smoke",[False,True])
def test_strategic_limits_valid(smoke):
    from fhp_evaluation.strategic_audit.board_export import validate_config
    resources=strategic.export_resources(smoke)
    assert validate_config(resources,4,52000)==resources


def test_resume_uses_original_bundle_without_rebuilding(tmp_path,monkeypatch):
    args=SimpleNamespace(run_id="vr7-eval2448-recovery",project="clever-overview-399515",
        region="europe-west1",bucket="gs://example-results",vr_bucket="gs://example-vr",
        ucv_bucket="gs://example-ucv",service_account="runner@clever-overview-399515.iam.gserviceaccount.com")
    job=batch.build_job(args,"a"*64)
    request=dict(protocol=batch.PROTOCOL,smoke=False,
        execution=dict(project=args.project,region=args.region,service_account=args.service_account),
        job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest())
    calls=[]
    def cloud(*cmd):
        calls.append(cmd)
        if cmd[:3]==("batch","jobs","list"):
            return json.dumps([dict(name="old",status=dict(state="FAILED"))])
        if cmd[:2]==("storage","cat"):
            return json.dumps(request if cmd[2].endswith("request.json") else job)
        if cmd[:3]==("batch","jobs","submit"):
            submitted=json.loads(Path(cmd[-1]).read_text())
            env=submitted["taskGroups"][0]["taskSpec"]["environment"]["variables"]
            assert env["VR7_SHA256"]=="a"*64 and env["VR7_RESUME"]=="1"
        return "{}"
    def no_bundle(*args):
        pytest.fail("Resume must not rebuild current source")
    monkeypatch.setattr(batch,"cloud",cloud)
    monkeypatch.setattr(batch,"bundle",no_bundle)
    monkeypatch.setattr(batch,"ROOT",tmp_path)
    batch.main(["resume","--run-id",args.run_id,"--project",args.project,"--region",args.region,
                "--bucket",args.bucket,"--service-account",args.service_account])
    assert not any(c[:2]==("storage","cp") for c in calls)
    assert sum(c[:3]==("batch","jobs","submit") for c in calls)==1


def fake_strategic():
    rows=[dict(r,path="/tmp/"+r["id"]) for r in sources.PINS["records"]]
    spec=strategic.make_spec(rows,ROOT,UCV,"development")
    result=dict(status="succeeded",provenance=dict(spec=spec,partition="development"),comparisons={})
    for c in spec["candidates"]:
        p=spec["policies"][c]
        value=p["training_seed"]+(10 if p["cohort"]=="ucv" else 0)+(1 if p["hours"]==48 else 2)
        for opp in spec["opponents"]:
            for tilt in spec["range_tilts"]:
                result["comparisons"][f"{c}__vs__{opp}__tilt_{tilt:g}"]=dict(
                    mean_root_response_gap_bb=value,
                    boards=[dict(board=[0,4,8],core=True,root_count=100)])
    return result


def test_strategic_paired_horizons_independent_families():
    result=fake_strategic()
    summary=strategic.summarise(result)
    assert summary["endpoints"]["0"]["vr_24h"]["seed_values"]==[2,3,4]
    assert summary["contrasts"]["0"]["vr_48_minus_24"]["seed_values"]==[-1,-1,-1]
    comparison=summary["contrasts"]["0"]["vr48_minus_ucv48"]
    assert comparison["mean"]==-10
    assert comparison["independent_seed_bootstrap_ci95"][0] < -10
    assert comparison["independent_seed_bootstrap_ci95"][1] > -10
    assert {p["family"] for p in result["provenance"]["spec"]["policies"].values()}=={"vr_deep","ucv","rule"}
    next(iter(result["comparisons"].values()))["boards"][0]["root_count"]=99
    with pytest.raises(ValueError,match="support differs"):
        strategic.summarise(result)


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        sources.contained(tmp_path,"../escape")
    with pytest.raises(ValueError):
        sources.contained(tmp_path,"/escape")


def test_corrupt_source_refused_before_loading(tmp_path,monkeypatch):
    pins=copy.deepcopy(sources.PINS)
    for row in pins["records"]:
        data=row["id"].encode()
        path=tmp_path/row["source_run"]/row["relative"]
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        row.update(sha256=sources.sha256(path),size_bytes=len(data))
    monkeypatch.setattr(sources,"PINS",pins)
    assert len(sources.records(tmp_path,"strategic"))==12
    assert len(sources.records(tmp_path,"matches"))==6
    row=pins["records"][0]
    (tmp_path/row["source_run"]/row["relative"]).write_bytes(b"broken")
    with pytest.raises(ValueError,match="Changed policy"):
        sources.records(tmp_path,"strategic")


def test_failed_audit_never_publishes_success(tmp_path,monkeypatch):
    from fhp_evaluation.strategic_audit import workflow,validation
    rows=[dict(r,path="/tmp/"+r["id"]) for r in sources.PINS["records"]]
    monkeypatch.setattr(strategic,"records",lambda *_:rows)
    monkeypatch.setattr(validation,"validate",lambda:{"passed":True})
    def fail(*args,**kwargs):
        assert kwargs["partition"]=="development"
        assert not kwargs["unlock_assessment"]
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(workflow,"run",fail)
    sources.write(tmp_path/"bundle.json",{})
    with pytest.raises(RuntimeError,match="simulated interruption"):
        strategic.main(["--input",str(tmp_path),"--output",str(tmp_path/"out"),
            "--suite-repo",str(SUITE),"--vr-repo",str(ROOT),"--ucv-repo",str(UCV),
            "--bundle-manifest",str(tmp_path/"bundle.json"),"--smoke"])
    assert not (tmp_path/"out/SUCCESS.json").exists()
    assert sources.read(tmp_path/"out/STATUS.json")["status"]=="incomplete"
    assert set(sources.read(tmp_path/"out/contract.json")["specs"])=={"development"}


def test_cached_shards_checked_without_recalculation(tmp_path):
    task=matches.build_tasks(VR,smoke=True)[0]
    contract=dict(identity={},tasks=portable([task]))
    result=dict(task=portable(task),summary=sample_summary([1,2]),elapsed_seconds=1)
    path=tmp_path/"task_results"/(task["id"]+".json")
    write_json(path,dict(signature=digest(contract),result_sha256=digest(result),result=result))
    assert execute_tasks([task],tmp_path,{},workers=1,seconds=1)==[result]
    result["summary"]["mean"]=999
    write_json(path,dict(signature=digest(contract),result_sha256="wrong",result=result))
    with pytest.raises(ValueError,match="Invalid cached"):
        execute_tasks([task],tmp_path,{},workers=1,seconds=1)


def test_two_task_batch_and_bundle(tmp_path):
    args=SimpleNamespace(run_id="vr7-eval2448-test",project="clever-overview-399515",region="europe-west1",
        service_account="fhp-vr-deep-runner@clever-overview-399515.iam.gserviceaccount.com",
        bucket="gs://clever-overview-399515-fhp-vr-deep-results",
        vr_bucket="gs://clever-overview-399515-fhp-vr-deep-results",
        ucv_bucket="gs://clever-overview-399515-fhp-escher-results")
    batch.validate(args)
    job=batch.build_job(args,"a"*64)
    group=job["taskGroups"][0]
    assert group["taskCount"]==group["parallelism"]==2 and group["taskCountPerNode"]==1
    assert group["taskSpec"]["maxRunDuration"]=="172800s"
    assert group["taskSpec"]["maxRetryCount"]==0
    assert group["taskSpec"]["runnables"][-1]["alwaysRun"]
    assert "sd_cfr" not in json.dumps(job)
    digest1,_=batch.bundle(SUITE,UCV,tmp_path/"a.tgz")
    digest2,_=batch.bundle(SUITE,UCV,tmp_path/"b.tgz")
    assert digest1==digest2
    with tarfile.open(tmp_path/"a.tgz") as archive:
        names=archive.getnames()
        assert "native/"+batch.MODULE_DIR+"/position_library.json" in names
        assert "ucv/fhp_escher/checkpointing.py" in names
        assert not any("cloud_outputs" in n or n.endswith((".pt",".pkl")) for n in names)
        assert not any(n.startswith("ucv/fhp_evaluation/") for n in names)


@pytest.mark.parametrize("name",["run_exp7_duration_evaluation.sh","vr_exp7_worker.sh","vr_exp7_sync.sh"])
def test_shell_syntax(name):
    subprocess.run(["bash","-n",str(ROOT/"gcp"/name)],check=True)
