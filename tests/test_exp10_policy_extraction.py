"""Scientific invariants and failure paths of the fixed-replay diagnosis."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.fhp.exp10_vr_deep_policy_extraction.config import contract, PINS
from experiments.fhp.exp10_vr_deep_policy_extraction.data import group_replay, digest, write_json, local_file
from experiments.fhp.exp10_vr_deep_policy_extraction.models import new_model, probabilities
from experiments.fhp.exp10_vr_deep_policy_extraction.fitting import fit_one
from experiments.fhp.exp10_vr_deep_policy_extraction.evaluation import planned_tasks, duplicate_values, moments
from experiments.fhp.exp10_vr_deep_policy_extraction.analysis import aggregate, pool_moments, bootstrap
from fhp_vr_deep.game import load_fhp_game
from gcp.exp10_policy_extraction_batch import build_job
from gcp.exp10_access_preflight import require_grant, check as check_access, SOURCE_BUCKETS


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def arrays(self):
        x=np.zeros((4,231),np.float32);x[2:,0]=1
        y=np.array([[.1,.9,0],[.7,.3,0],[0,.2,.8],[0,.9,.1]],np.float32)
        m=np.array([[1,1,0],[1,1,0],[0,1,1],[0,1,1]],np.float32)
        t=np.array([[1],[3],[2],[4]],np.float32)
        return x,y,m,t

    def test_grouping_preserves_weighted_row_ce_and_gradient(self):
        x,y,m,t=self.arrays();g=group_replay(x,y,m,t,4)
        model=new_model('ucv_refit',17)
        with torch.no_grad():model.output.bias.copy_(torch.tensor([.3,-.7,.2]))
        def ce(features,targets,masks,weights):
            logits=model(torch.from_numpy(features)).masked_fill(torch.from_numpy(masks)!=1,-1e20)
            return (-(torch.from_numpy(targets)*torch.log_softmax(logits,-1)).sum(1)*weights).mean()
        row=ce(x,y,m,torch.from_numpy((t[:,0]/4*2)**2));row.backward()
        grads=[p.grad.clone() for p in model.parameters()];model.zero_grad()
        grouped=ce(g.features,g.targets,g.masks,torch.tensor(g.masses*len(g.masses)/g.rows,dtype=torch.float32));grouped.backward()
        torch.testing.assert_close(row,grouped)
        for before,p in zip(grads,model.parameters()):torch.testing.assert_close(before,p.grad,rtol=1e-5,atol=1e-6)
        self.assertEqual(g.counts.tolist(),[2,2])

    def test_native_step_matches_existing_vr_loss(self):
        from types import SimpleNamespace
        from vr_deep_cfr.solver import AvePolicyTrainer
        arrays=self.arrays();g=group_replay(*arrays,4);cfg=contract(True)
        model=new_model('native_refit',cfg['init_seed_base']);opt=torch.optim.Adam(model.parameters(),lr=.001)
        trainer=SimpleNamespace(gamma=2.,loss_fn=torch.nn.MSELoss(),predict=lambda x,m:probabilities(model,x,m))
        loss=AvePolicyTrainer.compute_loss(trainer,tuple(torch.from_numpy(a) for a in arrays),4)
        loss.backward();opt.step()
        actual,_=fit_one(arrays,g,4,arm='native_refit',seed=0,replicate=0,cfg=cfg,steps=1)
        for a,b in zip(model.parameters(),actual.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_replay_and_mask_conflict_rejected(self):
        a=list(self.arrays());a[2][1]=[1,1,1]
        with self.assertRaisesRegex(ValueError,'mask conflict'):group_replay(*a,4)
        a=list(self.arrays());a[3][0]=0
        with self.assertRaises(ValueError):group_replay(*a,4)
        a=list(self.arrays());a[0][0,0]=np.nan
        with self.assertRaises(ValueError):group_replay(*a,4)

    def test_independent_seed_tasks_and_total_work(self):
        tasks=planned_tasks(contract());self.assertEqual(len(tasks),480)
        self.assertEqual(len({t['id'] for t in tasks}),480)
        self.assertEqual(len({t['deal_seed'] for t in tasks}),480)
        self.assertEqual(sum(t['deals']*(2 if t['kind']=='self' else 3) for t in tasks),6600000)
        self.assertEqual(tasks,planned_tasks(contract()))

    def test_duplicate_seats_cancel_identical_policy(self):
        class Uniform:
            def action_probabilities(self,state):
                legal=state.legal_actions();return {a:1/len(legal) for a in legal}
        a=Uniform();values=duplicate_values(load_fhp_game(),a,a,30,17)
        np.testing.assert_array_equal(values,np.zeros(30))

    def test_paired_moments_retain_covariance(self):
        a=np.array([1.,2,3,4]);b=a+2
        result=pool_moments([moments((a-b)[:2]),moments((a-b)[2:])])
        self.assertEqual(result['mean_mbb'],-2)
        self.assertEqual(result['conditional_mc_se_mbb'],0)
        self.assertGreater(pool_moments([moments(a)])['conditional_mc_se_mbb'],0)
        self.assertEqual(bootstrap(np.ones((3,3))*7)['ci_low'],7)

    def test_aggregate_rejects_missing_corrupt_and_incomplete_shards(self):
        cfg=contract(True)
        with tempfile.TemporaryDirectory() as root, patch('experiments.fhp.exp10_vr_deep_policy_extraction.analysis.check_fit',return_value={}):
            with self.assertRaises(FileNotFoundError):aggregate(root,cfg)
            task=planned_tasks(cfg)[0]
            row=dict(identity=digest(dict(task=task,contract=digest(cfg),fit_hash=digest({0:{}}))),task=task,scores={})
            row['checksum']=digest(row)
            path=Path(root)/'matches'/f"{task['id']}.json";write_json(path,row)
            with self.assertRaisesRegex(ValueError,'Incomplete metrics'):aggregate(root,cfg)
            row['checksum']='invalid';write_json(path,row)
            with self.assertRaisesRegex(ValueError,'corrupt'):aggregate(root,cfg)

    def test_allowlist_has_only_policy_arrays_and_pinned_sources(self):
        for pin in PINS['vr'].values():
            arrays=[name for name in pin['files'] if name.endswith('.npy')]
            self.assertEqual(len(arrays),4)
            self.assertTrue(all('/policy__' in name for name in arrays))
        with self.assertRaises(ValueError):local_file('/tmp/input','vr',0,'../../../../etc/passwd')

    def test_batch_script_parses_and_does_not_launch_training(self):
        kwargs=dict(run_id='vr10-test',bucket='gs://example-results',service_account='runner@example.iam.gserviceaccount.com',repo_ref='a'*40)
        for profile in [False,True]:
            job=build_job(**kwargs,profile_only=profile)
            script=job['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text']
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertEqual(' evaluate ' in script,not profile)
            self.assertIn('uv python install 3.11.16',script)
            self.assertNotIn('imageUri',json.dumps(job))
        with self.assertRaises(ValueError):build_job(**(kwargs|{'run_id':'bad; echo x'}))

    def test_access_check_uses_runner_not_submitter_or_another_runner(self):
        runner='fhp-vr-deep-runner@example.iam.gserviceaccount.com'
        binding=dict(role='roles/storage.objectAdmin',members=['serviceAccount:'+runner])
        policy=dict(bindings=[binding])
        self.assertEqual(require_grant(policy,'gs://bucket',runner,write=True),['roles/storage.objectAdmin'])
        with self.assertRaisesRegex(ValueError,'Cannot verify runner'):
            require_grant(policy,'gs://bucket','leduc-dream-runner@example.iam.gserviceaccount.com')
        for invalid in [dict(binding,condition={'expression':'true'}),
                        dict(binding,members=['user:submitter@example.com']),
                        dict(binding,role='roles/storage.objectCreator')]:
            with self.assertRaises(ValueError):require_grant(dict(bindings=[invalid]),'gs://bucket',runner,write=True)
        viewer=dict(binding,role='roles/storage.objectViewer')
        require_grant(dict(bindings=[viewer]),'gs://bucket',runner)
        with self.assertRaises(ValueError):require_grant(dict(bindings=[viewer]),'gs://bucket',runner,write=True)

    def test_preflight_checks_both_sources_and_output_with_correct_roles(self):
        from experiments.fhp.exp10_vr_deep_policy_extraction.config import VR_BUCKET, UCV_BUCKET
        self.assertEqual(set(SOURCE_BUCKETS),{VR_BUCKET,UCV_BUCKET})
        runner='runner@example.iam.gserviceaccount.com';seen=[]
        def cloud(bucket,project):
            self.assertEqual(project,'test-project');seen.append(bucket)
            role='roles/storage.objectAdmin' if bucket==VR_BUCKET else 'roles/storage.objectViewer'
            return dict(bindings=[dict(role=role,members=['serviceAccount:'+runner])])
        result=check_access(VR_BUCKET,runner,'test-project',cloud)
        self.assertEqual(set(seen),set(SOURCE_BUCKETS))
        self.assertEqual(result['status'],'passed')
        self.assertFalse(result['effective_access_verified'])
        def readonly(bucket,project):
            return dict(bindings=[dict(role='roles/storage.objectViewer',members=['serviceAccount:'+runner])])
        with self.assertRaises(ValueError):check_access(VR_BUCKET,runner,'test-project',readonly)


if __name__=='__main__':unittest.main()
