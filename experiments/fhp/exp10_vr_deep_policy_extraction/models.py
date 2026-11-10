"""Native VR MLP and UCV's exact card/context policy architecture.

StructuredMLP mirrors StructuredFHPMLP in fhp_escher/features.py at UCV
6ab85bbcd03627281e411243419d64ce9e9236d5. Same parameter names, initialisers,
231/104 split, branch width 64, and 192x192 trunk; no changed learner code.
"""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn
from vr_deep_cfr.solver import MLP,SonnetLinear,ZeroInitLinear
from fhp_vr_deep.rich_ae import make_encoder


class StructuredMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_encoder=SonnetLinear(104,64,True)
        self.context_encoder=SonnetLinear(127,64,True)
        self.trunk=nn.Sequential(SonnetLinear(128,192,True),SonnetLinear(192,192,True))
        self.output=ZeroInitLinear(192,3,False)

    def forward(self,x):
        return self.output(self.trunk(torch.cat([self.card_encoder(x[...,:104]),self.context_encoder(x[...,104:])],dim=-1)))


def new_model(arm,seed):
    old=np.random.get_state()
    try:
        np.random.seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return StructuredMLP() if arm=='ucv_refit' else MLP(231,[64,64,64],3)
    finally:np.random.set_state(old)


def probabilities(model,x,m):
    return torch.softmax(model(x).masked_fill(m!=1,-1e20),-1)


class Playable:
    def __init__(self,model):
        self.model=model.eval();self.feature_encoder=make_encoder()

    def action_probabilities(self,state,player_id=None):
        player=state.current_player() if player_id is None else player_id
        legal=state.legal_actions(player)
        if not legal:return {}
        with torch.no_grad():
            x=torch.from_numpy(self.feature_encoder.information_state(state,player))
            m=torch.tensor(state.legal_actions_mask(player),dtype=torch.float32)
            p=probabilities(self.model,x,m).numpy()[legal].astype(np.float64)
        p/=p.sum()
        return dict(zip(legal,p.tolist()))


def load_policy(path,kind):
    # Only pinned project-owned source artifacts or this experiment's own output.
    path=Path(path)
    if kind=='ucv':
        with path.open('rb') as f:payload=pickle.load(f)
        if (payload['feature_encoder']!=make_encoder().metadata()
            or payload['policy_model']['hidden_layers']!=[192,192]
            or payload['policy_model']['branch_width']!=64):raise ValueError('UCV anchor architecture')
        model=StructuredMLP()
    else:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('feature_encoder')!=make_encoder().metadata():raise ValueError('Policy encoder mismatch')
        if kind=='vr':
            if payload.get('type')!='vr_deep_cfr_policy_snapshot' or payload['policy_network_layers']!=[64,64,64]:raise ValueError('Native policy architecture')
            model=MLP(231,[64,64,64],3)
        else:
            if payload.get('type')!='vr10_extraction_policy':raise ValueError('Output policy identity')
            model=new_model(payload['arm'],0)
    model.load_state_dict(payload['policy_state_dict'],strict=True)
    return Playable(model)
