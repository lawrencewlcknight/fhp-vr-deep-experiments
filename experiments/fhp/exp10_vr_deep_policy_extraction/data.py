"""Pinned policy-only input slices. Never load critic replay or full learners."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
from fhp_vr_deep.io_utils import sha256_file
from .config import PINS, VR_BUCKET, UCV_BUCKET


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def local_file(root, algorithm, seed, relative):
    path=Path(root)/algorithm/str(seed)/relative
    if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Unsafe input path')
    return path


def fetch(root, vr_seeds=(0,1,2), anchors=True):
    """Explicit allowlist, ~952 MB per VR seed; no recursive cloud download."""
    for algorithm,seeds,bucket in [('vr',vr_seeds,VR_BUCKET),('ucv',range(3) if anchors else (),UCV_BUCKET)]:
        for seed in seeds:
            pin=PINS[algorithm][str(seed)]
            for name,sha in pin['files'].items():
                path=local_file(root,algorithm,seed,name)
                if path.is_file() and sha256_file(path)==sha:continue
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(path.suffix+'.download')
                uri=f"{bucket}/{PINS[algorithm+'_run']}/workers/{pin['worker']}/{name}"
                subprocess.run(['gcloud','storage','cp',uri,str(tmp)],check=True)
                if sha256_file(tmp)!=sha:raise ValueError(f'Source checksum mismatch: {uri}')
                tmp.replace(path)


def verify(root, algorithm, seed):
    pin=PINS[algorithm][str(seed)]
    for name,sha in pin['files'].items():
        path=local_file(root,algorithm,seed,name)
        if not path.is_file() or sha256_file(path)!=sha:raise ValueError(f'Missing/corrupt source {path}')
    return pin


@dataclass
class Groups:
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    masses: np.ndarray
    counts: np.ndarray
    rows: int


def group_replay(x,y,m,t,iteration):
    n=len(x);t=np.asarray(t,dtype=np.float64).reshape(-1)
    if not n or x.shape!=(n,231) or y.shape!=(n,3) or m.shape!=(n,3) or t.shape!=(n,):raise ValueError('Replay shape')
    for a in [x,y,m,t]:
        if not np.isfinite(a).all():raise ValueError('Nonfinite replay')
    if (iteration<=0 or np.any(t<1) or np.any(t>iteration) or np.any(y<0)
        or not np.allclose(y.sum(1),1,atol=1e-5) or not np.isin(m,[0,1]).all()
        or np.any(m.sum(1)==0) or np.any(y[m==0]!=0)):raise ValueError('Invalid replay values')
    # Same float32 input AND legal mask define a group. Iteration weights are
    # VR's gamma=2 objective weights, not uniform group weights or reach estimates.
    _,first,inverse,counts=np.unique(x,axis=0,return_index=True,return_inverse=True,return_counts=True)
    if not np.array_equal(m,m[first][inverse]):raise ValueError('Group mask conflict')
    weights=(t/iteration*2)**2
    masses=np.bincount(inverse,weights=weights,minlength=len(first))
    sums=np.column_stack([np.bincount(inverse,weights=weights*np.asarray(y[:,a],dtype=np.float64),minlength=len(first)) for a in range(3)])
    return Groups(np.asarray(x[first],dtype=np.float32),np.asarray(sums/masses[:,None],dtype=np.float32),
                  np.asarray(m[first],dtype=np.float32),masses,counts,n)


def load_replay(root,seed):
    pin=verify(root,'vr',seed)
    meta=json.loads(local_file(root,'vr',seed,'training_state/manifest.json').read_text())
    if (meta['is_smoke'] or meta['completed_target_hours']!=48 or meta['seed']!=seed
        or meta['iteration']!=pin['iteration'] or meta['training_config']['gamma']!=2.
        or meta['feature_encoder']['id']!='fhp_hand_board_strategic_ae_v1'):raise ValueError('Source contract mismatch')
    b=meta['buffers']['policy']; arrays=[]
    for k in ['infostate_buf','q_value_buf','q_value_mask_buf','iteration_buf']:
        spec=b['arrays'][k];a=np.load(local_file(root,'vr',seed,'training_state/'+spec['file']),mmap_mode='r',allow_pickle=False)
        if list(a.shape)!=spec['shape'] or a.dtype.str!=spec['dtype']:raise ValueError('Replay schema mismatch')
        arrays.append(a)
    if len(arrays[0])!=b['populated_rows'] or b['populated_rows']!=1000000:raise ValueError('Incomplete policy replay')
    return arrays,meta
