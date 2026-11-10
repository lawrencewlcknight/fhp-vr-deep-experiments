#!/usr/bin/env python3
"""Download verified analysis and fit diagnostics, never replay or model files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def copy(uri,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['gcloud','storage','cp',uri,str(path)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='gs://bucket/run-id')
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if not a.source.startswith('gs://'):p.error('GCS run prefix required')
    remote=a.source.rstrip('/')
    with tempfile.TemporaryDirectory() as tmp:
        receipt_path=Path(tmp)/'SUCCESS.json';copy(remote+'/analysis/SUCCESS.json',receipt_path)
        receipt=json.loads(receipt_path.read_text())
        for name,sha in receipt['files'].items():
            if Path(name).name!=name or Path(name).suffix not in {'.json','.csv','.md'}:raise ValueError('Unexpected analysis file')
            file=a.output/'analysis'/name;copy(remote+'/analysis/'+name,file)
            if hashlib.sha256(file.read_bytes()).hexdigest()!=sha:raise ValueError('Analysis checksum mismatch')
        copy(remote+'/experiment.json',a.output/'experiment.json')
        identity=json.loads((a.output/'experiment.json').read_text())
        cfg=identity['contract']
        digest=hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if digest!=receipt['contract_sha256']:raise ValueError('Wrong experiment contract')
        for seed in cfg['seeds']:
            base=f'fits/{seed}/';fit_receipt=Path(tmp)/f'fit{seed}.json'
            copy(remote+'/'+base+'SUCCESS.json',fit_receipt)
            fit=json.loads(fit_receipt.read_text());file=a.output/base/'result.json'
            copy(remote+'/'+base+'result.json',file)
            if fit['contract_sha256']!=digest or hashlib.sha256(file.read_bytes()).hexdigest()!=fit['files']['result.json']:
                raise ValueError('Fit diagnostics checksum mismatch')
        # Publish local completion marker only after every check has succeeded.
        (a.output/'analysis/SUCCESS.json').write_bytes(receipt_path.read_bytes())
    print(f'Analysis verified in {a.output}; no model or replay objects downloaded.')


if __name__=='__main__':main()
