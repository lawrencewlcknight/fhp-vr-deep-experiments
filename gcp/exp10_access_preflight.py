#!/usr/bin/env python3
"""Check the VM runner's explicit bucket grants before paying for a Batch VM.

This is a conservative allow-policy check, not an effective-access simulator.
Inherited/custom/conditional grants and organisation-level denies are not resolved.
No IAM policies are modified and no objects are downloaded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

SOURCE_BUCKETS = (
    'gs://clever-overview-399515-fhp-vr-deep-results',
    'gs://clever-overview-399515-fhp-escher-results',
)
READ_ROLES = {'roles/storage.objectViewer', 'roles/storage.objectUser',
              'roles/storage.objectAdmin', 'roles/storage.admin'}
WRITE_ROLES = READ_ROLES - {'roles/storage.objectViewer'}


def require_grant(policy, bucket, service_account, *, write=False):
    member = 'serviceAccount:' + service_account
    roles = {b.get('role') for b in policy.get('bindings', [])
             if member in b.get('members', []) and not b.get('condition')}
    accepted = roles & (WRITE_ROLES if write else READ_ROLES)
    if not accepted:
        raise ValueError(
            f"Cannot verify runner {'read/write' if write else 'read/list'} access: "
            f"{service_account} on {bucket}. The submitter's permissions are not "
            "the VM runner's permissions. This check requires a supported direct, "
            "unconditional bucket grant; inherited/custom/conditional access is "
            "not resolved. No IAM changes or job submission were made. "
            "For the standard Experiment 10 buckets, use "
            "fhp-vr-deep-runner@clever-overview-399515.iam.gserviceaccount.com.")
    return sorted(accepted)


def read_policy(bucket, project):
    result = subprocess.run(
        ['gcloud', 'storage', 'buckets', 'get-iam-policy', bucket,
         '--project', project, '--format=json'],
        capture_output=True, text=True, timeout=180, check=True)
    return json.loads(result.stdout)


def check(bucket, service_account, project, cloud=read_policy):
    bucket = bucket.rstrip('/')
    receipt = dict(status='passed', service_account=service_account,
                   checked_at=datetime.now(timezone.utc).isoformat(),
                   method='direct_unconditional_bucket_allow_grants',
                   effective_access_verified=False, bucket_grants=[])
    for source in sorted(set(SOURCE_BUCKETS) | {bucket}):
        policy = cloud(source, project)
        roles = require_grant(policy, source, service_account, write=source == bucket)
        receipt['bucket_grants'].append(dict(bucket=source, roles=roles, etag=policy.get('etag')))
        print(f'Runner bucket grant verified: {source} ({", ".join(roles)})', flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bucket', 'service-account', 'project']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    try:
        receipt = check(a.bucket, a.service_account, a.project)
    except (ValueError, subprocess.SubprocessError) as error:
        print(f'{error}\nAccess preflight stopped before VM submission.', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        return 1
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Access preflight passed: {a.receipt}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
