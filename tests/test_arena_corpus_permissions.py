"""Check the narrow deny overlay preserves capture, restore reads and testing."""
import fnmatch
import json
from pathlib import Path

import pytest

POLICY = json.loads((Path(__file__).resolve().parents[1] /
    'deploy/aws/arena-routine-operator-corpus-deny.json').read_text())
BUCKETS = ['leadpoet-attested-v2-artifacts-493765492819',
           'leadpoet-private-model-artifacts-493765492819']


def denied(action, resource):
    return any(s['Effect'] == 'Deny' and
               any(fnmatch.fnmatchcase(action, a) for a in s['Action']) and
               any(fnmatch.fnmatchcase(resource, r) for r in s['Resource'])
               for s in POLICY['Statement'])


@pytest.mark.parametrize('bucket', BUCKETS)
def test_ordinary_agent_cannot_delete_versions_or_disable_retention(bucket):
    arn = 'arn:aws:s3:::' + bucket
    for action in ['s3:DeleteObject', 's3:DeleteObjectVersion']:
        assert denied(action, arn + '/private/arena-training/retained-proof.json')
    for action in ['s3:DeleteBucket', 's3:PutLifecycleConfiguration',
                   's3:PutBucketVersioning', 's3:PutBucketObjectLockConfiguration']:
        assert denied(action, arn)


@pytest.mark.parametrize('bucket', BUCKETS)
def test_capture_and_version_restore_access_are_unchanged(bucket):
    arn = 'arn:aws:s3:::' + bucket
    for action in ['s3:PutObject', 's3:GetObject', 's3:GetObjectVersion',
                   's3:PutObjectRetention', 's3:PutObjectLegalHold',
                   's3:PutObjectTagging', 's3:AbortMultipartUpload',
                   's3:ListBucket', 's3:ListBucketVersions']:
        assert not denied(action, arn)
        assert not denied(action, arn + '/private/arena-training/retained-proof.json')


def test_overlay_grants_nothing_and_does_not_change_other_resources():
    assert all(s['Effect'] == 'Deny' for s in POLICY['Statement'])
    assert all('Condition' not in s and 'NotResource' not in s for s in POLICY['Statement'])
    assert not denied('s3:DeleteObject', 'arn:aws:s3:::leadpoet-lab-isolated-recovery/test')
    assert not denied('s3:DeleteBucket', 'arn:aws:s3:::leadpoet-lab-isolated-recovery')
