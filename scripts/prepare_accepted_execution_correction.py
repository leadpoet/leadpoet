"""Prepare one accepted execution correction through the frozen input builder.

The owning operator supplies its existing read-only service. This module loads
no credentials, starts no driver and calls no provider. Apply the private result
with the owner-only migration703 preparation function.
"""
from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path

from lab_arena import contracts
from lab_arena.service import MAX_OUTPUT_BYTES

ROUND_FIELDS = ('round_id', 'status', 'status_generation', 'stage_generation',
                'scoring_generation_floor', 'configuration_doc', 'participants', 'stage2_scoring_plan_doc')
SOURCE_FIELDS = ('run_id', 'assignment_id', 'round_id', 'submission_id', 'miner_hotkey',
                 'kind', 'stage', 'icp_position', 'attempt', 'status', 'terminal_cause',
                 'stage_generation', 'scoring_generation_floor', 'output_ref')
MAX_PREPARATION_BYTES = 8 * 1024 * 1024



class _FrozenOutputRead:
    """Bind this one canonical input build to the bytes hashed in its receipt."""

    def __init__(self, original, ref, raw):
        self.original, self.ref, self.raw = original, ref, raw

    def get(self, ref):
        return self.raw if ref == self.ref else self.original.get(ref)

    def get_bounded(self, ref, limit):
        if ref != self.ref:
            return self.original.get_bounded(ref, limit)
        if len(self.raw) > limit:
            raise ValueError('accepted execution correction retained output exceeds the requested bound')
        return self.raw


def prepare(service, run_id: str) -> dict:
    source = service._store.get_run(run_id)
    if not source:
        raise ValueError('corrected execution is missing')
    row = service._store.get_round(source['round_id'])
    submission = service._store.get_submission(source['submission_id'])
    if (not row or row['status'] != 'stage2_scoring'
            or source['status'] != 'accepted' or source['terminal_cause'] != 'accepted'
            or source['kind'] != 'execute' or source['stage'] != 2
            or not source.get('output_ref') or not submission or submission['is_king']
            or submission['status'] != 'frozen'
            or source['miner_hotkey'] != submission['miner_hotkey']):
        raise ValueError('corrected execution is not an accepted miner replacement')
    slots = [item for item in row['stage2_scoring_plan_doc']['work_items']
             if item['submission_id'] == source['submission_id']
             and item['icp_position'] == source['icp_position']]
    if len(slots) != 1 or slots[0]['scored_run_id'] == run_id:
        raise ValueError('original scoring slot differs or is already replaced')
    output = service._objects.get_bounded(source['output_ref'], MAX_OUTPUT_BYTES)
    requested = {'scored_run_id': run_id, 'submission_id': source['submission_id'],
                 'icp_position': source['icp_position'], 'output_ref': source['output_ref']}
    reader = copy(service)
    reader._objects = _FrozenOutputRead(service._objects, source['output_ref'], output)
    work = reader._scoring_work_items(row, 2, {'work_items': [requested]})
    if len(work) != 1 or any(work[0].get(k) != v for k, v in requested.items()):
        raise ValueError('accepted execution correction canonical scoring input differs')
    return {'schema_version': 'leadpoet.lab_arena.accepted_execution_preparation.v1',
            'expected_round': {key: row.get(key) for key in ROUND_FIELDS},
            'source': {key: source.get(key) for key in SOURCE_FIELDS},
            'output_sha256': contracts.hash_bytes(output), 'work_item': work[0]}


def write_preparation(service, run_id: str, path: Path) -> dict:
    """Write one exclusive private artifact; never print its data or credentials."""
    payload = prepare(service, run_id)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    if len(raw) > MAX_PREPARATION_BYTES:
        raise ValueError('accepted execution correction preparation exceeds its fixed size bound')
    descriptor = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(raw)
    if Path(path).read_bytes() != raw:
        raise ValueError('accepted execution correction private preparation readback differs')
    return {'path': str(path), 'sha256': contracts.hash_bytes(raw), 'run_id': run_id}
