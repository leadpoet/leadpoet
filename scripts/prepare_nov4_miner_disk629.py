"""Prepare the one Nov4 miner judgment through the existing frozen input builder.

The reviewed coordinator injects its existing read-only ArenaService, store and
object client. This module loads no credentials, starts no driver and calls no
provider. Apply the resulting private payload only with committed migration629.
"""
from __future__ import annotations

from copy import copy
import json
import os
from pathlib import Path

from lab_arena import contracts
from lab_arena.service import MAX_OUTPUT_BYTES

ROUND = 'arena-2026-11-04'
SUBMISSION = 'sub-2246da743ed59dcfe8b248e9d0ddfe50'
SOURCE = f'{ROUND}:{SUBMISSION}:2:7:3'
ROUND_FIELDS = ('round_id', 'status', 'status_generation', 'stage_generation',
                'scoring_generation_floor', 'configuration_doc', 'participants', 'stage2_scoring_plan_doc')
SOURCE_FIELDS = ('run_id', 'assignment_id', 'round_id', 'submission_id', 'miner_hotkey',
                 'kind', 'stage', 'icp_position', 'attempt', 'status', 'terminal_cause',
                 'stage_generation', 'scoring_generation_floor', 'output_ref')



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
            raise ValueError('miner629 retained output exceeds the requested bound')
        return self.raw


def prepare(service) -> dict:
    row = service._store.get_round(ROUND)
    source = service._store.get_run(SOURCE)
    submission = service._store.get_submission(SUBMISSION)
    if (not row or row['status'] not in ('stage2_scoring', 'stage2_judged', 'scored')
            or row['scoring_generation_floor'] != 0 or not source
            or source['status'] != 'accepted' or source['terminal_cause'] != 'accepted'
            or source['kind'] != 'execute' or source['stage'] != 2
            or source['icp_position'] != 7 or source['attempt'] != 3
            or not source.get('output_ref') or not submission or submission['is_king']
            or submission['status'] != 'frozen'
            or source['miner_hotkey'] != submission['miner_hotkey']):
        raise ValueError('miner629 source is not the exact accepted replacement')
    old = f'{ROUND}:{SUBMISSION}:2:7:2'
    slots = [item for item in row['stage2_scoring_plan_doc']['work_items']
             if item['submission_id'] == SUBMISSION and item['icp_position'] == 7]
    if len(slots) != 1 or slots[0]['scored_run_id'] != old:
        raise ValueError('miner629 original scoring slot differs or is already replaced')
    output = service._objects.get_bounded(source['output_ref'], MAX_OUTPUT_BYTES)
    requested = {'scored_run_id': SOURCE, 'submission_id': SUBMISSION,
                 'icp_position': 7, 'output_ref': source['output_ref']}
    reader = copy(service)
    reader._objects = _FrozenOutputRead(service._objects, source['output_ref'], output)
    work = reader._scoring_work_items(row, 2, {'work_items': [requested]})
    if len(work) != 1 or any(work[0].get(k) != v for k, v in requested.items()):
        raise ValueError('miner629 canonical scoring input differs')
    return {'schema_version': 'leadpoet.lab_arena.miner_disk629_preparation.v1',
            'expected_round': {key: row.get(key) for key in ROUND_FIELDS},
            'source': {key: source.get(key) for key in SOURCE_FIELDS},
            'output_sha256': contracts.hash_bytes(output), 'work_item': work[0]}


def write_preparation(service, path: Path) -> dict:
    """Write one exclusive private artifact; never print its data or credentials."""
    payload = prepare(service)
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    if len(raw) > 1024 * 1024:
        raise ValueError('miner629 preparation exceeds its fixed size bound')
    descriptor = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(raw)
    if Path(path).read_bytes() != raw:
        raise ValueError('miner629 private preparation readback differs')
    return {'path': str(path), 'sha256': contracts.hash_bytes(raw), 'run_id': SOURCE}
