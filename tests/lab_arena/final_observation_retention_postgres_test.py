"""Real accepted baseline/miner leases grant final capture only, with no row writes."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import capture_retention, trajectory, training_observations as obs
from lab_arena.service import LocalObjectStore, ServiceError
from lab_arena.store import hash_lease_token
from tests.lab_arena import final_provider_judge_diagnostic_postgres_test as native
from tests.lab_arena.training_observations_test import payload, observation

scenario = native.scenario
reset = native.reset


@pytest.mark.parametrize('actor', ['baseline','miner'])
def test_native_accepted_lease_replays_private_final_without_state_change(scenario, tmp_path, monkeypatch, actor, request):
    context, rid, pending = native.base.prepare(scenario, attempt=5)
    if actor == 'miner':
        pending = native.miner_pending(context, pending)
    worker = context[1].runner(0)
    request.addfinalizer(worker.close)
    claimed = worker.claim_one()
    assert claimed['status'] == 'leased'
    service = context[1].service
    owned = service.store.get_run(claimed['run_id'])
    claimed.update({key: owned[key] for key in ('kind','round_id','claim_request_hash')})
    objects = LocalObjectStore(tmp_path/'objects')
    monkeypatch.setattr(service.config, 'training_object_store', objects)
    monkeypatch.setattr(service, '_clock', lambda: datetime.now(timezone.utc))
    class Api:
        def training_observations(self, run_id, token, document, *, timeout_seconds=None):
            return service.handle_training_observations(run_id, token, document)
    config = SimpleNamespace(work_dir=tmp_path/'work', round_id=rid, identity=SimpleNamespace(hotkey='fixture-validator'),
        clock=service.now, claim_poll_seconds=30, api=Api())
    retention = capture_retention.InputRetention(config)
    document = payload(observation())
    path = retention.observations.retain(claimed, claimed['lease_token'], document)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result = {'schema_version':'leadpoet.lab_arena.run_result.v1','terminal_status':'accepted',
        'started_at':now,'finished_at':now,'resource_summary':{'wall_seconds':0.0,'cpu_seconds':0.0,
        'max_rss_bytes':0,'stdout_bytes':0,'stderr_bytes':0,'provider_call_count':0}}
    finish = trajectory.event('runtime.finished', {'status':'accepted','started_at':now,'finished_at':now,
        'resource_summary':result['resource_summary'],'exit_code':0,'timed_out':False,'company_count':0}, occurred_at=now)
    service.store.append_trajectory_events(claimed['run_id'], hash_lease_token(claimed['lease_token']), [finish])
    service.store.complete_attempt(run_id=claimed['run_id'], lease_token_hash=hash_lease_token(claimed['lease_token']),
        result=result, terminal_cause='accepted', output_ref='local-accepted-score')
    before = native.base.old.prior.snapshots(context)
    assert service.store.get_run(claimed['run_id'])['status'] == 'accepted'
    with pytest.raises(ServiceError):
        service._training_capture_run(claimed['run_id'], claimed['lease_token'])
    assert retention.observations.deliver(path, replay=True)
    assert not path.exists()
    assert obs.read_snapshot(objects, claimed['run_id'], 'score', claimed['lease_generation'], claimed['claim_request_hash'])['payload'] == document
    assert native.base.old.prior.snapshots(context) == before
