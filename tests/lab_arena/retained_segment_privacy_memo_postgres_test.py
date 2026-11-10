"""Native baseline/miner captures keep exact final bytes and database state."""
from types import SimpleNamespace

import pytest

from lab_arena import contracts, runner, training_observations as obs
from lab_arena.service import LocalObjectStore
from tests.lab_arena import final_provider_judge_diagnostic_postgres_test as native
from tests.lab_arena.retained_segment_privacy_memo_test import TEXT
from tests.lab_arena.training_observations_test import observation

scenario=native.scenario
reset=native.reset


@pytest.mark.parametrize('actor',['baseline','miner'])
def test_native_previous_segment_and_final_preserve_every_row(scenario,tmp_path,monkeypatch,actor,request):
    context,rid,pending=native.base.prepare(scenario,attempt=5)
    if actor=='miner': pending=native.miner_pending(context,pending)
    worker=context[1].runner(0)
    request.addfinalizer(worker.close)
    claimed=worker.claim_one()
    assert claimed['status']=='leased' and claimed['run_id']==pending['run_id']
    service=context[1].service
    owned=service.store.get_run(claimed['run_id'])
    claimed.update({key:owned[key] for key in ('kind','round_id','claim_request_hash')})
    objects=LocalObjectStore(tmp_path/'private-objects')
    monkeypatch.setattr(service.config,'training_object_store',objects)
    before=native.base.old.prior.snapshots(context)
    api=SimpleNamespace(training_observation_segment=service.handle_training_observation_segment)
    state=runner.RunState(lease=claimed,lease_token=claimed['lease_token'])
    journal=runner._ObservationJournal(api,claimed,claimed['lease_token'],state)
    for index in range(8):
        row=observation(ordinal=index,action=index)
        row['request']['admitted_parameters']['messages'][0]['content']=TEXT
        state.worker_observations.append(row)
    assert journal._prepare() and journal._send_pending()
    state.worker_observations.append(observation(ordinal=8,action=8))
    assert journal._prepare() and journal._send_pending()  # Native previous-read gate.
    document=state.worker_observations.document()
    final=service.handle_training_observations(claimed['run_id'],claimed['lease_token'],document)
    restored=obs.read_snapshot(objects,claimed['run_id'],'score',claimed['lease_generation'],claimed['claim_request_hash'])
    assert restored['payload']==document and final['payload_sha256']==contracts.document_hash(document)
    segments,partial,error=obs.read_journal_prefix(objects,claimed['run_id'],'score',claimed['lease_generation'],claimed['claim_request_hash'])
    assert len(segments)==2 and error is None and partial['complete'] is False
    assert partial['observations']==document['observations']
    assert native.base.old.prior.snapshots(context)==before
