"""Publication cannot treat an unavailable model route as a measured empty result."""
import base64
import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.broker import BrokerError
from lab_arena.service import ArenaService, _PublicationReads
from lab_arena.store import ArenaStore, ArenaStoreError


def case():
    run = dict(run_id="run", assignment_id="assignment", attempt=1, icp_position=0,
               submission_id="miner", kind="execute", status="accepted", stage=2,
               output_ref="saved-output")
    row = dict(round_id="round", status="scored", participants=[{"submission_id": "miner"}],
               configuration_doc={})
    call = dict(operation_id="openrouter.chat", provider_status=404,
                call_identity="native-call", request_hash="sha256:" + "a" * 64)
    events = [dict(run_id="run", event_kind="provider.request", operation_id="openrouter.chat", action_sequence=3),
              dict(run_id="run", event_kind="provider.response", action_sequence=3, call=call)]
    binding = dict(run_id="run", call_identity="native-call", operation_id="openrouter.chat",
                   provider="openrouter", funding_source="miner_key")
    body = {"error": {"code": 404, "message": "No endpoints found matching your data policy (Zero data retention)."}}
    ledger = [dict(binding, entry_id=1, entry_kind="reservation", entry_doc={
                  "action_sequence": 3, "request_hash": call["request_hash"]}),
              dict(binding, entry_id=2, entry_kind="dispatch"),
              dict(binding, entry_id=3, entry_kind="settlement", amount_microusd=0,
                   terminal_response={"status": 404, "headers": {}, "call_succeeded": False,
                                      "body_b64": base64.b64encode(json.dumps(body).encode()).decode()})]
    output = {"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION, "companies": []}
    calls = []
    service = ArenaService.__new__(ArenaService)
    service._objects = SimpleNamespace(get_bounded=lambda *a: json.dumps(output).encode())
    service._store = SimpleNamespace(
        latest_execution_model_reservation=lambda run_id: calls.append(("reservation", run_id)) or copy.deepcopy(ledger[0]),
        list_ledger=lambda **kw: calls.append(("ledger", kw)) or copy.deepcopy(ledger),
        list_runs=lambda *a, **kw: [copy.deepcopy(run)])
    reads = _PublicationReads(service, row, [run])
    return service, reads, events, ledger, output, calls


def test_native_route_denial_holds_and_preserves_every_saved_authority():
    service, reads, events, ledger, output, calls = case()
    original = copy.deepcopy((events, ledger, output, reads.runs))
    assert service._empty_execution_infrastructure_failure(reads) is True
    assert (events, ledger, output, reads.runs) == original
    assert [call[0] for call in calls] == ["reservation", "ledger"]


@pytest.mark.parametrize("succeeded", [None, True, 0, "false"])
def test_route_denial_with_missing_or_contradictory_success_is_not_recovery(succeeded):
    service, reads, _, ledger, _, _ = case()
    ledger[-1]["terminal_response"]["call_succeeded"] = succeeded
    with pytest.raises((ArenaStoreError, BrokerError)):
        service._empty_execution_infrastructure_failure(reads)
    service._round = lambda _: reads.round_row
    service._baseline_fully_succeeded = lambda _: True
    service._hold_incomplete_evaluation = lambda row, **kw: {"status": "retry", **kw}
    assert service.publish("round") == {"status": "retry", "reason": "execution_evidence_unavailable"}


@pytest.mark.parametrize("status,message", [(404, "Model does not exist"), (401, "Invalid key"),
    (402, "Insufficient credits"), (403, "Forbidden"), (200, "No endpoints found matching your data policy (Zero data retention).")])
def test_other_native_outcomes_do_not_become_infrastructure_holds(status, message):
    service, reads, _, ledger, _, _ = case()
    ledger[-1]["terminal_response"].update(status=status, body_b64=base64.b64encode(
        json.dumps({"error": {"code": status, "message": message}}).encode()).decode())
    assert service._empty_execution_infrastructure_failure(reads) is False


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
def test_later_issued_success_recovers_without_trusting_trajectory_metadata(operation):
    service, reads, events, ledger, _, calls = case()
    for row in ledger:
        row["operation_id"] = operation
    ledger[0]["entry_doc"]["action_sequence"] = 4
    ledger[-1]["terminal_response"].update(status=200, call_succeeded=True,
        body_b64=base64.b64encode(b'{"result":"valid response"}').decode())
    assert service._empty_execution_infrastructure_failure(reads) is False
    assert [call[0] for call in calls] == ["reservation", "ledger"]


def test_forged_later_trajectory_success_cannot_mask_native_route_failure():
    service, reads, events, _, _, _ = case()
    events.insert(0, dict(run_id="run", event_kind="provider.response", action_sequence=999,
                         call={"provider_status": 200, "operation_id": "openrouter.responses"}))
    assert service._empty_execution_infrastructure_failure(reads) is True


def test_nonempty_output_does_not_read_provider_history():
    service, reads, _, _, _, calls = case()
    reads.outputs_by_ref["saved-output"] = {"companies": [{"company_name": "already validated"}]}
    assert service._empty_execution_infrastructure_failure(reads) is False
    assert calls == []


def test_nonparticipant_runs_are_not_examined():
    service, reads, _, _, _, calls = case()
    reads.round_row["participants"] = []
    assert service._empty_execution_infrastructure_failure(reads) is False
    assert calls == []


@pytest.mark.parametrize("change", [lambda rows: rows.pop(1),
    lambda rows: rows[-1].update(run_id="other"),
    lambda rows: rows[-1].update(funding_source="other"),
    lambda rows: rows[0]["entry_doc"].update(action_sequence="9"),
    lambda rows: rows[0]["entry_doc"].update(request_hash=None),
    lambda rows: rows[-1].update(entry_id=0)])
def test_broken_native_binding_is_not_treated_as_a_valid_score(change):
    service, reads, _, ledger, _, _ = case()
    change(ledger)
    with pytest.raises(ArenaStoreError):
        service._empty_execution_infrastructure_failure(reads)


@pytest.mark.parametrize("broken", [False, True])
def test_actual_publish_holds_before_ranking_or_promotion(broken):
    service, reads, _, ledger, _, _ = case()
    if broken:
        ledger.pop(1)
    service._round = lambda _: reads.round_row
    service._baseline_fully_succeeded = lambda _: True
    holds = []
    service._hold_incomplete_evaluation = lambda row, **kw: holds.append(kw) or {"status": "retry", **kw}
    result = service.publish("round")
    assert result["status"] == "retry"
    assert holds == [{"reason": "execution_evidence_unavailable" if broken else "execution_infrastructure_incomplete"}]


def test_reservation_query_is_one_bounded_native_projection():
    calls = []
    store = ArenaStore.__new__(ArenaStore)
    store._transport = SimpleNamespace(select=lambda table, **kw: calls.append((table, kw)) or [])
    assert store.latest_execution_model_reservation("run") is None
    table, query = calls[0]
    assert table == "lab_arena_ledger"
    assert query["filters"] == {"run_id": "run", "provider": "openrouter", "entry_kind": "reservation"}
    assert query["order"] == "model_call_sequence" and query["descending"]
    assert query["limit"] == 1
    assert "terminal_response" not in query["columns"]


@pytest.mark.parametrize("terminal", [None, {"status": 200}])
def test_missing_or_malformed_native_terminal_cannot_prove_recovery(terminal):
    service, reads, _, ledger, _, _ = case()
    ledger[-1]["terminal_response"] = terminal
    with pytest.raises((ArenaStoreError, BrokerError, ValueError, KeyError)):
        service._empty_execution_infrastructure_failure(reads)
