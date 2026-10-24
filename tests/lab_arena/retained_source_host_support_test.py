"""Expand host readers before any grader emits retained source context."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, signal_cache_control as control, signal_judgments
from lab_arena.service import ArenaService, hash_lease_token
from lab_arena.training_export import assemble_example
from tests.lab_arena import signal_cache_control_test as socket_tests
from tests.lab_arena.training_export_signal_test import _signal_fixture

PAGES = {"https://acme.example/launch": {
    "final_url": "https://acme.example/launch",
    "text": "Acme announces a new product launch for its customers.",
}}


def _fixture_with_context(monkeypatch, pages):
    original = signal_judgments.build_scope
    with monkeypatch.context() as patch:
        def build(**kwargs):
            return original(**kwargs, **({"verified_source_pages": pages} if pages is not None else {}))
        patch.setattr(signal_judgments, "build_scope", build)
        return _signal_fixture()


@pytest.mark.parametrize("pages", [None, PAGES])
def test_actual_runner_socket_and_host_http_reader_accept_old_and_new_requests(monkeypatch, pages):
    request = socket_tests.request()
    if pages is not None:
        request["verified_source_pages"] = deepcopy(pages)
    monkeypatch.setattr(socket_tests, "request", lambda: deepcopy(request))
    # WorkerSocketServer and HttpArenaApiClient each use the HOST validator.
    socket_tests.test_socket_to_gateway_has_no_provider_debit(monkeypatch)


def test_optional_field_is_json_stable_and_invalid_type_is_rejected():
    old = socket_tests.request()
    assert control.validate_request(old) == old
    request = {**old, "verified_source_pages": PAGES}
    assert control.validate_request(json.loads(contracts.canonical_json(request))) == request
    with pytest.raises(ValueError):
        control.validate_request({**old, "verified_source_pages": [PAGES]})


def test_source_bytes_change_scope_without_changing_absent_context_identity(monkeypatch):
    old = _fixture_with_context(monkeypatch, None)["signal_judgment_rows"][0]["scope_doc"]
    empty = _fixture_with_context(monkeypatch, {})["signal_judgment_rows"][0]["scope_doc"]
    present = _fixture_with_context(monkeypatch, PAGES)["signal_judgment_rows"][0]["scope_doc"]
    changed_pages = deepcopy(PAGES)
    changed_pages[next(iter(PAGES))]["text"] += " Changed source bytes."
    changed = _fixture_with_context(monkeypatch, changed_pages)["signal_judgment_rows"][0]["scope_doc"]
    assert old == empty and "verified_source_pages" not in old
    assert old["cache_key"] != present["cache_key"] != changed["cache_key"]
    assert present["signal_input_hash"] != changed["signal_input_hash"]
    assert present["expected_urls"] == old["expected_urls"]
    assert present["expected_context_urls"] == old["expected_context_urls"] == []


@pytest.mark.parametrize("pages", [None, PAGES])
def test_actual_training_export_reconstructs_saved_optional_context(monkeypatch, pages):
    values = _fixture_with_context(monkeypatch, pages)
    result = assemble_example(**values)
    evidence = result["judgment"]["signal_authority_evidence"][0]
    assert evidence["authority"]["scope_doc"] == values["signal_judgment_rows"][0]["scope_doc"]
    if pages is not None:
        assert evidence["authority"]["scope_doc"]["verified_source_pages"] == pages
        values["signal_judgment_rows"][0]["scope_doc"]["verified_source_pages"][next(iter(PAGES))]["text"] += " Tampered bytes."
        invalid = assemble_example(**values)
        assert "signal_authority_unverified" in invalid["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("pages", [None, PAGES])
def test_gateway_reader_passes_optional_context_into_existing_scope(monkeypatch, pages):
    values = _signal_fixture()
    executed, run, _ = values["runs"]
    run.update(kind="score", status="leased", lease_token_hash=hash_lease_token("lease-test-only"))
    executed["status"] = "accepted"
    observed = []
    def claim(**kwargs):
        observed.append(kwargs["scope"])
        return {"status": "busy"}
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(get_run=lambda run_id: next((r for r in values["runs"] if r["run_id"] == run_id), None), claim_signal_judgment=claim)
    service._objects = values["objects"]
    service._require_round_ownership = lambda _: None
    service._hot_round = lambda _: values["round_row"]
    icp = json.loads(values["objects"].values["benchmark"])["icps"][0]
    service.evaluation_icps = lambda _: [icp]
    request = {**socket_tests.request(), "verified_company_identity": {"decision": "match", "observed_domain": "acme.example"}}
    if pages is not None:
        request["verified_source_pages"] = deepcopy(pages)
    assert service.handle_signal_judgment(run["run_id"], "lease-test-only", request) == {"status": "busy"}
    assert len(observed) == 1
    assert observed[0].get("verified_source_pages") == pages
