"""Request persistence uses the same validated immutable reservation proof."""

import copy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, training_capture as capture
from lab_arena.service import LocalObjectStore
from lab_arena.store import ArenaStoreUnavailable
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeLedgerStore, FakeTransport, make_broker,
)
from tests.lab_arena.training_capture_test import _execute


class OneReadStore(FakeLedgerStore):
    def __init__(self, *, ambiguous=True):
        super().__init__()
        self.ambiguous = ambiguous
        self.reserve_arguments = []
        self.ledger_reads = 0

    def reserve_call(self, **kwargs):
        self.reserve_arguments.append(copy.deepcopy(kwargs))
        result = super().reserve_call(**kwargs)
        if self.ambiguous and len(self.reserve_arguments) == 1:
            raise ArenaStoreUnavailable("first reservation response lost")
        return {**result, "idempotent": True}

    def list_ledger(self, **kwargs):
        self.ledger_reads += 1
        if self.ledger_reads > 1:
            raise ArenaStoreUnavailable("redundant ledger read unavailable")
        return copy.deepcopy(super().list_ledger(**kwargs))


@pytest.mark.parametrize("ambiguous", [False, True])
def test_validated_reservation_rows_persist_request_without_second_read(tmp_path, ambiguous):
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        store=OneReadStore(ambiguous=ambiguous),
        capture_request=capture.RequestCapture(objects).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})]),
    )
    result = _execute(provider)
    assert result.status == 200 and len(transport.sent) == 1
    receipt = next(iter(store.calls.values()))
    restored = capture.read_request(objects, receipt["call_doc"]["request_capture"])
    assert restored["outbound"]["body"] == json.loads(transport.sent[0]["body"])
    assert store.ledger_reads == 1
    assert store.log == ["reserve"] * (2 if ambiguous else 1) + ["dispatch", "settle"]


@pytest.mark.parametrize("damage", ["pointer", "missing_pointer", "request_hash", "order", "null_rows"])
def test_reuse_keeps_exact_pointer_match_and_rejects_invalid_proof(tmp_path, damage):
    class DamagedStore(OneReadStore):
        def list_ledger(self, **kwargs):
            rows = super().list_ledger(**kwargs)
            if damage == "null_rows":
                return None
            if damage == "pointer":
                rows[0]["entry_doc"]["request_capture"]["sha256"] = "sha256:" + "0" * 64
            elif damage == "missing_pointer":
                rows[0]["entry_doc"].pop("request_capture")
            elif damage == "request_hash":
                rows[0]["entry_doc"]["request_hash"] = "wrong"
            else:
                rows[0]["entry_id"] = 0
            return rows
    provider, store, transport = make_broker(
        store=DamagedStore(), capture_request=capture.RequestCapture(LocalObjectStore(tmp_path)).prepare,
        transport=FakeTransport([(200, {"usage": {"cost": "0.000001"}})]),
    )
    result = _execute(provider)
    pointer_only = damage in {"pointer", "missing_pointer"}
    assert result.status == (200 if pointer_only else 503)
    assert len(transport.sent) == int(pointer_only)
    assert store.ledger_reads == 1
    assert not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_first_proof_read_failure_and_early_identity_rejection_are_unchanged(tmp_path, wrong_identity):
    terminal = ArenaStoreUnavailable("first proof unavailable")
    class FailedProofStore(OneReadStore):
        def reserve_call(self, **kwargs):
            result = super().reserve_call(**kwargs)
            return {**result, "call_identity": "sha256:" + "0" * 64} if wrong_identity else result
        def list_ledger(self, **kwargs):
            self.ledger_reads += 1
            raise terminal
    provider, store, transport = make_broker(
        store=FailedProofStore(), capture_request=capture.RequestCapture(LocalObjectStore(tmp_path)).prepare,
    )
    if wrong_identity:
        assert _execute(provider).status == 503
    else:
        with pytest.raises(ArenaStoreUnavailable) as raised:
            _execute(provider)
        assert raised.value is terminal
    assert store.ledger_reads == (0 if wrong_identity else 1)
    assert transport.sent == [] and not list(tmp_path.rglob("*.gz"))


@pytest.mark.parametrize("kind,expected_status", [
    ("settlement", 200), ("dispatch", 409), ("uncertain", 409), ("recovery", 402),
])
def test_verified_nonreserved_states_keep_existing_replay_without_capture_repair(tmp_path, kind, expected_status):
    class TerminalStore(OneReadStore):
        def reserve_call(self, **kwargs):
            super().reserve_call(**kwargs)
            call = self.calls[kwargs["call_identity"]]
            call["kind"] = kind
            if kind == "settlement":
                call["actual"] = 1
                call["terminal"] = br._terminal_response_document(
                    200, {}, b'{"saved":true}', call_succeeded=True,
                )
            return self._view(call)
        def list_ledger(self, **kwargs):
            self.ledger_reads += 1
            return copy.deepcopy(FakeLedgerStore.list_ledger(self, **kwargs))
    provider, store, transport = make_broker(
        store=TerminalStore(), capture_request=capture.RequestCapture(LocalObjectStore(tmp_path)).prepare,
    )
    result = _execute(provider)
    assert result.status == expected_status
    # The existing uncertainty/billing replay still needs its fresh history.
    assert store.ledger_reads == (2 if kind == "uncertain" else 1)
    assert transport.sent == [] and not list(tmp_path.rglob("*.gz"))
    assert store.log == ["reserve", "reserve"]


@pytest.mark.parametrize("unavailable", [False, True])
def test_full_thinking_refusal_retains_own_capture_read(tmp_path, unavailable):
    from lab_arena import reasoning_capture
    from tests.lab_arena.reasoning_capture_test import _reviewed_registry

    class RefusalStore(FakeLedgerStore):
        def __init__(self):
            super().__init__()
            self.reads = 0
        def reserve_call(self, **kwargs):
            self.arguments = copy.deepcopy(kwargs)
            return {"status": "refused", "idempotent": False,
                    "call_identity": kwargs["call_identity"], "reason": "per_icp_quota"}
        def list_ledger(self, **kwargs):
            self.reads += 1
            if unavailable:
                raise ArenaStoreUnavailable("refusal capture unavailable")
            return [{"entry_kind": "refusal", "run_id": self.arguments["run_id"],
                     "call_identity": self.arguments["call_identity"], "amount_microusd": 0,
                     "entry_doc": {"reason": "per_icp_quota", "call": self.arguments["call_doc"]}}]
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        store=RefusalStore(), capture_request=capture.RequestCapture(objects).prepare,
    )
    result = _execute(provider, context=replace(CONTEXT, reasoning_capture_policy=reasoning_capture.POLICY,
                                               reasoning_route_registry=_reviewed_registry()))
    assert result.status == 402 and store.reads == 1 and transport.sent == []
    if unavailable:
        assert not list(tmp_path.rglob("*.gz"))
    else:
        pointer = store.arguments["call_doc"]["request_capture"]
        assert capture.read_request(objects, pointer)["operation_id"] == "openrouter.chat"


def test_source_fallback_revalidates_rebuilt_request_using_same_rows(tmp_path, monkeypatch):
    from tests.lab_arena.firecrawl_content_fallback_test import URL, _generic
    from tests.lab_arena.test_lab_arena_broker import HOST_KEYS

    validations = []
    original = br._reservation_readback_matches
    def validate(store, arguments, state, **kwargs):
        validations.append((copy.deepcopy(arguments), kwargs.get("ledger_entries")))
        return original(store, arguments, state, **kwargs)
    monkeypatch.setattr(br, "_reservation_readback_matches", validate)
    objects = LocalObjectStore(tmp_path)
    provider, store, transport = make_broker(
        store=OneReadStore(), capture_request=capture.RequestCapture(objects).prepare,
        credential_for=lambda _context, name: HOST_KEYS[name],
        funding_source_for=lambda _context: "miner_key",
        transport=FakeTransport([(200, _generic())]),
    )
    result = provider._execute_once(
        replace(CONTEXT, kind="score", round_id="arena-2026-10-13"),
        operation_id="scrapingdog.scrape", parameters={"url": URL}, action_sequence=1,
        timeout_ms=30_000, provider_attempt=2, champion_credential_retry=False,
        source_content_fallback=True,
    )
    assert result.status == 200 and len(transport.sent) == 1 and store.ledger_reads == 1
    assert len(validations) == 2 and validations[0][1] is validations[1][1]
    receipt = next(iter(store.calls.values()))
    pointer = receipt["call_doc"]["request_capture"]
    assert validations[-1][0]["call_doc"]["request_capture"] == pointer
    assert capture.read_request(objects, pointer)["outbound"]["body"] == json.loads(transport.sent[0]["body"])
