from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_outcome_store_v2 import (
    ProviderOutcomeStoreV2,
    ProviderOutcomeStoreV2Error,
)
from gateway.tee.provider_outcome_v2 import ProviderOutcomeLedgerV2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)


HASH = "sha256:" + "a" * 64
MASTER_KEY = bytes(range(32))
FIXED_NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


class _Broker:
    retry_policy_hashes = {"supabase": HASH}

    def __init__(self) -> None:
        self.rows = {}
        self.calls = []
        self.fail_reads = False
        self.fail_committed_appends = 0
        self.busy_appends = 0
        self.append_http_failure = None
        self.contention_contract = "current"
        self.tamper_next_response_hash = False

    def execute(self, request):
        self.calls.append(dict(request))
        assert request["provider_id"] == "supabase"
        if request["method"] == "POST":
            payload = json.loads(base64.b64decode(request["body_b64"]))
            row = payload["checkpoint_row"]
            if self.append_http_failure is not None:
                status, code = self.append_http_failure
                return self._result(
                    request,
                    status=status,
                    body=json.dumps({"code": code}).encode(),
                )
            if self.busy_appends > 0:
                self.busy_appends -= 1
                return self._result(
                    request,
                    status=200,
                    body=json.dumps(
                        {
                            "checkpoint_hash": row["checkpoint_hash"],
                            "status": "busy",
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                )
            lineage = (
                row["artifact_master_key_ref_hash"],
                row["utc_day"],
            )
            existing = next(
                (
                    item
                    for item in self.rows.values()
                    if item["checkpoint_hash"] == row["checkpoint_hash"]
                ),
                None,
            )
            if existing is not None:
                if existing != row:
                    return self._result(
                        request,
                        status=409,
                        body=b'{"code":"23505"}',
                    )
                return self._result(
                    request,
                    status=200,
                    body=json.dumps(
                        {
                            "status": "existing",
                            "checkpoint_hash": row["checkpoint_hash"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                )
            current = sorted(
                (
                    item
                    for (key_hash, day, _sequence), item in self.rows.items()
                    if (key_hash, day) == lineage
                ),
                key=lambda item: int(item["sequence"]),
                reverse=True,
            )
            expected_sequence = int(current[0]["sequence"]) + 1 if current else 1
            expected_previous = current[0]["checkpoint_hash"] if current else ""
            if (
                int(row["sequence"]) != expected_sequence
                or row["previous_checkpoint_hash"] != expected_previous
            ):
                if self.contention_contract != "legacy":
                    response = {
                        "status": "conflict",
                        "checkpoint_hash": row["checkpoint_hash"],
                    }
                    if self.contention_contract == "embedded":
                        response["head_checkpoint_row"] = (
                            current[0] if current else None
                        )
                    return self._result(
                        request,
                        status=200,
                        body=json.dumps(
                            response,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode(),
                    )
                return self._result(
                    request,
                    status=409,
                    body=b'{"code":"40001"}',
                )
            self.rows[
                (
                    row["artifact_master_key_ref_hash"],
                    row["utc_day"],
                    int(row["sequence"]),
                )
            ] = row
            if self.fail_committed_appends > 0:
                self.fail_committed_appends -= 1
                return self._failure(request, failure_code="unexpected_eof")
            return self._result(
                request,
                status=200,
                body=json.dumps(
                    {
                        "status": "inserted",
                        "checkpoint_hash": row["checkpoint_hash"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )
        if self.fail_reads:
            return self._failure(request)
        query = parse_qs(urlsplit(request["url"]).query)
        day = query["utc_day"][0].split("eq.", 1)[1]
        key_hash = query["artifact_master_key_ref_hash"][0].split("eq.", 1)[1]
        rows = [
            row
            for (row_key_hash, row_day, _sequence), row in self.rows.items()
            if row_key_hash == key_hash and row_day == day
        ]
        if "sequence" in query:
            sequence = int(query["sequence"][0].split("eq.", 1)[1])
            rows = [row for row in rows if int(row["sequence"]) == sequence]
        if query.get("order") == ["sequence.desc"]:
            rows.sort(key=lambda row: int(row["sequence"]), reverse=True)
        rows = rows[: int(query.get("limit", ["2"])[0])]
        return self._result(
            request,
            status=200,
            body=json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(),
        )

    def _result(self, request, *, status, body):
        ordinal = len(self.calls)
        parsed = urlsplit(request["url"])
        request_body = base64.b64decode(request["body_b64"])
        request_artifact = "sha256:" + ("%064x" % (ordinal * 2))[-64:]
        response_artifact = "sha256:" + ("%064x" % (ordinal * 2 + 1))[-64:]
        attempt = build_transport_attempt(
            request_id=("%032x" % ordinal)[-32:],
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="supabase",
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host=parsed.hostname,
            destination_port=443,
            path_hash=sha256_bytes(parsed.path.encode()),
            nonsecret_headers_hash=sha256_json(request["headers"]),
            body_hash=sha256_bytes(request_body),
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T12:00:00Z",
            terminal_status="authenticated_response",
            http_status=status,
            response_hash=sha256_bytes(body),
            request_artifact_hash=request_artifact,
            response_artifact_hash=sha256_bytes(body),
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T12:00:01Z",
        )
        result = {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": attempt,
            "encrypted_request_artifact_id": request_artifact,
            "encrypted_artifact_id": response_artifact,
        }
        if self.tamper_next_response_hash:
            self.tamper_next_response_hash = False
            result["transport_attempt"] = {
                **result["transport_attempt"],
                "response_hash": "sha256:" + "f" * 64,
            }
        return result

    def _failure(self, request, *, failure_code="timeout"):
        return {
            "terminal_status": "transport_failure",
            "failure_stage": "provider_transport",
            "failure_error_type": "ReadError",
            "transport_attempt": {
                "attempt_hash": "sha256:" + "f" * 64,
                "terminal_status": "transport_failure",
                "response_hash": None,
                "failure_code": failure_code,
            },
        }


class _TerminalCachingBroker(_Broker):
    """Match the production broker's logical-operation replay contract."""

    def __init__(self) -> None:
        super().__init__()
        self.terminals = {}

    def execute(self, request):
        key = (
            str(request["logical_operation_id"]),
            int(request["attempt_number"]),
        )
        fingerprint = sha256_json(request)
        existing = self.terminals.get(key)
        if existing is not None:
            if existing["fingerprint"] != fingerprint:
                raise RuntimeError(
                    "logical provider attempt was reused with different request"
                )
            return existing["result"]
        result = super().execute(request)
        self.terminals[key] = {
            "fingerprint": fingerprint,
            "result": result,
        }
        return result


def _vault(boot_hash=HASH):
    return EncryptedArtifactVaultV2(
        master_key=MASTER_KEY,
        boot_identity_hash=boot_hash,
        retention_days=30,
        clock=lambda: FIXED_NOW,
    )


def _document(clock_value="2026-07-10T12:00:00Z"):
    ledger = ProviderOutcomeLedgerV2(clock=lambda: clock_value)
    return ledger.record(
        provider_id="exa",
        endpoint_class="/search",
        evidence="recorded",
        status=200,
        live_call=True,
        cost_event={
            "billable": True,
            "cost_usd": "0.005",
            "cost_source": "exa_cost_dollars",
        },
    )


def test_outcome_checkpoint_reopens_after_coordinator_restart() -> None:
    broker = _Broker()
    first = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
    persisted = first.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="score-job-1",
        purpose="research_lab.company_score.v2",
    )

    restarted = ProviderOutcomeStoreV2(
        broker=broker,
        vault=_vault("sha256:" + "b" * 64),
    )
    restored = restarted.load_latest(
        utc_day="2026-07-10",
        job_id="provider-outcome-restore-2026-07-10",
        purpose="research_lab.provider_outcome_state.v2",
    )

    assert restored["found"] is True
    assert restored["checkpoint_hash"] == persisted["checkpoint_hash"]
    assert restored["state_document"]["sequence"] == 1
    assert restored["state_document"]["providers"]["exa"]["call_count"] == 1
    assert len(restored["transport_attempts"]) == 1


def test_outcome_checkpoint_chain_is_monotonic_and_collision_fails_closed() -> None:
    broker = _Broker()
    vault = _vault()
    store = ProviderOutcomeStoreV2(broker=broker, vault=vault)
    document = _document()
    first = store.persist(
        document,
        previous_checkpoint_hash="",
        job_id="job-1",
        purpose="research_lab.company_score.v2",
    )
    ledger = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:00Z",
        initial_document=document,
    )
    second_document = ledger.record(
        provider_id="exa",
        endpoint_class="/search",
        evidence="hit",
        status=200,
        live_call=False,
        cost_event={},
    )
    second = store.persist(
        second_document,
        previous_checkpoint_hash=first["checkpoint_hash"],
        job_id="job-2",
        purpose="research_lab.company_score.v2",
    )
    latest = store.load_latest(
        utc_day="2026-07-10",
        job_id="restore",
        purpose="research_lab.provider_outcome_state.v2",
    )
    assert latest["checkpoint_hash"] == second["checkpoint_hash"]
    assert latest["state_document"]["sequence"] == 2

    conflicting = {**second_document, "generated_at": "2026-07-10T12:00:01Z"}
    conflicting["generated_at_epoch"] = 1783684801.0
    from gateway.research_lab.provider_outcome_digest import _sidecar_document_hash

    conflicting["document_hash"] = _sidecar_document_hash(conflicting)
    conflict = store.persist(
        conflicting,
        previous_checkpoint_hash=first["checkpoint_hash"],
        job_id="job-3",
        purpose="research_lab.company_score.v2",
    )
    assert conflict["status"] == "conflict"
    assert len(conflict["transport_attempts"]) == 1
    assert len(broker.calls) == 6
    assert vault.job_artifacts(
        job_id="job-3",
        purpose="research_lab.company_score.v2",
    ) == ()


def test_sequential_checkpoint_readbacks_have_request_bound_operation_ids() -> None:
    broker = _TerminalCachingBroker()
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
    first_document = _document()
    first = store.persist(
        first_document,
        previous_checkpoint_hash="",
        job_id="shared-provider-preflight",
        purpose="research_lab.provider_preflight.v2",
    )
    ledger = ProviderOutcomeLedgerV2(
        clock=lambda: "2026-07-10T12:00:01Z",
        initial_document=first_document,
    )
    second_document = ledger.record(
        provider_id="scrapingdog",
        endpoint_class="/account",
        evidence="recorded",
        status=200,
        live_call=True,
        cost_event={},
    )

    second = store.persist(
        second_document,
        previous_checkpoint_hash=first["checkpoint_hash"],
        job_id="shared-provider-preflight",
        purpose="research_lab.provider_preflight.v2",
    )

    assert second["status"] == "persisted"
    read_calls = [call for call in broker.calls if call["method"] == "GET"]
    assert len(read_calls) == 2
    assert len({call["logical_operation_id"] for call in read_calls}) == 2
    assert "sequence=eq.1" in read_calls[0]["url"]
    assert "sequence=eq.2" in read_calls[1]["url"]


def test_outcome_checkpoint_restores_authenticated_embedded_conflict_head() -> None:
    broker = _Broker()
    broker.contention_contract = "embedded"
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
    first = store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job-first",
        purpose="research_lab.company_score.v2",
    )
    calls_before_conflict = len(broker.calls)

    conflict = store.persist(
        _document("2026-07-10T12:00:01Z"),
        previous_checkpoint_hash=first["checkpoint_hash"],
        job_id="job-conflict",
        purpose="research_lab.company_score.v2",
    )

    assert conflict["status"] == "conflict"
    assert conflict["head_checkpoint_hash"] == first["checkpoint_hash"]
    assert conflict["head_state_document"]["sequence"] == 1
    assert len(broker.calls) == calls_before_conflict + 1
    assert broker.calls[-1]["method"] == "POST"


def test_outcome_checkpoint_rejects_embedded_head_from_another_key_lineage() -> None:
    broker = _Broker()
    broker.contention_contract = "embedded"
    vault = _vault()
    store = ProviderOutcomeStoreV2(broker=broker, vault=vault)
    first = store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job-first",
        purpose="research_lab.company_score.v2",
    )
    stored = broker.rows[(vault.master_key_ref_hash, "2026-07-10", 1)]
    stored["artifact_master_key_ref_hash"] = "sha256:" + "b" * 64
    calls_before_conflict = len(broker.calls)

    with pytest.raises(
        ProviderOutcomeStoreV2Error,
        match="provider outcome checkpoint key lineage differs",
    ):
        store.persist(
            _document("2026-07-10T12:00:01Z"),
            previous_checkpoint_hash=first["checkpoint_hash"],
            job_id="job-conflict",
            purpose="research_lab.company_score.v2",
        )

    assert len(broker.calls) == calls_before_conflict + 1
    assert broker.calls[-1]["method"] == "POST"


def test_outcome_checkpoint_accepts_ambiguous_committed_append() -> None:
    broker = _Broker()
    broker.fail_committed_appends = 1
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())

    persisted = store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job",
        purpose="research_lab.company_score.v2",
    )

    assert persisted["status"] == "persisted"
    assert persisted["transport_attempts"][0]["terminal_status"] == "transport_failure"
    assert persisted["transport_attempts"][0]["failure_code"] == "unexpected_eof"
    assert persisted["transport_attempts"][1]["terminal_status"] == "authenticated_response"


def test_outcome_checkpoint_busy_append_is_retryable_without_readback() -> None:
    broker = _Broker()
    broker.busy_appends = 1
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())

    busy = store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job",
        purpose="research_lab.company_score.v2",
        attempt_number=7,
    )

    assert busy["status"] == "busy"
    assert len(broker.calls) == 1
    assert broker.calls[0]["attempt_number"] == 7


@pytest.mark.parametrize(
    ("http_status", "code"),
    (
        (503, "PGRST002"),
        (504, "PGRST003"),
    ),
)
def test_outcome_checkpoint_authenticated_append_failure_does_not_read_back(
    http_status: int,
    code: str,
) -> None:
    broker = _Broker()
    broker.append_http_failure = (http_status, code)
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())

    with pytest.raises(
        ProviderOutcomeStoreV2Error,
        match=(
            "provider outcome checkpoint authenticated append failed "
            rf"\(http_status={http_status} code={code}\)"
        ),
    ):
        store.persist(
            _document(),
            previous_checkpoint_hash="",
            job_id="job",
            purpose="research_lab.company_score.v2",
        )

    assert [call["method"] for call in broker.calls] == ["POST"]


@pytest.mark.parametrize(
    ("message", "expected_status"),
    (
        ("provider outcome checkpoint append is busy", "busy"),
        ("provider outcome checkpoint does not extend the current head", "conflict"),
    ),
)
def test_outcome_checkpoint_recognizes_legacy_sqlstate_responses(
    message: str,
    expected_status: str,
) -> None:
    body = json.dumps({"code": "40001", "message": message}).encode()
    result = {
        "terminal_status": "authenticated_response",
        "http_status": 409,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "transport_attempt": {"response_hash": sha256_bytes(body)},
    }

    assert ProviderOutcomeStoreV2._append_result(
        result,
        expected_checkpoint_hash=HASH,
    ) == {"status": expected_status}


def test_outcome_checkpoint_rejects_uncommitted_append_response_bytes() -> None:
    broker = _Broker()
    broker.tamper_next_response_hash = True
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())

    with pytest.raises(
        ProviderOutcomeStoreV2Error,
        match="provider outcome checkpoint append response commitments differ",
    ):
        store.persist(
            _document(),
            previous_checkpoint_hash="",
            job_id="job",
            purpose="research_lab.company_score.v2",
        )

    assert [call["method"] for call in broker.calls] == ["POST"]


def test_outcome_checkpoint_rejects_tampering_and_transport_failure() -> None:
    broker = _Broker()
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
    store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job",
        purpose="research_lab.company_score.v2",
    )
    row = broker.rows[(_vault().master_key_ref_hash, "2026-07-10", 1)]
    row["encrypted_checkpoint_doc"] = {
        **row["encrypted_checkpoint_doc"],
        "ciphertext_b64": base64.b64encode(b"tampered").decode("ascii"),
    }
    with pytest.raises(Exception):
        store.load_latest(
            utc_day="2026-07-10",
            job_id="restore",
            purpose="research_lab.provider_outcome_state.v2",
        )

    broker.fail_reads = True
    with pytest.raises(
        ProviderOutcomeStoreV2Error,
        match=(
            "authenticated read failed "
            r"\(terminal_status=transport_failure "
            r"http_status=0 failure_code=timeout "
            r"failure_stage=provider_transport "
            r"failure_error_type=ReadError\)"
        ),
    ):
        store.load_latest(
            utc_day="2026-07-11",
            job_id="restore",
            purpose="research_lab.provider_outcome_state.v2",
        )
