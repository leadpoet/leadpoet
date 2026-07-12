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

    def execute(self, request):
        self.calls.append(dict(request))
        assert request["provider_id"] == "supabase"
        if request["method"] == "POST":
            row = json.loads(base64.b64decode(request["body_b64"]))
            self.rows.setdefault((row["utc_day"], int(row["sequence"])), row)
            return self._result(request, status=201, body=b"")
        if self.fail_reads:
            return self._failure(request)
        query = parse_qs(urlsplit(request["url"]).query)
        day = query["utc_day"][0].split("eq.", 1)[1]
        rows = [
            row
            for (row_day, _sequence), row in self.rows.items()
            if row_day == day
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
            attempt_number=0,
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
        return {
            "terminal_status": "authenticated_response",
            "http_status": status,
            "body_b64": base64.b64encode(body).decode("ascii"),
            "transport_attempt": attempt,
            "encrypted_request_artifact_id": request_artifact,
            "encrypted_artifact_id": response_artifact,
        }

    def _failure(self, request):
        return {
            "terminal_status": "transport_failure",
            "transport_attempt": {
                "attempt_hash": "sha256:" + "f" * 64,
                "terminal_status": "transport_failure",
                "response_hash": None,
            },
        }


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
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
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
    with pytest.raises(ProviderOutcomeStoreV2Error, match="readback differs"):
        store.persist(
            conflicting,
            previous_checkpoint_hash=first["checkpoint_hash"],
            job_id="job-3",
            purpose="research_lab.company_score.v2",
        )


def test_outcome_checkpoint_rejects_tampering_and_transport_failure() -> None:
    broker = _Broker()
    store = ProviderOutcomeStoreV2(broker=broker, vault=_vault())
    store.persist(
        _document(),
        previous_checkpoint_hash="",
        job_id="job",
        purpose="research_lab.company_score.v2",
    )
    row = broker.rows[("2026-07-10", 1)]
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
    with pytest.raises(ProviderOutcomeStoreV2Error, match="authenticated read failed"):
        store.load_latest(
            utc_day="2026-07-11",
            job_id="restore",
            purpose="research_lab.provider_outcome_state.v2",
        )
