from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.provider_evidence_cache_store_v2 import (
    ProviderEvidenceCacheStoreV2,
    ProviderEvidenceCacheStoreV2Error,
)
from gateway.tee.provider_evidence_v2 import (
    REQUEST_SCHEMA_VERSION,
    ProviderEvidenceAuthorityV2,
)
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)


HASH = "sha256:" + "a" * 64
MASTER_KEY = bytes(range(32))
FIXED_NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _identity(key: Ed25519PrivateKey, *, boot_hash: str = HASH):
    return {
        "boot_identity_hash": boot_hash,
        "signing_pubkey": key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex(),
    }


def _request():
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "caller_job_id": "autoresearch-v2:cache-job",
        "purpose": "research_lab.candidate_decision.v2",
        "endpoint": {
            "endpoint_id": "exa.search",
            "provider_id": "exa",
            "method": "POST",
            "path": "/search",
            "params": [
                {
                    "name": "query",
                    "type": "string",
                    "required": True,
                    "location": "body",
                    "max_length": 300,
                }
            ],
            "est_cost_microusd": 5000,
            "description": "probe",
        },
        "upstream_base_url": "https://api.exa.ai",
        "query_params": {},
        "body_params": {"query": "developer tools"},
        "live_enabled": True,
        "timeout_seconds": 60,
    }


class _Broker:
    retry_policy_hashes = {"exa": HASH, "supabase": HASH}

    def __init__(self):
        self.rows = {}
        self.calls = []
        self.fail_reads = False
        self.provider_calls = 0

    def execute(self, request):
        self.calls.append(dict(request))
        provider_id = request["provider_id"]
        if provider_id == "exa":
            self.provider_calls += 1
            return self._result(
                request,
                status=200,
                body=b'{"results":[]}',
            )
        assert provider_id == "supabase"
        if request["method"] == "POST":
            row = json.loads(base64.b64decode(request["body_b64"]))
            key = (row["utc_day"], row["request_fingerprint"])
            self.rows.setdefault(key, row)
            return self._result(request, status=201, body=b"")
        if self.fail_reads:
            return self._failure(request)
        query = parse_qs(urlsplit(request["url"]).query)
        day = query["utc_day"][0].split("eq.", 1)[1]
        fingerprint = query["request_fingerprint"][0].split("eq.", 1)[1]
        row = self.rows.get((day, fingerprint))
        body = json.dumps([row] if row is not None else [], sort_keys=True).encode()
        return self._result(request, status=200, body=body)

    def _result(self, request, *, status, body):
        ordinal = len(self.calls)
        request_artifact = "sha256:" + ("%064x" % (ordinal * 2))[-64:]
        response_artifact = "sha256:" + ("%064x" % (ordinal * 2 + 1))[-64:]
        parsed = urlsplit(request["url"])
        request_body = base64.b64decode(request["body_b64"])
        attempt = build_transport_attempt(
            request_id=("%032x" % ordinal)[-32:],
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host=parsed.hostname,
            destination_port=443,
            path_hash=sha256_bytes(parsed.path.encode()),
            nonsecret_headers_hash=sha256_json(request["headers"]),
            body_hash=sha256_bytes(request_body),
            credential_ref_hash=HASH,
            retry_policy_hash=request["retry_policy_hash"],
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
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": request_artifact,
            "encrypted_artifact_id": response_artifact,
            "transport_attempt": attempt,
        }

    def _failure(self, request):
        ordinal = len(self.calls)
        return {
            "terminal_status": "transport_failure",
            "failure_code": "timeout",
            "encrypted_request_artifact_id": (
                "sha256:" + ("%064x" % (ordinal * 2))[-64:]
            ),
            "transport_attempt": {
                "attempt_hash": "sha256:" + ("%064x" % (ordinal + 100))[-64:],
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


def _recorded_terminal(broker, key, identity):
    authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
        clock=lambda: "2026-07-10T12:00:00Z",
    )
    return authority.resolve(_request())


def test_cache_store_persists_and_reopens_after_restart() -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    verified = []
    first_store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda value: verified.append(dict(value)),
    )

    persisted = first_store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="autoresearch-v2:cache-job",
        purpose="research_lab.candidate_decision.v2",
    )
    restarted_store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault("sha256:" + "b" * 64),
        source_boot_verifier=lambda value: verified.append(dict(value)),
    )
    loaded = restarted_store.load(
        utc_day="2026-07-10",
        request_fingerprint=terminal["record"]["request_fingerprint"],
        job_id="autoresearch-v2:cache-job-2",
        purpose="research_lab.candidate_decision.v2",
    )

    assert loaded["found"] is True
    assert loaded["payload"]["body_b64"] == terminal["body_b64"]
    assert loaded["payload"]["source_record"] == terminal["record"]
    assert loaded["cache_entry_hash"] == persisted["cache_entry_hash"]
    assert broker.provider_calls == 1
    assert verified


def test_cache_store_miss_and_transport_failure_are_distinct() -> None:
    broker = _Broker()
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
    )

    missing = store.load(
        utc_day="2026-07-10",
        request_fingerprint="0" * 64,
        job_id="job",
        purpose="research_lab.candidate_decision.v2",
    )
    assert missing["found"] is False
    assert len(missing["transport_attempts"]) == 1

    broker.fail_reads = True
    with pytest.raises(ProviderEvidenceCacheStoreV2Error, match="read failed"):
        store.load(
            utc_day="2026-07-10",
            request_fingerprint="1" * 64,
            job_id="job",
            purpose="research_lab.candidate_decision.v2",
        )


def test_cache_store_rejects_tampered_ciphertext_and_source_hash() -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
    )
    store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="job",
        purpose="research_lab.candidate_decision.v2",
    )
    fingerprint = terminal["record"]["request_fingerprint"]
    row = broker.rows[("2026-07-10", fingerprint)]
    row["encrypted_cache_doc"] = {
        **row["encrypted_cache_doc"],
        "ciphertext_b64": base64.b64encode(b"tampered").decode("ascii"),
    }

    with pytest.raises(Exception):
        store.load(
            utc_day="2026-07-10",
            request_fingerprint=fingerprint,
            job_id="job-2",
            purpose="research_lab.candidate_decision.v2",
        )


def test_cache_store_never_loads_another_utc_day() -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
    )
    store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="job",
        purpose="research_lab.candidate_decision.v2",
    )

    result = store.load(
        utc_day="2026-07-11",
        request_fingerprint=terminal["record"]["request_fingerprint"],
        job_id="job-2",
        purpose="research_lab.candidate_decision.v2",
    )
    assert result["found"] is False


def test_authority_restart_replays_without_second_provider_call() -> None:
    first_key = Ed25519PrivateKey.generate()
    first_identity = _identity(first_key)
    broker = _Broker()
    first_store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
    )
    first_authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: first_identity,
        sign_digest=first_key.sign,
        clock=lambda: "2026-07-10T12:00:00Z",
        cache_store=first_store,
    )
    recorded = first_authority.resolve(_request())

    second_key = Ed25519PrivateKey.generate()
    second_identity = _identity(
        second_key,
        boot_hash="sha256:" + "b" * 64,
    )
    restarted_store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(second_identity["boot_identity_hash"]),
        source_boot_verifier=lambda _value: None,
    )
    restarted_authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: second_identity,
        sign_digest=second_key.sign,
        clock=lambda: "2026-07-10T12:05:00Z",
        cache_store=restarted_store,
    )
    replayed = restarted_authority.resolve(_request())

    assert broker.provider_calls == 1
    assert recorded["evidence"] == "recorded"
    assert replayed["evidence"] == "hit"
    assert replayed["body_b64"] == recorded["body_b64"]
    assert replayed["source_record"]["evidence"] == "restored"
    assert replayed["source_boot_identity"] == second_identity
    assert replayed["source_record"]["source_record_hash"]
