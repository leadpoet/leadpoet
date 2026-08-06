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
    CACHE_RETRY_DELAYS_SECONDS,
    CACHE_TRANSPORT_ATTEMPTS,
    ProviderEvidenceCacheStoreV2,
    ProviderEvidenceCacheStoreV2Error,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_client_v2 import _ExecutionScope
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
        self.fail_posts = 0
        self.commit_failed_posts = False
        self.read_http_failure = None
        self.read_http_failures = 0
        self.post_http_failure = None
        self.post_http_failures = 0
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
            if self.post_http_failures:
                self.post_http_failures -= 1
                status, code = self.post_http_failure
                return self._result(
                    request,
                    status=status,
                    body=json.dumps({"code": code}).encode(),
                )
            if self.fail_posts:
                self.fail_posts -= 1
                if self.commit_failed_posts:
                    self.rows.setdefault(key, row)
                return self._failure(request)
            self.rows.setdefault(key, row)
            return self._result(request, status=201, body=b"")
        if self.fail_reads is True:
            return self._failure(request)
        if self.fail_reads:
            self.fail_reads -= 1
            return self._failure(request)
        if self.read_http_failures:
            self.read_http_failures -= 1
            status, code = self.read_http_failure
            return self._result(
                request,
                status=status,
                body=json.dumps({"code": code}).encode(),
            )
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
        parsed = urlsplit(request["url"])
        request_body = base64.b64decode(request["body_b64"])
        request_artifact = "sha256:" + ("%064x" % (ordinal * 2))[-64:]
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
            terminal_status="transport_failure",
            http_status=None,
            response_hash=None,
            request_artifact_hash=request_artifact,
            response_artifact_hash=None,
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code="unexpected_eof",
            completed_at="2026-07-10T12:00:01Z",
        )
        return {
            "terminal_status": "transport_failure",
            "failure_code": "unexpected_eof",
            "encrypted_request_artifact_id": request_artifact,
            "transport_attempt": attempt,
        }


class _DeduplicatingBroker(_Broker):
    def __init__(self):
        super().__init__()
        self.records = {}

    def execute(self, request):
        key = (request["logical_operation_id"], request["attempt_number"])
        if key in self.records:
            self.calls.append(dict(request))
            return dict(self.records[key])
        result = super().execute(request)
        self.records[key] = dict(result)
        return result


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
        sleeper=lambda _delay: None,
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


def test_cache_store_retries_transient_read_under_same_operation() -> None:
    broker = _Broker()
    broker.fail_reads = 1
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    missing = store.load(
        utc_day="2026-07-10",
        request_fingerprint="3" * 64,
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    assert missing["found"] is False
    assert [
        item["attempt_number"] for item in missing["transport_attempts"]
    ] == [0, 1]
    assert [
        item["terminal_status"] for item in missing["transport_attempts"]
    ] == ["transport_failure", "authenticated_response"]
    assert len(
        {item["logical_operation_id"] for item in missing["transport_attempts"]}
    ) == 1


@pytest.mark.parametrize(
    ("http_status", "code"),
    (
        (503, "PGRST002"),
        (504, "PGRST003"),
    ),
)
def test_cache_store_retries_authenticated_transient_read(
    http_status: int,
    code: str,
) -> None:
    broker = _Broker()
    broker.read_http_failure = (http_status, code)
    broker.read_http_failures = 1
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    missing = store.load(
        utc_day="2026-07-10",
        request_fingerprint="4" * 64,
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    assert missing["found"] is False
    assert [call["attempt_number"] for call in broker.calls] == [0, 1]
    assert [
        attempt["http_status"] for attempt in missing["transport_attempts"]
    ] == [http_status, 200]


@pytest.mark.parametrize("failure_kind", ("transport", "authenticated"))
def test_cache_store_recovers_read_across_extended_transient_window(
    failure_kind: str,
) -> None:
    broker = _Broker()
    if failure_kind == "transport":
        broker.fail_reads = CACHE_TRANSPORT_ATTEMPTS - 1
    else:
        broker.read_http_failure = (503, "PGRST002")
        broker.read_http_failures = CACHE_TRANSPORT_ATTEMPTS - 1
    sleeps = []
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=sleeps.append,
    )

    missing = store.load(
        utc_day="2026-07-10",
        request_fingerprint="7" * 64,
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    assert missing["found"] is False
    assert [call["attempt_number"] for call in broker.calls] == list(
        range(CACHE_TRANSPORT_ATTEMPTS)
    )
    assert sleeps == list(CACHE_RETRY_DELAYS_SECONDS[1:])
    assert missing["transport_attempts"][-1]["terminal_status"] == (
        "authenticated_response"
    )


@pytest.mark.parametrize("committed_before_eof", (False, True))
def test_cache_store_retries_transient_insert_under_same_operation(
    committed_before_eof: bool,
) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    broker.fail_posts = 1
    broker.commit_failed_posts = committed_before_eof
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    persisted = store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    insert_calls = [
        item
        for item in broker.calls
        if item["provider_id"] == "supabase" and item["method"] == "POST"
    ]
    assert [item["attempt_number"] for item in insert_calls] == [0, 1]
    assert len({item["logical_operation_id"] for item in insert_calls}) == 1
    insert_attempts = [
        item
        for item in persisted["transport_attempts"]
        if item["logical_operation_id"].endswith(":insert")
    ]
    assert [item["terminal_status"] for item in insert_attempts] == [
        "transport_failure",
        "authenticated_response",
    ]
    scope = _ExecutionScope(
        job_id="job",
        purpose="research_lab.private_model_run.v2",
        logical_operation_id="model-run",
        retry_policy_hashes={"supabase": HASH},
        default_timeout_ms=45_000,
        terminal_sink=None,
    )
    for attempt in persisted["transport_attempts"]:
        scope.record_intent(
            attempt["logical_operation_id"],
            attempt["attempt_number"],
        )
        scope.record_terminal(
            attempt["logical_operation_id"],
            attempt["attempt_number"],
            attempt["terminal_status"],
        )
    scope.assert_accepted_result_is_complete()


@pytest.mark.parametrize(
    ("http_status", "code"),
    (
        (503, "PGRST002"),
        (504, "PGRST003"),
    ),
)
def test_cache_store_retries_authenticated_transient_insert(
    http_status: int,
    code: str,
) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    broker.post_http_failure = (http_status, code)
    broker.post_http_failures = 1
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    persisted = store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    insert_attempts = [
        attempt
        for attempt in persisted["transport_attempts"]
        if attempt["logical_operation_id"].endswith(":insert")
    ]
    assert [attempt["attempt_number"] for attempt in insert_attempts] == [0, 1]
    assert [attempt["http_status"] for attempt in insert_attempts] == [
        http_status,
        201,
    ]


@pytest.mark.parametrize("failure_kind", ("transport", "authenticated"))
def test_cache_store_recovers_insert_across_extended_transient_window(
    failure_kind: str,
) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    if failure_kind == "transport":
        broker.fail_posts = CACHE_TRANSPORT_ATTEMPTS - 1
    else:
        broker.post_http_failure = (503, "PGRST002")
        broker.post_http_failures = CACHE_TRANSPORT_ATTEMPTS - 1
    sleeps = []
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=sleeps.append,
    )

    persisted = store.persist_recorded(
        terminal,
        utc_day="2026-07-10",
        job_id="job",
        purpose="research_lab.private_model_run.v2",
    )

    insert_calls = [
        item
        for item in broker.calls
        if item["provider_id"] == "supabase" and item["method"] == "POST"
    ]
    assert [item["attempt_number"] for item in insert_calls] == list(
        range(CACHE_TRANSPORT_ATTEMPTS)
    )
    assert sleeps == list(CACHE_RETRY_DELAYS_SECONDS[1:])
    assert persisted["cache_entry_hash"]
    assert len(broker.rows) == 1


def test_cache_store_authenticated_transient_exhaustion_remains_fail_closed() -> None:
    broker = _Broker()
    broker.read_http_failure = (503, "PGRST002")
    broker.read_http_failures = CACHE_TRANSPORT_ATTEMPTS
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(ProviderEvidenceCacheStoreV2Error, match="read failed"):
        store.load(
            utc_day="2026-07-10",
            request_fingerprint="5" * 64,
            job_id="job",
            purpose="research_lab.private_model_run.v2",
        )

    assert [call["attempt_number"] for call in broker.calls] == list(
        range(CACHE_TRANSPORT_ATTEMPTS)
    )


def test_cache_store_nontransient_response_is_not_retried() -> None:
    broker = _Broker()
    broker.read_http_failure = (401, "PGRST301")
    broker.read_http_failures = 1
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(ProviderEvidenceCacheStoreV2Error, match="read failed"):
        store.load(
            utc_day="2026-07-10",
            request_fingerprint="6" * 64,
            job_id="job",
            purpose="research_lab.private_model_run.v2",
        )

    assert len(broker.calls) == 1


def test_cache_store_exhausted_insert_remains_fail_closed() -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    terminal = _recorded_terminal(broker, key, identity)
    broker.fail_posts = CACHE_TRANSPORT_ATTEMPTS
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
        sleeper=lambda _delay: None,
    )

    with pytest.raises(
        ProviderEvidenceCacheStoreV2Error,
        match="authenticated insert failed",
    ):
        store.persist_recorded(
            terminal,
            utc_day="2026-07-10",
            job_id="job",
            purpose="research_lab.private_model_run.v2",
        )

    insert_calls = [
        item
        for item in broker.calls
        if item["provider_id"] == "supabase" and item["method"] == "POST"
    ]
    assert [item["attempt_number"] for item in insert_calls] == list(
        range(CACHE_TRANSPORT_ATTEMPTS)
    )
    assert not any(item["method"] == "GET" for item in broker.calls)


def test_repeated_cache_reads_have_request_bound_transport_identities() -> None:
    broker = _DeduplicatingBroker()
    store = ProviderEvidenceCacheStoreV2(
        broker=broker,
        vault=_vault(),
        source_boot_verifier=lambda _value: None,
    )
    context = ExecutionContextV2(
        job_id="job",
        purpose="research_lab.private_model_run.v2",
        epoch_id=1,
        provider_credential_ref_hashes={"supabase": HASH},
    )

    for attempt_number in (0, 1):
        result = store.load(
            utc_day="2026-07-10",
            request_fingerprint="2" * 64,
            job_id=context.job_id,
            purpose=context.purpose,
            attempt_number=attempt_number,
        )
        assert result["found"] is False
        for attempt in result["transport_attempts"]:
            context.record_transport(attempt)

    assert [item["attempt_number"] for item in context.transport_attempts] == [
        0,
        CACHE_TRANSPORT_ATTEMPTS,
    ]
    assert len({item["attempt_hash"] for item in context.transport_attempts}) == 2


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
