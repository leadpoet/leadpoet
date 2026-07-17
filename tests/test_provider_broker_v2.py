from __future__ import annotations

import base64
import threading
import time

import pytest

from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
    ProviderBrokerV2Error,
    _extract_tls_metadata,
    credential_reference_hash,
    credential_value_hash,
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
    measured_retry_policy_hashes,
    provider_registry_document,
    provider_registry_hash,
)
from leadpoet_canonical.attested_v2 import validate_transport_attempt
from leadpoet_canonical.attested_v2 import sha256_bytes


HASH = "sha256:" + "a" * 64
NOW = "2026-07-10T20:00:00Z"


def test_tls_metadata_supports_python39_positional_only_peer_certificate():
    class PositionalOnlyTLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return PositionalOnlyTLS()

    class Response:
        extensions = {"network_stream": Stream()}

    assert _extract_tls_metadata(Response()) == (
        sha256_bytes(b"peer-certificate"),
        "TLSv1.3",
    )


def test_provider_registry_hash_binds_measured_https_routes():
    document = provider_registry_document()
    assert document["transport"] == {
        "scheme": "https",
        "port": 443,
        "tls_termination": "gateway_coordinator_enclave",
        "plaintext_external_http": False,
    }
    assert set(document["routes"]) == set(BUILTIN_PROVIDER_ROUTES)
    assert document["routes"]["openrouter"]["hosts"] == ["openrouter.ai"]
    assert document["routes"]["dns"]["hosts"] == ["cloudflare-dns.com"]
    assert document["routes"]["rdap"]["hosts"] == ["rdap.org"]
    assert document["routes"]["bittensor_chain"]["hosts"] == [
        "entrypoint-finney.opentensor.ai"
    ]
    assert document["routes"]["bittensor_archive"] == {
        "hosts": ["archive.chain.opentensor.ai"],
        "path_prefixes": ["/"],
        "credential_slot": "",
        "credential_location": "none",
        "credential_name": "",
        "credential_prefix": "",
        "credential_header_aliases": [],
        "allowed_methods": ["POST"],
    }
    assert document["routes"]["arweave"] == {
        "hosts": ["arweave.net"],
        "path_prefixes": ["/"],
        "credential_slot": "",
        "credential_location": "none",
        "credential_name": "",
        "credential_prefix": "",
        "credential_header_aliases": [],
        "allowed_methods": ["GET"],
    }
    assert document["routes"]["supabase"] == {
        "hosts": ["qplwoislplkcegvdmbim.supabase.co"],
        "path_prefixes": ["/rest/v1/"],
        "credential_slot": "supabase_service_role",
        "credential_location": "header",
        "credential_name": "Authorization",
        "credential_prefix": "Bearer ",
        "credential_header_aliases": [{"name": "apikey", "prefix": ""}],
    }
    assert provider_registry_hash().startswith("sha256:")
    assert expected_provider_credential_slots() == (
        "deepline",
        "exa",
        "openrouter",
        "scrapingdog",
        "supabase_service_role",
        "truelist",
    )
    assert set(measured_retry_policy_hashes(HASH)) == set(BUILTIN_PROVIDER_ROUTES)
    assert set(expected_job_credential_slot_ref_hashes()) == {
        "egress_proxy",
        "openrouter_management",
    }


class FakeTransport:
    def __init__(self, *, error=None, delay=0.0):
        self.calls = []
        self.error = error
        self.delay = delay

    def __call__(self, **request):
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {
            "http_status": 503,
            "headers": {"content-type": "application/json", "set-cookie": "ignored"},
            "body": b'{"error":"provider unavailable"}',
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }


def _broker(transport):
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: NOW,
    )
    broker.provision_credentials(credentials)
    return broker


def _request(**overrides):
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": "operation-1",
        "job_id": "job-1",
        "purpose": "research_lab.provider_evidence.v2",
        "provider_id": "openrouter",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"content-type": "application/json", "x-title": "Leadpoet"},
        "body_b64": base64.b64encode(b'{"model":"model-1"}').decode("ascii"),
        "timeout_ms": 30000,
        "retry_policy_hash": HASH,
    }
    request.update(overrides)
    return request


def test_authenticated_provider_error_is_recorded_only_with_tls_evidence():
    transport = FakeTransport()
    result = _broker(transport).execute(_request())
    assert result["terminal_status"] == "authenticated_response"
    assert result["http_status"] == 503
    attempt = result["transport_attempt"]
    validate_transport_attempt(attempt)
    assert attempt["terminal_status"] == "authenticated_response"
    assert attempt["http_status"] == 503
    assert attempt["tls_protocol"] == "TLSv1.3"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer openrouter-secret"


def test_parent_or_network_error_cannot_masquerade_as_provider_status():
    transport = FakeTransport(error=RuntimeError("proxy generated 502"))
    result = _broker(transport).execute(_request())
    assert result == {
        "terminal_status": "transport_failure",
        "failure_code": "proxy_failure",
        "encrypted_request_artifact_id": result[
            "encrypted_request_artifact_id"
        ],
        "transport_attempt": result["transport_attempt"],
        "evidence_artifact_hashes": result["evidence_artifact_hashes"],
    }
    attempt = result["transport_attempt"]
    validate_transport_attempt(attempt)
    assert attempt["http_status"] is None
    assert attempt["response_hash"] is None
    assert attempt["request_artifact_hash"].startswith("sha256:")
    assert attempt["failure_code"] == "proxy_failure"


def test_one_logical_attempt_is_executed_and_charged_once_under_concurrency():
    transport = FakeTransport(delay=0.05)
    broker = _broker(transport)
    results = []
    errors = []

    def _call():
        try:
            results.append(broker.execute(_request()))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(transport.calls) == 1
    assert len(results) == 5
    assert {item["transport_attempt"]["attempt_hash"] for item in results} == {
        results[0]["transport_attempt"]["attempt_hash"]
    }


def test_attempt_id_cannot_be_reused_for_different_request():
    broker = _broker(FakeTransport())
    broker.execute(_request())
    changed_body = base64.b64encode(b'{"model":"different"}').decode("ascii")
    with pytest.raises(ProviderBrokerV2Error, match="different request"):
        broker.execute(_request(body_b64=changed_body))


def test_plaintext_unmeasured_destination_and_runner_credentials_fail_closed():
    broker = _broker(FakeTransport())
    with pytest.raises(ProviderBrokerV2Error, match="HTTPS"):
        broker.execute(_request(url="http://openrouter.ai/api/v1/chat/completions"))
    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(_request(url="https://example.com/api/v1/chat/completions"))
    with pytest.raises(ProviderBrokerV2Error, match="credential header"):
        broker.execute(_request(headers={"Authorization": "Bearer host-value"}))


def test_historical_settlement_routes_are_host_and_method_bound():
    broker = _broker(FakeTransport())
    broker.execute(
        _request(
            provider_id="bittensor_archive",
            method="POST",
            url="https://archive.chain.opentensor.ai/",
        )
    )
    broker.execute(
        _request(
            logical_operation_id="operation-arweave",
            provider_id="arweave",
            method="GET",
            url="https://arweave.net/" + "A" * 43,
            body_b64=base64.b64encode(b"").decode("ascii"),
        )
    )
    with pytest.raises(ProviderBrokerV2Error, match="method"):
        broker.execute(
            _request(
                logical_operation_id="operation-archive-get",
                provider_id="bittensor_archive",
                method="GET",
                url="https://archive.chain.opentensor.ai/",
            )
        )
    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(
            _request(
                logical_operation_id="operation-arweave-host",
                provider_id="arweave",
                method="GET",
                url="https://example.com/" + "A" * 43,
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )


def test_scrapingdog_key_is_injected_only_inside_coordinator_query():
    transport = FakeTransport()
    broker = _broker(transport)
    broker.execute(
        _request(
            provider_id="scrapingdog",
            url="https://api.scrapingdog.com/scrape?dynamic=true",
        )
    )
    assert "api_key=scrapingdog-secret" in transport.calls[0]["url"]
    attempt = broker.execute(
        _request(
            logical_operation_id="operation-2",
            provider_id="scrapingdog",
            url="https://api.scrapingdog.com/scrape?dynamic=true",
        )
    )["transport_attempt"]
    assert "scrapingdog-secret" not in str(attempt)


def test_source_add_provenance_static_route_uses_bounded_hash_only_artifacts():
    transport = FakeTransport()
    artifact_bodies = []
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }

    def sink(body, **_kwargs):
        artifact_bodies.append(bytes(body))
        return {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        }

    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=sink,
        clock=lambda: NOW,
    )
    broker.provision_credentials(credentials)
    result = broker.execute(
        _request(
            purpose="research_lab.source_add_provenance.v2",
            provider_id="scrapingdog",
            method="GET",
            url=(
                "https://api.scrapingdog.com/scrape?"
                "url=https%3A%2F%2Fdocs.example.com&dynamic=false"
            ),
            body_b64="",
            max_response_bytes=240_000,
            artifact_mode="hash_only",
        )
    )

    assert transport.calls[0]["max_response_bytes"] == 240_000
    assert b'"error":"provider unavailable"' not in artifact_bodies
    assert all(b"docs.example.com" not in body for body in artifact_bodies)
    assert result["transport_attempt"]["response_hash"].startswith("sha256:")


def test_unrelated_static_route_cannot_request_hash_only_artifacts():
    broker = _broker(FakeTransport())
    with pytest.raises(ProviderBrokerV2Error, match="measured SOURCE_ADD route"):
        broker.execute(
            _request(
                max_response_bytes=240_000,
                artifact_mode="hash_only",
            )
        )


def test_supabase_service_role_is_injected_only_for_measured_project():
    transport = FakeTransport()
    broker = _broker(transport)
    result = broker.execute(
        _request(
            provider_id="supabase",
            method="GET",
            url=(
                "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                "banned_hotkeys?select=hotkey"
            ),
            body_b64=base64.b64encode(b"").decode("ascii"),
        )
    )
    outbound = transport.calls[0]
    assert outbound["headers"]["Authorization"] == (
        "Bearer supabase-service-role-secret"
    )
    assert outbound["headers"]["apikey"] == "supabase-service-role-secret"
    assert "supabase-service-role-secret" not in str(result)

    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(
            _request(
                logical_operation_id="operation-wrong-project",
                provider_id="supabase",
                method="GET",
                url="https://attacker.supabase.co/rest/v1/banned_hotkeys?select=hotkey",
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )
    with pytest.raises(ProviderBrokerV2Error, match="credential header"):
        broker.execute(
            _request(
                logical_operation_id="operation-host-key",
                provider_id="supabase",
                method="GET",
                url=(
                    "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                    "banned_hotkeys?select=hotkey"
                ),
                headers={"apikey": "host-supplied"},
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )


def test_kms_unwrapped_slots_are_provisioned_individually_and_immutably():
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=FakeTransport(),
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: NOW,
    )
    status = broker.provision_credential(
        slot="openrouter",
        credential=credentials["openrouter"],
    )
    assert status["status"] == "provisioning"
    assert status["credential_slots"] == ["openrouter"]
    assert broker.provision_credential(
        slot="openrouter",
        credential=credentials["openrouter"],
    )["status"] == "provisioning"
    with pytest.raises(ProviderBrokerV2Error, match="hash mismatch"):
        broker.provision_credential(slot="openrouter", credential="wrong-secret")


def test_job_credential_lease_overrides_boot_key_for_only_that_job():
    transport = FakeTransport()
    broker = _broker(transport)
    miner_key = "miner-owned-openrouter-key"
    lease = broker.provision_job_credential(
        job_id="job-1",
        slot="openrouter",
        credential=miner_key,
        credential_value_hash_expected=credential_value_hash(miner_key),
    )
    assert lease["status"] == "ready"
    result = broker.execute(_request())
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer " + miner_key
    assert result["transport_attempt"]["credential_ref_hash"] == credential_value_hash(
        miner_key
    )

    released = broker.release_job_credentials("job-1")
    assert released["released_slot_count"] == 1
    broker.execute(
        _request(
            job_id="job-2",
            logical_operation_id="operation-2",
        )
    )
    assert transport.calls[1]["headers"]["Authorization"] == "Bearer openrouter-secret"


def test_job_scoped_tls_proxy_is_bound_to_transport_receipt():
    transport = FakeTransport()
    broker = _broker(transport)
    proxy_url = "https://worker-7:password@proxy.example.com:443"
    proxy_hash = credential_value_hash(proxy_url)
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=proxy_hash,
    )

    result = broker.execute(_request())

    assert transport.calls[0]["upstream_proxy_url"] == proxy_url
    assert result["transport_attempt"]["egress_proxy_ref_hash"] == proxy_hash
    validate_transport_attempt(result["transport_attempt"])
    assert proxy_url not in str(result)


@pytest.mark.parametrize(
    "proxy_url",
    (
        "http://proxy.example.com:443",
        "https://proxy.example.com:8443",
        "https://user@proxy.example.com:443",
    ),
)
def test_job_scoped_proxy_rejects_non_tls_or_incomplete_routes(proxy_url):
    broker = _broker(FakeTransport())
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=credential_value_hash(proxy_url),
    )

    with pytest.raises(ProviderBrokerV2Error, match="proxy"):
        broker.execute(_request())


def test_openrouter_management_route_requires_job_scoped_management_key():
    transport = FakeTransport()
    broker = _broker(transport)
    request = _request(
        provider_id="openrouter_management",
        method="GET",
        url="https://openrouter.ai/api/v1/workspaces",
        logical_operation_id="management-operation",
        body_b64=base64.b64encode(b"").decode("ascii"),
    )
    with pytest.raises(ProviderBrokerV2Error, match="credential slot"):
        broker.execute(request)

    management_key = "miner-management-key"
    broker.provision_job_credential(
        job_id="job-1",
        slot="openrouter_management",
        credential=management_key,
        credential_value_hash_expected=credential_value_hash(management_key),
    )
    broker.execute(request)
    assert transport.calls[-1]["headers"]["Authorization"] == (
        "Bearer " + management_key
    )

    with pytest.raises(ProviderBrokerV2Error, match="method"):
        broker.execute(
            _request(
                provider_id="openrouter_management",
                method="POST",
                url="https://openrouter.ai/api/v1/workspaces",
                logical_operation_id="management-post",
                body_b64=base64.b64encode(b"{}").decode("ascii"),
            )
        )
