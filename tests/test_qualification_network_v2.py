from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import dns.resolver
import pytest
import whois

from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    ProviderBrokerV2,
    credential_reference_hash,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from gateway.tee.qualification_network_v2 import SecureQualificationNetworkV2
from leadpoet_canonical.attested_v2 import sha256_bytes


HASH = "sha256:" + "a" * 64


class QualificationTransport:
    def __init__(self, *, fail_dns: bool = False):
        self.calls = []
        self.fail_dns = fail_dns

    def __call__(self, **request):
        self.calls.append(request)
        parsed = urlsplit(request["url"])
        if parsed.hostname == "cloudflare-dns.com":
            if self.fail_dns:
                raise TimeoutError("DNS request timed out")
            query = parse_qs(parsed.query)
            name = query["name"][0]
            record_type = query["type"][0]
            if name == "missing.example":
                body = {"Status": 3}
            else:
                type_number = {"A": 1, "MX": 15, "TXT": 16}[record_type]
                data = {
                    "A": "127.0.0.2",
                    "MX": "10 mail.example.com.",
                    "TXT": '"v=spf1 -all"',
                }[record_type]
                body = {
                    "Status": 0,
                    "Answer": [{"name": name + ".", "type": type_number, "data": data}],
                }
            return self._response(200, json.dumps(body).encode("utf-8"))
        if parsed.hostname == "rdap.org":
            return self._response(
                302,
                b"",
                {"location": "https://rdap.verisign.com/com/v1/domain/example.com"},
            )
        if parsed.hostname == "rdap.verisign.com":
            body = {
                "events": [
                    {"eventAction": "registration", "eventDate": "1995-08-14T04:00:00Z"},
                    {"eventAction": "last changed", "eventDate": "2025-08-14T10:00:00Z"},
                ],
                "entities": [
                    {
                        "roles": ["registrar"],
                        "vcardArray": [
                            "vcard",
                            [["fn", {}, "text", "Example Registrar"]],
                        ],
                    }
                ],
                "nameservers": [{"ldhName": "NS1.EXAMPLE.COM"}],
            }
            return self._response(200, json.dumps(body).encode("utf-8"))
        raise AssertionError("unexpected destination: %s" % request["url"])

    @staticmethod
    def _response(status, body, headers=None):
        return {
            "http_status": status,
            "headers": headers or {"content-type": "application/json"},
            "body": body,
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }


def _installed(transport):
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(value)
            for slot, value in credentials.items()
        },
        retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: "2026-07-10T20:00:00Z",
    )
    broker.provision_credentials(credentials)
    results = []

    def execute(request):
        result = broker.execute(request)
        results.append(result)
        return result

    client = BrokeredProviderTransportV2(execute)
    client.install()
    secure = SecureQualificationNetworkV2()
    secure.install()
    return client, secure, results


def _scope(client):
    return client.scope(
        job_id="qualification-1",
        purpose="qualification.lead_decision.v2",
        logical_operation_id="qualification-1",
        retry_policy_hashes={provider: HASH for provider in BUILTIN_PROVIDER_ROUTES},
    )


@pytest.mark.asyncio
async def test_dns_queries_use_authenticated_doh_with_dnspython_shapes():
    transport = QualificationTransport()
    client, secure, results = _installed(transport)
    try:
        with _scope(client):
            mx = await asyncio.get_running_loop().run_in_executor(
                None, lambda: dns.resolver.resolve("example.com", "MX")
            )
            txt = await asyncio.get_running_loop().run_in_executor(
                None, lambda: dns.resolver.resolve("example.com", "TXT")
            )
            address = await asyncio.get_running_loop().run_in_executor(
                None, lambda: dns.resolver.resolve("example.com", "A")
            )
            with pytest.raises(dns.resolver.NXDOMAIN):
                await asyncio.get_running_loop().run_in_executor(
                    None, lambda: dns.resolver.resolve("missing.example", "A")
                )
        assert str(mx[0]) == "10 mail.example.com."
        assert txt[0].strings == (b"v=spf1 -all",)
        assert str(address[0]) == "127.0.0.2"
        assert [row["transport_attempt"]["provider_id"] for row in results] == [
            "dns",
            "dns",
            "dns",
            "dns",
        ]
    finally:
        secure.restore()
        client.restore()


def test_whois_call_uses_authenticated_rdap_and_preserves_consumed_attributes():
    transport = QualificationTransport()
    client, secure, results = _installed(transport)
    try:
        with _scope(client):
            record = whois.whois("example.com")
        assert record.creation_date.isoformat() == "1995-08-14T04:00:00+00:00"
        assert record.updated_date.isoformat() == "2025-08-14T10:00:00+00:00"
        assert record.registrar == "Example Registrar"
        assert record.name_servers == ["NS1.EXAMPLE.COM"]
        assert [row["transport_attempt"]["provider_id"] for row in results] == [
            "rdap",
            "public_web",
        ]
    finally:
        secure.restore()
        client.restore()


@pytest.mark.asyncio
async def test_dns_transport_failure_is_terminal_and_never_falls_back_to_raw_dns():
    transport = QualificationTransport(fail_dns=True)
    client, secure, results = _installed(transport)
    try:
        from gateway.tee.provider_client_v2 import ProviderClientV2Error

        with pytest.raises(ProviderClientV2Error, match="did not authenticate"):
            with _scope(client):
                with pytest.raises(dns.resolver.Timeout):
                    await asyncio.get_running_loop().run_in_executor(
                        None, lambda: dns.resolver.resolve("example.com", "MX")
                    )
        assert len(results) == 1
        assert results[0]["transport_attempt"]["terminal_status"] == "transport_failure"
        assert results[0]["transport_attempt"]["failure_code"] == "timeout"
    finally:
        secure.restore()
        client.restore()
