"""Production readers use the fixed gateway API; parity stays isolated."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from Leadpoet.utils import public_supabase
from leadpoet_canonical import production_parity_boundary_v2 as boundary
from tests.test_gateway_epoch_cutover_fence import _active_state, _cutover


PRODUCTION_ORIGIN = "https://gateway.subnet71.com"


def test_public_and_measured_production_origins_match_active_template():
    assert public_supabase.SUPABASE_URL == PRODUCTION_ORIGIN
    assert boundary.PRODUCTION_SUPABASE_ORIGIN == PRODUCTION_ORIGIN
    template = Path(__file__).parents[1] / "env.example"
    assert "SUPABASE_URL=" + PRODUCTION_ORIGIN in template.read_text().splitlines()
    assert boundary.configured_supabase_origin_v2({}) == PRODUCTION_ORIGIN
    assert boundary.validate_production_parity_boundary_v2({}, network="finney", netuid=71) == PRODUCTION_ORIGIN


def test_public_rpc_keeps_fixed_origin_and_public_key_despite_service_environment(monkeypatch):
    requests = []
    monkeypatch.setenv("SUPABASE_URL", "https://untrusted.invalid")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_service-test-only")
    monkeypatch.setattr(public_supabase, "SUPABASE_ANON_KEY", "sb_publishable_public-test-only")

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        return BytesIO(b'{"status":"ready"}')

    monkeypatch.setattr(public_supabase, "urlopen", open_request)
    assert public_supabase.call_public_rpc("public_state", {"p_scope": "finney"}, timeout_seconds=8) == {"status": "ready"}
    request, timeout = requests[0]
    assert len(requests) == 1 and timeout == 8
    assert request.full_url == PRODUCTION_ORIGIN + "/rest/v1/rpc/public_state"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"p_scope": "finney"}
    assert request.get_header("Apikey") == "sb_publishable_public-test-only"
    assert request.get_header("Authorization") == "Bearer sb_publishable_public-test-only"


def test_public_rpc_outage_has_no_alternate_origin(monkeypatch):
    observed = []

    def unavailable(request, *, timeout):
        observed.append(request.full_url)
        raise URLError("unavailable")

    monkeypatch.setattr(public_supabase, "urlopen", unavailable)
    with pytest.raises(URLError):
        public_supabase.call_public_rpc("public_state")
    assert observed == [PRODUCTION_ORIGIN + "/rest/v1/rpc/public_state"]


def test_validator_epoch_read_reaches_fixed_gateway_public_rpc(monkeypatch):
    from gateway.utils import epoch

    cutover = _cutover()
    row = _active_state(cutover)
    observed = []
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setenv("BITTENSOR_NETWORK", "finney")
    monkeypatch.setenv("BITTENSOR_NETUID", "71")

    def open_request(request, *, timeout):
        observed.append(request.full_url)
        return BytesIO(json.dumps([row]).encode())

    monkeypatch.setattr(public_supabase, "urlopen", open_request)
    assert epoch._read_cutover_state_from_db_sync() == row
    assert observed == [PRODUCTION_ORIGIN + "/rest/v1/rpc/research_lab_stateful_subnet_epoch_cutover_public_state_v1"]


def test_gateway_epoch_read_preserves_configured_service_authority(monkeypatch):
    from gateway.db import client
    from gateway.utils import epoch

    row = _active_state(_cutover())
    tables = []

    class Query:
        def table(self, name):
            tables.append(name)
            return self

        def select(self, columns):
            return self

        def eq(self, key, value):
            return self

        def limit(self, count):
            return self

        def execute(self):
            return SimpleNamespace(data=[row])

    monkeypatch.setenv("SUPABASE_URL", PRODUCTION_ORIGIN)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "sb_secret_service-test-only")
    monkeypatch.setattr(client, "get_write_client", lambda: Query())
    monkeypatch.setattr(epoch, "call_public_rpc", lambda *a, **kw: pytest.fail("service authority must stay selected"))
    assert epoch._read_cutover_state_from_db_sync() == row
    assert tables == ["research_lab_stateful_subnet_epoch_cutover_state_v1"]
    assert epoch._cutover_authority_cache_scope() == ("configured_service", PRODUCTION_ORIGIN)


def _parity(origin="https://parity-a1.cloudfront.net"):
    return {
        boundary.PRODUCTION_PARITY_MODE_ENV: "enabled",
        boundary.PRODUCTION_PARITY_RUN_ID_ENV: "parity-a1",
        boundary.PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV: origin,
        boundary.PRODUCTION_PARITY_BENCHMARK_DATE_ENV: "2026-10-28",
    }


def test_run_scoped_parity_preserves_chain_and_database_isolation():
    document = boundary.validate_production_parity_boundary_document_v2(_parity(), network="finney", netuid=71)
    assert document["mode"] == "production-parity"
    assert document["supabase_origin"] == "https://parity-a1.cloudfront.net"
    assert document["chain_host"] == boundary.PRODUCTION_CHAIN_HOST
    assert document["chain_archive_host"] == boundary.PRODUCTION_CHAIN_ARCHIVE_HOST


@pytest.mark.parametrize("origin", [
    PRODUCTION_ORIGIN,
    "http://parity-a1.cloudfront.net",
    "https://parity-a1.cloudfront.net/rest/v1",
    "https://parity-a1.cloudfront.net.invalid",
    "https://user@parity-a1.cloudfront.net",
])
def test_parity_rejects_production_or_non_scoped_origin(origin):
    with pytest.raises(boundary.ProductionParityBoundaryV2Error):
        boundary.validate_production_parity_boundary_document_v2(_parity(origin), network="finney", netuid=71)
