"""Managed upstream errors, alternate research, exact costs and publication."""
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lab_arena import broker, contracts, deepline_catalog, runtime, scoring, shim
from tests.lab_arena.bound_account_refusal421_postgres_test import database, base_database
from tests.lab_arena.company_quality_round_test import QualityHarness
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.deepline_managed_account_test import managed_error
from tests.lab_arena.deepline_runtime_transition_postgres_test import (
    TOOL, ExactResearchTransport, FrozenResearchSandbox,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_DEEPLINE_KEY, assert_canary_absent, keypair, price_table,
)

class ManagedTransport(ExactResearchTransport):
    def __init__(self):
        super().__init__()
        self.failed = set()

    def send(self, **request):
        result = super().send(**request)
        if request["method"] == "POST" and json.loads(request["body"])["payload"].get("query") == "managed outage":
            native = json.loads(result.body)["job_id"]
            self.failed.add(native)
            return broker.ProviderResponse(402, {}, json.dumps(dict(
                managed_error("vendor", TOOL), job_id=native, status="failed",
            )).encode())
        if request["method"] == "GET":
            native = parse_qs(urlsplit(request["url"]).query)["request_id"][0]
            if native in self.failed:
                document = json.loads(result.body)
                document["recent"]["entries"][0].update(credits="0", delta="0")
                return broker.ProviderResponse(200, {}, json.dumps(document).encode())
        return result


class ResearchAfterManagedError(FrozenResearchSandbox):
    def run_icp(self, spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            saved = {name: os.environ.get(name) for name in (shim.WORKER_SOCKET_ENV, "LAB_ARENA_INPUT_PATH")}
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            os.environ["LAB_ARENA_INPUT_PATH"] = str(spec.input_dir / runtime.INPUT_FILE_NAME)
            try:
                status, _, body = shim.execute(method="POST",
                    url="https://code.deepline.com/api/v2/integrations/" + TOOL + "/execute",
                    headers={"content-type": "application/json"},
                    body=json.dumps({"payload": {"query": "managed outage"}}).encode(), timeout_ms=5000)
                assert status == 502
                assert json.loads(body) == {"error": {"code": "provider_unavailable"}}
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
        return super().run_icp(spec, **kwargs)


def test_managed_error_does_not_block_research_scoring_or_publication(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    with connect() as connection, connection.cursor() as cursor:
        for name in ("365-lab-arena-trajectories.sql", "366-lab-arena-trajectory-capacity.sql"):
            cursor.execute((Path(__file__).parents[2] / "scripts" / name).read_text())
    h = QualityHarness(connect, tmp_path, challengers=["ManagedResearch"], runners=["alpha"])
    miner = keypair("svc-miner-ManagedResearch").ss58_address
    h.chain.owned[miner] = [miner]
    h.service.config.defaults = replace(h.service.config.defaults,
        benchmark_icp_count=10, runner_slot_ceiling=2, promotion_margin=0.5,
        per_icp_cost_policy=True, integrity_from="2000-01-01T00:00:00Z",
        contacts_from=None, company_quality_from=None,
        intent_details_from="2026-01-01T00:00:00Z",
        execution_sequence_from="2026-01-01T00:00:00Z",
        benchmark_disclosure_from="2026-01-01T00:00:00Z")
    h.service.config.daily_icp_source = lambda **kwargs: {
        "status":"ready", "set_id":int(kwargs["set_id"]), "icps":deepcopy(daily_icps()[:10])}
    snapshot = deepline_catalog.freeze_catalog({"tools": [row(TOOL,
        pricing={"unit": "call", "usdPerUnit": 0.003, "creditsPerUnit": 0.03, "currency": "USD"})]})
    h.service.config.deepline_catalog_source = lambda **_: snapshot
    transport = ManagedTransport()
    h.service.config.broker_factory = lambda service, round_row: broker.Broker(
        store=service.store, key_for=lambda _: CANARY_DEEPLINE_KEY,
        credential_for=lambda *_: CANARY_DEEPLINE_KEY,
        funding_source_for=lambda context: service.store.provider_funding(context.run_id, "deepline")["funding_source"],
        price_table=price_table(), transport=transport, clock=h.clock)
    h.sandbox = ResearchAfterManagedError(snapshot, transport)
    original_runner = h.runner
    def runner(index):
        result = original_runner(index, parallel=1)
        result._config.proxy_worker_pool = _verified_test_pool(2)
        return result
    h.runner = runner
    h.round_id = "arena-2099-10-07-managed"
    h.clock.now = datetime.now(timezone.utc) + timedelta(seconds=1)
    configuration = h.service.create_round(h.clock.now + timedelta(minutes=30), round_id=h.round_id)
    assert configuration['execution_sequence_policy'] == contracts.BASELINE_SCORED_FIRST_POLICY
    assert configuration['scorer_policy']['scoring_adapter_version'] == 'qualification_integrity_v2'
    assert 'contact_policy' not in configuration
    h.submit('ManagedResearch', h.round_id)
    h.clock.advance_to(h.schedule()['submission_cutoff'])
    assert h.service.advance_round(h.round_id)['status'] == 'ok'
    participants = h.service.store.get_round(h.round_id)['participants']
    for participant in participants:
        h.flavors.setdefault(participant['submission_id'], 'PublicBaseline')
    h.clock.advance_to(h.schedule()['stage_1_start'])
    assert h.service.advance_round(h.round_id)['assignments'] == 10
    h.advance_until("published", runners=1)
    runs = h.service.store.list_runs(h.round_id, kind="execute")
    assert len(runs) == 10 * len(participants)
    assert len(h.service.store.list_runs(h.round_id, kind="score")) == len(runs)
    assert all(r["status"] == "accepted" for r in h.service.store.list_runs(h.round_id))
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT amount_microusd,terminal_response,funding_source FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND provider='deepline' AND entry_kind='settlement'", (h.round_id,))
        settlements = cursor.fetchall()
        cursor.execute("SELECT max(amount_microusd) FROM public.lab_arena_ledger "
            "WHERE round_id=%s AND provider='deepline' AND entry_kind='reservation'", (h.round_id,))
        assert cursor.fetchone()[0] == 0
    failed = [r for r in settlements if r[1]["call_succeeded"] is False]
    successful = [r for r in settlements if r[1]["call_succeeded"] is True]
    assert len(failed) == len(successful) == len(runs)
    assert {r[2] for r in failed} == {"host", "miner_key"}
    assert all(r[0] == 0 and "account_failure_evidence" not in r[1]
               and "credit_failure_proof" not in r[1] for r in failed)
    assert all(r[0] == 3000 for r in successful)
    assert h.service.store.get_round(h.round_id)["publication_doc"]
    assert_canary_absent(h, connect)
