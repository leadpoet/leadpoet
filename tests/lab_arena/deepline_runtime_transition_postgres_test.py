"""Frozen dynamic tools across the production runner, broker, ledger and publish path."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lab_arena import broker, deepline_catalog, runtime, runtime_version, scoring, shim
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _start_parallel_round, _verified_test_pool
from tests.lab_arena.test_deepline_catalog import row
from tests.lab_arena.test_lab_arena_service_round import (
    CANARY_DEEPLINE_KEY, Harness, InProcessApi, assert_canary_absent, keypair, price_table,
)

TOOL = "vendor_company_search"


class ExactResearchTransport:
    """Return exact provider receipts without dispatching paid work."""
    def __init__(self):
        self.requests = []
        self.identities = {}

    def send(self, **request):
        self.requests.append(request)
        assert request["headers"]["authorization"] == "Bearer " + CANARY_DEEPLINE_KEY
        if request["method"] == "POST":
            data = json.loads(request["body"])
            assert data["provider"] == "vendor" and data["operation"] == TOOL
            key = request["headers"]["idempotency-key"]
            assert key.startswith("arena:") and key not in self.identities
            native = "native-company-call-%d" % len(self.identities)
            self.identities[key] = native
            return broker.ProviderResponse(200, {}, json.dumps({
                "job_id": native, "status": "completed", "result": {"companies": []},
            }).encode())
        assert request["method"] == "GET"
        assert request["url"].startswith(broker.DEEPLINE_EXACT_BILLING_URL)
        native = parse_qs(urlsplit(request["url"]).query)["request_id"][0]
        assert native in self.identities.values()
        return broker.ProviderResponse(200, {}, json.dumps({"recent": {
            "request_id": native, "entries": [{
                "id": "usage-" + native, "request_id": native,
                "provider": "vendor", "operation": TOOL, "credits": "0.03", "delta": "-0.03",
                "charge_state": "posted", "charge_finality": "final", "metadata": {},
            }],
        }}).encode())


class FrozenResearchSandbox:
    def __init__(self, snapshot, transport):
        self.snapshot = snapshot
        self.transport = transport
        self.metadata_reads = 0

    def run_icp(self, spec, **_):
        input_path = spec.input_dir / runtime.INPUT_FILE_NAME
        document = json.loads(input_path.read_text())
        assert document["deepline_catalog"] == self.snapshot
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            assert document["companies"] == []
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(
                scoring.build_scoring_output(document["scored_run_id"], []),
            ).encode())
        saved = {name: os.environ.get(name) for name in (shim.WORKER_SOCKET_ENV, "LAB_ARENA_INPUT_PATH")}
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        os.environ["LAB_ARENA_INPUT_PATH"] = str(input_path)
        try:
            before = len(self.transport.requests)
            status, _, data = shim.execute(method="GET", url=deepline_catalog.CATALOG_URL,
                headers={}, body=b"", timeout_ms=5000)
            assert status == 200 and json.loads(data)["tools"][0]["toolId"] == TOOL
            status, _, data = shim.execute(method="GET",
                url="https://code.deepline.com/api/v2/integrations/" + TOOL + "/get",
                headers={"x-deepline-tool-meta-only": "1"}, body=b"", timeout_ms=5000)
            assert status == 200 and json.loads(data)["toolId"] == TOOL
            assert len(self.transport.requests) == before
            self.metadata_reads += 2
            status, headers, data = shim.execute(method="POST",
                url="https://code.deepline.com/api/v2/integrations/" + TOOL + "/execute",
                headers={"content-type": "application/json"},
                body=json.dumps({"payload": {"query": document["icp"]["prompt"][:200]}}).encode(),
                timeout_ms=5000)
            assert status == 200 and json.loads(data)["status"] == "completed"
            assert headers["x-leadpoet-settled-microusd"] == "3000"
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        spec.output_path.write_text(json.dumps({"companies": []}))
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec))


def test_dynamic_research_baseline_and_miner_complete_and_publish(database, tmp_path):
    connect = lambda: database[0].connect(**database[1])
    with connect() as connection, connection.cursor() as cursor:
        for name in ("365-lab-arena-trajectories.sql", "366-lab-arena-trajectory-capacity.sql"):
            cursor.execute((Path(__file__).resolve().parents[2] / "scripts" / name).read_text())
    h = Harness(connect, tmp_path, challengers=["DynamicResearch"], runners=["alpha"])
    miner = keypair("svc-miner-DynamicResearch").ss58_address
    h.chain.owned[miner] = [miner]
    h.service.config.defaults = replace(h.service.config.defaults,
        per_icp_cost_policy=True, integrity_from="2000-01-01T00:00:00Z")
    snapshot = deepline_catalog.freeze_catalog({"tools": [row(
        TOOL, pricing={"unit": "call", "usdPerUnit": 0.003, "creditsPerUnit": 0.03, "currency": "USD"},
    )]})
    h.service.config.deepline_catalog_source = lambda **_: snapshot
    transport = ExactResearchTransport()
    h.service.config.broker_factory = lambda service, round_row: broker.Broker(
        store=service.store, key_for=lambda _: CANARY_DEEPLINE_KEY,
        credential_for=lambda context, provider: CANARY_DEEPLINE_KEY,
        funding_source_for=lambda context: service.store.provider_funding(context.run_id, "deepline")["funding_source"],
        price_table=price_table(), transport=transport, clock=h.clock,
    )
    h.sandbox = FrozenResearchSandbox(snapshot, transport)
    quota_snapshots = []

    class AuditedApi(InProcessApi):
        def trajectory(self, run_id, lease_token, events):
            return self.service.handle_trajectory(run_id, lease_token, {"events": events})

        def provider(self, run_id, lease_token, frame):
            result = super().provider(run_id, lease_token, frame)
            if frame["operation_id"] == "deepline.execute":
                quota_snapshots.append(self.service.handle_quota_snapshot(run_id, lease_token))
            return result

    h.api_factory = lambda: AuditedApi(h.service)
    make_runner = h.runner

    def runner_with_native_worker(index):
        runner = make_runner(index, parallel=1)
        runner._config.proxy_worker_pool = _verified_test_pool(2)
        return runner

    h.runner = runner_with_native_worker
    participants = _start_parallel_round(h, "arena-2099-10-06-ff", slot_ceiling=2)
    configuration = h.service.store.get_round(h.round_id)["configuration_doc"]
    assert configuration["deepline_catalog"] == snapshot
    assert configuration["call_quotas"]["deepline"] == 0
    h.advance_until("published", runners=1)
    runs = h.service.store.list_runs(h.round_id, kind="execute")
    assert len(runs) == 20 * len(participants)
    assert all(run["status"] == "accepted" and run["terminal_cause"] == "accepted" for run in runs)
    assert len(quota_snapshots) == len(runs)
    assert all(item["providers"]["deepline"] == {
        "limit": 0, "used": 1, "remaining": None, "inflight": 0,
    } for item in quota_snapshots)
    assert h.sandbox.metadata_reads == 2 * len(runs)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT entry_kind, operation_id, amount_microusd, entry_doc, terminal_response "
            "FROM public.lab_arena_ledger WHERE round_id=%s AND provider='deepline' ORDER BY entry_id", (h.round_id,))
        entries = cursor.fetchall()
    assert all(entry[1] == "deepline.execute" for entry in entries)
    reservations = [entry for entry in entries if entry[0] == "reservation"]
    settlements = [entry for entry in entries if entry[0] == "settlement"]
    assert len(reservations) == len(settlements) == len(runs)
    assert all(entry[2] == 0 for entry in reservations)
    assert all(entry[2] == 3000 for entry in settlements)
    assert all(entry[3]["deepline_execution_key"].startswith("arena:") for entry in reservations)
    assert all(entry[4]["provider_cost"]["request_id"] in transport.identities.values() for entry in settlements)
    assert len(transport.requests) == 2 * len(runs)
    assert h.service.store.get_round(h.round_id)["status"] == "published"
    participants_by_id = {p["submission_id"]: p for p in h.service.store.get_round(h.round_id)["participants"]}
    audited_kinds = set()
    for run in h.service.store.list_runs(h.round_id):
        assert run["status"] == "accepted"
        role = "baseline" if participants_by_id[run["submission_id"]]["is_king"] else "miner"
        audited_kinds.add((role, run["kind"]))
        if run["result_doc"].get("schema_version") == "leadpoet.lab_arena.cached_run_result.v1":
            # A reuse is not a new validator execution. Follow its existing
            # immutable source pointer instead of inventing an executing SHA.
            source = h.service.store.get_run(run["result_doc"]["source_score_run_id"])
            assert source["kind"] == "score" and source["status"] == "accepted"
            assert source["result_doc"]["resource_summary"]["validator_source_commit"] == runtime_version.SOURCE_METADATA["validator_source_commit"]
            continue
        summary = run["result_doc"]["resource_summary"]
        assert summary["validator_source_commit"] == runtime_version.SOURCE_METADATA["validator_source_commit"]
        assert summary["gateway_claim_source_commit"] == runtime_version.SOURCE_METADATA["validator_source_commit"]
        events = h.service.store.list_trajectory_events(run["run_id"])
        started = [e for e in events if e["event_kind"] == "runtime.started"]
        assert len(started) == 1
        assert started[0]["content"]["validator_source_commit"] == summary["validator_source_commit"]
        assert started[0]["content"]["gateway_claim_source_commit"] == summary["gateway_claim_source_commit"]
        role = "baseline" if participants_by_id[run["submission_id"]]["is_king"] else "miner"
        for event in events:
            for key in ("round_id", "submission_id", "icp_position", "runner_hotkey", "attempt"):
                assert event[key] == run[key]
            assert event["model_role"] == role and event["run_kind"] == run["kind"]
        audited_kinds.add((role, run["kind"]))
    assert audited_kinds == {(role, kind) for role in ("baseline", "miner") for kind in ("execute", "score")}
    assert_canary_absent(h, connect)
