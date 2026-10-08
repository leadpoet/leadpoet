"""A provider credit refusal across the real broker and published round path.

All provider responses are local fixture data. No paid provider is contacted.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from lab_arena import broker, runner as rn, runtime, scoring, shim
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import database
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import _verified_test_pool
from tests.lab_arena import test_lab_arena_service_round as fixtures


REFUSAL_QUERY = "fixture-credit-refusal"


class CreditRefusalTransport(fixtures.FakeProviderTransport):
    def __init__(self):
        super().__init__()
        self.refused_dispatches = 0

    def send(self, **request):
        if request["method"] == "POST":
            document = json.loads(request["body"])
            if document.get("payload", {}).get("query") == REFUSAL_QUERY:
                self.refused_dispatches += 1
                return broker.ProviderResponse(
                    402,
                    {"content-type": "application/json"},
                    json.dumps({
                        "code": "INSUFFICIENT_CREDITS",
                        "error": "fixture account has insufficient credits",
                        "billing": {
                            "kind": "insufficient_credits",
                            "required_credits": "1",
                            "balance_credits": "0",
                            "needed_credits": "1",
                        },
                    }).encode(),
                )
        response = super().send(**request)
        if request["method"] == "POST" and response.status == 200:
            # The ordinary fixture's exact billing is free. Give each
            # successful call a final, nonzero provider receipt instead.
            with self._deepline_lock:
                self._deepline_jobs[-1]["credits"] = "0.03"
                self._deepline_jobs[-1]["delta"] = "-0.03"
        return response


class RefusalSandbox(fixtures.ModelSandbox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_refusals = 0

    def run_icp(self, spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
            return super().run_icp(spec, **kwargs)
        submission_id = spec.source_dir.parent.name.removeprefix("submission-")
        flavor = self.flavor_by_submission.get(submission_id)
        if flavor != "CreditMiner":
            return super().run_icp(spec, **kwargs)
        position = int(str(document["icp"]["icp_id"]).rsplit("_", 1)[-1]) - 1
        # Preserve actual completed research on the first ICP. The second ICP
        # has no research result; both see an unresolved credit refusal.
        if position == 0:
            result = super().run_icp(spec, **kwargs)
        else:
            spec.output_path.write_text(json.dumps({"companies": []}))
            result = runtime.fake_result(
                exit_code=0, output_bytes=runtime.read_output(spec)
            )
        with self.lock:
            previous = os.environ.get(shim.WORKER_SOCKET_ENV)
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                for _ in range(rn.MAX_REFUSED_FRAMES + 1):
                    try:
                        status, _headers, _body = shim.dispatch(
                            "deepline.execute",
                            {"tool": "exa_search", "payload": {"query": REFUSAL_QUERY}},
                            5000,
                        )
                    except shim.ShimError as exc:
                        assert "miner_credentials_unavailable" in str(exc)
                        self.local_refusals += 1
                    else:
                        assert status == 402
            finally:
                if previous is None:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
                else:
                    os.environ[shim.WORKER_SOCKET_ENV] = previous
        return result


def test_credit_refusal_preserves_partial_work_but_not_empty_success(
    database, tmp_path, monkeypatch,
):
    connect = lambda: database[0].connect(**database[1])
    with connect() as connection, connection.cursor() as cursor:
        for name in (
            "365-lab-arena-trajectories.sql",
            "366-lab-arena-trajectory-capacity.sql",
        ):
            cursor.execute((Path(__file__).resolve().parents[2] / "scripts" / name).read_text())
    h = fixtures.Harness(
        connect, tmp_path, challengers=["CreditMiner"], runners=["alpha"]
    )
    miner_hotkey = fixtures.keypair("svc-miner-CreditMiner").ss58_address
    h.chain.owned[miner_hotkey] = [miner_hotkey]
    h.service.config.defaults = replace(
        h.service.config.defaults,
        benchmark_icp_count=2,
        execution_sequence_from="2000-01-01T00:00:00Z",
        integrity_from="2000-01-01T00:00:00Z",
        per_icp_cost_policy=True,
    )
    h.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]),
        "icps": fixtures.daily_icps()[:2],
    }
    monkeypatch.setattr(fixtures, "deterministic_scorer", _baseline_first_judge)
    # The separate runner test covers the production threshold of 25. This
    # transition test keeps the PostgreSQL round small while exercising it.
    monkeypatch.setattr(rn, "MAX_REFUSED_FRAMES", 3)
    transport = CreditRefusalTransport()
    original_factory = h.service.config.broker_factory

    def broker_factory(service, round_row):
        instance = original_factory(service, round_row)
        instance._transport = transport
        return instance

    h.service.config.broker_factory = broker_factory
    h.sandbox = RefusalSandbox(
        flavor_by_submission=h.flavors, broken_submissions=h.broken
    )
    original_runner = h.runner

    def verified_runner(index, parallel=4):
        instance = original_runner(index, parallel=1)
        instance._config.proxy_worker_pool = _verified_test_pool(2)
        return instance

    h.runner = verified_runner
    assert fixtures._start_round(h, day=47, epoch=62_047) == 2
    h.clock.advance_to(h.schedule()["stage_1_start"])
    assert h.service.advance_round(h.round_id)["assignments"] == 2
    h.advance_until("published", runners=1, max_steps=100)

    round_row = h.service.store.get_round(h.round_id)
    miner_id = next(
        row["submission_id"] for row in round_row["participants"]
        if not row["is_king"]
    )
    miner_runs = [
        row for row in h.service.store.list_runs(h.round_id, kind="execute")
        if row["submission_id"] == miner_id
    ]
    by_position = {}
    for row in miner_runs:
        by_position.setdefault(row["icp_position"], []).append(row)
    assert set(by_position) == {0, 1}
    productive = [row for row in by_position[0] if row["status"] == "accepted"]
    assert len(productive) == 1
    assert productive[0]["output_ref"]
    assert productive[0]["per_icp_score"] > 0
    assert all(
        row["terminal_cause"] == "credential_error" and not row.get("output_ref")
        for row in by_position[1]
    )
    assert round_row["stage2_scoring_plan_doc"]["zero_rows"] == [{
        "submission_id": miner_id,
        "icp_position": 1,
        "cause": "credential_error",
    }]
    assert h.sandbox.local_refusals == len(miner_runs)
    assert transport.refused_dispatches == rn.MAX_REFUSED_FRAMES * len(miner_runs)

    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT entry_kind, amount_microusd, terminal_response, call_identity "
            "FROM public.lab_arena_ledger WHERE round_id=%s "
            "AND provider='deepline' ORDER BY entry_id", (h.round_id,)
        )
        ledger = cursor.fetchall()
    refusal_settlements = [
        row for row in ledger
        if row[0] == "settlement"
        and isinstance(row[2], dict)
        and row[2].get("status") == 402
    ]
    assert len(refusal_settlements) == transport.refused_dispatches, [
        (row[0], row[1], sorted(row[2]) if isinstance(row[2], dict) else None)
        for row in ledger
    ]
    assert all(row[1] == 0 for row in refusal_settlements)
    reservations = [row for row in ledger if row[0] == "reservation"]
    dispatches = [row for row in ledger if row[0] == "dispatch"]
    settlements = [row for row in ledger if row[0] == "settlement"]
    assert len(reservations) == len(dispatches) == len(settlements)
    assert len({row[3] for row in dispatches}) == len(dispatches)
    assert all(row[1] in (0, 3000) for row in settlements)
    assert any(row[1] == 3000 for row in settlements)
    publication = round_row["publication_doc"]
    miner_result = next(
        row for row in publication["final_ranking"]
        if row["submission_id"] == miner_id
    )
    assert miner_result["final_score"] > 0
    assert len(miner_result["cost_summary"]["per_icp"]) == 2
    summary = miner_result["cost_summary"]
    assert summary["competition_sourcing_microusd"] == 3000
    assert summary["judge"]["successful_microusd"] > 0
    public = h.service.public_results(h.round_id, miner_id)
    assert public["submission_scores"]["final"] == miner_result["final_score"]
    assert len(public["scores"]["stage_2"]) == 2
