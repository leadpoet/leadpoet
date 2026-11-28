import json

from lab_arena import broker
from scripts import reconcile_collection_generation as rc


def test_observer_pages_past_persistent_failures_and_sleeps_after_a_sweep(tmp_path):
    calls = []

    class Database:
        def rpc(self, function, params):
            assert function == "collection_pending_billing_v1"
            calls.append(tuple(params.values()))
            items = [
                {
                    "job": f"00000000-0000-0000-0000-00000000000{i}",
                    "request": "generate-1",
                    "provider": "openrouter",
                }
                for i in (1, 2, 3)
            ]
            if params["p_after_job"]:
                items = [i for i in items if i["job"] > params["p_after_job"]]
            return items[:2]

    results = []

    def unresolved(item):
        results.append(item["job"])
        return {"status": "unresolved", "generation_lookup_http": 404}

    state = tmp_path / "billing.json"
    assert (
        rc.reconcile_pending(Database(), unresolved, state, now=10000)["checked"] == 2
    )
    assert (
        rc.reconcile_pending(Database(), unresolved, state, now=10600)["checked"] == 1
    )
    assert (
        rc.reconcile_pending(Database(), unresolved, state, now=11200)["checked"] == 0
    )
    assert len(results) == len(set(results)) == 3
    assert (
        rc.reconcile_pending(Database(), unresolved, state, now=15000)["checked"] == 2
    )
    assert len(calls) == 3


def test_owner_generation_requires_control_receipt_from_same_account():
    item = {
        "job": "00000000-0000-0000-0000-000000000001",
        "request": "first",
        "provider": "judge",
        "source": "sha256:" + "a" * 64,
        "generation": "gen-missing",
        "control_generation": "gen-known",
        "control_cost": 900,
    }
    requests = []

    class Http:
        def send(self, **kw):
            assert kw["method"] == "GET" and kw["body"] == b""
            assert kw["headers"] == {"authorization": "Bearer owner-test"}
            gen = kw["url"].split("id=")[1]
            requests.append(gen)
            if gen == "gen-known":
                cost = self.control
            else:
                cost = "0.0003"
            return broker.ProviderResponse(
                200, {}, json.dumps({"data": {"id": gen, "total_cost": cost}}).encode()
            )

    class Db:
        def rpc(self, function, params):
            assert function == "collection_reconcile_launch_judge_v1"
            assert params["p_cost"] == 300 and params["p_source"] == item["source"]
            return {"status": "reconciled"}

    http = Http()
    http.control = "0.01"
    assert rc.reconcile_owner(Db(), http, item, "owner-test")["status"] == "unresolved"
    assert requests == ["gen-known"]
    http.control = "0.0009"
    assert rc.reconcile_owner(Db(), http, item, "owner-test")["status"] == "settled"
    assert requests == ["gen-known", "gen-known", "gen-missing"]


def test_busy_sweep_pages_beyond_two_and_checkpoints_before_deadline(tmp_path):
    from types import SimpleNamespace

    rows = [dict(job=f"00000000-0000-0000-0000-{i:012}", request="generate-1", provider="openrouter")
            for i in range(9)]

    def page(name, args):
        assert name == "collection_pending_billing_v1"
        return [r for r in rows if r["job"] > (args["p_after_job"] or "")][:2]

    consumed = []
    def read(row):
        consumed.append(row["job"])
        return {"status": "unresolved", "generation_lookup_http": 404}

    state = tmp_path / "billing.json"
    db = SimpleNamespace(rpc=page)
    run = lambda limit, deadline: rc.reconcile_pending(
        db, read, state, now=1000, max_calls=limit, deadline=deadline, monotonic=lambda: len(consumed))
    assert run(200, 3)["checked"] == 3
    assert json.loads(state.read_bytes())["cursor"] == [rows[2]["job"], "generate-1"]
    assert run(4, 100)["checked"] == 4
    assert run(200, 100)["checked"] == 2
    assert consumed == [r["job"] for r in rows]
    assert json.loads(state.read_bytes()) == {"next_sweep_at": 4600}
    assert run(200, 100)["checked"] == 0
