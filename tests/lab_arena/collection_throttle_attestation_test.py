import json
from copy import deepcopy
import pytest
from lab_arena import broker
from lab_arena.collection_broker import model_only_rejection


def approved():
    model = "moonshotai/kimi-k3.1-pro"
    return dict(
        model=model,
        messages=[{"role": "user", "content": "Synthetic scenario"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "local", "parameters": {"type": "object"}},
            }
        ],
        max_tokens=1024,
        reasoning={"effort": "low", "exclude": False},
        usage={"include": True},
        provider={
            "only": ["deepinfra/bf16"],
            "data_collection": "deny",
            "enforce_distillable_text": True,
            "allow_fallbacks": False,
            "zdr": True,
            "max_price": {"request": 0.0, "prompt": 0.01, "completion": 0.1},
        },
    )


def response(**changes):
    doc = {
        "id": "gen-demo",
        "error": {
            "code": 429,
            "message": "Provider rate limit",
            "metadata": {"error_type": "rate_limit_exceeded"},
        },
    }
    doc.update(changes)
    return broker.ProviderResponse(429, {}, json.dumps(doc).encode())


def test_typed_nonstreaming_refusal_without_hosted_work_is_auditable():
    b = approved()
    http = response()
    r = model_only_rejection(b, http)
    assert (
        r
        and r["http_status"] == 429
        and r["error_code"] == 429
        and r["no_delivered_output"]
    )
    assert (
        r["request_fee_ceiling"] == 0 and r["allowed_route"] == b["provider"]["only"][0]
    )
    assert "message" not in r and "Provider rate limit" not in str(r)


@pytest.mark.parametrize(
    "mutation",
    [
        "plugin",
        "stream",
        "search",
        "byok",
        "request_fee",
        "fallback",
        "structured",
        "partial",
        "usage",
        "embedded",
        "raw",
        "conflicting_code",
        "unknown_type",
        "nonzero",
        "provenance",
    ],
)
def test_any_missing_or_contradictory_evidence_preserves_a_hold(mutation):
    b = approved()
    r = response()
    if mutation == "plugin":
        b["plugins"] = [{"id": "file-parser"}]
    if mutation == "stream":
        b["stream"] = True
    if mutation == "search":
        b["tools"] = [{"type": "web_search"}]
    if mutation == "byok":
        b["provider"]["provider_keys"] = ["not-a-secret"]
    if mutation == "request_fee":
        b["provider"]["max_price"]["request"] = 0.01
    if mutation == "fallback":
        b["provider"]["allow_fallbacks"] = True
    if mutation == "structured":
        b["messages"][0]["content"] = [{"type": "image_url"}]
    if mutation == "partial":
        r = response(choices=[{"message": {"content": "partial"}}])
    if mutation == "usage":
        r = response(usage={"completion_tokens": 1})
    if mutation == "nonzero":
        r = response(usage={"cost": 0.01})
    if mutation == "embedded":
        r = broker.ProviderResponse(200, {}, r.body)
    if mutation == "raw":
        r = broker.ProviderResponse(429, {}, b"upstream limit")
    if mutation == "conflicting_code":
        r = response(error={"code": 400, "message": "different"})
    if mutation == "unknown_type":
        r = response(
            error={
                "code": 429,
                "message": "different",
                "metadata": {"error_type": "other"},
            }
        )
    if mutation == "provenance":
        r = broker.ProviderResponse(429, {}, r.body, internal_provenance="local")
    assert model_only_rejection(b, r) is None


def test_one_replacement_preserves_unknown_first_charge_and_uses_only_residual_slot_budget():
    from pathlib import Path
    from lab_arena import store
    from tests.lab_arena.lab_arena_pg_harness import (
        DEFAULT_MIGRATIONS,
        database_with_lab_arena_migration,
    )
    from tests.lab_arena.locked_collection_pipeline_test import registered, claim
    from scripts import reconcile_collection_generation

    directory = Path(__file__).resolve().parents[2] / "scripts"
    migrations = tuple(
        next(directory.glob(f"{prefix}-*.sql")).name
        for prefix in (
            720,
            721,
            722,
            723,
            724,
            725,
            726,
            727,
            728,
            729,
            732,
            733,
            738,
            740,
            741,
            742,
            743,
            744,
            745,
            746,
            747,
        )
    )
    for driver, dsn in database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS + migrations
    ):

        def connect():
            c = driver.connect(**dsn)
            c.autocommit = True
            return c

        db = store.PsycopgTransport(connect, role="lab_arena_service")
        miner = registered(db, "M")
        job = claim(miner, "small-bound", "P1-s1001")
        rid = job["job_id"]
        args = {
            "p_job": rid,
            "p_lease": store.hash_lease_token(job["lease_token"]),
            "p_request": "generate-1",
        }

        def record(j, n, http):
            a = dict(
                args,
                p_job=j["job_id"],
                p_lease=store.hash_lease_token(j["lease_token"]),
            )
            assert (
                db.rpc(
                    "collection_reserve_call_v1",
                    dict(a, p_provider="openrouter", p_hash="a" * 64, p_amount=n),
                )["status"]
                == "reserved"
            )
            db.rpc(
                "collection_record_call_v1",
                dict(
                    a,
                    p_hash="b" * 64,
                    p_provider_id="gen-eligible" if http == 429 else None,
                    p_cost=None,
                    p_status=http,
                ),
            )

        record(job, 31000, 429)
        with pytest.raises(store.ArenaStoreError):
            db.rpc(
                "collection_record_throttle_hold_v1",
                dict(
                    args,
                    p_original_request="c" * 64,
                    p_original_response="b" * 64,
                    p_proof="sha256:" + "d" * 64,
                ),
            )
        proof = dict(
            args,
            p_original_request="a" * 64,
            p_original_response="b" * 64,
            p_proof="sha256:" + "d" * 64,
        )
        assert (
            db.rpc("collection_record_throttle_hold_v1", proof)["status"]
            == "hold_preserved"
        )
        miner.complete(rid, job["lease_token"], "failed", {}, None, 7)
        retry = claim(miner, "small-bound", "P1-s1001")
        assert (retry["slot"], retry["attempt"], retry["model"]) == (
            job["slot"],
            2,
            job["model"],
        )
        assert retry["budget_microusd"] == job["budget_microusd"] - 31000
        assert retry["job_id"] != rid and retry["price_microusd"] == 0
        # The original remains unknown, a genuine later provider bill still wins.
        with driver.connect(**dsn) as con, con.cursor() as q:
            q.execute(
                "select cost_microusd from collection_call_receipts where job_id=%s",
                (rid,),
            )
            assert q.fetchone() == (None,)
        db.rpc(
            "collection_settle_generation_v1",
            dict(
                p_job=rid,
                p_request="generate-1",
                p_generation="gen-eligible",
                p_response_hash="b" * 64,
                p_lookup_hash="c" * 64,
                p_cost=150,
            ),
        )
        miner.complete(retry["job_id"], retry["lease_token"], "failed", {}, None, 7)
        assert claim(miner, "small-bound", "P1-s1001") is None
        db.close()
