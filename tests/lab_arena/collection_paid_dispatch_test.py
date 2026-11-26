from copy import deepcopy

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_broker import CollectionBroker, REFERENCE_MODEL, REFERENCE_ROUTE
from lab_arena.collection_generation import CAPTURE_V2
from lab_arena.collection_intake import GENERATOR_ROUTES
from tests.lab_arena.collection_judge_test import PROMPT, SPLIT
from scripts.run_locked_collection import load_batch, verify_live_panel


def paid_offer():
    from lab_arena.collection_intake import PAID_WORKFLOW_TERMS_HASH
    raw = {
        "mode": "live", "name": "paid-p7", "seed": "test-paid", "scenario_ids": ["P7-s40401"],
        "routes": {**{m: r.tag for m, r in GENERATOR_ROUTES.items()},
                   **{m: r.route for m, r in judge.QUALIFICATION_JUDGES.items()}},
        "control_slots": {}, "drift_slots": {}, "split_manifest": SPLIT,
        "terms_sha256": PAID_WORKFLOW_TERMS_HASH, "capture_version": CAPTURE_V2,
        "payment_rule": "verified_p7_v1", "budgets_microusd": {REFERENCE_MODEL: 50000},
        "fixed_quotes_microusd": {REFERENCE_MODEL: 8172},
    }
    raw["routes"][REFERENCE_MODEL] = REFERENCE_ROUTE
    return raw


def test_paid_offer_is_immutable_and_binds_generator_route():
    from lab_arena.collection_workflow import comparison_revisions
    raw = paid_offer()
    policy = load_batch(raw, PROMPT)
    agent, builder = comparison_revisions(raw)
    assert policy.batch_id.startswith("dv-") and len(policy.batch_id) == 67
    proof = dict(schema="leadpoet.collection.verified_p7_prices.v1", payment_rule=raw["payment_rule"],
                 live_batch_id=policy.batch_id, terms_sha256=raw["terms_sha256"], harness_commit=agent, builder_commit=builder,
                 judge_prompt_sha256=policy.prompt_sha256, capture_version=CAPTURE_V2,
                 split_manifest_sha256=judge.digest(SPLIT), routes={REFERENCE_MODEL: REFERENCE_ROUTE},
                 prices={"P":raw["fixed_quotes_microusd"]}, budgets={"P":raw["budgets_microusd"]})
    verify_live_panel(raw, policy, proof)
    class Schedule:
        def rpc(self, name, params):
            assert name == "collection_reward_schedule_v1" and params == {}
            return {"pilot_evidence": proof}
    meter = CollectionBroker(None, Schedule(), None, {"mode":"live", "batch_id":policy.batch_id},
                             {}, routes=raw["routes"])
    assert meter._approved_generator_route(REFERENCE_MODEL, REFERENCE_ROUTE)
    assert not meter._approved_generator_route(REFERENCE_MODEL, GENERATOR_ROUTES[REFERENCE_MODEL].tag)
    quoted = deepcopy(raw)
    quoted["fixed_quotes_microusd"][REFERENCE_MODEL] += 1
    assert load_batch(quoted, PROMPT).batch_id != policy.batch_id
    with pytest.raises(ValueError):
        verify_live_panel(quoted, load_batch(quoted, PROMPT), proof)
    proof["live_batch_id"] = "dv-" + "0" * 64
    assert not meter._approved_generator_route(REFERENCE_MODEL, REFERENCE_ROUTE)


@pytest.mark.parametrize("change", [
    {"mode":"pilot"}, {"scenario_ids":["C-s40401"]}, {"scenario_ids":["P7-sholdout"]},
    {"control_slots":{"0":True}}, {"fixed_quotes_microusd":{}}, {"panel_qualification_sha256":"a"*64},
    {"capture_version":None}, {"payment_rule":"other"},
])
def test_payment_rule_fails_closed(change):
    with pytest.raises(ValueError):
        load_batch(dict(paid_offer(), **change), PROMPT)
