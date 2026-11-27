from copy import deepcopy

import pytest

from lab_arena import collection_judge as judge
from lab_arena.collection_broker import REFERENCE_MODEL, REFERENCE_ROUTE, CollectionBroker
from lab_arena.collection_workflow import comparison_revisions
from scripts import run_locked_collection as runner
from tests.lab_arena.collection_judge_test import PROMPT
from tests.lab_arena.collection_paid_dispatch_test import paid_offer


def test_sales_evidence_offer_binds_its_task_family_and_measured_tariff():
    raw = paid_offer()
    raw.update(payment_rule="verified_sales_evidence_v1", scenario_ids=["P8-s61101"])
    policy = runner.load_batch(raw, PROMPT)
    harness, builder = comparison_revisions(raw)
    approval = dict(schema="leadpoet.collection.verified_task_prices.v1", payment_rule=raw["payment_rule"],
                    live_batch_id=policy.batch_id, terms_sha256=raw["terms_sha256"], harness_commit=harness,
                    builder_commit=builder, judge_prompt_sha256=policy.prompt_sha256,
                    capture_version=raw["capture_version"], split_manifest_sha256=judge.digest(raw["split_manifest"]),
                    routes={REFERENCE_MODEL: REFERENCE_ROUTE}, prices={"P8": raw["fixed_quotes_microusd"]},
                    budgets={"P8": raw["budgets_microusd"]})
    runner.verify_live_panel(raw, policy, approval)
    with pytest.raises(ValueError):
        runner.verify_live_panel(raw, policy, dict(approval, prices={"P":raw["fixed_quotes_microusd"]}))
    with pytest.raises(ValueError):
        runner.load_batch(dict(raw,scenario_ids=["P7-s61101", "P8-s61101"]), PROMPT)
    with pytest.raises(ValueError):
        runner.verify_live_panel(raw,policy,dict(approval,schema="leadpoet.collection.verified_p7_prices.v1"))


def test_broker_verifies_the_offer_signed_for_this_job():
    raw=paid_offer()
    bid=runner.load_batch(raw, PROMPT).batch_id
    class Registry:
        def rpc(self,name,args):
            assert (name,args)==("collection_offer_schedule_v1",{"p_offer":bid})
            return {"pilot_evidence":{"schema":"leadpoet.collection.verified_task_prices.v1",
                                     "live_batch_id":bid,"routes":{REFERENCE_MODEL:REFERENCE_ROUTE}}}
    b=CollectionBroker(None,Registry(),None,{"batch_id":bid,"mode":"live"}, {},routes=raw["routes"])
    assert b._approved_generator_route(REFERENCE_MODEL,REFERENCE_ROUTE)
    assert not b._approved_generator_route(REFERENCE_MODEL,"not-approved")


def test_multiple_signed_offers_get_fair_turns_and_no_double_dispatch(tmp_path):
    from scripts.run_locked_collection import poll_paid_offers
    from scripts.collection_review import write_new
    baseline=paid_offer()
    directory=tmp_path/"offers"
    directory.mkdir(mode=0o700)
    successor=deepcopy(baseline)
    successor["name"]="updated-prices"
    first=runner.load_batch(baseline,PROMPT).batch_id
    second=runner.load_batch(successor,PROMPT).batch_id
    write_new(directory/(second+".json"),successor)
    calls=[]
    def run(raw, *, before_claim):
        calls.append(raw["name"])
        before_claim()
        return 1
    assert poll_paid_offers(baseline,PROMPT,directory,run)==1
    assert poll_paid_offers(baseline,PROMPT,directory,run)==1
    assert set(calls)=={baseline["name"],successor["name"]}
    def ambiguous(raw, *, before_claim):
        before_claim()
        raise ValueError("unknown claim")
    with pytest.raises(ValueError,match="unknown claim"):
        poll_paid_offers(baseline,PROMPT,directory,ambiguous)
    assert first != second


def test_unpriced_or_invalid_offer_cannot_starve_a_valid_offer(tmp_path):
    from scripts.run_locked_collection import poll_paid_offers
    from scripts.collection_review import write_new
    baseline=paid_offer()
    write_new(tmp_path/("dv-"+"0"*64+".json"),{"unknown":"bad manifest"})
    saw=[]
    def run(raw, *, before_claim):
        saw.append(raw["name"])
        before_claim()
        return 1
    assert poll_paid_offers(baseline,PROMPT,tmp_path,run)==1
    assert saw==[baseline["name"]]
