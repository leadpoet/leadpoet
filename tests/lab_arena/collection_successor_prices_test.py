"""Comparable costs set future fixed offers. Measurements cannot become payable jobs."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from lab_arena import collection_intake as intake
from lab_arena.collection_workflow import comparison_revisions
from scripts import collection_price_successor as pricing
from tests.lab_arena.collection_paid_dispatch_test import paid_offer, PROMPT


def row(model, cost, cap=50000):
    return dict(
        job_id=str(uuid4()),
        model=model,
        cost_microusd=cost,
        budget_microusd=cap,
        scenario_id="P7-s60001",
        artifact_sha256="sha256:" + "a" * 64,
        harness_commit=comparison_revisions(paid_offer())[0],
        calls=2,
    )


def test_exact_mean_fee_and_fixed_are_bound_before_spend():
    source = paid_offer()
    model = next(iter(source["budgets_microusd"]))
    one, two = row(model, 6537), row(model, 10380, 30000)
    historic = {
        "quote_epoch": 26624,
        "payment_block": 9589716,
        "quoted_microusd": 34180,
        "received_microusd_at_signed_price": 28120,
        "block_hash": "0x" + "7" * 64,
    }
    paid, prices = pricing.build_offer(
        source,
        PROMPT,
        [one, two],
        ["P7-s91001"],
        "verified_p7_v1",
        "dv-" + "a" * 64,
        through=1,
        historical=historic,
    )
    assert paid["fixed_quotes_microusd"] == {model: 10574}  # ceil(1.25 * 8458.5)
    assert (
        paid["budgets_microusd"] == {model: 30000}
        and prices["historical_realization"] == historic
    )
    _, revised = pricing.build_offer(
        source,
        PROMPT,
        [one],
        ["P7-s91001"],
        "verified_p7_v1",
        "dv-" + "a" * 64,
        through=0,
        historical=historic,
    )
    assert prices["live_batch_id"] != revised["live_batch_id"]
    with pytest.raises(ValueError, match="duplicate"):
        pricing.build_offer(
            source, PROMPT, [one, one], ["P7-s91001"], "verified_p7_v1", "x"
        )
    with pytest.raises(ValueError, match="observed cost"):
        pricing.mean_quotes([dict(one, cost_microusd=one["budget_microusd"] + 1)])


def test_owner_measurements_require_actual_matching_provider_bills(monkeypatch):
    offer = paid_offer()
    model = next(iter(offer["budgets_microusd"]))
    case = {
        k: v for k, v in row(model, 100).items() if k not in ("cost_microusd", "calls")
    }
    case["scenario_id"] = "P8-s60202"
    ref = dict(offer, batch_id="owner-p8-" + "a" * 64, jobs=[case])
    originals = [("gen-a", 40, "a" * 64), ("gen-b", 60, "b" * 64)]

    def native(*_args, receipts_out):
        receipts_out.extend(originals)
        return {
            "eligible": True,
            "payment_rule": "verified_sales_evidence_v1",
            "calls": 2,
            "cost_microusd": 100,
        }, {}

    monkeypatch.setattr(pricing, "_native", native)

    def bill(gen):
        return SimpleNamespace(microusd={"gen-a": 40, "gen-b": 60}[gen]), "a" * 64

    measured = pricing.owner_samples(ref, None, bill)
    assert measured[0]["cost_microusd"] == 100 and measured[0]["calls"] == 2
    with pytest.raises(ValueError, match="confirmed native bill"):
        pricing.owner_samples(ref, None, lambda _: (None, {}))
    with pytest.raises(ValueError, match="differ"):
        pricing.owner_samples(
            ref,
            None,
            lambda gen: (
                SimpleNamespace(microusd=60 if gen == "gen-a" else 40),
                "f" * 64,
            ),
        )
    originals[1] = originals[0]
    with pytest.raises(ValueError, match="reused"):
        pricing.owner_samples(ref, None, bill)


def test_price_estimate_labels_and_published_selection(capsys):
    from lab_arena.collection_cli import _review_paid_offer

    model = intake.MODELS[1]
    bid = "dv-" + "c" * 64
    receipt = {
        "job_count": 5,
        "payment_rule": "verified_sales_evidence_v1",
        "live_batch_id": bid,
        "schema": intake.VERIFIED_TASK_PRICE_SCHEMA,
        "prices": {"P8": {model: 4000}},
        "budgets": {"P8": {model: 50000}},
        "historical_realization": {
            "quote_epoch": 26624,
            "payment_block": 9589716,
            "block_hash": "0x" + "f" * 64,
            "quoted_microusd": 34180,
            "received_microusd_at_signed_price": 28120,
        },
    }
    calls = []

    def rpc(fn, args):
        calls.append((fn, args))
        return {"pilot_evidence": receipt} if args == {"p_offer": bid} else {}

    published = intake.comparison_offer(SimpleNamespace(rpc=rpc), bid)
    assert published["reference_runs"] == 5 and published["offer_id"] == bid
    assert (
        _review_paid_offer(published, SimpleNamespace(accept_offer=bid), 50000) == bid
    )
    wording = capsys.readouterr().err
    assert "82.3%" in wording and "2.8%" in wording and "estimated" in wording
    assert calls == [("collection_offer_schedule_v1", {"p_offer": bid})]
    with pytest.raises(intake.IntakeError, match="unknown"):
        intake.comparison_offer(SimpleNamespace(rpc=rpc), "dv-" + "d" * 64)
