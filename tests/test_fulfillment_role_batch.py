from types import SimpleNamespace

import pytest

from gateway.fulfillment import scoring
from gateway.fulfillment.icp_checks import _role_decision_key, tier1_check
from qualification.scoring import role_batch_check


def _tier1_objects(role: str):
    lead = SimpleNamespace(
        role=role,
        role_type="Marketing",
        business="Example Company",
        industry="",
        sub_industry="",
        company_hq_country="",
        country="",
        employee_count="",
    )
    lead_output = SimpleNamespace(role=role, business=lead.business)
    icp = SimpleNamespace(
        industry=[],
        sub_industry=[],
        excluded_companies=[],
        target_role_types=[],
        target_roles=["Head of Marketing"],
        target_seniority="",
        company_country=[],
        contact_country=[],
        employee_count=[],
        company_stage="",
    )
    return lead, lead_output, icp


def test_tier1_uses_role_key_when_email_and_lead_id_are_absent():
    role = "Head of Communications and Marketing"
    lead, lead_output, icp = _tier1_objects(role)

    accepted = tier1_check(
        lead,
        lead_output,
        icp,
        set(),
        role_decisions={_role_decision_key(role): True},
    )
    rejected = tier1_check(
        lead,
        lead_output,
        icp,
        set(),
        role_decisions={_role_decision_key(role): False},
    )

    assert accepted is None
    assert rejected == "role_mismatch"


@pytest.mark.asyncio
async def test_batch_role_decisions_survive_no_email_handoff(monkeypatch):
    leads = [
        SimpleNamespace(
            role="Head of Communications and Marketing",
            email="",
            full_name="Person One",
        ),
        SimpleNamespace(
            role="  HEAD OF COMMUNICATIONS AND MARKETING  ",
            email="",
            full_name="Person Two",
        ),
        SimpleNamespace(
            role="Marketing Director",
            email="",
            full_name="Person Three",
        ),
    ]
    icp = SimpleNamespace(target_roles=["Head of Marketing"])
    observed_queue = []
    observed_decisions = []

    async def fake_batch_check(queue, target_roles):
        observed_queue.extend(queue)
        assert target_roles == ["Head of Marketing"]
        return {0: True, 1: False}

    async def fake_score_lead(lead, _icp, _seen, **kwargs):
        decisions = kwargs["role_decisions"]
        observed_decisions.append(decisions)
        return decisions.get(_role_decision_key(lead.role))

    monkeypatch.setattr(scoring, "FULFILLMENT_VERIFY_EMAIL", False)
    monkeypatch.setattr(
        "qualification.scoring.role_batch_check.batch_check",
        fake_batch_check,
    )
    monkeypatch.setattr(scoring, "score_fulfillment_lead", fake_score_lead)

    results = await scoring.score_fulfillment_batch(leads, icp)

    assert observed_queue == [
        {"id": 0, "role": "Head of Communications and Marketing"},
        {"id": 1, "role": "Marketing Director"},
    ]
    assert results == [True, True, False]
    assert all(
        decisions
        == {
            "head of communications and marketing": True,
            "marketing director": False,
        }
        for decisions in observed_decisions
    )


@pytest.mark.asyncio
async def test_role_batch_preserves_zero_numeric_id(monkeypatch):
    async def fake_judge_chunk(_http, _key, _target_roles, chunk):
        assert chunk == [{"id": 0, "role": "Head of Communications and Marketing"}]
        return [{"id": 0, "match": True, "reason": "same role family"}]

    monkeypatch.setattr(role_batch_check, "_judge_chunk", fake_judge_chunk)

    result = await role_batch_check.batch_check(
        [{"id": 0, "role": "Head of Communications and Marketing"}],
        ["Head of Marketing"],
        api_key="test-key",
    )

    assert result == {0: True}
