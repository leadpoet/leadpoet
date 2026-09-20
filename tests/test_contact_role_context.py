"""Regression tests for verified current-job duties in contact role checks."""

from __future__ import annotations

import asyncio
from typing import Any

from qualification.scoring.contact_verification import (
    contact_source_semantics,
    verify_contact,
)
from qualification.scoring.role_batch_check import _build_prompt


def _contact(role: str = "Workplace Manager") -> dict[str, Any]:
    return {
        "full_name": "Casey Rivera",
        "role": role,
        "linkedin_url": "https://www.linkedin.com/in/casey-rivera/",
        "location": {"country": "US", "region": "Washington", "city": "Seattle"},
        "email": "casey@example.test",
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "broker_call_id": "broker-role-context",
            "record_id": "profile-role-context",
        },
    }


def _company(role: str = "Workplace Manager") -> dict[str, Any]:
    return {
        "name": "Example Systems",
        "domain": "example.test",
        "linkedin_id": "example-systems",
        "contact": _contact(role),
    }


def _icp(*, seniority: str = "Manager") -> dict[str, Any]:
    return {
        "target_roles": ["Operations Manager"],
        "target_seniority": seniority,
        "contact_geography": {},
    }


def _profile(*, title: str = "Workplace Manager", duties: str = "") -> dict[str, Any]:
    return {
        "id": "profile-role-context",
        "linkedinUrl": "https://linkedin.com/in/casey-rivera",
        "firstName": "Casey",
        "lastName": "Rivera",
        "country": "United States",
        "region": "Washington",
        "city": "Seattle",
        "workEmail": "casey@example.test",
        "currentPosition": {
            "title": title,
            "description": duties,
            "company": {
                "name": "Example Systems",
                "domain": "example.test",
                "linkedinId": "example-systems",
            },
            "isCurrent": True,
        },
    }


def _source(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "harvestapi",
        "tool": "harvestapi_get_profile",
        "input": {"url": "https://linkedin.com/in/casey-rivera"},
        "response": {
            "status": "completed",
            "result": {"data": {"elements": [profile]}},
        },
        "call_identity": "broker-role-context",
    }


class _EmailValid:
    async def __call__(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert tool == "zerobounce_validate"
        assert payload == {"email": "casey@example.test"}
        return {"status": "completed", "result": {"data": {"status": "valid"}}}


def _verify(
    *,
    claimed_role: str = "Workplace Manager",
    profile_title: str = "Workplace Manager",
    duties: str,
    classify_role: Any,
) -> dict[str, Any]:
    return asyncio.run(
        verify_contact(
            _company(claimed_role),
            _icp(),
            source_evidence=_source(_profile(title=profile_title, duties=duties)),
            execute=_EmailValid(),
            classify_role=classify_role,
        )
    )


def test_verified_current_job_duties_reach_the_semantic_role_judge() -> None:
    observed: list[tuple[str, list[str], str, str]] = []
    duties = (
        "Leads workplace operations, site readiness, vendor budgets, service delivery, "
        "and office-opening readiness frameworks."
    )

    async def judge(role: str, targets: list[str], seniority: str, context: str) -> bool:
        observed.append((role, targets, seniority, context))
        return "office-opening" in context and "workplace operations" in context

    result = _verify(duties=duties, classify_role=judge)

    assert result["contact_qualified"] is True
    assert observed == [("Workplace Manager", ["Operations Manager"], "Manager", duties)]


def test_unrelated_workplace_duties_remain_rejected() -> None:
    async def judge(_role: str, _targets: list[str], _seniority: str, context: str) -> bool:
        return "site readiness" in context or "office opening" in context

    result = _verify(
        duties="Schedules social events and orders employee snacks.",
        classify_role=judge,
    )

    assert result["contact_qualified"] is False
    assert result["contact_verification"]["reason"] == "contact_role_not_targeted"


def test_missing_contact_never_reaches_the_role_judge() -> None:
    called = False

    async def judge(*_args: object) -> bool:
        nonlocal called
        called = True
        return True

    company = _company()
    company["contact"] = None
    result = asyncio.run(
        verify_contact(
            company,
            _icp(),
            source_evidence=_source(
                _profile(duties="Leads operations and office openings.")
            ),
            execute=_EmailValid(),
            classify_role=judge,
        )
    )

    assert called is False
    assert result["contact_qualified"] is False
    assert result["contact_verification"]["reason"] == "contact_claim_invalid"


def test_duties_cannot_override_the_claimed_title_check() -> None:
    called = False

    async def judge(*_args: object) -> bool:
        nonlocal called
        called = True
        return True

    result = _verify(
        claimed_role="Operations Manager",
        profile_title="Workplace Manager",
        duties="Leads all operations and new office openings.",
        classify_role=judge,
    )

    assert called is False
    assert result["contact_verification"]["reason"] == "contact_role_mismatch"


def test_duties_cannot_override_seniority() -> None:
    called = False

    async def judge(*_args: object) -> bool:
        nonlocal called
        called = True
        return True

    result = _verify(
        claimed_role="Workplace Coordinator",
        profile_title="Workplace Coordinator",
        duties="Coordinates operations and new office openings.",
        classify_role=judge,
    )

    assert called is False
    assert result["contact_verification"]["reason"] == "contact_role_not_targeted"


def test_role_duties_are_bounded_in_cache_semantics_and_prompt() -> None:
    duties = "x" * 5_000
    semantics = contact_source_semantics(_source(_profile(duties=duties)))
    projected = semantics["response"]["profiles"][0]["positions"][0]["duties"]
    prompt = _build_prompt(
        ["Operations Manager"],
        [{"id": "contact", "role": "Workplace Manager", "duties": duties}],
    )

    assert len(projected) == 4_000
    assert "x" * 4_000 in prompt
    assert "x" * 4_001 not in prompt
