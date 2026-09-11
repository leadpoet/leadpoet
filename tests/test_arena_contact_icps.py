import asyncio
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gateway.qualification.models import ICPPrompt
from gateway.tasks import icp_generator
from lab_arena import confirmation, contact_policy, integrity
from qualification.scoring.contact_verification import _deterministic_role_match


def _icp(role: str = "VP Engineering") -> dict:
    return {
        "icp_id": "icp-main",
        "prompt": "Find growing software companies with a recent product launch.",
        "industry": "Software",
        "sub_industry": "SaaS",
        "target_roles": [role],
        "target_seniority": "VP+",
        "contact_geography": {"countries": ["US"], "regions": [], "cities": []},
        "employee_count": ["51-200"],
        "company_stage": "Series A",
        "geography": "United States",
        "country": "United States",
        "product_service": "Revenue operations software",
        "required_attribute": "Sells subscription software",
        "intent_signal": "Launched a major product capability in the last year",
        "intent_signals": ["Launched a major product capability in the last year"],
        "intent_category": "PRODUCT_LAUNCH",
        "intent_max_age_days": 365,
        "max_companies": 5,
    }


def test_contact_projection_is_opt_in_and_legacy_fingerprint_is_unchanged() -> None:
    base = _icp()
    changed = deepcopy(base)
    changed["target_roles"] = ["Chief Information Officer"]
    changed["target_seniority"] = "C-Suite"
    changed["contact_geography"] = {"countries": ["CA"], "regions": [], "cities": []}

    legacy = integrity.agent_visible_icp(base)
    assert "contact_policy" not in legacy
    assert "target_roles" not in legacy
    assert integrity.requirement_fingerprint(base) == integrity.requirement_fingerprint(changed)

    projected = integrity.agent_visible_icp(base, contacts_required=True)
    assert projected["contact_policy"] == "contacts_v1"
    assert projected["target_roles"] == ["VP Engineering"]
    assert projected["contact_geography"]["countries"] == ["US"]
    assert integrity.requirement_fingerprint(
        base, contacts_required=True
    ) != integrity.requirement_fingerprint(changed, contacts_required=True)


def test_confirmation_dedup_includes_contact_requirements_only_when_enabled() -> None:
    main = [_icp()]
    candidates = []
    for index in range(5):
        candidate = _icp(f"Contact role {index}")
        candidate["icp_id"] = f"candidate-{index}"
        candidates.append(candidate)

    with pytest.raises(ValueError, match="repeats"):
        confirmation.build_bank("round", candidates, main)
    bank = confirmation.build_bank(
        "round", candidates, main, contacts_required=True
    )
    assert [row["target_roles"] for row in bank["icps"]] == [
        [f"Contact role {index}"] for index in range(5)
    ]


def test_contact_icp_model_round_trips_structured_geography() -> None:
    value = _icp()
    value["employee_count"] = "51-200"
    value["contact_policy"] = "contacts_v1"
    model = ICPPrompt.model_validate(value)
    dumped = model.model_dump_json()
    restored = ICPPrompt.model_validate_json(dumped)
    assert restored.contact_policy == "contacts_v1"
    assert restored.contact_geography == {
        "countries": ["US"],
        "regions": [],
        "cities": [],
    }
    with pytest.raises(ValueError):
        ICPPrompt.model_validate({**value, "contact_geography": {"country": ["US"]}})


@pytest.mark.parametrize("invalid", ["missing_roles", "invalid_country"])
def test_confirmation_rejects_incomplete_contact_requirements(invalid) -> None:
    candidates = [_icp(f"Contact role {index}") for index in range(5)]
    if invalid == "missing_roles":
        candidates[2]["target_roles"] = []
    else:
        candidates[2]["contact_geography"]["countries"] = ["not-a-country"]
    with pytest.raises(ValueError):
        confirmation.build_bank("round", candidates, [_icp()], contacts_required=True)


def test_template_contact_requirements_are_coherent_and_do_not_copy_hq() -> None:
    legacy = icp_generator.generate_single_icp("legacy", "Software", seed=7)
    assert legacy["target_roles"] == []
    assert legacy["target_seniority"] == ""
    assert "contact_geography" not in legacy

    contact_icp = icp_generator.generate_single_icp(
        "contact", "Software", seed=7, contacts_required=True
    )
    assert contact_icp["target_roles"]
    assert contact_icp["target_seniority"]
    assert all(role in contact_icp["prompt"] for role in contact_icp["target_roles"])
    assert contact_icp["contact_geography"] == {
        "countries": [],
        "regions": [],
        "cities": [],
    }


@pytest.mark.parametrize(
    ("product", "expected_roles"),
    [
        ("Email-marketing platform", ["Chief Marketing Officer", "VP Marketing"]),
        ("Retail inventory suite", ["Chief Operating Officer", "VP Operations"]),
        ("Email delivery platform", ["VP Engineering", "Head of IT"]),
    ],
)
def test_contact_product_keywords_match_whole_terms(
    product: str, expected_roles: list[str]
) -> None:
    roles, _seniority = icp_generator._contact_requirements_for_icp(
        {"product_service": product}, industry="Software"
    )
    assert roles == expected_roles


def test_generated_contact_icp_is_service_valid_and_storage_keeps_marker(
    monkeypatch,
) -> None:
    generated = icp_generator.canonicalize_generated_icp(
        _icp(),
        industry="Software",
        sub_industry="SaaS",
        contacts_required=True,
    )
    assert generated["contact_policy"] == "contacts_v1"
    assert contact_policy.validate_icp(generated)["contact_policy"] == "contacts_v1"

    captured = {}

    class Query:
        def upsert(self, value, *, on_conflict):
            assert on_conflict == "set_id"
            captured.update(deepcopy(value))
            return self

        def execute(self):
            return None

    class Client:
        def table(self, name):
            assert name == "qualification_private_icp_sets"
            return Query()

    monkeypatch.setitem(
        sys.modules,
        "gateway.db.client",
        SimpleNamespace(get_write_client=lambda: Client()),
    )
    active_from = datetime(2026, 9, 11, tzinfo=timezone.utc)
    assert asyncio.run(
        icp_generator.store_icp_set(
            set_id=20260911,
            icps=[generated],
            icp_set_hash=icp_generator.compute_icp_set_hash([generated]),
            industry_distribution={"Software": 1},
            active_from=active_from,
            active_until=active_from + timedelta(days=1),
        )
    )
    assert captured["icps"][0]["contact_policy"] == "contacts_v1"


@pytest.mark.parametrize(
    ("roles", "seniority", "expected_seniority"),
    [
        (["Head of Revenue Operations"], "VP+", ""),
        (["Director of Engineering"], "VP+", ""),
        (["Revenue Operations Manager"], "Director+", ""),
        (["Revenue Operations Lead"], "Director+", ""),
        (["Chief Revenue Officer", "VP Sales"], "VP+", "VP+"),
        (
            ["Head of Engineering", "Director of Engineering"],
            "Director+",
            "Director+",
        ),
        (["Customer Success Manager"], "Manager", "Manager"),
    ],
)
def test_generated_contact_seniority_never_rejects_an_explicit_role(
    roles: list[str],
    seniority: str,
    expected_seniority: str,
) -> None:
    prompt = "Find software companies whose buying process documents VP+ approval."
    source = _icp()
    source.update(
        {
            "prompt": prompt,
            "target_roles": roles,
            "target_seniority": seniority,
        }
    )

    generated = icp_generator.canonicalize_generated_icp(
        source,
        industry="Software",
        sub_industry="SaaS",
        contacts_required=True,
    )

    assert generated["target_roles"] == roles
    assert generated["target_seniority"] == expected_seniority
    assert generated["prompt"].startswith(prompt)
    assert "()" not in generated["prompt"]
    assert all(
        _deterministic_role_match(role, roles, generated["target_seniority"])
        is True
        for role in roles
    )


def test_legacy_canonicalization_does_not_repair_contact_fields() -> None:
    source = _icp("Head of Revenue Operations")
    source["target_seniority"] = "VP+"

    generated = icp_generator.canonicalize_generated_icp(
        source,
        industry="Software",
        sub_industry="SaaS",
    )

    assert generated["target_roles"] == ["Head of Revenue Operations"]
    assert generated["target_seniority"] == "VP+"
    assert generated["prompt"] == source["prompt"]


def test_contact_generation_prompt_keeps_natural_prompt_company_only(
    monkeypatch,
) -> None:
    captured = {}

    class Response:
        status_code = 503

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, **kwargs):
            captured.update(kwargs["json"])
            return Response()

    monkeypatch.setattr(icp_generator.httpx, "AsyncClient", Client)

    assert asyncio.run(
        icp_generator.generate_icps_with_openrouter(
            set_id=20260912,
            total_icps=1,
            api_key="test-only",
            contacts_required=True,
        )
    ) is None
    system_prompt = captured["messages"][0]["content"]
    assert "Never use job titles, seniority levels" in system_prompt
    assert "Keep the natural-language `prompt` company-only" in system_prompt
    assert "The natural-language `prompt` must name the same roles" not in system_prompt


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_contact_generation_environment_opt_in(monkeypatch, value: str) -> None:
    monkeypatch.setenv("LAB_ARENA_CONTACTS_GENERATION_ENABLED", value)
    assert icp_generator.contacts_generation_enabled()
