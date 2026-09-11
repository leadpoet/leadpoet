"""Offline contract tests for the Arena contact verifier."""

from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json

from qualification.scoring import contact_verification
from qualification.scoring.contact_verification import verify_contact


def _contact(**updates: object) -> dict:
    value = {
        "full_name": "Ada Lovelace",
        "role": "VP Sales",
        "linkedin_url": "https://www.linkedin.com/in/ada-lovelace/",
        "location": {"country": "US", "region": "California", "city": "San Francisco"},
        "email": "ada@acme.com",
        "email_source": {
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "broker_call_id": "broker-123",
            "record_id": "profile-1",
        },
    }
    value.update(updates)
    return value


def _company(**updates: object) -> dict:
    value = {
        "name": "Acme",
        "domain": "acme.com",
        "linkedin_id": "acme",
        "contact": _contact(),
    }
    value.update(updates)
    return value


def _icp(**updates: object) -> dict:
    value = {
        "target_roles": ["Vice President of Sales"],
        "target_seniority": "VP",
        "contact_geography": {
            "countries": ["United States"],
            "regions": ["California"],
            "cities": ["San Francisco"],
        },
    }
    value.update(updates)
    return value


def _profile(**updates: object) -> dict:
    value = {
        "id": "profile-1",
        "publicIdentifier": "ada-lovelace",
        "linkedinUrl": "https://linkedin.com/in/ada-lovelace",
        "firstName": "Ada",
        "lastName": "Lovelace",
        "country": "United States",
        "region": "California",
        "city": "San Francisco",
        "workEmail": "ada@acme.com",
        "currentPosition": {
            "title": "Vice President of Sales",
            "company": {
                "name": "Acme",
                "domain": "acme.com",
                "linkedinId": "acme",
            },
            "isCurrent": True,
        },
    }
    value.update(updates)
    return value


def _source(profile: dict | None = None, **updates: object) -> dict:
    value = {
        "provider": "harvestapi",
        "tool": "harvestapi_get_profile",
        "input": {"url": "https://linkedin.com/in/ada-lovelace", "findEmail": "true"},
        "response": {
            "status": "completed",
            "result": {"data": {"elements": [profile or _profile()]}},
        },
        "call_identity": "broker-123",
        "observed_at": "2026-09-11T12:00:00Z",
    }
    value.update(updates)
    return value


class ScriptedExecute:
    def __init__(self, scripts: dict[str, list[object]]) -> None:
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, tool: str, payload: dict) -> object:
        self.calls.append((tool, dict(payload)))
        values = self.scripts.get(tool)
        if not values:
            raise AssertionError(f"unexpected provider call: {tool}")
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _run(*, company: dict | None = None, icp: dict | None = None, source: dict | None = None, execute: ScriptedExecute | None = None) -> dict:
    return asyncio.run(
        verify_contact(
            company or _company(),
            icp or _icp(),
            source_evidence=source if source is not None else _source(),
            execute=execute,
        )
    )


def _zero(status: str) -> dict:
    return {"status": "completed", "result": {"data": {"status": status}}}


def test_default_execute_request_matches_the_closed_arena_operation(monkeypatch) -> None:
    from lab_arena import operations

    captured: dict = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"status": "completed", "result": {"data": {"status": "valid"}}}

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, url: str, *, json: dict, headers: dict):
            captured.update({"url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(contact_verification.httpx, "AsyncClient", Client)

    result = asyncio.run(
        contact_verification._default_execute(
            "zerobounce_validate", {"email": "ada@acme.com"}
        )
    )
    operation_id, parameters = operations.match_request(
        "POST",
        captured["url"],
        json.dumps(captured["json"]).encode("utf-8"),
        {},  # the sandbox runtime consumes the scorer's dummy credential
    )

    assert result["status"] == "completed"
    assert captured["headers"] == {
        "Authorization": "Bearer arena-trusted-scorer"
    }
    assert operation_id == "deepline.execute"
    assert parameters == {
        "tool": "zerobounce_validate",
        "payload": {"email": "ada@acme.com"},
    }


def test_catch_all_email_is_a_qualified_person_contact() -> None:
    execute = ScriptedExecute({"zerobounce_validate": [_zero("catch_all")]})

    result = _run(execute=execute)

    assert result["contact_qualified"] is True
    assert result["email_status"] == "catch_all"
    assert result["contact_verification"]["decision"] == "verified"
    assert result["contact_verification"]["reason"] == "contact_verified"
    assert result["contact_identity_key"].startswith("contact:")
    assert [call[0] for call in execute.calls] == ["zerobounce_validate"]
    assert "failure_class" not in result["verifier_gate_receipts"][0]


def test_valid_with_accept_all_substatus_is_reported_as_catch_all() -> None:
    execute = ScriptedExecute(
        {
            "zerobounce_validate": [
                {
                    "status": "completed",
                    "result": {
                        "data": {
                            "status": "valid",
                            "sub_status": "catchall_domain",
                            "address": "ada@acme.com",
                        }
                    },
                }
            ]
        }
    )

    result = _run(execute=execute)

    assert result["contact_qualified"] is True
    assert result["email_status"] == "catch_all"


def test_unsafe_email_evidence_overrides_valid_or_deliverable_status() -> None:
    zero = ScriptedExecute(
        {
            "zerobounce_validate": [
                {
                    "status": "valid",
                    "address": "ada@acme.com",
                    "do_not_mail": True,
                }
            ]
        }
    )
    bounce = ScriptedExecute(
        {
            "zerobounce_validate": [{"status": "unknown", "address": "ada@acme.com"}],
            "bounceban_verify_single": [
                {
                    "status": "success",
                    "result": "deliverable",
                    "email": "ada@acme.com",
                    "is_disposable": True,
                }
            ],
        }
    )

    zero_result = _run(execute=zero)
    bounce_result = _run(execute=bounce)

    assert zero_result["email_status"] == "invalid"
    assert zero_result["contact_verification"]["decision"] == "mismatch"
    assert bounce_result["email_status"] == "invalid"
    assert bounce_result["contact_verification"]["decision"] == "mismatch"


def test_provider_returned_email_must_match_the_requested_email() -> None:
    zero = ScriptedExecute(
        {
            "zerobounce_validate": [
                {"status": "valid", "address": "other@acme.com"}
            ]
        }
    )
    bounce = ScriptedExecute(
        {
            "zerobounce_validate": [
                {"status": "unknown", "address": "ada@acme.com"}
            ],
            "bounceban_verify_single": [
                {
                    "status": "success",
                    "result": "deliverable",
                    "email": "other@acme.com",
                }
            ],
        }
    )

    for result in (_run(execute=zero), _run(execute=bounce)):
        assert result["contact_verification"]["decision"] == "unavailable"
        assert result["contact_verification"]["reason"] == (
            "contact_provider_malformed_response"
        )
        assert result["verifier_gate_receipts"][0]["failure_class"] == (
            "contact_provider_error"
        )


def test_provider_person_name_mismatch_rejects_before_email_validation() -> None:
    execute = ScriptedExecute({})

    result = _run(source=_source(_profile(firstName="Grace", lastName="Hopper")), execute=execute)

    assert result["contact_qualified"] is False
    assert result["contact_verification"]["decision"] == "mismatch"
    assert result["contact_verification"]["reason"] == "contact_person_mismatch"
    assert execute.calls == []


def test_only_current_matching_employer_and_role_are_used() -> None:
    profile = _profile(
        currentPosition=None,
        experience=[
            {
                "title": "Vice President of Sales",
                "company": {"name": "Acme", "domain": "acme.com"},
                "endDate": {"year": 2024},
            },
            {
                "title": "Chief Technology Officer",
                "company": {"name": "OtherCo", "domain": "other.example"},
                "endDate": None,
            },
        ],
    )

    result = _run(source=_source(profile), execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "mismatch"
    assert result["contact_verification"]["reason"] == "contact_company_mismatch"


def test_exact_email_must_belong_to_the_matching_profile() -> None:
    result = _run(
        source=_source(_profile(workEmail="someone-else@acme.com")),
        execute=ScriptedExecute({}),
    )

    assert result["contact_verification"]["decision"] == "mismatch"
    assert result["contact_verification"]["reason"] == "contact_email_mismatch"


def test_country_or_supported_city_mismatch_rejects() -> None:
    country = _run(source=_source(_profile(country="Canada")), execute=ScriptedExecute({}))
    city = _run(source=_source(_profile(city="Los Angeles")), execute=ScriptedExecute({}))

    assert country["contact_verification"]["reason"] == "contact_location_mismatch"
    assert city["contact_verification"]["reason"] == "contact_location_mismatch"


def test_missing_provider_location_is_unverified_not_a_mismatch() -> None:
    profile = _profile(country=None, region=None, city=None)

    result = _run(source=_source(profile), execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "unverified"
    assert result["contact_verification"]["reason"] == "contact_location_unverified"
    assert "failure_class" not in result["verifier_gate_receipts"][0]


def test_missing_claimed_city_is_unverified_even_without_icp_city_filter() -> None:
    profile = _profile(city=None)
    icp = _icp(contact_geography={"countries": ["United States"]})

    result = _run(icp=icp, source=_source(profile), execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "unverified"
    assert result["contact_verification"]["reason"] == "contact_location_unverified"


def test_country_names_and_iso_codes_use_the_shared_normalizer() -> None:
    company = _company()
    company["contact"] = _contact(
        location={"country": "DE", "region": "Berlin", "city": "Berlin"}
    )
    profile = _profile(country="Germany", region="Berlin", city="Berlin")
    icp = _icp(
        contact_geography={
            "countries": ["Germany"],
            "regions": ["Berlin"],
            "cities": ["Berlin"],
        }
    )
    execute = ScriptedExecute({"zerobounce_validate": [_zero("valid")]})

    result = _run(company=company, icp=icp, source=_source(profile), execute=execute)

    assert result["contact_qualified"] is True


def test_same_company_name_cannot_override_linkedin_contradiction() -> None:
    profile = _profile(
        currentPosition={
            "title": "Vice President of Sales",
            "companyName": "Acme",
            "companyLinkedinUrl": "https://linkedin.com/company/unrelated/",
            "isCurrent": True,
        }
    )

    result = _run(source=_source(profile), execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "mismatch"
    assert result["contact_verification"]["reason"] == "contact_company_mismatch"


def test_employer_uses_registrable_domain_without_suffix_confusion() -> None:
    company = _company(name="Example", domain="example.com", linkedin_id=None)
    matching = _profile(
        currentPosition={
            "title": "Vice President of Sales",
            "company": {"name": "Example", "domain": "news.example.com"},
            "isCurrent": True,
        }
    )
    attacker = _profile(
        currentPosition={
            "title": "Vice President of Sales",
            "company": {
                "name": "Example",
                "domain": "example.com.attacker.test",
            },
            "isCurrent": True,
        }
    )
    execute = ScriptedExecute({"zerobounce_validate": [_zero("valid")]})

    matching_result = _run(
        company=company,
        source=_source(matching),
        execute=execute,
    )
    attacker_result = _run(
        company=company,
        source=_source(attacker),
        execute=ScriptedExecute({}),
    )

    assert matching_result["contact_qualified"] is True
    assert attacker_result["contact_verification"]["decision"] == "mismatch"
    assert attacker_result["contact_verification"]["reason"] == (
        "contact_company_mismatch"
    )


def test_semantic_role_judge_outage_is_retryable_unavailable() -> None:
    company = _company()
    company["contact"] = _contact(role="Vice President of Customer Growth")
    profile = _profile(
        currentPosition={
            "title": "Vice President of Customer Growth",
            "company": {"name": "Acme", "domain": "acme.com"},
            "isCurrent": True,
        }
    )

    async def unavailable(*_args: object) -> bool:
        raise RuntimeError("judge transport failed")

    result = asyncio.run(
        verify_contact(
            company,
            _icp(),
            source_evidence=_source(profile),
            execute=ScriptedExecute({}),
            classify_role=unavailable,
        )
    )

    assert result["contact_verification"]["decision"] == "unavailable"
    assert result["contact_verification"]["reason"] == "contact_role_provider_error"
    assert result["verifier_gate_receipts"][0]["failure_class"] == "contact_provider_error"


def test_nonexact_executive_titles_do_not_use_a_coarse_fast_path() -> None:
    cases = [
        ("Chief Technology Officer", ["Chief Information Security Officer"], "C-level"),
        ("Executive Assistant to CEO", ["Chief Executive Officer"], "C-level"),
        ("Vice President of Sales", ["Vice President of Sales"], "Director"),
    ]
    for role, target_roles, target_seniority in cases:
        company = _company()
        company["contact"] = _contact(role=role)
        profile = _profile(
            currentPosition={
                "title": role,
                "company": {"name": "Acme", "domain": "acme.com"},
                "isCurrent": True,
            }
        )
        result = _run(
            company=company,
            icp=_icp(
                target_roles=target_roles,
                target_seniority=target_seniority,
            ),
            source=_source(profile),
            execute=ScriptedExecute({}),
        )
        assert result["contact_qualified"] is False
        assert result["contact_verification"]["reason"] == "contact_role_not_targeted"


def test_seniority_plus_is_a_threshold() -> None:
    company = _company()
    company["contact"] = _contact(role="Chief Revenue Officer")
    profile = _profile(
        currentPosition={
            "title": "Chief Revenue Officer",
            "company": {"name": "Acme", "domain": "acme.com"},
            "isCurrent": True,
        }
    )
    execute = ScriptedExecute({"zerobounce_validate": [_zero("valid")]})

    result = _run(
        company=company,
        icp=_icp(
            target_roles=["Chief Revenue Officer"],
            target_seniority="VP+",
        ),
        source=_source(profile),
        execute=execute,
    )

    assert result["contact_qualified"] is True


def test_managing_partner_is_executive_but_assistant_is_not() -> None:
    executive_company = _company()
    executive_company["contact"] = _contact(role="Managing Partner")
    executive_profile = _profile(
        currentPosition={
            "title": "Managing Partner",
            "company": {"name": "Acme", "domain": "acme.com"},
            "isCurrent": True,
        }
    )
    executive = _run(
        company=executive_company,
        icp=_icp(target_roles=["Managing Partner"], target_seniority="VP+"),
        source=_source(executive_profile),
        execute=ScriptedExecute({"zerobounce_validate": [_zero("valid")]}),
    )

    assistant_company = _company()
    assistant_company["contact"] = _contact(role="Assistant to Managing Partner")
    assistant_profile = _profile(
        currentPosition={
            "title": "Assistant to Managing Partner",
            "company": {"name": "Acme", "domain": "acme.com"},
            "isCurrent": True,
        }
    )
    assistant = _run(
        company=assistant_company,
        icp=_icp(target_roles=["Managing Partner"], target_seniority="VP+"),
        source=_source(assistant_profile),
        execute=ScriptedExecute({}),
    )

    assert executive["contact_qualified"] is True
    assert assistant["contact_qualified"] is False
    assert assistant["contact_verification"]["reason"] == "contact_role_not_targeted"


def test_malformed_source_response_is_retryable_unavailable() -> None:
    source = _source()
    source["response"] = {"status": "completed", "result": {"data": "not-a-profile"}}

    result = _run(source=source, execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "unavailable"
    assert result["contact_verification"]["reason"] == "contact_provider_malformed_response"
    assert result["verifier_gate_receipts"] == [
        {
            "gate": "contact",
            "decision": "unavailable",
            "reason": "contact_provider_malformed_response",
            "failure_class": "contact_provider_error",
        }
    ]


def test_successful_empty_profile_result_is_unverified_not_unavailable() -> None:
    source = _source()
    source["response"] = {
        "status": "completed",
        "result": {"data": {"elements": []}},
    }

    result = _run(source=source, execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "unverified"
    assert result["contact_verification"]["reason"] == "contact_source_not_found"
    assert "failure_class" not in result["verifier_gate_receipts"][0]


def test_successful_provider_not_found_shape_is_unverified() -> None:
    source = _source()
    source["response"] = {
        "status": "completed",
        "result": {
            "data": {
                "status": 404,
                "element": None,
                "error": "profile not found",
            }
        },
    }

    result = _run(source=source, execute=ScriptedExecute({}))

    assert result["contact_verification"]["decision"] == "unverified"
    assert result["contact_verification"]["reason"] == "contact_source_not_found"


def test_unknown_zerobounce_polls_bounceban_then_completed_unknown_is_unverified(monkeypatch) -> None:
    sleeps: list[float] = []

    async def no_wait(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(contact_verification.asyncio, "sleep", no_wait)
    execute = ScriptedExecute(
        {
            "zerobounce_validate": [_zero("unknown")],
            "bounceban_verify_single": [{"status": "verifying", "id": "job-1"}],
            "bounceban_get_single_status": [
                {"status": "pending", "id": "job-1"},
                {"status": "completed", "result": {"status": "unknown"}},
            ],
        }
    )

    result = _run(execute=execute)

    assert result["contact_verification"]["decision"] == "unverified"
    assert result["contact_verification"]["reason"] == "contact_email_unverified"
    assert result["email_status"] == "unknown"
    assert [tool for tool, _ in execute.calls] == [
        "zerobounce_validate",
        "bounceban_verify_single",
        "bounceban_get_single_status",
        "bounceban_get_single_status",
    ]
    assert sleeps == [1.0, 2.0]


def test_retryable_email_timeout_is_retried_without_rechecking_completed_source() -> None:
    execute = ScriptedExecute(
        {"zerobounce_validate": [asyncio.TimeoutError(), _zero("valid")]}
    )

    result = _run(execute=execute)

    assert result["contact_qualified"] is True
    assert Counter(tool for tool, _ in execute.calls) == {"zerobounce_validate": 2}
    assert "source" in result["contact_verification"]["evidence_hashes"]


def test_known_invalid_email_never_falls_back_to_bounceban() -> None:
    execute = ScriptedExecute({"zerobounce_validate": [_zero("invalid")]})

    result = _run(execute=execute)

    assert result["contact_verification"]["decision"] == "mismatch"
    assert result["contact_verification"]["reason"] == "contact_email_invalid"
    assert result["email_status"] == "invalid"
    assert [tool for tool, _ in execute.calls] == ["zerobounce_validate"]


def test_bounceban_risky_only_passes_when_explained_by_catch_all() -> None:
    accepted = ScriptedExecute(
        {
            "zerobounce_validate": [_zero("unknown")],
            "bounceban_verify_single": [
                {"status": "success", "result": "risky", "is_accept_all": True}
            ],
        }
    )
    contradicted = ScriptedExecute(
        {
            "zerobounce_validate": [_zero("unknown")],
            "bounceban_verify_single": [
                {
                    "status": "success",
                    "result": "risky",
                    "is_accept_all": True,
                    "is_disposable": True,
                }
            ],
        }
    )

    accepted_result = _run(execute=accepted)
    contradicted_result = _run(execute=contradicted)

    assert accepted_result["contact_qualified"] is True
    assert accepted_result["email_status"] == "catch_all"
    assert contradicted_result["contact_verification"]["decision"] == "mismatch"
    assert contradicted_result["email_status"] == "invalid"


def test_invalid_broker_reference_and_role_mailbox_cannot_qualify() -> None:
    invalid_source = _source()
    invalid_source.update({"invalid": True, "reason": "email_source_reference_invalid"})
    invalid = _run(source=invalid_source, execute=ScriptedExecute({}))

    role_contact = _contact(email="sales@acme.com")
    role_company = _company(contact=role_contact)
    role_mailbox = _run(company=role_company, execute=ScriptedExecute({}))

    assert invalid["contact_verification"]["decision"] == "mismatch"
    assert invalid["contact_verification"]["reason"] == "email_source_reference_invalid"
    assert role_mailbox["contact_verification"]["reason"] == "contact_role_mailbox"


def test_record_only_evidence_is_refetched_and_must_match_returned_record() -> None:
    company = _company()
    company["contact"] = _contact(
        email_source={
            "provider": "harvestapi",
            "tool": "harvestapi_get_profile",
            "record_id": "profile-1",
        }
    )
    source = _source(response=None)
    source["call_identity"] = {"record_id": "profile-1"}
    execute = ScriptedExecute(
        {
            "harvestapi_get_profile": [
                {"status": "completed", "result": {"data": {"elements": [_profile()]}}}
            ],
            "zerobounce_validate": [_zero("valid")],
        }
    )

    result = _run(company=company, source=source, execute=execute)

    assert result["contact_qualified"] is True
    assert execute.calls[0] == (
        "harvestapi_get_profile",
        {
            "url": "https://linkedin.com/in/ada-lovelace",
            "findEmail": "true",
        },
    )


def test_actual_harvest_profile_shape_verifies_position_location_and_email() -> None:
    profile = {
        "id": "profile-1",
        "publicIdentifier": "ada-lovelace",
        "linkedinUrl": "https://www.linkedin.com/in/ada-lovelace/",
        "firstName": "Ada",
        "lastName": "Lovelace",
        "location": {
            "countryCode": "US",
            "linkedinText": "San Francisco, California, United States",
            "parsed": {
                "city": "San Francisco",
                "country": "US",
                "countryCode": "US",
                "countryFull": "United States",
                "state": "California",
                "regionCode": "CA",
            },
        },
        "currentPosition": [
            {
                "companyId": "acme",
                "companyLinkedinUrl": "https://linkedin.com/company/acme/",
                "companyName": "Acme, Inc.",
                "position": "Vice President of Sales",
                "endDate": None,
            }
        ],
        "emails": [
            {
                "email": "ada@acme.com",
                "catchAllDomain": False,
                "deliverable": True,
                "status": "valid",
            }
        ],
    }
    source = _source()
    source["response"] = {
        "status": "completed",
        "result": {"data": {"status": 200, "element": profile, "error": None}},
    }
    execute = ScriptedExecute({"zerobounce_validate": [_zero("valid")]})

    result = _run(source=source, execute=execute)

    assert result["contact_qualified"] is True
    assert result["contact_verification"]["subchecks"]["company"]["status"] == "pass"


def test_sanitized_tool_response_envelopes_work_for_harvest_and_zerobounce() -> None:
    source = _source()
    source["response"] = {
        "toolResponse": {
            "rawV2": {
                "status": 200,
                "element": _profile(),
                "error": None,
            }
        }
    }
    execute = ScriptedExecute(
        {
            "zerobounce_validate": [
                {
                    "toolResponse": {
                        "raw": {
                            "status": "valid",
                            "address": "ada@acme.com",
                        }
                    }
                }
            ]
        }
    )

    result = _run(source=source, execute=execute)

    assert result["contact_qualified"] is True
    assert result["email_status"] == "valid"


def test_sanitized_tool_response_envelope_works_for_bounceban() -> None:
    execute = ScriptedExecute(
        {
            "zerobounce_validate": [
                {
                    "status": "completed",
                    "result": {
                        "data": {
                            "status": "unknown",
                            "address": "ada@acme.com",
                        }
                    },
                }
            ],
            "bounceban_verify_single": [
                {
                    "toolResponse": {
                        "rawV2": {
                            "status": "success",
                            "result": "deliverable",
                            "email": "ada@acme.com",
                            "is_disposable": False,
                        }
                    }
                }
            ],
        }
    )

    result = _run(execute=execute)

    assert result["contact_qualified"] is True
    assert result["email_status"] == "valid"
