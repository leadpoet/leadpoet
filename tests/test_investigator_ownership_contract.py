"""Source-bound ownership contracts in the existing investigator only.

These controls test contract validation, not the model's semantic accuracy.
"""
from __future__ import annotations

import asyncio
import copy
import json

import pytest

from lab_arena.operations import validate_operation_request
from lab_arena.broker import _complete_declared_tool_calls
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    COMPANY_FIT_MATCH,
    COMPANY_FIT_MISMATCH,
    COMPANY_FIT_UNAVAILABLE,
    _company_name,
)


CONTROL_URL = "https://clear.example/ownership"
CLASSIFICATION_URL = "https://pinecrest.example/about"
ALIAS_URL = "https://pinecrest.example/identity"
CONTROL = "Pinecrest Capital holds 70 percent of all votes at Clear Capital."
CLASSIFICATION = "Pinecrest Capital is a private equity firm."
LITERAL = "Clear Capital is controlled by Pinecrest Capital, a private equity firm."
OWNERSHIP = {
    "controlling_owner": "Pinecrest Capital",
    "control_status": "current_control",
    "owner_type": "private_equity",
}


def finding(**overrides):
    raw = {
        "target": "stage", "status": "VERIFIED", "observed_value": "Private Equity",
        "evidence_url": CONTROL_URL, "evidence_quote": CONTROL,
        "supporting_evidence_url_1": CLASSIFICATION_URL,
        "supporting_evidence_quote_1": CLASSIFICATION,
        "supporting_evidence_url_2": "", "supporting_evidence_quote_2": "",
        "ownership": dict(OWNERSHIP), "reason": "Current voting control and same-owner PE identity.",
    }
    raw.update(overrides)
    return raw


def validate(raw, *, pages=None, company="Clear Capital", **kwargs):
    claims = investigator._validated_findings(
        {"findings": [raw]}, targets=(raw["target"],),
        fetched_pages=pages if pages is not None else {
            CONTROL_URL: raw["evidence_quote"], CLASSIFICATION_URL: CLASSIFICATION,
        },
        first_party_domains={"clear.example"},
        identity_names={_company_name(company)}, **kwargs,
    )
    assert claims is not None
    return claims[raw["target"]]


def project(claim, requested="Private Equity"):
    verdict = {
        "observed_company_stage": claim["observed_value"],
        "stage_matches": requested == "Private Equity",
        "dimension_evidence": {"stage": {
            "url": claim["evidence_url"], "quote": claim["evidence_quote"],
            "supporting_evidence": claim["supporting_evidence"],
        }},
    }
    return lead_scorer._decision_from_observed_stage(
        verdict, lead_scorer._normalize_company_stage(requested),
        validated_stage_finding=claim,
    )


def test_ownership_schema_is_nullable_complete_and_only_added_for_stage():
    def item(targets):
        return investigator._tools(targets)[-1]["parameters"]["properties"]["findings"]["items"]

    stage = item(("stage",))
    schema = stage["properties"]["ownership"]
    assert schema["type"] == ["object", "null"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(OWNERSHIP)
    assert "ownership" in stage["required"]
    assert "ownership" not in item(("industry",))["properties"]
    assert "ownership" not in item(("industry",))["required"]
    assert "ownership" in item(("stage", "industry"))["properties"]


def test_stage_ownership_tool_fits_existing_broker_depth_and_completed_shape():
    tools = [{
        "type": "function", "function": {key: value for key, value in tool.items() if key != "type"},
    } for tool in investigator._tools(("stage",))]
    request = {
        "model": investigator.INVESTIGATOR_MODEL,
        "messages": [
            {"role": "system", "content": investigator._SYSTEM_PROMPT},
            {"role": "user", "content": "Bounded test request."},
        ],
        "tools": tools, "tool_choice": "required", "max_tokens": 3000,
    }
    validate_operation_request("openrouter.chat", request)
    schema = tools[-1]["function"]["parameters"]["properties"]["findings"]["items"]
    raw = {key: "" for key in schema["required"]}
    raw.update(finding())
    raw["activity_role"] = "unresolved"
    message = {"tool_calls": [{
        "id": "bounded-ownership", "type": "function",
        "function": {"name": "submit_findings", "arguments": json.dumps({"findings": [raw]})},
    }]}
    assert _complete_declared_tool_calls(message, request)


@pytest.mark.parametrize("quote", [
    CONTROL,
    "Pinecrest-Capital holds 70 percent of all votes at Clear—Capital.",
    "Pinecrest Capital holds 70 percent of all votes at Clear   Capital.",
])
def test_semantic_ownership_binds_compact_multiword_company_and_punctuation(quote):
    assert not lead_scorer._stage_quote_supports_observation("private equity", quote)
    claim = validate(finding(evidence_quote=quote))
    assert claim["status"] == "VERIFIED"
    assert claim["ownership"] == OWNERSHIP
    assert claim["supporting_evidence"] == [{"url": CLASSIFICATION_URL, "quote": CLASSIFICATION}]
    assert project(claim) == COMPANY_FIT_MATCH


@pytest.mark.parametrize("company", ["Crédit Mutuel", "München Kapital", "Société Générale", "Straße Networks"])
def test_unicode_company_identity_uses_existing_compact_normalization(company):
    quote = f"Pinecrest Capital holds 70 percent of all votes at {company}."
    claim = validate(finding(evidence_quote=quote), company=company)
    assert claim["status"] == "VERIFIED"
    assert claim["ownership"] == OWNERSHIP
    assert project(claim) == COMPANY_FIT_MATCH


@pytest.mark.parametrize("quoted_company", [
    "PréCrédit Mutuel", "Crédit Mutuelle", "Crédit Mutuelx", "Crédit Mutuelé",
    "Crédit Mutuelétude", "Other Company",
])
def test_unicode_company_match_keeps_original_outer_word_boundaries(quoted_company):
    quote = f"Pinecrest Capital holds 70 percent of all votes at {quoted_company}."
    claim = validate(finding(evidence_quote=quote), company="Crédit Mutuel")
    assert claim["status"] == "UNPROVEN"
    assert project(claim) == COMPANY_FIT_UNAVAILABLE


def test_ownership_and_both_exact_support_spans_are_retained_without_mutating_input():
    alias = "Pinecrest Capital operates under the registered name Pinecrest Capital LP."
    raw = finding(supporting_evidence_url_2=ALIAS_URL, supporting_evidence_quote_2=alias)
    before = copy.deepcopy(raw)
    claim = validate(raw, pages={
        CONTROL_URL: CONTROL, CLASSIFICATION_URL: CLASSIFICATION, ALIAS_URL: alias,
    })
    assert claim["status"] == "VERIFIED"
    assert claim["supporting_evidence"] == [
        {"url": CLASSIFICATION_URL, "quote": CLASSIFICATION},
        {"url": ALIAS_URL, "quote": alias},
    ]
    assert claim["ownership"] == OWNERSHIP
    assert raw == before


@pytest.mark.parametrize("ownership", [
    {}, [], "private_equity", False, 1,
    {**OWNERSHIP, "extra": "unsupported"},
    {key: value for key, value in OWNERSHIP.items() if key != "owner_type"},
    {**OWNERSHIP, "controlling_owner": None},
    {**OWNERSHIP, "controlling_owner": []},
    {**OWNERSHIP, "controlling_owner": ""},
    {**OWNERSHIP, "controlling_owner": " Pinecrest Capital"},
    {**OWNERSHIP, "controlling_owner": "x" * 201},
    {**OWNERSHIP, "control_status": ["current_control"]},
    {**OWNERSHIP, "owner_type": {"type": "private_equity"}},
    *[{**OWNERSHIP, "control_status": value} for value in ("minority", "pending", "former", "unproven")],
    *[{**OWNERSHIP, "owner_type": value} for value in ("other", "unproven")],
])
def test_nonnull_malformed_or_unsupported_proof_cannot_escape_through_legacy_literal_gate(ownership):
    assert lead_scorer._stage_quote_supports_observation("private equity", LITERAL)
    claim = validate(finding(ownership=ownership, evidence_quote=LITERAL))
    assert claim["status"] == "UNPROVEN"
    assert "ownership" not in claim
    assert project(claim) == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("override", [
    {"ownership": {**OWNERSHIP, "controlling_owner": "Clear Capital"}},
    {"ownership": {**OWNERSHIP, "controlling_owner": "Other Capital"}},
    {"evidence_quote": CONTROL.replace("Clear Capital", "Different Company")},
    {"evidence_quote": CONTROL.replace("Clear Capital", "Unclear Capitalism")},
    {"evidence_quote": CONTROL.replace("Pinecrest Capital", "Pinecrest Capitalism")},
])
def test_entity_mismatches_and_word_prefix_collisions_reject(override):
    raw = finding(**override)
    claim = validate(raw, pages={
        CONTROL_URL: raw["evidence_quote"],
        CLASSIFICATION_URL: raw["supporting_evidence_quote_1"],
    })
    assert claim["status"] == "UNPROVEN"
    assert project(claim) == COMPANY_FIT_UNAVAILABLE


def test_first_party_domain_cannot_replace_explicit_primary_company_name():
    raw = finding(evidence_quote="Pinecrest Capital holds 70 percent of all our votes.")
    claim = validate(raw, identity_anchor={
        "submitted_domain": "clear.example", "observed_domain": "clear.example",
        "verified_domain": "clear.example",
    })
    assert claim["status"] == "UNPROVEN"


def test_compact_legal_company_identity_cannot_be_its_own_controlling_owner():
    assert not investigator._semantic_ownership_proof(
        {**OWNERSHIP, "controlling_owner": "Clear Capital Inc"},
        "Clear Capital Inc controls Clear Capital Inc.", {"clearcapitalinc"},
    )


@pytest.mark.parametrize("slot", [1, 2])
@pytest.mark.parametrize("defect", ["unloaded_url", "fabricated_quote", "missing_url", "missing_quote", "falsey_type"])
def test_any_invalid_supplied_support_rejects_even_when_primary_passes_legacy(slot, defect):
    raw = finding(evidence_quote=LITERAL)
    raw[f"supporting_evidence_url_{slot}"] = CLASSIFICATION_URL
    raw[f"supporting_evidence_quote_{slot}"] = CLASSIFICATION
    if defect == "unloaded_url":
        raw[f"supporting_evidence_url_{slot}"] = "https://unloaded.example/about"
    elif defect == "fabricated_quote":
        raw[f"supporting_evidence_quote_{slot}"] = "Pinecrest Capital invented an unsupported quote."
    elif defect == "missing_url":
        raw[f"supporting_evidence_url_{slot}"] = ""
    elif defect == "missing_quote":
        raw[f"supporting_evidence_quote_{slot}"] = ""
    else:
        raw[f"supporting_evidence_quote_{slot}"] = False
        raw[f"supporting_evidence_url_{slot}"] = ""
    claim = validate(raw)
    assert claim["status"] == "UNPROVEN"
    assert claim["supporting_evidence"] == []
    assert project(claim) == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("pages", [
    {CLASSIFICATION_URL: CLASSIFICATION},
    {CONTROL_URL: "Different source text.", CLASSIFICATION_URL: CLASSIFICATION},
    {CONTROL_URL: CLASSIFICATION, CLASSIFICATION_URL: CONTROL},
])
def test_primary_quote_requires_exact_loaded_source_at_its_url(pages):
    claim = validate(finding(), pages=pages)
    assert claim["status"] == "UNPROVEN"
    assert project(claim) == COMPANY_FIT_UNAVAILABLE


def test_source_bound_owner_alias_is_judged_by_existing_investigator():
    control = "Pinecrest Capital (Pinecrest) holds 70 percent of all votes at Clear Capital."
    classification = "Pinecrest is a private equity firm."
    raw = finding(
        evidence_quote=control, supporting_evidence_quote_1=classification,
    )
    claim = validate(raw, pages={CONTROL_URL: control, CLASSIFICATION_URL: classification})
    assert claim["status"] == "VERIFIED"
    assert claim["supporting_evidence"] == [{"url": CLASSIFICATION_URL, "quote": classification}]
    assert project(claim) == COMPANY_FIT_MATCH


def test_prompt_assigns_same_owner_classification_to_slot_one_and_existing_model():
    prompt = investigator._SYSTEM_PROMPT
    assert "Supporting quote 1 must classify that owner when the primary does not" in prompt
    assert "Slot 2 is extra alias/context proof" in prompt
    assert "use fetched context to bind owner aliases" in prompt


def test_primary_can_prove_both_links_while_second_slot_retains_additional_context():
    alias = "Pinecrest Capital is the registered manager of Pinecrest Fund IV."
    raw = finding(
        evidence_quote=LITERAL, supporting_evidence_url_1="", supporting_evidence_quote_1="",
        supporting_evidence_url_2=ALIAS_URL, supporting_evidence_quote_2=alias,
    )
    claim = validate(raw, pages={CONTROL_URL: LITERAL, ALIAS_URL: alias})
    assert claim["status"] == "VERIFIED"
    assert claim["ownership"] == OWNERSHIP
    assert claim["supporting_evidence"] == [{"url": ALIAS_URL, "quote": alias}]


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("quote, expected", [(LITERAL, "VERIFIED"), (CONTROL, "UNPROVEN")])
def test_missing_or_null_ownership_keeps_existing_literal_gate(missing, quote, expected):
    raw = finding(ownership=None, evidence_quote=quote, supporting_evidence_url_1="", supporting_evidence_quote_1="")
    if missing:
        del raw["ownership"]
    claim = validate(raw)
    assert claim["status"] == expected
    assert "ownership" not in claim


@pytest.mark.parametrize("ownership", [None, {**OWNERSHIP, "owner_type": "other"}])
def test_ownership_does_not_change_other_canonical_stage(ownership):
    quote = "Clear Capital common stock is listed on NASDAQ under ticker CLEAR."
    raw = finding(observed_value="Public", evidence_quote=quote, ownership=ownership)
    claim = validate(raw)
    assert claim["status"] == "VERIFIED"
    assert claim["observed_value"] == "Public"
    assert "ownership" not in claim


def test_unused_ownership_object_does_not_change_headcount_target():
    quote = "Clear Capital has 120 employees worldwide."
    claim = validate(finding(
        target="headcount", observed_value=120, evidence_quote=quote,
        ownership={"unused": "not a stage finding"},
    ))
    assert claim["status"] == "VERIFIED"
    assert claim["observed_value"] == 120
    assert "ownership" not in claim


def test_proven_pe_can_contradict_a_requested_venture_stage():
    claim = validate(finding(status="CONTRADICTED"))
    assert claim["status"] == "CONTRADICTED"
    assert project(claim, requested="Series B") == COMPANY_FIT_MISMATCH


@pytest.mark.parametrize("corrected", [True, False])
def test_contract_rejection_uses_existing_bounded_correction_and_source_provenance(monkeypatch, corrected):
    requests = []
    searches = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        ownership = dict(OWNERSHIP)
        if len(requests) == 1 or not corrected:
            ownership["control_status"] = "former"
        arguments = {"findings": [finding(ownership=ownership)]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(_session, query, *, key):
        del key
        searches.append(query)
        return {"results": []}

    async def unexpected_fetch(*_args, **_kwargs):
        pytest.fail("The bounded correction must use already loaded source text.")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 1)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    monkeypatch.setattr(investigator, "_fetch_page", unexpected_fetch)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Clear Capital", "website": "https://clear.example"},
        targets=("stage",), requested_stage="Private Equity",
        prior_observations={"submitted_source_urls": [CONTROL_URL, CLASSIFICATION_URL]},
        prefetched_pages={
            CONTROL_URL: {"final_url": CONTROL_URL, "text": CONTROL},
            CLASSIFICATION_URL: {"final_url": CLASSIFICATION_URL, "text": CLASSIFICATION},
        },
    ))
    assert len(requests) == 2, result
    assert len(searches) == 1
    assert result["usage"]["reasoning_turns"] == 2
    assert result["usage"]["search_calls"] == 1
    assert result["usage"]["fetch_calls"] == 0
    assert requests[-1]["tool_choice"]["function"]["name"] == "submit_findings"
    if corrected:
        claim = result["claims"]["stage"]
        assert claim["status"] == "VERIFIED", result
        assert claim["ownership"] == OWNERSHIP
        assert result["_validated_stage_finding"] == claim
        assert result["_completed_submit"] is True
        assert result["failure_reason"] == ""
        retained = result[investigator.PRIVATE_FETCHED_PAGES_KEY]
        assert retained[CONTROL_URL]["text"] == CONTROL
        assert retained[CLASSIFICATION_URL]["text"] == CLASSIFICATION
        assert project(claim) == COMPANY_FIT_MATCH
    else:
        assert result["_validated_stage_finding"] == {}
        assert result["claims"]["stage"]["status"] == "UNPROVEN"
        assert "ownership" not in result["claims"]["stage"]
        assert project(result["claims"]["stage"]) == COMPANY_FIT_UNAVAILABLE


@pytest.mark.parametrize("fail_turn", [1, 2])
def test_provider_failure_cannot_create_ownership_receipt(monkeypatch, fail_turn):
    requests = []

    async def fake_post(_session, _url, *, headers, payload):
        del headers
        requests.append(payload)
        if len(requests) == fail_turn:
            raise RuntimeError("bounded test provider failure")
        arguments = {"findings": [finding(ownership={**OWNERSHIP, "control_status": "former"})]}
        return 200, {"choices": [{"message": {"tool_calls": [{
            "id": str(len(requests)), "type": "function",
            "function": {"name": "submit_findings", "arguments": json.dumps(arguments)},
        }]}}]}

    async def fake_search(*_args, **_kwargs):
        return {"results": []}

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(investigator, "MAX_REASONING_TURNS", 1)
    monkeypatch.setattr(investigator, "MAX_SEARCH_CALLS", 1)
    monkeypatch.setattr(investigator, "_post_json", fake_post)
    monkeypatch.setattr(investigator, "_search_web", fake_search)
    result = asyncio.run(investigator.investigate_company_evidence(
        company_locator={"name": "Clear Capital", "website": "https://clear.example"},
        targets=("stage",), requested_stage="Private Equity",
        prior_observations={"submitted_source_urls": [CONTROL_URL, CLASSIFICATION_URL]},
        prefetched_pages={
            CONTROL_URL: {"final_url": CONTROL_URL, "text": CONTROL},
            CLASSIFICATION_URL: {"final_url": CLASSIFICATION_URL, "text": CLASSIFICATION},
        },
    ))
    assert len(requests) == fail_turn
    assert result == {"claims": {}, "failure_reason": "provider_error"}
