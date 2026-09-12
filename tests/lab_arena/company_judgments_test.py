from __future__ import annotations

import copy

import pytest

from lab_arena import company_judgments, contracts, scorer_entrypoint, scoring


def _icp() -> dict:
    return {
        "icp_id": "i-1",
        "prompt": "Find buyers",
        "industry": "Software",
        "employee_count": ["51-200"],
        "intent_signals": ["Product launch"],
    }


def _company(name: str) -> dict:
    return {
        "company_name": name,
        "company_website": "https://%s.example" % name.casefold(),
        "company_linkedin": (
            "https://www.linkedin.com/company/%s" % name.casefold()
        ),
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "New York",
        "fit_summary": "Software company",
        "fit_evidence_urls": ["https://%s.example/about" % name.casefold()],
        "intent_signals": [{
            "matched_icp_signal": 0,
            "description": "Product launch",
            "url": "https://%s.example/launch" % name.casefold(),
            "date": None,
            "snippet": "A new product",
            "why_now": "Recent launch",
        }],
    }


def _input(*, contacts: bool = False) -> dict:
    companies = [_company(name) for name in ("A", "B", "C", "D", "E")]
    icp = _icp()
    evidence = None
    adapter = "qualification_integrity_v2"
    if contacts:
        adapter = "qualification_contacts_v3"
        icp.update({
            "contact_policy": "contacts_v1",
            "target_roles": ["VP Engineering"],
            "target_seniority": "VP",
            "contact_geography": {
                "countries": ["US"], "regions": [], "cities": [],
            },
        })
        companies[0]["contact"] = {
            "full_name": "Ada Lovelace",
            "role": "VP Engineering",
            "linkedin_url": "https://www.linkedin.com/in/ada-lovelace/",
            "location": {"country": "US"},
            "email": "ada@acme.com",
            "email_source": {
                "provider": "harvestapi",
                "tool": "harvestapi_get_profile",
                "record_id": "profile-ada",
            },
        }
        evidence = {
            "profile-ada": {
                "provider": "harvestapi",
                "tool": "harvestapi_get_profile",
                "input": {
                    "url": "https://www.linkedin.com/in/ada-lovelace/",
                    "findEmail": "true",
                },
                "response": {
                    "data": [{
                        "id": "profile-ada",
                        "linkedinUrl": "https://www.linkedin.com/in/ada-lovelace/",
                        "fullName": "Ada Lovelace",
                        "currentPosition": {
                            "title": "VP Engineering", "companyName": "A",
                        },
                        "email": "ada@acme.com",
                    }]
                },
                "call_identity": "provider-call-1",
                "observed_at": "2026-09-11T00:00:00Z",
            }
        }
    return scoring.build_scoring_input(
        scored_run_id="execute-1",
        icp=icp,
        companies=companies,
        policy=scoring.build_scorer_policy(
            scoring_adapter_version=adapter, company_quality=True
        ),
        evaluation_date="2026-09-11",
        contact_source_evidence=evidence,
    )


def _refs(document: dict) -> list[dict]:
    return company_judgments.build_company_scopes(
        scoring_input=document,
        round_id="arena-2026-09-11",
        network_name="finney",
        netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
        company_quality_policy="company_quality_v1",
    )


def _raw(score: float = 81.0) -> dict:
    return {
        "final_score": score,
        "failure_reason": "",
        "company_identity_key": "domain:acme.com|name:a",
        "company_identity_alias_keys": ["domain:acme.com|name:a"],
        "verifier_gate_receipts": [{
            "gate": "company_fit",
            "decision": "match",
            "dimension_evidence": {
                "identity": {
                    "web_identity_receipt": {
                        "decision": "match",
                        "evidence_source": "company_web_reverification",
                        "observed_name": "Acme",
                        "observed_domain": "acme.com",
                        "observed_linkedin_slug": "acme",
                    }
                }
            },
        }],
        "intent_signals_detail": [],
    }


def _miss(ref: dict, *, slot: int = 0) -> dict:
    return {
        "company_index": ref["company_index"],
        "cache_key": ref["cache_key"],
        "company_input_hash": ref["company_input_hash"],
        "authority_slot": slot,
    }


def _new(ref: dict, *, slot: int = 0, score: float = 81.0) -> dict:
    return {**_miss(ref, slot=slot), "raw_judgment": _raw(score)}


def _evidence(ref: dict, *, slot: int = 0) -> dict:
    return company_judgments.build_evidence_snapshot(
        new_judgment=_new(ref, slot=slot),
        company_ref=ref,
        source_score_run_id="score-1",
        source_scored_run_id="execute-1",
        source_output_ref="arena/round/scores/score-1.json",
        source_output_hash="sha256:" + "b" * 64,
        source_runner_hotkey="validator-a",
        source_claim_request_id="c" * 32,
        source_claim_request_hash="sha256:" + "d" * 64,
        source_lease_generation=2,
        source_completion_request_hash="sha256:" + "e" * 64,
        runner_authority_exclusions=["miner-a", "validator-a"],
    )


def test_company_keys_reuse_four_of_five_after_one_real_change():
    original = _refs(_input())
    changed = _input()
    changed["companies"][2]["state"] = "California"
    updated = _refs(changed)
    assert sum(
        left["cache_key"] == right["cache_key"]
        for left, right in zip(original, updated)
    ) == 4


def test_company_keys_ignore_order_run_metadata_and_unused_prose():
    document = _input()
    original = _refs(document)
    reordered = copy.deepcopy(document)
    reordered["scored_run_id"] = "execute-2"
    reordered["companies"].reverse()
    reordered["companies"][0]["fit_summary"] = "ignored prose"
    reordered["companies"][0]["intent_signals"][0]["why_now"] = "ignored"
    observed = _refs(reordered)
    assert [item["cache_key"] for item in observed] == list(
        reversed([item["cache_key"] for item in original])
    )
    assert [item["company_index"] for item in observed] == list(range(5))


@pytest.mark.parametrize("change", ["date", "policy", "buyer"])
def test_round_date_policy_and_buyer_changes_do_not_reuse(change):
    document = _input()
    original = _refs(document)
    candidate = copy.deepcopy(document)
    if change == "date":
        candidate["evaluation_date"] = "2026-09-12"
    elif change == "policy":
        candidate["scorer_policy"]["fp_penalty_points"] += 1
    else:
        candidate["icp"]["industry"] = "Financial Services"
    assert all(
        left["cache_key"] != right["cache_key"]
        for left, right in zip(original, _refs(candidate))
    )


def test_contact_source_semantics_change_but_transport_metadata_does_not():
    document = _input(contacts=True)
    original = _refs(document)[0]["cache_key"]
    metadata = copy.deepcopy(document)
    metadata["contact_source_evidence"]["profile-ada"].update({
        "call_identity": "provider-call-2",
        "observed_at": "2026-09-12T00:00:00Z",
    })
    assert _refs(metadata)[0]["cache_key"] == original
    semantic = copy.deepcopy(document)
    semantic["contact_source_evidence"]["profile-ada"]["response"]["data"][0][
        "email"
    ] = "different@acme.com"
    assert _refs(semantic)[0]["cache_key"] != original


def test_contact_key_binds_claimed_and_actual_broker_call_identity():
    document = _input(contacts=True)
    contact = document["companies"][0]["contact"]
    contact["email_source"]["broker_call_id"] = "provider-call-1"
    source = document["contact_source_evidence"].pop("profile-ada")
    document["contact_source_evidence"]["provider-call-1"] = source
    original = _refs(document)[0]["cache_key"]

    changed_actual = copy.deepcopy(document)
    changed_actual["contact_source_evidence"]["provider-call-1"][
        "call_identity"
    ] = "provider-call-2"
    assert _refs(changed_actual)[0]["cache_key"] != original

    changed_claim = copy.deepcopy(document)
    changed_claim["companies"][0]["contact"]["email_source"][
        "broker_call_id"
    ] = "provider-call-2"
    changed_claim["contact_source_evidence"]["provider-call-2"] = (
        changed_claim["contact_source_evidence"].pop("provider-call-1")
    )
    changed_claim["contact_source_evidence"]["provider-call-2"][
        "call_identity"
    ] = "provider-call-2"
    assert _refs(changed_claim)[0]["cache_key"] != original


def test_raw_cache_rejects_context_and_provider_failures_but_keeps_negatives():
    negative = _raw(0.0)
    negative["failure_reason"] = "company fit mismatch"
    assert company_judgments.raw_judgment_is_cacheable(negative)
    assert not company_judgments.raw_judgment_is_cacheable({
        **negative, "company_index": 0,
    })
    assert not company_judgments.raw_judgment_is_cacheable({
        **negative, "failure_reason": "provider timeout",
    })
    unavailable = copy.deepcopy(negative)
    unavailable["verifier_gate_receipts"] = [{
        "gate": "contact",
        "decision": "unavailable",
        "failure_class": "contact_provider_error",
    }]
    assert not company_judgments.raw_judgment_is_cacheable(unavailable)
    intent_outage = copy.deepcopy(negative)
    intent_outage["intent_signals_detail"] = [{
        "judge_verdict": {"decision": "rejected_verifier_error"},
    }]
    assert not company_judgments.raw_judgment_is_cacheable(intent_outage)


def test_raw_cache_rejects_positive_fit_with_incomplete_verified_identity():
    incomplete = _raw()
    incomplete["verifier_gate_receipts"][0]["dimension_evidence"]["identity"][
        "web_identity_receipt"
    ]["observed_linkedin_slug"] = ""
    assert not company_judgments.raw_judgment_is_cacheable(incomplete)


def test_duplicate_destination_refs_share_one_miss_and_one_new_judgment():
    refs = _refs(_input())
    duplicate = {**copy.deepcopy(refs[0]), "company_index": 1}
    lease = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": [],
        "misses": [_miss(refs[0]), _miss(duplicate)],
    }
    assert company_judgments.validate_new_company_judgments(
        [_new(refs[0])], lease_context=lease
    )[0]["company_index"] == 0
    with pytest.raises(company_judgments.CompanyJudgmentError, match="every miss"):
        company_judgments.validate_new_company_judgments(
            [], lease_context=lease
        )


def test_evidence_is_immutable_hash_bound_and_exposes_intrinsic_verdict():
    ref = _refs(_input())[0]
    evidence = _evidence(ref, slot=3)
    evidence_hash = contracts.document_hash(evidence)
    assert company_judgments.validate_evidence_snapshot(
        evidence,
        cache_key=ref["cache_key"],
        company_input_hash=ref["company_input_hash"],
        authority_slot=3,
        evidence_hash=evidence_hash,
    )["raw_judgment"] == _raw()
    forged = copy.deepcopy(evidence)
    forged["raw_judgment"]["final_score"] = 99.0
    with pytest.raises(company_judgments.CompanyJudgmentError, match="binding"):
        company_judgments.validate_evidence_snapshot(
            forged,
            cache_key=ref["cache_key"],
            company_input_hash=ref["company_input_hash"],
            authority_slot=3,
            evidence_hash=evidence_hash,
        )


def test_lease_refuses_an_incompatible_or_inconsistent_hit():
    refs = _refs(_input())
    evidence = _evidence(refs[0], slot=0)
    hit = {
        **_miss(refs[0]),
        "evidence_hash": contracts.document_hash(evidence),
        "evidence_doc": evidence,
    }
    lease = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": [hit],
        "misses": [_miss(refs[index]) for index in range(1, 5)],
    }
    assert len(company_judgments.validate_lease_context(lease)["hits"]) == 1
    lease["misses"][0] = {**lease["misses"][0], "cache_key": hit["cache_key"]}
    with pytest.raises(company_judgments.CompanyJudgmentError, match="inconsistent"):
        company_judgments.validate_lease_context(lease)


def test_accepted_hits_do_not_invoke_the_scorer_or_create_new_judgments():
    document = _input()
    refs = _refs(document)
    hits = []
    for ref in refs:
        evidence = _evidence(ref)
        hits.append({
            **_miss(ref),
            "evidence_hash": contracts.document_hash(evidence),
            "evidence_doc": evidence,
        })
    lease = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": hits,
        "misses": [],
    }

    def scorer(*args, **kwargs):
        raise AssertionError("accepted cache hits must not call the scorer")

    scorer.company_quality = True
    scorer.integrity_policy = True
    scorer.contacts_required = False
    breakdowns, new_judgments = scoring.score_quality_work_item(
        {"scored_run_id": document["scored_run_id"]},
        icp=document["icp"],
        companies=document["companies"],
        scorer=scorer,
        cache_context=lease,
    )

    assert len(breakdowns) == 5
    assert new_judgments == []


def test_scorer_entrypoint_dispatches_quality_cache_and_returns_new_rows(
    monkeypatch,
):
    # The real entrypoint runs in a separate process. Restore its trusted-mode
    # environment when this in-process test finishes so later miner tests
    # exercise the miner boundary.
    monkeypatch.setenv(scorer_entrypoint.shim.TRUSTED_SCORER_ENV, "1")
    document = _input()
    ref = _refs(document)[0]
    lease = {
        "schema_version": company_judgments.LEASE_SCHEMA_VERSION,
        "hits": [],
        "misses": [_miss(ref)],
    }
    document["company_judgment_cache"] = lease
    expected_new = [_new(ref)]
    calls = []

    monkeypatch.setattr(
        scorer_entrypoint.scoring,
        "apply_policy_to_environment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scorer_entrypoint.scoring,
        "lab_scorer",
        lambda *args, **kwargs: object(),
    )

    def score_quality(item, **kwargs):
        calls.append((item, kwargs["cache_context"]))
        return ([{"final_score": 81.0}], expected_new)

    monkeypatch.setattr(
        scorer_entrypoint.scoring,
        "score_quality_work_item",
        score_quality,
    )
    output = scorer_entrypoint.score_input(document)
    assert calls == [({"scored_run_id": "execute-1"}, lease)]
    assert output["company_judgments"] == expected_new
