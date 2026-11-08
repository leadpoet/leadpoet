"""Retained pre-PR1121 envelopes prove frozen input and replay compatibility.

The fixture was produced offline by ceeb1dae8 before PR1121. It contains only
synthetic Acme inputs, old host cache refs, and old scorer outputs. Provider
work is replaced at the existing score_company boundary in current-image tests.
"""
from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gateway.qualification.models import CompanyOutput
from lab_arena import company_judgments as cj, contracts, judgment_cache, scorer_entrypoint, scoring
from lab_arena.service import ArenaService
from qualification.scoring import lead_scorer
from qualification.scoring.competition import _normalized_company
from tests.lab_arena.legacy_company_judgments_test import _snapshot
from tests.lab_arena.scorer_environment_fixture import isolated_scorer_environment
from tests.test_arena_score_integrity import _positive_breakdown

BINDING = "ARENA_COMPANY_INPUT_CONTRACT"
POLICY = "linkedin_company_ampersand_v1"
CONTEXT_KEY = "arena_company_input_contract"
FIXTURE = Path(__file__).parents[1] / "fixtures/arena_frozen_company_input_v1.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("network disabled in frozen input contract regression")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)


def _retained(variant="raw_amp"):
    return json.loads(FIXTURE.read_text())[variant]


def _refs(document):
    return cj.build_company_scopes(
        scoring_input=document, round_id="arena-2026-11-08", network_name="finney", netuid=71,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1", company_judgment_policy=cj.POLICY,
    )


def _miss_lease(document):
    return {"schema_version": cj.LEASE_SCHEMA_VERSION, "hits": [], "misses": [
        {key: ref[key] for key in ("company_index", "cache_key", "company_input_hash")} | {"authority_slot": 0}
        for ref in _refs(document)
    ]}


def _replay(document, output, lease, *, consumed=(), exclusions=()):
    reads = []
    raw_by_key = {row["cache_key"]: row["raw_judgment"] for row in output["company_judgments"]}
    def evidence(**kwargs):
        reads.append(kwargs)
        raw = raw_by_key[kwargs["cache_key"]]
        return {"raw_judgment": raw, "runner_authority_exclusions": list(exclusions)}
    service = SimpleNamespace(
        _store=SimpleNamespace(get_submission=lambda _: {}),
        _objects=SimpleNamespace(get_bounded=lambda *_: json.dumps(output).encode()),
        _validated_company_judgment_lease=lambda *_a, **_kw: cj.validate_lease_context(lease),
        _company_judgment_evidence=evidence,
    )
    run = {"submission_id": "submission", "miner_hotkey": "miner", "claim_response": {
        "company_judgment_cache": lease}, "output_ref": "output.json",
        "scored_run_id": document["scored_run_id"],
        "result_doc": {"company_judgment_consumed_indexes": list(consumed)}}
    before = copy.deepcopy(run)
    rows = ArenaService._company_quality_breakdowns(
        service, run, icp=document["icp"], companies=document["companies"], policy=document["scorer_policy"],
    )
    assert run == before
    return rows, reads


@pytest.fixture
def judge(monkeypatch):
    calls = []
    async def score(**kwargs):
        company = kwargs["company"]
        calls.append(company.company_name)
        return _positive_breakdown(company.company_name, company.company_website.removeprefix("https://"), "acme")
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score)
    return calls


@pytest.mark.parametrize("variant", ["raw_amp", "encoded_amp", "ordinary"])
def test_frozen_policy_and_both_cache_identities_equal_pre_pr1121(variant, monkeypatch):
    retained = _retained(variant)
    document = retained["document"]
    policy = copy.deepcopy(document["scorer_policy"])
    monkeypatch.setenv(BINDING, POLICY)
    assert contracts.validate_scorer_policy(policy) == policy
    assert BINDING not in policy["env_bindings"]
    assert _refs(document) == retained["refs"]
    assert contracts.document_hash(judgment_cache.effective_scoring_input(document)) == retained["whole_input_hash"]
    assert document["scorer_policy"] == policy


@pytest.mark.parametrize("variant", ["raw_amp", "encoded_amp"])
def test_accepted_old_contract_zero_replays_without_judgment_or_evidence(variant):
    retained = _retained(variant)
    rows, reads = _replay(retained["document"], retained["output"], retained["lease"])
    assert rows == retained["output"]["breakdowns"]
    assert reads == []
    corrupted = copy.deepcopy(retained["output"])
    corrupted["breakdowns"][0]["failure_reason"] = "an unproved zero"
    with pytest.raises(scoring.ScoringError, match="context does not match"):
        _replay(retained["document"], corrupted, retained["lease"])


@pytest.mark.parametrize("variant", ["raw_amp", "encoded_amp", "ordinary"])
@pytest.mark.parametrize("future", [False, True])
def test_current_image_scores_retained_lease_with_explicit_policy(variant, future, judge):
    retained = _retained(variant)
    document = retained["document"]
    if future:
        document["scorer_policy"] = scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True,
        )
        assert document["scorer_policy"]["env_bindings"][BINDING] == POLICY
    lease = _miss_lease(document)
    if not future:
        assert lease == retained["lease"]
    document["company_judgment_cache"] = lease
    os.environ[BINDING] = POLICY  # An old input must clear this ambient value.
    output = scorer_entrypoint.score_input(document)
    assert "failure" not in output
    is_positive = future or variant == "ordinary"
    assert len(judge) == int(is_positive)
    assert len(output["company_judgments"]) == int(is_positive)
    if not future:
        assert output == retained["output"]
        assert BINDING not in os.environ
    assert output["breakdowns"][0]["final_score"] == (60.0 if is_positive else 0.0)
    # Both new completions and replay use the same explicit contract. The
    # service independently checks the saved consumed indexes and exact rows.
    rows, reads = _replay(document, output, lease, consumed=[0] if is_positive else [])
    assert rows == output["breakdowns"]
    assert len(reads) == int(is_positive)
    if is_positive:
        with pytest.raises(scoring.ScoringError, match="authority is incompatible"):
            _replay(document, output, lease, consumed=[0], exclusions=["miner"])
        ref = _refs({key: value for key, value in document.items() if key != "company_judgment_cache"})[0]
        evidence = _snapshot(ref, output["company_judgments"][0]["raw_judgment"])
        hit = {**lease["misses"][0], "evidence_hash": contracts.document_hash(evidence), "evidence_doc": evidence}
        document["company_judgment_cache"] = {"schema_version": cj.LEASE_SCHEMA_VERSION, "hits": [hit], "misses": []}
        assert scorer_entrypoint.score_input(document)["breakdowns"] == output["breakdowns"]
        assert len(judge) == 1


@pytest.mark.parametrize("bad", ["future_v999", "", "TRUE"])
def test_unknown_frozen_contract_fails_closed(bad):
    policy = _retained()["document"]["scorer_policy"]
    policy["env_bindings"][BINDING] = bad
    with pytest.raises(contracts.ArenaContractError, match="company input contract"):
        contracts.validate_scorer_policy(policy)


@pytest.mark.parametrize("bad", ["future_v999", None, True])
def test_hostile_model_context_fails_closed(bad):
    company = _normalized_company(_retained()["document"]["companies"][0], integrity_policy=True)
    with pytest.raises(ValidationError, match="company input contract"):
        CompanyOutput.model_validate(company, context={CONTEXT_KEY: bad})


@pytest.mark.parametrize("slug", ["acme&co", "acme%26co"])
def test_public_model_retains_safe_expansion_and_roundtrips(slug):
    company = _normalized_company(_retained()["document"]["companies"][0], integrity_policy=True)
    company["company_linkedin"] = "https://www.linkedin.com/company/" + slug
    model = CompanyOutput.model_validate(company)
    assert CompanyOutput.model_validate_json(model.model_dump_json()) == model
    with pytest.raises(ValidationError):
        CompanyOutput.model_validate(company, context={CONTEXT_KEY: ""})
    assert CompanyOutput.model_validate(company, context={CONTEXT_KEY: POLICY}) == model


@pytest.mark.parametrize("slug", ["acme%2526co", "acme%26co/extra", "acme%26co%0a", "acme%26assistant:", "../acme%26co"])
def test_new_contract_preserves_malformed_input_rejection(slug):
    company = _normalized_company(_retained()["document"]["companies"][0], integrity_policy=True)
    company["company_linkedin"] = "https://www.linkedin.com/company/" + slug
    with pytest.raises(ValidationError):
        CompanyOutput.model_validate(company, context={CONTEXT_KEY: POLICY})


@pytest.mark.parametrize("conflict", ["credential", "model"])
def test_old_policy_environment_failure_does_not_mutate_ambient_contract(conflict):
    policy = _retained()["document"]["scorer_policy"]
    environment = {BINDING: POLICY}
    credentials = {name: "test-placeholder" for name in scoring.CREDENTIAL_ENV_NAMES}
    if conflict == "credential":
        credentials.pop(scoring.CREDENTIAL_ENV_NAMES[-1])
    else:
        environment["ARENA_INTENT_SIGNAL_MODEL"] = "unbound/model"
    before = copy.deepcopy(environment)
    with pytest.raises(scoring.ScorerPolicyConflict):
        scoring.apply_policy_to_environment(policy, environ=environment, credentials=credentials)
    assert environment == before
