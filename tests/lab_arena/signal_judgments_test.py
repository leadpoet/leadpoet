"""Gateway signal keys exclude sibling evidence and retain dynamic context."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring, signal_judgments
from lab_arena.service import ArenaService
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from tests.lab_arena import legacy_company_judgments_test as legacy
from tests.test_arena_score_integrity import _icp, _public_company


def _scope(
    company, *, signal_index=0, icp=None, identity=None, funding_context=None,
    image_digest=None,
):
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_judgment_cache=True,
    )
    policy["signal_judgment_policy"] = signal_judgments.POLICY
    document = scoring.build_scoring_input(
        scored_run_id="execution", icp=icp or _icp(), companies=[company],
        policy=policy, evaluation_date="2026-10-20",
    )
    return signal_judgments.build_scope(
        scoring_input=document, company_index=0, signal_index=signal_index,
        verified_company_identity=identity or {"decision": "match", "observed_domain": "acme.example"},
        verified_funding_stage_context=funding_context or [], round_id="round",
        network_name="finney", netuid=71,
        scorer_image_digest=image_digest or "sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
    )


def test_signal_key_reuses_a_when_other_submitted_signal_changes():
    first = _public_company("Acme", domain="acme.example")
    first["intent_signals"].append({**first["intent_signals"][0],
                                    "url": "https://acme.example/b"})
    second = deepcopy(first)
    second["intent_signals"][1]["url"] = "https://acme.example/c"
    assert _scope(first)["cache_key"] == _scope(second)["cache_key"]
    second["intent_signals"][0]["snippet"] = "New primary evidence text"
    assert _scope(first)["cache_key"] != _scope(second)["cache_key"]


def test_signal_key_ignores_other_company_prose_but_binds_identity():
    first = _public_company("Acme", domain="acme.example")
    second = deepcopy(first)
    second["fit_summary"] = "Changed fit summary"
    second["industry"] = "Manufacturing"
    assert _scope(first)["cache_key"] == _scope(second)["cache_key"]
    for field, changed in (("company_name", "Other"),
                           ("company_website", "https://other.example"),
                           ("company_linkedin", "https://www.linkedin.com/company/other")):
        variant = deepcopy(first)
        variant[field] = changed
        assert _scope(first)["cache_key"] != _scope(variant)["cache_key"]


def test_original_signal_source_is_credited_through_company_source(monkeypatch):
    scope = _scope(_public_company("Acme", domain="acme.example"))
    original = hotkey("original-signal-judge")
    judgment = {"terminal": "source-grounded"}
    row = {
        "cache_key": scope["cache_key"], "authority_slot": 0,
        "signal_input_hash": scope["signal_input_hash"], "scope_doc": scope,
        "judgment_doc": judgment, "judgment_hash": contracts.document_hash(judgment),
        "source_score_run_id": "original-score", "source_scored_run_id": "original-execute",
        "source_runner_hotkey": original,
        "source_claim_request_hash": "sha256:" + "b" * 64,
        "source_lease_generation": 1,
        "runner_authority_exclusions": [original],
    }
    use = {key: row[key] for key in (
        "cache_key", "authority_slot", "signal_input_hash",
        "source_score_run_id", "judgment_hash",
    )}
    use.update(run_id="company-score", company_index=0, signal_index=0)
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(
        get_signal_judgment_uses=lambda _: [use],
        get_signal_judgment=lambda *_: row,
    )
    service._same_scoring_membership = lambda *_args, **_kwargs: True
    monkeypatch.setattr(signal_judgments, "validate_judgment", lambda value, _: value)
    source_runs = {
        "original-score": {"runner_hotkey": original,
                           "claim_request_hash": "sha256:" + "b" * 64,
                           "lease_generation": 1},
        "original-execute": {"run_id": "original-execute"},
    }
    authorities = {}
    source_ids = set()
    service._add_signal_attribution(
        authorities, source_ids=source_ids,
        company_source_run_id="company-score", company_indexes={0},
        current_score_run_id="public-score", round_id="round",
        runs_by_id=source_runs,
    )
    assert authorities == {original: True}
    assert source_ids == {"original-score"}
    row["source_claim_request_hash"] = "sha256:" + "c" * 64
    with pytest.raises(scoring.ScoringError, match="source is invalid"):
        service._add_signal_attribution(
            {}, source_ids=set(), company_source_run_id="company-score",
            company_indexes={0}, current_score_run_id="public-score",
            round_id="round", runs_by_id=source_runs,
        )


def test_company_hit_resolves_original_company_index_for_signal_attribution(monkeypatch):
    current_document = legacy._input(("B", "A"))
    source_document = legacy._input(("A", "B"))
    current_refs = legacy._refs(current_document)
    source_refs = legacy._refs(source_document)
    hit = {"company_index": 1, "cache_key": current_refs[1]["cache_key"],
           "company_input_hash": current_refs[1]["company_input_hash"],
           "authority_slot": 0,
           "evidence_doc": {
               "cache_key": current_refs[1]["cache_key"],
               "company_input_hash": current_refs[1]["company_input_hash"],
               "source_score_run_id": "company-source",
               "source_scored_run_id": "source-execution",
               "source_runner_hotkey": "company-judge",
           }}
    miss = {"company_index": 0, "cache_key": current_refs[0]["cache_key"],
            "company_input_hash": current_refs[0]["company_input_hash"],
            "authority_slot": 0}
    lease = {"hits": [hit], "misses": [miss]}
    run = {"run_id": "public-score", "scored_run_id": "public-execution",
           "company_judgment_refs": current_refs,
           "claim_response": {"company_judgment_cache": lease},
           "result_doc": {"company_judgment_consumed_indexes": [1]}}
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(get_signal_judgment_uses=lambda _: [])
    service._round = lambda _: {"configuration_doc": {"scorer_policy": {
        "signal_judgment_policy": signal_judgments.POLICY}}}
    service._same_scoring_membership = lambda *_args, **_kwargs: True
    service._validated_attribution_source = lambda **kwargs: kwargs["source_runner_hotkey"]
    seen = []
    def record(attributed, **kwargs):
        seen.append(kwargs["company_indexes"])
        attributed["original-signal-judge"] = True
    service._add_signal_attribution = record
    monkeypatch.setattr(legacy.company_judgments, "validate_lease_context", lambda _: lease)
    ids = set()
    result = service._score_run_attribution(
        run, {}, round_id="arena-2026-10-20",
        runs_by_id={"company-source": {"company_judgment_refs": source_refs}},
        source_ids=ids,
    )
    assert seen == [{0}]
    assert result == {"company-judge": True, "original-signal-judge": True}
    assert ids == {"company-source"}


@pytest.mark.parametrize("source_status", ["leased", "failed"])
def test_terminal_signal_source_membership_survives_source_run_status(source_status):
    score = {
        "kind": "score", "status": source_status, "round_id": "round",
        "scored_run_id": "execution", "submission_id": "submission",
        "stage": 1, "icp_position": 0,
    }
    execution = {
        "kind": "execute", "status": "accepted", "round_id": "round",
        "run_id": "execution", "submission_id": "submission",
        "stage": 1, "icp_position": 0,
    }
    assert ArenaService._same_scoring_membership(
        score, execution, round_id="round", completed_signal=True
    )
    assert not ArenaService._same_scoring_membership(score, execution, round_id="round")


def test_signal_key_reuses_a_when_companion_order_changes():
    first = _public_company("Acme", domain="acme.example")
    first["intent_signals"].append({**first["intent_signals"][0],
                                    "url": "https://acme.example/b"})
    reversed_order = deepcopy(first)
    reversed_order["intent_signals"].reverse()
    assert _scope(first)["cache_key"] == _scope(reversed_order, signal_index=1)["cache_key"]


def test_signal_key_reuses_a_when_intent_paragraph_changes():
    first = _public_company("Acme", domain="acme.example")
    first.pop("fit_summary")
    first.pop("fit_evidence_urls")
    first["intent_signals"][0].pop("why_now")
    first["intent_signals"][0].pop("snippet")
    first["intent_details"] = "The product launch creates a timely buying trigger."
    changed = deepcopy(first)
    changed["intent_details"] = "The team has a different reason to buy this quarter."
    assert _scope(first)["cache_key"] == _scope(changed)["cache_key"]


def test_signal_key_changes_for_verified_context_icp_and_image():
    company = _public_company("Acme", domain="acme.example")
    original = _scope(company)["cache_key"]
    icp = _icp()
    icp["prompt"] = "Find product launches in a different buyer market"
    variants = (
        {"identity": {"decision": "match", "observed_domain": "other.example"}},
        {"funding_context": [{"url": "https://acme.example/funding", "quote": "Series A financing"}]},
        {"icp": icp},
        {"image_digest": "sha256:" + "b" * 64},
    )
    assert all(_scope(company, **variant)["cache_key"] != original for variant in variants)
