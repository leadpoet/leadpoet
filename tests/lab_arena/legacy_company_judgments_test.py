"""Company reuse preserves the current integrity grader and provider work."""
from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.usefixtures("scorer_process_environment")

from lab_arena import company_judgments, contracts, scorer_entrypoint, scoring, verify
from qualification.scoring import lead_scorer
from qualification.scoring.competition import (
    CompetitionCompanyScorer, apply_company_judgment_context, raw_company_judgment,
)
from tests.test_arena_score_integrity import _icp, _positive_breakdown, _public_company
from tests.lab_arena.company_judgments_test import _miss


def _input(names=("A", "B"), *, policy=None):
    return scoring.build_scoring_input(
        scored_run_id="execute-legacy", icp=_icp(),
        companies=[_public_company(name, domain=name.lower()+".example") for name in names],
        policy=policy or scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True,
        ), evaluation_date="2026-10-20",
    )


def _refs(document):
    return company_judgments.build_company_scopes(
        scoring_input=document, round_id="arena-2026-10-20", network_name="finney", netuid=71,
        scorer_image_digest="sha256:"+"a"*64,
        scorer_image_reference="registry/scorer@sha256:"+"a"*64,
        integrity_policy="arena_integrity_v1", company_judgment_policy=company_judgments.POLICY,
    )


def _snapshot(ref, raw, *, exclusions=None):
    return company_judgments.build_evidence_snapshot(
        new_judgment={**_miss(ref), "raw_judgment": raw}, company_ref=ref,
        source_score_run_id="score-source", source_scored_run_id="execute-source",
        source_output_ref="arena/source.json", source_output_hash="sha256:"+"b"*64,
        source_runner_hotkey="runner", source_claim_request_id="request",
        source_claim_request_hash="sha256:"+"c"*64, source_lease_generation=1,
        source_completion_request_hash="sha256:"+"d"*64, runner_authority_exclusions=["runner"] if exclusions is None else exclusions,
    )


def _lease(document, saved=None):
    saved = saved or {}
    hits, misses = [], []
    for ref in _refs(document):
        if ref["cache_key"] in saved:
            evidence = _snapshot(ref, saved[ref["cache_key"]])
            hits.append({**_miss(ref), "evidence_hash": contracts.document_hash(evidence), "evidence_doc": evidence})
        else:
            misses.append(_miss(ref))
    return {"schema_version": company_judgments.LEASE_SCHEMA_VERSION, "hits": hits, "misses": misses}


def _score(document, saved=None, **kwargs):
    document = copy.deepcopy(document)
    document["company_judgment_cache"] = _lease(document, saved)
    return scorer_entrypoint.score_input(document, **kwargs)


@pytest.fixture
def judge(monkeypatch):
    calls = []
    async def score_company(**kwargs):
        company = kwargs["company"]
        calls.append(company.company_name)
        return _positive_breakdown(company.company_name, company.company_website.removeprefix("https://"))
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    return calls


def test_across_companions_permutation_and_all_hit_preserves_first_judgment(judge):
    first = _score(_input())
    assert "failure" not in first
    saved = {row["cache_key"]: row["raw_judgment"] for row in first["company_judgments"]}
    second = _score(_input(("C", "A")), saved)
    assert judge == ["A", "B", "C"]
    assert [row["company_index"] for row in second["company_judgments"]] == [0]
    assert second["breakdowns"][1] == {**first["breakdowns"][0], "company_index": 1}
    saved.update({row["cache_key"]: row["raw_judgment"] for row in second["company_judgments"]})
    third = _score(_input(("B", "A")), saved)
    assert "failure" not in third and third["company_judgments"] == []
    assert judge == ["A", "B", "C"]
    empty = _score(_input(()))
    assert empty["breakdowns"] == [] and empty["company_judgments"] == []


@pytest.mark.parametrize("change", ["evidence", "date", "icp", "scorer"])
def test_effective_company_scope_binds_judgment_inputs(change):
    document = _input()
    before = _refs(document)[0]["cache_key"]
    if change == "evidence":
        document["companies"][0]["intent_signals"][0]["url"] += "-changed"
    elif change == "date":
        document["evaluation_date"] = "2026-10-21"
    elif change == "icp":
        document["icp"]["industry"] = "Manufacturing"
    else:
        document["scorer_policy"]["judge_models"]["company_fit"] = "anthropic/claude-sonnet-4"
    assert _refs(document)[0]["cache_key"] != before


@pytest.mark.parametrize("negative", [False, True])
def test_submitted_duplicates_use_existing_gates_and_never_judge_or_cache_alias(judge, monkeypatch, negative):
    if negative:
        async def no_primary(**kwargs):
            judge.append(kwargs["company"].company_name)
            row = _positive_breakdown("A", "a.example")
            row.update(final_score=0, intent_signals_detail=[])
            return row
        monkeypatch.setattr(lead_scorer, "score_company_competition_intent", no_primary)
    document = _input(("A", "A"))
    cached = _score(document)
    assert judge == ["A"]
    assert len(cached["company_judgments"]) == 1
    legacy_scorer = scoring.lab_scorer(scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2"))
    old = scoring.score_work_item({}, icp=document["icp"], companies=document["companies"], scorer=legacy_scorer)
    assert cached["breakdowns"] == old
    assert cached["breakdowns"][1]["duplicate_company"] is True
    # Subset persistence omits the alias; gateway replay must accept only a proved skip.
    missing = [cached["company_judgments"][0]["raw_judgment"], None]
    assert apply_company_judgment_context(document["companies"], missing, company_quality=False) == old
    with pytest.raises(ValueError, match="missing"):
        apply_company_judgment_context(document["companies"], [None, None], company_quality=False)


def test_post_judgment_alias_projection_preserves_intrinsic_raw(monkeypatch):
    calls = []
    async def same_observed(**kwargs):
        calls.append(kwargs["company"].company_name)
        return _positive_breakdown("Canonical", "canonical.example", "canonical")
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", same_observed)
    document = _input()
    cached = _score(document)
    assert calls == ["A", "B"]
    assert [row["raw_judgment"]["final_score"] for row in cached["company_judgments"]] == [60, 60]
    assert [row["final_score"] for row in cached["breakdowns"]] == [60, 0]
    legacy = scoring.lab_scorer(scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2"))
    assert cached["breakdowns"] == scoring.score_work_item({}, icp=document["icp"], companies=document["companies"], scorer=legacy)


def test_failure_keeps_prior_completed_intrinsic_and_retry_uses_it(monkeypatch):
    calls, checkpoints = [], []
    failed = True
    async def transient(**kwargs):
        name = kwargs["company"].company_name
        calls.append(name)
        if failed and name == "B":
            raise RuntimeError("provider unavailable")
        return _positive_breakdown(name, name.lower()+".example")
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", transient)
    output = _score(_input(), on_company_judgments=lambda rows: checkpoints.append(copy.deepcopy(rows)))
    assert output["failure"] == "judge_error"
    assert calls == ["A", "B", "B", "B"]
    assert len(output["company_judgments"]) == 1 and len(checkpoints) == 1
    saved = {row["cache_key"]: row["raw_judgment"] for row in output["company_judgments"]}
    failed = False
    retry = _score(_input(), saved)
    assert "failure" not in retry and calls[-1] == "B" and calls.count("A") == 1


def test_bucket_max_and_malformed_skips_are_not_cached_or_judged(judge):
    document = _input(("A", "B", "C"))
    document["companies"][0]["employee_count"] = "1-10"
    document["scorer_policy"]["max_scored_companies"] = 1
    output = _score(document)
    assert judge == ["B"]
    assert [row["company_index"] for row in output["breakdowns"]] == [1]
    assert [row["company_index"] for row in output["company_judgments"]] == [1]
    document = _input(("A",))
    document["companies"][0]["company_name"] = ""
    with pytest.raises(company_judgments.CompanyJudgmentError, match="invalid"):
        _score(document)
    assert judge == ["B"]


@pytest.mark.parametrize("fault", ["malformed", "primary", "paragraph", "transient"])
def test_malformed_or_unfinished_raw_cannot_be_cached(judge, fault):
    output = _score(_input(("A",)))
    raw = copy.deepcopy(output["company_judgments"][0]["raw_judgment"])
    if fault == "malformed":
        raw.pop("company_identity_key")
    elif fault == "primary":
        raw["intent_signals_detail"] = []
    elif fault == "paragraph":
        raw["verifier_gate_receipts"].append({"gate": "intent_details", "decision": "mismatch"})
    else:
        raw["verifier_gate_receipts"][0]["decision"] = "unavailable"
        raw["failure_reason"] = "provider_error"
    assert company_judgments.raw_judgment_is_cacheable(raw) is False


def test_legacy_raw_marker_cannot_weaken_quality_scope(judge):
    output = _score(_input(("A",)))
    row = output["company_judgments"][0]
    ref = _refs(_input(("A",)))[0]
    scope = ref["scope_doc"]
    scope.pop("company_judgment_policy")
    scope["company_quality_policy"] = "company_quality_v1"
    keyless = {key: value for key, value in scope.items() if key != "cache_key"}
    scope["cache_key"] = contracts.document_hash(keyless)
    ref["cache_key"] = scope["cache_key"]
    row["cache_key"] = scope["cache_key"]
    with pytest.raises(company_judgments.CompanyJudgmentError, match="mode"):
        _snapshot(ref, row["raw_judgment"])


def test_new_cache_policy_does_not_enable_quality_or_contacts():
    policy = _input()["scorer_policy"]
    assert policy["company_judgment_policy"] == company_judgments.POLICY
    assert "company_quality_policy" not in policy
    assert not scoring.lab_scorer(policy).company_quality
    for adapter in ("qualification_contacts_v3", "qualification_legacy_v1"):
        with pytest.raises(contracts.ArenaContractError):
            scoring.build_scorer_policy(scoring_adapter_version=adapter, company_judgment_cache=True)
    quality = scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2", company_quality=True)
    assert "company_judgment_policy" not in quality


@pytest.mark.parametrize("baseline", [False, True])
def test_gateway_replay_accepts_proved_skip_but_rejects_unskipped_missing(judge, baseline):
    import json
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    document = _input(("A", "A"))
    document["companies"][1]["intent_signals"][0]["url"] += "-different-evidence"
    refs = _refs(document)
    assert refs[0]["cache_key"] != refs[1]["cache_key"]
    output = _score(document)
    assert len(output["company_judgments"]) == 1 and judge == ["A"]
    evidence = _snapshot(refs[0], output["company_judgments"][0]["raw_judgment"])
    cache = {(refs[0]["cache_key"], 0): {"evidence_doc": evidence}}
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(
        get_submission=lambda *_: {"is_king": baseline},
        get_company_judgment=lambda *identity: cache.get(identity),
    )
    service._objects = SimpleNamespace(get_bounded=lambda *_: json.dumps(output).encode())
    service._validated_company_judgment_lease = lambda context, **_: context
    service._company_judgment_evidence = lambda **kwargs: cache[(kwargs["cache_key"], kwargs["authority_slot"])]["evidence_doc"]
    run = {"submission_id": "baseline" if baseline else "miner", "miner_hotkey": "miner",
           "scored_run_id": document["scored_run_id"], "output_ref": "score.json",
           "result_doc": {"company_judgment_consumed_indexes": [0]},
           "claim_response": {"company_judgment_cache": _lease(document)}}
    assert service._company_quality_breakdowns(run, icp=document["icp"],
        companies=document["companies"], policy=document["scorer_policy"]) == output["breakdowns"]
    # The second row now has an independent submitted identity, so its omission is invalid.
    independent = _input()["companies"]
    with pytest.raises(scoring.ScoringError, match="context"):
        service._company_quality_breakdowns(run, icp=document["icp"],
            companies=independent, policy=document["scorer_policy"])
    # An omitted slot stays omitted even if another assignment later judges it.
    cache[(refs[1]["cache_key"], 0)] = {"evidence_doc": {"raw_judgment": {"final_score": 99}, "runner_authority_exclusions": ["miner"]}}
    assert service._company_quality_breakdowns(run, icp=document["icp"],
        companies=document["companies"], policy=document["scorer_policy"]) == output["breakdowns"]


def test_atomic_checkpoint_dispatches_company_rows_for_cache_only_policy(tmp_path, monkeypatch):
    import json
    from arena_reviewer_runtime.checkpoint_io import run_scorer
    from lab_arena import quality_policy
    document = _input()
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(document))
    monkeypatch.setenv("LAB_ARENA_INPUT_PATH", str(input_path))
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))
    def interrupted(_document, *, on_company_judgments):
        on_company_judgments([])
        raise SystemExit("interrupted")
    with pytest.raises(SystemExit):
        run_scorer(score_input=interrupted, scoring=scoring, quality_policy=quality_policy)
    assert json.loads(output_path.read_text())["company_judgments"] == []
    assert not output_path.with_name("output.json.tmp").exists()


def test_provider_observation_uses_original_slot_and_binds_only_effective_evidence(monkeypatch):
    from lab_arena import provider_observations
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2", company_judgment_cache=True,
        intent_details=True, provider_observation_handoff=True,
    )
    document = _input(policy=policy)
    baseline = _refs(document)
    document["icp"][provider_observations.SCORING_ICP_KEY] = [{
        "company_index": 1, "company_domain": "b.example", "source_url": "https://b.example/news/launch",
        "first_observed_date": "2026-10-19",
    }]
    witnessed = _refs(document)
    assert witnessed[0]["cache_key"] == baseline[0]["cache_key"]
    assert witnessed[1]["cache_key"] != baseline[1]["cache_key"]
    observations = []
    async def score_company(**kwargs):
        observations.append(kwargs.get("provider_observation"))
        company = kwargs["company"]
        return _positive_breakdown(company.company_name, company.company_website.removeprefix("https://"))
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", score_company)
    output = _score(document)
    saved = {row["cache_key"]: row["raw_judgment"] for row in output["company_judgments"][:1]}
    observations.clear()
    second = _score(document, saved)
    assert "failure" not in second
    assert len(observations) == 1 and observations[0]["first_observed_date"] == "2026-10-19"


def test_activation_uses_submission_open_and_changes_only_cache_fields():
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    service = ArenaService.__new__(ArenaService)
    service._config = SimpleNamespace(defaults=SimpleNamespace(company_judgment_from="2026-10-19T00:00:00Z"))
    service._require_company_quality_schema = lambda: None
    document = {"integrity_policy": "arena_integrity_v1", "schedule": {"submission_open": "2026-10-18T00:00:00Z"},
                "scorer_policy": _input()["scorer_policy"]}
    document["scorer_policy"].pop("company_judgment_policy")
    old = copy.deepcopy(document)
    assert service._activate_company_judgments(document) is False and document == old
    document["schedule"]["submission_open"] = "2026-10-19T00:00:00Z"
    before = copy.deepcopy(document)
    assert service._activate_company_judgments(document) is True
    assert document.pop("company_judgment_policy") == company_judgments.POLICY
    assert document["scorer_policy"].pop("company_judgment_policy") == company_judgments.POLICY
    assert document == before


def test_store_uses_atomic_commit_v4_only_when_cache_activation_is_requested():
    from types import SimpleNamespace
    from lab_arena.store import ArenaStore
    calls = []
    store = ArenaStore(SimpleNamespace(rpc=lambda name, params: calls.append((name, params)) or {"status": "committed"}))
    kwargs = dict(participants=[], benchmark_ref="benchmark.json", evaluation_date="2026-10-20",
                  icp_set_date="2026-10-19", scorer_image_digest="sha256:"+"a"*64,
                  scorer_image_reference="scorer@sha256:"+"a"*64)
    store.commit_round_v2("arena-2026-10-20", **kwargs)
    store.commit_round_v2("arena-2026-10-20", **kwargs, company_judgment_policy=company_judgments.POLICY)
    assert calls[0][0] == "lab_arena_commit_round_v2"
    assert "p_company_judgment_policy" not in calls[0][1]
    assert calls[1][0] == "lab_arena_commit_round_v4"
    assert calls[1][1]["p_deepline_catalog"] is None
    assert calls[1][1]["p_company_judgment_policy"] == company_judgments.POLICY


def test_internal_model_incompatible_row_retains_old_zero_without_cache(judge):
    document = _input(("A",))
    document["companies"][0]["company_linkedin"] = "https://www.linkedin.com/company/cushman-&-wakefield"
    output = _score(document)
    assert "failure" not in output and output["company_judgments"] == []
    assert output["breakdowns"][0]["failure_reason"] == "company model contract incompatible"
    assert judge == []
    old = scoring.score_work_item({}, icp=document["icp"], companies=document["companies"],
        scorer=scoring.lab_scorer(scoring.build_scorer_policy(scoring_adapter_version="qualification_integrity_v2")))
    assert output["breakdowns"] == old


@pytest.mark.parametrize("source", [None, "another-score", "score-current"])
@pytest.mark.parametrize("consumed", [[], [0]])
def test_skipped_miss_does_not_acquire_authority_from_later_cache(source, consumed):
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    document = _input(("A",))
    refs = _refs(document)
    lease = _lease(document)
    run = {"run_id": "score-current", "scored_run_id": "execute-current", "runner_hotkey": "validator",
           "company_judgment_refs": refs, "claim_response": {"company_judgment_cache": lease},
           "result_doc": {"company_judgment_consumed_indexes": consumed}}
    service = ArenaService.__new__(ArenaService)
    service._same_scoring_membership = lambda *_a, **_k: True
    service._store = SimpleNamespace(get_company_judgment=lambda *_: {"source_score_run_id": source} if source else None)
    evidence = {"source_score_run_id": source, "source_scored_run_id": "execute-current", "source_runner_hotkey": "validator"}
    service._company_judgment_evidence = lambda **_: evidence
    service._validated_attribution_source = lambda **_: "validator"
    ids = set()
    if consumed and source != "score-current":
        with pytest.raises(scoring.ScoringError, match="missing or foreign"):
            service._score_run_attribution(run, {}, round_id="arena-2026-10-20", runs_by_id={}, source_ids=ids)
        return
    attribution = service._score_run_attribution(run, {}, round_id="arena-2026-10-20", runs_by_id={}, source_ids=ids)
    if consumed and source == "score-current":
        assert attribution == {"validator": False} and ids == {source}
    else:
        assert attribution == {} and ids == set()


def test_gateway_overwrites_forged_consumption_metadata_without_mutating_signed_body(judge):
    import json
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    document = _input(("A", "A"))
    document["companies"][1]["intent_signals"][0]["url"] += "-different-evidence"
    output = _score(document)
    refs, lease = _refs(document), _lease(document)
    score_run = {"run_id": "score-current", "round_id": "arena-2026-10-20", "kind": "score",
        "scored_run_id": document["scored_run_id"], "submission_id": "miner", "miner_hotkey": "miner",
        "runner_hotkey": "validator", "icp_position": 0, "company_judgment_refs": refs,
        "claim_request_id": "request", "claim_request_hash": "sha256:"+"c"*64, "lease_generation": 1,
        "claim_response": {"company_judgment_cache": lease, "runner_authority_exclusions": ["validator"]}}
    round_row = {"round_id": score_run["round_id"], "configuration_doc": {
        "integrity_policy": "arena_integrity_v1", "company_judgment_policy": company_judgments.POLICY,
        "scorer_policy": document["scorer_policy"]}}
    run_result = {"schema_version": contracts.RUN_RESULT_SCHEMA_VERSION, "terminal_status": "accepted",
        "started_at": "2026-10-20T00:00:00Z", "finished_at": "2026-10-20T00:00:01Z",
        "resource_summary": {"wall_seconds": 1, "cpu_seconds": 0, "max_rss_bytes": 0,
                             "stdout_bytes": 0, "stderr_bytes": 0, "provider_call_count": 1},
        "company_judgment_consumed_indexes": [1]}
    request = {"round_id": score_run["round_id"], "hotkey": "validator",
               "body": {"run_id": score_run["run_id"], "lease_token": "token", "result": run_result, "output": output}}
    original_hash = contracts.document_hash(request)
    completed = []
    service = ArenaService.__new__(ArenaService)
    service._request_round = lambda *_a, **_k: (request, round_row)
    service._require_validator_authority = lambda *_: None
    service._lease_token_for_run = lambda *_: "token"
    service.evaluation_icps = lambda *_: [document["icp"]]
    service._validated_company_judgment_lease = lambda context, **_: context
    service._store = SimpleNamespace(
        get_run=lambda run_id: score_run if run_id == score_run["run_id"] else {"output_ref": "execute.json"},
        get_submission=lambda *_: {"is_king": False},
        complete_attempt=lambda **kwargs: completed.append(kwargs) or {"status": "accepted"},
    )
    service._objects = SimpleNamespace(
        get_bounded=lambda *_: json.dumps({"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                                           "companies": document["companies"]}).encode(),
        put=lambda *_: None,
    )
    assert service.handle_complete(request)["status"] == "accepted"
    assert completed[0]["result"]["company_judgment_consumed_indexes"] == [0]
    assert len(completed[0]["company_judgment_evidence"]) == 1
    assert request["body"]["result"]["company_judgment_consumed_indexes"] == [1]
    assert contracts.document_hash(request) == original_hash


def test_consumed_indexes_exclude_skips_and_include_observed_alias_judgments(monkeypatch):
    async def same_observed(**_):
        return _positive_breakdown("Canonical", "canonical.example", "canonical")
    monkeypatch.setattr(lead_scorer, "score_company_competition_intent", same_observed)
    document = _input()
    output = _score(document)
    used = set()
    raw = [row["raw_judgment"] for row in output["company_judgments"]]
    apply_company_judgment_context(document["companies"], raw, company_quality=False, consumed_indexes=used)
    assert used == {0, 1}  # The second row was actually judged before observed-alias projection.
    used.clear()
    apply_company_judgment_context(document["companies"], raw, company_quality=False,
                                  scored_indexes=[1], consumed_indexes=used)
    assert used == {1}


@pytest.mark.parametrize("consumed", [[1], [0, 1]])
def test_attribution_filters_unused_foreign_hit_authority(judge, consumed):
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    document = _input()
    output = _score(document)
    saved = {row["cache_key"]: row["raw_judgment"] for row in output["company_judgments"]}
    lease = _lease(document, saved)
    lease["hits"][0]["evidence_doc"]["source_runner_hotkey"] = "foreign-validator"
    # Attribution reads validated persisted metadata; evidence hashing is tested separately.
    service = ArenaService.__new__(ArenaService)
    service._same_scoring_membership = lambda *_a, **_k: True
    service._validated_attribution_source = lambda **kwargs: kwargs["source_runner_hotkey"]
    service._store = SimpleNamespace()
    run = {"run_id": "score-current", "scored_run_id": document["scored_run_id"],
           "company_judgment_refs": _refs(document), "claim_response": {"company_judgment_cache": lease},
           "result_doc": {"company_judgment_consumed_indexes": consumed}}
    # Replace only expensive source binding for this public attribution unit.
    from unittest.mock import patch
    with patch.object(company_judgments, "validate_lease_context", return_value=lease):
        actual = service._score_run_attribution(run, {}, round_id="arena-2026-10-20", runs_by_id={})
    assert actual == ({"runner": True} if consumed == [1] else {"foreign-validator": True, "runner": True})


@pytest.mark.parametrize("indexes", [None, [True], [1, 0], [0, 0], [2], [-1], "0", [0.0]])
def test_attribution_fails_closed_on_missing_or_invalid_server_metadata(indexes):
    from types import SimpleNamespace
    from lab_arena.service import ArenaService
    document = _input()
    service = ArenaService.__new__(ArenaService)
    service._same_scoring_membership = lambda *_a, **_k: True
    service._store = SimpleNamespace()
    result = {} if indexes is None else {"company_judgment_consumed_indexes": indexes}
    run = {"run_id": "score-current", "scored_run_id": document["scored_run_id"],
        "company_judgment_refs": _refs(document), "claim_response": {"company_judgment_cache": _lease(document)},
        "result_doc": result}
    with pytest.raises(scoring.ScoringError, match="consumed"):
        service._score_run_attribution(run, {}, round_id="arena-2026-10-20", runs_by_id={})


def test_evidence_preserves_database_authority_order_instead_of_python_collation(judge):
    document = _input(("A",))
    raw = _score(document)["company_judgments"][0]["raw_judgment"]
    ref = _refs(document)[0]
    database_order = ["z-owner", "runner", "A-owner"]
    assert database_order != sorted(database_order)
    evidence = _snapshot(ref, raw, exclusions=database_order)
    assert evidence["runner_authority_exclusions"] == database_order
    assert company_judgments.validate_evidence_snapshot(
        evidence, cache_key=ref["cache_key"], company_input_hash=ref["company_input_hash"],
        authority_slot=0, evidence_hash=contracts.document_hash(evidence),
    ) == evidence
    for invalid in ([], ["runner", "runner"], ["runner", ""], ["runner", {}], ["owner"]):
        with pytest.raises(company_judgments.CompanyJudgmentError, match="authority"):
            _snapshot(ref, raw, exclusions=invalid)
