"""Partial company evidence remains reusable without accepting an incomplete ICP."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import company_judgments, contracts, runner, scorer_entrypoint, scoring
from lab_arena.service import ArenaService
from qualification.scoring.competition import apply_company_judgment_context
from tests.lab_arena.company_judgments_test import _input, _miss, _new, _refs, _evidence
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, scoring_lease
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model
from tests.test_arena_company_quality_policy import _positive_breakdown


def _lease(document):
    return {"schema_version": company_judgments.LEASE_SCHEMA_VERSION,
            "hits": [], "misses": [_miss(ref) for ref in _refs(document)]}


def test_failed_company_does_not_erase_completed_siblings_or_stop_later_companies(monkeypatch):
    document = _input()
    document["company_judgment_cache"] = _lease(document)
    attempted = []
    progress = []

    def judge(companies, buyer, reference):
        name = companies[0]["company_name"]
        attempted.append(name)
        if name == "B":
            raise scoring.ScoringError("provider unavailable", failure_reason="provider_error")
        raw = _positive_breakdown(name, name.lower() + ".example", name.lower())
        if name == "C":
            raw.update(final_score=0.0, failure_reason="company mismatch", intent_signals_detail=[])
            raw["verifier_gate_receipts"][0]["decision"] = "no_match"
        return apply_company_judgment_context(companies, [raw])

    judge.company_quality = judge.integrity_policy = True
    judge.contacts_required = False
    monkeypatch.setattr(scoring, "lab_scorer", lambda *_a, **_k: judge)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *_a, **_k: None)
    monkeypatch.setenv(scorer_entrypoint.shim.TRUSTED_SCORER_ENV, "1")
    output = scorer_entrypoint.score_input(document, on_company_judgments=lambda rows: progress.append(copy.deepcopy(rows)))
    assert output["failure"] == "judge_error"
    assert "breakdowns" not in output
    assert attempted == ["A", "B", "B", "B", "C", "D", "E"]
    assert [row["company_index"] for row in output["company_judgments"]] == [0, 2, 3, 4]
    assert [len(rows) for rows in progress] == [1, 2, 3, 4]
    assert output["company_judgments"][1]["raw_judgment"]["final_score"] == 0
    assert scoring.scoring_output_from_bytes(json.dumps(output).encode()) == output
    with pytest.raises(company_judgments.CompanyJudgmentError, match="every miss"):
        company_judgments.validate_new_company_judgments(output["company_judgments"], lease_context=document["company_judgment_cache"])


def test_atomic_output_survives_process_failure_after_completed_company(tmp_path, monkeypatch):
    document = _input()
    document["company_judgment_cache"] = _lease(document)
    rows = [_new(_refs(_input())[0])]
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(document))
    monkeypatch.setenv("LAB_ARENA_INPUT_PATH", str(input_path))
    monkeypatch.setenv("LAB_ARENA_OUTPUT_PATH", str(output_path))

    def interrupted(_document, *, on_company_judgments):
        on_company_judgments(rows)
        raise SystemExit("sandbox killed")

    monkeypatch.setattr(scorer_entrypoint, "score_input", interrupted)
    with pytest.raises(SystemExit):
        scorer_entrypoint.main()
    checkpoint = scoring.scoring_output_from_bytes(output_path.read_bytes())
    assert checkpoint == scoring.build_scoring_failure("execute-1", "judge_error", company_judgments=rows)
    assert not output_path.with_name("output.json.tmp").exists()


@pytest.mark.parametrize("timed_out", [False, True])
def test_runner_signs_partial_failure_and_timeout_without_accepting_icp(tmp_path, timed_out):
    document = _input()
    lease = scoring_lease()
    lease.update(scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1", company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]})
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)
    api = LoggingApi()
    api.leases = [lease]
    run_model(tmp_path, api, BridgingRuntime(output=output, calls=0, timed_out=timed_out))
    body = api.completions[0]["body"]
    assert body["result"]["terminal_status"] == ("judge_timeout" if timed_out else "judge_error")
    assert body["output"]["failure"] == body["result"]["terminal_status"]
    assert body["output"]["company_judgments"] == rows
    assert "breakdowns" not in body["output"]


@pytest.mark.parametrize("change", ["accepted", "wrong_run", "unleased", "wrong_slot", "provider_error"])
def test_runner_rejects_unbound_or_nonterminal_partial_output(change):
    document = _input()
    lease = {"scored_run_id": "execute-1", "company_judgment_cache": _lease(document)}
    output = scoring.build_scoring_failure("execute-1", "judge_error", company_judgments=[_new(_refs(document)[0])])
    if change == "accepted":
        output.pop("failure")
        output["breakdowns"] = []
    elif change == "wrong_run":
        output["scored_run_id"] = "foreign-run"
    elif change == "unleased":
        lease.pop("company_judgment_cache")
    elif change == "wrong_slot":
        output["company_judgments"][0]["authority_slot"] = 1
    else:
        output["company_judgments"][0]["raw_judgment"] = {"final_score": 0, "failure_reason": "provider timeout"}
    assert runner._partial_company_judgment_output(output, lease) is None


def _bound_source():
    document = _input()
    ref = _refs(document)[0]
    evidence = _evidence(ref)
    output = scoring.build_scoring_failure("execute-1", "judge_error", company_judgments=[_new(ref)])
    evidence["source_output_hash"] = contracts.document_hash(output)
    source = {"kind": "score", "status": "failed", "terminal_cause": "judge_error",
              "runner_hotkey": evidence["source_runner_hotkey"],
              "scored_run_id": evidence["source_scored_run_id"],
              "output_ref": evidence["source_output_ref"],
              "claim_request_id": evidence["source_claim_request_id"],
              "claim_request_hash": evidence["source_claim_request_hash"],
              "lease_generation": evidence["source_lease_generation"],
              "claim_response": {"runner_authority_exclusions": evidence["runner_authority_exclusions"], "company_judgment_cache": _lease(document)},
              "result_doc": {"terminal_status": "judge_error", "company_judgment_completion_request_hash": evidence["source_completion_request_hash"]}}
    cached = {"company_input_hash": ref["company_input_hash"], "scope_doc": ref["scope_doc"],
              "evidence_doc": evidence, "evidence_hash": contracts.document_hash(evidence),
              **{key: evidence[key] for key in ("source_score_run_id", "source_scored_run_id", "source_runner_hotkey")}}
    service = ArenaService.__new__(ArenaService)
    service._store = SimpleNamespace(get_company_judgment=lambda *_: cached, get_run=lambda *_: source)
    service._objects = SimpleNamespace(get_bounded=lambda *_: json.dumps(output).encode())
    return service, cached, source, output, ref


def test_gateway_accepts_failed_source_only_for_exact_completed_company():
    service, cached, _, _, ref = _bound_source()
    assert service._company_judgment_evidence(cache_key=ref["cache_key"], company_input_hash=ref["company_input_hash"], authority_slot=0) == cached["evidence_doc"]


@pytest.mark.parametrize("change", ["hash", "runner", "completion", "terminal", "slot", "blob", "accepted_blob", "unleased"])
def test_gateway_refuses_tampered_failed_source(change):
    service, cached, source, output, ref = _bound_source()
    if change == "hash":
        cached["evidence_hash"] = "sha256:" + "f" * 64
    elif change == "runner":
        source["runner_hotkey"] = "foreign-validator"
    elif change == "completion":
        source["result_doc"]["company_judgment_completion_request_hash"] = "sha256:" + "f" * 64
    elif change == "terminal":
        source["result_doc"]["terminal_status"] = "accepted"
    elif change == "slot":
        output["company_judgments"][0]["authority_slot"] = 1
        cached["evidence_doc"]["source_output_hash"] = contracts.document_hash(output)
        cached["evidence_hash"] = contracts.document_hash(cached["evidence_doc"])
    elif change == "blob":
        output["company_judgments"][0]["raw_judgment"]["final_score"] = 12
    elif change == "accepted_blob":
        output.pop("failure")
        output["breakdowns"] = []
        cached["evidence_doc"]["source_output_hash"] = contracts.document_hash(output)
        cached["evidence_hash"] = contracts.document_hash(cached["evidence_doc"])
    else:
        source["claim_response"].pop("company_judgment_cache")
    with pytest.raises(scoring.ScoringError):
        service._company_judgment_evidence(cache_key=ref["cache_key"], company_input_hash=ref["company_input_hash"], authority_slot=0)


@pytest.mark.parametrize("credential_failure", [False, True])
def test_timeout_partial_survives_provider_or_credential_override(tmp_path, credential_failure):
    from dataclasses import replace
    from tests.lab_arena.test_lab_arena_runner import RefusedJudgeRuntime, RefusingApi

    document = _input()
    leased = scoring_lease()
    leased.update(scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1", company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]})
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", reason="malformed_response", company_judgments=rows)

    class Api(RefusingApi):
        def provider(self, *args):
            result = super().provider(*args)
            if credential_failure:
                result["call"].update(funding_source="miner_key", provider_status=401)
            return result

    class TimedOut(RefusedJudgeRuntime):
        def run_icp(self, spec, **kwargs):
            return replace(super().run_icp(spec, **kwargs), timed_out=True)

    api = Api([leased], code="miner_credentials_unavailable" if credential_failure else "provider_unavailable")
    run_model(tmp_path, api, TimedOut(output=output))
    body = api.completions[0]["body"]
    assert body["result"]["terminal_status"] == ("credential_error" if credential_failure else "judge_error")
    assert body["output"]["failure"] == ("judge_timeout" if credential_failure else "judge_error")
    assert body["output"]["reason"] == "malformed_response"
    assert body["output"]["company_judgments"] == rows
