"""Partial company evidence remains reusable without accepting an incomplete ICP."""
from __future__ import annotations

import copy
import asyncio
import json
from types import SimpleNamespace

import pytest

from lab_arena import company_judgments, contracts, runner, scorer_entrypoint, scoring
from lab_arena.service import ArenaService
from qualification.scoring.competition import apply_company_judgment_context
from qualification.scoring import company_verification, lead_scorer
from qualification.scoring.company_fit_decision import company_fit_match
from tests.lab_arena.company_judgments_test import _input, _miss, _new, _refs, _evidence
from tests.lab_arena.test_lab_arena_runner import BridgingRuntime, scoring_lease
from tests.lab_arena.trajectory_runner_test import LoggingApi, run_model
from tests.test_arena_company_quality_policy import _positive_breakdown
from tests.test_homepage_web_identity_conflict import (
    HOMEPAGE, _company, _completed_qsic_review, _icp, _settle_qsic, _web_receipt,
)


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


def test_completed_identity_conflict_zero_survives_later_judge_error(monkeypatch):
    async def fetch(_session, _url):
        return 200, "https://qsic.com/", HOMEPAGE

    async def web(*_args, **_kwargs):
        return company_fit_match("web claims fit", details={
            "identity_decision": "match",
            "identity_receipt": _web_receipt(),
            "dimension_decisions": {
                "employee_size": "match", "industry": "match",
                "geography": "match",
            },
            "dimension_evidence": {
                key: {"url": "https://getqsic.com/", "quote": "QSIC"}
                for key in ("employee_size", "industry", "geography")
            },
        })

    monkeypatch.setattr(company_verification, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)
    bound = asyncio.run(lead_scorer._verify_company_fit(
        _company(), _icp(), 0, 0, set(), require_https_transport=True,
    ))
    identity = bound.details["dimension_evidence"]["identity"]["web_identity_receipt"]
    assert identity["reason_code"] == "homepage_web_identity_conflict_unresolved"
    assert {key: identity[key] for key in (
        "decision", "evidence_source", "submitted_name", "submitted_domain",
        "submitted_linkedin_slug", "observed_name", "observed_domain",
        "observed_linkedin_slug",
    )} == {
        "decision": "unavailable", "evidence_source": "company_web_reverification",
        "submitted_name": "qsic", "submitted_domain": "qsic.com",
        "submitted_linkedin_slug": "", "observed_name": "qsic",
        "observed_domain": "qsic.com", "observed_linkedin_slug": "qsic",
    }
    completed, verdict, icp = _completed_qsic_review(identity)
    settled = _settle_qsic(completed, verdict, icp)
    assert settled is not None
    assert settled.details["failure_class"] == "insufficient_fit_evidence"
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
        if name == "A":
            raw.update(final_score=0.0, icp_fit=0.0,
                       failure_reason="Company fit unavailable: unproven identity")
            raw["verifier_gate_receipts"][0] = settled.receipt("company_fit")
        return apply_company_judgment_context(companies, [raw])

    judge.company_quality = judge.integrity_policy = True
    judge.contacts_required = False
    monkeypatch.setattr(scoring, "lab_scorer", lambda *_a, **_k: judge)
    monkeypatch.setattr(scoring, "apply_policy_to_environment", lambda *_a, **_k: None)
    monkeypatch.setenv(scorer_entrypoint.shim.TRUSTED_SCORER_ENV, "1")
    output = scorer_entrypoint.score_input(
        document, on_company_judgments=lambda rows: progress.append(copy.deepcopy(rows)),
    )
    assert output["failure"] == "judge_error"
    assert attempted.count("A") == 1
    assert attempted.count("B") == 3
    assert [row["company_index"] for row in output["company_judgments"]] == [0, 2, 3, 4]
    zero = output["company_judgments"][0]["raw_judgment"]
    assert zero["final_score"] == 0.0
    assert zero["verifier_gate_receipts"][0]["failure_class"] == "insufficient_fit_evidence"
    assert progress[0][0]["company_index"] == 0
    assert scoring.scoring_output_from_bytes(json.dumps(output).encode()) == output


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
    source = {"run_id": evidence["source_score_run_id"],
              "round_id": ref["scope_doc"]["round_id"],
              "kind": "score", "status": "failed", "terminal_cause": "judge_error",
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
    service._round = lambda _: {"configuration_doc": {"scorer_policy": {}}}
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


@pytest.mark.parametrize("change", [
    "valid", "large_valid", "execute", "accepted", "wrong_run", "wrong_slot", "absent",
    "malformed", "nested", "oversized", "cleanup", "worker_cleanup", "stale",
])
def test_signaled_scorer_reports_only_strict_failed_company_checkpoint(
    tmp_path, monkeypatch, change,
):
    import signal
    from lab_arena import runner, runtime
    from tests.lab_arena.test_lab_arena_runner import lease as execute_lease

    document = _input()
    lease = scoring_lease()
    lease.update(
        scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1",
        company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]},
    )
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)
    if change == "execute":
        lease = execute_lease()
    elif change == "accepted":
        output = scoring.build_scoring_output("r1", [], company_judgments=rows)
    elif change == "wrong_run":
        output["scored_run_id"] = "another-execution"
    elif change == "wrong_slot":
        output["company_judgments"][0]["authority_slot"] += 1
    checkpoint_bytes = json.dumps(output).encode()
    if change == "large_valid":
        checkpoint_bytes += b" " * runtime.MAX_OUTPUT_BYTES
        assert len(checkpoint_bytes) < scoring.MAX_SCORING_OUTPUT_BYTES
    elif change == "absent":
        checkpoint_bytes = None
    elif change == "malformed":
        checkpoint_bytes = b'{"unfinished":'
    elif change == "nested":
        checkpoint_bytes = b"[" * 2000 + b"]" * 2000
    elif change == "oversized":
        checkpoint_bytes = b"x" * (scoring.MAX_SCORING_OUTPUT_BYTES + 1)

    class Interrupted:
        def run_icp(self, spec, **_kwargs):
            assert (spec.entry_command == runtime.SCORER_ENTRY_COMMAND) is (change != "execute")
            error = runtime.RuntimeHostError(
                reason="sandbox_launcher_signaled", launch_exit_code=-signal.SIGINT,
                checkpoint_bytes=checkpoint_bytes,
            )
            if change == "cleanup":
                raise runtime.SandboxCleanupError("cleanup incomplete", result=None) from error
            raise error

    class Api(LoggingApi):
        def complete(self, envelope):
            result = super().complete(envelope)
            return {"status": "stale"} if change == "stale" else result

    api = Api()
    api.leases = [lease]
    if change == "worker_cleanup":
        original_stop = runner.WorkerSocketServer.stop

        def failed_stop(server):
            original_stop(server)
            raise RuntimeError("worker cleanup failed")

        monkeypatch.setattr(runner.WorkerSocketServer, "stop", failed_stop)
    worker = run_model(tmp_path, api, Interrupted())
    events = [event for batch in api.batches for event in batch["events"]]
    host_errors = [event for event in events if event["kind"] == "runtime.error"]
    assert host_errors
    if change != "cleanup":
        assert any(event["content"].get("runtime_host_reason") == "sandbox_launcher_signaled"
                   and event["content"].get("status") == (
                       "judge_error" if change in {"valid", "large_valid", "stale"} else "abandoned"
                   ) for event in host_errors)
    if change in {"valid", "large_valid", "stale"}:
        assert worker.abandoned == 0
        assert len(api.completions) == 1
        body = api.completions[0]["body"]
        assert body["result"]["terminal_status"] == "judge_error"
        assert body["output"]["failure"] == "judge_error"
        assert body["output"]["company_judgments"] == rows
        assert "breakdowns" not in body["output"]
        if change == "stale":
            assert worker.completed[0]["result"]["status"] == "stale"
    else:
        assert worker.abandoned == 1
        assert api.completions == []
    assert "company_judgments" not in json.dumps(api.batches)


@pytest.mark.parametrize("failed_cleanup", [None, "delete", "umount"])
@pytest.mark.parametrize("large_checkpoint", [False, True])
def test_real_signaled_runtime_checkpoint_reaches_only_failed_completion(
    tmp_path, failed_cleanup, large_checkpoint,
):
    import signal
    from lab_arena import runtime
    from tests.lab_arena.test_lab_arena_runtime import FakeClock, FakeProcess, FakeRunner, make_config

    document = _input()
    lease = scoring_lease()
    lease.update(
        scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1",
        company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]},
    )
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)
    checkpoint_bytes = json.dumps(output).encode()
    if large_checkpoint:
        checkpoint_bytes += b" " * runtime.MAX_OUTPUT_BYTES
        assert len(checkpoint_bytes) < scoring.MAX_SCORING_OUTPUT_BYTES
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    host_config = make_config(host_dir)
    clock = FakeClock()

    class SignaledRuntime:
        def run_icp(self, spec, **_kwargs):
            assert spec.entry_command == runtime.SCORER_ENTRY_COMMAND

            def launch(argv):
                spec.output_path.write_bytes(checkpoint_bytes)
                return FakeProcess(argv, clock=clock, finish_at=clock(), returncode=-signal.SIGINT)

            return runtime.run_sandbox(
                host_config, spec,
                process_runner=FakeRunner(clock, fail={failed_cleanup} if failed_cleanup else (), run_process=launch),
                clock=clock, sleep=clock.sleep, rusage=lambda: (0.0, 0),
            )

    api = LoggingApi()
    api.leases = [lease]
    worker = run_model(tmp_path, api, SignaledRuntime())
    if failed_cleanup:
        assert worker.abandoned == 1
        assert api.completions == []
    else:
        assert worker.abandoned == 0
        body = api.completions[0]["body"]
        assert body["result"]["terminal_status"] == "judge_error"
        assert body["output"]["company_judgments"] == rows
        assert "breakdowns" not in body["output"]
        assert any(
            event["kind"] == "runtime.error"
            and event["content"].get("runtime_host_reason") == "sandbox_launcher_signaled"
            for batch in api.batches for event in batch["events"]
        )


@pytest.mark.parametrize("funding,error_code,provider_status,expected", [
    ("miner_key", "miner_credentials_unavailable", 401, "credential_error"),
    ("miner_key", "budget_refused", None, "credential_error"),
    ("miner_key", "budget_exhausted", None, "credential_error"),
    ("host", "provider_unavailable", 502, "judge_error"),
    ("host", "budget_refused", None, "judge_error"),
])
def test_interrupted_partial_uses_existing_provider_failure_classification(
    tmp_path, funding, error_code, provider_status, expected,
):
    import os
    import signal
    from lab_arena import runtime, shim
    from tests.lab_arena.test_lab_arena_runner import RefusingApi

    document = _input()
    lease = scoring_lease()
    lease.update(
        scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1",
        company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]},
    )
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)

    class Api(LoggingApi):
        def provider(self, run_id, lease_token, frame):
            response = RefusingApi.provider(self, run_id, lease_token, frame)
            response["call"].update(funding_source=funding, provider_status=provider_status)
            return response

    class Interrupted:
        def run_icp(self, spec, **_kwargs):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, _body = shim.dispatch(
                    "openrouter.chat", {"model": "openai/gpt-4o-mini", "messages": [
                        {"role": "user", "content": "judge"},
                    ]}, 5000,
                )
                assert status == 402
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            raise runtime.RuntimeHostError(
                reason="sandbox_launcher_signaled", launch_exit_code=-signal.SIGINT,
                checkpoint_bytes=json.dumps(output).encode(),
            )

    api = Api()
    api.code = error_code
    api.leases = [lease]
    worker = run_model(tmp_path, api, Interrupted())
    assert worker.abandoned == 0
    body = api.completions[0]["body"]
    assert body["result"]["terminal_status"] == expected
    assert body["output"]["failure"] == "judge_error"
    assert body["output"]["company_judgments"] == rows
    assert body["result"]["resource_summary"]["provider_call_count"] == 1
    assert any(
        event["kind"] == "runtime.error"
        and event["content"].get("runtime_host_reason") == "sandbox_launcher_signaled"
        for batch in api.batches for event in batch["events"]
    )


def test_interrupted_partial_preserves_successful_later_provider_call(tmp_path):
    import os
    import signal
    from lab_arena import runtime, shim
    from tests.lab_arena.codex_rate_limit_retry_test import identified_document, PARAMETERS

    document = _input()
    lease = scoring_lease()
    lease.update(
        scorer_policy=document["scorer_policy"], company_quality_policy="company_quality_v1",
        company_judgment_cache=_lease(document), scored_output={"companies": document["companies"]},
    )
    rows = [_new(_refs(document)[0])]
    output = scoring.build_scoring_failure("r1", "judge_error", company_judgments=rows)

    class Interrupted:
        def run_icp(self, spec, **_kwargs):
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                # Scorers do not use execute-only transparent Responses retries.
                # A later successful scorer request clears the latest unresolved
                # failure; the original host interruption still remains failed.
                for expected in (502, 200):
                    status, _headers, _body = shim.dispatch("openrouter.responses", PARAMETERS, 120_000)
                    assert status == expected
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
            raise runtime.RuntimeHostError(
                reason="sandbox_launcher_signaled", launch_exit_code=-signal.SIGINT,
                checkpoint_bytes=json.dumps(output).encode(),
            )

    api = LoggingApi()
    api.leases = [lease]
    api.broker_documents = [
        identified_document(0, retry_after=0),
        identified_document(1, provider_status=200, actual=7, status=200, error_code=None, idempotent=False),
    ]
    run_model(tmp_path, api, Interrupted())
    body = api.completions[0]["body"]
    assert body["result"]["terminal_status"] == "judge_error"
    assert body["output"]["company_judgments"] == rows
    summary = body["result"]["resource_summary"]
    assert summary["provider_call_count"] == 2
    assert summary["provider_unresolved_failure_count"] == 0
    assert body["result"]["failure_diagnostic"] == {
        "stage": "sandbox", "error_class": "judge_error",
    }
