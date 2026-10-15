"""Offline target-response provenance through the worker and homepage judge."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import tempfile

import aiohttp
import pytest
from yarl import URL

from lab_arena import operations, runner, scoring, scoring_provider_compat, shim
from qualification.scoring import company_verification
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import (
    company_fit_match, company_fit_unavailable, evaluate_company_identity,
)
from qualification.scoring.competition import (
    scorer_breakdown_has_company_local_verification_failure,
    scorer_breakdown_has_retryable_infrastructure_failure,
)
from tests.lab_arena.test_lab_arena_runner import FakeApi, lease
from tests.lab_arena.test_lab_arena_scoring import make_icp, scored_company
from tests.test_positive_semantic_fit_review import _company, _icp


URL_VALUE = "https://example.com/"
HTML = (
    '<html><head><title>Example Company</title></head><body>'
    '<a href="https://www.linkedin.com/company/example-company">LinkedIn</a>'
    'Example Company builds business software.</body></html>'
)


def _broker_document(case):
    """Use the real compatibility and sanitizer functions, without HTTP."""
    operation = shim.PAGE_FETCH_OPERATION
    call = {"operation_id": operation, "outcome": "settled", "status": 502}
    if case == "transport_recovered":
        case = "good200"
    if case in {"good200", "mismatch200", "target403", "target404", "target502"}:
        target_status = int(case[-3:])
        html = HTML if case == "good200" else (
            HTML.replace("Example Company", "Unrelated Business").replace(
                "example-company", "unrelated-business"
            ) if case == "mismatch200" else "Target error"
        )
        route = scoring_provider_compat.route_for(
            kind="score", funding_source="miner_key", round_id="arena-2026-10-15",
            operation_id=operation, parameters={"url": URL_VALUE},
        )
        envelope = {"status": "completed", "result": {"data": {
            "rawHtml": html, "metadata": {
                "sourceURL": URL_VALUE,
                "url": "https://other.example/" if case == "mismatch200" else URL_VALUE,
                "statusCode": target_status,
            },
        }}}
        status, headers, body, final_url = (
            scoring_provider_compat.adapt_response_with_trusted_url(
                route, status=200, headers={}, body=json.dumps(envelope).encode()
            )
        )
        status, headers, body = operations.sanitize_response(
            operation, status, headers, body, parameters={"url": URL_VALUE}
        )
        # This is the existing broker's post-sanitizer metadata handoff.
        headers[operations.TRUSTED_RESPONSE_URL_HEADER] = final_url
    else:
        # Exact retained d384 failure shape: generic broker 502, no target URL.
        raw_headers = (
            {operations.TRUSTED_RESPONSE_URL_HEADER: URL_VALUE}
            if case == "forged_target_header" else {}
        )
        status, headers, body = operations.sanitize_response(
            operation, 502, raw_headers, operations.GENERIC_UNAVAILABLE_BODY,
            parameters={"url": URL_VALUE},
        )
        assert operations.TRUSTED_RESPONSE_URL_HEADER not in headers
        if case == "d384_saved_shape":
            assert hashlib.sha256(body).hexdigest() == (
                "7bf697892f7c0f95010a89c676487c12b08e43887c5e2c7e69de896ca0559513"
            )
        if case in {"generic401", "generic402", "generic429"}:
            # Exact worker HTTP status, generic body, no account or target proof.
            status = int(case[-3:])
        if case == "malformed_url":
            headers[operations.TRUSTED_RESPONSE_URL_HEADER] = "https://127.0.0.1/"
        elif case == "duplicate_url":
            headers[operations.TRUSTED_RESPONSE_URL_HEADER] = URL_VALUE
            headers[operations.TRUSTED_RESPONSE_URL_HEADER.upper()] = URL_VALUE
        elif case == "wrong_operation":
            headers[operations.TRUSTED_RESPONSE_URL_HEADER] = URL_VALUE
    call["status"] = status
    return {
        "status": status, "headers": headers,
        "body_b64": base64.b64encode(body).decode(), "call": call,
    }


class _ShimSession:
    def __init__(self, **kwargs):
        self.headers = kwargs.get("headers", {})
        self.timeout = kwargs.get("timeout")
        self._raise_for_status = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, **kwargs):
        return aiohttp.client._RequestContextManager(
            shim._aiohttp_request(aiohttp, URL)(self, "GET", url, **kwargs)
        )


@pytest.mark.parametrize(
    "case,expected_decision,expected_code,expected_scorer_calls,fetches_per_call",
    [
        ("d384_saved_shape", "unavailable", "provider_error", 3, 1),
        ("generic401", "unavailable", "provider_error", 3, 1),
        ("generic402", "unavailable", "provider_error", 3, 1),
        ("generic429", "unavailable", "provider_error", 3, 1),
        ("forged_target_header", "unavailable", "provider_error", 3, 1),
        ("transport", "unavailable", "provider_error", 3, 2),
        ("transport_recovered", "match", "", 1, 2),
        ("malformed_url", "unavailable", "provider_error", 3, 2),
        ("duplicate_url", "unavailable", "provider_error", 3, 2),
        ("wrong_operation", "unavailable", "provider_error", 3, 2),
        ("target403", "unavailable", "source_blocked", 3, 1),
        ("target404", "unavailable", "", 3, 1),
        ("target502", "unavailable", "source_blocked", 3, 1),
        ("good200", "match", "", 1, 1),
        ("mismatch200", "mismatch", "", 1, 1),
    ],
)
def test_worker_homepage_provenance_reaches_real_scorer_admission(
    monkeypatch, case, expected_decision, expected_code,
    expected_scorer_calls, fetches_per_call,
):
    document = _broker_document(case)

    class Api(FakeApi):
        def provider(self, run_id, lease_token, frame):
            self.provider_frames.append(dict(frame))
            if case == "transport" or (
                case == "transport_recovered" and len(self.provider_frames) == 1
            ):
                raise runner.RunnerError("synthetic broker unavailable", http_status=503)
            return document

    api = Api([])
    state = runner.RunState(lease=lease(), lease_token="tok-r1")
    monkeypatch.setenv(shim.TRUSTED_SCORER_ENV, "1")
    monkeypatch.setattr(company_verification.aiohttp, "ClientSession", _ShimSession)
    monkeypatch.setattr(company_verification, "_TRANSIENT_FETCH_RETRY_DELAY_SECS", 0)
    if case == "wrong_operation":
        monkeypatch.setattr(
            shim, "_execute_with_response_url",
            lambda **_kwargs: shim._dispatch_with_response_url(
                "exa.contents", {"ids": [URL_VALUE]}, 1000
            ),
        )
    decisions = []

    def scorer(*_args):
        decision = asyncio.run(company_verification.verify_company_exists(
            "Example Company", URL_VALUE,
            company_linkedin="https://www.linkedin.com/company/example-company",
        ))
        decisions.append(decision)
        return [{
            "final_score": 90.0 if decision.decision == "match" else 0.0,
            "failure_reason": "", "intent_signals_detail": [],
            "verifier_gate_receipts": [decision.receipt("company_fit")],
        }]

    # Short local socket path, real worker framing/validation, no external calls.
    with tempfile.TemporaryDirectory(prefix="arena-prov-", dir="/tmp") as directory:
        socket_path = Path(directory) / "worker.sock"
        server = runner.WorkerSocketServer(socket_path, api, state)
        monkeypatch.setenv(shim.WORKER_SOCKET_ENV, str(socket_path))
        server.start()
        try:
            kwargs = dict(
                icp=make_icp(0), companies=[scored_company(0)], scorer=scorer,
                max_retries=3,
            )
            if expected_decision == "unavailable" and expected_code != "source_blocked":
                with pytest.raises(scoring.ScoringError):
                    scoring.score_work_item({"scored_run_id": case}, **kwargs)
            else:
                result = scoring.score_work_item({"scored_run_id": case}, **kwargs)
                assert result[0]["final_score"] == (
                    90.0 if expected_decision == "match" else 0.0
                )
                receipt = result[0]["verifier_gate_receipts"][0]
                assert receipt.get("failure_class", "") == (
                    "company_verification_exhausted"
                    if expected_code == "source_blocked" else ""
                )
                assert not scorer_breakdown_has_retryable_infrastructure_failure(result[0])
        finally:
            server.stop()

    assert len(decisions) == expected_scorer_calls
    assert all(decision.decision == expected_decision for decision in decisions)
    assert all(
        decision.details.get("failure_reason_code", "") == expected_code
        for decision in decisions
    )
    assert len(api.provider_frames) == expected_scorer_calls * fetches_per_call
    assert len(state.calls) == len(api.provider_frames)
    raw = {"verifier_gate_receipts": [decisions[-1].receipt("company_fit")]}
    assert scorer_breakdown_has_company_local_verification_failure(raw) == (
        expected_code == "source_blocked"
    )
    # Existing call records are retained; this change does not settle or replay them.
    if case in {"d384_saved_shape", "forged_target_header", "target502"}:
        assert all(call == document["call"] for call in state.calls)


@pytest.mark.parametrize("marker", [None, False, True, 0, ""])
def test_connection_provenance_requires_explicit_false_and_keeps_fetch_retries(
    monkeypatch, marker,
):
    calls = []

    class Session(_ShimSession):
        def get(self, *_args, **_kwargs):
            calls.append("get")
            error = aiohttp.ClientConnectionError("synthetic connection error")
            if marker is not None:
                error._lab_arena_response_url_validated = marker
            raise error

    monkeypatch.setattr(company_verification.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(company_verification, "_TRANSIENT_FETCH_RETRY_DELAY_SECS", 0)
    result = asyncio.run(company_verification.verify_company_exists(
        "Example Company", URL_VALUE,
    ))
    assert result.decision == "unavailable"
    assert result.details["failure_reason_code"] == (
        "provider_error" if marker is False else "source_blocked"
    )
    assert calls == ["get", "get"]


@pytest.mark.parametrize("web_identity_decision", ["unavailable", "mismatch"])
@pytest.mark.parametrize("transport", ["response502", "marked_connection"])
def test_opaque_homepage_failure_survives_completed_unproven_fit_assembly(
    monkeypatch, web_identity_decision, transport,
):
    fetch_calls = []

    class Session(_ShimSession):
        def get(self, url, **_kwargs):
            fetch_calls.append("get")
            if transport == "marked_connection":
                error = aiohttp.ClientConnectionError("synthetic worker unavailable")
                error._lab_arena_response_url_validated = False
                raise error
            return shim._AiohttpResponse(
                request_url=url, response_url="", status=502,
                headers={"content-type": "application/json"},
                body=operations.GENERIC_UNAVAILABLE_BODY,
            )

    async def precheck(*_args, **_kwargs):
        return company_fit_match()

    async def web(company, _icp, **kwargs):
        homepage = kwargs["verified_homepage_identity"]
        assert homepage.details["failure_reason_code"] == "provider_error"
        assert lead_scorer._homepage_identity_has_retryable_failure(homepage)
        receipt = evaluate_company_identity(
            submitted_name=company.company_name,
            submitted_website=company.company_website,
            submitted_linkedin=company.company_linkedin,
            observed_name="Other Company" if web_identity_decision == "mismatch" else "",
            observed_website="https://other.example" if web_identity_decision == "mismatch" else "",
            observed_linkedin="https://www.linkedin.com/company/other-company" if web_identity_decision == "mismatch" else "",
            evidence_source="company_web_reverification",
        )
        assert receipt["decision"] == web_identity_decision
        return company_fit_unavailable("completed bounded semantic review", details={
            "failure_class": "insufficient_fit_evidence",
            "dimension_decisions": {
                dimension: "unavailable"
                for dimension in ("employee_size", "industry", "geography", "stage")
            },
            "identity_decision": web_identity_decision,
            "identity_receipt": receipt,
            "investigation_receipt": {
                "completed": True, "failure_reason_code": "",
                "positive_semantic_review": True, "resolved": False,
            },
        })

    monkeypatch.setattr(company_verification.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(company_verification, "_TRANSIENT_FETCH_RETRY_DELAY_SECS", 0)
    monkeypatch.setattr(lead_scorer, "run_company_zero_checks", precheck)
    monkeypatch.setattr(lead_scorer, "_llm_reverify_company", web)
    fits = []

    def scorer(*_args):
        fit = asyncio.run(lead_scorer._verify_company_fit(
            _company(), _icp(), 0.0, 1.0, set(),
            require_https_transport=True, evidence_investigator=True,
        ))
        fits.append(fit)
        return [{
            "final_score": 0.0, "failure_reason": "",
            "verifier_gate_receipts": [fit.receipt("company_fit")],
        }]

    kwargs = dict(icp=make_icp(0), companies=[scored_company(0)], scorer=scorer)
    if web_identity_decision == "unavailable":
        with pytest.raises(scoring.ScoringError):
            scoring.score_work_item({"scored_run_id": "opaque-completed-fit"}, **kwargs)
        assert len(fits) == 3
        assert all(fit.details["failure_reason_code"] == "provider_error" for fit in fits)
        assert all(not fit.details.get("failure_class") for fit in fits)
    else:
        result = scoring.score_work_item({"scored_run_id": "independent-conflict"}, **kwargs)
        assert len(fits) == 1
        assert fits[0].decision == "mismatch"
        assert not scorer_breakdown_has_retryable_infrastructure_failure(result[0])
    assert len(fetch_calls) == len(fits) * (2 if transport == "marked_connection" else 1)
