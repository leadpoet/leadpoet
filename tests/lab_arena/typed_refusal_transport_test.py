"""Exercise the real Arena response parser, shim, and bounded scorer reader."""

import asyncio
import base64
import copy

import pytest

from lab_arena import broker, operations, runner, scoring_provider_compat, shim
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification
from qualification.scoring import lead_scorer
from tests.lab_arena.typed_investigator_fallback_test import _investigate


SOURCE = "https://acme.example/news"
API = "https://api.scrapingdog.com/scrape"
REFUSAL = b'{"error":{"code":"provider_request_refused"}}'
IDENTITY = "sha256:" + "a" * 64


def _compatibility_call():
    return {
        "operation_id": shim.PAGE_FETCH_OPERATION,
        "provider": "deepline",
        "action_sequence": 0,
        "error_code": "provider_request_refused",
        "call_identity": IDENTITY,
        "effective_operation_id": scoring_provider_compat.EFFECTIVE_OPERATION_ID,
        "compatibility_version": scoring_provider_compat.COMPATIBILITY_VERSION,
        "adapter": "generic_http_raw_html",
        "tool": scoring_provider_compat.GENERIC_HTTP_TOOL,
    }


def _response(status, body, *, trusted_url="", attested=False,
              forged_marker=False, call_changes=None):
    headers = {"content-type": "application/json", "content-length": str(len(body))}
    if trusted_url:
        headers[operations.TRUSTED_RESPONSE_URL_HEADER] = trusted_url
    if forged_marker:
        headers[shim.CALL_IDENTITY_HEADER] = IDENTITY
    call = _compatibility_call() if attested else {}
    if call_changes:
        call.update(call_changes)
    document = {
        "status": status,
        "headers": headers,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "call": call,
    }
    headers = runner.WorkerSocketServer._socket_response_headers(
        document, shim.PAGE_FETCH_OPERATION, 0,
    )
    parsed = shim._parse_worker_response_with_url({
        "status": status,
        "headers": headers,
        "body_b64": document["body_b64"],
    })
    return shim._AiohttpResponse(
        request_url=API, response_url=parsed[3], status=parsed[0],
        headers=parsed[1], body=parsed[2],
    )


class _Session:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url, **_kwargs):
        self.urls.append(url)
        return self.response


@pytest.mark.parametrize("version", [scoring_provider_compat.COMPATIBILITY_VERSION, scoring_provider_compat.SOURCE_HTTP_COMPATIBILITY_VERSION])
def test_exact_broker_refusal_crosses_real_reader_only_as_typed_error(monkeypatch, version):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    refused = broker._error_result("provider_request_refused", {})
    assert refused.status == 403 and refused.body == REFUSAL
    response = _response(refused.status, refused.body, attested=True,
                         call_changes={"compatibility_version": version})
    assert response._lab_arena_response_url_validated is False
    assert response.headers[shim.CALL_IDENTITY_HEADER] == IDENTITY
    session = _Session(response)
    result = asyncio.run(investigator._fetch_page(session, SOURCE))
    assert session.urls == [API]
    assert result == {"ok": False, "error": "provider_request_refused"}
    assert investigator._fetch_outcome(SOURCE, result)["error_class"] == (
        "provider_request_refused"
    )


@pytest.mark.parametrize("status,body", [
    (403, b'{"error":{"code":"unauthorized"}}'),
    (403, REFUSAL + b" "),
    (403, b'{"error":{"code":"provider_request_refused","extra":1}}'),
    (403, b'{"error":{"code":"provider_request_refused"}'),
    (403, REFUSAL),
    (401, REFUSAL),
    (402, REFUSAL),
    (429, REFUSAL),
    (502, REFUSAL),
])
def test_unvalidated_controls_remain_opaque(monkeypatch, status, body):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    session = _Session(_response(status, body))
    with pytest.raises(verification._ArenaProviderFetchError) as caught:
        asyncio.run(investigator._fetch_page(session, SOURCE))
    assert caught.value.broker_status == status
    assert caught.value.broker_error_code == ""
    assert session.urls == [API]


def test_exact_envelope_without_provider_route_remains_opaque(monkeypatch):
    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    with pytest.raises(verification._ArenaProviderFetchError):
        asyncio.run(investigator._fetch_page(
            _Session(_response(403, REFUSAL, attested=True)), SOURCE,
        ))


@pytest.mark.parametrize("status,body,attested,forged_marker,trusted_url,expected", [
    (403, REFUSAL, True, False, "", "source_blocked"),
    (403, REFUSAL, False, False, "", "provider_error"),
    (403, REFUSAL, False, True, "", "provider_error"),
    (403, REFUSAL + b" ", True, False, "", "provider_error"),
    (403, b'{"error":{"code":"unauthorized"}}', True, False, "", "provider_error"),
    (401, REFUSAL, True, False, "", "provider_error"),
    (402, REFUSAL, True, False, "", "provider_error"),
    (502, REFUSAL, True, False, "", "provider_error"),
    (403, REFUSAL, False, False, "https://thecudaily.com/", None),
])
def test_homepage_only_attested_request_refusal_is_source_blocked(
    monkeypatch, status, body, attested, forged_marker, trusted_url, expected,
):
    response = _response(
        status, body, attested=attested, forged_marker=forged_marker,
        trusted_url=trusted_url,
    )

    class _ClientSession(_Session):
        def __init__(self, **_kwargs):
            super().__init__(response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(verification.aiohttp, "ClientSession", _ClientSession)
    result = asyncio.run(verification.verify_company_exists(
        "Crux Analytics", "https://thecudaily.com/",
    ))
    assert result.decision == "unavailable"
    assert result.details.get("failure_reason_code") == expected
    assert not result.details.get("observed_homepage_organization_identity")


def test_trusted_target_url_path_is_unchanged(monkeypatch):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    response = _response(403, REFUSAL, trusted_url=SOURCE)
    assert response._lab_arena_response_url_validated is True
    result = asyncio.run(investigator._fetch_page(_Session(response), SOURCE))
    assert result == {"ok": False, "error": "provider_request_refused"}


def test_provider_cannot_forge_host_marker(monkeypatch):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    response = _response(403, REFUSAL, forged_marker=True)
    assert shim.CALL_IDENTITY_HEADER not in response.headers
    with pytest.raises(verification._ArenaProviderFetchError):
        asyncio.run(investigator._fetch_page(_Session(response), SOURCE))


@pytest.mark.parametrize("status,body", [
    (403, REFUSAL + b" "),
    (403, b'{"error":{"code":"provider_request_refused","extra":1}}'),
    (401, REFUSAL),
])
@pytest.mark.parametrize("version", [scoring_provider_compat.COMPATIBILITY_VERSION, scoring_provider_compat.SOURCE_HTTP_COMPATIBILITY_VERSION])
def test_valid_call_identity_does_not_turn_other_errors_into_refusals(status, body, version):
    response = _response(status, body, attested=True,
                         call_changes={"compatibility_version": version})
    assert response.headers[shim.CALL_IDENTITY_HEADER] == IDENTITY
    with pytest.raises(verification._ArenaProviderFetchError) as caught:
        asyncio.run(verification._fetch_bounded_html(_Session(response), SOURCE))
    assert caught.value.broker_error_code == ""


@pytest.mark.parametrize("changes", [
    {"operation_id": "deepline.execute"},
    {"provider": "openrouter"},
    {"action_sequence": 1},
    {"action_sequence": True},
    {"call_identity": "sha256:bad"},
    {"effective_operation_id": "openrouter.chat"},
    {"compatibility_version": "unknown"},
    {"adapter": "unknown"},
    {"tool": "unknown"},
    {"adapter": []},
    {"tool": {}},
    {"adapter": None},
    {"tool": None},
])
@pytest.mark.parametrize("version", [scoring_provider_compat.COMPATIBILITY_VERSION, scoring_provider_compat.SOURCE_HTTP_COMPATIBILITY_VERSION])
def test_host_does_not_mark_mismatched_call(changes, version):
    response = _response(403, REFUSAL, attested=True,
                         call_changes={"compatibility_version": version, **changes})
    assert shim.CALL_IDENTITY_HEADER not in response.headers


@pytest.mark.parametrize("marker", [
    "false", "sha256:123", "sha256:" + "z" * 64,
])
def test_shim_rejects_invalid_call_identity(marker):
    with pytest.raises(shim.ShimTransportError):
        shim._parse_worker_response_with_url({
            "status": 403,
            "headers": {shim.CALL_IDENTITY_HEADER: marker},
            "body_b64": base64.b64encode(REFUSAL).decode("ascii"),
        })


def test_complete_refusal_review_can_be_unproven_but_incomplete_claim_cannot(
    monkeypatch,
):
    call = _compatibility_call()
    document = {
        "status": 403,
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(REFUSAL).decode("ascii"),
        "call": call,
    }
    host_headers = runner.WorkerSocketServer._socket_response_headers(
        document, shim.PAGE_FETCH_OPERATION, 0,
    )
    status, headers, body, response_url = shim._parse_worker_response_with_url({
        "status": document["status"], "headers": host_headers,
        "body_b64": document["body_b64"],
    })
    assert response_url == "" and headers[shim.CALL_IDENTITY_HEADER] == IDENTITY
    result, calls = _investigate(
        monkeypatch, broker.BrokerResult(status, headers, body, call),
    )
    # Existing typed source refusals use the bounded exact-URL Exa fallback.
    assert calls == {"fetch": 1, "exa": 1, "model": 2}
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    outcomes = result["usage"]["fetch_outcomes"]
    assert len(outcomes) == 2
    assert outcomes[0]["ok"] is False
    assert outcomes[0]["error_class"] == "provider_request_refused"
    assert outcomes[0]["loaded_text_length"] == 0
    assert outcomes[1]["ok"] is True
    assert outcomes[1]["url_sha256"] == outcomes[0]["url_sha256"]
    assert outcomes[1]["loaded_text_length"] > 0
    receipt = {
        "gate": "company_evidence_investigation",
        "completed_submitted_findings": result["_completed_submit"],
        "targets": ["stage"],
        "claims": result["claims"],
        "failure_reason": result["failure_reason"],
        "usage": result["usage"],
    }
    assert lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("stage",), investigation_receipt=receipt,
    )
    unfinished = copy.deepcopy(receipt)
    unfinished["completed_submitted_findings"] = False
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("stage",), investigation_receipt=unfinished,
    )
    unsupported = copy.deepcopy(receipt)
    unsupported["claims"]["stage"]["status"] = "VERIFIED"
    assert not lead_scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("stage",), investigation_receipt=unsupported,
    )


@pytest.mark.parametrize("operation_id,parameters,fallback", [
    ("scrapingdog.scrape", {"url": SOURCE}, False),
    ("scrapingdog.scrape", {"url": SOURCE}, True),
    ("scrapingdog.scrape", {"url": "https://www.linkedin.com/feed/update/urn:li:activity:1234567890123456789/"}, False),
    ("scrapingdog.scrape", {"url": "https://api.ashbyhq.com/posting-api/job-board/example"}, False),
    ("scrapingdog.scrape", {"url": "https://boards-api.greenhouse.io/v1/boards/example/jobs/123456?content=true"}, False),
    ("scrapingdog.scrape", {"url": "https://boards-api.greenhouse.io/v1/boards/example/jobs"}, False),
    ("scrapingdog.scrape", {"url": "https://example.myworkdayjobs.com/wday/cxs/example/jobs/job/Example123"}, False),
    ("scrapingdog.scrape", {"url": "https://api.lever.co/v0/postings/example"}, False),
    ("scrapingdog.x_post", {"tweetId": "1234567890123456789"}, False),
    ("scrapingdog.linkedinjobs", {"job_id": "1234567890"}, False),
    ("scrapingdog.profile_post", {"id": "1234567890123456789"}, False),
])
def test_reviewed_compatibility_routes_keep_exact_success_receipts(
    operation_id, parameters, fallback,
):
    route = scoring_provider_compat.route_for(
        kind="score", funding_source="miner_key", round_id="arena-2026-10-21",
        operation_id=operation_id, parameters=parameters,
        source_content_fallback=fallback,
    )
    assert route is not None
    call = {
        **route.summary(), "operation_id": operation_id, "provider": "deepline",
        "tool": route.effective_parameters["tool"], "action_sequence": 0,
        "call_identity": IDENTITY, "outcome": "settled",
    }
    document = {
        "status": 200, "headers": {}, "body_b64": base64.b64encode(b"source").decode(),
        "call": call,
    }
    headers = runner.WorkerSocketServer._socket_response_headers(document, operation_id, 0)
    assert headers[shim.CALL_IDENTITY_HEADER] == IDENTITY
    state = runner.RunState(lease={"run_id": "r1", "kind": "score"}, lease_token="test-token")
    server = object.__new__(runner.WorkerSocketServer)
    server._state = state
    observation = {
        "ordinal": 0, "action_sequence": 0, "operation_id": operation_id,
        "request": {"admitted_parameters": parameters, "timeout_ms": 30_000},
        "protocol": "http", "contributes_to_provider_call_count": True,
    }
    response = {key: value for key, value in document.items() if key != "call"}
    server._finish_observation_unchecked(observation, response, document=document)
    snapshot = state.worker_observations.document()
    assert snapshot["response_receipt_count"] == 1
    receipt = snapshot["response_receipts"][0]
    assert receipt["operation_id"] == operation_id
    assert receipt["call_identity"] == IDENTITY and receipt["status"] == 200
    # A different provider or a route summary without its exact tool cannot
    # claim the same host response, even when it copies a valid identity.
    for changes in ({"provider": "openrouter"}, {"tool": "unknown"},
                    {"compatibility_version": "unknown"}, {"adapter": "unknown"},
                    {"effective_operation_id": "openrouter.chat"},
                    {"adapter": []}, {"tool": {}},
                    {"adapter": None}, {"tool": None},
                    {"action_sequence": True}, {"action_sequence": 1}):
        invalid = {**document, "call": {**call, **changes}}
        assert shim.CALL_IDENTITY_HEADER not in runner.WorkerSocketServer._socket_response_headers(
            invalid, operation_id, 0,
        )
        invalid_state = runner.RunState(lease={"run_id": "r1", "kind": "score"}, lease_token="test-token")
        server._state = invalid_state
        server._finish_observation_unchecked(dict(observation), response, document=invalid)
        assert invalid_state.worker_observations.document()["response_receipt_count"] == 0
