"""Exercise the real Arena response parser, shim, and bounded scorer reader."""

import asyncio
import base64
import copy

import pytest

from lab_arena import broker, operations, runner, shim
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import company_verification as verification
from qualification.scoring import lead_scorer
from tests.lab_arena.typed_investigator_fallback_test import _investigate


SOURCE = "https://acme.example/news"
API = "https://api.scrapingdog.com/scrape"
REFUSAL = b'{"error":{"code":"provider_request_refused"}}'
IDENTITY = "sha256:" + "a" * 64


def _response(status, body, *, trusted_url="", attested=False,
              forged_marker=False, call_changes=None):
    headers = {"content-type": "application/json", "content-length": str(len(body))}
    if trusted_url:
        headers[operations.TRUSTED_RESPONSE_URL_HEADER] = trusted_url
    if forged_marker:
        headers[shim.CALL_IDENTITY_HEADER] = IDENTITY
    call = ({
        "operation_id": shim.PAGE_FETCH_OPERATION,
        "provider": "deepline",
        "action_sequence": 0,
        "error_code": "provider_request_refused",
        "call_identity": IDENTITY,
    } if attested else {})
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


def test_exact_broker_refusal_crosses_real_reader_only_as_typed_error(monkeypatch):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test-placeholder")
    refused = broker._error_result("provider_request_refused", {})
    assert refused.status == 403 and refused.body == REFUSAL
    response = _response(refused.status, refused.body, attested=True)
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


def test_homepage_does_not_turn_refusal_into_source_absence(monkeypatch):
    response = _response(403, REFUSAL, attested=True)

    class _ClientSession(_Session):
        def __init__(self, **_kwargs):
            super().__init__(response)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(verification.aiohttp, "ClientSession", _ClientSession)
    result = asyncio.run(verification.verify_company_exists(
        "Acme", "https://acme.example/",
    ))
    assert result.decision == "unavailable"
    assert result.details["failure_reason_code"] == "provider_error"


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
def test_host_does_not_mark_wrong_status_or_body(status, body):
    response = _response(status, body, attested=True)
    assert shim.CALL_IDENTITY_HEADER not in response.headers


@pytest.mark.parametrize("changes", [
    {"operation_id": "deepline.execute"},
    {"provider": "openrouter"},
    {"action_sequence": 1},
    {"action_sequence": True},
    {"error_code": "provider_unavailable"},
    {"call_identity": "sha256:bad"},
])
def test_host_does_not_mark_mismatched_call(changes):
    response = _response(403, REFUSAL, attested=True, call_changes=changes)
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
    call = {
        "operation_id": shim.PAGE_FETCH_OPERATION,
        "provider": "deepline",
        "action_sequence": 0,
        "error_code": "provider_request_refused",
        "call_identity": IDENTITY,
    }
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
    assert calls == {"fetch": 1, "exa": 0, "model": 2}
    assert result["_completed_submit"] is True
    assert result["failure_reason"] == ""
    outcomes = result["usage"]["fetch_outcomes"]
    assert len(outcomes) == 1
    assert outcomes[0]["ok"] is False
    assert outcomes[0]["error_class"] == "provider_request_refused"
    assert outcomes[0]["loaded_text_length"] == 0
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
