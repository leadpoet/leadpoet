"""A proved source-local refusal is not a provider outage or identity proof."""
import asyncio
import copy
import gzip
import json
from pathlib import Path
from types import ModuleType

import pytest

from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring import company_verification as verification
from lab_arena import shim
from scripts import arena_reviewer_completed_merger_seam as seam
from scripts import arena_reviewer_source_local_refusal_seam as source_seam
from tests.reviewer_frozen_sources import frozen_source

CASE = json.loads(gzip.decompress((Path(__file__).parent / "fixtures" /
    "company_evidence" / "chegg_completed_missing_fit.json.gz").read_bytes()))
NATIVE_REFUSAL = "website fetch error: Arena provider response unavailable"
SOURCE_UNAVAILABLE = "website source unavailable: verified provider refusal"


def module_for(source):
    module = ModuleType("_typed_homepage_finality_frozen810")
    exec(compile(source, "exact_frozen810_finality.py", "exec"), vars(module))
    return module


@pytest.fixture(params=["current", "composed810"])
def scorer(request):
    if request.param == "current":
        return lead_scorer
    module = module_for(seam.convert_completed_merger_seam(
        frozen_source("completed_merger810", seam.FINALITY_FILE)))
    name = "qualification/scoring/company_verification.py"
    module._source_refusal_reader = module_for(source_seam.convert_source_local_refusal_seam(
        name, frozen_source("completed_merger810", name)))
    return module


def exercise(monkeypatch, scorer, *, code="source_blocked", reason=NATIVE_REFUSAL,
             completed=True, source_fault=False, missing_repair=False, proof=True):
    """Replay saved public fit observations; all external transports are closed.

    The original company, identity, fit verdicts, findings and page text remain
    unchanged. Intent evaluation is not called by this company-fit path. This
    does not replay reasoning or provider requests.
    """
    case = copy.deepcopy(CASE)
    company = scorer.CompanyOutput.model_validate(case["company"])
    icp = scorer.ICPPrompt.model_validate(case["icp"])
    events = []

    async def request(**kwargs):
        events.append(kwargs["telemetry_purpose"])
        assert len(events) <= 2
        if missing_repair and len(events) == 2:
            kwargs["diagnostic"]["failure_reason_code"] = "provider_error"
            return None, "provider_error"
        return copy.deepcopy(case["verdicts"][len(events) - 1]), ""

    async def investigate(**kwargs):
        result = copy.deepcopy(case["investigation"])
        pages = result["_server_fetched_pages"]
        # Replay the recorded native host result, including its separate paid
        # industry review. Reinterpreting an anonymous product quote without
        # that original review would be a different evidence path.
        assert tuple(kwargs["targets"]) == ("stage", "industry", "required_attribute")
        assert {k: v["status"] for k, v in result["claims"].items()} == {
            "stage": "VERIFIED", "industry": "VERIFIED", "required_attribute": "UNPROVEN",
        }
        for finding in result["claims"].values():
            if finding["status"] == "UNPROVEN":
                assert finding["evidence_url"] == finding["evidence_quote"] == ""
            else:
                assert investigator._quote_occurs(
                    finding["evidence_quote"], pages[finding["evidence_url"]]["text"],
                )
        if source_fault:
            result["usage"]["fetch_outcomes"].append({
                "url_sha256": "e" * 64, "ok": False,
                "error_class": "http_error", "loaded_text_length": 0,
            })
        if not completed:
            result.update(_completed_submit=False, failure_reason="provider_error")
        return result

    async def fetch(_session, url, **_kwargs):
        page = case["investigation"]["_server_fetched_pages"].get(url)
        assert page is not None, "offline replay cannot fetch an unrecorded page"
        return 200, page.get("final_url", url), page["text"]

    class OfflineSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def get(self, *_args, **_kwargs):
            raise AssertionError("offline replay must not contact providers")

        post = get

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "offline-placeholder")
    monkeypatch.setattr(scorer, "_request_company_reverify_json", request)
    monkeypatch.setattr(scorer, "investigate_company_evidence", investigate)
    monkeypatch.setattr(scorer, "_fetch_bounded_html", fetch)
    monkeypatch.setattr(scorer.aiohttp, "ClientSession", OfflineSession)
    if reason == NATIVE_REFUSAL and code == "source_blocked":
        reason = refusal_reason(getattr(scorer, "_source_refusal_reader", verification), proof=proof)
    homepage = scorer.company_fit_unavailable(reason, details={"failure_reason_code": code})
    result = asyncio.run(scorer._llm_reverify_company(
        company, icp, require_company_fit_dimensions=True,
        company_quality=False, evidence_investigator=True,
        verified_homepage_identity=homepage,
    ))
    return result, events


def refusal_reason(reader, *, proof=True, body=None, identity="sha256:" + "a" * 64):
    headers = {shim.CALL_IDENTITY_HEADER: identity}
    if proof:
        headers["x-leadpoet-source-unavailable"] = "verified-source-v1"

    class Session:
        def get(self, *_args, **_kwargs):
            return shim._AiohttpResponse(
                request_url="https://www.chegg.com/", response_url="", status=403,
                headers=headers,
                body=(b'{"error":{"code":"provider_request_refused"}}' if body is None else body),
            )

    with pytest.raises(reader._ArenaProviderFetchError) as error:
        asyncio.run(reader._fetch_bounded_html(Session(), "https://www.chegg.com/"))
    return str(error.value)


def test_native_completed_missing_fit_is_terminal_without_qualifying(monkeypatch, scorer):
    result, events = exercise(monkeypatch, scorer)
    assert events == ["lead_scorer_reverify", "lead_scorer_reverify_schema_repair"]
    assert result.decision == "unavailable"
    assert result.details["identity_decision"] == "unavailable"
    assert result.details["required_attribute_decision"] == "unavailable"
    assert result.details["failure_class"] == "insufficient_fit_evidence"
    assert result.details["_completed_investigation_settled"] is True


def test_exact_frozen_original_reproduces_the_native_retry_failure(monkeypatch):
    original = module_for(frozen_source("completed_merger810", seam.FINALITY_FILE))
    original._source_refusal_reader = module_for(frozen_source(
        "completed_merger810", "qualification/scoring/company_verification.py"))
    result, _ = exercise(monkeypatch, original)
    assert result.decision == "unavailable"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"


def test_original_frozen_reader_ignores_additive_proof():
    reader = module_for(frozen_source("completed_merger810", "qualification/scoring/company_verification.py"))
    assert refusal_reason(reader, proof=True) == NATIVE_REFUSAL


@pytest.mark.parametrize("changes", [
    {"code": "provider_error"}, {"code": ""},
    {"reason": "website unreachable: TimeoutError"},
    {"reason": "website fetch error: TimeoutError"},
    {"reason": "website returned HTTP 503"},
    {"completed": False}, {"source_fault": True}, {"missing_repair": True},
    {"proof": False},
])
def test_real_or_incomplete_failures_keep_retryable_classification(monkeypatch, scorer, changes):
    result, _ = exercise(monkeypatch, scorer, **changes)
    assert result.decision != "match"
    assert result.details.get("failure_class") != "insufficient_fit_evidence"


@pytest.mark.parametrize("reason,code,expected", [
    (SOURCE_UNAVAILABLE, "source_blocked", False),
    (NATIVE_REFUSAL, "source_blocked", True),
    (NATIVE_REFUSAL, "provider_error", True),
    (NATIVE_REFUSAL, "", True),
    ("website unreachable: TimeoutError", "source_blocked", True),
    ("website fetch error: TimeoutError", "source_blocked", True),
    ("website returned HTTP 503", "source_blocked", True),
])
def test_only_typed_native_source_refusal_changes_the_shared_predicate(scorer, reason, code, expected):
    result = scorer.company_fit_unavailable(reason, details={"failure_reason_code": code})
    assert scorer._homepage_identity_has_retryable_failure(result) is expected


@pytest.mark.parametrize("proof,body,identity,expected", [
    (True, None, "sha256:" + "a" * 64, SOURCE_UNAVAILABLE),
    (False, None, "sha256:" + "a" * 64, NATIVE_REFUSAL),
    (True, b'{"error":{"code":"provider_unavailable"}}', "sha256:" + "a" * 64, NATIVE_REFUSAL),
    (True, None, "forged", NATIVE_REFUSAL),
])
def test_source_proof_requires_exact_host_envelope(proof, body, identity, expected):
    assert refusal_reason(verification, proof=proof, body=body, identity=identity) == expected
