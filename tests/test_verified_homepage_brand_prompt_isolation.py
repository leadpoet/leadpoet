"""The host proof marker cannot change either model's identity prompt."""

import asyncio
from copy import deepcopy
from datetime import date
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_evidence_investigator as investigator
from qualification.scoring import lead_scorer
from qualification.scoring.company_fit_decision import CompanyFitDecisionResult
from qualification.scoring.evaluation_clock import use_evaluation_date
from tests.test_fit_evidence_url_compat import _icp


class CapturedPrompt(BaseException):
    """Stop at the actual model request without admitting any provider call."""


@pytest.mark.parametrize("boundary", ["web", "investigator"])
@pytest.mark.parametrize("marker", [True, False, 1, {"malformed": True}])
def test_actual_prompt_bytes_ignore_only_proof_marker_and_keep_server_anchor(monkeypatch, boundary, marker):
    fixture = json.loads((Path(__file__).parent / "fixtures/verified_homepage_brand_maa.json").read_text())
    homepage = CompanyFitDecisionResult(**fixture["homepage"])
    anchor = lead_scorer._verified_homepage_identity_anchor(homepage)
    assert anchor["verified_homepage_brand_alias"] is True
    anchor["verified_homepage_brand_alias"] = marker
    before = deepcopy(anchor)
    prior = {key: value for key, value in anchor.items() if key != "verified_homepage_brand_alias"}
    requests = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
    monkeypatch.setenv("EXA_API_KEY", "offline-test-key")

    async def web_request(**request):
        requests.append(request["prompt"].encode())
        raise CapturedPrompt()

    async def investigation_request(_session, _url, *, headers, payload, **_kwargs):
        requests.append(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        raise CapturedPrompt()

    monkeypatch.setattr(lead_scorer, "_request_company_reverify_json", web_request)
    monkeypatch.setattr(investigator, "_post_openrouter_json", investigation_request)
    for identity in (prior, anchor):
        with use_evaluation_date(date(2026, 10, 27)), pytest.raises(CapturedPrompt):
            if boundary == "web":
                monkeypatch.setattr(lead_scorer, "_verified_homepage_identity_anchor", lambda _value: identity)
                asyncio.run(lead_scorer._llm_reverify_company(
                    CompanyOutput.model_construct(**fixture["company"]), _icp(),
                    require_company_fit_dimensions=True, verified_homepage_identity=homepage,
                ))
            else:
                asyncio.run(investigator.investigate_company_evidence(
                    company_locator={"name": fixture["company"]["company_name"],
                                     "website": fixture["company"]["company_website"],
                                     "linkedin": fixture["company"]["company_linkedin"]},
                    targets=("industry",), requested_industry="Real estate",
                    verified_homepage_identity=identity,
                ))
    assert len(requests) == 2
    assert requests[1] == requests[0]
    assert b"verified_homepage_brand_alias" not in requests[1]
    assert anchor == before
    assert fixture["homepage"]["details"]["identity"]["verified_brand_name"] == "MAA"
