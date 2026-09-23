"""Verified first-party careers indexes get one bounded dynamic recovery."""

from __future__ import annotations

from pathlib import Path
import httpx
import pytest

from qualification.scoring import competition
from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import verification_helpers


FIXTURES = Path(__file__).parent / "fixtures" / "sep20_p1_dynamic_jobs"


class _ScrapingDogClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _html(name: str) -> str:
    return (FIXTURES / name).read_text()


def _identity(company: str, domain: str) -> dict:
    return {
        "decision": "match",
        "reason_code": "verifier_accepted",
        "submitted_name": company.casefold(),
        "submitted_domain": domain,
        "submitted_linkedin_slug": company.casefold(),
        "observed_name": company.casefold(),
        "observed_domain": domain,
        "observed_linkedin_slug": company.casefold(),
        "evidence_source": "company_web_reverification",
        "verified_legal_name_aliases": [],
    }


def _verdict(source_url: str, status: str, quote: str = "") -> dict:
    return {
        "answer": {
            "overall_verdict": (
                "qualified" if status == "supported" else "not_qualified"
            ),
            "overall_confidence": "high",
            "signal_evaluations": [{
                "claim": "The company has a current technical opening.",
                "signal_status": status,
                "confidence": "high" if status == "supported" else "medium",
                "same_entity_check": "pass",
                "verification_mode": "source_grounded",
                "source_accessibility": "accessible",
                "evidence_urls_used": [source_url] if status == "supported" else [],
                "supporting_quotes": [quote] if quote else [],
                "contradicting_quotes": [],
                "unsupported_parts": [],
                "risk_notes": [],
                "claim_matches_miner_date": "no_date_in_content",
            }],
        },
        "model": "test-model",
        "usage": {},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "company", "domain", "source_url", "role"),
    [
        (
            "trace3",
            "Trace3",
            "trace3.com",
            "https://www.trace3.com/careers",
            "Atlan Implementation Project Manager",
        ),
        (
            "divergeit",
            "DivergeIT",
            "divergeit.com",
            "https://divergeit.com/careers/",
            "Dedicated Support Engineer",
        ),
        (
            "synoptek",
            "Synoptek",
            "synoptek.com",
            "https://careers.synoptek.com/synoptek/",
            "Azure Solutions | USA | July 9",
        ),
    ],
)
async def test_full_verifier_recovers_rendered_first_party_job_listings(
    monkeypatch, label, company, domain, source_url, role
):
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=(
                _html(f"{label}_dynamic_sanitized.html")
                + (" Current openings published by the company." * 100
                   if label == "synoptek" else "")
            ),
            headers={},
        ),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(source_url, "supported", role)

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)
    if label == "trace3":
        assert not intent._looks_like_job_body(
            _html("trace3_dynamic_sanitized.html")
        )

    result = await intent.verify_three_stage(
        None,
        company_name=company,
        company_linkedin=f"https://www.linkedin.com/company/{company.casefold()}",
        company_website=f"https://{domain}",
        source_url=source_url,
        miner_claim="The company has a current technical opening.",
        target_signal_text="Cloud OR infrastructure OR implementation OR security role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=True,
        verified_company_identity=_identity(company, domain),
        integrity_policy=True,
    )

    assert result["client_ready"] is True
    assert result["decision"] == "approve"
    assert len(prompts) == 1
    assert role in prompts[0]
    assert len(prompts[0]) < intent.MAX_SCRAPED_CHARS
    assert len(client.calls) == 1
    assert result["job_publisher_relationship"] == "verified"
    assert result["verified_job_source_urls"] == [source_url]
    assert client.calls[0][1]["params"] == {
        "api_key": "test",
        "url": source_url,
        "dynamic": "true",
        "wait": "5000",
    }
    if label == "synoptek":
        assert (
            "Network Administrator | Las Vegas, United States | August 18"
            in prompts[0]
        )
        assert (
            "Salesforce Functional Consultant | Glendale, United States | August 17"
            in prompts[0]
        )


def test_verified_careers_subdomain_tenant_has_job_board_source_shape():
    source_url = "https://careers.synoptek.com/synoptek/"

    assert intent._is_careers_index_url(source_url)
    assert competition._evidence_source(
        source_url,
        company_website="https://www.synoptek.com/",
    ) == "job_board"
    assert competition._evidence_source(
        "https://careers.other.example/synoptek/",
        company_website="https://www.synoptek.com/",
    ) == "news"


@pytest.mark.asyncio
async def test_synoptek_unproved_exact_count_stays_rejected(monkeypatch):
    source_url = "https://careers.synoptek.com/synoptek/"
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=(
                _html("synoptek_dynamic_sanitized.html")
                + (" Current openings published by Synoptek." * 100)
            ),
            headers={},
        ),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(source_url, "partially_supported")

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)

    result = await intent.verify_three_stage(
        None,
        company_name="Synoptek",
        company_linkedin="https://www.linkedin.com/company/synoptek",
        company_website="https://synoptek.com",
        source_url=source_url,
        miner_claim="Synoptek has exactly 14 United States openings.",
        target_signal_text="Cloud OR infrastructure OR implementation role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=True,
        verified_company_identity=_identity("Synoptek", "synoptek.com"),
        integrity_policy=True,
    )

    assert len(prompts) == 1
    assert result["client_ready"] is False
    assert result["decision"] == "review"
    assert result["rejection_reason"] == "stage3_review"


@pytest.mark.asyncio
async def test_first_party_linked_greenhouse_root_enumerates_bound_jobs(
    monkeypatch,
):
    source_url = "https://www.trace3.com/careers"
    board_payload = {
        "jobs": [
            {
                "id": 8119730,
                "title": "Sr. Solutions Architect | Security",
                "absolute_url": (
                    "https://job-boards.greenhouse.io/trace3/jobs/8119730"
                ),
                "location": {"name": "Remote, United States"},
                "first_published": "2026-09-18T10:30:00-07:00",
            },
            {
                "id": 8104120,
                "title": "Atlan Implementation Project Manager",
                "absolute_url": (
                    "https://job-boards.greenhouse.io/trace3/jobs/8104120"
                ),
                "location": {"name": "Irvine, California"},
                "first_published": "2026-09-15T09:00:00-07:00",
            },
        ],
    }
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=(
                _html("trace3_board_root_sanitized.html")
                + (" Current public careers information." * 100)
            ),
            headers={},
        ),
        httpx.Response(200, json=board_payload, headers={}),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(
            source_url,
            "supported",
            "Sr. Solutions Architect | Security",
        )

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)

    result = await intent.verify_three_stage(
        None,
        company_name="Trace3",
        company_linkedin="https://www.linkedin.com/company/trace3",
        company_website="https://trace3.com",
        source_url=source_url,
        miner_claim="The company has a current technical opening.",
        target_signal_text="Cloud OR infrastructure OR implementation OR security role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=True,
        verified_company_identity=_identity("Trace3", "trace3.com"),
        integrity_policy=True,
    )

    assert result["client_ready"] is True
    assert result["job_publisher_relationship"] == "verified"
    assert result["verified_job_source_urls"] == [source_url]
    assert len(client.calls) == 2
    assert client.calls[1][1]["params"] == {
        "api_key": "test",
        "url": (
            "https://boards-api.greenhouse.io/v1/boards/trace3/jobs"
        ),
        "dynamic": "false",
    }
    assert "Remote, United States" in prompts[0]
    assert "2026-09-18T10:30:00-07:00" in prompts[0]
    assert "Irvine, California" in prompts[0]


@pytest.mark.asyncio
async def test_linked_greenhouse_empty_board_returns_no_listing(monkeypatch):
    client = _ScrapingDogClient([
        httpx.Response(200, json={"jobs": []}, headers={}),
        httpx.Response(200, json={"jobs": []}, headers={}),
    ])
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    result = await intent._scrape_linked_greenhouse_board(
        "https://job-boards.greenhouse.io/trace3?error=true",
        company_domain="trace3.com",
        company_name="Trace3",
    )

    assert result["ok"] is False
    assert result["observed_job_link_count"] == 0
    assert result["stage"] == "greenhouse_board_api_exhausted"
    assert len(client.calls) == 2


def test_linked_board_rejects_untrusted_and_misbound_roots():
    source_url = "https://www.trace3.com/careers"
    untrusted = (
        '<a href="https://attacker.example/trace3?error=true">Jobs</a>'
    )
    misbound = (
        '<a href="https://job-boards.greenhouse.io/acme?error=true">Jobs</a>'
    )

    assert intent._linked_greenhouse_board_url(
        untrusted,
        source_url,
        company_domain="trace3.com",
        company_name="Trace3",
    ) == ""
    assert intent._linked_greenhouse_board_url(
        misbound,
        source_url,
        company_domain="trace3.com",
        company_name="Trace3",
    ) == ""
    assert intent._greenhouse_board_identity(
        "https://job-boards.greenhouse.io/trace3?next=https://attacker.example"
    ) is None


@pytest.mark.asyncio
async def test_first_party_listing_receipt_does_not_autoapprove(monkeypatch):
    source_url = "https://www.trace3.com/careers"
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=_html("trace3_dynamic_sanitized.html"),
            headers={},
        ),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(source_url, "contradicted")

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)

    result = await intent.verify_three_stage(
        None,
        company_name="Trace3",
        company_linkedin="https://www.linkedin.com/company/trace3",
        company_website="https://trace3.com",
        source_url=source_url,
        miner_claim="The company has a current technical opening.",
        target_signal_text="Cloud infrastructure role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=True,
        verified_company_identity=_identity("Trace3", "trace3.com"),
    )

    assert len(prompts) == 1
    assert result["client_ready"] is False
    assert result["decision"] == "reject"
    assert result["rejection_reason"] == "stage3_contradicted"


@pytest.mark.asyncio
async def test_dynamic_fetch_threads_parsed_listing_count(monkeypatch):
    source_url = "https://www.trace3.com/careers"
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=_html("trace3_dynamic_sanitized.html"),
            headers={},
        ),
    ])
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    fetched = await intent._fetch_sd_then_exa(
        [source_url],
        prefer_dynamic_job_index=True,
    )

    assert fetched["results"][0]["meta"] == {
        "kind": "first_party_careers_index",
        "observed_job_link_count": 3,
    }
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_first_party_careers_page_without_job_links_stops_before_stage3(
    monkeypatch,
):
    source_url = "https://www.trace3.com/careers"
    shell = (
        "<html><body><h1>Trace3 Careers</h1>"
        '<a href="/about">About Trace3</a>'
        + ("<p>Learn about our workplace.</p>" * 150)
        + "</body></html>"
    )
    client = _ScrapingDogClient([
        httpx.Response(200, text=shell, headers={}),
        httpx.Response(200, text=shell, headers={}),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(source_url, "unable_to_verify")

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)

    result = await intent.verify_three_stage(
        None,
        company_name="Trace3",
        company_linkedin="https://www.linkedin.com/company/trace3",
        company_website="https://trace3.com",
        source_url=source_url,
        miner_claim="The company has a current technical opening.",
        target_signal_text="Cloud infrastructure role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=True,
        verified_company_identity=_identity("Trace3", "trace3.com"),
    )

    assert len(prompts) == 0
    assert len(client.calls) == 2
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("company", "domain", "verified_identity"),
    [
        ("Trace3", "trace3.com", None),
        ("Acme", "acme.com", _identity("Acme", "acme.com")),
    ],
    ids=("missing-independent-identity", "wrong-independent-identity"),
)
async def test_unverified_careers_domain_cannot_supply_listing_receipt(
    monkeypatch, company, domain, verified_identity,
):
    source_url = "https://www.trace3.com/careers"
    client = _ScrapingDogClient([
        httpx.Response(
            200,
            text=_html("trace3_dynamic_sanitized.html"),
            headers={},
        ),
    ])
    prompts = []

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(source_url, "unable_to_verify")

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)

    result = await intent.verify_three_stage(
        None,
        company_name=company,
        company_linkedin="",
        company_website=f"https://{domain}",
        source_url=source_url,
        miner_claim="The company has a current technical opening.",
        target_signal_text="Cloud infrastructure role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=False,
        verified_company_identity=verified_identity,
    )

    assert len(prompts) == 0
    assert len(client.calls) == 1
    assert result["rejection_reason"] == "job_body_not_in_fetched_content"


@pytest.mark.asyncio
async def test_dynamic_failure_preserves_accepted_baseline_and_stops_escalation(
    monkeypatch,
):
    baseline = _html("divergeit_baseline_incomplete.html")
    client = _ScrapingDogClient([
        httpx.Response(503, text="provider unavailable", headers={}),
        httpx.Response(200, text=baseline, headers={}),
    ])
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)

    result = await intent._scrape_sd_hardened(
        "https://divergeit.com/careers/",
        prefer_dynamic_job_index=True,
    )

    assert result["ok"] is True
    assert result["stage"] == "sd:baseline"
    assert "Senior Consultant" in result["content"]
    assert result["stage_history"] == [
        ("dynamic_render", "http_503"),
        ("baseline", "ok"),
    ]
    assert len(client.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("company_quality", [True, False])
async def test_full_verifier_preserves_visible_cards_after_dynamic_failure(
    monkeypatch, company_quality,
):
    source_url = "https://divergeit.com/careers/"
    page = _html("divergeit_visible_cards_sanitized.html")
    client = _ScrapingDogClient([
        httpx.Response(503, text="provider unavailable", headers={}),
        httpx.Response(200, text=page, headers={}),
    ])
    prompts = []

    class _ArticleOnlyTrafilatura:
        @staticmethod
        def extract(_content, **_kwargs):
            return (
                "Live openings. Review responsibilities and apply now. "
                "Senior Consultant Consulting Full-Time. "
                + "Current careers information. " * 20
            )

    async def call_openrouter(_client, _model, prompt):
        prompts.append(prompt)
        return _verdict(
            source_url,
            "supported",
            "Dedicated Support Engineer Managed Services Full-Time",
        )

    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "test")
    monkeypatch.setattr(intent.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(intent, "_call_openrouter", call_openrouter)
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _ArticleOnlyTrafilatura,
        raising=False,
    )

    result = await intent.verify_three_stage(
        None,
        company_name="DivergeIT",
        company_linkedin=(
            "https://www.linkedin.com/company/divergeit"
            if company_quality else ""
        ),
        company_website="https://divergeit.com",
        source_url=source_url,
        miner_claim="The company has current managed-services openings.",
        target_signal_text="Cloud OR infrastructure OR implementation OR security role",
        evidence_type="HIRING",
        declared_source="job_board",
        stage1_soft_reject=True,
        company_quality=company_quality,
        verified_company_identity=_identity("DivergeIT", "divergeit.com"),
        integrity_policy=True,
    )

    assert result["client_ready"] is True, result
    assert result["decision"] == "approve"
    assert len(prompts) == 1
    assert "Dedicated Support Engineer Managed Services Full-Time" in prompts[0]
    assert "Sr. Help Desk Technician IT Service Desk Full-Time" in prompts[0]
    assert "Technical Account Manager Managed Services Full-Time" in prompts[0]
    assert "Fake Script Role" not in prompts[0]
    assert "Fake Template Role" not in prompts[0]
    assert "Careers index noise" not in prompts[0]
    assert "Unrelated navigation noise" not in prompts[0]
    assert len(prompts[0]) < intent.MAX_SCRAPED_CHARS
    assert len(client.calls) == 2
    assert result["job_publisher_relationship"] == "verified"


def test_dynamic_recovery_scope_excludes_exact_postings_and_queries():
    assert intent._is_careers_index_url("https://example.com/about/careers")
    assert intent._is_careers_index_url("https://example.com/jobs/")
    assert not intent._is_careers_index_url(
        "https://example.com/careers/platform-engineer"
    )
    assert not intent._is_careers_index_url(
        "https://example.com/careers?department=engineering"
    )
    assert not intent._is_careers_index_url(
        "https://careers.example.com/tenant/jobs/12345"
    )
