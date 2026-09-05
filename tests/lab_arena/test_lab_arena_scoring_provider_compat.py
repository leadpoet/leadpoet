"""Bounded miner-score translation tests; no live provider calls."""

import json

import pytest

from lab_arena import scoring_provider_compat as compat


def envelope(data, *, status="completed"):
    return json.dumps(
        {"job_id": "test", "status": status, "result": {"data": data}}
    ).encode("utf-8")


def route(operation_id, parameters, *, round_id="arena-2026-09-04"):
    return compat.route_for(
        kind="score",
        funding_source="miner_key",
        round_id=round_id,
        operation_id=operation_id,
        parameters=parameters,
    )


def test_routes_apply_only_to_miner_funded_scoring():
    request = {"url": "https://example.com/about", "dynamic": False}
    assert route("scrapingdog.scrape", request) is not None
    assert compat.route_for(
        kind="execute",
        funding_source="miner_key",
        round_id="arena-2026-09-04",
        operation_id="scrapingdog.scrape",
        parameters=request,
    ) is None
    assert compat.route_for(
        kind="score",
        funding_source="host",
        round_id="arena-2026-09-04",
        operation_id="scrapingdog.scrape",
        parameters=request,
    ) is None


def test_firecrawl_route_requires_raw_html_and_exact_source_identity():
    requested_url = "https://example.com/about"
    selected = route(
        "scrapingdog.scrape", {"url": requested_url, "dynamic": True}
    )
    assert selected.effective_parameters == {
        "tool": compat.FIRECRAWL_TOOL,
        "payload": {
            "url": requested_url,
            "formats": ["rawHtml"],
            "onlyMainContent": False,
            "maxAge": 0,
            "timeout": 60_000,
            "storeInCache": False,
        },
    }
    status, headers, body = compat.adapt_response(
        selected,
        status=200,
        headers={},
        body=envelope(
            {
                "rawHtml": "<!doctype html><head><meta name='x'></head>",
                "metadata": {
                    "sourceURL": requested_url,
                    "url": requested_url,
                    "statusCode": 200,
                },
            }
        ),
    )
    assert status == 200 and headers["content-type"].startswith("text/html")
    assert b"<head>" in body

    with pytest.raises(compat.CompatibilityResponseError, match="source_url_mismatch"):
        compat.adapt_response(
            selected,
            status=200,
            headers={},
            body=envelope(
                {
                    "rawHtml": "<html></html>",
                    "metadata": {
                        "sourceURL": "https://other.example/",
                        "url": requested_url,
                        "statusCode": 200,
                    },
                }
            ),
        )


@pytest.mark.parametrize(
    ("url", "kind", "data"),
    [
        (
            "https://api.ashbyhq.com/posting-api/job-board/acme",
            "ashby",
            {"jobs": [], "meta": {}},
        ),
        (
            "https://boards-api.greenhouse.io/v1/boards/acme/jobs/12345?content=true",
            "greenhouse",
            {
                "id": 12345,
                "absolute_url": "https://boards.greenhouse.io/acme/jobs/12345",
                "title": "Engineer",
                "company_name": "Acme",
                "content": "Details",
            },
        ),
        (
            "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/careers/job/City/Role_R123",
            "workday",
            {"jobPostingInfo": {"title": "Engineer"}},
        ),
    ],
)
def test_exact_ats_json_uses_generic_http_without_html_conversion(url, kind, data):
    selected = route("scrapingdog.scrape", {"url": url, "dynamic": False})
    assert selected.adapter == "generic_ats_json:" + kind
    assert selected.effective_parameters == {
        "tool": compat.GENERIC_HTTP_TOOL,
        "payload": {
            "url": url,
            "method": "GET",
            "follow_redirects": False,
            "timeout_ms": 60_000,
        },
    }
    status, headers, body = compat.adapt_response(
        selected, status=200, headers={}, body=envelope(data)
    )
    assert status == 200 and headers["content-type"] == "application/json"
    assert json.loads(body) == data


def test_generic_http_preserves_explicit_target_error_and_rejects_unknown_success():
    selected = route(
        "scrapingdog.scrape",
        {
            "url": "https://boards-api.greenhouse.io/v1/boards/acme/jobs/12345?content=true",
            "dynamic": False,
        },
    )
    status, _headers, body = compat.adapt_response(
        selected,
        status=200,
        headers={},
        body=envelope({"status": 404, "error": "Job not found"}),
    )
    assert status == 404 and json.loads(body)["error"] == "Job not found"
    with pytest.raises(compat.CompatibilityResponseError, match="invalid_generic"):
        compat.adapt_response(
            selected, status=200, headers={}, body=envelope({"unexpected": True})
        )


def test_firecrawl_does_not_invent_success_without_target_status():
    url = "https://example.com/about"
    selected = route("scrapingdog.scrape", {"url": url})
    with pytest.raises(compat.CompatibilityResponseError, match="missing_firecrawl_target_status"):
        compat.adapt_response(selected, status=200, headers={}, body=envelope({
            "rawHtml": "<html>Access denied</html>",
            "metadata": {"sourceURL": url, "url": url},
        }))


def test_harvest_job_preserves_closed_and_stale_evidence_fields():
    selected = route(
        "scrapingdog.linkedinjobs",
        {"job_id": "4012345678"},
        round_id="arena-2026-09-04-restart",
    )
    assert selected.effective_parameters == {
        "tool": compat.HARVEST_JOB_TOOL,
        "payload": {"jobId": "4012345678"},
    }
    status, _headers, body = compat.adapt_response(
        selected,
        status=200,
        headers={},
        body=envelope(
            {
                "status": 200,
                "element": {
                    "id": "4012345678",
                    "title": "Engineer",
                    "company": {"name": "Acme"},
                    "descriptionText": "Build systems",
                    "location": {"linkedinText": "Seattle, WA"},
                    "postedDate": "2026-01-01T00:00:00Z",
                    "expireAt": "2026-08-01T00:00:00Z",
                    "jobState": "SUSPENDED",
                    "jobApplicationLimitReached": False,
                    "employmentType": "FULL_TIME",
                },
            }
        ),
    )
    result = json.loads(body)
    assert status == 200
    assert result["job_posting_time"] == "246 days ago"
    assert "applications closed" in result["jobs_status"]
    assert "expired" in result["jobs_status"]
    assert result["job_description"] == "Build systems"
    from qualification.scoring.intent_verification_three_stage import (
        _LINKEDIN_JOB_CLOSED_RE,
        _parse_relative_age_to_months,
    )

    assert _LINKEDIN_JOB_CLOSED_RE.search(result["jobs_status"])
    assert _parse_relative_age_to_months(result["job_posting_time"]) > 6


def test_harvest_post_and_twitter_post_match_legacy_shapes_and_ids():
    linkedin = route("scrapingdog.profile_post", {"id": "7489978607814144000"})
    assert linkedin.effective_parameters["payload"]["url"].endswith(
        "urn:li:activity:7489978607814144000/"
    )
    _status, _headers, body = compat.adapt_response(
        linkedin,
        status=200,
        headers={},
        body=envelope(
            {
                "status": 200,
                "element": {
                    "id": "7489978607814144000",
                    "content": "LinkedIn text",
                    "author": {"name": "Ada"},
                    "postedAt": {"date": "2026-09-01T00:00:00Z"},
                },
            }
        ),
    )
    assert json.loads(body)["post_results"]["post_text"] == "LinkedIn text"

    twitter = route("scrapingdog.x_post", {"tweetId": "2039690176843170070"})
    assert twitter.effective_parameters == {
        "tool": compat.TWITTER_POST_TOOL,
        "payload": {"tweet_ids": "2039690176843170070"},
    }
    _status, _headers, body = compat.adapt_response(
        twitter,
        status=200,
        headers={},
        body=envelope(
            {
                "status": 200,
                "tweets": [
                    {
                        "id": "2039690176843170070",
                        "text": "X text",
                        "createdAt": "Thu Apr 02 13:03:28 +0000 2026",
                        "author": {"userName": "ada", "name": "Ada"},
                    }
                ],
            }
        ),
    )
    assert json.loads(body) == {
        "created_at": "Thu Apr 02 13:03:28 +0000 2026",
        "full_tweet": "X text",
        "user": {"name": "Ada", "screen_name": "ada"},
    }

    with pytest.raises(compat.CompatibilityResponseError, match="id_mismatch"):
        compat.adapt_response(
            twitter,
            status=200,
            headers={},
            body=envelope(
                {"status": 200, "tweets": [{"id": "other", "text": "bad"}]}
            ),
        )


def test_incomplete_or_malformed_envelopes_fail_closed():
    selected = route(
        "scrapingdog.scrape",
        {"url": "https://example.com/about", "dynamic": False},
    )
    for body in (b"not json", envelope({}, status="queued")):
        with pytest.raises(compat.CompatibilityResponseError):
            compat.adapt_response(selected, status=200, headers={}, body=body)
