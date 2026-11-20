"""Nonblocking challenge scripts do not erase visible retrieved evidence."""

import asyncio
from unittest import mock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.verification_helpers import extract_article_body
from tests.test_scrapingdog_deadline_routing import _HttpxClient


TEXT = (
    "The company is hiring a data engineer to build and maintain production "
    "data pipelines. This role works with customers to improve data quality, "
    "reporting, and analytics. Engineers review changes, test each release, "
    "and maintain reliable software for the team. "
) * 14


def document(body, *, marker='window.features = {"recaptcha":false};', title="Data Engineer"):
    return (
        "<!doctype html><html><head><title>" + title + "</title><script>"
        + marker + "</script></head><body>" + body + "</body></html>"
    )


@pytest.mark.parametrize("marker", [
    'window.features = {"recaptcha":false,"turnstile_captcha_enabled":true};',
    'load("/cdn-cgi/challenge-platform/scripts/jsd/main.js");',
])
def test_rendered_evidence_with_nonblocking_antibot_scripts_is_usable(marker):
    body = document("<article><h1>Data Engineer</h1><p>" + TEXT + "</p></article>", marker=marker)
    assert intent._has_anti_bot_marker(body) is True
    assert intent._evaluate_sd_response(200, body) == "ok"
    assert "production data pipelines" in extract_article_body(body)
    assert "recaptcha" not in extract_article_body(body)


@pytest.mark.parametrize("body", [
    "<p>Loading</p>",
    "<script>" + TEXT + "</script><p>Loading</p>",
    "<template>" + TEXT + "</template><p>Loading</p>",
    '<div hidden>' + TEXT + '</div><p>Loading</p>',
    '<div aria-hidden="true">' + TEXT + '</div><p>Loading</p>',
    '<div style="display:none">' + TEXT + '</div><p>Loading</p>',
    '<style>.concealed{display:none}</style><div class="concealed">' + TEXT + '</div><p>Loading</p>',
    '<h1>Verify you are human</h1><p>' + TEXT + '</p>',
    '<h1>Access denied</h1><p>' + TEXT + '</p>',
    '<aside>Verify you are human</aside><p>' + TEXT + '</p>',
    '<nav>Access denied</nav><p>' + TEXT + '</p>',
])
def test_challenges_and_hidden_or_thin_bodies_still_fail(body):
    assert intent._evaluate_sd_response(200, document(body)) != "ok"


def test_challenge_title_and_non_html_markers_still_fail():
    assert intent._evaluate_sd_response(
        200, document("<p>" + TEXT + "</p>", title="Just a moment")
    ) == "anti_bot_marker"
    assert intent._evaluate_sd_response(200, "captcha " + TEXT) == "anti_bot_marker"
    assert intent._evaluate_sd_response(200, '{"captcha":"' + TEXT + '"}') == "anti_bot_marker"


@pytest.mark.parametrize("body", [
    "<p>Loading</p>",
    '<div hidden>' + TEXT + '</div><p>Loading</p>',
])
def test_long_document_title_does_not_supply_visible_body_proof(body):
    assert intent._evaluate_sd_response(200, document(body, title=TEXT)) != "ok"


def test_rendered_body_uses_one_existing_fetch_without_escalation():
    body = document("<article><h1>Data Engineer</h1><p>" + TEXT + "</p></article>")
    client = _HttpxClient([httpx.Response(200, text=body)])
    with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
            mock.patch.object(intent, "_try_wayback", new=mock.AsyncMock(side_effect=AssertionError("unexpected retry"))), \
            mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
        result = asyncio.run(intent._scrape_sd_hardened("https://arbitrary.example/jobs/engineer"))
    assert result["ok"] is True and result["stage"] == "sd:baseline"
    assert result["stage_history"] == [("baseline", "ok")]
    assert len(client.calls) == 1
    assert "production data pipelines" in result["content"]
    assert "recaptcha" not in result["content"]
