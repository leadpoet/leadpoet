"""Empty article bodies must render; hydration data never becomes evidence."""

import asyncio
from unittest import mock

import httpx
import pytest

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring.verification_helpers import extract_article_body
from tests.test_scrapingdog_deadline_routing import _HttpxClient


def _document(article, marker="self.__next_f.push"):
    return (
        "<!doctype html><html><head><title>Product announcement</title></head>"
        "<body><nav>Navigation</nav><aside>"
        + "Sidebar and unrelated cards. " * 100 + "</aside>"
        "<h1>Product announcement</h1><time>11 Jul 2026</time>"
        + article
        + "<script>" + marker + "('" + "Unrendered article data. " * 200
        + "');</script></body></html>"
    )


@pytest.mark.parametrize("marker", ["self.__next_f.push", "__NEXT_DATA__"])
@pytest.mark.parametrize("article", [
    '<article><div class="animate-spin"></div></article>',
    '<article> <script>Acme AI Chat is live.</script></article>',
    '<article><template>Acme AI Chat is live.</template></article>',
])
def test_empty_framework_article_is_a_shell_despite_long_chrome(marker, article):
    html = _document(article, marker)
    assert len(html) > 3000
    assert intent._evaluate_sd_response(200, html) == "js_shell"
    assert "Acme AI Chat is live" not in extract_article_body(html)


@pytest.mark.parametrize("article,marker", [
    ("<article><p>Acme AI Chat is live for customers.</p></article>", "self.__next_f.push"),
    ("<article></article><article><p>Real article body.</p></article>", "__NEXT_DATA__"),
    ("<p>Article body outside an article element.</p>", "self.__next_f.push"),
    ("<article></article>", "window.otherPageData"),
    ("<template><article></article></template><p>Real page body.</p>", "self.__next_f.push"),
])
def test_real_or_unbound_article_controls_keep_existing_admission(article, marker):
    assert intent._evaluate_sd_response(200, _document(article, marker)) == "ok"


def test_app_router_without_article_keeps_existing_low_density_admission():
    html = (
        '<html><body><svg><path d="' + "M1 2 " * 2000
        + '"></path></svg><p>A concise visible page.</p>'
        '<script>self.__next_f.push([1,"data"]);</script></body></html>'
    )
    assert intent._evaluate_sd_response(200, html) == "ok"


def test_empty_article_reuses_existing_dynamic_tier_without_admitting_script_data():
    shell = _document('<article><div class="animate-spin"></div></article>')
    rendered = _document(
        "<article><h1>Product announcement</h1><p>"
        + "Acme AI Chat is live and available to customers for credit research. " * 5
        + "</p></article>"
    )
    client = _HttpxClient([
        httpx.Response(200, text=shell),
        httpx.Response(200, text=rendered),
    ])
    with mock.patch.object(intent.httpx, "AsyncClient", return_value=client), \
            mock.patch.dict("os.environ", {"SCRAPINGDOG_API_KEY": "test"}):
        result = asyncio.run(intent._scrape_sd_hardened(
            "https://arbitrary.example/product-announcement"
        ))
    assert result["ok"] is True
    assert result["stage"] == "sd:dynamic_render"
    assert result["stage_history"] == [
        ("baseline", "js_shell"), ("dynamic_render", "ok"),
    ]
    assert "Acme AI Chat is live" in result["content"]
    assert "Unrendered article data" not in result["content"]
    assert len(client.calls) == 2
    assert client.calls[0][1]["params"].get("dynamic", "false") == "false"
    assert client.calls[1][1]["params"]["dynamic"] == "true"
