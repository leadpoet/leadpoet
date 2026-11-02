"""Exact archived publication metadata survives the normal intent fetch path."""
import asyncio
import json

import pytest

from qualification.scoring import intent_verification_three_stage as intent
from tests.test_intent_failed_archive_same_event import install_exact_failures
from tests.test_intent_wayback_empty_body import SOURCE, SNAPSHOT


def article(*, canonical=None, **fields):
    node = {"@type": "NewsArticle", "headline": "A supplier partnership",
            "datePublished": "2026-02-16T11:00:00+00:00", **fields}
    canonical = canonical or SNAPSHOT.replace("/https:", "mp_/https:")
    return (f'<html><head><link rel="canonical" href="{canonical}">'
            f'<script type="application/ld+json">{json.dumps(node)}</script>'
            '</head><body><h1>A supplier partnership</h1><p>'
            + 'The company announced a partnership with a data provider. ' * 25
            + '</p></body></html>')


@pytest.mark.parametrize("fields", [{}, {"mainEntityOfPage": SOURCE},
                                  {"mainEntityOfPage": {"@id": SOURCE}}])
def test_archive_date_reaches_native_fetch_and_prompt(monkeypatch, fields):
    calls = install_exact_failures(monkeypatch, archive=article(**fields))
    fetched = asyncio.run(intent._fetch_sd_then_exa([SOURCE]))
    row = fetched["results"][0]
    assert row["url"] == SOURCE
    assert row["source_publication_date"] == "2026-02-16"
    assert intent._project_contents_for_prompt(fetched)["results"][0][
        "source_publication_date"] == "2026-02-16"
    assert not any(method == "POST" for method, _ in calls)


@pytest.mark.parametrize("canonical", [SOURCE, SNAPSHOT,
    SNAPSHOT.replace("/https:", "mp_/https:")])
def test_bound_snapshot_keeps_existing_canonical_spellings(monkeypatch, canonical):
    install_exact_failures(monkeypatch, archive=article(canonical=canonical))
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result.get("source_publication_date") == "2026-02-16"


@pytest.mark.parametrize("final", [
    SNAPSHOT.replace(SOURCE, "https://foreign.example/article"),
    SNAPSHOT + "?other=1",
    SNAPSHOT.replace("web.archive.org", "web.archive.org.foreign.example"),
    SNAPSHOT.replace("web.archive.org", "user@web.archive.org"),
    SNAPSHOT.replace("web.archive.org", "web.archive.org:444"),
    SNAPSHOT + "#another-page",
])
def test_foreign_snapshot_response_cannot_supply_publication(monkeypatch, final):
    install_exact_failures(monkeypatch, final_snapshot=final,
                           archive=article(canonical=SOURCE))
    assert not asyncio.run(intent._try_wayback(SOURCE)).get("source_publication_date")


@pytest.mark.parametrize("fields", [
    {"canonical": SNAPSHOT + "?other=1"},
    {"canonical": SNAPSHOT.replace(SOURCE, "https://foreign.example/article")},
    {"canonical": SNAPSHOT.replace("web.archive.org", "web.archive.org.foreign.example")},
    {"canonical": SNAPSHOT.replace("web.archive.org", "user@web.archive.org")},
    {"canonical": SNAPSHOT.replace("web.archive.org", "web.archive.org:443")},
    {"canonical": SNAPSHOT + "#another-page"},
    {"mainEntityOfPage": "https://foreign.example/article"},
    {"datePublished": "2026-02-30"},
    {"headline": "A different article"},
])
def test_archive_wrapper_cannot_relax_article_binding(monkeypatch, fields):
    install_exact_failures(monkeypatch, archive=article(**fields))
    assert not asyncio.run(intent._try_wayback(SOURCE)).get("source_publication_date")


def test_direct_page_does_not_gain_archive_authority():
    assert intent._published_date_from_html(article(), SOURCE) == ""


def test_archive_publication_is_not_snapshot_date_or_event_date(monkeypatch):
    install_exact_failures(monkeypatch, archive=article())
    result = asyncio.run(intent._try_wayback(SOURCE))
    assert result.get("source_publication_date") == "2026-02-16"
    assert "event_date" not in result
    assert "2026-10-08" not in result.get("source_publication_date", "")
