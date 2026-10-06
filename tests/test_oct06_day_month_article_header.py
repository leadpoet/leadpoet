"""Keep a visible day-month article date through bounded body extraction."""
from pathlib import Path

from qualification.scoring import verification_helpers as helpers


_FIXTURE = Path(__file__).parent / "fixtures" / "intent_9fin_visible_article_header.html"


def _extract(monkeypatch, html):
    # The saved provider body begins after this exact visible article header.
    # Bind the extractor output to the same page, including its complete body.
    body_html = html.split("<!-- saved production body -->", 1)[1]
    body = helpers.visible_html_text("<html><body>" + body_html)

    class Extractor:
        @staticmethod
        def extract(*args, **kwargs):
            return body

    monkeypatch.setattr(helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(helpers, "_trafilatura", Extractor, raising=False)
    return helpers.extract_article_body(html)


def test_exact_production_day_month_header_survives(monkeypatch):
    result = _extract(monkeypatch, _FIXTURE.read_text())
    assert "11 Jul 2026" in result
    assert "AI Chat is live" in result
    assert "Your conversational research tool" in result


def test_unrelated_footer_date_does_not_replace_missing_header_date(monkeypatch):
    html = _FIXTURE.read_text().replace("11 Jul 2026", "")
    html = html.replace("</body>", "<footer>11 Jul 2026</footer></body>")
    result = _extract(monkeypatch, html)
    assert "11 Jul 2026" not in result
    assert "AI Chat is live" in result


def test_hidden_day_month_date_does_not_trigger_header_recovery(monkeypatch):
    html = _FIXTURE.read_text().replace(
        "11 Jul 2026", '<span hidden>11 Jul 2026</span>',
    )
    result = _extract(monkeypatch, html)
    assert "11 Jul 2026" not in result
    assert "AI Chat is live" in result
