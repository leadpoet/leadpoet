"""Keep visible article dates through extraction, without inferring event dates."""
from qualification.scoring import verification_helpers as helpers
from qualification.scoring.intent_verification_three_stage import _published_date_from_html


def _extract(monkeypatch, html, body):
    class Extractor:
        @staticmethod
        def extract(*args, **kwargs):
            return body
    monkeypatch.setattr(helpers, '_TRAFILATURA_AVAILABLE', True)
    monkeypatch.setattr(helpers, '_trafilatura', Extractor, raising=False)
    return helpers.extract_article_body(html)


def test_post_heading_date_survives_isomorphic_shaped_extraction(monkeypatch):
    heading = 'Isomorphic Labs Enters into a Research Collaboration with Johnson & Johnson'
    body = ('Isomorphic Labs announces a cross-modality, multi-target research collaboration '
            'with Johnson & Johnson. This partnership brings together Iso’s AI-first approach '
            'to drug discovery with Johnson & Johnson’s expertise in drug discovery and development.')
    html = f'<html><body><article><h1>{heading}</h1><div>January 20, 2026</div><p>{body}</p></article></body></html>'
    result = _extract(monkeypatch, html, body)
    assert heading in result and 'January 20, 2026' in result and body in result
    assert _published_date_from_html(html, 'https://example.test/article') == ''


def test_updated_and_event_dates_retain_their_visible_labels(monkeypatch):
    heading = 'Acme Opens New Manufacturing Facility'
    body = heading + '\n' + 'Acme opened a manufacturing facility and expanded its production capacity. ' * 5
    html = f'''<html><body><article><h1>{heading}</h1>
    <div>Published January 20, 2026. Updated September 29, 2026.</div>
    <p>{body}</p></article></body></html>'''
    result = _extract(monkeypatch, html, body)
    assert 'Published January 20, 2026. Updated September 29, 2026.' in result
    assert _published_date_from_html(html, 'https://example.test/article') == ''


def test_hidden_navigation_and_related_dates_cannot_trigger_recovery(monkeypatch):
    heading = 'Acme Opens New Manufacturing Facility'
    body = heading + '\n' + 'Acme opened a manufacturing facility and expanded its production capacity. ' * 5
    html = f'''<html><body><nav>October 1, 2026</nav><article><h1>{heading}</h1>
    <span hidden>January 20, 2026</span><span aria-hidden="true">February 20, 2026</span>
    <p>{body}</p><aside>March 20, 2026</aside>
    <section class="related-articles">April 20, 2026</section></article></body></html>'''
    result = _extract(monkeypatch, html, body)
    assert result == body
    assert '2026' not in result


def test_context_is_bounded_and_does_not_use_repeated_title_or_late_date(monkeypatch):
    heading = 'Acme Opens New Manufacturing Facility'
    body = heading + '\n' + 'Acme opened a manufacturing facility and expanded its production capacity. ' * 20
    html = f'''<html><head><title>{heading} January 1, 2026</title></head><body>
    <div>Archive date February 1, 2026</div><article><h1>{heading}</h1>
    <div>March 1, 2026</div><p>{body}</p><p>Later unrelated event April 1, 2026</p></article></body></html>'''
    result = _extract(monkeypatch, html, body)
    assert result.endswith(body)
    assert 'March 1, 2026' in result
    assert 'January 1, 2026' not in result and 'February 1, 2026' not in result
    assert 'April 1, 2026' not in result
    assert len(result) - len(body) <= 802
