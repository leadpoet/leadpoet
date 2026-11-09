"""Use the existing narrow entrance-animation rule for article extraction."""

import pytest

from qualification.scoring.verification_helpers import extract_article_body


CLAIM = (
    "On January 13, 2026, Acme released its new analytics assistant and MCP server. "
    "The assistant is now available to all customers and provides governed access "
    "to analytics data. The separate managed language model remains in private preview."
)


def page(*, attrs='class="txt-wrap wow fadeIn" style="visibility: hidden; animation-name: none;"',
         ancestor='', css=''):
    # Structure and animation attributes retained from the Sisense provider
    # response. Synthetic article text keeps the fixture small and focused.
    return f'''<html><head><title>Acme advances analytics | Acme</title><style>{css}</style></head>
      <body><nav>Menu Free trial Demo</nav><main class="content-main">
      <div {ancestor}><div {attrs}><h1>Acme advances analytics</h1><p>{CLAIM}</p></div></div>
      </main></body></html>'''


def test_scroll_reveal_article_survives_extraction():
    result = extract_article_body(page())
    assert CLAIM in result
    assert "private preview" in result  # Preserve limiting evidence too.


@pytest.mark.parametrize("kwargs", [
    {"ancestor": "hidden"}, {"ancestor": 'aria-hidden="true"'},
    {"ancestor": 'style="display:none"'}, {"css": ".txt-wrap { display:none; }"},
    {"attrs": 'class="txt-wrap wow fadeIn" hidden style="visibility:hidden;animation-name:none;"'},
    {"attrs": 'class="txt-wrap wow fadeIn" style="display:none;visibility:hidden;animation-name:none;"'},
    {"attrs": 'class="txt-wrap wow fadeOut" style="visibility:hidden;animation-name:none;"'},
    {"attrs": 'class="txt-wrap fadeIn" style="visibility:hidden;animation-name:none;"'},
    {"attrs": 'class="txt-wrap wow fadeIn" style="visibility:hidden;"'},
])
def test_hidden_and_non_entrance_content_stays_excluded(kwargs):
    assert CLAIM not in extract_article_body(page(**kwargs))
