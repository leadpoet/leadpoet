"""Bounded, source-local text from pages with large visible site menus."""

import asyncio

from qualification.scoring import company_evidence_investigator as investigator


# The first labels and long service list mirror the model-visible Cognizant
# fetch. The repeated middle is synthetic; no captured provider HTML is used.
MENU = (
    "AI-led mainframe modernization | Cognizant Careers News Events "
    "Investors US-EN Americas Argentina Brazil Canada "
    + "Products Solutions Services Industries " * 900
    + "Business operations to scale new ideas Communications media "
    "Digital finance Employee experience services "
)
BODY_QUOTE = (
    "Cognizant delivers end-to-end mainframe modernization services, "
    "spanning infrastructure, consulting, managed services and "
    "mainframe-as-a-service."
)


def test_visible_header_menu_cannot_displace_main_body_quote():
    # Synthetic rendering of the observed Cognizant capture: a visible div
    # menu fills the 24k prefix before the official page's body sentence.
    raw = (
        "<html><body><header><div class='mega-menu'>" + MENU +
        "</div></header><main><h1>AI-led mainframe modernization</h1>"
        f"<p>{BODY_QUOTE}</p></main></body></html>"
    )

    text = investigator._plain_text(raw)

    assert len(text) <= investigator.MAX_PAGE_CHARACTERS
    assert BODY_QUOTE in text
    assert investigator._quote_occurs(BODY_QUOTE, text)
    assert "Products Solutions Services" not in text


def test_main_body_keeps_article_date_and_feature_grid():
    raw = (
        "<html><body><header><div>" + MENU + "</div></header>"
        "<main><article><h1>Company expands its service</h1>"
        "<time>May 6, 2025</time><p>Company launched the service.</p>"
        "</article><div class='feature-grid'><p>Managed services for "
        "enterprise infrastructure.</p></div></main></body></html>"
    )

    text = investigator._plain_text(raw)

    assert "Company expands its service" in text
    assert "May 6, 2025" in text
    assert "Company launched the service." in text
    assert "Managed services for enterprise infrastructure." in text


def test_hidden_or_script_main_does_not_supply_a_quote():
    raw = (
        "<html><body><header><div>" + MENU + "</div></header>"
        f"<script><main>{BODY_QUOTE}</main></script>"
        f"<main hidden><p>{BODY_QUOTE}</p></main>"
        "<main><p>Visible company information.</p></main>"
        "</body></html>"
    )

    text = investigator._plain_text(raw)

    assert BODY_QUOTE not in text
    assert "Visible company information." in text


def test_reordered_header_and_main_cannot_make_a_new_contiguous_quote():
    raw = (
        "<html><body><header><div>" + MENU + "Managed</div></header>"
        "<main><p>services for enterprises.</p></main></body></html>"
    )

    text = investigator._plain_text(raw)

    assert "services for enterprises." in text
    assert not investigator._quote_occurs("Managed services for enterprises.", text)


def test_short_page_and_page_without_main_keep_existing_surface():
    short = "<html><body><header>Company news</header><main>Current facts</main></body></html>"
    long_without_main = "<html><body><div>" + MENU + "</div><p>Later facts.</p></body></html>"

    assert investigator._plain_text(short) == "Company news Current facts"
    assert investigator._plain_text(long_without_main) == MENU[:investigator.MAX_PAGE_CHARACTERS]


def test_main_already_in_original_window_keeps_existing_prefix():
    raw = (
        "<html><body><header>Company news</header><main>"
        "Original article title May 6, 2025 " + MENU +
        "Later text.</main></body></html>"
    )

    text = investigator._plain_text(raw)

    assert text.startswith("Company news Original article title May 6, 2025")
    assert "Later text." not in text


def test_main_projection_keeps_exact_fetched_source_identity(monkeypatch):
    url = "https://www.example.com/services/mainframe-modernization"
    raw = (
        "<html><body><header><div>" + MENU + "</div></header>"
        f"<main><p>{BODY_QUOTE}</p></main></body></html>"
    )

    async def fetch_html(_session, requested_url, **_kwargs):
        assert requested_url == url
        return 200, url, raw

    monkeypatch.delenv("SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.delenv("QUALIFICATION_SCRAPINGDOG_API_KEY", raising=False)
    monkeypatch.setattr(investigator, "_fetch_bounded_html", fetch_html)

    result = asyncio.run(investigator._fetch_page(object(), url))

    assert result["ok"] is True
    assert result["url"] == result["final_url"] == url
    assert investigator._quote_occurs(BODY_QUOTE, result["text"])
