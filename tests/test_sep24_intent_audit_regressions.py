"""September 24 source-delivery and prompt-routing regressions.

Prompt assertions prove that the bounded judge receives the intended rule and
evidence. They do not prove a live model's semantic verdict; provider-backed
validation remains separate.
"""

from qualification.scoring import intent_verification_three_stage as intent
from qualification.scoring import verification_helpers
from qualification.scoring.prompts import _common


def _row(
    *,
    company: str,
    claim: str,
    target: str,
    source_url: str,
    evidence_type: str,
) -> dict:
    return {
        "id": "signal-1",
        "company": company,
        "website": f"https://{company.casefold().replace(' ', '')}.example",
        "company_linkedin": "",
        "contact_linkedin": "",
        "claim": claim,
        "signal_date": "2026-03-31",
        "signal_type": "intent",
        "claimed_source_urls": [source_url],
        "_target_signal_text": target,
        "_evidence_type": evidence_type,
        "_integrity_policy": True,
        "_buyer_max_age_days": 365,
    }


def _prompts(row: dict, source_text: str) -> tuple[str, str]:
    return (
        intent._build_verification_prompt(row),
        intent._build_final_judge_prompt(
            row,
            {
                "results": [{
                    "url": row["claimed_source_urls"][0],
                    "title": "Exact submitted source",
                    "text": source_text,
                }],
                "statuses": [],
            },
        ),
    )


class _RelatedCardTrafilatura:
    @staticmethod
    def extract(_content, **_kwargs):
        return (
            "Bakong–RoamQR Linkage Expands to Enable Singapore Travellers to "
            "Pay via KHQR in Cambodia. " * 4
        )


def test_oxpay_wrong_related_extraction_recovers_visible_main_article(
    monkeypatch,
) -> None:
    html = """<html><body><main>
      <h1>Liquid Group Partners with OxPay</h1>
      <time>March 7, 2026</time>
      <div class="blog-post__body col-12">
        <p>SINGAPORE, 6 March 2026 — Liquid Group today announced a strategic
        partnership with OxPay Financial Limited to enhance QR payment acceptance
        for merchants in Singapore.</p>
        <p>Under this partnership, RoamQR will be integrated into OxPay's newly
        upgraded payments platform.</p>
      </div>
      <div class="blog-posts-grid">
        <div class="blog-index__post-content blog-index__post-content--small">
          <h4>Bakong–RoamQR Linkage Expands</h4>
          <p>Singapore travellers can pay via KHQR in Cambodia.</p>
        </div>
      </div>
    </main></body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _RelatedCardTrafilatura,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert "Liquid Group Partners with OxPay" in extracted
    assert "March 7, 2026" in extracted
    assert "strategic partnership with OxPay" in extracted
    assert "integrated into OxPay's newly upgraded payments platform" in extracted
    assert "Bakong" not in extracted
    assert "Cambodia" not in extracted


def test_visible_fallback_excludes_hidden_claims_and_nested_related_text(
    monkeypatch,
) -> None:
    html = """<html><head><style>
      .css-hidden { display: none; }
    </style></head><body><main>
      <h1>Acme Opens Its First Operating Plant</h1>
      <time>September 20, 2026</time>
      <article>
        <p>Acme opened its first operating plant on September 20, 2026.</p>
        <br></br><p>The plant began normal production that day.</p>
        <p hidden>Acme did not open the plant.</p>
        <p aria-hidden="true">Acme only proposed the plant.</p>
        <p style="visibility: hidden">The plant remains under construction.</p>
        <p class="css-hidden">Expected completion is in 2028.</p>
        <section class="related-articles">
          <h2>Beta plans a different factory</h2>
          <p>Beta expects construction to finish next year.</p>
        </section>
      </article>
    </main></body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        _RelatedCardTrafilatura,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert "Acme Opens Its First Operating Plant" in extracted
    assert "September 20, 2026" in extracted
    assert "Acme opened its first operating plant" in extracted
    assert "began normal production that day" in extracted
    assert "did not open" not in extracted
    assert "only proposed" not in extracted
    assert "under construction" not in extracted
    assert "Expected completion is in 2028" not in extracted
    assert "Beta plans" not in extracted


def test_conditional_css_does_not_hide_available_source_text(monkeypatch) -> None:
    html = """<html><head><style>
      .always-hidden { display: none; }
      @media (max-width: 600px) {
        .desktop-only { display: none; }
      }
      @supports (display: grid) {
        #grid-layout { visibility: hidden; }
      }
    </style></head><body><main>
      <p class="always-hidden">Unconditional hidden claim.</p>
      <p class="desktop-only">Desktop source evidence remains available.</p>
      <p id="grid-layout">Capability-dependent evidence remains available.</p>
    </main></body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", False)

    extracted = verification_helpers.extract_article_body(html)

    assert "Unconditional hidden claim" not in extracted
    assert "Desktop source evidence remains available" in extracted
    assert "Capability-dependent evidence remains available" in extracted


def test_same_headline_tokens_cannot_reintroduce_hidden_false_claim(
    monkeypatch,
) -> None:
    hidden_claim = (
        "Acme Opens Its First Operating Plant, but Acme did not open the plant "
        "and the project remains only a proposal. " * 4
    )

    class HiddenClaimTrafilatura:
        @staticmethod
        def extract(_content, **_kwargs):
            return hidden_claim

    html = """<html><body>
      <h1>Acme Opens Its First Operating Plant</h1>
      <article><p>Acme opened its first operating plant on September 20, 2026.
      The plant began normal production that day.</p>
      <p hidden>Acme did not open the plant and the project remains only a
      proposal.</p></article>
    </body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        HiddenClaimTrafilatura,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert "Acme opened its first operating plant" in extracted
    assert "began normal production" in extracted
    assert "did not open" not in extracted
    assert "only a proposal" not in extracted


def test_valid_main_extraction_keeps_same_event_date_without_using_fallback(
    monkeypatch,
) -> None:
    main_story = (
        "Liquid Group Partners with OxPay to Expand RoamQR Acceptance. "
        "On March 6, 2026, Liquid Group entered a strategic partnership with "
        "OxPay Financial Limited. " * 3
    )

    class MainStoryTrafilatura:
        @staticmethod
        def extract(_content, **_kwargs):
            return main_story

    html = f"""<html><body>
      <h1>Liquid Group Partners with OxPay to Expand RoamQR Acceptance</h1>
      <article><p>{main_story}</p></article>
      <div class="rich-text">A separate long generic rich-text block.</div>
    </body></html>"""
    monkeypatch.setattr(verification_helpers, "_TRAFILATURA_AVAILABLE", True)
    monkeypatch.setattr(
        verification_helpers,
        "_trafilatura",
        MainStoryTrafilatura,
        raising=False,
    )

    extracted = verification_helpers.extract_article_body(html)

    assert extracted == main_story
    assert "March 6, 2026" in extracted


def test_common_wealth_middle_event_month_reaches_final_judge_unchanged() -> None:
    source_url = "https://www.flowcap.com/post/common-wealth-case-study"
    event = (
        "In March 2026, Common Wealth closed a C$12 million Series A backed by "
        "a syndicate of institutional and individual investors."
    )
    source_text = "INTRO " * 4_000 + event + " LOOKING AHEAD" * 2_500
    assert len(source_text) <= _common.MAX_SCRAPED_CHARS
    row = _row(
        company="Common Wealth Retirement",
        claim="Common Wealth closed a C$12M Series A in March 2026.",
        target="Raised a funding round in the last 365 days.",
        source_url=source_url,
        evidence_type="FUNDING",
    )

    final_prompt = _prompts(row, source_text)[1]
    messages = intent._openrouter_user_messages(final_prompt)

    assert event in final_prompt
    assert source_url in final_prompt
    assert "".join(message["content"] for message in messages) == final_prompt


def test_grab_negative_control_rule_reaches_both_prompts() -> None:
    row = _row(
        company="Grab",
        claim="Grab entered agreements to acquire 60% of Atome Financial.",
        target="Entered a strategic partnership in the last 365 days.",
        source_url=(
            "https://www.grab.com/sg/press/others/"
            "grab-to-acquire-majority-stake-in-atome-financial/"
        ),
        evidence_type="PARTNERSHIP",
    )
    source_text = (
        "The parties entered definitive agreements for Grab to acquire a "
        "controlling 60% equity interest. The proposed transaction, if "
        "completed, is expected to close in 2027."
    )

    for prompt in _prompts(row, source_text):
        assert "requires positive evidence" in prompt
        assert "ownership transaction alone is\n        not a partnership" in prompt
        assert "target ICP does not explicitly exclude\n        acquisitions" in prompt


def test_oxpay_positive_control_source_and_rule_reach_final_prompt() -> None:
    row = _row(
        company="OxPay",
        claim=(
            "Liquid Group entered a strategic partnership with OxPay to expand "
            "RoamQR acceptance for Singapore merchants."
        ),
        target="Entered a strategic partnership in the last 365 days.",
        source_url=(
            "https://www.liquidgroup.sg/liquid-group-news/"
            "liquid-group-partners-with-oxpay-to-expand-roamqr-acceptance-"
            "for-singapore-merchants"
        ),
        evidence_type="PARTNERSHIP",
    )
    source_text = (
        "Liquid Group today announced a strategic partnership with OxPay "
        "Financial Limited. Under this partnership, RoamQR will be integrated "
        "into OxPay's payments platform."
    )

    stage_one, stage_three = _prompts(row, source_text)
    for prompt in (stage_one, stage_three):
        assert "partnership, alliance, joint venture" in prompt
        assert "Apply an OR alternative only when the exact source proves" in prompt
    assert source_text in stage_three


def test_corestack_completion_guidance_reaches_both_prompts() -> None:
    row = _row(
        company="CoreStack",
        claim="CoreStack acquired BetterCloud.",
        target="Acquired another company in the last 365 days.",
        source_url=(
            "https://www.corestack.io/blog/"
            "corestack-acquires-bettercloud-establishing-a-unified-agentic-"
            "governance-os-across-cloud-saas-and-ai/"
        ),
        evidence_type="ACQUISITION",
    )
    source_text = (
        "CoreStack and BetterCloud now provide services to over 2,000 customers. "
        "BetterCloud is part of CoreStack, and its new president will lead "
        "post-acquisition integration efforts."
    )

    for prompt in _prompts(row, source_text):
        assert "buyer now owns or controls" in prompt
        assert "active post-acquisition integration" in prompt
        assert "expected future closing do\n        not prove completion" in prompt


def test_faith_negative_control_rule_reaches_both_prompts() -> None:
    row = _row(
        company="Faith Technologies",
        claim="Faith Technologies plans three new manufacturing buildings.",
        target="Opened a new manufacturing facility in the last 365 days.",
        source_url=(
            "https://www.faithtechinc.com/news-and-insights/news/"
            "community-leaders-celebrate-groundbreaking-for-ftis-new-"
            "manufacturing-facility-in-monroe/"
        ),
        evidence_type="FACILITY_OPENING",
    )
    source_text = (
        "Leaders celebrated the groundbreaking, marking the official start of "
        "construction. The facility is expected to be completed in spring 2027."
    )

    for prompt in _prompts(row, source_text):
        assert "requires positive evidence that the facility\n        opened or began operations" in prompt
        assert "holding a groundbreaking" in prompt
        assert "expected completion date does not prove an opening" in prompt


def test_unrelated_product_launch_does_not_receive_event_type_rules() -> None:
    row = _row(
        company="Example",
        claim="Example launched a reporting API.",
        target="Launched a major product capability in the last 365 days.",
        source_url="https://example.com/news/reporting-api",
        evidence_type="PRODUCT_LAUNCH",
    )

    prompt = intent._build_verification_prompt(row)

    assert _common.PARTNERSHIP_BLOCK not in prompt
    assert _common.FACILITY_OPENING_BLOCK not in prompt
    assert _common.ACQUISITION_BLOCK not in prompt


def test_facility_category_does_not_turn_a_requested_plan_into_an_opening() -> None:
    row = _row(
        company="Example Manufacturing",
        claim="Example announced plans for a new manufacturing plant.",
        target="Announced plans for a new manufacturing plant in the last year.",
        source_url="https://example.com/news/planned-plant",
        evidence_type="FACILITY_OPENING",
    )

    prompt = intent._build_verification_prompt(row)

    assert _common.FACILITY_OPENING_BLOCK not in prompt
    assert "Announced plans for a new manufacturing plant" in prompt


def test_partnership_or_acquisition_keeps_each_alternative_independent() -> None:
    row = _row(
        company="Example",
        claim="Example acquired Beta.",
        target=(
            "Entered a strategic partnership OR acquired another company in "
            "the last year."
        ),
        source_url="https://example.com/news/acquires-beta",
        evidence_type="PARTNERSHIP",
    )

    prompt = intent._build_verification_prompt(row)

    assert _common.PARTNERSHIP_BLOCK in prompt
    assert _common.ACQUISITION_BLOCK in prompt
    assert "only to a chosen partnership alternative" in prompt
    assert "must not add a\n        partnership requirement" in prompt
    assert "buyer now owns or controls" in prompt
