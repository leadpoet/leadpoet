"""Source/role controls; model semantic interpretation needs live comparison."""
from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring.product_coverage import coverage_matches


CRITERION = "Manufactures interior and exterior surfaces"
URL = "https://orbit.example/products"


def validate(quote, role):
    raw = {"target": "industry", "status": "VERIFIED", "observed_industry": "Surface products",
           "activity_role": role, "evidence_url": URL, "evidence_quote": quote,
           "product_coverage": "VERIFIED"}
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages={URL: quote},
        fetched_final_urls={URL: URL}, first_party_domains={"orbit.example"},
        identity_names={"orbitproducts"}, requested_product_service=CRITERION,
        identity_anchor={"submitted_name": "Orbit Products", "submitted_domain": "orbit.example",
                         "observed_domain": "orbit.example", "verified_domain": "orbit.example"},
    )["industry"]


def test_explicit_parent_bound_operating_brand_product_passes_existing_source_guards():
    quote = ("Orbit Products supplies building surfaces through its operating brand Terra Tile. "
             "Terra Tile manufactures interior and exterior surfaces.")
    result = validate(quote, "supplier_operator")
    assert result["status"] == "VERIFIED"
    assert coverage_matches(result["product_coverage"], CRITERION)


def test_investment_portfolio_activity_cannot_replace_supplier_role():
    quote = ("Orbit Products is a financial investor with a minority investment in Terra Tile. "
             "Terra Tile manufactures interior and exterior surfaces.")
    # The semantic response identifies an investee's operation as third-party.
    # The host enforces that role; it does not parse ownership prose itself.
    result = validate(quote, "third_party")
    assert result["status"] == "UNPROVEN"
    assert result["activity_role"] == "unresolved"


def test_brand_product_quote_needs_parent_binding_when_homepage_identity_is_incomplete():
    result = validate("Terra Tile manufactures interior and exterior surfaces.", "supplier_operator")
    assert result["status"] == "UNPROVEN"
    assert not coverage_matches(result.get("product_coverage"), CRITERION)


def test_industry_instructions_distinguish_operating_brand_from_investment_and_stage():
    prompt = " ".join(inv._SYSTEM_PROMPT.split())
    assert "financial-investor portfolio" in prompt
    assert "Bound operating brands/divisions can establish its products" in prompt
    assert "quote their explicit parent binding" in prompt
    assert "This does not transfer identity or stage" in prompt
    assert len(inv._SYSTEM_PROMPT) <= 32000
