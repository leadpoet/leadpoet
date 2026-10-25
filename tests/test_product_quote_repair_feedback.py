"""Untrusted contiguous repair data, never automatic semantic admission."""
import asyncio
import json

import pytest

from qualification.scoring import company_evidence_investigator as inv
from qualification.scoring import product_coverage as pc
from tests.product_coverage_fixtures import coverage, trusted_plan

CRITERION = "Manufactures interior and exterior surfaces"
URL = "https://acmeproducts.example/products"
QUOTE = "The Company manufactures interior and exterior surfaces."
PAGE = 'Acme Products (“Acme”, or “the Company”) owns the operating brand Verda. ' + QUOTE
BRAND_URL = "https://acmeproducts.example/brands"
HOME_URL = "https://acmeproducts.example/"
BRAND_QUOTE = (
    "Verda® is a leading brand of manufactured stone tile for interior and "
    "exterior floors, walls, and countertops."
)
HOME_QUOTE = BRAND_QUOTE.replace("®", "")
HOME_PAGE = (
    "Acme Products supplies building surfaces through its operating brand Verda. "
    + HOME_QUOTE
)
NODES = [
    {"id": "p0", "span": CRITERION, "op": "all", "children": ["p1", "p2", "p3"]},
    {"id": "p1", "span": "Manufactures surfaces", "op": "leaf", "children": []},
    {"id": "p2", "span": "Interior surfaces", "op": "leaf", "children": []},
    {"id": "p3", "span": "Exterior surfaces", "op": "leaf", "children": []},
]


def raw_finding(quote=QUOTE, *, status="UNPROVEN", role="supplier_operator", url=URL):
    wire = json.loads(coverage(CRITERION, url, quote, nodes=NODES))
    for claim in wire["claims"]:
        claim["activity_role"] = role
    return {"target": "industry", "status": status, "observed_industry": "Surface manufacturing" if status == "VERIFIED" else "",
            "activity_role": role if status == "VERIFIED" else "unresolved",
            "evidence_url": url if status == "VERIFIED" else "", "evidence_quote": quote if status == "VERIFIED" else "",
            "product_coverage": json.dumps(wire), "reason": "Rejudge the complete criterion from loaded sources."}


def validated(raw, page=PAGE, *, url=URL, final_url=None):
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages={url: page},
        fetched_final_urls={url: final_url or url}, first_party_domains={"acmeproducts.example"},
        identity_names={"acmeproducts"}, requested_product_service=CRITERION,
        product_plan=pc.get_cached_plan(CRITERION, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL),
        identity_anchor={"submitted_name": "Acme Products", "submitted_domain": "acmeproducts.example",
                         "observed_name": "Acme Products", "observed_domain": "acmeproducts.example"},
    )["industry"]


def cross_page_validated(raw, *, home_page=HOME_PAGE, home_url=HOME_URL,
                         home_final_url=HOME_URL):
    pages = {BRAND_URL: "Our brands. " + BRAND_QUOTE, home_url: home_page}
    return inv._validated_findings(
        {"findings": [raw]}, targets=("industry",), fetched_pages=pages,
        fetched_final_urls={BRAND_URL: BRAND_URL, home_url: home_final_url},
        first_party_domains={"acmeproducts.example"}, identity_names={"acmeproducts"},
        requested_product_service=CRITERION,
        product_plan=pc.get_cached_plan(CRITERION, model=inv.POSITIVE_SEMANTIC_REVIEW_MODEL),
        identity_anchor={"submitted_name": "Acme Products", "submitted_domain": "acmeproducts.example",
                         "observed_name": "Acme Products", "observed_domain": "acmeproducts.example"},
    )["industry"]


def test_trademark_variant_finds_exact_contiguous_parent_brand_passage_on_loaded_page():
    assert not inv._quote_occurs(BRAND_QUOTE, HOME_PAGE)
    first = cross_page_validated(raw_finding(BRAND_QUOTE, url=BRAND_URL))
    assert first["status"] == "UNPROVEN"
    assert all(claim["status"] == "UNPROVEN" for claim in first["product_coverage"]["claims"])
    repairs = first["untrusted_product_citation_repairs"]
    assert len(repairs) == 3
    assert all(repair["source_url"] == HOME_URL for repair in repairs)
    candidate = repairs[0]["untrusted_suggested_quote"]
    assert HOME_QUOTE in repairs[0]["untrusted_source_context"]
    assert candidate in HOME_PAGE and "Acme Products" in candidate
    assert HOME_QUOTE in candidate and BRAND_QUOTE not in candidate
    assert len(candidate) <= 2000 and "not proof" in repairs[0]["instruction"]
    assert candidate not in json.dumps(first["product_coverage"])

    # The hint has no authority; only a fresh model submission can pass.
    repaired = cross_page_validated(raw_finding(candidate, status="VERIFIED", url=HOME_URL))
    assert repaired["status"] == "VERIFIED"
    assert pc.coverage_matches(repaired["product_coverage"], CRITERION)


@pytest.mark.parametrize("home_page,home_url,home_final_url", [
    ("Acme Products owns Verda. Verda® makes tile.", HOME_URL, HOME_URL),
    ("Other Products owns Verda. " + HOME_QUOTE, HOME_URL, HOME_URL),
    ("Acme Products owns Verda. " + HOME_QUOTE.replace("exterior", "outdoor"), HOME_URL, HOME_URL),
    (HOME_PAGE, "https://unbound.example/", "https://unbound.example/"),
    (HOME_PAGE, "http://acmeproducts.example/", "http://acmeproducts.example/"),
    (HOME_PAGE, HOME_URL, "https://unbound.example/"),
])
def test_cross_page_repair_rejects_short_overlap_wrong_entity_or_unsafe_source(
    home_page, home_url, home_final_url,
):
    first = cross_page_validated(
        raw_finding(BRAND_QUOTE, url=BRAND_URL),
        home_page=home_page, home_url=home_url, home_final_url=home_final_url,
    )
    assert first["status"] == "UNPROVEN"
    assert first["untrusted_product_citation_repairs"] == []


def test_short_common_trademark_quote_does_not_trigger_cross_page_hint():
    quote = "Verda® makes tile."
    first = cross_page_validated(
        raw_finding(quote, url=BRAND_URL),
        home_page="Acme Products owns Verda. Verda makes tile.",
    )
    assert first["status"] == "UNPROVEN"
    assert first["untrusted_product_citation_repairs"] == []


def test_candidate_is_exact_bounded_same_page_and_not_admitted_into_proof():
    result = validated(raw_finding())
    assert result["status"] == "UNPROVEN"
    assert all(c["evidence"] == [] for c in result["product_coverage"]["claims"])
    repairs = result["untrusted_product_citation_repairs"]
    assert len(repairs) == 3
    for repair in repairs:
        candidate = repair["untrusted_suggested_quote"]
        assert candidate in PAGE and QUOTE in candidate and "Acme Products" in candidate
        assert len(candidate) <= 2000 and "not proof" in repair["instruction"]
        assert candidate not in json.dumps(result["product_coverage"])
    candidate = repairs[0]["untrusted_suggested_quote"]
    # Only a new model submission plus all normal validators can accept it.
    assert validated(raw_finding(candidate, status="VERIFIED"))["status"] == "VERIFIED"


def test_root_quote_rejection_also_requires_bound_resubmission():
    first = validated(raw_finding(status="VERIFIED"))
    assert first["status"] == "UNPROVEN"
    assert first["reason"] == "source quote did not identify the investigated company"
    candidate = first["untrusted_product_citation_repairs"][0]["untrusted_suggested_quote"]
    assert validated(raw_finding(candidate, status="VERIFIED"))["status"] == "VERIFIED"


@pytest.mark.parametrize("role", ["customer_user", "third_party"])
def test_customer_or_parent_noise_is_not_automatic_supplier_proof(role):
    page = "Acme Products discusses a customer and its external vendor. " + QUOTE
    first = validated(raw_finding(), page)
    assert first["status"] == "UNPROVEN"
    candidate = first["untrusted_product_citation_repairs"][0]["untrusted_suggested_quote"]
    assert candidate in page  # Context can include unrelated activity.
    negative = validated(raw_finding(candidate, status="VERIFIED", role=role), page)
    assert negative["status"] == "UNPROVEN"
    assert not pc.coverage_matches(negative["product_coverage"], CRITERION)


@pytest.mark.parametrize("url,final_url", [
    ("https://unbound.example/products", "https://unbound.example/products"),
    (URL, "https://unbound.example/products"),
])
def test_unbound_or_redirected_source_has_no_expansion(url, final_url):
    raw = raw_finding(url=url)
    result = validated(raw, url=url, final_url=final_url)
    assert result["status"] == "UNPROVEN"
    assert not result["untrusted_product_citation_repairs"]


def test_expansion_does_not_cross_the_quote_bound_or_fabricate_a_name():
    assert inv._untrusted_contiguous_quote_candidate(QUOTE, "OtherCo. " + QUOTE, {"acmeproducts"}) == ""
    assert inv._untrusted_contiguous_quote_candidate(QUOTE, "Acme. Products are unrelated. " + QUOTE, {"acmeproducts"}) == ""
    assert inv._untrusted_contiguous_quote_candidate(QUOTE, "Acme Products " + "z" * 2001 + QUOTE, {"acmeproducts"}) == ""
    assert inv._untrusted_contiguous_quote_candidate("Invented quote", PAGE, {"acmeproducts"}) == ""
    page = PAGE.replace("manufactures interior", "manufactures\n interior")
    candidate = inv._untrusted_contiguous_quote_candidate(QUOTE, page, {"acmeproducts"})
    assert candidate in page and inv._quote_occurs(QUOTE, candidate)


def test_optional_repair_contexts_are_dropped_whole_to_fit_existing_message_bound():
    repairs = [{"claim_id": f"p{i}", "untrusted_suggested_quote": "q" * 2000,
                "untrusted_source_context": "s" * 800, "instruction": "context only"}
               for i in range(3)]
    document = {"required_metadata": "m" * 25000,
                "product_coverage": {"plan_id": "frozen", "untrusted_citation_repairs": repairs}}
    encoded = inv._bounded_message_json(document)
    decoded = json.loads(encoded)
    assert len(encoded) < 32000 and decoded["required_metadata"] == document["required_metadata"]
    retained = decoded["product_coverage"]["untrusted_citation_repairs"]
    assert len(retained) < 3
    assert all(r["untrusted_suggested_quote"] == "q" * 2000 for r in retained)


def test_global_unproven_positive_clause_rejections_get_context_and_bounded_rejudgment(monkeypatch):
    requests, searches = [], []
    accept_repair = True
    trusted_plan(CRITERION, NODES)

    async def post(_session, _url, *, headers, payload):
        from lab_arena.operations import validate_operation_request
        validate_operation_request("openrouter.chat", payload)
        requests.append(payload)
        turn = len(requests)
        if turn == 1:
            name, arguments = "search_web", {"query": "Acme Products surfaces"}
        else:
            raw = raw_finding()
            if turn == 3:
                feedback = json.loads(payload["messages"][-1]["content"])
                repairs = feedback["product_coverage"]["untrusted_citation_repairs"]
                assert len(repairs) == 3
                assert "requirements" not in feedback["product_coverage"]
                assert feedback["product_coverage"]["allowed_claim_ids"] == ["p1", "p2", "p3"]
                candidate = repairs[0]["untrusted_suggested_quote"]
                assert candidate in PAGE and "Acme Products" in candidate
                if accept_repair:
                    raw = raw_finding(candidate, status="VERIFIED")
            name, arguments = "submit_findings", {"findings": [raw]}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": str(turn), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}

    async def search(_session, query, **kwargs):
        searches.append(query)
        return {"results": []}

    async def forbidden_fetch(*args, **kwargs):
        pytest.fail("Quote repair must use the already loaded physical page")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    monkeypatch.setenv("EXA_API_KEY", "offline-test")
    monkeypatch.setattr(inv, "MAX_SEARCH_CALLS", 1)
    monkeypatch.setattr(inv, "_post_json", post)
    monkeypatch.setattr(inv, "_search_web", search)
    monkeypatch.setattr(inv, "_fetch_page", forbidden_fetch)
    result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme Products", "website": "https://acmeproducts.example"},
        targets=("industry",), requested_industry="Manufacturing", requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [URL], "observed_company_name": "Acme Products",
                            "observed_company_website": "https://acmeproducts.example"},
        prefetched_pages={URL: {"final_url": URL, "text": PAGE}},
    ))
    assert result["claims"]["industry"]["status"] == "VERIFIED"
    assert "untrusted_product_citation_repairs" not in result["claims"]["industry"]
    assert "product_positive_citation_rejected" not in result["claims"]["industry"]
    assert result["_completed_submit"] is True and len(requests) == 3
    assert result["usage"]["search_calls"] == len(searches) == 1 and result["usage"]["fetch_calls"] == 0

    # If no rejudgment can pass, private repair fields must not leak into the result.
    accept_repair = False
    requests.clear()
    searches.clear()
    rejected_result = asyncio.run(inv.investigate_company_evidence(
        company_locator={"name": "Acme Products", "website": "https://acmeproducts.example"},
        targets=("industry",), requested_industry="Manufacturing", requested_product_service=CRITERION,
        prior_observations={"submitted_source_urls": [URL], "observed_company_name": "Acme Products",
                            "observed_company_website": "https://acmeproducts.example"},
        prefetched_pages={URL: {"final_url": URL, "text": PAGE}},
    ))
    assert "untrusted_product_citation_repairs" not in rejected_result["claims"]["industry"]
    assert "product_positive_citation_rejected" not in rejected_result["claims"]["industry"]
