"""Expose loaded citation context without changing geography proof authority."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from qualification.scoring import company_evidence_investigator as i
from tests.test_company_evidence_investigator import _finding

NATIVE = json.loads((Path(__file__).parent / "fixtures" /
    "airwallex_geography_correction_context_20261025.json").read_text())
URL = NATIVE["source_url"]
PROFILE = NATIVE["profile_url"]
QUOTE = NATIVE["original_quote"]
ERROR = "headquarters evidence must use independently bound first-party requested and fetched final URLs"
KEY = "untrusted_alternative_first_party_source_context"
PAGES = {URL: NATIVE["source_text"], PROFILE: NATIVE["profile_text"]}
FINALS = {URL: URL, PROFILE: PROFILE}
IDENTITY = {"normalized_name": "airwallex", "registrable_dns_domain": "airwallex.com",
            "linkedin_company_slug": "airwallex"}


def geography(*, url=PROFILE, quote=QUOTE, status="VERIFIED"):
    return _finding("geography", status=status, observed_value="Singapore",
        observed_country="Singapore", observed_state="", evidence_url=url,
        evidence_quote=quote, reason="Model-authored headquarters judgment")


def hint(*, pages=PAGES, finals=FINALS, finding=None, reason=ERROR, names={"airwallex"}):
    return i._untrusted_geography_citation_context(
        [{"target": "geography", "reason": reason}],
        submitted_findings={"geography": finding or geography()},
        fetched_pages=pages, fetched_final_urls=finals,
        first_party_domains={"airwallex.com"}, identity_names=names)


def validate(finding, *, pages=PAGES, finals=FINALS):
    return i._validated_findings({"findings": [finding]}, targets=("geography",),
        fetched_pages=pages, fetched_final_urls=finals,
        first_party_domains={"airwallex.com"}, identity_names={"airwallex"},
        identity_anchor={"submitted_name": "Airwallex", "submitted_domain": "airwallex.com",
                         "observed_name": "Airwallex", "observed_domain": "airwallex.com"})["geography"]


def test_recorded_source_offers_one_neutral_contiguous_context():
    contexts = hint(pages={**PAGES, URL + "?another=1": NATIVE["source_text"]},
                    finals={**FINALS, URL + "?another=1": URL + "?another=1"})
    assert len(contexts) == 1
    context = contexts[0]
    assert context["target"] == "geography" and context["source_url"] == URL
    assert len(context["untrusted_source_context"]) <= 2000
    assert i._quote_occurs(QUOTE, context["untrusted_source_context"])
    assert i._quote_names_company(context["untrusted_source_context"], {"airwallex"})
    assert "Context only, not proof" in context["instruction"]
    assert not ({"status", "evidence_quote", "untrusted_suggested_quote"} & context.keys())


@pytest.mark.parametrize("case", [
    "wrong_final_host", "missing_final", "unloaded", "invented_quote",
    "unrelated_error", "other_company", "foreign_request",
    "private_url", "http_url", "empty_identity",
])
def test_source_selection_fails_closed(case):
    options = {"pages": {URL: NATIVE["source_text"]}, "finals": {URL: URL}}
    if case == "wrong_final_host": options["finals"] = {URL: "https://unrelated.example/page"}
    if case == "missing_final": options["finals"] = {}
    if case == "unloaded": options["pages"] = {}
    if case == "invented_quote": options["finding"] = geography(quote=QUOTE.replace("Singapore", "Sweden"))
    if case == "unrelated_error": options["reason"] = "submitted quote was not present in fetched source"
    if case == "other_company": options["pages"] = {URL: NATIVE["source_text"].replace("Airwallex", "Other Supplier").replace("airwallex", "other-supplier")}
    if case in {"foreign_request", "private_url", "http_url"}:
        bad = {"foreign_request": "https://unrelated.example/page",
               "private_url": "https://127.0.0.1/page",
               "http_url": "http://www.airwallex.com/page"}[case]
        options.update(pages={bad: NATIVE["source_text"]}, finals={bad: bad})
    if case == "empty_identity": options["names"] = set()
    assert hint(**options) == []


def test_context_limit_keeps_the_exact_span_or_omits_the_hint():
    # Do not clip a long submitted span into a misleading partial quotation.
    long_quote = "Airwallex " + "word " * 350 + "headquarters in Singapore."
    text = "prefix " * 200 + long_quote + " trailing"
    contexts = hint(pages={URL: text}, finding=geography(quote=long_quote))
    assert not i._quote_occurs(long_quote, i._source_context_for_quote(long_quote, text)[:2000])
    assert contexts == []


@pytest.mark.parametrize("case", ["cross_host_final", "fabricated", "unloaded_url", "nonfirstparty"])
def test_context_does_not_weaken_existing_proof_gates(case):
    finding = geography(url=URL, quote=NATIVE["named_quote"])
    options = {}
    if case == "cross_host_final": options["finals"] = {**FINALS, URL: "https://unrelated.example/page"}
    if case == "fabricated": finding["evidence_quote"] = NATIVE["named_quote"].replace("Singapore", "Sweden")
    if case == "unloaded_url": finding["evidence_url"] = "https://www.airwallex.com/unloaded"
    if case == "nonfirstparty":
        finding["evidence_url"] = "https://unrelated.example/page"
        options = {"pages": {"https://unrelated.example/page": NATIVE["source_text"]},
                   "finals": {"https://unrelated.example/page": "https://unrelated.example/page"}}
    assert validate(finding, **options)["status"] == "UNPROVEN"


def test_original_quote_remains_rejected_and_named_span_passes():
    assert validate(geography(url=URL))["reason"] == "source quote did not identify the investigated company"
    assert validate(geography(url=URL, quote=NATIVE["named_quote"]))["status"] == "VERIFIED"


@pytest.mark.parametrize("final", ["named", "repeat_unnamed", "unproven"])
def test_final_scoped_correction_receives_context_without_extra_research(monkeypatch, final):
    requests, research, corrections = [], [], []
    attribute_quote = "Founded in Melbourne in 2015, Airwallex holds 85+ licenses across North America, Europe, the Middle East, and Asia-Pacific, forming one of the most comprehensive financial infrastructures in the world."
    attribute = _finding("required_attribute", status="VERIFIED", activity_role="supplier_operator",
        evidence_url=URL, evidence_quote=attribute_quote, reason="Loaded licensing fact")

    async def post(_session, _url, *, headers, payload):
        requests.append(deepcopy(payload))
        turn = len(requests)
        if turn == 1:
            name, args = "fetch_page", {"url": URL}
        elif turn == 2:
            name, args = "submit_findings", {"findings": [geography(), attribute]}
        else:
            assert turn == 3
            assert [t["function"]["name"] for t in payload["tools"]] == ["submit_findings"]
            schema = payload["tools"][0]["function"]["parameters"]
            assert schema["properties"]["findings"]["items"]["properties"]["target"]["enum"] == ["geography"]
            feedback = json.loads(payload["messages"][-1]["content"])
            assert feedback["rejected_findings"][0]["source_url"] == PROFILE
            assert feedback["rejected_findings"][0]["reason"] == ERROR
            corrections.append(feedback)
            if final == "unproven":
                finding = geography(url="", quote="", status="UNPROVEN")
            else:
                finding = geography(url=URL, quote=NATIVE["named_quote"] if final == "named" else QUOTE)
            name, args = "submit_findings", {"findings": [finding]}
        return 200, {"choices": [{"message": {"tool_calls": [{"id": f"call-{turn}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}}]}

    async def deny(*_args, **_kwargs):
        research.append("unexpected research")
        raise AssertionError("only already-loaded pages may be used")

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-placeholder")
    monkeypatch.setenv("EXA_API_KEY", "offline-placeholder")
    monkeypatch.setattr(i, "MAX_REASONING_TURNS", 2)
    monkeypatch.setattr(i, "_post_json", post)
    monkeypatch.setattr(i, "_search_web", deny)
    monkeypatch.setattr(i, "_fetch_page", deny)
    result = asyncio.run(i.investigate_company_evidence(
        company_locator=NATIVE["company_locator"], targets=("geography", "required_attribute"),
        requested_geography="Singapore", requested_attribute="Holds financial-services licenses",
        positive_semantic_review=True, verified_homepage_identity=IDENTITY,
        prior_observations={"submitted_source_urls": list(PAGES),
                            "observed_company_name": "Airwallex",
                            "observed_company_website": "https://www.airwallex.com",
                            "observed_company_linkedin": PROFILE},
        prefetched_pages={u: {"final_url": FINALS[u], "text": t} for u, t in PAGES.items()}))
    assert len(requests) == 3 and research == []
    assert len(corrections) == 1
    contexts = corrections[0].get(KEY, [])
    assert len(contexts) == 1 and contexts[0]["source_url"] == URL
    assert i._quote_occurs(QUOTE, contexts[0]["untrusted_source_context"])
    assert result["claims"]["required_attribute"]["status"] == "VERIFIED"
    assert result["claims"]["required_attribute"]["evidence_quote"] == attribute_quote
    assert KEY not in json.dumps(result["claims"])
    if final == "repeat_unnamed":
        assert result["_completed_submit"] is False and result["failure_reason"] == "malformed_response"
    else:
        assert result["_completed_submit"] is True
        assert result["claims"]["geography"]["status"] == ("VERIFIED" if final == "named" else "UNPROVEN")


def test_optional_context_drops_whole_at_existing_message_bound(monkeypatch):
    context = hint()[0]
    document = {"ok": False, "rejected_findings": [{"target": "geography", "reason": ERROR}],
                KEY: [context]}
    baseline = {k: v for k, v in document.items() if k != KEY}
    monkeypatch.setattr(i, "OPENROUTER_MAX_CONTENT_CHARS", len(i._bounded_message_json(baseline)) + 20)
    bounded = json.loads(i._bounded_message_json(document))
    assert KEY not in bounded and bounded["rejected_findings"] == document["rejected_findings"]


def test_new_optional_context_does_not_displace_existing_citation_context(monkeypatch):
    baseline = {"ok": False, "rejected_findings": [{"target": "geography", "reason": ERROR}],
                "untrusted_non_rejected_source_context": [{"target": "stage", "source_context": "existing evidence"}]}
    limit = len(i._bounded_message_json(baseline)) + 10
    monkeypatch.setattr(i, "OPENROUTER_MAX_CONTENT_CHARS", limit)
    bounded = json.loads(i._bounded_message_json({**baseline, KEY: hint()}))
    assert bounded == baseline


@pytest.mark.parametrize("which", ["requested", "final"])
def test_both_bound_urls_must_pass_public_url_validation(monkeypatch, which):
    final_url = URL + "?fetched=1"
    reject = URL if which == "requested" else final_url
    original = i.public_http_url
    def public(value):
        if value == reject:
            raise ValueError("not public")
        return original(value)
    monkeypatch.setattr(i, "public_http_url", public)
    assert hint(pages={URL: NATIVE["source_text"]}, finals={URL: final_url}) == []
