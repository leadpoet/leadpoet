from copy import deepcopy
import pytest

from lab_arena import collection_research_evidence as ev

SITE = "https://acme.com/"
LI = "https://www.linkedin.com/company/acme/"
PRESS = "https://acme.com/news/launch"
CLAIM = "Acme Cloud introduced its Atlas service on November 21, 2026."


def fetch(url, body):
    return {
        "op": "deepline.execute",
        "request": {
            "tool": "firecrawl_scrape",
            "payload": {"url": url, "formats": ["rawHtml"]},
        },
        "response": {
            "status": "completed",
            "billing": {"cost_usd": 0.01},
            "result": {
                "data": {
                    "rawHtml": body,
                    "metadata": {"sourceURL": url, "url": url, "statusCode": 200},
                }
            },
        },
    }


def case():
    own = fetch(
        SITE,
        '<html><body>Acme Cloud supplies software. <a href="'
        + LI
        + '">Official account</a></body></html>',
    )
    li = fetch(
        LI,
        '<html><body>Acme Cloud. Industry Software. 11-50 employees. United States, Texas. <a href="'
        + SITE
        + '">Website</a></body></html>',
    )
    news = fetch(PRESS, "<html><body><article>" + CLAIM + "</article></body></html>")
    company = {
        "company_name": "Acme Cloud",
        "company_website": SITE,
        "company_linkedin": LI,
        "country": "United States",
        "state": "Texas",
        "industry": "Software",
        "employee_count": "11-50",
        "company_stage": "",
        "intent_details": CLAIM,
        "intent_signals": [
            {
                "date": "2026-11-21",
                "url": PRESS,
                "description": CLAIM,
                "matched_icp_signal": 0,
            }
        ],
    }
    return (
        dict(
            icp={"intent_signals": ["launched a product"], "intent_max_age_days": 30},
            evaluation_date="2026-11-23",
            labels={},
        ),
        company,
        {0: own, 1: li, 3: news},
    )


def test_new_company_grounded_without_pretending_native_verifier_decided():
    context, c, trace = case()
    checked = ev.apply_fresh_evidence(
        {
            "checks": {
                "native_coverage": False,
                "native_qualified": False,
                "distinct_companies": True,
            }
        },
        context,
        [c],
        trace,
    )
    assert checked["checks"] == {"distinct_companies": True, "source_coverage": True}
    assert (
        "same_run_verified_provider_sources" in checked["facts"]
        and "generate-research-3" in checked["facts"]
    )
    assert checked["reference_answer"] == "[]"


@pytest.mark.parametrize(
    "bad",
    [
        "generator_echo",
        "tool_result_echo",
        "redirect",
        "provider_error",
        "expired",
        "hidden",
        "wrong_subject",
        "wrong_domain",
        "no_link",
        "wrong_date",
        "false_detail",
        "changed_quote",
        "injected_source",
        "duplicate",
    ],
)
def test_unverified_fresh_evidence_never_substitutes_for_native_grade(bad):
    ctx, c, trace = case()
    if bad == "generator_echo":
        trace[3]["op"] = "openrouter.responses"
    if bad == "tool_result_echo":
        trace[3]["request"]["tool"] = "generic_http_request"
    if bad == "redirect":
        trace[3]["response"]["result"]["data"]["metadata"][
            "url"
        ] = "https://attacker.com/"
    if bad == "provider_error":
        trace[3]["response"]["result"]["data"]["metadata"]["statusCode"] = 404
    if bad == "expired":
        ctx["evaluation_date"] = "2027-03-03"
    if bad == "hidden":
        trace[3]["response"]["result"]["data"][
            "rawHtml"
        ] = f"<html><style>{CLAIM}</style></html>"
    if bad == "wrong_subject":
        c["company_name"] = "Other Corp."
    if bad == "wrong_domain":
        c["intent_signals"][0]["url"] = "https://acme.com.example.org/news/launch"
    if bad == "no_link":
        trace[1]["response"]["result"]["data"][
            "rawHtml"
        ] = "<html>Acme Cloud 11-50 Software United States Texas</html>"
    if bad == "wrong_date":
        c["intent_signals"][0]["date"] = "2026-11-22"
    if bad == "false_detail":
        c["intent_details"] += " Sales grew by 85%."
    if bad == "changed_quote":
        c["required_attribute"] = {
            "text": "growth",
            "passed": True,
            "evidence_url": PRESS,
            "evidence_quote": "Sales grew by 85%.",
        }
    if bad == "injected_source":
        trace[3]["request"]["payload"]["actions"] = [
            {"type": "executeJavascript", "script": "change page"}
        ]
    if bad == "duplicate":
        x = deepcopy(trace[3])
        x["response"]["result"]["data"]["rawHtml"] += " inconsistent"
        trace[4] = x
    assert ev.evidence_for(c, ctx, ev.verified_pages(trace)) is None
