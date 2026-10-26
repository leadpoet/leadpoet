"""Corporation uses the existing exact Organization identity proof gates."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gateway.qualification.models import CompanyOutput
from qualification.scoring import company_verification as verifier
from qualification.scoring import lead_scorer as scorer
from tests.test_company_verification_transport import _Response, _Session


CAPTURE = json.loads((Path(__file__).parent / "fixtures" /
    "ibm_20261025_corporation_identity.json").read_text())
PROFILE = "https://www.linkedin.com/company/example-company"
NODE = {"@type": "Corporation", "name": "NVS",
        "legalName": "North Valley Software Incorporated",
        "url": "https://example.com/", "sameAs": [PROFILE]}


def html(title, document):
    return (f'<html><head><title>{title}</title>'
            f'<script type="application/ld+json">{json.dumps(document)}</script>'
            '</head><body></body></html>').encode()


def handoff(monkeypatch, title, document, *, name=NODE["name"],
            domain="example.com", profile=PROFILE,
            legal_name=NODE["legalName"]):
    """Exercise fetch, homepage receipt, alias projection and web identity."""
    payload = html(title, document)
    calls = []

    class Session(_Session):
        def get(self, url, **kwargs):
            calls.append(url)
            return super().get(url, **kwargs)

    monkeypatch.setattr(verifier.aiohttp, "ClientSession", lambda **kwargs:
        Session(_Response(200, payload, f"https://{domain}/")))
    homepage = asyncio.run(verifier.verify_company_exists(name,
        f"https://{domain}/", company_linkedin=profile, company_quality=True))
    anchor = scorer._verified_homepage_identity_anchor(homepage)
    company = CompanyOutput.model_construct(company_name=name,
        company_website=f"https://{domain}/", company_linkedin=profile)
    receipt = scorer._web_identity_receipt(company, {
        "observed_company_name": legal_name,
        "observed_company_website": f"https://{domain}/",
        "observed_company_linkedin": profile,
    }, verified_homepage_identity=anchor,
       verified_homepage_transport_domain=domain,
       verified_rebrand_identity={"status": "UNPROVEN"}, company_quality=True)
    return homepage, anchor, receipt, calls


def test_current_public_corporation_metadata_binds_exact_legal_name(monkeypatch):
    node = CAPTURE["json_ld"]["@graph"][0]
    homepage, anchor, receipt, calls = handoff(monkeypatch,
        CAPTURE["title"], CAPTURE["json_ld"], name=node["name"],
        domain="ibm.com", profile=node["sameAs"][0], legal_name=node["legalName"])
    assert homepage.decision == "match"
    assert anchor["verified_legal_name_aliases"] == [node["legalName"]]
    assert receipt["decision"] == "match"
    assert receipt["reason_code"] == "verified_legal_name_initialism"
    assert calls == ["https://ibm.com/"]


@pytest.mark.parametrize("kind", ["Organization", "Corporation",
    ["Corporation"], ["Thing", "Corporation"]])
@pytest.mark.parametrize("container", ["root", "graph", "list"])
def test_exact_types_keep_existing_root_graph_and_array_support(monkeypatch,
        kind, container):
    node = {**NODE, "@type": kind}
    document = node if container == "root" else (
        {"@graph": [node]} if container == "graph" else [node])
    homepage, anchor, receipt, calls = handoff(monkeypatch, NODE["name"], document)
    assert homepage.decision == receipt["decision"] == "match"
    assert anchor["verified_legal_name_aliases"] == [NODE["legalName"]]
    assert calls == ["https://example.com/"]


@pytest.mark.parametrize("mutation", ["wrong-domain", "wrong-profile",
    "wrong-legal-name", "missing-legal-name", "multiple-profiles",
    "conflicting-legal-names", "nested-publisher", "graph-overflow"])
def test_corporation_does_not_bypass_existing_legal_identity_gates(monkeypatch,
        mutation):
    node = deepcopy(NODE)
    document = node
    if mutation == "wrong-domain": node["url"] = "https://wrong.example/"
    elif mutation == "wrong-profile": node["sameAs"] = [PROFILE + "-other"]
    elif mutation == "wrong-legal-name": node["legalName"] = "Other Company Incorporated"
    elif mutation == "missing-legal-name": node.pop("legalName")
    elif mutation == "multiple-profiles": node["sameAs"].append(PROFILE + "-other")
    elif mutation == "conflicting-legal-names":
        document = {"@graph": [node, {**node, "legalName": "Other Company Incorporated"}]}
    elif mutation == "nested-publisher":
        document = {"@type": "WebPage", "publisher": node}
    elif mutation == "graph-overflow": document = {"@graph": [node] * 21}
    _, _, receipt, _ = handoff(monkeypatch, NODE["name"], document)
    assert receipt["decision"] != "match"


@pytest.mark.parametrize("kind", ["Product", "Person", "LocalBusiness",
    "CorporationLike", ["Thing", "Product"]])
def test_unapproved_types_do_not_supply_company_metadata(kind):
    document = json.dumps({**NODE, "@type": kind})
    assert verifier._organization_identity_records(document) == []
    assert verifier._organization_same_as_urls(document) == []
