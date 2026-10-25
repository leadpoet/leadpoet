"""A valid recorded negative packet must not suppress source-fault recovery.

This replays the exact terminal classification block and caller predicate, not
an unrecorded model continuation or a full timed scorer invocation.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from qualification.scoring import company_evidence_investigator as i
from qualification.scoring import lead_scorer as scorer

FIXTURE = json.loads((Path(__file__).parent / "fixtures" /
    "cognizant_scoped_negative_typed415_20261025.json").read_text())


def validated_packet():
    d = FIXTURE["document"]
    locator, prior, home = (d[k] for k in (
        "company_locator", "prior_observations", "verified_homepage_identity"))
    domain = i._registrable_domain
    names = {re.sub("[^a-z0-9]+", "", i._normalized_span(value)) for value in (
        locator.get("name"), prior.get("observed_company_name"),
        home.get("normalized_name"), i.verified_homepage_brand_name(home)) if value}
    domains = {domain(value) for value in (locator.get("website"),
        prior.get("observed_company_website"), home.get("registrable_dns_domain"))
        if domain(value)}
    anchor = {
        "submitted_name": locator.get("name"),
        "submitted_domain": domain(locator.get("website")),
        "submitted_linkedin_slug": i.linkedin_company_page_slug(locator.get("linkedin")),
        "observed_name": prior.get("observed_company_name"),
        "observed_domain": domain(prior.get("observed_company_website")),
        "verified_domain": domain(home.get("registrable_dns_domain")),
        "verified_name": home.get("normalized_name"),
        "verified_linkedin_slug": home.get("linkedin_company_slug", ""),
        "verified_structured_alias_slug": "",
        "observed_linkedin_slug": i.linkedin_company_page_slug(prior.get("observed_company_linkedin")),
    }
    claims = i._validated_findings(
        {"findings": FIXTURE["findings"]}, targets=tuple(d["requested_targets"]),
        fetched_pages=FIXTURE["pages"], fetched_final_urls=FIXTURE["final_urls"],
        first_party_domains=domains, identity_names=names, identity_anchor=anchor,
        requested_stage=d["requested_stage"],
        requested_product_service=d["requested_product_service"])
    assert set(claims) == set(d["requested_targets"])
    assert all(i._submitted_finding_rejection(f, claims[f["target"]]) is None
               for f in FIXTURE["findings"])
    assert {k: v["status"] for k, v in claims.items()} == {
        "headcount": "VERIFIED", "geography": "UNPROVEN", "industry": "UNPROVEN",
        "stage": "VERIFIED", "required_attribute": "VERIFIED"}
    return claims


def terminal_classification(claims, outcomes):
    # The active finality code is nested in the async loop. Execute precisely
    # its pure statements, with the recorded accepted packet and outcomes.
    tree = ast.parse(Path(i.__file__).read_text())
    starts = [n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name) and n.target.id == "latest_source_faults"]
    assert len(starts) == 1
    body = next(value for node in ast.walk(tree) for _, value in ast.iter_fields(node)
                if isinstance(value, list) and starts[0] in value)
    start = body.index(starts[0])
    end = next(n for n in range(start, len(body)) if isinstance(body[n], ast.If)
               and isinstance(body[n].test, ast.Name) and body[n].test.id == "failure_reason")
    namespace = dict(vars(i))
    namespace.update(claims=deepcopy(claims), fetch_outcomes=deepcopy(outcomes),
        required_stage_incomplete=False, required_stage_discovery_failure_reason="",
        rejected=[], required_industry_research_denied=False)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=body[start:end], type_ignores=[])), i.__file__, "exec"), namespace)
    failure = namespace["failure_reason"]
    receipt = {"gate": "company_evidence_investigation",
        "targets": FIXTURE["document"]["requested_targets"],
        "completed_submitted_findings": not failure, "failure_reason": failure,
        "claims": claims, "usage": {"fetch_outcomes": outcomes}}
    return failure, scorer._has_explicitly_unproven_fit_dimensions(
        {}, ("industry", "geography"), investigation_receipt=receipt)


def test_recorded_scoped_negative_with_typed415_keeps_outer_retry_available():
    failure, terminal = terminal_classification(validated_packet(), FIXTURE["fetch_outcomes"])
    assert failure == "provider_error"
    assert terminal is False


@pytest.mark.parametrize("error,expected_failure,expected_terminal", [
    ("unsupported_source", "provider_error", False),
    ("http_503", "provider_error", False),
    ("source_not_found", "", True),
])
def test_recorded_negative_retains_fault_and_ordinary_completion_controls(
        error, expected_failure, expected_terminal):
    outcomes = deepcopy(FIXTURE["fetch_outcomes"])
    outcomes[-1]["error_class"] = error
    failure, terminal = terminal_classification(validated_packet(), outcomes)
    assert (failure, terminal) == (expected_failure, expected_terminal)
