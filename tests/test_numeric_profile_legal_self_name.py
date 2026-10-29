"""Exact numeric profile lookup can preserve its own leading legal identity."""

import ast
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from qualification.scoring import lead_scorer, linkedin_company_size as linkedin
from qualification.scoring.company_evidence_investigator import _validated_findings


NUMERIC = "https://www.linkedin.com/company/2305"
VANITY = "https://www.linkedin.com/company/insight"
SOURCE = "https://www.insight.com/en_US/what-we-do/services/managed-services.html"
QUOTE = (
    "Bring newfound flexibility and agility to your organization with Insight’s Managed IT Services. "
    "Clients working with Insight are reducing operational burden, getting better visibility and feedback, "
    "implementing sustainable governance models, addressing skill gaps, and driving strategic adoption of new solutions."
)
HOME = {"normalized_name": "insightenterprises", "registrable_dns_domain": "insight.com",
        "linkedin_company_slug": "2305"}
ELEMENT = {
    "id": "2305", "universalName": "insight", "name": "Insight", "linkedinUrl": VANITY + "/",
    "website": "https://www.insight.com/en_US/home.html",
    "description": (
        "Insight Enterprises, Inc. is a Fortune 500 solutions integrator helping organizations accelerate "
        "their digital journey to modernize their business and maximize the value of technology. Insight’s "
        "technical expertise spans cloud and edge-based transformation solutions, with global scale and "
        "optimization built on 33+ years of deep partnerships with the world’s leading and emerging technology providers. "
    ),
}
BODY = {"result": {"data": {"status": 200, "element": ELEMENT}}}


def proof(body=BODY, project=linkedin.project_structured_linkedin_company_identity):
    return project("insight.com", NUMERIC, body, observed_profile_url=VANITY,
                   expected_company_name=HOME["normalized_name"])


def web_identity():
    return lead_scorer._web_identity_receipt(
        SimpleNamespace(company_name="Insight Enterprises", company_website="https://insight.com/",
                        company_linkedin=""),
        {"observed_company_name": "Insight Enterprises, Inc.",
         "observed_company_website": "https://www.insight.com",
         "observed_company_linkedin": VANITY},
        verified_homepage_identity=HOME, verified_homepage_transport_domain="insight.com",
    )


def quote_finding(homepage, *, quote=QUOTE, source=SOURCE, loaded=QUOTE):
    web = web_identity()
    anchor = {"submitted_name": web["submitted_name"], "observed_name": web["observed_name"],
              "verified_name": homepage["normalized_name"], "submitted_domain": web["submitted_domain"],
              "observed_domain": web["observed_domain"], "verified_domain": homepage["registrable_dns_domain"],
              "submitted_linkedin_slug": "", "observed_linkedin_slug": "insight",
              "verified_linkedin_slug": homepage["linkedin_company_slug"]}
    finding = {"target": "industry", "status": "VERIFIED", "observed_value": None,
               "observed_industry": "IT Services and IT Consulting", "activity_role": "supplier_operator",
               "evidence_url": source, "evidence_quote": quote}
    return _validated_findings(
        {"findings": [finding]}, targets=("industry",), fetched_pages={SOURCE: loaded},
        fetched_final_urls={SOURCE: SOURCE}, first_party_domains={"insight.com"},
        identity_names={"insightenterprises", "insightenterprisesinc"}, identity_anchor=anchor,
    )["industry"]


def test_exact_captured_numeric_request_and_present_quote_recover_bound_identity():
    original = deepcopy(BODY)
    evidence = proof()
    assert evidence == {
        "name": "Insight Enterprises, Inc.", "provider": "harvestapi_get_company",
        "source_field": "description.leading_legal_name", "url": VANITY,
        "website": "https://insight.com/", "company_id": "2305", "requested_url": NUMERIC,
    }
    assert quote_finding(HOME)["status"] == "UNPROVEN"
    resolved, locator = lead_scorer._investigator_identity_context(
        "", HOME, web_identity(), "insight.com", structured_identity=evidence,
    )
    assert resolved == {**HOME, "linkedin_company_slug": "insight"} and locator == ""
    assert quote_finding(resolved)["status"] == "VERIFIED"
    assert BODY == original


def test_captured_case_fails_before_projection_change():
    # Exercise the exact pre-change function in the same module namespace.
    source = subprocess.check_output([
        "git", "show", "d38fbb16e:qualification/scoring/linkedin_company_size.py",
    ], text=True, cwd=Path(__file__).parents[1])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "project_structured_linkedin_company_identity")
    namespace = dict(vars(linkedin))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "captured-preimage", "exec"), namespace)
    assert proof(project=namespace[node.name]) is None


@pytest.mark.parametrize("field,value", [
    ("id", "9999"), ("id", True), ("id", "２３０５"), ("id", "2305 "), ("id", "23.05"),
    ("name", "Unrelated"), ("name", "Insights"), ("name", "Insight Ω"), ("name", "🦄 Insight"), ("name", "Insight!"), ("universalName", "unrelated"),
    ("universalName", None), ("website", "https://unrelated.example"),
    ("linkedinUrl", "https://www.linkedin.com/company/unrelated"),
    ("description", "Customers include Insight Enterprises, Inc. and other companies."),
    ("description", "Other Company is a provider serving Insight Enterprises, Inc."),
    ("description", "Insight Enterprises, Inc. is our customer."),
    ("description", "Insight Enterprises, Inc. is a customer of this company."),
    ("description", "Insight Enterprises, Inc. is a partner of this company."),
    ("description", "Insight Enterprises, Inc. is a client of this company."),
    ("description", "Insight Enterprises, Inc. is a competitor of this company."),
    ("description", "Insight Enterprises, Inc. is not the company described here."),
    ("description", "Insight Enterprises, Inc. is quoted as a technology provider."),
    ("description", '"Insight Enterprises, Inc. is a solutions integrator."'),
    ("description", "Insight\x00 Enterprises, Inc. is a solutions integrator."),
    ("description", "Insight\u200b Enterprises, Inc. is a solutions integrator."),
    ("description", "Insight Enterprises,\nInc. is a solutions integrator."),
    ("description", None),
])
def test_unbound_or_mention_only_main_profile_does_not_prove_alias(field, value):
    body = deepcopy(BODY)
    body["result"]["data"]["element"][field] = value
    assert proof(body) is None


def test_nested_other_company_and_ambiguous_main_profiles_do_not_prove_alias():
    unrelated = {"id": "9999", "name": "Unrelated", "website": "https://unrelated.example",
                 "linkedinUrl": "https://www.linkedin.com/company/unrelated", "similarOrganizations": [ELEMENT]}
    assert proof({"result": {"data": {"element": unrelated}}}) is None
    assert proof({"status": 200, "elements": [ELEMENT, dict(ELEMENT)]}) is None
    main = {**ELEMENT, "description": "", "similarOrganizations": [ELEMENT]}
    assert proof({"status": 200, "element": main}) is None


@pytest.mark.parametrize("fault", ["missing_quote", "unrelated_source", "wrong_company"])
def test_numeric_alias_does_not_change_quote_or_company_guards(fault):
    resolved, _ = lead_scorer._investigator_identity_context(
        "", HOME, web_identity(), "insight.com", structured_identity=proof(),
    )
    if fault == "missing_quote":
        # The second saved attempt's short quote was absent from its loaded page.
        options = {"quote": "As the leading Solutions Integrator, Insight solves your unique challenges."}
    elif fault == "unrelated_source":
        options = {"source": "https://unrelated.example/products"}
    else:
        resolved = {**resolved, "normalized_name": "Unrelated"}
        options = {}
    assert quote_finding(resolved, **options)["status"] == "UNPROVEN"


def test_ordinary_same_slug_identity_keeps_its_original_five_keys():
    evidence = linkedin.project_structured_linkedin_company_identity("insight.com", VANITY, BODY)
    assert set(evidence) == {"name", "provider", "source_field", "url", "website"}
    assert evidence["source_field"] == "name" and evidence["name"] == "Insight"


def _seam_preimages():
    from scripts.arena_reviewer_structured_homepage_alias_seam import SELF_NAME_CHANGES
    root = Path(__file__).parents[1]
    return {name: subprocess.check_output(["git", "show", "d38fbb16e:" + name], cwd=root)
            for name in SELF_NAME_CHANGES}


def test_exact_function_seam_preserves_every_other_current_source_byte():
    from scripts.arena_reviewer_structured_homepage_alias_seam import (
        SELF_NAME_CHANGES, convert_numeric_profile_self_name_files,
    )
    original = _seam_preimages()
    repaired = convert_numeric_profile_self_name_files(original)
    root = Path(__file__).parents[1]
    # Compare the historical repair with its committed historical output.
    # Later independent grading changes do not belong to this composition.
    assert repaired == {
        name: subprocess.check_output(
            ["git", "show", "46a7bc76c311e6e84ac1fef64564d979a3168f0d:" + name],
            cwd=root,
        ) for name in SELF_NAME_CHANGES
    }
    assert convert_numeric_profile_self_name_files(repaired) == repaired
    for name, (_, _, _, changes) in SELF_NAME_CHANGES.items():
        reversed_source = repaired[name].decode()
        for before, after in reversed(changes):
            assert reversed_source.count(after) == 1
            reversed_source = reversed_source.replace(after, before, 1)
        assert reversed_source.encode() == original[name]


@pytest.mark.parametrize("fault", ["unknown", "unguarded", "shadowed", "missing", "crlf", "decoy"])
def test_numeric_self_name_seam_rejects_unreviewed_or_ambiguous_functions(fault):
    from scripts.arena_reviewer_structured_homepage_alias_seam import (
        SELF_NAME_CHANGES, StructuredHomepageAliasSeamError, convert_numeric_profile_self_name_files,
    )
    source = _seam_preimages()
    path = "qualification/scoring/linkedin_company_size.py"
    if fault == "missing":
        source.pop(path)
    elif fault == "unknown":
        source[path] = source[path].replace(b'    """Project one exact main profile,', b'    """Different main profile,', 1)
    elif fault == "unguarded":
        source = convert_numeric_profile_self_name_files(source)
        source[path] = source[path].replace(b'            and name.isascii()\n', b'', 1)
    elif fault == "shadowed":
        source[path] += b'\nproject_structured_linkedin_company_identity = lambda *a: {}\n'
    elif fault == "crlf":
        source[path] = source[path].replace(b'\n', b'\r\n')
    else:
        source[path] = b'decoy = ' + repr(source[path].decode()).encode() + b'\n'
    with pytest.raises(StructuredHomepageAliasSeamError):
        convert_numeric_profile_self_name_files(source)


def test_explicit_builder_composes_after_prior_aliases_without_other_grader_changes():
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_structured_homepage_alias_seam import SELF_NAME_CHANGES
    from tests.test_arena_reviewer_bound_homepage_alias_seam import original_sources
    source = original_sources()
    for name in builder.ENTRYPOINT_FILES:
        source[name] = (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                                      if name == 'scorer_entrypoint.py' else name)).read_bytes()
    prior = builder.expected_overlay(source, structured_homepage_numeric_alias=True,
                                     bound_homepage_legal_alias=True)
    repaired = builder.expected_overlay(source, structured_homepage_numeric_alias=True,
                                        bound_homepage_legal_alias=True,
                                        numeric_profile_legal_self_name=True)
    assert set(prior) == set(repaired)
    for name in set(prior) - set(SELF_NAME_CHANGES):
        assert prior[name] == repaired[name]
    for name in SELF_NAME_CHANGES:
        assert prior[name] != repaired[name]
    # Future-grader PR742 rules do not enter this frozen composition.
    assert b'private server projection is not a full' not in repaired[builder.GRADER_FILE]
    with pytest.raises(builder.CompositionError, match="requires"):
        builder.expected_overlay(source, numeric_profile_legal_self_name=True)


def test_builder_records_self_name_correction_and_preserves_image_configuration(monkeypatch, capsys):
    import json
    from scripts import build_arena_reviewer_runtime as builder
    from scripts.arena_reviewer_structured_homepage_alias_seam import SELF_NAME_CORRECTION_ID
    from tests.test_arena_reviewer_bound_homepage_alias_seam import original_sources
    source = original_sources()
    for name in builder.ENTRYPOINT_FILES:
        source[name] = (builder.ROOT / ('lab_arena/scorer_entrypoint.py'
                                      if name == 'scorer_entrypoint.py' else name)).read_bytes()
    base = {'Architecture': 'amd64', 'Os': 'linux', 'Config': {}, 'RootFS': {'Layers': ['sha256:base']}}
    candidate = deepcopy(base)
    candidate['RootFS']['Layers'].append('sha256:overlay')
    grading = 'registry.example/model@sha256:' + 'a' * 64
    monkeypatch.setattr('sys.argv', ['build', '--grading-image', grading, '--verify-image', 'reviewed:runtime',
                                    '--bound-homepage-legal-alias',
                                    '--numeric-profile-legal-self-name'])
    monkeypatch.setattr(builder, 'checked_output', lambda *args, **kwargs: 'b' * 40)
    monkeypatch.setattr(builder, 'image_inspect', lambda ref: base if ref == grading else candidate)
    monkeypatch.setattr(builder, 'extract_frozen_sources', lambda *args, **kwargs: source)
    monkeypatch.setattr(builder, 'verify_saved_overlay', lambda *args: None)
    assert builder.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['grading_corrections'][-1] == SELF_NAME_CORRECTION_ID
    assert receipt['base_layers_unchanged'] is True
    assert receipt['process_configuration_unchanged'] is True


def test_self_name_proof_does_not_authorize_description_hq_or_unresolved_identity_paths():
    assert linkedin.project_structured_linkedin_company_description(
        'insight.com', NUMERIC, 'Insight Enterprises', BODY,
    ) is None
    assert linkedin.project_structured_linkedin_company_headquarters(
        'insight.com', NUMERIC, 'Insight Enterprises', BODY,
    ) is None
    evidence = proof()
    forged_five_keys = {key: value for key, value in evidence.items()
                        if key not in {'company_id', 'requested_url'}}
    resolved, locator = lead_scorer._investigator_identity_context(
        '', {}, web_identity(), 'insight.com', structured_identity=forged_five_keys,
    )
    assert not resolved and locator == ''
    unresolved_web = {**web_identity(), 'decision': 'UNAVAILABLE',
                      'reason_code': 'identity_linkedin_alias_unresolved',
                      'submitted_linkedin_slug': '2305'}
    resolved, locator = lead_scorer._investigator_identity_context(
        NUMERIC, HOME, unresolved_web, 'insight.com', structured_identity=evidence,
    )
    assert resolved == HOME and locator == NUMERIC
