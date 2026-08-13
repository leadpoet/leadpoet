"""Wrapper-contract conformance guard for the sourcing model.

The production sourcing flow is a harness built around frozen model symbols
(run_icp/adapter_metadata/qualify + the discovery/validation/client seams);
these tests lock the lab-side guard that protects that surface: the pure AST
verifier, the flag-gated candidate-build gate, and tolerance of the new
harness output shape in the lab's company normalizer.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from research_lab.sourcing_model_contract_check import (
    CONTRACT_PATH,
    CONTRACT_V11_PATH,
    CONTRACT_V12_PATH,
    CONTRACT_V13_PATH,
    CONTRACT_V7_PATH,
    PARITY_FIXTURE_PATH,
    PARITY_FIXTURE_V11_PATH,
    PARITY_FIXTURE_V12_PATH,
    PARITY_FIXTURE_V13_PATH,
    PARITY_FIXTURE_V7_PATH,
    load_wrapper_contract,
    resolve_reviewed_consumer_snapshot,
    reviewed_consumer_snapshots,
    verify_source_tree_contract,
)


def _write(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def _conforming_tree(
    root: Path,
    *,
    contract_snapshot: Path = CONTRACT_PATH,
    parity_snapshot: Path = PARITY_FIXTURE_PATH,
    runtime_version: int = 7,
) -> None:
    contract_document = json.loads(contract_snapshot.read_text(encoding="utf-8"))
    scrapingdog_constants = contract_document["exact_constants"][
        "sourcing_model/scrapingdog_signal_contract.py"
    ]
    _write(root, "requirements.txt", "httpx\n")
    _write(root, "research_lab_adapter.py", """
        ADAPTER_VERSION = "sourcing-model-research-lab-adapter:v3"
        COMPONENT_REGISTRY_VERSION = "sourcing-model-components:v2"

        def adapter_metadata():
            return {}

        def run_icp(icp, context=None):
            return []
    """)
    _write(root, "sourcing_model/__init__.py", "from .core import qualify\n")
    contract_path = root / "sourcing_model" / "consumer_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes(contract_snapshot.read_bytes())
    parity_path = root / "sourcing_model" / "consumer_parity_fixtures.json"
    parity_path.write_bytes(parity_snapshot.read_bytes())
    _write(
        root,
        "sourcing_model/consumer_parity.py",
        """
        def evaluate_all(document):
            return []

        def evaluate_parity_case(case):
            return {}

        def load_parity_fixtures(path=None):
            return {}

        def sha256_file(path):
            return "0" * 64

        def verify_expected_projections(document=None):
            return True
        """,
    )
    _write(root, "sourcing_model/clients.py", """
        import urllib.request

        def _exa_call(body):
            pass

        def agent_get(url):
            pass

        def agent_post(url, body):
            pass

        def exa_search(body):
            pass

        def sd_company(slug):
            pass

        def sd_scrape(url, dynamic=True):
            pass
    """)
    _write(root, "sourcing_model/core.py", """
        _GOAL_MAX_COMPANIES = 50
        _GOAL_MAX_ROUNDS = 8

        def _emitted_intent_evidence_url(c, icp, selected_url):
            pass

        def _fallback_sources(primary):
            pass

        def qualify(icp):
            return []
    """)
    _write(root, "sourcing_model/discovery.py", """
        def agent_results(icp, effort, timeout_s, avoid_companies):
            pass

        def apply_keep_gates(cand, source, today, max_age):
            pass

        def discover_goal_round(icp, source):
            pass

        def resolve_linkedin(name, anchor, agent_linkedin):
            pass
    """)
    _write(root, "sourcing_model/scoring.py", "SCORING = True\n")
    _write(root, "sourcing_model/firmographic_discovery.py", """
        def plan_for_icp(icp, *, target, allow_paid_escalation=True):
            return {}

        def policy_metadata():
            return {}
    """)
    _write(root, "sourcing_model/industry_taxonomy.py", """
        def canonical_categories(industry, subindustry):
            return ()

        def taxonomy_metadata():
            return {}
    """)
    _write(root, "sourcing_model/normalized_icp_discovery_plan.py", """
        CONTACT_PERSONA_SENIORITY_LEVELS = (
            "owner",
            "c_suite",
            "vp",
            "director",
            "manager",
            "individual_contributor",
        )
        DISCOVERY_PLAN_SCHEMA_VERSION = 2
        DISCOVERY_PLAN_CONTRACT_ID = "normalized-icp-discovery-plan:v2"

        def compile_normalized_icp_discovery_plan(normalized_icp):
            return {}
    """)
    _write(root, "sourcing_model/orchestrator.py", """
        def intent_source_for_category(category):
            return "news"

        def plan_branches(icp, *, max_companies):
            return []

        def run_branches(icp, qualify_fn, *, max_companies, runtime_options=None):
            return []
    """)
    _write(root, "sourcing_model/resilience.py", "POLICY = True\n")
    _write(root, "sourcing_model/routing/__init__.py", "")
    _write(root, "sourcing_model/routing/contracts.py", """
        def canonical_json(value):
            return "{}"

        def reject_secret_shaped_keys(value, *, path="$"):
            return None

        def sha256_payload(value):
            return "0" * 64
    """)
    _write(root, "sourcing_model/routing/compiler.py", """
        COMPILER_VERSION = "routing-compiler-v2"

        def compile_route(catalog, policy, context):
            return None
    """)
    _write(root, "sourcing_model/routing/defaults.py", """
        DEFAULT_CATALOG_VERSION = "sourcing-model-tools:v2"
        DEFAULT_POLICY_VERSION = "sourcing-model-routing:v2"

        def builtin_definitions():
            return ()

        def compile_candidate_route(
            *, structured_query, allow_paid_escalation,
            semantic_available, remaining_seconds, remaining_calls,
            remaining_results, credit_cap, cohort="control",
            catalog=None, policy=None
        ):
            return None

        def compile_intent_route(
            category, *, existing_evidence=False, available_tools=None,
            remaining_seconds=300, remaining_calls=8,
            remaining_results=1, credit_cap=8, cohort="control",
            catalog=None, policy=None
        ):
            return None

        def default_catalog(
            availability=None, *, state_overrides=None, additional_tools=(),
            additional_states=(), catalog_version=DEFAULT_CATALOG_VERSION
        ):
            return None

        def default_policy():
            return None

        def intent_source_for_category(
            category, *, catalog=None, policy=None, cohort="control"
        ):
            return "news"

        def intent_sources_for_category(
            category, *, catalog=None, policy=None, cohort="control"
        ):
            return ("news",)

        def routing_metadata():
            return {}

        def source_for_tool(tool_id):
            return "news"

        def tool_for_source(source):
            return "intent.news"
    """)
    _write(root, "sourcing_model/routing/policy.py", "POLICY = True\n")
    _write(root, "sourcing_model/routing/runtime.py", """
        RUNTIME_CATALOG_VERSION = "sourcing-model-runtime-tools:v__RUNTIME_VERSION__"
        RUNTIME_POLICY_VERSION = "sourcing-model-runtime-routing:v__RUNTIME_VERSION__"

        def runtime_tool_definitions():
            return ()

        def enhanced_scrapingdog_tool_definitions():
            return ()

        def runtime_policy():
            return None

        def runtime_catalog(
            availability=None, *, state_overrides=None, additional_tools=(),
            additional_states=(), catalog_version=RUNTIME_CATALOG_VERSION
        ):
            return None

        def candidate_route_eligibility(
            qualification_plan, *, deepline_available
        ):
            return False, ()

        def compile_candidate_acquisition_route(
            qualification_plan, *, backlog_available, registry_available,
            jobs_available, deepline_available, remaining_seconds,
            remaining_calls, remaining_results, credit_cap, cohort="control",
            catalog=None, policy=None
        ):
            return None

        def compile_intent_evidence_route(
            category, *, existing_evidence, available_tools,
            remaining_seconds, remaining_calls, credit_cap, cohort="control",
            catalog=None, policy=None
        ):
            return None

        def candidate_lane_for_tool(tool_id):
            return None

        def intent_tier_for_tool(tool_id):
            return None

        def runtime_routing_metadata():
            return {}
    """.replace("__RUNTIME_VERSION__", str(runtime_version)))
    _write(root, "sourcing_model/runtime_capabilities.py", """
        def capability_metadata():
            return {}

        def deadline():
            return None

        def register(name, implementation):
            pass
    """)
    _write(root, "sourcing_model/scrapingdog_intent.py", """
        def compile_scrapingdog_intent_request(
            tool_id, *, company_name, company_domain, signal, category,
            max_age_days, country="us", language="en"
        ):
            return {}

        def normalized_provider_date(value, *, today):
            return ""

        def select_scrapingdog_intent_evidence(
            request, candidates, *, today=None
        ):
            return None
    """)
    _write(root, "sourcing_model/scrapingdog_signal_contract.py", """
        SCHEMA_VERSION = __SCHEMA_VERSION__
        REQUEST_SCHEMA_VERSION = __REQUEST_SCHEMA_VERSION__
        EVIDENCE_SCHEMA_VERSION = "v1"

        def canonical_json(value):
            return "{}"

        def compile_request(
            tool_id, *, company_name, company_domain, verified_aliases=(),
            category, signal, subtype="active", signal_specific_terms=(),
            country="US", language="en", maximum_age_days=365,
            result_limit=20, call_budget=3, credit_budget=100,
            identifiers=None, requested_url=""
        ):
            return {}

        def contract_identity():
            return {}

        def route_for_signal(signal, *, available_tool_ids):
            return None

        def sha256_payload(value):
            return "0" * 64
    """.replace(
        "__SCHEMA_VERSION__", repr(scrapingdog_constants["SCHEMA_VERSION"])
    ).replace(
        "__REQUEST_SCHEMA_VERSION__",
        repr(scrapingdog_constants["REQUEST_SCHEMA_VERSION"]),
    ))
    _write(root, "verified_intent_event.py", """
        VERIFIED_INTENT_EVENT_SCHEMA_VERSION = "verified-intent-event:v1"

        def build_verified_intent_event(
            *, company_domain, category, event_subject, summary,
            supporting_urls
        ):
            return {}

        def intent_event_key(
            *, company_domain, category, event_subject
        ):
            return ""

        def verified_intent_event_contract_identity():
            return {}
    """)
    _write(root, "sourcing_model/validation.py", """
        def bonus_requirements(icp):
            pass

        def make_deps():
            pass

        async def validate_candidate(
            c, li, icp, deps, fetch_source, req_signal,
            seen_companies, seen_domains, today, seen_linkedin_companies=None
        ):
            pass
    """)


def test_contract_loads_and_declares_frozen_surface() -> None:
    contract = load_wrapper_contract()
    assert contract["contract_id"] == "leadpoet-sourcing-wrapper-contract-v8"
    assert (
        "sourcing_model/scrapingdog_signal_contract.py"
        in contract["required_files"]
    )
    assert "verified_intent_event.py" in contract["required_files"]
    assert "research_lab_adapter.py" in contract["functions"]
    assert contract["functions"]["research_lab_adapter.py"]["run_icp"] == [
        "icp",
        "context",
    ]
    assert contract["required_imports"]["sourcing_model/clients.py"] == [
        "urllib.request"
    ]


def test_exact_v7_and_v8_source_pairs_are_both_reviewed(tmp_path: Path) -> None:
    v8_root = tmp_path / "v8"
    _conforming_tree(v8_root)
    assert resolve_reviewed_consumer_snapshot(v8_root)["contract"][
        "contract_id"
    ].endswith("v8")
    assert verify_source_tree_contract(v8_root) == []

    v7_root = tmp_path / "v7"
    _conforming_tree(
        v7_root,
        contract_snapshot=CONTRACT_V7_PATH,
        parity_snapshot=PARITY_FIXTURE_V7_PATH,
        runtime_version=6,
    )
    assert resolve_reviewed_consumer_snapshot(v7_root)["contract"][
        "contract_id"
    ].endswith("v7")
    assert verify_source_tree_contract(v7_root) == []


def test_exact_v11_contract_and_parity_pair_is_reviewed(tmp_path: Path) -> None:
    root = tmp_path / "v11"
    contract = json.loads(CONTRACT_V11_PATH.read_text(encoding="utf-8"))
    contract_path = root / contract["canonical_path"]
    parity_path = root / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V11_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V11_PATH.read_bytes())

    resolved = resolve_reviewed_consumer_snapshot(root)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v11"
    )
    assert set(reviewed_consumer_snapshots()) == {
        "leadpoet-sourcing-wrapper-contract-v7",
        "leadpoet-sourcing-wrapper-contract-v8",
        "leadpoet-sourcing-wrapper-contract-v11",
        "leadpoet-sourcing-wrapper-contract-v12",
        "leadpoet-sourcing-wrapper-contract-v13",
        "leadpoet-sourcing-wrapper-contract-v27",
    }


def test_exact_v12_contact_contract_and_parity_pair_is_reviewed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v12"
    contract = json.loads(CONTRACT_V12_PATH.read_text(encoding="utf-8"))
    parity = json.loads(PARITY_FIXTURE_V12_PATH.read_text(encoding="utf-8"))
    contract_path = root / contract["canonical_path"]
    parity_path = root / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V12_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V12_PATH.read_bytes())

    resolved = resolve_reviewed_consumer_snapshot(root)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v12"
    )
    contact_cases = parity["contact_acquisition_parity_cases"]
    assert {case["fulfillment_mode"] for case in contact_cases} == {
        "company_only",
        "contact_optional",
        "contact_required",
    }
    projected_plans = [
        item["contact_acquisition_plan"]
        for item in parity["expected_contact_acquisition_projections"]
    ]
    assert next(
        plan for plan in projected_plans
        if plan["fulfillment_mode"] == "company_only"
    )["binding_requests"] == []
    optional = next(
        plan for plan in projected_plans
        if plan["fulfillment_mode"] == "contact_optional"
        and len(plan["ordered_tool_ids"]) == 3
    )
    assert [request["tool_id"] for request in optional["binding_requests"]] == (
        optional["ordered_tool_ids"]
    )
    assert all(
        request["roles_any_of"] == ["VP Sales", "Founder"]
        for request in optional["binding_requests"]
    )
    rendered = json.dumps(optional, sort_keys=True).casefold()
    for private_field in (
        "credential",
        "endpoint",
        "personal_email",
        "phone_number",
        "raw_provider",
    ):
        assert private_field not in rendered


def test_exact_v13_contact_verification_contract_and_parity_pair_is_reviewed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "v13"
    contract = json.loads(CONTRACT_V13_PATH.read_text(encoding="utf-8"))
    parity = json.loads(PARITY_FIXTURE_V13_PATH.read_text(encoding="utf-8"))
    contract_path = root / contract["canonical_path"]
    parity_path = root / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V13_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V13_PATH.read_bytes())

    resolved = resolve_reviewed_consumer_snapshot(root)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v13"
    )
    verification_cases = parity["contact_verification_parity_cases"]
    assert verification_cases
    assert len(parity["expected_contact_verification_projections"]) == len(
        verification_cases
    )
    decisions = [
        item["contact_verification_decision"]
        for item in parity["expected_contact_verification_projections"]
    ]
    assert {decision["status"] for decision in decisions} == {
        "accepted",
        "rejected",
        "unresolved",
    }
    assert all(decision["contract_id"] == "contact-verification:v1" for decision in decisions)


def test_mixed_reviewed_contract_and_parity_versions_fail_closed(
    tmp_path: Path,
) -> None:
    _conforming_tree(
        tmp_path,
        contract_snapshot=CONTRACT_V7_PATH,
        parity_snapshot=PARITY_FIXTURE_PATH,
        runtime_version=6,
    )

    assert resolve_reviewed_consumer_snapshot(tmp_path) is None
    assert (
        "model-owned compatibility contract/parity pair is not reviewed"
        in verify_source_tree_contract(tmp_path)
    )


def test_nonidentical_contract_and_parity_snapshots_fail_closed(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    contract = tmp_path / "sourcing_model" / "consumer_contract.json"
    contract.write_text(contract.read_text() + "\n", encoding="utf-8")
    violations = verify_source_tree_contract(tmp_path)
    assert any("compatibility contract differs" in item for item in violations)

    _conforming_tree(tmp_path)
    fixtures = (
        tmp_path / "sourcing_model" / "consumer_parity_fixtures.json"
    )
    fixtures.write_text(fixtures.read_text() + "\n", encoding="utf-8")
    violations = verify_source_tree_contract(tmp_path)
    assert any("parity fixtures differ" in item for item in violations)


def test_conforming_tree_has_no_violations(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    assert verify_source_tree_contract(tmp_path) == []


def test_missing_file_and_function_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    (tmp_path / "sourcing_model/validation.py").unlink()
    violations = verify_source_tree_contract(tmp_path)
    assert "missing required file: sourcing_model/validation.py" in violations
    assert not any("run_icp" in v for v in violations)


def test_parameter_drift_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "research_lab_adapter.py", """
        def adapter_metadata():
            return {}

        def run_icp(request, context=None):
            return []
    """)
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        v.startswith("exact parameter drift research_lab_adapter.py:run_icp")
        for v in violations
    )


def test_direct_entrypoints_reject_optional_or_required_trailing_parameters(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "research_lab_adapter.py", """
        def adapter_metadata(required=False):
            return {}

        def run_icp(icp, context=None, extra=None):
            return []
    """)
    violations = verify_source_tree_contract(tmp_path)
    assert any("exact parameter drift" in item and "adapter_metadata" in item for item in violations)
    assert any("exact parameter drift" in item and "run_icp" in item for item in violations)


@pytest.mark.parametrize(
    "signature",
    (
        "def run_icp(icp, context=None, *args):",
        "def run_icp(icp, context=None, **kwargs):",
        "def run_icp(icp, context=None, *, extra=None):",
        "def run_icp(icp, /, context=None):",
    ),
)
def test_direct_entrypoints_reject_parameter_kind_drift(
    tmp_path: Path,
    signature: str,
) -> None:
    _conforming_tree(tmp_path)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "def run_icp(icp, context=None):",
            signature,
        ),
        encoding="utf-8",
    )

    violations = verify_source_tree_contract(tmp_path)

    assert any(
        "exact parameter drift research_lab_adapter.py:run_icp" in item
        for item in violations
    )


def test_internal_seam_rejects_additional_required_parameter(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    clients = tmp_path / "sourcing_model" / "clients.py"
    clients.write_text(
        clients.read_text().replace(
            "def exa_search(body):",
            "def exa_search(body, required):",
        )
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any("required parameter drift" in item and "exa_search" in item for item in violations)


def test_wrapper_reachable_optional_parameter_surface_is_exact(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    module = tmp_path / "sourcing_model" / "firmographic_discovery.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace(
            ", allow_paid_escalation=True", ""
        ),
        encoding="utf-8",
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        "full parameter drift" in item
        and "firmographic_discovery.py:plan_for_icp" in item
        for item in violations
    )


def test_full_parameter_surface_rejects_variadic_escape(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    module = tmp_path / "sourcing_model" / "consumer_parity.py"
    module.write_text(
        module.read_text(encoding="utf-8").replace(
            "def evaluate_all(document):",
            "def evaluate_all(document, *args, **kwargs):",
        ),
        encoding="utf-8",
    )

    violations = verify_source_tree_contract(tmp_path)

    assert any(
        "full parameter drift" in item
        and "consumer_parity.py:evaluate_all" in item
        for item in violations
    )


def test_integer_floor_breach_reported(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "sourcing_model/core.py", """
        _GOAL_MAX_COMPANIES = 10
        _GOAL_MAX_ROUNDS = 8

        def _emitted_intent_evidence_url(c, icp, selected_url):
            pass

        def _fallback_sources(primary):
            pass

        def qualify(icp):
            return []
    """)
    violations = verify_source_tree_contract(tmp_path)
    assert any("integer floor breach" in v and "_GOAL_MAX_COMPANIES" in v for v in violations)


def test_unparseable_module_reported_not_raised(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    _write(tmp_path, "sourcing_model/core.py", "def qualify(icp:\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any(v.startswith("unparseable module sourcing_model/core.py") for v in violations)


def test_annotated_constant_assignment_conforms(tmp_path: Path) -> None:
    """``X: int = 50`` is behaviorally identical to ``X = 50`` and must not
    be reported as a missing constant (it would wrongly fail a legitimate
    candidate under enforce)."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text().replace(
            "_GOAL_MAX_COMPANIES = ", "_GOAL_MAX_COMPANIES: int = ", 1
        )
    )
    assert verify_source_tree_contract(tmp_path) == []


def test_non_utf8_coding_header_is_violation_not_crash(tmp_path: Path) -> None:
    """A legal PEP 263 latin-1 module must parse like the interpreter parses
    it; a truly undecodable file must surface as a violation — never as an
    exception that would let the build gate fail open."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    legal = ("# -*- coding: latin-1 -*-\n# café\n" + core.read_text()).encode(
        "latin-1"
    )
    compile(legal, "core.py", "exec")  # sanity: importable Python
    core.write_bytes(legal)
    assert verify_source_tree_contract(tmp_path) == []

    # Null byte: rejected by the parser with ValueError, must be a violation.
    core.write_bytes(b"_GOAL_MAX_COMPANIES = 50\x00\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        v.startswith(("unreadable module sourcing_model/core.py",
                      "unparseable module sourcing_model/core.py"))
        for v in violations
    )


def test_dynamic_constant_rebinding_is_violation(tmp_path: Path) -> None:
    """A conforming literal followed by a top-level non-literal rebinding
    means the runtime value is no longer statically verifiable — the earlier
    literal must not satisfy the floor check."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text()
        + "\n_GOAL_MAX_COMPANIES = min(2, _GOAL_MAX_ROUNDS)\n"
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any("_GOAL_MAX_COMPANIES" in v and "missing integer constant" in v
               for v in violations)

    # Augmented assignment likewise poisons the constant.
    _conforming_tree(tmp_path)
    core.write_text(core.read_text() + "\n_GOAL_MAX_ROUNDS -= 5\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any("_GOAL_MAX_ROUNDS" in v and "missing integer constant" in v
               for v in violations)

    # And a later conforming literal restores verifiability (last wins).
    _conforming_tree(tmp_path)
    core.write_text(core.read_text() + "\n_GOAL_MAX_COMPANIES = 60\n")
    assert verify_source_tree_contract(tmp_path) == []


def test_exact_constants_accept_annotated_literal_assignment(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    defaults = tmp_path / "sourcing_model" / "routing" / "defaults.py"
    defaults.write_text(
        defaults.read_text(encoding="utf-8").replace(
            'DEFAULT_POLICY_VERSION = "sourcing-model-routing:v2"',
            'DEFAULT_POLICY_VERSION: str = "sourcing-model-routing:v2"',
        ),
        encoding="utf-8",
    )

    assert verify_source_tree_contract(tmp_path) == []


@pytest.mark.parametrize(
    "mutation",
    (
        'DEFAULT_POLICY_VERSION += "-tampered"',
        "del DEFAULT_POLICY_VERSION",
        'if True:\n    DEFAULT_POLICY_VERSION = "tampered"',
        'DEFAULT_POLICY_VERSION = "tampered"',
    ),
)
def test_exact_constant_rebinding_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    _conforming_tree(tmp_path)
    defaults = tmp_path / "sourcing_model" / "routing" / "defaults.py"
    defaults.write_text(
        defaults.read_text(encoding="utf-8") + "\n" + mutation + "\n",
        encoding="utf-8",
    )

    violations = verify_source_tree_contract(tmp_path)

    assert any(
        "exact constant drift" in item and "DEFAULT_POLICY_VERSION" in item
        for item in violations
    )


def test_exact_constant_comparison_is_type_sensitive(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    plan = tmp_path / "sourcing_model" / "normalized_icp_discovery_plan.py"
    plan.write_text(
        plan.read_text(encoding="utf-8").replace(
            "DISCOVERY_PLAN_SCHEMA_VERSION = 2",
            "DISCOVERY_PLAN_SCHEMA_VERSION = True",
        ),
        encoding="utf-8",
    )

    violations = verify_source_tree_contract(tmp_path)

    assert any(
        "exact constant drift" in item
        and "DISCOVERY_PLAN_SCHEMA_VERSION" in item
        for item in violations
    )


def test_asyncness_drift_is_violation(tmp_path: Path) -> None:
    """Sync-vs-async is part of the callable surface: flipping it hands
    callers a coroutine (or breaks an await) despite identical parameters."""
    # Frozen-async function made sync → violation.
    _conforming_tree(tmp_path)
    val = tmp_path / "sourcing_model" / "validation.py"
    val.write_text(val.read_text().replace("async def validate_candidate", "def validate_candidate"))
    violations = verify_source_tree_contract(tmp_path)
    assert any("asyncness drift" in v and "validate_candidate" in v for v in violations)

    # Frozen-sync function made async → violation.
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(core.read_text().replace("def qualify(icp):", "async def qualify(icp):"))
    violations = verify_source_tree_contract(tmp_path)
    assert any("asyncness drift" in v and "qualify" in v for v in violations)


@pytest.mark.parametrize(
    "replacement",
    (
        "from urllib.request import urlopen",
        "import urllib.request as request",
    ),
)
def test_wrapper_reachable_import_must_remain_bound(
    tmp_path: Path,
    replacement: str,
) -> None:
    _conforming_tree(tmp_path)
    clients = tmp_path / "sourcing_model" / "clients.py"
    clients.write_text(
        clients.read_text().replace("import urllib.request", replacement)
    )

    violations = verify_source_tree_contract(tmp_path)

    assert (
        "missing bound import sourcing_model/clients.py:urllib.request"
        in violations
    )


def test_function_local_import_does_not_satisfy_wrapper_binding(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    clients = tmp_path / "sourcing_model" / "clients.py"
    clients.write_text(
        clients.read_text().replace(
            "import urllib.request",
            "def _bind_too_late():\n    import urllib.request",
        )
    )

    violations = verify_source_tree_contract(tmp_path)

    assert (
        "missing bound import sourcing_model/clients.py:urllib.request"
        in violations
    )


def test_simple_alias_rebinding_conforms(tmp_path: Path) -> None:
    """``qualify = _impl`` is a runtime-valid rebinding — the alias carries
    the implementation's surface instead of reporting a missing function."""
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(
        core.read_text().replace("def qualify(icp):", "def _qualify_impl(icp):")
        + "\nqualify = _qualify_impl\n"
    )
    assert verify_source_tree_contract(tmp_path) == []

    # Aliasing to a wrong-signature implementation still drifts.
    _conforming_tree(tmp_path)
    core.write_text(
        core.read_text().replace("def qualify(icp):", "def _qualify_impl(wrong_name):")
        + "\nqualify = _qualify_impl\n"
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any("parameter drift" in v and "qualify" in v for v in violations)


def test_deleted_function_is_missing(tmp_path: Path) -> None:
    _conforming_tree(tmp_path)
    core = tmp_path / "sourcing_model" / "core.py"
    core.write_text(core.read_text() + "\ndel qualify\n")
    violations = verify_source_tree_contract(tmp_path)
    assert any("missing function" in v and "qualify" in v for v in violations)


# ---------------------------------------------------------------------------
# The flag-gated candidate-build gate
# ---------------------------------------------------------------------------


def test_build_gate_shadow_logs_and_proceeds(tmp_path: Path, monkeypatch, caplog) -> None:
    from gateway.research_lab.code_build import _sourcing_contract_gate

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "shadow")
    _conforming_tree(tmp_path)
    (tmp_path / "research_lab_adapter.py").unlink()
    with caplog.at_level(
        "WARNING",
        logger="gateway.research_lab.code_build",
    ):
        _sourcing_contract_gate(tmp_path)  # must NOT raise in shadow
    assert any(
        "sourcing_contract_gate_shadow_violation" in rec.message for rec in caplog.records
    )


def test_build_gate_enforce_fails_broken_tree(tmp_path: Path, monkeypatch) -> None:
    from gateway.research_lab.code_build import (
        CodeEditPrivateTestError,
        _sourcing_contract_gate,
    )

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    _conforming_tree(tmp_path)
    (tmp_path / "research_lab_adapter.py").unlink()
    with pytest.raises(CodeEditPrivateTestError, match="wrapper contract violation"):
        _sourcing_contract_gate(tmp_path)
    # a conforming tree passes enforce silently
    _conforming_tree(tmp_path)
    _sourcing_contract_gate(tmp_path)


def test_build_gate_disabled_and_enforce_fails_closed(tmp_path: Path, monkeypatch) -> None:
    import gateway.research_lab.code_build as cb

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "disabled")
    cb._sourcing_contract_gate(tmp_path)  # empty tree, disabled -> no-op

    # Internal verifier failure fails closed in enforce mode.
    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    import research_lab.sourcing_model_contract_check as check_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("contract file unreadable")

    monkeypatch.setattr(check_mod, "verify_source_tree_contract", _boom)
    with pytest.raises(cb.CodeEditPrivateTestError, match="failed internally"):
        cb._sourcing_contract_gate(tmp_path)


# ---------------------------------------------------------------------------
# Harness output tolerance: the new flow's company shape through the lab
# normalizer (subindustry/hq_* mapping, extra fields ignored, intents mapped)
# ---------------------------------------------------------------------------


def test_lab_normalizer_tolerates_harness_company_shape() -> None:
    from research_lab.eval.evaluator import _normalize_company_output

    harness_company = {
        "company_name": "Acme Robotics",
        "domain": "acmerobotics.io",
        "company_website": "https://acmerobotics.io",
        "company_linkedin": "https://www.linkedin.com/company/acme-robotics",
        "industry": "Manufacturing",
        "subindustry": "Robotics",
        "hq_city": "Austin",
        "hq_state": "Texas",
        "hq_country": "United States",
        "employee_count": "51-200",
        "company_stage": "Series B",
        "description": "Industrial robotics automation",
        "intent": {
            "source": "news",
            "url": "https://technews.io/acme-expansion",
            "date": "2026-07-01",
            "signal": "Acme announced a facility expansion",
            "why_valid": "Direct first-party announcement",
        },
        "required_attribute": {
            "text": "Manufactures its own hardware",
            "passed": True,
            "evidence_url": "https://acmerobotics.io/products",
            "evidence_quote": "We design and manufacture our robots in-house",
        },
        "additional_intents": [
            {
                "category": "HIRING",
                "signal": "Hiring robotics engineers",
                "source": "job_listing",
                "url": "https://jobs.acmerobotics.io/1",
                "date": "2026-07-10",
                "why_valid": "Live posting",
                "points": 40,
            }
        ],
        "score": 87,
        "discovery_audit": {"lane": "exa_agent", "round": 2},
    }
    normalized = _normalize_company_output(harness_company)
    assert normalized["company_name"] == "Acme Robotics"
    assert normalized["sub_industry"] == "Robotics"      # subindustry mapped
    assert normalized["country"] == "United States"       # hq_country mapped
    assert normalized["employee_count"] == "51-200"
    # intents arrive as a scoreable list with the primary mapped to index 0
    signals = normalized.get("intent_signals") or []
    assert signals, "primary intent must map into intent_signals"
