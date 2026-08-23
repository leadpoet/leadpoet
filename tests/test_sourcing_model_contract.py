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
    CONTRACT_V26_PATH,
    CONTRACT_V46_PATH,
    CONTRACT_V47_PATH,
    CONTRACT_V52_PATH,
    CONTRACT_V52_82C_PATH,
    CONTRACT_V55_PATH,
    CONTRACT_V55_E55_PATH,
    CONTRACT_V7_PATH,
    PARITY_FIXTURE_PATH,
    PARITY_FIXTURE_V11_PATH,
    PARITY_FIXTURE_V12_PATH,
    PARITY_FIXTURE_V13_PATH,
    PARITY_FIXTURE_V26_PATH,
    PARITY_FIXTURE_V46_PATH,
    PARITY_FIXTURE_V47_PATH,
    PARITY_FIXTURE_V52_PATH,
    PARITY_FIXTURE_V52_82C_PATH,
    PARITY_FIXTURE_V55_PATH,
    PARITY_FIXTURE_V55_E55_PATH,
    PARITY_FIXTURE_V7_PATH,
    _resolve_reviewed_consumer_contract_pair,
    _reviewed_consumer_snapshot_for_source_hash,
    load_wrapper_contract,
    resolve_reviewed_consumer_snapshot,
    reviewed_consumer_profiles,
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


def test_exact_v7_and_v8_document_pairs_are_both_reviewed(tmp_path: Path) -> None:
    v8_root = tmp_path / "v8"
    _conforming_tree(v8_root)
    assert _resolve_reviewed_consumer_contract_pair(v8_root)["contract"][
        "contract_id"
    ].endswith("v8")
    assert resolve_reviewed_consumer_snapshot(v8_root) is None
    assert verify_source_tree_contract(v8_root) == []

    v7_root = tmp_path / "v7"
    _conforming_tree(
        v7_root,
        contract_snapshot=CONTRACT_V7_PATH,
        parity_snapshot=PARITY_FIXTURE_V7_PATH,
        runtime_version=6,
    )
    assert _resolve_reviewed_consumer_contract_pair(v7_root)["contract"][
        "contract_id"
    ].endswith("v7")
    assert resolve_reviewed_consumer_snapshot(v7_root) is None
    assert verify_source_tree_contract(v7_root) == []


def test_exact_v11_contract_and_parity_pair_is_reviewed(tmp_path: Path) -> None:
    root = tmp_path / "v11"
    contract = json.loads(CONTRACT_V11_PATH.read_text(encoding="utf-8"))
    contract_path = root / contract["canonical_path"]
    parity_path = root / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V11_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V11_PATH.read_bytes())

    resolved = _resolve_reviewed_consumer_contract_pair(root)

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
        "leadpoet-sourcing-wrapper-contract-v26",
        "leadpoet-sourcing-wrapper-contract-v46",
        "leadpoet-sourcing-wrapper-contract-v47",
        "leadpoet-sourcing-wrapper-contract-v52",
        "leadpoet-sourcing-wrapper-contract-v55",
    }


def test_exact_v26_contract_pair_and_keyword_only_surface_are_reviewed(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT_V26_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V26_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V26_PATH.read_bytes())
    _write(
        tmp_path,
        "sourcing_model/intent_freshness.py",
        """
        def build_source_date_proof(
            *, source_url, source_snapshot_sha256, source_metadata_sha256,
            fetched_at, category, source_kind, source_class, provenance,
            raw_date, evaluated_on, requested_maximum_age_days
        ):
            return None
        """,
    )

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)
    violations = verify_source_tree_contract(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v26"
    )
    assert not any(
        "intent_freshness.py:build_source_date_proof" in item
        for item in violations
    )


def test_exact_v46_contract_pair_is_reviewed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_V46_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V46_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V46_PATH.read_bytes())
    _write(
        tmp_path,
        "sourcing_model/corporate_filing_contract.py",
        """
        def compile_corporate_filing_request(
            *, company_name, company_domain, category, issuer_anchors,
            filing_forms, jurisdictions, reference_date, filters=None,
            maximum_age_days=365, used_filing_ids=(), used_filing_urls=(),
            query_variants=()
        ):
            return None

        def build_corporate_filing_envelope(*, request=None, **payload):
            return None
        """,
    )

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)
    violations = verify_source_tree_contract(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v46"
    )
    assert not any(
        "corporate_filing_contract.py" in item and "parameter drift" in item
        for item in violations
    )

    corporate_path = tmp_path / "sourcing_model/corporate_filing_contract.py"
    corporate_path.write_text(
        corporate_path.read_text(encoding="utf-8").replace(
            "**payload", "**unreviewed_payload"
        ),
        encoding="utf-8",
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        "build_corporate_filing_envelope" in item
        and "parameter drift" in item
        for item in violations
    )


def test_exact_v47_contract_pair_and_intent_outcome_surface_are_reviewed(
    tmp_path: Path,
) -> None:
    contract = json.loads(CONTRACT_V47_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V47_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V47_PATH.read_bytes())
    _write(
        tmp_path,
        "sourcing_model/intent_evidence_outcome.py",
        """
        def intent_evidence_outcome_contract_identity():
            return None

        def parse_intent_evidence_outcome(value):
            return None

        def project_intent_evidence_outcome(
            *, review_state, company_qualification, intent_verification,
            category, source, exact_source_retrieval, stage3_admission,
            reason_code, reference_date, maximum_age_days
        ):
            return None

        def project_intent_stage3_admission(*, category, stage1, source):
            return None
        """,
    )

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)
    violations = verify_source_tree_contract(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v47"
    )
    assert resolve_reviewed_consumer_snapshot(tmp_path) is None
    assert not any(
        "intent_evidence_outcome.py" in item and "parameter drift" in item
        for item in violations
    )

    outcome_path = tmp_path / "sourcing_model/intent_evidence_outcome.py"
    outcome_path.write_text(
        outcome_path.read_text(encoding="utf-8").replace(
            "*, review_state", "review_state"
        ),
        encoding="utf-8",
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        "project_intent_evidence_outcome" in item
        and "parameter drift" in item
        for item in violations
    )


def test_exact_v52_contract_and_parity_pair_is_reviewed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_V52_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V52_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V52_PATH.read_bytes())

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v52"
    )
    assert (
        "sourcing_model/routing/candidate_profiles.py"
        in resolved["contract"]["required_files"]
    )
    assert resolved["contract_sha256"] == (
        "sha256:2454c60c1e2614feef912aa6ea471307657dac7d418bdb3bdab5b105ddbb5932"
    )
    assert resolved["parity_sha256"] == (
        "sha256:7d18b358f7f6dcf1b58a175af43288a1db244c08af6fc5295116dbfe51976332"
    )
    releases = reviewed_consumer_snapshots()[
        "leadpoet-sourcing-wrapper-contract-v52"
    ]["release_identities"]
    assert releases == (
        {
            "source_tree_hash": (
                "sha256:603c4569fa35d6a66ee60596a44e37841aab1c6d794c3109349c1d6b7a5bcd85"
            ),
            "git_commit_sha": "6ed6289626b7e81c745daff97feabd237aa4ccee",
            "manifest_hash": (
                "sha256:e75c820acf1e2d1348aab3d34b85c3ae578fe8043d5ef97b28817a8b234bd3c0"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:5539652d47471773ca94221373ce01b5e610715177155a12ae8026da48e2ae52"
            ),
        },
        {
            "source_tree_hash": (
                "sha256:946fe12e38efa08c08631c864591bdf99c0538e6c450bdf4c33fbba3e167a969"
            ),
            "git_commit_sha": "ec5c0e7c7314e123c9fdafff63d2b809cb254cfd",
            "manifest_hash": (
                "sha256:ee0a1ad40a12d33dabd4d7fb68d4b9507cbfcbf2fe276a69e4a05cb82dc93f52"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:164cdf6b5c8c37af61d61c2d4b5c0a22fc248014be8289c0db2f242602459607"
            ),
        },
    )

    runtime_capabilities = tmp_path / "sourcing_model/runtime_capabilities.py"
    runtime_capabilities.write_text(
        'CAPABILITY_CONTRACT_VERSION = "sourcing-model-runtime-capabilities:v3"\n',
        encoding="utf-8",
    )
    violations = verify_source_tree_contract(tmp_path)
    assert not any("CAPABILITY_CONTRACT_VERSION" in item for item in violations)

    runtime_capabilities.write_text(
        'CAPABILITY_CONTRACT_VERSION = "sourcing-model-runtime-capabilities:v4"\n',
        encoding="utf-8",
    )
    violations = verify_source_tree_contract(tmp_path)
    assert any(
        "reviewed source constant drift "
        "sourcing_model/runtime_capabilities.py:CAPABILITY_CONTRACT_VERSION"
        in item
        for item in violations
    )

    parity_path.write_text("{}\n", encoding="utf-8")
    assert _resolve_reviewed_consumer_contract_pair(tmp_path) is None


def test_exact_v55_contract_and_parity_pair_is_reviewed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_V55_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V55_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V55_PATH.read_bytes())

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v55"
    )
    assert resolved["contract_sha256"] == (
        "sha256:02fcbcd84b2c887d0f6ba1515fba280267fc5d2571876f990acc865f4a038d2a"
    )
    assert resolved["parity_sha256"] == (
        "sha256:fe0e1faff8e45b432459dda2d5f5bf131aef2b5f60935d48395a814c7ed59573"
    )
    assert resolved["required_source_constants"] == {
        "sourcing_model/runtime_capabilities.py": {
            "CAPABILITY_CONTRACT_VERSION": "sourcing-model-runtime-capabilities:v3"
        }
    }
    assert resolved["release_identities"] == (
        {
            "source_tree_hash": (
                "sha256:a34a9158480dff89a53a7a5a3df27325239b7b64f476cb6f48c593520eca3858"
            ),
            "git_commit_sha": "0be5905ee24a4d8bb3ec6f316af3e8891f763919",
            "manifest_hash": (
                "sha256:2f4876077475c7c33135ec9b727e010d7e7845e3591d9edc9101e045dcaa8c01"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:ed7c03e744ba9bccd13e7608a6eabf7bcdd828dabaf02e09efd2774d7187d6a5"
            ),
        },
        {
            "source_tree_hash": (
                "sha256:2690deb3a6b9c8952e4ecd153458cfee1b0cebbd4edb79eb13129c3e96e673d5"
            ),
            "git_commit_sha": "cf6630732f7f8f16150d9dd3908dcd7f91ae7667",
            "manifest_hash": (
                "sha256:518022b4667471f866ef4cd66b1756f6d79ebe1757e44c9194ddd7687635eddd"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:a6cba9be0ff862883d9a7f33eccbb0212aee89d4949c4da10e14b0d5b0c21165"
            ),
        },
    )


def test_exact_v55_e55_revision_is_independently_reviewed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_V55_E55_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V55_E55_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V55_E55_PATH.read_bytes())

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v55"
    )
    assert resolved["contract_sha256"] == (
        "sha256:b89eda998cf8cf3d9ee80c4ccd2bd4e10e37d6e4bdd7be80e2dc70492d2c0ffd"
    )
    assert resolved["parity_sha256"] == (
        "sha256:b75f79a8b7c3eb72c24b14ceab7c84442e394dd8c738a627dbbb22ed4bf4271a"
    )
    assert resolved["release_identities"] == (
        {
            "source_tree_hash": (
                "sha256:491d6e76adf629b60d913062005191673f962db3cd5cd77223a68cf6262ac60f"
            ),
            "git_commit_sha": "e55e57f2be0ddadcc6b9c92c18b932dc2c354d21",
            "manifest_hash": (
                "sha256:af68f0fbd29c77f9ffe686dcbddbc1e5dd1cab6c8725c7c9669de367bd592928"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:f1ae9bc0ba2cd55450e4c1b1bbdb0030514dbf5afd380f29a09d5e95bdb0ade5"
            ),
        },
    )
    assert len(
        [
            profile
            for profile in reviewed_consumer_profiles()
            if profile["contract"]["contract_id"]
            == "leadpoet-sourcing-wrapper-contract-v55"
        ]
    ) == 2


def test_exact_v52_82c_revision_is_independently_reviewed(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_V52_82C_PATH.read_text(encoding="utf-8"))
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(CONTRACT_V52_82C_PATH.read_bytes())
    parity_path.write_bytes(PARITY_FIXTURE_V52_82C_PATH.read_bytes())

    resolved = _resolve_reviewed_consumer_contract_pair(tmp_path)

    assert resolved is not None
    assert resolved["contract"]["contract_id"] == (
        "leadpoet-sourcing-wrapper-contract-v52"
    )
    assert resolved["contract_sha256"] == (
        "sha256:48609451a69cf41a6a7615224e628417df4a27040a1b54c9958460cc76a48fc9"
    )
    assert resolved["parity_sha256"] == (
        "sha256:1e06b5bbe638356661494054363fbba8b8cba0181260b3396ce259f129d90e5d"
    )
    assert resolved["release_identities"] == (
        {
            "source_tree_hash": (
                "sha256:6835100e66840dab82a08d93abfeaba8cbaf51484c20e62a91c787c9d36366aa"
            ),
            "git_commit_sha": "82cfc8ecc1d57fd91f6a56ad4d2b7fd4fc4f2e43",
            "manifest_hash": (
                "sha256:168b4fb51a20cc82835d35905ae0dcf5bd39e6a1c2115b289dd6c9cb975c3652"
            ),
            "image_digest": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:1d4b55a84575559b2c8a13663d59b48985caa834fc4fb4fa34ba76c4f552b83f"
            ),
        },
    )
    assert len(
        [
            profile
            for profile in reviewed_consumer_profiles()
            if profile["contract"]["contract_id"]
            == "leadpoet-sourcing-wrapper-contract-v52"
        ]
    ) == 2

    release_manifest = {
        "model_artifact_hash": (
            "sha256:6835100e66840dab82a08d93abfeaba8cbaf51484c20e62a91c787c9d36366aa"
        ),
        "git_commit_sha": "82cfc8ecc1d57fd91f6a56ad4d2b7fd4fc4f2e43",
        "manifest_hash": (
            "sha256:168b4fb51a20cc82835d35905ae0dcf5bd39e6a1c2115b289dd6c9cb975c3652"
        ),
        "image_digest": (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
            "sourcing-model@sha256:1d4b55a84575559b2c8a13663d59b48985caa834fc4fb4fa34ba76c4f552b83f"
        ),
    }
    profiled = _reviewed_consumer_snapshot_for_source_hash(
        tmp_path,
        source_tree_hash=release_manifest["model_artifact_hash"],
        manifest=release_manifest,
    )
    assert profiled is not None
    assert profiled["contract_sha256"] == resolved["contract_sha256"]

    with pytest.raises(
        ValueError,
        match="reviewed legacy source manifest identity differs",
    ):
        _reviewed_consumer_snapshot_for_source_hash(
            tmp_path,
            source_tree_hash=release_manifest["model_artifact_hash"],
            manifest={**release_manifest, "git_commit_sha": "0" * 40},
        )

    parity_path.write_text("{}\n", encoding="utf-8")
    assert _resolve_reviewed_consumer_contract_pair(tmp_path) is None


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

    resolved = _resolve_reviewed_consumer_contract_pair(root)

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

    resolved = _resolve_reviewed_consumer_contract_pair(root)

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


def test_mixed_release_documents_do_not_bypass_semantic_admission(
    tmp_path: Path,
) -> None:
    _conforming_tree(
        tmp_path,
        contract_snapshot=CONTRACT_V7_PATH,
        parity_snapshot=PARITY_FIXTURE_PATH,
        runtime_version=6,
    )

    assert resolve_reviewed_consumer_snapshot(tmp_path) is None
    violations = verify_source_tree_contract(tmp_path)
    assert violations
    assert any("hard module semantic drift" in item for item in violations)


def test_nonidentical_legacy_documents_route_through_semantic_admission(
    tmp_path: Path,
) -> None:
    _conforming_tree(tmp_path)
    contract = tmp_path / "sourcing_model" / "consumer_contract.json"
    contract.write_text(contract.read_text() + "\n", encoding="utf-8")
    violations = verify_source_tree_contract(tmp_path)
    assert violations
    assert any("hard module semantic drift" in item for item in violations)

    _conforming_tree(tmp_path)
    fixtures = (
        tmp_path / "sourcing_model" / "consumer_parity_fixtures.json"
    )
    fixtures.write_text(fixtures.read_text() + "\n", encoding="utf-8")
    violations = verify_source_tree_contract(tmp_path)
    assert violations
    assert any("hard module semantic drift" in item for item in violations)


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


def test_build_gate_enforce_fails_broken_or_unreviewed_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from gateway.research_lab.code_build import (
        CodeEditPrivateTestError,
        _sourcing_contract_gate,
    )

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    _conforming_tree(tmp_path)
    (tmp_path / "research_lab_adapter.py").unlink()
    with pytest.raises(CodeEditPrivateTestError, match="wrapper contract violation"):
        _sourcing_contract_gate(tmp_path)
    # Exact legacy document conformance is not source admission. This
    # synthetic tree is not one of the signed rollback identities.
    _conforming_tree(tmp_path)
    with pytest.raises(CodeEditPrivateTestError, match="wrapper contract violation"):
        _sourcing_contract_gate(tmp_path)


def test_build_gate_disabled_and_enforce_fails_closed(tmp_path: Path, monkeypatch) -> None:
    import gateway.research_lab.code_build as cb

    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "disabled")
    cb._sourcing_contract_gate(tmp_path)  # empty tree, disabled -> no-op

    # Internal verifier failure fails closed in enforce mode.
    monkeypatch.setenv("RESEARCH_LAB_SOURCING_CONTRACT_CHECK", "enforce")
    def _boom(*args, **kwargs):
        raise RuntimeError("contract file unreadable")

    monkeypatch.setattr(cb, "source_tree_compatibility_admission", _boom)
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
