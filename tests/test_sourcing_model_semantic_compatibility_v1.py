from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.research_lab import promotion
from gateway.research_lab import model_authority_v2
from gateway.tee import model_sandbox_v2
from research_lab.eval import (
    PrivateModelArtifactManifest,
    build_local_private_artifact_manifest,
    compute_private_source_tree_hash,
)
from research_lab.eval.private_runtime import (
    PrivateModelRuntimeError,
    validate_sourcing_adapter_metadata,
)
from leadpoet_canonical.attested_v2 import sha256_json
import research_lab.sourcing_model_contract_check as compatibility
from tests.test_private_artifact_signature import artifact_mapping
from tests.test_sourcing_model_contract import _conforming_tree


def _write(root: Path, relative: str, body: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def _module_ast_hash(path: Path) -> str:
    tree = ast.parse(path.read_bytes())
    payload = ast.dump(
        tree,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ready_adapter_metadata() -> dict:
    routing_catalog = {"schema_version": 1}
    routing_policy = {"schema_version": 1}
    runtime_catalog = {
        "schema_version": 1,
        "tools": [
            {"tool_id": tool_id}
            for tool_id in (
                "candidate.backlog",
                "candidate.registry_feed",
                "candidate.jobs_feed",
                "candidate.deepline_firmographic",
                "candidate.model_semantic",
                "intent.existing_evidence",
                "intent.jobs_feed",
                "intent.company_search",
                "intent.first_party",
                "intent.newsroom",
            )
        ],
    }
    runtime_policy = {"schema_version": 1}
    return {
        "adapter_version": "sourcing-model-research-lab-adapter:v3",
        "scoring_adapter_version": "qualification-company-scorer:v1",
        "component_registry_version": "sourcing-model-components:v2",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "runtime_capabilities": [
            "deadline",
            "emit",
            "http_fetch",
            "probe_origin",
            "resolve_host",
        ],
        "resilience_policy_version": "sourcing-model-resilience:v1",
        "firmographic_discovery": {
            "firmographic_policy_version": "sourcing-model-firmographic-discovery:v1"
        },
        "industry_taxonomy": {"taxonomy_content_hash": "sha256:" + "d" * 64},
        "routing": {
            "compiler_version": "routing-compiler-v2",
            "catalog": routing_catalog,
            "catalog_sha256": sha256_json(routing_catalog).removeprefix("sha256:"),
            "policy": routing_policy,
            "policy_sha256": sha256_json(routing_policy).removeprefix("sha256:"),
            "intent_sources": ["company_site", "job_listing", "news"],
            "source_add_requires_manifest_sha256": True,
            "private_bindings_exposed": False,
        },
        "runtime_routing": {
            "compiler_version": "routing-compiler-v2",
            "catalog": runtime_catalog,
            "catalog_sha256": sha256_json(runtime_catalog).removeprefix("sha256:"),
            "policy": runtime_policy,
            "policy_sha256": sha256_json(runtime_policy).removeprefix("sha256:"),
            "candidate_tool_lanes": {
                "candidate.backlog": "backlog",
                "candidate.registry_feed": "registry_signal",
                "candidate.jobs_feed": "jobs_signal",
                "candidate.deepline_firmographic": "deepline_firmographic",
                "candidate.model_semantic": "model_semantic",
            },
            "intent_tool_tiers": {
                "intent.existing_evidence": "fused",
                "intent.jobs_feed": "jobs_feed",
                "intent.company_search": "company_search",
                "intent.first_party": "first_party",
                "intent.newsroom": "newsroom",
            },
            "private_bindings_exposed": False,
        },
        "component_registry": {
            "source_router": {
                "strategy_options": ["company_site", "job_listing", "news"],
            }
        },
    }


def _write_future_source(root: Path) -> None:
    _write(
        root,
        "sourcing_model/__init__.py",
        "from sourcing_model.core import qualify, MAX_LEADS",
    )
    _write(
        root,
        "sourcing_model/validation.py",
        '''
VALIDATION = True

def first_party_industry_run():
    class _Run:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    return _Run()

def qualification_reason_contract_identity():
    return {}
''',
    )
    _write(
        root,
        "qualification/scoring/company_fit_decision.py",
        '''
COMPANY_FIT_CONDITIONAL_DIMENSIONS = ["stage"]
COMPANY_FIT_DECISIONS = ["match", "mismatch", "unavailable"]
COMPANY_FIT_DECISION_CONTRACT_ID = "company-fit-decision:v1"
COMPANY_FIT_DECISION_PRECEDENCE = ["mismatch", "unavailable", "match"]
COMPANY_FIT_DIMENSIONS = ["identity", "employee_size", "industry", "geography", "stage"]
COMPANY_FIT_PASSING_OUTCOME = "match"
COMPANY_FIT_REQUIRED_DIMENSIONS = ["identity", "employee_size", "industry", "geography"]

def aggregate_company_fit_decisions(decisions, *, stage_required=False):
    values = list(decisions.values())
    if "mismatch" in values:
        return "mismatch"
    allowed = set(COMPANY_FIT_DIMENSIONS)
    required = set(COMPANY_FIT_REQUIRED_DIMENSIONS)
    if stage_required:
        required.add("stage")
    if set(decisions) - allowed or required - set(decisions):
        return "unavailable"
    return reconcile_company_fit_decisions(
        [decisions[name] for name in required]
    )

def company_fit_decision_contract_identity():
    return {
        "contract_id": COMPANY_FIT_DECISION_CONTRACT_ID,
        "outcomes": COMPANY_FIT_DECISIONS,
        "precedence": COMPANY_FIT_DECISION_PRECEDENCE,
        "passing_outcome": COMPANY_FIT_PASSING_OUTCOME,
        "required_dimensions": COMPANY_FIT_REQUIRED_DIMENSIONS,
        "conditional_dimensions": COMPANY_FIT_CONDITIONAL_DIMENSIONS,
    }

def reconcile_company_fit_decisions(decisions):
    if "mismatch" in decisions:
        return "mismatch"
    if not decisions or any(item != "match" for item in decisions):
        return "unavailable"
    return "match"

def strict_company_fit_boolean(value):
    return value if isinstance(value, bool) else None
''',
    )
    _write(
        root,
        "research_lab_adapter.py",
        '''
from sourcing_model import qualify
import sourcing_model.core as core
import sourcing_model.orchestrator as orchestrator
import sourcing_model.runtime_capabilities as runtime_capabilities
import sourcing_model.validation as validation
from qualification.scoring.company_fit_decision import company_fit_decision_contract_identity
from sourcing_model.routing.defaults import routing_metadata
from sourcing_model.routing.runtime import runtime_routing_metadata

ADAPTER_VERSION = "sourcing-model-research-lab-adapter:v7"
COMPONENT_REGISTRY_VERSION = "sourcing-model-components:v2"
RESEARCH_LAB_ICP_PROJECTION_SCHEMA_VERSION = "research-lab-icp-projection:v1"
RESEARCH_LAB_SIGNAL_PROFILE_RECEIPT_AUDIENCE = "private_evaluator_only"
RESEARCH_LAB_SIGNAL_PROFILE_RECEIPT_SCHEMA_VERSION = "research-lab-signal-profile-receipt:v1"
SCORING_ADAPTER_VERSION = "qualification-company-scorer:v1"
COMPONENT_REGISTRY = {}

def adapter_metadata():
    return {
        "adapter_version": ADAPTER_VERSION,
        "capability_contract_version": runtime_capabilities.CAPABILITY_CONTRACT_VERSION,
        "company_fit_decision": company_fit_decision_contract_identity(),
        "component_registry": COMPONENT_REGISTRY,
        "component_registry_version": COMPONENT_REGISTRY_VERSION,
        "routing": routing_metadata(),
        "runtime_capabilities": list(runtime_capabilities.capability_metadata()["capabilities"]),
        "runtime_routing": runtime_routing_metadata(),
        "scoring_adapter_version": SCORING_ADAPTER_VERSION,
    }

def run_icp(icp, context=None):
    return qualify(icp)
''',
    )
    _write(
        root,
        "sourcing_model/clients.py",
        '''
import urllib.request

def _exa_call(body):
    return None

def agent_get(url):
    return None

def agent_post(url, body):
    return None

def exa_search(body):
    return None

def has_keys():
    return False

def sd_company(slug):
    return None

def sd_scrape(url, dynamic=True):
    return None
''',
    )
    _write(
        root,
        "sourcing_model/core.py",
        '''
_GOAL_MAX_COMPANIES = 50
_GOAL_MAX_ROUNDS = 8
MAX_LEADS = 5

def _initialize_runtime():
    return "ready"

IMPORT_STATE = _initialize_runtime()

async def _qualify_async(icp, _progress=None):
    return []

def qualify(icp):
    return []
''',
    )
    _write(
        root,
        "sourcing_model/discovery.py",
        '''
def build_query(icp, source):
    return str(source)
''',
    )
    _write(
        root,
        "sourcing_model/orchestrator.py",
        '''
def flow_mode():
    return "branch"

def plan_branches(icp, *, max_companies):
    return []

def reset_run_state():
    return None

def run_branches(icp, qualify_fn, *, max_companies):
    return []
''',
    )
    _write(
        root,
        "sourcing_model/routing/compiler.py",
        '''
COMPILER_VERSION = "routing-compiler-v3"

def compile_route(catalog, policy, context):
    return None

def eligible_route_candidates(catalog, policy, context):
    return []
''',
    )
    _write(
        root,
        "sourcing_model/routing/defaults.py",
        '''
from .compiler import COMPILER_VERSION

def routing_metadata():
    return {
        "compiler_version": COMPILER_VERSION,
        "private_bindings_exposed": False,
        "source_add_requires_manifest_sha256": True,
    }
''',
    )
    _write(
        root,
        "sourcing_model/routing/runtime.py",
        '''
from .compiler import COMPILER_VERSION

def runtime_routing_metadata():
    return {
        "compiler_version": COMPILER_VERSION,
        "private_bindings_exposed": False,
    }
''',
    )
    _write(
        root,
        "sourcing_model/runtime_capabilities.py",
        '''
from enum import Enum

CAPABILITY_CONTRACT_VERSION = "sourcing-model-runtime-capabilities:v2"

class HostResolution(str, Enum):
    RESOLVED = "resolved"
    NXDOMAIN = "nxdomain"
    TIMEOUT = "timeout"
    INVALID = "invalid"

class OriginReachability(str, Enum):
    REACHABLE = "reachable"
    UNKNOWN = "unknown"

def default_deadline():
    return None

def default_emit(event):
    return None

def default_http_fetch(url, *, timeout=10.0, max_bytes=500000, accept=None):
    return None

def default_probe_origin(host):
    return OriginReachability.UNKNOWN

def default_resolve_host(name):
    return HostResolution.INVALID if not str(name or "").strip() else HostResolution.TIMEOUT

_DEFAULTS = {
    "deadline": default_deadline,
    "emit": default_emit,
    "http_fetch": default_http_fetch,
    "probe_origin": default_probe_origin,
    "resolve_host": default_resolve_host,
}
_registered = {}

class UnknownCapabilityError(KeyError):
    pass

def capability_metadata():
    return {
        "capability_contract_version": CAPABILITY_CONTRACT_VERSION,
        "capabilities": tuple(sorted(_DEFAULTS)),
        "host_registered": registered_capabilities(),
    }

def capability(name):
    if name not in _DEFAULTS:
        raise UnknownCapabilityError(name)
    return _registered.get(name) or _DEFAULTS[name]

def deadline():
    value = capability("deadline")()
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value

def emit(event):
    return capability("emit")(event)

def http_fetch(url, *, timeout=10.0, max_bytes=500000, accept=None):
    return capability("http_fetch")(
        url, timeout=timeout, max_bytes=max_bytes, accept=accept
    )

def is_terminally_unresolvable(resolution):
    return resolution is HostResolution.NXDOMAIN

def may_attempt(reachability):
    return reachability in (OriginReachability.REACHABLE, OriginReachability.UNKNOWN)

def probe_origin(host):
    return capability("probe_origin")(host)

def register(name, implementation):
    if name not in _DEFAULTS:
        raise UnknownCapabilityError(name)
    if not callable(implementation):
        raise TypeError(name)
    _registered[name] = implementation

def registered_capabilities():
    return tuple(sorted(_registered))

def reset():
    _registered.clear()

def resolve_host(name):
    return capability("resolve_host")(name)
''',
    )
    # These are the additive v47-style functions: they have no positional
    # parameters and a separately declared required keyword-only surface.
    _write(
        root,
        "sourcing_model/intent_evidence_outcome.py",
        '''
def project_intent_evidence_outcome(*, evidence, policy):
    return {}

def project_intent_stage3_admission(*, outcome, policy):
    return {}
''',
    )


def _install_future_tree(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    write_source: bool = True,
) -> tuple[dict, dict[str, str]]:
    if write_source:
        _write_future_source(root)
    policy = deepcopy(compatibility.semantic_compatibility_policy_v1())
    for relative, specification in policy["critical_binding_slices"].items():
        observed, violations = compatibility._critical_binding_slice_v1(
            ast.parse((root / relative).read_bytes()),
            roots=list(specification["roots"]),
            normalized_literals=set(
                (policy.get("opaque_constants") or {}).get(relative) or {}
            ),
        )
        assert not violations
        specification["sha256"] = observed
    for relative in policy["import_time_binding_slices"]:
        observed, violations = compatibility._critical_binding_slice_v1(
            ast.parse((root / relative).read_bytes()),
            roots=list(policy["callables"][relative]),
            normalized_literals=set(
                (policy.get("opaque_constants") or {}).get(relative) or {}
            ),
            strip_function_bodies=True,
        )
        assert not violations
        policy["import_time_binding_slices"][relative] = observed
    policy_hash = {"value": "sha256:" + "9" * 64}
    monkeypatch.setattr(
        compatibility,
        "semantic_compatibility_policy_v1",
        lambda: deepcopy(policy),
    )
    monkeypatch.setattr(
        compatibility,
        "semantic_compatibility_policy_hash_v1",
        lambda: policy_hash["value"],
    )

    functions: dict[str, dict[str, list[str]]] = {}
    required_keyword_only: dict[str, list[str]] = {}
    frozen_asyncness: dict[str, bool] = {}
    contract_declared = {
        ("research_lab_adapter.py", "adapter_metadata"),
        ("research_lab_adapter.py", "run_icp"),
        ("sourcing_model/core.py", "qualify"),
        ("sourcing_model/orchestrator.py", "plan_branches"),
        ("sourcing_model/orchestrator.py", "run_branches"),
        ("sourcing_model/routing/compiler.py", "compile_route"),
        ("sourcing_model/routing/compiler.py", "eligible_route_candidates"),
        ("sourcing_model/routing/defaults.py", "routing_metadata"),
        ("sourcing_model/routing/runtime.py", "runtime_routing_metadata"),
        ("sourcing_model/runtime_capabilities.py", "capability_metadata"),
        ("sourcing_model/runtime_capabilities.py", "deadline"),
        ("sourcing_model/runtime_capabilities.py", "register"),
    }
    for relative, callables in policy["callables"].items():
        for name, expected in callables.items():
            if (relative, name) not in contract_declared:
                continue
            functions.setdefault(relative, {})[name] = list(
                expected["positional"]
            )
            key = f"{relative}:{name}"
            required_keyword_only[key] = list(
                expected["required_keyword_only"]
            )
            frozen_asyncness[key] = bool(expected["is_async"])
    outcome_relative = "sourcing_model/intent_evidence_outcome.py"
    functions[outcome_relative] = {
        "project_intent_evidence_outcome": [],
        "project_intent_stage3_admission": [],
    }
    required_keyword_only.update(
        {
            f"{outcome_relative}:project_intent_evidence_outcome": [
                "evidence",
                "policy",
            ],
            f"{outcome_relative}:project_intent_stage3_admission": [
                "outcome",
                "policy",
            ],
        }
    )
    contract = {
        "schema_version": 1,
        "contract_id": "leadpoet-sourcing-wrapper-contract-v47",
        "canonical_path": policy["canonical_contract_path"],
        "parity_fixture_path": policy["canonical_parity_path"],
        "required_files": [
            policy["canonical_contract_path"],
            policy["canonical_parity_path"],
            *policy["required_files"],
            outcome_relative,
        ],
        "functions": functions,
        "full_parameters": {},
        "required_keyword_only": required_keyword_only,
        "frozen_asyncness": frozen_asyncness,
    }
    _write(
        root,
        policy["canonical_contract_path"],
        json.dumps(contract, sort_keys=True),
    )
    _write(
        root,
        policy["canonical_parity_path"],
        json.dumps(
            {
                "schema_version": 1,
                "intent_evidence_outcome_parity_cases": [],
                "expected_intent_evidence_outcome_projections": [],
            },
            sort_keys=True,
        ),
    )
    return policy, policy_hash


def _manifest(root: Path, *, source_hash: str = "") -> dict:
    contract_path = root / "sourcing_model/consumer_contract.json"
    parity_path = root / "sourcing_model/consumer_parity_fixtures.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        "compatibility_contract": {
            "contract_id": contract["contract_id"],
            "path": "sourcing_model/consumer_contract.json",
            "sha256": compatibility._snapshot_sha256(contract_path),
        },
        "consumer_parity_fixtures": {
            "path": "sourcing_model/consumer_parity_fixtures.json",
            "sha256": compatibility._snapshot_sha256(parity_path),
        },
        "model_artifact_hash": source_hash or compute_private_source_tree_hash(root),
        "git_commit_sha": "a" * 40,
        "manifest_hash": "sha256:" + "b" * 64,
        "image_digest": "private.invalid/model@sha256:" + "c" * 64,
    }


def _admit(root: Path, manifest: dict | None = None) -> dict:
    document = manifest or _manifest(root)
    return compatibility.source_tree_compatibility_admission_v1(
        root,
        manifest=document,
        source_tree_hash=str(document["model_artifact_hash"]),
        use_cache=True,
    )


@pytest.mark.parametrize(
    "consumer_api_version",
    (None, "research-lab-consumer-api:v2"),
)
def test_unknown_consumer_api_policy_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer_api_version: str | None,
) -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    if consumer_api_version is None:
        policy.pop("consumer_api_version")
    else:
        policy["consumer_api_version"] = consumer_api_version
    policy_path = tmp_path / "consumer-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setattr(
        compatibility,
        "SEMANTIC_COMPATIBILITY_POLICY_V1_PATH",
        policy_path,
    )

    with pytest.raises(ValueError, match="consumer API is unsupported"):
        compatibility.semantic_compatibility_policy_v1()


def test_consumer_policy_identity_rejects_hash_toctou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    hashes = iter(("sha256:" + "1" * 64, "sha256:" + "2" * 64))
    monkeypatch.setattr(
        compatibility,
        "semantic_compatibility_policy_v1",
        lambda: deepcopy(policy),
    )
    monkeypatch.setattr(
        compatibility,
        "semantic_compatibility_policy_hash_v1",
        lambda: next(hashes),
    )

    with pytest.raises(ValueError, match="changed during admission"):
        compatibility.semantic_compatibility_policy_identity_v1()


def test_consumer_policy_hash_binds_reviewed_legacy_release_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = compatibility.semantic_compatibility_policy_hash_v1()
    snapshot = compatibility.reviewed_consumer_snapshots()[
        "leadpoet-sourcing-wrapper-contract-v7"
    ]
    release = snapshot["release_identities"][0]
    policy = compatibility.semantic_compatibility_policy_v1()
    receipt = compatibility._semantic_compatibility_receipt(
        mode="legacy_exact",
        consumer_api_version=policy["consumer_api_version"],
        policy_hash=before,
        source_tree_hash=release["source_tree_hash"],
        manifest={},
        contract=snapshot["contract"],
        contract_hash=snapshot["contract_sha256"],
        parity_hash=snapshot["parity_sha256"],
        bindings={},
    )
    specs = deepcopy(compatibility.REVIEWED_CONSUMER_SNAPSHOT_SPECS)
    for spec in specs:
        if spec["contract_id"] != "leadpoet-sourcing-wrapper-contract-v7":
            continue
        changed_release = dict(spec["release_identities"][0])
        changed_release["image_digest"] = (
            "private.invalid/model@sha256:" + "f" * 64
        )
        spec["release_identities"] = (changed_release,)
        break
    monkeypatch.setattr(
        compatibility,
        "REVIEWED_CONSUMER_SNAPSHOT_SPECS",
        tuple(specs),
    )

    assert compatibility.semantic_compatibility_policy_hash_v1() != before
    with pytest.raises(ValueError, match="differs from signed artifact"):
        compatibility.validate_source_tree_compatibility_receipt_v1(
            receipt,
            manifest={},
            source_tree_hash=release["source_tree_hash"],
        )


def test_research_lab_files_remain_bound_to_canonical_source_identity(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "model.py", "VALUE = 1")
    before = compute_private_source_tree_hash(tmp_path)
    _write(tmp_path, ".research_lab/config.json", '{"model_readable": true}')

    assert compute_private_source_tree_hash(tmp_path) != before


def test_synthetic_legacy_rehearsal_fixture_cannot_claim_signed_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).parent / "restart_rehearsal")
    )
    from production_workflow_runner import (
        _exact_legacy_model_source_fixture_files,
    )

    files = _exact_legacy_model_source_fixture_files(
        contract_path=compatibility.CONTRACT_V7_PATH,
        parity_path=compatibility.PARITY_FIXTURE_V7_PATH,
        extra_files={
            "rehearsal_private_model.py": (
                b"def run_icp(icp, context):\n    return []\n"
            )
        },
    )
    for relative, body in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    source_hash = compute_private_source_tree_hash(tmp_path)

    assert compatibility.verify_source_tree_contract(tmp_path) == []
    assert compatibility.resolve_reviewed_consumer_snapshot(tmp_path) is None
    with pytest.raises(ValueError, match="compatibility admission failed"):
        compatibility.source_tree_compatibility_admission_v1(
            tmp_path,
            source_tree_hash=source_hash,
        )


def test_v47_shape_auto_admits_keyword_only_model_owned_additions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract = json.loads(
        (tmp_path / "sourcing_model/consumer_contract.json").read_text(
            encoding="utf-8"
        )
    )

    receipt = _admit(tmp_path)

    assert "consumer_api_version" not in contract
    assert "exact_signatures" not in contract
    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["consumer_api_version"] == "research-lab-consumer-api:v1"
    assert receipt["decision"] == "accepted"
    assert receipt["contract_id"].endswith("v47")
    assert receipt["bindings"] == {
        "adapter_version": "sourcing-model-research-lab-adapter:v7",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "component_registry_version": "sourcing-model-components:v2",
        "routing_compiler_version": "routing-compiler-v3",
        "scoring_adapter_version": "qualification-company-scorer:v1",
    }


@pytest.mark.parametrize(
    "marker",
    (
        "research_lab_adapter.py:dispatch_runner_initial_custody_v3",
        "sourcing_model/model_runner.py:model_runner_custody_metadata",
    ),
)
def test_one_custody_v3_marker_does_not_reclassify_legacy_contract(
    marker: str,
) -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    contract = {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v47",
        "exact_signatures": [marker],
    }

    assert not compatibility._typed_dispatch_custody_v3_requested(
        contract,
        policy=policy,
    )


def test_both_custody_v3_markers_make_unknown_identity_fail_closed() -> None:
    policy = compatibility.semantic_compatibility_policy_v1()
    contract = {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v999",
        "exact_signatures": [
            "research_lab_adapter.py:dispatch_runner_initial_custody_v3",
            "sourcing_model/model_runner.py:model_runner_custody_metadata",
        ],
    }

    assert compatibility._typed_dispatch_custody_v3_requested(
        contract,
        policy=policy,
    )


def test_v10_source_cannot_use_a_legacy_contract_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "sourcing-model-research-lab-adapter:v7",
            "sourcing-model-research-lab-adapter:v10",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="typed dispatch contract identity is not approved",
    ):
        _admit(tmp_path, _manifest(tmp_path))


def test_candidate_build_gate_auto_admits_future_surface_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.research_lab import code_build

    _install_future_tree(tmp_path, monkeypatch)

    admitted = code_build._sourcing_contract_gate(
        tmp_path,
        force_enforce=True,
    )

    assert admitted is not None
    source_tree_hash, receipt = admitted
    assert source_tree_hash == compute_private_source_tree_hash(tmp_path)
    assert receipt["admission_mode"] == "semantic_v1"

    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "return qualify(icp)",
            "return []",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        code_build.CodeEditPrivateTestError,
        match="hard module semantic drift",
    ):
        code_build._sourcing_contract_gate(
            tmp_path,
            force_enforce=True,
        )


def test_contract_id_is_release_identity_not_a_compatibility_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["contract_id"] = "leadpoet-sourcing-wrapper-contract-v999"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    receipt = _admit(tmp_path, _manifest(tmp_path))

    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["contract_id"] == contract["contract_id"]


def test_known_contract_with_modified_parity_uses_source_backed_semantic_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    parity_path = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    contract_path.write_bytes(compatibility.CONTRACT_V46_PATH.read_bytes())
    parity_path.write_bytes(
        compatibility.PARITY_FIXTURE_V46_PATH.read_bytes() + b"\n"
    )
    manifest = _manifest(tmp_path)

    receipt = _admit(tmp_path, manifest)

    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["contract_hash"] == manifest["compatibility_contract"]["sha256"]
    assert receipt["parity_hash"] == manifest["consumer_parity_fixtures"]["sha256"]

    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "return qualify(icp)",
            "return []",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_release_labels_and_additive_contract_revision_auto_admit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "sourcing-model-research-lab-adapter:v7",
            "sourcing-model-research-lab-adapter:v8",
        ),
        encoding="utf-8",
    )
    compiler = tmp_path / "sourcing_model/routing/compiler.py"
    compiler.write_text(
        compiler.read_text(encoding="utf-8").replace(
            "routing-compiler-v3",
            "routing-compiler-v4",
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["contract_id"] = "leadpoet-sourcing-wrapper-contract-v999"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    receipt = _admit(tmp_path, _manifest(tmp_path))

    assert receipt["bindings"]["adapter_version"].endswith(":v8")
    assert receipt["bindings"]["routing_compiler_version"] == "routing-compiler-v4"
    assert receipt["contract_id"].endswith("v999")


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    (
        (
            "research_lab_adapter.py",
            'ADAPTER_VERSION = "sourcing-model-research-lab-adapter:v7"',
            'ADAPTER_VERSION = "sourcing-model-research-lab-adapter:v8!"',
        ),
        (
            "research_lab_adapter.py",
            'ADAPTER_VERSION = "sourcing-model-research-lab-adapter:v7"',
            "ADAPTER_VERSION = object()",
        ),
        (
            "sourcing_model/routing/compiler.py",
            'COMPILER_VERSION = "routing-compiler-v3"',
            "COMPILER_VERSION = object()",
        ),
    ),
)
def test_release_labels_must_remain_unique_supported_literals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    old: str,
    new: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard release label drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_release_label_rebinding_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8")
        + '\nADAPTER_VERSION = "sourcing-model-research-lab-adapter:v9"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard release label drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_unseen_contract_explicit_supported_consumer_api_auto_admits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["consumer_api_version"] = "research-lab-consumer-api:v1"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    receipt = _admit(tmp_path, _manifest(tmp_path))

    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["consumer_api_version"] == contract["consumer_api_version"]


@pytest.mark.parametrize(
    "declared_consumer_api",
    ("research-lab-consumer-api:v2", 1, False),
)
def test_unseen_contract_explicit_unknown_consumer_api_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_consumer_api: object,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["consumer_api_version"] = declared_consumer_api
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported model consumer API"):
        _admit(tmp_path, _manifest(tmp_path))


def test_host_admission_rechecks_semantic_source_against_signed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "future-source"
    _install_future_tree(source, monkeypatch)
    artifact = PrivateModelArtifactManifest.from_mapping(
        build_local_private_artifact_manifest(
            source_path=source,
            git_commit_sha="d" * 40,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "e" * 64
            ),
            manifest_uri="s3://private/manifests/current.json",
            signature_ref="kms:signature",
            component_registry_version="sourcing-model-components:v2",
            scoring_adapter_version="qualification-company-scorer:v1",
        )
    )

    def extract_source(*, image_digest, source_dir, timeout_seconds):
        assert image_digest == artifact.image_digest
        assert timeout_seconds >= 120
        shutil.copytree(source, source_dir)
        return compute_private_source_tree_hash(source_dir), []

    monkeypatch.setattr(
        model_authority_v2,
        "_extract_parent_image_source",
        extract_source,
    )
    model_authority_v2._SOURCE_BUNDLE_CACHE.clear()
    model_authority_v2._SOURCE_COMPATIBILITY_RECEIPT_CACHE.clear()

    receipt = model_authority_v2.private_model_compatibility_receipt_v2(
        artifact,
        timeout_seconds=120,
    )

    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["consumer_api_version"] == "research-lab-consumer-api:v1"
    assert receipt["decision"] == "accepted"
    assert receipt["source_tree_hash"] == artifact.model_artifact_hash
    assert receipt["manifest_hash"] == artifact.manifest_hash
    assert receipt["image_digest"] == artifact.image_digest
    assert all(
        key[-2:]
        == (
            "research-lab-consumer-api:v1",
            compatibility.SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        )
        for key in model_authority_v2._SOURCE_BUNDLE_CACHE
    )


@pytest.mark.parametrize("mutation", ("schema_major", "capability_expansion"))
def test_unknown_schema_or_expanded_capability_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    if mutation == "schema_major":
        path = tmp_path / "sourcing_model/consumer_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        contract["schema_version"] = 2
        _write(tmp_path, "sourcing_model/consumer_contract.json", json.dumps(contract))
    else:
        path = tmp_path / "sourcing_model/runtime_capabilities.py"
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                '_DEFAULTS = {\n',
                '_DEFAULTS = {\n    "shell": object,\n',
                1,
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="compatibility admission failed") as captured:
        _admit(tmp_path, _manifest(tmp_path))

    if mutation == "schema_major":
        assert "unsupported model compatibility contract schema major" in str(
            captured.value
        )
    else:
        assert "hard module semantic drift" in str(captured.value)


def test_consumer_required_import_target_must_exist_as_regular_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    (tmp_path / "sourcing_model/validation.py").unlink()

    with pytest.raises(ValueError, match="missing required file"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize("mutation", ("redirect", "rebind"))
def test_package_qualify_export_must_remain_bound_to_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    package = tmp_path / "sourcing_model/__init__.py"
    body = package.read_text(encoding="utf-8")
    if mutation == "redirect":
        body = body.replace(
            "from sourcing_model.core import qualify, MAX_LEADS",
            (
                "from sourcing_model.core import MAX_LEADS\n"
                "from sourcing_model.discovery import build_query as qualify"
            ),
        )
    else:
        body += "\nqualify = lambda icp: []\n"
    package.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match="hard import binding drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_package_unrelated_export_remains_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    package = tmp_path / "sourcing_model/__init__.py"
    package.write_text(
        package.read_text(encoding="utf-8")
        + "\nfrom sourcing_model.core import _qualify_async as qualify_async\n",
        encoding="utf-8",
    )

    assert _admit(tmp_path, _manifest(tmp_path))["admission_mode"] == "semantic_v1"


@pytest.mark.parametrize(
    ("relative", "original", "replacement", "callable_name"),
    (
        (
            "sourcing_model/discovery.py",
            "def build_query(icp, source):",
            "def renamed_build_query(icp, source):",
            "build_query",
        ),
        (
            "sourcing_model/orchestrator.py",
            "def flow_mode():",
            "def flow_mode(mode):",
            "flow_mode",
        ),
        (
            "sourcing_model/validation.py",
            "def first_party_industry_run():",
            "def first_party_industry_run(mode):",
            "first_party_industry_run",
        ),
        (
            "sourcing_model/validation.py",
            "def qualification_reason_contract_identity():",
            "def qualification_reason_contract_identity(version):",
            "qualification_reason_contract_identity",
        ),
    ),
)
def test_adapter_dependency_abi_drift_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    original: str,
    replacement: str,
    callable_name: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compatibility admission failed") as raised:
        _admit(tmp_path, _manifest(tmp_path))
    assert callable_name in str(raised.value)


def _observe_future_runtime_dependencies(
    root: Path,
    compatibility_receipt: dict,
    *,
    execution_job_id: str,
) -> dict:
    observation_plan = model_sandbox_v2._runtime_probe_observation_plan_v1(
        compatibility_receipt,
        execution_job_id=execution_job_id,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            model_sandbox_v2._MEASURED_METADATA_BOOTSTRAP,
            "research_lab_adapter",
            "adapter_metadata",
        ],
        input=json.dumps(
            {
                "observation_plan": observation_plan,
            }
        ),
        text=True,
        capture_output=True,
        cwd=root,
        env={
            "HOME": str(root),
            "PATH": "",
            "LEADPOET_MODEL_SOURCE_ROOT": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert set(observed["runtime_observation"]) == {
        "invariants",
        "qualification_outcome_protocol",
    }
    assert observed["runtime_observation"]["qualification_outcome_protocol"] is None
    return {**observed, "observation_plan": observation_plan}


def test_adapter_dependency_runtime_protocols_match_consumer_expectations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    manifest = _manifest(tmp_path)
    receipt = _admit(tmp_path, manifest)
    observed = _observe_future_runtime_dependencies(
        tmp_path,
        receipt,
        execution_job_id="scoring-v2:run-model-sandbox-v2:" + "1" * 32,
    )

    probe = model_sandbox_v2._build_consumer_runtime_probe_from_observation_v1(
        observed["runtime_observation"],
        compatibility_receipt=receipt,
        metadata=observed["metadata"],
        expected_source_tree_hash=manifest["model_artifact_hash"],
        expected_manifest_hash=manifest["manifest_hash"],
        expected_image_digest=manifest["image_digest"],
        expected_module_name="research_lab_adapter",
        expected_callable_name="adapter_metadata",
        observation_plan=observed["observation_plan"],
    )

    assert probe["invariants"]["adapter_dependencies"] == {
        "build_query_returns_string": True,
        "first_party_industry_run_is_context_manager": True,
        "flow_mode_is_supported": True,
    }


@pytest.mark.parametrize(
    ("relative", "original", "replacement"),
    (
        ("sourcing_model/validation.py", "return _Run()", "return None"),
        ("sourcing_model/orchestrator.py", 'return "branch"', 'return "production"'),
        ("sourcing_model/discovery.py", "return str(source)", "return None"),
    ),
)
def test_adapter_dependency_runtime_protocol_drift_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    original: str,
    replacement: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path)
    receipt = _admit(tmp_path, manifest)
    observed = _observe_future_runtime_dependencies(
        tmp_path,
        receipt,
        execution_job_id="scoring-v2:run-model-sandbox-v2:" + "1" * 32,
    )

    with pytest.raises(
        model_sandbox_v2.ModelSandboxV2Error,
        match="^consumer runtime probe differs from host admission$",
    ):
        model_sandbox_v2._build_consumer_runtime_probe_from_observation_v1(
            observed["runtime_observation"],
            compatibility_receipt=receipt,
            metadata=observed["metadata"],
            expected_source_tree_hash=manifest["model_artifact_hash"],
            expected_manifest_hash=manifest["manifest_hash"],
            expected_image_digest=manifest["image_digest"],
            expected_module_name="research_lab_adapter",
            expected_callable_name="adapter_metadata",
            observation_plan=observed["observation_plan"],
        )


def test_adapter_dependency_bodies_and_unrelated_helpers_remain_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    mutations = {
        "sourcing_model/discovery.py": (
            "    return str(source)",
            "    return f'{source}:{icp!r}'",
        ),
        "sourcing_model/orchestrator.py": (
            '    return "branch"',
            '    return "legacy"',
        ),
        "sourcing_model/validation.py": (
            "            return False",
            "            return True",
        ),
    }
    for relative, (original, replacement) in mutations.items():
        path = tmp_path / relative
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                original,
                replacement,
                1,
            )
            + "\ndef unrelated_future_helper():\n    return 1\n",
            encoding="utf-8",
        )

    manifest = _manifest(tmp_path)
    receipt = _admit(tmp_path, manifest)
    assert receipt["admission_mode"] == "semantic_v1"
    observed = _observe_future_runtime_dependencies(
        tmp_path,
        receipt,
        execution_job_id="scoring-v2:run-model-sandbox-v2:" + "1" * 32,
    )
    model_sandbox_v2._build_consumer_runtime_probe_from_observation_v1(
        observed["runtime_observation"],
        compatibility_receipt=receipt,
        metadata=observed["metadata"],
        expected_source_tree_hash=manifest["model_artifact_hash"],
        expected_manifest_hash=manifest["manifest_hash"],
        expected_image_digest=manifest["image_digest"],
        expected_module_name="research_lab_adapter",
        expected_callable_name="adapter_metadata",
        observation_plan=observed["observation_plan"],
    )


def test_adapter_max_leads_boundary_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    core = tmp_path / "sourcing_model/core.py"
    core.write_text(
        core.read_text(encoding="utf-8").replace("MAX_LEADS = 5", "MAX_LEADS = 6"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact constant drift.*MAX_LEADS"):
        _admit(tmp_path, _manifest(tmp_path))


def test_model_owned_contract_cannot_self_authorize_hard_abi_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "def run_icp(icp, context=None):",
            "def run_icp(request, context=None):",
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["functions"]["research_lab_adapter.py"]["run_icp"] = [
        "request",
        "context",
    ]
    _write(tmp_path, "sourcing_model/consumer_contract.json", json.dumps(contract))

    with pytest.raises(ValueError) as captured:
        _admit(tmp_path, _manifest(tmp_path))

    assert "exact parameter drift research_lab_adapter.py:run_icp" in str(
        captured.value
    )


def test_adapter_entrypoint_body_is_consumer_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "    return qualify(icp)",
            "    return []",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    (
        "@identity\ndef run_icp(icp, context=None):",
        "def run_icp(icp, context=None):",
    ),
)
def test_hard_callable_decorators_and_rebinding_are_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    original = adapter.read_text(encoding="utf-8")
    if mutation.startswith("@"):
        original = original.replace(
            "def run_icp(icp, context=None):",
            mutation,
        )
    else:
        original += "\nrun_icp = lambda icp, context=None: []\n"
    adapter.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError) as captured:
        _admit(tmp_path, _manifest(tmp_path))

    message = str(captured.value).lower()
    if mutation.startswith("@"):
        assert "hard module semantic drift research_lab_adapter.py" in message
    else:
        assert "must have one definition, found 2" in message


@pytest.mark.parametrize(
    "injection",
    (
        'globals()["run_icp"] = lambda icp, context=None: []',
        'globals().update({"run_icp": lambda icp, context=None: []})',
        'vars()["run_icp"] = lambda icp, context=None: []',
        'exec("run_icp = lambda icp, context=None: []")',
        (
            "import gateway.tee.model_sandbox_v2 as trusted_probe\n"
            "trusted_probe.build_consumer_runtime_probe_v1 = "
            "lambda **kwargs: {}"
        ),
        "import research_lab.sourcing_model_contract_check as consumer_policy",
        (
            "import sys as shadow_sys\n"
            "shadow_sys.modules[__name__].run_icp = "
            "lambda icp, context=None: []"
        ),
        (
            "def replacement(icp, context=None):\n"
            "    return []\n"
            "run_icp.__code__ = replacement.__code__"
        ),
        (
            "getattr(run_icp, '__' + 'globals__')['run_icp'] = "
            "lambda icp, context=None: []"
        ),
        (
            "run_icp.__getattribute__('__' + 'globals__')['run_icp'] = "
            "lambda icp, context=None: []"
        ),
        (
            "aliased_entrypoint = run_icp\n"
            "getattr(aliased_entrypoint, '__' + 'globals__')['run_icp'] = "
            "lambda icp, context=None: []"
        ),
        (
            "def poison(_function):\n"
            "    global run_icp\n"
            "    run_icp = lambda icp, context=None: []\n"
            "    return _function\n"
            "@poison\n"
            "def trigger():\n"
            "    return None"
        ),
        (
            "def poison_default():\n"
            "    global run_icp\n"
            "    run_icp = lambda icp, context=None: []\n"
            "def trigger(value=poison_default()):\n"
            "    return value"
        ),
    ),
)
def test_reflective_and_evaluation_time_hard_abi_rebinding_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8") + "\n" + injection + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as captured:
        _admit(tmp_path, _manifest(tmp_path))

    message = str(captured.value).lower()
    assert "hard module semantic drift research_lab_adapter.py" in message


@pytest.mark.parametrize(
    "injection",
    (
        'import sys\ngetattr(sys, "mod" + "ules")',
        (
            'import sys\ngetattr('
            'getattr(sys, "_get" + "frame")(), "f_" + "globals")'
        ),
        'import sys\ngetattr(sys, "".join(["mod", "ules"]))',
    ),
)
def test_constructed_sys_namespace_access_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8") + "\n" + injection + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize(
    "helper_source",
    (
        (
            "import research_lab_adapter as target\n"
            "target.run_icp = lambda icp, context=None: []\n"
        ),
        "import gateway.tee.model_sandbox_v2\n",
        "import research_lab.sourcing_model_contract_check\n",
        "import __main__\n",
        'import sys\ngetattr(sys, "mod" + "ules")\n',
    ),
)
def test_transitive_local_helper_cannot_mutate_consumer_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_source: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8") + "\nimport additive_helper\n",
        encoding="utf-8",
    )
    (tmp_path / "additive_helper.py").write_text(
        helper_source,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="hard module semantic drift",
    ):
        _admit(tmp_path, _manifest(tmp_path))


def test_unimported_additive_reflective_helper_is_outside_adapter_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    (tmp_path / "unrelated_helper.py").write_text(
        "import inspect\nUNRELATED = inspect.currentframe\n",
        encoding="utf-8",
    )

    assert _admit(tmp_path, _manifest(tmp_path))["admission_mode"] == "semantic_v1"


def test_unconsumed_inert_function_and_literal_are_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    capabilities = tmp_path / "sourcing_model/runtime_capabilities.py"
    capabilities.write_text(
        capabilities.read_text(encoding="utf-8")
        + (
            "\nresult = 1\n"
            "def unrelated_reflective_helper(value: str = 'ok') -> str:\n"
            "    import inspect\n"
            "    inspect.currentframe()\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )

    assert _admit(tmp_path, _manifest(tmp_path))["admission_mode"] == "semantic_v1"


def test_dunder_namespace_literal_is_not_harmless_additive_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    capabilities = tmp_path / "sourcing_model/runtime_capabilities.py"
    capabilities.write_text(
        capabilities.read_text(encoding="utf-8") + "\n__builtins__ = {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="module namespace|module semantic"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize(
    ("relative", "mutation"),
    (
        (
            "sourcing_model/clients.py",
            "\nfrom arbitrary import *\n",
        ),
        (
            "sourcing_model/core.py",
            "\ndef identity(function):\n    return function\n"
            "\nqualify = identity(qualify)\n",
        ),
        (
            "sourcing_model/routing/compiler.py",
            "\nglobals()['COMPILER_VERSION'] = 'routing-compiler-evil'\n",
        ),
    ),
)
def test_noncritical_abi_modules_cannot_dynamically_rebind_consumer_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    path = tmp_path / relative
    path.write_text(
        path.read_text(encoding="utf-8") + mutation,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard import-time semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_required_abi_callable_decorator_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    core = tmp_path / "sourcing_model/core.py"
    core.write_text(
        core.read_text(encoding="utf-8").replace(
            "def qualify(icp):",
            "@identity\ndef qualify(icp):",
        )
        + "\ndef identity(function):\n    return function\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard import-time semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    (
        (
            "\ndef poison():\n"
            "    global qualify\n"
            "    qualify = lambda icp: []\n"
            "poison()\n"
        ),
        (
            "\ndef poison(function):\n"
            "    global qualify\n"
            "    qualify = lambda icp: []\n"
            "    return function\n"
            "@poison\n"
            "def trigger():\n"
            "    return None\n"
        ),
        (
            "\ndef poison():\n"
            "    global qualify\n"
            "    qualify = lambda icp: []\n"
            "    return None\n"
            "def trigger(value=poison()):\n"
            "    return value\n"
        ),
        (
            "\ndef poison(cls):\n"
            "    global qualify\n"
            "    qualify = lambda icp: []\n"
            "    return cls\n"
            "@poison\n"
            "class Trigger:\n"
            "    pass\n"
        ),
    ),
)
def test_abi_import_time_indirection_cannot_rebind_required_callables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    core = tmp_path / "sourcing_model/core.py"
    core.write_text(core.read_text(encoding="utf-8") + mutation, encoding="utf-8")

    with pytest.raises(ValueError, match="hard import-time semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_model_owned_abi_function_body_change_remains_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    core = tmp_path / "sourcing_model/core.py"
    core.write_text(
        core.read_text(encoding="utf-8").replace(
            "def qualify(icp):\n    return []",
            "def qualify(icp):\n    return [icp]",
        ),
        encoding="utf-8",
    )

    assert _admit(tmp_path, _manifest(tmp_path))["admission_mode"] == "semantic_v1"


def test_import_time_called_helper_body_is_consumer_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    core = tmp_path / "sourcing_model/core.py"
    core.write_text(
        core.read_text(encoding="utf-8").replace(
            'def _initialize_runtime():\n    return "ready"',
            'def _initialize_runtime():\n    return "poisoned"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard import-time semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_module_level_dynamic_attribute_hook_is_not_additive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    clients = tmp_path / "sourcing_model/clients.py"
    clients.write_text(
        clients.read_text(encoding="utf-8")
        + "\ndef __getattr__(name):\n    return lambda *args, **kwargs: None\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard import-time semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


@pytest.mark.parametrize(
    "mutation",
    (
        "\nimport sys\nsys.stdout = object()\n",
        "\nimport sys\nsys.settrace(lambda *args: None)\n",
        (
            "\ndef _known_case_branch(decisions):\n"
            "    return 'match' if decisions else 'unavailable'\n"
            "reconcile_company_fit_decisions = _known_case_branch\n"
        ),
    ),
)
def test_hostile_runtime_observation_cannot_authorize_critical_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    critical = tmp_path / "qualification/scoring/company_fit_decision.py"
    critical.write_text(
        critical.read_text(encoding="utf-8") + mutation,
        encoding="utf-8",
    )
    parity = tmp_path / "sourcing_model/consumer_parity_fixtures.json"
    parity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "producer_claimed_runtime_probe": "accepted",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_required_import_cannot_be_rebound_within_one_import_statement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "import sourcing_model.core as core",
            (
                "import sourcing_model.core as core, "
                "sourcing_model.validation as core"
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard import binding drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_wildcard_import_cannot_rebind_hard_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8")
        + "\nfrom sourcing_model.validation import *\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_malformed_model_owned_abi_maps_are_quarantined_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract_path = tmp_path / "sourcing_model/consumer_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["required_keyword_only"] = ["self-authorized"]
    _write(tmp_path, "sourcing_model/consumer_contract.json", json.dumps(contract))

    with pytest.raises(ValueError, match="keyword-only declaration is invalid"):
        _admit(tmp_path, _manifest(tmp_path))


def test_manifest_hash_hybrids_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    mismatched = _manifest(tmp_path)
    mismatched["consumer_parity_fixtures"]["sha256"] = "sha256:" + "0" * 64
    mismatched["manifest_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="manifest parity fixtures differ"):
        _admit(tmp_path, mismatched)

    mismatched = _manifest(tmp_path)
    mismatched["compatibility_contract"]["contract_id"] = (
        "leadpoet-sourcing-wrapper-contract-v999"
    )
    mismatched["manifest_hash"] = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="manifest compatibility contract differs"):
        _admit(tmp_path, mismatched)

def test_admission_cache_and_receipts_bind_source_manifest_image_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _policy, policy_hash = _install_future_tree(tmp_path, monkeypatch)
    compatibility.clear_source_tree_compatibility_admission_cache_v1()
    base_manifest = _manifest(tmp_path)
    receipts = [_admit(tmp_path, base_manifest)]

    changed_manifest = deepcopy(base_manifest)
    changed_manifest["manifest_hash"] = "sha256:" + "d" * 64
    receipts.append(_admit(tmp_path, changed_manifest))

    changed_image = deepcopy(base_manifest)
    changed_image["image_digest"] = (
        "private.invalid/model@sha256:" + "e" * 64
    )
    receipts.append(_admit(tmp_path, changed_image))

    _write(tmp_path, "additive_release_note.txt", "future additive release")
    changed_source_manifest = _manifest(tmp_path)
    changed_source_manifest["manifest_hash"] = "sha256:" + "f" * 64
    receipts.append(_admit(tmp_path, changed_source_manifest))

    policy_hash["value"] = "sha256:" + "1" * 64
    receipts.append(_admit(tmp_path, changed_source_manifest))

    receipt = receipts[0]
    body = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_hash"
    }
    assert receipt["receipt_hash"] == compatibility._sha256_json(body)
    for field, value in body.items():
        mutated = deepcopy(body)
        if isinstance(value, dict):
            mutated[field] = {"mutated": True}
        elif isinstance(value, int):
            mutated[field] = value + 1
        else:
            mutated[field] = str(value) + ":mutated"
        assert compatibility._sha256_json(mutated) != receipt["receipt_hash"]

    assert len({item["receipt_hash"] for item in receipts}) == len(receipts)
    assert all(
        item["consumer_api_version"] == "research-lab-consumer-api:v1"
        and item["decision"] == "accepted"
        for item in receipts
    )
    assert all(
        key[-2:]
        == (
            "research-lab-consumer-api:v1",
            compatibility.SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        )
        for key in compatibility._SEMANTIC_COMPATIBILITY_CACHE
    )


def test_caller_hash_cannot_seed_or_reuse_compatibility_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    compatibility.clear_source_tree_compatibility_admission_cache_v1()
    manifest = _manifest(tmp_path)
    forged_hash = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="caller source tree hash differs"):
        compatibility.source_tree_compatibility_admission_v1(
            tmp_path,
            manifest=manifest,
            source_tree_hash=forged_hash,
            use_cache=True,
        )
    assert not compatibility._SEMANTIC_COMPATIBILITY_CACHE

    accepted = _admit(tmp_path, manifest)
    assert accepted["source_tree_hash"] == manifest["model_artifact_hash"]
    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "def run_icp(icp, context=None):",
            "def run_icp(request, context=None):",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical extraction"):
        compatibility.source_tree_compatibility_admission_v1(
            tmp_path,
            manifest=manifest,
            source_tree_hash=manifest["model_artifact_hash"],
            use_cache=True,
        )


@pytest.mark.parametrize(
    "manifest_field",
    ("compatibility_contract", "consumer_parity_fixtures"),
)
def test_cached_admission_revalidates_signed_contract_and_parity_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_field: str,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    compatibility.clear_source_tree_compatibility_admission_cache_v1()
    manifest = _manifest(tmp_path)
    _admit(tmp_path, manifest)
    mutated = deepcopy(manifest)
    mutated[manifest_field]["sha256"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="cached receipt differs"):
        _admit(tmp_path, mutated)


@pytest.mark.parametrize("seed_cache", (False, True))
def test_source_tree_toctou_is_rejected_before_receipt_or_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_cache: bool,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    compatibility.clear_source_tree_compatibility_admission_cache_v1()
    manifest = _manifest(tmp_path)
    if seed_cache:
        _admit(tmp_path, manifest)
    observed = manifest["model_artifact_hash"]
    hashes = iter((observed, "sha256:" + "0" * 64))
    monkeypatch.setattr(
        compatibility,
        "compute_compatibility_source_tree_hash_v1",
        lambda _root: next(hashes),
    )

    expected = "cached admission" if seed_cache else "changed during admission"
    with pytest.raises(ValueError, match=expected):
        compatibility.source_tree_compatibility_admission_v1(
            tmp_path,
            manifest=manifest,
            source_tree_hash=observed,
            use_cache=seed_cache,
        )


def test_signed_legacy_profiles_and_manifest_identities_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_root = tmp_path / "legacy"
    _conforming_tree(
        legacy_root,
        contract_snapshot=compatibility.CONTRACT_V7_PATH,
        parity_snapshot=compatibility.PARITY_FIXTURE_V7_PATH,
        runtime_version=6,
    )
    snapshot = compatibility._resolve_reviewed_consumer_contract_pair(legacy_root)
    assert snapshot is not None
    assert {
        contract_id
        for contract_id, value in compatibility.reviewed_consumer_snapshots().items()
        if value["release_identities"]
    } == {
        "leadpoet-sourcing-wrapper-contract-v7",
        "leadpoet-sourcing-wrapper-contract-v8",
        "leadpoet-sourcing-wrapper-contract-v11",
        "leadpoet-sourcing-wrapper-contract-v12",
        "leadpoet-sourcing-wrapper-contract-v13",
        "leadpoet-sourcing-wrapper-contract-v26",
        "leadpoet-sourcing-wrapper-contract-v52",
        "leadpoet-sourcing-wrapper-contract-v55",
    }
    assert sum(
        len(value["release_identities"])
        for value in compatibility.reviewed_consumer_snapshots().values()
    ) == 13
    release = snapshot["release_identities"][0]
    assert compatibility._reviewed_consumer_snapshot_for_source_hash(
        legacy_root,
        source_tree_hash=release["source_tree_hash"],
    ) == snapshot
    field_map = {
        "source_tree_hash": "model_artifact_hash",
        "git_commit_sha": "git_commit_sha",
        "manifest_hash": "manifest_hash",
        "image_digest": "image_digest",
    }
    for profiled in compatibility.reviewed_consumer_snapshots().values():
        for identity in profiled["release_identities"]:
            manifest = {
                target: identity[source]
                for source, target in field_map.items()
            }
            receipt = {
                "admission_mode": "legacy_exact",
                "contract_id": profiled["contract"]["contract_id"],
                "source_tree_hash": identity["source_tree_hash"],
                "contract_hash": profiled["contract_sha256"],
                "parity_hash": profiled["parity_sha256"],
            }
            compatibility.validate_reviewed_legacy_release_manifest_identity_v1(
                receipt,
                manifest,
            )
            if profiled["contract_sha256"] == snapshot["contract_sha256"]:
                assert compatibility._reviewed_consumer_snapshot_for_source_hash(
                    legacy_root,
                    source_tree_hash=identity["source_tree_hash"],
                    manifest=manifest,
                ) == snapshot
            for target in field_map.values():
                hybrid = deepcopy(manifest)
                hybrid[target] = str(hybrid[target]) + ":hybrid"
                with pytest.raises(ValueError, match="reviewed signed release"):
                    compatibility.validate_reviewed_legacy_release_manifest_identity_v1(
                        receipt,
                        hybrid,
                    )
                if profiled["contract_sha256"] == snapshot["contract_sha256"]:
                    with pytest.raises(ValueError, match="manifest identity differs"):
                        compatibility._reviewed_consumer_snapshot_for_source_hash(
                            legacy_root,
                            source_tree_hash=identity["source_tree_hash"],
                            manifest=hybrid,
                        )


def test_current_v55_e55_signed_release_is_exact_and_hybrid_fails() -> None:
    profiled = next(
        profile
        for profile in compatibility.reviewed_consumer_profiles()
        if profile["contract_sha256"]
        == "sha256:b89eda998cf8cf3d9ee80c4ccd2bd4e10e37d6e4bdd7be80e2dc70492d2c0ffd"
    )
    release = profiled["release_identities"][0]
    manifest = {
        "model_artifact_hash": release["source_tree_hash"],
        "git_commit_sha": release["git_commit_sha"],
        "manifest_hash": release["manifest_hash"],
        "image_digest": release["image_digest"],
    }
    receipt = {
        "admission_mode": "legacy_exact",
        "contract_id": profiled["contract"]["contract_id"],
        "source_tree_hash": release["source_tree_hash"],
        "contract_hash": profiled["contract_sha256"],
        "parity_hash": profiled["parity_sha256"],
    }

    compatibility.validate_reviewed_legacy_release_manifest_identity_v1(
        receipt,
        manifest,
    )
    for field in ("git_commit_sha", "manifest_hash", "image_digest"):
        hybrid = {**manifest, field: str(manifest[field]) + ":hybrid"}
        with pytest.raises(ValueError, match="reviewed signed release"):
            compatibility.validate_reviewed_legacy_release_manifest_identity_v1(
                receipt,
                hybrid,
            )


def test_legacy_receipt_identity_rejects_hybrid_or_relabelled_release() -> None:
    snapshot = compatibility.reviewed_consumer_snapshots()[
        "leadpoet-sourcing-wrapper-contract-v7"
    ]
    release = snapshot["release_identities"][0]
    manifest = {
        "compatibility_contract": {
            "contract_id": snapshot["contract"]["contract_id"],
            "path": snapshot["contract"]["canonical_path"],
            "sha256": snapshot["contract_sha256"],
        },
        "consumer_parity_fixtures": {
            "path": snapshot["contract"]["parity_fixture_path"],
            "sha256": snapshot["parity_sha256"],
        },
        "model_artifact_hash": release["source_tree_hash"],
        "git_commit_sha": release["git_commit_sha"],
        "manifest_hash": release["manifest_hash"],
        "image_digest": release["image_digest"],
    }
    policy, policy_hash = compatibility.semantic_compatibility_policy_identity_v1()
    receipt = compatibility._semantic_compatibility_receipt(
        mode="legacy_exact",
        consumer_api_version=policy["consumer_api_version"],
        policy_hash=policy_hash,
        source_tree_hash=release["source_tree_hash"],
        manifest=manifest,
        contract=snapshot["contract"],
        contract_hash=snapshot["contract_sha256"],
        parity_hash=snapshot["parity_sha256"],
        bindings={},
    )
    accepted = compatibility.validate_source_tree_compatibility_receipt_v1(
        receipt,
        manifest=manifest,
        source_tree_hash=release["source_tree_hash"],
    )
    assert accepted["admission_mode"] == "legacy_exact"
    source_only_receipt = compatibility._semantic_compatibility_receipt(
        mode="legacy_exact",
        consumer_api_version=policy["consumer_api_version"],
        policy_hash=policy_hash,
        source_tree_hash=release["source_tree_hash"],
        manifest={},
        contract=snapshot["contract"],
        contract_hash=snapshot["contract_sha256"],
        parity_hash=snapshot["parity_sha256"],
        bindings={},
    )
    assert compatibility.validate_source_tree_compatibility_receipt_v1(
        source_only_receipt,
        manifest={},
        source_tree_hash=release["source_tree_hash"],
    )["admission_mode"] == "legacy_exact"
    wrong_contract_id = {
        **source_only_receipt,
        "contract_id": "leadpoet-sourcing-wrapper-contract-v999",
    }
    wrong_contract_id["receipt_hash"] = compatibility._sha256_json(
        {
            key: value
            for key, value in wrong_contract_id.items()
            if key != "receipt_hash"
        }
    )
    with pytest.raises(ValueError, match="differs from signed artifact"):
        compatibility.validate_source_tree_compatibility_receipt_v1(
            wrong_contract_id,
            manifest={},
            source_tree_hash=release["source_tree_hash"],
        )
    for field in (
        "model_artifact_hash",
        "git_commit_sha",
        "manifest_hash",
        "image_digest",
    ):
        hybrid = deepcopy(manifest)
        hybrid[field] = "f" * 40 if field == "git_commit_sha" else "sha256:" + "f" * 64
        if field == "image_digest":
            hybrid[field] = "private.invalid/model@sha256:" + "f" * 64
        with pytest.raises(ValueError, match="differs from signed artifact"):
            compatibility.validate_source_tree_compatibility_receipt_v1(
                receipt,
                manifest=hybrid,
                source_tree_hash=release["source_tree_hash"],
            )

    relabelled = {**source_only_receipt, "admission_mode": "semantic_v1"}
    relabelled["receipt_hash"] = compatibility._sha256_json(
        {key: value for key, value in relabelled.items() if key != "receipt_hash"}
    )
    with pytest.raises(ValueError, match="differs from signed artifact"):
        compatibility.validate_source_tree_compatibility_receipt_v1(
            relabelled,
            manifest={},
            source_tree_hash=release["source_tree_hash"],
        )


def test_exact_reviewed_pair_on_new_source_uses_semantic_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    contract = json.loads(
        compatibility.CONTRACT_V46_PATH.read_text(encoding="utf-8")
    )
    contract_path = tmp_path / contract["canonical_path"]
    parity_path = tmp_path / contract["parity_fixture_path"]
    contract_path.write_bytes(compatibility.CONTRACT_V46_PATH.read_bytes())
    parity_path.write_bytes(compatibility.PARITY_FIXTURE_V46_PATH.read_bytes())

    receipt = _admit(tmp_path, _manifest(tmp_path))

    assert receipt["admission_mode"] == "semantic_v1"
    assert receipt["contract_id"].endswith("v46")

    adapter = tmp_path / "research_lab_adapter.py"
    adapter.write_text(
        adapter.read_text(encoding="utf-8").replace(
            "return qualify(icp)",
            "return []",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hard module semantic drift"):
        _admit(tmp_path, _manifest(tmp_path))


def test_unprofiled_semantic_receipt_cannot_be_relabelled_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_future_tree(tmp_path, monkeypatch)
    manifest = _manifest(tmp_path)
    receipt = _admit(tmp_path, manifest)
    assert receipt["admission_mode"] == "semantic_v1"
    relabelled = {**receipt, "admission_mode": "legacy_exact"}
    relabelled["receipt_hash"] = compatibility._sha256_json(
        {key: value for key, value in relabelled.items() if key != "receipt_hash"}
    )

    with pytest.raises(ValueError, match="differs from signed artifact"):
        compatibility.validate_source_tree_compatibility_receipt_v1(
            relabelled,
            manifest=manifest,
            source_tree_hash=manifest["model_artifact_hash"],
            policy=compatibility.semantic_compatibility_policy_v1(),
            policy_hash=receipt["policy_hash"],
        )


def test_runtime_metadata_is_cross_bound_to_admitted_source() -> None:
    ready = _ready_adapter_metadata()
    ready.pop("qualification_outcome_protocol", None)
    ready["scoring_adapter_version"] = "qualification-company-scorer:v1"
    ready["company_fit_decision"] = {
        "contract_id": "company-fit-decision:v1",
        "outcomes": ["match", "mismatch", "unavailable"],
        "precedence": ["mismatch", "unavailable", "match"],
        "passing_outcome": "match",
        "required_dimensions": [
            "identity",
            "employee_size",
            "industry",
            "geography",
        ],
        "conditional_dimensions": ["stage"],
    }
    bindings = {
        "adapter_version": "sourcing-model-research-lab-adapter:v47-test",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "component_registry_version": "sourcing-model-components:v2",
        "routing_compiler_version": "routing-compiler-future",
        "scoring_adapter_version": "qualification-company-scorer:v1",
    }

    with pytest.raises(PrivateModelRuntimeError, match="reviewed adapter"):
        validate_sourcing_adapter_metadata(
            ready,
            expected_semantic_bindings=bindings,
            require_company_fit_contract=True,
        )
    ready["adapter_version"] = bindings["adapter_version"]
    with pytest.raises(PrivateModelRuntimeError, match="compiler version"):
        validate_sourcing_adapter_metadata(
            ready,
            expected_semantic_bindings=bindings,
            require_company_fit_contract=True,
        )
    ready["routing"]["compiler_version"] = bindings["routing_compiler_version"]
    ready["runtime_routing"]["compiler_version"] = bindings[
        "routing_compiler_version"
    ]
    expanded = deepcopy(ready)
    expanded["runtime_capabilities"].append("shell")
    with pytest.raises(PrivateModelRuntimeError, match="capability set differs"):
        validate_sourcing_adapter_metadata(
            expanded,
            expected_semantic_bindings=bindings,
            require_company_fit_contract=True,
        )
    missing_required = deepcopy(ready)
    missing_required["runtime_capabilities"].remove("resolve_host")
    with pytest.raises(PrivateModelRuntimeError, match="missing a Lab requirement"):
        validate_sourcing_adapter_metadata(
            missing_required,
            expected_semantic_bindings=bindings,
            require_company_fit_contract=True,
        )
    assert validate_sourcing_adapter_metadata(
        ready,
        expected_semantic_bindings=bindings,
        require_company_fit_contract=True,
    ) == ready


@pytest.mark.parametrize(
    ("contract_id", "adapter_version", "compiler_version"),
    (
        ("v46", "sourcing-model-research-lab-adapter:v7", "routing-compiler-v3"),
        ("v26", "sourcing-model-research-lab-adapter:v6", "routing-compiler-v2"),
        ("v13", "sourcing-model-research-lab-adapter:v3", "routing-compiler-v2"),
        ("v12", "sourcing-model-research-lab-adapter:v3", "routing-compiler-v2"),
        ("v11", "sourcing-model-research-lab-adapter:v3", "routing-compiler-v2"),
        ("v8", "sourcing-model-research-lab-adapter:v3", "routing-compiler-v2"),
        ("v7", "sourcing-model-research-lab-adapter:v3", "routing-compiler-v2"),
    ),
)
def test_reviewed_legacy_metadata_profiles_remain_cross_bound(
    contract_id: str,
    adapter_version: str,
    compiler_version: str,
) -> None:
    metadata = _ready_adapter_metadata()
    metadata["adapter_version"] = adapter_version
    metadata["routing"]["compiler_version"] = compiler_version
    metadata["runtime_routing"]["compiler_version"] = compiler_version
    bindings = {
        "adapter_version": adapter_version,
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "component_registry_version": "sourcing-model-components:v2",
        "routing_compiler_version": compiler_version,
        "scoring_adapter_version": "qualification-company-scorer:v1",
    }

    assert validate_sourcing_adapter_metadata(
        metadata,
        expected_semantic_bindings=bindings,
    ) == metadata, contract_id


def test_v46_legacy_receipt_rejects_historical_metadata_hybrid() -> None:
    metadata = _ready_adapter_metadata()
    v46_bindings = {
        "adapter_version": "sourcing-model-research-lab-adapter:v7",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "component_registry_version": "sourcing-model-components:v2",
        "routing_compiler_version": "routing-compiler-v3",
        "scoring_adapter_version": "qualification-company-scorer:v1",
    }

    with pytest.raises(PrivateModelRuntimeError, match="reviewed adapter"):
        validate_sourcing_adapter_metadata(
            metadata,
            expected_semantic_bindings=v46_bindings,
        )
    metadata["adapter_version"] = v46_bindings["adapter_version"]
    with pytest.raises(PrivateModelRuntimeError, match="compiler version"):
        validate_sourcing_adapter_metadata(
            metadata,
            expected_semantic_bindings=v46_bindings,
        )


@pytest.mark.asyncio
async def test_promotion_rereads_signed_pointer_after_semantic_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = PrivateModelArtifactManifest.from_mapping(artifact_mapping())
    changed = replace(
        artifact,
        image_digest="private.invalid/model@sha256:" + "f" * 64,
    )

    async def admitted(_artifact: PrivateModelArtifactManifest) -> dict:
        return {"admission_mode": "semantic_v1"}

    monkeypatch.setattr(
        promotion,
        "_preflight_private_artifact_compatibility",
        admitted,
    )
    monkeypatch.setattr(promotion, "_load_valid_artifact", lambda _uri: changed)

    with pytest.raises(RuntimeError, match="changed during compatibility"):
        await promotion._preflight_private_model_activation(
            SimpleNamespace(
                private_model_manifest_uri="s3://signed/current.json",
                private_repo_url="git@example.invalid/private.git",
                private_repo_branch="leadpoet-lab",
            ),
            artifact,
            pointer_uri="s3://signed/current.json",
            mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD,
            expected_branch_sha=artifact.git_commit_sha,
        )


@pytest.mark.asyncio
async def test_promotion_rereads_private_branch_after_semantic_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = PrivateModelArtifactManifest.from_mapping(artifact_mapping())

    async def admitted(_artifact: PrivateModelArtifactManifest) -> dict:
        return {"admission_mode": "semantic_v1"}

    monkeypatch.setattr(
        promotion,
        "_preflight_private_artifact_compatibility",
        admitted,
    )
    monkeypatch.setattr(
        promotion,
        "_load_valid_artifact",
        lambda _uri: artifact,
    )
    monkeypatch.setattr(
        promotion,
        "_resolve_private_repo_head_sha",
        lambda **_kwargs: "f" * 40,
    )

    with pytest.raises(RuntimeError, match="source branch changed"):
        await promotion._preflight_private_model_activation(
            SimpleNamespace(
                private_model_manifest_uri="s3://signed/current.json",
                private_repo_url="git@example.invalid/private.git",
                private_repo_branch="leadpoet-lab",
            ),
            artifact,
            pointer_uri="s3://signed/current.json",
            mode=promotion.PRIVATE_MODEL_ACTIVATION_MODE_EXACT_HEAD,
            expected_branch_sha=artifact.git_commit_sha,
        )
