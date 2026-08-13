"""Sourcing-model consumer-contract conformance checks.

The reviewed model-owned v7/v8/v11/v12/v13/v26 contracts are snapshotted byte-for-byte under
``research_lab/``. The exact function signatures
the Lab and production harness call
(``research_lab_adapter.run_icp``/``adapter_metadata``,
``sourcing_model.core.qualify``, the discovery/validation/client seams the
harness monkey-patches), module bindings reached through by the wrapper, and
integer floor constants. The new production flow is built AROUND these
symbols, so any model-source change that breaks them breaks both the lab
benchmark runtime and the harness.

``verify_source_tree_contract`` validates a model SOURCE TREE against that
contract using ``ast`` only — no imports, no execution, safe on untrusted
patched source.  Intended call sites:

* the candidate build path, so an autoresearch code-edit that would break the
  frozen adapter surface fails fast at build time instead of producing an
  image the benchmark cannot invoke (flag-gated, see code_build);
* local/CI checks against a model checkout.

Pure stdlib. A candidate is rejected unless its embedded canonical contract and
parity fixtures are a byte-identical pair from the reviewed consumer allowlist.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping

CONTRACT_PATH = Path(__file__).with_name("sourcing_model_contract.json")
PARITY_FIXTURE_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures.json"
)
CONTRACT_V7_PATH = Path(__file__).with_name("sourcing_model_contract_v7.json")
PARITY_FIXTURE_V7_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v7.json"
)
CONTRACT_V11_PATH = Path(__file__).with_name("sourcing_model_contract_v11.json")
PARITY_FIXTURE_V11_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v11.json"
)
CONTRACT_V12_PATH = Path(__file__).with_name("sourcing_model_contract_v12.json")
PARITY_FIXTURE_V12_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v12.json"
)
CONTRACT_V13_PATH = Path(__file__).with_name("sourcing_model_contract_v13.json")
PARITY_FIXTURE_V13_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v13.json"
)
CONTRACT_V26_PATH = Path(__file__).with_name("sourcing_model_contract_v26.json")
PARITY_FIXTURE_V26_PATH = Path(__file__).with_name(
    "sourcing_model_parity_fixtures_v26.json"
)
REVIEWED_CONSUMER_SNAPSHOT_SPECS = (
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v26",
        "contract_path": CONTRACT_V26_PATH,
        "contract_sha256": (
            "sha256:fb20751ddbc068d754913f5a6aea35d2330572acd267dd0e3a2906ff5c221a83"
        ),
        "parity_path": PARITY_FIXTURE_V26_PATH,
        "parity_sha256": (
            "sha256:28fd84abd9a0af578590c0744744a0e817624a5effe37f5449916b40e8557675"
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v13",
        "contract_path": CONTRACT_V13_PATH,
        "contract_sha256": (
            "sha256:9ab93592ae1e969bd08e50c73708513968b601c2b95e8d661a67cdcd3674f5da"
        ),
        "parity_path": PARITY_FIXTURE_V13_PATH,
        "parity_sha256": (
            "sha256:22638a5804681b3305606844359e6e69112937c21bda1cd34bb5edde93cdc7f0"
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v12",
        "contract_path": CONTRACT_V12_PATH,
        "contract_sha256": (
            "sha256:d681d2100a570c1e22447e3ac8bba53806ce01ae1f4cdad6aeba8eb8b6abaff3"
        ),
        "parity_path": PARITY_FIXTURE_V12_PATH,
        "parity_sha256": (
            "sha256:82b2cbd1cf9cf346b144d0d5cee8ec8d9ca4c02d97a52da2914313a1a5718dea"
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v11",
        "contract_path": CONTRACT_V11_PATH,
        "contract_sha256": (
            "sha256:2cd4d09b99db1f0ac523c3e57f361afb7c7ff1413392bd9aa5dfcee9efb81c01"
        ),
        "parity_path": PARITY_FIXTURE_V11_PATH,
        "parity_sha256": (
            "sha256:8b0d23b1664b5539e790c988afcb558c2aa4cf0ff925af0f7dbe2f9bc900fce4"
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v8",
        "contract_path": CONTRACT_PATH,
        "contract_sha256": (
            "sha256:080e7b199c3e1d27ae080e497b541b560a2e12d383a709d453e7a2dd320b8dfc"
        ),
        "parity_path": PARITY_FIXTURE_PATH,
        "parity_sha256": (
            "sha256:5527186b45294135639619d99bfcf076ec98035670f68843244ccd18fc3f80fe"
        ),
    },
    {
        "contract_id": "leadpoet-sourcing-wrapper-contract-v7",
        "contract_path": CONTRACT_V7_PATH,
        "contract_sha256": (
            "sha256:f2fea5a16de1dd1fafb1fa5259b161cd0dd8059fddaf30d8e9982d3eec391d10"
        ),
        "parity_path": PARITY_FIXTURE_V7_PATH,
        "parity_sha256": (
            "sha256:c39c48335a4877c091e6ca264f3f9411dbecd4992c09e9c77bdb789479076d3a"
        ),
    },
)

# These wrappers bind the fixed handoff into each stage. Routing policy,
# fallback, and tool metadata remain editable outside them. The projection is
# checked from source AST and does not import untrusted model code.
_ROUTER_STAGE_BINDINGS = {
    "compile_candidate_acquisition_route": "STAGE_CANDIDATE_ACQUISITION",
    "compile_candidate_enrichment_route": "STAGE_CANDIDATE_ENRICHMENT",
    "compile_intent_evidence_route": "STAGE_INTENT_EVIDENCE",
    "compile_contact_acquisition_route": "STAGE_CONTACT_ACQUISITION",
}
_DEFAULT_ROUTER_STAGE_BINDINGS = {
    "compile_candidate_route": "STAGE_CANDIDATE_ACQUISITION",
    "compile_intent_route": "STAGE_INTENT_EVIDENCE",
}
_PIPELINE_STRUCTURE_SCHEMA_VERSION = "leadpoet.sourcing_pipeline_structure.v1"
_PIPELINE_STAGE_VALUES = {
    "STAGE_CANDIDATE_ACQUISITION": "candidate_acquisition",
    "STAGE_CANDIDATE_ENRICHMENT": "candidate_enrichment",
    "STAGE_INTENT_EVIDENCE": "intent_evidence",
    "STAGE_CONTACT_ACQUISITION": "contact_acquisition",
}
_PIPELINE_IMPORTED_TOOL_IDS = {
    "TOOL_CONTACT_COMPANY_PEOPLE_SEARCH": "contact.company_people_search",
    "TOOL_CONTACT_COMPANY_PEOPLE_SEARCH_FALLBACK": (
        "contact.company_people_search_fallback"
    ),
    "TOOL_CONTACT_COMPANY_TITLE_ROSTER": "contact.company_title_roster",
    "TOOL_INTENT_SCRAPINGDOG_GOOGLE_NEWS": "intent.scrapingdog_google_news",
    "TOOL_INTENT_SCRAPINGDOG_GOOGLE_SEARCH": "intent.scrapingdog_google_search",
    "TOOL_INTENT_SCRAPINGDOG_YOUTUBE": "intent.scrapingdog_youtube_search",
}
_PIPELINE_KNOWN_IMPORTED_VALUES: Dict[str, Any] = {
    **_PIPELINE_STAGE_VALUES,
    **_PIPELINE_IMPORTED_TOOL_IDS,
    "EXECUTION_INVOKE": "invoke",
    "EXECUTION_OBSERVE": "observe",
    "EXECUTION_VIRTUAL": "virtual",
    "IDEMPOTENT": "idempotent",
    "RESUME_SAFE": "resume_safe",
    "NON_IDEMPOTENT": "non_idempotent",
    "COST_FREE": "free",
    "COST_METERED": "metered",
    "COST_PAID": "paid",
    "ORIGIN_BUILTIN": "builtin",
    "ORIGIN_SOURCE_ADD": "source_add",
    "IDENTITY_ANCHOR_NONE": "none",
    "IDENTITY_ANCHOR_VERIFIED_COMPANY_IDENTITY": "verified_company_identity",
    "CLAIM_VISIBILITY_VERIFIER_EVIDENCE": "verifier_evidence",
    "CLAIM_VISIBILITY_PRIVATE_CORROBORATING": "private_corroborating",
    "IDENTITY_POLICY_GENERIC": "generic",
    "IDENTITY_POLICY_EXACT_REGISTRABLE_DOMAIN_V1": (
        "exact_registrable_domain_v1"
    ),
    "PAGINATION_BOUNDED": "bounded",
    "PAGINATION_FORBIDDEN": "forbidden",
}
_PIPELINE_REVIEWED_SHARED_TOOL_OWNERSHIP = {
    ("definition", "intent.existing_evidence"): frozenset(
        {
            "sourcing_model/routing/defaults.py",
            "sourcing_model/routing/runtime.py",
        }
    ),
    ("policy", "intent.existing_evidence"): frozenset(
        {
            "sourcing_model/routing/defaults.py",
            "sourcing_model/routing/runtime.py",
        }
    ),
    ("definition", "intent.scrapingdog_google_news"): frozenset(
        {
            "sourcing_model/routing/runtime.py",
            "sourcing_model/scrapingdog_signal_contract.py",
        }
    ),
}
_PIPELINE_ROUTING_MODULES = (
    "sourcing_model/routing/defaults.py",
    "sourcing_model/routing/runtime.py",
)
_PIPELINE_MEMBERSHIP_MODULES = (
    *_PIPELINE_ROUTING_MODULES,
    "sourcing_model/scrapingdog_intent.py",
    "sourcing_model/scrapingdog_signal_contract.py",
)
_PIPELINE_EXACT_SYMBOLS = {
    "sourcing_model/routing/defaults.py": (
        "default_catalog",
    ),
    "sourcing_model/routing/runtime.py": (
        "SourceAddRoutingRegistration",
        "source_add_routing_structures",
        "enhanced_scrapingdog_tool_definitions",
        "runtime_catalog",
        "_ENHANCED_SCRAPINGDOG_TOOL_DEFINITIONS",
        "_SOURCE_ADD_TOOL_DEFINITIONS",
    ),
    "sourcing_model/scrapingdog_intent.py": tuple(
        _PIPELINE_IMPORTED_TOOL_IDS
    ),
    "sourcing_model/scrapingdog_signal_contract.py": (
        "ToolContract",
        "_tool",
    ),
}
_PIPELINE_COLLECTION_FUNCTIONS = {
    "sourcing_model/routing/defaults.py": {
        "builtin_definitions": ("definitions", "ToolDefinition"),
        "default_policy": ("policy", "PolicyStep"),
    },
    "sourcing_model/routing/runtime.py": {
        "runtime_tool_definitions": ("definitions", "ToolDefinition"),
        "runtime_policy": ("policy", "PolicyStep"),
    },
}
_PIPELINE_POLICY_MUTABLE_FIELDS = frozenset(
    {
        "forbidden_features",
        "intent_categories",
        "priority",
        "reason_code",
        "required",
        "required_features",
        "stop_on_success",
    }
)
_PIPELINE_TOOL_MUTABLE_FIELDS = frozenset(
    {
        "avoid_when",
        "best_for",
        "best_for_description",
        "capabilities",
        "cost_class",
        "evidence_types",
        "avoid_when_description",
        "idempotency",
        "intent_categories",
        "max_calls",
        "max_results",
        "revision",
        "timeout_seconds",
        "unit_cost",
    }
)
_PIPELINE_SIGNAL_TOOL_MUTABLE_FIELDS = frozenset(
    {
        "capability",
        "claim_visibility",
        "cost_credits",
        "evidence_types",
        "identity_anchor",
        "identity_policy",
        "pagination_mode",
        "routing_role",
        "signals",
        "source_classes",
        "supported_languages",
        "supported_regions",
    }
)
_PIPELINE_SOURCE_ADD_MUTABLE_FIELDS = frozenset(
    {
        "avoid_when",
        "avoid_when_description",
        "best_for",
        "best_for_description",
        "binding_requirements",
        "capabilities",
        "category_contracts",
        "cost_class",
        "evidence_types",
        "execution_mode",
        "idempotency",
        "intent_categories",
        "manifest_sha256",
        "max_calls",
        "max_results",
        "priority",
        "revision",
        "timeout_seconds",
        "unit_cost",
    }
)
_PIPELINE_CONSTRUCTOR_POSITIONAL_FIELDS = {
    "PolicyStep": (
        "tool_id",
        "stage",
        "priority",
        "required_features",
        "forbidden_features",
        "intent_categories",
        "stop_on_success",
        "required",
        "reason_code",
    ),
    "ToolDefinition": (),
    "_tool": (
        "tool_id",
        "stage",
        "capability",
        "routing_role",
        "signals",
        "source_classes",
        "evidence_types",
        "cost_credits",
    ),
    "SourceAddRoutingRegistration": (),
    "SourceAddCategoryContract": (),
    "RoutingPolicy": (),
}
_PIPELINE_CONSTRUCTOR_FIELDS = {
    "PolicyStep": frozenset(_PIPELINE_CONSTRUCTOR_POSITIONAL_FIELDS["PolicyStep"]),
    "ToolDefinition": frozenset(
        {
            "avoid_when",
            "avoid_when_description",
            "best_for",
            "best_for_description",
            "capabilities",
            "cost_class",
            "evidence_types",
            "execution_mode",
            "idempotency",
            "intent_categories",
            "manifest_sha256",
            "max_calls",
            "max_results",
            "origin",
            "revision",
            "stages",
            "timeout_seconds",
            "tool_id",
            "unit_cost",
        }
    ),
    "_tool": frozenset(
        {
            *_PIPELINE_CONSTRUCTOR_POSITIONAL_FIELDS["_tool"],
            "claim_visibility",
            "identity_anchor",
            "identity_policy",
            "pagination_mode",
            "supported_languages",
            "supported_regions",
        }
    ),
    "SourceAddRoutingRegistration": frozenset(
        {
            "avoid_when",
            "avoid_when_description",
            "best_for",
            "best_for_description",
            "binding_requirements",
            "capabilities",
            "category_contracts",
            "cost_class",
            "evidence_types",
            "execution_mode",
            "idempotency",
            "intent_categories",
            "manifest_sha256",
            "max_calls",
            "max_results",
            "priority",
            "provider_id",
            "revision",
            "stage",
            "timeout_seconds",
            "unit_cost",
        }
    ),
    "SourceAddCategoryContract": frozenset(
        {"capabilities", "category", "evidence_types", "requirements"}
    ),
    "RoutingPolicy": frozenset({"policy_version", "schema_version", "steps"}),
}
_PIPELINE_SOURCE_ADD_V7_FULL_FIELDS = frozenset(
    _PIPELINE_CONSTRUCTOR_FIELDS["SourceAddRoutingRegistration"]
    - {"execution_mode", "category_contracts", "binding_requirements"}
)
_PIPELINE_SOURCE_ADD_V7_LEGACY_FIELDS = frozenset(
    _PIPELINE_SOURCE_ADD_V7_FULL_FIELDS
    - {"best_for", "avoid_when", "best_for_description", "avoid_when_description"}
)
_PIPELINE_SOURCE_ADD_V7_ORIGINAL_FIELDS = frozenset(
    _PIPELINE_SOURCE_ADD_V7_LEGACY_FIELDS - {"intent_categories"}
)
_PIPELINE_SOURCE_ADD_V7_GUIDANCE_WITHOUT_CATEGORIES_FIELDS = frozenset(
    _PIPELINE_SOURCE_ADD_V7_FULL_FIELDS - {"intent_categories"}
)
_PIPELINE_SOURCE_ADD_V7_ACCEPTED_FIELD_SETS = (
    _PIPELINE_SOURCE_ADD_V7_FULL_FIELDS,
    _PIPELINE_SOURCE_ADD_V7_LEGACY_FIELDS,
    _PIPELINE_SOURCE_ADD_V7_ORIGINAL_FIELDS,
    _PIPELINE_SOURCE_ADD_V7_GUIDANCE_WITHOUT_CATEGORIES_FIELDS,
)
_PIPELINE_LIVENESS_FIELDS = {
    "PolicyStep": frozenset(
        {
            "forbidden_features",
            "intent_categories",
            "required",
            "required_features",
        }
    ),
    "ToolDefinition": frozenset(
        {
            "avoid_when",
            "capabilities",
            "cost_class",
            "execution_mode",
            "intent_categories",
            "max_calls",
            "max_results",
            "timeout_seconds",
            "unit_cost",
        }
    ),
    "_tool": frozenset(
        {
            "capability",
            "claim_visibility",
            "cost_credits",
            "identity_anchor",
            "identity_policy",
            "pagination_mode",
            "routing_role",
            "signals",
            "supported_languages",
            "supported_regions",
        }
    ),
    "SourceAddRoutingRegistration": frozenset(
        {
            "cost_class",
            "execution_mode",
            "intent_categories",
            "max_calls",
            "max_results",
            "timeout_seconds",
            "unit_cost",
        }
    ),
}
_PIPELINE_PROMPT_FUNCTIONS = {
    "sourcing_model/discovery.py": frozenset(
        {
            "_fallback_query",
            "_soft_context_query",
            "_source_query_prefix",
            "agent_request",
            "build_query",
            "build_query_variants",
            "evidence_characteristic_clause",
        }
    ),
}
_PIPELINE_PROMPT_LOCAL_BINDINGS = {
    "sourcing_model/discovery.py": {
        "_fallback_query": frozenset({"co"}),
        "_soft_context_query": frozenset({"context", "prefix"}),
        "build_query_variants": frozenset({"context"}),
        "agent_request": frozenset({"hard", "query", "signal_line"}),
    },
}
_PIPELINE_PROMPT_CALL_SINKS = {
    "sourcing_model/discovery.py": {
        "build_query_variants": frozenset({"add_variant"}),
        "agent_request": frozenset({"hard.append"}),
    },
}
_PIPELINE_PROMPT_BINDINGS = {
    "sourcing_model/discovery.py": frozenset(
        {
            "_CATEGORY_ALIASES",
            "_DATED_EVIDENCE_SUFFIX",
            "_EVIDENCE_CHARACTERISTICS",
            "_INTENT_CATEGORY_ALIASES",
            "_INTENT_PHRASE_FAMILIES",
        }
    ),
    "sourcing_model/routing/guidance.py": frozenset(
        {"GUIDANCE_SYSTEM_PROMPT"}
    ),
}
_PIPELINE_LITERAL_STRING_BINDINGS = {
    "sourcing_model/routing/guidance.py": frozenset(
        {"GUIDANCE_SYSTEM_PROMPT"}
    ),
}
_PIPELINE_EDIT_SURFACE_MODULES = (
    "sourcing_model/discovery.py",
    *_PIPELINE_MEMBERSHIP_MODULES,
)
_PIPELINE_OPTIONAL_EDIT_SURFACE_MODULES = (
    "sourcing_model/routing/guidance.py",
)


def _pipeline_edit_surface_modules(
    snapshot: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> tuple[str, ...]:
    required_files = set(snapshot["contract"].get("required_files", ()))
    return (
        *_PIPELINE_EDIT_SURFACE_MODULES,
        *(
            path
            for path in _PIPELINE_OPTIONAL_EDIT_SURFACE_MODULES
            if path in required_files
            or root is not None and (Path(root) / path).is_file()
        ),
    )
_PIPELINE_REFLECTION_NAMES = frozenset(
    {
        "__builtins__",
        "__dict__",
        "__globals__",
        "__import__",
        "__setitem__",
        "__delitem__",
        "eval",
        "exec",
        "globals",
        "locals",
        "vars",
    }
)
def load_wrapper_contract(path: Path | None = None) -> Dict[str, Any]:
    """Load and shape-check the reviewed model-owned contract snapshot."""
    document = json.loads(Path(path or CONTRACT_PATH).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(
            "Unsupported sourcing wrapper contract schema_version: "
            f"{document.get('schema_version')!r}"
        )
    for key in (
        "contract_id",
        "canonical_path",
        "parity_fixture_path",
        "required_files",
        "functions",
    ):
        if key not in document:
            raise ValueError(f"wrapper contract missing required key {key!r}")
    return document


def reviewed_consumer_snapshots() -> Dict[str, Dict[str, Any]]:
    """Return the exact contract/parity pairs accepted by this consumer.

    Contract ids are selectors only. Acceptance still requires byte equality
    with both files in the selected pair, so an artifact cannot mix a reviewed
    contract with unrelated parity projections.
    """

    snapshots: Dict[str, Dict[str, Any]] = {}
    for spec in REVIEWED_CONSUMER_SNAPSHOT_SPECS:
        contract_path = Path(spec["contract_path"])
        parity_path = Path(spec["parity_path"])
        contract_sha256 = _snapshot_sha256(contract_path)
        parity_sha256 = _snapshot_sha256(parity_path)
        if contract_sha256 != spec["contract_sha256"]:
            raise ValueError(
                f"reviewed sourcing contract hash differs: {contract_path.name}"
            )
        if parity_sha256 != spec["parity_sha256"]:
            raise ValueError(
                f"reviewed sourcing parity hash differs: {parity_path.name}"
            )
        document = load_wrapper_contract(contract_path)
        contract_id = str(spec["contract_id"])
        if document["contract_id"] != contract_id:
            raise ValueError(
                f"reviewed sourcing contract id differs: {contract_path.name}"
            )
        if contract_id in snapshots:
            raise ValueError(
                f"duplicate reviewed sourcing wrapper contract id: {contract_id}"
            )
        if not parity_path.is_file():
            raise ValueError(
                f"reviewed sourcing parity snapshot is missing: {parity_path.name}"
            )
        snapshots[contract_id] = {
            "contract": document,
            "contract_path": contract_path,
            "contract_sha256": contract_sha256,
            "parity_path": parity_path,
            "parity_sha256": parity_sha256,
        }
    return snapshots


def _snapshot_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"reviewed sourcing snapshot is unreadable: {path.name}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def resolve_reviewed_consumer_snapshot(root: Path) -> Dict[str, Any] | None:
    """Resolve a model tree to one exact reviewed contract/parity pair."""

    root = Path(root)
    matches: list[Dict[str, Any]] = []
    for snapshot in reviewed_consumer_snapshots().values():
        document = snapshot["contract"]
        candidate_contract_path = root / str(document["canonical_path"])
        candidate_parity_path = root / str(document["parity_fixture_path"])
        try:
            if (
                candidate_contract_path.is_file()
                and candidate_parity_path.is_file()
                and candidate_contract_path.read_bytes()
                == Path(snapshot["contract_path"]).read_bytes()
                and candidate_parity_path.read_bytes()
                == Path(snapshot["parity_path"]).read_bytes()
            ):
                matches.append(snapshot)
        except OSError:
            continue
    return matches[0] if len(matches) == 1 else None


def _function_signature(node: ast.AST) -> Dict[str, Any]:
    args = getattr(node, "args", None)
    if args is None:
        return {
            "params": [],
            "all_params": [],
            "positional_only": [],
            "vararg": None,
            "kwarg": None,
            "required_positional": 0,
            "required_keyword_only": [],
        }
    positional = list(args.posonlyargs + args.args)
    required_positional = max(0, len(positional) - len(args.defaults))
    required_keyword_only = [
        item.arg
        for item, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    ]
    return {
        "params": [item.arg for item in positional],
        "all_params": [
            item.arg for item in [*positional, *args.kwonlyargs]
        ],
        "positional_only": [item.arg for item in args.posonlyargs],
        "vararg": args.vararg.arg if args.vararg is not None else None,
        "kwarg": args.kwarg.arg if args.kwarg is not None else None,
        "required_positional": required_positional,
        "required_keyword_only": required_keyword_only,
    }


def _int_constant(value: ast.AST | None) -> int | None:
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, int)
        and not isinstance(value.value, bool)
    ):
        return value.value
    return None


def _module_symbols(tree: ast.Module) -> Dict[str, Any]:
    """Top-level function param-lists and integer constant assignments.

    Constants follow last-assignment-wins module semantics: a plain or
    annotated assignment of an integer literal records the value, and any
    later top-level rebinding of the same name to a non-literal (call,
    expression, augmented assignment) discards it — the value the runtime
    would see is no longer statically verifiable, which downstream reads as
    a missing-constant violation rather than silently trusting an earlier
    literal.
    """
    functions: Dict[str, Dict[str, Any]] = {}
    constants: Dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = {
                **_function_signature(node),
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            }
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _int_constant(node.value)
                if value is not None:
                    constants[target.id] = value
                    continue
                constants.pop(target.id, None)
                # Simple top-level alias (``qualify = _qualify_impl``) is a
                # runtime-valid rebinding — carry the aliased function's
                # surface instead of reporting it missing. Anything else
                # rebinding a function name makes it unverifiable.
                if isinstance(node.value, ast.Name) and node.value.id in functions:
                    functions[target.id] = dict(functions[node.value.id])
                else:
                    functions.pop(target.id, None)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                value = _int_constant(node.value)
                if value is not None:
                    constants[node.target.id] = value
                else:
                    constants.pop(node.target.id, None)
                    functions.pop(node.target.id, None)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            constants.pop(node.target.id, None)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants.pop(target.id, None)
                    functions.pop(target.id, None)
    return {"functions": functions, "constants": constants}


def _module_bound_imports(tree: ast.Module) -> set[str]:
    """Return dotted modules bound by plain, unaliased ``import a.b``.

    The site wrapper reaches ``clients.urllib.request.urlopen``. That chain is
    available after ``import urllib.request`` but not after
    ``from urllib.request import urlopen`` or
    ``import urllib.request as request``. Match the release bundle's contract
    check exactly.
    """
    bound: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname:
                continue
            parts = alias.name.split(".")
            for index in range(1, len(parts) + 1):
                bound.add(".".join(parts[:index]))
    return bound


def _module_scope_bindings(node: ast.AST) -> set[str]:
    """Return names rebound while evaluating ``node`` at module scope.

    Function and class bodies have their own scopes and are intentionally not
    traversed. Their names, decorators, bases, defaults, and annotations are
    evaluated at module scope and therefore still count.
    """

    bound: set[str] = set()

    class BindingVisitor(ast.NodeVisitor):
        def visit_Name(self, item: ast.Name) -> None:  # noqa: N802
            if isinstance(item.ctx, (ast.Store, ast.Del)):
                bound.add(item.id)

        def _visit_function_binding(
            self, item: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            bound.add(item.name)
            for decorator in item.decorator_list:
                self.visit(decorator)
            for default in item.args.defaults:
                self.visit(default)
            for default in item.args.kw_defaults:
                if default is not None:
                    self.visit(default)
            if item.returns is not None:
                self.visit(item.returns)

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_function_binding(item)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, item: ast.AsyncFunctionDef
        ) -> None:
            self._visit_function_binding(item)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:  # noqa: N802
            bound.add(item.name)
            for decorator in item.decorator_list:
                self.visit(decorator)
            for base in item.bases:
                self.visit(base)
            for keyword in item.keywords:
                self.visit(keyword.value)

        def visit_Lambda(self, item: ast.Lambda) -> None:  # noqa: N802
            for default in item.args.defaults:
                self.visit(default)
            for default in item.args.kw_defaults:
                if default is not None:
                    self.visit(default)

        def visit_Import(self, item: ast.Import) -> None:  # noqa: N802
            for alias in item.names:
                bound.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, item: ast.ImportFrom) -> None:  # noqa: N802
            for alias in item.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name)

    BindingVisitor().visit(node)
    return bound


def _literal_module_constants(
    tree: ast.Module,
    *,
    names: set[str],
) -> Dict[str, Any]:
    """Resolve exact constants with deterministic module last-write semantics.

    Only direct, single-name literal assignments are trusted. Conditional or
    dynamic writes poison the value unless a later direct literal assignment
    deterministically overwrites them.
    """

    constants: Dict[str, Any] = {}
    for node in tree.body:
        assigned_name = ""
        value_node: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned_name = node.targets[0].id
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            assigned_name = node.target.id
            value_node = node.value

        if assigned_name in names:
            if value_node is None:
                constants.pop(assigned_name, None)
                continue
            try:
                constants[assigned_name] = ast.literal_eval(value_node)
            except (TypeError, ValueError):
                constants.pop(assigned_name, None)
            continue

        for rebound_name in _module_scope_bindings(node) & names:
            constants.pop(rebound_name, None)
    return constants


def _same_literal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)):
            return False
        return len(actual) == len(expected) and all(
            _same_literal(left, right)
            for left, right in zip(actual, expected)
        )
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return actual.keys() == expected.keys() and all(
            _same_literal(actual[key], expected[key]) for key in expected
        )
    return type(actual) is type(expected) and actual == expected


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _has_single_unaliased_relative_import(
    tree: ast.Module,
    *,
    module: str,
    name: str,
) -> bool:
    bindings = []
    valid = []
    for node in tree.body:
        if name in _module_scope_bindings(node):
            bindings.append(node)
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == module
            and any(
                alias.name == name and alias.asname is None
                for alias in node.names
            )
        ):
            valid.append(node)
    return len(bindings) == 1 and len(valid) == 1 and bindings == valid


def _function_rebinds(function: ast.AST, names: set[str]) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id in names and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            return True
        if isinstance(node, ast.arg) and node.arg in names:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not function and node.name in names:
                return True
        if isinstance(node, ast.ExceptHandler) and node.name in names:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".", 1)[0]) in names:
                    return True
    return False


def _node_hash(node: ast.AST) -> str:
    encoded = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _top_level_binding_nodes(tree: ast.Module, name: str) -> list[ast.AST]:
    return [node for node in tree.body if name in _module_scope_bindings(node)]


def _simple_string_constants(tree: ast.Module) -> Dict[str, str]:
    """Resolve only literal or direct-name string constants without imports."""

    values: Dict[str, str] = {}
    pending: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        name = ""
        value: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            name = node.target.id
            value = node.value
        if name and value is not None:
            pending.append((name, value))
    changed = True
    while changed:
        changed = False
        for name, value in pending:
            resolved: str | None = None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                resolved = value.value
            elif isinstance(value, ast.Name):
                resolved = values.get(value.id)
            if resolved is not None and values.get(name) != resolved:
                values[name] = resolved
                changed = True
    return values


def _call_argument(
    call: ast.Call,
    *,
    keyword: str,
    position: int,
) -> ast.AST | None:
    matches = [item.value for item in call.keywords if item.arg == keyword]
    positional = call.args[position] if len(call.args) > position else None
    if len(matches) > 1 or (matches and positional is not None):
        return None
    return matches[0] if matches else positional


def _resolve_string(node: ast.AST | None, constants: Mapping[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return str(
            constants.get(node.id)
            or _PIPELINE_IMPORTED_TOOL_IDS.get(node.id)
            or ""
        )
    return ""


def _resolve_stages(
    node: ast.AST | None,
    constants: Mapping[str, str],
    *,
    sequence: bool,
) -> tuple[str, ...]:
    values: list[ast.AST]
    if sequence:
        if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return ()
        values = list(node.elts)
    else:
        values = [] if node is None else [node]
    stages: list[str] = []
    for value in values:
        if isinstance(value, ast.Name) and value.id in _PIPELINE_STAGE_VALUES:
            stage = _PIPELINE_STAGE_VALUES[value.id]
        else:
            stage = _resolve_string(value, constants)
        if stage not in set(_PIPELINE_STAGE_VALUES.values()):
            return ()
        stages.append(stage)
    return tuple(stages)


def _without_docstring(function: ast.AST) -> list[ast.stmt]:
    body = list(getattr(function, "body", ()))
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _collection_items(
    function: ast.AST,
    *,
    kind: str,
) -> tuple[list[ast.AST], str]:
    body = _without_docstring(function)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return [], "must contain only its collection return"
    value = body[0].value
    if kind == "definitions":
        if not isinstance(value, ast.Tuple):
            return [], "must return one literal tuple"
        return list(value.elts), ""
    if not isinstance(value, ast.Call) or _call_name(value) != "RoutingPolicy":
        return [], "must return one RoutingPolicy"
    if value.args:
        return [], "RoutingPolicy must use named declarative fields"
    if any(item.arg is None for item in value.keywords):
        return [], "RoutingPolicy cannot unpack keyword arguments"
    keyword_names = [str(item.arg) for item in value.keywords]
    if len(keyword_names) != len(set(keyword_names)):
        return [], "RoutingPolicy contains duplicate fields"
    if set(keyword_names) - _PIPELINE_CONSTRUCTOR_FIELDS["RoutingPolicy"]:
        return [], "RoutingPolicy contains unknown fields"
    for item in value.keywords:
        if item.arg == "steps":
            continue
        if not _is_pipeline_data_expression(item.value):
            return [], f"RoutingPolicy {item.arg} must be inert data"
    steps = [item.value for item in value.keywords if item.arg == "steps"]
    if len(steps) != 1 or not isinstance(steps[0], ast.Tuple):
        return [], "RoutingPolicy steps must be one literal tuple"
    return list(steps[0].elts), ""


def _is_pipeline_data_expression(node: ast.AST) -> bool:
    """Return whether ``node`` is inert declarative routing data.

    This is a positive grammar. Calls, attributes, subscripts, formatted
    expressions, comprehensions, and operators are code and are never made
    mutable by the routing-data allowlist.
    """

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool, type(None)))
    if isinstance(node, ast.Name):
        # Upper-case names are reviewed module constants. Their bindings stay
        # in the exact module skeleton, so the editable record may reference
        # them but cannot redefine or import them.
        return bool(re.fullmatch(r"_?[A-Z][A-Z0-9_]*", node.id))
    if isinstance(node, ast.Set):
        return False
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_pipeline_data_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and _is_pipeline_data_expression(key)
            and _is_pipeline_data_expression(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float)
        )
    return False


_PIPELINE_UNRESOLVED = object()
_PIPELINE_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,95}$")
_PIPELINE_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,95}$")
_PIPELINE_TOOL_CONTRACT_ID_RE = re.compile(
    r"^(?:candidate|intent)\.[a-z][a-z0-9_.-]{2,94}$"
)
_PIPELINE_TOOL_CONTRACT_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,95}$")
_PIPELINE_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")
_PIPELINE_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,79}$")
_PIPELINE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _pipeline_literal_value(
    node: ast.AST | None,
    values: Mapping[str, Any],
) -> Any:
    """Resolve the closed inert-data grammar without evaluating source."""

    if node is None:
        return _PIPELINE_UNRESOLVED
    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (str, int, float, bool, type(None)),
    ):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, _PIPELINE_UNRESOLVED)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _pipeline_literal_value(node.operand, values)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _PIPELINE_UNRESOLVED
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Set):
        return _PIPELINE_UNRESOLVED
    if isinstance(node, (ast.Tuple, ast.List)):
        output_items: list[Any] = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                expanded = _pipeline_literal_value(item.value, values)
                if not isinstance(expanded, tuple):
                    return _PIPELINE_UNRESOLVED
                output_items.extend(expanded)
            else:
                output_items.append(_pipeline_literal_value(item, values))
        output = tuple(output_items)
        if any(item is _PIPELINE_UNRESOLVED for item in output):
            return _PIPELINE_UNRESOLVED
        return output
    if isinstance(node, ast.Dict):
        output: Dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                return _PIPELINE_UNRESOLVED
            key = _pipeline_literal_value(key_node, values)
            value = _pipeline_literal_value(value_node, values)
            if key is _PIPELINE_UNRESOLVED or value is _PIPELINE_UNRESOLVED:
                return _PIPELINE_UNRESOLVED
            try:
                hash(key)
                output[key] = value
            except (TypeError, ValueError):
                return _PIPELINE_UNRESOLVED
        return output
    return _PIPELINE_UNRESOLVED


def _pipeline_static_values(tree: ast.Module) -> Dict[str, Any]:
    """Resolve local literal constants and reviewed imported enum values."""

    values: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            if alias.asname is None and alias.name in _PIPELINE_KNOWN_IMPORTED_VALUES:
                values[local_name] = _PIPELINE_KNOWN_IMPORTED_VALUES[alias.name]
    pending: Dict[str, ast.AST] = {}
    for node in tree.body:
        name, value = _assigned_name_and_value(node)
        if name and value is not None:
            pending[name] = value
    while pending:
        progressed = False
        for name, node in list(pending.items()):
            value = _pipeline_literal_value(node, values)
            if value is _PIPELINE_UNRESOLVED:
                continue
            values[name] = value
            pending.pop(name)
            progressed = True
        if not progressed:
            break
    return values


def _pipeline_static_values_before(
    tree: ast.Module,
    stop_node: ast.AST,
) -> Dict[str, Any]:
    """Resolve only bindings available before one eager module expression."""

    values: Dict[str, Any] = {}
    for node in tree.body:
        if node is stop_node:
            break
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                if alias.asname is None and alias.name in _PIPELINE_KNOWN_IMPORTED_VALUES:
                    values[local_name] = _PIPELINE_KNOWN_IMPORTED_VALUES[alias.name]
            continue
        name, value_node = _assigned_name_and_value(node)
        if not name or value_node is None:
            continue
        value = _pipeline_literal_value(value_node, values)
        if value is _PIPELINE_UNRESOLVED:
            values.pop(name, None)
        else:
            values[name] = value
    return values


def _pipeline_static_values_for_module(
    tree: ast.Module,
    *,
    trees: Mapping[str, ast.Module],
) -> Dict[str, Any]:
    """Resolve local data plus exact imported signal-category constants."""

    values = _pipeline_static_values(tree)
    intent_tree = trees.get("sourcing_model/scrapingdog_intent.py")
    if intent_tree is not None:
        imported_names = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and str(node.module or "").endswith("scrapingdog_intent")
            for alias in node.names
            if alias.name != "*"
        }
        intent_values = _pipeline_static_values(intent_tree)
        for name in imported_names:
            if name in intent_values:
                values[name] = intent_values[name]
    return values


def _pipeline_static_values_for_module_before(
    tree: ast.Module,
    stop_node: ast.AST,
    *,
    trees: Mapping[str, ast.Module],
) -> Dict[str, Any]:
    """Resolve eager data using Python module evaluation order."""

    values = _pipeline_static_values_before(tree, stop_node)
    intent_tree = trees.get("sourcing_model/scrapingdog_intent.py")
    if intent_tree is None:
        return values
    imported_names: set[str] = set()
    for node in tree.body:
        if node is stop_node:
            break
        if isinstance(node, ast.ImportFrom) and str(
            node.module or ""
        ).endswith("scrapingdog_intent"):
            imported_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            )
    intent_values = _pipeline_static_values(intent_tree)
    for name in imported_names:
        if name in intent_values:
            values[name] = intent_values[name]
    return values


def _pipeline_unresolved_name(node: ast.AST | None) -> str:
    return node.id if isinstance(node, ast.Name) else ""


def _pipeline_string_sequence(
    value: Any,
    *,
    maximum: int,
    pattern: re.Pattern[str] | None = None,
    required: bool = False,
) -> bool:
    if not isinstance(value, tuple):
        return False
    if required and not value:
        return False
    if len(value) > maximum:
        return False
    return all(
        isinstance(item, str)
        and bool(item.strip())
        and (pattern is None or pattern.fullmatch(item.strip()) is not None)
        for item in value
    )


def _pipeline_constructor_semantic_errors(
    call: ast.Call,
    constructor: str,
    *,
    values: Mapping[str, Any],
    bound_names: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Check constructor-required fields and fail-closed runtime bounds."""

    fields, structural_errors, executable_fields = _declarative_constructor_call(
        call,
        constructor,
    )
    if structural_errors:
        return ()
    required_fields = {
        "PolicyStep": {"tool_id", "stage", "priority"},
        "ToolDefinition": {"tool_id", "revision", "stages", "capabilities"},
        "_tool": {
            "tool_id",
            "stage",
            "capability",
            "routing_role",
            "signals",
            "source_classes",
            "evidence_types",
        },
        "SourceAddRoutingRegistration": {
            "provider_id",
            "stage",
            "priority",
            "capabilities",
        },
        "SourceAddCategoryContract": {
            "category",
            "capabilities",
            "evidence_types",
            "requirements",
        },
    }.get(constructor, set())
    errors = [
        f"missing required field {field}"
        for field in sorted(required_fields - set(fields))
    ]
    resolved = {
        field: _pipeline_literal_value(node, values)
        for field, node in fields.items()
        if field != "category_contracts"
    }
    reviewed_dynamic_fields = {
        ("ToolDefinition", "intent_categories"): {
            "GOOGLE_NEWS_RUNTIME_CATEGORIES",
            "SEARCH_INTENT_CATEGORIES",
            "YOUTUBE_INTENT_CATEGORIES",
        },
        ("PolicyStep", "intent_categories"): {
            "GOOGLE_NEWS_RUNTIME_CATEGORIES",
            "SEARCH_INTENT_CATEGORIES",
            "YOUTUBE_INTENT_CATEGORIES",
        },
    }
    for field, value in resolved.items():
        if value is _PIPELINE_UNRESOLVED:
            name = _pipeline_unresolved_name(fields.get(field))
            reviewed_dynamic = (
                name in bound_names
                and name
                in reviewed_dynamic_fields.get((constructor, field), set())
            )
            if field not in executable_fields and not reviewed_dynamic:
                errors.append(f"field {field} is not resolved inert data")

    def field(name: str, default: Any) -> Any:
        value = resolved.get(name, default)
        unresolved_name = _pipeline_unresolved_name(fields.get(name))
        if (
            value is _PIPELINE_UNRESOLVED
            and unresolved_name in bound_names
            and unresolved_name
            in reviewed_dynamic_fields.get((constructor, name), set())
        ):
            return default
        return value

    def is_opaque(name: str) -> bool:
        return name in executable_fields

    def bounded_number(
        name: str,
        default: int | float,
        minimum: float,
        maximum: float,
        *,
        integer: bool = False,
    ) -> None:
        value = field(name, default)
        valid_type = (
            isinstance(value, int) and not isinstance(value, bool)
            if integer
            else isinstance(value, (int, float)) and not isinstance(value, bool)
        )
        if (
            not valid_type
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            errors.append(f"field {name} is out of bounds")

    if constructor == "PolicyStep":
        tool_id = field("tool_id", "")
        stage = field("stage", "")
        priority = field("priority", _PIPELINE_UNRESOLVED)
        if not isinstance(tool_id, str) or not _PIPELINE_IDENTIFIER_RE.fullmatch(tool_id):
            errors.append("field tool_id is invalid")
        if stage not in set(_PIPELINE_STAGE_VALUES.values()):
            errors.append("field stage is invalid")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= 10_000
        ):
            errors.append("field priority is out of bounds")
        required_features = field("required_features", ())
        forbidden_features = field("forbidden_features", ())
        for name, sequence in (
            ("required_features", required_features),
            ("forbidden_features", forbidden_features),
        ):
            if not _pipeline_string_sequence(
                sequence,
                maximum=32,
                pattern=_PIPELINE_FEATURE_RE,
            ):
                errors.append(f"field {name} is invalid")
        if isinstance(required_features, tuple) and isinstance(
            forbidden_features,
            tuple,
        ) and set(required_features) & set(forbidden_features):
            errors.append("required and forbidden features overlap")
        categories = field("intent_categories", ())
        if not is_opaque("intent_categories") and (
            not _pipeline_string_sequence(categories, maximum=64) or any(
            len(item) > 80 for item in categories if isinstance(item, str)
            )
        ):
            errors.append("field intent_categories is invalid")
        for name in ("stop_on_success", "required"):
            if not isinstance(field(name, False), bool):
                errors.append(f"field {name} is invalid")
        reason_code = field("reason_code", "policy_selected")
        if not isinstance(reason_code, str) or not _PIPELINE_FEATURE_RE.fullmatch(
            reason_code
        ):
            errors.append("field reason_code is invalid")

    elif constructor == "ToolDefinition":
        tool_id = field("tool_id", "")
        revision = field("revision", "")
        stages = field("stages", ())
        capabilities = field("capabilities", ())
        if not isinstance(tool_id, str) or not _PIPELINE_IDENTIFIER_RE.fullmatch(tool_id):
            errors.append("field tool_id is invalid")
        if not isinstance(revision, str) or not _PIPELINE_VERSION_RE.fullmatch(revision):
            errors.append("field revision is invalid")
        if (
            not _pipeline_string_sequence(stages, maximum=4, required=True)
            or any(item not in set(_PIPELINE_STAGE_VALUES.values()) for item in stages)
        ):
            errors.append("field stages is invalid")
        if not _pipeline_string_sequence(
            capabilities,
            maximum=32,
            pattern=_PIPELINE_FEATURE_RE,
            required=True,
        ):
            errors.append("field capabilities is invalid")
        execution_mode = field("execution_mode", "invoke")
        if execution_mode not in {"invoke", "observe", "virtual"}:
            errors.append("field execution_mode is invalid")
        if field("idempotency", "idempotent") not in {
            "idempotent",
            "resume_safe",
            "non_idempotent",
        }:
            errors.append("field idempotency is invalid")
        cost_class = field("cost_class", "free")
        if cost_class not in {"free", "metered", "paid"}:
            errors.append("field cost_class is invalid")
        bounded_number("unit_cost", 0.0, 0.0, 10_000.0)
        unit_cost = field("unit_cost", 0.0)
        if isinstance(unit_cost, (int, float)) and not isinstance(unit_cost, bool):
            if cost_class == "free" and float(unit_cost) != 0.0:
                errors.append("free tool has nonzero unit_cost")
            if cost_class in {"metered", "paid"} and float(unit_cost) <= 0.0:
                errors.append("paid tool has nonpositive unit_cost")
        bounded_number("max_calls", 1, 1, 10_000, integer=True)
        bounded_number("max_results", 1, 1, 100_000, integer=True)
        bounded_number("timeout_seconds", 30.0, 0.1, 3_600.0)
        for name, maximum, required in (
            ("intent_categories", 64, False),
            ("evidence_types", 24, False),
            ("best_for", 32, False),
            ("avoid_when", 32, False),
        ):
            sequence = field(name, ())
            pattern = None if name == "intent_categories" else _PIPELINE_FEATURE_RE
            if not is_opaque(name) and (not _pipeline_string_sequence(
                sequence,
                maximum=maximum,
                pattern=pattern,
                required=required,
            ) or name == "intent_categories" and any(
                len(item) > 80 for item in sequence if isinstance(item, str)
            )):
                errors.append(f"field {name} is invalid")
        for name in ("best_for_description", "avoid_when_description"):
            value = field(name, "")
            if not isinstance(value, str) or len(" ".join(value.split())) > 500:
                errors.append(f"field {name} is invalid")
        origin = field("origin", "builtin")
        if origin not in {"builtin", "source_add"}:
            errors.append("field origin is invalid")
        manifest = field("manifest_sha256", None)
        if manifest is not None and (
            not isinstance(manifest, str)
            or not _PIPELINE_SHA256_RE.fullmatch(manifest)
        ):
            errors.append("field manifest_sha256 is invalid")
        if origin == "source_add" and manifest is None:
            errors.append("SOURCE_ADD tool is missing manifest_sha256")
        if origin == "source_add":
            errors.append("direct SOURCE_ADD tool is not registry-derived")

    elif constructor == "_tool":
        tool_id = field("tool_id", "")
        stage = field("stage", "")
        if not isinstance(tool_id, str) or not _PIPELINE_TOOL_CONTRACT_ID_RE.fullmatch(
            tool_id
        ):
            errors.append("field tool_id is invalid")
        if stage not in {
            "candidate_acquisition",
            "candidate_enrichment",
            "intent_evidence",
        }:
            errors.append("field stage is invalid")
        for name in ("capability", "routing_role"):
            value = field(name, "")
            if not isinstance(value, str) or not _PIPELINE_TOOL_CONTRACT_TOKEN_RE.fullmatch(
                value
            ):
                errors.append(f"field {name} is invalid")
        for name, maximum in (
            ("signals", 24),
            ("source_classes", 12),
            ("evidence_types", 16),
            ("supported_regions", 32),
            ("supported_languages", 32),
        ):
            sequence = field(
                name,
                ("global",)
                if name == "supported_regions"
                else ("en",)
                if name == "supported_languages"
                else (),
            )
            if not _pipeline_string_sequence(
                sequence,
                maximum=maximum,
                required=True,
            ) or any(len(item) > 80 for item in sequence if isinstance(item, str)):
                errors.append(f"field {name} is invalid")
        bounded_number("cost_credits", 5, 1, 1_000, integer=True)
        if field("identity_anchor", "none") not in {
            "none",
            "verified_company_identity",
        }:
            errors.append("field identity_anchor is invalid")
        if field("claim_visibility", "verifier_evidence") not in {
            "verifier_evidence",
            "private_corroborating",
        }:
            errors.append("field claim_visibility is invalid")
        if field("identity_policy", "generic") not in {
            "generic",
            "exact_registrable_domain_v1",
        }:
            errors.append("field identity_policy is invalid")
        if field("pagination_mode", "bounded") not in {"bounded", "forbidden"}:
            errors.append("field pagination_mode is invalid")
        identity_tuple = (
            field("identity_anchor", "none"),
            field("claim_visibility", "verifier_evidence"),
            field("identity_policy", "generic"),
            field("pagination_mode", "bounded"),
        )
        expected_identity = (
            (
                "verified_company_identity",
                "private_corroborating",
                "exact_registrable_domain_v1",
                "forbidden",
            )
            if stage == "candidate_enrichment"
            else ("none", "verifier_evidence", "generic", "bounded")
        )
        if identity_tuple != expected_identity:
            errors.append("tool identity policy does not match stage")

    elif constructor == "SourceAddRoutingRegistration":
        provider_id = field("provider_id", "")
        if not isinstance(provider_id, str) or not _PIPELINE_PROVIDER_ID_RE.fullmatch(
            provider_id
        ):
            errors.append("field provider_id is invalid")
        if field("stage", "") not in {
            "candidate_acquisition",
            "intent_evidence",
        }:
            errors.append("field stage is invalid")
        priority = field("priority", _PIPELINE_UNRESOLVED)
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= 10_000
        ):
            errors.append("field priority is out of bounds")
        if not _pipeline_string_sequence(
            field("capabilities", ()),
            maximum=32,
            pattern=_PIPELINE_FEATURE_RE,
            required=True,
        ):
            errors.append("field capabilities is invalid")
        if field("execution_mode", "invoke") not in {"invoke", "observe", "virtual"}:
            errors.append("field execution_mode is invalid")
        if field("idempotency", "idempotent") not in {
            "idempotent",
            "resume_safe",
            "non_idempotent",
        }:
            errors.append("field idempotency is invalid")
        cost_class = field("cost_class", "metered")
        if cost_class not in {"free", "metered", "paid"}:
            errors.append("field cost_class is invalid")
        bounded_number("unit_cost", 0.01, 0.0, 10_000.0)
        unit_cost = field("unit_cost", 0.01)
        if isinstance(unit_cost, (int, float)) and not isinstance(unit_cost, bool):
            if cost_class == "free" and float(unit_cost) != 0.0:
                errors.append("free SOURCE_ADD has nonzero unit_cost")
            if cost_class in {"metered", "paid"} and float(unit_cost) <= 0.0:
                errors.append("paid SOURCE_ADD has nonpositive unit_cost")
        bounded_number("max_calls", 1, 1, 10_000, integer=True)
        bounded_number("max_results", 1, 1, 100_000, integer=True)
        bounded_number("timeout_seconds", 30.0, 0.1, 3_600.0)
        for name, maximum in (
            ("intent_categories", 64),
            ("evidence_types", 24),
            ("binding_requirements", 32),
            ("best_for", 32),
            ("avoid_when", 32),
        ):
            if not _pipeline_string_sequence(
                field(name, ()),
                maximum=maximum,
                pattern=(
                    None
                    if name in {"intent_categories", "binding_requirements"}
                    else _PIPELINE_FEATURE_RE
                ),
            ):
                errors.append(f"field {name} is invalid")
            value = field(name, ())
            if name == "binding_requirements" and isinstance(value, tuple) and any(
                not isinstance(item, str) or len(item.strip()) > 160
                for item in value
            ):
                errors.append(f"field {name} is invalid")
        categories = field("intent_categories", ())
        if isinstance(categories, tuple) and any(
            isinstance(item, str) and len(item) > 80 for item in categories
        ):
            errors.append("field intent_categories is invalid")
        for name in ("best_for_description", "avoid_when_description"):
            value = field(name, "")
            if not isinstance(value, str) or len(" ".join(value.split())) > 500:
                errors.append(f"field {name} is invalid")
        revision = field("revision", None)
        manifest = field("manifest_sha256", None)
        if revision is not None and (
            not isinstance(revision, str)
            or not _PIPELINE_VERSION_RE.fullmatch(revision)
        ):
            errors.append("field revision is invalid")
        if manifest is not None and (
            not isinstance(manifest, str)
            or not _PIPELINE_SHA256_RE.fullmatch(manifest)
        ):
            errors.append("field manifest_sha256 is invalid")
        category_node = fields.get("category_contracts")
        if category_node is not None:
            if not isinstance(category_node, (ast.Tuple, ast.List)):
                errors.append("field category_contracts is invalid")
            else:
                for item in category_node.elts:
                    if not isinstance(item, ast.Call):
                        errors.append("field category_contracts is invalid")
                        continue
                    errors.extend(
                        f"category_contracts: {error}"
                        for error in _pipeline_constructor_semantic_errors(
                            item,
                            "SourceAddCategoryContract",
                            values=values,
                            bound_names=bound_names,
                        )
                    )

    elif constructor == "SourceAddCategoryContract":
        category = field("category", "")
        if not isinstance(category, str) or not category.strip() or len(category) > 80:
            errors.append("field category is invalid")
        for name in ("capabilities", "evidence_types", "requirements"):
            sequence = field(name, ())
            if not _pipeline_string_sequence(
                sequence,
                maximum=24,
                required=True,
            ) or any(
                not isinstance(item, str) or len(item.strip()) > 160
                for item in sequence
            ):
                errors.append(f"field {name} is invalid")
    return tuple(sorted(set(errors)))


def _pipeline_prompt_binding_errors(name: str, value: Any) -> tuple[str, ...]:
    """Validate the small reviewed discovery prompt-data schemas."""

    if name in {"_DATED_EVIDENCE_SUFFIX", "GUIDANCE_SYSTEM_PROMPT"}:
        return () if isinstance(value, str) else ("must be a string",)
    if name not in {
        "_CATEGORY_ALIASES",
        "_EVIDENCE_CHARACTERISTICS",
        "_INTENT_CATEGORY_ALIASES",
        "_INTENT_PHRASE_FAMILIES",
    }:
        return ("has no reviewed data schema",)
    if not isinstance(value, dict):
        return ("must be a string-keyed mapping",)
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            return ("contains an invalid mapping key",)
        if name in {"_CATEGORY_ALIASES", "_EVIDENCE_CHARACTERISTICS"}:
            if not isinstance(item, str):
                return ("contains an invalid mapping value",)
        elif not _pipeline_string_sequence(item, maximum=64, required=True):
            return ("contains an invalid mapping value",)
    return ()


def _source_add_registration_values(
    call: ast.Call,
    *,
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    fields, structural_errors, executable_fields = _declarative_constructor_call(
        call,
        "SourceAddRoutingRegistration",
    )
    if structural_errors or executable_fields:
        raise ValueError("SOURCE_ADD registration must use literal fields")
    output: Dict[str, Any] = {}
    for name, node in fields.items():
        if name != "category_contracts":
            value = _pipeline_literal_value(node, values)
            if value is _PIPELINE_UNRESOLVED:
                raise ValueError(f"SOURCE_ADD field {name} is unresolved")
            output[name] = value
            continue
        if not isinstance(node, (ast.Tuple, ast.List)):
            raise ValueError("SOURCE_ADD category_contracts must be a sequence")
        contracts: list[Dict[str, Any]] = []
        for item in node.elts:
            if not isinstance(item, ast.Call):
                raise ValueError("SOURCE_ADD category contract is not a constructor")
            nested, nested_errors, nested_executable = _declarative_constructor_call(
                item,
                "SourceAddCategoryContract",
            )
            if nested_errors or nested_executable:
                raise ValueError("SOURCE_ADD category contract fields differ")
            contract: Dict[str, Any] = {}
            for nested_name, nested_node in nested.items():
                nested_value = _pipeline_literal_value(nested_node, values)
                if nested_value is _PIPELINE_UNRESOLVED:
                    raise ValueError(
                        f"SOURCE_ADD category field {nested_name} is unresolved"
                    )
                contract[nested_name] = nested_value
            contracts.append(contract)
        output[name] = tuple(contracts)
    return output


def _normalize_source_add_registration_static(
    raw: Mapping[str, Any],
    *,
    contract_version: int,
) -> Dict[str, Any]:
    """Mirror the model-owned SOURCE_ADD constructor using literal data only."""

    value = dict(raw)
    provider_id = str(value.get("provider_id") or "").strip().lower()
    stage = value.get("stage")
    if not _PIPELINE_PROVIDER_ID_RE.fullmatch(provider_id):
        raise ValueError("provider_id is invalid")
    if stage not in {"candidate_acquisition", "intent_evidence"}:
        raise ValueError("stage is invalid")
    priority = value.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 10_000:
        raise ValueError("priority is invalid")
    capabilities = value.get("capabilities", ())
    if not _pipeline_string_sequence(
        capabilities,
        maximum=32,
        pattern=_PIPELINE_FEATURE_RE,
        required=True,
    ):
        raise ValueError("capabilities are invalid")
    capabilities = tuple(dict.fromkeys(item.strip() for item in capabilities))
    value.update(provider_id=provider_id, stage=stage, capabilities=capabilities)
    if contract_version == 7:
        value.setdefault("intent_categories", ())
        if "best_for" not in value:
            value.update(
                {
                    "best_for": (
                        ("icp.structured_eligible",)
                        if stage == "candidate_acquisition"
                        else ("intent.general",)
                    ),
                    "avoid_when": (),
                    "best_for_description": (
                        "Approved SOURCE_ADD company-discovery provider for "
                        "structured ICP acquisition."
                        if stage == "candidate_acquisition"
                        else "Approved SOURCE_ADD provider for company-scoped "
                        "intent-evidence discovery."
                    ),
                    "avoid_when_description": (
                        "Avoid when the consumer binding is unavailable, "
                        "unhealthy, outside its approved categories, or over "
                        "budget."
                    ),
                }
            )
        manifest = value.get("manifest_sha256")
        revision = value.get("revision")
        if (
            not isinstance(manifest, str)
            or not _PIPELINE_SHA256_RE.fullmatch(manifest)
            or revision != f"source-add-{manifest[:12]}"
        ):
            raise ValueError("v7 revision is not manifest-bound")
        expected = {
            "candidate_acquisition": {
                "priority": 80,
                "capabilities": ("candidate.provider_discovery",),
                "max_results": 100,
                "timeout_seconds": 60.0,
                "evidence_types": ("provider_database",),
            },
            "intent_evidence": {
                "priority": 35,
                "capabilities": ("intent.provider_evidence",),
                "max_results": 1,
                "timeout_seconds": 30.0,
                "evidence_types": ("external",),
            },
        }[str(stage)]
        if any(value.get(name) != expected_value for name, expected_value in expected.items()):
            raise ValueError("v7 stage contract is invalid")
        if value.get("idempotency") != "idempotent" or value.get("max_calls") != 1:
            raise ValueError("v7 execution contract is invalid")
        intent_categories = value.get("intent_categories", ())
        if not _pipeline_string_sequence(intent_categories, maximum=64):
            raise ValueError("v7 intent_categories are invalid")
        if stage == "candidate_acquisition" and intent_categories:
            raise ValueError("v7 candidate registration has intent categories")
        for name in ("evidence_types", "best_for"):
            if not _pipeline_string_sequence(
                value.get(name),
                maximum=32,
                pattern=_PIPELINE_FEATURE_RE,
                required=True,
            ):
                raise ValueError(f"v7 {name} is invalid")
        if not _pipeline_string_sequence(
            value.get("avoid_when", ()),
            maximum=32,
            pattern=_PIPELINE_FEATURE_RE,
        ):
            raise ValueError("v7 avoid_when is invalid")
        for name in ("best_for_description", "avoid_when_description"):
            text = value.get(name)
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError(f"v7 {name} is invalid")
        cost_class = value.get("cost_class")
        unit_cost = value.get("unit_cost")
        if cost_class not in {"free", "metered"} or (
            cost_class == "free" and unit_cost != 0.0
        ) or (
            cost_class == "metered"
            and (isinstance(unit_cost, bool) or not isinstance(unit_cost, (int, float)) or unit_cost <= 0)
        ):
            raise ValueError("v7 cost contract is invalid")
        return value

    execution_mode = value.get("execution_mode", "invoke")
    idempotency = value.get("idempotency", "idempotent")
    cost_class = value.get("cost_class", "metered")
    unit_cost = value.get("unit_cost", 0.01)
    max_calls = value.get("max_calls", 1)
    max_results = value.get("max_results", 1)
    timeout_seconds = value.get("timeout_seconds", 30.0)
    if execution_mode not in {"invoke", "observe", "virtual"}:
        raise ValueError("execution_mode is invalid")
    if idempotency not in {"idempotent", "resume_safe", "non_idempotent"}:
        raise ValueError("idempotency is invalid")
    if cost_class not in {"free", "metered", "paid"}:
        raise ValueError("cost_class is invalid")
    for name, number, minimum, maximum, integer in (
        ("unit_cost", unit_cost, 0.0, 10_000.0, False),
        ("max_calls", max_calls, 1, 10_000, True),
        ("max_results", max_results, 1, 100_000, True),
        ("timeout_seconds", timeout_seconds, 0.1, 3_600.0, False),
    ):
        if isinstance(number, bool) or not isinstance(number, int if integer else (int, float)):
            raise ValueError(f"{name} is invalid")
        if not math.isfinite(float(number)) or not minimum <= float(number) <= maximum:
            raise ValueError(f"{name} is invalid")
    rounded_unit_cost = round(float(unit_cost), 6)
    if (cost_class == "free" and rounded_unit_cost != 0.0) or (
        cost_class != "free" and rounded_unit_cost <= 0.0
    ):
        raise ValueError("cost_class and unit_cost differ")
    categories = tuple(
        dict.fromkeys(
            str(item or "").strip().upper()
            for item in value.get("intent_categories", ())
            if str(item or "").strip()
        )
    )
    if len(categories) > 64 or any(len(item) > 80 for item in categories):
        raise ValueError("intent_categories are invalid")
    evidence_types = value.get("evidence_types", ())
    if not _pipeline_string_sequence(
        evidence_types,
        maximum=24,
        pattern=_PIPELINE_FEATURE_RE,
    ):
        raise ValueError("evidence_types are invalid")
    evidence_types = tuple(dict.fromkeys(item.strip() for item in evidence_types))
    default_best_for = (
        ("icp.structured_eligible",)
        if stage == "candidate_acquisition"
        else ("intent.general",)
    )
    best_for = value.get("best_for") or default_best_for
    avoid_when = value.get("avoid_when", ())
    for name, sequence, required in (
        ("best_for", best_for, True),
        ("avoid_when", avoid_when, False),
    ):
        if not _pipeline_string_sequence(
            sequence,
            maximum=32,
            pattern=_PIPELINE_FEATURE_RE,
            required=required,
        ):
            raise ValueError(f"{name} is invalid")
    best_for = tuple(dict.fromkeys(item.strip() for item in best_for))
    avoid_when = tuple(dict.fromkeys(item.strip() for item in avoid_when))
    default_best = (
        "Approved SOURCE_ADD company-discovery provider for structured ICP acquisition."
        if stage == "candidate_acquisition"
        else "Approved SOURCE_ADD provider for company-scoped intent-evidence discovery."
    )
    best_description = " ".join(str(value.get("best_for_description") or default_best).split())
    avoid_description = " ".join(
        str(
            value.get("avoid_when_description")
            or "Avoid when the consumer binding is unavailable, unhealthy, outside its approved categories, or over budget."
        ).split()
    )
    if len(best_description) > 500 or len(avoid_description) > 500:
        raise ValueError("description is invalid")
    contracts: list[Dict[str, Any]] = []
    for raw_contract in value.get("category_contracts", ()):
        if not isinstance(raw_contract, Mapping) or set(raw_contract) != {
            "category",
            "capabilities",
            "evidence_types",
            "requirements",
        }:
            raise ValueError("category contract fields differ")
        category = str(raw_contract.get("category") or "").strip().upper()
        if not category or len(category) > 80:
            raise ValueError("category contract is invalid")
        contract: Dict[str, Any] = {"category": category}
        for name in ("capabilities", "evidence_types", "requirements"):
            sequence = raw_contract.get(name)
            if (
                not isinstance(sequence, tuple)
                or not sequence
                or len(sequence) > 24
                or any(not isinstance(item, str) or not item.strip() or len(item.strip()) > 160 for item in sequence)
            ):
                raise ValueError(f"category {name} is invalid")
            contract[name] = tuple(dict.fromkeys(item.strip() for item in sequence))
        contracts.append(contract)
    contract_categories = [item["category"] for item in contracts]
    if len(contract_categories) != len(set(contract_categories)):
        raise ValueError("duplicate category contract")
    if contracts and set(contract_categories) != set(categories):
        raise ValueError("category contracts do not match intent_categories")
    if any(
        not set(item["capabilities"]) <= set(capabilities)
        or not set(item["evidence_types"]) <= set(evidence_types)
        for item in contracts
    ):
        raise ValueError("category contract exceeds tool definition")
    binding_requirements = value.get("binding_requirements", ())
    if binding_requirements and (
        not isinstance(binding_requirements, tuple)
        or len(binding_requirements) > 32
        or any(not isinstance(item, str) or not item.strip() or len(item.strip()) > 160 for item in binding_requirements)
    ):
        raise ValueError("binding_requirements are invalid")
    binding_requirements = tuple(
        dict.fromkeys(item.strip() for item in binding_requirements)
    )
    normalized = {
        **value,
        "provider_id": provider_id,
        "stage": stage,
        "execution_mode": execution_mode,
        "idempotency": idempotency,
        "cost_class": cost_class,
        "unit_cost": rounded_unit_cost,
        "max_calls": max_calls,
        "max_results": max_results,
        "timeout_seconds": round(float(timeout_seconds), 3),
        "capabilities": capabilities,
        "intent_categories": categories,
        "evidence_types": evidence_types,
        "category_contracts": tuple(contracts),
        "binding_requirements": tuple(binding_requirements),
        "best_for": tuple(best_for),
        "avoid_when": tuple(avoid_when),
        "best_for_description": best_description,
        "avoid_when_description": avoid_description,
    }
    manifest = {
        "schema_version": "leadpoet.intent-source-binding-manifest:v1",
        "tool_id": ("candidate" if stage == "candidate_acquisition" else "intent")
        + ".source_add."
        + provider_id,
        "provider_id": provider_id,
        "stage": stage,
        "execution_mode": execution_mode,
        "cost_class": cost_class,
        "unit_cost": rounded_unit_cost,
        "max_calls": max_calls,
        "max_results": max_results,
        "timeout_seconds": normalized["timeout_seconds"],
        "capabilities": list(capabilities),
        "intent_categories": list(categories),
        "evidence_types": list(evidence_types),
        "category_contracts": [
            {
                "category": item["category"],
                "capabilities": list(item["capabilities"]),
                "evidence_types": list(item["evidence_types"]),
                "requirements": list(item["requirements"]),
            }
            for item in sorted(contracts, key=lambda item: str(item["category"]))
        ],
        "binding_requirements": list(binding_requirements),
    }
    digest = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    configured_digest = str(value.get("manifest_sha256") or "").strip()
    configured_revision = str(value.get("revision") or "").strip()
    if configured_digest and configured_digest != digest:
        raise ValueError("manifest_sha256 does not match binding manifest")
    if configured_revision and configured_revision != f"source-add-{digest[:12]}":
        raise ValueError("revision does not match binding manifest")
    normalized["manifest_sha256"] = digest
    normalized["revision"] = f"source-add-{digest[:12]}"
    return normalized


def _pipeline_routing_policy_errors(
    call: ast.Call,
    *,
    values: Mapping[str, Any],
) -> tuple[str, ...]:
    keyword_values = {
        str(keyword.arg): keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }
    errors: list[str] = []
    version = _pipeline_literal_value(keyword_values.get("policy_version"), values)
    if not isinstance(version, str) or not _PIPELINE_VERSION_RE.fullmatch(version):
        errors.append("policy_version is missing or invalid")
    schema = _pipeline_literal_value(keyword_values.get("schema_version"), values)
    if "schema_version" in keyword_values and schema != 1:
        errors.append("schema_version is invalid")
    steps = keyword_values.get("steps")
    if not isinstance(steps, ast.Tuple) or len(steps.elts) > 256:
        errors.append("steps are missing or out of bounds")
    return tuple(errors)


def _declarative_constructor_call(
    call: ast.Call,
    constructor: str,
) -> tuple[Dict[str, ast.AST], tuple[str, ...], tuple[str, ...]]:
    """Parse one constructor with a closed field set and no unpacking.

    ``structural_errors`` are never valid, including on a reviewed baseline.
    ``executable_fields`` are committed exactly for existing reviewed entries
    but make a newly added or changed entry ineligible for promotion. This
    keeps the three reviewed dynamic catalog expansions byte-bound while new
    routing data remains non-executable.
    """

    positional_fields = _PIPELINE_CONSTRUCTOR_POSITIONAL_FIELDS.get(constructor)
    allowed_fields = _PIPELINE_CONSTRUCTOR_FIELDS.get(constructor)
    if positional_fields is None or allowed_fields is None:
        return {}, ("unknown constructor",), ()
    structural_errors: list[str] = []
    executable_fields: list[str] = []
    values: Dict[str, ast.AST] = {}
    if not isinstance(call.func, ast.Name) or call.func.id != constructor:
        structural_errors.append("constructor binding is not direct")
    if any(isinstance(item, ast.Starred) for item in call.args):
        structural_errors.append("positional unpacking is not allowed")
    if len(call.args) > len(positional_fields):
        structural_errors.append("too many positional fields")
    for index, value in enumerate(call.args[: len(positional_fields)]):
        field = positional_fields[index]
        values[field] = value
        if not _is_pipeline_data_expression(value):
            executable_fields.append(field)
    for keyword in call.keywords:
        if keyword.arg is None:
            structural_errors.append("keyword unpacking is not allowed")
            continue
        field = str(keyword.arg)
        if field not in allowed_fields:
            structural_errors.append(f"unknown field {field}")
            continue
        if field in values:
            structural_errors.append(f"duplicate field {field}")
            continue
        values[field] = keyword.value
        if (
            constructor == "SourceAddRoutingRegistration"
            and field == "category_contracts"
        ):
            if not isinstance(keyword.value, (ast.Tuple, ast.List)):
                executable_fields.append(field)
                continue
            for item in keyword.value.elts:
                if not isinstance(item, ast.Call):
                    executable_fields.append(field)
                    continue
                _nested, nested_errors, nested_executable = (
                    _declarative_constructor_call(
                        item,
                        "SourceAddCategoryContract",
                    )
                )
                structural_errors.extend(
                    f"category_contracts: {error}" for error in nested_errors
                )
                if nested_executable:
                    executable_fields.append(field)
            continue
        if not _is_pipeline_data_expression(keyword.value):
            executable_fields.append(field)
    return (
        values,
        tuple(sorted(set(structural_errors))),
        tuple(sorted(set(executable_fields))),
    )


def _normalized_expansion(node: ast.AST, constructor: str) -> str:
    """Commit an existing dynamic expansion exactly.

    Dynamic comprehensions are part of the reviewed policy plumbing. They are
    not a writable routing-data surface. New tools enter through an approved
    literal catalog or SOURCE_ADD registration instead.
    """

    del constructor
    return ast.dump(node, include_attributes=False)


def _normalized_collection_entry(
    call: ast.Call,
    constructor: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Normalize only inert metadata fields of one declarative entry."""

    normalized = copy.deepcopy(call)
    _fields, structural_errors, executable_fields = (
        _declarative_constructor_call(call, constructor)
    )
    unsafe = list(executable_fields)
    mutable_positions: Mapping[int, str]
    mutable_keywords: frozenset[str]
    if constructor == "PolicyStep":
        mutable_positions = {2: "priority"}
        mutable_keywords = _PIPELINE_POLICY_MUTABLE_FIELDS
    elif constructor == "ToolDefinition":
        mutable_positions = {1: "revision"}
        mutable_keywords = _PIPELINE_TOOL_MUTABLE_FIELDS
    elif constructor == "_tool":
        mutable_positions = {
            2: "capability",
            3: "routing_role",
            4: "signals",
            5: "source_classes",
            6: "evidence_types",
            7: "cost_credits",
        }
        mutable_keywords = _PIPELINE_SIGNAL_TOOL_MUTABLE_FIELDS
    elif constructor == "SourceAddRoutingRegistration":
        mutable_positions = {}
        mutable_keywords = _PIPELINE_SOURCE_ADD_MUTABLE_FIELDS
    else:
        return (
            ast.dump(normalized, include_attributes=False),
            structural_errors,
            tuple(sorted(set(unsafe))),
        )

    for index, value in enumerate(normalized.args):
        field = mutable_positions.get(index)
        if field is None:
            continue
        if _is_pipeline_data_expression(value):
            normalized.args[index] = ast.Constant(value=f"<{field}>")
        else:
            unsafe.append(field)
    normalized_keywords: list[ast.keyword] = []
    for keyword in normalized.keywords:
        if keyword.arg not in mutable_keywords:
            normalized_keywords.append(keyword)
            continue
        field = str(keyword.arg)
        if _is_pipeline_data_expression(keyword.value) or (
            constructor == "SourceAddRoutingRegistration"
            and field == "category_contracts"
            and field not in executable_fields
        ):
            # Omit safe mutable named fields so adding, removing, or changing
            # one does not look like tool identity drift. Liveness-critical
            # values are committed separately below.
            continue
        unsafe.append(field)
        # Keep executable reviewed expressions exact. They can remain only if
        # the parent contains the identical expression.
        normalized_keywords.append(keyword)
    normalized.keywords = normalized_keywords
    return (
        ast.dump(normalized, include_attributes=False),
        structural_errors,
        tuple(sorted(set(unsafe))),
    )


def _membership_record(
    call: ast.Call,
    *,
    relative_path: str,
    container: str,
    constructor: str,
    constants: Mapping[str, str],
    static_values: Mapping[str, Any],
    bound_names: frozenset[str],
) -> Dict[str, Any] | None:
    call_fields, _shape_errors, _executable_fields = (
        _declarative_constructor_call(call, constructor)
    )
    entry_shape, invalid_call_shape, unsafe_mutable_fields = (
        _normalized_collection_entry(
        call,
        constructor,
        )
    )
    semantic_errors = _pipeline_constructor_semantic_errors(
        call,
        constructor,
        values=static_values,
        bound_names=bound_names,
    )
    if constructor == "ToolDefinition":
        kind = "definition"
        tool_node = _call_argument(call, keyword="tool_id", position=0)
        stage_node = _call_argument(call, keyword="stages", position=2)
        stages = _resolve_stages(stage_node, constants, sequence=True)
        tool_id = _resolve_string(tool_node, constants)
    elif constructor == "PolicyStep":
        kind = "policy"
        tool_node = _call_argument(call, keyword="tool_id", position=0)
        stage_node = _call_argument(call, keyword="stage", position=1)
        stages = _resolve_stages(stage_node, constants, sequence=False)
        tool_id = _resolve_string(tool_node, constants)
    elif constructor == "_tool":
        kind = "definition"
        tool_node = _call_argument(call, keyword="tool_id", position=0)
        stage_node = _call_argument(call, keyword="stage", position=1)
        stages = _resolve_stages(stage_node, constants, sequence=False)
        tool_id = _resolve_string(tool_node, constants)
    elif constructor == "SourceAddRoutingRegistration":
        kind = "registration"
        provider_node = _call_argument(call, keyword="provider_id", position=0)
        stage_node = _call_argument(call, keyword="stage", position=1)
        stages = _resolve_stages(stage_node, constants, sequence=False)
        provider_id = _resolve_string(provider_node, constants)
        tool_id = ""
        if provider_id and len(stages) == 1:
            prefix = (
                "candidate"
                if stages[0] == "candidate_acquisition"
                else "intent" if stages[0] == "intent_evidence" else ""
            )
            if prefix:
                tool_id = f"{prefix}.source_add.{provider_id}"
        tool_node = provider_node
    else:
        return None
    liveness_fields = {
        field: (
            ast.dump(call_fields[field], include_attributes=False)
            if field in call_fields
            else "<default>"
        )
        for field in sorted(_PIPELINE_LIVENESS_FIELDS.get(constructor, ()))
    }
    liveness_payload = json.dumps(
        {
            "constructor": constructor,
            "tool_id": tool_id,
            "stages": list(stages),
            "fields": liveness_fields,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "path": relative_path,
        "container": container,
        "kind": kind,
        "constructor": constructor,
        "tool_expr": (
            ast.dump(tool_node, include_attributes=False)
            if tool_node is not None
            else "<missing>"
        ),
        "stage_expr": (
            ast.dump(stage_node, include_attributes=False)
            if stage_node is not None
            else "<missing>"
        ),
        "tool_id": tool_id,
        "stages": list(stages),
        "entry_shape": entry_shape,
        "invalid_call_shape": list((*invalid_call_shape, *semantic_errors)),
        "declared_fields": sorted(call_fields),
        "liveness_key": "sha256:" + hashlib.sha256(liveness_payload).hexdigest(),
        "unsafe_mutable_fields": list(unsafe_mutable_fields),
    }


def _module_protected_binding_violations(
    tree: ast.Module,
    *,
    relative_path: str,
    wrapper_names: set[str],
) -> List[str]:
    """Reject rebinding and reflection that can redirect protected wrappers."""

    protected_names = {
        "compile_route",
        "RouteContext",
        "RoutingPolicy",
        "PolicyStep",
        "ToolDefinition",
        *set(_PIPELINE_STAGE_VALUES),
        *wrapper_names,
    }
    violations: List[str] = []
    for wrapper_name in sorted(wrapper_names):
        nodes = _top_level_binding_nodes(tree, wrapper_name)
        if (
            len(nodes) != 1
            or not isinstance(nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            violations.append(
                f"router entrypoint binding drift {relative_path}:{wrapper_name}"
            )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)) and set(
            node.names
        ) & protected_names:
            violations.append(f"protected router global mutation {relative_path}")
            break
        if (
            isinstance(node, ast.Name)
            and node.id in protected_names
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            violations.append(f"protected router binding mutation {relative_path}")
            break
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and (
                node.attr in protected_names
                or (
                    isinstance(node.value, ast.Name)
                    and node.value.id in protected_names
                )
            )
        ):
            violations.append(f"protected router attribute mutation {relative_path}")
            break
        if isinstance(node, ast.Name) and node.id in _PIPELINE_REFLECTION_NAMES:
            violations.append(f"protected router reflection {relative_path}")
            break
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _PIPELINE_REFLECTION_NAMES
        ):
            violations.append(f"protected router reflection {relative_path}")
            break
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in _PIPELINE_REFLECTION_NAMES
        ):
            violations.append(f"protected router reflective name {relative_path}")
            break
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"setattr", "delattr"}
        ):
            violations.append(f"protected router reflective mutation {relative_path}")
            break
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and (
                isinstance(node.args[0], ast.Name)
                and node.args[0].id in protected_names
                or isinstance(node.args[1], ast.Constant)
                and node.args[1].value in protected_names
            )
        ):
            violations.append(f"protected router reflective read {relative_path}")
            break
    return violations


def _assigned_name_and_value(node: ast.AST) -> tuple[str, ast.AST | None]:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id, node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return "", None


def _assignment_with_sentinel(node: ast.AST, value: str) -> ast.AST:
    normalized = copy.deepcopy(node)
    if isinstance(normalized, ast.Assign):
        normalized.value = ast.Constant(value=value)
    elif isinstance(normalized, ast.AnnAssign):
        normalized.value = ast.Constant(value=value)
    return normalized


def _collection_function_shell(function: ast.AST, kind: str) -> ast.AST:
    """Keep the callable and outer constructor; redact only its data list."""

    normalized = copy.deepcopy(function)
    body = _without_docstring(normalized)
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return normalized
    returned = body[0]
    if kind == "definitions":
        returned.value = ast.Constant(value="<routing-definitions>")
        return normalized
    value = returned.value
    if not isinstance(value, ast.Call) or _call_name(value) != "RoutingPolicy":
        return normalized
    step_keywords = [item for item in value.keywords if item.arg == "steps"]
    if len(step_keywords) == 1:
        step_keywords[0].value = ast.Constant(value="<routing-policy-steps>")
    return normalized


def _normalized_prompt_output_expression(node: ast.AST) -> ast.AST:
    """Redact only rendered prompt fragments, never selector expressions.

    Formatted values, mapping keys, call arguments, comparison values, and
    lookup keys remain exact AST. This lets loops revise text shown to a
    provider without turning a prompt edit into a control-flow or interface
    edit.
    """

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<prompt-text>"), node)
        return copy.deepcopy(node)
    if isinstance(node, ast.JoinedStr):
        normalized_values: list[ast.AST] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                normalized_values.append(
                    ast.copy_location(ast.Constant(value="<prompt-text>"), value)
                )
            else:
                # In particular, keep every FormattedValue expression exact.
                normalized_values.append(copy.deepcopy(value))
        normalized = copy.deepcopy(node)
        normalized.values = normalized_values
        return normalized
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add):
            return copy.deepcopy(node)
        normalized = copy.deepcopy(node)
        normalized.left = _normalized_prompt_output_expression(node.left)
        normalized.right = _normalized_prompt_output_expression(node.right)
        return normalized
    if isinstance(node, ast.BoolOp):
        # String truthiness controls which operands execute. Keep the complete
        # expression exact instead of treating it as rendered prompt text.
        return copy.deepcopy(node)
    if isinstance(node, ast.IfExp):
        normalized = copy.deepcopy(node)
        normalized.test = copy.deepcopy(node.test)
        normalized.body = _normalized_prompt_output_expression(node.body)
        normalized.orelse = _normalized_prompt_output_expression(node.orelse)
        return normalized
    if isinstance(node, (ast.List, ast.Tuple)):
        normalized = copy.deepcopy(node)
        normalized.elts = [
            _normalized_prompt_output_expression(value) for value in node.elts
        ]
        return normalized
    if isinstance(node, ast.Dict):
        normalized = copy.deepcopy(node)
        normalized.keys = [copy.deepcopy(value) for value in node.keys]
        normalized.values = [
            _normalized_prompt_output_expression(value) for value in node.values
        ]
        return normalized
    # Calls, names, attributes, subscripts, and all other executable
    # expressions are control/interface data and remain byte-semantic.
    return copy.deepcopy(node)


def _simple_assignment_target(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _prompt_call_sink(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
    ):
        return f"{call.func.value.id}.{call.func.attr}"
    return ""


class _PromptStringNormalizer(ast.NodeTransformer):
    """Normalize a reviewed function's explicit prompt-output positions."""

    def __init__(
        self,
        *,
        local_bindings: frozenset[str],
        call_sinks: frozenset[str],
    ) -> None:
        self._local_bindings = local_bindings
        self._call_sinks = call_sinks

    def visit_Assign(self, node: ast.Assign) -> ast.AST:  # noqa: N802
        normalized = copy.deepcopy(node)
        if (
            len(node.targets) == 1
            and _simple_assignment_target(node.targets[0]) in self._local_bindings
        ):
            normalized.value = _normalized_prompt_output_expression(node.value)
            return normalized
        return self.generic_visit(normalized)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:  # noqa: N802
        normalized = copy.deepcopy(node)
        if (
            _simple_assignment_target(node.target) in self._local_bindings
            and node.value is not None
        ):
            normalized.value = _normalized_prompt_output_expression(node.value)
            return normalized
        return self.generic_visit(normalized)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:  # noqa: N802
        normalized = copy.deepcopy(node)
        if _simple_assignment_target(node.target) in self._local_bindings:
            normalized.value = _normalized_prompt_output_expression(node.value)
            return normalized
        return self.generic_visit(normalized)

    def visit_Return(self, node: ast.Return) -> ast.AST:  # noqa: N802
        normalized = copy.deepcopy(node)
        if node.value is not None:
            normalized.value = _normalized_prompt_output_expression(node.value)
        return normalized

    def visit_Expr(self, node: ast.Expr) -> ast.AST:  # noqa: N802
        normalized = copy.deepcopy(node)
        if (
            isinstance(node.value, ast.Call)
            and _prompt_call_sink(node.value) in self._call_sinks
        ):
            normalized.value.args = [
                _normalized_prompt_output_expression(value)
                for value in node.value.args
            ]
            normalized.value.keywords = [copy.deepcopy(value) for value in node.value.keywords]
            return normalized
        return self.generic_visit(normalized)


def _module_edit_surface_projection(
    tree: ast.Module,
    *,
    relative_path: str,
) -> tuple[str, list[Dict[str, str]], List[str]]:
    """Hash all code except the reviewed declarative edit surface."""

    collection_functions = _PIPELINE_COLLECTION_FUNCTIONS.get(
        relative_path,
        {},
    )
    prompt_functions = _PIPELINE_PROMPT_FUNCTIONS.get(
        relative_path,
        frozenset(),
    )
    prompt_bindings = _PIPELINE_PROMPT_BINDINGS.get(
        relative_path,
        frozenset(),
    )
    normalized_body: list[ast.stmt] = []
    tool_constants: list[Dict[str, str]] = []
    violations: List[str] = []
    seen_tool_constants: set[str] = set()
    imported_bindings = {
        name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in _module_scope_bindings(node)
    }
    for node in tree.body:
        name, value = _assigned_name_and_value(node)
        if name.startswith("TOOL_") and name != "TOOL_CATALOG":
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                violations.append(
                    "routing tool constant is not a plain string assignment "
                    f"{relative_path}:{name}"
                )
                normalized_body.append(copy.deepcopy(node))
                continue
            if name in imported_bindings:
                violations.append(
                    f"routing tool constant shadows import {relative_path}:{name}"
                )
            if name in seen_tool_constants:
                violations.append(
                    f"duplicate routing tool constant {relative_path}:{name}"
                )
            seen_tool_constants.add(name)
            tool_constants.append(
                {"path": relative_path, "name": name, "value": value.value}
            )
            continue
        if (
            relative_path in _PIPELINE_ROUTING_MODULES
            and name in {
                "DEFAULT_CATALOG_VERSION",
                "DEFAULT_POLICY_VERSION",
                "RUNTIME_CATALOG_VERSION",
                "RUNTIME_POLICY_VERSION",
            }
        ):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and _PIPELINE_VERSION_RE.fullmatch(str(value.value))
            ):
                violations.append(
                    f"routing version is not a plain literal {relative_path}:{name}"
                )
            normalized_body.append(_assignment_with_sentinel(node, "<version>"))
            continue
        if (
            relative_path == "sourcing_model/routing/runtime.py"
            and name == "SOURCE_ADD_ROUTING_REGISTRATIONS"
        ):
            if not isinstance(node, ast.Assign):
                violations.append(
                    f"SOURCE_ADD registry is not a plain assignment {relative_path}:{name}"
                )
            normalized_body.append(
                _assignment_with_sentinel(node, "<approved-source-add-registry>")
            )
            continue
        if (
            relative_path == "sourcing_model/scrapingdog_signal_contract.py"
            and name == "TOOL_CATALOG"
        ):
            if not isinstance(node, ast.Assign):
                violations.append(
                    f"tool catalog is not a plain assignment {relative_path}:{name}"
                )
            normalized_body.append(
                _assignment_with_sentinel(node, "<tool-catalog>")
            )
            continue
        if name in prompt_bindings:
            if not isinstance(node, ast.Assign):
                violations.append(
                    f"prompt binding is not a plain assignment {relative_path}:{name}"
                )
                normalized_body.append(copy.deepcopy(node))
                continue
            string_only = name in _PIPELINE_LITERAL_STRING_BINDINGS.get(
                relative_path,
                frozenset(),
            )
            prompt_value = _pipeline_literal_value(
                value,
                _pipeline_static_values_before(tree, node),
            )
            semantic_errors = _pipeline_prompt_binding_errors(name, prompt_value)
            if (
                value is None
                or not _is_pipeline_data_expression(value)
                or (
                    string_only
                    and not (
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    )
                )
                or semantic_errors
            ):
                violations.append(
                    f"prompt binding is not reviewed data {relative_path}:{name}"
                )
                normalized_body.append(copy.deepcopy(node))
            else:
                normalized_body.append(
                    _assignment_with_sentinel(node, "<prompt-data>")
                )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            collection_spec = collection_functions.get(node.name)
            if collection_spec is not None:
                normalized_body.append(
                    _collection_function_shell(node, collection_spec[0])
                )
                continue
            if node.name in prompt_functions:
                local_bindings = _PIPELINE_PROMPT_LOCAL_BINDINGS.get(
                    relative_path,
                    {},
                ).get(node.name, frozenset())
                call_sinks = _PIPELINE_PROMPT_CALL_SINKS.get(
                    relative_path,
                    {},
                ).get(node.name, frozenset())
                normalized_body.append(
                    _PromptStringNormalizer(
                        local_bindings=local_bindings,
                        call_sinks=call_sinks,
                    ).visit(copy.deepcopy(node))
                )
                continue
        normalized_body.append(copy.deepcopy(node))
    normalized = ast.Module(body=normalized_body, type_ignores=tree.type_ignores)
    return (
        _node_hash(normalized),
        sorted(tool_constants, key=lambda item: (item["path"], item["name"])),
        violations,
    )


def _local_route_context_bindings(function: ast.AST) -> Dict[str, ast.Call]:
    """Resolve single-assignment local ``RouteContext`` values only."""

    store_counts: Counter[str] = Counter()
    candidates: Dict[str, ast.Call] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(
            node.ctx,
            (ast.Store, ast.Del),
        ):
            store_counts[node.id] += 1
        elif isinstance(node, ast.arg):
            store_counts[node.arg] += 1
        value: ast.AST | None = None
        name = ""
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target,
            ast.Name,
        ):
            name = node.target.id
            value = node.value
        if (
            name
            and isinstance(value, ast.Call)
            and _call_name(value) == "RouteContext"
        ):
            candidates[name] = value
    return {
        name: value
        for name, value in candidates.items()
        if store_counts[name] == 1
    }


def _compile_route_context(
    call: ast.Call,
    *,
    local_contexts: Mapping[str, ast.Call],
) -> ast.Call | None:
    context: ast.AST | None = call.args[2] if len(call.args) >= 3 else None
    keyword_context = [
        keyword.value for keyword in call.keywords if keyword.arg == "context"
    ]
    if keyword_context:
        if context is not None or len(keyword_context) != 1:
            return None
        context = keyword_context[0]
    if isinstance(context, ast.Name):
        context = local_contexts.get(context.id)
    if (
        not isinstance(context, ast.Call)
        or _call_name(context) != "RouteContext"
    ):
        return None
    return context


def _router_stage_binding_violations(
    tree: ast.Module,
    *,
    relative_path: str,
    expected_stage_bindings: Mapping[str, str],
    required_functions: set[str],
) -> List[str]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected_bindings = {
        name: stage
        for name, stage in expected_stage_bindings.items()
        if name in required_functions
    }
    violations: List[str] = _module_protected_binding_violations(
        tree,
        relative_path=relative_path,
        wrapper_names=set(expected_bindings),
    )
    if not _has_single_unaliased_relative_import(
        tree, module="compiler", name="compile_route"
    ) or not _has_single_unaliased_relative_import(
        tree, module="contracts", name="RouteContext"
    ):
        violations.append(
            f"router compiler binding drift {relative_path}"
        )
    for function_name, expected_stage in expected_bindings.items():
        function = functions.get(function_name)
        if function is None:
            continue
        if not _has_single_unaliased_relative_import(
            tree, module="contracts", name=expected_stage
        ):
            violations.append(
                "router stage constant binding drift "
                f"{relative_path}:{function_name}"
            )
            continue
        guarded_names = {"compile_route", "RouteContext", expected_stage}
        if _function_rebinds(function, guarded_names):
            violations.append(
                "router stage binding drift "
                f"{relative_path}:{function_name}: "
                f"expected {expected_stage}"
            )
            continue
        compile_route_calls: list[ast.Call] = []
        cross_stage_calls = False
        local_contexts = _local_route_context_bindings(function)
        route_context_calls: list[ast.Call] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            callee = _call_name(node)
            if callee == "compile_route":
                compile_route_calls.append(node)
            if callee == "RouteContext":
                route_context_calls.append(node)
            if callee in expected_bindings and callee != function_name:
                cross_stage_calls = True
        observed_stages: list[str] = []
        for call in compile_route_calls:
            context = _compile_route_context(
                call,
                local_contexts=local_contexts,
            )
            if context is None:
                observed_stages.append("<dynamic>")
                continue
            stage_keywords = [
                keyword.value
                for keyword in context.keywords
                if keyword.arg == "stage"
            ]
            if len(stage_keywords) != 1:
                observed_stages.append("<dynamic>")
                continue
            value = stage_keywords[0]
            observed_stages.append(
                value.id if isinstance(value, ast.Name) else "<dynamic>"
            )
        context_stages: list[str] = []
        for context in route_context_calls:
            stage_keywords = [
                keyword.value
                for keyword in context.keywords
                if keyword.arg == "stage"
            ]
            if len(stage_keywords) != 1:
                context_stages.append("<dynamic>")
                continue
            stage_value = stage_keywords[0]
            context_stages.append(
                stage_value.id
                if isinstance(stage_value, ast.Name)
                else "<dynamic>"
            )
        if (
            not compile_route_calls
            or not route_context_calls
            or cross_stage_calls
            or any(stage != expected_stage for stage in observed_stages)
            or any(stage != expected_stage for stage in context_stages)
        ):
            violations.append(
                "router stage binding drift "
                f"{relative_path}:{function_name}: "
                f"expected {expected_stage}"
            )
    return violations


def _assigned_value(node: ast.AST, name: str) -> ast.AST | None:
    if (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ):
        return node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    ):
        return node.value
    return None


def _pipeline_membership_projection(
    trees: Mapping[str, ast.Module],
    *,
    contract_id: str,
) -> tuple[Dict[str, Any], List[str]]:
    """Project tool ownership and policy membership without executing source."""

    contract_version_match = re.search(r"-v(\d+)$", contract_id)
    contract_version = (
        int(contract_version_match.group(1)) if contract_version_match else 0
    )
    allowed_routing_stages = {
        "candidate_acquisition",
        "intent_evidence",
    }
    if contract_version >= 11:
        allowed_routing_stages.add("candidate_enrichment")
    if contract_version >= 12:
        allowed_routing_stages.add("contact_acquisition")

    records: list[Dict[str, Any]] = []
    violations: List[str] = []
    coverage: Dict[str, set[str]] = {
        "definitions": set(),
        "policies": set(),
    }
    policy_expansion_stages: set[str] = set()
    ownership: Dict[str, Dict[str, Dict[str, list[str]]]] = {}
    definition_constructors: Dict[str, Dict[str, str]] = {}
    direct_membership_counts: Counter[tuple[str, str, str]] = Counter()
    logical_membership_paths: Dict[tuple[str, str], set[str]] = {}

    def add_record(record: Dict[str, Any]) -> None:
        records.append(record)
        invalid_call_shape = [
            str(item) for item in record.get("invalid_call_shape") or ()
        ]
        if invalid_call_shape:
            violations.append(
                "routing constructor shape drift "
                f"{record.get('path')}:{record.get('container')}: "
                + ", ".join(invalid_call_shape)
            )
        constructor = str(record.get("constructor") or "")
        observed_stages = {str(item) for item in record.get("stages") or ()}
        stage_vocabulary = (
            {
                "candidate_acquisition",
                "candidate_enrichment",
                "intent_evidence",
            }
            if constructor == "_tool" and contract_version >= 11
            else {"candidate_acquisition", "intent_evidence"}
            if constructor == "_tool"
            else {"candidate_acquisition", "intent_evidence"}
            if constructor == "SourceAddRoutingRegistration"
            else allowed_routing_stages
        )
        if observed_stages - stage_vocabulary:
            violations.append(
                f"routing stage is not available in {contract_id}: "
                f"{record.get('path')}:{record.get('container')}"
            )
        if (
            constructor == "_tool"
            and contract_version < 11
            and any(
                name in set(record.get("declared_fields") or ())
                for name in {
                    "identity_anchor",
                    "claim_visibility",
                    "identity_policy",
                    "pagination_mode",
                }
            )
        ):
            violations.append(
                f"routing constructor fields are not available in {contract_id}: "
                f"{record.get('path')}:{record.get('container')}"
            )
        if constructor == "SourceAddRoutingRegistration":
            declared_fields = set(record.get("declared_fields") or ())
            if contract_version == 7 and frozenset(
                declared_fields
            ) not in _PIPELINE_SOURCE_ADD_V7_ACCEPTED_FIELD_SETS:
                violations.append(
                    "SOURCE_ADD v7 registration fields differ from the contract"
                )
        kind = str(record.get("kind") or "")
        tool_id = str(record.get("tool_id") or "")
        stages = [str(item) for item in record.get("stages") or ()]
        path = str(record.get("path") or "")
        if path and tool_id:
            direct_key = (path, kind, tool_id)
            direct_membership_counts[direct_key] += 1
            if direct_membership_counts[direct_key] > 1:
                violations.append(
                    f"duplicate routing membership {path}:{kind}:{tool_id}"
                )
            logical_membership_paths.setdefault((kind, tool_id), set()).add(path)
        if kind == "registration":
            for coverage_kind in ("definitions", "policies"):
                coverage[coverage_kind].update(stages)
            return
        if kind not in {"definition", "policy"}:
            return
        plural = "definitions" if kind == "definition" else "policies"
        coverage[plural].update(stages)
        if not tool_id or len(stages) != 1:
            violations.append(
                "unresolved routing membership "
                f"{record.get('path')}:{record.get('container')}"
            )
            return
        path_doc = ownership.setdefault(
            path,
            {"definitions": {}, "policies": {}},
        )
        kind_doc = path_doc[plural]
        observed = set(kind_doc.get(tool_id, ()))
        observed.update(stages)
        kind_doc[tool_id] = sorted(observed)
        if kind == "definition":
            definition_constructors.setdefault(path, {})[tool_id] = str(
                record.get("constructor") or ""
            )

    for relative_path, functions_spec in _PIPELINE_COLLECTION_FUNCTIONS.items():
        tree = trees.get(relative_path)
        if tree is None:
            continue
        constants = dict(_PIPELINE_IMPORTED_TOOL_IDS)
        imported_intent_tree = trees.get("sourcing_model/scrapingdog_intent.py")
        if imported_intent_tree is not None:
            constants.update(_simple_string_constants(imported_intent_tree))
        constants.update(_simple_string_constants(tree))
        static_values = _pipeline_static_values_for_module(tree, trees=trees)
        bound_names = frozenset(
            name
            for top_level_node in tree.body
            for name in _module_scope_bindings(top_level_node)
        )
        for function_name, (kind, constructor) in functions_spec.items():
            functions = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
            ]
            if len(functions) != 1:
                violations.append(
                    f"routing collection binding drift {relative_path}:{function_name}"
                )
                continue
            items, error = _collection_items(functions[0], kind=kind)
            if error:
                violations.append(
                    f"routing collection shape drift {relative_path}:"
                    f"{function_name}: {error}"
                )
                continue
            if kind == "policy":
                returned = _without_docstring(functions[0])[0]
                assert isinstance(returned, ast.Return)
                assert isinstance(returned.value, ast.Call)
                policy_errors = _pipeline_routing_policy_errors(
                    returned.value,
                    values=static_values,
                )
                if policy_errors:
                    violations.append(
                        f"routing policy semantic drift {relative_path}:"
                        f"{function_name}: " + ", ".join(policy_errors)
                    )
            for index, item in enumerate(items):
                container = function_name
                if isinstance(item, ast.Call) and _call_name(item) == constructor:
                    record = _membership_record(
                        item,
                        relative_path=relative_path,
                        container=container,
                        constructor=constructor,
                        constants=constants,
                        static_values=static_values,
                        bound_names=bound_names,
                    )
                    if record is not None:
                        add_record(record)
                    continue
                if isinstance(item, ast.Starred):
                    records.append(
                        {
                            "path": relative_path,
                            "container": container,
                            "kind": "expansion",
                            "constructor": constructor,
                            "projection": _normalized_expansion(item, constructor),
                            "tool_id": "",
                            "stages": [],
                        }
                    )
                    nested_calls = [
                        node
                        for node in ast.walk(item)
                        if isinstance(node, ast.Call)
                        and _call_name(node) == constructor
                    ]
                    for nested in nested_calls:
                        if constructor == "PolicyStep":
                            stage_node = _call_argument(
                                nested,
                                keyword="stage",
                                position=1,
                            )
                            expansion_stages = _resolve_stages(
                                stage_node,
                                constants,
                                sequence=False,
                            )
                            coverage["policies"].update(expansion_stages)
                            policy_expansion_stages.update(expansion_stages)
                    continue
                violations.append(
                    f"routing collection item drift {relative_path}:{container}"
                )

    signal_path = "sourcing_model/scrapingdog_signal_contract.py"
    signal_tree = trees.get(signal_path)
    if signal_tree is not None:
        constants = _simple_string_constants(signal_tree)
        bound_names = frozenset(
            name
            for top_level_node in signal_tree.body
            for name in _module_scope_bindings(top_level_node)
        )
        catalog_nodes = _top_level_binding_nodes(signal_tree, "TOOL_CATALOG")
        if len(catalog_nodes) != 1:
            violations.append(f"routing tool catalog binding drift {signal_path}")
        else:
            static_values = _pipeline_static_values_for_module_before(
                signal_tree,
                catalog_nodes[0],
                trees=trees,
            )
            catalog_value = _assigned_value(catalog_nodes[0], "TOOL_CATALOG")
            if not isinstance(catalog_value, ast.Tuple):
                violations.append(f"routing tool catalog shape drift {signal_path}")
            else:
                for index, item in enumerate(catalog_value.elts):
                    if not isinstance(item, ast.Call) or _call_name(item) != "_tool":
                        violations.append(
                            f"routing tool catalog item drift {signal_path}:"
                            f"TOOL_CATALOG[{index}]"
                        )
                        continue
                    record = _membership_record(
                        item,
                        relative_path=signal_path,
                        container="TOOL_CATALOG",
                        constructor="_tool",
                        constants=constants,
                        static_values=static_values,
                        bound_names=bound_names,
                    )
                    if record is not None:
                        add_record(record)

    runtime_path = "sourcing_model/routing/runtime.py"
    runtime_tree = trees.get(runtime_path)
    if runtime_tree is not None:
        constants = dict(_PIPELINE_IMPORTED_TOOL_IDS)
        imported_intent_tree = trees.get("sourcing_model/scrapingdog_intent.py")
        if imported_intent_tree is not None:
            constants.update(_simple_string_constants(imported_intent_tree))
        constants.update(_simple_string_constants(runtime_tree))
        bound_names = frozenset(
            name
            for top_level_node in runtime_tree.body
            for name in _module_scope_bindings(top_level_node)
        )
        registry_nodes = _top_level_binding_nodes(
            runtime_tree, "SOURCE_ADD_ROUTING_REGISTRATIONS"
        )
        if len(registry_nodes) != 1:
            violations.append(
                f"SOURCE_ADD routing registry binding drift {runtime_path}"
            )
        else:
            static_values = _pipeline_static_values_for_module_before(
                runtime_tree,
                registry_nodes[0],
                trees=trees,
            )
            registry_value = _assigned_value(
                registry_nodes[0], "SOURCE_ADD_ROUTING_REGISTRATIONS"
            )
            if not isinstance(registry_value, ast.Tuple):
                violations.append(
                    f"SOURCE_ADD routing registry shape drift {runtime_path}"
                )
            else:
                for index, item in enumerate(registry_value.elts):
                    if (
                        not isinstance(item, ast.Call)
                        or _call_name(item) != "SourceAddRoutingRegistration"
                    ):
                        violations.append(
                            f"SOURCE_ADD routing membership drift {runtime_path}:"
                            f"SOURCE_ADD_ROUTING_REGISTRATIONS[{index}]"
                        )
                        continue
                    record = _membership_record(
                        item,
                        relative_path=runtime_path,
                        container="SOURCE_ADD_ROUTING_REGISTRATIONS",
                        constructor="SourceAddRoutingRegistration",
                        constants=constants,
                        static_values=static_values,
                        bound_names=bound_names,
                    )
                    if record is not None:
                        try:
                            _normalize_source_add_registration_static(
                                _source_add_registration_values(
                                    item,
                                    values=static_values,
                                ),
                                contract_version=(7 if contract_version == 7 else 8),
                            )
                        except (TypeError, ValueError) as exc:
                            violations.append(
                                "SOURCE_ADD registration semantic drift "
                                f"{runtime_path}: {exc}"
                            )
                        add_record(record)

    for path, kinds in ownership.items():
        for kind, tools in kinds.items():
            for tool_id, stages in tools.items():
                if len(stages) != 1:
                    violations.append(
                        f"tool stage ownership is not singular {path}:{kind}:{tool_id}"
                    )
        definitions = kinds["definitions"]
        policies = kinds["policies"]
        for tool_id in sorted(set(definitions) & set(policies)):
            if definitions[tool_id] != policies[tool_id]:
                violations.append(
                    f"tool policy stage differs from definition {path}:{tool_id}"
                )

    normalized_ownership = {
        path: {
            kind: {tool_id: list(stages) for tool_id, stages in sorted(tools.items())}
            for kind, tools in sorted(kinds.items())
        }
        for path, kinds in sorted(ownership.items())
    }
    global_ownership: Dict[str, set[str]] = {}
    for record in records:
        tool_id = str(record.get("tool_id") or "")
        stages = {str(item) for item in record.get("stages") or ()}
        if tool_id and stages:
            global_ownership.setdefault(tool_id, set()).update(stages)
    for tool_id, stages in sorted(global_ownership.items()):
        if len(stages) != 1:
            violations.append(
                f"global tool stage ownership is not singular: {tool_id}"
            )
    for logical_key, paths in sorted(logical_membership_paths.items()):
        if len(paths) <= 1:
            continue
        if paths != set(
            _PIPELINE_REVIEWED_SHARED_TOOL_OWNERSHIP.get(
                logical_key,
                frozenset(),
            )
        ):
            violations.append(
                "routing tool has unreviewed cross-catalog ownership: "
                f"{logical_key[0]}:{logical_key[1]}"
            )
    return (
        {
            "membership_records": sorted(
                records,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
            "tool_stage_ownership": normalized_ownership,
            "global_tool_stage_ownership": {
                tool_id: sorted(stages)
                for tool_id, stages in sorted(global_ownership.items())
            },
            "logical_tool_ownership": {
                f"{kind}:{tool_id}": sorted(paths)
                for (kind, tool_id), paths in sorted(
                    logical_membership_paths.items()
                )
            },
            "definition_constructors": {
                path: dict(sorted(items.items()))
                for path, items in sorted(definition_constructors.items())
            },
            "stage_coverage": {
                kind: sorted(stages) for kind, stages in sorted(coverage.items())
            },
            "policy_expansion_stages": sorted(policy_expansion_stages),
        },
        violations,
    )


def verify_sourcing_pipeline_structure(root: Path) -> List[str]:
    """Validate immutable router-to-stage bindings for a reviewed model tree."""

    root = Path(root)
    snapshot = resolve_reviewed_consumer_snapshot(root)
    if snapshot is None:
        return ["sourcing pipeline contract/parity pair is not reviewed"]
    violations: List[str] = []
    trees: Dict[str, ast.Module] = {}
    edit_surface_modules = _pipeline_edit_surface_modules(snapshot, root=root)
    for relative_path in edit_surface_modules:
        try:
            trees[relative_path] = ast.parse((root / relative_path).read_bytes())
        except SyntaxError as exc:
            violations.append(
                f"unparseable pipeline module {relative_path}: "
                f"{exc.msg} (line {exc.lineno})"
            )
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            violations.append(
                f"unreadable pipeline module {relative_path}: "
                f"{type(exc).__name__}"
            )
    for relative_path, stage_bindings in (
        ("sourcing_model/routing/defaults.py", _DEFAULT_ROUTER_STAGE_BINDINGS),
        ("sourcing_model/routing/runtime.py", _ROUTER_STAGE_BINDINGS),
    ):
        tree = trees.get(relative_path)
        if tree is None:
            continue
        required_functions = set(
            snapshot["contract"].get("functions", {}).get(relative_path, {})
        )
        violations.extend(
            _router_stage_binding_violations(
                tree,
                relative_path=relative_path,
                expected_stage_bindings=stage_bindings,
                required_functions=required_functions,
            )
        )
    if len(trees) == len(edit_surface_modules):
        _projection, membership_violations = _pipeline_membership_projection(
            trees,
            contract_id=str(snapshot["contract"]["contract_id"]),
        )
        violations.extend(membership_violations)
        for relative_path in edit_surface_modules:
            _hash, _constants, edit_surface_violations = (
                _module_edit_surface_projection(
                    trees[relative_path],
                    relative_path=relative_path,
                )
            )
            violations.extend(edit_surface_violations)
    return violations


def sourcing_pipeline_structure_document(root: Path) -> Dict[str, Any]:
    """Commit the reviewed router handoff wrappers without executing source."""

    root = Path(root)
    snapshot = resolve_reviewed_consumer_snapshot(root)
    if snapshot is None:
        raise ValueError("sourcing pipeline contract/parity pair is not reviewed")
    violations = verify_sourcing_pipeline_structure(root)
    if violations:
        raise ValueError("; ".join(violations))
    protected_hashes: Dict[str, str] = {}
    edit_surface_modules = _pipeline_edit_surface_modules(snapshot, root=root)
    trees = {
        relative_path: ast.parse((root / relative_path).read_bytes())
        for relative_path in edit_surface_modules
    }
    for relative_path, stage_bindings in (
        ("sourcing_model/routing/defaults.py", _DEFAULT_ROUTER_STAGE_BINDINGS),
        ("sourcing_model/routing/runtime.py", _ROUTER_STAGE_BINDINGS),
    ):
        tree = trees[relative_path]
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        required_functions = set(
            snapshot["contract"].get("functions", {}).get(relative_path, {})
        )
        for name in sorted(set(stage_bindings) & required_functions):
            node = functions.get(name)
            if node is None:
                raise ValueError(f"missing pipeline router function: {name}")
            protected_hashes[f"{relative_path}:{name}"] = _node_hash(node)
    for relative_path, names in _PIPELINE_EXACT_SYMBOLS.items():
        tree = trees[relative_path]
        for name in names:
            nodes = _top_level_binding_nodes(tree, name)
            if len(nodes) == 1:
                protected_hashes[f"{relative_path}:{name}"] = _node_hash(
                    nodes[0]
                )
    for relative_path in _PIPELINE_ROUTING_MODULES:
        tree = trees[relative_path]
        import_counts: Counter[str] = Counter()
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            bindings = ",".join(sorted(_module_scope_bindings(node)))
            import_counts[bindings] += 1
            protected_hashes[
                f"{relative_path}:import:{bindings}:"
                f"{import_counts[bindings]}"
            ] = _node_hash(node)
    edit_surface_hashes: Dict[str, str] = {}
    tool_constants: list[Dict[str, str]] = []
    for relative_path in edit_surface_modules:
        module_hash, module_constants, module_violations = (
            _module_edit_surface_projection(
                trees[relative_path],
                relative_path=relative_path,
            )
        )
        if module_violations:
            raise ValueError("; ".join(module_violations))
        edit_surface_hashes[relative_path] = module_hash
        tool_constants.extend(module_constants)
    membership, membership_violations = _pipeline_membership_projection(
        trees,
        contract_id=str(snapshot["contract"]["contract_id"]),
    )
    if membership_violations:
        raise ValueError("; ".join(membership_violations))
    body = {
        "schema_version": _PIPELINE_STRUCTURE_SCHEMA_VERSION,
        "contract_id": str(snapshot["contract"]["contract_id"]),
        "protected_symbol_hashes": protected_hashes,
        "edit_surface_hashes": edit_surface_hashes,
        "tool_constants": sorted(
            tool_constants,
            key=lambda item: (item["path"], item["name"]),
        ),
        **membership,
    }
    encoded_body = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        **body,
        "document_hash": "sha256:" + hashlib.sha256(encoded_body).hexdigest(),
    }


def sourcing_pipeline_preservation_errors(
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> List[str]:
    """Compare two independently measured pipeline structure documents."""

    if parent.get("schema_version") != _PIPELINE_STRUCTURE_SCHEMA_VERSION:
        return ["parent sourcing pipeline structure document is invalid"]
    if candidate.get("schema_version") != _PIPELINE_STRUCTURE_SCHEMA_VERSION:
        return ["candidate sourcing pipeline structure document is invalid"]
    if parent.get("contract_id") != candidate.get("contract_id"):
        return ["sourcing pipeline contract id changed"]
    parent_hashes = parent.get("protected_symbol_hashes")
    candidate_hashes = candidate.get("protected_symbol_hashes")
    if not isinstance(parent_hashes, Mapping) or not isinstance(
        candidate_hashes, Mapping
    ):
        return ["sourcing pipeline protected symbol document is invalid"]
    changed = sorted(
        set(parent_hashes) | set(candidate_hashes),
    )
    changed = [
        name
        for name in changed
        if parent_hashes.get(name) != candidate_hashes.get(name)
    ]
    errors = [f"immutable pipeline router changed: {name}" for name in changed]

    parent_edit_hashes = parent.get("edit_surface_hashes")
    candidate_edit_hashes = candidate.get("edit_surface_hashes")
    if not isinstance(parent_edit_hashes, Mapping) or not isinstance(
        candidate_edit_hashes,
        Mapping,
    ):
        errors.append("sourcing pipeline edit surface is invalid")
    else:
        changed_modules = sorted(set(parent_edit_hashes) | set(candidate_edit_hashes))
        for relative_path in changed_modules:
            if parent_edit_hashes.get(relative_path) != candidate_edit_hashes.get(
                relative_path
            ):
                errors.append(
                    f"non-declarative sourcing edit changed: {relative_path}"
                )

    parent_tool_constants = parent.get("tool_constants")
    candidate_tool_constants = candidate.get("tool_constants")
    parent_constant_map: Dict[tuple[str, str], str] = {}
    candidate_constant_map: Dict[tuple[str, str], str] = {}
    if not isinstance(parent_tool_constants, list) or not isinstance(
        candidate_tool_constants,
        list,
    ):
        errors.append("sourcing pipeline tool constants are invalid")
    else:
        def _tool_constant_key(item: Any) -> tuple[str, str]:
            if not isinstance(item, Mapping):
                return "<invalid>", "<invalid>"
            return str(item.get("path") or ""), str(item.get("name") or "")

        parent_constant_map = {
            _tool_constant_key(item): str(item.get("value") or "")
            for item in parent_tool_constants
            if isinstance(item, Mapping)
        }
        candidate_constant_map = {
            _tool_constant_key(item): str(item.get("value") or "")
            for item in candidate_tool_constants
            if isinstance(item, Mapping)
        }
        for key, value in parent_constant_map.items():
            if candidate_constant_map.get(key) != value:
                errors.append(f"existing routing tool constant changed: {key[0]}:{key[1]}")

    parent_records = parent.get("membership_records")
    candidate_records = candidate.get("membership_records")
    if not isinstance(parent_records, list) or not isinstance(
        candidate_records, list
    ):
        return errors + ["sourcing pipeline membership document is invalid"]

    def _record_key(item: Any) -> str:
        if not isinstance(item, Mapping):
            return "<invalid>"
        normalized = dict(item)
        # Liveness is compared per stage below. It is intentionally not part
        # of record identity so a loop can tune eligibility on some tools.
        normalized.pop("liveness_key", None)
        normalized.pop("declared_fields", None)
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    parent_counts = Counter(_record_key(item) for item in parent_records)
    candidate_counts = Counter(_record_key(item) for item in candidate_records)
    for record_key, count in sorted(parent_counts.items()):
        if candidate_counts[record_key] < count:
            errors.append("existing routing tool membership changed")
            break
    for record_key, count in sorted(candidate_counts.items()):
        if count <= parent_counts[record_key]:
            continue
        try:
            record = json.loads(record_key)
        except json.JSONDecodeError:
            errors.append("candidate routing membership record is invalid")
            continue
        if record.get("kind") == "expansion":
            errors.append("candidate added an unreviewed routing expansion")
        elif record.get("unsafe_mutable_fields"):
            errors.append("candidate routing metadata is executable")
        elif (
            not str(record.get("tool_id") or "")
            or len(record.get("stages") or ()) != 1
        ):
            errors.append("candidate routing membership is not explicit")

    def _stage_liveness_anchors(
        items: list[Any],
    ) -> Dict[tuple[str, str, str], set[str]]:
        anchors: Dict[tuple[str, str, str], set[str]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            if kind not in {"definition", "policy"}:
                continue
            stages = [str(value) for value in item.get("stages") or ()]
            tool_id = str(item.get("tool_id") or "")
            liveness_key = str(item.get("liveness_key") or "")
            path = str(item.get("path") or "")
            if not path or not tool_id or not liveness_key or len(stages) != 1:
                continue
            anchors.setdefault((path, kind, stages[0]), set()).add(
                f"{tool_id}:{liveness_key}"
            )
        return anchors

    def _paired_stage_liveness_anchors(
        items: list[Any],
    ) -> Dict[tuple[str, str], set[str]]:
        by_tool: Dict[tuple[str, str, str], Dict[str, str]] = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            if kind not in {"definition", "policy"}:
                continue
            stages = [str(value) for value in item.get("stages") or ()]
            path = str(item.get("path") or "")
            tool_id = str(item.get("tool_id") or "")
            liveness_key = str(item.get("liveness_key") or "")
            if not path or not tool_id or not liveness_key or len(stages) != 1:
                continue
            by_tool.setdefault((path, stages[0], tool_id), {})[kind] = liveness_key
        anchors: Dict[tuple[str, str], set[str]] = {}
        for (path, stage, tool_id), kinds in by_tool.items():
            if set(kinds) != {"definition", "policy"}:
                continue
            anchors.setdefault((path, stage), set()).add(
                f"{tool_id}:{kinds['definition']}:{kinds['policy']}"
            )
        return anchors

    parent_liveness = _stage_liveness_anchors(parent_records)
    candidate_liveness = _stage_liveness_anchors(candidate_records)
    parent_paired_liveness = _paired_stage_liveness_anchors(parent_records)
    candidate_paired_liveness = _paired_stage_liveness_anchors(candidate_records)
    for key, parent_anchors in sorted(parent_paired_liveness.items()):
        if not parent_anchors & candidate_paired_liveness.get(key, set()):
            errors.append(
                "routing stage lost every reviewed viable route pair: "
                f"{key[0]}:{key[1]}"
            )
    for key, parent_anchors in sorted(parent_liveness.items()):
        if (key[0], key[2]) in parent_paired_liveness:
            continue
        if not parent_anchors & candidate_liveness.get(key, set()):
            errors.append(
                "routing stage lost every reviewed viable path: "
                f"{key[0]}:{key[1]}:{key[2]}"
            )

    parent_ownership = parent.get("tool_stage_ownership")
    candidate_ownership = candidate.get("tool_stage_ownership")
    if not isinstance(parent_ownership, Mapping) or not isinstance(
        candidate_ownership, Mapping
    ):
        return errors + ["sourcing pipeline tool ownership document is invalid"]
    for path, parent_kinds in parent_ownership.items():
        candidate_kinds = candidate_ownership.get(path)
        if not isinstance(parent_kinds, Mapping) or not isinstance(
            candidate_kinds, Mapping
        ):
            errors.append(f"routing tool ownership changed: {path}")
            continue
        for kind in ("definitions", "policies"):
            parent_tools = parent_kinds.get(kind)
            candidate_tools = candidate_kinds.get(kind)
            if not isinstance(parent_tools, Mapping) or not isinstance(
                candidate_tools, Mapping
            ):
                errors.append(f"routing tool ownership changed: {path}:{kind}")
                continue
            for tool_id, stages in parent_tools.items():
                if candidate_tools.get(tool_id) != stages:
                    errors.append(
                        f"routing tool stage changed: {path}:{kind}:{tool_id}"
                    )

    parent_global_ownership = parent.get("global_tool_stage_ownership")
    candidate_global_ownership = candidate.get("global_tool_stage_ownership")
    if not isinstance(parent_global_ownership, Mapping) or not isinstance(
        candidate_global_ownership, Mapping
    ):
        errors.append("sourcing pipeline global tool ownership is invalid")
    else:
        for tool_id, stages in parent_global_ownership.items():
            if candidate_global_ownership.get(tool_id) != stages:
                errors.append(f"global routing tool stage changed: {tool_id}")
        for tool_id, stages in candidate_global_ownership.items():
            if not isinstance(stages, list) or len(stages) != 1:
                errors.append(
                    f"global routing tool stage is not singular: {tool_id}"
                )
                continue
            if tool_id in parent_global_ownership:
                continue
            expected_prefix = {
                "candidate_acquisition": "candidate.",
                "candidate_enrichment": "candidate.",
                "intent_evidence": "intent.",
                "contact_acquisition": "contact.",
            }.get(str(stages[0]), "")
            if not expected_prefix or not str(tool_id).startswith(expected_prefix):
                errors.append(f"new routing tool id does not match stage: {tool_id}")

    parent_logical_ownership = parent.get("logical_tool_ownership")
    candidate_logical_ownership = candidate.get("logical_tool_ownership")
    if not isinstance(parent_logical_ownership, Mapping) or not isinstance(
        candidate_logical_ownership,
        Mapping,
    ):
        errors.append("sourcing pipeline logical tool ownership is invalid")
    else:
        for logical_id, paths in parent_logical_ownership.items():
            if candidate_logical_ownership.get(logical_id) != paths:
                errors.append(
                    f"routing tool catalog ownership changed: {logical_id}"
                )
        for logical_id, paths in candidate_logical_ownership.items():
            if logical_id in parent_logical_ownership:
                continue
            if not isinstance(paths, list) or len(paths) != 1:
                errors.append(
                    f"new routing tool has cross-catalog ownership: {logical_id}"
                )

    constructors = candidate.get("definition_constructors")
    if not isinstance(constructors, Mapping):
        errors.append("sourcing pipeline constructor document is invalid")
        constructors = {}
    candidate_global_definitions: Dict[str, list[str]] = {}
    candidate_global_policies: Dict[str, list[str]] = {}
    for candidate_kinds in candidate_ownership.values():
        if not isinstance(candidate_kinds, Mapping):
            continue
        for tool_id, stages in (
            candidate_kinds.get("definitions", {}) or {}
        ).items():
            candidate_global_definitions[str(tool_id)] = list(stages)
        for tool_id, stages in (
            candidate_kinds.get("policies", {}) or {}
        ).items():
            candidate_global_policies[str(tool_id)] = list(stages)
    candidate_routing_tool_ids = set(candidate_global_definitions) | set(
        candidate_global_policies
    )
    for key, value in candidate_constant_map.items():
        if key in parent_constant_map:
            continue
        if value not in candidate_routing_tool_ids:
            errors.append(
                f"new routing tool constant has no catalog entry: {key[0]}:{key[1]}"
            )
    expansion_stages = candidate.get("policy_expansion_stages")
    if not isinstance(expansion_stages, list):
        errors.append("sourcing pipeline policy expansion document is invalid")
        expansion_stages = []
    for path, candidate_kinds in candidate_ownership.items():
        if not isinstance(candidate_kinds, Mapping):
            continue
        parent_kinds = parent_ownership.get(path, {})
        candidate_definitions = candidate_kinds.get("definitions", {})
        candidate_policies = candidate_kinds.get("policies", {})
        parent_definitions = (
            parent_kinds.get("definitions", {})
            if isinstance(parent_kinds, Mapping)
            else {}
        )
        parent_policies = (
            parent_kinds.get("policies", {})
            if isinstance(parent_kinds, Mapping)
            else {}
        )
        if not isinstance(candidate_definitions, Mapping) or not isinstance(
            candidate_policies, Mapping
        ):
            continue
        path_constructors = constructors.get(path, {})
        if not isinstance(path_constructors, Mapping):
            path_constructors = {}
        for tool_id, stages in candidate_definitions.items():
            if tool_id in parent_definitions:
                continue
            constructor = str(path_constructors.get(tool_id) or "")
            if (
                constructor == "_tool"
                and len(stages) == 1
                and stages[0] in expansion_stages
            ):
                continue
            if candidate_policies.get(tool_id) != stages:
                errors.append(
                    f"new routing tool has no same-stage policy: {path}:{tool_id}"
                )
        for tool_id, stages in candidate_policies.items():
            if tool_id in parent_policies:
                continue
            if candidate_definitions.get(tool_id) != stages:
                errors.append(
                    f"new routing policy has no same-stage tool: {path}:{tool_id}"
                )

    parent_coverage = parent.get("stage_coverage")
    candidate_coverage = candidate.get("stage_coverage")
    if not isinstance(parent_coverage, Mapping) or not isinstance(
        candidate_coverage, Mapping
    ):
        errors.append("sourcing pipeline stage coverage document is invalid")
    else:
        for kind in ("definitions", "policies"):
            if candidate_coverage.get(kind) != parent_coverage.get(kind):
                errors.append(f"routing stage coverage changed: {kind}")
    return list(dict.fromkeys(errors))


def verify_source_tree_contract(root: Path) -> List[str]:
    """Return every contract violation for the model source tree at ``root``.

    An empty list means the tree conforms.  Violations are stable, specific
    strings (missing file, unparseable module, missing function, parameter
    drift, integer floor breach) suitable for build receipts.
    """
    root = Path(root)
    detected_snapshot = resolve_reviewed_consumer_snapshot(root)
    violations: List[str] = []
    if detected_snapshot is not None:
        document = dict(detected_snapshot["contract"])
    else:
        document = load_wrapper_contract()
        violations.append(
            "model-owned compatibility contract/parity pair is not reviewed"
        )
    reviewed_snapshot = detected_snapshot or reviewed_consumer_snapshots()[
        str(document["contract_id"])
    ]
    expected_contract_path = Path(
        reviewed_snapshot["contract_path"]
        if reviewed_snapshot is not None
        else CONTRACT_PATH
    )
    expected_parity_path = Path(
        reviewed_snapshot["parity_path"]
        if reviewed_snapshot is not None
        else PARITY_FIXTURE_PATH
    )
    frozen_asyncness = dict(document.get("frozen_asyncness") or {})
    exact_signatures = set(document.get("exact_signatures") or ())
    frozen_required_keyword_only = dict(
        document.get("required_keyword_only") or {}
    )

    canonical_relative = str(document.get("canonical_path") or "")
    canonical_path = root / canonical_relative
    if canonical_path.is_file():
        try:
            if canonical_path.read_bytes() != expected_contract_path.read_bytes():
                violations.append(
                    "model-owned compatibility contract differs from the "
                    "reviewed Lab snapshot"
                )
        except OSError as exc:
            violations.append(
                "unable to compare model-owned compatibility contract: "
                f"{type(exc).__name__}"
            )
    parity_relative = str(document.get("parity_fixture_path") or "")
    parity_path = root / parity_relative
    if parity_path.is_file():
        try:
            if parity_path.read_bytes() != expected_parity_path.read_bytes():
                violations.append(
                    "model-owned parity fixtures differ from the reviewed "
                    "Lab snapshot"
                )
        except OSError as exc:
            violations.append(
                "unable to compare model-owned parity fixtures: "
                f"{type(exc).__name__}"
            )

    for relative in document.get("required_files", []):
        if not (root / relative).is_file():
            violations.append(f"missing required file: {relative}")

    parsed: Dict[str, ast.Module] = {}

    def _tree(relative: str) -> ast.Module | None:
        if relative in parsed:
            return parsed[relative]
        path = root / relative
        if not path.is_file():
            return None
        try:
            # Parse raw bytes so PEP 263 coding declarations are honored the
            # same way the interpreter honors them — a legal non-UTF-8 module
            # must parse, and an unreadable one must be a VIOLATION, never an
            # exception that lets the build gate fail open.
            tree = ast.parse(path.read_bytes())
        except SyntaxError as exc:
            violations.append(f"unparseable module {relative}: {exc.msg} (line {exc.lineno})")
            return None
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            violations.append(
                f"unreadable module {relative}: {type(exc).__name__}: {str(exc)[:120]}"
            )
            return None
        parsed[relative] = tree
        return tree

    for relative, functions in (document.get("functions") or {}).items():
        tree = _tree(relative)
        if tree is None:
            continue
        symbols = _module_symbols(tree)
        for name, expected_params in functions.items():
            actual = symbols["functions"].get(name)
            if actual is None:
                violations.append(f"missing function {relative}:{name}")
                continue
            expected = list(expected_params)
            actual_params = actual["params"]
            contract_key = f"{relative}:{name}"
            expected_full = (document.get("full_parameters") or {}).get(
                contract_key
            )
            if (
                expected_full is not None
                and (
                    actual["all_params"] != list(expected_full)
                    or actual["positional_only"]
                    or actual["vararg"] is not None
                    or actual["kwarg"] is not None
                )
            ):
                violations.append(
                    f"full parameter drift {relative}:{name}: expected "
                    f"{list(expected_full)}, found {actual['all_params']} "
                    f"(positional_only={actual['positional_only']}, "
                    f"vararg={actual['vararg']!r}, kwarg={actual['kwarg']!r})"
                )
            # Newer contracts separate the exact positional surface in
            # ``functions`` from the complete keyword-only surface in
            # ``full_parameters``. Older snapshots retain the original
            # all-parameter exactness.
            exact_actual = (
                actual["params"]
                if expected_full is not None
                else actual["all_params"]
            )
            if contract_key in exact_signatures and (
                exact_actual != expected
                or actual["positional_only"]
                or actual["vararg"] is not None
                or actual["kwarg"] is not None
            ):
                violations.append(
                    f"exact parameter drift {relative}:{name}: expected "
                    f"{expected}, found {exact_actual} "
                    f"(positional_only={actual['positional_only']}, "
                    f"vararg={actual['vararg']!r}, kwarg={actual['kwarg']!r})"
                )
            elif expected and actual_params[: len(expected)] != expected:
                violations.append(
                    f"parameter drift {relative}:{name}: expected leading "
                    f"parameters {expected}, found {actual_params}"
                )
            elif actual["required_positional"] > len(expected):
                violations.append(
                    f"required parameter drift {relative}:{name}: expected at most "
                    f"{len(expected)} required positional parameters, found "
                    f"{actual['required_positional']}"
                )
            expected_required_keyword_only = frozen_required_keyword_only.get(
                contract_key, []
            )
            if actual["required_keyword_only"] != expected_required_keyword_only:
                violations.append(
                    f"required keyword-only parameter drift {relative}:{name}: "
                    f"expected {expected_required_keyword_only}, found "
                    f"{actual['required_keyword_only']}"
                )
            frozen_async = frozen_asyncness.get(contract_key)
            if frozen_async is not None and actual["is_async"] != frozen_async:
                violations.append(
                    f"asyncness drift {relative}:{name}: frozen surface is "
                    f"{'async' if frozen_async else 'sync'}, found "
                    f"{'async' if actual['is_async'] else 'sync'}"
                )

    for relative, required_modules in (
        document.get("required_imports") or {}
    ).items():
        tree = _tree(relative)
        if tree is None:
            continue
        bound_imports = _module_bound_imports(tree)
        for module_name in required_modules:
            if str(module_name) not in bound_imports:
                violations.append(
                    f"missing bound import {relative}:{module_name}"
                )

    for relative, minimums in (document.get("integer_minimums") or {}).items():
        tree = _tree(relative)
        if tree is None:
            continue
        constants = _module_symbols(tree)["constants"]
        for name, floor in minimums.items():
            value = constants.get(name)
            if value is None:
                violations.append(f"missing integer constant {relative}:{name}")
            elif value < int(floor):
                violations.append(
                    f"integer floor breach {relative}:{name}: {value} < {floor}"
                )

    for relative, expected_values in (
        document.get("exact_constants") or {}
    ).items():
        tree = _tree(relative)
        if tree is None:
            continue
        constants = _literal_module_constants(
            tree,
            names={str(name) for name in expected_values},
        )
        missing = object()
        for name, expected in expected_values.items():
            actual = constants.get(name, missing)
            if actual is missing or not _same_literal(actual, expected):
                violations.append(
                    f"exact constant drift {relative}:{name}: expected "
                    f"{expected!r}, found "
                    f"{None if actual is missing else actual!r}"
                )

    runtime_tree = _tree("sourcing_model/routing/runtime.py")
    if runtime_tree is not None and detected_snapshot is not None:
        violations.extend(
            _router_stage_binding_violations(
                runtime_tree,
                relative_path="sourcing_model/routing/runtime.py",
                expected_stage_bindings=_ROUTER_STAGE_BINDINGS,
                required_functions=set(
                    document.get("functions", {}).get(
                        "sourcing_model/routing/runtime.py", {}
                    )
                ),
            )
        )
    defaults_tree = _tree("sourcing_model/routing/defaults.py")
    if defaults_tree is not None and detected_snapshot is not None:
        violations.extend(
            _router_stage_binding_violations(
                defaults_tree,
                relative_path="sourcing_model/routing/defaults.py",
                expected_stage_bindings=_DEFAULT_ROUTER_STAGE_BINDINGS,
                required_functions=set(
                    document.get("functions", {}).get(
                        "sourcing_model/routing/defaults.py", {}
                    )
                ),
            )
        )

    return violations
