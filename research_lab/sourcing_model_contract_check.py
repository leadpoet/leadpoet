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
            "cost_class",
            "execution_mode",
            "max_calls",
            "max_results",
            "timeout_seconds",
            "unit_cost",
        }
    ),
    "_tool": frozenset(
        {
            "claim_visibility",
            "cost_credits",
            "identity_anchor",
            "identity_policy",
            "pagination_mode",
        }
    ),
    "SourceAddRoutingRegistration": frozenset(
        {
            "cost_class",
            "execution_mode",
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
}
_PIPELINE_EDIT_SURFACE_MODULES = (
    "sourcing_model/discovery.py",
    *_PIPELINE_MEMBERSHIP_MODULES,
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
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
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
        "invalid_call_shape": list(invalid_call_shape),
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


class _PromptStringNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<prompt-text>"), node)
        return node


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
    for node in tree.body:
        name, value = _assigned_name_and_value(node)
        if name.startswith("TOOL_") and isinstance(value, ast.Constant) and isinstance(
            value.value,
            str,
        ):
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
            if not isinstance(value, ast.Constant) or not isinstance(
                value.value,
                str,
            ):
                violations.append(
                    f"routing version is not literal {relative_path}:{name}"
                )
            normalized_body.append(_assignment_with_sentinel(node, "<version>"))
            continue
        if (
            relative_path == "sourcing_model/routing/runtime.py"
            and name == "SOURCE_ADD_ROUTING_REGISTRATIONS"
        ):
            normalized_body.append(
                _assignment_with_sentinel(node, "<approved-source-add-registry>")
            )
            continue
        if (
            relative_path == "sourcing_model/scrapingdog_signal_contract.py"
            and name == "TOOL_CATALOG"
        ):
            normalized_body.append(
                _assignment_with_sentinel(node, "<tool-catalog>")
            )
            continue
        if name in prompt_bindings:
            if value is None or not _is_pipeline_data_expression(value):
                violations.append(
                    f"prompt binding is not inert data {relative_path}:{name}"
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
                normalized_body.append(
                    _PromptStringNormalizer().visit(copy.deepcopy(node))
                )
                continue
        normalized_body.append(copy.deepcopy(node))
    normalized = ast.Module(body=normalized_body, type_ignores=tree.type_ignores)
    return (
        _node_hash(normalized),
        sorted(tool_constants, key=lambda item: (item["path"], item["name"])),
        violations,
    )


def _compile_route_context(call: ast.Call) -> ast.Call | None:
    context: ast.AST | None = call.args[2] if len(call.args) >= 3 else None
    keyword_context = [
        keyword.value for keyword in call.keywords if keyword.arg == "context"
    ]
    if keyword_context:
        if context is not None or len(keyword_context) != 1:
            return None
        context = keyword_context[0]
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
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            callee = _call_name(node)
            if callee == "compile_route":
                compile_route_calls.append(node)
            if callee in expected_bindings and callee != function_name:
                cross_stage_calls = True
        observed_stages: list[str] = []
        for call in compile_route_calls:
            context = _compile_route_context(call)
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
        if (
            not compile_route_calls
            or cross_stage_calls
            or any(stage != expected_stage for stage in observed_stages)
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
) -> tuple[Dict[str, Any], List[str]]:
    """Project tool ownership and policy membership without executing source."""

    records: list[Dict[str, Any]] = []
    violations: List[str] = []
    coverage: Dict[str, set[str]] = {
        "definitions": set(),
        "policies": set(),
    }
    policy_expansion_stages: set[str] = set()
    ownership: Dict[str, Dict[str, Dict[str, list[str]]]] = {}
    definition_constructors: Dict[str, Dict[str, str]] = {}

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
        kind = str(record.get("kind") or "")
        tool_id = str(record.get("tool_id") or "")
        stages = [str(item) for item in record.get("stages") or ()]
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
        path = str(record["path"])
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
            for index, item in enumerate(items):
                container = function_name
                if isinstance(item, ast.Call) and _call_name(item) == constructor:
                    record = _membership_record(
                        item,
                        relative_path=relative_path,
                        container=container,
                        constructor=constructor,
                        constants=constants,
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
        catalog_nodes = _top_level_binding_nodes(signal_tree, "TOOL_CATALOG")
        if len(catalog_nodes) != 1:
            violations.append(f"routing tool catalog binding drift {signal_path}")
        else:
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
        registry_nodes = _top_level_binding_nodes(
            runtime_tree, "SOURCE_ADD_ROUTING_REGISTRATIONS"
        )
        if len(registry_nodes) != 1:
            violations.append(
                f"SOURCE_ADD routing registry binding drift {runtime_path}"
            )
        else:
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
                    )
                    if record is not None:
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
    for relative_path in _PIPELINE_EDIT_SURFACE_MODULES:
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
    if len(trees) == len(_PIPELINE_EDIT_SURFACE_MODULES):
        _projection, membership_violations = _pipeline_membership_projection(trees)
        violations.extend(membership_violations)
        for relative_path in _PIPELINE_EDIT_SURFACE_MODULES:
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
    trees = {
        relative_path: ast.parse((root / relative_path).read_bytes())
        for relative_path in _PIPELINE_EDIT_SURFACE_MODULES
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
    for relative_path in _PIPELINE_EDIT_SURFACE_MODULES:
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
    membership, membership_violations = _pipeline_membership_projection(trees)
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

    parent_liveness = _stage_liveness_anchors(parent_records)
    candidate_liveness = _stage_liveness_anchors(candidate_records)
    for key, parent_anchors in sorted(parent_liveness.items()):
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
