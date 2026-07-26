"""Sourcing-model wrapper-contract conformance checks.

The sourcing model's research-lab surface is frozen by the wrapper contract
(``research_lab/sourcing_model_contract.json``, mirrored from the site's
``sourcing-worker/release/model-contract.json``): required files, the exact
function signatures the lab and the production harness call
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

Pure stdlib.  The contract JSON is a derived mirror — when the model repo's
contract changes, copy it verbatim and note the source revision.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

CONTRACT_PATH = Path(__file__).with_name("sourcing_model_contract.json")

# Asyncness of the frozen surface at the mirrored contract revision. The
# contract JSON is a verbatim site mirror and carries no async information,
# but sync-vs-async IS part of the callable surface: the lab runtime and the
# harness call these seams the way the frozen source defines them, so a
# candidate flipping one (e.g. making ``qualify`` async) would hand callers a
# coroutine and break at invocation despite an identical parameter list.
# Only names present here are checked — a future contract revision that adds
# functions skips the asyncness check for them until this map is updated
# alongside the mirror.
FROZEN_ASYNCNESS: Dict[str, bool] = {
    "research_lab_adapter.py:adapter_metadata": False,
    "research_lab_adapter.py:run_icp": False,
    "sourcing_model/clients.py:_exa_call": False,
    "sourcing_model/clients.py:agent_get": False,
    "sourcing_model/clients.py:agent_post": False,
    "sourcing_model/clients.py:exa_search": False,
    "sourcing_model/clients.py:sd_company": False,
    "sourcing_model/clients.py:sd_scrape": False,
    "sourcing_model/core.py:_emitted_intent_evidence_url": False,
    "sourcing_model/core.py:_fallback_sources": False,
    "sourcing_model/core.py:qualify": False,
    "sourcing_model/discovery.py:agent_results": False,
    "sourcing_model/discovery.py:apply_keep_gates": False,
    "sourcing_model/discovery.py:discover_goal_round": False,
    "sourcing_model/discovery.py:resolve_linkedin": False,
    "sourcing_model/validation.py:bonus_requirements": False,
    "sourcing_model/validation.py:make_deps": False,
    "sourcing_model/validation.py:validate_candidate": True,
}

# These are invoked directly by the Lab/production wrappers, so an additional
# positional parameter changes the public callable even when it is optional.
# Internal monkey-patch seams remain call-compatible: optional trailing
# parameters are allowed, but additional required parameters are not.
EXACT_SIGNATURES = {
    "research_lab_adapter.py:adapter_metadata",
    "research_lab_adapter.py:run_icp",
    "sourcing_model/core.py:qualify",
}

FROZEN_REQUIRED_KEYWORD_ONLY: Dict[str, List[str]] = {
    "sourcing_model/firmographic_discovery.py:plan_for_icp": ["target"],
    "sourcing_model/orchestrator.py:run_branches": ["max_companies"],
}


def load_wrapper_contract(path: Path | None = None) -> Dict[str, Any]:
    """Load and shape-check the vendored wrapper contract."""
    document = json.loads(Path(path or CONTRACT_PATH).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(
            "Unsupported sourcing wrapper contract schema_version: "
            f"{document.get('schema_version')!r}"
        )
    for key in ("contract_id", "required_files", "functions"):
        if key not in document:
            raise ValueError(f"wrapper contract missing required key {key!r}")
    return document


def _function_signature(node: ast.AST) -> Dict[str, Any]:
    args = getattr(node, "args", None)
    if args is None:
        return {
            "params": [],
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
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.asname:
                continue
            parts = alias.name.split(".")
            for index in range(1, len(parts) + 1):
                bound.add(".".join(parts[:index]))
    return bound


def verify_source_tree_contract(
    root: Path, contract: Mapping[str, Any] | None = None
) -> List[str]:
    """Return every contract violation for the model source tree at ``root``.

    An empty list means the tree conforms.  Violations are stable, specific
    strings (missing file, unparseable module, missing function, parameter
    drift, integer floor breach) suitable for build receipts.
    """
    root = Path(root)
    document = dict(contract) if contract is not None else load_wrapper_contract()
    violations: List[str] = []

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
            if contract_key in EXACT_SIGNATURES and actual_params != expected:
                violations.append(
                    f"exact parameter drift {relative}:{name}: expected "
                    f"{expected}, found {actual_params}"
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
            expected_required_keyword_only = FROZEN_REQUIRED_KEYWORD_ONLY.get(
                contract_key, []
            )
            if actual["required_keyword_only"] != expected_required_keyword_only:
                violations.append(
                    f"required keyword-only parameter drift {relative}:{name}: "
                    f"expected {expected_required_keyword_only}, found "
                    f"{actual['required_keyword_only']}"
                )
            frozen_async = FROZEN_ASYNCNESS.get(contract_key)
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

    return violations
