"""Sourcing-model wrapper-contract conformance checks.

The sourcing model's research-lab surface is frozen by the wrapper contract
(``research_lab/sourcing_model_contract.json``, mirrored from the model repo's
``release/model-contract.json``): required files, the exact function
signatures the lab and the production harness call (``research_lab_adapter.
run_icp``/``adapter_metadata``, ``sourcing_model.core.qualify``, the discovery
/validation/client seams the harness monkey-patches), and integer floor
constants.  The new production flow is built AROUND these symbols, so any
model-source change that breaks them breaks both the lab benchmark runtime
and the harness.

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


def _function_params(node: ast.AST) -> List[str]:
    args = getattr(node, "args", None)
    if args is None:
        return []
    return [a.arg for a in args.posonlyargs + args.args]


def _module_symbols(tree: ast.Module) -> Dict[str, Any]:
    """Top-level function param-lists and integer constant assignments."""
    functions: Dict[str, List[str]] = {}
    constants: Dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = _function_params(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
                and not isinstance(node.value.value, bool)
            ):
                constants[target.id] = node.value.value
    return {"functions": functions, "constants": constants}


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
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            violations.append(f"unparseable module {relative}: {exc.msg} (line {exc.lineno})")
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
            if expected and actual[: len(expected)] != expected:
                violations.append(
                    f"parameter drift {relative}:{name}: expected leading "
                    f"parameters {expected}, found {actual}"
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
