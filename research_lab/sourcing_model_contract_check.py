"""Sourcing-model consumer-contract conformance checks.

The reviewed model-owned v7/v8/v11/v12 contracts are snapshotted byte-for-byte under
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
import hashlib
import json
from pathlib import Path
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
REVIEWED_CONSUMER_SNAPSHOT_SPECS = (
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
            if contract_key in exact_signatures and (
                actual["all_params"] != expected
                or actual["positional_only"]
                or actual["vararg"] is not None
                or actual["kwarg"] is not None
            ):
                violations.append(
                    f"exact parameter drift {relative}:{name}: expected "
                    f"{expected}, found {actual['all_params']} "
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

    return violations
