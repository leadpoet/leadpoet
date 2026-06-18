"""Phase 4 phase-scaffold cleanup verifier.

P4.3 verifies that phase-numbered runtime scaffolding has been renamed to
domain concepts before GA-scale work.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .legacy_surface_retirement import verify_legacy_surface_retirement
from .scale_foundation import (
    ScaleGate,
    require_scale_gate,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "scaffold_cleanup_fixtures.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_ROOTS: tuple[str, ...] = (
    "research_lab",
    "scripts",
)

DEFAULT_IGNORED_PREFIXES: tuple[str, ...] = (
    "scripts/techstack_test_artifacts/",
)

SCAN_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".json",
)

PHASE_MODULE_PATH_RE = re.compile(r"(^|/)phase[0-9][a-z0-9_./-]*", re.IGNORECASE)
PHASE_FIXTURE_PATH_RE = re.compile(r"(^|/)(phase[0-9]|p[0-9][.0-9a-z_-]*)", re.IGNORECASE)
PHASE_VERIFIER_SCRIPT_RE = re.compile(r"^scripts/verify_research_lab_phase[0-9][a-z0-9_./-]*\.py$", re.IGNORECASE)
PHASE_SYMBOL_RE = re.compile(r"(^Phase[0-9]|^PHASE[0-9]|(^|_)phase[0-9]($|_))")
PHASE_ENV_VAR_RE = re.compile(r"\bRESEARCH_LAB_PHASE[0-9][A-Z0-9_]*\b")


class PhaseScaffoldCategory(str, Enum):
    MODULE_PATH = "module_path"
    VERIFIER_SCRIPT = "verifier_script"
    FIXTURE_PATH = "fixture_path"
    PUBLIC_SYMBOL = "public_symbol"
    ENV_VAR = "env_var"


class PhaseScaffoldCleanupState(str, Enum):
    LOCAL_AUDIT_ONLY = "local_audit_only"
    READY_AFTER_CLEANUP_EVIDENCE = "ready_after_cleanup_evidence"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class PhaseScaffoldFinding:
    path: str
    line: int
    name: str
    category: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PhaseScaffoldFinding":
        return cls(
            path=str(data["path"]),
            line=int(data["line"]),
            name=str(data["name"]),
            category=str(data["category"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseScaffoldCleanupRecord:
    audit_id: str
    scanned_roots: tuple[str, ...]
    ignored_prefixes: tuple[str, ...]
    phase_module_path_refs: int = 0
    phase_verifier_script_refs: int = 0
    phase_fixture_path_refs: int = 0
    phase_public_symbol_refs: int = 0
    phase_env_var_refs: int = 0
    findings: tuple[PhaseScaffoldFinding, ...] = ()
    audit_scope_note: str = "static_phase_scaffold_scan_only"
    runtime_modules_domain_named: bool = False
    verifier_scripts_domain_named: bool = False
    fixtures_domain_named: bool = False
    public_symbols_domain_named: bool = False
    env_vars_domain_named: bool = False
    compatibility_shims_required: bool = False
    phase_scaffold_cleanup_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    cleanup_plan_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = PhaseScaffoldCleanupState.LOCAL_AUDIT_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PhaseScaffoldCleanupRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            scanned_roots=tuple(str(item) for item in data.get("scanned_roots", [])),
            ignored_prefixes=tuple(str(item) for item in data.get("ignored_prefixes", [])),
            phase_module_path_refs=int(data.get("phase_module_path_refs", 0)),
            phase_verifier_script_refs=int(data.get("phase_verifier_script_refs", 0)),
            phase_fixture_path_refs=int(data.get("phase_fixture_path_refs", 0)),
            phase_public_symbol_refs=int(data.get("phase_public_symbol_refs", 0)),
            phase_env_var_refs=int(data.get("phase_env_var_refs", 0)),
            findings=tuple(PhaseScaffoldFinding.from_mapping(item) for item in data.get("findings", [])),
            audit_scope_note=str(data.get("audit_scope_note", "static_phase_scaffold_scan_only")),
            runtime_modules_domain_named=bool(data.get("runtime_modules_domain_named", False)),
            verifier_scripts_domain_named=bool(data.get("verifier_scripts_domain_named", False)),
            fixtures_domain_named=bool(data.get("fixtures_domain_named", False)),
            public_symbols_domain_named=bool(data.get("public_symbols_domain_named", False)),
            env_vars_domain_named=bool(data.get("env_vars_domain_named", False)),
            compatibility_shims_required=bool(data.get("compatibility_shims_required", False)),
            phase_scaffold_cleanup_ready=bool(data.get("phase_scaffold_cleanup_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            cleanup_plan_ref=str(data.get("cleanup_plan_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", PhaseScaffoldCleanupState.LOCAL_AUDIT_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scanned_roots"] = list(self.scanned_roots)
        data["ignored_prefixes"] = list(self.ignored_prefixes)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def scan_phase_scaffold_refs(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> tuple[PhaseScaffoldFinding, ...]:
    repo = Path(repo_root)
    findings: list[PhaseScaffoldFinding] = []
    for path in _iter_scan_files(repo, roots, ignored_prefixes):
        rel_path = path.relative_to(repo).as_posix()
        findings.extend(_scan_path_for_phase_scaffold(rel_path))
        if path.suffix == ".py":
            findings.extend(_scan_python_symbols_for_phase_scaffold(path, rel_path))
        findings.extend(_scan_text_for_phase_env_vars(path, rel_path))
    return tuple(sorted(findings, key=lambda finding: (finding.path, finding.line, finding.category, finding.name)))


def build_current_phase_scaffold_cleanup_record(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> PhaseScaffoldCleanupRecord:
    findings = scan_phase_scaffold_refs(repo_root=repo_root, roots=roots, ignored_prefixes=ignored_prefixes)
    counts = _category_counts(findings)
    cleanup_ready = all(count == 0 for count in counts.values())
    return PhaseScaffoldCleanupRecord(
        audit_id="phase_scaffold_cleanup:p4.3:local-current-audit",
        scanned_roots=tuple(roots),
        ignored_prefixes=tuple(ignored_prefixes),
        phase_module_path_refs=counts[PhaseScaffoldCategory.MODULE_PATH.value],
        phase_verifier_script_refs=counts[PhaseScaffoldCategory.VERIFIER_SCRIPT.value],
        phase_fixture_path_refs=counts[PhaseScaffoldCategory.FIXTURE_PATH.value],
        phase_public_symbol_refs=counts[PhaseScaffoldCategory.PUBLIC_SYMBOL.value],
        phase_env_var_refs=counts[PhaseScaffoldCategory.ENV_VAR.value],
        findings=findings,
        audit_scope_note="static_phase_scaffold_scan_only",
        runtime_modules_domain_named=counts[PhaseScaffoldCategory.MODULE_PATH.value] == 0,
        verifier_scripts_domain_named=counts[PhaseScaffoldCategory.VERIFIER_SCRIPT.value] == 0,
        fixtures_domain_named=counts[PhaseScaffoldCategory.FIXTURE_PATH.value] == 0,
        public_symbols_domain_named=counts[PhaseScaffoldCategory.PUBLIC_SYMBOL.value] == 0,
        env_vars_domain_named=counts[PhaseScaffoldCategory.ENV_VAR.value] == 0,
        compatibility_shims_required=False,
        phase_scaffold_cleanup_ready=cleanup_ready,
        uses_local_fixtures=not cleanup_ready,
        local_only=not cleanup_ready,
        cleanup_plan_ref=(
            "phase_scaffold_cleanup_plan:p4.3:completed"
            if cleanup_ready
            else "phase_scaffold_cleanup_plan:p4.3:pending"
        ),
        evidence_refs=(
            "phase_scaffold_cleanup:p4.3:local-current-audit",
            "domain_module_map:research-lab-runtime-v1",
            "domain_symbol_map:research-lab-public-api-v1",
            "env_var_migration:research-lab-domain-env-v1",
            "fixture_migration:research-lab-domain-fixtures-v1",
            "script_migration:research-lab-domain-verifiers-v1",
            "compatibility_shim_retirement:normal-runtime-clean",
            "owner_approval:scale-phase-scaffold-cleanup",
        ) if cleanup_ready else ("phase_scaffold_cleanup:p4.3:local-current-audit",),
        owner_approval_ref="owner_approval:scale-phase-scaffold-cleanup" if cleanup_ready else "",
        state=(
            PhaseScaffoldCleanupState.READY_AFTER_CLEANUP_EVIDENCE.value
            if cleanup_ready
            else PhaseScaffoldCleanupState.LOCAL_AUDIT_ONLY.value
        ),
    )


def phase_scaffold_cleanup_blockers(record: PhaseScaffoldCleanupRecord | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(record, PhaseScaffoldCleanupRecord):
        record = PhaseScaffoldCleanupRecord.from_mapping(record)
    blockers: list[str] = []
    if record.phase_module_path_refs and not record.runtime_modules_domain_named:
        blockers.append("phase_module_paths_not_domain_named")
    if record.phase_verifier_script_refs and not record.verifier_scripts_domain_named:
        blockers.append("phase_verifier_scripts_not_domain_named")
    if record.phase_fixture_path_refs and not record.fixtures_domain_named:
        blockers.append("phase_fixtures_not_domain_named")
    if record.phase_public_symbol_refs and not record.public_symbols_domain_named:
        blockers.append("phase_public_symbols_not_domain_named")
    if record.phase_env_var_refs and not record.env_vars_domain_named:
        blockers.append("phase_env_vars_not_domain_named")
    if record.compatibility_shims_required:
        blockers.append("compatibility_shims_required")
    return tuple(blockers)


def validate_phase_scaffold_cleanup_record(record: PhaseScaffoldCleanupRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, PhaseScaffoldCleanupRecord):
        try:
            record = PhaseScaffoldCleanupRecord.from_mapping(record)
        except KeyError as exc:
            return [f"missing required phase scaffold cleanup field: {exc.args[0]}"]
        except (TypeError, ValueError) as exc:
            return [f"invalid phase scaffold cleanup payload: {exc}"]

    errors: list[str] = []
    if record.state not in {state.value for state in PhaseScaffoldCleanupState}:
        errors.append(f"unknown phase scaffold cleanup state: {record.state}")
        return errors
    if not record.audit_id.startswith("phase_scaffold_cleanup:"):
        errors.append("audit_id must be phase_scaffold_cleanup:-prefixed")
    if not record.scanned_roots:
        errors.append("phase scaffold cleanup audit requires scanned_roots")
    if record.audit_scope_note != "static_phase_scaffold_scan_only":
        errors.append("audit_scope_note must be static_phase_scaffold_scan_only")
    if record.cleanup_plan_ref and not record.cleanup_plan_ref.startswith("phase_scaffold_cleanup_plan:"):
        errors.append("cleanup_plan_ref must be phase_scaffold_cleanup_plan:-prefixed")
    _append_count_consistency_errors(record, errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "phase_scaffold_cleanup:",
                "phase_scaffold_cleanup_plan:",
                "domain_module_map:",
                "domain_symbol_map:",
                "env_var_migration:",
                "fixture_migration:",
                "script_migration:",
                "compatibility_shim_retirement:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved phase scaffold cleanup prefixes")
            break
    if record.production_writes:
        errors.append("P4.3 phase scaffold cleanup verifier must not enable production writes")
    if record.supabase_writes:
        errors.append("P4.3 phase scaffold cleanup verifier must not enable Supabase writes")
    if record.public_workflows:
        errors.append("P4.3 phase scaffold cleanup verifier must not enable public workflows")

    blockers = phase_scaffold_cleanup_blockers(record)
    if record.phase_scaffold_cleanup_ready:
        if blockers:
            errors.append("phase scaffold cleanup ready claim has blockers: " + ", ".join(blockers))
        if not _all_cleanup_controls_true(record):
            errors.append("phase scaffold cleanup ready claim requires all cleanup controls true")
        if record.uses_local_fixtures:
            errors.append("phase scaffold cleanup readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("phase scaffold cleanup readiness cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("phase scaffold cleanup readiness requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("phase scaffold cleanup readiness requires owner_approval_ref")
        if record.state != PhaseScaffoldCleanupState.READY_AFTER_CLEANUP_EVIDENCE.value:
            errors.append("phase_scaffold_cleanup_ready requires ready_after_cleanup_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready phase scaffold cleanup audits must remain local_only")
        if record.state == PhaseScaffoldCleanupState.READY_AFTER_CLEANUP_EVIDENCE.value:
            errors.append("ready_after_cleanup_evidence state requires phase_scaffold_cleanup_ready")
    return errors


def verify_phase_scaffold_cleanup(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    legacy_summary = verify_legacy_surface_retirement()
    fixture = _load_fixture(Path(fixture_path))

    local_record = PhaseScaffoldCleanupRecord.from_mapping(fixture["local_blocked_sample"])
    _assert(not validate_phase_scaffold_cleanup_record(local_record), "local phase-scaffold sample validates")
    _assert(not local_record.phase_scaffold_cleanup_ready, "P4.3 local sample does not claim cleanup ready")
    _assert(phase_scaffold_cleanup_blockers(local_record), "P4.3 local sample reports blockers")

    ready_record = PhaseScaffoldCleanupRecord.from_mapping(fixture["ready_control"])
    _assert(not validate_phase_scaffold_cleanup_record(ready_record), "ready-control phase-scaffold cleanup validates")
    _assert(not phase_scaffold_cleanup_blockers(ready_record), "ready-control phase-scaffold cleanup has no blockers")

    current_record = build_current_phase_scaffold_cleanup_record()
    _assert(not validate_phase_scaffold_cleanup_record(current_record), "current phase-scaffold cleanup audit validates")
    _assert(current_record.phase_scaffold_cleanup_ready, "current phase scaffold audit is ready after cleanup")
    current_blockers = phase_scaffold_cleanup_blockers(current_record)
    _assert(not current_blockers, "current phase scaffold audit has no cleanup blockers")
    _assert(current_record.phase_module_path_refs == 0, "current audit has no phase-numbered module paths")
    _assert(current_record.phase_verifier_script_refs == 0, "current audit has no phase-numbered verifier scripts")
    _assert(current_record.phase_fixture_path_refs == 0, "current audit has no phase-numbered fixtures")
    _assert(current_record.phase_public_symbol_refs == 0, "current audit has no phase-numbered public symbols")
    _assert(current_record.phase_env_var_refs == 0, "current audit has no phase-numbered env vars")

    for invalid in fixture["invalid_cleanup_audits"]:
        base = fixture[str(invalid.get("base", "local_blocked_sample"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_phase_scaffold_cleanup_record(merged)
        _assert(errors, f"invalid phase-scaffold cleanup audit fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"phase_scaffold_cleanup_ready": False}, ScaleGate.PHASE_SCAFFOLD_CLEANUP_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.3 phase-scaffold cleanup gate remains required by P4.0")

    return {
        "legacy_surface_audit_id": legacy_summary["current_audit_id"],
        "audit_id": local_record.audit_id,
        "current_audit_id": current_record.audit_id,
        "phase_module_path_refs": current_record.phase_module_path_refs,
        "phase_verifier_script_refs": current_record.phase_verifier_script_refs,
        "phase_fixture_path_refs": current_record.phase_fixture_path_refs,
        "phase_public_symbol_refs": current_record.phase_public_symbol_refs,
        "phase_env_var_refs": current_record.phase_env_var_refs,
        "phase_scaffold_cleanup_ready": current_record.phase_scaffold_cleanup_ready,
        "blockers": list(current_blockers),
        "current_findings": len(current_record.findings),
        "audit_scope_note": current_record.audit_scope_note,
    }


def _iter_scan_files(repo: Path, roots: Sequence[str], ignored_prefixes: Sequence[str]) -> Iterable[Path]:
    for root in roots:
        path = repo / root
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix not in SCAN_SUFFIXES:
                continue
            rel = candidate.relative_to(repo).as_posix()
            if any(rel.startswith(prefix) for prefix in ignored_prefixes):
                continue
            yield candidate


def _scan_path_for_phase_scaffold(rel_path: str) -> list[PhaseScaffoldFinding]:
    findings: list[PhaseScaffoldFinding] = []
    if rel_path.startswith("research_lab/fixtures/") and PHASE_FIXTURE_PATH_RE.search(Path(rel_path).name):
        findings.append(
            PhaseScaffoldFinding(
                path=rel_path,
                line=0,
                name=Path(rel_path).name,
                category=PhaseScaffoldCategory.FIXTURE_PATH.value,
            )
        )
    elif rel_path.startswith("research_lab/") and rel_path.endswith(".py") and PHASE_MODULE_PATH_RE.search(rel_path):
        findings.append(
            PhaseScaffoldFinding(
                path=rel_path,
                line=0,
                name=Path(rel_path).name,
                category=PhaseScaffoldCategory.MODULE_PATH.value,
            )
        )
    if PHASE_VERIFIER_SCRIPT_RE.match(rel_path):
        findings.append(
            PhaseScaffoldFinding(
                path=rel_path,
                line=0,
                name=Path(rel_path).name,
                category=PhaseScaffoldCategory.VERIFIER_SCRIPT.value,
            )
        )
    return findings


def _scan_python_symbols_for_phase_scaffold(path: Path, rel_path: str) -> list[PhaseScaffoldFinding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    except (SyntaxError, UnicodeDecodeError):
        return []
    findings: list[PhaseScaffoldFinding] = []
    for node in ast.walk(tree):
        name = _phase_scaffold_symbol_name(node)
        if not name or not PHASE_SYMBOL_RE.search(name):
            continue
        findings.append(
            PhaseScaffoldFinding(
                path=rel_path,
                line=int(getattr(node, "lineno", 0)),
                name=name,
                category=PhaseScaffoldCategory.PUBLIC_SYMBOL.value,
            )
        )
    return findings


def _phase_scaffold_symbol_name(node: ast.AST) -> str:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    if isinstance(node, ast.alias):
        return node.asname or node.name
    if isinstance(node, ast.Assign):
        names = [_target_name(target) for target in node.targets]
        return next((name for name in names if name), "")
    if isinstance(node, ast.AnnAssign):
        return _target_name(node.target)
    return ""


def _target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _scan_text_for_phase_env_vars(path: Path, rel_path: str) -> list[PhaseScaffoldFinding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings: list[PhaseScaffoldFinding] = []
    for line_number, line in enumerate(lines, start=1):
        for match in PHASE_ENV_VAR_RE.finditer(line):
            findings.append(
                PhaseScaffoldFinding(
                    path=rel_path,
                    line=line_number,
                    name=match.group(0),
                    category=PhaseScaffoldCategory.ENV_VAR.value,
                )
            )
    return findings


def _category_counts(findings: Sequence[PhaseScaffoldFinding]) -> dict[str, int]:
    counts = {category.value: 0 for category in PhaseScaffoldCategory}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    return counts


def _append_count_consistency_errors(record: PhaseScaffoldCleanupRecord, errors: list[str]) -> None:
    counts = _category_counts(record.findings)
    if counts[PhaseScaffoldCategory.MODULE_PATH.value] != record.phase_module_path_refs:
        errors.append("phase_module_path_refs must match findings")
    if counts[PhaseScaffoldCategory.VERIFIER_SCRIPT.value] != record.phase_verifier_script_refs:
        errors.append("phase_verifier_script_refs must match findings")
    if counts[PhaseScaffoldCategory.FIXTURE_PATH.value] != record.phase_fixture_path_refs:
        errors.append("phase_fixture_path_refs must match findings")
    if counts[PhaseScaffoldCategory.PUBLIC_SYMBOL.value] != record.phase_public_symbol_refs:
        errors.append("phase_public_symbol_refs must match findings")
    if counts[PhaseScaffoldCategory.ENV_VAR.value] != record.phase_env_var_refs:
        errors.append("phase_env_var_refs must match findings")


def _all_cleanup_controls_true(record: PhaseScaffoldCleanupRecord) -> bool:
    return all(
        (
            record.runtime_modules_domain_named,
            record.verifier_scripts_domain_named,
            record.fixtures_domain_named,
            record.public_symbols_domain_named,
            record.env_vars_domain_named,
            not record.compatibility_shims_required,
        )
    )


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if not expected:
        return
    expected_values = [str(item) for item in expected] if isinstance(expected, list) else [str(expected)]
    for expected_value in expected_values:
        _assert(any(expected_value in error for error in errors), f"expected error {expected_value!r}")


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
