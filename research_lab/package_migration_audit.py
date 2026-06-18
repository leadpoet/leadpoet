"""Phase 4 package-migration audit contracts.

P4.1 is a local static audit for the Phase-4 requirement that Research Lab
runtime code move toward a Leadpoet-native eval/scoring substrate and stop
depending on legacy qualification package imports before GA-scale work.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scale_foundation import (
    ScaleGate,
    require_scale_gate,
    verify_scale_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "package_migration_fixtures.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

LEGACY_IMPORT_PREFIXES: tuple[str, ...] = (
    "qualification",
    "gateway.qualification",
)

DEFAULT_SCAN_ROOTS: tuple[str, ...] = (
    "research_lab",
    "leadpoet_verifier",
    "scripts",
)

DEFAULT_IGNORED_PREFIXES: tuple[str, ...] = (
    "scripts/techstack_test_artifacts/",
)


class LegacyImportCategory(str, Enum):
    RESEARCH_LAB_RUNTIME = "research_lab_runtime"
    OPERATIONAL_SCRIPT = "operational_script"
    TEST_OR_ARCHIVE = "test_or_archive"


class PackageMigrationState(str, Enum):
    LOCAL_AUDIT_ONLY = "local_audit_only"
    READY_AFTER_MIGRATION_EVIDENCE = "ready_after_migration_evidence"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class LegacyImportFinding:
    path: str
    line: int
    import_name: str
    category: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LegacyImportFinding":
        return cls(
            path=str(data["path"]),
            line=int(data["line"]),
            import_name=str(data["import_name"]),
            category=str(data["category"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackageMigrationAuditRecord:
    audit_id: str
    scanned_roots: tuple[str, ...]
    ignored_prefixes: tuple[str, ...]
    leadpoet_native_package_ref: str
    leadpoet_native_package_present: bool = False
    research_lab_runtime_legacy_imports: int = 0
    operational_script_legacy_imports: int = 0
    test_or_archive_legacy_imports: int = 0
    findings: tuple[LegacyImportFinding, ...] = ()
    audit_scope_note: str = "static_ast_imports_only"
    package_migration_ready: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    migration_plan_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = PackageMigrationState.LOCAL_AUDIT_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PackageMigrationAuditRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            scanned_roots=tuple(str(item) for item in data.get("scanned_roots", [])),
            ignored_prefixes=tuple(str(item) for item in data.get("ignored_prefixes", [])),
            leadpoet_native_package_ref=str(data.get("leadpoet_native_package_ref", "")),
            leadpoet_native_package_present=bool(data.get("leadpoet_native_package_present", False)),
            research_lab_runtime_legacy_imports=int(data.get("research_lab_runtime_legacy_imports", 0)),
            operational_script_legacy_imports=int(data.get("operational_script_legacy_imports", 0)),
            test_or_archive_legacy_imports=int(data.get("test_or_archive_legacy_imports", 0)),
            findings=tuple(LegacyImportFinding.from_mapping(item) for item in data.get("findings", [])),
            audit_scope_note=str(data.get("audit_scope_note", "static_ast_imports_only")),
            package_migration_ready=bool(data.get("package_migration_ready", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            migration_plan_ref=str(data.get("migration_plan_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", PackageMigrationState.LOCAL_AUDIT_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scanned_roots"] = list(self.scanned_roots)
        data["ignored_prefixes"] = list(self.ignored_prefixes)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def scan_legacy_imports(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> tuple[LegacyImportFinding, ...]:
    repo = Path(repo_root)
    findings: list[LegacyImportFinding] = []
    for path in _iter_python_files(repo, roots, ignored_prefixes):
        rel_path = path.relative_to(repo).as_posix()
        findings.extend(_scan_python_file_for_legacy_imports(path, rel_path))
    return tuple(sorted(findings, key=lambda finding: (finding.path, finding.line, finding.import_name)))


def build_current_package_migration_audit_record(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> PackageMigrationAuditRecord:
    repo = Path(repo_root)
    findings = scan_legacy_imports(repo_root=repo, roots=roots, ignored_prefixes=ignored_prefixes)
    counts = _category_counts(findings)
    native_package_present = (repo / "research_lab" / "eval").is_dir()
    return PackageMigrationAuditRecord(
        audit_id="package_migration:p4.1:local-current-audit",
        scanned_roots=tuple(roots),
        ignored_prefixes=tuple(ignored_prefixes),
        leadpoet_native_package_ref="research_lab/eval:pending",
        leadpoet_native_package_present=native_package_present,
        research_lab_runtime_legacy_imports=counts[LegacyImportCategory.RESEARCH_LAB_RUNTIME.value],
        operational_script_legacy_imports=counts[LegacyImportCategory.OPERATIONAL_SCRIPT.value],
        test_or_archive_legacy_imports=counts[LegacyImportCategory.TEST_OR_ARCHIVE.value],
        findings=findings,
        audit_scope_note="static_ast_imports_only",
        package_migration_ready=False,
        uses_local_fixtures=True,
        local_only=True,
        migration_plan_ref="scale_package_migration_plan:p4.1:pending",
        evidence_refs=("package_migration:p4.1:local-current-audit",),
        state=PackageMigrationState.LOCAL_AUDIT_ONLY.value,
    )


def package_migration_blockers(record: PackageMigrationAuditRecord | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(record, PackageMigrationAuditRecord):
        record = PackageMigrationAuditRecord.from_mapping(record)
    blockers: list[str] = []
    if not record.leadpoet_native_package_present:
        blockers.append("leadpoet_native_package_missing")
    if record.research_lab_runtime_legacy_imports:
        blockers.append("research_lab_runtime_legacy_imports")
    if record.operational_script_legacy_imports:
        blockers.append("operational_script_legacy_imports")
    return tuple(blockers)


def validate_package_migration_audit_record(record: PackageMigrationAuditRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, PackageMigrationAuditRecord):
        record = PackageMigrationAuditRecord.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in PackageMigrationState}:
        errors.append(f"unknown package migration state: {record.state}")
        return errors
    if not record.audit_id.startswith("package_migration:"):
        errors.append("audit_id must be package_migration:-prefixed")
    if not record.scanned_roots:
        errors.append("package migration audit requires scanned_roots")
    if record.audit_scope_note != "static_ast_imports_only":
        errors.append("audit_scope_note must be static_ast_imports_only")
    if not record.leadpoet_native_package_ref.startswith("research_lab/eval:"):
        errors.append("leadpoet_native_package_ref must be research_lab/eval:-prefixed")
    if record.migration_plan_ref and not record.migration_plan_ref.startswith("scale_package_migration_plan:"):
        errors.append("migration_plan_ref must be scale_package_migration_plan:-prefixed")
    _append_count_consistency_errors(record, errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith((
            "package_migration:",
            "scale_package_migration_plan:",
            "model_pipeline_exit_gate:",
            "owner_approval:",
        )):
            errors.append("evidence_refs must use approved package migration prefixes")
            break
    if record.production_writes:
        errors.append("P4.1 package migration audit must not enable production writes")
    if record.supabase_writes:
        errors.append("P4.1 package migration audit must not enable Supabase writes")
    if record.public_workflows:
        errors.append("P4.1 package migration audit must not enable public workflows")

    blockers = package_migration_blockers(record)
    if record.package_migration_ready:
        if blockers:
            errors.append("package migration ready claim has blockers: " + ", ".join(blockers))
        if record.uses_local_fixtures:
            errors.append("package migration readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("package migration readiness cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("package migration readiness requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("package migration readiness requires owner_approval_ref")
        if record.state != PackageMigrationState.READY_AFTER_MIGRATION_EVIDENCE.value:
            errors.append("package_migration_ready requires ready_after_migration_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready package migration audits must remain local_only")
        if record.state == PackageMigrationState.READY_AFTER_MIGRATION_EVIDENCE.value:
            errors.append("ready_after_migration_evidence state requires package_migration_ready")
    return errors


def verify_package_migration_audit(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    scale_summary = verify_scale_foundation()
    fixture = _load_fixture(Path(fixture_path))

    local_record = PackageMigrationAuditRecord.from_mapping(fixture["local_audit"])
    _assert(not validate_package_migration_audit_record(local_record), "local package migration audit validates")
    _assert(not local_record.package_migration_ready, "P4.1 local audit does not claim migration ready")
    _assert(package_migration_blockers(local_record), "P4.1 local fixture reports migration blockers")

    ready_record = PackageMigrationAuditRecord.from_mapping(fixture["ready_control"])
    _assert(not validate_package_migration_audit_record(ready_record), "ready-control package migration audit validates")
    _assert(not package_migration_blockers(ready_record), "ready-control package migration audit has no blockers")

    current_record = build_current_package_migration_audit_record()
    _assert(not validate_package_migration_audit_record(current_record), "current package migration audit validates")
    _assert(
        current_record.research_lab_runtime_legacy_imports == 0,
        "Research Lab package has no direct legacy qualification imports",
    )
    _assert(
        not current_record.package_migration_ready,
        "current package migration audit remains not ready",
    )

    for invalid in fixture["invalid_audits"]:
        base = fixture[str(invalid.get("base", "local_audit"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_package_migration_audit_record(merged)
        _assert(errors, f"invalid package migration audit fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"package_migration_ready": False}, ScaleGate.PACKAGE_MIGRATION_READY)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.1 package migration gate remains required by P4.0")

    return {
        "scale_readiness_id": scale_summary["readiness_id"],
        "audit_id": local_record.audit_id,
        "current_audit_id": current_record.audit_id,
        "leadpoet_native_package_present": current_record.leadpoet_native_package_present,
        "research_lab_runtime_legacy_imports": current_record.research_lab_runtime_legacy_imports,
        "operational_script_legacy_imports": current_record.operational_script_legacy_imports,
        "test_or_archive_legacy_imports": current_record.test_or_archive_legacy_imports,
        "package_migration_ready": current_record.package_migration_ready,
        "blockers": list(package_migration_blockers(current_record)),
        "current_findings": len(current_record.findings),
    }


def _iter_python_files(repo: Path, roots: Sequence[str], ignored_prefixes: Sequence[str]) -> Iterable[Path]:
    for root in roots:
        path = repo / root
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else path.rglob("*.py")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix != ".py":
                continue
            rel = candidate.relative_to(repo).as_posix()
            if any(rel.startswith(prefix) for prefix in ignored_prefixes):
                continue
            yield candidate


def _scan_python_file_for_legacy_imports(path: Path, rel_path: str) -> list[LegacyImportFinding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    except SyntaxError:
        return []
    findings: list[LegacyImportFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy_import(alias.name):
                    findings.append(
                        LegacyImportFinding(
                            path=rel_path,
                            line=int(getattr(node, "lineno", 0)),
                            import_name=alias.name,
                            category=_categorize_path(rel_path),
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_legacy_import(module):
                findings.append(
                    LegacyImportFinding(
                        path=rel_path,
                        line=int(getattr(node, "lineno", 0)),
                        import_name=module,
                        category=_categorize_path(rel_path),
                    )
                )
    return findings


def _is_legacy_import(import_name: str) -> bool:
    return any(import_name == prefix or import_name.startswith(prefix + ".") for prefix in LEGACY_IMPORT_PREFIXES)


def _categorize_path(rel_path: str) -> str:
    if rel_path.startswith(("research_lab/", "leadpoet_verifier/")):
        return LegacyImportCategory.RESEARCH_LAB_RUNTIME.value
    if rel_path.startswith("scripts/verify_") or "/test" in rel_path or rel_path.endswith("_test.py"):
        return LegacyImportCategory.TEST_OR_ARCHIVE.value
    if rel_path.startswith("scripts/"):
        return LegacyImportCategory.OPERATIONAL_SCRIPT.value
    return LegacyImportCategory.TEST_OR_ARCHIVE.value


def _category_counts(findings: Sequence[LegacyImportFinding]) -> dict[str, int]:
    counts = {category.value: 0 for category in LegacyImportCategory}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    return counts


def _append_count_consistency_errors(record: PackageMigrationAuditRecord, errors: list[str]) -> None:
    counts = _category_counts(record.findings)
    if counts[LegacyImportCategory.RESEARCH_LAB_RUNTIME.value] != record.research_lab_runtime_legacy_imports:
        errors.append("research_lab_runtime_legacy_imports must match findings")
    if counts[LegacyImportCategory.OPERATIONAL_SCRIPT.value] != record.operational_script_legacy_imports:
        errors.append("operational_script_legacy_imports must match findings")
    if counts[LegacyImportCategory.TEST_OR_ARCHIVE.value] != record.test_or_archive_legacy_imports:
        errors.append("test_or_archive_legacy_imports must match findings")


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
