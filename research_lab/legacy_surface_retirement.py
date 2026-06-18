"""Phase 4 legacy model-surface retirement guard.

P4.2 is a local static audit for the old qualification model-submission,
payment, and submission-credit surface. It does not remove routes, edit SQL, or
disable production behavior; it records whether the surface is still present and
fails closed until retirement or hard-gating evidence exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .package_migration_audit import verify_package_migration_audit
from .scale_foundation import (
    ScaleGate,
    require_scale_gate,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "legacy_surface_fixtures.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_ROOTS: tuple[str, ...] = (
    "gateway",
    "qualification",
    "scripts",
)

DEFAULT_IGNORED_PREFIXES: tuple[str, ...] = (
    "scripts/techstack_test_artifacts/",
)

SCAN_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".sql",
    ".plist",
)


class LegacySurfaceCategory(str, Enum):
    ROUTE = "route"
    TABLE = "table"
    PAYMENT = "payment"
    SUBMISSION_CREDIT = "submission_credit"
    ARTIFACT_COMPETITION = "artifact_competition"


class LegacySurfaceState(str, Enum):
    LOCAL_AUDIT_ONLY = "local_audit_only"
    READY_AFTER_RETIREMENT_EVIDENCE = "ready_after_retirement_evidence"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class LegacySurfaceMarker:
    marker_name: str
    needle: str
    category: str


LEGACY_SURFACE_MARKERS: tuple[LegacySurfaceMarker, ...] = (
    LegacySurfaceMarker("qualification_model_route", "/qualification/model", LegacySurfaceCategory.ROUTE.value),
    LegacySurfaceMarker("qualification_models_table", "qualification_models", LegacySurfaceCategory.TABLE.value),
    LegacySurfaceMarker("qualification_payments_table", "qualification_payments", LegacySurfaceCategory.PAYMENT.value),
    LegacySurfaceMarker("submission_credits_field", "submission_credits", LegacySurfaceCategory.SUBMISSION_CREDIT.value),
    LegacySurfaceMarker("submission_credit_phrase", "submission credit", LegacySurfaceCategory.SUBMISSION_CREDIT.value),
    LegacySurfaceMarker("model_submission_phrase", "model submission", LegacySurfaceCategory.ARTIFACT_COMPETITION.value),
    LegacySurfaceMarker("submitted_model_phrase", "submitted model", LegacySurfaceCategory.ARTIFACT_COMPETITION.value),
    LegacySurfaceMarker(
        "qualification_agent_competition_phrase",
        "lead qualification agent competition",
        LegacySurfaceCategory.ARTIFACT_COMPETITION.value,
    ),
)


@dataclass(frozen=True)
class LegacySurfaceFinding:
    path: str
    line: int
    marker_name: str
    category: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LegacySurfaceFinding":
        return cls(
            path=str(data["path"]),
            line=int(data["line"]),
            marker_name=str(data["marker_name"]),
            category=str(data["category"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LegacySurfaceRetirementRecord:
    audit_id: str
    scanned_roots: tuple[str, ...]
    ignored_prefixes: tuple[str, ...]
    legacy_route_refs: int = 0
    legacy_table_refs: int = 0
    legacy_payment_refs: int = 0
    submission_credit_refs: int = 0
    artifact_competition_refs: int = 0
    findings: tuple[LegacySurfaceFinding, ...] = ()
    audit_scope_note: str = "static_text_marker_scan_only"
    routes_hard_gated: bool = False
    legacy_tables_retired_or_read_only: bool = False
    legacy_payments_disabled: bool = False
    submission_credits_disabled: bool = False
    artifact_competition_disabled: bool = False
    legacy_surface_retired: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    retirement_plan_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False
    state: str = LegacySurfaceState.LOCAL_AUDIT_ONLY.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LegacySurfaceRetirementRecord":
        return cls(
            audit_id=str(data["audit_id"]),
            scanned_roots=tuple(str(item) for item in data.get("scanned_roots", [])),
            ignored_prefixes=tuple(str(item) for item in data.get("ignored_prefixes", [])),
            legacy_route_refs=int(data.get("legacy_route_refs", 0)),
            legacy_table_refs=int(data.get("legacy_table_refs", 0)),
            legacy_payment_refs=int(data.get("legacy_payment_refs", 0)),
            submission_credit_refs=int(data.get("submission_credit_refs", 0)),
            artifact_competition_refs=int(data.get("artifact_competition_refs", 0)),
            findings=tuple(LegacySurfaceFinding.from_mapping(item) for item in data.get("findings", [])),
            audit_scope_note=str(data.get("audit_scope_note", "static_text_marker_scan_only")),
            routes_hard_gated=bool(data.get("routes_hard_gated", False)),
            legacy_tables_retired_or_read_only=bool(data.get("legacy_tables_retired_or_read_only", False)),
            legacy_payments_disabled=bool(data.get("legacy_payments_disabled", False)),
            submission_credits_disabled=bool(data.get("submission_credits_disabled", False)),
            artifact_competition_disabled=bool(data.get("artifact_competition_disabled", False)),
            legacy_surface_retired=bool(data.get("legacy_surface_retired", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            retirement_plan_ref=str(data.get("retirement_plan_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            production_writes=bool(data.get("production_writes", False)),
            supabase_writes=bool(data.get("supabase_writes", False)),
            public_workflows=bool(data.get("public_workflows", False)),
            state=str(data.get("state", LegacySurfaceState.LOCAL_AUDIT_ONLY.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scanned_roots"] = list(self.scanned_roots)
        data["ignored_prefixes"] = list(self.ignored_prefixes)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def scan_legacy_model_surface(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> tuple[LegacySurfaceFinding, ...]:
    repo = Path(repo_root)
    findings: list[LegacySurfaceFinding] = []
    for path in _iter_scan_files(repo, roots, ignored_prefixes):
        rel_path = path.relative_to(repo).as_posix()
        findings.extend(_scan_file_for_legacy_surface_markers(path, rel_path))
    return tuple(sorted(findings, key=lambda finding: (finding.path, finding.line, finding.category, finding.marker_name)))


def build_current_legacy_surface_retirement_record(
    *,
    repo_root: Path | str = REPO_ROOT,
    roots: Sequence[str] = DEFAULT_SCAN_ROOTS,
    ignored_prefixes: Sequence[str] = DEFAULT_IGNORED_PREFIXES,
) -> LegacySurfaceRetirementRecord:
    findings = scan_legacy_model_surface(repo_root=repo_root, roots=roots, ignored_prefixes=ignored_prefixes)
    counts = _category_counts(findings)
    return LegacySurfaceRetirementRecord(
        audit_id="legacy_surface_retirement:p4.2:local-current-audit",
        scanned_roots=tuple(roots),
        ignored_prefixes=tuple(ignored_prefixes),
        legacy_route_refs=counts[LegacySurfaceCategory.ROUTE.value],
        legacy_table_refs=counts[LegacySurfaceCategory.TABLE.value],
        legacy_payment_refs=counts[LegacySurfaceCategory.PAYMENT.value],
        submission_credit_refs=counts[LegacySurfaceCategory.SUBMISSION_CREDIT.value],
        artifact_competition_refs=counts[LegacySurfaceCategory.ARTIFACT_COMPETITION.value],
        findings=findings,
        audit_scope_note="static_text_marker_scan_only",
        routes_hard_gated=False,
        legacy_tables_retired_or_read_only=False,
        legacy_payments_disabled=False,
        submission_credits_disabled=False,
        artifact_competition_disabled=False,
        legacy_surface_retired=False,
        uses_local_fixtures=True,
        local_only=True,
        retirement_plan_ref="legacy_surface_retirement_plan:p4.2:pending",
        evidence_refs=("legacy_surface_retirement:p4.2:local-current-audit",),
        state=LegacySurfaceState.LOCAL_AUDIT_ONLY.value,
    )


def legacy_surface_retirement_blockers(record: LegacySurfaceRetirementRecord | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(record, LegacySurfaceRetirementRecord):
        record = LegacySurfaceRetirementRecord.from_mapping(record)
    blockers: list[str] = []
    if record.legacy_route_refs and not record.routes_hard_gated:
        blockers.append("legacy_route_surface_not_hard_gated")
    if record.legacy_table_refs and not record.legacy_tables_retired_or_read_only:
        blockers.append("legacy_tables_not_retired_or_read_only")
    if record.legacy_payment_refs and not record.legacy_payments_disabled:
        blockers.append("legacy_payments_not_disabled")
    if record.submission_credit_refs and not record.submission_credits_disabled:
        blockers.append("submission_credits_not_disabled")
    if record.artifact_competition_refs and not record.artifact_competition_disabled:
        blockers.append("artifact_competition_not_disabled")
    return tuple(blockers)


def validate_legacy_surface_retirement_record(record: LegacySurfaceRetirementRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, LegacySurfaceRetirementRecord):
        try:
            record = LegacySurfaceRetirementRecord.from_mapping(record)
        except KeyError as exc:
            return [f"missing required legacy surface retirement field: {exc.args[0]}"]
        except (TypeError, ValueError) as exc:
            return [f"invalid legacy surface retirement payload: {exc}"]

    errors: list[str] = []
    if record.state not in {state.value for state in LegacySurfaceState}:
        errors.append(f"unknown legacy surface retirement state: {record.state}")
        return errors
    if not record.audit_id.startswith("legacy_surface_retirement:"):
        errors.append("audit_id must be legacy_surface_retirement:-prefixed")
    if not record.scanned_roots:
        errors.append("legacy surface retirement audit requires scanned_roots")
    if record.audit_scope_note != "static_text_marker_scan_only":
        errors.append("audit_scope_note must be static_text_marker_scan_only")
    if record.retirement_plan_ref and not record.retirement_plan_ref.startswith("legacy_surface_retirement_plan:"):
        errors.append("retirement_plan_ref must be legacy_surface_retirement_plan:-prefixed")
    _append_count_consistency_errors(record, errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "legacy_surface_retirement:",
                "legacy_surface_retirement_plan:",
                "route_gate:",
                "table_retirement:",
                "payment_disablement:",
                "submission_credit_retirement:",
                "artifact_competition_retirement:",
                "scale_package_migration_plan:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved legacy surface retirement prefixes")
            break
    if record.production_writes:
        errors.append("P4.2 legacy surface retirement guard must not enable production writes")
    if record.supabase_writes:
        errors.append("P4.2 legacy surface retirement guard must not enable Supabase writes")
    if record.public_workflows:
        errors.append("P4.2 legacy surface retirement guard must not enable public workflows")

    blockers = legacy_surface_retirement_blockers(record)
    if record.legacy_surface_retired:
        if blockers:
            errors.append("legacy surface retired claim has blockers: " + ", ".join(blockers))
        if not _all_retirement_controls_true(record):
            errors.append("legacy surface retired claim requires all retirement controls true")
        if record.uses_local_fixtures:
            errors.append("legacy surface retirement cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("legacy surface retirement cannot be claimed by a local_only record")
        if not record.evidence_refs:
            errors.append("legacy surface retirement requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("legacy surface retirement requires owner_approval_ref")
        if record.state != LegacySurfaceState.READY_AFTER_RETIREMENT_EVIDENCE.value:
            errors.append("legacy_surface_retired requires ready_after_retirement_evidence state")
    else:
        if not record.local_only:
            errors.append("not-ready legacy surface retirement audits must remain local_only")
        if record.state == LegacySurfaceState.READY_AFTER_RETIREMENT_EVIDENCE.value:
            errors.append("ready_after_retirement_evidence state requires legacy_surface_retired")
    return errors


def verify_legacy_surface_retirement(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    package_summary = verify_package_migration_audit()
    fixture = _load_fixture(Path(fixture_path))

    local_record = LegacySurfaceRetirementRecord.from_mapping(fixture["local_blocked_sample"])
    _assert(not validate_legacy_surface_retirement_record(local_record), "local legacy-surface blocked sample validates")
    _assert(not local_record.legacy_surface_retired, "P4.2 local sample does not claim retirement")
    _assert(legacy_surface_retirement_blockers(local_record), "P4.2 local sample reports blockers")

    ready_record = LegacySurfaceRetirementRecord.from_mapping(fixture["ready_control"])
    _assert(not validate_legacy_surface_retirement_record(ready_record), "ready-control legacy-surface retirement validates")
    _assert(not legacy_surface_retirement_blockers(ready_record), "ready-control legacy-surface retirement has no blockers")

    current_record = build_current_legacy_surface_retirement_record()
    _assert(not validate_legacy_surface_retirement_record(current_record), "current legacy-surface retirement audit validates")
    _assert(not current_record.legacy_surface_retired, "current legacy surface audit remains not retired")
    current_blockers = legacy_surface_retirement_blockers(current_record)
    _assert(current_blockers, "current legacy surface audit has retirement blockers")
    _assert(current_record.legacy_route_refs > 0, "current audit detects legacy qualification model routes")
    _assert(current_record.legacy_table_refs > 0, "current audit detects legacy qualification model tables")
    _assert(current_record.legacy_payment_refs > 0, "current audit detects legacy payment surface")
    _assert(current_record.submission_credit_refs > 0, "current audit detects submission credits")
    _assert(current_record.artifact_competition_refs > 0, "current audit detects artifact-competition mechanics")

    for invalid in fixture["invalid_retirement_audits"]:
        base = fixture[str(invalid.get("base", "local_blocked_sample"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_legacy_surface_retirement_record(merged)
        _assert(errors, f"invalid legacy-surface retirement audit fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    try:
        require_scale_gate({"legacy_surface_retired": False}, ScaleGate.LEGACY_SURFACE_RETIRED)
    except ValueError:
        pass
    else:
        raise AssertionError("P4.2 legacy-surface retirement gate remains required by P4.0")

    return {
        "package_migration_audit_id": package_summary["current_audit_id"],
        "audit_id": local_record.audit_id,
        "current_audit_id": current_record.audit_id,
        "legacy_route_refs": current_record.legacy_route_refs,
        "legacy_table_refs": current_record.legacy_table_refs,
        "legacy_payment_refs": current_record.legacy_payment_refs,
        "submission_credit_refs": current_record.submission_credit_refs,
        "artifact_competition_refs": current_record.artifact_competition_refs,
        "legacy_surface_retired": current_record.legacy_surface_retired,
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


def _scan_file_for_legacy_surface_markers(path: Path, rel_path: str) -> list[LegacySurfaceFinding]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    findings: list[LegacySurfaceFinding] = []
    for line_number, line in enumerate(lines, start=1):
        lowered = line.lower()
        for marker in LEGACY_SURFACE_MARKERS:
            if marker.needle in lowered:
                findings.append(
                    LegacySurfaceFinding(
                        path=rel_path,
                        line=line_number,
                        marker_name=marker.marker_name,
                        category=marker.category,
                    )
                )
    return findings


def _category_counts(findings: Sequence[LegacySurfaceFinding]) -> dict[str, int]:
    counts = {category.value: 0 for category in LegacySurfaceCategory}
    for finding in findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
    return counts


def _append_count_consistency_errors(record: LegacySurfaceRetirementRecord, errors: list[str]) -> None:
    counts = _category_counts(record.findings)
    if counts[LegacySurfaceCategory.ROUTE.value] != record.legacy_route_refs:
        errors.append("legacy_route_refs must match findings")
    if counts[LegacySurfaceCategory.TABLE.value] != record.legacy_table_refs:
        errors.append("legacy_table_refs must match findings")
    if counts[LegacySurfaceCategory.PAYMENT.value] != record.legacy_payment_refs:
        errors.append("legacy_payment_refs must match findings")
    if counts[LegacySurfaceCategory.SUBMISSION_CREDIT.value] != record.submission_credit_refs:
        errors.append("submission_credit_refs must match findings")
    if counts[LegacySurfaceCategory.ARTIFACT_COMPETITION.value] != record.artifact_competition_refs:
        errors.append("artifact_competition_refs must match findings")


def _all_retirement_controls_true(record: LegacySurfaceRetirementRecord) -> bool:
    return all(
        (
            record.routes_hard_gated,
            record.legacy_tables_retired_or_read_only,
            record.legacy_payments_disabled,
            record.submission_credits_disabled,
            record.artifact_competition_disabled,
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
