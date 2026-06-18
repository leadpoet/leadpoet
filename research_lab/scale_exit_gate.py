"""Phase 4 exit-gate verifier contracts.

P4.8 composes P4.1-P4.7 into a local capstone readiness record. It marks the
Phase 4 local build complete, but it remains fail-closed for production scale
until measured evidence is supplied for every scale gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .agent_track_ga import verify_research_lab_agent_track_ga
from .confidential_gpu_training import PROTECTED_CONFIDENTIAL_GPU_KEYS, PROTECTED_CONFIDENTIAL_GPU_MARKERS, verify_research_lab_confidential_gpu_training
from .legacy_surface_retirement import verify_legacy_surface_retirement
from .package_migration_audit import verify_package_migration_audit
from .scale_foundation import (
    SCALE_GATES,
    PROTECTED_SCALE_KEYS,
    PROTECTED_SCALE_MARKERS,
    ScaleBuildStatus,
    ScaleGate,
    ScaleReadinessStatus,
    ScaleWorkflowGuards,
    assert_scale_workflows_disabled,
    default_scale_workflow_guards,
    scale_build_complete,
    verify_scale_foundation,
)
from .model_pipeline_exit_gate import verify_model_pipeline_exit_gate
from .phase_scaffold_cleanup import verify_phase_scaffold_cleanup
from .research_slice_governance import verify_research_lab_research_slice_governance
from .stage3_experiment import verify_research_lab_stage3_experiment


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "scale_exit_gate_fixtures.json"

SCALE_EXIT_GATE_CONTRACT_VERSION = "scale_exit_gate:v1:local_contract"

PROTECTED_SCALE_EXIT_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_SCALE_KEYS)
    | set(PROTECTED_CONFIDENTIAL_GPU_KEYS)
    | {
        "production_deploy_token",
        "public_weight_blob",
        "raw_scale_evidence",
        "research_slice_wallet_key",
        "treasury_private_key",
    }
)

PROTECTED_SCALE_EXIT_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_SCALE_MARKERS)
        | set(PROTECTED_CONFIDENTIAL_GPU_MARKERS)
        | {
            "production deploy token",
            "public weight blob",
            "raw scale evidence",
            "research slice wallet",
            "treasury private key",
        }
    )
)


class ScaleExitGateState(str, Enum):
    LOCAL_NOT_READY = "local_not_ready"
    READY_AFTER_SCALE_EVIDENCE = "ready_after_scale_evidence"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ScaleExitGateReadinessRecord:
    exit_gate_id: str
    scale_readiness_ref: str
    model_pipeline_exit_gate_ref: str
    package_migration_ref: str
    legacy_surface_ref: str
    phase_scaffold_ref: str
    confidential_gpu_ref: str
    stage3_success_ref: str
    workspace_api_ga_ref: str
    research_slice_governance_ref: str
    scale_readiness: ScaleReadinessStatus
    build_status: ScaleBuildStatus
    evidence_refs: tuple[str, ...]
    owner_approval_ref: str = ""
    local_code_complete: bool = False
    production_scale_ready: bool = False
    package_migration_ready: bool = False
    legacy_surface_retired: bool = False
    phase_scaffold_cleanup_ready: bool = False
    confidential_gpu_evidence_ready: bool = False
    stage3_experiment_evidence_ready: bool = False
    workspace_api_ga_evidence_ready: bool = False
    research_slice_governance_ready: bool = False
    model_weight_publication_enabled: bool = False
    public_agent_track_enabled: bool = False
    research_slice_raise_executed: bool = False
    production_deployment_requested: bool = False
    production_writes_enabled: bool = False
    supabase_writes_enabled: bool = False
    public_workflows_enabled: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    state: str = ScaleExitGateState.LOCAL_NOT_READY.value
    contract_version: str = SCALE_EXIT_GATE_CONTRACT_VERSION

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScaleExitGateReadinessRecord":
        return cls(
            exit_gate_id=str(data["exit_gate_id"]),
            scale_readiness_ref=str(data["scale_readiness_ref"]),
            model_pipeline_exit_gate_ref=str(data["model_pipeline_exit_gate_ref"]),
            package_migration_ref=str(data["package_migration_ref"]),
            legacy_surface_ref=str(data["legacy_surface_ref"]),
            phase_scaffold_ref=str(data["phase_scaffold_ref"]),
            confidential_gpu_ref=str(data["confidential_gpu_ref"]),
            stage3_success_ref=str(data["stage3_success_ref"]),
            workspace_api_ga_ref=str(data["workspace_api_ga_ref"]),
            research_slice_governance_ref=str(data["research_slice_governance_ref"]),
            scale_readiness=ScaleReadinessStatus.from_mapping(data.get("scale_readiness", {})),
            build_status=ScaleBuildStatus.from_mapping(data.get("build_status", {})),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            local_code_complete=bool(data.get("local_code_complete", False)),
            production_scale_ready=bool(data.get("production_scale_ready", False)),
            package_migration_ready=bool(data.get("package_migration_ready", False)),
            legacy_surface_retired=bool(data.get("legacy_surface_retired", False)),
            phase_scaffold_cleanup_ready=bool(data.get("phase_scaffold_cleanup_ready", False)),
            confidential_gpu_evidence_ready=bool(data.get("confidential_gpu_evidence_ready", False)),
            stage3_experiment_evidence_ready=bool(data.get("stage3_experiment_evidence_ready", False)),
            workspace_api_ga_evidence_ready=bool(data.get("workspace_api_ga_evidence_ready", False)),
            research_slice_governance_ready=bool(data.get("research_slice_governance_ready", False)),
            model_weight_publication_enabled=bool(data.get("model_weight_publication_enabled", False)),
            public_agent_track_enabled=bool(data.get("public_agent_track_enabled", False)),
            research_slice_raise_executed=bool(data.get("research_slice_raise_executed", False)),
            production_deployment_requested=bool(data.get("production_deployment_requested", False)),
            production_writes_enabled=bool(data.get("production_writes_enabled", False)),
            supabase_writes_enabled=bool(data.get("supabase_writes_enabled", False)),
            public_workflows_enabled=bool(data.get("public_workflows_enabled", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            state=str(data.get("state", ScaleExitGateState.LOCAL_NOT_READY.value)),
            contract_version=str(data.get("contract_version", SCALE_EXIT_GATE_CONTRACT_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale_readiness"] = self.scale_readiness.to_dict()
        data["build_status"] = self.build_status.to_dict()
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def scale_exit_gate_check_results(
    record: ScaleExitGateReadinessRecord | Mapping[str, Any],
) -> dict[str, bool]:
    if not isinstance(record, ScaleExitGateReadinessRecord):
        record = ScaleExitGateReadinessRecord.from_mapping(record)
    scale = record.scale_readiness
    return {
        ScaleGate.MODEL_PIPELINE_EXIT_GATE_READY.value: bool(scale.model_pipeline_exit_gate_ready),
        ScaleGate.PACKAGE_MIGRATION_READY.value: (
            bool(scale.package_migration_ready)
            and record.package_migration_ready
            and record.package_migration_ref.startswith("package_migration:")
        ),
        ScaleGate.LEGACY_SURFACE_RETIRED.value: (
            bool(scale.legacy_surface_retired)
            and record.legacy_surface_retired
            and record.legacy_surface_ref.startswith("legacy_surface_retirement:")
        ),
        ScaleGate.PHASE_SCAFFOLD_CLEANUP_READY.value: (
            bool(scale.phase_scaffold_cleanup_ready)
            and record.phase_scaffold_cleanup_ready
            and record.phase_scaffold_ref.startswith("phase_scaffold_cleanup:")
        ),
        ScaleGate.CONFIDENTIAL_GPU_EVIDENCE_READY.value: (
            bool(scale.confidential_gpu_evidence_ready)
            and record.confidential_gpu_evidence_ready
            and record.confidential_gpu_ref.startswith("confidential_gpu_training:")
        ),
        ScaleGate.STAGE3_EXPERIMENT_EVIDENCE_READY.value: (
            bool(scale.stage3_experiment_evidence_ready)
            and record.stage3_experiment_evidence_ready
            and record.stage3_success_ref.startswith("stage3_success:")
        ),
        ScaleGate.WORKSPACE_API_GA_EVIDENCE_READY.value: (
            bool(scale.workspace_api_ga_evidence_ready)
            and record.workspace_api_ga_evidence_ready
            and record.workspace_api_ga_ref.startswith("agent_track_ga:")
        ),
        ScaleGate.RESEARCH_SLICE_GOVERNANCE_READY.value: (
            bool(scale.research_slice_governance_ready)
            and record.research_slice_governance_ready
            and record.research_slice_governance_ref.startswith("research_slice_decision:")
        ),
    }


def scale_exit_gate_missing_checks(
    record: ScaleExitGateReadinessRecord | Mapping[str, Any],
) -> tuple[str, ...]:
    return tuple(
        check for check, passed in scale_exit_gate_check_results(record).items()
        if not passed
    )


def validate_scale_exit_gate_readiness_record(
    record: ScaleExitGateReadinessRecord | Mapping[str, Any],
    *,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_scale_exit_payload_errors(raw)
    if not isinstance(record, ScaleExitGateReadinessRecord):
        try:
            record = ScaleExitGateReadinessRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Phase 4 exit-gate field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Phase 4 exit-gate payload: {exc}")
            return errors
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    if record.contract_version != SCALE_EXIT_GATE_CONTRACT_VERSION:
        errors.append("contract_version must match Phase 4 exit-gate contract")
    if record.state not in {state.value for state in ScaleExitGateState}:
        errors.append(f"unknown Phase 4 exit-gate state: {record.state}")
        return errors
    if not record.exit_gate_id.startswith("scale_exit_gate:"):
        errors.append("exit_gate_id must be scale_exit_gate:-prefixed")
    _append_prefix_errors(
        record,
        errors,
        (
            ("scale_readiness_ref", "scale_readiness:"),
            ("model_pipeline_exit_gate_ref", "model_pipeline_exit_gate:"),
            ("package_migration_ref", "package_migration:"),
            ("legacy_surface_ref", "legacy_surface_retirement:"),
            ("phase_scaffold_ref", "phase_scaffold_cleanup:"),
            ("confidential_gpu_ref", "confidential_gpu_training:"),
            ("stage3_success_ref", "stage3_success:"),
            ("workspace_api_ga_ref", "agent_track_ga:"),
            ("research_slice_governance_ref", "research_slice_decision:"),
        ),
    )
    if tuple(scale_exit_gate_check_results(record).keys()) != SCALE_GATES:
        errors.append("Phase 4 exit-gate checks must match Phase 4 scale gates")
    if record.scale_readiness.to_dict() != {
        "model_pipeline_exit_gate_ready": record.scale_readiness.model_pipeline_exit_gate_ready,
        "package_migration_ready": record.package_migration_ready,
        "legacy_surface_retired": record.legacy_surface_retired,
        "phase_scaffold_cleanup_ready": record.phase_scaffold_cleanup_ready,
        "confidential_gpu_evidence_ready": record.confidential_gpu_evidence_ready,
        "stage3_experiment_evidence_ready": record.stage3_experiment_evidence_ready,
        "workspace_api_ga_evidence_ready": record.workspace_api_ga_evidence_ready,
        "research_slice_governance_ready": record.research_slice_governance_ready,
    }:
        errors.append("Phase 4 exit-gate scale flags must mirror scale_readiness")
    _append_live_enablement_errors(record, errors)
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "scale_readiness:",
                "model_pipeline_exit_gate:",
                "package_migration:",
                "legacy_surface_retirement:",
                "phase_scaffold_cleanup:",
                "confidential_gpu_training:",
                "stage3_success:",
                "agent_track_ga:",
                "research_slice_decision:",
                "owner_approval:",
            )
        ):
            errors.append("Phase 4 exit-gate evidence_refs must use approved prefixes")
            break
    missing = scale_exit_gate_missing_checks(record)
    if record.production_scale_ready:
        if not record.local_code_complete:
            errors.append("production_scale_ready requires local_code_complete")
        if not scale_build_complete(record.build_status):
            errors.append("Phase 4 exit requires all Phase 4 build fields complete")
        if record.uses_local_fixtures:
            errors.append("Phase 4 exit cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 4 exit cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 4 exit missing checks: " + ", ".join(missing))
        if not record.owner_approval_ref:
            errors.append("Phase 4 exit requires owner_approval_ref")
        for required_ref in (
            record.scale_readiness_ref,
            record.model_pipeline_exit_gate_ref,
            record.package_migration_ref,
            record.legacy_surface_ref,
            record.phase_scaffold_ref,
            record.confidential_gpu_ref,
            record.stage3_success_ref,
            record.workspace_api_ga_ref,
            record.research_slice_governance_ref,
        ):
            if required_ref not in record.evidence_refs:
                errors.append("Phase 4 exit evidence_refs must include all core evidence refs")
                break
        if record.state != ScaleExitGateState.READY_AFTER_SCALE_EVIDENCE.value:
            errors.append("production_scale_ready requires ready_after_scale_evidence state")
    else:
        if not record.local_code_complete:
            errors.append("Phase 4 exit local verifier should mark local_code_complete")
        if not record.local_only:
            errors.append("not-ready Phase 4 exit records must remain local_only")
        if record.state == ScaleExitGateState.READY_AFTER_SCALE_EVIDENCE.value:
            errors.append("ready_after_scale_evidence state requires production_scale_ready")
    return errors


def verify_scale_exit_gate(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    model_pipeline_summary = verify_model_pipeline_exit_gate()
    scale_summary = verify_scale_foundation()
    package_summary = verify_package_migration_audit()
    legacy_summary = verify_legacy_surface_retirement()
    scaffold_summary = verify_phase_scaffold_cleanup()
    gpu_summary = verify_research_lab_confidential_gpu_training()
    stage3_summary = verify_research_lab_stage3_experiment()
    agent_summary = verify_research_lab_agent_track_ga()
    slice_summary = verify_research_lab_research_slice_governance()
    fixture = _load_fixture(Path(fixture_path))

    _assert(fixture["scale_gate_values"] == list(SCALE_GATES), "P4.8 fixture covers all Phase 4 scale gates")
    local_exit = ScaleExitGateReadinessRecord.from_mapping(fixture["local_exit_gate"])
    _assert(not validate_scale_exit_gate_readiness_record(local_exit), "local Phase 4 exit-gate record validates")
    _assert(local_exit.local_code_complete, "P4.8 local verifier marks code complete")
    _assert(not local_exit.production_scale_ready, "P4.8 local verifier does not claim production scale readiness")
    _assert(scale_exit_gate_missing_checks(local_exit), "P4.8 reports missing measured scale checks")
    _assert(local_exit.scale_readiness_ref == scale_summary["readiness_id"], "P4.8 pins P4.0 readiness ref")
    _assert(local_exit.model_pipeline_exit_gate_ref == model_pipeline_summary["exit_gate_id"], "P4.8 pins P3.7 exit-gate ref")
    _assert(local_exit.package_migration_ref == package_summary["audit_id"], "P4.8 pins P4.1 audit ref")
    _assert(local_exit.legacy_surface_ref == legacy_summary["current_audit_id"], "P4.8 pins P4.2 current audit ref")
    _assert(local_exit.phase_scaffold_ref == scaffold_summary["current_audit_id"], "P4.8 pins P4.3 current audit ref")
    _assert(
        local_exit.phase_scaffold_cleanup_ready == scaffold_summary["phase_scaffold_cleanup_ready"],
        "P4.8 mirrors P4.3 cleanup readiness",
    )
    _assert(
        local_exit.scale_readiness.phase_scaffold_cleanup_ready == scaffold_summary["phase_scaffold_cleanup_ready"],
        "P4.8 scale readiness mirrors P4.3 cleanup readiness",
    )
    _assert(local_exit.confidential_gpu_ref == gpu_summary["training_run_id"], "P4.8 pins P4.4 training ref")
    _assert(local_exit.stage3_success_ref == stage3_summary["success_claim_id"], "P4.8 pins P4.5 success ref")
    _assert(local_exit.workspace_api_ga_ref == agent_summary["ga_readiness_id"], "P4.8 pins P4.6 GA ref")
    _assert(local_exit.research_slice_governance_ref == slice_summary["decision_id"], "P4.8 pins P4.7 decision ref")

    measured_exit = ScaleExitGateReadinessRecord.from_mapping(fixture["measured_exit_gate"])
    measured_errors = validate_scale_exit_gate_readiness_record(measured_exit)
    _assert(not measured_errors, "fully supplied measured Phase 4 exit gate validates")

    for invalid in fixture["invalid_exit_gates"]:
        base = fixture[str(invalid.get("base", "local_exit_gate"))]
        merged = _deep_merge(dict(base), dict(invalid.get("overrides", {})))
        errors = validate_scale_exit_gate_readiness_record(merged)
        _assert(errors, f"invalid Phase 4 exit gate fails: {invalid['id']}")
        _assert_expected_error(errors, invalid)

    unsafe_errors = validate_scale_exit_gate_readiness_record(
        local_exit,
        guards=fixture["unsafe_scale_guards"],
    )
    _assert(unsafe_errors, "unsafe Phase 4 guards block P4.8 verifier")
    _assert_expected_error(unsafe_errors, fixture["unsafe_scale_guards"])

    return {
        "exit_gate_id": local_exit.exit_gate_id,
        "local_code_complete": local_exit.local_code_complete,
        "production_scale_ready": local_exit.production_scale_ready,
        "missing_checks": list(scale_exit_gate_missing_checks(local_exit)),
        "measured_control_validates": not measured_errors,
        "scale_gates": list(SCALE_GATES),
        "package_migration_ready": local_exit.package_migration_ready,
        "legacy_surface_retired": local_exit.legacy_surface_retired,
        "phase_scaffold_cleanup_ready": local_exit.phase_scaffold_cleanup_ready,
        "confidential_gpu_evidence_ready": local_exit.confidential_gpu_evidence_ready,
        "stage3_experiment_evidence_ready": local_exit.stage3_experiment_evidence_ready,
        "workspace_api_ga_evidence_ready": local_exit.workspace_api_ga_evidence_ready,
        "research_slice_governance_ready": local_exit.research_slice_governance_ready,
    }


def _append_prefix_errors(record: Any, errors: list[str], checks: Sequence[tuple[str, str]]) -> None:
    for field, prefix in checks:
        if not getattr(record, field).startswith(prefix):
            errors.append(f"{field} must be {prefix}-prefixed")


def _append_live_enablement_errors(record: ScaleExitGateReadinessRecord, errors: list[str]) -> None:
    if record.model_weight_publication_enabled:
        errors.append("P4.8 must not enable model-weight publication")
    if record.public_agent_track_enabled:
        errors.append("P4.8 must not enable public agent track")
    if record.research_slice_raise_executed:
        errors.append("P4.8 must not execute research-slice raises")
    if record.production_deployment_requested:
        errors.append("P4.8 must not request production deployment")
    if record.production_writes_enabled:
        errors.append("P4.8 must not enable production writes")
    if record.supabase_writes_enabled:
        errors.append("P4.8 must not enable Supabase writes")
    if record.public_workflows_enabled:
        errors.append("P4.8 must not enable public workflows")


def _protected_scale_exit_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_scale_exit_material(record))
    if not found:
        return []
    return ["P4.8 Phase 4 exit-gate payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_scale_exit_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_SCALE_EXIT_KEYS and not key_text.endswith(
                ("_ref", "_refs", "_hash", "_hashes")
            ):
                found.add(key_path)
            found.update(_find_protected_scale_exit_material(nested, key_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.update(_find_protected_scale_exit_material(nested, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_SCALE_EXIT_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


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
