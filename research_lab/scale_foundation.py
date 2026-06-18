"""Phase 4 foundation contracts for the Leadpoet Research Lab.

Phase 4 is scale and weights work. P4.0 defines local readiness records and
guards, but does not start confidential-GPU training, Stage-3 experiments,
agent-track GA, model-weight publication, research-slice changes, production
writes, or Supabase writes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .loop_foundation import TRUTHY_ENV_VALUES
from .model_pipeline_exit_gate import (
    PROTECTED_MODEL_PIPELINE_EXIT_KEYS,
    PROTECTED_MODEL_PIPELINE_EXIT_MARKERS,
    verify_model_pipeline_exit_gate,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "scale_foundation_fixtures.json"


class ScaleGate(str, Enum):
    MODEL_PIPELINE_EXIT_GATE_READY = "model_pipeline_exit_gate_ready"
    PACKAGE_MIGRATION_READY = "package_migration_ready"
    LEGACY_SURFACE_RETIRED = "legacy_surface_retired"
    PHASE_SCAFFOLD_CLEANUP_READY = "phase_scaffold_cleanup_ready"
    CONFIDENTIAL_GPU_EVIDENCE_READY = "confidential_gpu_evidence_ready"
    STAGE3_EXPERIMENT_EVIDENCE_READY = "stage3_experiment_evidence_ready"
    WORKSPACE_API_GA_EVIDENCE_READY = "workspace_api_ga_evidence_ready"
    RESEARCH_SLICE_GOVERNANCE_READY = "research_slice_governance_ready"


class ScaleReadinessState(str, Enum):
    LOCAL_CONTRACT_READY = "local_contract_ready"
    AWAITING_SCALE_EVIDENCE = "awaiting_scale_evidence"
    READY_AFTER_SCALE_EVIDENCE = "ready_after_scale_evidence"
    BLOCKED = "blocked"


SCALE_GATES: tuple[str, ...] = tuple(gate.value for gate in ScaleGate)

PROTECTED_SCALE_KEYS: frozenset[str] = frozenset(
    set(PROTECTED_MODEL_PIPELINE_EXIT_KEYS)
    | {
        "aws_secret_access_key",
        "confidential_gpu_key",
        "customer_training_row",
        "gpu_session_token",
        "kms_plaintext_key",
        "live_weight_checkpoint",
        "model_weight_blob",
        "private_training_example",
        "raw_axis_a_trace",
        "raw_axis_b_trace",
        "raw_stage3_trace",
        "research_slice_wallet_key",
        "sealed_eval_details",
        "stage3_reward_label",
        "workspace_api_secret",
    }
)

PROTECTED_SCALE_MARKERS: tuple[str, ...] = tuple(
    sorted(
        set(PROTECTED_MODEL_PIPELINE_EXIT_MARKERS)
        | {
            "aws secret",
            "confidential gpu key",
            "customer training row",
            "gpu session token",
            "kms plaintext",
            "live weight checkpoint",
            "model weight blob",
            "private training",
            "raw axis-a",
            "raw axis-b",
            "raw stage3",
            "research slice wallet",
            "sealed eval",
            "stage3 reward label",
            "workspace api secret",
        }
    )
)


@dataclass(frozen=True)
class ScaleWorkflowGuards:
    confidential_gpu_training: bool = False
    stage3_end_to_end_experiments: bool = False
    agent_track_ga: bool = False
    model_weight_publication: bool = False
    research_slice_governance_change: bool = False
    legacy_surface_enabled: bool = False
    phase_scaffold_runtime_public: bool = False
    production_writes: bool = False
    supabase_writes: bool = False
    public_workflows: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScaleWorkflowGuards":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleReadinessStatus:
    model_pipeline_exit_gate_ready: bool = False
    package_migration_ready: bool = False
    legacy_surface_retired: bool = False
    phase_scaffold_cleanup_ready: bool = False
    confidential_gpu_evidence_ready: bool = False
    stage3_experiment_evidence_ready: bool = False
    workspace_api_ga_evidence_ready: bool = False
    research_slice_governance_ready: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScaleReadinessStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleBuildStatus:
    scale_readiness_contracts: bool = False
    package_migration_audit: bool = False
    legacy_surface_retirement_guard: bool = False
    phase_scaffold_cleanup_verifier: bool = False
    confidential_gpu_contracts: bool = False
    stage3_experiment_contracts: bool = False
    agent_track_ga_gate: bool = False
    research_slice_governance_gate: bool = False
    scale_exit_gate_verifier: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScaleBuildStatus":
        fields = cls.__dataclass_fields__
        return cls(**{name: bool(data.get(name, False)) for name in fields})

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleReadinessClaimRecord:
    readiness_id: str
    scale_readiness: ScaleReadinessStatus
    build_status: ScaleBuildStatus
    local_code_complete: bool = False
    production_scale_ready: bool = False
    confidential_gpu_claimed_valid: bool = False
    stage3_success_claimed: bool = False
    agent_track_ga_claimed: bool = False
    model_weights_publication_claimed: bool = False
    research_slice_raise_claimed: bool = False
    uses_local_fixtures: bool = True
    local_only: bool = True
    model_pipeline_exit_gate_ref: str = ""
    evidence_refs: tuple[str, ...] = ()
    owner_approval_ref: str = ""
    state: str = ScaleReadinessState.AWAITING_SCALE_EVIDENCE.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ScaleReadinessClaimRecord":
        return cls(
            readiness_id=str(data["readiness_id"]),
            scale_readiness=ScaleReadinessStatus.from_mapping(data.get("scale_readiness", {})),
            build_status=ScaleBuildStatus.from_mapping(data.get("build_status", {})),
            local_code_complete=bool(data.get("local_code_complete", False)),
            production_scale_ready=bool(data.get("production_scale_ready", False)),
            confidential_gpu_claimed_valid=bool(data.get("confidential_gpu_claimed_valid", False)),
            stage3_success_claimed=bool(data.get("stage3_success_claimed", False)),
            agent_track_ga_claimed=bool(data.get("agent_track_ga_claimed", False)),
            model_weights_publication_claimed=bool(data.get("model_weights_publication_claimed", False)),
            research_slice_raise_claimed=bool(data.get("research_slice_raise_claimed", False)),
            uses_local_fixtures=bool(data.get("uses_local_fixtures", True)),
            local_only=bool(data.get("local_only", True)),
            model_pipeline_exit_gate_ref=str(data.get("model_pipeline_exit_gate_ref", "")),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            owner_approval_ref=str(data.get("owner_approval_ref", "")),
            state=str(data.get("state", ScaleReadinessState.AWAITING_SCALE_EVIDENCE.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale_readiness"] = self.scale_readiness.to_dict()
        data["build_status"] = self.build_status.to_dict()
        data["evidence_refs"] = list(self.evidence_refs)
        return data


def default_scale_workflow_guards() -> ScaleWorkflowGuards:
    return ScaleWorkflowGuards()


def scale_workflow_guards_from_env(env: Optional[Mapping[str, str]] = None) -> ScaleWorkflowGuards:
    env = env or {}
    return ScaleWorkflowGuards(
        confidential_gpu_training=_truthy(env.get("RESEARCH_LAB_SCALE_CONFIDENTIAL_GPU_TRAINING")),
        stage3_end_to_end_experiments=_truthy(env.get("RESEARCH_LAB_SCALE_STAGE3_END_TO_END_EXPERIMENTS")),
        agent_track_ga=_truthy(env.get("RESEARCH_LAB_SCALE_AGENT_TRACK_GA")),
        model_weight_publication=_truthy(env.get("RESEARCH_LAB_SCALE_MODEL_WEIGHT_PUBLICATION")),
        research_slice_governance_change=_truthy(env.get("RESEARCH_LAB_SCALE_RESEARCH_SLICE_GOVERNANCE_CHANGE")),
        legacy_surface_enabled=_truthy(env.get("RESEARCH_LAB_SCALE_LEGACY_SURFACE_ENABLED")),
        phase_scaffold_runtime_public=_truthy(env.get("RESEARCH_LAB_SCALE_PHASE_SCAFFOLD_RUNTIME_PUBLIC")),
        production_writes=_truthy(env.get("RESEARCH_LAB_SCALE_PRODUCTION_WRITES")),
        supabase_writes=_truthy(env.get("RESEARCH_LAB_SCALE_SUPABASE_WRITES")),
        public_workflows=_truthy(env.get("RESEARCH_LAB_SCALE_PUBLIC_WORKFLOWS")),
    )


def default_scale_readiness_status() -> ScaleReadinessStatus:
    return ScaleReadinessStatus()


def validate_scale_workflow_guards(guards: ScaleWorkflowGuards | Mapping[str, Any]) -> list[str]:
    if not isinstance(guards, ScaleWorkflowGuards):
        guards = ScaleWorkflowGuards.from_mapping(guards)
    enabled = [name for name, value in guards.to_dict().items() if value]
    if not enabled:
        return []
    return ["Phase 4 local build must keep these workflows disabled: " + ", ".join(enabled)]


def assert_scale_workflows_disabled(guards: ScaleWorkflowGuards | Mapping[str, Any]) -> None:
    errors = validate_scale_workflow_guards(guards)
    if errors:
        raise ValueError("; ".join(errors))


def scale_gate_ready(status: ScaleReadinessStatus | Mapping[str, Any], gate: ScaleGate | str) -> bool:
    if not isinstance(status, ScaleReadinessStatus):
        status = ScaleReadinessStatus.from_mapping(status)
    gate_value = gate.value if isinstance(gate, ScaleGate) else str(gate)
    if gate_value not in ScaleGate._value2member_map_:
        raise ValueError(f"unknown Phase 4 scale gate: {gate_value}")
    return bool(getattr(status, gate_value))


def require_scale_gate(status: ScaleReadinessStatus | Mapping[str, Any], gate: ScaleGate | str) -> None:
    if not scale_gate_ready(status, gate):
        gate_value = gate.value if isinstance(gate, ScaleGate) else str(gate)
        raise ValueError(f"Phase 4 scale gate is not ready: {gate_value}")


def scale_missing_scale_gates(status: ScaleReadinessStatus | Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(status, ScaleReadinessStatus):
        status = ScaleReadinessStatus.from_mapping(status)
    return tuple(gate for gate in SCALE_GATES if not scale_gate_ready(status, gate))


def scale_build_complete(status: ScaleBuildStatus | Mapping[str, Any]) -> bool:
    if not isinstance(status, ScaleBuildStatus):
        status = ScaleBuildStatus.from_mapping(status)
    return all(status.to_dict().values())


def validate_scale_readiness_claim(
    record: ScaleReadinessClaimRecord | Mapping[str, Any],
    *,
    guards: Optional[ScaleWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    raw = record if isinstance(record, Mapping) else record.to_dict()
    errors = _protected_scale_payload_errors(raw)
    if not isinstance(record, ScaleReadinessClaimRecord):
        try:
            record = ScaleReadinessClaimRecord.from_mapping(record)
        except KeyError as exc:
            errors.append(f"missing required Phase 4 readiness field: {exc.args[0]}")
            return errors
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid Phase 4 readiness payload: {exc}")
            return errors
    try:
        assert_scale_workflows_disabled(guards or default_scale_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))

    if record.state not in {state.value for state in ScaleReadinessState}:
        errors.append(f"unknown Phase 4 readiness state: {record.state}")
        return errors
    if not record.readiness_id.startswith("scale_readiness:"):
        errors.append("readiness_id must be scale_readiness:-prefixed")
    if record.model_pipeline_exit_gate_ref and not record.model_pipeline_exit_gate_ref.startswith("model_pipeline_exit_gate:"):
        errors.append("model_pipeline_exit_gate_ref must be model_pipeline_exit_gate:-prefixed")
    for evidence_ref in record.evidence_refs:
        if not evidence_ref.startswith(
            (
                "model_pipeline_exit_gate:",
                "package_migration:",
                "legacy_retirement:",
                "phase_scaffold_cleanup:",
                "confidential_gpu:",
                "gpu_attestation:",
                "kms_policy:",
                "stage3_experiment:",
                "heldout_eval:",
                "workspace_api_ga:",
                "workspace_api_beta:",
                "research_slice_governance:",
                "market_yield:",
                "counterfactual_gate:",
                "model_weight_release:",
                "owner_approval:",
            )
        ):
            errors.append("evidence_refs must use approved Phase 4 prefixes")
            break

    missing = scale_missing_scale_gates(record.scale_readiness)
    if record.production_scale_ready:
        if not record.local_code_complete:
            errors.append("production_scale_ready requires local_code_complete")
        if not scale_build_complete(record.build_status):
            errors.append("Phase 4 scale readiness requires all Phase 4 build fields true")
        if record.uses_local_fixtures:
            errors.append("Phase 4 scale readiness cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Phase 4 scale readiness cannot be claimed by a local_only record")
        if missing:
            errors.append("Phase 4 scale readiness claim is missing gates: " + ", ".join(missing))
        if not record.evidence_refs:
            errors.append("Phase 4 scale readiness claim requires evidence_refs")
        if not record.owner_approval_ref:
            errors.append("Phase 4 scale readiness claim requires owner_approval_ref")
        if record.state != ScaleReadinessState.READY_AFTER_SCALE_EVIDENCE.value:
            errors.append("production_scale_ready requires ready_after_scale_evidence state")
    else:
        if not record.local_code_complete:
            errors.append("Phase 4 local verifier should mark local_code_complete")
        if not record.local_only:
            errors.append("not-ready Phase 4 records must remain local_only")
        if record.state == ScaleReadinessState.READY_AFTER_SCALE_EVIDENCE.value:
            errors.append("ready_after_scale_evidence state requires production_scale_ready")

    _validate_scale_success_claims(record, errors)
    return errors


def verify_scale_foundation(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    model_pipeline_summary = verify_model_pipeline_exit_gate()
    fixture = _load_fixture(Path(fixture_path))

    _assert(
        fixture["scale_gate_values"] == list(SCALE_GATES),
        "fixture Phase 4 scale gates match code",
    )

    disabled_guards = ScaleWorkflowGuards.from_mapping(fixture["workflow_guards"]["disabled"])
    _assert(not validate_scale_workflow_guards(disabled_guards), "disabled Phase 4 workflow guards validate")

    invalid_guard_errors = validate_scale_workflow_guards(fixture["workflow_guards"]["invalid"])
    _assert(invalid_guard_errors, "enabled Phase 4 workflow guards fail closed")
    _assert_expected_error(invalid_guard_errors, fixture["workflow_guards"]["invalid"])

    env_guards = scale_workflow_guards_from_env(fixture["workflow_guard_env"])
    env_errors = validate_scale_workflow_guards(env_guards)
    _assert(env_errors, "truthy Phase 4 env flags fail closed")
    _assert_expected_error(env_errors, fixture["workflow_guard_env"])

    scale_status = ScaleReadinessStatus.from_mapping(fixture["scale_readiness_status"])
    for gate in SCALE_GATES:
        _assert(not scale_gate_ready(scale_status, gate), f"local Phase 4 scale gate defaults false: {gate}")
        try:
            require_scale_gate(scale_status, gate)
        except ValueError:
            pass
        else:
            raise AssertionError(f"missing Phase 4 scale gate raises: {gate}")

    local_claim = ScaleReadinessClaimRecord.from_mapping(fixture["readiness_claims"]["local_foundation_ready"])
    _assert(not validate_scale_readiness_claim(local_claim), "local Phase 4 foundation-ready claim validates")
    _assert(local_claim.local_code_complete, "P4.0 fixture marks local code complete")
    _assert(not local_claim.production_scale_ready, "P4.0 fixture does not claim scale readiness")
    _assert(not scale_build_complete(local_claim.build_status), "P4.0 does not claim later P4 slices built")
    _assert(scale_missing_scale_gates(local_claim.scale_readiness), "P4.0 fixture reports missing scale gates")
    _assert(
        local_claim.model_pipeline_exit_gate_ref == model_pipeline_summary["exit_gate_id"],
        "P4.0 fixture pins the P3.7 exit-gate ref",
    )
    _assert(not model_pipeline_summary["production_operational_ready"], "P3.7 remains not operationally ready")

    for record in fixture["readiness_claims"]["invalid"]:
        if record.get("raw"):
            merged = dict(record.get("overrides", {}))
        else:
            base = fixture["readiness_claims"][str(record.get("base", "local_foundation_ready"))]
            merged = _deep_merge(dict(base), dict(record.get("overrides", {})))
        errors = validate_scale_readiness_claim(merged)
        _assert(errors, f"invalid Phase 4 readiness claim fails: {record['id']}")
        _assert_expected_error(errors, record)

    unsafe_errors = validate_scale_readiness_claim(
        local_claim,
        guards=fixture["workflow_guards"]["invalid"],
    )
    _assert(unsafe_errors, "unsafe Phase 4 guards block readiness verifier")
    _assert_expected_error(unsafe_errors, fixture["workflow_guards"]["invalid"])

    return {
        "model_pipeline_exit_gate_id": model_pipeline_summary["exit_gate_id"],
        "model_pipeline_production_operational_ready": model_pipeline_summary["production_operational_ready"],
        "scale_gates": len(SCALE_GATES),
        "workflow_guard_fields": len(ScaleWorkflowGuards.__dataclass_fields__),
        "local_code_complete": local_claim.local_code_complete,
        "production_scale_ready": local_claim.production_scale_ready,
        "missing_scale_gates": list(scale_missing_scale_gates(local_claim.scale_readiness)),
        "readiness_id": local_claim.readiness_id,
    }


def _validate_scale_success_claims(record: ScaleReadinessClaimRecord, errors: list[str]) -> None:
    scale = record.scale_readiness
    if record.confidential_gpu_claimed_valid:
        if record.uses_local_fixtures:
            errors.append("confidential GPU validity cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("confidential GPU validity cannot be claimed by a local_only record")
        if not (scale.model_pipeline_exit_gate_ready and scale.confidential_gpu_evidence_ready):
            errors.append("confidential GPU validity requires Phase 3 exit and confidential GPU evidence")
        if not any(ref.startswith(("confidential_gpu:", "gpu_attestation:", "kms_policy:")) for ref in record.evidence_refs):
            errors.append("confidential GPU validity requires confidential_gpu, gpu_attestation, or kms_policy evidence")
        if not record.owner_approval_ref:
            errors.append("confidential GPU validity requires owner_approval_ref")
    if record.stage3_success_claimed:
        if record.uses_local_fixtures:
            errors.append("Stage-3 success cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Stage-3 success cannot be claimed by a local_only record")
        if not (scale.model_pipeline_exit_gate_ready and scale.stage3_experiment_evidence_ready):
            errors.append("Stage-3 success requires Phase 3 exit and Stage-3 experiment evidence")
        if not any(ref.startswith(("stage3_experiment:", "heldout_eval:")) for ref in record.evidence_refs):
            errors.append("Stage-3 success requires stage3_experiment or heldout_eval evidence")
        if not record.owner_approval_ref:
            errors.append("Stage-3 success requires owner_approval_ref")
    if record.agent_track_ga_claimed:
        if record.uses_local_fixtures:
            errors.append("Agent-track GA cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("Agent-track GA cannot be claimed by a local_only record")
        if not (scale.model_pipeline_exit_gate_ready and scale.workspace_api_ga_evidence_ready):
            errors.append("Agent-track GA requires Phase 3 exit and Workspace API GA evidence")
        if not any(ref.startswith(("workspace_api_ga:", "workspace_api_beta:")) for ref in record.evidence_refs):
            errors.append("Agent-track GA requires workspace_api_ga or workspace_api_beta evidence")
        if not record.owner_approval_ref:
            errors.append("Agent-track GA requires owner_approval_ref")
    if record.model_weights_publication_claimed:
        if record.uses_local_fixtures:
            errors.append("model-weight publication cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("model-weight publication cannot be claimed by a local_only record")
        if not (scale.confidential_gpu_evidence_ready and scale.stage3_experiment_evidence_ready):
            errors.append("model-weight publication requires confidential GPU and Stage-3 evidence")
        if not any(ref.startswith("model_weight_release:") for ref in record.evidence_refs):
            errors.append("model-weight publication requires model_weight_release evidence")
        if not record.owner_approval_ref:
            errors.append("model-weight publication requires owner_approval_ref")
    if record.research_slice_raise_claimed:
        if record.uses_local_fixtures:
            errors.append("research-slice raise cannot be claimed from local fixtures")
        if record.local_only:
            errors.append("research-slice raise cannot be claimed by a local_only record")
        if not (scale.model_pipeline_exit_gate_ready and scale.research_slice_governance_ready):
            errors.append("research-slice raise requires Phase 3 exit and governance evidence")
        if not any(ref.startswith(("research_slice_governance:", "market_yield:", "counterfactual_gate:")) for ref in record.evidence_refs):
            errors.append("research-slice raise requires governance, market_yield, or counterfactual_gate evidence")
        if not record.owner_approval_ref:
            errors.append("research-slice raise requires owner_approval_ref")


def _protected_scale_payload_errors(record: Any) -> list[str]:
    found = sorted(_find_protected_scale_material(record))
    if not found:
        return []
    return ["Phase 4 readiness payload contains protected material keys/markers: " + ", ".join(found)]


def _find_protected_scale_material(value: Any, path: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            key_path = f"{path}.{key_text}" if path else key_text
            if key_text in PROTECTED_SCALE_KEYS and not key_text.endswith(("_ref", "_refs")):
                found.add(key_path)
            found.update(_find_protected_scale_material(item, key_path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.update(_find_protected_scale_material(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        for marker in PROTECTED_SCALE_MARKERS:
            if marker in lowered:
                found.add(path or "<string>")
    return found


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in TRUTHY_ENV_VALUES


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
