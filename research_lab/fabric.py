"""Phase 1 tiered run-fabric record contracts.

P1.2 defines the shape of eval-service, sandbox, notary-proxy, and attestation
records without provisioning hosts, starting enclaves, routing network egress,
releasing sealed keys, or claiming production-valid attestation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Sequence

from leadpoet_verifier.attestation import validate_attestation_response_shape

from .canonical import sha256_json
from .evidence import build_evidence_bundle
from .notary import LocalSnapshotStore, NotarySigner, capture_snapshot
from .loop_foundation import (
    LoopWorkflowGuards,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    verify_loop_foundation,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "fabric_fixtures.json"


class FabricRunStatus(str, Enum):
    PLANNED = "planned"
    RECORDED = "recorded"
    CRASH = "crash"
    TIMEOUT = "timeout"


class SandboxJobStatus(str, Enum):
    QUEUED = "queued"
    COMPLETED = "completed"
    CRASH = "crash"
    TIMEOUT = "timeout"
    GUARDRAIL_BREACH = "guardrail_breach"


class AttestationAnchorState(str, Enum):
    SHAPE_RECORDED_ONLY = "shape_recorded_only"
    STUB_NOT_ANCHORED = "stub_not_anchored"


@dataclass(frozen=True)
class SandboxResourceCaps:
    cpu_cores: int
    memory_mb: int
    disk_mb: int
    wall_clock_cap_s: int
    max_cost_cents: int
    network_disabled: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SandboxResourceCaps":
        return cls(
            cpu_cores=int(data["cpu_cores"]),
            memory_mb=int(data["memory_mb"]),
            disk_mb=int(data["disk_mb"]),
            wall_clock_cap_s=int(data["wall_clock_cap_s"]),
            max_cost_cents=int(data["max_cost_cents"]),
            network_disabled=bool(data.get("network_disabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalServiceRunRecord:
    run_id: str
    ticket_id: str
    run_role: str
    rung: str
    status: str
    enclave_measurement_ref: str
    verifier_image_ref: str
    verifier_hash: str
    judge_version_hash: str
    egress_policy_ref: str
    evidence_bundle_refs: tuple[str, ...]
    attestation_ref: str
    icp_set_commitment: str
    output_hash: str
    cost_ledger_hash: str
    lab_only: bool = True
    enclave_started: bool = False
    sealed_keys_released: bool = False
    production_valid_attestation: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvalServiceRunRecord":
        return cls(
            run_id=str(data["run_id"]),
            ticket_id=str(data["ticket_id"]),
            run_role=str(data["run_role"]),
            rung=str(data["rung"]),
            status=str(data["status"]),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            verifier_image_ref=str(data["verifier_image_ref"]),
            verifier_hash=str(data["verifier_hash"]),
            judge_version_hash=str(data["judge_version_hash"]),
            egress_policy_ref=str(data["egress_policy_ref"]),
            evidence_bundle_refs=tuple(str(ref) for ref in data.get("evidence_bundle_refs", [])),
            attestation_ref=str(data["attestation_ref"]),
            icp_set_commitment=str(data["icp_set_commitment"]),
            output_hash=str(data["output_hash"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            lab_only=bool(data.get("lab_only", True)),
            enclave_started=bool(data.get("enclave_started", False)),
            sealed_keys_released=bool(data.get("sealed_keys_released", False)),
            production_valid_attestation=bool(data.get("production_valid_attestation", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_bundle_refs"] = list(self.evidence_bundle_refs)
        return data


@dataclass(frozen=True)
class ResearchSandboxJobRecord:
    job_id: str
    ticket_id: str
    image_ref: str
    sandbox_policy_ref: str
    resource_caps: SandboxResourceCaps
    mounted_artifact_refs: tuple[str, ...]
    status: str
    estimated_cost_cents: int
    actual_cost_cents: int = 0
    crash_reason: Optional[str] = None
    timeout_s: Optional[int] = None
    guardrail_reason: Optional[str] = None
    lab_only: bool = True
    provisioned_host: bool = False
    network_enabled: bool = False
    enclave_attested: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResearchSandboxJobRecord":
        return cls(
            job_id=str(data["job_id"]),
            ticket_id=str(data["ticket_id"]),
            image_ref=str(data["image_ref"]),
            sandbox_policy_ref=str(data["sandbox_policy_ref"]),
            resource_caps=SandboxResourceCaps.from_mapping(data["resource_caps"]),
            mounted_artifact_refs=tuple(str(ref) for ref in data.get("mounted_artifact_refs", [])),
            status=str(data["status"]),
            estimated_cost_cents=int(data["estimated_cost_cents"]),
            actual_cost_cents=int(data.get("actual_cost_cents", 0)),
            crash_reason=data.get("crash_reason"),
            timeout_s=data.get("timeout_s"),
            guardrail_reason=data.get("guardrail_reason"),
            lab_only=bool(data.get("lab_only", True)),
            provisioned_host=bool(data.get("provisioned_host", False)),
            network_enabled=bool(data.get("network_enabled", False)),
            enclave_attested=bool(data.get("enclave_attested", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resource_caps"] = self.resource_caps.to_dict()
        data["mounted_artifact_refs"] = list(self.mounted_artifact_refs)
        return data


@dataclass(frozen=True)
class NotaryProxyEgressStub:
    egress_id: str
    run_id: str
    url: str
    method: str
    vsock_route_ref: str
    egress_policy_ref: str
    snapshot_ref: str
    content_hash: str
    normalized_text_hash: str
    snapshot_signature: str
    evidence_bundle_ref: str
    live_network_performed: bool = False
    artifact_raw_egress_access: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "NotaryProxyEgressStub":
        return cls(
            egress_id=str(data["egress_id"]),
            run_id=str(data["run_id"]),
            url=str(data["url"]),
            method=str(data.get("method", "GET")),
            vsock_route_ref=str(data["vsock_route_ref"]),
            egress_policy_ref=str(data["egress_policy_ref"]),
            snapshot_ref=str(data["snapshot_ref"]),
            content_hash=str(data["content_hash"]),
            normalized_text_hash=str(data["normalized_text_hash"]),
            snapshot_signature=str(data["snapshot_signature"]),
            evidence_bundle_ref=str(data["evidence_bundle_ref"]),
            live_network_performed=bool(data.get("live_network_performed", False)),
            artifact_raw_egress_access=bool(data.get("artifact_raw_egress_access", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttestationAnchorStub:
    attestation_ref: str
    run_id: str
    enclave_measurement_ref: str
    engine_version_ref: str
    verifier_image_ref: str
    verifier_hash: str
    judge_version_hash: str
    pcr0_allowlist_ref: str
    icp_set_commitment: str
    output_hash: str
    evidence_bundle_merkle_root: str
    cost_ledger_hash: str
    attestation_response: dict[str, Any]
    anchor_state: str = AttestationAnchorState.SHAPE_RECORDED_ONLY.value
    allowlist_checked: bool = False
    cryptographic_verification_performed: bool = False
    production_valid: bool = False
    signed_anchor_ref: str = "anchor:pending"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AttestationAnchorStub":
        return cls(
            attestation_ref=str(data["attestation_ref"]),
            run_id=str(data["run_id"]),
            enclave_measurement_ref=str(data["enclave_measurement_ref"]),
            engine_version_ref=str(data["engine_version_ref"]),
            verifier_image_ref=str(data["verifier_image_ref"]),
            verifier_hash=str(data["verifier_hash"]),
            judge_version_hash=str(data["judge_version_hash"]),
            pcr0_allowlist_ref=str(data["pcr0_allowlist_ref"]),
            icp_set_commitment=str(data["icp_set_commitment"]),
            output_hash=str(data["output_hash"]),
            evidence_bundle_merkle_root=str(data["evidence_bundle_merkle_root"]),
            cost_ledger_hash=str(data["cost_ledger_hash"]),
            attestation_response=dict(data["attestation_response"]),
            anchor_state=str(data.get("anchor_state", AttestationAnchorState.SHAPE_RECORDED_ONLY.value)),
            allowlist_checked=bool(data.get("allowlist_checked", False)),
            cryptographic_verification_performed=bool(data.get("cryptographic_verification_performed", False)),
            production_valid=bool(data.get("production_valid", False)),
            signed_anchor_ref=str(data.get("signed_anchor_ref", "anchor:pending")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_eval_service_run_record(record: EvalServiceRunRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, EvalServiceRunRecord):
        record = EvalServiceRunRecord.from_mapping(record)
    errors: list[str] = []
    if record.status not in {status.value for status in FabricRunStatus}:
        errors.append(f"unknown fabric run status: {record.status}")
    if record.rung not in {"L0", "L1", "L2", "L3", "L4", "anchor"}:
        errors.append(f"unknown eval rung: {record.rung}")
    for field in (
        "run_id",
        "ticket_id",
        "run_role",
        "enclave_measurement_ref",
        "verifier_image_ref",
        "verifier_hash",
        "judge_version_hash",
        "egress_policy_ref",
        "attestation_ref",
        "icp_set_commitment",
        "output_hash",
        "cost_ledger_hash",
    ):
        if not getattr(record, field):
            errors.append(f"{field} is required")
    if not record.evidence_bundle_refs:
        errors.append("evidence_bundle_refs must not be empty")
    if not record.lab_only:
        errors.append("eval-service records must remain lab_only in P1.2")
    if record.enclave_started:
        errors.append("P1.2 must not start a live enclave")
    if record.sealed_keys_released:
        errors.append("P1.2 must not release sealed keys")
    if record.production_valid_attestation:
        errors.append("P1.2 must not mark attestation as production-valid")
    return errors


def validate_sandbox_job_record(record: ResearchSandboxJobRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, ResearchSandboxJobRecord):
        record = ResearchSandboxJobRecord.from_mapping(record)
    errors = validate_sandbox_resource_caps(record.resource_caps)
    if record.status not in {status.value for status in SandboxJobStatus}:
        errors.append(f"unknown sandbox job status: {record.status}")
    if not record.mounted_artifact_refs:
        errors.append("mounted_artifact_refs must not be empty")
    if record.estimated_cost_cents < 0 or record.actual_cost_cents < 0:
        errors.append("cost metadata must be non-negative")
    if record.status == SandboxJobStatus.CRASH.value and not record.crash_reason:
        errors.append("crash status requires crash_reason")
    if record.status == SandboxJobStatus.TIMEOUT.value and not record.timeout_s:
        errors.append("timeout status requires timeout_s")
    if record.status == SandboxJobStatus.GUARDRAIL_BREACH.value and not record.guardrail_reason:
        errors.append("guardrail_breach status requires guardrail_reason")
    if not record.lab_only:
        errors.append("sandbox jobs must remain lab_only in P1.2")
    if record.provisioned_host:
        errors.append("P1.2 must not provision sandbox hosts")
    if record.network_enabled:
        errors.append("P1.2 sandbox jobs must not enable network")
    if record.enclave_attested:
        errors.append("Phase-1 research jobs are sandbox-isolated, not enclave-attested")
    return errors


def validate_sandbox_resource_caps(caps: SandboxResourceCaps | Mapping[str, Any]) -> list[str]:
    if not isinstance(caps, SandboxResourceCaps):
        caps = SandboxResourceCaps.from_mapping(caps)
    errors: list[str] = []
    for field in ("cpu_cores", "memory_mb", "disk_mb", "wall_clock_cap_s", "max_cost_cents"):
        if getattr(caps, field) <= 0:
            errors.append(f"{field} must be positive")
    if not caps.network_disabled:
        errors.append("sandbox resource caps must keep network_disabled=true")
    return errors


def validate_notary_proxy_egress_stub(record: NotaryProxyEgressStub | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, NotaryProxyEgressStub):
        record = NotaryProxyEgressStub.from_mapping(record)
    errors: list[str] = []
    if record.method not in {"GET", "POST"}:
        errors.append("method must be GET or POST")
    for field in (
        "egress_id",
        "run_id",
        "url",
        "vsock_route_ref",
        "egress_policy_ref",
        "snapshot_ref",
        "content_hash",
        "normalized_text_hash",
        "snapshot_signature",
        "evidence_bundle_ref",
    ):
        if not getattr(record, field):
            errors.append(f"{field} is required")
    if not record.content_hash.startswith("sha256:"):
        errors.append("content_hash must be sha256-prefixed")
    if not record.normalized_text_hash.startswith("sha256:"):
        errors.append("normalized_text_hash must be sha256-prefixed")
    if record.live_network_performed:
        errors.append("P1.2 notary-proxy egress stubs must not perform live network egress")
    if record.artifact_raw_egress_access:
        errors.append("artifact code must not receive raw egress access")
    return errors


def validate_attestation_anchor_stub(record: AttestationAnchorStub | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, AttestationAnchorStub):
        record = AttestationAnchorStub.from_mapping(record)
    errors: list[str] = []
    if record.anchor_state not in {state.value for state in AttestationAnchorState}:
        errors.append(f"unknown anchor_state: {record.anchor_state}")
    for field in (
        "attestation_ref",
        "run_id",
        "enclave_measurement_ref",
        "engine_version_ref",
        "verifier_image_ref",
        "verifier_hash",
        "judge_version_hash",
        "pcr0_allowlist_ref",
        "icp_set_commitment",
        "output_hash",
        "evidence_bundle_merkle_root",
        "cost_ledger_hash",
    ):
        if not getattr(record, field):
            errors.append(f"{field} is required")
    shape = validate_attestation_response_shape(record.attestation_response)
    if not shape["passed"]:
        errors.extend(f"attestation response shape: {error}" for error in shape["errors"])
    if record.allowlist_checked:
        errors.append("P1.2 records allowlist inputs but does not claim allowlist verification")
    if record.cryptographic_verification_performed:
        errors.append("P1.2 must not claim Nitro/COSE cryptographic verification")
    if record.production_valid:
        errors.append("P1.2 must not mark attestation as production-valid")
    if record.signed_anchor_ref != "anchor:pending":
        errors.append("P1.2 anchor stubs must remain pending, not signed/anchored")
    return errors


def validate_fabric_bundle(
    *,
    eval_run: EvalServiceRunRecord | Mapping[str, Any],
    sandbox_job: ResearchSandboxJobRecord | Mapping[str, Any],
    egress_stub: NotaryProxyEgressStub | Mapping[str, Any],
    attestation_anchor: AttestationAnchorStub | Mapping[str, Any],
    guards: LoopWorkflowGuards | Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(validate_eval_service_run_record(eval_run))
    errors.extend(validate_sandbox_job_record(sandbox_job))
    errors.extend(validate_notary_proxy_egress_stub(egress_stub))
    errors.extend(validate_attestation_anchor_stub(attestation_anchor))
    return errors


def build_notary_proxy_egress_stub(
    *,
    egress_id: str,
    run_id: str,
    url: str,
    snapshot: Mapping[str, Any],
    evidence_bundle_ref: str,
    egress_policy_ref: str,
    vsock_route_ref: str,
    method: str = "GET",
) -> NotaryProxyEgressStub:
    return NotaryProxyEgressStub(
        egress_id=egress_id,
        run_id=run_id,
        url=url,
        method=method,
        vsock_route_ref=vsock_route_ref,
        egress_policy_ref=egress_policy_ref,
        snapshot_ref=str(snapshot["snapshot_ref"]),
        content_hash=str(snapshot["content_hash"]),
        normalized_text_hash=str(snapshot["normalized_text_hash"]),
        snapshot_signature=str(snapshot["signature"]),
        evidence_bundle_ref=evidence_bundle_ref,
    )


def verify_research_lab_fabric(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    foundation_summary = verify_loop_foundation()
    fixture = _load_fixture(Path(fixture_path))

    with tempfile.TemporaryDirectory(prefix="leadpoet_p12_snapshot_") as tmpdir:
        snapshot = capture_snapshot(
            url=fixture["local_snapshot"]["url"],
            content=fixture["local_snapshot"]["content"],
            store=LocalSnapshotStore(tmpdir),
            signer=NotarySigner("loop-fabric-local-test", key_id="loop-fabric"),
            fetch_ts=fixture["local_snapshot"]["fetch_ts"],
            metadata={"source": "p1.2-local-fixture"},
        )
        evidence_bundle = build_evidence_bundle(
            artifact_hash=fixture["eval_run"]["output_hash"],
            snapshots=[snapshot],
            run_id=fixture["eval_run"]["run_id"],
            created_at=fixture["local_snapshot"]["fetch_ts"],
            validate=True,
        )

    evidence_bundle_ref = evidence_bundle["bundle_hash"]
    eval_run_data = dict(fixture["eval_run"])
    eval_run_data["evidence_bundle_refs"] = [evidence_bundle_ref]
    eval_run = EvalServiceRunRecord.from_mapping(eval_run_data)

    egress_stub = build_notary_proxy_egress_stub(
        egress_id=fixture["egress_stub"]["egress_id"],
        run_id=eval_run.run_id,
        url=fixture["local_snapshot"]["url"],
        snapshot=snapshot.to_schema_snapshot(),
        evidence_bundle_ref=evidence_bundle_ref,
        egress_policy_ref=eval_run.egress_policy_ref,
        vsock_route_ref=fixture["egress_stub"]["vsock_route_ref"],
        method=fixture["egress_stub"]["method"],
    )
    sandbox_job = ResearchSandboxJobRecord.from_mapping(fixture["sandbox_job"])
    attestation_anchor = AttestationAnchorStub.from_mapping(fixture["attestation_anchor"])

    _assert(not validate_eval_service_run_record(eval_run), "eval-service run record validates")
    _assert(not validate_sandbox_job_record(sandbox_job), "sandbox job record validates")
    _assert(not validate_notary_proxy_egress_stub(egress_stub), "notary-proxy egress stub validates")
    _assert(not validate_attestation_anchor_stub(attestation_anchor), "attestation anchor stub validates")
    _assert(
        not validate_fabric_bundle(
            eval_run=eval_run,
            sandbox_job=sandbox_job,
            egress_stub=egress_stub,
            attestation_anchor=attestation_anchor,
        ),
        "fabric bundle validates",
    )

    for invalid in fixture["invalid_records"]:
        kind = invalid["kind"]
        record = _invalid_record_payload(fixture, invalid)
        if kind == "eval_run":
            errors = validate_eval_service_run_record(record)
        elif kind == "sandbox_job":
            errors = validate_sandbox_job_record(record)
        elif kind == "egress_stub":
            errors = validate_notary_proxy_egress_stub(record)
        elif kind == "attestation_anchor":
            errors = validate_attestation_anchor_stub(record)
        elif kind == "fabric_bundle_guards":
            errors = validate_fabric_bundle(
                eval_run=eval_run,
                sandbox_job=sandbox_job,
                egress_stub=egress_stub,
                attestation_anchor=attestation_anchor,
                guards=record,
            )
        else:
            raise AssertionError(f"unknown invalid fixture kind: {kind}")
        _assert(errors, f"invalid {kind} fixture fails: {invalid['id']}")
        expected = invalid.get("expected_error_contains")
        if expected:
            _assert(
                any(str(expected) in error for error in errors),
                f"invalid {kind} fixture has expected error {expected!r}: {invalid['id']}",
            )

    return {
        "foundation_invalid_release_records": foundation_summary["invalid_release_records"],
        "evidence_bundle_ref": evidence_bundle_ref,
        "snapshot_ref": snapshot.snapshot_ref,
        "invalid_records": len(fixture["invalid_records"]),
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _invalid_record_payload(fixture: Mapping[str, Any], invalid: Mapping[str, Any]) -> dict[str, Any]:
    if "record" in invalid:
        return dict(invalid["record"])
    base_name = invalid.get("base")
    if not base_name:
        raise AssertionError(f"invalid fixture needs record or base: {invalid.get('id')}")
    base = dict(fixture[str(base_name)])
    return _deep_merge(base, dict(invalid.get("overrides", {})))


def _deep_merge(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
