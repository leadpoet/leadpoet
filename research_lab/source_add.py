"""Phase 1 SOURCE_ADD coverage-track records.

P1.5 defines the launch-surface evidence-adapter contracts, sandbox-review
stubs, trial-yield bounty records, and IP-assignment acknowledgements. It does
not execute adapter code, accept public components, settle bounties, or write to
production systems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .canonical import sha256_json
from .fabric import (
    ResearchSandboxJobRecord,
    SandboxJobStatus,
    validate_sandbox_job_record,
    verify_research_lab_fabric,
)
from .loop_foundation import (
    ArtifactReleaseState,
    LoopWorkflowGuards,
    ReleasePolicyRecord,
    VisibilityPolicy,
    assert_loop_workflows_disabled,
    default_loop_workflow_guards,
    validate_release_policy,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "source_add_fixtures.json"

SOURCE_ADD_OUTPUT_FIELDS: tuple[str, ...] = (
    "evidence_refs",
    "snapshot_refs",
    "content_hashes",
    "normalized_text_hashes",
    "metadata_refs",
)

RAW_OUTPUT_FIELDS: tuple[str, ...] = (
    "content",
    "raw_content",
    "html",
    "raw_html",
    "text",
    "page_text",
    "scraped_content",
    "body",
    "raw_response",
)

# These are defense-in-depth tripwires for declared schemas. The hard guarantee
# is the ref/hash-only adapter contract plus the disabled sandbox execution path.
RAW_CREDENTIAL_KEYS: tuple[str, ...] = (
    "api_key",
    "password",
    "secret",
    "token",
    "credential_value",
    "raw_credential",
)


class SourceAddSourceKind(str, Enum):
    WEB = "web"
    FILING = "filing"
    NEWS = "news"
    REGISTRY = "registry"
    PROCUREMENT = "procurement"
    SOCIAL = "social"


class SourceAddReviewStatus(str, Enum):
    PLANNED = "planned"
    RECORDED = "recorded"
    ACCEPTABLE_PENDING_HUMAN_GATE = "acceptable_pending_human_gate"
    REJECTED = "rejected"


class SourceAddBountyState(str, Enum):
    CALCULATED_NOT_PAYABLE = "calculated_not_payable"
    REJECTED_NOT_PAYABLE = "rejected_not_payable"


class SourceAddIPState(str, Enum):
    ACKNOWLEDGED_PENDING_ACCEPTANCE = "acknowledged_pending_acceptance"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SourceAddAdapterManifest:
    adapter_id: str
    miner_ref: str
    source_name: str
    source_kind: str
    declared_base_domains: tuple[str, ...]
    output_schema_ref: str
    allowed_output_fields: tuple[str, ...]
    submitted_artifact_ref: str
    code_bundle_hash: str
    sandbox_policy_ref: str
    max_trial_cost_cents: int
    max_request_cost_cents: int
    max_latency_ms: int
    credential_policy: str = "no_credentials"
    credential_ref: str = ""
    fixture_refs: tuple[str, ...] = ()
    arbitrary_code_execution_enabled: bool = False
    live_network_enabled: bool = False
    component_publicly_accepted: bool = False
    visibility_policy: str = VisibilityPolicy.DEFAULT_PRIVATE.value
    artifact_release_state: str = ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddAdapterManifest":
        return cls(
            adapter_id=str(data["adapter_id"]),
            miner_ref=str(data["miner_ref"]),
            source_name=str(data["source_name"]),
            source_kind=str(data["source_kind"]),
            declared_base_domains=tuple(str(item) for item in data.get("declared_base_domains", [])),
            output_schema_ref=str(data["output_schema_ref"]),
            allowed_output_fields=tuple(str(item) for item in data.get("allowed_output_fields", [])),
            submitted_artifact_ref=str(data["submitted_artifact_ref"]),
            code_bundle_hash=str(data["code_bundle_hash"]),
            sandbox_policy_ref=str(data["sandbox_policy_ref"]),
            max_trial_cost_cents=int(data["max_trial_cost_cents"]),
            max_request_cost_cents=int(data["max_request_cost_cents"]),
            max_latency_ms=int(data["max_latency_ms"]),
            credential_policy=str(data.get("credential_policy", "no_credentials")),
            credential_ref=str(data.get("credential_ref", "")),
            fixture_refs=tuple(str(item) for item in data.get("fixture_refs", [])),
            arbitrary_code_execution_enabled=bool(data.get("arbitrary_code_execution_enabled", False)),
            live_network_enabled=bool(data.get("live_network_enabled", False)),
            component_publicly_accepted=bool(data.get("component_publicly_accepted", False)),
            visibility_policy=str(data.get("visibility_policy", VisibilityPolicy.DEFAULT_PRIVATE.value)),
            artifact_release_state=str(
                data.get("artifact_release_state", ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["declared_base_domains"] = list(self.declared_base_domains)
        data["allowed_output_fields"] = list(self.allowed_output_fields)
        data["fixture_refs"] = list(self.fixture_refs)
        return data


@dataclass(frozen=True)
class SourceAddTrialOutputRecord:
    output_ref: str
    adapter_id: str
    icp_ref: str
    evidence_refs: tuple[str, ...]
    snapshot_refs: tuple[str, ...]
    content_hashes: tuple[str, ...]
    normalized_text_hashes: tuple[str, ...]
    metadata_refs: tuple[str, ...] = ()
    output_schema_ref: str = "schema:source-add-output:v1"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddTrialOutputRecord":
        return cls(
            output_ref=str(data["output_ref"]),
            adapter_id=str(data["adapter_id"]),
            icp_ref=str(data["icp_ref"]),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", [])),
            snapshot_refs=tuple(str(item) for item in data.get("snapshot_refs", [])),
            content_hashes=tuple(str(item) for item in data.get("content_hashes", [])),
            normalized_text_hashes=tuple(str(item) for item in data.get("normalized_text_hashes", [])),
            metadata_refs=tuple(str(item) for item in data.get("metadata_refs", [])),
            output_schema_ref=str(data.get("output_schema_ref", "schema:source-add-output:v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field in SOURCE_ADD_OUTPUT_FIELDS:
            data[field] = list(getattr(self, field))
        return data


@dataclass(frozen=True)
class SourceAddSandboxReviewRecord:
    review_id: str
    adapter_id: str
    sandbox_job: ResearchSandboxJobRecord
    trial_icp_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    measured_trial_yield: float
    status: str = SourceAddReviewStatus.RECORDED.value
    lab_only: bool = True
    adapter_code_executed: bool = False
    arbitrary_code_executed_outside_sandbox: bool = False
    component_publicly_accepted: bool = False
    bounty_payment_enabled: bool = False
    acceptance_human_gate_passed: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddSandboxReviewRecord":
        return cls(
            review_id=str(data["review_id"]),
            adapter_id=str(data["adapter_id"]),
            sandbox_job=ResearchSandboxJobRecord.from_mapping(data["sandbox_job"]),
            trial_icp_refs=tuple(str(item) for item in data.get("trial_icp_refs", [])),
            output_refs=tuple(str(item) for item in data.get("output_refs", [])),
            measured_trial_yield=float(data["measured_trial_yield"]),
            status=str(data.get("status", SourceAddReviewStatus.RECORDED.value)),
            lab_only=bool(data.get("lab_only", True)),
            adapter_code_executed=bool(data.get("adapter_code_executed", False)),
            arbitrary_code_executed_outside_sandbox=bool(data.get("arbitrary_code_executed_outside_sandbox", False)),
            component_publicly_accepted=bool(data.get("component_publicly_accepted", False)),
            bounty_payment_enabled=bool(data.get("bounty_payment_enabled", False)),
            acceptance_human_gate_passed=bool(data.get("acceptance_human_gate_passed", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sandbox_job"] = self.sandbox_job.to_dict()
        data["trial_icp_refs"] = list(self.trial_icp_refs)
        data["output_refs"] = list(self.output_refs)
        return data


@dataclass(frozen=True)
class SourceAddBountyBand:
    band_ref: str
    min_trial_yield: float
    max_trial_yield: Optional[float]
    bounty_cents: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddBountyBand":
        max_yield = data.get("max_trial_yield")
        return cls(
            band_ref=str(data["band_ref"]),
            min_trial_yield=float(data["min_trial_yield"]),
            max_trial_yield=None if max_yield is None else float(max_yield),
            bounty_cents=int(data["bounty_cents"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceAddBountyRecord:
    bounty_ref: str
    adapter_id: str
    review_id: str
    miner_ref: str
    measured_trial_yield: float
    band_ref: str
    bounty_cents: int
    state: str = SourceAddBountyState.CALCULATED_NOT_PAYABLE.value
    payment_enabled: bool = False
    settlement_ref: str = ""
    paid_at: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddBountyRecord":
        return cls(
            bounty_ref=str(data["bounty_ref"]),
            adapter_id=str(data["adapter_id"]),
            review_id=str(data["review_id"]),
            miner_ref=str(data["miner_ref"]),
            measured_trial_yield=float(data["measured_trial_yield"]),
            band_ref=str(data["band_ref"]),
            bounty_cents=int(data["bounty_cents"]),
            state=str(data.get("state", SourceAddBountyState.CALCULATED_NOT_PAYABLE.value)),
            payment_enabled=bool(data.get("payment_enabled", False)),
            settlement_ref=str(data.get("settlement_ref", "")),
            paid_at=str(data.get("paid_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceAddIPAssignmentAcknowledgement:
    assignment_ref: str
    adapter_id: str
    miner_ref: str
    assigned_to_ref: str
    rights_scope: str
    state: str = SourceAddIPState.ACKNOWLEDGED_PENDING_ACCEPTANCE.value
    effective_on_acceptance_only: bool = True
    public_component_accepted: bool = False
    signature_ref: str = "signature:pending"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceAddIPAssignmentAcknowledgement":
        return cls(
            assignment_ref=str(data["assignment_ref"]),
            adapter_id=str(data["adapter_id"]),
            miner_ref=str(data["miner_ref"]),
            assigned_to_ref=str(data["assigned_to_ref"]),
            rights_scope=str(data["rights_scope"]),
            state=str(data.get("state", SourceAddIPState.ACKNOWLEDGED_PENDING_ACCEPTANCE.value)),
            effective_on_acceptance_only=bool(data.get("effective_on_acceptance_only", True)),
            public_component_accepted=bool(data.get("public_component_accepted", False)),
            signature_ref=str(data.get("signature_ref", "signature:pending")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_source_add_adapter_manifest(
    manifest: SourceAddAdapterManifest | Mapping[str, Any],
) -> list[str]:
    raw = manifest if isinstance(manifest, Mapping) else manifest.to_dict()
    if not isinstance(manifest, SourceAddAdapterManifest):
        manifest = SourceAddAdapterManifest.from_mapping(manifest)
    errors: list[str] = []
    if manifest.source_kind not in {kind.value for kind in SourceAddSourceKind}:
        errors.append(f"unknown source_kind: {manifest.source_kind}")
    if not manifest.declared_base_domains:
        errors.append("declared_base_domains must not be empty")
    if not manifest.output_schema_ref:
        errors.append("output_schema_ref is required")
    if not manifest.code_bundle_hash.startswith("sha256:"):
        errors.append("code_bundle_hash must be sha256-prefixed")
    if not manifest.submitted_artifact_ref:
        errors.append("submitted_artifact_ref is required")
    if not manifest.fixture_refs:
        errors.append("fixture_refs must not be empty")
    if manifest.max_trial_cost_cents <= 0 or manifest.max_request_cost_cents <= 0:
        errors.append("cost caps must be positive")
    if manifest.max_request_cost_cents > manifest.max_trial_cost_cents:
        errors.append("max_request_cost_cents cannot exceed max_trial_cost_cents")
    if manifest.max_trial_cost_cents > 5000:
        errors.append("max_trial_cost_cents exceeds P1.5 launch cap")
    if manifest.max_latency_ms <= 0:
        errors.append("max_latency_ms must be positive")
    disallowed = sorted(set(manifest.allowed_output_fields) - set(SOURCE_ADD_OUTPUT_FIELDS))
    if disallowed:
        errors.append("allowed_output_fields contains disallowed fields: " + ", ".join(disallowed))
    if "evidence_refs" not in manifest.allowed_output_fields:
        errors.append("allowed_output_fields must include evidence_refs")
    if any(field in manifest.allowed_output_fields for field in RAW_OUTPUT_FIELDS):
        errors.append("allowed_output_fields must not include raw scraped content fields")
    if manifest.credential_policy not in {"no_credentials", "credential_ref_only"}:
        errors.append("credential_policy must be no_credentials or credential_ref_only")
    if manifest.credential_policy == "no_credentials" and manifest.credential_ref:
        errors.append("credential_ref must be empty when credential_policy=no_credentials")
    if _contains_raw_credential_key(raw):
        errors.append("manifest must not contain raw credential fields")
    if manifest.arbitrary_code_execution_enabled:
        errors.append("arbitrary_code_execution_enabled must remain false")
    if manifest.live_network_enabled:
        errors.append("live_network_enabled must remain false")
    if manifest.component_publicly_accepted:
        errors.append("component_publicly_accepted must remain false in P1.5")
    errors.extend(
        validate_release_policy(
            ReleasePolicyRecord(
                artifact_ref=manifest.adapter_id,
                artifact_type="source_add_adapter_manifest",
                visibility_policy=manifest.visibility_policy,
                artifact_release_state=manifest.artifact_release_state,
                reason="SOURCE_ADD adapter manifests remain private until acceptance gate",
            )
        )
    )
    if manifest.artifact_release_state != ArtifactReleaseState.PRIVATE_LIVE_CHAMPION.value:
        errors.append("SOURCE_ADD adapter manifest must remain private in P1.5")
    return errors


def validate_source_add_trial_output(output: SourceAddTrialOutputRecord | Mapping[str, Any]) -> list[str]:
    raw = output if isinstance(output, Mapping) else output.to_dict()
    if not isinstance(output, SourceAddTrialOutputRecord):
        output = SourceAddTrialOutputRecord.from_mapping(output)
    errors: list[str] = []
    if _contains_any_key(raw, RAW_OUTPUT_FIELDS):
        errors.append("adapter trial outputs must not contain raw scraped content fields")
    if not output.evidence_refs:
        errors.append("evidence_refs must not be empty")
    if not output.snapshot_refs:
        errors.append("snapshot_refs must not be empty")
    if not output.content_hashes:
        errors.append("content_hashes must not be empty")
    if not output.normalized_text_hashes:
        errors.append("normalized_text_hashes must not be empty")
    for field in ("content_hashes", "normalized_text_hashes"):
        bad = [value for value in getattr(output, field) if not value.startswith("sha256:")]
        if bad:
            errors.append(f"{field} must be sha256-prefixed")
    for evidence_ref in output.evidence_refs:
        if not (evidence_ref.startswith("evidence:") or evidence_ref.startswith("sha256:")):
            errors.append("evidence_refs must be evidence: or sha256: references")
    return errors


def validate_source_add_sandbox_review(
    review: SourceAddSandboxReviewRecord | Mapping[str, Any],
    *,
    guards: Optional[LoopWorkflowGuards | Mapping[str, Any]] = None,
) -> list[str]:
    if not isinstance(review, SourceAddSandboxReviewRecord):
        review = SourceAddSandboxReviewRecord.from_mapping(review)
    errors: list[str] = []
    try:
        assert_loop_workflows_disabled(guards or default_loop_workflow_guards())
    except ValueError as exc:
        errors.append(str(exc))
    errors.extend(validate_sandbox_job_record(review.sandbox_job))
    if review.status not in {status.value for status in SourceAddReviewStatus}:
        errors.append(f"unknown source-add review status: {review.status}")
    if not review.lab_only:
        errors.append("SOURCE_ADD sandbox review must remain lab_only")
    if not review.trial_icp_refs:
        errors.append("trial_icp_refs must not be empty")
    if not review.output_refs:
        errors.append("output_refs must not be empty")
    if review.measured_trial_yield < 0:
        errors.append("measured_trial_yield must be non-negative")
    if review.adapter_code_executed:
        errors.append("P1.5 must not execute adapter code")
    if review.arbitrary_code_executed_outside_sandbox:
        errors.append("arbitrary code must not execute outside sandbox")
    if review.component_publicly_accepted:
        errors.append("P1.5 must not accept public components")
    if review.bounty_payment_enabled:
        errors.append("P1.5 must not enable bounty payments")
    if review.acceptance_human_gate_passed:
        errors.append("acceptance_human_gate_passed must remain false in local P1.5")
    if review.sandbox_job.status not in {
        SandboxJobStatus.QUEUED.value,
        SandboxJobStatus.GUARDRAIL_BREACH.value,
        SandboxJobStatus.CRASH.value,
        SandboxJobStatus.TIMEOUT.value,
    }:
        errors.append("P1.5 sandbox review records must not claim completed adapter execution")
    return errors


def select_source_add_bounty_band(
    measured_trial_yield: float,
    bands: Sequence[SourceAddBountyBand | Mapping[str, Any]],
) -> SourceAddBountyBand:
    normalized = [band if isinstance(band, SourceAddBountyBand) else SourceAddBountyBand.from_mapping(band) for band in bands]
    matches = [
        band
        for band in normalized
        if measured_trial_yield >= band.min_trial_yield
        and (band.max_trial_yield is None or measured_trial_yield < band.max_trial_yield)
    ]
    if not matches:
        raise ValueError("no bounty band matches measured trial yield")
    return sorted(matches, key=lambda band: (-band.min_trial_yield, band.band_ref))[0]


def build_source_add_bounty_record(
    *,
    manifest: SourceAddAdapterManifest | Mapping[str, Any],
    review: SourceAddSandboxReviewRecord | Mapping[str, Any],
    bands: Sequence[SourceAddBountyBand | Mapping[str, Any]],
) -> SourceAddBountyRecord:
    if not isinstance(manifest, SourceAddAdapterManifest):
        manifest = SourceAddAdapterManifest.from_mapping(manifest)
    if not isinstance(review, SourceAddSandboxReviewRecord):
        review = SourceAddSandboxReviewRecord.from_mapping(review)
    manifest_errors = validate_source_add_adapter_manifest(manifest)
    review_errors = validate_source_add_sandbox_review(review)
    if manifest_errors or review_errors:
        raise ValueError("; ".join(manifest_errors + review_errors))
    band = select_source_add_bounty_band(review.measured_trial_yield, bands)
    payload = {
        "adapter_id": manifest.adapter_id,
        "review_id": review.review_id,
        "miner_ref": manifest.miner_ref,
        "measured_trial_yield": review.measured_trial_yield,
        "band_ref": band.band_ref,
        "bounty_cents": band.bounty_cents,
    }
    record = SourceAddBountyRecord(
        bounty_ref="source_add_bounty:" + sha256_json(payload).split(":", 1)[1][:16],
        **payload,
    )
    errors = validate_source_add_bounty_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def validate_source_add_bounty_record(record: SourceAddBountyRecord | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, SourceAddBountyRecord):
        record = SourceAddBountyRecord.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in SourceAddBountyState}:
        errors.append(f"unknown SOURCE_ADD bounty state: {record.state}")
    if record.bounty_cents < 0:
        errors.append("bounty_cents must be non-negative")
    if record.payment_enabled:
        errors.append("payment_enabled must remain false")
    if record.settlement_ref:
        errors.append("settlement_ref must remain empty")
    if record.paid_at:
        errors.append("paid_at must remain empty")
    return errors


def validate_source_add_ip_assignment(
    record: SourceAddIPAssignmentAcknowledgement | Mapping[str, Any],
) -> list[str]:
    if not isinstance(record, SourceAddIPAssignmentAcknowledgement):
        record = SourceAddIPAssignmentAcknowledgement.from_mapping(record)
    errors: list[str] = []
    if record.state not in {state.value for state in SourceAddIPState}:
        errors.append(f"unknown SOURCE_ADD IP state: {record.state}")
    if not record.assigned_to_ref:
        errors.append("assigned_to_ref is required")
    if "source_add_adapter" not in record.rights_scope:
        errors.append("rights_scope must cover source_add_adapter")
    if not record.effective_on_acceptance_only:
        errors.append("IP assignment must be effective on acceptance only")
    if record.public_component_accepted:
        errors.append("P1.5 must not mark public component accepted")
    if record.signature_ref != "signature:pending":
        errors.append("P1.5 IP acknowledgement must remain signature:pending")
    return errors


def verify_research_lab_source_add(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    fabric_summary = verify_research_lab_fabric()
    fixture = _load_fixture(Path(fixture_path))

    manifest = SourceAddAdapterManifest.from_mapping(fixture["adapter_manifest"])
    _assert(not validate_source_add_adapter_manifest(manifest), "valid SOURCE_ADD manifest passes")
    for record in fixture["invalid_manifests"]:
        errors = validate_source_add_adapter_manifest(record)
        _assert(errors, f"invalid manifest fails: {record['adapter_id']}")
        _assert_expected_error(errors, record)

    outputs = [SourceAddTrialOutputRecord.from_mapping(item) for item in fixture["trial_outputs"]]
    for output in outputs:
        _assert(not validate_source_add_trial_output(output), f"valid trial output passes: {output.output_ref}")
    for output in fixture["invalid_trial_outputs"]:
        errors = validate_source_add_trial_output(output)
        _assert(errors, f"invalid trial output fails: {output['output_ref']}")
        _assert_expected_error(errors, output)

    review = SourceAddSandboxReviewRecord.from_mapping(fixture["sandbox_review"])
    _assert(not validate_source_add_sandbox_review(review), "sandbox review validates")
    _assert(review.sandbox_job.resource_caps.network_disabled, "review sandbox caps keep network disabled")
    _assert(not review.adapter_code_executed, "review does not execute adapter code")
    for record in fixture["invalid_sandbox_reviews"]:
        errors = validate_source_add_sandbox_review(record)
        _assert(errors, f"invalid sandbox review fails: {record['review_id']}")
        _assert_expected_error(errors, record)
    unsafe_guard_errors = validate_source_add_sandbox_review(review, guards=fixture["unsafe_workflow_guards"])
    _assert(unsafe_guard_errors, "unsafe workflow guards fail sandbox review")

    bands = [SourceAddBountyBand.from_mapping(item) for item in fixture["bounty_bands"]]
    bounty = build_source_add_bounty_record(manifest=manifest, review=review, bands=bands)
    _assert(bounty.band_ref == fixture["expected_bounty_band_ref"], "bounty band selected from measured yield")
    _assert(not validate_source_add_bounty_record(bounty), "bounty record validates")
    for record in fixture["invalid_bounty_records"]:
        errors = validate_source_add_bounty_record(record)
        _assert(errors, f"invalid bounty record fails: {record['bounty_ref']}")
        _assert_expected_error(errors, record)

    ip_ack = SourceAddIPAssignmentAcknowledgement.from_mapping(fixture["ip_assignment"])
    _assert(not validate_source_add_ip_assignment(ip_ack), "IP assignment acknowledgement validates")
    for record in fixture["invalid_ip_assignments"]:
        errors = validate_source_add_ip_assignment(record)
        _assert(errors, f"invalid IP assignment fails: {record['assignment_ref']}")
        _assert_expected_error(errors, record)

    return {
        "fabric_invalid_records": fabric_summary["invalid_records"],
        "adapter_id": manifest.adapter_id,
        "trial_outputs": len(outputs),
        "measured_trial_yield": review.measured_trial_yield,
        "bounty_ref": bounty.bounty_ref,
        "bounty_band_ref": bounty.band_ref,
        "bounty_cents": bounty.bounty_cents,
    }


def _contains_any_key(value: Any, keys: Sequence[str]) -> bool:
    key_set = {key.lower() for key in keys}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in key_set:
                return True
            if _contains_any_key(nested, keys):
                return True
    elif isinstance(value, list):
        return any(_contains_any_key(item, keys) for item in value)
    return False


def _contains_raw_credential_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).lower()
            if normalized_key in RAW_CREDENTIAL_KEYS:
                return True
            if normalized_key.endswith("_ref"):
                continue
            if _contains_raw_credential_key(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_raw_credential_key(item) for item in value)
    return False


def _assert_expected_error(errors: Sequence[str], record: Mapping[str, Any]) -> None:
    expected = record.get("expected_error_contains")
    if expected:
        _assert(any(str(expected) in error for error in errors), f"expected error {expected!r}")


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
