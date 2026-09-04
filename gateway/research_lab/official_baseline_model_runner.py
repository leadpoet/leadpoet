"""Exact immutable-model execution boundary for the official baseline.

This module deliberately contains no provider implementation.  A v3 release
can enter official scoring only when the process is given a protected,
append-only action authority, an artifact-owned benchmark projector, and an
append-only terminal-record authority.  Missing dependencies fail before the
first model or provider call; the legacy runner is selected only by its exact
signed compatibility documents.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from gateway.qualification.models import CompanyOutput
from gateway.research_lab.common_model_experiment import (
    CommonModelExperimentRecoveryError,
    ExactModelActionDispatcher,
    ExactModelExperimentCoordinator,
    ExactModelUnitResult,
    ModelTransitionRepository,
)
from research_lab.canonical import sha256_json
from research_lab.eval import DockerPrivateModelSpec, PrivateModelArtifactManifest
from research_lab.model_runner_protocol import ExactModelRunnerRegistration


OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_execution.v1"
)
OFFICIAL_BASELINE_RUN_SCHEMA_VERSION = "leadpoet.research_lab.official_baseline_run.v1"
OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION = "model-runner-benchmark-projection:v1"
OFFICIAL_BASELINE_TERMINAL_RECORD_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_terminal_record.v1"
)
OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_unit_completion.v1"
)
OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_unit_closure.v1"
)
OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_provider_frontier.v1"
)
OFFICIAL_BASELINE_CHECKPOINT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_checkpoint.v1"
)
OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_authority_preflight.v1"
)
OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_run_registration.v1"
)
OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_authorization.v1"
)
OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_terminal_known.v1"
)
OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_terminal_uncertain.v1"
)
OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_replay_identity.v1"
)
OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_run_registration_result.v1"
)
OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_reservation_result.v1"
)
OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_terminal_result.v1"
)
OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_action_replay_result.v1"
)
EXACT_MODEL_RUNNER_FAMILY = "exact_model_runner:v3"
LEGACY_MODEL_RUNNER_FAMILY = "attested_private_model:v2"
LAB_RAW_ICP_SOURCE_SCHEMA = "leadpoet-research-lab-benchmark-icp:v1"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_TERMINAL_REF_RE = re.compile(r"baseline_terminal:[0-9a-f]{64}")
_PROVIDER_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class OfficialBaselineModelError(RuntimeError):
    """The official baseline cannot cross the exact model boundary safely."""


class OfficialBaselineReleaseSelectionError(OfficialBaselineModelError):
    """The signed release does not select one supported runner family."""


class OfficialBaselineAuthorityUnavailable(OfficialBaselineModelError):
    """A v3 release has no protected baseline execution authority."""


class OfficialBaselineProviderPendingError(CommonModelExperimentRecoveryError):
    """An authorized asynchronous provider run is still reconciling."""


def _require_sha256(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise OfficialBaselineModelError(f"{label} is invalid")
    return normalized


def _artifact_extension(
    artifact: PrivateModelArtifactManifest,
    name: str,
) -> Any:
    extensions = artifact.signed_extensions
    return extensions.get(name) if isinstance(extensions, Mapping) else None


@dataclass(frozen=True)
class OfficialBaselineReleaseSelection:
    """One runner family selected only from the signed artifact manifest."""

    runner_family: str
    selection_document: Mapping[str, Any]

    @property
    def selection_sha256(self) -> str:
        return sha256_json(dict(self.selection_document))

    @property
    def is_exact(self) -> bool:
        return self.runner_family == EXACT_MODEL_RUNNER_FAMILY


def select_official_baseline_release(
    artifact: PrivateModelArtifactManifest,
) -> OfficialBaselineReleaseSelection:
    """Select exact v3 or an exact legacy drain; never runtime-fallback.

    New raw releases must carry a closed ``official_baseline_execution``
    extension plus their signed ``model_release_identity``.  An older release
    is legacy only when it has no model release identity and carries both exact
    signed qualify-compatibility documents.  Unknown/mixed releases fail.
    """

    if not isinstance(artifact, PrivateModelArtifactManifest):
        raise OfficialBaselineReleaseSelectionError(
            "official baseline artifact is invalid"
        )
    execution = _artifact_extension(artifact, "official_baseline_execution")
    release_identity = _artifact_extension(artifact, "model_release_identity")
    if execution is not None:
        if not isinstance(execution, Mapping) or set(execution) != {
            "schema_version",
            "runner_family",
            "execution_mode",
            "release_identity_sha256",
            "protocol_generation_sha256",
            "benchmark_projection_sha256",
            "protected_action_authority_sha256",
        }:
            raise OfficialBaselineReleaseSelectionError(
                "official baseline exact release selection is not closed"
            )
        if (
            execution.get("schema_version")
            != OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION
            or execution.get("runner_family") != EXACT_MODEL_RUNNER_FAMILY
            or execution.get("execution_mode") != "measured_lab"
            or not isinstance(release_identity, Mapping)
        ):
            raise OfficialBaselineReleaseSelectionError(
                "official baseline exact release selection is invalid"
            )
        for field in (
            "release_identity_sha256",
            "protocol_generation_sha256",
            "benchmark_projection_sha256",
            "protected_action_authority_sha256",
        ):
            _require_sha256(execution.get(field), field)
        if sha256_json(dict(release_identity)) != execution["release_identity_sha256"]:
            raise OfficialBaselineReleaseSelectionError(
                "official baseline release identity differs from selection"
            )
        return OfficialBaselineReleaseSelection(
            runner_family=EXACT_MODEL_RUNNER_FAMILY,
            selection_document=dict(execution),
        )

    if release_identity is not None:
        raise OfficialBaselineReleaseSelectionError(
            "raw model release is missing official baseline selection"
        )
    compatibility = artifact.compatibility_contract
    parity = artifact.consumer_parity_fixtures
    if (
        not isinstance(compatibility, Mapping)
        or set(compatibility) != {"contract_id", "path", "sha256"}
        or not isinstance(parity, Mapping)
        or set(parity) != {"path", "sha256"}
    ):
        raise OfficialBaselineReleaseSelectionError(
            "official baseline release does not select a legacy drain"
        )
    _require_sha256(compatibility.get("sha256"), "legacy contract hash")
    _require_sha256(parity.get("sha256"), "legacy parity hash")
    if (
        not str(compatibility.get("contract_id") or "").strip()
        or not str(compatibility.get("path") or "").strip()
        or not str(parity.get("path") or "").strip()
    ):
        raise OfficialBaselineReleaseSelectionError(
            "official baseline legacy release identity is invalid"
        )
    document = {
        "schema_version": OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION,
        "runner_family": LEGACY_MODEL_RUNNER_FAMILY,
        "legacy_release_sha256": sha256_json(
            {
                "model_artifact_hash": artifact.model_artifact_hash,
                "manifest_hash": artifact.manifest_hash,
                "git_commit_sha": artifact.git_commit_sha,
                "compatibility_contract": dict(compatibility),
                "consumer_parity_fixtures": dict(parity),
            }
        ),
    }
    return OfficialBaselineReleaseSelection(
        runner_family=LEGACY_MODEL_RUNNER_FAMILY,
        selection_document=document,
    )


@dataclass(frozen=True)
class ArtifactBenchmarkProjection:
    """Outputs and the immutable artifact's projection receipt."""

    outputs: tuple[Mapping[str, Any], ...]
    projection_receipt: Mapping[str, Any]


class ImmutableArtifactBenchmarkProjector(Protocol):
    """Artifact-owned terminal-result -> CompanyOutput projection only."""

    @property
    def artifact_key(self) -> str: ...

    @property
    def protocol_generation_sha256(self) -> str: ...

    @property
    def projection_identity_sha256(self) -> str: ...

    def project_company_outputs(
        self,
        *,
        start_request: Mapping[str, Any],
        terminal_result: Mapping[str, Any],
    ) -> ArtifactBenchmarkProjection: ...


class ArtifactProtocolBenchmarkProjector:
    """Thin projection adapter bound to one immutable runner generation."""

    def __init__(self, registration: ExactModelRunnerRegistration) -> None:
        if not isinstance(registration, ExactModelRunnerRegistration):
            raise OfficialBaselineModelError(
                "artifact benchmark projector registration is invalid"
            )
        generation = registration.protocol_generation
        if not generation.supports_official_baseline:
            raise OfficialBaselineModelError(
                "artifact benchmark projection bundle is unavailable"
            )
        self._registration = registration
        self._projection_identity_sha256 = (
            generation.official_contract_sha256(
                "benchmark_projection_contract"
            )
        )

    @property
    def artifact_key(self) -> str:
        return self._registration.key

    @property
    def protocol_generation_sha256(self) -> str:
        return (
            self._registration.protocol_generation.protocol_generation_sha256
        )

    @property
    def projection_identity_sha256(self) -> str:
        return self._projection_identity_sha256

    def project_company_outputs(
        self,
        *,
        start_request: Mapping[str, Any],
        terminal_result: Mapping[str, Any],
    ) -> ArtifactBenchmarkProjection:
        value = self._registration.protocol.project_runner_result_for_benchmark(
            terminal_result,
            start_request=start_request,
        )
        if not isinstance(value, Mapping):
            raise OfficialBaselineModelError(
                "artifact benchmark projection is invalid"
            )
        companies = value.get("companies")
        if not isinstance(companies, list) or any(
            not isinstance(item, Mapping) for item in companies
        ):
            raise OfficialBaselineModelError(
                "artifact benchmark projection companies are invalid"
            )
        return ArtifactBenchmarkProjection(
            outputs=tuple(dict(item) for item in companies),
            projection_receipt=dict(value),
        )


class ProtectedOfficialBaselineAuthority(Protocol):
    """Protected append-only provider/verifier authority.

    ``dispatcher_for_unit`` must implement dispatch-or-replay: it computes the
    deterministic attempt key and checks durable terminal/uncertain state
    before any authorization or network call.  A known terminal attempt is
    replayed byte-exactly; an uncertain consumed call is never redispatched.
    """

    @property
    def authority_identity_sha256(self) -> str: ...

    def preflight_run(
        self,
        *,
        run_identity: Mapping[str, Any],
        registration: ExactModelRunnerRegistration,
    ) -> Mapping[str, Any]: ...

    def dispatcher_for_unit(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
    ) -> ExactModelActionDispatcher: ...

    def transition_repository_for_unit(
        self,
        *,
        run_identity: Mapping[str, Any],
        unit_ref: str,
    ) -> ModelTransitionRepository: ...

    def close_unit(self, *, completion: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def load_frontier(self, *, run_sha256: str, unit_ref: str) -> Mapping[str, Any]: ...


class OfficialBaselineAttemptStore(Protocol):
    """Frozen append-only SQL/RPC adapter surface for protected actions.

    Responses use, respectively,
    ``OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION``,
    ``OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION``,
    ``OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION``,
    ``OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION``,
    ``OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION``, and
    ``OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION``.  Documents are
    closed versioned mappings.  Implementations persist only
    identities, protected-job/receipt refs, hashes, and accounting custody;
    provider responses and projected companies remain in the protected job
    and terminal authorities.
    """

    def register_run(self, *, registration: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def reserve_action(
        self, *, authorization: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def record_terminal_known(
        self, *, terminal: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def record_terminal_uncertain(
        self, *, uncertainty: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def load_replay(self, *, identity: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close_unit(self, *, closure: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def load_frontier(self, *, run_sha256: str, unit_ref: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class OfficialBaselineDependencyContext:
    """Already-frozen official-run inputs passed to dependency construction.

    A factory must consume these exact objects.  It must never resolve a new
    active artifact or independently select a release while a benchmark is in
    progress.
    """

    artifact: PrivateModelArtifactManifest
    artifact_pointer_uri: str
    artifact_pointer_manifest_hash: str
    selection: OfficialBaselineReleaseSelection
    spec: DockerPrivateModelSpec
    benchmark_date: str
    rolling_window_hash: str
    benchmark_attempt: int
    evaluation_epoch: int
    parent_graphs: tuple[Mapping[str, Any], ...]
    worker_index: int
    worker_ref: str
    evidence_proxy_url: str
    evidence_proxy_capability_sha256: str
    evidence_proxy_ready_provider_ids: tuple[str, ...]

    @property
    def source_branch(self) -> str:
        """Return the branch frozen by the verified mutable pointer."""

        try:
            pointer = urlsplit(str(self.artifact_pointer_uri or ""))
            archive = urlsplit(str(self.artifact.manifest_uri or ""))
            pointer_port = pointer.port
            archive_port = archive.port
        except (AttributeError, ValueError) as exc:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline artifact pointer identity is invalid"
            ) from exc
        pointer_parts = tuple(
            part for part in pointer.path.split("/") if part
        )
        archive_parts = tuple(
            part for part in archive.path.split("/") if part
        )
        branch = pointer_parts[-2] if len(pointer_parts) >= 3 else ""
        commit = str(self.artifact.git_commit_sha or "")
        if (
            pointer.scheme != "s3"
            or archive.scheme != "s3"
            or not pointer.netloc
            or not archive.netloc
            or pointer.username is not None
            or pointer.password is not None
            or archive.username is not None
            or archive.password is not None
            or pointer_port is not None
            or archive_port is not None
            or pointer.query
            or pointer.fragment
            or archive.query
            or archive.fragment
            or pointer.path != "/" + "/".join(pointer_parts)
            or archive.path != "/" + "/".join(archive_parts)
            or len(pointer_parts) < 4
            or pointer_parts[-3] != "branches"
            or branch not in {"main", "leadpoet-lab"}
            or pointer_parts[-1] != "current.json"
            or pointer_parts.count("branches") != 1
            or not archive_parts
            or "branches" in archive_parts
            or re.fullmatch(r"[0-9a-f]{40}", commit) is None
            or archive_parts[-1] != f"{commit}.json"
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline artifact pointer identity is invalid"
            )
        return branch

    def validate(self) -> None:
        try:
            proxy = urlsplit(str(self.evidence_proxy_url or ""))
            proxy_port = proxy.port
        except ValueError as exc:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline evidence proxy context is invalid"
            ) from exc
        ready_provider_ids = (
            tuple(self.evidence_proxy_ready_provider_ids)
            if isinstance(self.evidence_proxy_ready_provider_ids, tuple)
            else ()
        )
        if not isinstance(self.artifact, PrivateModelArtifactManifest):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline frozen artifact is invalid"
            )
        source_branch = self.source_branch
        expected_selection = select_official_baseline_release(self.artifact)
        if (
            not self.selection.is_exact
            or self.selection != expected_selection
            or self.spec.image_digest != self.artifact.image_digest
            or _SHA256_RE.fullmatch(
                str(self.artifact_pointer_manifest_hash or "")
            )
            is None
            or self.artifact_pointer_manifest_hash
            != self.artifact.manifest_hash
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.benchmark_date)
            or _SHA256_RE.fullmatch(str(self.rolling_window_hash or "")) is None
            or type(self.benchmark_attempt) is not int
            or self.benchmark_attempt < 0
            or type(self.evaluation_epoch) is not int
            or self.evaluation_epoch < 0
            or type(self.worker_index) is not int
            or self.worker_index < 0
            or not str(self.worker_ref or "").strip()
            or any(not isinstance(item, Mapping) for item in self.parent_graphs)
            or proxy.scheme != "http"
            or proxy.hostname not in {"127.0.0.1", "::1"}
            or proxy_port is None
            or proxy_port < 1
            or proxy_port > 65535
            or proxy.username is not None
            or proxy.password is not None
            or proxy.path not in {"", "/"}
            or bool(proxy.query)
            or bool(proxy.fragment)
            or _SHA256_RE.fullmatch(
                str(self.evidence_proxy_capability_sha256 or "")
            )
            is None
            or ready_provider_ids != tuple(sorted(set(ready_provider_ids)))
            or not isinstance(self.evidence_proxy_ready_provider_ids, tuple)
            or any(
                _PROVIDER_ID_RE.fullmatch(str(item or "")) is None
                for item in ready_provider_ids
            )
            or source_branch not in {"main", "leadpoet-lab"}
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline frozen dependency context differs"
            )


class OfficialBaselineTerminalAuthority(Protocol):
    """Append-only full terminal storage; checkpoints retain hashes/refs only."""

    def persist_terminal_record(
        self,
        *,
        record_identity_sha256: str,
        record: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def load_terminal_record(
        self, *, terminal_record_ref: str
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class OfficialBaselineExactDependencies:
    registration: ExactModelRunnerRegistration
    projector: ImmutableArtifactBenchmarkProjector
    protected_authority: ProtectedOfficialBaselineAuthority
    terminal_authority: OfficialBaselineTerminalAuthority


@dataclass(frozen=True)
class OfficialBaselineModelOutput:
    company_outputs: tuple[Mapping[str, Any], ...]
    model_receipt: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    replayed_transition_count: int


def _validate_company_outputs(
    values: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(values, (list, tuple)):
        raise OfficialBaselineModelError(
            "artifact benchmark projection outputs are invalid"
        )
    normalized: list[Mapping[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise OfficialBaselineModelError(
                "artifact benchmark projection output is invalid"
            )
        raw = dict(value)
        try:
            parsed = CompanyOutput.model_validate(raw)
            roundtrip = parsed.model_dump(mode="json", exclude_unset=True)
        except Exception as exc:  # noqa: BLE001 - artifact output is fail closed
            raise OfficialBaselineModelError(
                "artifact benchmark projection is not CompanyOutput"
            ) from exc
        if roundtrip != raw:
            raise OfficialBaselineModelError(
                "artifact benchmark projection changed during CompanyOutput validation"
            )
        normalized.append(raw)
    return tuple(normalized)


def _validate_projection(
    projection: ArtifactBenchmarkProjection,
    *,
    start_request: Mapping[str, Any],
    release_identity: Mapping[str, Any],
    model_receipt: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    if not isinstance(projection, ArtifactBenchmarkProjection):
        raise OfficialBaselineModelError(
            "artifact benchmark projector returned an invalid result"
        )
    receipt = projection.projection_receipt
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version",
        "start_request_sha256",
        "release_identity_sha256",
        "model_receipt_sha256",
        "companies",
        "companies_sha256",
        "projection_sha256",
    }:
        raise OfficialBaselineModelError(
            "artifact benchmark projection receipt is not closed"
        )
    companies = receipt.get("companies")
    outputs = _validate_company_outputs(companies)
    if tuple(projection.outputs) != outputs:
        raise OfficialBaselineModelError(
            "artifact benchmark projection outputs differ from its receipt"
        )

    def model_wire_sha256(value: Any) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    body = {
        "schema_version": OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION,
        "start_request_sha256": model_wire_sha256(dict(start_request)),
        "release_identity_sha256": model_wire_sha256(dict(release_identity)),
        "model_receipt_sha256": model_wire_sha256(dict(model_receipt)),
        "companies": [dict(value) for value in outputs],
        "companies_sha256": model_wire_sha256([dict(value) for value in outputs]),
    }
    if dict(receipt) != {
        **body,
        "projection_sha256": model_wire_sha256(body),
    }:
        raise OfficialBaselineModelError(
            "artifact benchmark projection receipt differs"
        )
    return outputs, dict(receipt)


def validate_official_baseline_provider_closure(
    value: Any,
    *,
    expected_completion: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "run_sha256",
        "unit_ref",
        "protocol_generation_sha256",
        "raw_input_sha256",
        "start_request_sha256",
        "terminal_result_sha256",
        "model_receipt_sha256",
        "projection_sha256",
        "ordered_attempt_keys",
        "ordered_attempt_sha256s",
        "provider_frontier_sha256",
        "closure_ref",
        "closure_sha256",
        "idempotent",
    }:
        raise OfficialBaselineModelError(
            "official baseline provider closure is not closed"
        )
    normalized = dict(value)
    expected_identity = {
        key: item
        for key, item in expected_completion.items()
        if key != "schema_version"
    }
    if (
        normalized.get("schema_version")
        != OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION
        or any(normalized.get(key) != item for key, item in expected_identity.items())
        or type(normalized.get("idempotent")) is not bool
    ):
        raise OfficialBaselineModelError(
            "official baseline provider closure identity differs"
        )
    keys = normalized.get("ordered_attempt_keys")
    hashes = normalized.get("ordered_attempt_sha256s")
    if (
        not isinstance(keys, list)
        or not isinstance(hashes, list)
        or len(keys) != len(hashes)
        or len(set(keys)) != len(keys)
        or any(not _SHA256_RE.fullmatch(str(item or "")) for item in keys)
        or any(not _SHA256_RE.fullmatch(str(item or "")) for item in hashes)
    ):
        raise OfficialBaselineModelError(
            "official baseline provider closure attempts are invalid"
        )
    frontier = {
        "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
        "ordered_attempt_keys": list(keys),
        "ordered_attempt_sha256s": list(hashes),
    }
    if normalized.get("provider_frontier_sha256") != sha256_json(frontier):
        raise OfficialBaselineModelError(
            "official baseline provider frontier hash differs"
        )
    body = {
        "schema_version": OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION,
        **expected_identity,
        "ordered_attempt_keys": list(keys),
        "ordered_attempt_sha256s": list(hashes),
        "provider_frontier_sha256": normalized["provider_frontier_sha256"],
    }
    closure_sha = sha256_json(body)
    if normalized.get("closure_sha256") != closure_sha or normalized.get(
        "closure_ref"
    ) != "baseline_closure:" + closure_sha.removeprefix("sha256:"):
        raise OfficialBaselineModelError(
            "official baseline provider closure hash differs"
        )
    return {
        **body,
        "closure_ref": normalized["closure_ref"],
        "closure_sha256": closure_sha,
    }


def validate_official_baseline_checkpoint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "release_selection_sha256",
        "artifact_key_sha256",
        "protocol_generation_sha256",
        "unit_ref",
        "raw_input_sha256",
        "start_request_sha256",
        "terminal_result_sha256",
        "model_receipt_sha256",
        "projection_identity_sha256",
        "projection_sha256",
        "provider_frontier_sha256",
        "provider_closure_ref",
        "provider_closure_sha256",
        "terminal_record_ref",
        "terminal_record_sha256",
        "checkpoint_sha256",
    }:
        raise OfficialBaselineModelError("official baseline checkpoint is not closed")
    normalized = dict(value)
    if normalized.get("schema_version") != OFFICIAL_BASELINE_CHECKPOINT_SCHEMA_VERSION:
        raise OfficialBaselineModelError(
            "official baseline checkpoint schema is invalid"
        )
    for field in (
        "release_selection_sha256",
        "artifact_key_sha256",
        "protocol_generation_sha256",
        "raw_input_sha256",
        "start_request_sha256",
        "terminal_result_sha256",
        "model_receipt_sha256",
        "projection_identity_sha256",
        "projection_sha256",
        "provider_frontier_sha256",
        "provider_closure_sha256",
        "terminal_record_sha256",
        "checkpoint_sha256",
    ):
        _require_sha256(normalized.get(field), field)
    if not re.fullmatch(
        r"baseline_icp:[0-9a-f]{64}", str(normalized.get("unit_ref") or "")
    ):
        raise OfficialBaselineModelError("official baseline checkpoint unit is invalid")
    if not _TERMINAL_REF_RE.fullmatch(str(normalized.get("terminal_record_ref") or "")):
        raise OfficialBaselineModelError(
            "official baseline terminal record reference is invalid"
        )
    if not re.fullmatch(
        r"baseline_closure:[0-9a-f]{64}",
        str(normalized.get("provider_closure_ref") or ""),
    ):
        raise OfficialBaselineModelError(
            "official baseline provider closure reference is invalid"
        )
    body = dict(normalized)
    checkpoint_hash = body.pop("checkpoint_sha256")
    if sha256_json(body) != checkpoint_hash:
        raise OfficialBaselineModelError("official baseline checkpoint hash differs")
    return normalized


class ExactOfficialBaselineRunner:
    """Run and validate one raw benchmark ICP under one exact v3 release."""

    def __init__(
        self,
        *,
        artifact: PrivateModelArtifactManifest,
        selection: OfficialBaselineReleaseSelection,
        dependencies: OfficialBaselineExactDependencies | None,
        spec: DockerPrivateModelSpec,
        benchmark_date: str,
        rolling_window_hash: str,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        worker_index: int = 0,
    ) -> None:
        if not selection.is_exact:
            raise OfficialBaselineReleaseSelectionError(
                "exact official baseline runner requires a v3 selection"
            )
        if dependencies is None:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline v3 protected authority is unavailable"
            )
        if not isinstance(dependencies.registration, ExactModelRunnerRegistration):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline exact registration is unavailable"
            )
        self.artifact = artifact
        self.selection = selection
        self.dependencies = dependencies
        self.spec = spec
        self.parent_graphs = tuple(parent_graphs)
        self.worker_index = int(worker_index)
        self._benchmark_date = str(benchmark_date)
        self._rolling_window_hash = _require_sha256(
            rolling_window_hash, "official baseline rolling window hash"
        )
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        selection = self.selection.selection_document
        registration = self.dependencies.registration
        registration.validate_identity()
        generation = registration.protocol_generation
        if generation.family != "model-runner-protocol:v3":
            raise OfficialBaselineModelError(
                "official baseline exact registration is not raw v3"
            )
        release_identity_hash = sha256_json(
            dict(registration.protocol.release_identity)
        )
        if (
            release_identity_hash != selection["release_identity_sha256"]
            or generation.protocol_generation_sha256
            != selection["protocol_generation_sha256"]
        ):
            raise OfficialBaselineModelError(
                "official baseline registration differs from signed selection"
            )
        projector = self.dependencies.projector
        if (
            str(projector.artifact_key) != registration.key
            or str(projector.protocol_generation_sha256)
            != generation.protocol_generation_sha256
            or str(projector.projection_identity_sha256)
            != selection["benchmark_projection_sha256"]
        ):
            raise OfficialBaselineModelError(
                "official baseline artifact projection differs from selection"
            )
        authority = self.dependencies.protected_authority
        if (
            str(authority.authority_identity_sha256)
            != selection["protected_action_authority_sha256"]
        ):
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline protected authority differs from selection"
            )
        # All cheap signed identity checks precede the immutable OCI
        # preflight.  Construction happens once before any benchmark unit is
        # claimed, so a mismatched capability/release bundle cannot consume a
        # queue item or provider reservation.
        registration.preflight(execution_mode="full_company")
        preflight = authority.preflight_run(
            run_identity=self.run_identity,
            registration=registration,
        )
        expected_preflight = {
            "schema_version": OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION,
            "run_sha256": self.run_sha256,
            "artifact_key_sha256": sha256_json({"artifact_key": registration.key}),
            "protocol_generation_sha256": generation.protocol_generation_sha256,
            "authority_identity_sha256": authority.authority_identity_sha256,
            "ready": True,
        }
        if not isinstance(preflight, Mapping) or dict(preflight) != expected_preflight:
            raise OfficialBaselineAuthorityUnavailable(
                "official baseline protected authority preflight failed"
            )

    @property
    def run_identity(self) -> Mapping[str, Any]:
        registration = self.dependencies.registration
        return {
            "schema_version": OFFICIAL_BASELINE_RUN_SCHEMA_VERSION,
            "benchmark_date": self._benchmark_date,
            "rolling_window_hash": self._rolling_window_hash,
            "model_artifact_hash": self.artifact.model_artifact_hash,
            "manifest_hash": self.artifact.manifest_hash,
            "release_selection_sha256": self.selection.selection_sha256,
            "artifact_key_sha256": sha256_json({"artifact_key": registration.key}),
            "protocol_generation_sha256": (
                registration.protocol_generation.protocol_generation_sha256
            ),
            "projection_identity_sha256": (
                self.dependencies.projector.projection_identity_sha256
            ),
            "authority_identity_sha256": (
                self.dependencies.protected_authority.authority_identity_sha256
            ),
        }

    @property
    def run_sha256(self) -> str:
        return sha256_json(dict(self.run_identity))

    def with_worker_index(self, worker_index: int) -> "ExactOfficialBaselineRunner":
        clone = object.__new__(type(self))
        clone.__dict__ = {**self.__dict__, "worker_index": int(worker_index)}
        return clone

    def with_spec(self, spec: DockerPrivateModelSpec) -> "ExactOfficialBaselineRunner":
        clone = object.__new__(type(self))
        clone.__dict__ = {**self.__dict__, "spec": spec}
        return clone

    def run_icp(
        self,
        *,
        raw_icp: Mapping[str, Any],
        icp_ref: str,
        target_count: int,
        attempt_ordinal: int = 0,
        expected_checkpoint: Mapping[str, Any] | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> OfficialBaselineModelOutput:
        registration = self.dependencies.registration
        generation_sha256 = registration.protocol_generation.protocol_generation_sha256
        if (
            not isinstance(attempt_ordinal, int)
            or isinstance(attempt_ordinal, bool)
            or attempt_ordinal < 0
        ):
            raise OfficialBaselineModelError(
                "official baseline attempt ordinal is invalid"
            )
        unit_identity: dict[str, Any] = {
            "run_sha256": self.run_sha256,
            "icp_ref": str(icp_ref),
            "raw_icp_sha256": sha256_json(dict(raw_icp)),
        }
        # Preserve every existing first-attempt unit identity. A later scoring
        # round is a new paid provider attempt, while replaying the same round
        # must remain exactly idempotent across gateway restarts.
        if attempt_ordinal:
            unit_identity["attempt_ordinal"] = attempt_ordinal
        unit_ref = "baseline_icp:" + sha256_json(
            unit_identity
        ).removeprefix("sha256:")
        model_input = registration.protocol.build_raw_input(
            raw_icp,
            source_schema=LAB_RAW_ICP_SOURCE_SCHEMA,
        )
        authority = self.dependencies.protected_authority
        coordinator = ExactModelExperimentCoordinator(
            experiment_hash=self.run_sha256,
            registration=registration,
            dispatcher=authority.dispatcher_for_unit(
                run_identity=self.run_identity,
                unit_ref=unit_ref,
            ),
            transitions=authority.transition_repository_for_unit(
                run_identity=self.run_identity,
                unit_ref=unit_ref,
            ),
        )
        unit = coordinator.run_unit(
            variant_id="official_baseline",
            unit_ref=unit_ref,
            model_input=model_input,
            execution_mode="full_company",
            target_count=int(target_count),
            evaluated_on=self._benchmark_date,
            progress_callback=progress_callback,
        )
        return self._finish_unit(
            unit=unit,
            raw_input=model_input,
            expected_checkpoint=expected_checkpoint,
        )

    def _finish_unit(
        self,
        *,
        unit: ExactModelUnitResult,
        raw_input: Mapping[str, Any],
        expected_checkpoint: Mapping[str, Any] | None,
    ) -> OfficialBaselineModelOutput:
        terminal = dict(unit.terminal_result)
        model_receipt = terminal.get("model_receipt")
        if not isinstance(model_receipt, Mapping):
            raise OfficialBaselineModelError(
                "official baseline terminal model receipt is missing"
            )
        terminal_sha = sha256_json(terminal)
        projection_value = self.dependencies.projector.project_company_outputs(
            start_request=unit.start_request,
            terminal_result=terminal,
        )
        outputs, projection_receipt = _validate_projection(
            projection_value,
            start_request=unit.start_request,
            release_identity=(
                self.dependencies.registration.protocol.release_identity
            ),
            model_receipt=model_receipt,
        )
        projection_sha256 = "sha256:" + projection_receipt["projection_sha256"]
        unit_completion = {
            "schema_version": OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION,
            "run_sha256": self.run_sha256,
            "unit_ref": unit.unit_ref,
            "protocol_generation_sha256": unit.protocol_generation_sha256,
            "raw_input_sha256": sha256_json(dict(raw_input)),
            "start_request_sha256": sha256_json(dict(unit.start_request)),
            "terminal_result_sha256": terminal_sha,
            "model_receipt_sha256": sha256_json(dict(model_receipt)),
            "projection_sha256": projection_sha256,
        }
        closure = self.dependencies.protected_authority.close_unit(
            completion=unit_completion
        )
        closure = validate_official_baseline_provider_closure(
            closure, expected_completion=unit_completion
        )
        readback_closure = self.dependencies.protected_authority.load_frontier(
            run_sha256=self.run_sha256,
            unit_ref=unit.unit_ref,
        )
        if (
            validate_official_baseline_provider_closure(
                readback_closure, expected_completion=unit_completion
            )
            != closure
        ):
            raise OfficialBaselineModelError(
                "official baseline provider frontier readback differs"
            )
        provider_frontier = {
            "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
            "ordered_attempt_keys": list(closure["ordered_attempt_keys"]),
            "ordered_attempt_sha256s": list(closure["ordered_attempt_sha256s"]),
        }
        record = {
            "schema_version": OFFICIAL_BASELINE_TERMINAL_RECORD_SCHEMA_VERSION,
            "run_identity": dict(self.run_identity),
            "unit_ref": unit.unit_ref,
            "raw_input": dict(raw_input),
            "start_request": dict(unit.start_request),
            "terminal_result": terminal,
            "model_receipt": dict(model_receipt),
            "projection_receipt": projection_receipt,
            "company_outputs": [dict(value) for value in outputs],
            "provider_frontier": provider_frontier,
        }
        record_identity = sha256_json(
            {
                "run_sha256": self.run_sha256,
                "unit_ref": unit.unit_ref,
            }
        )
        persisted = self.dependencies.terminal_authority.persist_terminal_record(
            record_identity_sha256=record_identity,
            record=record,
        )
        if not isinstance(persisted, Mapping) or set(persisted) != {
            "terminal_record_ref",
            "terminal_record_sha256",
        }:
            raise OfficialBaselineModelError(
                "official baseline terminal persistence result is invalid"
            )
        terminal_ref = str(persisted.get("terminal_record_ref") or "")
        terminal_record_sha = _require_sha256(
            persisted.get("terminal_record_sha256"),
            "official baseline terminal record hash",
        )
        if not _TERMINAL_REF_RE.fullmatch(
            terminal_ref
        ) or terminal_record_sha != sha256_json(record):
            raise OfficialBaselineModelError(
                "official baseline terminal persistence differs"
            )
        readback = self.dependencies.terminal_authority.load_terminal_record(
            terminal_record_ref=terminal_ref
        )
        if not isinstance(readback, Mapping) or dict(readback) != record:
            raise OfficialBaselineModelError(
                "official baseline terminal readback differs"
            )
        checkpoint_body = {
            "schema_version": OFFICIAL_BASELINE_CHECKPOINT_SCHEMA_VERSION,
            "release_selection_sha256": self.selection.selection_sha256,
            "artifact_key_sha256": self.run_identity["artifact_key_sha256"],
            "protocol_generation_sha256": unit.protocol_generation_sha256,
            "unit_ref": unit.unit_ref,
            "raw_input_sha256": sha256_json(dict(raw_input)),
            "start_request_sha256": sha256_json(dict(unit.start_request)),
            "terminal_result_sha256": terminal_sha,
            "model_receipt_sha256": sha256_json(dict(model_receipt)),
            "projection_identity_sha256": (
                self.dependencies.projector.projection_identity_sha256
            ),
            "projection_sha256": projection_sha256,
            "provider_frontier_sha256": closure["provider_frontier_sha256"],
            "provider_closure_ref": closure["closure_ref"],
            "provider_closure_sha256": closure["closure_sha256"],
            "terminal_record_ref": terminal_ref,
            "terminal_record_sha256": terminal_record_sha,
        }
        checkpoint = validate_official_baseline_checkpoint(
            {**checkpoint_body, "checkpoint_sha256": sha256_json(checkpoint_body)}
        )
        if expected_checkpoint is not None and checkpoint != (
            validate_official_baseline_checkpoint(expected_checkpoint)
        ):
            raise OfficialBaselineModelError(
                "official baseline reconstructed checkpoint differs"
            )
        return OfficialBaselineModelOutput(
            company_outputs=outputs,
            model_receipt=dict(model_receipt),
            checkpoint=checkpoint,
            replayed_transition_count=unit.replayed_transition_count,
        )


__all__ = [
    "ArtifactBenchmarkProjection",
    "ArtifactProtocolBenchmarkProjector",
    "EXACT_MODEL_RUNNER_FAMILY",
    "ExactOfficialBaselineRunner",
    "ImmutableArtifactBenchmarkProjector",
    "LEGACY_MODEL_RUNNER_FAMILY",
    "OFFICIAL_BASELINE_ACTION_REPLAY_RESULT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_REPLAY_IDENTITY_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_AUTHORIZATION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_RESERVATION_RESULT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_TERMINAL_KNOWN_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_TERMINAL_RESULT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_ACTION_TERMINAL_UNCERTAIN_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_AUTHORITY_PREFLIGHT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_CHECKPOINT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_EXECUTION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_PROJECTION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_RUN_REGISTRATION_RESULT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_RUN_REGISTRATION_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_RUN_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_TERMINAL_RECORD_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_UNIT_CLOSURE_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_UNIT_COMPLETION_SCHEMA_VERSION",
    "OfficialBaselineAuthorityUnavailable",
    "OfficialBaselineAttemptStore",
    "OfficialBaselineDependencyContext",
    "OfficialBaselineExactDependencies",
    "OfficialBaselineModelError",
    "OfficialBaselineModelOutput",
    "OfficialBaselineProviderPendingError",
    "OfficialBaselineReleaseSelection",
    "OfficialBaselineReleaseSelectionError",
    "OfficialBaselineTerminalAuthority",
    "ProtectedOfficialBaselineAuthority",
    "select_official_baseline_release",
    "validate_official_baseline_checkpoint",
    "validate_official_baseline_provider_closure",
]
