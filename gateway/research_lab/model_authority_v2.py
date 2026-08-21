"""Drop-in Research Lab model runner backed by the measured scoring EIF.

The parent may extract bytes from an immutable compatibility image, but the
scoring enclave independently reconstructs the source tree and requires it to
match the signed model artifact before executing it in a fresh runsc sandbox.
No parent credential or host filesystem path crosses the authority boundary.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import concurrent.futures
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Mapping, Sequence

from gateway.research_lab.attested_scoring_v2 import (
    AttestedScoringV2Error,
    execute_scoring_v2,
)
from gateway.research_lab.code_build import _extract_parent_image_source
from gateway.research_lab.v2_authority import load_source_add_catalog_snapshot_v2
from gateway.research_lab.tee_protocol import legacy_v1_enabled
from gateway.tee.model_sandbox_v2 import (
    MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
    ModelSandboxV2Error,
    provider_evidence_tape_input_root,
    validate_consumer_runtime_probe_v1,
)
from gateway.tee.scoring_executor_v2 import (
    MODEL_COMPATIBILITY_PURPOSE_V2,
    OP_RUN_MODEL_SANDBOX_V2,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_job_envelope_v2,
    build_source_add_runtime_catalog_v2,
    source_add_runtime_credential_refs_v2,
    validate_source_add_runtime_catalog_v2,
)
from gateway.tee.source_bundle_v2 import (
    build_source_bundle_v2,
)
from gateway.utils.tee_artifact_store_v2 import TEEArtifactStoreV2Error
from leadpoet_canonical.attested_v2 import (
    canonical_json,
    sha256_json,
    validate_transport_attempt,
)
from research_lab.eval import (
    DockerPrivateModelSpec,
    PrivateModelArtifactManifest,
    PrivateModelRuntimeError,
    compute_private_source_tree_hash,
    ensure_private_model_outputs,
    validate_private_model_artifact_manifest,
)
from research_lab.eval.private_runtime import (
    DockerPrivateModelRunner,
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    QUALIFICATION_OUTCOME_MAX_FAILURE_CLASSES_V2,
    QUALIFICATION_OUTCOME_PROTOCOL_MAJOR_V2,
    QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2,
    QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2,
    qualification_outcome_required_route_terminal_satisfies_v2,
    qualification_outcome_failure_class_valid_v2,
    context_with_runtime_options,
    validate_sourcing_adapter_metadata,
    validate_sourcing_runtime_receipt_entries,
    canonicalize_private_model_icp,
    publish_attested_receipt_hash,
    publish_incontainer_trace_entries,
    validate_qualification_outcome_envelope_v2,
)
from research_lab.eval.provider_costs import summarize_provider_cost_trace_entries
from research_lab.eval.provider_evidence_cache import icp_evidence_cache_key
from research_lab.sourcing_model_contract_check import (
    SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
    semantic_compatibility_policy_identity_v1,
    source_tree_compatibility_admission,
    source_tree_compatibility_admission_v1,
    validate_source_tree_compatibility_receipt,
)


_SOURCE_BUNDLE_CACHE_SIZE = 8
_SOURCE_BUNDLE_CACHE: "OrderedDict[tuple[str, str, str, str, str, str], dict[str, Any]]" = OrderedDict()
_SOURCE_COMPATIBILITY_RECEIPT_CACHE: "OrderedDict[tuple[str, str, str, str, str, str], dict[str, Any]]" = OrderedDict()
_SOURCE_BUNDLE_CACHE_LOCK = threading.Lock()
_SOURCE_BUNDLE_BUILD_LOCKS: dict[
    tuple[str, str, str, str, str, str], tuple[threading.Lock, int]
] = {}
_MEASURED_COMPATIBILITY_CACHE: "OrderedDict[tuple[str, ...], dict[str, Any]]" = (
    OrderedDict()
)
_MEASURED_COMPATIBILITY_FUTURES: dict[
    tuple[str, ...], concurrent.futures.Future
] = {}
_MEASURED_COMPATIBILITY_CACHE_LOCK = threading.Lock()
_PRIVATE_REPO_URL_ENV = "RESEARCH_LAB_PRIVATE_REPO_URL"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MEASURED_COMPATIBILITY_ADMISSION_SCHEMA_V1 = (
    "leadpoet.measured-model-compatibility-admission.v1"
)
logger = logging.getLogger(__name__)
V2_PROVIDER_PROFILE_ENV = "LEADPOET_V2_PROVIDER_CREDENTIAL_PROFILE"
PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND = "provider_evidence_tape_v2"
RETRYABLE_ATTESTED_PROVIDER_TRANSPORT_MARKER = (
    "retryable_attested_provider_transport_failure"
)
RETRYABLE_ATTESTED_ARTIFACT_PERSISTENCE_MARKER = (
    "retryable_attested_artifact_persistence_failure"
)
_RETRYABLE_ARTIFACT_PERSISTENCE_FAILURE_MARKERS = (
    "authenticated_http_408",
    "authenticated_http_429",
    "authenticated_http_500",
    "authenticated_http_502",
    "authenticated_http_503",
    "authenticated_http_504",
    "connection_refused",
    "connection_reset",
    "dns_failure",
    "gateway timeout",
    "proxy_failure",
    "read operation timed out",
    "timeout",
    "tls_failure",
    "unexpected_eof",
)
_MODEL_INVOCATION_TIMEOUT_OVERHEAD_SECONDS = 120.0
_MODEL_INVOCATION_ATTESTED_PHASES = 2.0
_MODEL_INVOCATION_PERSISTENCE_RESERVE_SECONDS = 300.0
_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "DEEPLINE_API_KEY",
        "EXA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
    }
)
_HOST_ONLY_ENV_NAMES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "RESEARCH_LAB_EVIDENCE_PROXY_URL",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR",
        "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH",
        "RESEARCH_LAB_SCORING_CACHE_DIR",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        V2_PROVIDER_PROFILE_ENV,
    }
)


class AttestedPrivateModelRunnerV2Error(PrivateModelRuntimeError):
    """The measured model result or one of its commitments is invalid."""

    def __init__(
        self,
        message: str,
        *,
        authority: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.authority = (
            deepcopy(dict(authority))
            if isinstance(authority, Mapping)
            else None
        )


MODEL_QUALIFICATION_AUTHORITY_SCHEMA_V1 = (
    "leadpoet.model-qualification-authority.v1"
)
_PLAIN_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_model_qualification_authority_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the payload-free host projection of model-owned authority."""

    if not isinstance(value, Mapping):
        raise AttestedPrivateModelRunnerV2Error(
            "model qualification authority is invalid"
        )
    document = deepcopy(dict(value))
    fields = {
        "schema_version",
        "source_commit",
        "git_commit_sha",
        "source_tree_hash",
        "model_artifact_digest",
        "manifest_hash",
        "model_manifest_sha256",
        "image_digest",
        "protocol_major",
        "protocol_minor",
        "contract_sha256",
        "completion_state",
        "disposition",
        "retryable",
        "failure_classes",
        "partial_company_count",
        "invocation_sha256",
        "input_hash",
        "route_completion_receipt_sha256",
        "provider_terminal_observation_hash",
        "host_provider_observation_root",
        "execution_receipt_hash",
        "authority_hash",
    }
    failures = document.get("failure_classes")
    body = {
        key: item for key, item in document.items() if key != "authority_hash"
    }
    completion_state = document.get("completion_state")
    disposition = document.get("disposition")
    if (
        set(document) != fields
        or document.get("schema_version")
        != MODEL_QUALIFICATION_AUTHORITY_SCHEMA_V1
        or not _GIT_SHA_RE.fullmatch(str(document.get("git_commit_sha") or ""))
        or document.get("source_commit") != document.get("git_commit_sha")
        or not _SHA256_RE.fullmatch(str(document.get("source_tree_hash") or ""))
        or document.get("model_artifact_digest")
        != document.get("source_tree_hash")
        or not _SHA256_RE.fullmatch(str(document.get("manifest_hash") or ""))
        or document.get("model_manifest_sha256")
        != document.get("manifest_hash")
        or re.fullmatch(
            r"[^\s@]+@sha256:[0-9a-f]{64}",
            str(document.get("image_digest") or ""),
        )
        is None
        or document.get("protocol_major") != QUALIFICATION_OUTCOME_PROTOCOL_MAJOR_V2
        or type(document.get("protocol_minor")) is not int
        or document["protocol_minor"] < 0
        or not _PLAIN_SHA256_RE.fullmatch(
            str(document.get("contract_sha256") or "")
        )
        or completion_state not in {"complete", "incomplete"}
        or disposition
        not in {
            "complete_nonempty",
            "complete_confirmed_empty",
            "incomplete_retryable",
            "incomplete_terminal",
        }
        or type(document.get("retryable")) is not bool
        or not isinstance(failures, list)
        or failures != sorted(failures)
        or len(failures) > QUALIFICATION_OUTCOME_MAX_FAILURE_CLASSES_V2
        or len(set(failures)) != len(failures)
        or any(
            not qualification_outcome_failure_class_valid_v2(item)
            for item in failures
        )
        or type(document.get("partial_company_count")) is not int
        or document["partial_company_count"] < 0
        or not _PLAIN_SHA256_RE.fullmatch(
            str(document.get("invocation_sha256") or "")
        )
        or any(
            not _SHA256_RE.fullmatch(str(document.get(field) or ""))
            for field in (
                "input_hash",
                "provider_terminal_observation_hash",
                "host_provider_observation_root",
                "execution_receipt_hash",
            )
        )
        or not _PLAIN_SHA256_RE.fullmatch(
            str(document.get("route_completion_receipt_sha256") or "")
        )
        or document.get("authority_hash") != sha256_json(body)
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model qualification authority differs from verified evidence"
        )
    if (
        (completion_state == "complete")
        != disposition.startswith("complete_")
        or document["retryable"]
        != (disposition == "incomplete_retryable")
        or bool(failures) != (completion_state == "incomplete")
        or (completion_state == "complete" and document["partial_company_count"])
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model qualification authority state is inconsistent"
        )
    return document


class QualificationOutcomeIncompleteV2Error(AttestedPrivateModelRunnerV2Error):
    """A signed model run completed technically but not semantically."""

    def __init__(
        self,
        message: str,
        *,
        model_qualification_authority: Mapping[str, Any],
        partial_companies: Sequence[Mapping[str, Any]] = (),
        authority: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_authority = validate_model_qualification_authority_v1(
            model_qualification_authority
        )
        normalized_partial = list(
            ensure_private_model_outputs(
                list(partial_companies),
                context_label="incomplete qualification outcome",
                require_non_empty=False,
            )
        )
        if len(normalized_partial) != normalized_authority[
            "partial_company_count"
        ]:
            raise AttestedPrivateModelRunnerV2Error(
                "incomplete qualification partial count differs"
            )
        super().__init__(message, authority=authority)
        self.model_qualification_authority = normalized_authority
        self.partial_companies = tuple(
            deepcopy(dict(company)) for company in normalized_partial
        )
        self.retryable = bool(normalized_authority["retryable"])


class QualificationOutcomeCompleteV2(list):
    """List-compatible complete result with invocation-local authority."""

    def __init__(
        self,
        companies: Sequence[Mapping[str, Any]],
        *,
        model_qualification_authority: Mapping[str, Any],
    ) -> None:
        normalized_authority = validate_model_qualification_authority_v1(
            model_qualification_authority
        )
        if normalized_authority["completion_state"] != "complete":
            raise AttestedPrivateModelRunnerV2Error(
                "complete qualification result has incomplete authority"
            )
        normalized_companies = list(
            ensure_private_model_outputs(
                list(companies),
                context_label="complete qualification outcome",
                require_non_empty=False,
            )
        )
        expected_disposition = (
            "complete_nonempty"
            if normalized_companies
            else "complete_confirmed_empty"
        )
        if normalized_authority["disposition"] != expected_disposition:
            raise AttestedPrivateModelRunnerV2Error(
                "complete qualification company count differs from authority"
            )
        super().__init__(deepcopy(normalized_companies))
        self.companies = tuple(
            deepcopy(dict(company)) for company in normalized_companies
        )
        self.model_qualification_authority = normalized_authority


def _host_provider_observation_v1(
    attempts: Sequence[Mapping[str, Any]],
    sandbox_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Corroborate the sandbox's payload-free view against signed attempts."""

    if not isinstance(attempts, (list, tuple)) or not isinstance(
        sandbox_observation, Mapping
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation is unavailable"
        )
    normalized_attempts: dict[str, dict[str, Any]] = {}
    latest_by_operation: dict[str, tuple[int, dict[str, Any]]] = {}
    for raw_attempt in attempts:
        if not isinstance(raw_attempt, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "model provider observation attempt is invalid"
            )
        attempt = dict(raw_attempt)
        try:
            validate_transport_attempt(attempt)
        except Exception as exc:
            raise AttestedPrivateModelRunnerV2Error(
                "model provider observation attempt is invalid"
            ) from exc
        attempt_hash = str(attempt["attempt_hash"])
        if attempt_hash in normalized_attempts:
            raise AttestedPrivateModelRunnerV2Error(
                "model provider observation attempt is duplicated"
            )
        normalized_attempts[attempt_hash] = attempt
        operation_id = str(attempt["logical_operation_id"])
        ordinal = int(attempt["attempt_number"])
        current = latest_by_operation.get(operation_id)
        if current is None or ordinal > current[0]:
            latest_by_operation[operation_id] = (ordinal, attempt)

    sandbox = dict(sandbox_observation)
    raw_latest_hashes = sandbox.get("latest_terminal_attempt_hashes")
    raw_successful_hashes = sandbox.get(
        "successful_latest_terminal_attempt_hashes"
    )
    if not isinstance(raw_latest_hashes, list) or not isinstance(
        raw_successful_hashes, list
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation hashes are invalid"
        )
    latest_hashes = [str(item) for item in raw_latest_hashes]
    successful_hashes = [str(item) for item in raw_successful_hashes]
    if (
        latest_hashes != sorted(latest_hashes)
        or len(set(latest_hashes)) != len(latest_hashes)
        or any(item not in normalized_attempts for item in latest_hashes)
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation differs from signed attempts"
        )
    selected = [normalized_attempts[item] for item in latest_hashes]
    if any(
        latest_by_operation[str(item["logical_operation_id"])][1][
            "attempt_hash"
        ]
        != item["attempt_hash"]
        for item in selected
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation is not the latest attempt"
        )
    expected_successful = sorted(
        str(item["attempt_hash"])
        for item in selected
        if item.get("terminal_status")
        in {"authenticated_response", "attested_local_response"}
        and type(item.get("http_status")) is int
        and 200 <= int(item["http_status"]) <= 299
    )
    accepted_count = sum(
        1
        for item in selected
        if item.get("terminal_status")
        in {"authenticated_response", "attested_local_response"}
    )
    expected_counts = {
        "request_intent_count": sandbox.get("request_intent_count"),
        "terminal_count": sandbox.get("terminal_count"),
        "latest_operation_count": len(selected),
        "accepted_latest_terminal_count": accepted_count,
        "successful_latest_terminal_count": len(expected_successful),
        "failed_latest_terminal_count": len(selected) - accepted_count,
        "unresolved_latest_terminal_count": (
            len(selected) - len(expected_successful)
        ),
    }
    if (
        type(expected_counts["request_intent_count"]) is not int
        or type(expected_counts["terminal_count"]) is not int
        or expected_counts["request_intent_count"] < len(selected)
        or expected_counts["request_intent_count"]
        != expected_counts["terminal_count"]
        or successful_hashes != expected_successful
        or any(sandbox.get(key) != item for key, item in expected_counts.items())
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation differs from signed terminals"
        )
    required_terminals = sandbox.get("required_route_terminals")
    required_commitments = sandbox.get("required_route_commitments")
    if not isinstance(required_terminals, list) or not isinstance(
        required_commitments, list
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model required-route observation is invalid"
        )
    if (
        required_commitments != sorted(required_commitments)
        or len(set(required_commitments)) != len(required_commitments)
        or len(required_commitments)
        > QUALIFICATION_OUTCOME_MAX_REQUIRED_ROUTE_OUTCOMES_V2
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(item or "")) is None
            for item in required_commitments
        )
        or len(required_terminals) != len(required_commitments)
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model required-route commitments are invalid"
        )
    safe_required_terminals = []
    successful_required = 0
    for index, raw_terminal in enumerate(required_terminals):
        if not isinstance(raw_terminal, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "model required-route terminal is invalid"
            )
        terminal = dict(raw_terminal)
        attempt = normalized_attempts.get(str(terminal.get("attempt_hash") or ""))
        if (
            attempt is None
            or index >= len(required_commitments)
            or terminal.get("route_commitment") != required_commitments[index]
            or terminal.get("terminal_status") != attempt["terminal_status"]
            or terminal.get("http_status") != attempt["http_status"]
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "model required-route terminal differs from signed attempt"
            )
        successful = (
            attempt["terminal_status"]
            in {"authenticated_response", "attested_local_response"}
            and type(attempt["http_status"]) is int
            and 200 <= attempt["http_status"] <= 299
        )
        successful_required += int(successful)
        safe_required_terminals.append(
            {
                "route_commitment": str(terminal["route_commitment"]),
                "attempt_hash": str(attempt["attempt_hash"]),
                "provider_id": str(attempt["provider_id"]),
                "terminal_status": str(attempt["terminal_status"]),
                "http_status": attempt.get("http_status"),
                "failure_code": attempt.get("failure_code"),
            }
        )
    expected_required_counts = {
        "required_route_count": len(required_commitments),
        "successful_required_route_count": successful_required,
        "unresolved_required_route_count": (
            len(required_commitments) - successful_required
        ),
    }
    if any(
        sandbox.get(key) != item
        for key, item in expected_required_counts.items()
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model required-route counts differ from signed attempts"
        )
    safe_attempts = [
        {
            "attempt_hash": str(item["attempt_hash"]),
            "provider_id": str(item["provider_id"]),
            "terminal_status": str(item["terminal_status"]),
            "http_status": item.get("http_status"),
            "failure_code": item.get("failure_code"),
        }
        for item in selected
    ]
    body = {
        "schema_version": "leadpoet.host-provider-observation.v1",
        **expected_counts,
        **expected_required_counts,
        "required_route_commitments": list(required_commitments),
        "required_route_terminals": safe_required_terminals,
        "latest_terminals": safe_attempts,
    }
    return {**body, "observation_root": sha256_json(body)}


def _model_qualification_authority_v1(
    *,
    envelope: Mapping[str, Any],
    input_doc: Mapping[str, Any],
    sandbox_result: Mapping[str, Any],
    outcome: Mapping[str, Any],
    artifact: PrivateModelArtifactManifest,
) -> dict[str, Any]:
    """Join model semantics to exact input, artifact, receipt, and transport."""

    validated_envelope = validate_qualification_outcome_envelope_v2(envelope)
    receipt = dict(validated_envelope["route_completion_receipt"])
    expected_invocation_sha256 = sha256_json(dict(input_doc)).removeprefix(
        "sha256:"
    )
    sandbox_observation = sandbox_result.get(
        "provider_terminal_observation"
    )
    if not isinstance(sandbox_observation, Mapping) or sandbox_result.get(
        "provider_terminal_observation_hash"
    ) != sha256_json(dict(sandbox_observation)):
        raise AttestedPrivateModelRunnerV2Error(
            "model provider observation commitment differs"
        )
    host_observation = _host_provider_observation_v1(
        outcome.get("transport_attempts") or (),
        sandbox_observation,
    )
    execution_receipt = outcome.get("execution_receipt")
    execution_graph = outcome.get("execution_receipt_graph")
    execution_receipt_hash = (
        str(execution_receipt.get("receipt_hash") or "")
        if isinstance(execution_receipt, Mapping)
        else ""
    )
    if (
        receipt.get("invocation_sha256") != expected_invocation_sha256
        or sandbox_result.get("input_hash") != sha256_json(dict(input_doc))
        or not _SHA256_RE.fullmatch(execution_receipt_hash)
        or not isinstance(execution_graph, Mapping)
        or execution_graph.get("root_receipt_hash")
        != execution_receipt_hash
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model qualification input or execution authority differs"
        )
    disposition = str(receipt["disposition"])
    bound_outcomes = receipt.get("extensions", {}).get(
        QUALIFICATION_OUTCOME_REQUIRED_ROUTE_OUTCOMES_EXTENSION_V2
    )
    bound_commitments = (
        [item.get("commitment") for item in bound_outcomes]
        if isinstance(bound_outcomes, list)
        and all(isinstance(item, Mapping) for item in bound_outcomes)
        else []
    )
    required_terminals = host_observation["required_route_terminals"]
    if (
        receipt.get("probe") is not None
        or not isinstance(bound_outcomes, list)
        or bound_commitments
        != host_observation["required_route_commitments"]
        or receipt["route_summary"]["attempted"]
        != len(bound_commitments)
        or any(
            outcome.get("commitment") != terminal.get("route_commitment")
            or (
                outcome.get("state") in {"completed", "confirmed_empty"}
                and not qualification_outcome_required_route_terminal_satisfies_v2(
                    outcome.get("state"),
                    terminal.get("terminal_status"),
                    terminal.get("http_status"),
                )
            )
            for outcome, terminal in zip(bound_outcomes, required_terminals)
        )
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "qualification outcome lacks exact required-route authority"
        )
    if disposition.startswith("complete_") and (
        host_observation["required_route_count"] <= 0
        or any(
            outcome.get("state") not in {"completed", "confirmed_empty"}
            or not qualification_outcome_required_route_terminal_satisfies_v2(
                outcome.get("state"),
                terminal.get("terminal_status"),
                terminal.get("http_status"),
            )
            for outcome, terminal in zip(bound_outcomes, required_terminals)
        )
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "complete outcome lacks successful provider completion"
        )
    companies = list(validated_envelope["companies"])
    body = {
        "schema_version": MODEL_QUALIFICATION_AUTHORITY_SCHEMA_V1,
        "source_commit": artifact.git_commit_sha,
        "git_commit_sha": artifact.git_commit_sha,
        "source_tree_hash": artifact.model_artifact_hash,
        "model_artifact_digest": artifact.model_artifact_hash,
        "manifest_hash": artifact.manifest_hash,
        "model_manifest_sha256": artifact.manifest_hash,
        "image_digest": artifact.image_digest,
        "protocol_major": int(validated_envelope["protocol_major"]),
        "protocol_minor": int(validated_envelope["protocol_minor"]),
        "contract_sha256": str(validated_envelope["contract_sha256"]),
        "completion_state": str(validated_envelope["completion_state"]),
        "disposition": disposition,
        "retryable": bool(receipt["retryable"]),
        "failure_classes": list(receipt["failure_classes"]),
        "partial_company_count": (
            len(companies)
            if validated_envelope["completion_state"] == "incomplete"
            else 0
        ),
        "invocation_sha256": expected_invocation_sha256,
        "input_hash": str(sandbox_result["input_hash"]),
        "route_completion_receipt_sha256": str(receipt["receipt_sha256"]),
        "provider_terminal_observation_hash": str(
            sandbox_result["provider_terminal_observation_hash"]
        ),
        "host_provider_observation_root": str(
            host_observation["observation_root"]
        ),
        "execution_receipt_hash": execution_receipt_hash,
    }
    return validate_model_qualification_authority_v1(
        {**body, "authority_hash": sha256_json(body)}
    )


def _model_invocation_timeout_seconds(model_timeout_seconds: float) -> float:
    """Bound measured execution plus its required artifact-lineage attestation."""

    model_timeout = max(1.0, float(model_timeout_seconds))
    attested_phase_timeout = (
        model_timeout + _MODEL_INVOCATION_TIMEOUT_OVERHEAD_SECONDS
    )
    return (
        attested_phase_timeout * _MODEL_INVOCATION_ATTESTED_PHASES
        + _MODEL_INVOCATION_PERSISTENCE_RESERVE_SECONDS
    )


def has_retryable_attested_provider_transport_failure(
    error: AttestedScoringV2Error,
) -> bool:
    """Recognize retryable model-scope failures from verified receipt evidence."""

    if "execution_providerclientv2error" not in str(error).lower():
        return False
    authority = error.authority
    if not isinstance(authority, Mapping):
        return False
    attempts = authority.get("transport_attempts")
    if not isinstance(attempts, list):
        return False
    latest: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            return False
        logical_operation_id = str(attempt.get("logical_operation_id") or "")
        if isinstance(attempt.get("attempt_number"), bool):
            return False
        try:
            attempt_number = int(attempt.get("attempt_number"))
        except (TypeError, ValueError):
            return False
        if not logical_operation_id or attempt_number < 0:
            return False
        current = latest.get(logical_operation_id)
        if current is None or attempt_number > current[0]:
            latest[logical_operation_id] = (attempt_number, dict(attempt))
    for _attempt_number, attempt in latest.values():
        terminal_status = str(attempt.get("terminal_status") or "")
        if terminal_status == "transport_failure":
            return True
        if terminal_status not in {
            "authenticated_response",
            "attested_local_response",
        }:
            continue
        http_status = attempt.get("http_status")
        if isinstance(http_status, bool):
            continue
        try:
            status = int(http_status)
        except (TypeError, ValueError):
            continue
        provider_id = str(attempt.get("provider_id") or "").strip().lower()
        if status in {408, 429} or 500 <= status <= 599:
            return True
        if provider_id == "scrapingdog" and status == 400:
            return True
    return False


def _run_private_source_git(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> str:
    """Run private-source Git without exposing credentialed remotes in errors."""

    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AttestedPrivateModelRunnerV2Error(
            "exact private model source checkout failed"
        ) from exc
    if result.returncode != 0:
        raise AttestedPrivateModelRunnerV2Error(
            "exact private model source checkout failed"
        )
    return result.stdout.strip()


def _checkout_private_source_for_artifact(
    artifact: PrivateModelArtifactManifest,
    *,
    destination: Path,
    timeout_seconds: int,
) -> str:
    repo_url = str(os.getenv(_PRIVATE_REPO_URL_ENV) or "").strip()
    commit_sha = str(artifact.git_commit_sha or "").strip().lower()
    if not repo_url:
        raise AttestedPrivateModelRunnerV2Error(
            f"{_PRIVATE_REPO_URL_ENV} is required when the immutable image contains only the runtime source closure"
        )
    if not _GIT_SHA_RE.fullmatch(commit_sha):
        raise AttestedPrivateModelRunnerV2Error(
            "private model artifact commit is not a full Git SHA"
        )
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    _run_private_source_git(
        ["init", "--quiet"],
        cwd=destination,
        timeout_seconds=timeout_seconds,
    )
    _run_private_source_git(
        ["remote", "add", "origin", repo_url],
        cwd=destination,
        timeout_seconds=timeout_seconds,
    )
    _run_private_source_git(
        ["fetch", "--quiet", "--depth", "1", "origin", commit_sha],
        cwd=destination,
        timeout_seconds=timeout_seconds,
    )
    _run_private_source_git(
        ["checkout", "--quiet", "--detach", "FETCH_HEAD"],
        cwd=destination,
        timeout_seconds=timeout_seconds,
    )
    observed_commit = _run_private_source_git(
        ["rev-parse", "HEAD"],
        cwd=destination,
        timeout_seconds=timeout_seconds,
    ).lower()
    if observed_commit != commit_sha:
        raise AttestedPrivateModelRunnerV2Error(
            "checked-out private model source commit differs from its signed artifact"
        )
    return compute_private_source_tree_hash(destination)


class _LegacyPrivateModelRunnerAdapter:
    """Current-commit host runner with the V2 runner's public interface."""

    def __init__(
        self,
        *,
        artifact: PrivateModelArtifactManifest | Mapping[str, Any],
        spec: DockerPrivateModelSpec | Mapping[str, Any],
        model_kind: str,
        worker_index: int,
        epoch_id: int | None = None,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        **_kwargs: Any,
    ) -> None:
        self.artifact = (
            artifact
            if isinstance(artifact, PrivateModelArtifactManifest)
            else PrivateModelArtifactManifest.from_mapping(artifact)
        )
        errors = validate_private_model_artifact_manifest(self.artifact)
        if errors:
            raise AttestedPrivateModelRunnerV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        self.spec = (
            spec
            if isinstance(spec, DockerPrivateModelSpec)
            else DockerPrivateModelSpec.from_mapping(spec)
        )
        if self.spec.image_digest != self.artifact.image_digest:
            raise AttestedPrivateModelRunnerV2Error(
                "legacy model runner image differs from the signed artifact"
            )
        if model_kind not in {"private", "candidate"}:
            raise AttestedPrivateModelRunnerV2Error(
                "legacy model runner kind is invalid"
            )
        self.model_kind = model_kind
        self.worker_index = int(worker_index)
        self.epoch_id = int(epoch_id) if epoch_id is not None else None
        self.parent_graphs = tuple(dict(item) for item in parent_graphs)
        self._runner = DockerPrivateModelRunner(self.spec)

    def with_spec(self, spec: DockerPrivateModelSpec) -> "_LegacyPrivateModelRunnerAdapter":
        return _LegacyPrivateModelRunnerAdapter(
            artifact=self.artifact,
            spec=spec,
            model_kind=self.model_kind,
            worker_index=self.worker_index,
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
        )

    def with_worker_index(self, worker_index: int) -> "_LegacyPrivateModelRunnerAdapter":
        return _LegacyPrivateModelRunnerAdapter(
            artifact=self.artifact,
            spec=self.spec,
            model_kind=self.model_kind,
            worker_index=int(worker_index),
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
        )

    def attested_receipts(self) -> list[dict[str, Any]]:
        return []

    def attested_authorities(self) -> list[dict[str, Any]]:
        return []

    def measured_compatibility_admission(self) -> dict[str, Any]:
        raise AttestedPrivateModelRunnerV2Error(
            "legacy model runtime has no measured compatibility admission"
        )

    async def __call__(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        return list(await asyncio.to_thread(self._runner, icp, context))

    def metadata(self) -> Mapping[str, Any]:
        return self._runner.metadata()


@contextmanager
def _source_bundle_build_lock(
    cache_key: tuple[str, str, str, str, str, str],
) -> Any:
    with _SOURCE_BUNDLE_CACHE_LOCK:
        build_lock, users = _SOURCE_BUNDLE_BUILD_LOCKS.get(
            cache_key,
            (threading.Lock(), 0),
        )
        _SOURCE_BUNDLE_BUILD_LOCKS[cache_key] = (build_lock, users + 1)
    try:
        with build_lock:
            yield
    finally:
        with _SOURCE_BUNDLE_CACHE_LOCK:
            current = _SOURCE_BUNDLE_BUILD_LOCKS.get(cache_key)
            if current is not None and current[0] is build_lock:
                if current[1] == 1:
                    _SOURCE_BUNDLE_BUILD_LOCKS.pop(cache_key, None)
                else:
                    _SOURCE_BUNDLE_BUILD_LOCKS[cache_key] = (
                        build_lock,
                        current[1] - 1,
                    )


def _validated_cached_source_bundle_and_receipt_v2(
    cache_key: tuple[str, str, str, str, str, str],
    *,
    artifact: PrivateModelArtifactManifest,
    policy: Mapping[str, Any],
    policy_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Validate immutable cached commitments without repeating tree I/O."""

    with _SOURCE_BUNDLE_CACHE_LOCK:
        cached_bundle = _SOURCE_BUNDLE_CACHE.get(cache_key)
        cached_receipt = _SOURCE_COMPATIBILITY_RECEIPT_CACHE.get(cache_key)
        if cached_bundle is None or cached_receipt is None:
            if cached_bundle is not None or cached_receipt is not None:
                _SOURCE_BUNDLE_CACHE.pop(cache_key, None)
                _SOURCE_COMPATIBILITY_RECEIPT_CACHE.pop(cache_key, None)
            return None
        bundle = deepcopy(cached_bundle)
        receipt = deepcopy(cached_receipt)
    try:
        receipt = validate_source_tree_compatibility_receipt(
            receipt,
            manifest=artifact,
            source_tree_hash=artifact.model_artifact_hash,
            policy=policy,
            policy_hash=policy_hash,
        )
    except ValueError:
        with _SOURCE_BUNDLE_CACHE_LOCK:
            _SOURCE_BUNDLE_CACHE.pop(cache_key, None)
            _SOURCE_COMPATIBILITY_RECEIPT_CACHE.pop(cache_key, None)
        return None
    with _SOURCE_BUNDLE_CACHE_LOCK:
        # Do not return a snapshot that another thread replaced while its
        # archive was being validated.
        if (
            _SOURCE_BUNDLE_CACHE.get(cache_key) != cached_bundle
            or _SOURCE_COMPATIBILITY_RECEIPT_CACHE.get(cache_key)
            != cached_receipt
        ):
            return None
        _SOURCE_BUNDLE_CACHE.move_to_end(cache_key)
        _SOURCE_COMPATIBILITY_RECEIPT_CACHE.move_to_end(cache_key)
    return bundle, receipt


def _source_bundle_and_compatibility_receipt_for_artifact(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, policy_hash = semantic_compatibility_policy_identity_v1()
    cache_key = (
        artifact.model_artifact_hash,
        artifact.manifest_hash,
        artifact.image_digest,
        policy_hash,
        str(policy["consumer_api_version"]),
        SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
    )
    with _source_bundle_build_lock(cache_key):
        cached = _validated_cached_source_bundle_and_receipt_v2(
            cache_key,
            artifact=artifact,
            policy=policy,
            policy_hash=policy_hash,
        )
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory(prefix="research-lab-model-v2-source-") as tmp:
            source_root = Path(tmp) / "app"
            observed_tree_hash, _paths = _extract_parent_image_source(
                image_digest=artifact.image_digest,
                source_dir=source_root,
                timeout_seconds=max(120, int(timeout_seconds)),
            )
            if observed_tree_hash != artifact.model_artifact_hash:
                logger.info(
                    "research_lab_v2_image_contains_runtime_source_closure "
                    "artifact=%s image_source=%s",
                    artifact.model_artifact_hash,
                    observed_tree_hash,
                )
                observed_tree_hash = _checkout_private_source_for_artifact(
                    artifact,
                    destination=source_root,
                    timeout_seconds=max(120, int(timeout_seconds)),
                )
            if observed_tree_hash != artifact.model_artifact_hash:
                raise AttestedPrivateModelRunnerV2Error(
                    "exact private model source differs from its signed artifact"
                )
            try:
                compatibility_receipt = source_tree_compatibility_admission(
                    source_root,
                    manifest=artifact,
                    source_tree_hash=observed_tree_hash,
                    use_cache=True,
                )
                compatibility_receipt = (
                    validate_source_tree_compatibility_receipt(
                        compatibility_receipt,
                        manifest=artifact,
                        source_tree_hash=artifact.model_artifact_hash,
                        policy=policy,
                        policy_hash=policy_hash,
                    )
                )
            except ValueError as exc:
                raise AttestedPrivateModelRunnerV2Error(str(exc)) from exc
            if compute_private_source_tree_hash(source_root) != observed_tree_hash:
                raise AttestedPrivateModelRunnerV2Error(
                    "private model source changed during compatibility admission"
                )
            bundle = build_source_bundle_v2(source_root)
        if bundle.get("source_tree_hash") != artifact.model_artifact_hash:
            raise AttestedPrivateModelRunnerV2Error(
                "model source bundle differs from its signed artifact"
            )
        with _SOURCE_BUNDLE_CACHE_LOCK:
            _SOURCE_BUNDLE_CACHE[cache_key] = deepcopy(bundle)
            _SOURCE_COMPATIBILITY_RECEIPT_CACHE[cache_key] = deepcopy(
                compatibility_receipt
            )
            _SOURCE_BUNDLE_CACHE.move_to_end(cache_key)
            _SOURCE_COMPATIBILITY_RECEIPT_CACHE.move_to_end(cache_key)
            while len(_SOURCE_BUNDLE_CACHE) > _SOURCE_BUNDLE_CACHE_SIZE:
                expired_key, _expired_bundle = _SOURCE_BUNDLE_CACHE.popitem(
                    last=False
                )
                _SOURCE_COMPATIBILITY_RECEIPT_CACHE.pop(expired_key, None)
        return deepcopy(bundle), deepcopy(compatibility_receipt)


def _source_bundle_for_artifact(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    bundle, _receipt = _source_bundle_and_compatibility_receipt_for_artifact(
        artifact,
        timeout_seconds=timeout_seconds,
    )
    return bundle


def private_model_compatibility_receipt_v2(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Return the host-side source admission bound to an immutable artifact."""

    _bundle, receipt = _source_bundle_and_compatibility_receipt_for_artifact(
        artifact,
        timeout_seconds=timeout_seconds,
    )
    return receipt


def _validate_measured_compatibility_admission_v2(
    admission: Mapping[str, Any],
    *,
    artifact: PrivateModelArtifactManifest,
    spec: DockerPrivateModelSpec,
    host_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the enclave-observed runtime admission on the host."""

    normalized = dict(admission)
    expected_fields = {
        "schema_version",
        "decision",
        "admission_mode",
        "consumer_api_version",
        "compatibility_policy_hash",
        "compatibility_admission_hash",
        "source_tree_hash",
        "manifest_hash",
        "image_digest",
        "module_name",
        "callable_name",
        "consumer_runtime_probe_hash",
        "adapter_metadata_hash",
        "execution_receipt_hash",
        "receipt_hash",
    }
    body = {
        key: value for key, value in normalized.items() if key != "receipt_hash"
    }
    if (
        set(normalized) != expected_fields
        or normalized.get("schema_version")
        != MEASURED_COMPATIBILITY_ADMISSION_SCHEMA_V1
        or normalized.get("decision")
        != SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION
        or normalized.get("admission_mode") != host_receipt.get("admission_mode")
        or normalized.get("consumer_api_version")
        != host_receipt.get("consumer_api_version")
        or normalized.get("compatibility_policy_hash")
        != host_receipt.get("policy_hash")
        or normalized.get("compatibility_admission_hash")
        != host_receipt.get("receipt_hash")
        or normalized.get("source_tree_hash") != artifact.model_artifact_hash
        or normalized.get("manifest_hash") != artifact.manifest_hash
        or normalized.get("image_digest") != artifact.image_digest
        or normalized.get("module_name") != spec.module_name
        or normalized.get("callable_name") != "adapter_metadata"
        or not _SHA256_RE.fullmatch(
            str(normalized.get("consumer_runtime_probe_hash") or "")
        )
        or not _SHA256_RE.fullmatch(
            str(normalized.get("adapter_metadata_hash") or "")
        )
        or not _SHA256_RE.fullmatch(
            str(normalized.get("execution_receipt_hash") or "")
        )
        or normalized.get("receipt_hash") != sha256_json(body)
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "measured model compatibility admission differs from host admission"
        )
    return normalized


def _combined_compatibility_proof_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    spec: DockerPrivateModelSpec,
    host_receipt: Mapping[str, Any],
    measured_admission: Mapping[str, Any],
) -> dict[str, Any]:
    measured = _validate_measured_compatibility_admission_v2(
        measured_admission,
        artifact=artifact,
        spec=spec,
        host_receipt=host_receipt,
    )
    combined_body = {
        "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "host_compatibility_receipt_hash": host_receipt["receipt_hash"],
        "measured_runtime_receipt_hash": measured["receipt_hash"],
        "measured_runtime_probe_hash": measured[
            "consumer_runtime_probe_hash"
        ],
    }
    return {
        **dict(host_receipt),
        "measured_runtime_admission": measured,
        "measured_runtime_decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
        "measured_runtime_probe_hash": measured[
            "consumer_runtime_probe_hash"
        ],
        "combined_receipt_hash": sha256_json(combined_body),
    }


def _measured_compatibility_cache_key_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    spec: DockerPrivateModelSpec,
    host_receipt: Mapping[str, Any],
) -> tuple[str, ...]:
    return (
        artifact.model_artifact_hash,
        artifact.manifest_hash,
        artifact.image_digest,
        str(host_receipt["policy_hash"]),
        str(host_receipt["receipt_hash"]),
        spec.module_name,
        "adapter_metadata",
    )


async def _execute_measured_compatibility_preflight_v2(
    *,
    artifact: PrivateModelArtifactManifest,
    spec: DockerPrivateModelSpec,
    host_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one measured metadata admission without a model provider broker."""

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=spec,
        model_kind="private",
        worker_index=0,
        epoch_id=0,
    )
    await asyncio.to_thread(runner.metadata)
    return runner.measured_compatibility_admission()


async def preflight_private_model_compatibility_v2(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
    measured_preflight_executor: Any = None,
) -> dict[str, Any]:
    """Bind immutable host admission to one measured runtime admission."""

    host_receipt = await asyncio.to_thread(
        private_model_compatibility_receipt_v2,
        artifact,
        timeout_seconds=timeout_seconds,
    )
    spec = DockerPrivateModelSpec(
        image_digest=artifact.image_digest,
        timeout_seconds=int(timeout_seconds),
        env_passthrough=(),
        extra_env={},
    )
    executor = (
        measured_preflight_executor
        or _execute_measured_compatibility_preflight_v2
    )
    cache_enabled = measured_preflight_executor is None
    cache_key = _measured_compatibility_cache_key_v2(
        artifact=artifact,
        spec=spec,
        host_receipt=host_receipt,
    )
    leader_future: concurrent.futures.Future | None = None
    if cache_enabled:
        while True:
            with _MEASURED_COMPATIBILITY_CACHE_LOCK:
                cached = deepcopy(
                    _MEASURED_COMPATIBILITY_CACHE.get(cache_key)
                )
                pending = _MEASURED_COMPATIBILITY_FUTURES.get(cache_key)
                if cached is None and pending is None:
                    pending = concurrent.futures.Future()
                    _MEASURED_COMPATIBILITY_FUTURES[cache_key] = pending
                    leader_future = pending
            if cached is not None:
                try:
                    proof = _combined_compatibility_proof_v2(
                        artifact=artifact,
                        spec=spec,
                        host_receipt=host_receipt,
                        measured_admission=cached,
                    )
                except AttestedPrivateModelRunnerV2Error:
                    with _MEASURED_COMPATIBILITY_CACHE_LOCK:
                        if _MEASURED_COMPATIBILITY_CACHE.get(cache_key) == cached:
                            _MEASURED_COMPATIBILITY_CACHE.pop(cache_key, None)
                    continue
                with _MEASURED_COMPATIBILITY_CACHE_LOCK:
                    _MEASURED_COMPATIBILITY_CACHE.move_to_end(cache_key)
                return proof
            if leader_future is not None:
                break
            shared = await asyncio.shield(asyncio.wrap_future(pending))
            return _combined_compatibility_proof_v2(
                artifact=artifact,
                spec=spec,
                host_receipt=host_receipt,
                measured_admission=shared,
            )
    try:
        measured_result = executor(
            artifact=artifact,
            spec=spec,
            host_receipt=host_receipt,
        )
        if inspect.isawaitable(measured_result):
            try:
                measured_result = await asyncio.wait_for(
                    measured_result,
                    timeout=_model_invocation_timeout_seconds(timeout_seconds),
                )
            except asyncio.TimeoutError as exc:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured compatibility preflight timed out"
                ) from exc
        if not isinstance(measured_result, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "measured compatibility preflight result is invalid"
            )
        proof = _combined_compatibility_proof_v2(
            artifact=artifact,
            spec=spec,
            host_receipt=host_receipt,
            measured_admission=measured_result,
        )
    except BaseException as exc:
        if leader_future is not None:
            with _MEASURED_COMPATIBILITY_CACHE_LOCK:
                _MEASURED_COMPATIBILITY_FUTURES.pop(cache_key, None)
            leader_future.set_exception(exc)
            leader_future.exception()
        raise
    if leader_future is not None:
        with _MEASURED_COMPATIBILITY_CACHE_LOCK:
            measured_admission = deepcopy(proof["measured_runtime_admission"])
            _MEASURED_COMPATIBILITY_CACHE[cache_key] = measured_admission
            _MEASURED_COMPATIBILITY_CACHE.move_to_end(cache_key)
            while len(_MEASURED_COMPATIBILITY_CACHE) > _SOURCE_BUNDLE_CACHE_SIZE:
                _MEASURED_COMPATIBILITY_CACHE.popitem(last=False)
            _MEASURED_COMPATIBILITY_FUTURES.pop(cache_key, None)
        leader_future.set_result(deepcopy(measured_admission))
    return proof


async def source_bundle_for_artifact_v2(
    artifact: PrivateModelArtifactManifest,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _source_bundle_for_artifact,
        artifact,
        timeout_seconds=timeout_seconds,
    )


def _provider_evidence_cache(
    spec: DockerPrivateModelSpec,
    *,
    canonical_icp: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extra_env = dict(spec.extra_env or {})
    cache_path = ""
    cache_dir = str(
        extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or ""
    ).strip()
    if cache_dir and canonical_icp is not None:
        cache_path = str(
            Path(cache_dir) / (icp_evidence_cache_key(canonical_icp) + ".json")
        )
    if not cache_path:
        cache_path = str(
            extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH") or ""
        ).strip()
    if not cache_path or not Path(cache_path).is_file():
        return {}
    try:
        document = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache is unreadable"
        ) from exc
    if not isinstance(document, Mapping):
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache is not an object"
        )
    return dict(document)


def _provider_evidence_cache_ref(
    canonical_icp: Mapping[str, Any] | None,
) -> str:
    return icp_evidence_cache_key(canonical_icp) if canonical_icp is not None else ""


def _require_tape_receipt(
    graph: Mapping[str, Any],
    *,
    cache_ref: str,
    cache_hash: str,
) -> Mapping[str, Any]:
    expected_input_root = provider_evidence_tape_input_root(cache_ref, cache_hash)
    matches = [
        item
        for item in graph.get("receipts") or ()
        if isinstance(item, Mapping)
        and item.get("role") == "gateway_scoring"
        and item.get("purpose") == "research_lab.provider_evidence_tape.v2"
        and item.get("status") == "succeeded"
        and item.get("input_root") == expected_input_root
        and item.get("output_root") == cache_hash
    ]
    if len(matches) != 1:
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache has no unique measured tape receipt"
        )
    return matches[0]


async def _load_provider_evidence_tape_graphs(
    *,
    cache_ref: str,
    cache_hash: str,
) -> tuple[dict[str, Any], ...]:
    from gateway.research_lab.attested_v2_store import (
        load_business_artifact_graph_v2,
        load_receipt_graph_v2,
    )

    lineage_graph = await load_business_artifact_graph_v2(
        artifact_kind=PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND,
        artifact_ref=cache_ref,
        artifact_hash=cache_hash,
    )
    try:
        _require_tape_receipt(
            lineage_graph,
            cache_ref=cache_ref,
            cache_hash=cache_hash,
        )
        return (dict(lineage_graph),)
    except AttestedPrivateModelRunnerV2Error:
        receipts = {
            str(item.get("receipt_hash") or ""): item
            for item in lineage_graph.get("receipts") or ()
            if isinstance(item, Mapping)
        }
        lineage_root = receipts.get(
            str(lineage_graph.get("root_receipt_hash") or "")
        )
        parent_hashes = (
            list(lineage_root.get("parent_receipt_hashes") or ())
            if isinstance(lineage_root, Mapping)
            else []
        )
        if (
            not isinstance(lineage_root, Mapping)
            or lineage_root.get("role") != "gateway_coordinator"
            or lineage_root.get("purpose")
            != "leadpoet.artifact_persistence.v2"
            or lineage_root.get("status") != "succeeded"
            or len(parent_hashes) != 1
        ):
            raise
        source_graph = await load_receipt_graph_v2(str(parent_hashes[0]))
        _require_tape_receipt(
            source_graph,
            cache_ref=cache_ref,
            cache_hash=cache_hash,
        )
        return (dict(source_graph), dict(lineage_graph))


async def _persist_provider_evidence_tape_link(
    *,
    receipt_hash: str,
    cache_ref: str,
    cache_hash: str,
) -> dict[str, Any]:
    from gateway.research_lab.attested_v2_store import (
        AttestedV2StoreError,
        persist_business_artifact_links_v2,
    )

    artifact = {
        "artifact_kind": PROVIDER_EVIDENCE_TAPE_ARTIFACT_KIND,
        "artifact_ref": cache_ref,
        "artifact_hash": cache_hash,
    }
    try:
        return await persist_business_artifact_links_v2(
            receipt_hash=receipt_hash,
            artifacts=(artifact,),
        )
    except AttestedV2StoreError as exc:
        if "stored row conflicts at receipt_hash" not in str(exc):
            raise

        # An interrupted/retried baseline may reproduce the exact immutable
        # tape under a new measured receipt. The artifact table deliberately
        # retains its first receipt owner, so validate and reuse that owner's
        # complete graph instead of treating exact replay as corruption.
        existing_graphs = await _load_provider_evidence_tape_graphs(
            cache_ref=cache_ref,
            cache_hash=cache_hash,
        )
        owner_receipt_hash = str(
            existing_graphs[-1].get("root_receipt_hash") or ""
        ).lower()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", owner_receipt_hash) is None:
            raise AttestedPrivateModelRunnerV2Error(
                "provider evidence tape owner receipt is invalid"
            ) from exc
        canonical_link = {"receipt_hash": owner_receipt_hash, **artifact}
        return {
            "business_artifact_link_count": 1,
            "business_artifact_link_set_hash": sha256_json([canonical_link]),
        }


def _write_provider_evidence_cache(
    *,
    cache_ref: str,
    cache_document: Mapping[str, Any],
) -> str:
    cache_dir = str(
        os.getenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or ""
    ).strip()
    if not cache_dir:
        return ""
    destination_dir = Path(cache_dir)
    if not destination_dir.is_dir():
        raise AttestedPrivateModelRunnerV2Error(
            "provider evidence cache directory is unavailable"
        )
    destination = destination_dir / (cache_ref + ".json")
    encoded = canonical_json(dict(cache_document)).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=destination.name + ".tmp.",
        dir=str(destination_dir),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def _measured_environment(
    spec: DockerPrivateModelSpec,
    *,
    additional_credential_env_names: Sequence[str] = (),
) -> dict[str, str]:
    credential_env_names = _CREDENTIAL_ENV_NAMES | frozenset(
        str(item) for item in additional_credential_env_names
    )
    environment = {}
    for name, value in dict(spec.extra_env or {}).items():
        normalized_name = str(name)
        if normalized_name in credential_env_names or normalized_name in _HOST_ONLY_ENV_NAMES:
            continue
        environment[normalized_name] = str(value)
    for name in spec.env_passthrough:
        if name in credential_env_names or name in _HOST_ONLY_ENV_NAMES:
            continue
        # The legacy runner only forwards names present in the process env.
        import os

        if name in os.environ:
            environment[str(name)] = str(os.environ[name])
    return environment


def _measured_environment_for_provider_cost_scope(
    spec: DockerPrivateModelSpec,
    *,
    provider_cost_scope: str,
    additional_credential_env_names: Sequence[str] = (),
) -> dict[str, str]:
    environment = _measured_environment(
        spec,
        additional_credential_env_names=additional_credential_env_names,
    )
    environment[PROVIDER_COST_EVALUATION_SCOPE_ENV] = str(provider_cost_scope)
    return environment


def _empty_model_compatibility_provider_profile(
    profile: str,
    *,
    execution_role: str,
    worker_index: int,
    require_egress_proxy: bool,
) -> dict[str, Any]:
    """Return the measured metadata profile without reading credential files."""

    if (
        str(profile or "default") != "default"
        or execution_role != "gateway_scoring"
        or int(worker_index) < 0
        or require_egress_proxy
    ):
        raise AttestedPrivateModelRunnerV2Error(
            "model compatibility provider profile is not isolated"
        )
    return {
        "profile": "default",
        "execution_role": "gateway_scoring",
        "worker_index": int(worker_index),
        "egress_proxy_required": False,
        "credential_ref_hashes": {},
        "envelopes": [],
        "profile_hash": sha256_json(
            {
                "schema_version": "leadpoet.provider_profile.v2",
                "profile": "default",
                "execution_role": "gateway_scoring",
                "worker_index": int(worker_index),
                "egress_proxy_required": False,
                "credential_ref_hashes": {},
            }
        ),
    }


class AttestedPrivateModelRunnerV2:
    """The existing model-runner interface with V2 enclave authority."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls is AttestedPrivateModelRunnerV2 and legacy_v1_enabled():
            return _LegacyPrivateModelRunnerAdapter(*args, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        *,
        artifact: PrivateModelArtifactManifest | Mapping[str, Any],
        spec: DockerPrivateModelSpec | Mapping[str, Any],
        model_kind: str,
        worker_index: int,
        epoch_id: int | None = None,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        execute: Any = execute_scoring_v2,
        catalog_snapshot_loader: Any = None,
        _shared_state: dict[str, Any] | None = None,
    ) -> None:
        self.artifact = (
            artifact
            if isinstance(artifact, PrivateModelArtifactManifest)
            else PrivateModelArtifactManifest.from_mapping(artifact)
        )
        errors = validate_private_model_artifact_manifest(self.artifact)
        if errors:
            raise AttestedPrivateModelRunnerV2Error(
                "model artifact is invalid: " + "; ".join(errors)
            )
        self.spec = (
            spec
            if isinstance(spec, DockerPrivateModelSpec)
            else DockerPrivateModelSpec.from_mapping(spec)
        )
        if self.spec.image_digest != self.artifact.image_digest:
            raise AttestedPrivateModelRunnerV2Error(
                "model runner image differs from the signed artifact"
            )
        if model_kind not in {"private", "candidate"}:
            raise AttestedPrivateModelRunnerV2Error("model runner kind is invalid")
        self.model_kind = model_kind
        self.worker_index = int(worker_index)
        self.epoch_id = int(epoch_id) if epoch_id is not None else None
        if self.epoch_id is not None and self.epoch_id < 0:
            raise AttestedPrivateModelRunnerV2Error("model authority epoch is invalid")
        self.parent_graphs = tuple(dict(item) for item in parent_graphs)
        self._execute = execute
        self._catalog_snapshot_loader = catalog_snapshot_loader
        self._shared_state = _shared_state or {
            "sequence": 0,
            "receipts": [],
            "authorities": [],
            "compatibility_admissions": [],
            "generated_caches": {},
            "evidence_summaries": {},
            "catalog_snapshot_futures": {},
            "lock": threading.Lock(),
        }

    def with_spec(self, spec: DockerPrivateModelSpec) -> "AttestedPrivateModelRunnerV2":
        return AttestedPrivateModelRunnerV2(
            artifact=self.artifact,
            spec=spec,
            model_kind=self.model_kind,
            worker_index=self.worker_index,
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
            execute=self._execute,
            catalog_snapshot_loader=self._catalog_snapshot_loader,
            _shared_state=self._shared_state,
        )

    def with_worker_index(self, worker_index: int) -> "AttestedPrivateModelRunnerV2":
        return AttestedPrivateModelRunnerV2(
            artifact=self.artifact,
            spec=self.spec,
            model_kind=self.model_kind,
            worker_index=int(worker_index),
            epoch_id=self.epoch_id,
            parent_graphs=self.parent_graphs,
            execute=self._execute,
            catalog_snapshot_loader=self._catalog_snapshot_loader,
            _shared_state=self._shared_state,
        )

    def attested_receipts(self) -> list[dict[str, Any]]:
        with self._shared_state["lock"]:
            return [dict(item) for item in self._shared_state["receipts"]]

    def attested_authorities(self) -> list[dict[str, Any]]:
        with self._shared_state["lock"]:
            return [dict(item) for item in self._shared_state["authorities"]]

    def _retain_attested_authority(self, authority: Mapping[str, Any]) -> str:
        """Retain one already-verified success or failure graph by its root.

        Failed enclave jobs are still authoritative observations.  Their root
        must reach the per-attempt checkpoint instead of disappearing when the
        public runner interface raises ``PrivateModelRuntimeError``.
        """

        outcome = dict(authority)
        graph = outcome.get("receipt_graph")
        receipt = outcome.get("receipt")
        if not isinstance(receipt, Mapping):
            return ""
        receipt_hash = str(receipt.get("receipt_hash") or "").lower()
        graph_root = (
            str(graph.get("root_receipt_hash") or "").lower()
            if isinstance(graph, Mapping)
            else (
                receipt_hash
                if getattr(self, "_execute", execute_scoring_v2)
                is not execute_scoring_v2
                else ""
            )
        )
        if (
            not _SHA256_RE.fullmatch(receipt_hash)
            or graph_root != receipt_hash
        ):
            return ""
        with self._shared_state["lock"]:
            receipts = self._shared_state["receipts"]
            if not any(
                item.get("receipt_hash") == receipt_hash for item in receipts
            ):
                receipts.append(dict(receipt))
            authorities = self._shared_state["authorities"]
            if not any(
                str(
                    dict(item.get("receipt_graph") or {}).get(
                        "root_receipt_hash"
                    )
                    or dict(item.get("receipt") or {}).get("receipt_hash")
                    or ""
                ).lower()
                == receipt_hash
                for item in authorities
                if isinstance(item, Mapping)
            ):
                authorities.append(outcome)
        publish_attested_receipt_hash(receipt_hash)
        return receipt_hash

    def measured_compatibility_admission(self) -> dict[str, Any]:
        with self._shared_state["lock"]:
            admissions = self._shared_state.get("compatibility_admissions") or []
            if not admissions:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured model compatibility admission is unavailable"
                )
            return deepcopy(admissions[-1])

    async def _load_catalog_snapshot(self, *, epoch_id: int) -> Mapping[str, Any]:
        """Load one exact measured catalog outcome per shared runner epoch."""

        with self._shared_state["lock"]:
            futures = self._shared_state.setdefault("catalog_snapshot_futures", {})
            future = futures.get(epoch_id)
            leader = future is None
            if leader:
                future = concurrent.futures.Future()
                futures[epoch_id] = future

        if leader:
            catalog_loader = (
                self._catalog_snapshot_loader or load_source_add_catalog_snapshot_v2
            )
            try:
                outcome = await catalog_loader(epoch_id=epoch_id)
            except BaseException as exc:
                future.set_exception(exc)
                # Mark the exception observed for the leader-only case while
                # preserving it for any followers already awaiting this future.
                future.exception()
                with self._shared_state["lock"]:
                    if futures.get(epoch_id) is future:
                        del futures[epoch_id]
                raise
            future.set_result(outcome)
            return outcome

        return await asyncio.shield(asyncio.wrap_future(future))

    async def __call__(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        canonical_icp = canonicalize_private_model_icp(icp)
        cache_ref = _provider_evidence_cache_ref(canonical_icp)
        cache_document = _provider_evidence_cache(
            self.spec,
            canonical_icp=canonical_icp,
        )
        cache_parent_graphs = ()
        if cache_document:
            cache_parent_graphs = await _load_provider_evidence_tape_graphs(
                cache_ref=cache_ref,
                cache_hash=sha256_json(cache_document),
            )
        run_mode = str(dict(context or {}).get("mode") or "")
        evidence_mode = (
            "record"
            if self.model_kind == "private" and run_mode == "private_baseline"
            else "cache_live"
            if cache_document
            else "live"
        )
        result = await self._invoke_operation(
            operation="run_icp",
            input_doc={
                "icp": canonical_icp,
                "context": context_with_runtime_options(
                    context,
                    outer_timeout_seconds=self.spec.timeout_seconds,
                ),
            },
            provider_evidence_cache=cache_document,
            provider_evidence_cache_ref=cache_ref,
            provider_evidence_mode=evidence_mode,
            provider_snapshot_bundle={},
            provider_snapshot_tree_hash="",
            provider_snapshot_manifest_hash="",
            provider_cost_scope_override="",
            provider_cost_cap_microusd=0,
            provider_call_cap=0,
            publish_provider_evidence_cache=True,
            additional_parent_graphs=cache_parent_graphs,
        )
        if isinstance(result, QualificationOutcomeCompleteV2):
            return result
        return list(
            ensure_private_model_outputs(
                result,
                context_label="V2 measured private model",
                require_non_empty=False,
            )
        )

    async def run_with_provider_evidence(
        self,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        provider_evidence_cache: Mapping[str, Any],
        provider_evidence_mode: str,
        cache_parent_graphs: Sequence[Mapping[str, Any]] = (),
        provider_snapshot_bundle: Mapping[str, Any] | None = None,
        provider_snapshot_tree_hash: str = "",
        provider_snapshot_manifest_hash: str = "",
        provider_cost_scope: str = "",
        provider_cost_cap_microusd: int = 0,
        provider_call_cap: int = 0,
    ) -> list[Mapping[str, Any]]:
        """Run one ICP under an explicitly committed tree-evaluation tape mode."""

        canonical_icp = canonicalize_private_model_icp(icp)
        result = await self._invoke_operation(
            operation="run_icp",
            input_doc={
                "icp": canonical_icp,
                "context": context_with_runtime_options(
                    context,
                    outer_timeout_seconds=self.spec.timeout_seconds,
                ),
            },
            provider_evidence_cache=dict(provider_evidence_cache),
            provider_evidence_cache_ref=_provider_evidence_cache_ref(canonical_icp),
            provider_evidence_mode=str(provider_evidence_mode),
            provider_snapshot_bundle=dict(provider_snapshot_bundle or {}),
            provider_snapshot_tree_hash=str(provider_snapshot_tree_hash or ""),
            provider_snapshot_manifest_hash=str(
                provider_snapshot_manifest_hash or ""
            ),
            provider_cost_scope_override=str(provider_cost_scope or ""),
            provider_cost_cap_microusd=int(provider_cost_cap_microusd),
            provider_call_cap=int(provider_call_cap),
            publish_provider_evidence_cache=False,
            additional_parent_graphs=cache_parent_graphs,
        )
        if isinstance(result, QualificationOutcomeCompleteV2):
            return result
        return list(
            ensure_private_model_outputs(
                result,
                context_label="V2 measured tree evaluation",
                require_non_empty=False,
            )
        )

    def generated_provider_evidence_cache(
        self, cache_ref: str
    ) -> dict[str, Any]:
        with self._shared_state["lock"]:
            return dict(
                self._shared_state.get("generated_caches", {}).get(cache_ref) or {}
            )

    def provider_evidence_summary(self, cache_ref: str) -> dict[str, Any]:
        with self._shared_state["lock"]:
            return dict(
                self._shared_state.get("evidence_summaries", {}).get(cache_ref)
                or {}
            )

    def metadata(self) -> Mapping[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # ``_execute_operation`` validates measured metadata against the
            # exact semantic bindings in the host compatibility receipt before
            # returning it.  A second legacy-only validation here would reject
            # compatible future adapter release identities after that stronger
            # validation already succeeded.
            return dict(
                asyncio.run(
                    self._invoke_operation(
                        operation="metadata",
                        input_doc={},
                        provider_evidence_cache={},
                        provider_evidence_cache_ref="",
                        provider_evidence_mode="",
                        provider_snapshot_bundle={},
                        provider_snapshot_tree_hash="",
                        provider_snapshot_manifest_hash="",
                        provider_cost_scope_override="",
                        provider_cost_cap_microusd=0,
                        provider_call_cap=0,
                        publish_provider_evidence_cache=False,
                    )
                )
            )
        raise AttestedPrivateModelRunnerV2Error(
            "synchronous model metadata cannot run on an active event loop"
        )

    async def _invoke_operation(self, **kwargs: Any) -> Any:
        """Keep measured bridge failures inside the model-runner contract."""

        try:
            return await asyncio.wait_for(
                self._execute_operation(**kwargs),
                timeout=_model_invocation_timeout_seconds(
                    self.spec.timeout_seconds
                ),
            )
        except asyncio.TimeoutError as exc:
            raise AttestedPrivateModelRunnerV2Error(
                "measured model invocation timed out"
            ) from exc
        except AttestedScoringV2Error as exc:
            message = str(exc)
            if has_retryable_attested_provider_transport_failure(exc):
                message = "%s; %s" % (
                    message,
                    RETRYABLE_ATTESTED_PROVIDER_TRANSPORT_MARKER,
                )
            authority = exc.authority
            if isinstance(authority, Mapping):
                self._retain_attested_authority(authority)
            raise AttestedPrivateModelRunnerV2Error(
                message,
                authority=authority,
            ) from exc
        except TEEArtifactStoreV2Error as exc:
            message = str(exc)
            if any(
                marker in message.lower()
                for marker in _RETRYABLE_ARTIFACT_PERSISTENCE_FAILURE_MARKERS
            ):
                message = "%s; %s" % (
                    message,
                    RETRYABLE_ATTESTED_ARTIFACT_PERSISTENCE_MARKER,
                )
            raise AttestedPrivateModelRunnerV2Error(message) from exc

    async def _execute_operation(
        self,
        *,
        operation: str,
        input_doc: Mapping[str, Any],
        provider_evidence_cache: Mapping[str, Any],
        provider_evidence_cache_ref: str,
        provider_evidence_mode: str,
        provider_snapshot_bundle: Mapping[str, Any],
        provider_snapshot_tree_hash: str,
        provider_snapshot_manifest_hash: str,
        provider_cost_scope_override: str,
        provider_cost_cap_microusd: int,
        provider_call_cap: int,
        publish_provider_evidence_cache: bool,
        additional_parent_graphs: Sequence[Mapping[str, Any]] = (),
    ) -> Any:
        source_bundle, compatibility_receipt = await asyncio.to_thread(
            _source_bundle_and_compatibility_receipt_for_artifact,
            self.artifact,
            timeout_seconds=self.spec.timeout_seconds,
        )
        callable_name = (
            self.spec.callable_name if operation == "run_icp" else "adapter_metadata"
        )
        argv = [self.spec.module_name, callable_name]
        scope_doc: dict[str, Any] = {
            "image_digest": self.spec.image_digest,
            "argv": argv,
            "stdin_payload": dict(input_doc),
        }
        evaluation_scope = str(
            dict(self.spec.extra_env or {}).get(
                PROVIDER_COST_EVALUATION_SCOPE_ENV,
                "",
            )
            or ""
        ).strip()
        if evaluation_scope:
            scope_doc["evaluation_scope"] = evaluation_scope
        metadata_operation = operation == "metadata"
        if metadata_operation and (
            input_doc
            or provider_evidence_cache
            or provider_evidence_cache_ref
            or provider_evidence_mode
            or provider_snapshot_bundle
            or provider_snapshot_tree_hash
            or provider_snapshot_manifest_hash
            or provider_cost_scope_override
            or provider_cost_cap_microusd
            or provider_call_cap
            or publish_provider_evidence_cache
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "model compatibility metadata inputs are not isolated"
            )
        provider_cost_scope = (
            ""
            if metadata_operation
            else str(provider_cost_scope_override or "") or sha256_json(scope_doc)
        )
        cache_hash = sha256_json(dict(provider_evidence_cache))
        image_hash = "sha256:" + self.artifact.image_digest.rsplit("@sha256:", 1)[1]
        execution_epoch = (
            self.epoch_id
            if self.epoch_id is not None
            else int(
                dict(input_doc.get("context") or {}).get("evaluation_epoch") or 0
            )
        )
        if metadata_operation:
            runtime_catalog: dict[str, Any] = {}
            provider_catalog_evidence: dict[str, Any] = {}
            provisioned_sources: list[Mapping[str, Any]] = []
            dynamic_provider_refs: dict[str, str] = {}
            model_environment: dict[str, str] = {}
            catalog_parent_graphs: tuple[Mapping[str, Any], ...] = ()
            catalog_input_hashes: tuple[str, ...] = ()
            purpose = MODEL_COMPATIBILITY_PURPOSE_V2
            provider_profile = "default"
            provider_profile_loader = _empty_model_compatibility_provider_profile
            envelope_builder = None
            require_egress_proxy = False
        else:
            catalog_outcome = await self._load_catalog_snapshot(
                epoch_id=int(execution_epoch)
            )
            catalog_result = catalog_outcome.get("result")
            catalog_graph = catalog_outcome.get("receipt_graph")
            catalog_lineage_receipt = catalog_outcome.get("receipt")
            catalog_execution_graph = catalog_outcome.get(
                "execution_receipt_graph"
            ) or catalog_outcome.get("receipt_graph")
            catalog_execution_receipt = catalog_outcome.get(
                "execution_receipt"
            ) or catalog_outcome.get("receipt")
            if (
                not isinstance(catalog_result, Mapping)
                or not isinstance(catalog_graph, Mapping)
                or not isinstance(catalog_lineage_receipt, Mapping)
                or not isinstance(catalog_execution_graph, Mapping)
                or not isinstance(catalog_execution_receipt, Mapping)
                or catalog_graph.get("root_receipt_hash")
                != catalog_lineage_receipt.get("receipt_hash")
                or catalog_execution_graph.get("root_receipt_hash")
                != catalog_execution_receipt.get("receipt_hash")
            ):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured SOURCE_ADD catalog authority is unavailable"
                )
            provisioned_sources = catalog_result.get("provisioned_sources")
            private_registry_rows = catalog_result.get("private_registry_rows")
            if (
                catalog_result.get("schema_version")
                != "leadpoet.source_add_catalog_snapshot.v2"
                or not isinstance(provisioned_sources, list)
                or any(
                    not isinstance(item, Mapping)
                    for item in provisioned_sources
                )
                or not isinstance(private_registry_rows, list)
                or any(
                    not isinstance(item, Mapping)
                    for item in private_registry_rows
                )
            ):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured SOURCE_ADD catalog result is invalid"
                )
            try:
                runtime_catalog = validate_source_add_runtime_catalog_v2(
                    catalog_result.get("runtime_catalog") or {}
                )
                derived_runtime_catalog = build_source_add_runtime_catalog_v2(
                    [dict(item) for item in provisioned_sources]
                )
            except Exception as exc:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured SOURCE_ADD runtime catalog is invalid"
                ) from exc
            catalog_root = str(catalog_graph.get("root_receipt_hash") or "")
            catalog_execution_root = str(
                catalog_execution_graph.get("root_receipt_hash") or ""
            )
            if (
                runtime_catalog != derived_runtime_catalog
                or catalog_result.get("provisioned_sources_hash")
                != sha256_json([dict(item) for item in provisioned_sources])
                or catalog_result.get("private_registry_rows_hash")
                != sha256_json([dict(item) for item in private_registry_rows])
                or catalog_result.get("runtime_catalog_hash")
                != runtime_catalog["catalog_hash"]
                or catalog_execution_receipt.get("role")
                != "gateway_coordinator"
                or catalog_execution_receipt.get("purpose")
                != "research_lab.source_add_catalog_snapshot.v2"
                or catalog_execution_receipt.get("status") != "succeeded"
                or catalog_execution_receipt.get("output_root")
                != sha256_json(dict(catalog_result))
            ):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured SOURCE_ADD catalog commitment differs"
                )
            dynamic_provider_refs = source_add_runtime_credential_refs_v2(
                runtime_catalog
            )
            dynamic_credential_env_names = tuple(
                str(env_name)
                for route in runtime_catalog["routes"]
                for env_name in route["credential_env_refs"]
            )
            model_environment = _measured_environment_for_provider_cost_scope(
                self.spec,
                provider_cost_scope=provider_cost_scope,
                additional_credential_env_names=dynamic_credential_env_names,
            )
            purpose = (
                "research_lab.private_model_run.v2"
                if self.model_kind == "private"
                else "research_lab.candidate_hybrid_discovery.v2"
                if provider_evidence_mode == "record"
                else "research_lab.candidate_model_run.v2"
            )
            provider_catalog_evidence = {
                "result": dict(catalog_result),
                "root_receipt_hash": catalog_execution_root,
            }
            catalog_parent_graphs = (
                catalog_execution_graph,
                catalog_graph,
            )
            catalog_input_hashes = (
                catalog_root,
                catalog_execution_root,
                str(catalog_result["provisioned_sources_hash"]),
                str(catalog_result["private_registry_rows_hash"]),
                str(catalog_result["runtime_catalog_hash"]),
                *dynamic_provider_refs.values(),
            )
            provider_profile = str(
                dict(self.spec.extra_env or {}).get(
                    V2_PROVIDER_PROFILE_ENV,
                    "default",
                )
                or "default"
            )
            provider_profile_loader = None
            envelope_builder = lambda job_id: [
                envelope
                for source_row in provisioned_sources
                for envelope in (
                    build_source_add_job_envelope_v2(
                        source_row,
                        job_id=job_id,
                    ),
                )
                if envelope is not None
            ]
            require_egress_proxy = None
        with self._shared_state["lock"]:
            sequence = int(self._shared_state["sequence"])
            self._shared_state["sequence"] = sequence + 1
        parent_graph_by_root = {
            str(graph.get("root_receipt_hash") or ""): dict(graph)
            for graph in (
                *self.parent_graphs,
                *additional_parent_graphs,
                *catalog_parent_graphs,
            )
        }
        if "" in parent_graph_by_root:
            raise AttestedPrivateModelRunnerV2Error(
                "model authority parent graph root is missing"
            )
        outcome = await self._execute(
            operation=OP_RUN_MODEL_SANDBOX_V2,
            purpose=purpose,
            epoch_id=int(execution_epoch),
            sequence=sequence,
            payload={
                "schema_version": MODEL_SANDBOX_REQUEST_SCHEMA_VERSION,
                "model_kind": self.model_kind,
                "operation": operation,
                "artifact": self.artifact.to_dict(),
                "source_bundle": source_bundle,
                "module_name": self.spec.module_name,
                "callable_name": callable_name,
                "input": dict(input_doc),
                "environment": model_environment,
                "provider_evidence_cache": dict(provider_evidence_cache),
                "provider_evidence_cache_ref": provider_evidence_cache_ref,
                "provider_evidence_mode": provider_evidence_mode,
                "provider_snapshot_bundle": dict(provider_snapshot_bundle),
                "provider_snapshot_tree_hash": provider_snapshot_tree_hash,
                "provider_snapshot_manifest_hash": provider_snapshot_manifest_hash,
                "provider_cost_scope": provider_cost_scope,
                "provider_cost_cap_microusd": int(provider_cost_cap_microusd),
                "provider_call_cap": int(provider_call_cap),
                "provider_runtime_catalog": runtime_catalog,
                "provider_catalog_evidence": provider_catalog_evidence,
            },
            worker_index=self.worker_index,
            provider_credential_profile=provider_profile,
            provider_credential_ref_hashes=dynamic_provider_refs,
            provider_profile_loader=provider_profile_loader,
            require_egress_proxy=require_egress_proxy,
            additional_job_credential_envelope_builder=envelope_builder,
            parent_graphs=tuple(
                parent_graph_by_root[key] for key in sorted(parent_graph_by_root)
            ),
            input_artifact_hashes=(
                self.artifact.model_artifact_hash,
                self.artifact.manifest_hash,
                image_hash,
                str(source_bundle["archive_sha256"]),
                str(compatibility_receipt["policy_hash"]),
                str(compatibility_receipt["receipt_hash"]),
                cache_hash,
                *(
                    (
                        str(provider_snapshot_bundle["archive_sha256"]),
                        provider_snapshot_tree_hash,
                        provider_snapshot_manifest_hash,
                    )
                    if provider_snapshot_bundle
                    else ()
                ),
                *catalog_input_hashes,
            ),
            timeout_seconds=max(1.0, float(self.spec.timeout_seconds) + 120.0),
        )
        result = outcome.get("result")
        if not isinstance(result, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model result is missing"
            )
        expected = {
            "schema_version": "leadpoet.model_sandbox_result.v2",
            "model_kind": self.model_kind,
            "operation": operation,
            "model_artifact_hash": self.artifact.model_artifact_hash,
            "model_manifest_hash": self.artifact.manifest_hash,
            "compatibility_image_digest": self.artifact.image_digest,
            "source_bundle_hash": source_bundle["archive_sha256"],
            "compatibility_policy_hash": compatibility_receipt["policy_hash"],
            "compatibility_admission_hash": compatibility_receipt[
                "receipt_hash"
            ],
            "input_hash": sha256_json(dict(input_doc)),
            "provider_evidence_cache_hash": cache_hash,
            "provider_evidence_cache_ref": provider_evidence_cache_ref,
            "provider_evidence_mode": provider_evidence_mode,
            "provider_snapshot_archive_hash": (
                str(provider_snapshot_bundle.get("archive_sha256") or "")
                if provider_snapshot_bundle
                else sha256_json({})
            ),
            "provider_snapshot_tree_hash": (
                provider_snapshot_tree_hash or sha256_json({})
            ),
            "provider_snapshot_manifest_hash": (
                provider_snapshot_manifest_hash or sha256_json({})
            ),
            "provider_cost_cap_microusd": int(provider_cost_cap_microusd),
            "provider_call_cap": int(provider_call_cap),
            "provider_runtime_catalog_hash": (
                sha256_json({})
                if metadata_operation
                else runtime_catalog["catalog_hash"]
            ),
        }
        if any(result.get(name) != value for name, value in expected.items()):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model result commitments differ"
            )
        consumer_runtime_probe_hash = ""
        adapter_metadata_hash = ""
        if operation == "metadata":
            probe = result.get("consumer_runtime_probe")
            probe_hash = str(result.get("consumer_runtime_probe_hash") or "")
            if (
                not isinstance(probe, Mapping)
                or probe_hash != sha256_json(dict(probe))
                or not isinstance(result.get("output"), Mapping)
            ):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured consumer runtime probe commitment differs"
                )
            try:
                measured_metadata = validate_sourcing_adapter_metadata(
                    result["output"],
                    expected_semantic_bindings=dict(
                        compatibility_receipt.get("bindings") or {}
                    ),
                    require_company_fit_contract=(
                        compatibility_receipt.get("admission_mode")
                        == "semantic_v1"
                    ),
                )
                validate_consumer_runtime_probe_v1(
                    probe,
                    compatibility_receipt=compatibility_receipt,
                    metadata=measured_metadata,
                    expected_source_tree_hash=self.artifact.model_artifact_hash,
                    expected_manifest_hash=self.artifact.manifest_hash,
                    expected_image_digest=self.artifact.image_digest,
                    expected_module_name=self.spec.module_name,
                    expected_callable_name="adapter_metadata",
                )
            except (ModelSandboxV2Error, PrivateModelRuntimeError) as exc:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured consumer runtime probe differs from host admission"
                ) from exc
            consumer_runtime_probe_hash = probe_hash
            adapter_metadata_hash = sha256_json(dict(measured_metadata))
        trace_entries = result.get("trace_entries")
        if not isinstance(trace_entries, list) or sha256_json(trace_entries) != result.get(
            "trace_entries_hash"
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model trace commitment differs"
            )
        if operation == "run_icp":
            try:
                validate_sourcing_runtime_receipt_entries(
                    trace_entries,
                    expected_runtime_options=dict(
                        input_doc.get("context") or {}
                    )["runtime_options"],
                )
            except (KeyError, PrivateModelRuntimeError) as exc:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured model sourcing runtime receipt is invalid"
                ) from exc
        cost_summary = summarize_provider_cost_trace_entries(trace_entries)
        if result.get("output_hash") != sha256_json(result.get("output")):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model output commitment differs"
            )
        qualification_envelope: dict[str, Any] | None = None
        if (
            operation == "run_icp"
            and compatibility_receipt.get("admission_mode")
            == "qualification_protocol_v2"
        ):
            try:
                qualification_envelope = (
                    validate_qualification_outcome_envelope_v2(
                        result.get("output")
                    )
                )
            except PrivateModelRuntimeError as exc:
                raise AttestedPrivateModelRunnerV2Error(
                    "measured model qualification outcome is invalid"
                ) from exc
        generated_cache = result.get("generated_provider_evidence_cache")
        generated_cache_hash = str(
            result.get("generated_provider_evidence_cache_hash") or ""
        )
        if not isinstance(generated_cache, Mapping) or generated_cache_hash != sha256_json(
            dict(generated_cache)
        ):
            raise AttestedPrivateModelRunnerV2Error(
                "measured provider evidence tape commitment differs"
            )
        publish_incontainer_trace_entries(trace_entries)
        receipt = outcome.get("receipt")
        if not isinstance(receipt, Mapping):
            raise AttestedPrivateModelRunnerV2Error(
                "measured model receipt is missing"
            )
        measured_compatibility_admission = None
        if operation == "metadata":
            measured_admission_body = {
                "schema_version": MEASURED_COMPATIBILITY_ADMISSION_SCHEMA_V1,
                "decision": SEMANTIC_COMPATIBILITY_ACCEPTED_DECISION,
                "admission_mode": compatibility_receipt["admission_mode"],
                "consumer_api_version": compatibility_receipt[
                    "consumer_api_version"
                ],
                "compatibility_policy_hash": compatibility_receipt["policy_hash"],
                "compatibility_admission_hash": compatibility_receipt[
                    "receipt_hash"
                ],
                "source_tree_hash": self.artifact.model_artifact_hash,
                "manifest_hash": self.artifact.manifest_hash,
                "image_digest": self.artifact.image_digest,
                "module_name": self.spec.module_name,
                "callable_name": "adapter_metadata",
                "consumer_runtime_probe_hash": consumer_runtime_probe_hash,
                "adapter_metadata_hash": adapter_metadata_hash,
                "execution_receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
            measured_compatibility_admission = {
                **measured_admission_body,
                "receipt_hash": sha256_json(measured_admission_body),
            }
        if generated_cache:
            graph = outcome.get("execution_receipt_graph")
            if not isinstance(graph, Mapping):
                raise AttestedPrivateModelRunnerV2Error(
                    "measured provider evidence tape graph is missing"
                )
            _require_tape_receipt(
                graph,
                cache_ref=provider_evidence_cache_ref,
                cache_hash=generated_cache_hash,
            )
            await _persist_provider_evidence_tape_link(
                receipt_hash=str(receipt.get("receipt_hash") or ""),
                cache_ref=provider_evidence_cache_ref,
                cache_hash=generated_cache_hash,
            )
            if publish_provider_evidence_cache:
                _write_provider_evidence_cache(
                    cache_ref=provider_evidence_cache_ref,
                    cache_document=generated_cache,
                )
            with self._shared_state["lock"]:
                self._shared_state.setdefault("generated_caches", {})[
                    provider_evidence_cache_ref
                ] = dict(generated_cache)
                self._shared_state.setdefault("evidence_summaries", {})[
                    provider_evidence_cache_ref
                ] = {
                    "cache_hash": generated_cache_hash,
                    "trace_entries_hash": str(result.get("trace_entries_hash") or ""),
                    "cost_summary": dict(cost_summary),
                }
        model_qualification_authority: dict[str, Any] | None = None
        retained_outcome = dict(outcome)
        if qualification_envelope is not None:
            model_qualification_authority = _model_qualification_authority_v1(
                envelope=qualification_envelope,
                input_doc=input_doc,
                sandbox_result=result,
                outcome=outcome,
                artifact=self.artifact,
            )
            retained_outcome["model_qualification_authority"] = deepcopy(
                model_qualification_authority
            )
        receipt_hash = self._retain_attested_authority(retained_outcome)
        if not receipt_hash:
            raise AttestedPrivateModelRunnerV2Error(
                "measured model receipt authority root differs"
            )
        with self._shared_state["lock"]:
            if measured_compatibility_admission is not None:
                admissions = self._shared_state.setdefault(
                    "compatibility_admissions", []
                )
                if not any(
                    item.get("receipt_hash")
                    == measured_compatibility_admission["receipt_hash"]
                    for item in admissions
                ):
                    admissions.append(deepcopy(measured_compatibility_admission))
        if qualification_envelope is not None:
            companies = list(qualification_envelope["companies"])
            if qualification_envelope["completion_state"] == "incomplete":
                raise QualificationOutcomeIncompleteV2Error(
                    "model qualification outcome is incomplete",
                    model_qualification_authority=(
                        model_qualification_authority or {}
                    ),
                    partial_companies=companies,
                    authority=retained_outcome,
                )
            return QualificationOutcomeCompleteV2(
                companies,
                model_qualification_authority=(
                    model_qualification_authority or {}
                ),
            )
        return result.get("output")


def retry_attested_model_runner_v2(
    runner: AttestedPrivateModelRunnerV2,
    *,
    extra_env: Mapping[str, str],
) -> AttestedPrivateModelRunnerV2:
    return runner.with_spec(
        replace(
            runner.spec,
            extra_env=dict(extra_env),
            pull_before_run=False,
        )
    )
