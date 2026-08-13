"""Runtime helpers for private Research Lab model artifacts.

The public subnet repo must not import private champion code directly. These
helpers load a private model adapter from an immutable artifact checkout/image
boundary and execute it through a small JSON contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import contextvars
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_bytes, sha256_json
from research_lab.employee_buckets import (
    normalize_employee_count_bucket as _normalize_linkedin_employee_count_bucket,
    normalize_employee_count_buckets,
)
from research_lab.sourcing_model_contract_check import (
    resolve_reviewed_consumer_snapshot,
    reviewed_consumer_snapshots,
)

logger = logging.getLogger(__name__)

SECRET_MARKERS = (
    "sk-or-",
    "sb_secret_",
    "aws_secret_access_key",
    "openrouter_api_key",
    "openrouter_management_key",
    "scrapingdog_api_key",
    "exa_api_key",
    "deepline_api_key",
    "raw_secret",
    "service_role",
)
PROVIDER_ERROR_MARKER = "research_lab_private_runtime_provider_error"
# In-container trace capture (training-data tee): the diagnostics bootstrap
# emits one single-line marker per hooked HTTP call (success AND failure) so
# the host can reconstruct the sourcing model's own LLM/search/fetch behavior.
# Pure observation: emission never changes what the hooked calls return, and
# any capture failure is swallowed.
INCONTAINER_TRACE_MARKER = "research_lab_private_runtime_trace"
SOURCING_BRANCH_RECEIPT_MARKER = "sourcing_branch_receipt"
SOURCING_RUNTIME_EVENT_MARKER = "sourcing_runtime_event"
INCONTAINER_TRACE_CAPTURE_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE"
INCONTAINER_TRACE_MAX_CALL_BYTES_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_MAX_CALL_BYTES"
INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV = "RESEARCH_LAB_INCONTAINER_TRACE_MAX_TOTAL_BYTES"
INCONTAINER_TRACE_DEFAULT_MAX_CALL_BYTES = 16384
INCONTAINER_TRACE_DEFAULT_MAX_TOTAL_BYTES = 524288
# P13 corpus-mode budgets: the small diagnostics caps above silently truncate
# real LLM answers with tool-call JSON and Exa/scrape payloads. When an S3
# persistence prefix is configured (the traces are corpus data, not just
# diagnostics), these larger defaults apply unless the operator set the caps
# explicitly.
INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES = 262144  # 256 KiB per call body
INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES = 16777216  # 16 MiB per run
# Forwarded into model containers so operators can disable/tune capture
# per-fleet. Capture defaults ON when the env is absent, so forwarding only
# matters for opting out or resizing the caps.
INCONTAINER_TRACE_ENV_PASSTHROUGH = (
    INCONTAINER_TRACE_CAPTURE_ENV,
    INCONTAINER_TRACE_MAX_CALL_BYTES_ENV,
    INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV,
)
PROVIDER_COST_ENV_PASSTHROUGH = (
    "RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP",
    "RESEARCH_LAB_SCRAPINGDOG_COST_PER_CREDIT_USD",
    "RESEARCH_LAB_SCRAPINGDOG_UNKNOWN_ENDPOINT_CREDITS",
    "RESEARCH_LAB_PROVIDER_COST_UNKNOWN_ENDPOINT_POLICY",
    "RESEARCH_LAB_PROVIDER_COST_SOFT_STOP",
)
PROVIDER_COST_EVALUATION_SCOPE_ENV = "RESEARCH_LAB_PROVIDER_COST_EVALUATION_SCOPE"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
SOURCING_MODEL_MAX_RUNTIME_CAP_SECONDS = 1500.0
SOURCING_MODEL_MAX_AGENT_TIMEOUT_SECONDS = 900
# Match the adapter's result/receipt finalization window before sandbox kill.
SOURCING_MODEL_MAX_FINALIZATION_RESERVE_SECONDS = 60.0
EXPECTED_SOURCING_ADAPTER_VERSIONS = frozenset(
    str(snapshot["contract"]["exact_constants"]["research_lab_adapter.py"]["ADAPTER_VERSION"])
    for snapshot in reviewed_consumer_snapshots().values()
)
EXPECTED_COMPONENT_REGISTRY_VERSION = "sourcing-model-components:v2"
EXPECTED_ROUTING_COMPILER_VERSION = "routing-compiler-v2"
REQUIRED_RUNTIME_CANDIDATE_TOOLS = {
    "candidate.backlog",
    "candidate.registry_feed",
    "candidate.jobs_feed",
    "candidate.deepline_firmographic",
    "candidate.model_semantic",
}
REQUIRED_RUNTIME_INTENT_TOOLS = {
    "intent.existing_evidence",
    "intent.jobs_feed",
    "intent.company_search",
    "intent.first_party",
    "intent.newsroom",
}
LAB_CONSUMER_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "sourcing_model_contract.json"
)
LAB_CONSUMER_PARITY_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sourcing_model_parity_fixtures.json"
)
_LAB_CONSUMER_CONTRACT = json.loads(
    LAB_CONSUMER_CONTRACT_PATH.read_text(encoding="utf-8")
)
EXPECTED_CONSUMER_CONTRACT_ID = str(
    _LAB_CONSUMER_CONTRACT["contract_id"]
)
EXPECTED_CONSUMER_CONTRACT_PATH = str(
    _LAB_CONSUMER_CONTRACT["canonical_path"]
)
EXPECTED_CONSUMER_PARITY_FIXTURE_PATH = str(
    _LAB_CONSUMER_CONTRACT["parity_fixture_path"]
)
_REVIEWED_CONSUMER_MANIFEST_PAIRS = tuple(
    (
        {
            "contract_id": contract_id,
            "path": str(snapshot["contract"]["canonical_path"]),
            "sha256": str(snapshot["contract_sha256"]),
        },
        {
            "path": str(snapshot["contract"]["parity_fixture_path"]),
            "sha256": str(snapshot["parity_sha256"]),
        },
    )
    for contract_id, snapshot in sorted(reviewed_consumer_snapshots().items())
)
DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID = (
    "alias/leadpoet-research-lab-artifact-signing"
)
DEFAULT_ENV_PASSTHROUGH = (
    "EXA_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "DEEPLINE_API_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
) + INCONTAINER_TRACE_ENV_PASSTHROUGH + PROVIDER_COST_ENV_PASSTHROUGH
PROVIDER_KEY_ENV_PASSTHROUGH = (
    "EXA_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "DEEPLINE_API_KEY",
)
PROVIDER_PROXY_ENV_PASSTHROUGH = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

class PrivateModelRuntimeError(RuntimeError):
    """Raised when the private model artifact cannot be executed safely."""


def private_model_env_passthrough(*, include_proxy: bool = False) -> tuple[str, ...]:
    """Provider env names for private model containers.

    Worker processes may carry global Webshare proxy vars for their own network
    path. Do not implicitly forward those into provider-backed model containers:
    Exa, ScrapingDog, and OpenRouter are API services, and ScrapingDog performs
    its own upstream proxying. Operators can opt in when testing a proxy-specific
    provider path.

    The in-container trace-capture controls always ride along so operators can
    disable or resize the capture per-fleet without a code change.
    """
    if include_proxy:
        return PROVIDER_KEY_ENV_PASSTHROUGH + PROVIDER_PROXY_ENV_PASSTHROUGH + INCONTAINER_TRACE_ENV_PASSTHROUGH
    return PROVIDER_KEY_ENV_PASSTHROUGH + INCONTAINER_TRACE_ENV_PASSTHROUGH


def incontainer_trace_corpus_env() -> dict[str, str]:
    """P13: container env cap overrides for corpus-mode capture.

    Returns the larger corpus byte budgets when an in-container persistence
    prefix is configured and the operator has not explicitly set the caps;
    empty otherwise (diagnostics mode keeps the small defaults)."""
    prefix = str(os.getenv("RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX") or "").strip()
    if not prefix:
        return {}
    env: dict[str, str] = {}
    if not str(os.getenv(INCONTAINER_TRACE_MAX_CALL_BYTES_ENV) or "").strip():
        env[INCONTAINER_TRACE_MAX_CALL_BYTES_ENV] = str(
            INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES
        )
    if not str(os.getenv(INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV) or "").strip():
        env[INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV] = str(
            INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES
        )
    return env


def _default_docker_platform() -> str:
    return os.getenv("RESEARCH_LAB_PRIVATE_MODEL_DOCKER_PLATFORM", DEFAULT_DOCKER_PLATFORM).strip() or DEFAULT_DOCKER_PLATFORM


def canonicalize_private_model_icp(icp: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt Research Lab ICPs to the private sourcing-model contract.

    The private sourcing model is intentionally kept outside this public repo,
    but its stable adapter expects a canonical ICP shape with
    ``required_attribute``, ``intent_signal``, ``intent_category``,
    ``geography``, and ``employee_count``. Research Lab benchmark rows already
    contain the same information, but may use public qualification-era names
    such as ``product_service``, ``intent_signals``, ``target_geography``, or
    ``company_size``. Normalize only at the execution boundary so benchmark
    hashes and persisted private ICP rows remain unchanged.
    """
    if not isinstance(icp, Mapping):
        raise PrivateModelRuntimeError("private model ICP payload must be an object")
    normalized = dict(icp)

    industry = _first_text(normalized.get("industry"), normalized.get("market"))
    sub_industry = _first_text(normalized.get("sub_industry"), normalized.get("subindustry"))
    product_service = _first_text(
        normalized.get("product_service"),
        normalized.get("required_attribute"),
        normalized.get("solution"),
        normalized.get("offering"),
    )
    geography = _first_text(
        normalized.get("geography"),
        normalized.get("country"),
        normalized.get("target_geography"),
        normalized.get("hq_country"),
        default="United States",
    )
    raw_employee_count = normalized.get("employee_count")
    employee_count = _normalize_employee_count_bucket(
        _first_text(
            raw_employee_count,
            normalized.get("company_size"),
            normalized.get("company_size_bucket"),
            normalized.get("employee_range"),
            default="51-200",
        )
    )
    intent_signal = _intent_signal_text(normalized)
    required_attribute = _required_attribute(
        normalized.get("required_attribute"),
        industry=industry,
        sub_industry=sub_industry,
        product_service=product_service,
    )

    normalized["industry"] = industry
    normalized["sub_industry"] = sub_industry
    normalized["geography"] = geography
    normalized["country"] = _first_text(normalized.get("country"), geography)
    employee_count_buckets = employee_count_buckets_for_icp(
        {
            "employee_count": raw_employee_count if raw_employee_count is not None else employee_count,
            "employee_count_buckets": normalized.get("employee_count_buckets"),
            "employee_counts": normalized.get("employee_counts"),
            "company_size": normalized.get("company_size"),
            "company_size_bucket": normalized.get("company_size_bucket"),
            "employee_range": normalized.get("employee_range"),
        }
    )
    normalized["employee_count"] = employee_count_buckets if len(employee_count_buckets) > 1 else employee_count
    normalized.pop("employee_count_buckets", None)
    normalized.pop("employee_counts", None)
    normalized["product_service"] = product_service or required_attribute
    normalized["required_attribute"] = required_attribute
    normalized["intent_signal"] = intent_signal
    normalized["intent_signal_text"] = _first_text(normalized.get("intent_signal_text"), intent_signal)
    normalized["intent_signals"] = _intent_signals_list(normalized.get("intent_signals"), intent_signal)
    normalized["intent_category"] = _intent_category(normalized.get("intent_category"), intent_signal)
    normalized["intent_max_age_days"] = _positive_int(normalized.get("intent_max_age_days"), default=365)
    normalized["bonus_intents"] = _bonus_intents_list(normalized.get("bonus_intents"))
    normalized["company_stage"] = _first_text(normalized.get("company_stage"), default="Any")
    normalized["prompt"] = _first_text(normalized.get("prompt"), _prompt_from_icp(normalized))
    normalized["target_roles"] = normalized.get("target_roles") if isinstance(normalized.get("target_roles"), list) else []
    normalized["target_seniority"] = _first_text(normalized.get("target_seniority"))
    if not _first_text(normalized.get("icp_id")):
        normalized["icp_id"] = sha256_json(
            {
                "industry": industry,
                "sub_industry": sub_industry,
                "geography": geography,
                "employee_count": employee_count_buckets if len(employee_count_buckets) > 1 else employee_count,
                "product_service": normalized["product_service"],
                "intent_signal": intent_signal,
            }
        )[:18]

    if not normalized["industry"] or not normalized["intent_signal"]:
        raise PrivateModelRuntimeError("private model ICP is missing industry or intent signal after canonicalization")
    return normalized


def employee_count_buckets_for_icp(icp: Mapping[str, Any]) -> list[str]:
    """Return the relaxed Research Lab employee buckets for an ICP."""

    if not isinstance(icp, Mapping):
        raise PrivateModelRuntimeError("private model ICP payload must be an object")
    raw_employee_count = icp.get("employee_count")
    primary = _normalize_employee_count_bucket(
        _first_text(
            raw_employee_count,
            icp.get("company_size"),
            icp.get("company_size_bucket"),
            icp.get("employee_range"),
            default="51-200",
        )
    )
    explicit = icp.get("employee_count_buckets") or icp.get("employee_counts")
    if explicit is None and (
        isinstance(raw_employee_count, (list, tuple))
        or any(sep in str(raw_employee_count or "") for sep in ("|", ";"))
        or " or " in str(raw_employee_count or "").lower()
    ):
        explicit = raw_employee_count
    if explicit is None:
        return [primary]
    return normalize_employee_count_buckets(
        explicit,
        primary_bucket=primary,
        expand_single=False,
    )


def ensure_private_model_outputs(
    outputs: Any,
    *,
    context_label: str,
    require_non_empty: bool,
) -> list[Mapping[str, Any]]:
    """Validate the adapter's JSON-list contract at the evaluation boundary."""
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes, bytearray)):
        raise PrivateModelRuntimeError(f"{context_label} adapter must return a JSON array")
    normalized = list(outputs)
    if require_non_empty and not normalized:
        raise PrivateModelRuntimeError(f"{context_label} adapter returned zero companies")
    if not all(isinstance(item, Mapping) for item in normalized):
        raise PrivateModelRuntimeError(f"{context_label} adapter returned a non-object company row")
    if _contains_secret_material(normalized):
        raise PrivateModelRuntimeError(f"{context_label} adapter returned raw secret material")
    return normalized


@dataclass(frozen=True)
class PrivateModelAdapterSpec:
    source_path: Path
    module_name: str = "research_lab_adapter"
    callable_name: str = "run_icp"
    python_executable: str = sys.executable
    timeout_seconds: int = 900
    env_passthrough: tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH
    extra_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PrivateModelAdapterSpec":
        return cls(
            source_path=Path(str(data["source_path"])).expanduser().resolve(),
            module_name=str(data.get("module_name") or "research_lab_adapter"),
            callable_name=str(data.get("callable_name") or "run_icp"),
            python_executable=str(data.get("python_executable") or sys.executable),
            timeout_seconds=int(data.get("timeout_seconds") or 900),
            env_passthrough=tuple(str(item) for item in data.get("env_passthrough", DEFAULT_ENV_PASSTHROUGH)),
            extra_env={str(k): str(v) for k, v in dict(data.get("extra_env") or {}).items()},
        )


class SubprocessPrivateModelRunner:
    """Execute a private model adapter in a separate Python process."""

    def __init__(self, spec: PrivateModelAdapterSpec | Mapping[str, Any]):
        self.spec = spec if isinstance(spec, PrivateModelAdapterSpec) else PrivateModelAdapterSpec.from_mapping(spec)
        if not self.spec.source_path.exists():
            raise PrivateModelRuntimeError(f"private model source path does not exist: {self.spec.source_path}")

    def __call__(self, icp: Mapping[str, Any], context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        payload = {
            "icp": canonicalize_private_model_icp(icp),
            "context": context_with_runtime_options(
                context,
                outer_timeout_seconds=self.spec.timeout_seconds,
            ),
        }
        env = _build_subprocess_env(self.spec)
        command = [
            self.spec.python_executable,
            "-c",
            _ADAPTER_BOOTSTRAP,
            str(self.spec.source_path),
            self.spec.module_name,
            self.spec.callable_name,
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
                text=True,
                capture_output=True,
                timeout=self.spec.timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest whatever trace the container emitted before the kill;
            # timeouts are exactly where the in-flight behavior matters.
            _collect_incontainer_trace(_timeout_stderr_text(exc))
            raise PrivateModelRuntimeError("private model adapter timed out") from exc

        stderr_text = _collect_incontainer_trace(completed.stderr)
        if completed.returncode != 0:
            stderr = _sanitize_text(stderr_text)[-1200:]
            raise PrivateModelRuntimeError(f"private model adapter failed with code {completed.returncode}: {stderr}")
        validate_sourcing_runtime_receipt(
            completed.stderr,
            expected_runtime_options=payload["context"]["runtime_options"],
        )

        try:
            decoded = _loads_adapter_stdout(completed.stdout)
        except json.JSONDecodeError as exc:
            stdout = _sanitize_text(completed.stdout)[-800:]
            stderr = _sanitize_text(stderr_text)[-800:]
            raise PrivateModelRuntimeError(
                f"private model adapter returned invalid JSON: stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        _raise_on_empty_provider_error(decoded, stderr_text, context_label="private model")
        return ensure_private_model_outputs(
            decoded,
            context_label="private model",
            require_non_empty=False,
        )

    def metadata(self) -> Mapping[str, Any]:
        env = _build_subprocess_env(self.spec)
        command = [
            self.spec.python_executable,
            "-c",
            _ADAPTER_METADATA_BOOTSTRAP,
            str(self.spec.source_path),
            self.spec.module_name,
            "adapter_metadata",
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.spec.timeout_seconds,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            stderr = _sanitize_text(completed.stderr)[-1200:]
            raise PrivateModelRuntimeError(f"private model metadata failed with code {completed.returncode}: {stderr}")
        try:
            decoded = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PrivateModelRuntimeError("private model metadata returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise PrivateModelRuntimeError("private model metadata must return a JSON object")
        if _contains_secret_material(decoded):
            raise PrivateModelRuntimeError("private model metadata returned raw secret material")
        return validate_sourcing_adapter_metadata(decoded)


@dataclass(frozen=True)
class DockerPrivateModelSpec:
    image_digest: str
    module_name: str = "research_lab_adapter"
    callable_name: str = "run_icp"
    timeout_seconds: int = 900
    docker_executable: str = "docker"
    platform: str = field(default_factory=_default_docker_platform)
    env_passthrough: tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH
    extra_env: Mapping[str, str] = field(default_factory=dict)
    pull_before_run: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DockerPrivateModelSpec":
        return cls(
            image_digest=str(data["image_digest"]),
            module_name=str(data.get("module_name") or "research_lab_adapter"),
            callable_name=str(data.get("callable_name") or "run_icp"),
            timeout_seconds=int(data.get("timeout_seconds") or 900),
            docker_executable=str(data.get("docker_executable") or "docker"),
            platform=str(data.get("platform") or os.getenv("RESEARCH_LAB_PRIVATE_MODEL_DOCKER_PLATFORM") or DEFAULT_DOCKER_PLATFORM),
            env_passthrough=tuple(str(item) for item in data.get("env_passthrough", DEFAULT_ENV_PASSTHROUGH)),
            extra_env={str(k): str(v) for k, v in dict(data.get("extra_env") or {}).items()},
            pull_before_run=bool(data.get("pull_before_run", True)),
        )


class DockerPrivateModelRunner:
    """Execute a private model adapter from an immutable ECR image digest."""

    def __init__(self, spec: DockerPrivateModelSpec | Mapping[str, Any]):
        self.spec = spec if isinstance(spec, DockerPrivateModelSpec) else DockerPrivateModelSpec.from_mapping(spec)
        if "@sha256:" not in self.spec.image_digest:
            raise PrivateModelRuntimeError("docker private model image must be an immutable digest")
        if self.spec.pull_before_run:
            self._pull_image()

    def __call__(self, icp: Mapping[str, Any], context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        payload = {
            "icp": canonicalize_private_model_icp(icp),
            "context": context_with_runtime_options(
                context,
                outer_timeout_seconds=self.spec.timeout_seconds,
            ),
        }
        decoded = self._run_json(
            bootstrap=_DOCKER_ADAPTER_BOOTSTRAP,
            argv=(self.spec.module_name, self.spec.callable_name),
            stdin_payload=payload,
            expected_runtime_options=payload["context"]["runtime_options"],
        )
        return ensure_private_model_outputs(
            decoded,
            context_label="private model",
            require_non_empty=False,
        )

    def metadata(self) -> Mapping[str, Any]:
        decoded = self._run_json(
            bootstrap=_DOCKER_METADATA_BOOTSTRAP,
            argv=(self.spec.module_name, "adapter_metadata"),
            stdin_payload={},
        )
        if not isinstance(decoded, Mapping):
            raise PrivateModelRuntimeError("private model metadata must return a JSON object")
        if _contains_secret_material(decoded):
            raise PrivateModelRuntimeError("private model metadata returned raw secret material")
        return validate_sourcing_adapter_metadata(decoded)

    def _pull_image(self) -> None:
        completed = subprocess.run(
            [self.spec.docker_executable, "pull", *_docker_platform_args(self.spec), self.spec.image_digest],
            text=True,
            capture_output=True,
            timeout=self.spec.timeout_seconds,
            env=_build_docker_process_env(self.spec),
            check=False,
        )
        if completed.returncode != 0:
            stderr = _sanitize_text(completed.stderr)[-1200:]
            raise PrivateModelRuntimeError(f"docker pull failed with code {completed.returncode}: {stderr}")

    def _run_json(
        self,
        *,
        bootstrap: str,
        argv: Sequence[str],
        stdin_payload: Mapping[str, Any],
        expected_runtime_options: Mapping[str, Any] | None = None,
    ) -> Any:
        # Provider evidence cache: bind-mount the recorded baseline cache
        # read-only and repoint the env at the in-container path (the trailing
        # -e overrides the passthrough name). The per-ICP directory form is
        # the intended one — each run is only ever handed evidence for the ICP
        # in its own payload, never the rest of the window's sealed ICPs; the
        # single-file form remains for callers that already scope the file
        # themselves. An absent host file leaves the run uncached.
        evidence_cache_args: list[str] = []
        extra_env = dict(self.spec.extra_env or {})
        host_cache_path = ""
        cache_dir = str(extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR") or "").strip()
        if cache_dir and isinstance(stdin_payload, dict):
            payload_icp = stdin_payload.get("icp")
            if isinstance(payload_icp, dict):
                from .provider_evidence_cache import icp_evidence_cache_key

                host_cache_path = os.path.join(
                    cache_dir, f"{icp_evidence_cache_key(payload_icp)}.json"
                )
        if not host_cache_path:
            host_cache_path = str(
                extra_env.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH") or ""
            ).strip()
        if host_cache_path and os.path.isfile(host_cache_path):
            container_cache_path = "/research_lab_provider_evidence_cache.json"
            evidence_cache_args = [
                "-v",
                f"{host_cache_path}:{container_cache_path}:ro",
                "-e",
                f"RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH={container_cache_path}",
            ]
        provider_cost_scope_doc: dict[str, Any] = {
            "image_digest": self.spec.image_digest,
            "argv": list(argv),
            "stdin_payload": stdin_payload,
        }
        evaluation_scope = str(extra_env.get(PROVIDER_COST_EVALUATION_SCOPE_ENV) or "").strip()
        if evaluation_scope:
            if len(evaluation_scope) > 512 or _contains_secret_material(evaluation_scope):
                raise PrivateModelRuntimeError("provider cost evaluation scope is unsafe")
            provider_cost_scope_doc["evaluation_scope"] = evaluation_scope
        provider_cost_scope = sha256_json(provider_cost_scope_doc)
        provider_cost_args = [
            "-e",
            f"RESEARCH_LAB_PROVIDER_COST_SCOPE={provider_cost_scope}",
        ]
        command = [
            self.spec.docker_executable,
            "run",
            "--rm",
            "-i",
            *_docker_platform_args(self.spec),
            *_docker_env_args(self.spec),
            *provider_cost_args,
            *evidence_cache_args,
            self.spec.image_digest,
            "python",
            "-c",
            bootstrap,
            *argv,
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(stdin_payload, separators=(",", ":"), sort_keys=True),
                text=True,
                capture_output=True,
                timeout=self.spec.timeout_seconds,
                env=_build_docker_process_env(self.spec),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            # Harvest whatever trace the container emitted before the kill;
            # timeouts are exactly where the in-flight behavior matters.
            _collect_incontainer_trace(_timeout_stderr_text(exc))
            raise PrivateModelRuntimeError("docker private model adapter timed out") from exc
        stderr_text = _collect_incontainer_trace(completed.stderr)
        if completed.returncode != 0:
            stderr = _sanitize_text(stderr_text)[-1200:]
            raise PrivateModelRuntimeError(f"docker private model adapter failed with code {completed.returncode}: {stderr}")
        if expected_runtime_options is not None:
            validate_sourcing_runtime_receipt(
                completed.stderr,
                expected_runtime_options=expected_runtime_options,
            )
        try:
            decoded = _loads_adapter_stdout(completed.stdout)
        except json.JSONDecodeError as exc:
            stdout = _sanitize_text(completed.stdout)[-800:]
            stderr = _sanitize_text(stderr_text)[-800:]
            raise PrivateModelRuntimeError(
                f"docker private model adapter returned invalid JSON: stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        _raise_on_empty_provider_error(decoded, stderr_text, context_label="docker private model")
        return decoded


def load_private_artifact_manifest(uri: str) -> dict[str, Any]:
    """Load a private artifact manifest from S3 or a local JSON file."""
    if not uri:
        raise PrivateModelRuntimeError("private model manifest URI is required")
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(uri)
        try:
            import boto3
        except Exception as exc:
            raise PrivateModelRuntimeError("boto3 is required to load S3 private manifests") from exc
        response = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read().decode("utf-8")
    else:
        raw = Path(uri).expanduser().read_text(encoding="utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise PrivateModelRuntimeError("private model manifest must be a JSON object")
    if _contains_secret_material(decoded):
        raise PrivateModelRuntimeError("private model manifest contains raw secret material")
    return dict(decoded)


def verify_private_artifact_manifest_signature(
    manifest: Any,
    *,
    key_id: str = DEFAULT_PRIVATE_MODEL_ARTIFACT_SIGNING_KMS_KEY_ID,
    s3_client: Any = None,
    kms_client: Any = None,
) -> dict[str, Any]:
    """Verify that KMS signed the final, self-consistent manifest hash.

    Successful default-client verifications are cached by immutable manifest
    hash, signature URI, and key identity. Failures are never cached.
    """

    manifest_hash = _private_manifest_field(manifest, "manifest_hash")
    signature_ref = _private_manifest_field(manifest, "signature_ref")
    resolved_key_id = str(key_id or "").strip()
    if not manifest_hash.startswith("sha256:") or len(manifest_hash) != 71:
        raise PrivateModelRuntimeError(
            "private artifact manifest signature requires a sha256 manifest hash"
        )
    if not signature_ref.startswith("s3://"):
        raise PrivateModelRuntimeError(
            "private artifact manifest signature_ref must be an s3:// URI"
        )
    if not resolved_key_id:
        raise PrivateModelRuntimeError(
            "private artifact manifest signing KMS key id is required"
        )
    payload = _private_manifest_hash_payload(manifest)
    expected_manifest_hash = sha256_json(payload)
    if manifest_hash != expected_manifest_hash:
        raise PrivateModelRuntimeError(
            "private artifact manifest hash does not match its payload"
        )
    _verify_consumer_contract_manifest(payload)
    if s3_client is None and kms_client is None:
        return dict(
            _verify_private_artifact_manifest_signature_cached(
                manifest_hash,
                signature_ref,
                resolved_key_id,
            )
        )
    return _verify_private_artifact_manifest_signature_uncached(
        manifest_hash=manifest_hash,
        signature_ref=signature_ref,
        key_id=resolved_key_id,
        s3_client=s3_client,
        kms_client=kms_client,
    )


def _private_manifest_field(manifest: Any, field: str) -> str:
    if isinstance(manifest, Mapping):
        value = manifest.get(field)
    else:
        value = getattr(manifest, field, "")
    return str(value or "").strip()


def _private_manifest_hash_payload(manifest: Any) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        to_dict = getattr(manifest, "to_dict", None)
        if not callable(to_dict):
            raise PrivateModelRuntimeError(
                "private artifact manifest must be a mapping or manifest object"
            )
        payload = dict(to_dict())
    payload.pop("manifest_hash", None)
    return payload


def _verify_consumer_contract_manifest(payload: Mapping[str, Any]) -> None:
    contract = payload.get("compatibility_contract")
    fixtures = payload.get("consumer_parity_fixtures")
    matching_contracts = [
        expected_fixtures
        for expected_contract, expected_fixtures in _REVIEWED_CONSUMER_MANIFEST_PAIRS
        if contract == expected_contract
    ]
    if not matching_contracts:
        raise PrivateModelRuntimeError(
            "private artifact compatibility contract differs from the "
            "reviewed Lab snapshots"
        )
    if fixtures not in matching_contracts:
        raise PrivateModelRuntimeError(
            "private artifact parity fixtures differ from the reviewed Lab "
            "contract pair"
        )


@lru_cache(maxsize=256)
def _verify_private_artifact_manifest_signature_cached(
    manifest_hash: str,
    signature_ref: str,
    key_id: str,
) -> dict[str, Any]:
    return _verify_private_artifact_manifest_signature_uncached(
        manifest_hash=manifest_hash,
        signature_ref=signature_ref,
        key_id=key_id,
    )


def _verify_private_artifact_manifest_signature_uncached(
    *,
    manifest_hash: str,
    signature_ref: str,
    key_id: str,
    s3_client: Any = None,
    kms_client: Any = None,
) -> dict[str, Any]:
    signature_bucket, signature_key = _parse_s3_uri(signature_ref)
    if s3_client is None or kms_client is None:
        try:
            import boto3
        except Exception as exc:
            raise PrivateModelRuntimeError(
                "boto3 is required to verify private artifact manifests"
            ) from exc
        s3_client = s3_client or boto3.client("s3")
        kms_client = kms_client or boto3.client("kms")
    try:
        response = s3_client.get_object(
            Bucket=signature_bucket,
            Key=signature_key,
        )
        encoded_signature = response["Body"].read()
        if isinstance(encoded_signature, bytes):
            encoded_signature = encoded_signature.decode("ascii")
        signature = base64.b64decode(str(encoded_signature).strip(), validate=True)
        verification = kms_client.verify(
            KeyId=key_id,
            Message=manifest_hash.encode("utf-8"),
            MessageType="RAW",
            Signature=signature,
            SigningAlgorithm="ECDSA_SHA_256",
        )
    except Exception as exc:
        raise PrivateModelRuntimeError(
            "private artifact manifest KMS signature verification failed"
        ) from exc
    if not bool(verification.get("SignatureValid")):
        raise PrivateModelRuntimeError(
            "private artifact manifest KMS signature was rejected"
        )
    return {
        "verified": True,
        "manifest_hash": manifest_hash,
        "signature_ref": signature_ref,
        "key_id": str(verification.get("KeyId") or key_id),
        "signing_algorithm": str(
            verification.get("SigningAlgorithm") or "ECDSA_SHA_256"
        ),
    }


def sign_digest_with_kms(
    *,
    key_id: str,
    digest_hash: str,
    signature_uri_prefix: str = "",
) -> str:
    """KMS-sign a sha256 digest and return a verifier-facing signature ref."""
    if not key_id:
        raise PrivateModelRuntimeError("KMS key id is required for score-bundle signing")
    if not digest_hash.startswith("sha256:"):
        raise PrivateModelRuntimeError("KMS digest signing requires a sha256: hash")
    try:
        import boto3
    except Exception as exc:
        raise PrivateModelRuntimeError("boto3 is required for KMS signing") from exc
    digest = bytes.fromhex(digest_hash.split(":", 1)[1])
    kms = boto3.client("kms")
    response = kms.sign(
        KeyId=key_id,
        Message=digest,
        MessageType="DIGEST",
        SigningAlgorithm="ECDSA_SHA_256",
    )
    signature = bytes(response["Signature"])
    signature_hash = sha256_bytes(signature)
    if signature_uri_prefix:
        bucket, prefix = _parse_s3_uri(signature_uri_prefix.rstrip("/") + "/placeholder")
        object_prefix = prefix.rsplit("/", 1)[0]
        key = f"{object_prefix}/{digest_hash.split(':', 1)[1]}.signature.json"
        payload = {
            "schema_version": "1.0",
            "signature_type": "aws_kms_ecdsa_sha256",
            "key_id": key_id,
            "message_hash": digest_hash,
            "signature_hash": signature_hash,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
        }
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"
    return f"kms-signature:{key_id}:{signature_hash}"


def compute_private_source_tree_hash(path: Path | str) -> str:
    """Compute a deterministic hash of a private checkout without reading .git/env."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise PrivateModelRuntimeError(f"source path does not exist: {root}")
    digest_inputs: list[tuple[str, str]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root).as_posix()
        if _excluded_source_path(rel):
            continue
        digest_inputs.append((rel, sha256_bytes(file_path.read_bytes())))
    return sha256_json(digest_inputs)


def build_local_private_artifact_manifest(
    *,
    source_path: Path | str,
    git_commit_sha: str,
    image_digest: str,
    manifest_uri: str,
    signature_ref: str,
    component_registry_version: str,
    scoring_adapter_version: str,
    build_id: str = "",
    config_payload: Mapping[str, Any] | None = None,
    consumer_contract_id: str = "",
) -> dict[str, Any]:
    """Build the same manifest shape private CI publishes.

    Official production manifests must point at an immutable private ECR digest
    and S3 URI. This helper exists so the CI script and local verification use
    one canonical hash shape.
    """
    reviewed_snapshots = reviewed_consumer_snapshots()
    source_snapshot = resolve_reviewed_consumer_snapshot(Path(source_path))
    requested_contract_id = str(consumer_contract_id or "").strip()
    if source_snapshot is None:
        raise PrivateModelRuntimeError(
            "private model source has no reviewed contract/parity pair"
        )
    if requested_contract_id and requested_contract_id not in reviewed_snapshots:
        raise PrivateModelRuntimeError(
            "private model consumer contract id is not reviewed"
        )
    if (
        requested_contract_id
        and requested_contract_id
        != str(source_snapshot["contract"]["contract_id"])
    ):
        raise PrivateModelRuntimeError(
            "private model source contract differs from requested contract id"
        )
    contract = source_snapshot["contract"]
    payload = {
        "model_artifact_hash": compute_private_source_tree_hash(source_path),
        "git_commit_sha": str(git_commit_sha),
        "image_digest": str(image_digest),
        "config_hash": sha256_json(config_payload or {}),
        "component_registry_version": str(component_registry_version),
        "scoring_adapter_version": str(scoring_adapter_version),
        "compatibility_contract": {
            "contract_id": str(contract["contract_id"]),
            "path": str(contract["canonical_path"]),
            "sha256": str(source_snapshot["contract_sha256"]),
        },
        "consumer_parity_fixtures": {
            "path": str(contract["parity_fixture_path"]),
            "sha256": str(source_snapshot["parity_sha256"]),
        },
        "manifest_uri": str(manifest_uri),
        "signature_ref": str(signature_ref),
        "build_id": str(build_id),
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _build_subprocess_env(spec: PrivateModelAdapterSpec) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
    }
    for name in spec.env_passthrough:
        if name in os.environ:
            env[name] = os.environ[name]
    for name, value in spec.extra_env.items():
        env[name] = value
    return env


def _build_docker_process_env(spec: DockerPrivateModelSpec) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUNBUFFERED": "1",
    }
    for name in spec.env_passthrough:
        if name in os.environ:
            env[name] = os.environ[name]
    for name, value in spec.extra_env.items():
        env[name] = value
    return env


def _docker_env_args(spec: DockerPrivateModelSpec) -> list[str]:
    args: list[str] = []
    names = set(spec.env_passthrough) | set(spec.extra_env)
    for name in sorted(names):
        if name in os.environ or name in spec.extra_env:
            args.extend(["-e", name])
    return args


def _docker_platform_args(spec: DockerPrivateModelSpec) -> list[str]:
    platform = str(spec.platform or "").strip()
    return ["--platform", platform] if platform else []


def _redacted_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_secret_material(context):
        raise PrivateModelRuntimeError("run context contains raw secret material")
    return dict(context)


def context_with_runtime_options(
    context: Mapping[str, Any],
    *,
    outer_timeout_seconds: float,
) -> dict[str, Any]:
    """Bind model work to the host's actual per-invocation timeout.

    The outer process/container timeout is authoritative. Candidate-supplied
    options may request a smaller agent budget, but may never widen the cap or
    consume the final serialization reserve.
    """

    outer = max(10.0, float(outer_timeout_seconds))
    cap = min(SOURCING_MODEL_MAX_RUNTIME_CAP_SECONDS, max(10.0, outer - 3.0))
    reserve = min(
        SOURCING_MODEL_MAX_FINALIZATION_RESERVE_SECONDS,
        max(2.0, cap * 0.1),
    )
    incoming = dict((context or {}).get("runtime_options") or {})
    requested_agent = incoming.get("agent_timeout_seconds")
    try:
        agent_timeout = int(requested_agent)
    except (TypeError, ValueError):
        agent_timeout = int(max(30.0, cap - reserve))
    agent_timeout = max(
        5,
        min(
            agent_timeout,
            SOURCING_MODEL_MAX_AGENT_TIMEOUT_SECONDS,
            int(max(5.0, cap - reserve)),
        ),
    )
    result = _redacted_context(context)
    result["runtime_options"] = {
        "runtime_cap_seconds": cap,
        "finalization_reserve_seconds": reserve,
        "agent_timeout_seconds": agent_timeout,
    }
    return result


def validate_sourcing_adapter_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless an artifact declares the Lab runtime contract."""

    document = dict(metadata)
    if document.get("adapter_version") not in EXPECTED_SOURCING_ADAPTER_VERSIONS:
        raise PrivateModelRuntimeError(
            "private model does not declare a reviewed adapter contract"
        )
    if (
        document.get("component_registry_version")
        != EXPECTED_COMPONENT_REGISTRY_VERSION
    ):
        raise PrivateModelRuntimeError(
            "private model does not declare the supported component registry v2"
        )
    required_capabilities = {
        "deadline",
        "emit",
        "http_fetch",
        "probe_origin",
        "resolve_host",
    }
    if document.get("capability_contract_version") != (
        "sourcing-model-runtime-capabilities:v2"
    ):
        raise PrivateModelRuntimeError(
            "private model does not declare runtime capability contract v2"
        )
    capabilities = {
        str(item) for item in document.get("runtime_capabilities") or ()
    }
    if capabilities != required_capabilities:
        raise PrivateModelRuntimeError(
            "private model runtime capability set differs from the Lab contract"
        )
    if not str(document.get("resilience_policy_version") or "").startswith(
        "sourcing-model-resilience:"
    ):
        raise PrivateModelRuntimeError(
            "private model does not declare a resilience policy"
        )
    firmographic = document.get("firmographic_discovery")
    if not isinstance(firmographic, Mapping) or not str(
        firmographic.get("firmographic_policy_version") or ""
    ).startswith("sourcing-model-firmographic-discovery:"):
        raise PrivateModelRuntimeError(
            "private model does not declare firmographic discovery policy"
        )
    taxonomy = document.get("industry_taxonomy")
    taxonomy_hash = (
        str(taxonomy.get("taxonomy_content_hash") or "")
        if isinstance(taxonomy, Mapping)
        else ""
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", taxonomy_hash):
        raise PrivateModelRuntimeError(
            "private model does not declare a valid industry taxonomy hash"
        )
    routing = document.get("routing")
    if not isinstance(routing, Mapping):
        raise PrivateModelRuntimeError(
            "private model does not declare model-owned routing metadata"
        )
    if routing.get("compiler_version") != EXPECTED_ROUTING_COMPILER_VERSION:
        raise PrivateModelRuntimeError(
            "private model routing compiler version is unsupported"
        )
    for field in ("catalog_sha256", "policy_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(routing.get(field) or "")):
            raise PrivateModelRuntimeError(
                f"private model routing {field} is invalid"
            )
    catalog = routing.get("catalog")
    policy = routing.get("policy")
    if not isinstance(catalog, Mapping) or not isinstance(policy, Mapping):
        raise PrivateModelRuntimeError(
            "private model routing catalog and policy must be structured"
        )
    for name, payload in (("catalog", catalog), ("policy", policy)):
        expected_hash = sha256_json(payload).removeprefix("sha256:")
        if routing.get(f"{name}_sha256") != expected_hash:
            raise PrivateModelRuntimeError(
                f"private model routing {name} hash does not match its payload"
            )
    if routing.get("private_bindings_exposed") is not False:
        raise PrivateModelRuntimeError(
            "private model routing metadata exposes private bindings"
        )
    if routing.get("source_add_requires_manifest_sha256") is not True:
        raise PrivateModelRuntimeError(
            "private model routing does not require source manifest attestation"
        )
    runtime_routing = document.get("runtime_routing")
    if not isinstance(runtime_routing, Mapping):
        raise PrivateModelRuntimeError(
            "private model does not declare runtime routing metadata"
        )
    if (
        runtime_routing.get("compiler_version")
        != EXPECTED_ROUTING_COMPILER_VERSION
    ):
        raise PrivateModelRuntimeError(
            "private model runtime routing compiler version is unsupported"
        )
    for field in ("catalog_sha256", "policy_sha256"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(runtime_routing.get(field) or "")
        ):
            raise PrivateModelRuntimeError(
                f"private model runtime routing {field} is invalid"
            )
    runtime_catalog = runtime_routing.get("catalog")
    runtime_policy = runtime_routing.get("policy")
    if not isinstance(runtime_catalog, Mapping) or not isinstance(
        runtime_policy, Mapping
    ):
        raise PrivateModelRuntimeError(
            "private model runtime routing catalog and policy must be structured"
        )
    for name, payload in (
        ("catalog", runtime_catalog),
        ("policy", runtime_policy),
    ):
        expected_hash = sha256_json(payload).removeprefix("sha256:")
        if runtime_routing.get(f"{name}_sha256") != expected_hash:
            raise PrivateModelRuntimeError(
                f"private model runtime routing {name} hash does not match its payload"
            )
    candidate_tools = set(
        dict(runtime_routing.get("candidate_tool_lanes") or {})
    )
    intent_tools = set(dict(runtime_routing.get("intent_tool_tiers") or {}))
    if not REQUIRED_RUNTIME_CANDIDATE_TOOLS <= candidate_tools:
        raise PrivateModelRuntimeError(
            "private model runtime candidate tool catalog is missing required tools"
        )
    if not REQUIRED_RUNTIME_INTENT_TOOLS <= intent_tools:
        raise PrivateModelRuntimeError(
            "private model runtime intent tool catalog is missing required tools"
        )
    catalog_tool_ids = {
        str(item.get("tool_id") or "")
        for item in runtime_catalog.get("tools") or ()
        if isinstance(item, Mapping)
    }
    if not candidate_tools | intent_tools <= catalog_tool_ids:
        raise PrivateModelRuntimeError(
            "private model runtime routing mappings reference uncataloged tools"
        )
    if runtime_routing.get("private_bindings_exposed") is not False:
        raise PrivateModelRuntimeError(
            "private model runtime routing metadata exposes private bindings"
        )
    intent_sources = routing.get("intent_sources")
    if (
        not isinstance(intent_sources, list)
        or not intent_sources
        or any(not str(source or "").strip() for source in intent_sources)
    ):
        raise PrivateModelRuntimeError(
            "private model routing intent source catalog is invalid"
        )
    component_registry = document.get("component_registry")
    source_router = (
        component_registry.get("source_router")
        if isinstance(component_registry, Mapping)
        else None
    )
    if not isinstance(source_router, Mapping) or list(
        source_router.get("strategy_options") or ()
    ) != list(intent_sources):
        raise PrivateModelRuntimeError(
            "source-router strategies differ from model-owned routing metadata"
        )
    return document


def validate_sourcing_runtime_receipt(
    stderr: str,
    *,
    expected_runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one complete receipt from every successful sourcing call."""

    return validate_sourcing_runtime_receipt_entries(
        parse_sourcing_runtime_lines(stderr),
        expected_runtime_options=expected_runtime_options,
    )


def validate_sourcing_runtime_receipt_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate receipts already decoded by the measured sandbox."""

    receipts = [
        dict(entry)
        for entry in entries
        if entry.get("kind") == "sourcing_branch_receipt"
    ]
    if len(receipts) != 1:
        raise PrivateModelRuntimeError(
            f"private model emitted {len(receipts)} sourcing branch receipts; expected 1"
        )
    receipt = receipts[0]
    try:
        expected_cap = float(expected_runtime_options["runtime_cap_seconds"])
        receipt_cap = float(receipt.get("runtime_cap_seconds") or 0.0)
    except (TypeError, ValueError) as exc:
        raise PrivateModelRuntimeError(
            "private model receipt runtime cap is not numeric"
        ) from exc
    if not math.isfinite(expected_cap) or expected_cap <= 0.0:
        raise PrivateModelRuntimeError(
            "Lab runtime allocation does not contain a finite positive cap"
        )
    if (
        not math.isfinite(receipt_cap)
        or abs(receipt_cap - expected_cap) > 0.001
    ):
        raise PrivateModelRuntimeError(
            "private model receipt runtime cap differs from the Lab allocation"
        )
    capability_contract = receipt.get("capability_contract")
    registered = (
        set(capability_contract.get("host_registered") or ())
        if isinstance(capability_contract, Mapping)
        else set()
    )
    if not {"deadline", "emit", "probe_origin", "resolve_host"}.issubset(
        registered
    ):
        raise PrivateModelRuntimeError(
            "private model receipt does not prove Lab capability registration"
        )
    taxonomy = receipt.get("industry_taxonomy")
    if not isinstance(taxonomy, Mapping) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        str(taxonomy.get("taxonomy_content_hash") or ""),
    ):
        raise PrivateModelRuntimeError(
            "private model receipt does not bind an industry taxonomy"
        )
    firmographic = receipt.get("firmographic_discovery")
    if not isinstance(firmographic, Mapping) or not isinstance(
        firmographic.get("plan"), Mapping
    ):
        raise PrivateModelRuntimeError(
            "private model receipt does not bind a firmographic discovery plan"
        )
    branches = receipt.get("branches")
    if not isinstance(branches, list) or not branches:
        raise PrivateModelRuntimeError(
            "private model receipt does not contain compiled intent branches"
        )
    catalog_hashes: set[str] = set()
    policy_hashes: set[str] = set()
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise PrivateModelRuntimeError(
                "private model intent branch receipt is malformed"
            )
        for field in (
            "route_plan_sha256",
            "route_policy_sha256",
            "route_catalog_sha256",
            "route_context_sha256",
        ):
            value = str(branch.get(field) or "")
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise PrivateModelRuntimeError(
                    f"private model intent branch {field} is invalid"
                )
        route_tools = branch.get("route_tool_ids")
        route_sources = branch.get("route_sources")
        if (
            not isinstance(route_tools, list)
            or not route_tools
            or any(not str(tool or "").strip() for tool in route_tools)
            or not isinstance(route_sources, list)
            or not route_sources
            or any(not str(source or "").strip() for source in route_sources)
        ):
            raise PrivateModelRuntimeError(
                "private model intent branch route is empty or malformed"
            )
        if not isinstance(branch.get("source_override"), bool):
            raise PrivateModelRuntimeError(
                "private model intent branch source override is not boolean"
            )
        compiled_source = str(branch.get("compiled_source") or "")
        selected_source = str(branch.get("source") or "")
        if compiled_source != str(route_sources[0]):
            raise PrivateModelRuntimeError(
                "private model intent branch compiled source differs from its route"
            )
        if not branch["source_override"] and selected_source != compiled_source:
            raise PrivateModelRuntimeError(
                "private model intent branch source differs without an override"
            )
        if branch["source_override"] and not selected_source:
            raise PrivateModelRuntimeError(
                "private model intent branch override source is empty"
            )
        catalog_hashes.add(str(branch["route_catalog_sha256"]))
        policy_hashes.add(str(branch["route_policy_sha256"]))
    if len(catalog_hashes) != 1 or len(policy_hashes) != 1:
        raise PrivateModelRuntimeError(
            "private model intent branches used inconsistent routing contracts"
        )
    return receipt


# Provider failures that end the whole sourcing run: credit/auth/quota
# rejections and provider-infra failures. Request-shaped rejections of
# model-generated inputs (Scrapingdog 400 "Something went wrong", 404 on an
# invalid URL, 410 Gone, 503 on a bad scrape target) are model-output
# behavior — the provider answered, the input was wrong — so an empty run
# that only carries those stands as a legitimate zero-company result.
_LOOP_ENDING_PROVIDER_STATUSES = frozenset({401, 402, 403, 429, 500, 502, 504})
_PROVIDER_CREDIT_MARKERS = (
    "out of credits",
    "insufficient credits",
    "payment required",
    "quota",
)
_PROVIDER_OUTAGE_TEXT_MARKERS = (
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "remotedisconnected",
    "incompleteread",
)
_PROVIDER_ERROR_STATUS_CODES = (400, 401, 402, 403, 404, 408, 409, 410, 429, 500, 502, 503, 504)


def _provider_error_line_is_loop_ending(line: str) -> bool:
    """True only for provider errors that mean the provider itself failed.

    Credit/quota markers win regardless of status (providers report exhausted
    credits under several 4xx codes). A recognized request-shaped status
    (400/404/408/409/410/503) is the provider correctly rejecting a
    model-generated input, not an outage. Text markers (timeouts, connection
    resets) only decide lines with no parseable status — e.g. transport
    failures reaching the evidence proxy.
    """
    lowered = line.lower()
    if any(marker in lowered for marker in _PROVIDER_CREDIT_MARKERS):
        return True
    status = 0
    for code in _PROVIDER_ERROR_STATUS_CODES:
        if f"http error {code}" in lowered or f"status={code}" in lowered or f'"status":{code}' in lowered:
            status = code
            break
    if status:
        return status in _LOOP_ENDING_PROVIDER_STATUSES
    return any(marker in lowered for marker in _PROVIDER_OUTAGE_TEXT_MARKERS)


def _raise_on_empty_provider_error(decoded: Any, stderr: str, *, context_label: str) -> None:
    if decoded != []:
        return
    provider_error = _provider_error_text(stderr)
    if not provider_error:
        return
    loop_ending_lines = [
        line for line in provider_error.splitlines() if _provider_error_line_is_loop_ending(line)
    ]
    if not loop_ending_lines:
        # Only request-shaped provider rejections (invalid model-generated
        # URLs and the like): the empty result is scored as-is instead of
        # marking the run unhealthy.
        logger.warning(
            "research_lab_private_runtime_empty_run_request_errors_only context=%s "
            "provider_error_lines=%d",
            context_label,
            len(provider_error.splitlines()),
        )
        return
    raise PrivateModelRuntimeError(
        f"{context_label} provider-backed sourcing failed before returning companies: "
        + "\n".join(loop_ending_lines)[-1200:]
    )


def _provider_error_text(stderr: str) -> str:
    sanitized = _sanitize_text(stderr)
    lines = [line.strip() for line in sanitized.splitlines() if PROVIDER_ERROR_MARKER in line]
    if not lines:
        return ""
    return "\n".join(lines)[-1200:]


def incontainer_trace_capture_enabled() -> bool:
    """Host-side mirror of the bootstrap's capture gate (default ON)."""
    raw = str(os.getenv(INCONTAINER_TRACE_CAPTURE_ENV) or "").strip().lower()
    if not raw:
        return True
    return raw in {"1", "true", "yes", "on"}


# Ambient per-task collector for in-container trace entries. The evaluator
# installs a list before invoking a model runner (asyncio.to_thread copies the
# context into the worker thread, so sync runners see the same list object);
# runners publish parsed entries into it. When no collector is installed —
# e.g. direct gateway runner calls outside the evaluator — publishing is a
# no-op so nothing accumulates.
_INCONTAINER_TRACE_COLLECTOR: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "research_lab_incontainer_trace_collector",
    default=None,
)

# Successful V2 model and scorer calls publish only their receipt hashes into
# this task-local sidecar. Baseline retries share runner/scorer objects across
# concurrent ICPs, so reading those objects' process-wide receipt histories
# cannot identify which receipts actually produced one durable ICP result.
_ATTESTED_RECEIPT_HASH_COLLECTOR: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "research_lab_attested_receipt_hash_collector",
    default=None,
)


def begin_incontainer_trace_collection() -> tuple[list[dict[str, Any]], contextvars.Token]:
    """Install an in-container trace collector for the current context."""
    entries: list[dict[str, Any]] = []
    token = _INCONTAINER_TRACE_COLLECTOR.set(entries)
    return entries, token


def end_incontainer_trace_collection(token: contextvars.Token) -> None:
    _INCONTAINER_TRACE_COLLECTOR.reset(token)


def publish_incontainer_trace_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    """Append decoded trace entries to the ambient collector, if any."""
    collector = _INCONTAINER_TRACE_COLLECTOR.get()
    if collector is None:
        return
    collector.extend(dict(entry) for entry in entries if isinstance(entry, Mapping))


def begin_attested_receipt_hash_collection() -> tuple[set[str], contextvars.Token]:
    """Install a successful-receipt collector for the current scoring attempt."""

    receipt_hashes: set[str] = set()
    token = _ATTESTED_RECEIPT_HASH_COLLECTOR.set(receipt_hashes)
    return receipt_hashes, token


def end_attested_receipt_hash_collection(token: contextvars.Token) -> None:
    _ATTESTED_RECEIPT_HASH_COLLECTOR.reset(token)


def publish_attested_receipt_hash(receipt_hash: str) -> None:
    """Publish one successful V2 receipt to the current attempt, if present."""

    collector = _ATTESTED_RECEIPT_HASH_COLLECTOR.get()
    if collector is None:
        return
    collector.add(str(receipt_hash or "").strip().lower())


def parse_incontainer_trace_lines(stderr: str) -> list[dict[str, Any]]:
    """Decode ``INCONTAINER_TRACE_MARKER`` lines from captured runner stderr.

    Malformed lines are skipped; entries keep stderr emission order (each also
    carries an in-process ``seq``). Retention is byte-budgeted: one runner
    invocation covers a single ICP, and an unbounded base64 trace of every
    provider exchange can reach hundreds of MB — decoded dicts held per ICP
    were a driver of scorer-worker memory exhaustion. Entries past the budget
    are dropped (newest-first ordering preserved for what is kept) and the
    drop is logged with a count, never silently.
    """
    max_bytes = _trace_capture_max_bytes()
    entries: list[dict[str, Any]] = []
    kept_bytes = 0
    dropped = 0
    for line in (stderr or "").splitlines():
        marker_index = line.find(INCONTAINER_TRACE_MARKER)
        if marker_index < 0:
            continue
        payload = line[marker_index + len(INCONTAINER_TRACE_MARKER):].strip()
        if not payload:
            continue
        if max_bytes > 0 and kept_bytes + len(payload) > max_bytes:
            dropped += 1
            continue
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, Mapping):
            entries.append(dict(decoded))
            kept_bytes += len(payload)
    if dropped:
        logger.warning(
            "incontainer_trace_capture_truncated kept_entries=%d kept_bytes=%d "
            "dropped_entries=%d max_bytes=%d",
            len(entries),
            kept_bytes,
            dropped,
            max_bytes,
        )
    return entries


def parse_sourcing_runtime_lines(stderr: str) -> list[dict[str, Any]]:
    """Decode bounded, structured model receipts/events from stderr."""

    entries: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    markers = (
        (SOURCING_BRANCH_RECEIPT_MARKER, "sourcing_branch_receipt"),
        (SOURCING_RUNTIME_EVENT_MARKER, "sourcing_runtime_event"),
    )
    for line in (stderr or "").splitlines():
        cursor = 0
        while cursor < len(line):
            matches = (
                (marker_index, marker, kind)
                for marker, kind in markers
                if (marker_index := line.find(marker, cursor)) >= 0
            )
            try:
                marker_index, marker, kind = min(matches, key=lambda item: item[0])
            except ValueError:
                break
            payload_start = marker_index + len(marker)
            untrimmed_payload = line[payload_start:]
            payload = untrimmed_payload.lstrip()
            payload_offset = payload_start + len(untrimmed_payload) - len(payload)
            if not payload:
                break
            try:
                decoded, consumed = decoder.raw_decode(payload[: 256 * 1024 + 1])
            except json.JSONDecodeError:
                cursor = payload_start
                continue
            if consumed > 256 * 1024:
                cursor = payload_offset + consumed
                continue
            if isinstance(decoded, Mapping):
                entries.append({"kind": kind, **dict(decoded)})
            cursor = payload_offset + max(1, consumed)
    return entries


def _trace_capture_max_bytes() -> int:
    """Per-invocation retention budget for decoded trace entries (0 disables)."""
    try:
        return int(os.getenv("RESEARCH_LAB_INCONTAINER_TRACE_MAX_BYTES", str(64 * 1024 * 1024)))
    except ValueError:
        return 64 * 1024 * 1024


def strip_incontainer_trace_lines(text: str) -> str:
    """Remove trace marker lines from stderr bound for diagnostics/error fields.

    Trace lines carry base64 payload material; letting them flow into exception
    messages or audit-visible diagnostics would bloat them and could trip the
    protected-material scanners. Non-trace lines (including provider-error
    marker lines) are preserved.
    """
    if not text:
        return text or ""
    markers = (
        INCONTAINER_TRACE_MARKER,
        SOURCING_BRANCH_RECEIPT_MARKER,
        SOURCING_RUNTIME_EVENT_MARKER,
    )
    return "\n".join(
        line for line in text.splitlines()
        if not any(marker in line for marker in markers)
    )


def _collect_incontainer_trace(stderr: str) -> str:
    """Publish trace entries from runner stderr and return the stripped text.

    Publishing is gated on the host-side capture flag; stripping is
    unconditional so trace payloads never leak into diagnostics even when a
    container emitted them while the host has capture disabled. Never raises.
    """
    if not stderr:
        return stderr or ""
    runtime_entries = parse_sourcing_runtime_lines(stderr)
    if runtime_entries:
        try:
            publish_incontainer_trace_entries(runtime_entries)
        except Exception:
            pass
    if incontainer_trace_capture_enabled():
        try:
            publish_incontainer_trace_entries(parse_incontainer_trace_lines(stderr))
        except Exception:
            # Capture is pure observation: a parse/publish failure must never
            # fail the run.
            pass
    return strip_incontainer_trace_lines(stderr)


def _timeout_stderr_text(exc: subprocess.TimeoutExpired) -> str:
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", "replace")
    return str(stderr or "")


def _loads_adapter_stdout(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass

    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("adapter stdout did not contain a JSON payload", stdout, 0)


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            nested = _first_text(*value)
            if nested:
                return nested
            continue
        if isinstance(value, Mapping):
            nested = _first_text(
                value.get("description"),
                value.get("signal"),
                value.get("text"),
                value.get("label"),
                value.get("name"),
                value.get("value"),
            )
            if nested:
                return nested
            continue
        text = " ".join(str(value).strip().split())
        if text:
            return text
    return default


def _normalize_employee_count_bucket(value: Any, *, default: str = "51-200") -> str:
    return _normalize_linkedin_employee_count_bucket(value, default=default)


def _intent_signal_text(icp: Mapping[str, Any]) -> str:
    return _first_text(
        icp.get("intent_signal_text"),
        icp.get("intent_signal"),
        icp.get("intent"),
        icp.get("buying_intent"),
        icp.get("intent_signals"),
    )


def _intent_signals_list(value: Any, fallback: str) -> list[str]:
    if isinstance(value, str):
        signals = [_first_text(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        signals = [_first_text(item) for item in value]
    else:
        signals = []
    signals = [signal for signal in signals if signal]
    if not signals and fallback:
        signals = [fallback]
    return signals


def _bonus_intents_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        signal = _first_text(item.get("intent_signal"), item.get("signal"))
        category = _intent_category(item.get("intent_category") or item.get("category"), signal)
        if not signal or not category:
            continue
        normalized.append(
            {
                "intent_signal": signal,
                "intent_category": category,
                "intent_max_age_days": _positive_int(item.get("intent_max_age_days") or item.get("max_age"), default=365),
            }
        )
    return normalized


def _required_attribute(
    value: Any,
    *,
    industry: str,
    sub_industry: str,
    product_service: str,
) -> str:
    existing = _first_text(value)
    if existing:
        return existing
    if product_service:
        return f"The company offers or provides {product_service}"
    if sub_industry:
        return f"The company operates in {sub_industry}"
    if industry:
        return f"The company operates in {industry}"
    return "The company matches the target customer profile"


def _intent_category(value: Any, signal: str) -> str:
    explicit = _first_text(value).strip().upper().replace(" ", "_").replace("-", "_")
    if explicit:
        return explicit
    text = signal.lower()
    if any(word in text for word in ("hiring", "job", "role", "career", "recruit")):
        return "HIRING"
    if any(word in text for word in ("tech stack", "installed", "uses ", "using ", "software", "tool")):
        return "TECHSTACK"
    if any(word in text for word in ("linkedin", "posted", "social", "tweet", "x.com")):
        return "SOCIAL_POSTING"
    if any(word in text for word in ("funding", "raised", "series ", "seed round", "financing")):
        return "FUNDING"
    if any(word in text for word in ("acquired", "acquisition", "merger", "bought")):
        return "ACQUISITION"
    if any(word in text for word in ("partner", "partnership", "integration")):
        return "PARTNERSHIP"
    if any(word in text for word in ("launch", "launched", "announced", "released", "new product")):
        return "PRODUCT_LAUNCH"
    if any(word in text for word in ("executive", "ceo", "cfo", "cto", "appointed", "joined")):
        return "LEADERSHIP_CHANGE"
    if any(word in text for word in ("expanded", "expansion", "new market", "opened")):
        return "MARKET_EXPANSION"
    if any(word in text for word in ("regulatory", "clearance", "certification", "approved")):
        return "REGULATORY_CLEARANCE"
    if any(word in text for word in ("factory", "facility", "store", "warehouse", "office opening")):
        return "FACILITY_OPENING"
    return "SALES_GROWTH"


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _prompt_from_icp(icp: Mapping[str, Any]) -> str:
    employee_count = _employee_count_display(icp.get("employee_count"))
    return " ".join(
        part
        for part in (
            f"{icp.get('industry', '')} companies",
            f"in {icp.get('geography', '')}" if icp.get("geography") else "",
            f"with {employee_count} employees" if employee_count else "",
            f"that {str(icp.get('required_attribute', '')).removeprefix('The company ').strip()}"
            if icp.get("required_attribute")
            else "",
            f"showing intent: {icp.get('intent_signal', '')}" if icp.get("intent_signal") else "",
        )
        if part
    )


def _employee_count_display(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        buckets = [_first_text(item) for item in value]
        buckets = [bucket for bucket in buckets if bucket]
        return " or ".join(buckets)
    return _first_text(value)


def _excluded_source_path(rel: str) -> bool:
    parts = rel.split("/")
    if any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"} for part in parts):
        return True
    if rel.endswith((".pyc", ".pyo", ".env", ".pem", ".key")):
        return True
    if rel == ".env" or rel.startswith(".env."):
        return True
    return False


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_secret_material(key) or _contains_secret_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False


def _sanitize_text(text: str) -> str:
    sanitized = text or ""
    for env_name in (
        "EXA_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "DEEPLINE_API_KEY",
    ):
        value = os.environ.get(env_name)
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    for marker in SECRET_MARKERS:
        sanitized = sanitized.replace(marker, "[REDACTED]")
    sanitized = re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[REDACTED]", sanitized)
    sanitized = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]", sanitized)
    return sanitized


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise PrivateModelRuntimeError(f"expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, sep, key = without_scheme.partition("/")
    if not bucket or not sep or not key:
        raise PrivateModelRuntimeError(f"invalid s3 URI: {uri}")
    return bucket, key


_PROVIDER_DIAGNOSTICS_BOOTSTRAP = r"""
import os
import re
import sys
import urllib.error
import urllib.request

_research_lab_original_urlopen = urllib.request.urlopen
_research_lab_trusted_v2_transport = (
    getattr(_research_lab_original_urlopen, "__module__", "")
    == "gateway.tee.sandbox_http_shim_v2"
)

# Provider evidence cache: when the runner supplies a recorded baseline
# request->response cache, an identical provider request always replays the
# same recorded canonical response (including recorded settled failures)
# instead of calling the provider live, so provider variance between runs
# cannot change the evaluation.
# Fingerprint canonicalization mirrors
# research_lab/eval/provider_evidence_cache.py and MUST stay in sync with it.
def _research_lab_load_evidence_cache():
    # Replay is governed solely by whether a cache is supplied; runs that
    # must observe live providers (the seeding baseline) are simply never
    # handed a cache. Recording and replay compose: hits replay recorded
    # evidence, live calls are captured at full fidelity.
    path = (os.environ.get("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH") or "").strip()
    if not path:
        return {}
    try:
        import json as _json
        import time as _time
        with open(path, "r", encoding="utf-8") as handle:
            doc = _json.load(handle)
        # Evidence expires at 00:00 UTC: a cache recorded on a previous UTC
        # day is ignored so the same input is recorded fresh for the new day.
        if isinstance(doc, dict):
            stamp = str(doc.get("utc_day") or "")
            if stamp and stamp != _time.strftime("%Y-%m-%d", _time.gmtime()):
                return {}
        entries = doc.get("entries") if isinstance(doc, dict) else None
        if not isinstance(entries, dict):
            return {}
        cache = {}
        for key, record in entries.items():
            # Canonical form is one record per fingerprint; settle a legacy
            # sequence the same way the builder does (last success wins,
            # otherwise the final failure).
            if isinstance(record, list):
                settled = None
                for item in record:
                    if not isinstance(item, dict) or not isinstance(item.get("status"), int):
                        continue
                    if settled is None or item["status"] < 400 or settled["status"] >= 400:
                        settled = item
                record = settled
            if isinstance(record, dict) and isinstance(record.get("status"), int):
                cache[str(key)] = record
        return cache
    except Exception:
        return {}

_research_lab_evidence_cache = _research_lab_load_evidence_cache()

import threading as _research_lab_threading

# Flags provider traffic made outside the instrumented urlopen path (for
# example an HTTP client layered directly on http.client) while replay is
# active, so the evaluation can tell the run used live evidence that replay
# never saw.
_research_lab_in_urlopen = _research_lab_threading.local()

_research_lab_evidence_proxy = (os.environ.get("RESEARCH_LAB_EVIDENCE_PROXY_URL") or "").strip().rstrip("/")
_research_lab_provider_cost_scope = (os.environ.get("RESEARCH_LAB_PROVIDER_COST_SCOPE") or "").strip()
_research_lab_provider_cost_cap_usd = (os.environ.get("RESEARCH_LAB_PROVIDER_COST_CAP_USD_PER_ICP") or "").strip()
_research_lab_provider_cost_soft_stop = (os.environ.get("RESEARCH_LAB_PROVIDER_COST_SOFT_STOP") or "1").strip().lower() not in {"0", "false", "no", "off"}
_research_lab_proxy_routes = (
    ("api.exa.ai", "/exa"),
    ("api.scrapingdog.com", "/sd"),
    ("openrouter.ai", "/or"),
    ("code.deepline.com", "/deepline"),
)

def _research_lab_uses_evidence_transport(proxied_url, target):
    if proxied_url is not None:
        return True
    if not _research_lab_trusted_v2_transport:
        return False
    try:
        import urllib.parse as _urlparse
        host = (_urlparse.urlsplit(str(target or "")).hostname or "").lower()
        return any(
            host == provider_host
            for provider_host, _route in _research_lab_proxy_routes
        )
    except Exception:
        return False

def _research_lab_proxy_rewrite(url):
    # Provider URL -> host proxy route, or None when not a provider call.
    if not _research_lab_evidence_proxy:
        return None
    try:
        import urllib.parse as _urlparse
        split = _urlparse.urlsplit(str(url or ""))
        host = (split.hostname or "").lower()
        for provider_host, route in _research_lab_proxy_routes:
            if host == provider_host:
                path = split.path or "/"
                query = ("?" + split.query) if split.query else ""
                return _research_lab_evidence_proxy + route + path + query
    except Exception:
        return None
    return None

def _research_lab_proxy_headers():
    headers = {}
    if _research_lab_provider_cost_scope:
        headers["X-Research-Lab-Cost-Scope"] = _research_lab_provider_cost_scope
    if _research_lab_provider_cost_cap_usd:
        headers["X-Research-Lab-Cost-Cap-Usd"] = _research_lab_provider_cost_cap_usd
    if _research_lab_provider_cost_soft_stop:
        headers["X-Research-Lab-Budget-Soft-Stop"] = "1"
    return headers

def _research_lab_decode_cost_event_header(headers):
    try:
        if headers is None:
            return None
        raw = headers.get("X-Research-Lab-Provider-Cost-Event")
        if not raw:
            return None
        import base64 as _cost_base64
        import json as _cost_json
        decoded = _cost_base64.b64decode(str(raw)).decode("utf-8")
        doc = _cost_json.loads(decoded)
        if not isinstance(doc, dict):
            return None
        text = _cost_json.dumps(doc, sort_keys=True, separators=(",", ":")).lower()
        for marker in ("sk-or-", "api_key", "service_role", "raw_secret", "hidden_prompt", "provider_output"):
            if marker in text:
                return None
        return doc
    except Exception:
        return None

def _research_lab_emit_evidence_marker(headers, method, target, request_body):
    cost_event = _research_lab_decode_cost_event_header(headers)
    if cost_event is not None:
        _research_lab_emit_trace(
            method,
            target,
            request_body,
            None,
            None,
            "provider_cost",
            "",
            phase="provider_cost",
            extra={"provider_cost_event": cost_event},
        )
    try:
        kind = str(headers.get("X-Research-Lab-Evidence") or "") if headers is not None else ""
    except Exception:
        kind = ""
    if kind == "hit":
        _research_lab_emit_trace(method, target, request_body, None, None, "cache_hit", "", phase="cache_hit")
    elif kind == "recorded":
        _research_lab_emit_trace(method, target, request_body, None, None, "cache_miss", "", phase="cache_miss")
    elif kind == "budget_soft_stop":
        _research_lab_emit_trace(
            method,
            target,
            request_body,
            None,
            None,
            "budget_soft_stop",
            "",
            phase="budget_soft_stop",
        )

def _research_lab_install_httpclient_watch():
    if not _research_lab_evidence_cache:
        return
    try:
        import http.client as _http_client
        _original_httpclient_request = _http_client.HTTPConnection.request

        def _research_lab_watched_request(self, method, url, *args, **kwargs):
            if not getattr(_research_lab_in_urlopen, "active", False):
                _research_lab_emit_trace(
                    method,
                    "%s%s" % (getattr(self, "host", ""), url),
                    None, None, None,
                    "uninstrumented_http", "", phase="uninstrumented_http",
                )
            return _original_httpclient_request(self, method, url, *args, **kwargs)

        _http_client.HTTPConnection.request = _research_lab_watched_request
    except Exception:
        pass

def _research_lab_evidence_fingerprint(method, url, body):
    import hashlib as _hashlib
    import json as _json
    import urllib.parse as _urlparse
    auth_params = ("api_key", "apikey", "api-key", "key", "token", "access_token", "x-api-key")
    method_part = str(method or "GET").upper()
    try:
        split = _urlparse.urlsplit(str(url or ""))
        pairs = [
            (k, v)
            for k, v in _urlparse.parse_qsl(split.query, keep_blank_values=True)
            if k.lower() not in auth_params
        ]
        pairs.sort()
        url_part = "|".join(
            (
                (split.scheme or "").lower(),
                (split.netloc or "").lower(),
                split.path or "",
                _urlparse.urlencode(pairs),
            )
        )
    except Exception:
        url_part = str(url or "")
    if body is None:
        body_bytes = b""
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        body_bytes = str(body).encode("utf-8", "replace")
    if body_bytes:
        try:
            parsed = _json.loads(body_bytes.decode("utf-8"))
            body_bytes = _json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except Exception:
            pass
    digest = _hashlib.sha256()
    digest.update(method_part.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(url_part.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(body_bytes)
    return digest.hexdigest()

class _ResearchLabCachedHeaders(object):
    def __init__(self, body_len):
        self._values = {"Content-Length": str(body_len)}

    def get(self, name, default=None):
        return self._values.get(str(name), default)

    def get_content_charset(self, failobj=None):
        return "utf-8"

    def items(self):
        return list(self._values.items())

class _ResearchLabCachedResponse(object):
    def __init__(self, url, status, body, headers=None):
        import io as _io
        self._stream = _io.BytesIO(body)
        self.url = url
        self.status = status
        self.code = status
        self.headers = headers if headers is not None else _ResearchLabCachedHeaders(len(body))
        self.reason = ""

    def read(self, *args):
        return self._stream.read(*args)

    def peek(self, *args):
        return self._stream.getbuffer().tobytes()[self._stream.tell():]

    def readline(self, *args):
        return self._stream.readline(*args)

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def info(self):
        return self.headers

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

def _research_lab_serve_cached_response(record, method, target, request_body):
    import urllib.error as _urlerror
    status = int(record.get("status") or 0)
    try:
        body = _research_lab_base64.b64decode(record.get("body_b64") or "")
    except Exception:
        body = b""
    if status >= 400:
        _research_lab_emit_trace(
            method, target, request_body, status, body,
            "error", "cached provider failure replayed", phase="cache_hit",
        )
        raise _urlerror.HTTPError(
            str(target), status, "cached provider failure replayed",
            _ResearchLabCachedHeaders(len(body)), _ResearchLabCachedResponse(str(target), status, body),
        )
    _research_lab_emit_trace(
        method, target, request_body, status, body, "success", "", phase="cache_hit",
    )
    return _ResearchLabCachedResponse(str(target), status, body)

def _research_lab_sanitize(text):
    text = str(text or "")
    for env_name in (
        "EXA_API_KEY",
        "SCRAPINGDOG_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "OPENROUTER_API_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
    ):
        value = os.environ.get(env_name)
        if value:
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)(api_key=)[^&\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)[^\s]+", r"\1[REDACTED]", text)
    return text

def _research_lab_http_error_details(exc):
    parts = [f"{type(exc).__name__}: {exc}"]
    status = getattr(exc, "code", None) or getattr(exc, "status", None)
    reason = getattr(exc, "reason", None)
    if status is not None:
        parts.append(f"status={status}")
    if reason:
        parts.append(f"reason={reason}")
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read(1200)
            if isinstance(body, bytes):
                body = body.decode("utf-8", "replace")
            body = _research_lab_sanitize(str(body)).replace("\n", " ")[:900]
            if body:
                parts.append(f"body={body}")
        except Exception as body_exc:
            parts.append(f"body_unavailable={type(body_exc).__name__}")
    return "; ".join(parts)

def _research_lab_emit_provider_error(details, target):
    # Single line: the host-side parser matches marker lines individually.
    message = _research_lab_sanitize(f"{details}; url={target}").replace("\n", " ")
    sys.stderr.write("research_lab_private_runtime_provider_error " + message[-900:] + "\n")

# ---------------------------------------------------------------------------
# In-container trace capture: tee every hooked HTTP call (success AND failure)
# to stderr as a single-line marker so the host can collect the sourcing
# model's own provider traffic as training data. Pure observation — emission
# never changes what a hooked call returns, and every capture step is wrapped
# so a failure inside the tee can never break the model's request.
# ---------------------------------------------------------------------------
import base64 as _research_lab_base64
import hashlib as _research_lab_hashlib
import json as _research_lab_json

_research_lab_trace_flag_raw = (os.environ.get("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE") or "").strip().lower()
_research_lab_trace_enabled = True if not _research_lab_trace_flag_raw else _research_lab_trace_flag_raw in ("1", "true", "yes", "on")

def _research_lab_trace_env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default

_research_lab_trace_max_call_bytes = _research_lab_trace_env_int("RESEARCH_LAB_INCONTAINER_TRACE_MAX_CALL_BYTES", 16384)
_research_lab_trace_max_total_bytes = _research_lab_trace_env_int("RESEARCH_LAB_INCONTAINER_TRACE_MAX_TOTAL_BYTES", 524288)
# Evidence recording mode: the run's provider responses seed the replay cache,
# and a truncated body can never be replayed, so recording lifts the capture
# budgets far above any real provider payload instead of truncating.
_research_lab_evidence_record = (
    (os.environ.get("RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD") or "").strip().lower()
    in ("1", "true", "yes", "on")
)
if _research_lab_evidence_record:
    _research_lab_trace_max_call_bytes = max(_research_lab_trace_max_call_bytes, 1 << 27)
    _research_lab_trace_max_total_bytes = max(_research_lab_trace_max_total_bytes, 1 << 30)
# Metadata guard: bodies are budgeted by the caps above, but a pathological
# call loop could still grow stderr through per-entry overhead alone. Stop
# emitting entirely after this many entries (~350B each => ~1.75MB worst case).
_research_lab_trace_max_entries = 5000
if _research_lab_evidence_record:
    _research_lab_trace_max_entries = max(_research_lab_trace_max_entries, 200000)
_research_lab_trace_state = {"seq": 0, "body_bytes_emitted": 0}

def _research_lab_trace_sanitize_text(text):
    text = _research_lab_sanitize(text)
    # Defense in depth for key-shaped strings echoed inside bodies that are
    # not present in this process env.
    return re.sub(r"sk-or-[A-Za-z0-9_\-]+", "[REDACTED]", text)

def _research_lab_trace_provider_class(url):
    # P13: name the provider instead of collapsing everything non-core to
    # "other" — the tool-call sequence is training data and the class is the
    # cheapest queryable signal about what the model reached for.
    try:
        host = ""
        try:
            import urllib.parse as _research_lab_urlparse
            host = (_research_lab_urlparse.urlsplit(str(url)).hostname or "").lower()
        except Exception:
            host = ""
        blob = host or str(url).lower()
        # "exa" needs an exact-host match ("example.com" contains "exa").
        if host in ("exa.ai", "api.exa.ai") or host.endswith(".exa.ai"):
            return "search"
        for marker, provider_class in (
            ("openrouter", "llm"),
            ("api.openai", "llm"),
            ("api.anthropic", "llm"),
            ("generativelanguage.googleapis", "llm"),
            ("serper", "search"),
            ("serpapi", "search"),
            ("googleapis.com/customsearch", "search"),
            ("scrapingdog", "fetch"),
            ("firecrawl", "fetch"),
            ("scrapingbee", "fetch"),
            ("peopledatalabs", "enrichment_pdl"),
            ("pdl.sh", "enrichment_pdl"),
            ("hunter.io", "enrichment_hunter"),
            ("apollo.io", "enrichment_apollo"),
            ("clearbit", "enrichment_clearbit"),
            ("rocketreach", "enrichment_rocketreach"),
            ("prospeo", "enrichment_prospeo"),
            ("findymail", "enrichment_findymail"),
            ("zerobounce", "verification_zerobounce"),
            ("truelist", "verification_truelist"),
            ("linkedin", "linkedin"),
            ("crustdata", "enrichment_crustdata"),
        ):
            if marker in blob:
                return provider_class
        return "other"
    except Exception:
        return "other"

def _research_lab_trace_url(url):
    try:
        text = _research_lab_trace_sanitize_text(str(url))
        text = re.sub(r"://[^/@\s]+@", "://[REDACTED]@", text)
        return text.replace("\n", " ")[:2000]
    except Exception:
        return ""

def _research_lab_trace_body_to_bytes(body):
    # Returns sanitized utf-8 bytes, or None when the body is absent or not
    # capturable without touching a live stream. Bodies are text-decoded with
    # replacement so redaction always runs; provider payloads are JSON/text.
    try:
        if body is None:
            return None
        if isinstance(body, (bytes, bytearray)):
            text = bytes(body).decode("utf-8", "replace")
        else:
            text = str(body)
        return _research_lab_trace_sanitize_text(text).encode("utf-8")
    except Exception:
        return None

def _research_lab_emit_trace(
    method,
    url,
    request_body,
    response_status,
    response_body,
    outcome,
    error_text="",
    request_capture_incomplete=False,
    response_capture_incomplete=False,
    phase="call",
    extra=None,
):
    if not _research_lab_trace_enabled:
        return
    try:
        state = _research_lab_trace_state
        if state["seq"] >= _research_lab_trace_max_entries:
            return
        state["seq"] += 1
        entry = {
            "seq": state["seq"],
            "phase": str(phase),
            "provider_class": _research_lab_trace_provider_class(url),
            "method": str(method or "").upper()[:16],
            "url_redacted": _research_lab_trace_url(url),
            "outcome": str(outcome),
            "error": _research_lab_trace_sanitize_text(str(error_text or "")).replace("\n", " ")[:300],
        }
        truncated = bool(request_capture_incomplete or response_capture_incomplete)
        for label, body in (("request", request_body), ("response", response_body)):
            raw = _research_lab_trace_body_to_bytes(body)
            if raw is None:
                entry[label + "_body_b64"] = ""
                entry[label + "_sha256"] = ""
                entry[label + "_byte_len"] = 0
                continue
            byte_len = len(raw)
            entry[label + "_sha256"] = _research_lab_hashlib.sha256(raw).hexdigest() if byte_len else ""
            entry[label + "_byte_len"] = byte_len
            kept = raw
            if byte_len > _research_lab_trace_max_call_bytes:
                kept = raw[:_research_lab_trace_max_call_bytes]
                truncated = True
            if kept and state["body_bytes_emitted"] + len(kept) > _research_lab_trace_max_total_bytes:
                # Per-run budget exhausted: drop the body, keep hash/length.
                kept = b""
                truncated = True
            state["body_bytes_emitted"] += len(kept)
            entry[label + "_body_b64"] = _research_lab_base64.b64encode(kept).decode("ascii") if kept else ""
        try:
            entry["response_status"] = int(response_status) if response_status is not None else None
        except Exception:
            entry["response_status"] = None
        entry["truncated"] = truncated
        if isinstance(extra, dict):
            for key, value in extra.items():
                if key not in entry:
                    entry[key] = value
        # Single line: base64 has no newlines and json.dumps escapes the rest,
        # so the host-side per-line parser always sees one complete record.
        sys.stderr.write(
            "research_lab_private_runtime_trace "
            + _research_lab_json.dumps(entry, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
    except Exception:
        pass

def _research_lab_urlopen(req, *args, **kwargs):
    target = getattr(req, "full_url", req if isinstance(req, str) else "")
    method = "GET"
    request_body = None
    request_incomplete = False
    try:
        if hasattr(req, "get_method"):
            method = req.get_method()
            raw_data = getattr(req, "data", None)
        else:
            raw_data = args[0] if args else kwargs.get("data")
            if raw_data is not None:
                method = "POST"
        if isinstance(raw_data, (bytes, bytearray, str)):
            request_body = raw_data
        elif raw_data is not None:
            request_incomplete = True
    except Exception:
        request_incomplete = True
    _proxied_url = _research_lab_proxy_rewrite(target) if not request_incomplete else None
    if _proxied_url is not None:
        try:
            _headers = dict(getattr(req, "headers", None) or {})
            _headers.update(_research_lab_proxy_headers())
            req = urllib.request.Request(_proxied_url, data=getattr(req, "data", None) if hasattr(req, "get_method") else (args[0] if args else kwargs.get("data")), headers=_headers, method=method)
            args = ()
            kwargs = {k: v for k, v in kwargs.items() if k != "data"}
            target = _proxied_url
        except Exception:
            _proxied_url = None
    if _research_lab_evidence_cache and _proxied_url is None:
        _cached_record = None
        if not request_incomplete:
            try:
                _cached_record = _research_lab_evidence_cache.get(
                    _research_lab_evidence_fingerprint(method, target, request_body)
                )
            except Exception:
                _cached_record = None
        if _cached_record is not None:
            return _research_lab_serve_cached_response(_cached_record, method, target, request_body)
        # No replayable record (or an unreadable request body): this call is
        # live, so mark it as fresh evidence either way.
        _research_lab_emit_trace(
            method, target, request_body, None, None,
            "cache_miss", "", phase="cache_miss",
        )
    _research_lab_in_urlopen.active = True
    try:
        response = _research_lab_original_urlopen(req, *args, **kwargs)
    except Exception as exc:
        _research_lab_in_urlopen.active = False
        if _research_lab_uses_evidence_transport(_proxied_url, target):
            _research_lab_emit_evidence_marker(getattr(exc, "headers", None), method, target, request_body)
        _research_lab_emit_provider_error(_research_lab_http_error_details(exc), target)
        _research_lab_emit_trace(
            method,
            target,
            request_body,
            getattr(exc, "code", None) or getattr(exc, "status", None),
            None,
            "error",
            "%s: %s" % (type(exc).__name__, exc),
            request_capture_incomplete=request_incomplete,
            response_capture_incomplete=True,
        )
        raise
    _research_lab_in_urlopen.active = False
    if _research_lab_uses_evidence_transport(_proxied_url, target):
        _research_lab_emit_evidence_marker(getattr(response, "headers", None), method, target, request_body)
    if _research_lab_evidence_record:
        _record_status = getattr(response, "status", None) or getattr(response, "code", None)
        try:
            _record_body = response.read()
        except Exception:
            _record_body = None
        if _record_body is not None:
            _research_lab_emit_trace(
                method,
                target,
                request_body,
                _record_status,
                _record_body,
                "success",
                "",
                request_capture_incomplete=request_incomplete,
                response_capture_incomplete=False,
            )
            return _ResearchLabCachedResponse(
                str(target), _record_status, _record_body,
                headers=getattr(response, "headers", None),
            )
    response_body = None
    response_incomplete = True
    try:
        # peek() never consumes the stream the model will read. It returns at
        # most one buffer's worth, so mark the capture complete only when it
        # provably covers the whole body (Content-Length match).
        peeked = response.peek() if hasattr(response, "peek") else b""
        response_body = peeked or b""
        headers = getattr(response, "headers", None)
        content_length = headers.get("Content-Length") if headers is not None else None
        if content_length is not None and int(content_length) == len(response_body):
            response_incomplete = False
    except Exception:
        response_body = None
        response_incomplete = True
    _research_lab_emit_trace(
        method,
        target,
        request_body,
        getattr(response, "status", None) or getattr(response, "code", None),
        response_body,
        "success",
        "",
        request_capture_incomplete=request_incomplete,
        response_capture_incomplete=response_incomplete,
    )
    return response

urllib.request.urlopen = _research_lab_urlopen
_research_lab_install_httpclient_watch()

def _research_lab_generic_error_details(exc):
    # httpx / requests / aiohttp analog of _research_lab_http_error_details:
    # best-effort status/reason/body extraction, never raising.
    parts = [f"{type(exc).__name__}: {exc}"]
    response = getattr(exc, "response", None)
    status = getattr(exc, "code", None) or getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None) or getattr(response, "status", None)
    if status is not None and not callable(status):
        parts.append(f"status={status}")
    reason = getattr(exc, "reason", None)
    if reason is None and response is not None:
        reason = getattr(response, "reason_phrase", None) or getattr(response, "reason", None)
    if reason and not callable(reason):
        parts.append(f"reason={reason}")
    if response is not None:
        try:
            body = getattr(response, "text", None)
            if callable(body):
                # aiohttp's text() is an async method; skip rather than await.
                body = None
            if body is None:
                raw = getattr(response, "content", None)
                if isinstance(raw, bytes):
                    body = raw.decode("utf-8", "replace")
                elif isinstance(raw, str):
                    body = raw
            if isinstance(body, str) and body:
                body = _research_lab_sanitize(body).replace("\n", " ")[:900]
                parts.append(f"body={body}")
        except Exception as body_exc:
            parts.append(f"body_unavailable={type(body_exc).__name__}")
    return "; ".join(parts)

def _research_lab_httpx_request_body(request):
    # request.content raises for streaming bodies; never force-read those.
    try:
        return request.content, False
    except Exception:
        return None, True

def _research_lab_httpx_response_body(response):
    # For stream=False (the default) httpx has already buffered the body, so
    # .content is a cache read. For streaming responses it raises and we skip
    # the body rather than consume the model's stream.
    try:
        return response.content, False
    except Exception:
        return None, True

def _research_lab_patch_httpx():
    # Bug #35: httpx failures previously read as "model returned nothing".
    try:
        import httpx
    except Exception:
        return
    try:
        _research_lab_original_httpx_send = httpx.Client.send

        def _research_lab_httpx_send(self, request, *args, **kwargs):
            request_body, request_incomplete = _research_lab_httpx_request_body(request)
            _proxied_url = _research_lab_proxy_rewrite(getattr(request, "url", ""))
            if _proxied_url is not None:
                try:
                    request.url = httpx.URL(_proxied_url)
                    for _key, _value in _research_lab_proxy_headers().items():
                        request.headers[_key] = _value
                except Exception:
                    _proxied_url = None
            try:
                response = _research_lab_original_httpx_send(self, request, *args, **kwargs)
            except Exception as exc:
                if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                    _research_lab_emit_evidence_marker(getattr(getattr(exc, "response", None), "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(request, "url", ""),
                )
                _research_lab_emit_trace(
                    getattr(request, "method", ""),
                    getattr(request, "url", ""),
                    request_body,
                    None,
                    None,
                    "error",
                    "%s: %s" % (type(exc).__name__, exc),
                    request_capture_incomplete=request_incomplete,
                    response_capture_incomplete=True,
                )
                raise
            response_body, response_incomplete = _research_lab_httpx_response_body(response)
            if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                _research_lab_emit_evidence_marker(getattr(response, "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
            _research_lab_emit_trace(
                getattr(request, "method", ""),
                getattr(request, "url", ""),
                request_body,
                getattr(response, "status_code", None),
                response_body,
                "success",
                "",
                request_capture_incomplete=request_incomplete,
                response_capture_incomplete=response_incomplete,
            )
            return response

        httpx.Client.send = _research_lab_httpx_send
    except Exception:
        pass
    try:
        _research_lab_original_httpx_async_send = httpx.AsyncClient.send

        async def _research_lab_httpx_async_send(self, request, *args, **kwargs):
            request_body, request_incomplete = _research_lab_httpx_request_body(request)
            _proxied_url = _research_lab_proxy_rewrite(getattr(request, "url", ""))
            if _proxied_url is not None:
                try:
                    request.url = httpx.URL(_proxied_url)
                    for _key, _value in _research_lab_proxy_headers().items():
                        request.headers[_key] = _value
                except Exception:
                    _proxied_url = None
            try:
                response = await _research_lab_original_httpx_async_send(self, request, *args, **kwargs)
            except Exception as exc:
                if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                    _research_lab_emit_evidence_marker(getattr(getattr(exc, "response", None), "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(request, "url", ""),
                )
                _research_lab_emit_trace(
                    getattr(request, "method", ""),
                    getattr(request, "url", ""),
                    request_body,
                    None,
                    None,
                    "error",
                    "%s: %s" % (type(exc).__name__, exc),
                    request_capture_incomplete=request_incomplete,
                    response_capture_incomplete=True,
                )
                raise
            response_body, response_incomplete = _research_lab_httpx_response_body(response)
            if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                _research_lab_emit_evidence_marker(getattr(response, "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
            _research_lab_emit_trace(
                getattr(request, "method", ""),
                getattr(request, "url", ""),
                request_body,
                getattr(response, "status_code", None),
                response_body,
                "success",
                "",
                request_capture_incomplete=request_incomplete,
                response_capture_incomplete=response_incomplete,
            )
            return response

        httpx.AsyncClient.send = _research_lab_httpx_async_send
    except Exception:
        pass
    try:
        _research_lab_original_httpx_raise = httpx.Response.raise_for_status

        def _research_lab_httpx_raise_for_status(self, *args, **kwargs):
            try:
                return _research_lab_original_httpx_raise(self, *args, **kwargs)
            except Exception as exc:
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(self, "url", ""),
                )
                raise

        httpx.Response.raise_for_status = _research_lab_httpx_raise_for_status
    except Exception:
        pass

def _research_lab_patch_requests():
    # Bug #35: requests rides urllib3, not urllib.request, so the urlopen hook
    # never saw its failures.
    try:
        import requests
    except Exception:
        return
    try:
        _research_lab_original_requests_send = requests.Session.send

        def _research_lab_requests_send(self, request, *args, **kwargs):
            request_body = getattr(request, "body", None)
            request_incomplete = request_body is not None and not isinstance(
                request_body, (bytes, bytearray, str)
            )
            if request_incomplete:
                # File-like/generator bodies cannot be captured without
                # consuming them.
                request_body = None
            _proxied_url = _research_lab_proxy_rewrite(getattr(request, "url", ""))
            if _proxied_url is not None:
                try:
                    request.url = _proxied_url
                    for _key, _value in _research_lab_proxy_headers().items():
                        request.headers[_key] = _value
                except Exception:
                    _proxied_url = None
            stream = bool(kwargs.get("stream"))
            try:
                response = _research_lab_original_requests_send(self, request, *args, **kwargs)
            except Exception as exc:
                if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                    _research_lab_emit_evidence_marker(getattr(getattr(exc, "response", None), "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(request, "url", ""),
                )
                _research_lab_emit_trace(
                    getattr(request, "method", ""),
                    getattr(request, "url", ""),
                    request_body,
                    None,
                    None,
                    "error",
                    "%s: %s" % (type(exc).__name__, exc),
                    request_capture_incomplete=request_incomplete,
                    response_capture_incomplete=True,
                )
                raise
            response_body = None
            response_incomplete = True
            if not stream:
                # Non-streaming sends already buffered .content inside
                # Session.send, so this is a cache read, never a consume.
                try:
                    response_body = response.content
                    response_incomplete = False
                except Exception:
                    response_body = None
                    response_incomplete = True
            if _research_lab_uses_evidence_transport(_proxied_url, getattr(request, "url", "")):
                _research_lab_emit_evidence_marker(getattr(response, "headers", None), getattr(request, "method", ""), getattr(request, "url", ""), request_body)
            _research_lab_emit_trace(
                getattr(request, "method", ""),
                getattr(request, "url", ""),
                request_body,
                getattr(response, "status_code", None),
                response_body,
                "success",
                "",
                request_capture_incomplete=request_incomplete,
                response_capture_incomplete=response_incomplete,
            )
            return response

        requests.Session.send = _research_lab_requests_send
    except Exception:
        pass
    try:
        _research_lab_original_requests_raise = requests.models.Response.raise_for_status

        def _research_lab_requests_raise_for_status(self, *args, **kwargs):
            try:
                return _research_lab_original_requests_raise(self, *args, **kwargs)
            except Exception as exc:
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(self, "url", ""),
                )
                raise

        requests.models.Response.raise_for_status = _research_lab_requests_raise_for_status
    except Exception:
        pass

def _research_lab_patch_aiohttp():
    # Bug #35: aiohttp failures were equally invisible to the urlopen hook.
    try:
        import aiohttp
    except Exception:
        return
    try:
        _research_lab_original_aiohttp_request = aiohttp.ClientSession._request

        async def _research_lab_aiohttp_request(self, method, str_or_url, *args, **kwargs):
            request_body = None
            request_incomplete = False
            try:
                data = kwargs.get("data")
                json_payload = kwargs.get("json")
                if isinstance(data, (bytes, bytearray, str)):
                    request_body = data
                elif json_payload is not None:
                    request_body = _research_lab_json.dumps(json_payload, sort_keys=True)
                elif data is not None:
                    request_incomplete = True
            except Exception:
                request_incomplete = True
            _proxied_url = _research_lab_proxy_rewrite(str_or_url)
            if _proxied_url is not None:
                str_or_url = _proxied_url
                headers = dict(kwargs.get("headers") or {})
                headers.update(_research_lab_proxy_headers())
                kwargs["headers"] = headers
            try:
                response = await _research_lab_original_aiohttp_request(self, method, str_or_url, *args, **kwargs)
            except Exception as exc:
                if _research_lab_uses_evidence_transport(_proxied_url, str_or_url):
                    _research_lab_emit_evidence_marker(getattr(exc, "headers", None), method, str_or_url, request_body)
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    str_or_url,
                )
                _research_lab_emit_trace(
                    method,
                    str_or_url,
                    request_body,
                    None,
                    None,
                    "error",
                    "%s: %s" % (type(exc).__name__, exc),
                    request_capture_incomplete=request_incomplete,
                    response_capture_incomplete=True,
                )
                raise
            # A real ClientResponse is still streaming here, but the measured
            # V2 transport returns an aiohttp-compatible response whose body is
            # already buffered. Capture that immutable buffer without reading
            # the response. Otherwise the shim response bypasses
            # ClientResponse.read and its successful body never becomes
            # replayable provider evidence.
            response_body = getattr(response, "_body", None)
            if not isinstance(response_body, (bytes, bytearray)):
                response_body = None
            response_incomplete = response_body is None
            if _research_lab_uses_evidence_transport(_proxied_url, str_or_url):
                _research_lab_emit_evidence_marker(getattr(response, "headers", None), method, str_or_url, request_body)
            _research_lab_emit_trace(
                method,
                str_or_url,
                request_body,
                getattr(response, "status", None),
                response_body,
                "success",
                "",
                request_capture_incomplete=request_incomplete,
                response_capture_incomplete=response_incomplete,
            )
            return response

        aiohttp.ClientSession._request = _research_lab_aiohttp_request
    except Exception:
        pass
    try:
        _research_lab_original_aiohttp_read = aiohttp.ClientResponse.read

        async def _research_lab_aiohttp_read(self, *args, **kwargs):
            # aiohttp caches the payload in _body on first read; only the first
            # materialization emits, and the returned bytes are teed after the
            # fact so the model sees exactly what it would have seen.
            already_cached = getattr(self, "_body", None) is not None
            body = await _research_lab_original_aiohttp_read(self, *args, **kwargs)
            if not already_cached:
                _research_lab_emit_trace(
                    getattr(self, "method", ""),
                    getattr(self, "url", ""),
                    None,
                    getattr(self, "status", None),
                    body,
                    "success",
                    "",
                    phase="response_body",
                )
            return body

        aiohttp.ClientResponse.read = _research_lab_aiohttp_read
    except Exception:
        pass
    try:
        _research_lab_original_aiohttp_raise = aiohttp.ClientResponse.raise_for_status

        def _research_lab_aiohttp_raise_for_status(self, *args, **kwargs):
            try:
                return _research_lab_original_aiohttp_raise(self, *args, **kwargs)
            except Exception as exc:
                _research_lab_emit_provider_error(
                    _research_lab_generic_error_details(exc),
                    getattr(self, "url", ""),
                )
                raise

        aiohttp.ClientResponse.raise_for_status = _research_lab_aiohttp_raise_for_status
    except Exception:
        pass

_research_lab_patch_httpx()
_research_lab_patch_requests()
_research_lab_patch_aiohttp()

def _research_lab_emit_trace_bootstrap():
    # P13/P19: assert at bootstrap which client libraries the trace tee
    # actually patched, so an unhooked client (e.g. a new SDK doing its own
    # streaming) is a visible capture gap instead of a silent one. One line,
    # parsed host-side alongside the per-call trace markers.
    if not _research_lab_trace_enabled:
        return
    try:
        patched = {"urllib": True}
        for lib, probe in (
            ("httpx", lambda: __import__("httpx").Client.send.__name__),
            ("requests", lambda: __import__("requests").Session.send.__name__),
            ("aiohttp", lambda: __import__("aiohttp").ClientSession._request.__name__),
        ):
            try:
                patched[lib] = probe().startswith("_research_lab")
            except Exception:
                patched[lib] = None  # library not installed in this image
        sys.stderr.write(
            "research_lab_private_runtime_capture_bootstrap "
            + _research_lab_json.dumps(
                {
                    "schema_version": "1.0",
                    "patched_clients": patched,
                    "max_call_bytes": _research_lab_trace_max_call_bytes,
                    "max_total_bytes": _research_lab_trace_max_total_bytes,
                    "streaming_posture": "request_only_for_streams",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except Exception:
        pass

_research_lab_emit_trace_bootstrap()

def _research_lab_patch_strict_qualify(adapter_module):
    try:
        import asyncio
        import sourcing_model
        import sourcing_model.core as core
    except Exception:
        return

    def _strict_qualify(icp):
        if not isinstance(icp, dict) or not (icp.get("industry") or icp.get("intent_signal_text") or icp.get("intent_signal")):
            return []
        try:
            import sourcing_model.clients as clients
            has_keys = clients.has_keys()
        except Exception:
            try:
                has_keys = bool(core._exa_key() and core._sd_key() and core._or_key())
            except Exception:
                has_keys = False
        if not has_keys:
            missing = []
            if not os.environ.get("EXA_API_KEY"):
                missing.append("EXA_API_KEY")
            if not (os.environ.get("SCRAPINGDOG_API_KEY") or os.environ.get("QUALIFICATION_SCRAPINGDOG_API_KEY")):
                missing.append("SCRAPINGDOG_API_KEY")
            if not (
                os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
                or os.environ.get("OPENROUTER_KEY")
            ):
                missing.append("OPENROUTER_API_KEY")
            raise RuntimeError("Missing API keys for private sourcing model: " + ", ".join(missing))
        return asyncio.run(core._qualify_async(icp))

    try:
        sourcing_model.qualify = _strict_qualify
        if hasattr(adapter_module, "qualify"):
            adapter_module.qualify = _strict_qualify
    except Exception as exc:
        message = _research_lab_sanitize(f"{type(exc).__name__}: {exc}")
        sys.stderr.write("research_lab_private_runtime_diagnostic_warning strict_qualify_patch_failed " + message[-500:] + "\n")
"""


_ADAPTER_BOOTSTRAP = _PROVIDER_DIAGNOSTICS_BOOTSTRAP + r"""
import contextlib
import importlib
import json
import sys
import time

source_path, module_name, callable_name = sys.argv[1:4]
sys.path.insert(0, source_path)
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
try:
    from sourcing_model import runtime_capabilities as _sourcing_caps
    _runtime_options = dict((payload.get("context") or {}).get("runtime_options") or {})
    _runtime_cap = max(10.0, float(_runtime_options.get("runtime_cap_seconds") or 900.0))
    _runtime_reserve = max(0.0, float(_runtime_options.get("finalization_reserve_seconds") or 0.0))
    _runtime_deadline = time.monotonic() + max(1.0, _runtime_cap - _runtime_reserve)
    _sourcing_caps.register("deadline", lambda: _runtime_deadline)
    _sourcing_caps.register("resolve_host", lambda _name: _sourcing_caps.HostResolution.TIMEOUT)
    _sourcing_caps.register("probe_origin", lambda _host: _sourcing_caps.OriginReachability.UNKNOWN)
    def _emit_sourcing_event(event):
        if isinstance(event, dict):
            sys.stderr.write("sourcing_runtime_event " + json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    _sourcing_caps.register("emit", _emit_sourcing_event)
except ImportError:
    # A non-sourcing adapter may still fail before returning. Successful Lab
    # sourcing calls are rejected host-side if they do not emit the receipt.
    pass
except (AttributeError, TypeError, ValueError) as exc:
    raise RuntimeError("failed to register sourcing runtime capabilities") from exc
_research_lab_patch_strict_qualify(module)
fn = getattr(module, callable_name)
with contextlib.redirect_stdout(sys.stderr):
    result = fn(payload["icp"], payload.get("context") or {})
sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


_ADAPTER_METADATA_BOOTSTRAP = r"""
import importlib
import json
import sys

source_path, module_name, callable_name = sys.argv[1:4]
sys.path.insert(0, source_path)
module = importlib.import_module(module_name)
fn = getattr(module, callable_name)
result = fn()
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


_DOCKER_ADAPTER_BOOTSTRAP = _PROVIDER_DIAGNOSTICS_BOOTSTRAP + r"""
import contextlib
import importlib
import json
import logging
import sys
import time

for _research_lab_logger_name in (
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "aiohttp",
    "openai",
):
    logging.getLogger(_research_lab_logger_name).setLevel(logging.WARNING)

module_name, callable_name = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
try:
    from sourcing_model import runtime_capabilities as _sourcing_caps
    _runtime_options = dict((payload.get("context") or {}).get("runtime_options") or {})
    _runtime_cap = max(10.0, float(_runtime_options.get("runtime_cap_seconds") or 900.0))
    _runtime_reserve = max(0.0, float(_runtime_options.get("finalization_reserve_seconds") or 0.0))
    _runtime_deadline = time.monotonic() + max(1.0, _runtime_cap - _runtime_reserve)
    _sourcing_caps.register("deadline", lambda: _runtime_deadline)
    _sourcing_caps.register("resolve_host", lambda _name: _sourcing_caps.HostResolution.TIMEOUT)
    _sourcing_caps.register("probe_origin", lambda _host: _sourcing_caps.OriginReachability.UNKNOWN)
    def _emit_sourcing_event(event):
        if isinstance(event, dict):
            sys.stderr.write("sourcing_runtime_event " + json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    _sourcing_caps.register("emit", _emit_sourcing_event)
except ImportError:
    # A non-sourcing adapter may still fail before returning. Successful Lab
    # sourcing calls are rejected host-side if they do not emit the receipt.
    pass
except (AttributeError, TypeError, ValueError) as exc:
    raise RuntimeError("failed to register sourcing runtime capabilities") from exc
_research_lab_patch_strict_qualify(module)
fn = getattr(module, callable_name)
with contextlib.redirect_stdout(sys.stderr):
    result = fn(payload["icp"], payload.get("context") or {})
sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""


_DOCKER_METADATA_BOOTSTRAP = r"""
import importlib
import json
import sys

module_name, callable_name = sys.argv[1:3]
module = importlib.import_module(module_name)
fn = getattr(module, callable_name)
result = fn()
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""
