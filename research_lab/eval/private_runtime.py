"""Runtime helpers for private Research Lab model artifacts.

The public subnet repo must not import private champion code directly. These
helpers load a private model adapter from an immutable artifact checkout/image
boundary and execute it through a small JSON contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_bytes, sha256_json

SECRET_MARKERS = (
    "sk-or-",
    "sb_secret_",
    "aws_secret_access_key",
    "openrouter_api_key",
    "scrapingdog_api_key",
    "exa_api_key",
    "raw_secret",
    "service_role",
)
PROVIDER_ERROR_MARKER = "research_lab_private_runtime_provider_error"
DEFAULT_DOCKER_PLATFORM = "linux/amd64"
DEFAULT_ENV_PASSTHROUGH = (
    "EXA_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
PROVIDER_KEY_ENV_PASSTHROUGH = (
    "EXA_API_KEY",
    "SCRAPINGDOG_API_KEY",
    "QUALIFICATION_SCRAPINGDOG_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)
PROVIDER_PROXY_ENV_PASSTHROUGH = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)
LINKEDIN_EMPLOYEE_BUCKETS = (
    "0-1",
    "2-10",
    "11-50",
    "51-200",
    "201-500",
    "501-1,000",
    "1,001-5,000",
    "5,001-10,000",
    "10,001+",
)
LEGACY_EMPLOYEE_BUCKET_MAP = {
    "10-50": "11-50",
    "50-200": "51-200",
    "200-500": "201-500",
    "500-1000": "501-1,000",
    "501-1000": "501-1,000",
    "1000-5000": "1,001-5,000",
    "1001-5000": "1,001-5,000",
    "5000-10000": "5,001-10,000",
    "5001-10000": "5,001-10,000",
    "5000+": "5,001-10,000",
    "10000+": "10,001+",
    "10001+": "10,001+",
}


class PrivateModelRuntimeError(RuntimeError):
    """Raised when the private model artifact cannot be executed safely."""


def private_model_env_passthrough(*, include_proxy: bool = False) -> tuple[str, ...]:
    """Provider env names for private model containers.

    Worker processes may carry global Webshare proxy vars for their own network
    path. Do not implicitly forward those into provider-backed model containers:
    Exa, ScrapingDog, and OpenRouter are API services, and ScrapingDog performs
    its own upstream proxying. Operators can opt in when testing a proxy-specific
    provider path.
    """
    if include_proxy:
        return PROVIDER_KEY_ENV_PASSTHROUGH + PROVIDER_PROXY_ENV_PASSTHROUGH
    return PROVIDER_KEY_ENV_PASSTHROUGH


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
    employee_count = _normalize_employee_count_bucket(
        _first_text(
            normalized.get("employee_count"),
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
    normalized["employee_count"] = employee_count
    normalized["product_service"] = product_service or required_attribute
    normalized["required_attribute"] = required_attribute
    normalized["intent_signal"] = intent_signal
    normalized["intent_signal_text"] = _first_text(normalized.get("intent_signal_text"), intent_signal)
    normalized["intent_signals"] = _intent_signals_list(normalized.get("intent_signals"), intent_signal)
    normalized["intent_category"] = _intent_category(normalized.get("intent_category"), intent_signal)
    normalized["intent_max_age_days"] = _positive_int(normalized.get("intent_max_age_days"), default=365)
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
                "employee_count": employee_count,
                "product_service": normalized["product_service"],
                "intent_signal": intent_signal,
            }
        )[:18]

    if not normalized["industry"] or not normalized["intent_signal"]:
        raise PrivateModelRuntimeError("private model ICP is missing industry or intent signal after canonicalization")
    return normalized


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
        payload = {"icp": canonicalize_private_model_icp(icp), "context": _redacted_context(context)}
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
            raise PrivateModelRuntimeError("private model adapter timed out") from exc

        if completed.returncode != 0:
            stderr = _sanitize_text(completed.stderr)[-1200:]
            raise PrivateModelRuntimeError(f"private model adapter failed with code {completed.returncode}: {stderr}")

        try:
            decoded = _loads_adapter_stdout(completed.stdout)
        except json.JSONDecodeError as exc:
            stdout = _sanitize_text(completed.stdout)[-800:]
            stderr = _sanitize_text(completed.stderr)[-800:]
            raise PrivateModelRuntimeError(
                f"private model adapter returned invalid JSON: stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        _raise_on_empty_provider_error(decoded, completed.stderr, context_label="private model")
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
        return decoded


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
        payload = {"icp": canonicalize_private_model_icp(icp), "context": _redacted_context(context)}
        decoded = self._run_json(
            bootstrap=_DOCKER_ADAPTER_BOOTSTRAP,
            argv=(self.spec.module_name, self.spec.callable_name),
            stdin_payload=payload,
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
        return decoded

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
    ) -> Any:
        command = [
            self.spec.docker_executable,
            "run",
            "--rm",
            "-i",
            *_docker_platform_args(self.spec),
            *_docker_env_args(self.spec),
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
            raise PrivateModelRuntimeError("docker private model adapter timed out") from exc
        if completed.returncode != 0:
            stderr = _sanitize_text(completed.stderr)[-1200:]
            raise PrivateModelRuntimeError(f"docker private model adapter failed with code {completed.returncode}: {stderr}")
        try:
            decoded = _loads_adapter_stdout(completed.stdout)
        except json.JSONDecodeError as exc:
            stdout = _sanitize_text(completed.stdout)[-800:]
            stderr = _sanitize_text(completed.stderr)[-800:]
            raise PrivateModelRuntimeError(
                f"docker private model adapter returned invalid JSON: stdout={stdout!r} stderr={stderr!r}"
            ) from exc
        _raise_on_empty_provider_error(decoded, completed.stderr, context_label="docker private model")
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
) -> dict[str, Any]:
    """Build the same manifest shape private CI publishes.

    Official production manifests must point at an immutable private ECR digest
    and S3 URI. This helper exists so the CI script and local verification use
    one canonical hash shape.
    """
    payload = {
        "model_artifact_hash": compute_private_source_tree_hash(source_path),
        "git_commit_sha": str(git_commit_sha),
        "image_digest": str(image_digest),
        "config_hash": sha256_json(config_payload or {}),
        "component_registry_version": str(component_registry_version),
        "scoring_adapter_version": str(scoring_adapter_version),
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


def _raise_on_empty_provider_error(decoded: Any, stderr: str, *, context_label: str) -> None:
    if decoded != []:
        return
    provider_error = _provider_error_text(stderr)
    if not provider_error:
        return
    raise PrivateModelRuntimeError(
        f"{context_label} provider-backed sourcing failed before returning companies: "
        f"{provider_error}"
    )


def _provider_error_text(stderr: str) -> str:
    sanitized = _sanitize_text(stderr)
    lines = [line.strip() for line in sanitized.splitlines() if PROVIDER_ERROR_MARKER in line]
    if not lines:
        return ""
    return "\n".join(lines)[-1200:]


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
    raw = " ".join(str(value or "").strip().split())
    if raw in LINKEDIN_EMPLOYEE_BUCKETS:
        return raw
    key = raw.replace(",", "")
    normalized = LEGACY_EMPLOYEE_BUCKET_MAP.get(key) or LEGACY_EMPLOYEE_BUCKET_MAP.get(raw)
    if normalized:
        return normalized
    return default


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
    return " ".join(
        part
        for part in (
            f"{icp.get('industry', '')} companies",
            f"in {icp.get('geography', '')}" if icp.get("geography") else "",
            f"with {icp.get('employee_count', '')} employees" if icp.get("employee_count") else "",
            f"that {str(icp.get('required_attribute', '')).removeprefix('The company ').strip()}"
            if icp.get("required_attribute")
            else "",
            f"showing intent: {icp.get('intent_signal', '')}" if icp.get("intent_signal") else "",
        )
        if part
    )


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

def _research_lab_urlopen(req, *args, **kwargs):
    try:
        return _research_lab_original_urlopen(req, *args, **kwargs)
    except Exception as exc:
        target = getattr(req, "full_url", req if isinstance(req, str) else "")
        message = _research_lab_sanitize(f"{_research_lab_http_error_details(exc)}; url={target}")
        sys.stderr.write("research_lab_private_runtime_provider_error " + message[-900:] + "\n")
        raise

urllib.request.urlopen = _research_lab_urlopen

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
        if not (core._exa_key() and core._sd_key() and core._or_key()):
            raise RuntimeError("Missing API keys for private sourcing model")
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

source_path, module_name, callable_name = sys.argv[1:4]
sys.path.insert(0, source_path)
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
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
import sys

module_name, callable_name = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
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
