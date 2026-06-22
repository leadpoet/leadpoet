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


class PrivateModelRuntimeError(RuntimeError):
    """Raised when the private model artifact cannot be executed safely."""


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
        payload = {"icp": dict(icp), "context": _redacted_context(context)}
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
            decoded = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PrivateModelRuntimeError("private model adapter returned invalid JSON") from exc
        if not isinstance(decoded, list):
            raise PrivateModelRuntimeError("private model adapter must return a JSON array")
        if _contains_secret_material(decoded):
            raise PrivateModelRuntimeError("private model adapter returned raw secret material")
        return decoded

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
        payload = {"icp": dict(icp), "context": _redacted_context(context)}
        decoded = self._run_json(
            bootstrap=_DOCKER_ADAPTER_BOOTSTRAP,
            argv=(self.spec.module_name, self.spec.callable_name),
            stdin_payload=payload,
        )
        if not isinstance(decoded, list):
            raise PrivateModelRuntimeError("private model adapter must return a JSON array")
        if _contains_secret_material(decoded):
            raise PrivateModelRuntimeError("private model adapter returned raw secret material")
        return decoded

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
            [self.spec.docker_executable, "pull", self.spec.image_digest],
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
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PrivateModelRuntimeError("docker private model adapter returned invalid JSON") from exc


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


def _redacted_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_secret_material(context):
        raise PrivateModelRuntimeError("run context contains raw secret material")
    return dict(context)


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
    for marker in SECRET_MARKERS:
        sanitized = sanitized.replace(marker, "[REDACTED]")
    return sanitized


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise PrivateModelRuntimeError(f"expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, sep, key = without_scheme.partition("/")
    if not bucket or not sep or not key:
        raise PrivateModelRuntimeError(f"invalid s3 URI: {uri}")
    return bucket, key


_ADAPTER_BOOTSTRAP = r"""
import importlib
import json
import sys

source_path, module_name, callable_name = sys.argv[1:4]
sys.path.insert(0, source_path)
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
fn = getattr(module, callable_name)
result = fn(payload["icp"], payload.get("context") or {})
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
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


_DOCKER_ADAPTER_BOOTSTRAP = r"""
import importlib
import json
import sys

module_name, callable_name = sys.argv[1:3]
payload = json.load(sys.stdin)
module = importlib.import_module(module_name)
fn = getattr(module, callable_name)
result = fn(payload["icp"], payload.get("context") or {})
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
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
