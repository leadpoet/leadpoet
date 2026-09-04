"""Run the public PydanticAI baseline with ordinary process boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping
import uuid

from qualification.competition_models import validate_companies
from gateway.utils.docker_lifecycle import (
    DockerLifecycleError,
    shared_docker_lifecycle,
)


logger = logging.getLogger(__name__)

BASELINE_ID = "leadpoet/pydantic-harness"
BASELINE_REPOSITORY = "https://github.com/leadpoet/pydantic-harness.git"
BASELINE_BRANCH = "main"
BASELINE_ENTRYPOINT = "harness.run_icp"
RESULT_SENTINEL = "PYDANTIC_HARNESS_RESULT_JSON="
PROVIDER_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "DEEPLINE_API_KEY",
    "SCRAPINGDOG_API_KEY",
)
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 12 * 60
DEFAULT_BUILD_TIMEOUT_SECONDS = 15 * 60
MAX_COMPANIES = 5
MAX_OUTPUT_BYTES = 2_000_000


class PublicBaselineRunError(RuntimeError):
    """The public baseline could not complete a bounded run."""


CommandExecutor = Callable[[list[str], str, float], tuple[int, str, str]]


def _subprocess_execute(
    argv: list[str], input_text: str, timeout_seconds: float
) -> tuple[int, str, str]:
    completed = subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _parse_result(stdout: str) -> dict[str, Any]:
    if len(stdout.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise PublicBaselineRunError("baseline output exceeded the size limit")
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        if not line.startswith(RESULT_SENTINEL):
            continue
        try:
            value = json.loads(line[len(RESULT_SENTINEL) :])
        except json.JSONDecodeError as exc:
            raise PublicBaselineRunError("baseline returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise PublicBaselineRunError("baseline result must be an object")
        return value
    raise PublicBaselineRunError("baseline result marker is missing")


def _redact_error(value: str) -> str:
    text = str(value or "")
    for env_name in PROVIDER_ENV_NAMES:
        secret = str(os.getenv(env_name, "") or "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:2000]


def _cleanup_container(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning(
            "public_baseline_container_cleanup_failed container=%s",
            container_name,
        )


@dataclass(frozen=True)
class PublicBaselineExecution:
    companies: list[dict[str, Any]]
    usage: dict[str, Any]
    provider_calls: list[dict[str, Any]]
    provider_cost_usd: float
    model_cost_usd: float | None
    combined_cost_usd: float | None
    latency_seconds: float


class PublicBaselineDockerRunner:
    """Build the public branch once, then start one clean container per ICP."""

    def __init__(
        self,
        *,
        execute: CommandExecutor | None = None,
        attempt_timeout_seconds: int = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        build_timeout_seconds: int = DEFAULT_BUILD_TIMEOUT_SECONDS,
    ) -> None:
        self._execute = execute or _subprocess_execute
        self._uses_real_docker = execute is None
        self.attempt_timeout_seconds = max(1, int(attempt_timeout_seconds))
        self.build_timeout_seconds = max(1, int(build_timeout_seconds))
        self.image_tag = "leadpoet-public-baseline-" + uuid.uuid4().hex
        self._built = False
        self._model = ""
        self._model_pricing: dict[str, Any] = {}

    def ensure_image(self) -> None:
        if self._built:
            return
        dockerfile_dir = Path(__file__).with_name("public_baseline_image")
        argv = [
            "docker",
            "build",
            "--pull",
            "--no-cache",
            "--build-arg",
            f"BASELINE_REPOSITORY={BASELINE_REPOSITORY}",
            "--build-arg",
            f"BASELINE_BRANCH={BASELINE_BRANCH}",
            "-t",
            self.image_tag,
            str(dockerfile_dir),
        ]
        lifecycle = (
            shared_docker_lifecycle(timeout_seconds=self.build_timeout_seconds)
            if self._uses_real_docker
            else nullcontext()
        )
        try:
            with lifecycle:
                code, _stdout, stderr = self._execute(
                    argv, "", float(self.build_timeout_seconds)
                )
        except (DockerLifecycleError, subprocess.TimeoutExpired) as exc:
            raise PublicBaselineRunError("public baseline image build timed out") from exc
        if code != 0:
            raise PublicBaselineRunError(
                "public baseline image build failed: " + _redact_error(stderr)
            )
        self._built = True

    def _run_container(
        self, command: list[str], payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.ensure_image()
        container_name = "leadpoet-public-baseline-run-" + uuid.uuid4().hex
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "256",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
        ]
        for env_name in PROVIDER_ENV_NAMES:
            if os.getenv(env_name):
                argv.extend(("-e", env_name))
        argv.extend(("-e", "HOME=/tmp"))
        argv.append(self.image_tag)
        argv.extend(command)
        lifecycle = (
            shared_docker_lifecycle(timeout_seconds=self.attempt_timeout_seconds)
            if self._uses_real_docker
            else nullcontext()
        )
        try:
            with lifecycle:
                code, stdout, stderr = self._execute(
                    argv,
                    json.dumps(dict(payload), ensure_ascii=False, allow_nan=False),
                    float(self.attempt_timeout_seconds),
                )
        except (DockerLifecycleError, subprocess.TimeoutExpired) as exc:
            if self._uses_real_docker:
                _cleanup_container(container_name)
            raise PublicBaselineRunError("public baseline container timed out") from exc
        except BaseException:
            if self._uses_real_docker:
                _cleanup_container(container_name)
            raise
        result = _parse_result(stdout)
        if code != 0 or not bool(result.get("ok")):
            detail = str(result.get("error") or stderr or f"container exit {code}")
            raise PublicBaselineRunError(_redact_error(detail))
        return result

    def preflight(self) -> dict[str, Any]:
        result = self._run_container(["preflight"], {})
        model = str(result.get("selected_model") or "").strip()
        pricing = result.get("model_pricing")
        if not model or not isinstance(pricing, Mapping):
            raise PublicBaselineRunError("public baseline preflight returned no model")
        self._model = model
        self._model_pricing = dict(pricing)
        return {
            "model": self._model,
            "provider_status": dict(result.get("deepline") or {}),
        }

    def run_icp(
        self,
        icp: Mapping[str, Any],
        *,
        evaluation_date: str,
        max_companies: int = MAX_COMPANIES,
    ) -> PublicBaselineExecution:
        if not self._model:
            raise PublicBaselineRunError("public baseline preflight is required")
        company_limit = min(MAX_COMPANIES, max(1, int(max_companies)))
        result = self._run_container(
            [
                "run",
                "--model",
                self._model,
                "--model-pricing-json",
                json.dumps(self._model_pricing, separators=(",", ":")),
                "--evaluation-date",
                str(evaluation_date),
                "--max-companies",
                str(company_limit),
            ],
            dict(icp),
        )
        raw_companies = result.get("companies")
        if not isinstance(raw_companies, list) or len(raw_companies) > MAX_COMPANIES:
            raise PublicBaselineRunError("baseline returned an invalid company list")
        try:
            companies = validate_companies(
                raw_companies,
                max_companies=MAX_COMPANIES,
            )
        except (TypeError, ValueError) as exc:
            raise PublicBaselineRunError(
                "baseline output contains a non-public URL or invalid field"
            ) from exc
        usage = result.get("usage")
        provider_calls = result.get("provider_calls")
        if not isinstance(usage, Mapping) or not isinstance(provider_calls, list):
            raise PublicBaselineRunError("baseline returned invalid usage data")
        return PublicBaselineExecution(
            companies=companies,
            usage=dict(usage),
            provider_calls=[dict(value) for value in provider_calls if isinstance(value, Mapping)],
            provider_cost_usd=float(
                result.get("estimated_provider_cost_usd") or 0.0
            ),
            model_cost_usd=(
                float(result["model_cost_usd"])
                if result.get("model_cost_usd") is not None
                else None
            ),
            combined_cost_usd=(
                float(result["estimated_combined_cost_usd"])
                if result.get("estimated_combined_cost_usd") is not None
                else None
            ),
            latency_seconds=float(result.get("latency_seconds") or 0.0),
        )

    def close(self) -> None:
        if not self._built or not self._uses_real_docker:
            return
        try:
            subprocess.run(
                ["docker", "image", "rm", "-f", self.image_tag],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            logger.warning("public_baseline_image_cleanup_failed image=%s", self.image_tag)
        finally:
            self._built = False

    def __enter__(self) -> "PublicBaselineDockerRunner":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()
