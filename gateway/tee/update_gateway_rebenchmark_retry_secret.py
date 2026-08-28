"""Apply narrow production rebenchmark runtime configuration changes.

The retry operation remains intentionally fixed: it can only move provider
retry rounds from its default 1 (absent) or explicit 1 to explicit 2, and retry
concurrency from its default 2 (absent) or explicit 2 to explicit 1.  The
first-pass concurrency operation accepts an exact expected old value and a
bounded new value.  Both operations preserve every unrelated secret field,
verify the exact persisted version, and restore the prior version if readback
fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Mapping
import uuid

from gateway.tee.scoring_executor import (
    SCORING_RUNTIME_ENV_NAMES,
    configuration_hash as scoring_configuration_hash,
)


DEFAULT_SECRET_ID = "leadpoet/prod/gateway/env"
DEFAULT_BACKUP_DIRECTORY = Path(
    "/home/ec2-user/.config/leadpoet/env-backups"
)
_RETRY_ROUNDS_ENV = "RESEARCH_LAB_BENCHMARK_PROVIDER_RETRY_ROUNDS"
_RETRY_CONCURRENCY_ENV = "RESEARCH_LAB_BENCHMARK_RETRY_CONCURRENCY"
_TARGET_VALUES = {
    _RETRY_ROUNDS_ENV: "2",
    _RETRY_CONCURRENCY_ENV: "1",
}
_TARGET_NAMES = frozenset(_TARGET_VALUES)
_FIRST_PASS_CONCURRENCY_ENV = "RESEARCH_LAB_BENCHMARK_CONCURRENCY"
_MAX_FIRST_PASS_CONCURRENCY = 64
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class GatewayRebenchmarkRetryUpdateError(RuntimeError):
    """The retry configuration could not be changed without losing state."""


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for raw_name, value in pairs:
        name = str(raw_name)
        if name in decoded:
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway secret JSON contains a duplicate name"
            )
        decoded[name] = value
    return decoded


def _parse_shell_environment(
    raw: str,
    *,
    target_names: frozenset[str] = _TARGET_NAMES,
) -> dict[str, str]:
    """Parse restart-hydrated KEY=VALUE records as data, never shell code."""

    parsed: dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = [line]
        assignment = parts[0] if len(parts) == 1 else line
        if "=" not in assignment:
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway secret environment is malformed"
            )
        name, value = assignment.split("=", 1)
        name = name.strip()
        if not _ENVIRONMENT_NAME_RE.fullmatch(name):
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway secret environment contains an invalid name"
            )
        if name in target_names and name in parsed:
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway secret environment contains a duplicate target setting"
            )
        parsed[name] = value
    return parsed


def _parse_environment(
    raw: str,
    *,
    target_names: frozenset[str] = _TARGET_NAMES,
) -> tuple[dict[str, str], str]:
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_json_object_without_duplicates,
        )
    except json.JSONDecodeError:
        return _parse_shell_environment(
            raw,
            target_names=target_names,
        ), "shell"
    if not isinstance(decoded, Mapping):
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret JSON must contain an object"
        )
    return {str(name): str(value) for name, value in decoded.items()}, "json"


def _render_environment(
    raw: str,
    *,
    document_format: str,
    values: Mapping[str, str],
    target_names: frozenset[str] = _TARGET_NAMES,
) -> str:
    if document_format == "json":
        decoded = json.loads(
            raw,
            object_pairs_hook=_json_object_without_duplicates,
        )
        decoded.update(values)
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"))

    kept_records = []
    records = re.split(r"(\r\n|\n|\r|\x00)", raw)
    for index in range(0, len(records), 2):
        raw_line = records[index]
        separator = records[index + 1] if index + 1 < len(records) else ""
        candidate = raw_line.strip()
        if candidate.startswith("export "):
            candidate = candidate[len("export ") :].strip()
        name = (
            candidate.split("=", 1)[0].strip()
            if "=" in candidate
            else ""
        )
        if name not in target_names:
            kept_records.append(raw_line + separator)
    rendered = "".join(kept_records)
    if rendered and not rendered.endswith(("\n", "\r", "\x00")):
        rendered += "\n"
    rendered += "\n".join(
        "export %s=%s" % (name, shlex.quote(value))
        for name, value in sorted(values.items())
    )
    return rendered + "\n"


def _secret_string(response: Mapping[str, Any]) -> str:
    value = response.get("SecretString")
    if not isinstance(value, str):
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret is not stored as text"
        )
    return value


def _write_backup(
    raw_secret: str,
    *,
    backup_directory: Path,
    now: datetime,
    operation_label: str = "rebenchmark-retry-extension",
) -> Path:
    backup_directory = Path(backup_directory)
    backup_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(backup_directory, 0o700)
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = backup_directory / (
        "gateway-secret.before-%s.%s.%s.env"
        % (operation_label, timestamp, uuid.uuid4().hex[:12])
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw_secret)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _restore_prior_secret(
    secrets_client: Any,
    *,
    secret_id: str,
    prior_secret: str,
) -> None:
    try:
        secrets_client.put_secret_value(
            SecretId=secret_id,
            SecretString=prior_secret,
            ClientRequestToken=str(uuid.uuid4()),
        )
    except Exception as exc:
        raise GatewayRebenchmarkRetryUpdateError(
            "candidate verification failed and automatic secret restoration "
            "also failed; use the protected local backup"
        ) from exc


def _configuration_hash(environment: Mapping[str, str]) -> str:
    return scoring_configuration_hash(
        {name: environment.get(name) for name in SCORING_RUNTIME_ENV_NAMES}
    )


def update_gateway_rebenchmark_retry_secret(
    *,
    secrets_client: Any,
    expected_prior_scoring_configuration_hash: str,
    apply: bool = False,
    secret_id: str = DEFAULT_SECRET_ID,
    backup_directory: Path = DEFAULT_BACKUP_DIRECTORY,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify or apply the fixed 1→2 extension against a checkpoint hash."""

    expected_prior_hash = str(
        expected_prior_scoring_configuration_hash or ""
    ).strip().lower()
    if not _SHA256_RE.fullmatch(expected_prior_hash):
        raise GatewayRebenchmarkRetryUpdateError(
            "expected prior scoring configuration hash is invalid"
        )

    initial_response = secrets_client.get_secret_value(SecretId=secret_id)
    initial_version = str(initial_response.get("VersionId") or "")
    if not initial_version:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret version identity is unavailable"
        )
    initial_secret = _secret_string(initial_response)
    environment, document_format = _parse_environment(initial_secret)

    current_rounds = environment.get(_RETRY_ROUNDS_ENV)
    current_concurrency = environment.get(_RETRY_CONCURRENCY_ENV)
    if (
        current_rounds == _TARGET_VALUES[_RETRY_ROUNDS_ENV]
        and current_concurrency == _TARGET_VALUES[_RETRY_CONCURRENCY_ENV]
    ):
        prior_candidates = []
        for prior_rounds in (None, "1"):
            prior_environment = dict(environment)
            if prior_rounds is None:
                prior_environment.pop(_RETRY_ROUNDS_ENV, None)
            else:
                prior_environment[_RETRY_ROUNDS_ENV] = prior_rounds
            prior_candidates.append(_configuration_hash(prior_environment))
        if expected_prior_hash not in prior_candidates:
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway scoring configuration does not match the expected "
                "checkpoint hash"
            )
        return {
            "status": "already_applied",
            "secret_id": secret_id,
            "scoring_configuration_hash": _configuration_hash(environment),
        }
    if current_rounds not in {None, "1"}:
        raise GatewayRebenchmarkRetryUpdateError(
            "provider retry rounds must have the expected old value or be unset"
        )
    if current_concurrency not in {None, "2"}:
        raise GatewayRebenchmarkRetryUpdateError(
            "retry concurrency must have the expected old value or be unset"
        )

    prior_scoring_hash = _configuration_hash(environment)
    if prior_scoring_hash != expected_prior_hash:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway scoring configuration does not match the expected "
            "checkpoint hash"
        )
    candidate_secret = _render_environment(
        initial_secret,
        document_format=document_format,
        values=_TARGET_VALUES,
    )
    candidate_environment, _ = _parse_environment(candidate_secret)
    before_unrelated = {
        name: value
        for name, value in environment.items()
        if name not in _TARGET_NAMES
    }
    after_unrelated = {
        name: value
        for name, value in candidate_environment.items()
        if name not in _TARGET_NAMES
    }
    if before_unrelated != after_unrelated:
        raise GatewayRebenchmarkRetryUpdateError(
            "candidate gateway secret changes unrelated environment values"
        )
    if any(
        candidate_environment.get(name) != value
        for name, value in _TARGET_VALUES.items()
    ):
        raise GatewayRebenchmarkRetryUpdateError(
            "candidate gateway secret did not preserve the fixed retry state"
        )
    current_scoring_hash = _configuration_hash(candidate_environment)

    if not apply:
        return {
            "status": "verified",
            "secret_id": secret_id,
            "prior_scoring_configuration_hash": prior_scoring_hash,
            "current_scoring_configuration_hash": current_scoring_hash,
        }

    backup_path = _write_backup(
        initial_secret,
        backup_directory=backup_directory,
        now=now or datetime.now(timezone.utc),
    )
    current_response = secrets_client.get_secret_value(SecretId=secret_id)
    if (
        str(current_response.get("VersionId") or "") != initial_version
        or _secret_string(current_response) != initial_secret
    ):
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret changed concurrently; no update was applied"
        )

    candidate_token = str(uuid.uuid4())
    secrets_client.put_secret_value(
        SecretId=secret_id,
        SecretString=candidate_secret,
        ClientRequestToken=candidate_token,
    )
    try:
        persisted = secrets_client.get_secret_value(
            SecretId=secret_id,
            VersionId=candidate_token,
        )
        description = secrets_client.describe_secret(SecretId=secret_id)
        stages = description.get("VersionIdsToStages", {}).get(candidate_token, [])
        if (
            _secret_string(persisted) != candidate_secret
            or "AWSCURRENT" not in stages
        ):
            raise GatewayRebenchmarkRetryUpdateError(
                "persisted gateway secret failed exact readback verification"
            )
    except Exception as exc:
        description = secrets_client.describe_secret(SecretId=secret_id)
        stages = description.get("VersionIdsToStages", {}).get(candidate_token, [])
        if "AWSCURRENT" in stages:
            _restore_prior_secret(
                secrets_client,
                secret_id=secret_id,
                prior_secret=initial_secret,
            )
        if isinstance(exc, GatewayRebenchmarkRetryUpdateError):
            raise
        raise GatewayRebenchmarkRetryUpdateError(
            "persisted gateway secret could not be verified"
        ) from exc

    return {
        "status": "updated",
        "secret_id": secret_id,
        "backup_path": str(backup_path),
        "prior_scoring_configuration_hash": prior_scoring_hash,
        "current_scoring_configuration_hash": current_scoring_hash,
    }


def _bounded_first_pass_concurrency(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise GatewayRebenchmarkRetryUpdateError(f"{label} is invalid")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GatewayRebenchmarkRetryUpdateError(f"{label} is invalid") from exc
    if not 1 <= normalized <= _MAX_FIRST_PASS_CONCURRENCY:
        raise GatewayRebenchmarkRetryUpdateError(
            f"{label} must be between 1 and {_MAX_FIRST_PASS_CONCURRENCY}"
        )
    return normalized


def update_gateway_rebenchmark_concurrency_secret(
    *,
    secrets_client: Any,
    expected_prior_scoring_configuration_hash: str,
    expected_current_concurrency: int,
    target_concurrency: int,
    apply: bool = False,
    secret_id: str = DEFAULT_SECRET_ID,
    backup_directory: Path = DEFAULT_BACKUP_DIRECTORY,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify or apply one exact, bounded first-pass concurrency change."""

    expected_prior_hash = str(
        expected_prior_scoring_configuration_hash or ""
    ).strip().lower()
    if not _SHA256_RE.fullmatch(expected_prior_hash):
        raise GatewayRebenchmarkRetryUpdateError(
            "expected prior scoring configuration hash is invalid"
        )
    expected_concurrency = _bounded_first_pass_concurrency(
        expected_current_concurrency,
        label="expected current first-pass concurrency",
    )
    target = _bounded_first_pass_concurrency(
        target_concurrency,
        label="target first-pass concurrency",
    )
    target_names = frozenset({_FIRST_PASS_CONCURRENCY_ENV})

    initial_response = secrets_client.get_secret_value(SecretId=secret_id)
    initial_version = str(initial_response.get("VersionId") or "")
    if not initial_version:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret version identity is unavailable"
        )
    initial_secret = _secret_string(initial_response)
    environment, document_format = _parse_environment(
        initial_secret,
        target_names=target_names,
    )
    raw_current = environment.get(_FIRST_PASS_CONCURRENCY_ENV)
    try:
        current_concurrency = int(raw_current) if raw_current is not None else 1
    except (TypeError, ValueError, OverflowError) as exc:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway first-pass concurrency is invalid"
        ) from exc
    if not 1 <= current_concurrency <= _MAX_FIRST_PASS_CONCURRENCY:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway first-pass concurrency is outside the supported bound"
        )

    if current_concurrency == target:
        prior_candidates: list[str] = []
        explicit_prior = dict(environment)
        explicit_prior[_FIRST_PASS_CONCURRENCY_ENV] = str(expected_concurrency)
        prior_candidates.append(_configuration_hash(explicit_prior))
        if expected_concurrency == 1:
            default_prior = dict(environment)
            default_prior.pop(_FIRST_PASS_CONCURRENCY_ENV, None)
            prior_candidates.append(_configuration_hash(default_prior))
        if expected_prior_hash not in prior_candidates:
            raise GatewayRebenchmarkRetryUpdateError(
                "gateway scoring configuration does not match the expected "
                "checkpoint hash"
            )
        return {
            "status": "already_applied",
            "secret_id": secret_id,
            "prior_first_pass_concurrency": expected_concurrency,
            "current_first_pass_concurrency": target,
            "scoring_configuration_hash": _configuration_hash(environment),
        }
    if current_concurrency != expected_concurrency:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway first-pass concurrency does not match the expected old value"
        )

    prior_scoring_hash = _configuration_hash(environment)
    if prior_scoring_hash != expected_prior_hash:
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway scoring configuration does not match the expected "
            "checkpoint hash"
        )
    candidate_secret = _render_environment(
        initial_secret,
        document_format=document_format,
        values={_FIRST_PASS_CONCURRENCY_ENV: str(target)},
        target_names=target_names,
    )
    candidate_environment, _ = _parse_environment(
        candidate_secret,
        target_names=target_names,
    )
    before_unrelated = {
        name: value
        for name, value in environment.items()
        if name not in target_names
    }
    after_unrelated = {
        name: value
        for name, value in candidate_environment.items()
        if name not in target_names
    }
    if before_unrelated != after_unrelated:
        raise GatewayRebenchmarkRetryUpdateError(
            "candidate gateway secret changes unrelated environment values"
        )
    if candidate_environment.get(_FIRST_PASS_CONCURRENCY_ENV) != str(target):
        raise GatewayRebenchmarkRetryUpdateError(
            "candidate gateway secret did not preserve the target concurrency"
        )
    current_scoring_hash = _configuration_hash(candidate_environment)

    if not apply:
        return {
            "status": "verified",
            "secret_id": secret_id,
            "prior_first_pass_concurrency": expected_concurrency,
            "current_first_pass_concurrency": target,
            "prior_scoring_configuration_hash": prior_scoring_hash,
            "current_scoring_configuration_hash": current_scoring_hash,
        }

    backup_path = _write_backup(
        initial_secret,
        backup_directory=backup_directory,
        now=now or datetime.now(timezone.utc),
        operation_label="rebenchmark-concurrency",
    )
    current_response = secrets_client.get_secret_value(SecretId=secret_id)
    if (
        str(current_response.get("VersionId") or "") != initial_version
        or _secret_string(current_response) != initial_secret
    ):
        raise GatewayRebenchmarkRetryUpdateError(
            "gateway secret changed concurrently; no update was applied"
        )

    candidate_token = str(uuid.uuid4())
    secrets_client.put_secret_value(
        SecretId=secret_id,
        SecretString=candidate_secret,
        ClientRequestToken=candidate_token,
    )
    try:
        persisted = secrets_client.get_secret_value(
            SecretId=secret_id,
            VersionId=candidate_token,
        )
        description = secrets_client.describe_secret(SecretId=secret_id)
        stages = description.get("VersionIdsToStages", {}).get(candidate_token, [])
        if (
            _secret_string(persisted) != candidate_secret
            or "AWSCURRENT" not in stages
        ):
            raise GatewayRebenchmarkRetryUpdateError(
                "persisted gateway secret failed exact readback verification"
            )
    except Exception as exc:
        description = secrets_client.describe_secret(SecretId=secret_id)
        stages = description.get("VersionIdsToStages", {}).get(candidate_token, [])
        if "AWSCURRENT" in stages:
            _restore_prior_secret(
                secrets_client,
                secret_id=secret_id,
                prior_secret=initial_secret,
            )
        if isinstance(exc, GatewayRebenchmarkRetryUpdateError):
            raise
        raise GatewayRebenchmarkRetryUpdateError(
            "persisted gateway secret could not be verified"
        ) from exc

    return {
        "status": "updated",
        "secret_id": secret_id,
        "backup_path": str(backup_path),
        "prior_first_pass_concurrency": expected_concurrency,
        "current_first_pass_concurrency": target,
        "prior_scoring_configuration_hash": prior_scoring_hash,
        "current_scoring_configuration_hash": current_scoring_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the fixed production rebenchmark retry extension without "
            "displaying any secret values."
        )
    )
    parser.add_argument("--secret-id", default=DEFAULT_SECRET_ID)
    parser.add_argument(
        "--expected-prior-scoring-configuration-hash",
        required=True,
        help="Exact full durable checkpoint configuration hash.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the verified new secret version; the default is read-only.",
    )
    parser.add_argument(
        "--backup-directory",
        type=Path,
        default=DEFAULT_BACKUP_DIRECTORY,
    )
    parser.add_argument(
        "--expected-current-first-pass-concurrency",
        type=int,
        help=(
            "Exact current first-pass concurrency for a bounded concurrency "
            "change."
        ),
    )
    parser.add_argument(
        "--target-first-pass-concurrency",
        type=int,
        help=(
            "Bounded first-pass concurrency to persist instead of applying "
            "the fixed retry extension."
        ),
    )
    args = parser.parse_args()

    import boto3

    secrets_client = boto3.client("secretsmanager", region_name="us-east-1")
    if args.target_first_pass_concurrency is not None:
        if args.expected_current_first_pass_concurrency is None:
            parser.error(
                "--expected-current-first-pass-concurrency is required with "
                "--target-first-pass-concurrency"
            )
        result = update_gateway_rebenchmark_concurrency_secret(
            secrets_client=secrets_client,
            expected_prior_scoring_configuration_hash=str(
                args.expected_prior_scoring_configuration_hash
            ),
            expected_current_concurrency=(
                args.expected_current_first_pass_concurrency
            ),
            target_concurrency=args.target_first_pass_concurrency,
            apply=bool(args.apply),
            secret_id=str(args.secret_id),
            backup_directory=args.backup_directory,
        )
    else:
        if args.expected_current_first_pass_concurrency is not None:
            parser.error(
                "--expected-current-first-pass-concurrency requires "
                "--target-first-pass-concurrency"
            )
        result = update_gateway_rebenchmark_retry_secret(
            secrets_client=secrets_client,
            expected_prior_scoring_configuration_hash=str(
                args.expected_prior_scoring_configuration_hash
            ),
            apply=bool(args.apply),
            secret_id=str(args.secret_id),
            backup_directory=args.backup_directory,
        )
    if "current_first_pass_concurrency" in result:
        print(
            "Gateway rebenchmark first-pass concurrency %s; "
            "prior=%s current=%s scoring_configuration_hash=%s%s"
            % (
                result["status"],
                result["prior_first_pass_concurrency"],
                result["current_first_pass_concurrency"],
                result.get("current_scoring_configuration_hash")
                or result["scoring_configuration_hash"],
                (
                    " backup=%s" % result["backup_path"]
                    if result.get("backup_path")
                    else ""
                ),
            )
        )
        return 0
    if result["status"] == "already_applied":
        print(
            "Gateway rebenchmark retry extension was already applied; "
            "scoring_configuration_hash=%s"
            % result["scoring_configuration_hash"]
        )
    elif result["status"] == "verified":
        print(
            "Gateway rebenchmark retry extension verified without writing; "
            "prior_scoring_configuration_hash=%s "
            "current_scoring_configuration_hash=%s"
            % (
                result["prior_scoring_configuration_hash"],
                result["current_scoring_configuration_hash"],
            )
        )
    else:
        print(
            "Gateway rebenchmark retry extension updated and read back; "
            "prior_scoring_configuration_hash=%s "
            "current_scoring_configuration_hash=%s backup=%s"
            % (
                result["prior_scoring_configuration_hash"],
                result["current_scoring_configuration_hash"],
                result["backup_path"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
