"""Safely migrate the production gateway secret to V2 worker proxy profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import shlex
from typing import Any, Callable, Mapping
import uuid

from gateway.research_lab.config import (
    LEGACY_SCORING_PROXY_PREFIXES,
)
from gateway.tee.provider_broker_v2 import _validated_tls_proxy_url
from gateway.tee.proxy_transport_preflight_v2 import (
    verify_worker_proxy_fleets_v2,
)


DEFAULT_SECRET_ID = "leadpoet/prod/gateway/env"
DEFAULT_BACKUP_DIRECTORY = Path(
    "/home/ec2-user/.config/leadpoet/env-backups"
)
_TARGET_ENVIRONMENT = {
    "gateway_scoring": {
        "proxy": "RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1",
        "count": "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT",
        "legacy_prefixes": LEGACY_SCORING_PROXY_PREFIXES,
    },
}
_TARGET_NAMES = frozenset(
    str(configuration[field])
    for configuration in _TARGET_ENVIRONMENT.values()
    for field in ("proxy", "count")
)
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GatewayV2ProxySecretUpdateError(RuntimeError):
    """The production secret could not be migrated without losing state."""


def _configured_proxies(
    environment: Mapping[str, str], prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    """Read legacy proxy slots without importing the retired worker supervisor."""

    proxies: list[str] = []
    seen: set[str] = set()
    for index in range(1, 501):
        for prefix in prefixes:
            value = str(environment.get(f"{prefix}_{index}", "")).strip()
            if value and value not in seen:
                proxies.append(value)
                seen.add(value)
                break
    for prefix in prefixes:
        value = str(environment.get(prefix, "")).strip()
        if value and value not in seen:
            proxies.append(value)
            seen.add(value)
    return tuple(proxies)


def _parse_shell_environment(raw: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise GatewayV2ProxySecretUpdateError(
                "gateway secret environment is malformed"
            ) from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise GatewayV2ProxySecretUpdateError(
                "gateway secret environment is malformed"
            )
        name, value = parts[0].split("=", 1)
        if not _ENVIRONMENT_NAME_RE.fullmatch(name):
            raise GatewayV2ProxySecretUpdateError(
                "gateway secret environment contains an invalid name"
            )
        parsed[name] = value
    return parsed


def _parse_environment(raw: str) -> tuple[dict[str, str], str]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return _parse_shell_environment(raw), "shell"
    if not isinstance(decoded, Mapping):
        raise GatewayV2ProxySecretUpdateError(
            "gateway secret JSON must contain an object"
        )
    return {str(name): str(value) for name, value in decoded.items()}, "json"


def _render_updated_environment(
    raw: str,
    *,
    document_format: str,
    values: Mapping[str, str],
) -> str:
    if document_format == "json":
        decoded = json.loads(raw)
        decoded.update(values)
        return json.dumps(decoded, sort_keys=True, separators=(",", ":"))

    kept_lines = []
    for raw_line in raw.splitlines():
        candidate = raw_line.strip()
        if candidate.startswith("export "):
            candidate = candidate[len("export ") :].strip()
        name = candidate.split("=", 1)[0] if "=" in candidate else ""
        if name in _TARGET_NAMES:
            continue
        kept_lines.append(raw_line)
    kept_lines.extend(
        "export %s=%s" % (name, shlex.quote(value))
        for name, value in sorted(values.items())
    )
    return "\n".join(kept_lines).rstrip() + "\n"


def _worker_count(
    environment: Mapping[str, str],
    *,
    count_environment: str,
    legacy_prefixes: tuple[str, ...],
) -> int:
    raw_count = str(environment.get(count_environment) or "").strip()
    if raw_count:
        if not raw_count.isdigit() or not 1 <= int(raw_count) <= 500:
            raise GatewayV2ProxySecretUpdateError(
                "%s must be an integer from 1 through 500"
                % count_environment
            )
        return int(raw_count)
    legacy_count = len(_configured_proxies(environment, legacy_prefixes))
    if not 1 <= legacy_count <= 500:
        raise GatewayV2ProxySecretUpdateError(
            "%s is absent and no legacy worker capacity can be preserved"
            % count_environment
        )
    return legacy_count


def _validated_proxy_values(
    *,
    scoring_proxy: str,
    proxy_fleet_probe: Callable[..., None],
) -> dict[str, str]:
    values = {
        "gateway_scoring": str(scoring_proxy or "").strip(),
    }
    for role, value in values.items():
        try:
            _validated_tls_proxy_url(value)
        except Exception as exc:
            raise GatewayV2ProxySecretUpdateError(
                "%s proxy must be a complete HTTP CONNECT or HTTPS proxy URL"
                % role
            ) from exc
    try:
        proxy_fleet_probe(
            {
                role: (value,)
                for role, value in values.items()
            }
        )
    except Exception as exc:
        raise GatewayV2ProxySecretUpdateError(
            "V2 worker proxy live CONNECT validation failed"
        ) from exc
    return values


def _write_backup(
    raw_secret: str,
    *,
    backup_directory: Path,
    now: datetime,
) -> Path:
    backup_directory = Path(backup_directory)
    backup_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(backup_directory, 0o700)
    timestamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = backup_directory / (
        "gateway-secret.before-v2-proxy.%s.%s.env"
        % (timestamp, uuid.uuid4().hex[:12])
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(raw_secret)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _secret_string(response: Mapping[str, Any]) -> str:
    value = response.get("SecretString")
    if not isinstance(value, str):
        raise GatewayV2ProxySecretUpdateError(
            "gateway secret is not stored as text"
        )
    return value


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
        raise GatewayV2ProxySecretUpdateError(
            "candidate verification failed and automatic secret restoration "
            "also failed; use the protected local backup"
        ) from exc


def update_gateway_v2_proxy_secret(
    *,
    secrets_client: Any,
    secret_id: str,
    backup_directory: Path,
    scoring_proxy: str,
    proxy_fleet_probe: Callable[..., None] = verify_worker_proxy_fleets_v2,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate and atomically version the gateway secret without exposing it."""

    initial_response = secrets_client.get_secret_value(SecretId=secret_id)
    initial_version = str(initial_response.get("VersionId") or "")
    if not initial_version:
        raise GatewayV2ProxySecretUpdateError(
            "gateway secret version identity is unavailable"
        )
    initial_secret = _secret_string(initial_response)
    environment, document_format = _parse_environment(initial_secret)
    proxies = _validated_proxy_values(
        scoring_proxy=scoring_proxy,
        proxy_fleet_probe=proxy_fleet_probe,
    )

    target_values: dict[str, str] = {}
    worker_counts: dict[str, int] = {}
    for role, configuration in _TARGET_ENVIRONMENT.items():
        count = _worker_count(
            environment,
            count_environment=str(configuration["count"]),
            legacy_prefixes=configuration["legacy_prefixes"],
        )
        worker_counts[role] = count
        target_values[str(configuration["proxy"])] = proxies[role]
        target_values[str(configuration["count"])] = str(count)

    candidate_secret = _render_updated_environment(
        initial_secret,
        document_format=document_format,
        values=target_values,
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
        raise GatewayV2ProxySecretUpdateError(
            "candidate gateway secret changes unrelated environment values"
        )
    if any(
        candidate_environment.get(name) != value
        for name, value in target_values.items()
    ):
        raise GatewayV2ProxySecretUpdateError(
            "candidate gateway secret did not preserve the validated proxy state"
        )

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
        raise GatewayV2ProxySecretUpdateError(
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
        stages = (
            description.get("VersionIdsToStages", {})
            .get(candidate_token, [])
        )
        if (
            _secret_string(persisted) != candidate_secret
            or "AWSCURRENT" not in stages
        ):
            raise GatewayV2ProxySecretUpdateError(
                "persisted gateway secret failed exact readback verification"
            )
    except Exception as exc:
        description = secrets_client.describe_secret(SecretId=secret_id)
        stages = (
            description.get("VersionIdsToStages", {})
            .get(candidate_token, [])
        )
        if "AWSCURRENT" in stages:
            _restore_prior_secret(
                secrets_client,
                secret_id=secret_id,
                prior_secret=initial_secret,
            )
        if isinstance(exc, GatewayV2ProxySecretUpdateError):
            raise
        raise GatewayV2ProxySecretUpdateError(
            "persisted gateway secret could not be verified"
        ) from exc

    return {
        "backup_path": str(backup_path),
        "secret_id": secret_id,
        "worker_counts": worker_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install validated V2 worker proxies in the production gateway "
            "secret without displaying credentials."
        )
    )
    parser.add_argument("--secret-id", default=DEFAULT_SECRET_ID)
    parser.add_argument(
        "--backup-directory",
        type=Path,
        default=DEFAULT_BACKUP_DIRECTORY,
    )
    args = parser.parse_args()

    scoring_proxy = getpass.getpass(
        "Enter RESEARCH_LAB_V2_SCORING_HTTPS_PROXY_1: "
    )

    import boto3

    result = update_gateway_v2_proxy_secret(
        secrets_client=boto3.client("secretsmanager", region_name="us-east-1"),
        secret_id=str(args.secret_id),
        backup_directory=args.backup_directory,
        scoring_proxy=scoring_proxy,
    )
    print(
        "Gateway V2 proxy secret updated and read back successfully; "
        "scoring_workers=%d backup=%s"
        % (
            result["worker_counts"]["gateway_scoring"],
            result["backup_path"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
