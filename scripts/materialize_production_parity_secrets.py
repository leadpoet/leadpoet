#!/usr/bin/env python3
"""Create run-scoped staging secrets without exposing their values.

The materializer copies candidate-independent runtime/provider configuration
from production, but it never copies production persistence, chain, wallet, or
service identity boundaries. Those values are generated for the run or must be
present in the staging overlay secret.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import boto3
from botocore.exceptions import BotoCoreError, ClientError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leadpoet_canonical.production_parity_wallet import (
    ProductionParityWalletError,
    install_base,
    normalize_spec,
)
from leadpoet_canonical.production_parity_epoch_authority import (
    ProductionParityEpochAuthorityError,
    cutover_manifest_path,
    normalize_spec as normalize_epoch_authority_spec,
)


SCHEMA_VERSION = "leadpoet.production_parity_infra.v1"
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STATIC_AWS_KEYS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
}
FORCED_BOUNDARY_KEYS = {
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "BITTENSOR_NETWORK",
    "BITTENSOR_NETUID",
    "SUBTENSOR_NETWORK",
    "NETUID",
    "EXPECTED_CHAIN",
    "GATEWAY_URL",
    "VALIDATOR_V2_GATEWAY_URL",
    "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON",
    "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH",
    "LEADPOET_SENTRY_API_TOKEN",
    "LEADPOET_SENTRY_DSN",
    "LEADPOET_SENTRY_ENABLED",
    "GATEWAY_OTEL_ENABLED",
    "GATEWAY_OTEL_ENDPOINT",
    "GATEWAY_OTEL_METRICS_ENDPOINT",
    "GATEWAY_OTEL_TOKEN",
    "GITHUB_TOKEN",
    "LEADPOET_INTERNAL_SECRET",
    "RESEARCH_LAB_INTERNAL_API_KEY",
    "LEADPOET_PRODUCTION_PARITY_MODE",
    "LEADPOET_PRODUCTION_PARITY_RUN_ID",
    "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN",
    "LEADPOET_PRODUCTION_PARITY_CHAIN_HOST",
    "LEADPOET_PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST",
    "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_API_MANAGEMENT_KEY",
    "OR_MANAGEMENT_KEY",
}
WRITE_BOUNDARY_RE = re.compile(
    r"(?:^|_)(?:ARWEAVE|BUCKET|DATABASE|DB|DYNAMO|ENDPOINT|HOST|HOTKEY|KMS|"
    r"PRIVATE_KEY|QUEUE|REDIS|STREAM|TABLE|TOPIC|URL|WALLET)(?:_|$)"
)


class SecretMaterializationError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SecretMaterializationError("parity infrastructure config is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise SecretMaterializationError("parity infrastructure config schema differs")
    return value


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SecretMaterializationError(f"{field} must be an object")
    return dict(value)


def _parse_environment_document(raw: str, *, field: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    values: dict[str, str] = {}
    if isinstance(parsed, Mapping):
        items = parsed.items()
    else:
        rows: list[tuple[str, str]] = []
        for raw_line in raw.replace("\x00", "\n").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise SecretMaterializationError(f"{field} contains a non-assignment")
            key, value = line.split("=", 1)
            raw_value = value.strip()
            try:
                tokens = shlex.split(raw_value, comments=False, posix=True)
            except ValueError as exc:
                raise SecretMaterializationError(
                    f"{field} contains invalid shell quoting"
                ) from exc
            if not tokens and not raw_value:
                parsed_value = ""
            elif len(tokens) == 1:
                parsed_value = tokens[0]
            else:
                raise SecretMaterializationError(
                    f"{field} contains an unquoted multi-token value"
                )
            rows.append((key.strip(), parsed_value))
        items = rows
    for raw_key, raw_value in items:
        key = str(raw_key or "").strip()
        if not ENV_KEY_RE.fullmatch(key) or key in values:
            raise SecretMaterializationError(f"{field} contains an invalid key")
        if isinstance(raw_value, (dict, list)):
            value = json.dumps(raw_value, sort_keys=True, separators=(",", ":"))
        elif raw_value is None:
            value = ""
        else:
            value = str(raw_value)
        values[key] = value
    if not values:
        raise SecretMaterializationError(f"{field} is empty")
    return values


def _secret_string(client: Any, secret_id: str, *, field: str) -> str:
    value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    if not isinstance(value, str) or not value:
        raise SecretMaterializationError(f"{field} has no string value")
    return value


def _jwt(secret: str, role: str) -> str:
    now = int(time.time())
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    header = encode(b'{"alg":"HS256","typ":"JWT"}')
    payload = encode(
        json.dumps(
            {
                "aud": "authenticated",
                "exp": now + 86400 * 30,
                "iat": now - 5,
                "iss": "leadpoet-production-parity",
                "role": role,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = encode(hmac.new(secret.encode("ascii"), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _text_set(value: Any, *, field: str) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        raise SecretMaterializationError(f"{field} must be a list")
    result = {str(item or "").strip() for item in value}
    if "" in result or len(result) != len(value):
        raise SecretMaterializationError(f"{field} must contain unique nonempty values")
    return result


def _build_runtime(
    *,
    source: Mapping[str, str],
    overlay: Mapping[str, Any],
    generated: Mapping[str, str],
    allowed_production_boundaries: set[str],
    required_overlay_keys: set[str],
    field: str,
) -> dict[str, str]:
    normalized_overlay = {
        str(key): (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else "" if value is None else str(value)
        )
        for key, value in overlay.items()
    }
    if any(not ENV_KEY_RE.fullmatch(key) for key in normalized_overlay):
        raise SecretMaterializationError(f"{field} overlay contains an invalid key")
    result = {key: value for key, value in source.items() if key not in STATIC_AWS_KEYS}
    result.update(normalized_overlay)
    result.update(generated)
    missing = sorted(required_overlay_keys - set(normalized_overlay) - set(generated))
    if missing:
        raise SecretMaterializationError(
            f"{field} staging overlay is missing required keys: {','.join(missing)}"
        )
    unsafe = sorted(
        key
        for key in source
        if key not in STATIC_AWS_KEYS
        and (
            key in FORCED_BOUNDARY_KEYS
            or (
                WRITE_BOUNDARY_RE.search(key)
                and key not in allowed_production_boundaries
            )
        )
        and key not in normalized_overlay
        and key not in generated
    )
    if unsafe:
        raise SecretMaterializationError(
            f"{field} would inherit unclassified production boundaries: {','.join(unsafe)}"
        )
    for key in FORCED_BOUNDARY_KEYS & set(source):
        if key in result and result[key] == source[key]:
            raise SecretMaterializationError(f"{field} retained production boundary {key}")
    return dict(sorted(result.items()))


def _create_secret(
    client: Any,
    *,
    name: str,
    value: Mapping[str, str],
    kms_key_id: str,
    run_id: str,
    candidate_sha: str,
) -> None:
    client.create_secret(
        Name=name,
        Description=f"Disposable Leadpoet parity runtime {run_id}",
        KmsKeyId=kms_key_id,
        SecretString=json.dumps(dict(value), sort_keys=True, separators=(",", ":")),
        Tags=[
            {"Key": "leadpoet:parity-run", "Value": run_id},
            {"Key": "leadpoet:candidate-sha", "Value": candidate_sha},
        ],
    )


def _secret_names(run_id: str) -> dict[str, str]:
    if not RUN_RE.fullmatch(run_id):
        raise SecretMaterializationError("secret run identity is invalid")
    prefix = f"leadpoet/staging/production-parity/{run_id}"
    return {
        "gateway": prefix + "/gateway",
        "validator": prefix + "/validator",
        "database": prefix + "/database",
        "auditor-a": prefix + "/auditor-a",
        "auditor-b": prefix + "/auditor-b",
        "dashboard": prefix + "/dashboard",
    }


def _cleanup_state(
    *,
    run_id: str,
    candidate_sha: str,
    region: str,
) -> dict[str, Any]:
    if not SHA_RE.fullmatch(candidate_sha) or not region:
        raise SecretMaterializationError("secret cleanup identity is invalid")
    return {
        "schema_version": "leadpoet.production_parity_secret_state.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "region": region,
        "secret_ids": _secret_names(run_id),
        "status": "creating",
    }


def _delete_secret(client: Any, name: str, *, attempts: int = 5) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            client.delete_secret(SecretId=name, ForceDeleteWithoutRecovery=True)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return False
            if attempt == attempts:
                raise
        except BotoCoreError:
            if attempt == attempts:
                raise
        time.sleep(min(2 ** (attempt - 1), 8))
    raise SecretMaterializationError("secret cleanup retry budget exhausted")


def materialize(
    *,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    candidate_sha: str,
    client: Any,
) -> dict[str, Any]:
    run_id = str(state.get("run_id") or "")
    if not RUN_RE.fullmatch(run_id) or not SHA_RE.fullmatch(candidate_sha):
        raise SecretMaterializationError("parity run or candidate identity is invalid")
    if state.get("candidate_sha") != candidate_sha:
        raise SecretMaterializationError("stack and secret candidate identities differ")
    outputs = _mapping(state.get("outputs"), field="stack outputs")
    gateway_domain = str(outputs.get("GatewayDomain") or "").strip()
    database_domain = str(outputs.get("DatabaseDomain") or "").strip()
    dashboard_domain = str(outputs.get("DashboardDomain") or "").strip()
    if not gateway_domain or not database_domain or not dashboard_domain:
        raise SecretMaterializationError("stack DNS outputs are incomplete")
    settings = _mapping(config.get("secret_materialization"), field="secret_materialization")
    runtime = _mapping(config.get("runtime_metadata"), field="runtime_metadata")
    wallet_settings = _mapping(
        config.get("wallet_artifacts"), field="wallet_artifacts"
    )
    epoch_authority_settings = _mapping(
        config.get("epoch_authority_artifact"),
        field="epoch_authority_artifact",
    )
    source_gateway_id = str(settings.get("production_gateway_secret_id") or "")
    source_validator_id = str(settings.get("production_validator_secret_id") or "")
    overlay_id = str(settings.get("staging_overlay_secret_id") or "")
    kms_key_id = str(settings.get("staging_kms_key_id") or "")
    if not all((source_gateway_id, source_validator_id, overlay_id, kms_key_id)):
        raise SecretMaterializationError("secret materialization identities are incomplete")
    gateway_source = _parse_environment_document(
        _secret_string(client, source_gateway_id, field="production gateway secret"),
        field="production gateway secret",
    )
    validator_source = _parse_environment_document(
        _secret_string(client, source_validator_id, field="production validator secret"),
        field="production validator secret",
    )
    overlay_doc = json.loads(_secret_string(client, overlay_id, field="staging overlay secret"))
    overlay = _mapping(overlay_doc, field="staging overlay secret")
    gateway_overlay = _mapping(overlay.get("gateway"), field="staging gateway overlay")
    validator_overlay = _mapping(overlay.get("validator"), field="staging validator overlay")
    dashboard_overlay = _mapping(
        overlay.get("dashboard"), field="staging dashboard overlay"
    )
    auditor_overlays = overlay.get("auditors")
    if not isinstance(auditor_overlays, list) or len(auditor_overlays) != 2:
        raise SecretMaterializationError(
            "staging overlay requires exactly two auditor environments"
        )

    jwt_secret = secrets.token_urlsafe(48)
    postgres_password = secrets.token_urlsafe(36)
    authenticator_password = secrets.token_urlsafe(36)
    internal_secret = secrets.token_urlsafe(48)
    research_lab_internal_key = secrets.token_urlsafe(48)
    supabase_url = f"https://{database_domain}"
    gateway_url = f"https://{gateway_domain}"
    network = str(runtime.get("network") or "test")
    netuid = str(int(runtime.get("netuid") or 1))
    chain_endpoint = str(runtime.get("testnet_chain_endpoint") or "").strip()
    parsed_chain = urlsplit(chain_endpoint)
    chain_host = str(parsed_chain.hostname or "").lower()
    if (
        network != "test"
        or parsed_chain.scheme != "wss"
        or parsed_chain.port not in (None, 443)
        or chain_host != "test.finney.opentensor.ai"
        or parsed_chain.path not in ("", "/")
        or parsed_chain.query
        or parsed_chain.fragment
    ):
        raise SecretMaterializationError("staging testnet boundary is invalid")
    try:
        epoch_authority_spec = normalize_epoch_authority_spec(
            epoch_authority_settings,
            network=network,
            netuid=int(netuid),
        )
    except ProductionParityEpochAuthorityError as exc:
        raise SecretMaterializationError(
            "staging testnet epoch authority is invalid"
        ) from exc
    raw_auditor_wallets = wallet_settings.get("audit_validators")
    if not isinstance(raw_auditor_wallets, list) or len(raw_auditor_wallets) != 2:
        raise SecretMaterializationError(
            "exactly two staging audit wallet artifacts are required"
        )
    try:
        wallet_specs = {
            "primary-validator": normalize_spec(
                _mapping(
                    wallet_settings.get("primary_validator"),
                    field="primary validator wallet artifact",
                ),
                role="primary-validator",
                network=network,
                netuid=int(netuid),
            ),
            "auditor-a": normalize_spec(
                _mapping(
                    raw_auditor_wallets[0], field="auditor A wallet artifact"
                ),
                role="auditor-a",
                network=network,
                netuid=int(netuid),
            ),
            "auditor-b": normalize_spec(
                _mapping(
                    raw_auditor_wallets[1], field="auditor B wallet artifact"
                ),
                role="auditor-b",
                network=network,
                netuid=int(netuid),
            ),
        }
    except ProductionParityWalletError as exc:
        raise SecretMaterializationError(
            "staging wallet artifact configuration is invalid"
        ) from exc
    staging_hotkeys = {
        str(item["expected_hotkey"]) for item in wallet_specs.values()
    }
    production_hotkeys = {
        value
        for source in (gateway_source, validator_source)
        for key, value in source.items()
        if "HOTKEY" in key and re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{40,64}", value)
    }
    if len(staging_hotkeys) != 3 or staging_hotkeys & production_hotkeys:
        raise SecretMaterializationError(
            "staging and production hotkey identities are not isolated"
        )
    generated_common = {
        "SUPABASE_URL": supabase_url,
        "SUPABASE_ANON_KEY": _jwt(jwt_secret, "anon"),
        "SUPABASE_SERVICE_ROLE_KEY": _jwt(jwt_secret, "service_role"),
        "BITTENSOR_NETWORK": network,
        "BITTENSOR_NETUID": netuid,
        "SUBTENSOR_NETWORK": network,
        "NETUID": netuid,
        "EXPECTED_CHAIN": chain_endpoint,
        "GATEWAY_URL": gateway_url,
        "VALIDATOR_V2_GATEWAY_URL": gateway_url,
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
        "LEADPOET_PARITY_CANDIDATE_SHA": candidate_sha,
        "LEADPOET_PRODUCTION_PARITY_MODE": "enabled",
        "LEADPOET_PRODUCTION_PARITY_RUN_ID": run_id,
        "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN": supabase_url,
        "LEADPOET_PRODUCTION_PARITY_CHAIN_HOST": chain_host,
        "LEADPOET_PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST": chain_host,
        "BT_SUBTENSOR_NETWORK": network,
        "VALIDATOR_SUBTENSOR_NETWORK": network,
        "VALIDATOR_NETUID": netuid,
        "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON": "",
        "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH": str(
            cutover_manifest_path(run_id)
        ),
        "LEADPOET_SENTRY_API_TOKEN": "",
        "LEADPOET_SENTRY_DSN": "",
        "LEADPOET_SENTRY_ENABLED": "0",
        "GATEWAY_OTEL_ENABLED": "0",
        "GATEWAY_OTEL_ENDPOINT": "",
        "GATEWAY_OTEL_METRICS_ENDPOINT": "",
        "GATEWAY_OTEL_TOKEN": "",
        "LEADPOET_INTERNAL_SECRET": internal_secret,
        "RESEARCH_LAB_INTERNAL_API_KEY": research_lab_internal_key,
    }
    wallet_environments: dict[str, dict[str, str]] = {}
    for role, spec in wallet_specs.items():
        base = install_base(run_id, role)
        wallet_environments[role] = {
            "VALIDATOR_WALLET_ROOT": str(base / "wallets"),
            "VALIDATOR_WALLET_NAME": str(spec["wallet_name"]),
            "VALIDATOR_WALLET_HOTKEY": str(spec["wallet_hotkey"]),
            "VALIDATOR_V2_HOTKEY_CONFIG": str(
                base / "validator-hotkey-config-v2.json"
            ),
            "VALIDATOR_V2_HOTKEY_ENVELOPE": str(
                base / "validator-hotkey-envelope-v2.json"
            ),
            "BT_WALLET_PATH": str(base / "wallets"),
            "BT_WALLET_NAME": str(spec["wallet_name"]),
            "BT_WALLET_HOTKEY": str(spec["wallet_hotkey"]),
        }
    allowed = _text_set(
        settings.get("allowed_production_boundary_keys"),
        field="allowed_production_boundary_keys",
    )
    required_gateway = _text_set(
        settings.get("required_gateway_overlay_keys"),
        field="required_gateway_overlay_keys",
    )
    required_validator = _text_set(
        settings.get("required_validator_overlay_keys"),
        field="required_validator_overlay_keys",
    )
    required_auditor = _text_set(
        settings.get("required_auditor_overlay_keys"),
        field="required_auditor_overlay_keys",
    )
    required_dashboard = _text_set(
        settings.get("required_dashboard_overlay_keys"),
        field="required_dashboard_overlay_keys",
    )
    gateway = _build_runtime(
        source=gateway_source,
        overlay=gateway_overlay,
        generated=generated_common,
        allowed_production_boundaries=allowed,
        required_overlay_keys=required_gateway,
        field="gateway",
    )
    validator = _build_runtime(
        source=validator_source,
        overlay=validator_overlay,
        generated={
            **generated_common,
            **wallet_environments["primary-validator"],
        },
        allowed_production_boundaries=allowed,
        required_overlay_keys=required_validator,
        field="validator",
    )
    auditors: list[dict[str, str]] = []
    for index, raw_overlay in enumerate(auditor_overlays):
        auditor_overlay = _mapping(
            raw_overlay, field=f"staging auditor overlay {index}"
        )
        auditors.append(
            _build_runtime(
                source=validator_source,
                overlay=auditor_overlay,
                generated={
                    **generated_common,
                    "AUDITOR_WEIGHT_PROTOCOL": "authoritative_v2",
                    **wallet_environments[f"auditor-{'a' if index == 0 else 'b'}"],
                },
                allowed_production_boundaries=allowed,
                required_overlay_keys=required_auditor,
                field=f"auditor {index}",
            )
        )
    dashboard = _build_runtime(
        source={},
        overlay=dashboard_overlay,
        generated={
            "NEXT_PUBLIC_SUPABASE_URL": supabase_url,
            "NEXT_PUBLIC_SUPABASE_ANON_KEY": generated_common[
                "SUPABASE_ANON_KEY"
            ],
            "SUPABASE_SECRET_KEY": generated_common[
                "SUPABASE_SERVICE_ROLE_KEY"
            ],
            "NEXT_PUBLIC_SITE_URL": f"https://{dashboard_domain}",
            "GATEWAY_URL": gateway_url,
            "RESEARCH_LAB_ALERT_MONITOR_ENABLED": "false",
            "RESEARCH_LAB_EVENT_MONITOR_ENABLED": "false",
        },
        allowed_production_boundaries=set(),
        required_overlay_keys=required_dashboard,
        field="dashboard",
    )
    names = _secret_names(run_id)
    _create_secret(
        client,
        name=names["gateway"],
        value=gateway,
        kms_key_id=kms_key_id,
        run_id=run_id,
        candidate_sha=candidate_sha,
    )
    try:
        _create_secret(
            client,
            name=names["validator"],
            value=validator,
            kms_key_id=kms_key_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
        )
        for index, key in enumerate(("auditor-a", "auditor-b")):
            _create_secret(
                client,
                name=names[key],
                value=auditors[index],
                kms_key_id=kms_key_id,
                run_id=run_id,
                candidate_sha=candidate_sha,
            )
        _create_secret(
            client,
            name=names["dashboard"],
            value=dashboard,
            kms_key_id=kms_key_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
        )
        _create_secret(
            client,
            name=names["database"],
            value={
                "AUTHENTICATOR_PASSWORD": authenticator_password,
                "JWT_SECRET": jwt_secret,
                "POSTGRES_PASSWORD": postgres_password,
            },
            kms_key_id=kms_key_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
        )
    except Exception:
        for name in names.values():
            try:
                _delete_secret(client, name)
            except Exception:
                pass
        raise
    return {
        "schema_version": "leadpoet.production_parity_secret_state.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "region": str(config.get("region") or ""),
        "secret_ids": names,
        "status": "ready",
        "gateway_domain": gateway_domain,
        "database_domain": database_domain,
        "dashboard_domain": dashboard_domain,
        "source_key_counts": {
            "gateway": len(gateway_source),
            "validator": len(validator_source),
        },
        "materialized_key_counts": {
            "gateway": len(gateway),
            "validator": len(validator),
            "auditor-a": len(auditors[0]),
            "auditor-b": len(auditors[1]),
            "dashboard": len(dashboard),
        },
        "wallet_identities": {
            role: {
                "expected_hotkey": spec["expected_hotkey"],
                "artifact_sha256": spec["sha256"],
                "artifact_version_id": spec["version_id"],
            }
            for role, spec in wallet_specs.items()
        },
        "epoch_authority_identity": {
            "artifact_sha256": epoch_authority_spec["sha256"],
            "artifact_version_id": epoch_authority_spec["version_id"],
            "mapping_hash": epoch_authority_spec["mapping_hash"],
            "network_genesis_hash": epoch_authority_spec[
                "network_genesis_hash"
            ],
            "netuid": epoch_authority_spec["netuid"],
        },
    }


def cleanup(*, state: Mapping[str, Any], client: Any) -> dict[str, Any]:
    run_id = str(state.get("run_id") or "")
    if not RUN_RE.fullmatch(run_id):
        raise SecretMaterializationError("secret state run identity is invalid")
    ids = _mapping(state.get("secret_ids"), field="secret_ids")
    expected_prefix = f"leadpoet/staging/production-parity/{run_id}/"
    expected_ids = _secret_names(run_id)
    if ids != expected_ids:
        raise SecretMaterializationError("secret cleanup scope differs")
    names = [
        str(ids.get(key) or "")
        for key in (
            "gateway",
            "validator",
            "database",
            "auditor-a",
            "auditor-b",
            "dashboard",
        )
    ]
    if any(not name.startswith(expected_prefix) for name in names):
        raise SecretMaterializationError("secret cleanup scope is invalid")
    deleted = 0
    absent = 0
    failures: list[str] = []
    for name in names:
        try:
            if _delete_secret(client, name):
                deleted += 1
            else:
                absent += 1
        except (BotoCoreError, ClientError):
            failures.append(name)
    if failures:
        raise SecretMaterializationError(
            "run-scoped secret cleanup failed: " + ",".join(failures)
        )
    return {
        "run_id": run_id,
        "cleaned_secret_count": len(names),
        "deleted_secret_count": deleted,
        "already_absent_secret_count": absent,
    }


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--stack-state", type=Path, required=True)
    create.add_argument("--candidate-sha", required=True)
    create.add_argument("--state", type=Path, required=True)
    delete = subparsers.add_parser("delete")
    delete.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            config = _load(args.config)
            stack_state = json.loads(args.stack_state.read_text(encoding="utf-8"))
            region = str(config.get("region") or "")
            stack_doc = _mapping(stack_state, field="stack state")
            candidate_sha = str(args.candidate_sha).lower()
            provisional = _cleanup_state(
                run_id=str(stack_doc.get("run_id") or ""),
                candidate_sha=candidate_sha,
                region=region,
            )
            if stack_doc.get("candidate_sha") != candidate_sha:
                raise SecretMaterializationError(
                    "stack and secret candidate identities differ"
                )
            _write_state(args.state, provisional)
            result = materialize(
                config=config,
                state=stack_doc,
                candidate_sha=candidate_sha,
                client=boto3.client("secretsmanager", region_name=region),
            )
            _write_state(args.state, result)
        else:
            state = json.loads(args.state.read_text(encoding="utf-8"))
            state_doc = _mapping(state, field="secret state")
            result = cleanup(
                state=state_doc,
                client=boto3.client(
                    "secretsmanager", region_name=str(state_doc.get("region") or "")
                ),
            )
    except (
        OSError,
        ValueError,
        BotoCoreError,
        ClientError,
        SecretMaterializationError,
    ) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
