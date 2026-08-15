"""Fail-closed external boundary selection for disposable parity deployments."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
import re
from typing import Dict, Mapping, Optional
from urllib.parse import urlsplit


PRODUCTION_SUPABASE_ORIGIN = "https://qplwoislplkcegvdmbim.supabase.co"
PRODUCTION_CHAIN_HOST = "entrypoint-finney.opentensor.ai"
PRODUCTION_CHAIN_ARCHIVE_HOST = "archive.chain.opentensor.ai"
PRODUCTION_PARITY_CHAIN_HOST = "test.finney.opentensor.ai"
PRODUCTION_PARITY_MODE_ENV = "LEADPOET_PRODUCTION_PARITY_MODE"
PRODUCTION_PARITY_RUN_ID_ENV = "LEADPOET_PRODUCTION_PARITY_RUN_ID"
PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV = (
    "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN"
)
PRODUCTION_PARITY_CHAIN_HOST_ENV = "LEADPOET_PRODUCTION_PARITY_CHAIN_HOST"
PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST_ENV = (
    "LEADPOET_PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST"
)
PRODUCTION_PARITY_ENV_NAMES = (
    PRODUCTION_PARITY_MODE_ENV,
    PRODUCTION_PARITY_RUN_ID_ENV,
    PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV,
    PRODUCTION_PARITY_CHAIN_HOST_ENV,
    PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST_ENV,
)

_RUN_ID_RE = re.compile(r"^[a-z0-9-]{6,40}$")
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class ProductionParityBoundaryV2Error(ValueError):
    """A parity-only external boundary is incomplete or unsafe."""


def _optional_text(environment: Mapping[str, object], name: str) -> Optional[str]:
    value = environment.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProductionParityBoundaryV2Error(f"{name} must be text or null")
    normalized = value.strip()
    return normalized or None


def _validate_hostname(value: str, *, field: str) -> str:
    hostname = str(value or "").strip().lower()
    if not _HOST_RE.fullmatch(hostname):
        raise ProductionParityBoundaryV2Error(f"{field} is invalid")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return hostname
    raise ProductionParityBoundaryV2Error(f"{field} cannot be an IP address")


def _parity_configuration(
    environment: Mapping[str, object],
) -> Optional[Dict[str, str]]:
    values = {
        name: _optional_text(environment, name)
        for name in PRODUCTION_PARITY_ENV_NAMES
    }
    configured = {name for name, value in values.items() if value is not None}
    if not configured:
        return None
    if configured != set(PRODUCTION_PARITY_ENV_NAMES):
        raise ProductionParityBoundaryV2Error(
            "production-parity boundary configuration is incomplete"
        )
    if values[PRODUCTION_PARITY_MODE_ENV] != "enabled":
        raise ProductionParityBoundaryV2Error(
            "production-parity boundary mode is invalid"
        )
    run_id = str(values[PRODUCTION_PARITY_RUN_ID_ENV])
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ProductionParityBoundaryV2Error(
            "production-parity run identity is invalid"
        )
    origin = str(values[PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV])
    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ProductionParityBoundaryV2Error(
            "production-parity Supabase origin port is invalid"
        ) from exc
    hostname = _validate_hostname(
        str(parsed.hostname or ""), field="production-parity Supabase hostname"
    )
    expected_prefix = f"database-{run_id}."
    if (
        parsed.scheme.lower() != "https"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not hostname.startswith(expected_prefix)
        or hostname == PRODUCTION_SUPABASE_ORIGIN.split("://", 1)[1]
    ):
        raise ProductionParityBoundaryV2Error(
            "production-parity Supabase origin is outside the run-scoped TLS boundary"
        )
    chain_host = _validate_hostname(
        str(values[PRODUCTION_PARITY_CHAIN_HOST_ENV]),
        field="production-parity chain host",
    )
    archive_host = _validate_hostname(
        str(values[PRODUCTION_PARITY_CHAIN_ARCHIVE_HOST_ENV]),
        field="production-parity archive host",
    )
    if (
        chain_host != PRODUCTION_PARITY_CHAIN_HOST
        or archive_host != PRODUCTION_PARITY_CHAIN_HOST
    ):
        raise ProductionParityBoundaryV2Error(
            "production-parity chain boundary is not the reviewed test network"
        )
    return {
        "run_id": run_id,
        "supabase_origin": f"https://{hostname}"
        + (":443" if port == 443 else ""),
        "chain_host": chain_host,
        "chain_archive_host": archive_host,
    }


def validate_production_parity_boundary_document_v2(
    environment: Mapping[str, object],
    *,
    network: str,
    netuid: int,
) -> Dict[str, object]:
    """Return every external boundary after deployment-level validation."""

    parity = _parity_configuration(environment)
    if parity is None:
        return {
            "mode": "production",
            "supabase_origin": PRODUCTION_SUPABASE_ORIGIN,
            "chain_host": PRODUCTION_CHAIN_HOST,
            "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
        }
    if str(network or "").strip().lower() != "test" or int(netuid) == 71:
        raise ProductionParityBoundaryV2Error(
            "production-parity boundary requires the isolated test network"
        )
    return {"mode": "production-parity", **parity}


def validate_production_parity_boundary_v2(
    environment: Mapping[str, object],
    *,
    network: str,
    netuid: int,
) -> str:
    """Return the committed Supabase origin after deployment-level validation."""

    return str(
        validate_production_parity_boundary_document_v2(
            environment, network=network, netuid=netuid
        )["supabase_origin"]
    )


def configured_boundary_document_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    source = os.environ if environment is None else environment
    parity = _parity_configuration(source)
    if parity is None:
        return {
            "mode": "production",
            "supabase_origin": PRODUCTION_SUPABASE_ORIGIN,
            "chain_host": PRODUCTION_CHAIN_HOST,
            "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
        }
    return {"mode": "production-parity", **parity}


def configured_supabase_origin_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> str:
    """Resolve the already-attested database boundary."""

    return str(configured_boundary_document_v2(environment)["supabase_origin"])


def configured_chain_source_boundary_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """Resolve the already-attested live/archive chain hosts."""

    document = configured_boundary_document_v2(environment)
    return {
        "chain_host": str(document["chain_host"]),
        "chain_archive_host": str(document["chain_archive_host"]),
    }


def configured_chain_signing_profile_path_v2(
    production_profile_path: Path,
    *,
    environment: Optional[Mapping[str, object]] = None,
) -> Path:
    """Select one measured profile without accepting a caller-chosen path."""

    source = os.environ if environment is None else environment
    parity = _parity_configuration(source)
    path = Path(production_profile_path)
    if parity is None:
        return path
    return path.with_name("chain_signing_profile_test_v2.json")
