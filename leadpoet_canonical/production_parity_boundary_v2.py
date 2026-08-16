"""Fail-closed database boundary for disposable production-parity runs.

Parity deliberately changes only the Supabase-compatible data plane and the
daily benchmark date. Chain reads, signing profiles, netuid, and network stay
identical to production; irreversible chain writes are stopped by the external
submission-capture adapter rather than by changing measured validator policy.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import os
from pathlib import Path
import re
from typing import Dict, Mapping, Optional
from urllib.parse import urlsplit


PRODUCTION_SUPABASE_ORIGIN = "https://qplwoislplkcegvdmbim.supabase.co"
PRODUCTION_CHAIN_HOST = "entrypoint-finney.opentensor.ai"
PRODUCTION_CHAIN_ARCHIVE_HOST = "archive.chain.opentensor.ai"
PRODUCTION_PARITY_MODE_ENV = "LEADPOET_PRODUCTION_PARITY_MODE"
PRODUCTION_PARITY_RUN_ID_ENV = "LEADPOET_PRODUCTION_PARITY_RUN_ID"
PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV = (
    "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN"
)
PRODUCTION_PARITY_BENCHMARK_DATE_ENV = (
    "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE"
)
PRODUCTION_PARITY_ENV_NAMES = (
    PRODUCTION_PARITY_MODE_ENV,
    PRODUCTION_PARITY_RUN_ID_ENV,
    PRODUCTION_PARITY_SUPABASE_ORIGIN_ENV,
    PRODUCTION_PARITY_BENCHMARK_DATE_ENV,
)

_RUN_ID_RE = re.compile(r"^[a-z0-9-]{6,40}$")
_CLOUDFRONT_HOST_RE = re.compile(r"^[a-z0-9-]+\.cloudfront\.net$")


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
    hostname = str(parsed.hostname or "").strip().lower()
    if (
        parsed.scheme.lower() != "https"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not _CLOUDFRONT_HOST_RE.fullmatch(hostname)
        or origin.rstrip("/") == PRODUCTION_SUPABASE_ORIGIN
    ):
        raise ProductionParityBoundaryV2Error(
            "production-parity Supabase origin is outside the run-scoped TLS boundary"
        )
    benchmark_date = str(values[PRODUCTION_PARITY_BENCHMARK_DATE_ENV])
    try:
        normalized_date = date.fromisoformat(benchmark_date).isoformat()
    except ValueError as exc:
        raise ProductionParityBoundaryV2Error(
            "production-parity benchmark date is invalid"
        ) from exc
    return {
        "run_id": run_id,
        "supabase_origin": f"https://{hostname}" + (":443" if port == 443 else ""),
        "benchmark_date": normalized_date,
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
            "benchmark_date": None,
            "chain_host": PRODUCTION_CHAIN_HOST,
            "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
        }
    if str(network or "").strip().lower() != "finney" or int(netuid) != 71:
        raise ProductionParityBoundaryV2Error(
            "production-parity must retain the production network and netuid"
        )
    return {
        "mode": "production-parity",
        **parity,
        "chain_host": PRODUCTION_CHAIN_HOST,
        "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
    }


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
            "benchmark_date": None,
            "chain_host": PRODUCTION_CHAIN_HOST,
            "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
        }
    return {
        "mode": "production-parity",
        **parity,
        "chain_host": PRODUCTION_CHAIN_HOST,
        "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
    }


def configured_supabase_origin_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> str:
    """Resolve the already-attested database boundary."""

    return str(configured_boundary_document_v2(environment)["supabase_origin"])


def production_parity_enabled_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> bool:
    """Return true only for a complete, validated parity configuration."""

    return configured_boundary_document_v2(environment)["mode"] == "production-parity"


def configured_chain_source_boundary_v2(
    environment: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """Return the production chain boundary in every mode."""

    configured_boundary_document_v2(environment)
    return {
        "chain_host": PRODUCTION_CHAIN_HOST,
        "chain_archive_host": PRODUCTION_CHAIN_ARCHIVE_HOST,
    }


def configured_chain_signing_profile_path_v2(
    production_profile_path: Path,
    *,
    environment: Optional[Mapping[str, object]] = None,
) -> Path:
    """Retain the measured production signing profile in every parity mode."""

    configured_boundary_document_v2(environment)
    return Path(production_profile_path)


def configured_rebenchmark_now_v2(
    *,
    environment: Optional[Mapping[str, object]] = None,
    now: Optional[datetime] = None,
) -> datetime:
    """Return UTC now, replacing only the date inside a complete parity run."""

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    document = configured_boundary_document_v2(environment)
    benchmark_date = document.get("benchmark_date")
    if benchmark_date is None:
        return current
    selected = date.fromisoformat(str(benchmark_date))
    return current.replace(year=selected.year, month=selected.month, day=selected.day)
