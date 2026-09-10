"""Protected provider preflight adapter for qualification."""
from __future__ import annotations
import uuid
from typing import Any, Mapping
from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
from gateway.tee.scoring_executor_v2 import OP_PROVIDER_PREFLIGHT_V2, PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION

class ResearchLabV2AuthorityError(RuntimeError):
    """The protected preflight result is missing or invalid."""


async def execute_provider_preflight_v2(
    *,
    scope_key: str,
    worker_index: int,
    settings: Mapping[str, Any],
    force: bool = False,
    provider_credential_profile: str = "provider_preflight",
    execute: Any = execute_scoring_v2,
) -> dict[str, Any]:
    measurement_id = uuid.uuid4().hex
    outcome = await execute(
        operation=OP_PROVIDER_PREFLIGHT_V2,
        purpose="research_lab.provider_preflight.v2",
        epoch_id=0,
        # Keep the receipt sequence inside the V2 INTEGER schema. Freshness is
        # committed by measurement_id in the payload so every requested probe
        # derives a new enclave job instead of replaying a terminal job for up
        # to the execution manager's one-hour retention window.
        sequence=0,
        payload={
            "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
            "measurement_id": measurement_id,
            "scope_key": str(scope_key),
            "force": bool(force),
            "settings": dict(settings),
        },
        worker_index=int(worker_index),
        provider_credential_profile=provider_credential_profile,
    )
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise ResearchLabV2AuthorityError("provider preflight result is missing")
    return dict(result)
