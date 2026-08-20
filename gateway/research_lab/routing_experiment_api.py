"""Internal Research Lab API for immutable routing experiment specifications.

These endpoints are deliberately narrower than the routing worker API.  They
validate and persist an immutable ``RoutingExperimentV2Spec`` or read the
persisted spec.  They never construct a runtime service, claim a worker lease,
load a provider adapter, or execute a provider call.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
from typing import Any, Callable, Mapping, Optional, Protocol

from fastapi import APIRouter, Body, Header, HTTPException, Request

from .config import ResearchLabGatewayConfig
from .routing_experiment_store import (
    RoutingExperimentStoreError,
    SupabaseRoutingExperimentStore,
)
from .routing_execution_envelope import RoutingExperimentExecutionEnvelopeV2
from research_lab.routing_experiments import (
    RoutingExperimentError,
    RoutingExperimentV2Spec,
)


ROUTING_EXPERIMENT_API_SCHEMA_VERSION = (
    "leadpoet.research_lab.routing_experiment_api.v2"
)


def _require_internal_key(
    config: ResearchLabGatewayConfig,
    provided: Optional[str],
) -> None:
    """Authenticate the internal-only surface without exposing the key."""

    configured = str(config.internal_api_key or "")
    if not configured:
        raise HTTPException(
            status_code=403,
            detail="Research Lab internal API key is not configured",
        )
    if not provided or not secrets.compare_digest(str(provided), configured):
        raise HTTPException(status_code=401, detail="invalid Research Lab internal API key")


def _authorize(config: ResearchLabGatewayConfig, provided: Optional[str]) -> None:
    if not config.api_enabled:
        raise HTTPException(status_code=403, detail="Research Lab gateway API is disabled")
    _require_internal_key(config, provided)


def _spec_response(
    spec: RoutingExperimentV2Spec,
    *,
    request_status: str,
) -> dict[str, Any]:
    """Return bounded status metadata, never model payloads or unit refs."""

    return {
        "schema_version": ROUTING_EXPERIMENT_API_SCHEMA_VERSION,
        "experiment_hash": spec.experiment_hash(),
        "experiment_id": spec.experiment_id,
        "status": request_status,
        "execution_started": False,
        "provider_execution": "not_requested_by_api",
        "receipt_execution_mode": spec.receipt_execution_mode,
        "allow_live_credit_spend": spec.allow_live_credit_spend,
        "baseline_variant_id": spec.baseline_variant_id,
        "variants": [
            {
                "variant_id": item.variant_id,
                "stage": item.stage,
                "change_kind": item.change_kind,
            }
            for item in spec.variants
        ],
    }


class RoutingExperimentSpecAdmissionAuthority(Protocol):
    """Exact model/runtime admission that runs before the first SQL write."""

    def admit(
        self, spec: RoutingExperimentV2Spec
    ) -> RoutingExperimentExecutionEnvelopeV2 | None: ...


class FailClosedRoutingExperimentSpecAdmissionAuthority:
    def admit(
        self, spec: RoutingExperimentV2Spec
    ) -> RoutingExperimentExecutionEnvelopeV2 | None:
        del spec
        raise RoutingExperimentError(
            "routing experiment model/runtime admission is unavailable"
        )


@dataclass(frozen=True)
class RoutingExperimentApiService:
    """Store-only application service used by all API aliases."""

    store_factory: Callable[[], SupabaseRoutingExperimentStore] = (
        SupabaseRoutingExperimentStore
    )
    admission_authority: RoutingExperimentSpecAdmissionAuthority = (
        FailClosedRoutingExperimentSpecAdmissionAuthority()
    )

    def parse_spec(self, payload: Mapping[str, Any]) -> RoutingExperimentV2Spec:
        if not isinstance(payload, Mapping):
            raise RoutingExperimentError("routing experiment request must be an object")
        try:
            spec = RoutingExperimentV2Spec.from_mapping(dict(payload))
            if dict(payload) != spec.to_dict():
                raise RoutingExperimentError(
                    "routing experiment request is not canonical"
                )
            return spec
        except Exception as exc:  # noqa: BLE001 - convert model errors at API boundary
            if isinstance(exc, RoutingExperimentError):
                raise
            raise RoutingExperimentError(
                "routing experiment specification is malformed"
            ) from exc

    def submit(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = self.parse_spec(payload)
        execution_envelope = self.admission_authority.admit(spec)
        if spec.allow_live_credit_spend and not isinstance(
            execution_envelope, RoutingExperimentExecutionEnvelopeV2
        ):
            raise RoutingExperimentError(
                "routing live experiment admission did not return an execution envelope"
            )
        # This is the sole write. Supabase's RPC enforces the append-only,
        # hash/idempotency contract; no worker service is created here.
        try:
            result = self.store_factory().submit(
                spec, execution_envelope=execution_envelope
            )
        except RoutingExperimentStoreError:
            raise
        authority = {}
        if isinstance(result, Mapping):
            # The SQL authority currently returns only these bounded fields.
            # Keep the response whitelist explicit if the RPC gains fields in
            # a later migration.
            authority = {
                key: result[key]
                for key in ("experiment_hash", "idempotent")
                if key in result
            }
        return {
            **_spec_response(spec, request_status="submitted"),
            "authority": authority,
        }

    def status(self, experiment_hash: str) -> dict[str, Any]:
        store = self.store_factory()
        try:
            spec = store.load_spec(experiment_hash)
            request = store.execution_request(experiment_hash)
        except RoutingExperimentStoreError:
            raise
        if spec is None:
            raise KeyError("routing experiment was not found")
        return _spec_response(
            spec,
            request_status="queued" if request is not None else "submitted",
        )

    def request_execution(self, experiment_hash: str) -> dict[str, Any]:
        store = self.store_factory()
        spec = store.load_spec(experiment_hash)
        if spec is None:
            raise KeyError("routing experiment was not found")
        result = store.request_execution(experiment_hash)
        return {
            **_spec_response(spec, request_status="queued"),
            "execution_requested": True,
            "request_hash": str(result["request_hash"]),
        }


# The API has no active module-global service.  A reviewed process bootstrap
# installs one into app.state.  Until that happens every route fails closed
# before the store is constructed or an SQL write is attempted.
service: RoutingExperimentApiService | None = None


def install_routing_experiment_api_service(
    installed: RoutingExperimentApiService | None,
    *,
    app: Any | None = None,
) -> None:
    """Install the bootstrap-selected API service.

    ``installed=None`` deliberately installs the fail-closed state.  The app
    state is mandatory so a module import can never become an active product
    composition by itself.
    """

    if installed is not None and not isinstance(installed, RoutingExperimentApiService):
        raise TypeError("routing experiment API service is invalid")
    state = getattr(app, "state", None)
    if state is None:
        raise TypeError("routing experiment API app state is unavailable")
    state.routing_experiment_api_service = installed


def _service_for_request(request: Request) -> RoutingExperimentApiService:
    state = getattr(getattr(request, "app", None), "state", None)
    installed = getattr(state, "routing_experiment_api_service", None)
    if not isinstance(installed, RoutingExperimentApiService):
        raise RoutingExperimentError(
            "Research Lab routing experiment API composition is unavailable"
        )
    return installed
router = APIRouter(
    prefix="/research-lab/routing-experiments",
    tags=["research-lab-routing-experiments"],
)


def _service_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RoutingExperimentStoreError):
        return HTTPException(
            status_code=503,
            detail="Research Lab routing experiment authority is unavailable",
        )
    if isinstance(exc, (RoutingExperimentError, ValueError, TypeError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(
        status_code=503,
        detail="Research Lab routing experiment authority is unavailable",
    )


@router.post("")
@router.post("/submit")
@router.post("/request")
async def submit_routing_experiment_spec(
    request: Request,
    payload: Mapping[str, Any] = Body(...),
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    """Persist one immutable spec; this endpoint never starts execution."""

    config = ResearchLabGatewayConfig.from_env()
    _authorize(config, x_leadpoet_internal_key)
    try:
        return _service_for_request(request).submit(payload)
    except Exception as exc:  # noqa: BLE001 - map authority failures safely
        raise _service_error(exc) from exc


@router.get("/{experiment_hash}")
@router.get("/{experiment_hash}/status")
async def get_routing_experiment_status(
    request: Request,
    experiment_hash: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    """Read one immutable spec without claiming or executing it."""

    config = ResearchLabGatewayConfig.from_env()
    _authorize(config, x_leadpoet_internal_key)
    try:
        return _service_for_request(request).status(experiment_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - map authority failures safely
        raise _service_error(exc) from exc


@router.post("/{experiment_hash}/request")
async def request_routing_experiment_execution(
    request: Request,
    experiment_hash: str,
    x_leadpoet_internal_key: Optional[str] = Header(default=None),
):
    """Append one durable worker request without executing a provider in API.

    The independent worker remains the only execution entrypoint.  This route
    persists only a bounded queue fact. The API process cannot claim a lease,
    construct a provider request, or execute a provider call.
    """

    config = ResearchLabGatewayConfig.from_env()
    _authorize(config, x_leadpoet_internal_key)
    try:
        response = _service_for_request(request).request_execution(experiment_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - map authority failures safely
        raise _service_error(exc) from exc
    return response


__all__ = [
    "ROUTING_EXPERIMENT_API_SCHEMA_VERSION",
    "RoutingExperimentApiService",
    "RoutingExperimentSpecAdmissionAuthority",
    "FailClosedRoutingExperimentSpecAdmissionAuthority",
    "install_routing_experiment_api_service",
    "get_routing_experiment_status",
    "request_routing_experiment_execution",
    "router",
    "service",
    "submit_routing_experiment_spec",
]
