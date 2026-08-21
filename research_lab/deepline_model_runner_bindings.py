"""Deepline transport bindings for reviewed common-runner tool IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .common_model_runner_host import (
    HostActionBinding,
    HostActionResult,
    ModelRunnerHostError,
)


HOST_PROVIDER_RESPONSE_SCHEMA_VERSION = "host-provider-response:v1"
DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION = "model-runner-deepline-action:v1"


@dataclass(frozen=True)
class DeeplineToolContract:
    deepline_tool_id: str
    timeout_seconds: float
    maximum_cost_credits: float


DEEPLINE_TOOL_CONTRACTS = {
    "intent.source_add.predictleads_financing": DeeplineToolContract(
        deepline_tool_id="predictleads_company_financing_events",
        timeout_seconds=30,
        maximum_cost_credits=0.56,
    ),
    "intent.source_add.predictleads_jobs": DeeplineToolContract(
        deepline_tool_id="predictleads_company_job_openings",
        timeout_seconds=30,
        maximum_cost_credits=0.56,
    ),
    "intent.source_add.predictleads_connections": DeeplineToolContract(
        deepline_tool_id="predictleads_company_connections",
        timeout_seconds=30,
        maximum_cost_credits=1.68,
    ),
}


@dataclass(frozen=True)
class DeeplineCallReceipt:
    status_code: int
    body: Mapping[str, Any]
    calls: int
    cost_credits: float
    latency_ms: float
    provider_request_id: str | None = None


class DeeplineRunnerClient(Protocol):
    """Credentialed Lab client that performs the host-owned network call."""

    def execute_tool(
        self,
        *,
        tool_id: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        timeout_seconds: float,
    ) -> DeeplineCallReceipt: ...


def _execute_deepline_action(
    *,
    client: DeeplineRunnerClient,
    contract: DeeplineToolContract,
    action: Mapping[str, Any],
) -> HostActionResult:
    arguments = action.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ModelRunnerHostError("model Deepline action arguments are invalid")
    idempotency_key = str(action.get("idempotency_key") or "")
    if not idempotency_key:
        raise ModelRunnerHostError("model Deepline action is not idempotent")
    tool_id = str(action.get("tool_id") or "")
    receipt = client.execute_tool(
        tool_id=contract.deepline_tool_id,
        payload={
            "schema_version": DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION,
            "model_tool_id": tool_id,
            "arguments": dict(arguments),
        },
        idempotency_key=idempotency_key,
        timeout_seconds=contract.timeout_seconds,
    )
    if not isinstance(receipt, DeeplineCallReceipt):
        raise ModelRunnerHostError("Deepline client receipt is invalid")
    if receipt.calls < 1:
        raise ModelRunnerHostError("Deepline call receipt has no call")
    if not 0 <= receipt.cost_credits <= contract.maximum_cost_credits:
        raise ModelRunnerHostError("Deepline call exceeded its cost limit")
    if not 200 <= receipt.status_code < 300:
        return HostActionResult(
            outcome="failed",
            reason_code=f"deepline_http_{receipt.status_code}",
            provider_response=None,
            calls=receipt.calls,
            cost_credits=receipt.cost_credits,
            latency_ms=receipt.latency_ms,
            provider_request_id=receipt.provider_request_id,
        )
    return HostActionResult(
        outcome="succeeded",
        reason_code="deepline_completed",
        provider_response={
            "schema_version": HOST_PROVIDER_RESPONSE_SCHEMA_VERSION,
            "provider": "deepline",
            "status_code": receipt.status_code,
            "body": dict(receipt.body),
        },
        calls=receipt.calls,
        cost_credits=receipt.cost_credits,
        latency_ms=receipt.latency_ms,
        provider_request_id=receipt.provider_request_id,
    )


def build_deepline_runner_binding(
    *,
    model_tool_id: str,
    binding_contract_sha256: str,
    client: DeeplineRunnerClient,
) -> HostActionBinding:
    """Bind one reviewed model tool ID without adding host routing policy."""

    contract = DEEPLINE_TOOL_CONTRACTS.get(model_tool_id)
    if contract is None:
        raise ModelRunnerHostError("Deepline model tool ID is not reviewed")
    return HostActionBinding(
        action_type="execute_intent_tool",
        tool_id=model_tool_id,
        binding_contract_sha256=binding_contract_sha256,
        dispatch=lambda action: _execute_deepline_action(
            client=client,
            contract=contract,
            action=action,
        ),
    )


__all__ = [
    "DEEPLINE_RUNNER_REQUEST_SCHEMA_VERSION",
    "DEEPLINE_TOOL_CONTRACTS",
    "DeeplineCallReceipt",
    "DeeplineRunnerClient",
    "DeeplineToolContract",
    "build_deepline_runner_binding",
]
