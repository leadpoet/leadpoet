"""Validator-side Research Lab allocation fetch and verification."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
import http.client
import json
import os
import re
import socket
import sys
import time
from typing import Any, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zlib

from leadpoet_verifier.economics import DEFAULT_RESEARCH_LAB_EMISSION_PERCENT, allocate_research_lab_epoch

from .canonical import sha256_json
TRUTHY_VALUES = {"1", "true", "yes", "on"}
PERCENT_EPSILON = 0.000001
WEIGHT_INPUT_FETCH_TIMEOUT_SECONDS = 90
# The block-180 preparation has about 720 seconds before submission starts.
# Give its measured cold build enough time, but keep the in-window budget at 90.
ALLOCATION_PREPARATION_FETCH_TIMEOUT_SECONDS = 480

# Context keeps the larger budget limited to the early preparation task without
# changing the protected weight-submission guard.
ALLOCATION_PREPARATION_FETCH_BUDGET: "ContextVar[Optional[int]]" = ContextVar(
    "leadpoet_allocation_preparation_fetch_budget",
    default=None,
)


def resolve_allocation_fetch_budget(timeout_seconds: float) -> float:
    """Return the larger of the caller's budget and any ambient preparation budget."""

    ambient = ALLOCATION_PREPARATION_FETCH_BUDGET.get()
    if ambient is None:
        return float(timeout_seconds)
    return float(max(float(timeout_seconds), float(ambient)))
# Bounded in-window retry for the weight-path allocation fetch. A single
# transient gateway failure (connection refused, 5xx, a brief restart blip)
# must not cost the validator the whole epoch's weight submission. The retry
# budget is the caller's total timeout, so the sequence can never run past the
# on-chain submission window; a single slow response that consumes the budget
# behaves exactly like the previous single-attempt fetch.
ALLOCATION_FETCH_MAX_ATTEMPTS = 4
ALLOCATION_FETCH_RETRY_DELAY_SECONDS = 2.0
ALLOCATION_FETCH_MIN_ATTEMPT_BUDGET_SECONDS = 5.0


def _request_headers(*, include_internal_key: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if include_internal_key:
        internal_key = (
            os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", "").strip()
            or os.getenv("LEADPOET_INTERNAL_SECRET", "").strip()
        )
        if internal_key:
            headers["x-leadpoet-internal-key"] = internal_key
    return headers


def _validator_lab_cap_ceiling_percent() -> float:
    raw = os.getenv("RESEARCH_LAB_EMISSION_PERCENT", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)
    return float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)


def _argv_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return str(sys.argv[index + 1] or "")


def _is_production_subnet(data: Mapping[str, Any] | None = None) -> bool:
    data = data or {}
    network = str(
        data.get("BITTENSOR_NETWORK")
        or data.get("SUBTENSOR_NETWORK")
        or os.getenv("BITTENSOR_NETWORK")
        or os.getenv("SUBTENSOR_NETWORK")
        or _argv_value("--subtensor_network")
        or ""
    ).strip().lower()
    netuid = str(
        data.get("BITTENSOR_NETUID")
        or data.get("NETUID")
        or os.getenv("BITTENSOR_NETUID")
        or os.getenv("NETUID")
        or _argv_value("--netuid")
        or ""
    ).strip()
    return network == "finney" and netuid == "71"


def _default_for_prod(data: Mapping[str, Any] | None = None) -> bool:
    return _is_production_subnet(data)


def _truthy_with_default(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str) and not value.strip():
        return bool(default)
    return _truthy(value)


@dataclass(frozen=True)
class ResearchLabValidatorFlags:
    fetch_enabled: bool = False
    reimbursements_enabled: bool = True
    weight_mutation_enabled: bool = False
    production_writes_enabled: bool = False
    submit_on_chain_enabled: bool = False
    fulfillment_mutation_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ResearchLabValidatorFlags":
        data = data or {}
        prod_default = _default_for_prod(data)
        return cls(
            fetch_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_VALIDATOR_FETCH_ENABLED", data.get("fetch_enabled")),
                prod_default,
            ),
            reimbursements_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_REIMBURSEMENTS_ENABLED", data.get("reimbursements_enabled", True)),
                True,
            ),
            weight_mutation_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_WEIGHT_MUTATION_ENABLED", data.get("weight_mutation_enabled")),
                prod_default,
            ),
            production_writes_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_PRODUCTION_WRITES_ENABLED", data.get("production_writes_enabled")),
                prod_default,
            ),
            submit_on_chain_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED", data.get("submit_on_chain_enabled")),
                prod_default,
            ),
            fulfillment_mutation_enabled=_truthy(
                data.get("RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED", data.get("fulfillment_mutation_enabled", False))
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def enabled_mutation_flags(self) -> list[str]:
        return [
            name
            for name, enabled in {
                "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.weight_mutation_enabled,
                "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": self.production_writes_enabled,
                "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": self.submit_on_chain_enabled,
                "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
            }.items()
            if enabled
        ]

    def live_allocation_enabled(self) -> bool:
        return self.reimbursements_enabled or self.weight_mutation_enabled or self.submit_on_chain_enabled


















def _is_retryable_allocation_fetch_error(exc: BaseException) -> bool:
    """A transient gateway failure worth another in-window attempt.

    5xx and 429 responses and connection/timeout errors are transient; a 4xx
    (other than 429) is a client-side rejection that will not resolve on retry.
    """
    if isinstance(exc, HTTPError):
        return exc.code >= 500 or exc.code == 429
    if isinstance(
        exc,
        (
            URLError,
            socket.timeout,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            json.JSONDecodeError,
            UnicodeDecodeError,
            _RetryableAllocationResponseError,
        ),
    ):
        return True
    return False


class _RetryableAllocationResponseError(RuntimeError):
    """A complete retry may recover a truncated or corrupted transfer."""


# Sanity bound for decompressing a gzip allocation response. The handoff is
# ~10 MB serialized today; this is a zip-bomb guard, not a policy limit, so it
# is set far above any realistic handoff size.
_ALLOCATION_RESPONSE_MAX_WIRE_BYTES = 64 * 1024 * 1024
_ALLOCATION_RESPONSE_MAX_LOGICAL_BYTES = 256 * 1024 * 1024


def _decode_allocation_response_gzip(wire: bytes) -> bytes:
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        body = decompressor.decompress(
            wire, _ALLOCATION_RESPONSE_MAX_LOGICAL_BYTES + 1
        )
    except zlib.error as exc:
        raise _RetryableAllocationResponseError(
            "allocation response gzip is invalid"
        ) from exc
    if len(body) > _ALLOCATION_RESPONSE_MAX_LOGICAL_BYTES:
        raise RuntimeError("allocation response gzip exceeds size limit")
    if not decompressor.eof:
        raise _RetryableAllocationResponseError(
            "allocation response gzip is truncated"
        )
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise RuntimeError("allocation response gzip failed validation")
    return body


def _fetch_allocation_json(
    url: str,
    *,
    deadline_seconds: float,
    max_attempts: int = ALLOCATION_FETCH_MAX_ATTEMPTS,
    retry_delay_seconds: float = ALLOCATION_FETCH_RETRY_DELAY_SECONDS,
) -> dict[str, Any]:
    """GET one allocation JSON within a total wall-clock budget.

    The whole retry sequence is bounded by ``deadline_seconds`` so it can never
    exceed the caller's on-chain submission window. Each attempt is given the
    time remaining in the budget; a single slow response that consumes the
    budget therefore behaves exactly like the previous single-attempt fetch,
    while a fast transient failure leaves budget for another attempt.
    """
    deadline = time.monotonic() + max(1.0, float(deadline_seconds))
    attempts = max(1, int(max_attempts))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        request = Request(
            url,
            headers={
                **_request_headers(include_internal_key=True),
                # The attested handoff is multi-MB, hash-dense JSON that
                # compresses several-fold; ask for gzip so a cold-epoch fetch
                # spends its 90s budget on the gateway build, not the
                # transfer. Gateways without response compression ignore this
                # header and keep returning identity unchanged.
                "Accept-Encoding": "gzip",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=remaining) as response:
                raw = response.read(_ALLOCATION_RESPONSE_MAX_WIRE_BYTES + 1)
                if len(raw) > _ALLOCATION_RESPONSE_MAX_WIRE_BYTES:
                    raise RuntimeError(
                        "allocation response exceeds wire size limit"
                    )
                declared_encoding = str(
                    response.headers.get("Content-Encoding") or ""
                ).strip().lower()
                if declared_encoding == "gzip":
                    raw = _decode_allocation_response_gzip(raw)
                elif declared_encoding not in ("", "identity"):
                    raise RuntimeError(
                        "unsupported allocation response encoding: %s"
                        % declared_encoding
                    )
                return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - reclassified below
            last_error = exc
            remaining_after = deadline - time.monotonic()
            if (
                attempt >= attempts
                or not _is_retryable_allocation_fetch_error(exc)
                or remaining_after <= ALLOCATION_FETCH_MIN_ATTEMPT_BUDGET_SECONDS
            ):
                raise
            delay = min(
                float(retry_delay_seconds) * (2 ** (attempt - 1)),
                remaining_after - ALLOCATION_FETCH_MIN_ATTEMPT_BUDGET_SECONDS,
            )
            print(
                "research_lab_allocation_fetch_retry attempt=%d/%d "
                "delay_seconds=%.1f remaining_seconds=%.1f type=%s error=%s"
                % (
                    attempt,
                    attempts,
                    max(0.0, delay),
                    remaining_after,
                    type(exc).__name__,
                    str(exc)[:200],
                ),
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("research lab allocation fetch exhausted without a response")


def fetch_research_lab_allocation_bundle(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = WEIGHT_INPUT_FETCH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch the live Research Lab allocation bundle for an epoch.

    The internal key authenticates the validator so the gateway persists the
    submission-time snapshot; anonymous callers get a read-only computation.
    """
    base = gateway_url.rstrip("/")
    return _fetch_allocation_json(
        f"{base}/research-lab/allocations/live/{int(epoch)}",
        deadline_seconds=timeout_seconds,
    )


def fetch_research_lab_attested_allocation_bundle(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = WEIGHT_INPUT_FETCH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch the additive enclave receipt for one live allocation bundle."""

    base = gateway_url.rstrip("/")
    return _fetch_allocation_json(
        f"{base}/research-lab/allocations/attested/{int(epoch)}",
        deadline_seconds=resolve_allocation_fetch_budget(timeout_seconds),
    )






def verify_research_lab_allocation_bundle(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.fetch_enabled:
        errors.append("validator_fetch_disabled")
    if not validator_flags.live_allocation_enabled():
        errors.append("validator_live_research_lab_weight_flags_disabled")
    if _contains_secret_material(bundle):
        errors.append("allocation_bundle_contains_raw_secret_material")
    if bundle.get("bundle_type") != "research_lab_live_allocation_bundle":
        errors.append("unexpected_allocation_bundle_type")
    if not bundle.get("submission_allowed") or not bundle.get("on_chain_submission_allowed"):
        errors.append("gateway_live_research_lab_weight_submission_disabled")

    source_state = bundle.get("source_state")
    source_state_hash = bundle.get("source_state_hash")
    if not isinstance(source_state, Mapping) or not source_state_hash:
        errors.append("allocation_source_state_required")
        source_state = {}
    elif sha256_json(source_state) != source_state_hash:
        errors.append("allocation_source_state_hash_diverged")
    else:
        try:
            if int(source_state.get("epoch")) != int(bundle.get("epoch")):
                errors.append("allocation_source_state_epoch_diverged")
        except (TypeError, ValueError):
            errors.append("allocation_source_state_epoch_invalid")
        if source_state.get("netuid") is not None and bundle.get("netuid") is not None:
            try:
                if int(source_state.get("netuid")) != int(bundle.get("netuid")):
                    errors.append("allocation_source_state_netuid_diverged")
            except (TypeError, ValueError):
                errors.append("allocation_source_state_netuid_invalid")

    allocation_doc = bundle.get("allocation_doc")
    allocation_hash = bundle.get("allocation_hash")
    if not isinstance(allocation_doc, Mapping) or not allocation_hash:
        errors.append("allocation_doc_and_hash_required")
        allocation_doc = {}
    else:
        expected_payload = {k: v for k, v in dict(allocation_doc).items() if k != "allocation_hash"}
        if sha256_json(expected_payload) != allocation_hash:
            errors.append("allocation_hash_diverged")
        if allocation_doc.get("allocation_hash") != allocation_hash:
            errors.append("allocation_doc_hash_field_diverged")

    lab_cap = float(allocation_doc.get("lab_cap_percent") or 0.0) if allocation_doc else 0.0
    validator_lab_cap_ceiling = _validator_lab_cap_ceiling_percent()
    paid = sum(
        float(allocation_doc.get(field) or 0.0)
        for field in (
            "source_add_alpha_percent",
            "reimbursement_alpha_percent",
            "champion_alpha_percent",
            "queued_champion_alpha_percent",
            "unallocated_percent",
        )
    )
    if lab_cap < 0 or lab_cap > 100:
        errors.append("invalid_lab_cap_percent")
    if lab_cap > validator_lab_cap_ceiling + PERCENT_EPSILON:
        errors.append("allocation_lab_cap_exceeds_validator_policy")
    if paid > lab_cap + PERCENT_EPSILON:
        errors.append("allocation_exceeds_lab_cap")

    recomputed_allocation_hash: str | None = None
    policy = source_state.get("policy") if isinstance(source_state, Mapping) else None
    source_add = source_state.get("source_add_obligations") if isinstance(source_state, Mapping) else None
    reimbursements = source_state.get("reimbursement_obligations") if isinstance(source_state, Mapping) else None
    fallback_reimbursements = (
        source_state.get("fallback_reimbursement_obligations")
        if isinstance(source_state, Mapping)
        else None
    )
    champions = source_state.get("champion_obligations") if isinstance(source_state, Mapping) else None
    if not isinstance(policy, Mapping):
        errors.append("allocation_policy_required")
    if not isinstance(reimbursements, list):
        errors.append("allocation_reimbursement_obligations_must_be_array")
        reimbursements = []
    if fallback_reimbursements is None:
        fallback_reimbursements = []
    elif not isinstance(fallback_reimbursements, list):
        errors.append(
            "allocation_fallback_reimbursement_obligations_must_be_array"
        )
        fallback_reimbursements = []
    if source_add is None:
        source_add = []
    elif not isinstance(source_add, list):
        errors.append("allocation_source_add_obligations_must_be_array")
        source_add = []
    if not isinstance(champions, list):
        errors.append("allocation_champion_obligations_must_be_array")
        champions = []
    if isinstance(policy, Mapping):
        try:
            policy_lab_cap = float(policy.get("research_lab_emission_percent") or 0.0)
        except (TypeError, ValueError):
            errors.append("allocation_policy_lab_cap_invalid")
            policy_lab_cap = 0.0
        if policy_lab_cap > validator_lab_cap_ceiling + PERCENT_EPSILON:
            errors.append("allocation_policy_cap_exceeds_validator_policy")
        if abs(policy_lab_cap - lab_cap) > PERCENT_EPSILON:
            errors.append("allocation_policy_cap_diverged_from_doc")
        if isinstance(source_state, Mapping) and source_state.get("policy_id") is not None:
            if str(policy.get("policy_id") or "") != str(source_state.get("policy_id") or ""):
                errors.append("allocation_policy_id_diverged")
        try:
            allocation_epoch = int(source_state.get("epoch", bundle.get("epoch")))
            recomputed = allocate_research_lab_epoch(
                allocation_epoch,
                policy,
                reimbursements,
                champions,
                active_source_add_obligations=source_add,
                fallback_reimbursement_obligations=fallback_reimbursements,
            )
            recomputed_allocation_hash = str(recomputed.get("allocation_hash") or "")
            if allocation_hash and recomputed_allocation_hash != str(allocation_hash):
                errors.append("allocation_recompute_hash_diverged")
            if allocation_doc and dict(recomputed) != dict(allocation_doc):
                errors.append("allocation_recompute_doc_diverged")
        except Exception as exc:
            errors.append(f"allocation_recompute_failed:{str(exc)[:120]}")

    return {
        "passed": not errors,
        "errors": errors,
        "epoch": bundle.get("epoch"),
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": source_state_hash,
        "allocation_hash": allocation_hash,
        "recomputed_allocation_hash": recomputed_allocation_hash,
        "validator_lab_cap_ceiling_percent": validator_lab_cap_ceiling,
        "allocation_doc": dict(allocation_doc or {}),
        "on_chain_submission_allowed": not errors,
    }










def build_research_lab_allocation_component(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verification = verify_research_lab_allocation_bundle(bundle, flags=flags)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    allocation_doc = dict(verification["allocation_doc"])
    return {
        "epoch": int(bundle["epoch"]),
        "shadow_only": False,
        "read_only": False,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": bundle.get("source_state_hash"),
        "allocation_hash": verification["allocation_hash"],
        "allocation_doc": allocation_doc,
        "observability": dict(bundle.get("observability") or {}),
    }








def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if any(
                marker in lowered_key
                for marker in (
                    "api_key",
                    "raw_secret",
                    "raw_openrouter",
                    "credential",
                    "private_model_manifest_doc",
                    "candidate_patch_manifest",
                    "image_digest",
                    "proxy_url",
                )
            ):
                return True
            if _contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(
            marker in lowered
            for marker in (
                "sk-or-",
                "raw_openrouter_key",
                "openrouter_api_key",
                "raw_secret",
                "hidden_icp",
                "icp_plaintext",
                ".dkr.ecr.",
                "private_repo",
                "judge_prompt",
            )
        )
    return False








def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES
