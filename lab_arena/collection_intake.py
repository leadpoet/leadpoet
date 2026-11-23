"""Miner consent, key enrollment and trusted job leases for locked collection.

Only the gateway/collection worker can construct this object. The public routes
expose terms, authenticated enrollment, revocation and the signer's statement.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from lab_arena import contracts, store
from lab_arena.credentials import OPENROUTER_FUNDING_FIELDS


class GeneratorRoute(NamedTuple):
    revision: str
    tag: str


# Slot order is part of the batch. Evidence for both APIs is in docs/locked-collection.md.
GENERATOR_ROUTES = {
    "moonshotai/kimi-k3.1-pro": GeneratorRoute(
        "moonshotai/kimi-k3.1-pro-20261023", "deepinfra/bf16"
    ),
    "deepseek/deepseek-v4-pro-0813": GeneratorRoute(
        "deepseek/deepseek-v4-pro-20260813", "deepinfra/fp8"
    ),
    "mistralai/mistral-large-4-0": GeneratorRoute(
        "mistralai/mistral-large-4-0-20261006", "mistral/zdr"
    ),
    "qwen/qwen3.8-27b": GeneratorRoute(
        "qwen/qwen3.8-27b-20260814", "deepinfra/fp8"
    ),
    "nvidia/nemotron-3.5-super-vl": GeneratorRoute(
        "nvidia/nemotron-3.5-super-vl-20261015", "deepinfra/fp4"
    ),
    "deepseek/deepseek-v4.1-flash-1017": GeneratorRoute(
        "deepseek/deepseek-v4.1-flash-20261017", "particle-ai/us/flex"
    ),
}
MODELS = tuple(GENERATOR_ROUTES)
PROVIDERS = ("openrouter",)
TASK_SCOPE = "synthetic_workflows"
SYNTHETIC_SCENARIO = re.compile(r"(?:C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-6])-s[1-9][0-9]{0,7}")
SCOPE = {
    op: "lab_arena.collection." + op + ".v1"
    for op in ("register", "revoke", "statement")
}
TERMS = (
    "I authorize Leadpoet to spend on my OpenRouter key for centrally assigned simulated sales workflows "
    "within the rolling 24-hour limit I sign below. I will keep input/output logging, product-use sharing "
    "and external trace broadcast disabled for this key. The scenarios use invented data and local tools. "
    "My enrollment does not authorize live web research, third-party enrichment or Deepline charges. "
    "Leadpoet funds verification and ranking. The read-only management key is used only to verify the runtime key. "
    "Revocation stops new calls. Already dispatched calls can still be billed. I will keep private run data confidential. "
    "Pilot jobs incur real provider charges and earn no on-chain reward. Authentic completed live jobs earn the "
    "assigned fixed run quote under the epoch settlement rules; spending more does not increase "
    "the quote. If total quotes exceed available epoch emissions, payouts scale proportionally. Unpaid amounts do not carry forward."
)
TERMS_HASH = contracts.hash_bytes(TERMS.encode())
LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
SHA = re.compile(r"sha256:[0-9a-f]{64}")
MAX_MICROUSD = 10_000_000_000
_SIGNATURES = {
    "register": (
        ("id", "uuid"),
        ("hotkey", "text"),
        ("owner", "text"),
        ("mode", "text"),
        ("consent", "jsonb"),
        ("cap", "bigint"),
        ("openrouter", "jsonb"),
        ("deepline", "jsonb"),
    ),
    "revoke": (("id", "uuid"), ("hotkey", "text"), ("mode", "text")),
    "claim": (
        ("job", "uuid"),
        ("token_hash", "text"),
        ("batch", "text"),
        ("mode", "text"),
        ("commit", "text"),
        ("scenarios", "jsonb"),
        ("models", "jsonb"),
        ("prices", "jsonb"),
        ("budgets", "jsonb"),
        ("terms_hash", "text"),
    ),
    "credential": (("job", "uuid"), ("token_hash", "text"), ("provider", "text")),
    "finish": (
        ("job", "uuid"),
        ("token_hash", "text"),
        ("status", "text"),
        ("quality", "jsonb"),
        ("artifact", "text"),
        ("epoch", "bigint"),
    ),
    "prepare_launch": (
        ("job", "uuid"), ("token_hash", "text"), ("quality", "jsonb"), ("artifact", "text"),
    ),
    "launch_finish": (("job", "uuid"), ("anchor", "jsonb")),
    "statement": (("hotkey", "text"), ("mode", "text")),
}


class IntakeError(ValueError):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


def live_finney_arena():
    # Match the production gateway defaults in lab_arena.wiring.
    return (
        os.environ.get("LAB_ARENA_MODE") == "live"
        and os.environ.get("LAB_ARENA_NETWORK", "finney") == "finney"
        and os.environ.get("LAB_ARENA_NETUID", "71") == "71"
    )


def mode_from_environment():
    mode = os.environ.get("LAB_ARENA_COLLECTION_MODE", "off")
    if mode not in {"off", "pilot", "live"}:
        raise IntakeError("collection_mode_invalid", 503)
    return mode


def terms(mode):
    return {
        "mode": mode,
        "task_scope": TASK_SCOPE,
        "terms": TERMS,
        "terms_sha256": TERMS_HASH,
        "network": "finney",
        "netuid": 71,
    }


def scope_id(mode):
    return f"finney:71:locked-collection:{mode}"


def _micros(value, minimum):
    if type(value) is not int or not minimum <= value <= MAX_MICROUSD:
        raise IntakeError("collection_money_invalid")
    return value


def _uuid(value):
    try:
        parsed = uuid.UUID(value)
        if str(parsed) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise IntakeError("collection_id_invalid")


def register_signatures():
    store.FUNCTION_SIGNATURES.setdefault(
        "collection_reward_statement_v1", (("p_hotkey", "text"),)
    )
    for name, fields in _SIGNATURES.items():
        signature = tuple(("p_" + k, t) for k, t in fields)
        existing = store.FUNCTION_SIGNATURES.setdefault(
            "collection_" + name + "_v1", signature
        )
        if tuple(existing) != signature:
            raise RuntimeError("collection RPC signature conflict")


class CollectionIntake:
    def __init__(self, transport, credential_manager, mode):
        if mode not in ("pilot", "live"):
            raise IntakeError("collection_inactive", 404)
        register_signatures()
        self.transport, self.credential_manager, self.mode = (
            transport,
            credential_manager,
            mode,
        )

    def _rpc(self, name, **params):
        result = self.transport.rpc(
            f"collection_{name}_v1", {"p_" + k: v for k, v in params.items()}
        )
        if isinstance(result, Mapping) and "error" in result:
            raise IntakeError(result["error"], 409)
        return result

    def register(self, signed, owner, credentials):
        """Caller authenticates the scoped signed envelope. Only that envelope is retained."""
        body, hk = signed["body"], signed["hotkey"]
        if (
            set(body) != {"credentials_sha256", "terms_sha256", "daily_cap_microusd", "task_scope"}
            or body["task_scope"] != TASK_SCOPE
            or body["terms_sha256"] != TERMS_HASH
            or signed["round_id"] != scope_id(self.mode)
            or not isinstance(credentials, dict)
            or set(credentials) != set(OPENROUTER_FUNDING_FIELDS)
            or not all(isinstance(v, str) for v in credentials.values())
            or body["credentials_sha256"] != contracts.document_hash(credentials)
        ):
            raise IntakeError("collection_consent_invalid")
        cap = _micros(body["daily_cap_microusd"], 1)
        registration = str(uuid.UUID(hex=signed["request_id"]))
        prefix = "collection:" + registration
        encrypted = self.credential_manager.validate_openrouter_funding(
            credentials, submission_id=prefix, miner_hotkey=hk
        )
        key = {
            "key_id": str(uuid.uuid5(uuid.UUID(registration), "openrouter")),
            "fingerprint": contracts.hash_bytes(credentials["openrouter_api_key"].encode()),
            "ciphertext_b64": encrypted,
        }
        return self._rpc(
            "register",
            id=registration,
            hotkey=hk,
            owner=owner,
            mode=self.mode,
            consent=signed,
            cap=cap,
            openrouter=key,
            deepline=None,
        )

    def revoke(self, hotkey, registration_id):
        return self._rpc(
            "revoke", id=_uuid(registration_id), hotkey=hotkey, mode=self.mode
        )

    def statement(self, hotkey):
        statement = self._rpc("statement", hotkey=hotkey, mode=self.mode)
        if self.mode == "live":
            statement["weight_targets"] = self.transport.rpc(
                "collection_reward_statement_v1", {"p_hotkey": hotkey}
            )
        return statement

    def claim(
        self,
        batch_id,
        scenario_ids,
        harness_commit,
        prices_microusd,
        budgets_microusd,
        *,
        job_id=None,
        lease_token=None,
    ):
        if not isinstance(batch_id, str) or not LABEL.fullmatch(batch_id):
            raise IntakeError("collection_batch_invalid")
        if not isinstance(harness_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", harness_commit
        ):
            raise IntakeError("collection_harness_unpinned")
        if (
            not isinstance(scenario_ids, Sequence)
            or isinstance(scenario_ids, (str, bytes))
            or not 1 <= len(scenario_ids) <= 4096
            or any(
                not isinstance(s, str) or not LABEL.fullmatch(s) for s in scenario_ids
            )
            or len(set(scenario_ids)) != len(scenario_ids)
        ):
            raise IntakeError("collection_scenarios_invalid")
        if not all(SYNTHETIC_SCENARIO.fullmatch(s) for s in scenario_ids):
            raise IntakeError("collection_synthetic_scope_required")
        for values, minimum in (
            (prices_microusd, 1 if self.mode == "live" else 0),
            (budgets_microusd, 1),
        ):
            if not isinstance(values, Mapping) or set(values) != set(MODELS):
                raise IntakeError("collection_price_coverage_invalid")
            for amount in values.values():
                _micros(amount, minimum)
        if (job_id is None) != (lease_token is None):
            raise IntakeError("collection_claim_identity_invalid")
        token = lease_token if lease_token is not None else store.new_lease_token()
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
            raise IntakeError("collection_lease_invalid")
        job = self._rpc(
            "claim",
            job=_uuid(job_id) if job_id else str(uuid.uuid4()),
            token_hash=store.hash_lease_token(token),
            batch=batch_id,
            mode=self.mode,
            commit=harness_commit,
            scenarios=list(scenario_ids),
            models=list(MODELS),
            prices=dict(prices_microusd),
            budgets=dict(budgets_microusd),
            terms_hash=TERMS_HASH,
        )
        return dict(job, lease_token=token) if job else None

    def credential(self, job_id, lease_token, provider):
        if provider not in PROVIDERS:
            raise IntakeError("collection_provider_invalid")
        row = self._rpc(
            "credential",
            job=_uuid(job_id),
            token_hash=store.hash_lease_token(lease_token),
            provider=provider,
        )
        if not row:
            raise IntakeError("collection_key_unavailable", 403)
        return self.credential_manager.runtime_key(row, provider)

    def complete(self, job_id, lease_token, status, quality, artifact_sha256, epoch, *, prepare_launch=False):
        if (
            status not in ("accepted", "rejected", "failed")
            or type(epoch) is not int
            or epoch < 0
        ):
            raise IntakeError("collection_outcome_invalid")
        if (artifact_sha256 is not None or status != "failed") and not SHA.fullmatch(
            str(artifact_sha256)
        ):
            raise IntakeError("collection_artifact_invalid")
        _validate_quality(quality)
        if prepare_launch:
            if status != "accepted" or self.mode != "live":
                raise IntakeError("collection_launch_invalid")
            return self._rpc(
                "prepare_launch", job=_uuid(job_id), token_hash=store.hash_lease_token(lease_token),
                quality=quality, artifact=artifact_sha256,
            )
        return self._rpc(
            "finish",
            job=_uuid(job_id),
            token_hash=store.hash_lease_token(lease_token),
            status=status,
            quality=quality,
            artifact=artifact_sha256,
            epoch=epoch,
        )


def _validate_quality(summary):
    if not isinstance(summary, dict) or set(summary) - {
        "scores",
        "failure_tags",
        "training_status",
    }:
        raise IntakeError("collection_quality_invalid")
    if "training_status" in summary and summary["training_status"] not in {
        "code_rejected",
        "rejected",
        "salvaged",
        "needs_review",
        "cannot_judge",
        "accepted",
    }:
        raise IntakeError("collection_training_status_invalid")
    if "scores" in summary:
        scores = summary["scores"]
        if (
            not isinstance(scores, dict)
            or set(scores)
            != {"truth", "decision", "completeness", "efficiency", "safety"}
            or any(type(v) is not int or not 0 <= v <= 3 for v in scores.values())
        ):
            raise IntakeError("collection_scores_invalid")
    if "failure_tags" in summary:
        tags = summary["failure_tags"]
        from lab_arena.collection_judge import TAGS

        if (
            not isinstance(tags, list)
            or len(tags) > 5
            or any(type(tag) is not str or tag not in TAGS for tag in tags)
        ):
            raise IntakeError("collection_tags_invalid")
