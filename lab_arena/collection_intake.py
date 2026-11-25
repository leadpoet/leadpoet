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
SYNTHETIC_SCENARIO = re.compile(
    r"(?:C[0-9]{2}|K[0-9]{2}|B[1-8]|E[1-7]|P[1-7])-s[1-9][0-9]{0,7}"
)
SCOPE = {
    op: "lab_arena.collection." + op + ".v1"
    for op in ("register", "revoke", "statement")
}
CORRECTION_SCOPE = "synthetic_corrections"
# Keep historical consent identifiable without upgrading its permissions.
LEGACY_COMPARISON_TERMS_HASH = "sha256:df00d01897792eca6eb2aa88eab4a16dcba253ef3c8998581a124dead92686ca"
LEGACY_CORRECTION_TERMS_HASHES = {
    "pilot": "sha256:41588b51710c8218e5dce56ef12386709e4ebd2409ca76fe4427251425cc6119",
    "live": "sha256:88d5b6886024e7aa5960c3d76586e2831d96650c071a348b145d19a7e155ba6c",
}

SYNTHETIC_TRAINING_TERMS = (
    "I authorize Leadpoet to charge my dedicated OpenRouter key up to my signed rolling 24-hour limit "
    "for the synthetic sales tasks, scope and payment mode in this enrollment. The subnet assigns the "
    "agent and model. Tasks may include model comparisons, repeat runs or continuations from a saved state. "
    "Failed and partial attempts can incur charges. Pilot work pays no emissions. In live mode an authentic "
    "completed assignment earns its approved fixed quote, regardless of spend or model performance. "
    "If quotes exceed the epoch's available emissions, each quote is scaled proportionally. Unpaid "
    "differences do not carry forward. Leadpoet pays for private verification. "
    "Leadpoet may record messages, original reasoning exposed by the provider, tools, results, evidence, "
    "task states, costs, checks and outcomes from successful, failed and partial runs. I assign to Leadpoet "
    "any rights I hold in those outputs. Where assignment is unavailable, I grant Leadpoet a perpetual, "
    "worldwide, royalty-free, transferable and sublicensable license to store, evaluate, train on, publish "
    "and license them, including as complete trajectories or chosen/rejected pairs. Other rights holders' "
    "licenses and privacy protections still apply. I will keep private run data confidential, keep no "
    "copies beyond my funding statements, and disable input/output logging, sharing and external trace "
    "broadcast on the enrolled key. The management key is used only to verify the runtime key. I may "
    "revoke access at any time. Revocation blocks new calls. Already dispatched calls may still be billed. "
    "This authorization covers only synthetic tasks using OpenRouter and local tools. Live research, hosted "
    "search and Deepline charges require separate authorization. Changing mode or scope requires a new signature."
)
SYNTHETIC_TRAINING_TERMS_HASH = contracts.hash_bytes(SYNTHETIC_TRAINING_TERMS.encode())
# These public names are used by existing miners and configuration tooling.
TERMS = CORRECTION_TERMS = LIVE_CORRECTION_TERMS = SYNTHETIC_TRAINING_TERMS
TERMS_HASH = SYNTHETIC_TRAINING_TERMS_HASH


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


def correction_terms_hash(mode="pilot"):
    if mode not in ("pilot", "live"):
        raise IntakeError("collection_scope_unavailable", 422)
    setting = "LAB_ARENA_CORRECTION_TERMS_SHA256" if mode == "pilot" else "LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256"
    digest = SYNTHETIC_TRAINING_TERMS_HASH
    if os.getenv(setting, digest) != digest:
        raise IntakeError("collection_correction_terms_unavailable", 503)
    return digest


def require_correction_grant(mode, digest):
    """Keep signed legacy work valid within its original mode and task scope."""
    current = correction_terms_hash(mode)
    if digest not in (current, LEGACY_CORRECTION_TERMS_HASHES[mode]):
        raise IntakeError("collection_correction_terms_changed")
    return digest


def terms(mode, task_scope=TASK_SCOPE):
    if task_scope not in (TASK_SCOPE, CORRECTION_SCOPE) or mode not in ("pilot", "live"):
        raise IntakeError("collection_scope_unavailable", 422)
    if task_scope == CORRECTION_SCOPE:
        correction_terms_hash(mode)
    return {
        "mode": mode,
        "task_scope": task_scope,
        "max_selectable_slots_per_utc_day": MAX_DAILY_SLOTS,
        "terms": SYNTHETIC_TRAINING_TERMS,
        "terms_sha256": SYNTHETIC_TRAINING_TERMS_HASH,
        "network": "finney",
        "netuid": 71,
    }


def open_correction_quotes(transport):
    register_signatures()
    quotes = transport.rpc("collection_open_correction_quotes_v1", {})
    if not isinstance(quotes, list) or not quotes:
        raise IntakeError("collection_compensated_corrections_unavailable", 503)
    return quotes


def scope_id(mode):
    return f"finney:71:locked-collection:{mode}"


MAX_DAILY_SLOTS = 6
REGISTRATION_FIELDS = {"credentials_sha256", "terms_sha256", "daily_cap_microusd", "task_scope"}


def signed_daily_slots(body):
    if set(body) - {"max_slots_per_utc_day"} != REGISTRATION_FIELDS:
        raise IntakeError("collection_consent_invalid")
    slots = body.get("max_slots_per_utc_day", 1)
    if type(slots) is not int or not 1 <= slots <= MAX_DAILY_SLOTS:
        raise IntakeError("collection_consent_invalid")
    if "max_slots_per_utc_day" in body and body["terms_sha256"] != SYNTHETIC_TRAINING_TERMS_HASH:
        raise IntakeError("collection_consent_invalid")
    return slots


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
        "collection_priced_claim_v1",
        tuple(("p_" + name, kind) for name, kind in _SIGNATURES["claim"]) + (("p_expected_model", "text"),),
    )
    store.FUNCTION_SIGNATURES.setdefault("collection_training_consents_v1", (("p_batch", "text"),))
    store.FUNCTION_SIGNATURES.setdefault("collection_open_correction_quotes_v1", ())
    store.FUNCTION_SIGNATURES.setdefault("collection_correction_tariff_v1", (("p_batch", "text"),))
    store.FUNCTION_SIGNATURES.setdefault("collection_experiment_costs_v1", (("p_batch", "text"),))
    store.FUNCTION_SIGNATURES.setdefault("collection_correction_measurement_v1", (("p_batch", "text"),))
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
        signed_daily_slots(body)
        if (
            body["terms_sha256"] != terms(self.mode, body["task_scope"])["terms_sha256"]
            or signed["round_id"] != scope_id(self.mode)
            or not isinstance(credentials, dict)
            or set(credentials) != set(OPENROUTER_FUNDING_FIELDS)
            or not all(isinstance(v, str) for v in credentials.values())
            or body["credentials_sha256"] != contracts.document_hash(credentials)
        ):
            raise IntakeError("collection_consent_invalid")
        if self.mode == "live" and body["task_scope"] == CORRECTION_SCOPE:
            open_correction_quotes(self.transport)
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
        terms_hash=TERMS_HASH,
        expected_model="*",
        job_id=None,
        lease_token=None,
    ):
        if terms_hash not in (LEGACY_COMPARISON_TERMS_HASH, SYNTHETIC_TRAINING_TERMS_HASH):
            raise IntakeError("collection_terms_changed")
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
        return self._lease(token, expected_model,
            job=_uuid(job_id) if job_id else str(uuid.uuid4()),
            token_hash=store.hash_lease_token(token),
            batch=batch_id,
            mode=self.mode,
            commit=harness_commit,
            scenarios=list(scenario_ids),
            models=list(MODELS),
            prices=dict(prices_microusd),
            budgets=dict(budgets_microusd),
            terms_hash=terms_hash,
        )

    def _lease(self, token, expected_model, **arguments):
        if expected_model == "*":
            job = self._rpc("claim", **arguments)
        else:
            job = self._rpc("priced_claim", expected_model=expected_model, **arguments)
            if expected_model is None:
                return job
        return dict(job, lease_token=token) if job else None

    def claim_experiment(self, experiment, harness_commit, *, fixed_quote=0, expected_model="*"):
        """Claim exactly one frozen slot under its own mode and signed consent."""
        if self.mode != experiment.raw["mode"] or not re.fullmatch(r"xp-[0-9a-f]{64}", experiment.batch_id):
            raise IntakeError("collection_correction_pilot_required")
        terms_hash = require_correction_grant(self.mode, experiment.raw["terms_sha256"])
        _micros(fixed_quote, 0 if self.mode == "pilot" else 1)
        if self.mode == "pilot" and fixed_quote:
            raise IntakeError("pilot_quote_must_be_zero")
        if not re.fullmatch("[0-9a-f]{40}", harness_commit):
            raise IntakeError("collection_harness_unpinned")
        token = store.new_lease_token()
        models = [arm["model"] for _, arm, _ in experiment.slots]
        return self._lease(token, expected_model,
            job=str(uuid.uuid4()), token_hash=store.hash_lease_token(token),
            batch=experiment.batch_id, mode=self.mode, commit=harness_commit,
            scenarios=[case["scenario_id"] for case, _, _ in experiment.slots], models=models,
            prices=dict.fromkeys(models, fixed_quote),
            budgets={arm["model"]:arm["budget_microusd"] for _, arm, _ in experiment.slots},
            terms_hash=terms_hash,
        )

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
        "claim_rejected",
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
