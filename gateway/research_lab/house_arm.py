"""House-funded baseline arm scheduler (fableanalysis §9.3 / fablefollowup 4.3).

Opens Research Lab loops funded by the house through the EXISTING miner queue —
zero new execution machinery and no new tables. House runs are ordinary runs
distinguished only by tags (``ticket_doc.arm = "house"`` +
``ticket_doc.house_policy_ref``) and by the dedicated house hotkey.

Safety posture — everything defaults to NOT spending:
  * ``RESEARCH_LAB_HOUSE_ARM_ENABLED`` (master flag, default FALSE) must be
    truthy AND the caller must pass ``dry_run=False`` (CLI ``--no-dry-run``)
    before a single row is written. Dry-run is pure read-only planning.
  * The daily budget is clamped hard at the
    ``BaselineArmOperatingPolicyRecord`` policy max ($500/day) regardless of
    arguments; today's already-recorded house spend is subtracted first.
  * The scripts/43 + scripts/54/67 DB capacity triggers apply unchanged:
    active loops per hotkey are capped by the shared gateway policy. The
    queued event carries the same
    ``autoresearch_capacity`` doc miner loops carry so admission stays atomic.

Ticket-opening path: the exact store primitives the miner API uses —
``research_loop_tickets`` insert (mirroring ``store.create_ticket``'s row
shape; see ``_create_house_ticket``), ``create_loop_start_payment``,
``create_ticket_event`` ("funded" / "queued"), and ``create_queue_event`` with
``autoresearch_queue_capacity_doc``. Funding uses the house OpenRouter key
REF from ``RESEARCH_LAB_HOUSE_OPENROUTER_KEY_REF`` (an ``encrypted_ref:...``
row in ``research_lab_openrouter_key_refs`` registered for the house hotkey,
or an env-mapped ephemeral ref) — raw keys are rejected, matching how miner
key refs are stored.

Direction selection: round-robin over the loop-direction planner lanes (read
from ``research_lab.code_editing``'s planner context). When the §9.4
meta-allocator priors module lands, its ``CellYieldPriorRecord`` selection
becomes the aiming input here in place of round-robin (fablefollowup 4.4).

Scheduling: cron/manual for now (``python3 -m gateway.research_lab.house_arm``);
deliberately NOT wired into the hosted worker pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# When admin/house-arm commands are launched from /home/ec2-user/gateway, the
# gateway package directory shadows the shared top-level research_lab package.
_GATEWAY_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = Path(__file__).resolve().parents[2]
_package_parent_str = str(_PACKAGE_PARENT)
if _package_parent_str in sys.path:
    sys.path.remove(_package_parent_str)
sys.path.insert(0, _package_parent_str)
_existing_research_lab = sys.modules.get("research_lab")
_existing_file = str(getattr(_existing_research_lab, "__file__", "") or "")
if _existing_file:
    try:
        if os.path.commonpath([str(Path(_existing_file).resolve()), str(_GATEWAY_ROOT)]) == str(_GATEWAY_ROOT):
            sys.modules.pop("research_lab", None)
    except (OSError, ValueError):
        pass

from research_lab.baseline_arm_ops import (
    BASELINE_DAILY_BUDGET_MAX_CENTS,
    BASELINE_DAILY_BUDGET_MIN_CENTS,
    BaselineArmOperatingPolicyRecord,
    REFERENCE_BASELINE_MODEL_ID,
    RESEARCH_LAB_ARM_B_MODEL_ID,
    validate_baseline_arm_operating_policy_record,
)
from research_lab.counterfactual_gate import (
    CounterfactualYieldRecord,
    build_matched_budget_comparison,
)

from .config import DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS, ResearchLabGatewayConfig
from .maintenance import (
    autoresearch_queue_capacity_doc,
    dumps_status,
    get_autoresearch_maintenance_state,
)
from .public_activity import safe_project_public_loop_activity
from .store import (
    canonical_hash,
    create_loop_start_payment,
    create_queue_event,
    create_ticket_event,
    deterministic_uuid,
    insert_row,
    payment_ref_exists,
    select_all,
    select_many,
    select_one,
)


logger = logging.getLogger(__name__)

HOUSE_ARM_ENABLED_ENV = "RESEARCH_LAB_HOUSE_ARM_ENABLED"
HOUSE_HOTKEY_ENV = "RESEARCH_LAB_HOUSE_HOTKEY"
HOUSE_OPENROUTER_KEY_REF_ENV = "RESEARCH_LAB_HOUSE_OPENROUTER_KEY_REF"

HOUSE_ARM_TAG = "house"
HOUSE_POLICY_REF = "baseline_arm_policy:house:v1"
HOUSE_TICKET_SOURCE = "gateway_research_lab_house_arm"
# §9.4 aiming input placeholder until the meta-allocator priors module lands.
HOUSE_ALLOCATOR_SELECTION_REF = "meta_allocator_selection:house_round_robin:v0"
HOUSE_COMPARISON_METHODOLOGY_REF = "counterfactual_methodology:matched_budget_house_arm:v1"
HOUSE_SHARED_PRIOR_CAVEAT = (
    "Both arms run the same champion, eval harness, and shared priors; verified "
    "points are scaled linearly to the matched budget for comparison."
)

# Fallback copy of the loop-direction planner lanes. ``planner_lanes()`` reads
# the live list from research_lab.code_editing's planner context first so the
# round-robin follows the planner; this constant only guards against import or
# parsing failures.
DEFAULT_PLANNER_LANES: tuple[str, ...] = (
    "icp_normalization",
    "query_construction",
    "source_routing",
    "provider_fallback",
    "intent_evidence_quality",
    "company_fit_filtering",
    "openrouter_model_selection",
    "output_ranking",
)

_RAW_KEY_MARKERS = ("sk-or-", "sk_or_")
ACTIVE_HOUSE_QUEUE_STATUSES = ("paused", "queued", "started")


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def house_arm_enabled() -> bool:
    """Master spend flag — default FALSE; house loops spend real money."""

    return _truthy_env(HOUSE_ARM_ENABLED_ENV)


def house_hotkey() -> str:
    return str(os.getenv(HOUSE_HOTKEY_ENV, "")).strip()


def house_openrouter_key_ref() -> str:
    return str(os.getenv(HOUSE_OPENROUTER_KEY_REF_ENV, "")).strip()


def house_key_handling(key_ref: str) -> str:
    """Mirror the miner key-handling values (encrypted_ref | ephemeral_ref)."""

    if str(key_ref).startswith("encrypted_ref:openrouter:"):
        return "encrypted_ref"
    return "ephemeral_ref"


def planner_lanes() -> tuple[str, ...]:
    """Read the loop-direction planner lane list from research_lab.code_editing.

    The lanes are defined inline in the planner-context builder rather than as
    a module constant, so extract them from the rendered context JSON; fall
    back to the pinned copy if extraction fails.
    """

    try:
        from research_lab.code_editing import build_loop_direction_planner_messages

        messages = build_loop_direction_planner_messages(
            ticket={},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            runtime_source_index={},
            budget_context={},
        )
        user_content = str(messages[-1]["content"])
        context_json = user_content.split("Context JSON:\n", 1)[1]
        lanes = json.loads(context_json).get("allowed_lanes") or []
        cleaned = tuple(str(lane).strip() for lane in lanes if str(lane).strip())
        if cleaned:
            return cleaned
    except Exception as exc:  # noqa: BLE001 - fall back to the pinned lane list
        logger.warning("house_arm_planner_lane_extraction_failed: %s", str(exc)[:200])
    return DEFAULT_PLANNER_LANES


def next_house_lane(last_lane: str | None, lanes: Sequence[str] | None = None) -> str:
    """Round-robin lane rotation. Allocator priors (§9.4, another module)
    replace this as the aiming input once live."""

    lane_list = list(lanes if lanes is not None else planner_lanes())
    if not lane_list:
        raise ValueError("house arm has no planner lanes to rotate over")
    if last_lane in lane_list:
        return lane_list[(lane_list.index(last_lane) + 1) % len(lane_list)]
    return lane_list[0]


def house_lane_focus(lane: str) -> str:
    """One-line public research focus the planner treats as the miner brief."""

    return (
        f"House baseline arm ({lane}): improve the {lane.replace('_', ' ')} "
        "stage of the qualification pipeline with one safe, testable mechanism."
    )


def build_house_arm_operating_policy(config: ResearchLabGatewayConfig) -> BaselineArmOperatingPolicyRecord:
    """§9.3 operating policy in its validated (inert) P1.9 shape.

    The record carries the hard $200–$500/day clamps consumed here; the live
    decision to spend is the HOUSE_ARM env flag + ``dry_run=False``, never a
    record field (the P1.9 validators reject spend/scheduler-enabled records
    by design, which keeps this policy honest as configuration-only).
    """

    island = (config.allowed_research_islands or ("generalist",))[0]
    policy = BaselineArmOperatingPolicyRecord(
        policy_id=HOUSE_POLICY_REF,
        island=str(island),
        house_arm_ref="baseline_house_arm:leadpoet_research_lab",
        reference_model_id=REFERENCE_BASELINE_MODEL_ID,
        allocator_directed_model_id=RESEARCH_LAB_ARM_B_MODEL_ID,
        daily_budget_min_cents=BASELINE_DAILY_BUDGET_MIN_CENTS,
        daily_budget_max_cents=BASELINE_DAILY_BUDGET_MAX_CENTS,
        budget_currency="USD",
        island_scope_refs=(f"island:{island}",),
        exploration_objective_refs=("objective:planner_lane_round_robin:v0",),
        comparison_methodology_ref=HOUSE_COMPARISON_METHODOLOGY_REF,
    )
    errors = validate_baseline_arm_operating_policy_record(policy)
    if errors:
        raise ValueError("house arm operating policy invalid: " + "; ".join(errors))
    return policy


def house_per_loop_usd(config: ResearchLabGatewayConfig) -> float:
    """Planned house outlay per loop counted against the daily clamp.

    Per fablefollowup 4.3 the loop-start fee comes from config
    ``loop_start_fee_usd``; the compute budget is the real money that leaves
    through the house OpenRouter key, so both count (conservative)."""

    return float(config.loop_start_fee_usd) + float(config.default_compute_budget_usd)


def _utc_day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _house_run_id(ticket_id: str) -> str:
    """Deterministic run id (one run per house ticket) so a crashed opening
    pass can be resumed idempotently instead of leaking paid-but-unqueued runs."""

    return deterministic_uuid("house_arm_run", str(ticket_id))


def _payment_spend_usd(row: Mapping[str, Any]) -> float:
    """House spend recorded on one loop-start payment row: fee + compute budget."""

    try:
        fee = float(row.get("required_usd") or 0.0)
    except (TypeError, ValueError):
        fee = 0.0
    doc = row.get("verification_doc") if isinstance(row.get("verification_doc"), Mapping) else {}
    try:
        compute = float(doc.get("compute_budget_usd") or 0.0)
    except (TypeError, ValueError):
        compute = 0.0
    return fee + compute


async def _todays_house_spend_usd(hotkey: str, *, now: datetime) -> float:
    rows = await select_all(
        "research_loop_start_payments",
        columns="payment_id,required_usd,verification_doc,created_at,payment_status",
        filters=(
            ("miner_hotkey", hotkey),
            ("created_at", "gte", _utc_day_start(now).isoformat()),
        ),
        max_rows=2000,
    )
    return round(
        sum(_payment_spend_usd(row) for row in rows if str(row.get("payment_status") or "") == "verified"),
        6,
    )


def _row_is_fresh(row: Mapping[str, Any], config: ResearchLabGatewayConfig, *, now: datetime) -> bool:
    """Same stale-active window the miner admission path applies."""

    stale_after_seconds = max(
        60,
        int(config.active_loop_stale_after_seconds or DEFAULT_ACTIVE_LOOP_STALE_AFTER_SECONDS),
    )
    raw_status_at = row.get("current_status_at")
    if not raw_status_at:
        return True
    try:
        status_at = datetime.fromisoformat(str(raw_status_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if status_at.tzinfo is None:
        status_at = status_at.replace(tzinfo=timezone.utc)
    age_seconds = (now.astimezone(timezone.utc) - status_at.astimezone(timezone.utc)).total_seconds()
    return age_seconds <= stale_after_seconds


def _ticket_doc(ticket: Mapping[str, Any]) -> Mapping[str, Any]:
    doc = ticket.get("ticket_doc")
    return doc if isinstance(doc, Mapping) else {}


def _is_house_ticket(ticket: Mapping[str, Any] | None, hotkey: str) -> bool:
    if not ticket:
        return False
    if str(ticket.get("miner_hotkey") or "").strip() != hotkey:
        return False
    return str(_ticket_doc(ticket).get("arm") or "") == HOUSE_ARM_TAG


async def _house_tickets(hotkey: str, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = await select_all(
        "research_loop_ticket_current",
        filters=(("miner_hotkey", hotkey),),
        order_by=(("created_at", True),),
        max_rows=limit,
    )
    return [row for row in rows if _is_house_ticket(row, hotkey)]


async def _open_house_queue_rows(
    config: ResearchLabGatewayConfig,
    hotkey: str,
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    """Active (queued/started/paused, fresh) queue rows on house-arm tickets."""

    open_rows: list[dict[str, Any]] = []
    ticket_cache: dict[str, dict[str, Any] | None] = {}
    for status in ACTIVE_HOUSE_QUEUE_STATUSES:
        rows = await select_all(
            "research_loop_run_queue_current",
            columns="run_id,ticket_id,current_queue_status,current_status_at",
            filters=(("current_queue_status", status),),
            order_by=(("current_status_at", True),),
            max_rows=10000,
        )
        for row in rows:
            if not _row_is_fresh(row, config, now=now):
                continue
            ticket_id = str(row.get("ticket_id") or "")
            if ticket_id not in ticket_cache:
                ticket_cache[ticket_id] = await select_one(
                    "research_loop_ticket_current",
                    filters=(("ticket_id", ticket_id),),
                )
            if _is_house_ticket(ticket_cache[ticket_id], hotkey):
                open_rows.append(dict(row))
    return open_rows


def _validate_house_key_ref_shape(key_ref: str) -> str | None:
    if not key_ref:
        return f"{HOUSE_OPENROUTER_KEY_REF_ENV} is not set"
    lowered = key_ref.lower()
    if any(marker in lowered for marker in _RAW_KEY_MARKERS) or "api_key" in lowered:
        return (
            f"{HOUSE_OPENROUTER_KEY_REF_ENV} looks like a raw OpenRouter key; "
            "register it and store only the key ref (never raw keys)"
        )
    return None


async def _validate_house_key_ref(config: ResearchLabGatewayConfig, hotkey: str, key_ref: str) -> str | None:
    """Mirror the miner key-ref validation: encrypted refs must be registered
    (preflight passed) for the house hotkey; ephemeral refs must resolve to a
    non-empty env var. Returns an error string or None."""

    shape_error = _validate_house_key_ref_shape(key_ref)
    if shape_error:
        return shape_error
    if house_key_handling(key_ref) == "encrypted_ref":
        row = await select_one(
            "research_lab_openrouter_key_refs",
            columns="key_ref,miner_hotkey,preflight_status",
            filters=(("key_ref", key_ref), ("miner_hotkey", hotkey)),
        )
        if not row:
            return "house OpenRouter key ref is not registered for the house hotkey"
        if str(row.get("preflight_status") or "") != "passed":
            return "house OpenRouter key ref has not passed preflight"
        return None
    env_name = _ephemeral_key_env_name(config, key_ref)
    if not env_name:
        return "house OpenRouter key ref is not mapped to a key env var"
    if not os.getenv(env_name):
        return f"house OpenRouter key env var is empty: {env_name}"
    return None


def _ephemeral_key_env_name(config: ResearchLabGatewayConfig, key_ref: str) -> str:
    if config.miner_openrouter_key_ref_env_map_json:
        try:
            mapping = json.loads(config.miner_openrouter_key_ref_env_map_json)
        except json.JSONDecodeError:
            return ""
        if isinstance(mapping, Mapping) and mapping.get(str(key_ref)):
            return str(mapping[str(key_ref)])
    return str(config.miner_openrouter_key_env_var or "")


def _house_budget_doc(config: ResearchLabGatewayConfig) -> dict[str, Any]:
    """Budget doc in the exact shape the miner loop-start writes; the max is
    pinned to the requested budget so a house run can never overspend its slot."""

    tier, _model, _tier_doc = config.resolve_auto_research_model(
        str(config.default_auto_research_model_tier or "default")
    )
    requested = float(config.default_compute_budget_usd)
    return {
        "research_model_tier": str(tier),
        "requested_compute_budget_usd": requested,
        "max_compute_budget_usd": requested,
        "budget_policy_version": "research-lab-budget:v1",
    }


async def _create_house_ticket(
    *,
    hotkey: str,
    idempotency_key: str,
    island: str,
    lane: str,
    key_ref: str,
    key_handling: str,
    loop_start_fee_usd: float,
    budget_doc: Mapping[str, Any],
    policy_ref: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create the house ticket with the identical row shape, idempotency, and
    hash discipline as ``store.create_ticket``.

    Mirrored (not called) because ``store.create_ticket`` hard-codes its
    ticket_doc keys and cannot carry the §9.3 arm tags, and the ticket hash is
    computed over the full payload so the tags cannot be added after the fact.
    Only this dict literal is duplicated — payment/queue logic reuses the store
    functions directly. Follow-up: add an ``extra_ticket_doc`` parameter to
    ``store.create_ticket`` and delete this mirror.
    """

    ticket_id = deterministic_uuid("ticket", hotkey, idempotency_key)
    existing_ticket = await select_one("research_loop_tickets", filters=(("ticket_id", ticket_id),))
    if existing_ticket:
        events = await select_many(
            "research_loop_ticket_events",
            filters=(("ticket_id", ticket_id),),
            order_by=(("seq", False),),
            limit=1,
        )
        if events:
            return existing_ticket, events[0]
        event = await create_ticket_event(
            ticket_id=ticket_id,
            event_type="opened",
            actor_hotkey=hotkey,
            reason="house_arm_ticket_created",
            event_doc={"ticket_hash": existing_ticket.get("ticket_hash"), "arm": HOUSE_ARM_TAG},
        )
        return existing_ticket, event

    ticket_payload = {
        "ticket_id": ticket_id,
        "schema_version": "1.0",
        "miner_hotkey": hotkey,
        "island": island,
        "brief_id": None,
        "brief_sanitized_ref": f"house_arm_brief:{lane}:{idempotency_key}",
        "requested_loop_count": 1,
        "ticket_status": "opened",
        "loop_start_fee_required_usd": float(loop_start_fee_usd),
        "loop_start_fee_payment_ref": None,
        "miner_openrouter_key_ref": key_ref,
        "miner_openrouter_key_handling": key_handling,
        "miner_openrouter_preflight_status": "passed" if key_handling == "encrypted_ref" else "not_run",
        "ticket_hash": "",
        "ticket_doc": {
            "idempotency_key_hash": canonical_hash(idempotency_key),
            "source": HOUSE_TICKET_SOURCE,
            "brief_public_summary": house_lane_focus(lane),
            "research_model_tier": str(budget_doc["research_model_tier"]),
            "requested_compute_budget_usd": float(budget_doc["requested_compute_budget_usd"]),
            "max_compute_budget_usd": float(budget_doc["max_compute_budget_usd"]),
            "budget_policy_version": str(budget_doc["budget_policy_version"]),
            # §9.3 arm tags: house runs are ordinary runs distinguished by tag.
            "arm": HOUSE_ARM_TAG,
            "house_policy_ref": policy_ref,
            "house_lane": lane,
            # Env NAME only (never key material) for funding auditability.
            "house_funding_key_ref_env": HOUSE_OPENROUTER_KEY_REF_ENV,
        },
    }
    ticket_payload["ticket_hash"] = canonical_hash(ticket_payload)
    ticket = await insert_row("research_loop_tickets", ticket_payload)
    event = await create_ticket_event(
        ticket_id=ticket_id,
        event_type="opened",
        actor_hotkey=hotkey,
        reason="house_arm_ticket_created",
        event_doc={"ticket_hash": ticket_payload["ticket_hash"], "arm": HOUSE_ARM_TAG, "house_lane": lane},
    )
    return ticket, event


def _bittensor_network_netuid() -> tuple[str, int]:
    """Lazy chain-constant lookup (kept out of module import for CLI/tests)."""

    from gateway.qualification.utils.chain import BITTENSOR_NETUID, BITTENSOR_NETWORK

    return str(BITTENSOR_NETWORK), int(BITTENSOR_NETUID)


async def _ensure_house_payment(
    *,
    ticket_id: str,
    run_id: str,
    hotkey: str,
    loop_start_fee_usd: float,
    budget_doc: Mapping[str, Any],
    policy_ref: str,
    lane: str,
) -> dict[str, Any]:
    """Record the house loop-start funding through the same payments table.

    There is no on-chain extrinsic for house funding, so the payment row uses
    a synthetic, per-run-unique ``house_arm:{run_id}`` block ref (the table
    only requires ``payment_ref = block_hash || ':' || extrinsic_index``).
    ``payment_kind`` stays ``loop_start`` so the worker budget path is
    byte-identical to miner runs; house-ness lives in the ``arm`` tag.
    """

    block_hash = f"house_arm:{run_id}"
    extrinsic_index = 0
    payment_ref = f"{block_hash}:{extrinsic_index}"
    if await payment_ref_exists(block_hash, extrinsic_index):
        existing = await select_one(
            "research_loop_start_payments",
            filters=(("payment_ref", payment_ref),),
        )
        if existing:
            return existing
    network, netuid = _bittensor_network_netuid()
    return await create_loop_start_payment(
        ticket_id=ticket_id,
        payment_ref=payment_ref,
        block_hash=block_hash,
        extrinsic_index=extrinsic_index,
        network=network,
        netuid=netuid,
        miner_hotkey=hotkey,
        payment_info={
            "call_function": "research_lab_house_arm_internal_funding",
            "amount_rao": 0,
            "amount_tao": 0.0,
            "amount_usd": 0.0,
            "tao_price_at_payment": 0.0,
            "sender_coldkey": None,
            "destination": "research_lab_house_arm_internal",
        },
        required_usd=float(loop_start_fee_usd),
        payment_kind="loop_start",
        run_id=run_id,
        compute_budget_usd=float(budget_doc["requested_compute_budget_usd"]),
        extra_verification_doc={
            "research_model_tier": str(budget_doc["research_model_tier"]),
            "max_compute_budget_usd": float(budget_doc["max_compute_budget_usd"]),
            "arm": HOUSE_ARM_TAG,
            "house_policy_ref": policy_ref,
            "house_lane": lane,
            "funding_source": "house_openrouter_key",
        },
    )


async def _ticket_has_event(ticket_id: str, event_type: str) -> bool:
    rows = await select_many(
        "research_loop_ticket_events",
        filters=(("ticket_id", ticket_id), ("event_type", event_type)),
        limit=1,
    )
    return bool(rows)


async def _queue_row_for_run(run_id: str) -> dict[str, Any] | None:
    return await select_one(
        "research_loop_run_queue_current",
        filters=(("run_id", run_id),),
    )


async def _open_one_house_loop(
    config: ResearchLabGatewayConfig,
    *,
    ticket: Mapping[str, Any],
    hotkey: str,
    key_ref: str,
    key_handling: str,
    budget_doc: Mapping[str, Any],
    policy_ref: str,
    lane: str,
) -> dict[str, Any]:
    """Fund + queue one house loop on an existing house ticket, idempotently.

    Emulates the miner ``/loop-start`` sequence with the same store functions:
    payment row -> "funded" ticket event -> "queued" queue event (carrying the
    capacity doc so the scripts/43/54 trigger admits atomically) -> "queued"
    ticket event -> public card projection. Every step skips work that already
    exists so a crashed pass can be re-run safely.
    """

    ticket_id = str(ticket["ticket_id"])
    run_id = _house_run_id(ticket_id)
    fee_usd = float(ticket.get("loop_start_fee_required_usd") or config.loop_start_fee_usd)

    payment = await _ensure_house_payment(
        ticket_id=ticket_id,
        run_id=run_id,
        hotkey=hotkey,
        loop_start_fee_usd=fee_usd,
        budget_doc=budget_doc,
        policy_ref=policy_ref,
        lane=lane,
    )
    arm_doc = {"arm": HOUSE_ARM_TAG, "house_policy_ref": policy_ref, "house_lane": lane}
    if not await _ticket_has_event(ticket_id, "funded"):
        await create_ticket_event(
            ticket_id=ticket_id,
            event_type="funded",
            actor_hotkey=hotkey,
            reason="house_arm_loop_funded",
            event_doc={
                "payment_id": payment["payment_id"],
                "payment_ref": payment["payment_ref"],
                "miner_openrouter_key_ref": key_ref,
                "miner_openrouter_key_handling": key_handling,
                **arm_doc,
                **budget_doc,
            },
        )

    existing_queue_row = await _queue_row_for_run(run_id)
    if existing_queue_row:
        return {
            "ticket_id": ticket_id,
            "run_id": run_id,
            "lane": lane,
            "status": "already_queued",
            "queue_status": existing_queue_row.get("current_queue_status"),
        }

    # The scripts/43 + scripts/54 DB trigger enforces global capacity and the
    # one-active-loop-per-hotkey rule atomically on this insert.
    await create_queue_event(
        run_id=run_id,
        ticket_id=ticket_id,
        event_type="queued",
        queue_priority=0,
        reason="house_arm_loop_queued",
        event_doc={
            "payment_id": payment["payment_id"],
            "payment_ref": payment["payment_ref"],
            "payment_kind": "loop_start",
            "miner_openrouter_key_ref": key_ref,
            "miner_openrouter_key_handling": key_handling,
            "requested_loop_count": 1,
            **arm_doc,
            **autoresearch_queue_capacity_doc(config),
            **budget_doc,
        },
    )
    await create_ticket_event(
        ticket_id=ticket_id,
        event_type="queued",
        actor_hotkey=hotkey,
        reason="house_arm_loop_queued",
        event_doc={
            "payment_id": payment["payment_id"],
            "payment_ref": payment["payment_ref"],
            "run_id": run_id,
            **arm_doc,
            **budget_doc,
        },
    )
    await safe_project_public_loop_activity(
        ticket_id,
        source_ref=f"loop_start_queue:{run_id}",
        reason="paid_loop_queued",
        config=config,
    )
    return {"ticket_id": ticket_id, "run_id": run_id, "lane": lane, "status": "queued"}


def _is_capacity_conflict(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "research_lab_queue_capacity_conflict" in message
        or "research_lab_queue_hotkey_conflict" in message
        or "23505" in message
    )


async def open_house_loops(
    *,
    budget_usd_today: float,
    max_open_loops: int,
    dry_run: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Scheduler entry point: open house-funded loops within the policy clamp.

    Never spends unless BOTH the ``RESEARCH_LAB_HOUSE_ARM_ENABLED`` master
    flag is truthy AND ``dry_run=False``. The effective daily budget is
    ``min(budget_usd_today, policy max $500) - today's recorded house spend``,
    and the scripts/43/54/67 per-hotkey guard caps concurrent house loops
    according to the shared gateway policy regardless of ``max_open_loops``.
    """

    now = now or datetime.now(timezone.utc)
    config = ResearchLabGatewayConfig.from_env()
    enabled = house_arm_enabled()
    hotkey = house_hotkey()
    key_ref = house_openrouter_key_ref()
    result: dict[str, Any] = {
        "ok": False,
        "action": "open-house-loops",
        "dry_run": bool(dry_run),
        "house_arm_enabled": enabled,
        "opened": [],
        "planned": [],
        "skipped": [],
    }

    if budget_usd_today <= 0:
        result["error"] = "budget_usd_today must be positive"
        return result
    if max_open_loops < 1:
        result["error"] = "max_open_loops must be at least 1"
        return result
    if not dry_run and not enabled:
        result["error"] = (
            f"{HOUSE_ARM_ENABLED_ENV} is not enabled; refusing to spend "
            "(dry-run remains available)"
        )
        return result
    if not hotkey:
        result["error"] = f"{HOUSE_HOTKEY_ENV} is not set"
        return result
    result["house_hotkey_prefix"] = hotkey[:16]

    try:
        policy = build_house_arm_operating_policy(config)
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    result["house_policy_ref"] = policy.policy_id

    key_error = await _validate_house_key_ref(config, hotkey, key_ref)
    if key_error:
        result["error"] = key_error
        return result
    key_handling = house_key_handling(key_ref)

    maintenance = await get_autoresearch_maintenance_state()
    if maintenance.get("paused"):
        result["error"] = "research lab auto-research is paused for maintenance"
        result["maintenance"] = {key: maintenance.get(key) for key in ("paused", "reason", "status_at")}
        return result

    # Hard clamp at the policy max regardless of arguments.
    policy_max_usd = policy.daily_budget_max_cents / 100.0
    policy_min_usd = policy.daily_budget_min_cents / 100.0
    effective_budget_usd = min(float(budget_usd_today), policy_max_usd)
    todays_spend_usd = await _todays_house_spend_usd(hotkey, now=now)
    remaining_usd = round(max(0.0, effective_budget_usd - todays_spend_usd), 6)
    per_loop_usd = house_per_loop_usd(config)
    result["budget"] = {
        "requested_usd": float(budget_usd_today),
        "policy_daily_min_usd": policy_min_usd,
        "policy_daily_max_usd": policy_max_usd,
        "effective_budget_usd": effective_budget_usd,
        "below_policy_floor": float(budget_usd_today) < policy_min_usd,
        "todays_spend_usd": todays_spend_usd,
        "remaining_usd": remaining_usd,
        "per_loop_usd": per_loop_usd,
        "loop_start_fee_usd": float(config.loop_start_fee_usd),
        "compute_budget_usd": float(config.default_compute_budget_usd),
    }

    open_rows = await _open_house_queue_rows(config, hotkey, now=now)
    result["open_house_loops"] = len(open_rows)
    result["open_house_run_ids"] = [str(row.get("run_id") or "") for row in open_rows]

    capacity_doc = autoresearch_queue_capacity_doc(config)
    global_capacity = int(capacity_doc.get("autoresearch_capacity") or 0)
    if global_capacity <= 0:
        result["skipped"].append("autoresearch capacity is closed (no proxies/workers configured)")
        result["ok"] = True
        return result

    budget_doc = _house_budget_doc(config)
    house_tickets = await _house_tickets(hotkey)
    day_key = now.astimezone(timezone.utc).strftime("%Y%m%d")

    # Crash recovery first: a house ticket funded earlier but never queued
    # (payment row without a queue row) is completed before any new opening,
    # so already-counted budget is not stranded.
    orphaned: list[dict[str, Any]] = []
    for ticket in house_tickets:
        run_id = _house_run_id(str(ticket["ticket_id"]))
        if await _queue_row_for_run(run_id):
            continue
        if not await payment_ref_exists(f"house_arm:{run_id}", 0):
            continue
        orphaned.append(dict(ticket))

    lanes = planner_lanes()
    last_lane = None
    if house_tickets:
        last_lane = str(_ticket_doc(house_tickets[0]).get("house_lane") or "") or None

    # House uses the same per-hotkey active-loop cap as miner submissions.
    # Repaired orphans re-enter the queue, so they consume slots first.
    hotkey_cap = max(1, int(getattr(config, "max_active_autoresearch_loops_per_hotkey", 2) or 1))
    hotkey_slots = max(0, hotkey_cap - len(open_rows) - len(orphaned))
    budget_slots = int(remaining_usd // per_loop_usd) if per_loop_usd > 0 else 0
    request_slots = max(0, int(max_open_loops) - len(open_rows) - len(orphaned))
    to_open = max(0, min(hotkey_slots, budget_slots, request_slots))
    if hotkey_slots == 0:
        result["skipped"].append(
            f"house hotkey already reached active loop cap ({hotkey_cap}; scripts/43/54/67)"
            if open_rows
            else "orphaned house opening consumes the remaining hotkey slot this pass"
        )
    if budget_slots == 0 and hotkey_slots > 0:
        result["skipped"].append("remaining daily budget is below the per-loop cost")
    if request_slots == 0 and hotkey_slots > 0:
        result["skipped"].append("max_open_loops already reached")
    result["to_open"] = to_open

    if dry_run:
        for ticket in orphaned:
            result["planned"].append(
                {
                    "action": "repair_orphaned_opening",
                    "ticket_id": str(ticket["ticket_id"]),
                    "run_id": _house_run_id(str(ticket["ticket_id"])),
                    "lane": str(_ticket_doc(ticket).get("house_lane") or ""),
                }
            )
        lane = last_lane
        for index in range(to_open):
            lane = next_house_lane(lane, lanes)
            result["planned"].append(
                {
                    "action": "open_house_loop",
                    "lane": lane,
                    "idempotency_key": f"house-arm:{day_key}:{len(house_tickets) + index:03d}:{lane}",
                    "per_loop_usd": per_loop_usd,
                    "budget_doc": dict(budget_doc),
                }
            )
        result["ok"] = True
        return result

    for ticket in orphaned:
        lane = str(_ticket_doc(ticket).get("house_lane") or "") or lanes[0]
        try:
            outcome = await _open_one_house_loop(
                config,
                ticket=ticket,
                hotkey=hotkey,
                key_ref=key_ref,
                key_handling=key_handling,
                budget_doc=budget_doc,
                policy_ref=policy.policy_id,
                lane=lane,
            )
            outcome["action"] = "repaired_orphaned_opening"
            result["opened"].append(outcome)
        except Exception as exc:  # noqa: BLE001 - report per-loop, keep the pass alive
            stage = "capacity_conflict" if _is_capacity_conflict(exc) else "open_failed"
            result["skipped"].append(
                {"ticket_id": str(ticket["ticket_id"]), "stage": stage, "error": str(exc)[:240]}
            )

    lane = last_lane
    for index in range(to_open):
        lane = next_house_lane(lane, lanes)
        idempotency_key = f"house-arm:{day_key}:{len(house_tickets) + index:03d}:{lane}"
        try:
            ticket, _event = await _create_house_ticket(
                hotkey=hotkey,
                idempotency_key=idempotency_key,
                island=policy.island,
                lane=lane,
                key_ref=key_ref,
                key_handling=key_handling,
                loop_start_fee_usd=float(config.loop_start_fee_usd),
                budget_doc=budget_doc,
                policy_ref=policy.policy_id,
            )
            outcome = await _open_one_house_loop(
                config,
                ticket=ticket,
                hotkey=hotkey,
                key_ref=key_ref,
                key_handling=key_handling,
                budget_doc=budget_doc,
                policy_ref=policy.policy_id,
                lane=lane,
            )
            outcome["action"] = "opened_house_loop"
            result["opened"].append(outcome)
        except Exception as exc:  # noqa: BLE001 - report per-loop, keep the pass alive
            stage = "capacity_conflict" if _is_capacity_conflict(exc) else "open_failed"
            result["skipped"].append(
                {"idempotency_key": idempotency_key, "lane": lane, "stage": stage, "error": str(exc)[:240]}
            )
            logger.warning(
                "research_lab_house_arm_open_failed lane=%s stage=%s error=%s",
                lane,
                stage,
                str(exc)[:240],
            )

    result["ok"] = not any(
        isinstance(item, Mapping) and item.get("stage") == "open_failed" for item in result["skipped"]
    )
    return result


async def house_arm_status(*, now: datetime | None = None) -> dict[str, Any]:
    """Read-only status: open house loops, today's spend, remaining clamp."""

    now = now or datetime.now(timezone.utc)
    config = ResearchLabGatewayConfig.from_env()
    hotkey = house_hotkey()
    result: dict[str, Any] = {
        "ok": True,
        "action": "house-arm-status",
        "house_arm_enabled": house_arm_enabled(),
        "house_hotkey_configured": bool(hotkey),
        "house_openrouter_key_ref_configured": bool(house_openrouter_key_ref()),
    }
    if not hotkey:
        result["error"] = f"{HOUSE_HOTKEY_ENV} is not set"
        result["ok"] = False
        return result
    result["house_hotkey_prefix"] = hotkey[:16]
    try:
        policy = build_house_arm_operating_policy(config)
    except ValueError as exc:
        result["error"] = str(exc)
        result["ok"] = False
        return result
    todays_spend_usd = await _todays_house_spend_usd(hotkey, now=now)
    open_rows = await _open_house_queue_rows(config, hotkey, now=now)
    result.update(
        {
            "house_policy_ref": policy.policy_id,
            "policy_daily_min_usd": policy.daily_budget_min_cents / 100.0,
            "policy_daily_max_usd": policy.daily_budget_max_cents / 100.0,
            "todays_spend_usd": todays_spend_usd,
            "remaining_clamp_usd": round(
                max(0.0, policy.daily_budget_max_cents / 100.0 - todays_spend_usd), 6
            ),
            "per_loop_usd": house_per_loop_usd(config),
            "open_house_loops": len(open_rows),
            "open_house_runs": [
                {
                    "run_id": row.get("run_id"),
                    "ticket_id": row.get("ticket_id"),
                    "queue_status": row.get("current_queue_status"),
                    "status_at": row.get("current_status_at"),
                }
                for row in open_rows
            ],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Matched-budget comparison (admin `house-arm-comparison`, read-only)
# ---------------------------------------------------------------------------


def _arm_metrics_template() -> dict[str, Any]:
    return {
        "candidates_total": 0,
        "candidates_scored": 0,
        "keeps": 0,
        "deltas": [],
        "verified_points": 0.0,
        "spend_usd": 0.0,
        "run_ids": set(),
        "payment_refs": [],
    }


async def _score_bundle_mean_delta(score_bundle_id: str) -> float | None:
    if not score_bundle_id:
        return None
    row = await select_one(
        "research_evaluation_score_bundle_current",
        filters=(("score_bundle_id", score_bundle_id),),
    )
    if not row:
        return None
    doc = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
    aggregates = doc.get("aggregates") if isinstance(doc.get("aggregates"), Mapping) else {}
    try:
        return float(aggregates.get("mean_delta"))
    except (TypeError, ValueError):
        return None


async def collect_house_arm_comparison_metrics(
    *,
    start_iso: str,
    end_exclusive_iso: str,
    hotkey: str,
    improvement_threshold_points: float,
) -> dict[str, dict[str, Any]]:
    """Read-only per-arm yield metrics over [start, end): candidates scored,
    deltas, keeps, spend. Arm split: house = the house hotkey, miner = rest."""

    arms = {"house": _arm_metrics_template(), "miner": _arm_metrics_template()}

    candidates = await select_all(
        "research_lab_candidate_evaluation_current",
        columns=(
            "candidate_id,run_id,ticket_id,miner_hotkey,created_at,"
            "current_candidate_status,current_reason,current_score_bundle_id"
        ),
        filters=(
            ("created_at", "gte", start_iso),
            ("created_at", "lt", end_exclusive_iso),
        ),
        order_by=(("created_at", True),),
        max_rows=5000,
    )
    for candidate in candidates:
        arm = arms["house" if str(candidate.get("miner_hotkey") or "").strip() == hotkey else "miner"]
        arm["candidates_total"] += 1
        if candidate.get("run_id"):
            arm["run_ids"].add(str(candidate["run_id"]))
        if str(candidate.get("current_candidate_status") or "") != "scored":
            continue
        arm["candidates_scored"] += 1
        mean_delta = await _score_bundle_mean_delta(str(candidate.get("current_score_bundle_id") or ""))
        if mean_delta is None:
            continue
        arm["deltas"].append(round(mean_delta, 6))
        arm["verified_points"] += max(0.0, mean_delta)
        if mean_delta >= improvement_threshold_points:
            arm["keeps"] += 1

    payments = await select_all(
        "research_loop_start_payments",
        columns="payment_id,payment_ref,miner_hotkey,required_usd,verification_doc,payment_status,created_at",
        filters=(
            ("created_at", "gte", start_iso),
            ("created_at", "lt", end_exclusive_iso),
        ),
        order_by=(("created_at", True),),
        max_rows=10000,
    )
    for payment in payments:
        if str(payment.get("payment_status") or "") != "verified":
            continue
        arm = arms["house" if str(payment.get("miner_hotkey") or "").strip() == hotkey else "miner"]
        arm["spend_usd"] += _payment_spend_usd(payment)
        arm["payment_refs"].append(str(payment.get("payment_id") or ""))
        doc = payment.get("verification_doc") if isinstance(payment.get("verification_doc"), Mapping) else {}
        if doc.get("run_id"):
            arm["run_ids"].add(str(doc["run_id"]))

    for arm in arms.values():
        arm["spend_usd"] = round(float(arm["spend_usd"]), 6)
        arm["verified_points"] = round(float(arm["verified_points"]), 6)
        arm["run_count"] = len(arm.pop("run_ids"))
        arm["mean_delta"] = round(sum(arm["deltas"]) / len(arm["deltas"]), 6) if arm["deltas"] else None
    return arms


def _matched_yield_record(
    *,
    arm_kind: str,
    quarter_ref: str,
    matched_budget_cents: int,
    own_budget_cents: int,
    verified_points: float,
    keeps: int,
    run_count: int,
    payment_refs: Sequence[str],
    window_ref: str,
) -> CounterfactualYieldRecord:
    """Build one arm's yield record scaled to the matched budget.

    ``build_matched_budget_comparison`` requires equal budgets, so each arm's
    verified points are scaled linearly to the matched budget; points-per-
    dollar (the reported yield) is invariant under this scaling.
    """

    scale = matched_budget_cents / own_budget_cents
    record = CounterfactualYieldRecord.from_mapping(
        {
            "yield_id": "counterfactual_yield:pending",
            "arm_kind": arm_kind,
            "quarter_ref": quarter_ref,
            "budget_cents": matched_budget_cents,
            "verified_points": round(verified_points * scale, 6),
            "novelty_weighted_points": round(float(keeps) * scale, 6),
            "run_count": max(1, run_count),
            "receipt_refs": tuple(
                f"receipt_v2:loop_start_payment:{ref}" for ref in list(payment_refs)[:20]
            )
            or (f"receipt_v2:loop_start_payment:none:{window_ref}",),
            "ledger_audit_refs": (f"cost_ledger_audit:research_loop_start_payments:{window_ref}",),
            "anchor_proposal_ref": f"anchor_proposal:house_arm_comparison:{window_ref}",
            "allocator_selection_refs": (HOUSE_ALLOCATOR_SELECTION_REF,)
            if arm_kind == "allocator_directed"
            else (),
            "brief_refs": (f"brief_sanitized:aggregate:{window_ref}",)
            if arm_kind == "miner_briefed"
            else (),
            "source_data_state": "measured_lab_only",
            "uses_local_fixtures": False,
            "measured_data_ready": True,
            "production_data_claimed": False,
            "local_only": True,
        }
    )
    from research_lab.canonical import sha256_json

    data = record.to_dict()
    data["yield_id"] = "counterfactual_yield:" + sha256_json(record.identity_payload()).split(":", 1)[1][:16]
    return CounterfactualYieldRecord.from_mapping(data)


async def build_house_arm_comparison_report(
    *,
    start_date: str,
    end_date: str,
    config: ResearchLabGatewayConfig | None = None,
) -> dict[str, Any]:
    """Read-only §9.3 honesty report: miner vs house yield per dollar over a
    date range, via ``counterfactual_gate.build_matched_budget_comparison``."""

    config = config or ResearchLabGatewayConfig.from_env()
    hotkey = house_hotkey()
    result: dict[str, Any] = {
        "ok": False,
        "action": "house-arm-comparison",
        "read_only": True,
        "start_date": start_date,
        "end_date": end_date,
    }
    if not hotkey:
        result["error"] = f"{HOUSE_HOTKEY_ENV} is not set"
        return result

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_exclusive = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    except ValueError as exc:
        result["error"] = f"invalid date (expected YYYY-MM-DD): {exc}"
        return result
    if end_exclusive <= start:
        result["error"] = "end_date must not be before start_date"
        return result

    arms = await collect_house_arm_comparison_metrics(
        start_iso=start.isoformat(),
        end_exclusive_iso=end_exclusive.isoformat(),
        hotkey=hotkey,
        improvement_threshold_points=float(config.improvement_threshold_points),
    )
    result["arms"] = {
        name: {key: value for key, value in metrics.items() if key != "payment_refs"}
        for name, metrics in arms.items()
    }

    house_cents = int(round(arms["house"]["spend_usd"] * 100))
    miner_cents = int(round(arms["miner"]["spend_usd"] * 100))
    if house_cents <= 0 or miner_cents <= 0:
        result["ok"] = True
        result["status"] = "insufficient_data"
        result["detail"] = "both arms need verified loop-start spend in the window to compare"
        return result

    matched_cents = min(house_cents, miner_cents)
    window_ref = f"{start_date}:{end_date}"
    quarter = (start.month - 1) // 3 + 1
    quarter_ref = f"quarter:{start.year}-Q{quarter}:house_arm_window:{window_ref}"
    try:
        miner_yield = _matched_yield_record(
            arm_kind="miner_briefed",
            quarter_ref=quarter_ref,
            matched_budget_cents=matched_cents,
            own_budget_cents=miner_cents,
            verified_points=float(arms["miner"]["verified_points"]),
            keeps=int(arms["miner"]["keeps"]),
            run_count=int(arms["miner"]["run_count"]),
            payment_refs=arms["miner"]["payment_refs"],
            window_ref=window_ref,
        )
        house_yield = _matched_yield_record(
            arm_kind="allocator_directed",
            quarter_ref=quarter_ref,
            matched_budget_cents=matched_cents,
            own_budget_cents=house_cents,
            verified_points=float(arms["house"]["verified_points"]),
            keeps=int(arms["house"]["keeps"]),
            run_count=int(arms["house"]["run_count"]),
            payment_refs=arms["house"]["payment_refs"],
            window_ref=window_ref,
        )
        comparison = build_matched_budget_comparison(
            miner_yield=miner_yield,
            allocator_yield=house_yield,
            methodology_ref=HOUSE_COMPARISON_METHODOLOGY_REF,
            shared_prior_caveat=HOUSE_SHARED_PRIOR_CAVEAT,
        )
    except ValueError as exc:
        result["error"] = f"comparison build failed: {exc}"
        return result

    result["ok"] = True
    result["status"] = "compared"
    result["matched_budget_usd"] = matched_cents / 100.0
    result["comparison"] = comparison.to_dict()
    result["summary"] = {
        "miner_yield_points_per_1000_usd": comparison.miner_yield_points_per_1000_usd,
        "house_yield_points_per_1000_usd": comparison.allocator_yield_points_per_1000_usd,
        "delta_points_per_1000_usd": comparison.delta_yield_points_per_1000_usd,
        "miners_add_signal": comparison.passed_gate,
    }
    return result


# ---------------------------------------------------------------------------
# CLI: python3 -m gateway.research_lab.house_arm --open --budget-usd 200 --dry-run
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Leadpoet Research Lab house-funded baseline arm (§9.3). "
            "Dry-run by default; spending requires BOTH "
            f"{HOUSE_ARM_ENABLED_ENV}=true and --no-dry-run."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--open", action="store_true", help="Open house loops within the daily clamp")
    mode.add_argument("--status", action="store_true", help="Open house loops, today's spend, remaining clamp")
    parser.add_argument("--budget-usd", type=float, default=200.0, help="Today's house budget (clamped at policy max)")
    parser.add_argument(
        "--max-open-loops",
        type=int,
        default=1,
        help="Max open house loops; the shared per-hotkey guard still applies",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Plan without writing (default)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help=f"Actually open loops (also requires {HOUSE_ARM_ENABLED_ENV}=true)",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.status:
        return await house_arm_status()
    return await open_house_loops(
        budget_usd_today=args.budget_usd,
        max_open_loops=args.max_open_loops,
        dry_run=args.dry_run,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
