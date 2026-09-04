"""Measured PostgREST reads for authoritative V2 weight inputs.

The coordinator owns every query shape. Callers may supply only the small,
typed values named by a policy; they cannot select another table, project,
column set, ordering, page size, retry policy, or timeout.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from uuid import UUID

from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_SUPABASE_ORIGIN,
    configured_supabase_origin_v2,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


SUPABASE_WEIGHT_SOURCE_ORIGIN = PRODUCTION_SUPABASE_ORIGIN
SUPABASE_SOURCE_SCHEMA_VERSION = "leadpoet.supabase_source.v2"
SUPABASE_READ_TIMEOUT_MS = 45_000
SUPABASE_PAGE_SIZE = 1_000
SUPABASE_RETRY_BACKOFF_SECONDS = (1.0, 3.0)
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)


class SupabaseSourceV2Error(RuntimeError):
    """A measured database read did not end in an authenticated valid page."""


@dataclass(frozen=True)
class SupabaseQueryV2:
    policy_id: str
    table: str
    select: str
    parameter_names: Tuple[str, ...]
    max_pages: int
    page_size: int = SUPABASE_PAGE_SIZE
    order: str = ""
    limit: int = 0


QUERY_POLICIES = {
    "qualification_epoch_assignment": SupabaseQueryV2(
        policy_id="qualification_epoch_assignment",
        table="transparency_log",
        select="payload",
        parameter_names=("epoch_id",),
        max_pages=1,
        order="ts.desc",
        limit=1,
    ),
    "qualification_leads_by_ids": SupabaseQueryV2(
        policy_id="qualification_leads_by_ids",
        table="leads_private",
        select=(
            "lead_id,lead_blob,lead_blob_hash,miner_hotkey,"
            "first_name,last_name,email,role,company_name,linkedin,website,"
            "company_linkedin,industry,sub_industry,city,state,country,"
            "hq_city,hq_state,hq_country,employee_count,description"
        ),
        parameter_names=("lead_ids",),
        max_pages=1,
    ),
    "banned_hotkeys": SupabaseQueryV2(
        policy_id="banned_hotkeys",
        table="banned_hotkeys",
        select="hotkey",
        parameter_names=(),
        max_pages=20,
        order="hotkey.asc",
    ),
    "fulfillment_active_rewards": SupabaseQueryV2(
        policy_id="fulfillment_active_rewards",
        table="fulfillment_score_consensus",
        select="consensus_id,miner_hotkey,reward_pct,reward_expires_epoch",
        parameter_names=("epoch_id",),
        max_pages=50,
        order="consensus_id.asc",
    ),
    "fulfillment_leaderboard_winners": SupabaseQueryV2(
        policy_id="fulfillment_leaderboard_winners",
        table="fulfillment_score_consensus",
        select="miner_hotkey,reward_pct,computed_at",
        parameter_names=("window_start", "window_end"),
        max_pages=50,
    ),
    # The governing Lab Arena reward basis for one weight epoch (labarena.md
    # 13.4): the newest published round whose effective epoch is at most the
    # weight epoch, read from the signed-columns view with the service role.
    "lab_arena_reward_basis": SupabaseQueryV2(
        policy_id="lab_arena_reward_basis",
        table="lab_arena_reward_basis_v1",
        select="round_id,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc",
        parameter_names=("epoch_id",),
        max_pages=1,
        order="effective_reward_epoch.desc",
        limit=1,
    ),
    "research_lab_allocation_current": SupabaseQueryV2(
        policy_id="research_lab_allocation_current",
        table="research_lab_emission_allocation_current",
        select="allocation_id,epoch,netuid,snapshot_status,allocation_hash,allocation_doc,created_at",
        parameter_names=("epoch_id", "netuid"),
        max_pages=1,
        order="created_at.desc",
        limit=1,
    ),
    "latest_native_compute_allocation_authority": SupabaseQueryV2(
        policy_id="latest_native_compute_allocation_authority",
        table="research_lab_finalized_allocation_epochs_v2",
        select="epoch_id,netuid",
        parameter_names=("epoch_id", "netuid"),
        max_pages=1,
        order="epoch_id.desc",
        limit=1,
    ),
    "latest_legacy_compute_allocation_authority": SupabaseQueryV2(
        policy_id="latest_legacy_compute_allocation_authority",
        table="research_lab_legacy_finalized_allocation_migrations_v2",
        select="netuid,epoch_id,allocation_hash,allocation_doc",
        parameter_names=("epoch_id", "netuid"),
        max_pages=1,
        order="epoch_id.desc",
        limit=1,
    ),
    "legacy_allocation_by_hash": SupabaseQueryV2(
        policy_id="legacy_allocation_by_hash",
        table="research_lab_emission_allocation_snapshots",
        select=(
            "allocation_id,epoch,netuid,snapshot_status,allocation_hash,"
            "allocation_doc,created_at"
        ),
        parameter_names=("allocation_hash", "epoch_id", "netuid"),
        max_pages=1,
        order="created_at.asc",
        limit=2,
    ),
    "score_bundle_by_id": SupabaseQueryV2(
        policy_id="score_bundle_by_id",
        table="research_evaluation_score_bundle_current",
        select=(
            "score_bundle_id,score_bundle_hash,score_bundle_doc,current_event_status,"
            "current_status_at"
        ),
        parameter_names=("score_bundle_id",),
        max_pages=1,
        limit=2,
    ),
    "allocation_reimbursement_schedules": SupabaseQueryV2(
        policy_id="allocation_reimbursement_schedules",
        table="research_reimbursement_schedules",
        select=(
            "schedule_id,award_id,schedule_status,start_epoch,epoch_count,"
            "total_microusd,entries"
        ),
        parameter_names=("epoch_id", "start_epoch"),
        max_pages=50,
        order="start_epoch.asc,schedule_id.asc",
    ),
    "allocation_reimbursement_awards": SupabaseQueryV2(
        policy_id="allocation_reimbursement_awards",
        table="research_reimbursement_award_current",
        select=(
            "award_id,run_id,miner_hotkey,island,current_award_status,award_status,"
            "run_day,target_reimbursement_microusd,participation_score,"
            "participation_fraction,rebate_rate,eligible_cost_microusd,"
            "reimbursement_epochs,loop_start_fee_included,input_hash"
        ),
        parameter_names=("award_ids",),
        max_pages=2,
        order="award_id.asc",
    ),
    "allocation_champion_rewards": SupabaseQueryV2(
        policy_id="allocation_champion_rewards",
        table="research_lab_champion_reward_current",
        select=(
            "champion_reward_id,score_bundle_id,candidate_id,run_id,miner_hotkey,"
            "miner_uid,island,evaluation_epoch,current_reward_status,start_epoch,"
            "epoch_count,improvement_points,threshold_points,"
            "desired_alpha_percent,input_hash,anchored_hash"
        ),
        parameter_names=("epoch_id", "include_paid"),
        max_pages=50,
        order="champion_reward_id.asc",
    ),
    "champion_reward_by_id": SupabaseQueryV2(
        policy_id="champion_reward_by_id",
        table="research_lab_champion_reward_current",
        select=(
            "champion_reward_id,score_bundle_id,candidate_id,run_id,miner_hotkey,"
            "miner_uid,island,policy_id,evaluation_epoch,start_epoch,epoch_count,"
            "improvement_points,threshold_points,desired_alpha_percent,"
            "source_score_bundle_hash,input_hash,anchored_hash,current_reward_status"
        ),
        parameter_names=("champion_reward_id",),
        max_pages=1,
        limit=2,
    ),
    "allocation_source_add_rewards": SupabaseQueryV2(
        policy_id="allocation_source_add_rewards",
        table="research_lab_source_add_reward_current",
        select=(
            "reward_ref,adapter_id,miner_hotkey,leg,reward_kind,alpha_percent,"
            "reward_epochs,start_epoch,current_reward_status,trigger_evidence_doc,"
            "public_label,desired_alpha_percent,epoch_count,created_at"
        ),
        parameter_names=("epoch_id",),
        max_pages=50,
        order="created_at.asc,reward_ref.asc",
    ),
    "source_add_reward_by_ref": SupabaseQueryV2(
        policy_id="source_add_reward_by_ref",
        table="research_lab_source_add_reward_current",
        select=(
            "reward_ref,adapter_id,miner_hotkey,leg,reward_kind,alpha_percent,"
            "reward_epochs,start_epoch,current_reward_status,trigger_evidence_doc,"
            "public_label,desired_alpha_percent,epoch_count"
        ),
        parameter_names=("reward_ref",),
        max_pages=1,
        limit=2,
    ),
    "source_add_rewards_by_adapter": SupabaseQueryV2(
        policy_id="source_add_rewards_by_adapter",
        table="research_lab_source_add_reward_current",
        select="reward_ref,adapter_id,leg,current_reward_status",
        parameter_names=("adapter_id",),
        max_pages=1,
        order="reward_ref.asc",
    ),
    "source_add_submission_by_id": SupabaseQueryV2(
        policy_id="source_add_submission_by_id",
        table="research_lab_source_add_submission_current",
        select=(
            "submission_id,adapter_id,miner_hotkey,stage,precheck_status,"
            "precheck_doc,submission_doc,source_identity_hash,"
            "source_identity_version"
        ),
        parameter_names=("submission_id",),
        max_pages=1,
        limit=2,
    ),
    "source_add_probe_config_by_submission": SupabaseQueryV2(
        policy_id="source_add_probe_config_by_submission",
        table="research_lab_source_add_probe_config_current",
        select=(
            "config_ref,submission_id,adapter_id,config_status,probe_doc,"
            "credential_envelope,created_at"
        ),
        parameter_names=("submission_id",),
        max_pages=1,
        limit=2,
    ),
    "source_add_functional_probe_by_submission": SupabaseQueryV2(
        policy_id="source_add_functional_probe_by_submission",
        table="research_lab_source_add_functional_probe_current",
        select=(
            "attempt_ref,submission_id,adapter_id,evaluation_mode,config_ref,"
            "result_status,route_hash,response_hash,status_class,content_type,"
            "byte_count,duration_ms,reason_codes,receipt_hash,"
            "business_artifact_hash,result_doc,created_at"
        ),
        parameter_names=("submission_id",),
        max_pages=1,
        limit=2,
    ),
    "source_add_provisioning_smoke_by_submission": SupabaseQueryV2(
        policy_id="source_add_provisioning_smoke_by_submission",
        table="research_lab_source_add_provisioning_smoke_current",
        select=(
            "attempt_ref,submission_id,adapter_id,evaluation_mode,config_ref,"
            "result_status,receipt_hash,business_artifact_hash,result_doc,created_at"
        ),
        parameter_names=("submission_id",),
        max_pages=1,
        limit=2,
    ),
    "source_add_leg1_events_since": SupabaseQueryV2(
        policy_id="source_add_leg1_events_since",
        table="research_lab_source_add_reward_events",
        select="reward_ref,created_at,reason",
        parameter_names=("day_start",),
        max_pages=2,
        order="created_at.asc,reward_ref.asc",
    ),
    "source_add_provisioning_by_adapter": SupabaseQueryV2(
        policy_id="source_add_provisioning_by_adapter",
        table="research_lab_source_add_provisioning_current",
        select=(
            "provision_ref,catalog_id,submission_id,adapter_id,miner_hotkey,"
            "registry_provider_id,provision_status"
        ),
        parameter_names=("adapter_id",),
        max_pages=1,
        order="adapter_id.asc",
        limit=2,
    ),
    "source_add_provisioning_eligible": SupabaseQueryV2(
        policy_id="source_add_provisioning_eligible",
        table="research_lab_source_add_provisioning_current",
        select=(
            "provision_ref,catalog_id,submission_id,adapter_id,miner_hotkey,"
            "source_identity_hash,registry_provider_id,provision_status,"
            "provision_doc,credential_envelope,source_name,declared_base_domains,"
            "catalog_doc,accepted_at"
        ),
        parameter_names=(),
        max_pages=1,
        limit=200,
    ),
    "provider_registry_recent": SupabaseQueryV2(
        policy_id="provider_registry_recent",
        table="research_lab_provider_registry",
        select="registry_hash,provider_count,registry_doc,created_at",
        parameter_names=(),
        max_pages=1,
        order="created_at.desc",
        limit=20,
    ),
    "reimbursement_ticket_by_id": SupabaseQueryV2(
        policy_id="reimbursement_ticket_by_id",
        table="research_loop_ticket_current",
        select=(
            "ticket_id,miner_hotkey,island,brief_sanitized_ref,"
            "miner_openrouter_key_ref,ticket_doc,created_at,current_status_at"
        ),
        parameter_names=("ticket_id",),
        max_pages=1,
        limit=1,
    ),
    "reimbursement_receipt_by_id": SupabaseQueryV2(
        policy_id="reimbursement_receipt_by_id",
        table="research_loop_receipt_current",
        select=(
            "receipt_id,run_id,ticket_id,loop_start_payment_id,"
            "loop_start_credit_id,current_receipt_status"
        ),
        parameter_names=("receipt_id",),
        max_pages=1,
        limit=1,
    ),
    "reimbursement_payment_by_id": SupabaseQueryV2(
        policy_id="reimbursement_payment_by_id",
        table="research_loop_start_payments",
        select="payment_id,ticket_id,payment_status,verification_doc,verified_at",
        parameter_names=("payment_id",),
        max_pages=1,
        limit=1,
    ),
    "reimbursement_queue_events_by_run": SupabaseQueryV2(
        policy_id="reimbursement_queue_events_by_run",
        table="research_loop_run_queue_events",
        select="run_id,ticket_id,seq,event_type,event_doc,created_at",
        parameter_names=("run_id",),
        max_pages=1,
        order="seq.desc,created_at.desc",
        limit=200,
    ),
    "reimbursement_participation_tickets": SupabaseQueryV2(
        policy_id="reimbursement_participation_tickets",
        table="research_loop_ticket_current",
        select=(
            "ticket_id,miner_hotkey,island,brief_sanitized_ref,"
            "created_at,current_status_at"
        ),
        parameter_names=("island",),
        max_pages=50,
        order="created_at.desc,ticket_id.asc",
    ),
    "reimbursement_queue_by_ticket": SupabaseQueryV2(
        policy_id="reimbursement_queue_by_ticket",
        table="research_loop_run_queue_current",
        select="run_id,ticket_id,current_queue_status,current_status_at",
        parameter_names=("ticket_id",),
        max_pages=1,
        order="current_status_at.desc,run_id.asc",
        limit=100,
    ),
    "reimbursement_cap_awards_by_day": SupabaseQueryV2(
        policy_id="reimbursement_cap_awards_by_day",
        table="research_reimbursement_award_current",
        select=(
            "award_id,miner_hotkey,island,run_day,current_award_status,"
            "award_status,target_reimbursement_microusd"
        ),
        parameter_names=("run_day",),
        max_pages=50,
        order="award_id.asc",
    ),
    "allocation_history": SupabaseQueryV2(
        policy_id="allocation_history",
        table="research_lab_emission_allocation_current",
        select="epoch,netuid,allocation_hash,allocation_doc",
        parameter_names=("netuid", "start_epoch", "end_epoch"),
        max_pages=100,
        order="epoch.desc",
    ),
    "finalized_allocation_authorities": SupabaseQueryV2(
        policy_id="finalized_allocation_authorities",
        table="research_lab_finalized_allocation_epochs_v2",
        select=(
            "bundle_hash,schema_version,netuid,epoch_id,block,validator_hotkey,"
            "root_receipt_hash,weights_hash,snapshot_hash,bundle_doc,"
            "weight_submission_event_hash,publication_receipt_hash,"
            "transparency_event_hash,durable_readback_hash,publication_doc,"
            "weight_finalization_event_hash,finalization_receipt_hash,"
            "extrinsic_authorization_hash,extrinsic_hash,finalized_block,"
            "finalized_block_hash,state_transition_hash,finalization_doc"
        ),
        parameter_names=("netuid", "start_epoch", "end_epoch"),
        max_pages=500,
        page_size=2,
        order="epoch_id.asc,validator_hotkey.asc",
    ),
    "latest_finalized_allocation_authority_summaries": SupabaseQueryV2(
        policy_id="latest_finalized_allocation_authority_summaries",
        table="research_lab_finalized_allocation_epochs_v2",
        select=(
            "bundle_hash,netuid,epoch_id,validator_hotkey,finalized_block,"
            "finalized_block_hash,finalization_receipt_hash"
        ),
        parameter_names=("netuid",),
        max_pages=1,
        order="finalized_block.desc,bundle_hash.asc",
        limit=2,
    ),
    "finalized_allocation_authority_by_bundle_hash": SupabaseQueryV2(
        policy_id="finalized_allocation_authority_by_bundle_hash",
        table="research_lab_finalized_allocation_epochs_v2",
        select=(
            "bundle_hash,schema_version,netuid,epoch_id,block,validator_hotkey,"
            "root_receipt_hash,weights_hash,snapshot_hash,bundle_doc,"
            "weight_submission_event_hash,publication_receipt_hash,"
            "transparency_event_hash,durable_readback_hash,publication_doc,"
            "weight_finalization_event_hash,finalization_receipt_hash,"
            "extrinsic_authorization_hash,extrinsic_hash,finalized_block,"
            "finalized_block_hash,state_transition_hash,finalization_doc"
        ),
        parameter_names=("netuid", "bundle_hash"),
        max_pages=1,
        limit=1,
    ),
    "finalized_authority_by_chain_vector": SupabaseQueryV2(
        policy_id="finalized_authority_by_chain_vector",
        table="research_lab_finalized_weight_vector_candidates_v1",
        select=(
            "bundle_hash,schema_version,netuid,epoch_id,block,validator_hotkey,"
            "root_receipt_hash,weights_hash,snapshot_hash,bundle_doc,"
            "weight_submission_event_hash,publication_receipt_hash,"
            "transparency_event_hash,durable_readback_hash,publication_doc,"
            "weight_finalization_event_hash,finalization_receipt_hash,"
            "extrinsic_authorization_hash,extrinsic_hash,finalized_block,"
            "finalized_block_hash,state_transition_hash,finalization_doc"
        ),
        parameter_names=(
            "netuid",
            "uids",
            "weights_u16",
            "source_epoch_id",
            "validator_hotkey",
            "finalized_block",
            "finalized_block_hash",
        ),
        max_pages=1,
        order="finalized_block.desc,bundle_hash.asc",
        limit=100,
    ),
    "compact_finalized_authority_cutover": SupabaseQueryV2(
        policy_id="compact_finalized_authority_cutover",
        table="research_lab_compact_weight_authorities_v2",
        select="epoch_id",
        parameter_names=("netuid",),
        max_pages=1,
        order="epoch_id.asc",
        limit=1,
    ),
    "latest_compact_finalized_authority_summaries": SupabaseQueryV2(
        policy_id="latest_compact_finalized_authority_summaries",
        table="research_lab_compact_weight_authorities_v2",
        select=(
            "bundle_hash,compact_submission_hash,netuid,epoch_id,"
            "validator_hotkey,authority_hash,lineage_id,"
            "finalization_receipt_hash"
        ),
        parameter_names=("netuid",),
        max_pages=1,
        order="epoch_id.desc,bundle_hash.asc",
        limit=2,
    ),
    "compact_finalized_authority_by_bundle_hash": SupabaseQueryV2(
        policy_id="compact_finalized_authority_by_bundle_hash",
        table="research_lab_compact_weight_authorities_v2",
        select=(
            "bundle_hash,compact_submission_hash,netuid,epoch_id,"
            "validator_hotkey,authority_stage,schema_version,lineage_id,"
            "authority_hash,publication_receipt_hash,"
            "compact_finalization_hash,finalization_receipt_hash,authority_doc"
        ),
        parameter_names=("netuid", "bundle_hash"),
        max_pages=1,
        limit=1,
    ),
    "compact_finalized_authority_by_identity": SupabaseQueryV2(
        policy_id="compact_finalized_authority_by_identity",
        table="research_lab_compact_weight_authorities_v2",
        select=(
            "bundle_hash,compact_submission_hash,netuid,epoch_id,"
            "validator_hotkey,authority_stage,schema_version,lineage_id,"
            "authority_hash,publication_receipt_hash,"
            "compact_finalization_hash,finalization_receipt_hash,authority_doc"
        ),
        parameter_names=(
            "netuid",
            "source_epoch_id",
            "validator_hotkey",
        ),
        max_pages=1,
        order="bundle_hash.asc",
        limit=11,
    ),
    "legacy_finalized_allocation_migrations": SupabaseQueryV2(
        policy_id="legacy_finalized_allocation_migrations",
        table="research_lab_legacy_finalized_allocation_migrations_v2",
        select=(
            "netuid,epoch_id,schema_version,allocation_hash,settlement_hash,"
            "settlement_receipt_hash,allocation_doc,settlement_doc"
        ),
        parameter_names=("netuid", "start_epoch", "end_epoch"),
        max_pages=100,
        order="epoch_id.asc",
    ),
    "chain_realized_epoch_settlements": SupabaseQueryV2(
        policy_id="chain_realized_epoch_settlements",
        table="research_lab_chain_realized_epoch_settlements_v1",
        select=(
            "netuid,epoch_id,schema_version,settlement_hash,"
            "settlement_receipt_hash,settlement_doc"
        ),
        parameter_names=("netuid", "start_epoch", "end_epoch"),
        max_pages=100,
        order="epoch_id.asc",
    ),
    "chain_realized_settlement_activation": SupabaseQueryV2(
        policy_id="chain_realized_settlement_activation",
        table="research_lab_chain_realized_settlement_activation_v1",
        select=(
            "netuid,schema_version,first_epoch_id,source_bundle_hash,"
            "source_bundle_epoch_id,source_finalized_block"
        ),
        parameter_names=("netuid",),
        max_pages=1,
        order="first_epoch_id.asc",
        limit=1,
    ),
    "chain_realized_obligation_credits": SupabaseQueryV2(
        policy_id="chain_realized_obligation_credits",
        table="research_lab_chain_realized_obligation_credits_v1",
        select=(
            "netuid,epoch_id,settlement_hash,schema_version,obligation_kind,"
            "obligation_source_id,miner_hotkey,miner_uid,"
            "observed_chain_alpha_percent,lab_attributed_alpha_percent,"
            "scheduled_alpha_percent,credited_alpha_percent,"
            "champion_credit_policy,credit_hash,"
            "credit_receipt_hash,credit_doc"
        ),
        parameter_names=("netuid", "start_epoch", "end_epoch"),
        max_pages=100,
        order="epoch_id.asc,obligation_kind.asc,obligation_source_id.asc",
    ),
    "allocation_settlement_frontier_activation": SupabaseQueryV2(
        policy_id="allocation_settlement_frontier_activation",
        table="research_lab_allocation_settlement_frontier_activation_v2",
        select=(
            "netuid,schema_version,first_allocation_epoch,first_frontier_hash,"
            "source_receipt_hash"
        ),
        parameter_names=("netuid",),
        max_pages=1,
        limit=1,
    ),
    "allocation_settlement_frontiers": SupabaseQueryV2(
        policy_id="allocation_settlement_frontiers",
        table="research_lab_allocation_settlement_frontiers_v2",
        select=(
            "netuid,allocation_epoch,settled_through_epoch,schema_version,"
            "frontier_hash,predecessor_frontier_hash,source_receipt_hash,"
            "source_state_hash,frontier_doc"
        ),
        parameter_names=("netuid", "before_epoch"),
        max_pages=1,
        order="allocation_epoch.desc",
        limit=1,
    ),
    "allocation_settlement_frontier_by_epoch": SupabaseQueryV2(
        policy_id="allocation_settlement_frontier_by_epoch",
        table="research_lab_allocation_settlement_frontiers_v2",
        select=(
            "netuid,allocation_epoch,settled_through_epoch,schema_version,"
            "frontier_hash,predecessor_frontier_hash,source_receipt_hash,"
            "source_state_hash,frontier_doc"
        ),
        parameter_names=("netuid", "allocation_epoch"),
        max_pages=1,
        limit=1,
    ),
    "legacy_weight_bundles_by_epoch": SupabaseQueryV2(
        policy_id="legacy_weight_bundles_by_epoch",
        table="published_weight_bundles",
        select=(
            "netuid,epoch_id,block,uids,weights_u16,weights_hash,validator_hotkey,"
            "validator_enclave_pubkey,validator_signature,validator_pcr0,"
            "pcr0_commit_hash,weight_submission_event_hash,created_at"
        ),
        parameter_names=("netuid", "epoch_id"),
        max_pages=1,
        order="validator_hotkey.asc,created_at.asc",
        limit=100,
    ),
    "legacy_audit_anchor_by_epoch": SupabaseQueryV2(
        policy_id="legacy_audit_anchor_by_epoch",
        table="research_lab_arweave_epoch_audit_anchor_current",
        select=(
            "anchor_id,epoch,netuid,audit_kind,allocation_hash,weights_hash,"
            "payload_hash,transparency_event_hash,current_anchor_status,"
            "current_transparency_event_hash,current_tee_sequence,"
            "current_checkpoint_number,current_checkpoint_merkle_root,"
            "current_arweave_tx_id,current_status_at"
        ),
        parameter_names=("netuid", "epoch_id"),
        max_pages=1,
        order="current_status_at.desc",
        limit=2,
    ),
    "legacy_transparency_event_by_hash": SupabaseQueryV2(
        policy_id="legacy_transparency_event_by_hash",
        table="transparency_log",
        select=(
            "event_type,event_hash,payload_hash,enclave_pubkey,signed_log_entry,"
            "tee_sequence,created_at"
        ),
        parameter_names=("event_hash",),
        max_pages=1,
        limit=2,
    ),
    "attested_business_artifact_by_ref": SupabaseQueryV2(
        policy_id="attested_business_artifact_by_ref",
        table="research_lab_attested_business_artifact_links_v2",
        select="receipt_hash,artifact_kind,artifact_ref,artifact_hash",
        parameter_names=("artifact_kind", "artifact_ref", "artifact_hash"),
        max_pages=1,
        order="created_at.desc",
        limit=1,
    ),
    "attested_receipt_by_hash": SupabaseQueryV2(
        policy_id="attested_receipt_by_hash",
        table="research_lab_attested_execution_receipts_v2",
        select="receipt_hash,role,purpose,epoch_id,output_root,boot_identity_hash,receipt_doc",
        parameter_names=("receipt_hash",),
        max_pages=1,
        limit=1,
    ),
    "attested_execution_result_by_receipt": SupabaseQueryV2(
        policy_id="attested_execution_result_by_receipt",
        table="research_lab_attested_execution_results_v2",
        select=(
            "receipt_hash,schema_version,role,operation,purpose,job_id,epoch_id,"
            "sequence,release_hash,input_root,output_root,artifact_root,"
            "result_hash,artifact_hashes,result_doc"
        ),
        parameter_names=("receipt_hash",),
        max_pages=1,
        limit=1,
    ),
    "latest_attested_allocation_execution_results": SupabaseQueryV2(
        policy_id="latest_attested_allocation_execution_results",
        table="research_lab_attested_execution_results_v2",
        select=(
            "receipt_hash,schema_version,role,operation,purpose,job_id,epoch_id,"
            "sequence,release_hash,input_root,output_root,artifact_root,"
            "result_hash,artifact_hashes,result_doc"
        ),
        parameter_names=("through_epoch",),
        max_pages=1,
        order="epoch_id.desc,receipt_hash.asc",
        limit=100,
    ),
    "attested_artifact_by_ref": SupabaseQueryV2(
        policy_id="attested_artifact_by_ref",
        table="research_lab_attested_artifact_links_v2",
        select="receipt_hash,artifact_kind,artifact_ref,artifact_hash",
        parameter_names=("artifact_kind", "artifact_ref"),
        max_pages=1,
        limit=1,
    ),
    "sourcing_epoch_inputs": SupabaseQueryV2(
        policy_id="sourcing_epoch_inputs",
        table="validator_sourcing_epoch_inputs_v2",
        select="epoch_id,epoch_hash,receipt_hash,source_doc,receipt_doc",
        parameter_names=("start_epoch", "end_epoch"),
        max_pages=1,
        order="epoch_id.asc",
        limit=30,
    ),
    "sourcing_epoch_inputs_empty": SupabaseQueryV2(
        policy_id="sourcing_epoch_inputs_empty",
        table="validator_sourcing_epoch_inputs_v2",
        select="epoch_id,epoch_hash,receipt_hash,source_doc,receipt_doc",
        parameter_names=(),
        max_pages=1,
        order="epoch_id.asc",
        limit=1,
    ),
}


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SupabaseSourceV2Error("%s must be an integer" % field)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SupabaseSourceV2Error("%s must be an integer" % field) from exc
    if result < 0:
        raise SupabaseSourceV2Error("%s must be non-negative" % field)
    return result


def _identifier(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 512 or any(
        character in normalized for character in ("\x00", "\r", "\n")
    ):
        raise SupabaseSourceV2Error("%s is invalid" % field)
    return normalized


def _raw_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise SupabaseSourceV2Error("%s is invalid" % field)
    return normalized


def _content_hash(value: Any, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized):
        raise SupabaseSourceV2Error("%s is invalid" % field)
    return normalized


def _uuid(value: Any, field: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise SupabaseSourceV2Error("%s is not a UUID" % field) from exc


def _filters(policy: SupabaseQueryV2, parameters: Mapping[str, Any]) -> Sequence[Tuple[str, str]]:
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        policy.parameter_names
    ):
        raise SupabaseSourceV2Error("Supabase policy parameters are invalid")
    if policy.policy_id == "qualification_epoch_assignment":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        return (
            ("event_type", "eq.EPOCH_INITIALIZATION"),
            ("payload->>epoch_id", "eq.%d" % epoch_id),
        )
    if policy.policy_id == "qualification_leads_by_ids":
        values = parameters["lead_ids"]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or len(values) > 200
        ):
            raise SupabaseSourceV2Error("lead_ids must contain 1-200 UUIDs")
        normalized = []
        for value in values:
            try:
                normalized.append(str(UUID(str(value))))
            except (TypeError, ValueError, AttributeError) as exc:
                raise SupabaseSourceV2Error("lead_id is not a UUID") from exc
        if len(normalized) != len(set(normalized)):
            raise SupabaseSourceV2Error("lead_ids are duplicated")
        return (("lead_id", "in.(%s)" % ",".join(normalized)),)
    if policy.policy_id == "banned_hotkeys":
        return ()
    if policy.policy_id == "fulfillment_active_rewards":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        return (
            ("reward_pct", "not.is.null"),
            ("reward_expires_epoch", "gt.%d" % epoch_id),
        )
    if policy.policy_id == "fulfillment_leaderboard_winners":
        window_start = _identifier(parameters["window_start"], "window_start")
        window_end = _identifier(parameters["window_end"], "window_end")
        if not _TIMESTAMP_RE.fullmatch(window_start):
            raise SupabaseSourceV2Error("window_start is not an ISO timestamp")
        if not _TIMESTAMP_RE.fullmatch(window_end):
            raise SupabaseSourceV2Error("window_end is not an ISO timestamp")
        try:
            start_value = datetime.fromisoformat(window_start.replace("Z", "+00:00"))
            end_value = datetime.fromisoformat(window_end.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SupabaseSourceV2Error("leaderboard window is invalid") from exc
        if (
            start_value.tzinfo is None
            or end_value.tzinfo is None
            or end_value.astimezone(timezone.utc)
            < start_value.astimezone(timezone.utc)
        ):
            raise SupabaseSourceV2Error("leaderboard window is inverted")
        return (
            ("is_winner", "eq.true"),
            ("computed_at", "gte.%s" % window_start),
            ("computed_at", "lte.%s" % window_end),
        )
    if policy.policy_id == "lab_arena_reward_basis":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        return (("effective_reward_epoch", "lte.%d" % epoch_id),)
    if policy.policy_id == "research_lab_allocation_current":
        return (
            ("epoch", "eq.%d" % _non_negative_int(parameters["epoch_id"], "epoch_id")),
            ("netuid", "eq.%d" % _non_negative_int(parameters["netuid"], "netuid")),
        )
    if policy.policy_id == "latest_native_compute_allocation_authority":
        return (
            (
                "epoch_id",
                "lt.%d"
                % _non_negative_int(parameters["epoch_id"], "epoch_id"),
            ),
            (
                "netuid",
                "eq.%d" % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "bundle_doc->weight_snapshot->calculation_snapshot"
                "->research_lab_allocation_doc->reimbursement_allocations",
                "not.eq.[]",
            ),
            (
                "bundle_doc->weight_snapshot->calculation_snapshot"
                "->research_lab_allocation_doc"
                "->>historical_compute_fallback_source_epoch",
                "is.null",
            ),
        )
    if policy.policy_id == "latest_legacy_compute_allocation_authority":
        return (
            (
                "epoch_id",
                "lt.%d"
                % _non_negative_int(parameters["epoch_id"], "epoch_id"),
            ),
            (
                "netuid",
                "eq.%d" % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            ("allocation_doc->reimbursement_allocations", "not.eq.[]"),
            (
                "allocation_doc->>historical_compute_fallback_source_epoch",
                "is.null",
            ),
        )
    if policy.policy_id == "legacy_allocation_by_hash":
        allocation_hash = str(parameters["allocation_hash"] or "").strip().lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", allocation_hash):
            raise SupabaseSourceV2Error("allocation_hash is invalid")
        return (
            ("allocation_hash", "eq.%s" % allocation_hash),
            ("epoch", "eq.%d" % _non_negative_int(parameters["epoch_id"], "epoch_id")),
            ("netuid", "eq.%d" % _non_negative_int(parameters["netuid"], "netuid")),
        )
    if policy.policy_id == "score_bundle_by_id":
        score_bundle_id = _identifier(parameters["score_bundle_id"], "score_bundle_id")
        if not re.fullmatch(r"score_bundle:[0-9a-f]{64}", score_bundle_id):
            raise SupabaseSourceV2Error("score_bundle_id is invalid")
        return (("score_bundle_id", "eq.%s" % score_bundle_id),)
    if policy.policy_id == "allocation_reimbursement_schedules":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        start_epoch = _non_negative_int(parameters["start_epoch"], "start_epoch")
        if start_epoch > epoch_id:
            raise SupabaseSourceV2Error("reimbursement schedule range is inverted")
        return (
            ("schedule_status", "eq.scheduled"),
            ("start_epoch", "lte.%d" % epoch_id),
            ("start_epoch", "gte.%d" % start_epoch),
        )
    if policy.policy_id == "allocation_reimbursement_awards":
        award_ids = parameters["award_ids"]
        if (
            not isinstance(award_ids, (list, tuple))
            or not award_ids
            or len(award_ids) > 1_000
        ):
            raise SupabaseSourceV2Error("award_ids must contain 1-1000 values")
        normalized = []
        for value in award_ids:
            identifier = _identifier(value, "award_id")
            if not re.fullmatch(r"reimbursement_award:sha256:[0-9a-f]{64}", identifier):
                raise SupabaseSourceV2Error("award_id is invalid")
            normalized.append(identifier)
        if len(normalized) != len(set(normalized)):
            raise SupabaseSourceV2Error("award_ids are duplicated")
        return (
            ("award_id", "in.(%s)" % ",".join(normalized)),
            ("current_award_status", "eq.awarded"),
        )
    if policy.policy_id == "allocation_champion_rewards":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        include_paid = parameters["include_paid"]
        if not isinstance(include_paid, bool):
            raise SupabaseSourceV2Error("include_paid must be boolean")
        statuses = (
            "active,queued,partially_paid,paid"
            if include_paid
            else "active,queued,partially_paid"
        )
        return (
            ("current_reward_status", "in.(%s)" % statuses),
            ("start_epoch", "lte.%d" % epoch_id),
        )
    if policy.policy_id == "champion_reward_by_id":
        reward_id = _identifier(
            parameters["champion_reward_id"], "champion_reward_id"
        )
        if not re.fullmatch(r"champion_reward:sha256:[0-9a-f]{64}", reward_id):
            raise SupabaseSourceV2Error("champion_reward_id is invalid")
        return (("champion_reward_id", "eq.%s" % reward_id),)
    if policy.policy_id == "allocation_source_add_rewards":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        return (
            ("current_reward_status", "in.(active,queued,partially_paid)"),
            ("start_epoch", "lte.%d" % epoch_id),
        )
    if policy.policy_id == "source_add_reward_by_ref":
        reward_ref = _identifier(parameters["reward_ref"], "reward_ref")
        if not re.fullmatch(r"source_add_reward:[0-9a-f]{16}", reward_ref):
            raise SupabaseSourceV2Error("reward_ref is invalid")
        return (("reward_ref", "eq.%s" % reward_ref),)
    if policy.policy_id == "source_add_rewards_by_adapter":
        return (
            (
                "adapter_id",
                "eq.%s" % _identifier(parameters["adapter_id"], "adapter_id"),
            ),
        )
    if policy.policy_id in {
        "source_add_submission_by_id",
        "source_add_probe_config_by_submission",
        "source_add_functional_probe_by_submission",
        "source_add_provisioning_smoke_by_submission",
    }:
        submission_id = _identifier(parameters["submission_id"], "submission_id")
        if not re.fullmatch(r"source_add_submission:[0-9a-f]{16}", submission_id):
            raise SupabaseSourceV2Error("submission_id is invalid")
        filters = [("submission_id", "eq.%s" % submission_id)]
        if policy.policy_id == "source_add_probe_config_by_submission":
            filters.append(("config_status", "eq.active"))
        return tuple(filters)
    if policy.policy_id == "source_add_leg1_events_since":
        day_start = _identifier(parameters["day_start"], "day_start")
        if not _TIMESTAMP_RE.fullmatch(day_start):
            raise SupabaseSourceV2Error("day_start is not an ISO timestamp")
        return (
            (
                "reason",
                "in.(leg1_provenance_precheck_passed,leg1_functional_probe_passed)",
            ),
            ("created_at", "gte.%s" % day_start),
        )
    if policy.policy_id == "source_add_provisioning_by_adapter":
        return (
            (
                "adapter_id",
                "eq.%s" % _identifier(parameters["adapter_id"], "adapter_id"),
            ),
            ("provision_status", "eq.provisioned_autoresearch_eligible"),
        )
    if policy.policy_id in {
        "source_add_provisioning_eligible",
        "provider_registry_recent",
    }:
        if policy.policy_id == "source_add_provisioning_eligible":
            return (("provision_status", "eq.provisioned_autoresearch_eligible"),)
        return ()
    if policy.policy_id in {
        "reimbursement_ticket_by_id",
        "reimbursement_queue_by_ticket",
    }:
        ticket_id = _uuid(parameters["ticket_id"], "ticket_id")
        return (("ticket_id", "eq.%s" % ticket_id),)
    if policy.policy_id == "reimbursement_receipt_by_id":
        receipt_id = _uuid(parameters["receipt_id"], "receipt_id")
        return (("receipt_id", "eq.%s" % receipt_id),)
    if policy.policy_id == "reimbursement_payment_by_id":
        payment_id = _uuid(parameters["payment_id"], "payment_id")
        return (("payment_id", "eq.%s" % payment_id),)
    if policy.policy_id == "reimbursement_queue_events_by_run":
        run_id = _uuid(parameters["run_id"], "run_id")
        return (("run_id", "eq.%s" % run_id),)
    if policy.policy_id == "reimbursement_participation_tickets":
        island = _identifier(parameters["island"], "island")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", island):
            raise SupabaseSourceV2Error("island is invalid")
        return (("island", "eq.%s" % island),)
    if policy.policy_id == "reimbursement_cap_awards_by_day":
        run_day = _identifier(parameters["run_day"], "run_day")
        try:
            datetime.strptime(run_day, "%Y-%m-%d")
        except ValueError as exc:
            raise SupabaseSourceV2Error("run_day is invalid") from exc
        return (
            ("current_award_status", "eq.awarded"),
            ("run_day", "eq.%s" % run_day),
        )
    if policy.policy_id == "allocation_history":
        start_epoch = _non_negative_int(parameters["start_epoch"], "start_epoch")
        end_epoch = _non_negative_int(parameters["end_epoch"], "end_epoch")
        if end_epoch < start_epoch:
            raise SupabaseSourceV2Error("allocation history range is inverted")
        return (
            ("netuid", "eq.%d" % _non_negative_int(parameters["netuid"], "netuid")),
            ("epoch", "gte.%d" % start_epoch),
            ("epoch", "lte.%d" % end_epoch),
        )
    if policy.policy_id in {
        "finalized_allocation_authorities",
        "legacy_finalized_allocation_migrations",
        "chain_realized_epoch_settlements",
        "chain_realized_obligation_credits",
    }:
        start_epoch = _non_negative_int(parameters["start_epoch"], "start_epoch")
        end_epoch = _non_negative_int(parameters["end_epoch"], "end_epoch")
        if end_epoch < start_epoch:
            raise SupabaseSourceV2Error(
                "finalized allocation authority range is inverted"
            )
        return (
            ("netuid", "eq.%d" % _non_negative_int(parameters["netuid"], "netuid")),
            ("epoch_id", "gte.%d" % start_epoch),
            ("epoch_id", "lte.%d" % end_epoch),
        )
    if policy.policy_id == "latest_finalized_allocation_authority_summaries":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
        )
    if policy.policy_id in {
        "compact_finalized_authority_cutover",
        "latest_compact_finalized_authority_summaries",
    }:
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            ("authority_stage", "eq.finalized"),
        )
    if policy.policy_id == "compact_finalized_authority_by_bundle_hash":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "bundle_hash",
                "eq.%s"
                % _content_hash(parameters["bundle_hash"], "bundle_hash"),
            ),
            ("authority_stage", "eq.finalized"),
        )
    if policy.policy_id == "compact_finalized_authority_by_identity":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "epoch_id",
                "eq.%d"
                % _non_negative_int(
                    parameters["source_epoch_id"], "source_epoch_id"
                ),
            ),
            (
                "validator_hotkey",
                "eq.%s"
                % _identifier(
                    parameters["validator_hotkey"], "validator_hotkey"
                ),
            ),
            ("authority_stage", "eq.finalized"),
        )
    if policy.policy_id == "finalized_allocation_authority_by_bundle_hash":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "bundle_hash",
                "eq.%s"
                % _content_hash(parameters["bundle_hash"], "bundle_hash"),
            ),
        )
    if policy.policy_id == "chain_realized_settlement_activation":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
        )
    if policy.policy_id == "allocation_settlement_frontier_activation":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
        )
    if policy.policy_id == "allocation_settlement_frontiers":
        before_epoch = _non_negative_int(
            parameters["before_epoch"], "before_epoch"
        )
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            ("allocation_epoch", "lt.%d" % before_epoch),
        )
    if policy.policy_id == "allocation_settlement_frontier_by_epoch":
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "allocation_epoch",
                "eq.%d"
                % _non_negative_int(
                    parameters["allocation_epoch"], "allocation_epoch"
                ),
            ),
        )
    if policy.policy_id == "finalized_authority_by_chain_vector":
        raw_uids = parameters["uids"]
        raw_weights = parameters["weights_u16"]
        if (
            not isinstance(raw_uids, list)
            or not isinstance(raw_weights, list)
            or len(raw_uids) != len(raw_weights)
            or not raw_uids
        ):
            raise SupabaseSourceV2Error(
                "finalized authority vector is invalid"
            )
        uids: list[int] = []
        weights: list[int] = []
        for raw_uid, raw_weight in zip(raw_uids, raw_weights):
            if isinstance(raw_uid, bool) or isinstance(raw_weight, bool):
                raise SupabaseSourceV2Error(
                    "finalized authority vector is invalid"
                )
            try:
                uid = int(raw_uid)
                weight = int(raw_weight)
            except (TypeError, ValueError) as exc:
                raise SupabaseSourceV2Error(
                    "finalized authority vector is invalid"
                ) from exc
            if uid < 0 or not 1 <= weight <= 65535:
                raise SupabaseSourceV2Error(
                    "finalized authority vector is invalid"
                )
            uids.append(uid)
            weights.append(weight)
        if uids != sorted(set(uids)):
            raise SupabaseSourceV2Error(
                "finalized authority vector is not canonical"
            )
        return (
            (
                "netuid",
                "eq.%d"
                % _non_negative_int(parameters["netuid"], "netuid"),
            ),
            (
                "epoch_id",
                "eq.%d"
                % _non_negative_int(
                    parameters["source_epoch_id"],
                    "source_epoch_id",
                ),
            ),
            (
                "validator_hotkey",
                "eq.%s"
                % _identifier(
                    parameters["validator_hotkey"],
                    "validator_hotkey",
                ),
            ),
            (
                "finalized_block",
                "eq.%d"
                % _non_negative_int(
                    parameters["finalized_block"],
                    "finalized_block",
                ),
            ),
            (
                "finalized_block_hash",
                "eq.%s"
                % _raw_hash(
                    parameters["finalized_block_hash"],
                    "finalized_block_hash",
                ),
            ),
            ("uids", "eq.%s" % json.dumps(uids, separators=(",", ":"))),
            (
                "weights_u16",
                "eq.%s" % json.dumps(weights, separators=(",", ":")),
            ),
        )
    if policy.policy_id in {
        "legacy_weight_bundles_by_epoch",
        "legacy_audit_anchor_by_epoch",
    }:
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        netuid = _non_negative_int(parameters["netuid"], "netuid")
        filters = [
            ("netuid", "eq.%d" % netuid),
            (("epoch_id" if policy.policy_id == "legacy_weight_bundles_by_epoch" else "epoch"), "eq.%d" % epoch_id),
        ]
        if policy.policy_id == "legacy_audit_anchor_by_epoch":
            filters.extend(
                (
                    ("audit_kind", "eq.active"),
                    ("current_anchor_status", "eq.checkpointed"),
                )
            )
        return tuple(filters)
    if policy.policy_id == "legacy_transparency_event_by_hash":
        event_hash = str(parameters["event_hash"] or "").strip().lower()
        if event_hash.startswith("sha256:"):
            event_hash = event_hash.split(":", 1)[1]
        if not re.fullmatch(r"[0-9a-f]{64}", event_hash):
            raise SupabaseSourceV2Error("event_hash is invalid")
        return (("event_hash", "eq.%s" % event_hash),)
    if policy.policy_id == "attested_business_artifact_by_ref":
        artifact_hash = str(parameters["artifact_hash"] or "").strip().lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact_hash):
            raise SupabaseSourceV2Error("artifact_hash is invalid")
        return (
            (
                "artifact_kind",
                "eq.%s" % _identifier(parameters["artifact_kind"], "artifact_kind"),
            ),
            (
                "artifact_ref",
                "eq.%s" % _identifier(parameters["artifact_ref"], "artifact_ref"),
            ),
            ("artifact_hash", "eq.%s" % artifact_hash),
        )
    if policy.policy_id == "attested_receipt_by_hash":
        return (("receipt_hash", "eq.%s" % _identifier(parameters["receipt_hash"], "receipt_hash")),)
    if policy.policy_id == "attested_execution_result_by_receipt":
        return (
            (
                "receipt_hash",
                "eq.%s"
                % _content_hash(parameters["receipt_hash"], "receipt_hash"),
            ),
        )
    if policy.policy_id == "latest_attested_allocation_execution_results":
        return (
            ("role", "eq.gateway_coordinator"),
            ("operation", "eq.research_lab_allocation"),
            ("purpose", "eq.research_lab.allocation.v2"),
            (
                "epoch_id",
                "lte.%d"
                % _non_negative_int(
                    parameters["through_epoch"],
                    "through_epoch",
                ),
            ),
        )
    if policy.policy_id == "attested_artifact_by_ref":
        return (
            ("artifact_kind", "eq.%s" % _identifier(parameters["artifact_kind"], "artifact_kind")),
            ("artifact_ref", "eq.%s" % _identifier(parameters["artifact_ref"], "artifact_ref")),
        )
    if policy.policy_id == "sourcing_epoch_inputs":
        start_epoch = _non_negative_int(parameters["start_epoch"], "start_epoch")
        end_epoch = _non_negative_int(parameters["end_epoch"], "end_epoch")
        if end_epoch < start_epoch:
            raise SupabaseSourceV2Error("sourcing epoch range is inverted")
        return (("epoch_id", "gte.%d" % start_epoch), ("epoch_id", "lte.%d" % end_epoch))
    if policy.policy_id == "sourcing_epoch_inputs_empty":
        return (("epoch_id", "lt.0"),)
    raise SupabaseSourceV2Error("Supabase query policy is unsupported")


class SupabaseSourceReaderV2:
    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hash: str,
        origin: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hash = str(retry_policy_hash or "")
        self._origin = str(origin or configured_supabase_origin_v2())
        self._sleep = sleep

    def read(
        self,
        *,
        policy_id: str,
        parameters: Mapping[str, Any],
        job_id: str,
        purpose: str,
        record_transport: Callable[[Mapping[str, Any]], None],
        record_artifact: Callable[[str], None],
    ) -> list[Dict[str, Any]]:
        policy = QUERY_POLICIES.get(str(policy_id or ""))
        if policy is None:
            raise SupabaseSourceV2Error("Supabase query policy is not measured")
        if not 1 <= policy.page_size <= SUPABASE_PAGE_SIZE:
            raise SupabaseSourceV2Error("Supabase policy page size is invalid")
        filters = _filters(policy, parameters)
        rows = []
        for page_index in range(policy.max_pages):
            page = self._read_page(
                policy=policy,
                filters=filters,
                page_index=page_index,
                job_id=job_id,
                purpose=purpose,
                record_transport=record_transport,
                record_artifact=record_artifact,
            )
            rows.extend(page)
            if policy.limit or len(page) < policy.page_size:
                break
        else:
            raise SupabaseSourceV2Error("Supabase query exceeded its measured page limit")
        return rows

    def _read_page(
        self,
        *,
        policy: SupabaseQueryV2,
        filters: Sequence[Tuple[str, str]],
        page_index: int,
        job_id: str,
        purpose: str,
        record_transport: Callable[[Mapping[str, Any]], None],
        record_artifact: Callable[[str], None],
    ) -> list[Dict[str, Any]]:
        query = [("select", policy.select), *filters]
        if policy.order:
            query.append(("order", policy.order))
        if policy.limit:
            query.append(("limit", str(policy.limit)))
        url = "%s/rest/v1/%s?%s" % (
            self._origin,
            policy.table,
            urlencode(query),
        )
        start = page_index * policy.page_size
        end = start + policy.page_size - 1
        query_scope_hash = sha256_json(
            {
                "schema_version": "leadpoet.supabase_query_scope.v2",
                "policy_id": policy.policy_id,
                "filters": [list(item) for item in filters],
                "page_size": policy.page_size,
            }
        )
        logical_operation_id = "%s:%s:%s:page-%d" % (
            job_id,
            policy.policy_id,
            query_scope_hash,
            page_index,
        )
        last_error = "unavailable"
        for attempt_number in range(len(SUPABASE_RETRY_BACKOFF_SECONDS) + 1):
            result = dict(
                self._execute_provider(
                    {
                        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                        "logical_operation_id": logical_operation_id,
                        "job_id": job_id,
                        "purpose": purpose,
                        "provider_id": "supabase",
                        "attempt_number": attempt_number,
                        "method": "GET",
                        "url": url,
                        "headers": {
                            "accept": "application/json",
                            "range": "%d-%d" % (start, end),
                            "range-unit": "items",
                        },
                        "body_b64": base64.b64encode(b"").decode("ascii"),
                        "timeout_ms": SUPABASE_READ_TIMEOUT_MS,
                        "retry_policy_hash": self._retry_policy_hash,
                    }
                )
            )
            attempt = result.get("transport_attempt")
            if not isinstance(attempt, Mapping):
                raise SupabaseSourceV2Error("Supabase terminal attempt is missing")
            record_transport(attempt)
            record_artifact(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                record_artifact(str(attempt["response_artifact_hash"]))
            if (
                result.get("terminal_status") == "authenticated_response"
                and 200 <= int(result.get("http_status") or 0) < 300
            ):
                try:
                    body = base64.b64decode(
                        str(result.get("body_b64") or ""), validate=True
                    )
                    parsed = json.loads(body.decode("utf-8"))
                except Exception as exc:
                    last_error = "malformed_json"
                else:
                    if not isinstance(parsed, list) or any(
                        not isinstance(item, Mapping) for item in parsed
                    ):
                        last_error = "response_not_row_array"
                    else:
                        if sha256_bytes(body) != attempt.get("response_hash"):
                            raise SupabaseSourceV2Error(
                                "Supabase response body hash differs from terminal record"
                            )
                        return [dict(item) for item in parsed]
            else:
                last_error = str(
                    result.get("failure_code")
                    or "http_%s" % result.get("http_status")
                )
            if attempt_number < len(SUPABASE_RETRY_BACKOFF_SECONDS):
                self._sleep(SUPABASE_RETRY_BACKOFF_SECONDS[attempt_number])
        raise SupabaseSourceV2Error(
            "Supabase %s page %d failed: %s"
            % (policy.policy_id, page_index, last_error)
        )
