#!/usr/bin/env python3.11
"""Exercise settlement-critical candidate migrations in disposable PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from gateway.research_lab import store as research_lab_store
from gateway.research_lab.champion_settlement_v2 import (
    CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
    ChampionSettlementV2Error,
    _preliminary_finalized_bundle_authority_v1,
    build_chain_realized_settlement_package_v1,
    validate_legacy_settlement_migrations_v2,
)
from gateway.research_lab.attested_v2_store import (
    _execution_result_storage_row_v2,
    boot_storage_row,
    receipt_storage_row,
    transport_storage_row,
)
from gateway.tee.supabase_schema_preflight_v2 import (
    REQUIRED_SUPABASE_V2_RPCS,
    REQUIRED_SUPABASE_V2_SCHEMA,
    SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256,
    SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256,
    SOURCE_ADD_MINER_STATUS_CONTRACT_AUTHORITY_SHA256,
    SOURCE_ADD_MINER_STATUS_PAGE_AUTHORITY_SHA256,
    SOURCE_ADD_MINER_STATUS_VIEW_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256,
    SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256,
)
from gateway.tee.coordinator_chain_realized_settlement_v1 import (
    OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
)
from gateway.tee.coordinator_executor_v2 import (
    CoordinatorExecutorV2,
    coordinator_receipt_output_v2,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from leadpoet_canonical.attested_v2 import (
    build_receipt_graph,
    build_transport_attempt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.allocation_settlement_frontier_v2 import (
    build_allocation_settlement_frontier_v2,
)
from leadpoet_canonical.allocation_settlement_frontier_bootstrap_v2 import (
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
    build_allocation_settlement_frontier_bootstrap_v2,
    frontier_bootstrap_artifact_hashes_v2,
)
from leadpoet_canonical.legacy_settlement_v2 import (
    LEGACY_SETTLEMENT_SCHEMA_VERSION,
    validate_legacy_settlement_document_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    build_weight_extrinsic_authorization_v2,
    chain_signing_profiles,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch
from tests.restart_rehearsal.fixture_contract import (
    load_rehearsal_current_settlement_epoch_id,
    load_rehearsal_metagraph_hotkeys,
    validate_rehearsal_finalized_authority_epochs,
)
from tests.restart_rehearsal.sanitized_weight_fixture import (
    NOW,
    SanitizedWeightFixture,
)


ALLOCATION_CANDIDATE_MIGRATION = (
    "33-research-lab-candidate-evaluation-queue.sql"
)
EVENT_PROJECTIONS_MIGRATION = "29-research-lab-event-projections.sql"
ALLOCATION_AUTO_RESEARCH_MIGRATION = (
    "34-research-lab-auto-research-loop-events.sql"
)
ALLOCATION_SCHEMA_MIGRATION = "35-research-lab-emission-allocator.sql"
ALLOCATION_SCORING_AUDIT_MIGRATION = (
    "36-research-lab-gateway-scoring-audit.sql"
)
ALLOCATION_PROMOTION_MIGRATION = (
    "37-research-lab-promotion-and-public-benchmarks.sql"
)
ATOMIC_CLAIM_GUARDS_MIGRATION = (
    "42-research-lab-atomic-claim-guards.sql"
)
QUEUE_CAPACITY_GUARD_MIGRATION = (
    "43-research-lab-queue-capacity-guard.sql"
)
MAINTENANCE_PAUSE_MIGRATION = "44-research-lab-maintenance-pause.sql"
PAUSED_CAPACITY_AGING_MIGRATION = (
    "48-research-lab-paused-capacity-aging.sql"
)
RESUME_REQUEUE_HOTKEY_GUARD_MIGRATION = (
    "54-research-lab-resume-requeue-hotkey-guard.sql"
)
HOTKEY_ACTIVE_LOOP_CAP_MIGRATION = (
    "67-research-lab-hotkey-active-loop-cap.sql"
)
ALLOCATION_IMAGE_BUILD_MIGRATIONS = (
    "46-research-lab-code-edit-candidate-images.sql",
    "47-research-lab-disable-new-patch-candidates.sql",
    "52-research-lab-image-build-candidate-current-view.sql",
)
SOURCE_ADD_PRE_V2_MIGRATIONS = (
    "72-research-lab-source-experiments.sql",
    "74-research-lab-source-add-provenance-precheck.sql",
    "78-research-lab-source-add-catalog-provisioning.sql",
    "79-research-lab-source-add-llm-leg2-evidence.sql",
    "82-research-lab-source-add-llm-only-leg2.sql",
    "84-expand-source-add-source-kinds.sql",
)
ALLOCATION_CONTAINMENT_MIGRATION = (
    "87-research-lab-source-add-allocation-containment.sql"
)
GIT_TREE_AUTORESEARCH_MIGRATION = (
    "95-research-lab-git-tree-autoresearch.sql"
)
SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION = (
    "96-research-lab-source-add-functional-workflow.sql"
)
GIT_TREE_ROOT_REPLACEMENT_MIGRATION = (
    "115-research-lab-git-tree-root-replacement.sql"
)
MAINTENANCE_LEASE_MIGRATION = "118-research-lab-maintenance-lease.sql"
MIGRATIONS_BEFORE_TRANSPORT_FIX = (
    "86-research-lab-attested-v2-authority.sql",
    "89-research-lab-provider-evidence-cache-v2.sql",
    "90-research-lab-provider-outcome-checkpoints-v2.sql",
    "99-research-lab-v2-champion-settlement.sql",
    "104-research-lab-attested-result-replay-v2.sql",
    MAINTENANCE_LEASE_MIGRATION,
    "125-research-lab-artifact-key-lineage.sql",
    "126-research-lab-chain-realized-settlement.sql",
    "127-research-lab-chain-unattributed-settlement.sql",
)
TRANSPORT_FIX_MIGRATION = "128-research-lab-chain-settlement-transport-purposes.sql"
TRANSPORT_TERMINAL_MIGRATION = (
    "129-research-lab-attested-local-transport.sql"
)
PROVIDER_OUTCOME_APPEND_MIGRATION = (
    "130-research-lab-provider-outcome-append.sql"
)
PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION = (
    "131-research-lab-provider-outcome-backpressure.sql"
)
PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION = (
    "133-research-lab-provider-outcome-contention-status.sql"
)
PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION = (
    "134-research-lab-provider-outcome-head-contention.sql"
)
ANCESTRY_CHECKPOINT_MIGRATION = (
    "136-research-lab-ancestry-checkpoint-sidecars.sql"
)
ALLOCATION_SETTLEMENT_FRONTIER_MIGRATION = (
    "137-research-lab-allocation-settlement-frontier.sql"
)
ANCESTRY_CHECKPOINT_BOOTSTRAP_PURPOSE_MIGRATION = (
    "138-research-lab-ancestry-checkpoint-bootstrap-purpose.sql"
)
ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_MIGRATION = (
    "139-research-lab-allocation-frontier-bootstrap.sql"
)
ALLOCATION_SETTLEMENT_FRONTIER_HISTORICAL_SOURCE_MIGRATION = (
    "140-research-lab-allocation-frontier-historical-source.sql"
)
ALLOCATION_SETTLEMENT_FRONTIER_SOURCE_CONTRACT_MIGRATION = (
    "141-research-lab-allocation-frontier-source-contract.sql"
)
SOURCE_CATALOG_RESULT_REPLAY_MIGRATION = (
    "142-research-lab-source-catalog-result-replay.sql"
)
COMPACT_ANCESTRY_CHECKPOINT_MIGRATION = (
    "143-research-lab-compact-ancestry-checkpoints.sql"
)
PROVIDER_PERSISTENCE_BATCH_MIGRATION = (
    "144-research-lab-provider-persistence-batches.sql"
)
SOURCE_ADD_ADMISSION_CONTROL_MIGRATION = (
    "145-research-lab-source-add-admission-control.sql"
)
SOURCE_CATALOG_AUTH_METADATA_MIGRATION = (
    "147-research-lab-source-catalog-auth-metadata.sql"
)
ATOMIC_CREDIT_RESUME_MIGRATION = (
    "148-research-lab-atomic-credit-resume.sql"
)
COMPACT_WEIGHT_SETTLEMENT_AUTHORITY_MIGRATION = (
    "149-research-lab-compact-weight-settlement-authority.sql"
)
ANCESTRY_DISCLOSURE_ROOT_FAST_PATH_MIGRATION = (
    "155-research-lab-ancestry-disclosure-root-fast-path.sql"
)
PRODUCTION_PARITY_READER_MIGRATION = (
    "156-production-parity-readonly-role.sql"
)
SOURCE_ADD_POST_ACCEPT_LEG1_MIGRATION = (
    "169-research-lab-source-add-post-accept-leg1.sql"
)
SOURCE_ADD_PROVIDER_ORIGIN_UNIQUENESS_MIGRATION = (
    "170-research-lab-source-add-provider-origin-uniqueness.sql"
)
SOURCE_ADD_DUPLICATE_PRIVACY_MIGRATION = (
    "171-research-lab-source-add-duplicate-privacy.sql"
)
SOURCE_ADD_CLAIM_CONTROL_MIGRATION = (
    "172-research-lab-source-add-claim-control.sql"
)
SOURCE_ADD_LEG1_RELEASE_POLICY_MIGRATION = (
    "173-research-lab-source-add-leg1-release-policy.sql"
)
SOURCE_ADD_RESTART_STATE_RESTORE_MIGRATION = (
    "174-research-lab-source-add-restart-state-restore.sql"
)
SOURCE_ADD_PROVENANCE_LEG1_MIGRATION = (
    "175-research-lab-source-add-provenance-leg1.sql"
)
SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION = (
    "176-research-lab-source-add-provenance-origin-repair.sql"
)
SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION = (
    "177-research-lab-source-add-provenance-authority-acl.sql"
)
SOURCE_ADD_MINER_STATUS_MIGRATION = (
    "178-research-lab-source-add-miner-status.sql"
)
LAB_ARENA_MIGRATIONS = (
    "179-lab-arena-v1.sql",
    "180-lab-arena-daily-competition.sql",
    "181-lab-arena-source-submissions.sql",
    "182-lab-arena-source-execution.sql",
    "183-lab-arena-miner-reward-basis.sql",
    "184-lab-arena-scoring-failure-isolation.sql",
)
CHAMPION_LIFETIME_CREDIT_MIGRATION = (
    "132-research-lab-champion-lifetime-credit.sql"
)
EXPECTED_APPLIED_MIGRATIONS = (
    EVENT_PROJECTIONS_MIGRATION,
    ALLOCATION_CANDIDATE_MIGRATION,
    ALLOCATION_AUTO_RESEARCH_MIGRATION,
    ALLOCATION_SCHEMA_MIGRATION,
    ALLOCATION_SCORING_AUDIT_MIGRATION,
    ALLOCATION_PROMOTION_MIGRATION,
    ATOMIC_CLAIM_GUARDS_MIGRATION,
    QUEUE_CAPACITY_GUARD_MIGRATION,
    MAINTENANCE_PAUSE_MIGRATION,
    *ALLOCATION_IMAGE_BUILD_MIGRATIONS[:2],
    PAUSED_CAPACITY_AGING_MIGRATION,
    ALLOCATION_IMAGE_BUILD_MIGRATIONS[2],
    RESUME_REQUEUE_HOTKEY_GUARD_MIGRATION,
    HOTKEY_ACTIVE_LOOP_CAP_MIGRATION,
    *SOURCE_ADD_PRE_V2_MIGRATIONS,
    MIGRATIONS_BEFORE_TRANSPORT_FIX[0],
    ALLOCATION_CONTAINMENT_MIGRATION,
    MIGRATIONS_BEFORE_TRANSPORT_FIX[1],
    MIGRATIONS_BEFORE_TRANSPORT_FIX[2],
    GIT_TREE_AUTORESEARCH_MIGRATION,
    SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION,
    MIGRATIONS_BEFORE_TRANSPORT_FIX[3],
    MIGRATIONS_BEFORE_TRANSPORT_FIX[4],
    GIT_TREE_ROOT_REPLACEMENT_MIGRATION,
    *MIGRATIONS_BEFORE_TRANSPORT_FIX[5:],
    TRANSPORT_FIX_MIGRATION,
    TRANSPORT_TERMINAL_MIGRATION,
    PROVIDER_OUTCOME_APPEND_MIGRATION,
    PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION,
    CHAMPION_LIFETIME_CREDIT_MIGRATION,
    PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION,
    PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION,
    ANCESTRY_CHECKPOINT_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_MIGRATION,
    ANCESTRY_CHECKPOINT_BOOTSTRAP_PURPOSE_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_HISTORICAL_SOURCE_MIGRATION,
    ALLOCATION_SETTLEMENT_FRONTIER_SOURCE_CONTRACT_MIGRATION,
    SOURCE_CATALOG_RESULT_REPLAY_MIGRATION,
    COMPACT_ANCESTRY_CHECKPOINT_MIGRATION,
    PROVIDER_PERSISTENCE_BATCH_MIGRATION,
    SOURCE_ADD_ADMISSION_CONTROL_MIGRATION,
    SOURCE_CATALOG_AUTH_METADATA_MIGRATION,
    ATOMIC_CREDIT_RESUME_MIGRATION,
    COMPACT_WEIGHT_SETTLEMENT_AUTHORITY_MIGRATION,
    ANCESTRY_DISCLOSURE_ROOT_FAST_PATH_MIGRATION,
    PRODUCTION_PARITY_READER_MIGRATION,
    SOURCE_ADD_POST_ACCEPT_LEG1_MIGRATION,
    SOURCE_ADD_PROVIDER_ORIGIN_UNIQUENESS_MIGRATION,
    SOURCE_ADD_DUPLICATE_PRIVACY_MIGRATION,
    SOURCE_ADD_CLAIM_CONTROL_MIGRATION,
    SOURCE_ADD_LEG1_RELEASE_POLICY_MIGRATION,
    SOURCE_ADD_RESTART_STATE_RESTORE_MIGRATION,
    SOURCE_ADD_PROVENANCE_LEG1_MIGRATION,
    SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION,
    SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION,
    SOURCE_ADD_MINER_STATUS_MIGRATION,
    *LAB_ARENA_MIGRATIONS,
)
EXPECTED_POSTGRES_CONTRACT_CHECKS = (
    "maintenance_lease_contract_valid",
    "pre_128_transport_rejected",
    "post_128_transport_persisted",
    "transport_contract_valid",
    "pre_129_attested_local_transport_rejected",
    "post_129_attested_local_transport_persisted",
    "transport_terminal_contract_valid",
    "pre_133_provider_outcome_contract_rejected",
    "post_133_provider_outcome_contract_valid",
    "pre_134_provider_outcome_head_contract_rejected",
    "post_134_provider_outcome_head_contract_valid",
    "post_136_ancestry_checkpoint_contract_valid",
    "post_137_allocation_settlement_frontier_contract_valid",
    "post_138_ancestry_checkpoint_bootstrap_purpose_valid",
    "post_139_allocation_frontier_bootstrap_contract_valid",
    "post_141_allocation_frontier_source_contract_valid",
    "post_142_source_catalog_replay_contract_valid",
    "post_143_compact_checkpoint_contract_valid",
    "post_144_provider_persistence_batch_contract_valid",
    "post_096_source_add_functional_workflow_valid",
    "post_145_source_add_admission_control_contract_valid",
    "post_147_source_catalog_auth_metadata_contract_valid",
    "post_148_atomic_credit_resume_contract_valid",
    "post_149_compact_weight_settlement_contract_valid",
    "post_155_ancestry_disclosure_lookup_contract_valid",
    "post_156_production_parity_reader_contract_valid",
    "post_169_source_add_post_accept_leg1_valid",
    "post_170_source_add_provider_origin_contract_valid",
    "post_171_source_add_duplicate_privacy_valid",
    "post_172_source_add_claim_control_valid",
    "post_173_source_add_leg1_release_policy_valid",
    "post_174_source_add_restart_state_restore_valid",
    "post_175_source_add_provenance_leg1_valid",
    "post_176_source_add_provenance_origin_repair_valid",
    "post_178_source_add_miner_status_valid",
    "post_184_lab_arena_schema_valid",
    "credit_resume_identical_replay_idempotent",
    "credit_resume_differing_replay_rejected",
    "credit_resume_invalid_heads_rejected",
    "provider_outcome_append_atomic",
    "provider_outcome_batch_append_atomic",
    "provider_evidence_cache_put_atomic",
    "provider_outcome_contention_zero_rollback",
    "provider_outcome_conflict_head_exact",
    "pre_132_lifetime_credit_rejected",
    "post_132_lifetime_credit_persisted",
    "lifetime_credit_rpc_idempotent",
    "grandfathered_credit_unchanged",
    "lifetime_credit_contract_valid",
    "finalized_view_projection_exact",
    "finalized_view_seed_available",
    "historical_compute_schema_migrations_applied",
    "historical_compute_finalized_authority_seed_available",
    "historical_compute_allocation_conserved",
    "historical_compute_release_identity_bound",
    "settlement_authority_parsed",
    "measured_settlement_receipt_projection_exact",
    "tampered_weight_receipt_rejected",
    "required_schema_migrations_declared",
)
EXPECTED_FINALIZED_VIEW_COLUMNS = (
    "bundle_hash",
    "schema_version",
    "netuid",
    "epoch_id",
    "block",
    "validator_hotkey",
    "root_receipt_hash",
    "weights_hash",
    "snapshot_hash",
    "bundle_doc",
    "weight_submission_event_hash",
    "publication_receipt_hash",
    "transparency_event_hash",
    "durable_readback_hash",
    "publication_doc",
    "weight_finalization_event_hash",
    "finalization_receipt_hash",
    "extrinsic_authorization_hash",
    "extrinsic_hash",
    "finalized_block",
    "finalized_block_hash",
    "state_transition_hash",
    "finalization_doc",
)
EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE = {
    "event_id": "40000000-0000-0000-0000-000000000147",
    "event_hash": "sha256:" + "2" * 64,
    "identical_replay": True,
    "concurrent_replay_serialized": True,
    "differing_replay_rejected": True,
    "invalid_arguments_rejected": True,
    "stale_head_rejected": True,
    "empty_head_rejected": True,
    "wrong_paused_head_rejected": True,
    "rpc_security_contract_valid": True,
    "queue_capacity_guard_exercised": True,
    "hotkey_capacity_guard_exercised": True,
    "row_counts": {
        "resumed_run": 2,
        "empty_run": 0,
        "wrong_paused_run": 1,
        "capacity_closed_run": 1,
        "hotkey_capacity_closed_run": 1,
        "concurrent_run": 2,
    },
}
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
SYSTEM_BINARY_DIRS = tuple(
    Path(value)
    for value in ("/usr/local/sbin", "/usr/sbin", "/sbin", "/usr/bin", "/bin")
)
ALLOCATION_MIGRATION_PREREQUISITES_SQL = """
CREATE SCHEMA extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
CREATE SCHEMA auth;
CREATE FUNCTION auth.role()
RETURNS TEXT
LANGUAGE SQL
STABLE
AS $$ SELECT current_user::TEXT $$;
CREATE TABLE public.research_evaluation_score_bundles (
    score_bundle_id TEXT PRIMARY KEY,
    score_bundle_doc JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.research_loop_tickets (
    ticket_id UUID PRIMARY KEY,
    miner_hotkey TEXT NOT NULL
);
CREATE TABLE public.research_loop_balance_ledger (
    ledger_entry_id UUID PRIMARY KEY,
    miner_hotkey TEXT NOT NULL,
    ticket_id UUID REFERENCES public.research_loop_tickets(ticket_id)
        ON DELETE RESTRICT,
    amount_microusd BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE public.research_weight_input_snapshots (
    weight_input_snapshot_id UUID PRIMARY KEY,
    snapshot_status TEXT NOT NULL CHECK (
        snapshot_status IN ('shadow', 'candidate', 'active', 'tombstoned')
    )
);
CREATE TABLE public.research_loop_receipts (
    receipt_id UUID PRIMARY KEY
);
CREATE TABLE public.research_loop_start_payments (
    payment_id UUID PRIMARY KEY
);
CREATE TABLE public.research_reimbursement_awards (
    award_id TEXT PRIMARY KEY
);
CREATE TABLE public.qualification_private_icp_sets (
    set_id BIGINT PRIMARY KEY,
    icps JSONB NOT NULL,
    icp_set_hash TEXT,
    industry_distribution JSONB,
    active_from TIMESTAMPTZ,
    active_until TIMESTAMPTZ,
    generation_seed TEXT,
    is_active BOOLEAN NOT NULL DEFAULT FALSE
);
ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.qualification_private_icp_sets
    FROM PUBLIC, anon, authenticated;
"""
GIT_TREE_CANDIDATE_PREREQUISITES_SQL = """
ALTER TABLE public.research_lab_candidate_artifacts
    ADD COLUMN git_tree_id TEXT NULL,
    ADD COLUMN git_tree_node_id TEXT NULL,
    ADD COLUMN git_tree_root_commit TEXT NULL,
    ADD COLUMN git_tree_node_commit TEXT NULL,
    ADD COLUMN git_tree_lineage_hash TEXT NULL;
"""


class PostgresContractProbeError(RuntimeError):
    """The candidate migration-backed V2 contract is not production-ready."""


def _sql_without_comments(value: str) -> str:
    value = re.sub(r"/\*.*?\*/", " ", value, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", value)


def _validate_required_migration_declarations(
    source_root: Path,
) -> dict[str, int]:
    documents: dict[str, str] = {}

    def migration_sql(name: str) -> str:
        if name not in documents:
            path = source_root / name
            if not path.is_file():
                raise PostgresContractProbeError(
                    "required migration is missing: %s" % name
                )
            documents[name] = _sql_without_comments(path.read_text(encoding="utf-8"))
        return documents[name]

    for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        sql = migration_sql(migration)
        declaration = re.compile(
            r"\b(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)"
            r"|ALTER\s+TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
            r"(?:public\.)?%s\b" % re.escape(relation),
            flags=re.IGNORECASE,
        )
        declaration_match = declaration.search(sql)
        if declaration_match is None:
            raise PostgresContractProbeError(
                "required relation is not declared migration=%s relation=%s"
                % (migration, relation)
            )
        for column in columns:
            statement_end = sql.find(";", declaration_match.start())
            declaration_statement = sql[
                declaration_match.start() : None if statement_end < 0 else statement_end
            ]
            inherited_view_columns = (
                "VIEW" in declaration_statement.upper()
                and re.search(r"\b[a-z][a-z0-9_]*\s*\.\s*\*", declaration_statement)
                is not None
            )
            if (
                re.search(r"\b%s\b" % re.escape(column), sql) is None
                and not inherited_view_columns
            ):
                raise PostgresContractProbeError(
                    "required column is not declared migration=%s "
                    "relation=%s column=%s" % (migration, relation, column)
                )
    for migration, function_name in REQUIRED_SUPABASE_V2_RPCS:
        sql = migration_sql(migration)
        declaration = re.compile(
            r"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+"
            r"(?:public\.)?%s\s*\(" % re.escape(function_name),
            flags=re.IGNORECASE,
        )
        if declaration.search(sql) is None:
            raise PostgresContractProbeError(
                "required RPC is not declared migration=%s rpc=%s"
                % (migration, function_name)
            )
    return {
        "migration_count": len(documents),
        "relation_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "rpc_probe_count": len(REQUIRED_SUPABASE_V2_RPCS),
    }


class DisposablePostgres:
    def __init__(self, *, state_root: Path):
        self.state_root = state_root
        self.root = Path(tempfile.mkdtemp(prefix="leadpoet-postgres-v2-", dir="/tmp"))
        self.data = self.root / "data"
        self.socket = self.root / "socket"
        self.port = 55432
        self.database = "leadpoet_rehearsal"
        self.started = False
        account = pwd.getpwnam("postgres")
        os.chown(self.root, account.pw_uid, account.pw_gid)
        self.socket.mkdir()
        os.chown(self.socket, account.pw_uid, account.pw_gid)

    @staticmethod
    def _binary(name: str) -> str:
        if not IDENTIFIER_RE.fullmatch(name):
            raise PostgresContractProbeError(
                "postgres binary name is invalid: %s" % name
            )
        candidates = []
        resolved = shutil.which(name)
        if resolved is not None:
            candidates.append(Path(resolved))
        candidates.extend(directory / name for directory in SYSTEM_BINARY_DIRS)
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        raise PostgresContractProbeError("postgres binary is unavailable: %s" % name)

    def _as_postgres(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            self._binary("runuser"),
            "-u",
            "postgres",
            "--",
            *argv,
        ]
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise PostgresContractProbeError(
                "postgres command failed executable=%s stderr=%s"
                % (Path(argv[0]).name, result.stderr.strip())
            )
        return result

    def start(self) -> None:
        self._as_postgres(
            [
                self._binary("initdb"),
                "--pgdata",
                str(self.data),
                "--auth=trust",
                "--no-locale",
                "--encoding=UTF8",
            ]
        )
        self._as_postgres(
            [
                self._binary("pg_ctl"),
                "--pgdata",
                str(self.data),
                "--log",
                str(self.root / "postgres.log"),
                "--options",
                "-k %s -p %d -c listen_addresses=''" % (self.socket, self.port),
                "--wait",
                "start",
            ]
        )
        self.started = True
        self.psql(
            """
            CREATE ROLE anon NOLOGIN;
            CREATE ROLE authenticated NOLOGIN;
            CREATE ROLE service_role NOLOGIN;
            CREATE DATABASE leadpoet_rehearsal;
            """,
            database="postgres",
        )

    def stop(self) -> None:
        if self.started:
            self._as_postgres(
                [
                    self._binary("pg_ctl"),
                    "--pgdata",
                    str(self.data),
                    "--wait",
                    "--mode",
                    "fast",
                    "stop",
                ],
                check=False,
            )
            self.started = False
        shutil.rmtree(self.root, ignore_errors=True)

    def psql(
        self,
        sql: str,
        *,
        database: str | None = None,
        check: bool = True,
        tuples_only: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        argv = [
            self._binary("psql"),
            "--no-psqlrc",
            "--host",
            str(self.socket),
            "--port",
            str(self.port),
            "--username",
            "postgres",
            "--dbname",
            database or self.database,
            "--set",
            "ON_ERROR_STOP=1",
        ]
        if tuples_only:
            argv.extend(["--tuples-only", "--no-align"])
        return self._as_postgres(argv, input_text=sql, check=check)

    def apply_migration(self, path: Path) -> None:
        result = self.psql(path.read_text(encoding="utf-8"), check=False)
        if result.returncode != 0:
            raise PostgresContractProbeError(
                "migration failed path=%s stderr=%s"
                % (path.name, result.stderr.strip())
            )


def _json_insert_sql(table: str, row: Mapping[str, Any]) -> str:
    if not IDENTIFIER_RE.fullmatch(table):
        raise PostgresContractProbeError(
            "fixture table identifier is invalid: %s" % table
        )
    columns = tuple(row)
    if not columns or any(not IDENTIFIER_RE.fullmatch(name) for name in columns):
        raise PostgresContractProbeError(
            "fixture row columns are invalid for %s" % table
        )
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError("fixture JSON delimiter collision")
    selected = ",".join(columns)
    return (
        "INSERT INTO public.%s (%s) "
        "SELECT %s FROM pg_catalog.json_populate_record("
        "NULL::public.%s, $leadpoet$%s$leadpoet$::json);\n"
        % (table, selected, selected, table, payload)
    )


def _deterministic_seed_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(row), "created_at": NOW}


def _postgres_literal(value: Any) -> str:
    """Return one strict literal for the disposable-Postgres adapter."""

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (Mapping, list, tuple)):
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if "$leadpoet$" in encoded:
            raise PostgresContractProbeError("Postgres JSON delimiter collision")
        return "$leadpoet$%s$leadpoet$::jsonb" % encoded
    encoded = str(value)
    if "$leadpoet$" in encoded:
        raise PostgresContractProbeError("Postgres text delimiter collision")
    return "$leadpoet$%s$leadpoet$" % encoded


def _postgres_rpc(
    database: DisposablePostgres,
    function_name: str,
    params: Mapping[str, Any],
) -> Any:
    if not IDENTIFIER_RE.fullmatch(function_name):
        raise PostgresContractProbeError("invalid RPC identifier")
    arguments = []
    for name, value in params.items():
        if not IDENTIFIER_RE.fullmatch(name):
            raise PostgresContractProbeError("invalid RPC parameter identifier")
        arguments.append("%s => %s" % (name, _postgres_literal(value)))
    result = database.psql(
        "SELECT pg_catalog.to_jsonb(public.%s(%s))::text;"
        % (function_name, ",".join(arguments)),
        check=False,
        tuples_only=True,
    )
    if result.returncode != 0:
        raise PostgresContractProbeError(
            "Postgres RPC failed function=%s stderr=%s"
            % (function_name, result.stderr.strip())
        )
    payload = result.stdout.strip()
    return json.loads(payload) if payload else None


def _postgres_select_one(
    database: DisposablePostgres,
    table: str,
    *,
    columns: str,
    filters: Sequence[tuple[Any, ...]],
) -> dict[str, Any] | None:
    if not IDENTIFIER_RE.fullmatch(table):
        raise PostgresContractProbeError("invalid relation identifier")
    if columns.strip() == "*":
        projection = "*"
    else:
        projected = tuple(part.strip() for part in columns.split(","))
        if not projected or any(
            not IDENTIFIER_RE.fullmatch(part) for part in projected
        ):
            raise PostgresContractProbeError("invalid relation projection")
        projection = ",".join(projected)
    predicates = []
    for raw_filter in filters:
        if len(raw_filter) != 2:
            raise PostgresContractProbeError(
                "disposable Postgres adapter only permits equality filters"
            )
        column, value = raw_filter
        if not IDENTIFIER_RE.fullmatch(str(column)):
            raise PostgresContractProbeError("invalid filter identifier")
        predicates.append(
            "%s IS NOT DISTINCT FROM %s" % (column, _postgres_literal(value))
        )
    where = " AND ".join(predicates) if predicates else "TRUE"
    result = database.psql(
        "SELECT pg_catalog.to_jsonb(selected)::text FROM "
        "(SELECT %s FROM public.%s WHERE %s LIMIT 1) AS selected;"
        % (projection, table, where),
        tuples_only=True,
    ).stdout.strip()
    return dict(json.loads(result)) if result else None


def _postgres_insert_row(
    database: DisposablePostgres,
    table: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if not IDENTIFIER_RE.fullmatch(table):
        raise PostgresContractProbeError("invalid insert relation identifier")
    columns = tuple(row)
    if not columns or any(not IDENTIFIER_RE.fullmatch(name) for name in columns):
        raise PostgresContractProbeError("invalid insert columns")
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError("insert JSON delimiter collision")
    selected = ",".join(columns)
    result = database.psql(
        "WITH inserted AS ("
        "INSERT INTO public.%s (%s) "
        "SELECT %s FROM pg_catalog.json_populate_record("
        "NULL::public.%s, $leadpoet$%s$leadpoet$::json) RETURNING *"
        ") SELECT pg_catalog.to_jsonb(inserted)::text FROM inserted;"
        % (table, selected, selected, table, payload),
        tuples_only=True,
    ).stdout.strip()
    if not result:
        raise PostgresContractProbeError("insert returned no row: %s" % table)
    return dict(json.loads(result))


def _credit_resume_rpc(
    database: DisposablePostgres,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    required = (
        "p_run_id",
        "p_ticket_id",
        "p_expected_event_seq",
        "p_expected_event_hash",
        "p_event_id",
        "p_anchored_hash",
        "p_queue_priority",
        "p_worker_ref",
        "p_reason",
        "p_event_doc",
    )
    if set(params) != set(required):
        raise PostgresContractProbeError("credit resume RPC parameters differ")
    arguments = ",".join(
        "%s => %s" % (name, _postgres_literal(params[name]))
        for name in required
    )
    result = database.psql(
        "SELECT pg_catalog.to_jsonb(resumed)::text "
        "FROM public.resume_research_lab_credit_blocked_run_v1(%s) AS resumed;"
        % arguments,
        check=False,
        tuples_only=True,
    )
    if result.returncode != 0:
        raise PostgresContractProbeError(
            "credit resume RPC failed: %s" % result.stderr.strip()
        )
    payload = result.stdout.strip()
    if not payload:
        raise PostgresContractProbeError("credit resume RPC returned no row")
    return dict(json.loads(payload))


def _credit_resume_rejection(
    database: DisposablePostgres,
    params: Mapping[str, Any],
    *,
    expected_error: str,
) -> None:
    required = (
        "p_run_id",
        "p_ticket_id",
        "p_expected_event_seq",
        "p_expected_event_hash",
        "p_event_id",
        "p_anchored_hash",
        "p_queue_priority",
        "p_worker_ref",
        "p_reason",
        "p_event_doc",
    )
    arguments = ",".join(
        "%s => %s" % (name, _postgres_literal(params[name]))
        for name in required
    )
    result = database.psql(
        "SELECT pg_catalog.to_jsonb(resumed)::text "
        "FROM public.resume_research_lab_credit_blocked_run_v1(%s) AS resumed;"
        % arguments,
        check=False,
        tuples_only=True,
    )
    if result.returncode == 0 or expected_error not in result.stderr:
        raise PostgresContractProbeError(
            "credit resume rejection differed expected=%s stderr=%s"
            % (expected_error, result.stderr.strip())
        )


def _atomic_credit_resume_postgres_contract(
    database: DisposablePostgres,
) -> dict[str, Any]:
    rpc_signature = (
        "public.resume_research_lab_credit_blocked_run_v1("
        "uuid,uuid,integer,text,uuid,text,integer,text,text,jsonb)"
    )
    rpc_catalog = json.loads(
        database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'security_definer', p.prosecdef,
                'config', pg_catalog.to_jsonb(p.proconfig),
                'service_role_execute', EXISTS (
                    SELECT 1
                    FROM pg_catalog.aclexplode(
                        COALESCE(
                            p.proacl,
                            pg_catalog.acldefault('f', p.proowner)
                        )
                    ) AS acl
                    JOIN pg_catalog.pg_roles AS role
                      ON role.oid = acl.grantee
                    WHERE role.rolname = 'service_role'
                      AND acl.privilege_type = 'EXECUTE'
                ),
                'anon_execute', EXISTS (
                    SELECT 1
                    FROM pg_catalog.aclexplode(
                        COALESCE(
                            p.proacl,
                            pg_catalog.acldefault('f', p.proowner)
                        )
                    ) AS acl
                    JOIN pg_catalog.pg_roles AS role
                      ON role.oid = acl.grantee
                    WHERE role.rolname = 'anon'
                      AND acl.privilege_type = 'EXECUTE'
                ),
                'authenticated_execute', EXISTS (
                    SELECT 1
                    FROM pg_catalog.aclexplode(
                        COALESCE(
                            p.proacl,
                            pg_catalog.acldefault('f', p.proowner)
                        )
                    ) AS acl
                    JOIN pg_catalog.pg_roles AS role
                      ON role.oid = acl.grantee
                    WHERE role.rolname = 'authenticated'
                      AND acl.privilege_type = 'EXECUTE'
                ),
                'public_execute', EXISTS (
                    SELECT 1
                    FROM pg_catalog.aclexplode(
                        COALESCE(
                            p.proacl,
                            pg_catalog.acldefault('f', p.proowner)
                        )
                    ) AS acl
                    WHERE acl.grantee = 0
                      AND acl.privilege_type = 'EXECUTE'
                )
            )::text
            FROM pg_catalog.pg_proc AS p
            WHERE p.oid = %s::pg_catalog.regprocedure;
            """
            % _postgres_literal(rpc_signature),
            tuples_only=True,
        ).stdout.strip()
    )
    config = rpc_catalog.pop("config", None)
    search_path_values = (
        [
            value.split("=", 1)[1]
            for value in config
            if isinstance(value, str) and value.startswith("search_path=")
        ]
        if isinstance(config, list)
        else []
    )
    if (
        rpc_catalog
        != {
            "security_definer": True,
            "service_role_execute": True,
            "anon_execute": False,
            "authenticated_execute": False,
            "public_execute": False,
        }
        or search_path_values not in ([""], ['""'])
    ):
        raise PostgresContractProbeError(
            "atomic credit resume RPC security contract differs"
        )
    ticket_id = "20000000-0000-0000-0000-000000000147"
    capacity_ticket_id = "20000000-0000-0000-0000-000000000148"
    run_id = "10000000-0000-0000-0000-000000000147"
    empty_run_id = "10000000-0000-0000-0000-000000000148"
    wrong_paused_run_id = "10000000-0000-0000-0000-000000000149"
    capacity_closed_run_id = "10000000-0000-0000-0000-000000000150"
    hotkey_capacity_closed_run_id = "10000000-0000-0000-0000-000000000151"
    concurrent_run_id = "10000000-0000-0000-0000-000000000152"
    paused_hash = "sha256:" + "1" * 64
    resumed_hash = "sha256:" + "2" * 64
    wrong_paused_hash = "sha256:" + "7" * 64
    capacity_closed_hash = "sha256:" + "9" * 64
    hotkey_capacity_closed_hash = "sha256:" + "b" * 64
    concurrent_paused_hash = "sha256:" + "d" * 64
    concurrent_resumed_hash = "sha256:" + "e" * 64
    _postgres_insert_row(
        database,
        "research_loop_tickets",
        {
            "ticket_id": ticket_id,
            "miner_hotkey": "5F-rehearsal-credit-resume-miner",
        },
    )
    _postgres_insert_row(
        database,
        "research_loop_run_queue_events",
        {
            "event_id": "30000000-0000-0000-0000-000000000147",
            "schema_version": "1.0",
            "run_id": run_id,
            "ticket_id": ticket_id,
            "seq": 4,
            "event_type": "paused",
            "queue_priority": 3,
            "worker_ref": "worker:credit-blocked",
            "reason": "blocked_for_credit",
            "anchored_hash": paused_hash,
            "event_doc": {"schema_version": "1.0"},
        },
    )
    common = {
        "p_run_id": run_id,
        "p_ticket_id": ticket_id,
        "p_expected_event_seq": 4,
        "p_expected_event_hash": paused_hash,
        "p_event_id": "40000000-0000-0000-0000-000000000147",
        "p_anchored_hash": resumed_hash,
        "p_queue_priority": 3,
        "p_worker_ref": "miner:credit-topup",
        "p_reason": "credit_topup_resume",
        "p_event_doc": {
            "schema_version": "1.0",
            "autoresearch_capacity_policy": "proxy_worker_capacity:v1",
            "autoresearch_capacity": 1,
            "autoresearch_hotkey_capacity": 1,
            "active_loop_stale_after_seconds": 300,
            "resume_source": "miner_credit_topup_resume",
            "previous_event_hash": paused_hash,
        },
    }
    first = _credit_resume_rpc(database, common)
    replay = _credit_resume_rpc(database, common)
    if (
        first != replay
        or first.get("event_id") != common["p_event_id"]
        or first.get("run_id") != run_id
        or first.get("ticket_id") != ticket_id
        or first.get("seq") != 5
        or first.get("event_type") != "queued"
        or first.get("reason") != "credit_topup_resume"
        or first.get("anchored_hash") != resumed_hash
        or first.get("event_doc") != common["p_event_doc"]
    ):
        raise PostgresContractProbeError(
            "credit resume append or identical replay differed"
        )

    _credit_resume_rejection(
        database,
        {**common, "p_anchored_hash": "sha256:" + "3" * 64},
        expected_error="research_lab_credit_resume_replay_differs",
    )
    invalid_arguments = {
        **common,
        "p_event_doc": {
            **common["p_event_doc"],
            "resume_source": "noncanonical_resume",
        },
    }
    _credit_resume_rejection(
        database,
        invalid_arguments,
        expected_error="research_lab_credit_resume_invalid_arguments",
    )
    _credit_resume_rejection(
        database,
        {
            **common,
            "p_event_id": "40000000-0000-0000-0000-000000000148",
            "p_anchored_hash": "sha256:" + "4" * 64,
        },
        expected_error="research_lab_credit_resume_head_conflict",
    )
    empty = {
        **common,
        "p_run_id": empty_run_id,
        "p_expected_event_seq": 0,
        "p_expected_event_hash": "sha256:" + "5" * 64,
        "p_event_id": "40000000-0000-0000-0000-000000000149",
        "p_anchored_hash": "sha256:" + "6" * 64,
    }
    empty["p_event_doc"] = {
        **common["p_event_doc"],
        "previous_event_hash": empty["p_expected_event_hash"],
    }
    _credit_resume_rejection(
        database,
        empty,
        expected_error="research_lab_credit_resume_head_conflict",
    )
    _postgres_insert_row(
        database,
        "research_loop_tickets",
        {
            "ticket_id": capacity_ticket_id,
            "miner_hotkey": "5F-rehearsal-capacity-other-miner",
        },
    )
    _postgres_insert_row(
        database,
        "research_loop_run_queue_events",
        {
            "event_id": "30000000-0000-0000-0000-000000000149",
            "schema_version": "1.0",
            "run_id": wrong_paused_run_id,
            "ticket_id": ticket_id,
            "seq": 0,
            "event_type": "paused",
            "queue_priority": 3,
            "worker_ref": "worker:maintenance",
            "reason": "maintenance_pause",
            "anchored_hash": wrong_paused_hash,
            "event_doc": {"schema_version": "1.0"},
        },
    )
    wrong_paused = {
        **common,
        "p_run_id": wrong_paused_run_id,
        "p_expected_event_seq": 0,
        "p_expected_event_hash": wrong_paused_hash,
        "p_event_id": "40000000-0000-0000-0000-000000000150",
        "p_anchored_hash": "sha256:" + "8" * 64,
    }
    wrong_paused["p_event_doc"] = {
        **common["p_event_doc"],
        "previous_event_hash": wrong_paused_hash,
    }
    _credit_resume_rejection(
        database,
        wrong_paused,
        expected_error="research_lab_credit_resume_head_conflict",
    )
    _postgres_insert_row(
        database,
        "research_loop_run_queue_events",
        {
            "event_id": "30000000-0000-0000-0000-000000000150",
            "schema_version": "1.0",
            "run_id": capacity_closed_run_id,
            "ticket_id": capacity_ticket_id,
            "seq": 0,
            "event_type": "paused",
            "queue_priority": 3,
            "worker_ref": "worker:credit-blocked",
            "reason": "blocked_for_credit",
            "anchored_hash": capacity_closed_hash,
            "event_doc": {"schema_version": "1.0"},
        },
    )
    capacity_closed = {
        **common,
        "p_run_id": capacity_closed_run_id,
        "p_ticket_id": capacity_ticket_id,
        "p_expected_event_seq": 0,
        "p_expected_event_hash": capacity_closed_hash,
        "p_event_id": "40000000-0000-0000-0000-000000000151",
        "p_anchored_hash": "sha256:" + "a" * 64,
        "p_event_doc": {
            **common["p_event_doc"],
            "autoresearch_capacity": 1,
            "autoresearch_hotkey_capacity": 10,
            "previous_event_hash": capacity_closed_hash,
        },
    }
    _credit_resume_rejection(
        database,
        capacity_closed,
        expected_error="research_lab_queue_capacity_conflict",
    )
    _postgres_insert_row(
        database,
        "research_loop_run_queue_events",
        {
            "event_id": "30000000-0000-0000-0000-000000000151",
            "schema_version": "1.0",
            "run_id": hotkey_capacity_closed_run_id,
            "ticket_id": ticket_id,
            "seq": 0,
            "event_type": "paused",
            "queue_priority": 3,
            "worker_ref": "worker:credit-blocked",
            "reason": "blocked_for_credit",
            "anchored_hash": hotkey_capacity_closed_hash,
            "event_doc": {"schema_version": "1.0"},
        },
    )
    hotkey_capacity_closed = {
        **common,
        "p_run_id": hotkey_capacity_closed_run_id,
        "p_expected_event_seq": 0,
        "p_expected_event_hash": hotkey_capacity_closed_hash,
        "p_event_id": "40000000-0000-0000-0000-000000000152",
        "p_anchored_hash": "sha256:" + "c" * 64,
        "p_event_doc": {
            **common["p_event_doc"],
            "autoresearch_capacity": 10,
            "autoresearch_hotkey_capacity": 1,
            "previous_event_hash": hotkey_capacity_closed_hash,
        },
    }
    _credit_resume_rejection(
        database,
        hotkey_capacity_closed,
        expected_error="research_lab_queue_hotkey_conflict",
    )
    _postgres_insert_row(
        database,
        "research_loop_run_queue_events",
        {
            "event_id": "30000000-0000-0000-0000-000000000152",
            "schema_version": "1.0",
            "run_id": concurrent_run_id,
            "ticket_id": ticket_id,
            "seq": 0,
            "event_type": "paused",
            "queue_priority": 3,
            "worker_ref": "worker:credit-blocked",
            "reason": "blocked_for_credit",
            "anchored_hash": concurrent_paused_hash,
            "event_doc": {"schema_version": "1.0"},
        },
    )
    concurrent_params = {
        **common,
        "p_run_id": concurrent_run_id,
        "p_expected_event_seq": 0,
        "p_expected_event_hash": concurrent_paused_hash,
        "p_event_id": "40000000-0000-0000-0000-000000000153",
        "p_anchored_hash": concurrent_resumed_hash,
        "p_event_doc": {
            **common["p_event_doc"],
            "autoresearch_capacity": 10,
            "autoresearch_hotkey_capacity": 2,
            "previous_event_hash": concurrent_paused_hash,
        },
    }
    concurrent_arguments = ",".join(
        "%s => %s" % (name, _postgres_literal(concurrent_params[name]))
        for name in (
            "p_run_id",
            "p_ticket_id",
            "p_expected_event_seq",
            "p_expected_event_hash",
            "p_event_id",
            "p_anchored_hash",
            "p_queue_priority",
            "p_worker_ref",
            "p_reason",
            "p_event_doc",
        )
    )

    def delayed_concurrent_resume() -> dict[str, Any]:
        result = database.psql(
            "BEGIN; "
            "SELECT pg_catalog.to_jsonb(resumed)::text "
            "FROM public.resume_research_lab_credit_blocked_run_v1(%s) "
            "AS resumed; "
            "SELECT pg_catalog.pg_sleep(0.75); COMMIT;"
            % concurrent_arguments,
            check=False,
            tuples_only=True,
        )
        if result.returncode != 0:
            raise PostgresContractProbeError(
                "delayed concurrent credit resume failed: %s"
                % result.stderr.strip()
            )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return dict(json.loads(line))
        raise PostgresContractProbeError(
            "delayed concurrent credit resume returned no row"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(delayed_concurrent_resume)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            lock_observed = database.psql(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_locks
                    WHERE locktype = 'advisory'
                      AND granted
                );
                """,
                tuples_only=True,
            ).stdout.strip()
            if lock_observed == "t":
                break
            if first_future.done():
                first_future.result()
                raise PostgresContractProbeError(
                    "concurrent credit resume lock was not observable"
                )
            time.sleep(0.01)
        else:
            raise PostgresContractProbeError(
                "concurrent credit resume lock observation timed out"
            )
        second_future = executor.submit(
            _credit_resume_rpc,
            database,
            concurrent_params,
        )
        concurrent_first = first_future.result(timeout=10)
        concurrent_second = second_future.result(timeout=10)
    if (
        concurrent_first != concurrent_second
        or concurrent_first.get("event_id") != concurrent_params["p_event_id"]
        or concurrent_first.get("anchored_hash")
        != concurrent_params["p_anchored_hash"]
    ):
        raise PostgresContractProbeError(
            "concurrent credit resume replay was not exactly serialized"
        )
    capacity_trigger_enabled = database.psql(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_trigger
            WHERE tgrelid =
                  'public.research_loop_run_queue_events'::pg_catalog.regclass
              AND tgname = 'guard_research_loop_queue_capacity_insert'
              AND tgenabled <> 'D'
              AND NOT tgisinternal
        );
        """,
        tuples_only=True,
    ).stdout.strip()
    if capacity_trigger_enabled != "t":
        raise PostgresContractProbeError(
            "production queue-capacity trigger was not exercised"
        )
    capacity_guard_definition = database.psql(
        """
        SELECT pg_catalog.pg_get_functiondef(
            'public.guard_research_lab_queue_capacity()'::pg_catalog.regprocedure
        );
        """,
        tuples_only=True,
    ).stdout
    if not all(
        marker in capacity_guard_definition
        for marker in (
            "hotkey_capacity_text",
            "same_hotkey_count >= hotkey_capacity",
        )
    ):
        raise PostgresContractProbeError(
            "production queue-capacity guard is not the migration-67 definition"
        )
    counts = json.loads(
        database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'resumed_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000147'::uuid
                ),
                'empty_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000148'::uuid
                ),
                'wrong_paused_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000149'::uuid
                ),
                'capacity_closed_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000150'::uuid
                ),
                'hotkey_capacity_closed_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000151'::uuid
                ),
                'concurrent_run', pg_catalog.count(*) FILTER (
                    WHERE run_id = '10000000-0000-0000-0000-000000000152'::uuid
                )
            )::text
            FROM public.research_loop_run_queue_events;
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    if counts != {
        "resumed_run": 2,
        "empty_run": 0,
        "wrong_paused_run": 1,
        "capacity_closed_run": 1,
        "hotkey_capacity_closed_run": 1,
        "concurrent_run": 2,
    }:
        raise PostgresContractProbeError(
            "credit resume rejection persisted an extra row"
        )
    evidence = {
        "event_id": str(first["event_id"]),
        "event_hash": str(first["anchored_hash"]),
        "identical_replay": True,
        "concurrent_replay_serialized": True,
        "differing_replay_rejected": True,
        "invalid_arguments_rejected": True,
        "stale_head_rejected": True,
        "empty_head_rejected": True,
        "wrong_paused_head_rejected": True,
        "rpc_security_contract_valid": True,
        "queue_capacity_guard_exercised": True,
        "hotkey_capacity_guard_exercised": True,
        "row_counts": counts,
    }
    if evidence != EXPECTED_ATOMIC_CREDIT_RESUME_EVIDENCE:
        raise PostgresContractProbeError(
            "atomic credit resume evidence differs from the release contract"
        )
    return evidence


def _provider_outcome_append_sql(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError(
            "provider outcome checkpoint JSON delimiter collision"
        )
    return (
        "SELECT public.append_research_lab_provider_outcome_checkpoint_v2("
        "$leadpoet$%s$leadpoet$::jsonb)::text;\n" % payload
    )


def _provider_outcome_batch_append_sql(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError(
            "provider outcome checkpoint batch JSON delimiter collision"
        )
    return (
        "SELECT public.append_research_lab_provider_outcome_checkpoints_v2("
        "$leadpoet$%s$leadpoet$::jsonb)::text;\n" % payload
    )


def _provider_cache_put_sql(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in payload:
        raise PostgresContractProbeError(
            "provider evidence cache JSON delimiter collision"
        )
    return (
        "SELECT public.put_research_lab_provider_evidence_cache_v2("
        "$leadpoet$%s$leadpoet$::jsonb)::text;\n" % payload
    )


def _provider_persistence_batch_contract(
    database: DisposablePostgres,
) -> dict[str, Any]:
    def checkpoint_row(
        sequence: int,
        checkpoint_hash: str,
        previous_hash: str,
        suffix: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.provider_outcome_checkpoint_row.v2",
            "artifact_master_key_ref_hash": "sha256:" + "8" * 64,
            "utc_day": "2026-07-11",
            "sequence": sequence,
            "checkpoint_hash": checkpoint_hash,
            "previous_checkpoint_hash": previous_hash,
            "state_document_hash": sha256_json(
                {"provider_persistence_batch_state": suffix}
            ),
            "checkpoint_artifact_id": sha256_json(
                {"provider_persistence_batch_artifact": sequence}
            ),
            "encrypted_checkpoint_doc": {
                "schema_version": "leadpoet.encrypted_artifact.v2",
                "fixture": "batch-%d" % sequence,
            },
        }

    batch = []
    previous = ""
    for sequence, suffix in enumerate(("1", "2", "3", "4", "5"), start=1):
        checkpoint_hash = sha256_json(
            {"provider_persistence_batch_checkpoint": sequence}
        )
        batch.append(
            checkpoint_row(sequence, checkpoint_hash, previous, suffix)
        )
        previous = checkpoint_hash
    inserted = json.loads(
        database.psql(
            _provider_outcome_batch_append_sql(batch),
            tuples_only=True,
        ).stdout.strip()
    )
    expected_inserted = {
        "status": "inserted",
        "checkpoint_hash": batch[-1]["checkpoint_hash"],
        "checkpoint_count": len(batch),
    }
    if inserted != expected_inserted:
        raise PostgresContractProbeError(
            "provider outcome batch insert result differs"
        )
    replayed = json.loads(
        database.psql(
            _provider_outcome_batch_append_sql(batch),
            tuples_only=True,
        ).stdout.strip()
    )
    if replayed != {**expected_inserted, "status": "existing"}:
        raise PostgresContractProbeError(
            "provider outcome batch replay result differs"
        )
    durable_count = int(
        database.psql(
            """
            SELECT pg_catalog.count(*)
            FROM public.research_lab_provider_outcome_checkpoints_v2
            WHERE artifact_master_key_ref_hash =
                  'sha256:8888888888888888888888888888888888888888888888888888888888888888'
              AND utc_day = DATE '2026-07-11';
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    if durable_count != len(batch):
        raise PostgresContractProbeError(
            "provider outcome batch durable row count differs"
        )

    stale = [
        checkpoint_row(
            7,
            sha256_json({"provider_persistence_batch_checkpoint": 7}),
            batch[-1]["checkpoint_hash"],
            "6",
        )
    ]
    conflict = json.loads(
        database.psql(
            _provider_outcome_batch_append_sql(stale),
            tuples_only=True,
        ).stdout.strip()
    )
    if (
        conflict.get("status") != "conflict"
        or conflict.get("checkpoint_hash") != stale[-1]["checkpoint_hash"]
        or conflict.get("checkpoint_count") != 1
        or conflict.get("head_checkpoint_row") != batch[-1]
    ):
        raise PostgresContractProbeError(
            "provider outcome batch conflict head differs"
        )

    encrypted_cache_doc = {
        "schema_version": "leadpoet.encrypted_artifact.v2",
        "artifact_id": "sha256:" + "a" * 64,
        "plaintext_hash": "sha256:" + "b" * 64,
        "ciphertext_hash": "sha256:" + "c" * 64,
        "nonce_b64": "bm9uY2U=",
        "aad_b64": "YWFk",
        "encryption_context_hash": "sha256:" + "d" * 64,
        "ciphertext_b64": "Y2lwaGVydGV4dA==",
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2026-08-10T12:00:00Z",
    }
    cache_row = {
        "schema_version": "leadpoet.provider_evidence_cache_row.v2",
        "artifact_master_key_ref_hash": "sha256:" + "e" * 64,
        "utc_day": "2026-07-11",
        "request_fingerprint": "f" * 64,
        "cache_entry_hash": "sha256:" + "0" * 64,
        "cache_artifact_id": encrypted_cache_doc["artifact_id"],
        "source_record_hash": "sha256:" + "1" * 64,
        "source_boot_identity_hash": "sha256:" + "2" * 64,
        "response_body_hash": "sha256:" + "3" * 64,
        "encrypted_cache_doc": encrypted_cache_doc,
    }
    cache_inserted = json.loads(
        database.psql(
            _provider_cache_put_sql(cache_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if cache_inserted != {
        "status": "inserted",
        "cache_entry_hash": cache_row["cache_entry_hash"],
        "cache_row": cache_row,
    }:
        raise PostgresContractProbeError(
            "provider cache atomic put result differs"
        )
    cache_replayed = json.loads(
        database.psql(
            _provider_cache_put_sql(cache_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if cache_replayed != {**cache_inserted, "status": "existing"}:
        raise PostgresContractProbeError(
            "provider cache atomic replay result differs"
        )

    schema = json.loads(
        database.psql(
            "SELECT public.research_lab_provider_persistence_batch_contract_v1()::text;",
            tuples_only=True,
        ).stdout.strip()
    )
    if schema != {
        "schema_version": "leadpoet.provider_persistence_batch_contract.v1",
        "cache_put": "atomic_exact_row",
        "outcome_append": "atomic_contiguous_batch",
        "outcome_batch_max": 32,
        "conflict_head_checkpoint_row": "encrypted_or_null",
    }:
        raise PostgresContractProbeError(
            "provider persistence batch schema differs"
        )
    return {
        "batch_size": len(batch),
        "durable_count": durable_count,
        "batch_replay_exact": True,
        "batch_conflict_head_exact": True,
        "cache_put_exact": True,
        "cache_replay_exact": True,
        "schema": schema,
    }


def _provider_outcome_append_contract(
    database: DisposablePostgres,
) -> dict[str, Any]:
    key_hash = "sha256:" + "a" * 64

    def rollback_count() -> int:
        database.psql("SELECT pg_catalog.pg_stat_clear_snapshot();")
        return int(
            database.psql(
                """
                SELECT xact_rollback
                FROM pg_catalog.pg_stat_database
                WHERE datname = pg_catalog.current_database();
                """,
                tuples_only=True,
            ).stdout.strip()
        )

    def row(
        *,
        sequence: int,
        checkpoint_hash: str,
        previous_checkpoint_hash: str,
        suffix: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": "leadpoet.provider_outcome_checkpoint_row.v2",
            "artifact_master_key_ref_hash": key_hash,
            "utc_day": "2026-07-10",
            "sequence": sequence,
            "checkpoint_hash": checkpoint_hash,
            "previous_checkpoint_hash": previous_checkpoint_hash,
            "state_document_hash": "sha256:" + suffix * 64,
            "checkpoint_artifact_id": "sha256:" + suffix.upper().lower() * 64,
            "encrypted_checkpoint_doc": {
                "schema_version": "leadpoet.encrypted_artifact.v2",
                "fixture": suffix,
            },
        }

    first_hash = "sha256:" + "b" * 64
    first = row(
        sequence=1,
        checkpoint_hash=first_hash,
        previous_checkpoint_hash="",
        suffix="c",
    )
    inserted = json.loads(
        database.psql(
            _provider_outcome_append_sql(first),
            tuples_only=True,
        ).stdout.strip()
    )
    if inserted != {"status": "inserted", "checkpoint_hash": first_hash}:
        raise PostgresContractProbeError(
            "provider outcome first append result differs"
        )
    existing = json.loads(
        database.psql(
            _provider_outcome_append_sql(first),
            tuples_only=True,
        ).stdout.strip()
    )
    if existing != {"status": "existing", "checkpoint_hash": first_hash}:
        raise PostgresContractProbeError(
            "provider outcome idempotent append result differs"
        )
    rollback_count_before_contention = rollback_count()

    siblings = (
        row(
            sequence=2,
            checkpoint_hash="sha256:" + "d" * 64,
            previous_checkpoint_hash=first_hash,
            suffix="e",
        ),
        row(
            sequence=2,
            checkpoint_hash="sha256:" + "f" * 64,
            previous_checkpoint_hash=first_hash,
            suffix="1",
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda value: database.psql(
                    _provider_outcome_append_sql(value),
                    check=False,
                    tuples_only=True,
                ),
                siblings,
            )
        )
    if any(result.returncode != 0 for result in results):
        raise PostgresContractProbeError(
            "provider outcome expected contention surfaced as a SQL error"
        )
    outcomes = [json.loads(result.stdout.strip()) for result in results]
    accepted = [
        outcome for outcome in outcomes if outcome.get("status") == "inserted"
    ]
    rejected = [
        outcome
        for outcome in outcomes
        if outcome.get("status") in {"busy", "conflict"}
    ]
    if len(accepted) != 1 or len(rejected) != 1:
        raise PostgresContractProbeError(
            "provider outcome concurrent append did not select one head"
        )
    accepted_hash = accepted[0].get("checkpoint_hash")
    accepted_row = next(
        (
            dict(candidate)
            for candidate in siblings
            if candidate["checkpoint_hash"] == accepted_hash
        ),
        None,
    )
    if accepted_row is None:
        raise PostgresContractProbeError(
            "provider outcome append accepted an unknown candidate"
        )
    rejected_outcome = rejected[0]
    if set(rejected_outcome) not in (
        {"status", "checkpoint_hash"},
        {"status", "checkpoint_hash", "head_checkpoint_row"},
    ):
        raise PostgresContractProbeError(
            "provider outcome contention response fields differ"
        )
    rejected_hash = rejected_outcome.get("checkpoint_hash")
    if (
        rejected_hash not in {candidate["checkpoint_hash"] for candidate in siblings}
        or rejected_hash == accepted_hash
    ):
        raise PostgresContractProbeError(
            "provider outcome contention response lost candidate identity"
        )
    if rejected_outcome["status"] == "busy":
        if set(rejected_outcome) != {"status", "checkpoint_hash"}:
            raise PostgresContractProbeError(
                "provider outcome busy response fields differ"
            )
    elif (
        set(rejected_outcome)
        != {"status", "checkpoint_hash", "head_checkpoint_row"}
        or rejected_outcome.get("head_checkpoint_row") != accepted_row
    ):
        raise PostgresContractProbeError(
            "provider outcome concurrent conflict omitted its durable head"
        )
    row_count = int(
        database.psql(
            """
            SELECT pg_catalog.count(*)
            FROM public.research_lab_provider_outcome_checkpoints_v2
            WHERE artifact_master_key_ref_hash =
                  'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
              AND utc_day = DATE '2026-07-10';
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    if row_count != 2:
        raise PostgresContractProbeError(
            "provider outcome lineage contains an unexpected row count"
        )

    stale_row = next(
        dict(candidate)
        for candidate in siblings
        if candidate["checkpoint_hash"] != accepted_hash
    )
    stale = json.loads(
        database.psql(
            _provider_outcome_append_sql(stale_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if stale != {
        "status": "conflict",
        "checkpoint_hash": stale_row["checkpoint_hash"],
        "head_checkpoint_row": accepted_row,
    }:
        raise PostgresContractProbeError(
            "provider outcome stale append did not return the exact durable head"
        )

    empty_conflict_row = row(
        sequence=2,
        checkpoint_hash="sha256:" + "4" * 64,
        previous_checkpoint_hash="sha256:" + "5" * 64,
        suffix="6",
    )
    empty_conflict_row["artifact_master_key_ref_hash"] = "sha256:" + "9" * 64
    empty_conflict = json.loads(
        database.psql(
            _provider_outcome_append_sql(empty_conflict_row),
            tuples_only=True,
        ).stdout.strip()
    )
    if empty_conflict != {
        "status": "conflict",
        "checkpoint_hash": empty_conflict_row["checkpoint_hash"],
        "head_checkpoint_row": None,
    }:
        raise PostgresContractProbeError(
            "provider outcome empty-lineage conflict response differs"
        )

    third_hash = "sha256:" + "2" * 64
    third = row(
        sequence=3,
        checkpoint_hash=third_hash,
        previous_checkpoint_hash=accepted_hash,
        suffix="3",
    )
    lock_sql = """
        BEGIN;
        SELECT pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtext('research_lab_provider_outcome_checkpoint_v2'),
            pg_catalog.hashtext(
                'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                || ':2026-07-10'
            )
        );
        SELECT pg_catalog.pg_sleep(2);
        COMMIT;
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        holder = executor.submit(database.psql, lock_sql)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            granted = int(
                database.psql(
                    """
                    SELECT pg_catalog.count(*)
                    FROM pg_catalog.pg_locks
                    WHERE locktype = 'advisory' AND granted;
                    """,
                    tuples_only=True,
                ).stdout.strip()
            )
            if granted:
                break
            time.sleep(0.02)
        else:
            raise PostgresContractProbeError(
                "provider outcome contention fixture did not acquire its lock"
            )
        started = time.monotonic()
        busy = database.psql(
            _provider_outcome_append_sql(third),
            check=False,
            tuples_only=True,
        )
        busy_elapsed = time.monotonic() - started
        busy_result = (
            json.loads(busy.stdout.strip())
            if busy.returncode == 0 and busy.stdout.strip()
            else {}
        )
        if busy.returncode != 0 or busy_result != {
            "status": "busy",
            "checkpoint_hash": third_hash,
        }:
            raise PostgresContractProbeError(
                "provider outcome contention did not return the busy contract"
            )
        if busy_elapsed >= 1.0:
            raise PostgresContractProbeError(
                "provider outcome contention occupied a database session"
            )
        holder.result(timeout=3.0)

    third_inserted = json.loads(
        database.psql(
            _provider_outcome_append_sql(third),
            tuples_only=True,
        ).stdout.strip()
    )
    if third_inserted != {
        "status": "inserted",
        "checkpoint_hash": third_hash,
    }:
        raise PostgresContractProbeError(
            "provider outcome append did not recover after contention"
        )
    rollback_count_after_contention = rollback_count()
    if rollback_count_after_contention != rollback_count_before_contention:
        raise PostgresContractProbeError(
            "provider outcome expected contention rolled back a transaction"
        )
    return {
        "first_checkpoint_hash": first_hash,
        "candidate_sibling_hashes": sorted(
            item["checkpoint_hash"] for item in siblings
        ),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "row_count": row_count + 1,
        "contention_rollback_delta": (
            rollback_count_after_contention
            - rollback_count_before_contention
        ),
        "durable_head_conflict_verified": True,
        "empty_head_conflict_verified": True,
    }


def _settlement_fixture(
    *,
    candidate_sha: str,
    epoch_id: int,
    source_root: Path | None = None,
) -> tuple[
    list[tuple[str, dict[str, Any]]],
    dict[str, Any],
    SanitizedWeightFixture,
]:
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    bundle = fixture.bundle()
    verified = validate_published_weight_bundle_v2(bundle)
    coordinator_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    weight_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    bundle_row = {
        "bundle_hash": verified["bundle_hash"],
        "schema_version": bundle["schema_version"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "block": verified["block"],
        "validator_hotkey": verified["validator_hotkey"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "weights_hash": verified["weights_hash"],
        "snapshot_hash": verified["snapshot_hash"],
        "bundle_doc": bundle,
    }
    durable_readback_hash = sha256_json(bundle_row)
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified["bundle_hash"],
        "root_receipt_hash": verified["root_receipt_hash"],
        "durable_readback_hash": durable_readback_hash,
        "transparency_event_hash": sha256_json(
            {"kind": "transparency", "epoch_id": epoch_id}
        ),
    }
    publication_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="gateway.weights.publication.v2",
        job_id="postgres-contract-publication",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json({"kind": "publication", "epoch_id": epoch_id}),
        output_root=sha256_json(publication_doc),
        parents=[verified["root_receipt_hash"]],
        sequence=800,
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": verified["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": durable_readback_hash,
        }
    )
    publication_row = {
        "weight_submission_event_hash": submission_event_hash,
        "bundle_hash": verified["bundle_hash"],
        "publication_receipt_hash": publication_receipt["receipt_hash"],
        "transparency_event_hash": publication_doc["transparency_event_hash"],
        "durable_readback_hash": durable_readback_hash,
        "publication_doc": publication_doc,
    }
    candidate_root = source_root
    if candidate_root is None:
        module_path = Path(__file__).resolve()
        if len(module_path.parents) < 3:
            raise PostgresContractProbeError(
                "candidate source root is required outside the repository"
            )
        local_candidate = module_path.parents[2]
        if not (
            local_candidate
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).is_file():
            raise PostgresContractProbeError(
                "candidate source root is required outside the repository"
            )
        candidate_root = local_candidate
    profile_manifest = json.loads(
        (
            candidate_root
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile = next(
        item
        for item in chain_signing_profiles(profile_manifest)
        if int(item["spec_version"])
        == int(profile_manifest["spec_version"])
    )
    block = int(verified["block"])
    authorization = build_weight_extrinsic_authorization_v2(
        profile=profile,
        validator_hotkey=verified["validator_hotkey"],
        hotkey_public_key_hex=hashlib.sha256(
            b"postgres-contract-hotkey:" + candidate_sha.encode("ascii")
        ).hexdigest(),
        epoch_id=verified["epoch_id"],
        netuid=verified["netuid"],
        subnet_epoch_index=verified["epoch_id"],
        weight_receipt_hash=verified["weight_receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash=verified["weights_hash"],
        sparse_uids=bundle["weight_result"]["sparse_uids"],
        sparse_weights_u16=bundle["weight_result"]["sparse_weights_u16"],
        commitment=hashlib.sha512(
            b"postgres-contract-commitment:"
            + str(epoch_id).encode("ascii")
        ).digest(),
        reveal_round=epoch_id + 1,
        era_current=block,
        nonce=epoch_id,
        block_hash=hashlib.sha256(
            f"postgres-contract-block:{block}".encode("ascii")
        ).hexdigest(),
    )
    extrinsic_signature = hashlib.sha512(
        b"postgres-contract-signature:"
        + authorization["authorization_hash"].encode("ascii")
    ).hexdigest()
    extrinsic_hash = "0x" + hashlib.sha256(
        b"postgres-contract-extrinsic:"
        + authorization["authorization_hash"].encode("ascii")
    ).hexdigest()
    extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": verified["validator_hotkey"],
        "signature": extrinsic_signature,
        "extrinsic_hash": extrinsic_hash,
    }
    extrinsic_receipt = fixture.receipt(
        role="validator_weights",
        purpose="validator.set_weights_extrinsic.v2",
        job_id=f"postgres-contract-extrinsic-{epoch_id}",
        key=fixture.weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[verified["weight_receipt_hash"]],
        sequence=801,
    )
    finalization_job_id = f"postgres-contract-finalization-{epoch_id}"
    finalization_attempts = [
        fixture.source_attempt(
            category="weight-finalization",
            job_id=finalization_job_id,
            purpose="validator.weights.finalized.v2",
            sequence=900,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="GET",
        ),
        fixture.source_attempt(
            category="weight-finalization-archive",
            job_id=finalization_job_id,
            purpose="validator.weights.finalized.v2",
            sequence=901,
            provider_id="bittensor_archive",
            host="archive.chain.opentensor.ai",
            method="GET",
        ),
    ]
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": verified["validator_hotkey"],
        "netuid": verified["netuid"],
        "epoch_id": verified["epoch_id"],
        "weights_hash": verified["weights_hash"],
        "weight_receipt_hash": verified["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": extrinsic_signature,
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": block + 1,
        "finalized_block_hash": sha256_json(
            {"kind": "finalized-block", "epoch_id": epoch_id}
        )[7:],
        "state_transition_hash": sha256_json(
            {"kind": "state-transition", "epoch_id": epoch_id}
        ),
    }
    finalization_receipt = fixture.receipt(
        role="validator_weights",
        purpose="validator.weights.finalized.v2",
        job_id=finalization_job_id,
        key=fixture.weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [
                    extrinsic_receipt["receipt_hash"]
                ],
            }
        ),
        output_root=sha256_json(finalization_doc),
        parents=[extrinsic_receipt["receipt_hash"]],
        sequence=802,
        transport_root=merkle_root(
            [item["attempt_hash"] for item in finalization_attempts],
            domain="leadpoet-transport-v2",
        ),
        artifact_root=merkle_root(
            [
                item[field]
                for item in finalization_attempts
                for field in (
                    "request_artifact_hash",
                    "response_artifact_hash",
                )
            ],
            domain="leadpoet-artifact-v2",
        ),
    )
    finalization_graph = build_receipt_graph(
        root_receipt_hash=finalization_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            *[
                receipt
                for receipt in bundle["receipt_graph"]["receipts"]
                if receipt["purpose"] != "validator.hotkey_signature.v2"
            ],
            extrinsic_receipt,
            finalization_receipt,
        ],
        transport_attempts=[
            *bundle["receipt_graph"]["transport_attempts"],
            *finalization_attempts,
        ],
    )
    verified_finalization = validate_weight_finalization_submission_v2(
        {
            "schema_version": "leadpoet.weight_finalization_submission.v2",
            "validator_hotkey": verified["validator_hotkey"],
            "weight_submission_event_hash": submission_event_hash,
            "finalization": finalization_doc,
            "receipt_graph": finalization_graph,
        },
        chain_signing_profile=profile_manifest,
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": verified["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": finalization_doc[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": finalization_doc["extrinsic_hash"],
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc["finalized_block_hash"],
            "state_transition_hash": finalization_doc["state_transition_hash"],
        }
    )
    finalization_row = {
        "weight_finalization_event_hash": finalization_event_hash,
        "weight_submission_event_hash": submission_event_hash,
        "bundle_hash": verified["bundle_hash"],
        "finalization_receipt_hash": verified_finalization[
            "finalization_receipt_hash"
        ],
        "extrinsic_authorization_hash": finalization_doc[
            "extrinsic_authorization_hash"
        ],
        "extrinsic_hash": finalization_doc["extrinsic_hash"],
        "finalized_block": finalization_doc["finalized_block"],
        "finalized_block_hash": finalization_doc["finalized_block_hash"],
        "state_transition_hash": finalization_doc["state_transition_hash"],
        "finalization_doc": finalization_doc,
    }
    graph_receipts = [
        *bundle["receipt_graph"]["receipts"],
        publication_receipt,
        extrinsic_receipt,
        finalization_receipt,
    ]
    all_transport_attempts = [
        *bundle["receipt_graph"]["transport_attempts"],
        *finalization_attempts,
    ]
    attempts_by_scope: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for attempt in all_transport_attempts:
        scope = (str(attempt["job_id"]), str(attempt["purpose"]))
        attempts_by_scope.setdefault(scope, []).append(attempt)
    rows = [
        *[
            (
                "research_lab_attested_boot_identities_v2",
                boot_storage_row(identity),
            )
            for identity in bundle["receipt_graph"]["boot_identities"]
        ],
        *[
            (
                "research_lab_attested_transport_attempts_v2",
                transport_storage_row(attempt),
            )
            for attempt in all_transport_attempts
        ],
        *[
            (
                "research_lab_attested_execution_receipts_v2",
                receipt_storage_row(receipt),
            )
            for receipt in graph_receipts
        ],
        *[
            (
                "research_lab_attested_receipt_edges_v2",
                {
                    "child_receipt_hash": receipt["receipt_hash"],
                    "parent_receipt_hash": parent_hash,
                },
            )
            for receipt in graph_receipts
            for parent_hash in receipt["parent_receipt_hashes"]
        ],
        *[
            (
                "research_lab_attested_receipt_transport_v2",
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "attempt_hash": attempt["attempt_hash"],
                },
            )
            for receipt in graph_receipts
            for attempt in attempts_by_scope.get(
                (str(receipt["job_id"]), str(receipt["purpose"])),
                [],
            )
        ],
        ("research_lab_attested_weight_bundles_v2", bundle_row),
        ("research_lab_attested_publication_events_v2", publication_row),
        ("research_lab_attested_weight_finalizations_v2", finalization_row),
    ]
    return rows, verified, fixture


def _deduplicate_settlement_fixture_rows(
    rows: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    key_fields = {
        "research_lab_attested_boot_identities_v2": (
            "boot_identity_hash",
        ),
        "research_lab_attested_transport_attempts_v2": ("attempt_hash",),
        "research_lab_attested_execution_receipts_v2": ("receipt_hash",),
        "research_lab_attested_receipt_edges_v2": (
            "child_receipt_hash",
            "parent_receipt_hash",
        ),
        "research_lab_attested_receipt_transport_v2": (
            "receipt_hash",
            "attempt_hash",
        ),
        "research_lab_attested_weight_bundles_v2": ("bundle_hash",),
        "research_lab_attested_publication_events_v2": (
            "weight_submission_event_hash",
        ),
        "research_lab_attested_weight_finalizations_v2": (
            "weight_finalization_event_hash",
        ),
    }
    deduplicated: list[tuple[str, dict[str, Any]]] = []
    seen: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
    for table, row in rows:
        fields = key_fields.get(table)
        if fields is None:
            raise PostgresContractProbeError(
                "settlement fixture table has no declared key: %s" % table
            )
        identity = (table, tuple(row.get(field) for field in fields))
        existing = seen.get(identity)
        if existing is not None:
            if existing != row:
                raise PostgresContractProbeError(
                    "settlement fixture key was reused with different content"
                )
            continue
        seen[identity] = row
        deduplicated.append((table, row))
    return deduplicated


def _settlement_graph_seed_rows(
    rows: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    graph_tables = (
        "research_lab_attested_boot_identities_v2",
        "research_lab_attested_execution_receipts_v2",
        "research_lab_attested_receipt_edges_v2",
        "research_lab_attested_receipt_transport_v2",
        "research_lab_attested_transport_attempts_v2",
    )
    return {
        table: [
            _deterministic_seed_row(row)
            for row_table, row in rows
            if row_table == table
        ]
        for table in graph_tables
    }


def _relation_contract(database: DisposablePostgres) -> dict[str, Any]:
    result = database.psql(
        """
        SELECT pg_catalog.json_build_object(
            'relations',
            COALESCE(
                (
                    SELECT pg_catalog.json_object_agg(name, relation)
                    FROM (
                        SELECT
                            class.relname AS name,
                            pg_catalog.json_build_object(
                                'kind', class.relkind,
                                'columns', pg_catalog.json_agg(
                                    attribute.attname
                                    ORDER BY attribute.attnum
                                )
                            ) AS relation
                        FROM pg_catalog.pg_class class
                        JOIN pg_catalog.pg_namespace namespace
                          ON namespace.oid = class.relnamespace
                        JOIN pg_catalog.pg_attribute attribute
                          ON attribute.attrelid = class.oid
                         AND attribute.attnum > 0
                         AND NOT attribute.attisdropped
                        WHERE namespace.nspname = 'public'
                          AND class.relkind IN ('r', 'p', 'v', 'm')
                        GROUP BY class.relname, class.relkind
                        ORDER BY class.relname
                    ) relations
                ),
                '{}'::json
            ),
            'rpcs',
            COALESCE(
                (
                    SELECT pg_catalog.json_agg(name ORDER BY name)
                    FROM (
                        SELECT DISTINCT procedure.proname AS name
                        FROM pg_catalog.pg_proc procedure
                        JOIN pg_catalog.pg_namespace namespace
                          ON namespace.oid = procedure.pronamespace
                        WHERE namespace.nspname = 'public'
                    ) procedures
                ),
                '[]'::json
            )
        )::text;
        """,
        tuples_only=True,
    )
    return json.loads(result.stdout.strip())


def _maintenance_lease_contract(
    database: DisposablePostgres,
) -> dict[str, Any]:
    def acquire(holder_ref: str, ttl_seconds: int) -> dict[str, Any]:
        result = database.psql(
            """
            SELECT public.research_lab_acquire_maintenance_lease(
                'restart-rehearsal',
                '%s',
                %d
            )::text;
            """
            % (holder_ref, ttl_seconds),
            tuples_only=True,
        )
        document = json.loads(result.stdout.strip())
        if set(document) != {"acquired", "holder_ref", "expires_at"}:
            raise PostgresContractProbeError(
                "maintenance lease RPC response shape differs"
            )
        return document

    first = acquire("worker-a", 180)
    contender = acquire("worker-b", 180)
    renewal = acquire("worker-a", 180)
    if (
        first.get("acquired") is not True
        or first.get("holder_ref") != "worker-a"
        or contender.get("acquired") is not False
        or contender.get("holder_ref") != "worker-a"
        or renewal.get("acquired") is not True
        or renewal.get("holder_ref") != "worker-a"
    ):
        raise PostgresContractProbeError(
            "maintenance lease acquire, contention, or renewal differs"
        )

    database.psql(
        """
        UPDATE public.research_lab_maintenance_lease
        SET expires_at = pg_catalog.now() - INTERVAL '1 second'
        WHERE lease_name = 'restart-rehearsal';
        """
    )
    takeover = acquire("worker-b", 180)
    row = json.loads(
        database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'row_count', pg_catalog.count(*),
                'holder_ref', pg_catalog.min(holder_ref),
                'timestamps_valid', pg_catalog.bool_and(
                    acquired_at <= updated_at
                    AND updated_at < expires_at
                )
            )::text
            FROM public.research_lab_maintenance_lease
            WHERE lease_name = 'restart-rehearsal';
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    invalid = database.psql(
        """
        SELECT public.research_lab_acquire_maintenance_lease(
            'restart-rehearsal-invalid',
            'worker-a',
            0
        )::text;
        """,
        check=False,
    )
    if (
        takeover.get("acquired") is not True
        or takeover.get("holder_ref") != "worker-b"
        or row != {
            "row_count": 1,
            "holder_ref": "worker-b",
            "timestamps_valid": True,
        }
        or invalid.returncode == 0
        or "maintenance lease arguments are invalid" not in invalid.stderr
    ):
        raise PostgresContractProbeError(
            "maintenance lease expiry takeover or fail-closed validation differs"
        )
    return {
        "schema_version": "leadpoet.maintenance_lease_contract.v1",
        "atomic_acquire": True,
        "live_contention_rejected": True,
        "same_holder_renewed": True,
        "expired_holder_replaced": True,
        "invalid_ttl_rejected": True,
    }


def _measured_settlement_receipt_contract(
    *,
    authority: Mapping[str, Any],
    verified_bundle: Mapping[str, Any],
    fixture: SanitizedWeightFixture,
) -> dict[str, Any]:
    calculation = authority["bundle_doc"]["weight_snapshot"][
        "calculation_snapshot"
    ]
    hotkeys = list(calculation["metagraph_hotkeys"])
    hotkeys[0] = str(verified_bundle["validator_hotkey"])
    finalized_block = int(authority["finalized_block"])
    weights = [
        [int(uid), int(weight)]
        for uid, weight in zip(
            verified_bundle["uids"],
            verified_bundle["weights_u16"],
        )
    ]
    observation = {
        "schema_version": CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V1,
        "netuid": int(verified_bundle["netuid"]),
        "epoch_id": int(verified_bundle["epoch_id"]),
        "official_subnet_epoch_id": int(verified_bundle["epoch_id"]),
        "cutover_mapping_hash": sha256_json({"cutover": "rehearsal"}),
        "close_block": finalized_block + 50,
        "close_block_hash": "3" * 64,
        "close_state_root": "4" * 64,
        "next_epoch_block": finalized_block + 51,
        "next_epoch_block_hash": "5" * 64,
        "validator_hotkey": str(verified_bundle["validator_hotkey"]),
        "validator_uid": 0,
        "metagraph_hotkeys": hotkeys,
        "weights": weights,
        "weights_storage_key": "0x01",
        "last_update_storage_key": "0x02",
        "last_update_block": finalized_block,
        "last_update_block_hash": str(authority["finalized_block_hash"]),
        "last_update_official_subnet_epoch_id": int(
            verified_bundle["epoch_id"]
        ),
        "active_source_epoch_id": int(verified_bundle["epoch_id"]),
        "weights_vector_hash": sha256_json(
            {
                "uids": [item[0] for item in weights],
                "weights_u16": [item[1] for item in weights],
            }
        ),
    }
    package = build_chain_realized_settlement_package_v1(
        observation=observation,
        authority=authority,
    )
    executor = CoordinatorExecutorV2(
        chain_realized_settlement_resolver=lambda _payload, _context: package
    )

    async def execute() -> Any:
        return await executor(
            OP_ATTEST_CHAIN_REALIZED_SETTLEMENT_V1,
            {
                "schema_version": (
                    "leadpoet.chain_realized_settlement_request.v1"
                ),
                "netuid": int(verified_bundle["netuid"]),
                "epoch_id": int(verified_bundle["epoch_id"]),
            },
            ExecutionContextV2(
                job_id="restart-rehearsal-chain-settlement",
                purpose="research_lab.chain_realized_epoch_settlement.v1",
                epoch_id=int(verified_bundle["epoch_id"]),
            ),
        )

    measured = asyncio.run(execute())
    if measured.output != package:
        raise PostgresContractProbeError(
            "measured chain settlement output differs"
        )
    if measured.receipt_output != package["settlement_doc"]:
        raise PostgresContractProbeError(
            "measured chain settlement receipt projection differs"
        )
    if sha256_json(measured.receipt_output) != package["settlement_hash"]:
        raise PostgresContractProbeError(
            "measured chain settlement receipt root differs"
        )
    coordinator_boot = next(
        identity
        for identity in authority["bundle_doc"]["receipt_graph"][
            "boot_identities"
        ]
        if identity["physical_role"] == "gateway_coordinator"
    )
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.chain_realized_epoch_settlement.v1",
        job_id="restart-rehearsal-chain-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {
                "schema_version": (
                    "leadpoet.chain_realized_settlement_request.v1"
                ),
                "netuid": int(verified_bundle["netuid"]),
                "epoch_id": int(verified_bundle["epoch_id"]),
            }
        ),
        output_root=package["settlement_hash"],
        parents=[verified_bundle["root_receipt_hash"]],
        sequence=802,
    )
    return {
        "settlement_hash": package["settlement_hash"],
        "credit_count": len(package["credits"]),
        "package": package,
        "receipt": receipt,
    }


def _settlement_persistence_rows(
    *,
    package: Mapping[str, Any],
    receipt_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settlement_doc = dict(package["settlement_doc"])
    settlement_hash = str(package["settlement_hash"])
    settlement_row = {
        "netuid": int(settlement_doc["netuid"]),
        "epoch_id": int(settlement_doc["epoch_id"]),
        "schema_version": str(settlement_doc["schema_version"]),
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": receipt_hash,
        "settlement_doc": settlement_doc,
    }
    credit_rows = []
    for item in package["credits"]:
        document = dict(item["credit_doc"])
        credit_rows.append(
            {
                "netuid": int(document["netuid"]),
                "epoch_id": int(document["epoch_id"]),
                "settlement_hash": settlement_hash,
                "schema_version": str(document["schema_version"]),
                "obligation_kind": str(document["obligation_kind"]),
                "obligation_source_id": str(
                    document["obligation_source_id"]
                ),
                "miner_hotkey": str(document["miner_hotkey"]),
                "miner_uid": int(document["miner_uid"]),
                "observed_chain_alpha_percent": str(
                    document["observed_chain_alpha_percent"]
                ),
                "lab_attributed_alpha_percent": str(
                    document["lab_attributed_alpha_percent"]
                ),
                "scheduled_alpha_percent": str(
                    document["scheduled_alpha_percent"]
                ),
                "credited_alpha_percent": str(
                    document["credited_alpha_percent"]
                ),
                "champion_credit_policy": str(
                    document["champion_credit_policy"]
                ),
                "credit_hash": str(item["credit_hash"]),
                "credit_receipt_hash": receipt_hash,
                "credit_doc": document,
            }
        )
    credit_rows.sort(key=lambda row: str(row["credit_hash"]))
    return settlement_row, credit_rows


def _json_rpc_sql(
    function_name: str,
    first: Mapping[str, Any],
    second: Sequence[Mapping[str, Any]],
) -> str:
    if not IDENTIFIER_RE.fullmatch(function_name):
        raise PostgresContractProbeError("fixture RPC identifier is invalid")
    first_json = json.dumps(dict(first), sort_keys=True, separators=(",", ":"))
    second_json = json.dumps(
        [dict(item) for item in second],
        sort_keys=True,
        separators=(",", ":"),
    )
    if "$leadpoet$" in first_json or "$leadpoet$" in second_json:
        raise PostgresContractProbeError("fixture RPC JSON delimiter collision")
    return (
        "SELECT public.%s("
        "$leadpoet$%s$leadpoet$::jsonb,"
        "$leadpoet$%s$leadpoet$::jsonb"
        ")::text;\n"
        % (function_name, first_json, second_json)
    )


def _historical_v1_settlement_rows(
    *,
    fixture: SanitizedWeightFixture,
    coordinator_boot: Mapping[str, Any],
    netuid: int,
    epoch_id: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    credit_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_obligation_credit.v1"
        ),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "obligation_kind": "champion",
        "obligation_source_id": "grandfathered-champion",
        "miner_hotkey": "lab-hotkey",
        "miner_uid": 2,
        "observed_chain_alpha_percent": "5",
        "lab_attributed_alpha_percent": "5",
        "scheduled_alpha_percent": "5",
        "credited_alpha_percent": "5",
        "attribution_doc": {"fixture": "grandfathered"},
        "observation_doc": {"fixture": "grandfathered"},
    }
    credit_hash = sha256_json(credit_doc)
    settlement_doc = {
        "schema_version": (
            "leadpoet.research_lab_chain_realized_epoch_settlement.v1"
        ),
        "netuid": netuid,
        "epoch_id": epoch_id,
        "credit_hashes": [credit_hash],
        "observation_summary": {"fixture": "grandfathered"},
    }
    settlement_hash = sha256_json(settlement_doc)
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.chain_realized_epoch_settlement.v1",
        job_id="restart-rehearsal-grandfathered-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {"kind": "grandfathered-settlement", "epoch_id": epoch_id}
        ),
        output_root=settlement_hash,
        parents=[],
        sequence=803,
    )
    settlement_row = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "schema_version": settlement_doc["schema_version"],
        "settlement_hash": settlement_hash,
        "settlement_receipt_hash": receipt["receipt_hash"],
        "settlement_doc": settlement_doc,
    }
    credit_row = {
        "netuid": netuid,
        "epoch_id": epoch_id,
        "settlement_hash": settlement_hash,
        "schema_version": credit_doc["schema_version"],
        "obligation_kind": credit_doc["obligation_kind"],
        "obligation_source_id": credit_doc["obligation_source_id"],
        "miner_hotkey": credit_doc["miner_hotkey"],
        "miner_uid": credit_doc["miner_uid"],
        "observed_chain_alpha_percent": (
            credit_doc["observed_chain_alpha_percent"]
        ),
        "lab_attributed_alpha_percent": (
            credit_doc["lab_attributed_alpha_percent"]
        ),
        "scheduled_alpha_percent": credit_doc["scheduled_alpha_percent"],
        "credited_alpha_percent": credit_doc["credited_alpha_percent"],
        "credit_hash": credit_hash,
        "credit_receipt_hash": receipt["receipt_hash"],
        "credit_doc": credit_doc,
    }
    return receipt_storage_row(receipt), settlement_row, credit_row


def _single_relation_row(
    database: DisposablePostgres,
    *,
    relation: str,
    where_sql: str,
) -> dict[str, Any]:
    if not IDENTIFIER_RE.fullmatch(relation):
        raise PostgresContractProbeError(
            "fixture relation identifier is invalid: %s" % relation
        )
    result = database.psql(
        """
        SELECT pg_catalog.row_to_json(row_value)::TEXT
        FROM public.%s row_value
        WHERE %s;
        """
        % (relation, where_sql),
        tuples_only=True,
    )
    rows = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise PostgresContractProbeError(
            "fixture relation returned %d rows: %s" % (len(rows), relation)
        )
    return rows[0]


def _load_coordinator_release_identity(
    path: Path,
    *,
    candidate_sha: str,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        roles = document["gateway_roles"]
        identity = roles["gateway_coordinator"]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PostgresContractProbeError(
            "candidate coordinator release identity is unavailable"
        ) from exc
    if (
        not isinstance(document, dict)
        or document.get("commit_sha") != candidate_sha
        or not isinstance(roles, dict)
        or not isinstance(identity, dict)
        or identity.get("commit_sha") != candidate_sha
    ):
        raise PostgresContractProbeError(
            "candidate coordinator release identity commit differs"
        )
    required = {
        "commit_sha",
        "dependency_lock_hash",
        "execution_manifest_hash",
        "pcr0",
    }
    if required - set(identity):
        raise PostgresContractProbeError(
            "candidate coordinator release identity fields are incomplete"
        )
    return dict(identity)


def _historical_compute_reimbursements(
    *,
    source_root: Path,
    source_epoch: int,
) -> list[dict[str, Any]]:
    metagraph_hotkeys = load_rehearsal_metagraph_hotkeys(source_root)
    return [
        {
            "uid": 2,
            "miner_hotkey": metagraph_hotkeys[2],
            "source_id": "reimbursement_schedule:restart-rehearsal-compute-2",
            "island": "generalist",
            "status": "active",
            "start_epoch": source_epoch,
            "epoch_count": 20,
            "target_reimbursement_microusd": 1_000_000,
            "eligible_compute_microusd": 1_000_000,
        },
        {
            "uid": 3,
            "miner_hotkey": metagraph_hotkeys[3],
            "source_id": "reimbursement_schedule:restart-rehearsal-compute-3",
            "island": "generalist",
            "status": "active",
            "start_epoch": source_epoch,
            "epoch_count": 20,
            "target_reimbursement_microusd": 3_000_000,
            "eligible_compute_microusd": 3_000_000,
        },
    ]


def _historical_compute_source_epoch(
    native_finalized_epochs: Sequence[int],
) -> int:
    """Place the legacy fixture immediately before native V2 authority."""

    normalized = sorted({int(epoch) for epoch in native_finalized_epochs})
    if not normalized or normalized[0] <= 0:
        raise PostgresContractProbeError(
            "historical compute source epoch is unavailable"
        )
    return normalized[0] - 1


def _historical_compute_allocation_seed_rows(
    *,
    database: DisposablePostgres,
    source_root: Path,
    candidate_sha: str,
    source_epoch: int,
    netuid: int,
    coordinator_release_identity: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Persist and read back one finalized prior compute allocation."""

    source_epoch = int(source_epoch)
    if source_epoch < 0:
        raise PostgresContractProbeError(
            "historical compute source epoch is unavailable"
        )
    policy = {
        "policy_id": "restart-rehearsal-no-burn-v2",
        "enabled": True,
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_max_cost_multiplier_with_champions": 2.0,
        "champion_threshold_points": 1.0,
        "champion_min_alpha_percent": 7.0,
        "champion_extra_alpha_percent_per_point": 0.3,
        "champion_max_alpha_percent": 30.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.5,
        "usd_per_0_1_percent_epoch": 1.0,
        "enable_conservative": False,
        "enable_champ_cap": False,
    }
    reimbursements = _historical_compute_reimbursements(
        source_root=source_root,
        source_epoch=source_epoch,
    )
    allocation = allocate_research_lab_epoch(
        source_epoch,
        policy,
        reimbursements,
        [],
    )
    if (
        allocation.get("allocation_hash") != sha256_json(
            {
                key: value
                for key, value in allocation.items()
                if key != "allocation_hash"
            }
        )
        or float(allocation.get("reimbursement_alpha_percent") or 0) != 20.0
        or float(allocation.get("unallocated_percent") or 0) != 0.0
    ):
        raise PostgresContractProbeError(
            "historical compute allocation did not conserve the Lab cap"
        )

    allocation_hash = str(allocation["allocation_hash"])
    snapshot_row = {
        "allocation_id": "lab_allocation:" + allocation_hash,
        "schema_version": "1.0",
        "epoch": source_epoch,
        "netuid": int(netuid),
        "policy_id": str(policy["policy_id"]),
        "snapshot_status": "active",
        "lab_cap_alpha_percent": allocation["lab_cap_percent"],
        "reimbursement_alpha_percent": allocation[
            "reimbursement_alpha_percent"
        ],
        "champion_alpha_percent": allocation["champion_alpha_percent"],
        "queued_champion_alpha_percent": allocation[
            "queued_champion_alpha_percent"
        ],
        "unallocated_alpha_percent": allocation["unallocated_percent"],
        "input_hash": allocation["input_hash"],
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
        "source_add_alpha_percent": allocation.get(
            "source_add_alpha_percent",
            0,
        ),
    }

    def hash_ref(label: str) -> str:
        return sha256_json({"fixture": label, "epoch": source_epoch})

    settlement_body = {
        "schema_version": LEGACY_SETTLEMENT_SCHEMA_VERSION,
        "netuid": int(netuid),
        "epoch_id": source_epoch,
        "allocation_hash": allocation_hash,
        "allocation_doc": allocation,
        "validator_hotkey": (
            "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
        ),
        "legacy_bundle_weights_hash": hash_ref("legacy-weights").split(
            ":", 1
        )[1],
        "legacy_bundle_block": source_epoch * 360 + 99,
        "chain_compare_hash": hash_ref("chain-compare"),
        "chain_vector_tolerance_u16": 1,
        "chain_target_block": (source_epoch + 1) * 360 - 1,
        "chain_target_block_hash": hash_ref("chain-target-block"),
        "chain_finalized_head_block": (source_epoch + 1) * 360,
        "validator_uid": 0,
        "weights_storage_key_hash": hash_ref("weights-storage-key"),
        "audit_event_hash": hash_ref("audit-event"),
        "audit_payload_hash": hash_ref("audit-payload"),
        "checkpoint_merkle_root": hash_ref("checkpoint-merkle-root"),
        "checkpoint_number": 1,
        "checkpoint_event_sequence": 1,
        "arweave_tx_id": "R" * 43,
    }
    settlement_doc = validate_legacy_settlement_document_v2(
        {
            **settlement_body,
            "settlement_hash": sha256_json(settlement_body),
        }
    )

    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=source_epoch,
    )
    coordinator_config_hash = sha256_json(
        {
            "candidate_sha": candidate_sha,
            "fixture": "historical-compute-coordinator-config",
        }
    )
    coordinator_boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=coordinator_config_hash,
        release_identity=coordinator_release_identity,
        boot_nonce_context="historical-compute-settlement",
    )
    settlement_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.legacy_finalized_allocation.v2",
        job_id="restart-rehearsal-historical-compute-settlement",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=coordinator_config_hash,
        input_root=sha256_json(
            {
                "kind": "historical-compute-allocation",
                "allocation_hash": allocation_hash,
            }
        ),
        output_root=sha256_json(settlement_doc),
        parents=[],
        sequence=804,
    )
    settlement_receipt_hash = str(settlement_receipt["receipt_hash"])
    migration_row = {
        "netuid": int(netuid),
        "epoch_id": source_epoch,
        "schema_version": LEGACY_SETTLEMENT_SCHEMA_VERSION,
        "allocation_hash": allocation_hash,
        "settlement_hash": settlement_doc["settlement_hash"],
        "settlement_receipt_hash": settlement_receipt_hash,
        "allocation_doc": allocation,
        "settlement_doc": settlement_doc,
    }
    graph = build_receipt_graph(
        root_receipt_hash=settlement_receipt_hash,
        boot_identities=[coordinator_boot],
        receipts=[settlement_receipt],
        transport_attempts=[],
        host_operations=[],
    )
    validated = validate_legacy_settlement_migrations_v2(
        [migration_row],
        receipt_graphs={settlement_receipt_hash: graph},
    )
    if (
        len(validated) != 1
        or validated[0].get("allocation_hash") != allocation_hash
        or validated[0].get("allocation_doc") != allocation
    ):
        raise PostgresContractProbeError(
            "historical compute finalized authority did not validate"
        )

    database.psql(
        "".join(
            (
                _json_insert_sql(
                    "research_lab_emission_allocation_snapshots",
                    _deterministic_seed_row(snapshot_row),
                ),
                _json_insert_sql(
                    "research_lab_attested_boot_identities_v2",
                    _deterministic_seed_row(
                        boot_storage_row(coordinator_boot)
                    ),
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_receipts_v2",
                    _deterministic_seed_row(
                        receipt_storage_row(settlement_receipt)
                    ),
                ),
                _json_insert_sql(
                    "research_lab_legacy_finalized_allocation_migrations_v2",
                    _deterministic_seed_row(migration_row),
                ),
            )
        )
    )

    rows = {
        "research_lab_emission_allocation_current": [
            _single_relation_row(
                database,
                relation="research_lab_emission_allocation_current",
                where_sql="epoch = %d AND netuid = %d"
                % (source_epoch, int(netuid)),
            )
        ],
        "research_lab_legacy_finalized_allocation_migrations_v2": [
            _single_relation_row(
                database,
                relation=(
                    "research_lab_legacy_finalized_allocation_migrations_v2"
                ),
                where_sql="epoch_id = %d AND netuid = %d"
                % (source_epoch, int(netuid)),
            )
        ],
        "research_lab_attested_boot_identities_v2": [
            _single_relation_row(
                database,
                relation="research_lab_attested_boot_identities_v2",
                where_sql="boot_identity_hash = '%s'"
                % coordinator_boot["boot_identity_hash"],
            )
        ],
        "research_lab_attested_execution_receipts_v2": [
            _single_relation_row(
                database,
                relation="research_lab_attested_execution_receipts_v2",
                where_sql="receipt_hash = '%s'" % settlement_receipt_hash,
            )
        ],
    }
    return rows


def _allocation_settlement_frontier_contract(
    *,
    database: DisposablePostgres,
    fixture: SanitizedWeightFixture,
    verified_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise migration 137 with a signed production-shaped allocation."""

    epoch_id = int(verified_bundle["epoch_id"])
    netuid = int(verified_bundle["netuid"])
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=netuid,
        allocation_epoch=epoch_id,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    source_state = {
        "epoch": epoch_id,
        "netuid": netuid,
        "settlement_frontier": frontier,
    }
    source_state_hash = sha256_json(source_state)
    allocation = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": epoch_id,
        "netuid": netuid,
        "allocations": [],
    }
    result = {
        "allocation": allocation,
        "allocation_inputs": {},
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        {source_state_hash, str(frontier["frontier_hash"])}
    )
    fixture_bundle = fixture.bundle()
    coordinator_boot = next(
        identity
        for identity in fixture_bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="postgres-contract-allocation-frontier",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {"kind": "allocation-frontier", "epoch_id": epoch_id}
        ),
        output_root=sha256_json(
            coordinator_receipt_output_v2(
                "research_lab_allocation",
                result,
            )
        ),
        artifact_root=merkle_root(
            artifact_hashes,
            domain="leadpoet-artifact-v2",
        ),
        sequence=805,
    )
    execution_row = _execution_result_storage_row_v2(
        operation="research_lab_allocation",
        result=result,
        receipt=receipt,
        artifact_hashes=artifact_hashes,
        release_hash=sha256_json(
            {
                "candidate_sha": fixture.candidate_sha,
                "contract": "allocation-settlement-frontier",
            }
        ),
    )
    database.psql(
        "".join(
            (
                _json_insert_sql(
                    "research_lab_attested_execution_receipts_v2",
                    receipt_storage_row(receipt),
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_results_v2",
                    execution_row,
                ),
            )
        )
    )
    frontier_payload = json.dumps(
        frontier,
        sort_keys=True,
        separators=(",", ":"),
    )
    if "$leadpoet$" in frontier_payload:
        raise PostgresContractProbeError(
            "allocation settlement frontier JSON delimiter collision"
        )
    request_sql = """
        SELECT public.persist_research_lab_allocation_settlement_frontier_v2(
            $leadpoet$%s$leadpoet$::jsonb,
            '%s',
            '%s'
        )::text;
    """ % (
        frontier_payload,
        receipt["receipt_hash"],
        source_state_hash,
    )
    persisted = json.loads(
        database.psql(request_sql, tuples_only=True).stdout.strip()
    )
    replayed = json.loads(
        database.psql(request_sql, tuples_only=True).stdout.strip()
    )
    counts = json.loads(
        database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'frontiers', (
                    SELECT pg_catalog.count(*)
                    FROM public.research_lab_allocation_settlement_frontiers_v2
                ),
                'activations', (
                    SELECT pg_catalog.count(*)
                    FROM public.research_lab_allocation_settlement_frontier_activation_v2
                )
            )::text;
            """,
            tuples_only=True,
        ).stdout.strip()
    )
    expected_identity = {
        "netuid": netuid,
        "allocation_epoch": epoch_id,
        "frontier_hash": frontier["frontier_hash"],
        "source_receipt_hash": receipt["receipt_hash"],
        "source_state_hash": source_state_hash,
    }
    if (
        persisted != {"status": "persisted", **expected_identity}
        or replayed != {"status": "already_persisted", **expected_identity}
        or counts != {"frontiers": 1, "activations": 1}
    ):
        raise PostgresContractProbeError(
            "post-137 allocation settlement frontier contract differs"
        )
    return {
        "frontier_hash": frontier["frontier_hash"],
        "source_receipt_hash": receipt["receipt_hash"],
        "idempotent_replay": True,
        "frontier_count": 1,
        "activation_count": 1,
    }


def _allocation_settlement_frontier_bootstrap_contract(
    *,
    database: DisposablePostgres,
    fixture: SanitizedWeightFixture,
    verified_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise migration 139 with signed source and bootstrap receipts."""

    epoch_id = int(verified_bundle["epoch_id"])
    netuid = int(verified_bundle["netuid"]) + 1
    source_state = {
        "epoch": epoch_id,
        "netuid": netuid,
        "policy": {"enable_champ_cap": True},
        "champion_obligation_count": 0,
        "champion_obligations": [],
        "source_add_obligation_count": 0,
        "source_add_obligations": [],
        "skipped": {"champions": [], "source_add": []},
    }
    source_state_hash = sha256_json(source_state)
    allocation = {
        "schema_version": "leadpoet.research_lab_allocation.v2",
        "epoch": epoch_id,
        "netuid": netuid,
        "allocations": [],
    }
    source_result = {
        "allocation": allocation,
        "allocation_inputs": {},
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    source_artifacts = sorted(
        {source_state_hash, sha256_json(allocation)}
    )
    fixture_bundle = fixture.bundle()
    coordinator_boot = next(
        identity
        for identity in fixture_bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    source_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="postgres-contract-allocation-frontier-bootstrap-source",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {"kind": "allocation-frontier-bootstrap-source", "epoch_id": epoch_id}
        ),
        output_root=sha256_json(
            coordinator_receipt_output_v2(
                "research_lab_allocation",
                source_result,
            )
        ),
        artifact_root=merkle_root(
            source_artifacts,
            domain="leadpoet-artifact-v2",
        ),
        sequence=806,
    )
    source_execution = _execution_result_storage_row_v2(
        operation="research_lab_allocation",
        result=source_result,
        receipt=source_receipt,
        artifact_hashes=source_artifacts,
        release_hash=sha256_json(
            {
                "candidate_sha": fixture.candidate_sha,
                "contract": "allocation-frontier-bootstrap-source",
            }
        ),
    )
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=netuid,
        allocation_epoch=epoch_id,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    bootstrap = build_allocation_settlement_frontier_bootstrap_v2(
        netuid=netuid,
        bootstrap_epoch=epoch_id,
        allocation_source_receipt_hash=source_receipt["receipt_hash"],
        source_state_hash=source_state_hash,
        frontier=frontier,
    )
    bootstrap_artifacts = list(frontier_bootstrap_artifact_hashes_v2(bootstrap))
    bootstrap_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE,
        job_id="postgres-contract-allocation-frontier-bootstrap",
        key=fixture.coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json(
            {"kind": "allocation-frontier-bootstrap", "epoch_id": epoch_id}
        ),
        output_root=sha256_json(
            coordinator_receipt_output_v2(
                ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
                bootstrap,
            )
        ),
        parents=[source_receipt["receipt_hash"]],
        artifact_root=merkle_root(
            bootstrap_artifacts,
            domain="leadpoet-artifact-v2",
        ),
        sequence=807,
    )
    bootstrap_execution = _execution_result_storage_row_v2(
        operation=ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION,
        result=bootstrap,
        receipt=bootstrap_receipt,
        artifact_hashes=bootstrap_artifacts,
        release_hash=sha256_json(
            {
                "candidate_sha": fixture.candidate_sha,
                "contract": "allocation-frontier-bootstrap",
            }
        ),
    )
    database.psql(
        "".join(
            (
                _json_insert_sql(
                    "research_lab_attested_execution_receipts_v2",
                    receipt_storage_row(source_receipt),
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_results_v2",
                    source_execution,
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_receipts_v2",
                    receipt_storage_row(bootstrap_receipt),
                ),
                _json_insert_sql(
                    "research_lab_attested_execution_results_v2",
                    bootstrap_execution,
                ),
            )
        )
    )
    frontier_payload = json.dumps(frontier, sort_keys=True, separators=(",", ":"))
    if "$leadpoet$" in frontier_payload:
        raise PostgresContractProbeError(
            "allocation frontier bootstrap JSON delimiter collision"
        )
    request_sql = """
        SELECT public.persist_research_lab_allocation_frontier_bootstrap_v2(
            $leadpoet$%s$leadpoet$::jsonb,
            '%s',
            '%s'
        )::text;
    """ % (
        frontier_payload,
        bootstrap_receipt["receipt_hash"],
        source_state_hash,
    )
    persisted = json.loads(
        database.psql(request_sql, tuples_only=True).stdout.strip()
    )
    replayed = json.loads(
        database.psql(request_sql, tuples_only=True).stdout.strip()
    )
    counts = json.loads(
        database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'frontiers', (
                    SELECT pg_catalog.count(*)
                    FROM public.research_lab_allocation_settlement_frontiers_v2
                    WHERE netuid = %d
                ),
                'activations', (
                    SELECT pg_catalog.count(*)
                    FROM public.research_lab_allocation_settlement_frontier_activation_v2
                    WHERE netuid = %d
                )
            )::text;
            """ % (netuid, netuid),
            tuples_only=True,
        ).stdout.strip()
    )
    expected_identity = {
        "netuid": netuid,
        "allocation_epoch": epoch_id,
        "frontier_hash": frontier["frontier_hash"],
        "source_receipt_hash": bootstrap_receipt["receipt_hash"],
        "source_state_hash": source_state_hash,
    }
    if (
        persisted != {"status": "persisted", **expected_identity}
        or replayed != {"status": "already_persisted", **expected_identity}
        or counts != {"frontiers": 1, "activations": 1}
    ):
        raise PostgresContractProbeError(
            "post-139 allocation frontier bootstrap contract differs"
        )
    rejected = database.psql(
        request_sql.replace(
            bootstrap_receipt["receipt_hash"],
            source_receipt["receipt_hash"],
        ),
        check=False,
    )
    if rejected.returncode == 0 or "allocation_frontier_bootstrap_authority_invalid" not in rejected.stderr:
        raise PostgresContractProbeError(
            "post-139 unmeasured frontier bootstrap did not fail closed"
        )
    return {
        "frontier_hash": frontier["frontier_hash"],
        "allocation_source_receipt_hash": source_receipt["receipt_hash"],
        "bootstrap_receipt_hash": bootstrap_receipt["receipt_hash"],
        "idempotent_replay": True,
        "unmeasured_source_rejected": True,
        "frontier_count": 1,
        "activation_count": 1,
    }


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    declaration_counts = _validate_required_migration_declarations(args.source_root)
    coordinator_release_identity = _load_coordinator_release_identity(
        args.release_build_input,
        candidate_sha=args.candidate_sha,
    )
    database = DisposablePostgres(state_root=args.state_root)
    try:
        database.start()
        scripts = args.source_root / "scripts"
        applied = []
        database.psql(ALLOCATION_MIGRATION_PREREQUISITES_SQL)
        database.apply_migration(scripts / EVENT_PROJECTIONS_MIGRATION)
        applied.append(EVENT_PROJECTIONS_MIGRATION)
        database.apply_migration(
            scripts / ALLOCATION_CANDIDATE_MIGRATION
        )
        applied.append(ALLOCATION_CANDIDATE_MIGRATION)
        database.apply_migration(
            scripts / ALLOCATION_AUTO_RESEARCH_MIGRATION
        )
        applied.append(ALLOCATION_AUTO_RESEARCH_MIGRATION)
        database.apply_migration(scripts / ALLOCATION_SCHEMA_MIGRATION)
        applied.append(ALLOCATION_SCHEMA_MIGRATION)
        for name in (
            ALLOCATION_SCORING_AUDIT_MIGRATION,
            ALLOCATION_PROMOTION_MIGRATION,
            ATOMIC_CLAIM_GUARDS_MIGRATION,
            QUEUE_CAPACITY_GUARD_MIGRATION,
            MAINTENANCE_PAUSE_MIGRATION,
            *ALLOCATION_IMAGE_BUILD_MIGRATIONS[:2],
            PAUSED_CAPACITY_AGING_MIGRATION,
            ALLOCATION_IMAGE_BUILD_MIGRATIONS[2],
            RESUME_REQUEUE_HOTKEY_GUARD_MIGRATION,
            HOTKEY_ACTIVE_LOOP_CAP_MIGRATION,
        ):
            database.apply_migration(scripts / name)
            applied.append(name)
        for name in SOURCE_ADD_PRE_V2_MIGRATIONS:
            database.apply_migration(scripts / name)
            applied.append(name)
        candidate_view_columns = {
            row.strip()
            for row in database.psql(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'research_lab_candidate_evaluation_current';
                """,
                tuples_only=True,
            ).stdout.splitlines()
            if row.strip()
        }
        required_candidate_view_columns = {
            "candidate_kind",
            "candidate_model_manifest_hash",
            "candidate_model_manifest_doc",
            "candidate_source_diff_hash",
            "candidate_build_doc",
        }
        if not required_candidate_view_columns.issubset(candidate_view_columns):
            raise PostgresContractProbeError(
                "candidate evaluation view is missing image-build columns: %s"
                % sorted(required_candidate_view_columns - candidate_view_columns)
            )
        database.psql(GIT_TREE_CANDIDATE_PREREQUISITES_SQL)
        for name in MIGRATIONS_BEFORE_TRANSPORT_FIX:
            database.apply_migration(scripts / name)
            applied.append(name)
            if name == "86-research-lab-attested-v2-authority.sql":
                database.apply_migration(
                    scripts / ALLOCATION_CONTAINMENT_MIGRATION
                )
                applied.append(ALLOCATION_CONTAINMENT_MIGRATION)
            if name == "90-research-lab-provider-outcome-checkpoints-v2.sql":
                database.apply_migration(
                    scripts / GIT_TREE_AUTORESEARCH_MIGRATION
                )
                applied.append(GIT_TREE_AUTORESEARCH_MIGRATION)
                database.apply_migration(
                    scripts / SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION
                )
                applied.append(SOURCE_ADD_FUNCTIONAL_WORKFLOW_MIGRATION)
            if name == "104-research-lab-attested-result-replay-v2.sql":
                database.apply_migration(
                    scripts / GIT_TREE_ROOT_REPLACEMENT_MIGRATION
                )
                applied.append(GIT_TREE_ROOT_REPLACEMENT_MIGRATION)

        maintenance_lease = _maintenance_lease_contract(database)

        fixture = SanitizedWeightFixture(
            candidate_sha=args.candidate_sha,
            epoch_id=args.epoch_id,
        )
        attempt = fixture.source_attempt(
            category="chain-settlement-contract",
            job_id="postgres-contract-transport",
            purpose="research_lab.chain_weight_observation.v1",
            sequence=700,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="WSS",
        )
        attempt_sql = _json_insert_sql(
            "research_lab_attested_transport_attempts_v2",
            transport_storage_row(attempt),
        )
        rejected = database.psql(attempt_sql, check=False)
        if rejected.returncode == 0:
            raise PostgresContractProbeError(
                "pre-128 V1 transport evidence unexpectedly persisted"
            )
        if (
            "research_lab_attested_transport_attempts_v2_purpose_check"
            not in rejected.stderr
        ):
            raise PostgresContractProbeError(
                "pre-128 transport rejection differed: %s" % rejected.stderr.strip()
            )

        database.apply_migration(scripts / TRANSPORT_FIX_MIGRATION)
        applied.append(TRANSPORT_FIX_MIGRATION)
        database.psql(attempt_sql)

        contract_result = database.psql(
            """
            SELECT
                public.research_lab_attested_transport_purpose_contract_v2()
                ::text;
            """,
            tuples_only=True,
        )
        transport_contract = json.loads(contract_result.stdout.strip())
        definition = str(transport_contract.get("constraint_definition") or "")
        if (
            transport_contract.get("constraint_valid") is not True
            or "research_lab.chain_weight_observation.v1" not in definition
            or "research_lab.chain_realized_epoch_settlement.v1" not in definition
        ):
            raise PostgresContractProbeError(
                "post-128 transport purpose contract is incomplete"
            )

        local_attempt = build_transport_attempt(
            request_id="f" * 32,
            logical_operation_id="provider-preflight-local-cache",
            job_id="postgres-contract-local-transport",
            purpose="research_lab.provider_preflight.v2",
            provider_id="exa",
            attempt_number=0,
            method="POST",
            destination_host="api.exa.ai",
            destination_port=443,
            path_hash=sha256_json({"path": "/search"}),
            nonsecret_headers_hash=sha256_json({"accept": "application/json"}),
            body_hash=sha256_json({"query": "rehearsal"}),
            credential_ref_hash=sha256_json({"credential": "attested-local"}),
            retry_policy_hash=sha256_json({"retry": "provider-preflight"}),
            timeout_ms=30000,
            started_at="2026-07-10T00:00:00Z",
            terminal_status="attested_local_response",
            http_status=200,
            response_hash=sha256_json({"response": "cached"}),
            request_artifact_hash=sha256_json({"artifact": "local-request"}),
            response_artifact_hash=sha256_json({"artifact": "local-response"}),
            tls_peer_chain_hash=None,
            tls_protocol=None,
            failure_code=None,
            completed_at="2026-07-10T00:00:01Z",
        )
        local_attempt_sql = _json_insert_sql(
            "research_lab_attested_transport_attempts_v2",
            transport_storage_row(local_attempt),
        )
        local_rejected = database.psql(local_attempt_sql, check=False)
        if local_rejected.returncode == 0:
            raise PostgresContractProbeError(
                "pre-129 attested local transport unexpectedly persisted"
            )
        if (
            "check constraint" not in local_rejected.stderr.lower()
            or "transport_attempts" not in local_rejected.stderr
        ):
            raise PostgresContractProbeError(
                "pre-129 attested local rejection differed: %s"
                % local_rejected.stderr.strip()
            )

        database.apply_migration(scripts / TRANSPORT_TERMINAL_MIGRATION)
        applied.append(TRANSPORT_TERMINAL_MIGRATION)
        database.psql(local_attempt_sql)
        terminal_contract_result = database.psql(
            """
            SELECT
                public.research_lab_attested_transport_terminal_contract_v2()
                ::text;
            """,
            tuples_only=True,
        )
        terminal_contract = json.loads(
            terminal_contract_result.stdout.strip()
        )
        terminal_constraints = terminal_contract.get("constraints")
        if not isinstance(terminal_constraints, Mapping) or set(
            terminal_constraints
        ) != {
            "research_lab_transport_terminal_status_v2_check",
            "research_lab_transport_terminal_shape_v2_check",
        }:
            raise PostgresContractProbeError(
                "post-129 transport terminal contract is incomplete"
            )
        for constraint in terminal_constraints.values():
            definition = str(constraint.get("constraint_definition") or "")
            if (
                constraint.get("constraint_valid") is not True
                or "attested_local_response" not in definition
            ):
                raise PostgresContractProbeError(
                    "post-129 transport terminal constraint is invalid"
                )

        database.apply_migration(scripts / PROVIDER_OUTCOME_APPEND_MIGRATION)
        applied.append(PROVIDER_OUTCOME_APPEND_MIGRATION)
        database.apply_migration(
            scripts / PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_BACKPRESSURE_MIGRATION)
        pre_contention_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v2()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_contention_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v2"
            not in pre_contention_contract.stderr
            or "does not exist" not in pre_contention_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-133 provider outcome contention contract did not fail closed"
            )
        pre_head_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v3()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_head_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v3"
            not in pre_head_contract.stderr
            or "does not exist" not in pre_head_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-134 provider outcome head contract did not fail closed"
            )

        prior_rows, prior_verified, _prior_fixture = _settlement_fixture(
            candidate_sha=args.candidate_sha,
            epoch_id=args.epoch_id - 1,
            source_root=args.source_root,
        )
        rows, verified, fixture = _settlement_fixture(
            candidate_sha=args.candidate_sha,
            epoch_id=args.epoch_id,
            source_root=args.source_root,
        )
        settlement_fixture_rows = _deduplicate_settlement_fixture_rows(
            (*prior_rows, *rows)
        )
        database.psql(
            "".join(
                _json_insert_sql(table, row)
                for table, row in settlement_fixture_rows
            )
        )
        view_result = database.psql(
            """
            SELECT pg_catalog.row_to_json(authority)::text
            FROM public.research_lab_finalized_allocation_epochs_v2 authority;
            """,
            tuples_only=True,
        )
        view_rows = [
            json.loads(line) for line in view_result.stdout.splitlines() if line.strip()
        ]
        if len(view_rows) != 2:
            raise PostgresContractProbeError(
                "finalized allocation view returned %d rows" % len(view_rows)
            )
        view_rows_by_epoch = {
            int(row["epoch_id"]): row for row in view_rows
        }
        if set(view_rows_by_epoch) != {args.epoch_id - 1, args.epoch_id}:
            raise PostgresContractProbeError(
                "finalized allocation view epochs differ"
            )
        prior_view_row = view_rows_by_epoch[args.epoch_id - 1]
        view_row = view_rows_by_epoch[args.epoch_id]
        if tuple(view_row) != EXPECTED_FINALIZED_VIEW_COLUMNS:
            raise PostgresContractProbeError(
                "finalized allocation view columns differ: %s" % ",".join(view_row)
            )
        if "weight_receipt_hash" in view_row:
            raise PostgresContractProbeError(
                "finalized allocation view synthesized weight_receipt_hash"
            )
        authority = _preliminary_finalized_bundle_authority_v1(view_row)
        prior_authority = _preliminary_finalized_bundle_authority_v1(
            prior_view_row
        )
        if authority["weight_receipt_hash"] != verified["weight_receipt_hash"]:
            raise PostgresContractProbeError(
                "settlement authority weight receipt differs"
            )
        measured_settlement = _measured_settlement_receipt_contract(
            authority=authority,
            verified_bundle=verified,
            fixture=fixture,
        )
        package = measured_settlement.pop("package")
        settlement_receipt = measured_settlement.pop("receipt")
        if (
            package["settlement_doc"]["schema_version"]
            != "leadpoet.research_lab_chain_realized_epoch_settlement.v3"
            or not package["credits"]
        ):
            raise PostgresContractProbeError(
                "marked lifetime settlement fixture is incomplete"
            )
        coordinator_boot = next(
            identity
            for identity in authority["bundle_doc"]["receipt_graph"][
                "boot_identities"
            ]
            if identity["physical_role"] == "gateway_coordinator"
        )
        historical_receipt, historical_settlement, historical_credit = (
            _historical_v1_settlement_rows(
                fixture=fixture,
                coordinator_boot=coordinator_boot,
                netuid=int(verified["netuid"]),
                epoch_id=int(verified["epoch_id"]) - 1,
            )
        )
        activation_row = {
            "netuid": int(verified["netuid"]),
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_activation.v1"
            ),
            "first_epoch_id": int(verified["epoch_id"]) - 1,
            "source_bundle_hash": str(prior_verified["bundle_hash"]),
            "source_bundle_epoch_id": int(verified["epoch_id"]) - 1,
            "source_finalized_block": int(
                prior_authority["finalized_block"]
            ),
        }
        database.psql(
            "".join(
                (
                    _json_insert_sql(
                        "research_lab_attested_execution_receipts_v2",
                        historical_receipt,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_settlement_activation_v1",
                        activation_row,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_epoch_settlements_v1",
                        historical_settlement,
                    ),
                    _json_insert_sql(
                        "research_lab_chain_realized_obligation_credits_v1",
                        historical_credit,
                    ),
                )
            )
        )

        settlement_row, credit_rows = _settlement_persistence_rows(
            package=package,
            receipt_hash=str(settlement_receipt["receipt_hash"]),
        )
        lifetime_rpc = (
            "persist_research_lab_chain_realized_lifetime_settlement_v2"
        )
        persistence_sql = _json_rpc_sql(
            lifetime_rpc,
            settlement_row,
            credit_rows,
        )
        pre_lifetime = database.psql(persistence_sql, check=False)
        if pre_lifetime.returncode == 0 or (
            lifetime_rpc not in pre_lifetime.stderr
            or "does not exist" not in pre_lifetime.stderr
        ):
            raise PostgresContractProbeError(
                "pre-132 lifetime persistence did not fail closed: %s"
                % pre_lifetime.stderr.strip()
            )

        database.apply_migration(scripts / CHAMPION_LIFETIME_CREDIT_MIGRATION)
        applied.append(CHAMPION_LIFETIME_CREDIT_MIGRATION)
        historical_result = database.psql(
            """
            SELECT pg_catalog.json_build_object(
                'schema_version', schema_version,
                'champion_credit_policy', champion_credit_policy,
                'document_has_policy', credit_doc ? 'champion_credit_policy',
                'credited_alpha_percent', credited_alpha_percent::TEXT
            )::text
            FROM public.research_lab_chain_realized_obligation_credits_v1
            WHERE netuid = 71
              AND epoch_id = %d
              AND obligation_source_id = 'grandfathered-champion';
            """
            % (int(verified["epoch_id"]) - 1),
            tuples_only=True,
        )
        historical_contract = json.loads(historical_result.stdout.strip())
        if historical_contract != {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_obligation_credit.v1"
            ),
            "champion_credit_policy": "scheduled_bonus_v1",
            "document_has_policy": False,
            "credited_alpha_percent": "5.000000000000",
        }:
            raise PostgresContractProbeError(
                "migration 132 changed grandfathered settlement credit"
            )

        database.psql(
            _json_insert_sql(
                "research_lab_attested_execution_receipts_v2",
                receipt_storage_row(settlement_receipt),
            )
        )
        first_persistence = json.loads(
            database.psql(
                persistence_sql,
                tuples_only=True,
            ).stdout.strip()
        )
        repeated_persistence = json.loads(
            database.psql(
                persistence_sql,
                tuples_only=True,
            ).stdout.strip()
        )
        expected_credit_hashes = sorted(
            str(row["credit_hash"]) for row in credit_rows
        )
        expected_persistence = {
            "schema_version": (
                "leadpoet.research_lab_chain_realized_"
                "settlement_persistence.v1"
            ),
            "netuid": int(verified["netuid"]),
            "epoch_id": int(verified["epoch_id"]),
            "settlement_hash": str(package["settlement_hash"]),
            "settlement_receipt_hash": str(
                settlement_receipt["receipt_hash"]
            ),
            "credit_count": len(credit_rows),
            "credit_hashes": expected_credit_hashes,
        }
        if (
            first_persistence != expected_persistence
            or repeated_persistence != expected_persistence
        ):
            raise PostgresContractProbeError(
                "lifetime settlement persistence is not exact and idempotent"
            )
        lifetime_contract_result = database.psql(
            """
            SELECT
                public.research_lab_champion_lifetime_credit_contract_v1()
                ::text;
            """,
            tuples_only=True,
        )
        lifetime_contract = json.loads(
            lifetime_contract_result.stdout.strip()
        )
        lifetime_constraints = lifetime_contract.get("constraints")
        if (
            lifetime_contract.get("schema_version")
            != (
                "leadpoet.research_lab_champion_"
                "lifetime_credit_contract.v1"
            )
            or lifetime_contract.get("champion_credit_policy")
            != "accelerated_lifetime_cap_v1"
            or lifetime_contract.get("credit_policy_column") is not True
            or not isinstance(lifetime_constraints, Mapping)
            or set(lifetime_constraints)
            != {
                "research_lab_chain_settlement_schema_check",
                "research_lab_chain_settlement_champion_policy_check",
                "research_lab_chain_credit_schema_policy_check",
                "research_lab_chain_credit_policy_amount_check",
            }
            or any(
                constraint.get("validated") is not True
                for constraint in lifetime_constraints.values()
            )
        ):
            raise PostgresContractProbeError(
                "post-131 lifetime credit contract is incomplete"
            )

        database.apply_migration(
            scripts / PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_CONTENTION_STATUS_MIGRATION)
        contention_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_provider_outcome_contention_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if contention_contract != {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v2"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
        }:
            raise PostgresContractProbeError(
                "post-133 provider outcome contention contract differs"
            )
        pre_head_contract = database.psql(
            """
            SELECT public.research_lab_provider_outcome_contention_contract_v3()
                   ::text;
            """,
            check=False,
        )
        if (
            pre_head_contract.returncode == 0
            or "research_lab_provider_outcome_contention_contract_v3"
            not in pre_head_contract.stderr
            or "does not exist" not in pre_head_contract.stderr
        ):
            raise PostgresContractProbeError(
                "pre-134 provider outcome head contract did not fail closed"
            )

        database.apply_migration(
            scripts / PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION
        )
        applied.append(PROVIDER_OUTCOME_HEAD_CONTENTION_MIGRATION)
        head_contention_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_provider_outcome_contention_contract_v3()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if head_contention_contract != {
            "schema_version": (
                "leadpoet.provider_outcome_contention_contract.v3"
            ),
            "lock_contention_status": "busy",
            "stale_lineage_status": "conflict",
            "candidate_checkpoint_hash": True,
            "conflict_head_checkpoint_row": "encrypted_or_null",
        }:
            raise PostgresContractProbeError(
                "post-134 provider outcome head contract differs"
            )
        database.apply_migration(scripts / ANCESTRY_CHECKPOINT_MIGRATION)
        applied.append(ANCESTRY_CHECKPOINT_MIGRATION)
        checkpoint_catalog = _relation_contract(database)
        checkpoint_relations = checkpoint_catalog["relations"]
        checkpoint_rpcs = set(checkpoint_catalog["rpcs"])
        required_checkpoint_relations = {
            "research_lab_attested_ancestry_checkpoints_v2",
            "research_lab_attested_ancestry_activations_v2",
        }
        if (
            not required_checkpoint_relations <= set(checkpoint_relations)
            or "persist_research_lab_ancestry_checkpoint_v2"
            not in checkpoint_rpcs
        ):
            raise PostgresContractProbeError(
                "post-136 ancestry checkpoint contract is incomplete"
            )
        database.apply_migration(
            scripts / ALLOCATION_SETTLEMENT_FRONTIER_MIGRATION
        )
        applied.append(ALLOCATION_SETTLEMENT_FRONTIER_MIGRATION)
        frontier_contract = _allocation_settlement_frontier_contract(
            database=database,
            fixture=fixture,
            verified_bundle=verified,
        )
        database.apply_migration(
            scripts / ANCESTRY_CHECKPOINT_BOOTSTRAP_PURPOSE_MIGRATION
        )
        applied.append(ANCESTRY_CHECKPOINT_BOOTSTRAP_PURPOSE_MIGRATION)
        bootstrap_purpose_constraint = database.psql(
            """
            SELECT public.research_lab_ancestry_checkpoint_bootstrap_contract_v2()
                   ::text;
            """,
            tuples_only=True,
        ).stdout.strip()
        bootstrap_purpose_contract = json.loads(
            bootstrap_purpose_constraint
        )
        if (
            bootstrap_purpose_contract.get("schema_version")
            != "leadpoet.ancestry_checkpoint_bootstrap_contract.v2"
            or bootstrap_purpose_contract.get("operation")
            != "ancestry_checkpoint_bootstrap_v2"
            or bootstrap_purpose_contract.get("purpose")
            != "research_lab.ancestry_checkpoint_bootstrap.v2"
            or bootstrap_purpose_contract.get("constraint_valid") is not True
            or "research_lab.ancestry_checkpoint_bootstrap.v2"
            not in str(
                bootstrap_purpose_contract.get("constraint_definition") or ""
            )
        ):
            raise PostgresContractProbeError(
                "post-138 ancestry checkpoint bootstrap purpose is incomplete"
            )
        database.apply_migration(
            scripts / ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_MIGRATION
        )
        applied.append(ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_MIGRATION)
        database.apply_migration(
            scripts
            / ALLOCATION_SETTLEMENT_FRONTIER_HISTORICAL_SOURCE_MIGRATION
        )
        applied.append(
            ALLOCATION_SETTLEMENT_FRONTIER_HISTORICAL_SOURCE_MIGRATION
        )
        database.apply_migration(
            scripts / ALLOCATION_SETTLEMENT_FRONTIER_SOURCE_CONTRACT_MIGRATION
        )
        applied.append(ALLOCATION_SETTLEMENT_FRONTIER_SOURCE_CONTRACT_MIGRATION)
        historical_source_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_allocation_frontier_historical_source_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if historical_source_contract != {
            "schema_version": (
                "leadpoet.allocation_frontier_historical_source_contract.v1"
            ),
            "persistence_rpc": (
                "persist_research_lab_allocation_frontier_bootstrap_v2"
            ),
            "settlement_frontier_compatibility": "missing_or_null",
        }:
            raise PostgresContractProbeError(
                "post-141 historical source capability is incomplete"
            )
        database.apply_migration(
            scripts / SOURCE_CATALOG_RESULT_REPLAY_MIGRATION
        )
        applied.append(SOURCE_CATALOG_RESULT_REPLAY_MIGRATION)
        source_catalog_replay_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_catalog_replay_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        source_catalog_constraints = source_catalog_replay_contract.get(
            "constraints"
        )
        source_catalog_constraint_definitions = (
            "\n".join(
                str(constraint.get("constraint_definition") or "")
                for constraint in source_catalog_constraints.values()
            )
            if isinstance(source_catalog_constraints, Mapping)
            else ""
        )
        if (
            source_catalog_replay_contract.get("schema_version")
            != "leadpoet.source_catalog_replay_contract.v2"
            or source_catalog_replay_contract.get("operation")
            != "source_add_catalog_snapshot_v2"
            or source_catalog_replay_contract.get("purpose")
            != "research_lab.source_add_catalog_snapshot.v2"
            or not isinstance(source_catalog_constraints, Mapping)
            or set(source_catalog_constraints)
            != {
                "research_lab_attested_execution_results_v2_operation_check",
                "research_lab_attested_execution_results_v2_purpose_check",
                "research_lab_attested_exec_results_v2_op_purpose_check",
            }
            or any(
                constraint.get("constraint_valid") is not True
                for constraint in source_catalog_constraints.values()
            )
            or "source_add_catalog_snapshot_v2"
            not in source_catalog_constraint_definitions
            or "research_lab.source_add_catalog_snapshot.v2"
            not in source_catalog_constraint_definitions
        ):
            raise PostgresContractProbeError(
                "post-142 source-catalog replay contract differs"
            )
        database.apply_migration(
            scripts / COMPACT_ANCESTRY_CHECKPOINT_MIGRATION
        )
        applied.append(COMPACT_ANCESTRY_CHECKPOINT_MIGRATION)
        compact_checkpoint_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_compact_checkpoint_graph_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if compact_checkpoint_contract != {
            "schema_version": (
                "leadpoet.compact_checkpoint_graph_contract.v1"
            ),
            "checkpoint_graph_schema_version": (
                "leadpoet.attested_checkpointed_receipt_graph.v4"
            ),
            "legacy_checkpoint_graph_schema_version": (
                "leadpoet.attested_checkpointed_receipt_graph.v3"
            ),
            "new_row_constraint_enabled": True,
            "historical_rows_append_only": True,
            "sidecar_trigger_enabled": True,
        }:
            raise PostgresContractProbeError(
                "post-143 compact ancestry checkpoint contract differs"
            )
        database.apply_migration(
            scripts / PROVIDER_PERSISTENCE_BATCH_MIGRATION
        )
        applied.append(PROVIDER_PERSISTENCE_BATCH_MIGRATION)
        database.apply_migration(
            scripts / SOURCE_ADD_ADMISSION_CONTROL_MIGRATION
        )
        applied.append(SOURCE_ADD_ADMISSION_CONTROL_MIGRATION)
        source_add_admission_control_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_admission_control_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_admission_control_contract != {
            "schema_version": (
                "leadpoet.source_add_admission_control_contract.v1"
            ),
            "control_row_present": True,
            "trigger_enabled": True,
            "pause_rpc": "research_lab_source_add_set_paused",
            "admission_trigger": "trg_source_add_work_admission_control",
        }:
            raise PostgresContractProbeError(
                "post-145 source-add admission-control contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_CATALOG_AUTH_METADATA_MIGRATION
        )
        # The migration drops and recreates one validated CHECK; rerunning it
        # must preserve both existing rows and the exact fail-closed contract.
        database.apply_migration(
            scripts / SOURCE_CATALOG_AUTH_METADATA_MIGRATION
        )
        applied.append(SOURCE_CATALOG_AUTH_METADATA_MIGRATION)
        source_catalog_auth_metadata_contract = json.loads(
            database.psql(
                """
                WITH safe AS (
                    SELECT jsonb_build_object(
                        'schema_version',
                        'leadpoet.source_add_catalog_snapshot.v2',
                        'provisioned_sources',
                        jsonb_build_array(jsonb_build_object(
                            'provision_doc', jsonb_build_object(
                                'provider_registry_entry',
                                jsonb_build_object(
                                    'id', 'builtwith_trends',
                                    'auth_kind', 'header',
                                    'auth_name', 'Authorization'
                                )
                            )
                        )),
                        'provisioned_sources_hash',
                        'sha256:' || repeat('1', 64),
                        'private_registry_rows', '[]'::jsonb,
                        'private_registry_rows_hash',
                        'sha256:' || repeat('2', 64),
                        'runtime_catalog', jsonb_build_object(
                            'routes', jsonb_build_array(jsonb_build_object(
                                'provider_id', 'builtwith_trends',
                                'auth_kind', 'header',
                                'auth_name', 'Authorization',
                                'request_headers', '{}'::jsonb
                            ))
                        ),
                        'runtime_catalog_hash',
                        'sha256:' || repeat('3', 64)
                    ) AS doc
                )
                SELECT jsonb_build_object(
                    'valid_correlated_auth_metadata',
                    public.research_lab_attested_execution_result_secret_free_v2(
                        'source_add_catalog_snapshot_v2', doc
                    ),
                    'generic_authorization_rejected',
                    NOT public.research_lab_attested_execution_result_secret_free_v2(
                        'research_lab_allocation',
                        '{"authorization":"forbidden"}'::jsonb
                    ),
                    'private_registry_authorization_rejected',
                    NOT public.research_lab_attested_execution_result_secret_free_v2(
                        'source_add_catalog_snapshot_v2',
                        jsonb_set(
                            doc,
                            '{private_registry_rows}',
                            '[{"authorization":"forbidden"}]'::jsonb
                        )
                    ),
                    'request_header_authorization_rejected',
                    NOT public.research_lab_attested_execution_result_secret_free_v2(
                        'source_add_catalog_snapshot_v2',
                        jsonb_set(
                            doc,
                            '{runtime_catalog,routes,0,request_headers}',
                            '{"Authorization":"forbidden"}'::jsonb
                        )
                    ),
                    'uncorrelated_auth_metadata_rejected',
                    NOT public.research_lab_attested_execution_result_secret_free_v2(
                        'source_add_catalog_snapshot_v2',
                        jsonb_set(
                            doc,
                            '{runtime_catalog,routes,0,provider_id}',
                            '"different_provider"'::jsonb
                        )
                    ),
                    'proxy_authorization_rejected',
                    NOT public.research_lab_attested_execution_result_secret_free_v2(
                        'source_add_catalog_snapshot_v2',
                        jsonb_set(
                            doc,
                            '{runtime_catalog,routes,0,request_headers}',
                            '{"Proxy-Authorization":"forbidden"}'::jsonb
                        )
                    ),
                    'constraint_valid', EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'public.research_lab_attested_execution_results_v2'
                                ::regclass
                          AND conname =
                            'research_lab_attested_execution_results_v2_result_doc_check'
                          AND convalidated
                    )
                )::text
                FROM safe;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_catalog_auth_metadata_contract != {
            "valid_correlated_auth_metadata": True,
            "generic_authorization_rejected": True,
            "private_registry_authorization_rejected": True,
            "request_header_authorization_rejected": True,
            "uncorrelated_auth_metadata_rejected": True,
            "proxy_authorization_rejected": True,
            "constraint_valid": True,
        }:
            raise PostgresContractProbeError(
                "post-147 SOURCE_ADD catalog auth metadata contract differs"
            )
        database.apply_migration(scripts / ATOMIC_CREDIT_RESUME_MIGRATION)
        applied.append(ATOMIC_CREDIT_RESUME_MIGRATION)
        atomic_credit_resume = _atomic_credit_resume_postgres_contract(database)
        database.apply_migration(
            scripts / COMPACT_WEIGHT_SETTLEMENT_AUTHORITY_MIGRATION
        )
        applied.append(COMPACT_WEIGHT_SETTLEMENT_AUTHORITY_MIGRATION)
        compact_weight_settlement_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_compact_weight_settlement_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if compact_weight_settlement_contract != {
            "schema_version": (
                "leadpoet.research_lab_compact_weight_settlement_contract.v1"
            ),
            "max_authority_bytes": 8_388_608,
            "size_constraint_valid": True,
            "append_only_trigger_enabled": True,
            "identity_unique_constraint_enabled": True,
            "row_level_security_enabled": True,
            "finalized_stage_supported": True,
        }:
            raise PostgresContractProbeError(
                "post-149 compact weight settlement contract differs"
            )
        database.apply_migration(
            scripts / ANCESTRY_DISCLOSURE_ROOT_FAST_PATH_MIGRATION
        )
        applied.append(ANCESTRY_DISCLOSURE_ROOT_FAST_PATH_MIGRATION)
        disclosure_lookup_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_ancestry_disclosure_lookup_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if disclosure_lookup_contract != {
            "schema_version": (
                "leadpoet.ancestry-disclosure-lookup-contract.v1"
            ),
            "persistence_rpc": "persist_research_lab_ancestry_checkpoint_v2",
            "root_witness_key": "root_receipt_hash",
            "non_root_fallback": "lineage_sequence_disclosure_scan",
        }:
            raise PostgresContractProbeError(
                "post-155 ancestry disclosure lookup contract differs"
            )
        database.apply_migration(scripts / PRODUCTION_PARITY_READER_MIGRATION)
        applied.append(PRODUCTION_PARITY_READER_MIGRATION)
        parity_reader_contract = json.loads(
            database.psql(
                """
                SELECT public.leadpoet_production_parity_reader_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            parity_reader_contract.get("schema_version")
            != "leadpoet.production-parity-reader-contract.v1"
            or parity_reader_contract.get("database_name")
            != "leadpoet_rehearsal"
            or parity_reader_contract.get("reader_role")
            != "leadpoet_parity_reader"
            or parity_reader_contract.get("superuser") is not False
            or parity_reader_contract.get("bypass_rls") is not True
            or parity_reader_contract.get("createdb") is not False
            or parity_reader_contract.get("createrole") is not False
            or parity_reader_contract.get("inherit") is not False
            or parity_reader_contract.get("replication") is not False
            or parity_reader_contract.get("connection_limit") != 2
            or parity_reader_contract.get("default_read_only") is not True
            or parity_reader_contract.get("membership_count") != 0
            or parity_reader_contract.get("schema_create_capable") is not False
            or parity_reader_contract.get("table_write_capable") is not False
            or parity_reader_contract.get("sequence_write_capable") is not False
        ):
            raise PostgresContractProbeError(
                "post-156 production parity reader contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_POST_ACCEPT_LEG1_MIGRATION
        )
        applied.append(SOURCE_ADD_POST_ACCEPT_LEG1_MIGRATION)
        post_accept_leg1_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_post_accept_leg1_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if post_accept_leg1_contract != {
            "schema_version": (
                "leadpoet.source_add_post_accept_leg1_contract.v1"
            ),
            "daily_cap": 10,
            "leg1_alpha_percent": 1.0,
            "leg1_reward_epochs": 20,
            "function_authority_sha256": (
                "sha256:035b4dc17bc8e8b63524df2c123892aa"
                "3ddaf0a01d08c69fc2d756921e8e96be"
            ),
            "functions": {
                "configure_probe_v2": True,
                "finalize_provision_v2": True,
                "reject_current_builtin_v2": True,
                "reserve_leg1_slot_v2": True,
                "finalize_leg1_v2": True,
                "finalize_provision_smoke_v2": True,
            },
            "triggers": {
                "acceptance": True,
                "eligible": True,
                "leg1_work": True,
                "leg1_slot": True,
                "leg1_obligation": True,
                "leg1_initial_event": True,
            },
            "permissions": {
                "service_role_exists": True,
                "v2_callable": True,
                "legacy_not_callable": True,
            },
        }:
            raise PostgresContractProbeError(
                "post-169 SOURCE_ADD post-accept Leg 1 contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_PROVIDER_ORIGIN_UNIQUENESS_MIGRATION
        )
        applied.append(SOURCE_ADD_PROVIDER_ORIGIN_UNIQUENESS_MIGRATION)
        source_add_provider_origin_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_provider_origin_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_provider_origin_contract != {
            "schema_version": (
                "leadpoet.source_add_provider_origin_contract.v1"
            ),
            "identity_version": "v1",
            "identity_scope": "normalized_exact_host",
            "admission_rpc": "research_lab_source_add_admit_v2",
            "recheck_rpc": "research_lab_source_add_requeue_provenance_v2",
            "owner_count": 0,
            "reserved_count": 0,
            "coverage_complete": True,
            "collision_free": True,
            "submission_trigger_enabled": True,
            "catalog_trigger_enabled": True,
            "provision_trigger_enabled": True,
            "terminal_release_trigger_enabled": True,
            "append_only_trigger_enabled": True,
            "row_level_security_enabled": True,
            "service_role_policy_enabled": True,
        }:
            raise PostgresContractProbeError(
                "post-170 SOURCE_ADD provider-origin contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_DUPLICATE_PRIVACY_MIGRATION
        )
        applied.append(SOURCE_ADD_DUPLICATE_PRIVACY_MIGRATION)
        source_add_duplicate_privacy_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_duplicate_privacy_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_duplicate_privacy_contract != {
            "schema_version": (
                "leadpoet.source_add_duplicate_privacy_contract.v1"
            ),
            "admission_rpc": "research_lab_source_add_admit_v3",
            "admission_signature": (
                "jsonb,text,text,text,text,text,integer,integer,integer,integer"
            ),
            "compatibility_rpc": "research_lab_source_add_admit_v2",
            "compatibility_signature": (
                "jsonb,text,text,text,text,text,integer,integer,integer"
            ),
            "compatibility_cooldown_seconds": 20,
            "cooldown_parameter_min_seconds": 1,
            "cooldown_parameter_max_seconds": 3600,
            "cooldown_clock": "clock_timestamp_after_advisory_locks",
            "cooldown_source": "durable_miner_provenance_work",
            "duplicate_precedes_cooldown": True,
            "lock_order": [
                "provider_origin_or_identity",
                "hotkey",
                "submission_or_work",
            ],
            "function_authority_sha256": (
                "sha256:26bf34c94725b855f81c2e48b6afbd72"
                "d68db36a4aeffb5642494a5da32233e0"
            ),
            "functions": {
                "admit_v1": True,
                "admit_v2_compatibility": True,
                "admit_v3": True,
                "provider_origin_hash_v1": True,
                "provider_origin_host_v1": True,
            },
            "permissions": {
                "service_role_exists": True,
                "v3_service_role_callable": True,
                "v2_service_role_callable": True,
                "contract_service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        }:
            raise PostgresContractProbeError(
                "post-171 SOURCE_ADD duplicate-privacy contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_CLAIM_CONTROL_MIGRATION
        )
        applied.append(SOURCE_ADD_CLAIM_CONTROL_MIGRATION)
        source_add_claim_control_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_claim_control_contract_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_claim_control_contract != {
            "schema_version": (
                "leadpoet.source_add_claim_control_contract.v1"
            ),
            "control_lock": "source-add-control",
            "pause_rpc": "research_lab_source_add_set_paused",
            "pause_signature": "boolean,text,text",
            "claim_rpc": "research_lab_source_add_claim_work",
            "claim_signature": "text,integer",
            "acquire_guard_rpc": (
                "research_lab_source_add_acquire_restart_guard_v1"
            ),
            "acquire_guard_signature": "text,text,bigint,integer,text",
            "guard_state_rpc": (
                "research_lab_source_add_restart_guard_state_v1"
            ),
            "guard_state_signature": "",
            "release_guard_rpc": (
                "research_lab_source_add_release_restart_guard_v1"
            ),
            "release_guard_signature": "text,text,bigint,text",
            "guard_state_result_fields": [
                "schema_version",
                "paused",
                "guard_active",
                "guard_commitment",
                "owner_commitment",
                "guard_generation",
                "owner_generation_commitment",
                "guard_expires_at",
            ],
            "acquire_guard_result_fields": [
                "schema_version",
                "paused",
                "guard_active",
                "guard_commitment",
                "owner_commitment",
                "guard_generation",
                "owner_generation_commitment",
                "guard_expires_at",
            ],
            "release_guard_result_fields": [
                "schema_version",
                "released",
                "paused",
                "guard_active",
                "guard_generation",
                "owner_generation_commitment",
            ],
            "guard_id_format": "^source_add_restart_guard:[0-9a-f]{64}$",
            "guard_commitment": "sha256_utf8_guard_id",
            "owner_id_format": "^source_add_restart_owner:[0-9a-f]{64}$",
            "owner_commitment": "sha256_utf8_owner_id",
            "owner_generation_commitment": (
                "sha256_utf8_owner_commitment_colon_decimal_generation"
            ),
            "guard_lease_min_seconds": 60,
            "guard_lease_max_seconds": 14400,
            "active_guard_replay_extends_lease": True,
            "acquire_compare_and_swap": "expected_generation",
            "different_owner_takeover_increments_generation": True,
            "expired_reacquire_increments_generation": True,
            "generation_retained_after_release": True,
            "resume_requires_guard_clear": True,
            "expired_guard_recovery": (
                "explicit_reacquire_then_exact_release"
            ),
            "release_keeps_paused": True,
            "restart_quiescence_rpc": (
                "research_lab_source_add_restart_quiescence_v1"
            ),
            "restart_quiescence_signature": "text,text,bigint",
            "restart_quiescence_schema_version": (
                "leadpoet.source_add_restart_quiescence.v1"
            ),
            "restart_quiescence_result_fields": [
                "schema_version",
                "paused",
                "guard_active",
                "guard_matches",
                "owner_matches",
                "generation_matches",
                "guard_commitment",
                "owner_commitment",
                "guard_generation",
                "owner_generation_commitment",
                "guard_expires_at",
                "leased_work_count",
                "quiescent",
            ],
            "lock_before_paused_read": True,
            "leased_scope": "all_leased_regardless_of_expiry",
            "migration_requires_paused": True,
            "migration_requires_zero_leased": True,
            "function_authority_sha256": (
                "sha256:890a1e42b6dd28eb1c8515c3b8c33d31"
                "a9974058fbd2c43393bb0880c0ca21e6"
            ),
            "functions": {
                "admission_guard": True,
                "acquire_restart_guard_v1": True,
                "claim_work": True,
                "pause": True,
                "release_restart_guard_v1": True,
                "restart_guard_state_v1": True,
                "restart_quiescence_v1": True,
            },
            "permissions": {
                "service_role_exists": True,
                "acquire_guard_service_role_callable": True,
                "claim_service_role_callable": True,
                "pause_service_role_callable": True,
                "quiescence_service_role_callable": True,
                "release_guard_service_role_callable": True,
                "guard_state_service_role_callable": True,
                "contract_service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        }:
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD claim-control contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_LEG1_RELEASE_POLICY_MIGRATION
        )
        applied.append(SOURCE_ADD_LEG1_RELEASE_POLICY_MIGRATION)
        source_add_leg1_release_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_post_accept_leg1_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_leg1_release_contract != {
            "schema_version": (
                "leadpoet.source_add_post_accept_leg1_contract.v2"
            ),
            "daily_cap": 50,
            "leg1_alpha_percent": 0.2,
            "leg1_reward_epochs": 20,
            "function_authority_sha256": (
                "sha256:6c09aa3c6b82b3fe666c6739c4f71a51"
                "ea8d6445e3e5a52ab08a4e2f8fa8d9ec"
            ),
            "functions": {
                "configure_probe_v2": True,
                "finalize_provision_v2": True,
                "reject_current_builtin_v2": True,
                "post_accept_contract_v1": True,
                "reserve_leg1_slot_v2": True,
                "finalize_leg1_v2": True,
                "reserve_leg1_slot_v3": True,
                "finalize_leg1_v3": True,
                "finalize_provision_smoke_v2": True,
            },
            "triggers": {
                "acceptance": True,
                "eligible": True,
                "leg1_work": True,
                "leg1_slot": True,
                "leg1_obligation": True,
                "leg1_initial_event": True,
            },
            "permissions": {
                "service_role_exists": True,
                "candidate_callable": True,
                "rollback_v2_callable": True,
                "legacy_not_callable": True,
            },
        }:
            raise PostgresContractProbeError(
                "post-173 SOURCE_ADD Leg 1 release contract differs"
            )
        rehearsal_guard_id = "source_add_restart_guard:" + "d" * 64
        rehearsal_guard_commitment = "sha256:" + hashlib.sha256(
            rehearsal_guard_id.encode("utf-8")
        ).hexdigest()
        rehearsal_owner_id = "source_add_restart_owner:" + "e" * 64
        rehearsal_owner_commitment = "sha256:" + hashlib.sha256(
            rehearsal_owner_id.encode("utf-8")
        ).hexdigest()
        rehearsal_owner_generation_commitment = (
            "sha256:"
            + hashlib.sha256(
                f"{rehearsal_owner_commitment}:1".encode("utf-8")
            ).hexdigest()
        )
        source_add_restart_guard_state = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_restart_guard_state_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_restart_guard_state != {
            "schema_version": "leadpoet.source_add_restart_guard_state.v1",
            "paused": True,
            "guard_active": False,
            "guard_commitment": "",
            "owner_commitment": "",
            "guard_generation": 0,
            "owner_generation_commitment": "",
            "guard_expires_at": None,
        }:
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD restart guard initial state differs"
            )
        source_add_restart_guard = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_acquire_restart_guard_v1(
                    :'guard_id', :'owner_id', 0, 14400,
                    'operator:restart-rehearsal'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            source_add_restart_guard.get("schema_version")
            != "leadpoet.source_add_restart_guard.v1"
            or source_add_restart_guard.get("paused") is not True
            or source_add_restart_guard.get("guard_active") is not True
            or source_add_restart_guard.get("guard_commitment")
            != rehearsal_guard_commitment
            or source_add_restart_guard.get("owner_commitment")
            != rehearsal_owner_commitment
            or source_add_restart_guard.get("guard_generation") != 1
            or source_add_restart_guard.get(
                "owner_generation_commitment"
            )
            != rehearsal_owner_generation_commitment
            or not isinstance(
                source_add_restart_guard.get("guard_expires_at"), str
            )
        ):
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD restart guard differs"
            )
        source_add_restart_quiescence = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_restart_quiescence_v1(
                    :'guard_id', :'owner_id', 1
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            source_add_restart_quiescence.get("schema_version")
            != "leadpoet.source_add_restart_quiescence.v1"
            or source_add_restart_quiescence.get("paused") is not True
            or source_add_restart_quiescence.get("guard_active") is not True
            or source_add_restart_quiescence.get("guard_matches") is not True
            or source_add_restart_quiescence.get("owner_matches") is not True
            or source_add_restart_quiescence.get("generation_matches")
            is not True
            or source_add_restart_quiescence.get("guard_commitment")
            != rehearsal_guard_commitment
            or source_add_restart_quiescence.get("owner_commitment")
            != rehearsal_owner_commitment
            or source_add_restart_quiescence.get("guard_generation") != 1
            or source_add_restart_quiescence.get(
                "owner_generation_commitment"
            )
            != rehearsal_owner_generation_commitment
            or source_add_restart_quiescence.get("leased_work_count") != 0
            or source_add_restart_quiescence.get("quiescent") is not True
        ):
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD restart quiescence differs"
            )
        source_add_restart_guard_release = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_release_restart_guard_v1(
                    :'guard_id', :'owner_id', 1,
                    'operator:restart-rehearsal'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_restart_guard_release != {
            "schema_version": "leadpoet.source_add_restart_guard_release.v1",
            "released": True,
            "paused": True,
            "guard_active": False,
            "guard_generation": 1,
            "owner_generation_commitment": (
                rehearsal_owner_generation_commitment
            ),
        }:
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD restart guard release differs"
            )
        source_add_restart_guard_state = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_restart_guard_state_v1()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            source_add_restart_guard_state.get("paused") is not True
            or source_add_restart_guard_state.get("guard_active") is not False
            or source_add_restart_guard_state.get("guard_commitment") != ""
            or source_add_restart_guard_state.get("owner_commitment") != ""
            or source_add_restart_guard_state.get("guard_generation") != 1
            or source_add_restart_guard_state.get(
                "owner_generation_commitment"
            )
            != ""
            or source_add_restart_guard_state.get("guard_expires_at")
            is not None
        ):
            raise PostgresContractProbeError(
                "post-172 SOURCE_ADD restart guard released state differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_RESTART_STATE_RESTORE_MIGRATION
        )
        applied.append(SOURCE_ADD_RESTART_STATE_RESTORE_MIGRATION)
        source_add_restart_state_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_claim_control_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        expected_restart_state_contract = {
            "schema_version": "leadpoet.source_add_claim_control_contract.v2",
            "control_lock": "source-add-control",
            "pause_rpc": "research_lab_source_add_set_paused",
            "pause_signature": "boolean,text,text",
            "claim_rpc": "research_lab_source_add_claim_work",
            "claim_signature": "text,integer",
            "acquire_guard_rpc": (
                "research_lab_source_add_acquire_restart_guard_v2"
            ),
            "acquire_guard_signature": "text,text,bigint,integer,text",
            "guard_state_rpc": (
                "research_lab_source_add_restart_guard_state_v2"
            ),
            "guard_state_signature": "",
            "release_guard_rpc": (
                "research_lab_source_add_release_restart_guard_v2"
            ),
            "release_guard_signature": "text,text,bigint,text",
            "restart_quiescence_rpc": (
                "research_lab_source_add_restart_quiescence_v1"
            ),
            "restart_quiescence_signature": "text,text,bigint",
            "guard_state_result_fields": [
                "schema_version",
                "paused",
                "guard_active",
                "guard_commitment",
                "owner_commitment",
                "guard_generation",
                "owner_generation_commitment",
                "guard_expires_at",
                "restore_paused",
            ],
            "acquire_guard_result_fields": [
                "schema_version",
                "paused",
                "guard_active",
                "guard_commitment",
                "owner_commitment",
                "guard_generation",
                "owner_generation_commitment",
                "guard_expires_at",
                "restore_paused",
            ],
            "release_guard_result_fields": [
                "schema_version",
                "released",
                "paused",
                "guard_active",
                "guard_generation",
                "owner_generation_commitment",
                "restored_pre_restart_state",
            ],
            "restore_state_column": "restart_guard_restore_paused",
            "acquire_captures_pre_restart_paused": True,
            "renewal_preserves_restore_state": True,
            "expired_takeover_preserves_restore_state": True,
            "operator_pause_wins": True,
            "release_restores_pre_restart_state": True,
            "failed_restart_keeps_paused": True,
            "rollback_v1_contract_schema_version": (
                "leadpoet.source_add_claim_control_contract.v1"
            ),
            "rollback_v1_contract_sha256": (
                SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256
            ),
            "migration_requires_paused": True,
            "migration_requires_zero_leased": True,
            "migration_requires_guard_clear": True,
            "function_authority_sha256": (
                SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
            ),
            "functions": {
                "admission_guard": True,
                "acquire_restart_guard_v1": True,
                "acquire_restart_guard_v2": True,
                "claim_work": True,
                "pause": True,
                "release_restart_guard_v1": True,
                "release_restart_guard_v2": True,
                "restart_guard_state_v1": True,
                "restart_guard_state_v2": True,
                "restart_quiescence_v1": True,
                "restore_trigger_v2": True,
            },
            "permissions": {
                "service_role_exists": True,
                "service_role_callable": True,
                "anon_callable": False,
                "authenticated_callable": False,
            },
        }
        if source_add_restart_state_contract != expected_restart_state_contract:
            raise PostgresContractProbeError(
                "post-174 SOURCE_ADD restart-state contract differs"
            )

        # A paused pre-state must remain paused after exact release.
        paused_guard = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_acquire_restart_guard_v2(
                    :'guard_id', :'owner_id', 1, 14400,
                    'operator:restart-rehearsal-v2-paused'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if paused_guard.get("restore_paused") is not True:
            raise PostgresContractProbeError(
                "post-174 paused SOURCE_ADD restart state was not captured"
            )
        paused_release = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_release_restart_guard_v2(
                    :'guard_id', :'owner_id', 2,
                    'operator:restart-rehearsal-v2-paused'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            paused_release.get("paused") is not True
            or paused_release.get("restored_pre_restart_state") is not True
        ):
            raise PostgresContractProbeError(
                "post-174 paused SOURCE_ADD restart state was not restored"
            )

        # An active pre-state is guarded as paused, then restored active only
        # by the exact owner/generation release.
        database.psql(
            """
            SELECT public.research_lab_source_add_set_paused(
                FALSE, 'restart_rehearsal_active_prestate',
                'operator:restart-rehearsal-v2-active'
            );
            """
        )
        active_guard = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_acquire_restart_guard_v2(
                    :'guard_id', :'owner_id', 2, 14400,
                    'operator:restart-rehearsal-v2-active'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            active_guard.get("paused") is not True
            or active_guard.get("restore_paused") is not False
        ):
            raise PostgresContractProbeError(
                "post-174 active SOURCE_ADD restart state was not captured"
            )
        active_release = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_release_restart_guard_v2(
                    :'guard_id', :'owner_id', 3,
                    'operator:restart-rehearsal-v2-active'
                )::text;
                """
                .replace(":'guard_id'", "'" + rehearsal_guard_id + "'")
                .replace(":'owner_id'", "'" + rehearsal_owner_id + "'"),
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            active_release.get("paused") is not False
            or active_release.get("restored_pre_restart_state") is not True
        ):
            raise PostgresContractProbeError(
                "post-174 active SOURCE_ADD restart state was not restored"
            )
        restored_state = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_restart_guard_state_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if (
            restored_state.get("paused") is not False
            or restored_state.get("guard_active") is not False
            or restored_state.get("restore_paused") is not None
            or restored_state.get("guard_generation") != 3
        ):
            raise PostgresContractProbeError(
                "post-174 SOURCE_ADD restored durable state differs"
            )
        database.psql(
            """
            SELECT public.research_lab_source_add_set_paused(
                TRUE, 'restart_rehearsal_provenance_leg1_migration',
                'operator:restart-rehearsal-v2-provenance-leg1'
            );
            """
        )
        database.apply_migration(
            scripts / SOURCE_ADD_PROVENANCE_LEG1_MIGRATION
        )
        applied.append(SOURCE_ADD_PROVENANCE_LEG1_MIGRATION)
        source_add_provenance_leg1_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_post_accept_leg1_contract_v3()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_provenance_leg1_contract != {
            "schema_version": (
                "leadpoet.source_add_post_accept_leg1_contract.v3"
            ),
            "daily_cap": 50,
            "leg1_alpha_percent": 0.2,
            "leg1_reward_epochs": 20,
            "approval_boundary": "provenance_precheck_passed",
            "backfill_policy": "all_exact_attested_provenance",
            "public_trigger_fields": [
                "precheck_status",
                "provenance_artifact_hash",
                "provenance_precheck_passed",
                "provenance_receipt_hash",
                "provenance_result_hash",
                "submission_id",
            ],
            "authority_view": (
                "research_lab_source_add_provenance_leg1_authority_v1"
            ),
            "function_authority_sha256": (
                SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
            ),
            "trigger_authority_sha256": (
                SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
            ),
            "view_authority_sha256": (
                SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256
            ),
            "functions": {
                "configure_probe_v3": True,
                "enqueue_leg1_after_provenance_v1": True,
                "enqueue_provision_smoke_v2": True,
                "finalize_leg1_v4": True,
                "finalize_provision_smoke_v3": True,
                "finalize_provision_v3": True,
                "reject_current_builtin_v3": True,
                "reconcile_provenance_leg1_v1": True,
                "reserve_leg1_slot_v4": True,
            },
            "triggers": {
                "automatic_enqueue": True,
                "eligible_v2": True,
                "eligible_v3": True,
                "leg1_initial_event_v3": True,
                "leg1_obligation_v3": True,
                "leg1_slot_v3": True,
                "leg1_work_v3": True,
            },
            "columns": {
                "intent_approval_kind": True,
                "intent_provenance_artifact_hash": True,
                "intent_provenance_receipt_hash": True,
                "slot_approval_kind": True,
            },
            "permissions": {
                "service_role_exists": True,
                "candidate_callable": True,
                "internal_not_callable": True,
                "rollback_v2_callable": True,
            },
        }:
            raise PostgresContractProbeError(
                "post-175 SOURCE_ADD provenance Leg 1 contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION
        )
        applied.append(SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_MIGRATION)
        database.apply_migration(
            scripts / SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION
        )
        applied.append(SOURCE_ADD_PROVENANCE_AUTHORITY_ACL_MIGRATION)
        source_add_provenance_origin_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_post_accept_leg1_contract_v4()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        expected_provenance_origin_contract = dict(
            source_add_provenance_leg1_contract
        )
        expected_provenance_origin_contract.update(
            {
                "schema_version": (
                    "leadpoet.source_add_post_accept_leg1_contract.v4"
                ),
                "required_migration": (
                    "scripts/176-research-lab-source-add-provenance-origin-"
                    "repair.sql"
                ),
                "backfill_policy": (
                    "earliest_exact_attested_provenance_per_provider_origin"
                ),
                "provider_origin_scope": "normalized_exact_host",
                "provider_origin_winner_order": [
                    "provenance_created_at",
                    "submission_id",
                ],
                "cancelled_intents_are_authority": False,
                "view_authority_sha256": (
                    SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256
                ),
                "repair_function_authority_sha256": (
                    SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256
                ),
            }
        )
        if (
            source_add_provenance_origin_contract
            != expected_provenance_origin_contract
        ):
            raise PostgresContractProbeError(
                "post-176 SOURCE_ADD provenance-origin contract differs"
            )
        database.apply_migration(
            scripts / SOURCE_ADD_MINER_STATUS_MIGRATION
        )
        applied.append(SOURCE_ADD_MINER_STATUS_MIGRATION)
        source_add_miner_status_contract = json.loads(
            database.psql(
                """
                SELECT public.research_lab_source_add_miner_status_contract_v1()
                       ::TEXT;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if source_add_miner_status_contract != {
            "schema_version": (
                "leadpoet.source_add_miner_status_contract.v1"
            ),
            "view_name": "research_lab_source_add_miner_status_v1",
            "page_rpc": "research_lab_source_add_miner_status_page_v1",
            "page_signature": "text,text,integer",
            "view_columns": [
                "schema_version",
                "submission_id",
                "miner_hotkey",
                "source_name",
                "submitted_at",
                "updated_at",
                "decision_status",
                "decision_reason_code",
                "decision_reason",
                "reward_status",
                "alpha_percent",
                "reward_epochs",
                "start_epoch",
                "end_epoch",
            ],
            "view_security_invoker": True,
            "view_security_barrier": True,
            "page_security_invoker": True,
            "page_stable": True,
            "view_authority_sha256": (
                SOURCE_ADD_MINER_STATUS_VIEW_AUTHORITY_SHA256
            ),
            "page_authority_sha256": (
                SOURCE_ADD_MINER_STATUS_PAGE_AUTHORITY_SHA256
            ),
            "contract_authority_sha256": (
                SOURCE_ADD_MINER_STATUS_CONTRACT_AUTHORITY_SHA256
            ),
            "permissions": {
                "view_service_role_select": True,
                "view_anon_select": False,
                "view_authenticated_select": False,
                "view_public_select": False,
                "page_service_role_callable": True,
                "page_anon_callable": False,
                "page_authenticated_callable": False,
                "page_public_callable": False,
                "contract_service_role_callable": True,
                "contract_anon_callable": False,
                "contract_authenticated_callable": False,
            },
        }:
            raise PostgresContractProbeError(
                "post-178 SOURCE_ADD miner status contract differs"
            )
        # The active agent competition schema follows SOURCE_ADD migration 178.
        for migration in LAB_ARENA_MIGRATIONS:
            database.apply_migration(scripts / migration)
            applied.append(migration)
        lab_arena_schema_contract = json.loads(
            database.psql(
                """
                SELECT public.lab_arena_schema_version_v1()::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        if lab_arena_schema_contract != {
            "schema_version": "leadpoet.lab_arena.schema_version.v1",
            "version": 184,
        }:
            raise PostgresContractProbeError(
                "post-184 Lab Arena schema contract differs"
            )
        allocation_frontier_bootstrap_contract = (
            _allocation_settlement_frontier_bootstrap_contract(
                database=database,
                fixture=fixture,
                verified_bundle=verified,
            )
        )
        allocation_frontier_bootstrap_schema = json.loads(
            database.psql(
                """
                SELECT public.research_lab_allocation_frontier_bootstrap_contract_v2()
                       ::text;
                """,
                tuples_only=True,
            ).stdout.strip()
        )
        bootstrap_constraints = allocation_frontier_bootstrap_schema.get(
            "constraints"
        )
        if (
            allocation_frontier_bootstrap_schema.get("schema_version")
            != "leadpoet.allocation_frontier_bootstrap_contract.v2"
            or allocation_frontier_bootstrap_schema.get("operation")
            != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION
            or allocation_frontier_bootstrap_schema.get("purpose")
            != ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE
            or allocation_frontier_bootstrap_schema.get("persistence_rpc")
            != "persist_research_lab_allocation_frontier_bootstrap_v2"
            or not isinstance(bootstrap_constraints, Mapping)
            or len(bootstrap_constraints) != 4
            or any(
                constraint.get("constraint_valid") is not True
                for constraint in bootstrap_constraints.values()
            )
        ):
            raise PostgresContractProbeError(
                "post-139 allocation frontier bootstrap schema is incomplete"
            )
        provider_outcome_append = _provider_outcome_append_contract(database)
        provider_persistence_batch = _provider_persistence_batch_contract(
            database
        )
        historical_compute_seed_rows = (
            _historical_compute_allocation_seed_rows(
                database=database,
                source_root=args.source_root,
                candidate_sha=args.candidate_sha,
                source_epoch=_historical_compute_source_epoch(
                    (
                        int(prior_verified["epoch_id"]),
                        int(verified["epoch_id"]),
                    )
                ),
                netuid=int(verified["netuid"]),
                coordinator_release_identity=coordinator_release_identity,
            )
        )
        graph_seed_rows = _settlement_graph_seed_rows(
            settlement_fixture_rows
        )
        merged_seed_rows = {
            table: [
                *graph_seed_rows.get(table, ()),
                *historical_compute_seed_rows.get(table, ()),
            ]
            for table in sorted(
                set(graph_seed_rows) | set(historical_compute_seed_rows)
            )
        }
        merged_seed_rows["research_lab_finalized_allocation_epochs_v2"] = [
            prior_view_row,
            view_row,
        ]
        try:
            validate_rehearsal_finalized_authority_epochs(merged_seed_rows)
        except ValueError as exc:
            raise PostgresContractProbeError(str(exc)) from exc

        tampered = copy.deepcopy(view_row)
        tampered["finalization_doc"]["weight_receipt_hash"] = "sha256:" + "0" * 64
        try:
            _preliminary_finalized_bundle_authority_v1(tampered)
        except ChampionSettlementV2Error as exc:
            if "weight_receipt_hash" not in str(exc):
                raise PostgresContractProbeError(
                    "tampered authority failed for the wrong reason: %s" % exc
                ) from exc
        else:
            raise PostgresContractProbeError(
                "tampered settlement authority was accepted"
            )

        if tuple(applied) != EXPECTED_APPLIED_MIGRATIONS:
            raise PostgresContractProbeError(
                "applied migration sequence differs from the rehearsal contract"
            )
        contract = _relation_contract(database)
        view_columns = contract["relations"][
            "research_lab_finalized_allocation_epochs_v2"
        ]["columns"]
        if tuple(view_columns) != EXPECTED_FINALIZED_VIEW_COLUMNS:
            raise PostgresContractProbeError(
                "catalog and finalized view projections differ"
            )
        return {
            "schema_version": "leadpoet.restart_rehearsal.postgres_contract.v1",
            "candidate_sha": args.candidate_sha,
            "applied_migrations": applied,
            "relations": contract["relations"],
            "rpcs": contract["rpcs"],
            "maintenance_lease": maintenance_lease,
            "atomic_credit_resume": atomic_credit_resume,
            "compact_weight_settlement_contract": (
                compact_weight_settlement_contract
            ),
            "lab_arena_schema_contract": lab_arena_schema_contract,
            "source_add_miner_status_contract": (
                source_add_miner_status_contract
            ),
            "checks": {
                name: True for name in EXPECTED_POSTGRES_CONTRACT_CHECKS
            },
            "seed_rows": merged_seed_rows,
            "measured_settlement": measured_settlement,
            "champion_lifetime_credit": {
                "policy": "accelerated_lifetime_cap_v1",
                "credit_count": len(credit_rows),
                "credit_hashes": expected_credit_hashes,
                "historical_contract": historical_contract,
                "persistence": first_persistence,
            },
            "allocation_settlement_frontier": frontier_contract,
            "allocation_settlement_frontier_bootstrap": (
                allocation_frontier_bootstrap_contract
            ),
            "provider_outcome_append": provider_outcome_append,
            "provider_persistence_batch": provider_persistence_batch,
            "provider_outcome_contention_contract": head_contention_contract,
            "required_schema_declarations": declaration_counts,
        }
    finally:
        database.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--release-build-input", type=Path, required=True)
    parser.add_argument("--epoch-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.candidate_sha):
        raise SystemExit("candidate SHA must be lowercase full-length hex")
    if args.epoch_id is None:
        args.epoch_id = load_rehearsal_current_settlement_epoch_id(
            args.source_root
        )
    args.state_root.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_probe(args)
    except Exception as exc:
        diagnostic = {
            "schema_version": "leadpoet.restart_rehearsal.failure.v1",
            "stage": "migration_backed_v2_settlement_contract",
            "candidate_sha": args.candidate_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(
            "REHEARSAL_POSTGRES_CONTRACT_ERROR "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        raise
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output.chmod(0o600)
    print(
        "REHEARSAL_POSTGRES_CONTRACT_OK "
        + json.dumps(result["checks"], sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
