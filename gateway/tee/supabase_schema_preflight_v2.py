"""Read-only PostgREST schema gate for the Arena gateway release."""
from __future__ import annotations
import json
from typing import Any, Dict, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CODE_REVIEW_MIGRATION = "scripts/207-lab-arena-code-review.sql"
PARTICIPATION_MIGRATION = "scripts/20260911173147_lab_arena_validator_participation.sql"
PRIVATE_ARENA_MIGRATIONS = frozenset({CODE_REVIEW_MIGRATION, PARTICIPATION_MIGRATION})
REQUIRED_SUPABASE_V2_SCHEMA = (
    (PARTICIPATION_MIGRATION, "lab_arena_runs", ("runner_hotkey", "participation_accepted_at")),
    (CODE_REVIEW_MIGRATION, "lab_arena_submissions", ("submission_id", "code_review_status", "code_review_doc")),
    ("scripts/125-research-lab-artifact-key-lineage.sql", "research_lab_provider_evidence_cache_v2", ("artifact_master_key_ref_hash",)),
    ("scripts/101-stateful-subnet-epoch-authority.sql", "research_lab_stateful_subnet_epoch_cutovers_v1", ("mapping_hash", "network_genesis_hash", "netuid", "first_subnet_epoch_index", "first_settlement_epoch_id")),
    ("scripts/101-stateful-subnet-epoch-authority.sql", "research_lab_stateful_subnet_epoch_cutover_state_v1", ("lifecycle_state", "mapping_hash", "network_genesis_hash", "netuid", "updated_at")),
    ("scripts/197-lab-arena-reward-chain-scope.sql", "lab_arena_reward_basis_v1", ("round_id", "effective_reward_epoch", "reward_basis_hash", "reward_basis_doc", "signing_key_doc", "king_outcome", "king_hotkey", "king_start_epoch", "published_at", "arena_network_name", "arena_netuid")),
    ("scripts/202-arena-accepted-weight-state.sql", "lab_arena_accepted_weight_states", ("network", "netuid", "epoch", "state_hash", "state_doc", "created_at")),
    ("scripts/202-arena-accepted-weight-state.sql", "lab_arena_chain_outcomes", ("network", "netuid", "epoch", "validator_hotkey", "request_id", "extrinsic_hash", "outcome_doc", "created_at")),
)
REQUIRED_SUPABASE_V2_RPCS = (
    (PARTICIPATION_MIGRATION, "lab_arena_has_recent_participation_v1"),
    (CODE_REVIEW_MIGRATION, "lab_arena_code_review_schema_v1"),
    (CODE_REVIEW_MIGRATION, "lab_arena_begin_submission_review"),
    (CODE_REVIEW_MIGRATION, "lab_arena_finish_submission_review"),
    ("scripts/144-research-lab-provider-persistence-batches.sql", "put_research_lab_provider_evidence_cache_v2"),
    ("scripts/144-research-lab-provider-persistence-batches.sql", "research_lab_provider_persistence_batch_contract_v1"),
    ("scripts/101-stateful-subnet-epoch-authority.sql", "research_lab_stateful_subnet_epoch_cutover_public_state_v1"),
    ("scripts/197-lab-arena-reward-chain-scope.sql", "lab_arena_schema_version_v1"),
    ("scripts/202-arena-accepted-weight-state.sql", "lab_arena_publish_weight_state_v1"),
    ("scripts/202-arena-accepted-weight-state.sql", "lab_arena_record_chain_outcome_v1"),
    ("scripts/202-arena-accepted-weight-state.sql", "lab_arena_weight_state_schema_v1"),
    ("scripts/203-retire-legacy-incentive-weight-bridge.sql", "lab_arena_incentive_retirement_schema_v1"),
)
SCHEMA_CAPABILITIES = (
    ("lab_arena_code_review_schema_v1", {
        "schema_version": "leadpoet.lab_arena.code_review.v1",
        "version": 207,
        "claim_ttl_seconds": 600,
        "retry_backoff_seconds": 60,
        "max_attempts": 3,
    }),
    ("lab_arena_schema_version_v1", {"schema_version": "leadpoet.lab_arena.schema_version.v1", "version": 197}),
    ("lab_arena_weight_state_schema_v1", {"schema_version": "leadpoet.lab_arena.weight_state_schema.v1", "version": 202}),
    ("lab_arena_incentive_retirement_schema_v1", {"schema_version": "leadpoet.lab_arena.incentive_retirement_schema.v1", "version": 203}),
)
POSTGRES_IDENTIFIER_MAX_BYTES = 63
PRIVATE_ARENA_CAPABILITIES = frozenset({"lab_arena_code_review_schema_v1"})

class SupabaseSchemaPreflightV2Error(RuntimeError):
    """The selected release cannot use the live PostgREST schema."""

def _read_json(request: Request, *, opener: Any, timeout_seconds: float, label: str) -> Any:
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status, encoded = int(response.getcode()), response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(f"{label} failed (HTTP {exc.code})") from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(f"{label} failed") from exc
    if not 200 <= status < 300:
        raise SupabaseSchemaPreflightV2Error(f"{label} failed (HTTP {status})")
    try:
        return json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(f"{label} response is invalid") from exc

def verify_required_supabase_v2_schema(parent_environment: Mapping[str, str], *, opener: Any = urlopen, timeout_seconds: float = 10.0, defer_incentive_retirement: bool = False) -> Dict[str, Any]:
    """Verify only live Arena and generic scoring dependencies."""
    supabase_url = str(parent_environment.get("SUPABASE_URL") or "").rstrip("/")
    arena_supabase_url = str(
        parent_environment.get("LAB_ARENA_SUPABASE_URL") or supabase_url
    ).rstrip("/")
    service_role_key = str(parent_environment.get("SUPABASE_SERVICE_ROLE_KEY") or "")
    if not supabase_url or not service_role_key:
        raise SupabaseSchemaPreflightV2Error("prepared parent environment lacks Supabase V2 schema credentials")
    headers = {"Accept": "application/json", "Authorization": f"Bearer {service_role_key}", "apikey": service_role_key}
    arena_service_key = str(parent_environment.get("LAB_ARENA_SERVICE_KEY") or "").strip()
    arena_service_jwt = str(parent_environment.get("LAB_ARENA_SERVICE_JWT") or "").strip()
    arena_anon_key = str(parent_environment.get("LAB_ARENA_SUPABASE_ANON_KEY") or "").strip()
    if bool(arena_service_key) == bool(arena_service_jwt):
        raise SupabaseSchemaPreflightV2Error(
            "prepared parent environment lacks one Arena schema credential"
        )
    if arena_service_key:
        if not arena_service_key.startswith("sb_secret_"):
            raise SupabaseSchemaPreflightV2Error(
                "prepared parent environment has an invalid Arena schema credential"
            )
        arena_headers = {"Accept": "application/json", "apikey": arena_service_key}
    else:
        if not arena_anon_key or arena_service_jwt.count(".") != 2:
            raise SupabaseSchemaPreflightV2Error(
                "prepared parent environment has an invalid Arena schema credential"
            )
        arena_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {arena_service_jwt}",
            "apikey": arena_anon_key,
        }
    migrations = set()
    for migration, table, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        probe_url = arena_supabase_url if migration in PRIVATE_ARENA_MIGRATIONS else supabase_url
        probe_headers = arena_headers if migration in PRIVATE_ARENA_MIGRATIONS else headers
        request = Request(f"{probe_url}/rest/v1/{table}?{urlencode({'select': ','.join(columns), 'limit': '0'})}", headers=probe_headers)
        try:
            with opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                response.read(1)
        except HTTPError as exc:
            raise SupabaseSchemaPreflightV2Error(f"required schema is unavailable for {table}; apply {migration} before restart (HTTP {exc.code})") from exc
        except Exception as exc:
            raise SupabaseSchemaPreflightV2Error(f"Supabase schema probe failed for {table}") from exc
        if not 200 <= status < 300:
            raise SupabaseSchemaPreflightV2Error(f"required schema is unavailable for {table}; apply {migration} before restart (HTTP {status})")
        migrations.add(migration)
    required_rpcs = tuple(
        row for row in REQUIRED_SUPABASE_V2_RPCS
        if not (defer_incentive_retirement and row[1] == "lab_arena_incentive_retirement_schema_v1")
    )
    required_capabilities = tuple(
        row for row in SCHEMA_CAPABILITIES
        if not (defer_incentive_retirement and row[0] == "lab_arena_incentive_retirement_schema_v1")
    )
    for migration, function_name in required_rpcs:
        if len(function_name.encode()) > POSTGRES_IDENTIFIER_MAX_BYTES:
            raise SupabaseSchemaPreflightV2Error(f"required RPC identifier exceeds PostgreSQL's identifier limit: {function_name}")
        migrations.add(migration)
    schema_paths = {}
    for authority, probe_url, probe_headers in (
        ("service_role", supabase_url, headers),
        ("lab_arena_service", arena_supabase_url, arena_headers),
    ):
        schema_document = _read_json(Request(f"{probe_url}/rest/v1/", headers={**probe_headers, "Accept": "application/openapi+json"}), opener=opener, timeout_seconds=timeout_seconds, label=f"Supabase {authority} RPC schema probe")
        paths = schema_document.get("paths") if isinstance(schema_document, Mapping) else None
        if not isinstance(paths, Mapping):
            raise SupabaseSchemaPreflightV2Error(
                f"Supabase {authority} RPC schema document is invalid"
            )
        schema_paths[authority] = paths
    for migration, function_name in required_rpcs:
        authority = (
            "lab_arena_service"
            if migration in PRIVATE_ARENA_MIGRATIONS else "service_role"
        )
        paths = schema_paths[authority]
        if f"/rpc/{function_name}" not in paths:
            raise SupabaseSchemaPreflightV2Error(f"required RPC is unavailable for {function_name}; apply {migration} before restart")
    capabilities: Dict[str, Any] = {}
    for function_name, expected in required_capabilities:
        probe_url = (
            arena_supabase_url
            if function_name in PRIVATE_ARENA_CAPABILITIES else supabase_url
        )
        probe_headers = (
            arena_headers
            if function_name in PRIVATE_ARENA_CAPABILITIES else headers
        )
        value = _read_json(Request(f"{probe_url}/rest/v1/rpc/{function_name}", data=b"{}", headers={**probe_headers, "Content-Type": "application/json"}, method="POST"), opener=opener, timeout_seconds=timeout_seconds, label=f"{function_name} capability probe")
        if value != expected:
            raise SupabaseSchemaPreflightV2Error(f"{function_name} capability differs: expected {expected}")
        capabilities[function_name] = dict(expected)
    return {"status": "ready", "probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA) + len(schema_paths) + len(required_capabilities), "table_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA), "rpc_probe_count": len(required_rpcs), "data_probe_count": len(required_capabilities), "schema_document_probe_count": len(schema_paths), "schema_capabilities": capabilities, "migration_files": sorted(migrations), "incentive_retirement_deferred": bool(defer_incentive_retirement)}
