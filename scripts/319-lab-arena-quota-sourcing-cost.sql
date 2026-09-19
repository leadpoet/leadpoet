-- Opt-in sourcing cost visibility. The original counter-only RPC is unchanged.
-- No admission, settlement, eligibility, or output decision is changed here.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_run_quota_snapshot_v2(
  p_run_id TEXT, p_lease_token_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $quota_cost_snapshot$
DECLARE
  v_snapshot JSONB;
  v_run public.lab_arena_runs;
  v_cost JSONB;
  v_execution JSONB;
BEGIN
  -- Reuse the exact lease authorization and round/run lock of version 1.
  -- Scope comes from that lease, never from model-supplied identities.
  v_snapshot := public.lab_arena_run_quota_snapshot_v1(
    p_run_id, p_lease_token_hash
  );
  SELECT * INTO STRICT v_run
  FROM public.lab_arena_runs
  WHERE run_id = p_run_id;
  IF v_run.kind IS DISTINCT FROM 'execute' THEN
    RAISE EXCEPTION 'lab_arena_sourcing_cost_unavailable'
      USING ERRCODE = '22023';
  END IF;
  -- This existing function selects every execute attempt for this ICP and
  -- excludes every judge call. Expose its costs, not its zero-pair verdict.
  v_cost := public.lab_arena_icp_cost_eligibility(
    v_run.round_id, v_run.submission_id, v_run.icp_position, 0
  );
  v_execution := v_cost -> 'execution';
  RETURN v_snapshot || pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.quota_snapshot.v2',
    'sourcing_cost', pg_catalog.jsonb_build_object(
      'settled_microusd', v_execution -> 'settled_microusd',
      'reserved_or_uncertain_microusd',
        v_execution -> 'reserved_or_uncertain_microusd',
      'successful_microusd', v_execution -> 'successful_microusd',
      'success_unresolved_microusd',
        v_execution -> 'success_unresolved_microusd',
      'inflight_calls', v_execution -> 'inflight_calls',
      'success_unresolved_calls',
        v_execution -> 'success_unresolved_calls',
      'admission_cap_microusd',
        v_cost -> 'execution_icp_cap_microusd',
      'per_qualified_pair_cap_microusd',
        v_cost -> 'cost_per_company_cap_microusd'
    )
  );
END;
$quota_cost_snapshot$;
ALTER FUNCTION public.lab_arena_run_quota_snapshot_v2(TEXT, TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_run_quota_snapshot_v2(TEXT, TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_run_quota_snapshot_v2(TEXT, TEXT)
  TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMENT ON FUNCTION public.lab_arena_run_quota_snapshot_v2(TEXT, TEXT) IS
  'Passive active-lease per-ICP sourcing costs across execute attempts; no admission or qualification guarantee. Version 1 remains unchanged.';
COMMIT;
