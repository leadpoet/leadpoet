-- Read-only, lease-scoped provider quota visibility for an active Arena run.
-- The snapshot exposes counters only. It does not reserve capacity, renew the
-- lease, change admission, or promise that a later provider call will succeed.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_274_requires_current_ledger$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__lock_current_lease(text,text)'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_ledger') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
     ) THEN
    RAISE EXCEPTION 'apply the current Lab Arena ledger schema before 274';
  END IF;
END;
$lab_arena_274_requires_current_ledger$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_run_quota_snapshot_v1(
  p_run_id TEXT,
  p_lease_token_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_run_quota_snapshot_v1$
DECLARE
  v_run public.lab_arena_runs;
  v_configuration JSONB;
  v_limits JSONB;
  v_snapshot JSONB;
BEGIN
  -- This takes the same round/run locks as provider admission. The function
  -- performs no update, but the lock makes the three counters one point-in-time
  -- view relative to reservations and terminal ledger transitions.
  v_run := public.lab_arena__lock_current_lease(
    p_run_id, p_lease_token_hash
  );

  SELECT rounds.configuration_doc
  INTO v_configuration
  FROM public.lab_arena_rounds AS rounds
  WHERE rounds.round_id = v_run.round_id;

  v_limits := CASE v_run.kind
    WHEN 'score' THEN v_configuration -> 'scoring_call_quotas'
    ELSE v_configuration -> 'call_quotas'
  END;

  IF pg_catalog.jsonb_typeof(v_limits) IS DISTINCT FROM 'object'
     OR COALESCE((v_limits ->> 'scrapingdog')::INTEGER, 0) < 1
     OR COALESCE((v_limits ->> 'deepline')::INTEGER, 0) < 1
     OR COALESCE((v_limits ->> 'openrouter')::INTEGER, 0) < 1 THEN
    RAISE EXCEPTION 'lab_arena_quota_missing' USING ERRCODE = '22023';
  END IF;

  WITH fixed_providers(provider, quota_limit) AS (
    VALUES
      ('scrapingdog'::TEXT, (v_limits ->> 'scrapingdog')::INTEGER),
      ('deepline'::TEXT, (v_limits ->> 'deepline')::INTEGER),
      ('openrouter'::TEXT, (v_limits ->> 'openrouter')::INTEGER)
  ),
  latest_heads AS (
    SELECT DISTINCT ON (ledger.provider, ledger.call_identity)
      ledger.provider,
      ledger.call_identity,
      ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id
      AND ledger.provider IN ('scrapingdog', 'deepline', 'openrouter')
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.provider, ledger.call_identity, ledger.entry_id DESC
  ),
  counts AS (
    SELECT
      providers.provider,
      providers.quota_limit,
      COUNT(heads.call_identity) FILTER (
        WHERE heads.entry_kind IN (
          'reservation', 'dispatch', 'settlement', 'uncertain'
        )
      )::INTEGER AS used,
      COUNT(heads.call_identity) FILTER (
        WHERE heads.entry_kind IN ('reservation', 'dispatch')
      )::INTEGER AS inflight
    FROM fixed_providers AS providers
    LEFT JOIN latest_heads AS heads ON heads.provider = providers.provider
    GROUP BY providers.provider, providers.quota_limit
  )
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.quota_snapshot.v1',
    'providers', pg_catalog.jsonb_build_object(
      'scrapingdog', (
        SELECT pg_catalog.jsonb_build_object(
          'limit', quota_limit,
          'used', used,
          'remaining', CASE WHEN used < quota_limit
            THEN quota_limit - used ELSE 0 END,
          'inflight', inflight
        ) FROM counts WHERE provider = 'scrapingdog'
      ),
      'deepline', (
        SELECT pg_catalog.jsonb_build_object(
          'limit', quota_limit,
          'used', used,
          'remaining', CASE WHEN used < quota_limit
            THEN quota_limit - used ELSE 0 END,
          'inflight', inflight
        ) FROM counts WHERE provider = 'deepline'
      ),
      'openrouter', (
        SELECT pg_catalog.jsonb_build_object(
          'limit', quota_limit,
          'used', used,
          'remaining', CASE WHEN used < quota_limit
            THEN quota_limit - used ELSE 0 END,
          'inflight', inflight
        ) FROM counts WHERE provider = 'openrouter'
      )
    )
  ) INTO v_snapshot;

  RETURN v_snapshot;
END;
$lab_arena_run_quota_snapshot_v1$;

ALTER FUNCTION public.lab_arena_run_quota_snapshot_v1(TEXT, TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_run_quota_snapshot_v1(TEXT, TEXT)
  FROM PUBLIC;

DO $lab_arena_274_function_acl$
DECLARE
  role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role']
  LOOP
    IF EXISTS (
      SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
    ) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_run_quota_snapshot_v1(TEXT, TEXT) FROM %I',
        role_name
      );
    END IF;
  END LOOP;
  GRANT EXECUTE ON FUNCTION public.lab_arena_run_quota_snapshot_v1(TEXT, TEXT)
    TO lab_arena_service;
END;
$lab_arena_274_function_acl$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;

COMMENT ON FUNCTION public.lab_arena_run_quota_snapshot_v1(TEXT, TEXT) IS
  'Read-only point-in-time active-lease quota counters; no reservation, renewal, admission, or future availability guarantee.';

COMMIT;
