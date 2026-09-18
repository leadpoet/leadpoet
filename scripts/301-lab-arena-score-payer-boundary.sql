-- Keep paid provider calls on the submission owner's account for both execution
-- and scoring. A promoted miner remains the owner behind the daily baseline;
-- only a baseline without a promoted owner may use organizer credentials.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $score_payer_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_provider_funding(text,text)'
     ) IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_runs') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
     ) THEN
    RAISE EXCEPTION 'apply the current champion funding schema before 301';
  END IF;
END;
$score_payer_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_provider_funding(
  p_run_id TEXT, p_provider TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $score_payer_boundary$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_baseline BOOLEAN;
  v_champion BOOLEAN;
  v_source TEXT;
BEGIN
  IF p_provider NOT IN ('openrouter','deepline','scrapingdog') THEN
    RAISE EXCEPTION 'lab_arena_provider_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = p_run_id;
  IF NOT FOUND THEN RETURN pg_catalog.jsonb_build_object('status','missing'); END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = v_run.round_id;
  v_baseline := v_run.submission_id =
      'baseline-' || pg_catalog.regexp_replace(v_run.round_id,'^arena-','')
    AND v_run.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey'
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_submissions
      WHERE submission_id = v_run.submission_id AND is_king
    );
  v_champion := v_baseline
    AND v_round.champion_funding_frozen
    AND v_round.champion_submission_id IS NOT NULL;
  v_source := CASE
    -- Execution keeps its immutable per-run fallback snapshot. Scoring always
    -- remains on the promoted miner account and cannot inherit host fallback.
    WHEN v_champion AND v_run.kind = 'execute'
      THEN v_run.champion_funding_sources ->> p_provider
    WHEN v_champion AND v_run.kind = 'score' THEN 'miner_key'
    WHEN v_baseline THEN 'host'
    ELSE 'miner_key'
  END;
  IF v_source IS NULL OR v_source NOT IN ('host','miner_key') THEN
    RAISE EXCEPTION 'lab_arena_funding_snapshot_invalid' USING ERRCODE = '23514';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status','available',
    'funding_source',v_source,
    'champion_funding',v_champion,
    'credential_submission_id',CASE WHEN v_source = 'miner_key' THEN
      CASE WHEN v_champion THEN v_round.champion_submission_id
           ELSE v_run.submission_id END END,
    'credential_miner_hotkey',CASE WHEN v_source = 'miner_key' THEN
      CASE WHEN v_champion THEN v_round.champion_hotkey
           ELSE v_run.miner_hotkey END END,
    'restart_required',v_champion AND v_run.kind = 'execute' AND (
      v_run.champion_restart_required OR (
        v_source = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers)
      )
    )
  );
END;
$score_payer_boundary$;

ALTER FUNCTION public.lab_arena_provider_funding(TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_provider_funding(TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_provider_funding(TEXT,TEXT)
  TO lab_arena_service;

COMMIT;
