-- Frozen catalog rounds admit Deepline work by confirmed spend, not raw calls.
-- Preserve every positive historical quota and the existing no-hold admission.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $deepline_budget_only$
DECLARE
  v_definition TEXT;
  v_signature TEXT;
  v_old TEXT;
  v_new TEXT;
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',
    'public.lab_arena_run_quota_snapshot_v1(text,text)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_signature::pg_catalog.regprocedure)
      INTO v_definition;
    IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_budget_only') > 0 THEN
      CONTINUE;
    END IF;
    IF v_signature LIKE '%reserve_call%' THEN
      IF pg_catalog.strpos(v_definition, 'lab_arena_confirmed_cost_admission') = 0
         OR pg_catalog.strpos(v_definition, 'lab_arena_confirmed_score_admission') = 0 THEN
        RAISE EXCEPTION 'apply confirmed-cost admission migration 321 first';
      END IF;
      v_old := 'IF v_quota IS NULL OR v_quota < 1 THEN';
      v_new := $replace$-- lab_arena_deepline_budget_only: zero has meaning only for a frozen
  -- catalog round under the existing confirmed-cost policy.
  IF v_quota IS NULL OR v_quota < 0 OR (v_quota = 0 AND NOT COALESCE((
       p_provider = 'deepline'
       AND v_round.configuration_doc #>> '{deepline_catalog,schema_version}'
           = 'leadpoet.lab_arena.deepline_catalog.v1'
       AND v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_per_icp_v1'
     ), FALSE)) THEN$replace$;
      IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
        RAISE EXCEPTION 'Deepline quota admission preimage differs';
      END IF;
      v_definition := pg_catalog.replace(v_definition, v_old, v_new);
      v_definition := pg_catalog.replace(v_definition,
        'IF v_consumed >= v_quota THEN',
        'IF v_quota > 0 AND v_consumed >= v_quota THEN');
      v_definition := pg_catalog.replace(v_definition,
        'IF v_consumed >= v_stage_quota THEN',
        'IF v_quota > 0 AND v_consumed >= v_stage_quota THEN');
    ELSE
      v_old := $replace$OR COALESCE((v_limits ->> 'deepline')::INTEGER, 0) < 1$replace$;
      v_new := $replace$-- lab_arena_deepline_budget_only
     OR (v_limits ->> 'deepline') IS NULL
     OR (v_limits ->> 'deepline')::INTEGER < 0
     OR ((v_limits ->> 'deepline')::INTEGER = 0 AND NOT COALESCE((
       v_configuration #>> '{deepline_catalog,schema_version}'
           = 'leadpoet.lab_arena.deepline_catalog.v1'
       AND v_configuration ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_per_icp_v1'
     ), FALSE))$replace$;
      IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
        RAISE EXCEPTION 'Deepline quota snapshot preimage differs';
      END IF;
      v_definition := pg_catalog.replace(v_definition, v_old, v_new);
      -- Only Deepline can have a zero limit; other quota validation stays exact.
      v_definition := pg_catalog.replace(v_definition,
        'CASE WHEN used < quota_limit',
        'CASE WHEN quota_limit = 0 THEN NULL WHEN used < quota_limit');
    END IF;
    EXECUTE v_definition;
  END LOOP;
END;
$deepline_budget_only$;

-- Recovery metadata comes from the immutable pre-dispatch reservation.
DO $deepline_exact_recovery_list$
DECLARE
  v_definition TEXT;
  v_old TEXT := $replace$'request_id', candidate.request_id,$replace$;
  v_new TEXT := $replace$'request_id', candidate.request_id,
            'execution_key', candidate.execution_key,
            'billing_provider', candidate.billing_provider,
            'operation_aliases', candidate.operation_aliases,$replace$;
  v_select_old TEXT := $replace$reservation.entry_doc ->> 'tool' AS operation,$replace$;
  v_select_new TEXT := $replace$-- lab_arena_deepline_execution_key_recovery
            reservation.entry_doc ->> 'deepline_execution_key' AS execution_key,
            reservation.entry_doc ->> 'deepline_billing_provider' AS billing_provider,
            reservation.entry_doc -> 'deepline_operation_aliases' AS operation_aliases,
            reservation.entry_doc ->> 'tool' AS operation,$replace$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_execution_key_recovery') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(v_definition, v_select_old) = 0 THEN
      RAISE EXCEPTION 'Deepline exact recovery list preimage differs';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    EXECUTE pg_catalog.replace(v_definition, v_select_old, v_select_new);
  END IF;
END;
$deepline_exact_recovery_list$;

-- V2 reuses the exact legacy settlement guards. The new provider request ID
-- is separately bound to the pre-dispatch key; it never replaces the candidate
-- identity or updates an existing ledger row.
DO $deepline_exact_recovery_settlement$
DECLARE
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_native_billing_identity') = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_billing_reconciliation') = 0 THEN
    RAISE EXCEPTION 'apply exact native and closed Deepline recovery first';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'FUNCTION public.lab_arena_reconcile_deepline_cost_v1(',
    'FUNCTION public.lab_arena_reconcile_deepline_cost_v2(');
  v_definition := pg_catalog.replace(v_definition,
    'p_cost_units text)',
    'p_cost_units text, p_execution_key text, p_recovered_request_id text)');
  -- New API IDs are opaque. Equality to the retained candidate and key,
  -- rather than one historical provider ID format, is the safety boundary.
  v_old := $replace$'^(ctx-tool-[0-9a-f]{32}|[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64})$'$replace$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'Deepline native identity validation preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old,
    $replace$'^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'$replace$);
  v_old := $replace$  IF COALESCE(p_round_id, '') = ''$replace$;
  v_new := $replace$  IF COALESCE(p_execution_key, '') IS DISTINCT FROM
       'arena:' || pg_catalog.substr(p_call_identity, 8)
     OR COALESCE(p_recovered_request_id, '') !~
        '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$' THEN
    RAISE EXCEPTION 'lab_arena_deepline_execution_recovery_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF COALESCE(p_round_id, '') = ''$replace$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'Deepline exact recovery input preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  v_old := $replace$  IF v_head.entry_kind = 'settlement'$replace$;
  v_new := $replace$  -- lab_arena_deepline_execution_key_recovery: only this immutable
  -- reservation can authorize binding a recovered native provider identity.
  SELECT * INTO v_reservation
  FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  IF v_reservation.run_id IS DISTINCT FROM p_run_id
     OR v_reservation.round_id IS DISTINCT FROM p_round_id
     OR v_reservation.entry_doc ->> 'deepline_execution_key'
        IS DISTINCT FROM p_execution_key
     OR (v_head.entry_kind = 'uncertain'
         AND v_head.entry_doc #>> '{call,deepline_job_id}' IS NOT NULL
         AND v_head.entry_doc #>> '{call,deepline_job_id}'
             IS DISTINCT FROM p_recovered_request_id) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  IF v_head.entry_kind = 'settlement'$replace$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'Deepline exact recovery head preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  v_definition := pg_catalog.replace(v_definition,
    $replace$       AND v_head.amount_microusd = p_actual_microusd$replace$,
    $replace$       AND v_head.entry_doc ->> 'deepline_execution_key' = p_execution_key
       AND v_head.entry_doc ->> 'deepline_recovered_request_id' = p_recovered_request_id
       AND v_head.amount_microusd = p_actual_microusd$replace$);
  v_definition := pg_catalog.replace(v_definition,
    $replace$      'request_id', p_request_id$replace$,
    $replace$      'request_id', p_recovered_request_id$replace$);
  v_definition := pg_catalog.replace(v_definition,
    $replace$      'deepline_operation', p_operation,$replace$,
    $replace$      'deepline_operation', p_operation,
      'deepline_execution_key', p_execution_key,
      'deepline_recovered_request_id', p_recovered_request_id,$replace$);
  EXECUTE v_definition;
END;
$deepline_exact_recovery_settlement$;
ALTER FUNCTION public.lab_arena_reconcile_deepline_cost_v2(
  TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_reconcile_deepline_cost_v2(
  TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_reconcile_deepline_cost_v2(
  TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT,BIGINT,TEXT,TEXT,TEXT
) TO lab_arena_service;
-- Add the catalog in the same write that freezes the benchmark. Existing
-- catalogs and every other policy field remain immutable.
DO $deepline_catalog_commit_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := $replace$(NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference') =
          (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference')$replace$;
  v_new TEXT := $replace$(
        (NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference') =
          (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference')
        OR (
          -- lab_arena_deepline_catalog_commit_guard
          NOT OLD.configuration_doc ? 'deepline_catalog'
          AND OLD.benchmark_ref IS NULL
          AND OLD.icp_set_date IS NULL
          AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id = OLD.round_id)
          AND NEW.configuration_doc #>> '{deepline_catalog,schema_version}'
              = 'leadpoet.lab_arena.deepline_catalog.v1'
          AND NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
              = 'successful_calls_per_icp_v1'
          AND NEW.configuration_doc #>> '{call_quotas,deepline}' = '0'
          AND NEW.configuration_doc #>> '{scoring_call_quotas,deepline}' = '0'
          AND (NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference'
               - 'deepline_catalog' - 'call_quotas' - 'scoring_call_quotas') =
              (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference'
               - 'call_quotas' - 'scoring_call_quotas')
          AND (NEW.configuration_doc -> 'call_quotas') =
              pg_catalog.jsonb_set(OLD.configuration_doc -> 'call_quotas', '{deepline}', '0')
          AND (NEW.configuration_doc -> 'scoring_call_quotas') =
              pg_catalog.jsonb_set(OLD.configuration_doc -> 'scoring_call_quotas', '{deepline}', '0')
        )
      )$replace$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_rounds_write_once_v1()'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_catalog_commit_guard') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'Deepline catalog commit guard preimage differs';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$deepline_catalog_commit_guard$;

DO $deepline_catalog_commit$
DECLARE
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  v_definition := pg_catalog.replace(v_definition,
    'FUNCTION public.lab_arena_commit_round_v2(', 'FUNCTION public.lab_arena_commit_round_v3(');
  v_definition := pg_catalog.replace(v_definition,
    'p_scorer_image_reference text)', 'p_scorer_image_reference text, p_deepline_catalog jsonb)');
  v_old := $replace$  UPDATE public.lab_arena_rounds
  SET status = 'committed',$replace$;
  v_new := $replace$  -- lab_arena_deepline_catalog_commit: validation cannot bypass the
  -- existing participant/date/scorer checks and has no partial config write.
  IF pg_catalog.jsonb_typeof(p_deepline_catalog) IS DISTINCT FROM 'object'
     OR p_deepline_catalog ->> 'schema_version' IS DISTINCT FROM
        'leadpoet.lab_arena.deepline_catalog.v1'
     OR pg_catalog.octet_length(p_deepline_catalog::TEXT) > 4194304
     OR pg_catalog.jsonb_typeof(p_deepline_catalog -> 'tools') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_deepline_catalog -> 'tools') NOT BETWEEN 1 AND 4096
     OR COALESCE(p_deepline_catalog ->> 'catalog_hash', '') !~ '^[0-9a-f]{64}$'
     OR p_deepline_catalog ->> 'policy_version' IS DISTINCT FROM
        'leadpoet.lab_arena.deepline_company_research.v1'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR v_round.configuration_doc ->> 'execution_icp_cap_microusd' IS DISTINCT FROM '4000000'
     OR pg_catalog.jsonb_typeof(p_deepline_catalog -> 'allow_people') IS DISTINCT FROM 'boolean'
     OR (p_deepline_catalog ->> 'allow_people')::BOOLEAN IS DISTINCT FROM
        COALESCE(v_round.configuration_doc ->> 'contact_policy' = 'contacts_v1', FALSE)
     OR (v_round.configuration_doc ? 'deepline_catalog'
         AND v_round.configuration_doc -> 'deepline_catalog' IS DISTINCT FROM p_deepline_catalog)
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.icp_set_date IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id = p_round_id) THEN
    RAISE EXCEPTION 'lab_arena_deepline_catalog_commit_invalid' USING ERRCODE = '22023';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'committed',$replace$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'Deepline catalog atomic commit preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  v_old := $replace$configuration_doc = v_round.configuration_doc || pg_catalog.jsonb_build_object($replace$;
  v_new := $replace$configuration_doc = pg_catalog.jsonb_set(pg_catalog.jsonb_set(
        v_round.configuration_doc || pg_catalog.jsonb_build_object('deepline_catalog', p_deepline_catalog),
        '{call_quotas,deepline}', '0'), '{scoring_call_quotas,deepline}', '0')
        || pg_catalog.jsonb_build_object($replace$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'Deepline catalog configuration commit preimage differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$deepline_catalog_commit$;
ALTER FUNCTION public.lab_arena_commit_round_v3(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_commit_round_v3(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_commit_round_v3(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB)
  TO lab_arena_service;

-- Dynamic rollout starts only after its exact RPCs and quota guards exist.
CREATE OR REPLACE FUNCTION public.lab_arena_deepline_catalog_schema_v1()
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $deepline_catalog_schema$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)'
     ) IS NULL
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
     ), 'lab_arena_deepline_budget_only') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena_run_quota_snapshot_v1(text,text)'::pg_catalog.regprocedure
     ), 'lab_arena_deepline_budget_only') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena_rounds_write_once_v1()'::pg_catalog.regprocedure
     ), 'lab_arena_deepline_catalog_commit_guard') = 0 THEN
    RAISE EXCEPTION 'lab_arena_deepline_catalog_schema_incomplete' USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.deepline_catalog_schema.v1', 'version', 415
  );
END;
$deepline_catalog_schema$;
ALTER FUNCTION public.lab_arena_deepline_catalog_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_deepline_catalog_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_deepline_catalog_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
