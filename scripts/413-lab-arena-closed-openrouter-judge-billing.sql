-- Recover exact OpenRouter judge bills after publication or cancellation.
-- Published scores, sourcing admission, promotion and reward state do not change.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $closed_openrouter_413$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_updated TEXT;
  v_old_hash TEXT;
  v_new_hash TEXT;
  v_identity JSONB;
  v_expected_volatility TEXT;
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_reconcile_openrouter_cost_v1(text,text,text,bigint,text,text,bigint,text)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
           pg_catalog.jsonb_build_array(o.rolname, p.proacl::TEXT,
             p.prosecdef, p.provolatile, p.proconfig)
    INTO v_definition, v_identity
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles o ON o.oid = p.proowner
    WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
    IF v_signature LIKE 'public.lab_arena_list_%' THEN
      v_old_hash := '07875b14d14625c3027e1b6c2fea5baf';
      v_new_hash := '5f4c076529fec10150a9f4e719016f6f';
      v_expected_volatility := 's';
      v_updated := pg_catalog.replace(pg_catalog.replace(v_definition, $old$            AND rounds.status NOT IN ('open', 'published', 'cancelled')$old$, $new$            -- lab_arena_closed_openrouter_judge_billing_413
            AND (
              rounds.status NOT IN ('open', 'published', 'cancelled')
              OR (rounds.status IN ('published', 'cancelled')
                  AND runs.kind = 'score'
                  AND runs.status IN ('accepted', 'failed')
                  AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
                      IN ('successful_calls_v1', 'successful_calls_per_icp_v1'))
            )$new$), $old$uncertainty.entry_doc #>> '{call,reason}' =
                'missing_provider_cost'$old$, $new$(uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
                 OR (uncertainty.entry_doc #>> '{call,reason}' = 'settle_failure'
                     AND uncertainty.entry_doc #>> '{call,failure_stage}' = 'settlement'
                     AND uncertainty.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception')))$new$);
    ELSE
      v_old_hash := '89090bb81c76374fadc4ef9b745adb58';
      v_new_hash := '088a417cf90e0c919792029dcf788ca8';
      v_expected_volatility := 'v';
      v_updated := pg_catalog.replace(pg_catalog.replace(
        pg_catalog.replace(v_definition, $old$  IF v_round.status IN ('open', 'published', 'cancelled') THEN$old$, $new$  IF v_round.status = 'open' THEN$new$),
        $old$  SELECT * INTO v_submission
  FROM public.lab_arena_submissions$old$, $new$  -- lab_arena_closed_openrouter_judge_billing_413: only audit costs change.
  IF v_round.status IN ('published', 'cancelled') AND NOT (
       v_run.kind = 'score' AND v_run.status IN ('accepted', 'failed')
       AND COALESCE(v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy', '')
           IN ('successful_calls_v1', 'successful_calls_per_icp_v1')
     ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions$new$
      ), $old$v_head.entry_doc #>> '{call,reason}' IS DISTINCT FROM
        'missing_provider_cost'$old$, $new$NOT COALESCE((
       v_head.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
       OR (v_head.entry_doc #>> '{call,reason}' = 'settle_failure'
           AND v_head.entry_doc #>> '{call,failure_stage}' = 'settlement'
           AND v_head.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception'))
     ), FALSE)$new$);
    END IF;
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
         'lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE, v_expected_volatility, ARRAY['search_path=pg_catalog, public']
       ) THEN
      RAISE EXCEPTION 'lab_arena_closed_openrouter_413_security_shape_changed';
    END IF;
    IF pg_catalog.md5(v_definition) = v_new_hash THEN
      CONTINUE;
    END IF;
    IF pg_catalog.md5(v_definition) <> v_old_hash
       OR pg_catalog.md5(v_updated) <> v_new_hash THEN
      RAISE EXCEPTION 'lab_arena_closed_openrouter_413_preimage_changed';
    END IF;
    EXECUTE v_updated;
    IF pg_catalog.md5(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE)) <> v_new_hash THEN
      RAISE EXCEPTION 'lab_arena_closed_openrouter_413_readback_changed';
    END IF;
  END LOOP;
END;
$closed_openrouter_413$;

-- Fail closed on an unexpected existing generic selector or legacy dependency.
DO $closed_provider_preimage_413$
DECLARE
  v_oid REGPROCEDURE;
  v_name TEXT;
  v_hash TEXT;
  v_expected TEXT;
BEGIN
  FOREACH v_name IN ARRAY ARRAY['deepline', 'provider'] LOOP
    v_oid := pg_catalog.to_regprocedure('public.lab_arena_next_closed_' || v_name ||
      '_reconciliation_v1(text,text,integer,text,bigint)');
    IF v_oid IS NULL AND v_name = 'provider' THEN
      CONTINUE;
    END IF;
    v_expected := CASE v_name WHEN 'deepline' THEN '835bb0eaf4c842778a5ea8d8da90b19c'
      ELSE 'e44109bc96f0be89c600079ab07a1ee7' END;
    SELECT pg_catalog.md5(pg_catalog.pg_get_functiondef(p.oid)) INTO v_hash
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid = p.proowner
    WHERE p.oid = v_oid AND o.rolname = 'lab_arena_owner'
      AND p.proacl::TEXT = '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile = 's'
      AND p.proconfig = ARRAY['search_path=pg_catalog, public'];
    IF v_hash IS DISTINCT FROM v_expected THEN
      RAISE EXCEPTION 'lab_arena_closed_provider_413_preimage_changed';
    END IF;
  END LOOP;
END;
$closed_provider_preimage_413$;

-- Keep the old Deepline-only selector unchanged for an N-1 gateway.
CREATE OR REPLACE FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1(
  p_mode TEXT, p_network_name TEXT, p_netuid INTEGER,
  p_round_id TEXT, p_after_entry_id BIGINT
)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $closed_provider_413$
  SELECT COALESCE((
    SELECT candidate.doc FROM (
      SELECT original.doc || pg_catalog.jsonb_build_object('provider', 'deepline') AS doc
      FROM (SELECT public.lab_arena_next_closed_deepline_reconciliation_v1(
          p_mode, p_network_name, p_netuid, p_round_id, p_after_entry_id
        ) AS doc) original
      WHERE original.doc ->> 'status' = 'ok'
      UNION ALL
      SELECT pg_catalog.jsonb_build_object(
        'status', 'ok', 'provider', 'openrouter',
        'uncertain_entry_id', uncertainty.entry_id,
        'round_id', rounds.round_id, 'run_id', uncertainty.run_id
      )
      FROM public.lab_arena_ledger uncertainty
      JOIN public.lab_arena_runs runs ON runs.run_id = uncertainty.run_id
        AND runs.round_id = uncertainty.round_id
        AND runs.submission_id = uncertainty.submission_id
        AND runs.miner_hotkey = uncertainty.miner_hotkey
        AND runs.stage = uncertainty.stage
      JOIN public.lab_arena_rounds rounds ON rounds.round_id = uncertainty.round_id
      WHERE uncertainty.provider = 'openrouter'
        AND uncertainty.entry_kind = 'uncertain'
        AND uncertainty.operation_id IN ('openrouter.chat', 'openrouter.responses')
        AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
        AND (uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
             OR (uncertainty.entry_doc #>> '{call,reason}' = 'settle_failure'
                 AND uncertainty.entry_doc #>> '{call,failure_stage}' = 'settlement'
                 AND uncertainty.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception')))
        AND uncertainty.entry_doc #>> '{call,openrouter_generation_id}'
            ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
        AND uncertainty.entry_doc #>> '{call,credential_fingerprint}' ~ '^sha256:[0-9a-f]{64}$'
        AND runs.kind = 'score' AND runs.status IN ('accepted', 'failed')
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger later
          WHERE later.call_identity = uncertainty.call_identity AND later.entry_id > uncertainty.entry_id)
        AND EXISTS (SELECT 1 FROM public.lab_arena_ledger reservation
          WHERE reservation.call_identity = uncertainty.call_identity AND reservation.entry_kind = 'reservation'
            AND reservation.run_id = uncertainty.run_id AND reservation.round_id = uncertainty.round_id
            AND reservation.submission_id = uncertainty.submission_id AND reservation.miner_hotkey = uncertainty.miner_hotkey
            AND reservation.stage = uncertainty.stage AND reservation.provider = uncertainty.provider
            AND reservation.operation_id = uncertainty.operation_id AND reservation.funding_source = uncertainty.funding_source)
        AND EXISTS (SELECT 1 FROM public.lab_arena_ledger dispatch
          WHERE dispatch.call_identity = uncertainty.call_identity AND dispatch.entry_kind = 'dispatch'
            AND dispatch.run_id = uncertainty.run_id)
        AND rounds.status IN ('published', 'cancelled')
        AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
            IN ('successful_calls_v1', 'successful_calls_per_icp_v1')
        AND rounds.configuration_doc ->> 'mode' = p_mode
        AND rounds.configuration_doc ->> 'network_name' = p_network_name
        AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
        AND (COALESCE(p_round_id, '') = '' OR rounds.round_id = p_round_id)
    ) candidate
    ORDER BY ((candidate.doc ->> 'uncertain_entry_id')::BIGINT <= COALESCE(p_after_entry_id, 0)),
             (candidate.doc ->> 'uncertain_entry_id')::BIGINT
    LIMIT 1
  ), pg_catalog.jsonb_build_object('status', 'none'));
$closed_provider_413$;

DO $closed_provider_acl_413$
DECLARE
  v_granted_create BOOLEAN := FALSE;
BEGIN
  IF NOT pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE') THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
    v_granted_create := TRUE;
  END IF;
  ALTER FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
    OWNER TO lab_arena_owner;
  REVOKE ALL ON FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
    FROM PUBLIC, anon, authenticated, service_role;
  GRANT EXECUTE ON FUNCTION public.lab_arena_next_closed_provider_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
    TO lab_arena_service;
  IF v_granted_create THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles o ON o.oid = p.proowner
    WHERE p.oid = 'public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)'::REGPROCEDURE
      AND pg_catalog.md5(pg_catalog.pg_get_functiondef(p.oid)) = 'e44109bc96f0be89c600079ab07a1ee7'
      AND o.rolname = 'lab_arena_owner'
      AND p.proacl::TEXT = '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile = 's'
      AND p.proconfig = ARRAY['search_path=pg_catalog, public']
  ) THEN
    RAISE EXCEPTION 'lab_arena_closed_provider_413_readback_changed';
  END IF;
END;
$closed_provider_acl_413$;
NOTIFY pgrst, 'reload schema';
COMMIT;
