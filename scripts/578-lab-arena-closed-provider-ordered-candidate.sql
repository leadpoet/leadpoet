-- Keep the same global cursor minimum while proving only the first valid
-- OpenRouter candidate in cursor order. Every existing scope and billing proof
-- remains required. OFFSET 0 fences ordered candidates and correlated proofs;
-- a branch LIMIT alone still validates all candidates before its sort.
-- No data, index, payer, settlement, history, timeout or publication change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $closed_provider_order578$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)';
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
BEGIN
  IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(
      'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)')),'sha256'),'hex')
      IS DISTINCT FROM 'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89' THEN
    RAISE EXCEPTION 'Closed provider order inner selector preimage differs';
  END IF;
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR security_before IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Closed provider order security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
      '89a779452595cbab4537fec883dfaa0191eb327244a311421d297c73fecf1476' THEN
    RAISE EXCEPTION 'Closed provider order exact preimage differs';
  END IF;
  updated:=replace(definition,$old_branch578$      SELECT pg_catalog.jsonb_build_object(
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
            IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
        AND rounds.configuration_doc ->> 'mode' = p_mode
        AND rounds.configuration_doc ->> 'network_name' = p_network_name
        AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
        AND (COALESCE(p_round_id, '') = '' OR rounds.round_id = p_round_id)$old_branch578$,
    $new_branch578$      (SELECT pg_catalog.jsonb_build_object(
        'status', 'ok', 'provider', 'openrouter',
        'uncertain_entry_id', uncertainty.entry_id,
        'round_id', uncertainty.round_id, 'run_id', uncertainty.run_id
      )
      FROM (
        SELECT uncertainty.entry_id, uncertainty.round_id, uncertainty.run_id,
          uncertainty.submission_id, uncertainty.miner_hotkey, uncertainty.stage,
          uncertainty.call_identity, uncertainty.provider, uncertainty.operation_id,
          uncertainty.funding_source, uncertainty.entry_doc,
          uncertainty.entry_id <= COALESCE(p_after_entry_id, 0) AS cursor_wrapped
        FROM public.lab_arena_ledger uncertainty
        JOIN public.lab_arena_runs runs ON runs.run_id = uncertainty.run_id
          AND runs.round_id = uncertainty.round_id
          AND runs.submission_id = uncertainty.submission_id
          AND runs.miner_hotkey = uncertainty.miner_hotkey
          AND runs.stage = uncertainty.stage
        JOIN public.lab_arena_rounds rounds ON rounds.round_id = uncertainty.round_id
        WHERE uncertainty.provider = 'openrouter'
          AND uncertainty.entry_kind = 'uncertain'
          AND runs.kind = 'score' AND runs.status IN ('accepted', 'failed')
          AND rounds.status IN ('published', 'cancelled')
          AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
              IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
          AND rounds.configuration_doc ->> 'mode' = p_mode
          AND rounds.configuration_doc ->> 'network_name' = p_network_name
          AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
          AND (COALESCE(p_round_id, '') = '' OR rounds.round_id = p_round_id)
        ORDER BY cursor_wrapped, uncertainty.entry_id
        OFFSET 0
      ) uncertainty
      CROSS JOIN LATERAL (
        SELECT 1
        WHERE uncertainty.operation_id IN ('openrouter.chat', 'openrouter.responses')
          AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
          AND (uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
               OR (uncertainty.entry_doc #>> '{call,reason}' = 'settle_failure'
                   AND uncertainty.entry_doc #>> '{call,failure_stage}' = 'settlement'
                   AND uncertainty.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception')))
          AND uncertainty.entry_doc #>> '{call,openrouter_generation_id}'
              ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
          AND uncertainty.entry_doc #>> '{call,credential_fingerprint}' ~ '^sha256:[0-9a-f]{64}$'
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
        OFFSET 0
      ) proof
      ORDER BY uncertainty.cursor_wrapped, uncertainty.entry_id
      LIMIT 1)$new_branch578$);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM
      '91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17' THEN
    RAISE EXCEPTION 'Closed provider order target differs';
  END IF;
  EXECUTE updated;
  SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO security_after
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')
      IS DISTINCT FROM '91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17' OR security_after IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Closed provider order readback differs';
  END IF;
END;
$closed_provider_order578$;
NOTIFY pgrst,'reload schema';
COMMIT;
