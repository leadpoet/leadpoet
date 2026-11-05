-- Validate exact billing candidates in cursor order, stopping at the existing limit.
-- Keep every original JSON, round/run, latest-head, reservation and dispatch proof.
-- Thin candidates retain the implied non-null binding so existing indexes apply;
-- full JSON is read by primary key only after the ordered OFFSET 0 fence.
-- No data, cost, timeout, index, ownership, privilege or settlement change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $openrouter_ordered_cost632$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_list_openrouter_cost_reconciliations_v1(text,text,bigint,integer)';
  target OID;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  prerequisite RECORD;
  index_definition TEXT;
  index_owner TEXT;
BEGIN
  -- Verify both existing access paths before accepting installation or replay.
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena_ledger_openrouter_uncertain_run_entry_idx',
     'CREATE INDEX lab_arena_ledger_openrouter_uncertain_run_entry_idx ON public.lab_arena_ledger USING btree (run_id, entry_id) WHERE ((provider = ''openrouter''::text) AND (entry_kind = ''uncertain''::text))'),
    ('public.lab_arena_ledger_round_call_head_idx',
     'CREATE INDEX lab_arena_ledger_round_call_head_idx ON public.lab_arena_ledger USING btree (round_id) WHERE (call_identity IS NOT NULL)')
  ) AS required(name,definition) LOOP
    SELECT pg_get_indexdef(i.indexrelid),pg_get_userbyid(c.relowner)
      INTO index_definition,index_owner
    FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
    JOIN pg_am am ON am.oid=c.relam
    WHERE i.indexrelid=to_regclass(prerequisite.name)
      AND i.indrelid='public.lab_arena_ledger'::regclass
      AND am.amname='btree' AND NOT i.indisunique
      AND i.indisvalid AND i.indisready;
    IF index_owner IS DISTINCT FROM 'lab_arena_owner'
       OR index_definition IS DISTINCT FROM prerequisite.definition THEN
      RAISE EXCEPTION 'OpenRouter ordered cost index absent, invalid, or wrong shape: %',prerequisite.name USING ERRCODE='55000';
    END IF;
  END LOOP;

  target:=to_regprocedure(signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
    INTO definition,metadata,identity
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
    RAISE EXCEPTION 'OpenRouter ordered cost selector security differs' USING ERRCODE='55000';
  END IF;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex')=
      '4554bbd62c6fe41589f8142cf4b92d6845ea2813f466bd77110e809dd6030e72' THEN RETURN; END IF;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex') IS DISTINCT FROM
      'e2d75725cb2fa7c5e901f27a0625b86c9a9d5f0727fcbfafb5ebcb235d46ab78' THEN
    RAISE EXCEPTION 'OpenRouter ordered cost selector preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,$old_candidate632$          SELECT
            uncertainty.entry_id AS uncertain_entry_id,
            uncertainty.round_id,
            uncertainty.run_id,
            uncertainty.submission_id,
            uncertainty.miner_hotkey,
            runs.assignment_id,
            uncertainty.stage,
            runs.icp_position,
            runs.attempt,
            runs.kind,
            uncertainty.call_identity,
            uncertainty.entry_doc #>>
              '{call,openrouter_generation_id}' AS generation_id,
            uncertainty.entry_doc #>>
              '{call,credential_fingerprint}' AS credential_fingerprint,
            uncertainty.funding_source,
            runs.status AS run_status,
            pg_catalog.to_char(
              runs.lease_expires_at AT TIME ZONE 'UTC',
              'YYYY-MM-DD"T"HH24:MI:SS"Z"'
            ) AS lease_expires_at
          FROM public.lab_arena_ledger AS uncertainty
          JOIN public.lab_arena_runs AS runs
            ON runs.run_id = uncertainty.run_id
           AND runs.round_id = uncertainty.round_id
           AND runs.submission_id = uncertainty.submission_id
           AND runs.miner_hotkey = uncertainty.miner_hotkey
           AND runs.stage = uncertainty.stage
          JOIN public.lab_arena_rounds AS rounds
            ON rounds.round_id = uncertainty.round_id
          WHERE uncertainty.round_id = p_round_id
            AND (COALESCE(p_run_id, '') = '' OR uncertainty.run_id = p_run_id)
            -- lab_arena_closed_openrouter_judge_billing_413
            AND (
              rounds.status NOT IN ('open', 'published', 'cancelled')
              OR (rounds.status IN ('published', 'cancelled')
                  AND runs.kind = 'score'
                  AND runs.status IN ('accepted', 'failed')
                  AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
                      IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1'))
            )
            AND uncertainty.entry_kind = 'uncertain'
            AND uncertainty.provider = 'openrouter'
            AND uncertainty.operation_id IN ('openrouter.chat', 'openrouter.responses')
            AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
            AND (uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
                 OR (uncertainty.entry_doc #>> '{call,reason}' = 'settle_failure'
                     AND uncertainty.entry_doc #>> '{call,failure_stage}' = 'settlement'
                     AND uncertainty.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception')))
            AND uncertainty.entry_doc #>>
                '{call,openrouter_generation_id}' ~
                '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
            AND uncertainty.entry_doc #>>
                '{call,credential_fingerprint}' ~ '^sha256:[0-9a-f]{64}$'
            AND NOT EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS later
              WHERE later.call_identity = uncertainty.call_identity
                AND later.entry_id > uncertainty.entry_id
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS reservation
              WHERE reservation.call_identity = uncertainty.call_identity
                AND reservation.entry_kind = 'reservation'
                AND reservation.run_id = uncertainty.run_id
                AND reservation.round_id = uncertainty.round_id
                AND reservation.submission_id = uncertainty.submission_id
                AND reservation.miner_hotkey = uncertainty.miner_hotkey
                AND reservation.stage = uncertainty.stage
                AND reservation.provider = uncertainty.provider
                AND reservation.operation_id = uncertainty.operation_id
                AND reservation.funding_source = uncertainty.funding_source
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS dispatch
              WHERE dispatch.call_identity = uncertainty.call_identity
                AND dispatch.entry_kind = 'dispatch'
                AND dispatch.run_id = uncertainty.run_id
            )
          ORDER BY
            uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0),
            uncertainty.entry_id
          LIMIT CASE
            WHEN p_limit BETWEEN 1 AND 20 THEN p_limit
            ELSE 0
          END
$old_candidate632$,$new_candidate632$          SELECT
            uncertainty.entry_id AS uncertain_entry_id,
            uncertainty.round_id,
            uncertainty.run_id,
            uncertainty.submission_id,
            uncertainty.miner_hotkey,
            uncertainty.assignment_id,
            uncertainty.stage,
            uncertainty.icp_position,
            uncertainty.attempt,
            uncertainty.kind,
            uncertainty.call_identity,
            proof.entry_doc #>>
              '{call,openrouter_generation_id}' AS generation_id,
            proof.entry_doc #>>
              '{call,credential_fingerprint}' AS credential_fingerprint,
            uncertainty.funding_source,
            uncertainty.run_status AS run_status,
            pg_catalog.to_char(
              uncertainty.lease_expires_at AT TIME ZONE 'UTC',
              'YYYY-MM-DD"T"HH24:MI:SS"Z"'
            ) AS lease_expires_at
          FROM (
            SELECT uncertainty.entry_id, uncertainty.round_id, uncertainty.run_id,
              uncertainty.submission_id, uncertainty.miner_hotkey, uncertainty.stage,
              uncertainty.call_identity, uncertainty.provider, uncertainty.operation_id,
              uncertainty.funding_source, runs.assignment_id, runs.icp_position,
              runs.attempt, runs.kind, runs.status AS run_status, runs.lease_expires_at,
              uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0) AS cursor_wrapped
          FROM public.lab_arena_ledger AS uncertainty
          JOIN public.lab_arena_runs AS runs
            ON runs.run_id = uncertainty.run_id
           AND runs.round_id = uncertainty.round_id
           AND runs.submission_id = uncertainty.submission_id
           AND runs.miner_hotkey = uncertainty.miner_hotkey
           AND runs.stage = uncertainty.stage
          JOIN public.lab_arena_rounds AS rounds
            ON rounds.round_id = uncertainty.round_id
          WHERE uncertainty.round_id = p_round_id
            AND (COALESCE(p_run_id, '') = '' OR uncertainty.run_id = p_run_id)
            -- lab_arena_closed_openrouter_judge_billing_413
            AND (
              rounds.status NOT IN ('open', 'published', 'cancelled')
              OR (rounds.status IN ('published', 'cancelled')
                  AND runs.kind = 'score'
                  AND runs.status IN ('accepted', 'failed')
                  AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
                      IN ('successful_calls_v1', 'successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1'))
            )
            AND uncertainty.call_identity IS NOT NULL
            AND uncertainty.entry_kind = 'uncertain'
            AND uncertainty.provider = 'openrouter'
            AND uncertainty.operation_id IN ('openrouter.chat', 'openrouter.responses')
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty
          CROSS JOIN LATERAL (
            SELECT metadata.entry_doc
            FROM public.lab_arena_ledger AS metadata
            WHERE metadata.entry_id = uncertainty.entry_id
            AND metadata.entry_doc ->> 'reason' = 'worker_reported'
            AND (metadata.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
                 OR (metadata.entry_doc #>> '{call,reason}' = 'settle_failure'
                     AND metadata.entry_doc #>> '{call,failure_stage}' = 'settlement'
                     AND metadata.entry_doc #>> '{call,error_class}' IN ('ArenaStoreError', 'Exception')))
            AND metadata.entry_doc #>>
                '{call,openrouter_generation_id}' ~
                '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
            AND metadata.entry_doc #>>
                '{call,credential_fingerprint}' ~ '^sha256:[0-9a-f]{64}$'
            AND NOT EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS later
              WHERE later.call_identity = uncertainty.call_identity
                AND later.entry_id > uncertainty.entry_id
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS reservation
              WHERE reservation.call_identity = uncertainty.call_identity
                AND reservation.entry_kind = 'reservation'
                AND reservation.run_id = uncertainty.run_id
                AND reservation.round_id = uncertainty.round_id
                AND reservation.submission_id = uncertainty.submission_id
                AND reservation.miner_hotkey = uncertainty.miner_hotkey
                AND reservation.stage = uncertainty.stage
                AND reservation.provider = uncertainty.provider
                AND reservation.operation_id = uncertainty.operation_id
                AND reservation.funding_source = uncertainty.funding_source
            )
            AND EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS dispatch
              WHERE dispatch.call_identity = uncertainty.call_identity
                AND dispatch.entry_kind = 'dispatch'
                AND dispatch.run_id = uncertainty.run_id
            )
            OFFSET 0
          ) AS proof
          ORDER BY uncertainty.cursor_wrapped, uncertainty.entry_id
          LIMIT CASE
            WHEN p_limit BETWEEN 1 AND 20 THEN p_limit
            ELSE 0
          END
$new_candidate632$);
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(updated,'UTF8')),'hex') IS DISTINCT FROM
      '4554bbd62c6fe41589f8142cf4b92d6845ea2813f466bd77110e809dd6030e72' THEN
    RAISE EXCEPTION 'OpenRouter ordered cost selector postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(pg_get_functiondef(target),'UTF8')),'hex')
       IS DISTINCT FROM '4554bbd62c6fe41589f8142cf4b92d6845ea2813f466bd77110e809dd6030e72'
     OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
    RAISE EXCEPTION 'OpenRouter ordered cost selector readback differs' USING ERRCODE='55000';
  END IF;
END;
$openrouter_ordered_cost632$;
NOTIFY pgrst,'reload schema';
COMMIT;
