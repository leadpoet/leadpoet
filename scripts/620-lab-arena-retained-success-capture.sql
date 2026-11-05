-- Retain exact sanitized OpenRouter successes across billing uncertainty.
-- Change only the existing aggregate selectors. Costs, ledger rows, token
-- parsing, function signatures, owners, ACLs and role memberships stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $capture_terminal620$
DECLARE
  target RECORD;
  function_oid OID;
  definition TEXT;
  metadata JSONB;
  source TEXT;
  updated TEXT;
  old_selector CONSTANT TEXT := $old620$    SELECT reserved.started_at, heads.created_at AS finished_at,
      l.terminal_response -> 'body_b64' AS body
    FROM heads
    JOIN reserved ON reserved.call_identity = heads.call_identity
    JOIN public.lab_arena_ledger AS l ON l.entry_id = heads.entry_id
    WHERE l.terminal_response -> 'call_succeeded' = 'true'::JSONB
      AND jsonb_typeof(l.terminal_response -> 'status') = 'number'
      AND (l.terminal_response ->> 'status') ~ '^2[0-9][0-9]$'$old620$;
  new_selector CONSTANT TEXT := $new620$    SELECT reserved.started_at, heads.created_at AS finished_at,
      selected.terminal -> 'body_b64' AS body
    FROM heads
    JOIN reserved ON reserved.call_identity = heads.call_identity
    JOIN public.lab_arena_ledger AS l ON l.entry_id = heads.entry_id
    JOIN public.lab_arena_ledger AS reservation
      ON reservation.call_identity = heads.call_identity AND reservation.entry_kind = 'reservation'
    LEFT JOIN LATERAL (
      -- Only the explicit sanitized reply for this exact reservation can
      -- survive a billing-only head. Private upstream receipts are not replies.
      SELECT u.entry_doc #> '{call,openrouter_terminal_response}' AS terminal
      FROM public.lab_arena_ledger AS u
      WHERE u.run_id = l.run_id AND u.call_identity = l.call_identity
        AND u.entry_kind = 'uncertain' AND u.entry_id > reservation.entry_id AND u.entry_id <= l.entry_id
        AND u.round_id = l.round_id AND u.submission_id = l.submission_id
        AND u.provider = l.provider AND u.operation_id = l.operation_id
        AND u.miner_hotkey = l.miner_hotkey AND u.stage = l.stage AND u.funding_source = l.funding_source
        AND reservation.run_id = l.run_id AND reservation.round_id = l.round_id
        AND reservation.submission_id = l.submission_id AND reservation.provider = l.provider
        AND reservation.operation_id = l.operation_id
        AND reservation.miner_hotkey = l.miner_hotkey AND reservation.stage = l.stage
        AND reservation.funding_source = l.funding_source
        AND l.round_id = p_round_id AND l.submission_id = p_submission_id
        AND (u.entry_id = l.entry_id OR (
          l.entry_kind = 'settlement'
          AND l.entry_doc -> 'openrouter_delayed_reconciliation' = 'true'::JSONB
          AND NOT l.entry_doc ? 'deepline_delayed_reconciliation'
          AND l.entry_doc -> 'reconciled_uncertainty_entry_id' = to_jsonb(u.entry_id)))
        AND u.entry_doc #>> '{call,reason}' IN ('missing_provider_cost', 'settle_failure')
        AND u.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
        AND jsonb_typeof(u.entry_doc #> '{call,provider_status}') = 'number'
        AND (u.entry_doc #>> '{call,provider_status}') ~ '^2[0-9][0-9]$'
        AND (u.entry_doc #>> '{call,credential_fingerprint}') ~ '^sha256:[0-9a-f]{64}$'
        AND NOT EXISTS (SELECT 1 FROM unnest(ARRAY[
          'credential_fingerprint', 'openrouter_generation_id', 'openrouter_model', 'request_hash']) AS field(name)
          WHERE reservation.entry_doc ? field.name AND (u.entry_doc -> 'call') ? field.name
            AND reservation.entry_doc -> field.name IS DISTINCT FROM u.entry_doc -> 'call' -> field.name)
        AND (NOT (u.entry_doc -> 'call') ? 'openrouter_generation_id'
          OR (u.entry_doc #>> '{call,openrouter_generation_id}') ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$')
        AND u.entry_doc #> '{call,openrouter_terminal_response,call_succeeded}' = 'true'::JSONB
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,status}') = 'number'
        AND (u.entry_doc #>> '{call,openrouter_terminal_response,status}') ~ '^2[0-9][0-9]$'
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,headers}') = 'object'
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,body_b64}') = 'string'
        AND length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') % 4 = 0
        AND (u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') ~ '^[A-Za-z0-9+/]*={0,2}$'
        AND length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') / 4 * 3
          - (length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}')
            - length(rtrim(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}', '='))) <= 16777216
        AND (u.entry_doc #> '{call,openrouter_terminal_response}') - ARRAY[
          'status', 'headers', 'body_b64', 'call_succeeded'] = '{}'::JSONB
        AND NOT (u.entry_doc #> '{call,openrouter_terminal_response}') ?| ARRAY[
          'judgment_cache_source_call_identity', 'judgment_cache_source_run_id']
    ) AS retained ON TRUE
    CROSS JOIN LATERAL (
      SELECT CASE WHEN l.terminal_response -> 'call_succeeded' = 'true'::JSONB
        AND jsonb_typeof(l.terminal_response -> 'status') = 'number'
        AND (l.terminal_response ->> 'status') ~ '^2[0-9][0-9]$'
        THEN l.terminal_response ELSE retained.terminal END AS terminal
    ) AS selected
    WHERE selected.terminal -> 'call_succeeded' = 'true'::JSONB
      AND jsonb_typeof(selected.terminal -> 'status') = 'number'
      AND (selected.terminal ->> 'status') ~ '^2[0-9][0-9]$'$new620$;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('leadpoet.lab-arena-roles.v1', 0));
  FOR target IN SELECT * FROM (VALUES
    (1, '635bc03c9716b4e7980badcfc8333ad727125bad36b0f698fecaa8c7c26bb25c', '448f050af374c82c4431d86f5da0e6282f92c2032b6bcd29062b6ff7c1c4677d'),
    (2, '54642ee1cf1949b52f668fe74c8ca12b3651cfc2467ba1e949032b6fe89f8229', '5148a96c50e2952ec109b4770b59af66e527374126e3c194fd7435ccd4fc407e')
  ) AS versions(version, preimage, postimage)
  LOOP
    function_oid := pg_catalog.to_regprocedure(pg_catalog.format(
      'public.lab_arena_capture_summary_v%s(text,text)', target.version));
    SELECT pg_catalog.pg_get_functiondef(p.oid), p.prosrc, pg_catalog.to_jsonb(p)-'prosrc'
      INTO definition, source, metadata FROM pg_catalog.pg_proc p WHERE p.oid=function_oid;
    IF definition IS NULL THEN
      RAISE EXCEPTION 'capture terminal620 function missing' USING ERRCODE='55000';
    END IF;
    IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
        p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
      IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
      RAISE EXCEPTION 'capture terminal620 security differs' USING ERRCODE='55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(source,'sha256'),'hex')=target.postimage THEN
      CONTINUE;
    END IF;
    IF pg_catalog.encode(extensions.digest(source,'sha256'),'hex')<>target.preimage
       OR length(source)-length(pg_catalog.replace(source,old_selector,''))<>length(old_selector) THEN
      RAISE EXCEPTION 'capture terminal620 preimage differs' USING ERRCODE='55000';
    END IF;
    updated := pg_catalog.replace(definition,old_selector,new_selector);
    EXECUTE updated;
    IF (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=function_oid)
         IS DISTINCT FROM metadata
       OR pg_catalog.encode(extensions.digest((SELECT p.prosrc FROM pg_catalog.pg_proc p
         WHERE p.oid=function_oid),'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'capture terminal620 catalog readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$capture_terminal620$;
NOTIFY pgrst, 'reload schema';
COMMIT;
