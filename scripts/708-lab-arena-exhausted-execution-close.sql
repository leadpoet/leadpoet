-- build_scoring_plan classifies an exhausted provider failure as incomplete.
-- Record the same outcome at closure, using the existing latest-run marker.
-- No rows from stages closed before installation are modified.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $close708$
DECLARE
  target RECORD;
  definition TEXT;
  identity JSONB;
  aggregates TEXT;
  old_aggregates CONSTANT TEXT := $aggregates$,
      (pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]
        AS latest_attempt,
      (pg_catalog.array_agg(runs.status ORDER BY runs.attempt DESC))[1]
        AS latest_status,
      (pg_catalog.array_agg(
        COALESCE(runs.terminal_cause, '') ORDER BY runs.attempt DESC
      ))[1] AS latest_cause,
      (pg_catalog.array_agg(
        runs.terminal_doc ORDER BY runs.attempt DESC
      ))[1] AS latest_terminal_doc$aggregates$;
  old_exclusions CONSTANT TEXT := $exclude$
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'provider_error'
    )
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'stage_closed'
      AND COALESCE(
        outcomes.latest_terminal_doc @>
          '{"deadline_provider_retry_exhausted":true}'::JSONB,
        FALSE
      )
    )$exclude$;
BEGIN
  FOR target IN
    SELECT * FROM (VALUES
      ('public.lab_arena_close_stage(text,smallint)',
       '96db06408278ccad9cca9566457518c2951c944d33149c8efe2336c8cca164aa',
       'd4261afcdc7a08c474ffd3e27befd824fc0d1915d595fcf2f662eed6944e2513'),
      ('public.lab_arena_close_parallel_execution_v1(text)',
       'a4f0ea976da82d09ae19ef7867090ec819cda465a4f5307d30df54dbef8a7609',
       'c4347b7d9b547609dbd41ac9f1e868844b51fe442b609252f5ae6724f1bfdac8')
    ) AS functions(signature, preimage, postimage)
  LOOP
    SELECT pg_get_functiondef(p.oid), to_jsonb(p)-'prosrc'
      INTO STRICT definition, identity
      FROM pg_proc p WHERE p.oid=target.signature::regprocedure;
    IF pg_get_userbyid((identity->>'proowner')::oid) <> 'lab_arena_owner'
       OR NOT (identity->>'prosecdef')::boolean
       OR identity->'proacl' IS DISTINCT FROM
         '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::jsonb THEN
      RAISE EXCEPTION 'Arena execution close authority differs: %',target.signature;
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>target.preimage THEN
      RAISE EXCEPTION 'Arena execution close preimage differs: %',target.signature;
    END IF;
    aggregates := CASE WHEN target.signature='public.lab_arena_close_stage(text,smallint)'
      THEN replace(old_aggregates,'array_agg(runs.attempt ', 'array_agg(public.lab_arena_execution_attempt_count(runs) ')
      ELSE old_aggregates END;
    definition := replace(replace(definition,aggregates,''),old_exclusions,'');
    IF encode(extensions.digest(definition,'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'Arena execution close postimage differs: %',target.signature;
    END IF;
    EXECUTE definition;
    IF NOT EXISTS (SELECT 1 FROM pg_proc p WHERE p.oid=target.signature::regprocedure
        AND to_jsonb(p)-'prosrc'=identity
        AND encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')=target.postimage) THEN
      RAISE EXCEPTION 'Arena execution close readback differs: %',target.signature;
    END IF;
  END LOOP;
END;
$close708$;
COMMIT;
