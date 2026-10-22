-- Bound judge continuation by the round's own scoring deadline.
-- Script 472 made a failed judgment retry until an accepted judgment exists,
-- with no ceiling (attempt<2147483647). Retries that are still useful are the
-- ones that land before the round closes scoring; afterwards the judgment can
-- no longer be recorded, so every further attempt is spend without an outcome.
-- This bounds the lane at {schedule,final_scoring_close} and leaves every
-- pre-deadline retry exactly as it is today — accepted judgments have been
-- observed as late as attempt 30, so no attempt ceiling is imposed here.
-- Rounds without a scheduled close keep their current behaviour.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';

CREATE OR REPLACE FUNCTION public.lab_arena__judge_deadline_open_v1(p_round_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $deadline$
  -- TRUE while the round can still record a judgment. A round that carries no
  -- scheduled scoring close keeps its existing unbounded behaviour.
  SELECT COALESCE((SELECT
      (round_row.configuration_doc#>>'{schedule,final_scoring_close}') IS NULL
      OR pg_catalog.clock_timestamp()
         < (round_row.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
    FROM public.lab_arena_rounds round_row
    WHERE round_row.round_id=p_round_id),FALSE);
$deadline$;
ALTER FUNCTION public.lab_arena__judge_deadline_open_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__judge_deadline_open_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Patch only exact existing source shapes, retaining their grants and owners.
DO $deadline_patch$
DECLARE p RECORD; definition TEXT; owner_name TEXT; secure BOOLEAN; current_signature TEXT;
BEGIN
  FOR p IN SELECT * FROM (VALUES
    ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)',
     $old$OR NOT public.lab_arena_judge_continuation_enabled_v1(p_round.round_id)$old$,
     $new$OR NOT public.lab_arena_judge_continuation_enabled_v1(p_round.round_id)
    OR NOT public.lab_arena__judge_deadline_open_v1(p_round.round_id)$new$),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
     $old$(v_run.kind='score' AND v_run.attempt<2147483647$old$,
     $new$(v_run.kind='score' AND public.lab_arena__judge_deadline_open_v1(v_run.round_id)$new$)
  ) AS patches(signature,old_text,new_text) LOOP
    IF current_signature IS DISTINCT FROM p.signature THEN
      IF current_signature IS NOT NULL THEN EXECUTE definition; END IF;
      SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
        INTO definition,owner_name,secure FROM pg_proc proc
        WHERE proc.oid=to_regprocedure(p.signature);
      IF definition IS NULL OR owner_name<>'lab_arena_owner' OR secure IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'judge deadline function authority differs: %',p.signature
          USING ERRCODE='55000';
      END IF;
      current_signature:=p.signature;
    END IF;
    IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
    IF (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text)<>1 THEN
      RAISE EXCEPTION 'judge deadline anchor differs: %',p.signature USING ERRCODE='55000';
    END IF;
    definition:=replace(definition,p.old_text,p.new_text);
  END LOOP;
  IF current_signature IS NOT NULL THEN EXECUTE definition; END IF;
END;
$deadline_patch$;
NOTIFY pgrst,'reload schema';
COMMIT;
