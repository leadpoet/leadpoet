-- A score submission has one shared provider budget across all ICP assignments.
-- Serialize its live score leases so an unknown-price provider call cannot
-- reserve the remaining budget while a sibling ICP waits and fails admission.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_236_score_submission_serialization$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
$old$;
  v_new TEXT := $new$
    -- lab_arena_score_submission_serialization: a score submission and kind
    -- share one provider budget, so only one current-generation lease may run.
    AND (
      runs.kind <> 'score'
      OR NOT EXISTS (
        SELECT 1
        FROM public.lab_arena_runs AS active_score
        WHERE active_score.round_id = runs.round_id
          AND active_score.stage_generation = runs.stage_generation
          AND active_score.submission_id = runs.submission_id
          AND active_score.kind = 'score'
          AND active_score.status = 'leased'
          AND active_score.lease_expires_at > pg_catalog.clock_timestamp()
      )
    )
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
           'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
         )
  INTO v_definition;

  IF pg_catalog.strpos(v_definition, v_new) > 0 THEN
    RETURN;
  END IF;

  IF pg_catalog.strpos(
       v_definition, 'lab_arena_score_submission_serialization'
     ) > 0 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment serialization is partial';
  END IF;

  -- Patch only the current composed claim function. These markers protect the
  -- scoring authority, accepted-judgment, company-cache, and champion-funding
  -- changes installed by migrations 212, 213, 217, and 227.
  IF pg_catalog.strpos(v_definition, 'runner_authority_exclusions') = 0
     OR pg_catalog.strpos(v_definition, 'judgment_group_leader') = 0
     OR pg_catalog.strpos(v_definition, 'company_judgment_cache') = 0
     OR pg_catalog.strpos(v_definition, 'champion_funding_sources') = 0
     OR pg_catalog.strpos(v_definition, 'lab-arena-claim-control') = 0 THEN
    RAISE EXCEPTION 'apply the current Arena claim migrations before migration 236';
  END IF;
  IF (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
       / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment serialization shape unexpected';
  END IF;

  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$lab_arena_236_score_submission_serialization$;

COMMIT;
