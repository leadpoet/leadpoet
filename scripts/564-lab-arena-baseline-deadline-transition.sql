-- Let challengers start after the frozen baseline scoring deadline while an
-- ordinary judge retry continues. This grants no final publication authority.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE arena564_schema_acl ON COMMIT DROP AS
SELECT nspacl AS acl, has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
FROM pg_namespace WHERE nspname='public';
DO $create564$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena564_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$create564$;

DO $preflight564$
DECLARE p RECORD;
BEGIN
  SELECT proc.*, language.lanname INTO p FROM pg_proc proc
  JOIN pg_language language ON language.oid=proc.prolang
  WHERE proc.oid=to_regprocedure('public.lab_arena__baseline_transition_ready_v1(text,text)');
  IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex') <> '3ec29a8396fa3246e774fc7b21ad6bb0a5dd56c15b81fce950d20b83d622ee0c'
    OR pg_get_userbyid(p.proowner) <> 'lab_arena_owner'
    OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
    OR NOT p.prosecdef OR p.provolatile <> 's' OR p.lanname <> 'plpgsql'
    OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena baseline transition helper source/security differs';
  END IF;
END;
$preflight564$;

CREATE OR REPLACE FUNCTION public.lab_arena__baseline_transition_ready_v1(
  p_round_id TEXT, p_submission_id TEXT
) RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $ready564$
DECLARE
  r public.lab_arena_rounds;
  source public.lab_arena_runs;
  judge public.lab_arena_runs;
  position INTEGER;
  total INTEGER;
  first_count INTEGER;
  missing INTEGER := 0;
BEGIN
  SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id;
  IF NOT FOUND OR r.status <> 'stage1_judged'
    OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
    OR r.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
    OR COALESCE(r.configuration_doc->>'sourcing_cost_eligibility_policy','')
      NOT IN ('successful_calls_per_icp_v1','successful_calls_flat_per_icp_v1')
    OR NOT public.lab_arena_judge_continuation_enabled_v1(p_round_id)
    OR jsonb_typeof(r.configuration_doc#>'{schedule,stage_1_scoring_close}') IS DISTINCT FROM 'string'
    OR now() < (r.configuration_doc#>>'{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
    OR (SELECT count(*) FROM jsonb_array_elements(r.participants) p
      WHERE p->'is_king'='true'::JSONB) <> 1
    OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(r.participants) p
      WHERE p->>'submission_id'=p_submission_id AND p->'is_king'='true'::JSONB)
    THEN RETURN FALSE; END IF;
  first_count := (r.configuration_doc->>'stage_1_icp_count')::INTEGER;
  total := first_count + (r.configuration_doc->>'stage_2_icp_count')::INTEGER;
  IF total IS NULL OR total NOT BETWEEN 2 AND 100 OR first_count < 1
    OR total <= first_count THEN RETURN FALSE; END IF;
  FOR position IN 0..total-1 LOOP
    -- Require the preserved accepted source, including its exact baseline
    -- identity. Missing executions retain only the existing publication proof.
    IF (SELECT count(*) FROM public.lab_arena_runs run
      WHERE run.round_id=p_round_id AND run.submission_id=p_submission_id
        AND run.kind='execute' AND run.stage=1 AND run.icp_position=position
        AND run.status='accepted') <> 1 THEN
      RETURN FALSE;
    END IF;
    SELECT * INTO STRICT source FROM public.lab_arena_runs run
    WHERE run.round_id=p_round_id AND run.submission_id=p_submission_id
      AND run.kind='execute' AND run.stage=1 AND run.icp_position=position
      AND run.status='accepted';
    IF source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
      OR source.result_doc IS NULL OR source.assignment_id IS DISTINCT FROM
        p_round_id||':'||p_submission_id||':1:'||position::TEXT
      OR source.run_id IS DISTINCT FROM source.assignment_id||':'||source.attempt::TEXT
      OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(r.participants) p
        WHERE p->>'submission_id'=p_submission_id AND p->'is_king'='true'::JSONB
          AND p->>'miner_hotkey'=source.miner_hotkey) THEN RETURN FALSE; END IF;
    IF source.per_icp_score IS NOT NULL THEN
      IF NOT COALESCE((public.lab_arena__integrity_submission_summary(
        p_round_id,p_submission_id,ARRAY[position])->>'valid')::BOOLEAN,FALSE) THEN
        RETURN FALSE;
      END IF;
      CONTINUE;
    END IF;
    missing := missing + 1;
    -- Accepted current-generation authority must be scored normally; an old
    -- accepted cache alias must never be replaced by an incomplete result.
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs accepted
      WHERE accepted.round_id=p_round_id AND accepted.kind='score'
        AND accepted.scored_run_id=source.run_id AND accepted.status='accepted'
        AND accepted.scoring_generation_floor=r.scoring_generation_floor) THEN
      RETURN FALSE;
    END IF;
    SELECT * INTO judge FROM public.lab_arena_runs run
    WHERE run.round_id=p_round_id AND run.submission_id=p_submission_id
      AND run.kind='score' AND run.scored_run_id=source.run_id
      AND run.scoring_generation_floor=r.scoring_generation_floor
    ORDER BY run.stage_generation DESC,run.attempt DESC,run.run_id DESC LIMIT 1;
    IF NOT FOUND OR judge.status NOT IN ('pending','leased') OR judge.attempt <= 1
      OR NOT public.lab_arena__judge_continuation_v1(r,judge) THEN
      RETURN FALSE;
    END IF;
  END LOOP;
  RETURN missing > 0;
END;
$ready564$;
ALTER FUNCTION public.lab_arena__baseline_transition_ready_v1(TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_transition_ready_v1(TEXT,TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $transition564$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_transition_round(text,text,text,jsonb)';
  definition TEXT;
  identity JSONB;
  hash TEXT;
  old_text CONSTANT TEXT := $old$          AND public.lab_arena__publication_execution_incomplete_v1(
            p_round_id, baseline ->> 'submission_id'
          )$old$;
  new_text CONSTANT TEXT := $new$          -- lab_arena_baseline_deadline_transition_564: stage readiness
          -- permits a live retry; final publication keeps its separate proof.
          AND (public.lab_arena__publication_execution_incomplete_v1(
            p_round_id, baseline ->> 'submission_id'
          ) OR public.lab_arena__baseline_transition_ready_v1(
            p_round_id, baseline ->> 'submission_id'
          ))$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(pg_get_userbyid(p.proowner),
    p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO definition,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=to_regprocedure(signature);
  IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public","statement_timeout=600s"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Arena baseline transition RPC security differs';
  END IF;
  hash := encode(extensions.digest(definition,'sha256'),'hex');
  IF hash='90fd40203320e777844a38c430aa7f12c12ed99e32df9beabebb50bb7a068a17' THEN RETURN; END IF;
  IF hash <> '7fd92dc70dfad3a16090d9ad40ac739c9e94d74db3f81b6ac6c311baeac2c4cd' THEN
    RAISE EXCEPTION 'Arena baseline transition RPC preimage differs';
  END IF;
  IF length(definition)-length(replace(definition,old_text,'')) <> length(old_text) THEN
    RAISE EXCEPTION 'Arena baseline transition RPC fragment differs';
  END IF;
  definition := replace(definition,old_text,new_text);
  IF encode(extensions.digest(definition,'sha256'),'hex') <> '90fd40203320e777844a38c430aa7f12c12ed99e32df9beabebb50bb7a068a17' THEN
    RAISE EXCEPTION 'Arena baseline transition RPC postimage differs';
  END IF;
  EXECUTE definition;
  IF encode(extensions.digest(pg_get_functiondef(signature::REGPROCEDURE),'sha256'),'hex') <> '90fd40203320e777844a38c430aa7f12c12ed99e32df9beabebb50bb7a068a17' THEN
    RAISE EXCEPTION 'Arena baseline transition RPC readback differs';
  END IF;
END;
$transition564$;

DO $restore564$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena564_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')
    IS DISTINCT FROM (SELECT acl FROM pg_temp.arena564_schema_acl) THEN
    RAISE EXCEPTION 'Arena baseline transition schema ACL changed';
  END IF;
END;
$restore564$;
COMMIT;
