-- Existing and future baseline-first rounds can execute miners after all
-- baseline source positions are accepted. Frozen configuration and judgments
-- remain unchanged; complete scoring still gates publication and rewards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena572_acl ON COMMIT DROP AS
SELECT nspacl AS acl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper572$
DECLARE p RECORD;
BEGIN
  SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
  WHERE proc.oid=to_regprocedure('public.lab_arena__baseline_execution_ready_v1(text)');
  IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'fe4e47738d0debc59a20e347d758fcd9248bb342348ab0c98d173f5e5147f7f1'
    OR pg_get_userbyid(p.proowner)<>'lab_arena_owner'
    OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
    OR NOT p.prosecdef OR p.provolatile<>'s' OR p.lanname<>'plpgsql'
    OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena concurrent baseline helper source/security differs'; END IF;
END;
$helper572$;

CREATE OR REPLACE FUNCTION public.lab_arena__baseline_execution_ready_v1(p_round_id TEXT)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $ready572$
DECLARE r public.lab_arena_rounds; b JSONB; source public.lab_arena_runs;
  position INTEGER; total INTEGER;
BEGIN
  SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id;
  IF NOT FOUND OR r.status NOT IN ('stage1_scoring','stage1_judged','stage2')
    OR r.publication_doc IS NOT NULL OR r.benchmark_ref IS NULL
    OR r.stage1_scoring_plan_doc IS NULL
    OR now() >= (r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
    OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
    OR NOT public.lab_arena_judge_continuation_enabled_v1(p_round_id)
    OR (SELECT count(*) FROM jsonb_array_elements(r.participants) p
      WHERE p->'is_king'='true'::JSONB)<>1 THEN RETURN FALSE; END IF;
  SELECT p INTO b FROM jsonb_array_elements(r.participants) p WHERE p->'is_king'='true'::JSONB;
  total := (r.configuration_doc->>'stage_1_icp_count')::INTEGER
    + (r.configuration_doc->>'stage_2_icp_count')::INTEGER;
  IF total IS NULL OR total NOT BETWEEN 2 AND 100 THEN RETURN FALSE; END IF;
  FOR position IN 0..total-1 LOOP
    IF (SELECT count(*) FROM public.lab_arena_runs run WHERE run.round_id=p_round_id
      AND run.submission_id=b->>'submission_id' AND run.stage=1 AND run.kind='execute'
      AND run.icp_position=position AND run.status='accepted')<>1 THEN RETURN FALSE; END IF;
    SELECT * INTO STRICT source FROM public.lab_arena_runs run WHERE run.round_id=p_round_id
      AND run.submission_id=b->>'submission_id' AND run.stage=1 AND run.kind='execute'
      AND run.icp_position=position AND run.status='accepted';
    IF source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
      OR source.result_doc IS NULL OR source.miner_hotkey IS DISTINCT FROM b->>'miner_hotkey'
      OR source.assignment_id IS DISTINCT FROM p_round_id||':'||(b->>'submission_id')||':1:'||position::TEXT
      OR source.run_id IS DISTINCT FROM source.assignment_id||':'||source.attempt::TEXT
      OR EXISTS (SELECT 1 FROM public.lab_arena_runs other WHERE other.round_id=p_round_id
        AND other.assignment_id=source.assignment_id AND other.run_id<>source.run_id
        AND other.status IN ('pending','leased','submitted')) THEN RETURN FALSE; END IF;
  END LOOP;
  RETURN TRUE;
END;
$ready572$;
ALTER FUNCTION public.lab_arena__baseline_execution_ready_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_execution_ready_v1(TEXT)
FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $patch572$
DECLARE d TEXT; metadata JSONB; target OID;
  old_gate CONSTANT TEXT := $old$  IF v_round.status <> v_expected THEN$old$;
  new_gate CONSTANT TEXT := $new$  -- lab_arena_concurrent_baseline_judging_572
  IF v_round.status <> v_expected AND NOT (
    p_stage=2 AND v_round.status IN ('stage1_scoring','stage1_judged')
    AND clock_timestamp() < (v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
    AND public.lab_arena__baseline_execution_ready_v1(p_round_id)
  ) THEN$new$;
  old_update CONSTANT TEXT := $old$      stage_generation = v_generation
  WHERE round_id = p_round_id;$old$;
  new_update CONSTANT TEXT := $new$      stage_generation = v_generation,
      finalists = CASE WHEN p_stage=2 AND v_round.status IN ('stage1_scoring','stage1_judged')
        THEN COALESCE((SELECT jsonb_agg(p->>'submission_id' ORDER BY p->>'submission_id')
          FROM jsonb_array_elements(v_round.participants) p
          WHERE COALESCE((p->>'is_king')::BOOLEAN,FALSE)=FALSE),'[]'::JSONB)
        ELSE finalists END
  WHERE round_id = p_round_id;$new$;
BEGIN
  target:='public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='v' AND l.lanname='plpgsql'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public','statement_timeout=60s']::TEXT[];
  IF encode(extensions.digest(d,'sha256'),'hex')<>'fd1d9ea1870b749014e0685c10a6c3c1a3fdf68c8042dc02bf8d32162851b05f' THEN
    IF encode(extensions.digest(d,'sha256'),'hex')<>'58b4f914cc2eab90f99d76da4a336aed83f7030f5172147a1b22cfdd9207cf75'
      OR length(d)-length(replace(d,old_gate,''))<>length(old_gate)
      OR length(d)-length(replace(d,old_update,''))<>length(old_update) THEN
      RAISE EXCEPTION 'Arena concurrent baseline stage preimage differs'; END IF;
    EXECUTE replace(replace(d,old_gate,new_gate),old_update,new_update);
  END IF;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'fd1d9ea1870b749014e0685c10a6c3c1a3fdf68c8042dc02bf8d32162851b05f' THEN
    RAISE EXCEPTION 'Arena concurrent baseline stage metadata differs'; END IF;
END;
$patch572$;

DO $claim572$
DECLARE d TEXT; metadata JSONB; target OID;
  old_order CONSTANT TEXT := $old$  ORDER BY
    -- lab_arena_baseline_recovery_priority_v1:$old$;
  new_order CONSTANT TEXT := $new$  ORDER BY
    -- lab_arena_concurrent_baseline_priority_572: keep baseline judges ahead
    -- of the miner backlog; normal admission and retry backoff still apply.
    CASE WHEN runs.kind='score' AND runs.stage=1
      AND v_round.configuration_doc->>'execution_sequence_policy'='baseline_scored_first_v1'
      AND EXISTS (SELECT 1 FROM jsonb_array_elements(v_round.participants) p
        WHERE p->>'submission_id'=runs.submission_id AND p->'is_king'='true'::JSONB)
      THEN 0 ELSE 1 END,
    -- lab_arena_baseline_recovery_priority_v1:$new$;
  old_barrier CONSTANT TEXT := $old$    ) OR NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs baseline$old$;
  new_barrier CONSTANT TEXT := $new$    ) OR public.lab_arena__baseline_execution_ready_v1(p_round_id) OR NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs baseline$new$;
BEGIN
  target:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='v' AND l.lanname='plpgsql'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[];
  IF encode(extensions.digest(d,'sha256'),'hex')<>'a01a0cd11bba2abb77f74d04ee780cab32a04ff2c08eabbb448b90ae280e92f7' THEN
    IF encode(extensions.digest(d,'sha256'),'hex')<>'0b86688d858e6a9a17b011e812541a53041099414df50b3e886a1ede07c35c01'
      OR length(d)-length(replace(d,old_order,''))<>length(old_order)
      OR length(d)-length(replace(d,old_barrier,''))<>length(old_barrier) THEN
      RAISE EXCEPTION 'Arena concurrent baseline claim preimage differs'; END IF;
    EXECUTE replace(replace(d,old_order,new_order),old_barrier,new_barrier);
  END IF;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a01a0cd11bba2abb77f74d04ee780cab32a04ff2c08eabbb448b90ae280e92f7' THEN
    RAISE EXCEPTION 'Arena concurrent baseline claim metadata differs'; END IF;
END;
$claim572$;
DO $restore572$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena572_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT acl FROM pg_temp.arena572_acl) THEN
    RAISE EXCEPTION 'Arena concurrent baseline schema ACL changed'; END IF;
END;
$restore572$;
COMMIT;
