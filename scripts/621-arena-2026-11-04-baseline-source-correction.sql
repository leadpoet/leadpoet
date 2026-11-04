-- Authorized November 4 source correction. Preserve the frozen benchmark,
-- scorer, all miner work and every historical payload and charge.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena621_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- This owner-only predicate admits only the ten roots named by the exact
-- append-only correction receipt. It grants no additional execution retries.
DO $helper_guard621_source621$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena__source_correction621_v1(public.lab_arena_runs)');
 IF NOT FOUND AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
   WHERE round_id='arena-2026-11-04' AND request_id='nov04-baseline-source621-v1') THEN
  RAISE EXCEPTION 'Source correction621 receipt exists but helper is missing'; END IF;
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'f6902036486533a2907a8ede1413ece42ae36d078b0550cd460bd40eaad8c343'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' OR p.prokind<>'f'
  OR (SELECT lanname FROM pg_language WHERE oid=p.prolang)<>'plpgsql'
  OR p.proretset
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Source correction621 installed helper differs'; END IF;
END;
$helper_guard621_source621$;
CREATE OR REPLACE FUNCTION public.lab_arena__source_correction621_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $source621$
DECLARE r public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 item JSONB; original public.lab_arena_runs;
BEGIN
 IF p_run.round_id IS DISTINCT FROM 'arena-2026-11-04'
  OR p_run.submission_id IS DISTINCT FROM 'baseline-2026-11-04'
  OR p_run.kind IS DISTINCT FROM 'execute' OR p_run.stage IS DISTINCT FROM 1
  OR p_run.icp_position NOT BETWEEN 0 AND 9 OR p_run.scoring_generation_floor<>0
  OR p_run.assignment_id IS DISTINCT FROM p_run.round_id||':'||p_run.submission_id||':1:'||p_run.icp_position
  OR p_run.run_id IS DISTINCT FROM p_run.assignment_id||':'||p_run.attempt
  OR p_run.champion_restart_required THEN RETURN FALSE; END IF;
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_run.round_id;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_run.round_id AND request_id='nov04-baseline-source621-v1'
  AND recovery_doc->>'operation'='baseline_source_supersession_v1'
  AND new_round_doc->'configuration_doc'=r.configuration_doc
  AND new_round_doc->'participants'=r.participants
  AND new_round_doc->>'champion_hotkey'=r.champion_hotkey
  AND new_round_doc->>'champion_submission_id'=r.champion_submission_id;
 IF NOT FOUND OR r.scoring_generation_floor<>0
  OR p_run.miner_hotkey IS DISTINCT FROM r.configuration_doc->>'baseline_hotkey'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions s
   WHERE s.submission_id=p_run.submission_id AND s.status='frozen' AND s.is_king
    AND s.miner_hotkey=p_run.miner_hotkey
    AND (to_jsonb(s)-'updated_at')=h.recovery_doc->'new_submission_doc') THEN RETURN FALSE; END IF;
 SELECT value INTO item FROM jsonb_array_elements(h.recovery_doc->'work_items')
 WHERE value->>'icp_position'=p_run.icp_position::TEXT
  AND p_run.attempt BETWEEN (value->>'new_attempt')::INTEGER
   AND (value->>'new_attempt')::INTEGER+(r.configuration_doc->>'max_attempts_per_assignment')::INTEGER-1;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 -- The frozen normal attempt budget starts at this source generation's root.
 -- Only complete failed infrastructure lineage can consume its next attempt.
 IF p_run.attempt>(item->>'new_attempt')::INTEGER AND NOT (
   SELECT count(*)=p_run.attempt-(item->>'new_attempt')::INTEGER
    AND coalesce(bool_and(parent.status='failed' AND parent.kind='execute'
     AND parent.terminal_cause IN('provider_error','lease_expired')
     AND parent.output_ref IS NULL AND NOT parent.champion_restart_required
     AND (parent.terminal_cause='lease_expired'
       OR parent.result_doc->>'terminal_status'=parent.terminal_cause)
     AND parent.run_id=parent.assignment_id||':'||parent.attempt),FALSE)
   FROM public.lab_arena_runs parent WHERE parent.assignment_id=p_run.assignment_id
    AND parent.attempt BETWEEN (item->>'new_attempt')::INTEGER AND p_run.attempt-1
 ) THEN RETURN FALSE; END IF;
 SELECT * INTO original FROM public.lab_arena_runs WHERE run_id=item->>'old_run_id';
 RETURN COALESCE(original.status='failed' AND original.terminal_cause='stage_closed'
  AND original.terminal_doc=jsonb_build_object('operation','baseline_source_supersession_v1',
    'request_id',h.request_id,'new_execution_run_id',item->>'new_run_id')
  AND (to_jsonb(original)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
    ((h.recovery_doc->'superseded_runs'->original.run_id)-
      ARRAY['status','terminal_cause','terminal_doc','updated_at']),FALSE);
END;
$source621$;
ALTER FUNCTION public.lab_arena__source_correction621_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__source_correction621_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE TEMP TABLE arena621_final_functions(signature TEXT,sha256 TEXT,security JSONB) ON COMMIT DROP;
INSERT INTO arena621_final_functions VALUES
('public.lab_arena_runs_terminal_v1()','3d4003101e7638b211885ccdb7fe697106dc85326959ed577276894f99e3dbcf',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":false,"provolatile":"v","proconfig":["search_path=pg_catalog"]}$security621$::jsonb),
('public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)','8b0c5de078a5d45786164d4ad5b202c31b6f63885cb069b45163677fca84bda4',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb),
('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','4a993c5395f6aef99a692b58fb1001fc2f25259edca80ab8be4c503b7431b700',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb),
('public.lab_arena_recover_baseline_v1(text,jsonb)','a9694575a9c092b0e67ee0a2fcb8f7d69633b35bb228c45dcaa03dd42bfdbaea',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb),
('public.lab_arena_submissions_frozen_v1()','c7f91e17eaca4d726472d3e80c2af452ba4854ced1e3a200d64c9a1304e20ad4',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":false,"provolatile":"v","proconfig":["search_path=pg_catalog"]}$security621$::jsonb),
('public.lab_arena_source_digests_immutable_v1()','33d21ea2929eb3ec750435ca6e774b73e9cb0dee83f438269d3e1671a2655b78',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}","prosecdef":false,"provolatile":"v","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb),
('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb),
('public.lab_arena_freeze_champion_funding(text)','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151',$security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"]}$security621$::jsonb);
DO $security621$
DECLARE expected RECORD; actual JSONB;
BEGIN
 FOR expected IN SELECT * FROM pg_temp.arena621_final_functions LOOP
  SELECT jsonb_build_object('owner',pg_get_userbyid(p.proowner),'acl',p.proacl::TEXT,
   'prosecdef',p.prosecdef,'provolatile',p.provolatile,'proconfig',p.proconfig) INTO actual
   FROM pg_proc p WHERE p.oid=to_regprocedure(expected.signature);
  IF actual IS DISTINCT FROM expected.security THEN
   RAISE EXCEPTION 'Source correction621 security differs: %',expected.signature; END IF;
 END LOOP;
END;
$security621$;
DO $patch621$
DECLARE p RECORD; expected RECORD; target OID; d TEXT; metadata JSONB;
BEGIN
 FOR expected IN SELECT * FROM pg_temp.arena621_final_functions LOOP
  target:=to_regprocedure(expected.signature);
  SELECT pg_get_functiondef(proc.oid),to_jsonb(proc)-'prosrc' INTO d,metadata FROM pg_proc proc WHERE proc.oid=target;
  IF d IS NULL OR (SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid=target)<>'lab_arena_owner' THEN
    RAISE EXCEPTION 'Source correction621 function owner differs: %',expected.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=expected.sha256 THEN CONTINUE; END IF;
  FOR p IN SELECT * FROM (VALUES
 (1,'public.lab_arena_runs_terminal_v1()','a59bf87b8823352c01c4ee4714e4475a857e0c6eda3092e447760ad3afcbd21e','3d4003101e7638b211885ccdb7fe697106dc85326959ed577276894f99e3dbcf',$old621_0$BEGIN
$old621_0$,$new621_0$BEGIN
  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-04'
   AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1
   AND OLD.kind IN('execute','score') AND OLD.status='accepted' AND OLD.terminal_cause='accepted'
   AND OLD.icp_position BETWEEN 0 AND 9 AND OLD.scoring_generation_floor=0
   AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
   AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
   AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h,
     jsonb_array_elements(h.recovery_doc->'work_items') item
    WHERE h.round_id=OLD.round_id AND h.request_id='nov04-baseline-source621-v1'
     AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
     AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
     AND h.recovery_doc->'superseded_runs'->OLD.run_id=to_jsonb(OLD)
     AND OLD.icp_position=(item->>'icp_position')::INTEGER
     AND OLD.run_id IN(item->>'old_run_id',item->>'old_judge_run_id')
     AND NEW.terminal_doc=jsonb_build_object('operation','baseline_source_supersession_v1',
      'request_id',h.request_id,'new_execution_run_id',item->>'new_run_id')) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new621_0$),
 (2,'public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)','d7003c494b9ab2ae572562713ef8714ac6f7a26158ca3cd2b01d2489ac43d781','8b0c5de078a5d45786164d4ad5b202c31b6f63885cb069b45163677fca84bda4',$old621_1$  IF p_run.kind='score' THEN$old621_1$,$new621_1$  IF public.lab_arena__source_correction621_v1(p_run) THEN RETURN TRUE; END IF;
  IF p_run.kind='score' AND EXISTS(SELECT 1 FROM public.lab_arena_runs source
    WHERE source.run_id=p_run.scored_run_id AND source.status='accepted'
     AND source.terminal_cause='accepted' AND source.output_ref IS NOT NULL
     AND (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)=
       (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
     AND public.lab_arena__source_correction621_v1(source)) THEN RETURN TRUE; END IF;
  IF p_run.kind='score' THEN$new621_1$),
 (3,'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','05818529181013b76affbe6253814cce6f6cbc2d4bdd0377f5ce57ef89191606','8eae70c79442f802bc76e2bd39f63503b4c074dedc2f4b83e05ef1df132279f3',$old621_2$      WHERE recovered_source.run_id=p_run.scored_run_id
        AND recovered_source.kind='execute' AND recovered_source.attempt>
          (p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
$old621_2$,$new621_2$      WHERE recovered_source.run_id=p_run.scored_run_id
        AND recovered_source.kind='execute' AND (recovered_source.attempt>
          (p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
$new621_2$),
 (4,'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','8eae70c79442f802bc76e2bd39f63503b4c074dedc2f4b83e05ef1df132279f3','9c38b30a84e293d01819de445916e747edd9965985ddd867afefd69ac460cfcc',$old621_3$          (p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
        AND public.lab_arena__baseline_recovery_run_v1(recovered_source))
$old621_3$,$new621_3$          (p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
          OR public.lab_arena__source_correction621_v1(recovered_source))
        AND public.lab_arena__baseline_recovery_run_v1(recovered_source))
$new621_3$),
 (5,'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','9c38b30a84e293d01819de445916e747edd9965985ddd867afefd69ac460cfcc','4a993c5395f6aef99a692b58fb1001fc2f25259edca80ab8be4c503b7431b700',$old621_4$    ) THEN RETURN TRUE; END IF;
  -- The operator-created first attempt of this grading revision is a root;
$old621_4$,$new621_4$    ) THEN RETURN TRUE; END IF;
  IF EXISTS(SELECT 1 FROM public.lab_arena_runs source,
    public.lab_arena_scoring_supersessions h,jsonb_array_elements(h.recovery_doc->'work_items') item
    WHERE source.run_id=p_run.scored_run_id AND public.lab_arena__source_correction621_v1(source)
     AND h.round_id=p_round.round_id AND h.request_id='nov04-baseline-source621-v1'
     AND item->>'icp_position'=source.icp_position::TEXT
     AND item->>'new_judge_attempt'=p_run.attempt::TEXT
     AND p_run.run_id=source.assignment_id||':score:'||p_run.attempt
     AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs later
       WHERE later.assignment_id=p_run.assignment_id AND later.attempt>p_run.attempt)) THEN RETURN TRUE; END IF;
  -- The operator-created first attempt of this grading revision is a root;
$new621_4$),
 (6,'public.lab_arena_recover_baseline_v1(text,jsonb)','f395bb52ebe3f91bd3b7702594a10f3e953e2a32df113bffdb55c7444fa6e3a4','4820898e83da994fc5850750d06e0c9aeb3139ff02ac5b7b6273ee7a67021be1',$old621_5$  v_new public.lab_arena_runs; v_source public.lab_arena_runs;
  v_item JSONB; v_created INTEGER:=0; v_state JSONB; v_cache public.lab_arena_judgment_cache;
BEGIN
$old621_5$,$new621_5$  v_new public.lab_arena_runs; v_source public.lab_arena_runs;
  v_item JSONB; v_score_attempt INTEGER; v_created INTEGER:=0; v_state JSONB; v_cache public.lab_arena_judgment_cache;
BEGIN
$new621_5$),
 (7,'public.lab_arena_recover_baseline_v1(text,jsonb)','4820898e83da994fc5850750d06e0c9aeb3139ff02ac5b7b6273ee7a67021be1','309d26f9d6c51665b5290d958eee5d210d9dff9ed61a08057e5f74ce90c03be4',$old621_6$      IF public.lab_arena__company_judgment_policy_v1(v_round) THEN
        INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
$old621_6$,$new621_6$      IF public.lab_arena__company_judgment_policy_v1(v_round) THEN
        v_score_attempt:=1;
        IF public.lab_arena__source_correction621_v1(v_source) THEN
          SELECT (item->>'new_judge_attempt')::INTEGER INTO STRICT v_score_attempt
          FROM public.lab_arena_scoring_supersessions h,jsonb_array_elements(h.recovery_doc->'work_items') item
          WHERE h.round_id=p_round_id AND h.request_id='nov04-baseline-source621-v1'
            AND item->>'icp_position'=v_source.icp_position::TEXT;
        END IF;
        INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
$new621_6$),
 (8,'public.lab_arena_recover_baseline_v1(text,jsonb)','309d26f9d6c51665b5290d958eee5d210d9dff9ed61a08057e5f74ce90c03be4','f52946ab995dcec33dfcb0f6ee66a92a5845d33fff919b315ff4d465529c0519',$old621_7$          icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
        VALUES(v_source.assignment_id||':score:1',v_source.assignment_id||':score',p_round_id,
          v_source.submission_id,v_source.miner_hotkey,1,v_source.icp_position,1,'pending',
          v_round.stage_generation,'score',v_source.run_id,v_item->'company_judgment_refs');
$old621_7$,$new621_7$          icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
        VALUES(v_source.assignment_id||':score:'||v_score_attempt,v_source.assignment_id||':score',p_round_id,
          v_source.submission_id,v_source.miner_hotkey,1,v_source.icp_position,v_score_attempt,'pending',
          v_round.stage_generation,'score',v_source.run_id,v_item->'company_judgment_refs');
$new621_7$),
 (9,'public.lab_arena_recover_baseline_v1(text,jsonb)','f52946ab995dcec33dfcb0f6ee66a92a5845d33fff919b315ff4d465529c0519','a9694575a9c092b0e67ee0a2fcb8f7d69633b35bb228c45dcaa03dd42bfdbaea',$old621_8$    AND r.stage_generation<>v_round.stage_generation AND public.lab_arena__baseline_recovery_run_v1(r);
  RETURN jsonb_build_object('status','ok','queued',v_created);
$old621_8$,$new621_8$    AND r.stage_generation<>v_round.stage_generation AND public.lab_arena__baseline_recovery_run_v1(r);
  IF p_round_id='arena-2026-11-04'
    AND (SELECT count(*)=10 AND count(DISTINCT icp_position)=10 FROM public.lab_arena_runs a
     WHERE a.round_id=p_round_id AND a.kind='execute' AND a.status='accepted'
      AND a.terminal_cause='accepted' AND a.output_ref IS NOT NULL
      AND public.lab_arena__source_correction621_v1(a)) THEN
    PERFORM public.lab_arena__source_correction_plan621_v1(p_round_id);
  END IF;
  RETURN jsonb_build_object('status','ok','queued',v_created);
$new621_8$),
 (10,'public.lab_arena_submissions_frozen_v1()','73a0ebc54d35f76b18fc0b1a83b3ab36ed9f793748274b3ea32ea96584da6fb1','c7f91e17eaca4d726472d3e80c2af452ba4854ced1e3a200d64c9a1304e20ad4',$old621_9$BEGIN
$old621_9$,$new621_9$BEGIN
  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.is_king AND OLD.status='frozen'
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.request_id='nov04-baseline-source621-v1'
       AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
       AND h.recovery_doc->'old_submission_doc'=to_jsonb(OLD)-'updated_at'
       AND h.recovery_doc->'new_submission_doc'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new621_9$),
 (11,'public.lab_arena_source_digests_immutable_v1()','59294764e68968e6591cefcf724a1551892396578233bc624f510241875775de','33d21ea2929eb3ec750435ca6e774b73e9cb0dee83f438269d3e1671a2655b78',$old621_10$BEGIN
$old621_10$,$new621_10$BEGIN
  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.is_king AND OLD.status='frozen'
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.request_id='nov04-baseline-source621-v1'
       AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
       AND h.recovery_doc->'old_submission_doc'=to_jsonb(OLD)-'updated_at'
       AND h.recovery_doc->'new_submission_doc'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new621_10$),
 (12,'public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','6cb941811025501152d69891d190b61341acabce0b710b02322f4325a62dbf59','caf315c8c6c4c15d946ba70b00b0a7c4f893f42ada5e77b3af22fe4fdea8575b',$old621_11$  IF current_round.champion_funding_frozen OR current_round.status <> 'open' THEN$old621_11$,$new621_11$  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=current_round.arena_network_name
      AND correction.arena_netuid=current_round.arena_netuid
      AND correction.evaluation_date<current_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
  IF current_round.champion_funding_frozen OR current_round.status <> 'open' THEN$new621_11$),
 (13,'public.lab_arena_freeze_champion_funding(text)','558a380c6831c6747e8c5a47e59777549e8c1d66d0a0c3a88c058c8ccfb9a9f8','7633073e713c6969e550c1316d8531426337ada442e563f6bdc8d399bc4c3151',$old621_12$CREATE OR REPLACE FUNCTION public.lab_arena_freeze_champion_funding(p_round_id text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002'; END IF;
$old621_12$,$new621_12$CREATE OR REPLACE FUNCTION public.lab_arena_freeze_champion_funding(p_round_id text)
 RETURNS jsonb
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002'; END IF;
  -- The promotion lock and target row lock cover this fresh receipt read.
  IF EXISTS(
    SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
    WHERE h.old_round_doc->>'status'='published'
      AND correction.status NOT IN('published','cancelled')
      AND correction.arena_network_name=v_round.arena_network_name
      AND correction.arena_netuid=v_round.arena_netuid
      AND correction.evaluation_date<v_round.evaluation_date) THEN
    RETURN jsonb_build_object('status','promotion_pending');
  END IF;
$new621_12$)
 ) edits(ordinal,signature,preimage,postimage,old_text,new_text)
 WHERE edits.signature=expected.signature ORDER BY edits.ordinal LOOP
   IF encode(extensions.digest(d,'sha256'),'hex')<>p.preimage
    OR length(d)-length(replace(d,p.old_text,''))<>length(p.old_text) THEN
    RAISE EXCEPTION 'Source correction621 function preimage or anchor differs: %',p.signature; END IF;
   d:=replace(d,p.old_text,p.new_text);
   IF encode(extensions.digest(d,'sha256'),'hex')<>p.postimage THEN
    RAISE EXCEPTION 'Source correction621 source hunk differs: %',p.signature; END IF;
  END LOOP;
  IF encode(extensions.digest(d,'sha256'),'hex')<>expected.sha256 THEN
   RAISE EXCEPTION 'Source correction621 final source differs: %',expected.signature; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected.sha256 THEN
    RAISE EXCEPTION 'Source correction621 readback differs: %',expected.signature; END IF;
 END LOOP;
END;
$patch621$;
-- Install only the canonical plan derived from all ten genuinely accepted new
-- executions. A second receipt authorizes this one write-once plan transition.
DO $helper_guard621_plan621$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena__source_correction_plan621_v1(text)');
 IF NOT FOUND AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
   WHERE round_id='arena-2026-11-04' AND request_id='nov04-baseline-source621-v1') THEN
  RAISE EXCEPTION 'Source correction621 receipt exists but helper is missing'; END IF;
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'ec48228de35ef8a4d1419235e6180831f9256f3458a41d64ae8ded3012d65d09'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v'
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' OR p.prokind<>'f'
  OR (SELECT lanname FROM pg_language WHERE oid=p.prolang)<>'plpgsql'
  OR p.proretset
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Source correction621 installed helper differs'; END IF;
END;
$helper_guard621_plan621$;
CREATE OR REPLACE FUNCTION public.lab_arena__source_correction_plan621_v1(p_round_id TEXT)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public AS $plan621$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 plan JSONB; revision BIGINT;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round_id AND request_id='nov04-baseline-source621-v1'
  AND recovery_doc->>'operation'='baseline_source_supersession_v1';
 IF NOT FOUND OR p_round_id<>'arena-2026-11-04' OR r.status<>'stage2_scoring'
  OR r.configuration_doc IS DISTINCT FROM h.new_round_doc->'configuration_doc'
  OR (SELECT count(*)<>10 OR count(DISTINCT icp_position)<>10 FROM public.lab_arena_runs a
    WHERE a.round_id=p_round_id AND a.status='accepted' AND a.terminal_cause='accepted'
      AND a.output_ref IS NOT NULL AND public.lab_arena__source_correction621_v1(a)) THEN
  RAISE EXCEPTION 'Source correction621 complete source plan missing'; END IF;
 SELECT jsonb_build_object('schema_version','leadpoet.lab_arena.scoring_plan.v1',
   'round_id',p_round_id,'stage',1,'execution_sequence_policy','baseline_scored_first_v1',
   'zero_rows','[]'::JSONB,'work_items',jsonb_agg(jsonb_build_object(
     'scored_run_id',a.run_id,'submission_id',a.submission_id,
     'icp_position',a.icp_position,'output_ref',a.output_ref) ORDER BY a.run_id)) INTO plan
 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status='accepted'
  AND public.lab_arena__source_correction621_v1(a);
 IF r.stage1_scoring_plan_doc=plan THEN RETURN; END IF;
 IF r.stage1_scoring_plan_doc IS DISTINCT FROM h.old_round_doc->'stage1_scoring_plan_doc'
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
   WHERE round_id=p_round_id AND request_id='nov04-baseline-source621-plan-v1') THEN
  RAISE EXCEPTION 'Source correction621 original plan differs'; END IF;
 n:=r; n.stage1_scoring_plan_doc:=plan;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(p_round_id,revision,'nov04-baseline-source621-plan-v1',
  'sha256:'||encode(extensions.digest(plan::TEXT,'sha256'),'hex'),'nov04-baseline-source621-plan',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','baseline_source_replacement_plan_v1','scoring_generation_floor',0));
 UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc=plan WHERE round_id=p_round_id;
END;
$plan621$;
ALTER FUNCTION public.lab_arena__source_correction_plan621_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__source_correction_plan621_v1(TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
-- Retired source authorities remain immutable evidence. Fresh company and
-- signal lookups treat only those exact retired baseline authorities as misses.
DO $cache621$
DECLARE p RECORD; target OID; d TEXT; metadata JSONB;
BEGIN
 FOR p IN SELECT * FROM (VALUES
('public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)','47a767d79cc8a47ad3d653f940e5987ce1ae210701ec41f97e781b498b1ce250','b2f39b095a1596b35244d9da68ab591b3c50b733d8d1e1b5dcfbcd5668f9ee32',$oldcache621_0$  AND j.signal_input_hash=v_hash$oldcache621_0$,$newcache621_0$  AND j.signal_input_hash=v_hash
  AND NOT EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions retired621
    WHERE retired621.round_id='arena-2026-11-04'
      AND retired621.request_id='nov04-baseline-source621-v1'
      AND retired621.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND retired621.recovery_doc->'superseded_runs' ? j.source_scored_run_id
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'kind'='execute'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'round_id'='arena-2026-11-04'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')$newcache621_0$,$cacheacl621${lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}$cacheacl621$),
('public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamp with time zone)','d29b525fe8eeb48957446c3331adef049764523044546c57418f6d57e39c63dd','8a55713ca73871eddc438082750411cea633c691440d350c17efe40a402fdd53',$oldcache621_1$      AND accepted.company_input_hash = v_ref ->> 'company_input_hash'$oldcache621_1$,$newcache621_1$      AND accepted.company_input_hash = v_ref ->> 'company_input_hash'
  AND NOT EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions retired621
    WHERE retired621.round_id='arena-2026-11-04'
      AND retired621.request_id='nov04-baseline-source621-v1'
      AND retired621.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND retired621.recovery_doc->'superseded_runs' ? accepted.source_scored_run_id
      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'kind'='execute'
      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'round_id'='arena-2026-11-04'
      AND retired621.recovery_doc->'superseded_runs'->accepted.source_scored_run_id->>'submission_id'='baseline-2026-11-04')$newcache621_1$,$cacheacl621${lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}$cacheacl621$)
 ) edits(signature,preimage,postimage,old_text,new_text,acl) LOOP
  target:=to_regprocedure(p.signature);
  SELECT pg_get_functiondef(proc.oid),to_jsonb(proc)-'prosrc' INTO d,metadata
   FROM pg_proc proc WHERE proc.oid=target;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig)
    FROM pg_proc WHERE oid=target) IS DISTINCT FROM jsonb_build_array('lab_arena_owner',p.acl,TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'Source correction621 cache security differs: %',p.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=p.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>p.preimage
    OR length(d)-length(replace(d,p.old_text,''))<>length(p.old_text) THEN
   RAISE EXCEPTION 'Source correction621 cache preimage differs: %',p.signature; END IF;
  EXECUTE replace(d,p.old_text,p.new_text);
  IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>p.postimage THEN
   RAISE EXCEPTION 'Source correction621 cache readback differs: %',p.signature; END IF;
 END LOOP;
END;
$cache621$;

-- Historical calls retain their financial totals. Only the exact31 original
-- source attempt IDs recorded before the correction leave competition budgets.
DO $retired_helper_guard621$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena__retired_source621_v1(public.lab_arena_runs)');
 IF NOT FOUND AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
   WHERE round_id='arena-2026-11-04' AND request_id='nov04-baseline-source621-v1') THEN
  RAISE EXCEPTION 'Source correction621 receipt exists but helper is missing'; END IF;
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'48ac3348ea7b751ad47ce57494e2c857f78e2942563a13dc71a0ad8ed2708ba3'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' OR p.prokind<>'f'
  OR (SELECT lanname FROM pg_language WHERE oid=p.prolang)<>'sql'
  OR p.proretset
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Source correction621 installed retired-source helper differs'; END IF;
END;
$retired_helper_guard621$;
CREATE OR REPLACE FUNCTION public.lab_arena__retired_source621_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $retired621$
 SELECT COALESCE(p_run.round_id='arena-2026-11-04' AND p_run.submission_id='baseline-2026-11-04'
  AND p_run.stage=1 AND p_run.kind IN('execute','score') AND p_run.scoring_generation_floor=0
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-baseline-source621-v1'
    AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
    AND h.recovery_doc->'retired_source_run_hashes' ? p_run.run_id
    AND h.new_round_doc->'configuration_doc'=r.configuration_doc
    AND h.old_round_doc#>>'{configuration_doc,reasoning_baseline_source_sha256}' IS DISTINCT FROM
      h.new_round_doc#>>'{configuration_doc,reasoning_baseline_source_sha256}'),FALSE);
$retired621$;
ALTER FUNCTION public.lab_arena__retired_source621_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__retired_source621_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $cost621$
DECLARE expected RECORD; edit RECORD; target OID; d TEXT; metadata JSONB;
BEGIN
 FOR expected IN SELECT * FROM (VALUES
('public.lab_arena__execute_assignment_calls_v1(text,text)','fbaaeba7355ed524201a5d5381be8791b31424e45875e0d7b6bda5b1a57bece6','9db0dfe3420016d430269efca49878c21f69a489914fef7930958915c3cc87b8',$cost_security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","language":"sql"}$cost_security621$::JSONB),
('public.lab_arena__submission_stage_consumed(text,smallint,text,text)','488e418fcd9c479b6898ac661bb64392a0fcb7cd271039bc28c062c1923e78e9','cf3f9729010d77f0a2f379cdc394b636aa18542df965a3fb7e6023e538f601e7',$cost_security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","language":"sql"}$cost_security621$::JSONB),
('public.lab_arena__successful_icp_cost_state(text,text,integer)','a227265beb00a1918e26079c1b878ab846dd5849d1928e79bfd4fcf519ba4b97','91734409c42ab7cc3b2592dcda1fca79e622d9dae3a9a437a9cb4c6c8febf06a',$cost_security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","language":"sql"}$cost_security621$::JSONB),
('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','2b95e5566b8fed2ddee1fae3525a6bbc0ab5d598001eee650b45653f2dee22a0','8740ff3538cdd0800178711ea8cda5af273b9ede0e8e319d4f1a1a66037fd4ab',$cost_security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"v","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","language":"plpgsql"}$cost_security621$::JSONB)
 ) functions(signature,preimage,postimage,security) LOOP
  target:=to_regprocedure(expected.signature);
  SELECT pg_get_functiondef(proc.oid),to_jsonb(proc)-'prosrc' INTO d,metadata
   FROM pg_proc proc WHERE proc.oid=target;
  IF d IS NULL OR (SELECT jsonb_build_object('owner',pg_get_userbyid(proc.proowner),
    'acl',proc.proacl::TEXT,'prosecdef',proc.prosecdef,'provolatile',proc.provolatile,
    'proconfig',proc.proconfig,'proisstrict',proc.proisstrict,'proleakproof',proc.proleakproof,
    'proparallel',proc.proparallel,'prokind',proc.prokind,'language',lang.lanname)
    FROM pg_proc proc JOIN pg_language lang ON lang.oid=proc.prolang WHERE proc.oid=target)
    IS DISTINCT FROM expected.security THEN
   RAISE EXCEPTION 'Source correction621 cost security differs: %',expected.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=expected.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>expected.preimage THEN
   RAISE EXCEPTION 'Source correction621 cost preimage differs: %',expected.signature; END IF;
  FOR edit IN SELECT * FROM (VALUES
('public.lab_arena__execute_assignment_calls_v1(text,text)',1,$oldcost621_0$   WHERE r.assignment_id=p_assignment_id AND r.kind='execute' AND l.provider=p_provider
     AND l.call_identity IS NOT NULL ORDER BY l.call_identity,l.entry_id DESC) heads
$oldcost621_0$,$newcost621_0$   WHERE r.assignment_id=p_assignment_id AND r.kind='execute' AND l.provider=p_provider
     AND NOT public.lab_arena__retired_source621_v1(r)
     AND l.call_identity IS NOT NULL ORDER BY l.call_identity,l.entry_id DESC) heads
$newcost621_0$),
('public.lab_arena__submission_stage_consumed(text,smallint,text,text)',1,$oldcost621_1$    JOIN public.lab_arena_runs AS runs ON runs.run_id = requested.run_id
    WHERE runs.kind = p_kind
  )
$oldcost621_1$,$newcost621_1$    JOIN public.lab_arena_runs AS runs ON runs.run_id = requested.run_id
    WHERE runs.kind = p_kind AND NOT public.lab_arena__retired_source621_v1(runs)
  )
$newcost621_1$),
('public.lab_arena__successful_icp_cost_state(text,text,integer)',1,$oldcost621_2$  WITH matching_runs AS MATERIALIZED (
    SELECT runs.run_id, public.lab_arena__execute_infrastructure_cost_exempt_v1(runs) AS infrastructure_exempt
    FROM (
$oldcost621_2$,$newcost621_2$  WITH matching_runs AS MATERIALIZED (
    SELECT runs.run_id, public.lab_arena__execute_infrastructure_cost_exempt_v1(runs) AS infrastructure_exempt,
      public.lab_arena__retired_source621_v1(runs) AS source_retired
    FROM (
$newcost621_2$),
('public.lab_arena__successful_icp_cost_state(text,text,integer)',2,$oldcost621_3$      ledger.entry_kind, ledger.amount_microusd, COALESCE(runs.infrastructure_exempt,FALSE) AS infrastructure_exempt,
      CASE
$oldcost621_3$,$newcost621_3$      ledger.entry_kind, ledger.amount_microusd, COALESCE(runs.infrastructure_exempt,FALSE) AS infrastructure_exempt,
      runs.source_retired,
      CASE
$newcost621_3$),
('public.lab_arena__successful_icp_cost_state(text,text,integer)',3,$oldcost621_4$  ) || CASE WHEN EXISTS (SELECT 1 FROM public.lab_arena_rounds r
    WHERE r.round_id=p_round_id AND r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1')
  THEN jsonb_build_object(
$oldcost621_4$,$newcost621_4$  ) || CASE WHEN EXISTS (SELECT 1 FROM public.lab_arena_rounds r
    WHERE r.round_id=p_round_id AND r.configuration_doc->>'infrastructure_retry_cost_policy'='trusted_infrastructure_v1') OR bool_or(source_retired)
  THEN jsonb_build_object(
$newcost621_4$),
('public.lab_arena__successful_icp_cost_state(text,text,integer)',4,$oldcost621_5$    'competition_settled_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND NOT infrastructure_exempt),0)::BIGINT,
    'competition_successful_microusd', COALESCE(SUM(amount_microusd)
$oldcost621_5$,$newcost621_5$    'competition_settled_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND NOT infrastructure_exempt AND NOT source_retired),0)::BIGINT,
    'competition_successful_microusd', COALESCE(SUM(amount_microusd)
$newcost621_5$),
('public.lab_arena__successful_icp_cost_state(text,text,integer)',5,$oldcost621_6$    'competition_successful_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND call_succeeded IS TRUE AND NOT infrastructure_exempt),0)::BIGINT,
    'excluded_infrastructure_settled_microusd', COALESCE(SUM(amount_microusd)
$oldcost621_6$,$newcost621_6$    'competition_successful_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind='settlement' AND call_succeeded IS TRUE AND NOT infrastructure_exempt AND NOT source_retired),0)::BIGINT,
    'excluded_infrastructure_settled_microusd', COALESCE(SUM(amount_microusd)
$newcost621_6$),
('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',1,$oldcost621_7$        -- submission-wide, but only authoritative settlements consume it.
        v_spent := (
          public.lab_arena__successful_call_cost_state(
            v_run.submission_id, 'score', NULL
          ) ->> 'settled_microusd'
        )::BIGINT;
        IF (v_dynamic OR p_amount_microusd > 0)
$oldcost621_7$,$newcost621_7$        -- submission-wide, but only authoritative settlements consume it.
        WITH eligible_score_runs AS MATERIALIZED (
          SELECT score_source.run_id
          FROM (SELECT DISTINCT ledger.run_id FROM public.lab_arena_ledger ledger
            WHERE ledger.submission_id=v_run.submission_id AND ledger.call_identity IS NOT NULL) scoped
          JOIN public.lab_arena_runs score_source ON score_source.run_id=scoped.run_id
          WHERE score_source.kind='score' AND NOT public.lab_arena__retired_source621_v1(score_source)
        )
        SELECT COALESCE(SUM(head.amount_microusd),0)::BIGINT INTO v_spent
        FROM (
          SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind,ledger.amount_microusd
          FROM public.lab_arena_ledger ledger
          JOIN eligible_score_runs source ON source.run_id=ledger.run_id
          WHERE ledger.submission_id=v_run.submission_id AND ledger.call_identity IS NOT NULL
          ORDER BY ledger.call_identity,ledger.entry_id DESC
        ) head WHERE head.entry_kind='settlement';
        IF (v_dynamic OR p_amount_microusd > 0)
$newcost621_7$)
,
('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',2,$source_quota_old621$    AND (public.lab_arena_execution_attempt_count(v_run)<v_run.attempt
$source_quota_old621$,$source_quota_new621$    AND (public.lab_arena_execution_attempt_count(v_run)<v_run.attempt
      OR public.lab_arena__source_correction621_v1(v_run)
$source_quota_new621$)
,
('public.lab_arena__execute_assignment_calls_v1(text,text)',2,$assignment_perf_old621$CREATE OR REPLACE FUNCTION public.lab_arena__execute_assignment_calls_v1(p_assignment_id text, p_provider text)
 RETURNS bigint
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
 SELECT count(*) FROM (SELECT DISTINCT ON(l.call_identity) l.entry_kind
   FROM public.lab_arena_runs r JOIN public.lab_arena_ledger l ON l.run_id=r.run_id
   WHERE r.assignment_id=p_assignment_id AND r.kind='execute' AND l.provider=p_provider
     AND NOT public.lab_arena__retired_source621_v1(r)
     AND l.call_identity IS NOT NULL ORDER BY l.call_identity,l.entry_id DESC) heads
 WHERE entry_kind IN ('reservation','dispatch','settlement','uncertain');
$function$
$assignment_perf_old621$,$assignment_perf_new621$CREATE OR REPLACE FUNCTION public.lab_arena__execute_assignment_calls_v1(p_assignment_id text, p_provider text)
 RETURNS bigint
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
 WITH eligible_runs AS MATERIALIZED (
  SELECT r.run_id FROM public.lab_arena_runs r
  WHERE r.assignment_id=p_assignment_id AND r.kind='execute'
    AND NOT public.lab_arena__retired_source621_v1(r)
 )
 SELECT count(*) FROM (SELECT DISTINCT ON(l.call_identity) l.entry_kind
   FROM eligible_runs r JOIN public.lab_arena_ledger l ON l.run_id=r.run_id
   WHERE l.provider=p_provider AND l.call_identity IS NOT NULL
   ORDER BY l.call_identity,l.entry_id DESC) heads
 WHERE entry_kind IN ('reservation','dispatch','settlement','uncertain');
$function$
$assignment_perf_new621$)
 ) edits(signature,ordinal,old_text,new_text)
 WHERE edits.signature=expected.signature ORDER BY edits.ordinal LOOP
   IF length(d)-length(replace(d,edit.old_text,''))<>length(edit.old_text) THEN
    RAISE EXCEPTION 'Source correction621 cost anchor differs: %',expected.signature; END IF;
   d:=replace(d,edit.old_text,edit.new_text);
  END LOOP;
  IF encode(extensions.digest(d,'sha256'),'hex')<>expected.postimage THEN
   RAISE EXCEPTION 'Source correction621 cost source differs: %',expected.signature; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected.postimage THEN
   RAISE EXCEPTION 'Source correction621 cost readback differs: %',expected.signature; END IF;
 END LOOP;
END;
$cost621$;

DO $funding621$
DECLARE target OID:='public.lab_arena_champion_funding_immutable_v1()'::regprocedure;
 d TEXT; metadata JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(proowner),proacl::TEXT,prosecdef,provolatile,proconfig)
  FROM pg_proc WHERE oid=target) IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',FALSE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
  RAISE EXCEPTION 'Source correction621 funding security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='56f2036dc4d88a46d9fb1257dfde6a4e291947b88d7cb56434ee545d2f237a46' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7f129885317b9d8920a3db55a1f2f29f2b4a61af817eb9c16624dae71a9e308d'
  OR length(d)-length(replace(d,E'BEGIN\n',''))<>length(E'BEGIN\n') THEN
  RAISE EXCEPTION 'Source correction621 funding preimage differs'; END IF;
 EXECUTE replace(d,E'BEGIN\n',E'BEGIN\n'||$funding_add621$  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_rounds' THEN
  IF OLD.round_id='arena-2026-11-04'
   AND OLD.champion_funding_frozen AND NEW.champion_funding_frozen
   AND OLD.champion_fallback_providers=ARRAY[]::TEXT[]
   AND NEW.champion_fallback_providers=OLD.champion_fallback_providers
   AND NEW.champion_submission_id='sub-28594d902080fb5213f4344c09ffb531'
   AND NEW.champion_hotkey='5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq'
   AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
     WHERE h.round_id=OLD.round_id AND h.request_id='nov04-baseline-source621-v1'
      AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
      AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND h.old_round_doc=to_jsonb(OLD)-'updated_at'
      AND (h.new_round_doc-ARRAY['arena_network_name','arena_netuid'])=
       (to_jsonb(NEW)-ARRAY['updated_at','arena_network_name','arena_netuid'])) THEN RETURN NEW; END IF;
  END IF;
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_rounds' THEN
   IF NOT OLD.champion_funding_frozen AND NEW.champion_funding_frozen AND EXISTS(
      SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_rounds correction ON correction.round_id=h.round_id
      WHERE h.old_round_doc->>'status'='published'
       AND correction.status NOT IN('published','cancelled')
       AND correction.arena_network_name=OLD.arena_network_name
       AND correction.arena_netuid=OLD.arena_netuid
       AND correction.evaluation_date<OLD.evaluation_date) THEN
     RAISE EXCEPTION 'baseline_source_correction_pending' USING ERRCODE='42501';
   END IF;
  END IF;
$funding_add621$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'56f2036dc4d88a46d9fb1257dfde6a4e291947b88d7cb56434ee545d2f237a46' THEN
  RAISE EXCEPTION 'Source correction621 funding readback differs'; END IF;
END;
$funding621$;

-- Export only the narrow source receipt needed to verify this baseline's
-- immutable replacement archive. No provider/private artifact payload is read.
DO $history621$
DECLARE target OID:='public.lab_arena_scoring_policy_history_v1(text)'::regprocedure;
 d TEXT; metadata JSONB;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_object('owner',pg_get_userbyid(proc.proowner),
    'acl',proc.proacl::TEXT,'prosecdef',proc.prosecdef,'provolatile',proc.provolatile,
    'proconfig',proc.proconfig,'proisstrict',proc.proisstrict,'proleakproof',proc.proleakproof,
    'proparallel',proc.proparallel,'prokind',proc.prokind,'language',lang.lanname)
    FROM pg_proc proc JOIN pg_language lang ON lang.oid=proc.prolang WHERE proc.oid=target)
   IS DISTINCT FROM $history_security621${"owner":"lab_arena_owner","acl":"{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}","prosecdef":true,"provolatile":"s","proconfig":["search_path=pg_catalog, public"],"proisstrict":false,"proleakproof":false,"proparallel":"u","prokind":"f","language":"sql"}$history_security621$::JSONB THEN
   RAISE EXCEPTION 'Source correction621 source history security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='873cdf9c0e5ad4a6427617fa7377a64af17a83d631e24e7ca9ac81c476eeadf3' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c36bbea6177d420460a14f62512351196a34cffe2eb14415751579cbce7fa0ec' THEN
  RAISE EXCEPTION 'Source correction621 source history preimage differs'; END IF;
 d:=replace(d,$history_old621$ SELECT jsonb_build_object('round_id',p_round_id,'revisions',$history_old621$,$history_new621$ SELECT jsonb_build_object('round_id',p_round_id,'source_corrections',coalesce((
  SELECT jsonb_agg(jsonb_build_object(
   'operation','baseline_source_supersession_v1','request_id',source_history.request_id,
   'request_hash',source_history.request_hash,'round_id',source_history.round_id,
   'submission_id',baseline.submission_id,'miner_hotkey',baseline.miner_hotkey,
   'source_ref',baseline.source_ref,'source_size_bytes',baseline.source_size_bytes,
   'source_archive_sha256',baseline.source_archive_sha256,'source_normalized_sha256',baseline.source_normalized_sha256,
   'baseline_source_url',r.configuration_doc->>'baseline_source_url',
   'reasoning_baseline_source_sha256',r.configuration_doc->>'reasoning_baseline_source_sha256',
   'run_ids',coalesce((SELECT jsonb_agg(a.run_id ORDER BY a.run_id)
     FROM jsonb_array_elements(source_history.recovery_doc->'work_items') item
     JOIN public.lab_arena_runs a ON a.round_id=source_history.round_id
       AND a.submission_id=baseline.submission_id AND a.icp_position=(item->>'icp_position')::INTEGER
     WHERE a.kind='execute' AND public.lab_arena__source_correction621_v1(a)),'[]'::JSONB))
    ORDER BY source_history.generation_floor)
  FROM public.lab_arena_scoring_supersessions source_history
  JOIN public.lab_arena_rounds r ON r.round_id=source_history.round_id
  JOIN public.lab_arena_submissions baseline ON baseline.round_id=r.round_id
    AND baseline.submission_id='baseline-2026-11-04' AND baseline.is_king AND baseline.status='frozen'
  WHERE source_history.round_id=p_round_id AND source_history.round_id='arena-2026-11-04'
   AND source_history.request_id='nov04-baseline-source621-v1'
   AND source_history.recovery_doc->>'operation'='baseline_source_supersession_v1'
   AND source_history.new_round_doc->'configuration_doc'=r.configuration_doc
   AND source_history.new_round_doc->'participants'=r.participants
   AND source_history.new_round_doc->>'champion_hotkey'=r.champion_hotkey
   AND source_history.new_round_doc->>'champion_submission_id'=r.champion_submission_id
   AND source_history.recovery_doc->'new_submission_doc'=to_jsonb(baseline)-'updated_at'
 ),'[]'::JSONB),'revisions',$history_new621$);
 d:=replace(d,$history_floor_old621$'nov03_captify_checkpoint_continuation614_v1') THEN$history_floor_old621$,$history_floor_new621$'nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1') THEN$history_floor_new621$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'873cdf9c0e5ad4a6427617fa7377a64af17a83d631e24e7ca9ac81c476eeadf3' THEN
  RAISE EXCEPTION 'Source correction621 source history source differs'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'873cdf9c0e5ad4a6427617fa7377a64af17a83d631e24e7ca9ac81c476eeadf3' THEN
  RAISE EXCEPTION 'Source correction621 source history readback differs'; END IF;
END;
$history621$;

-- Bind the correction to an exact terminal snapshot. These are hashes only;
-- provider payloads and private ICP contents remain in their original storage.
DO $correct621$
DECLARE rid CONSTANT TEXT:='arena-2026-11-04'; request CONSTANT TEXT:='nov04-baseline-source621-v1';
 manifest CONSTANT JSONB:=$manifest621${"accepted_run_hashes":{"arena-2026-11-04:baseline-2026-11-04:1:0:2":"e788ac92ae8baea7db81a38b7ce3699e4b87335fdf8e6ef994672878ce0d3cee","arena-2026-11-04:baseline-2026-11-04:1:0:score:1":"dac7986e203c44ec73598cde933183ca24c61dc95a0414a57b8a077f6aa526e7","arena-2026-11-04:baseline-2026-11-04:1:1:2":"86e20ffe1c1af5ff1430336681f5a05c8ade941ff6374ee2c54383a6faa0c04f","arena-2026-11-04:baseline-2026-11-04:1:1:score:1":"3995cabeab4b46c9cb3d87ac7440030b34564d28abe87b07dec7a2eae7b771d6","arena-2026-11-04:baseline-2026-11-04:1:2:2":"03aac325486254509e5bac264beec6e13935a8cda0f24c1db4ecef5a98ebc441","arena-2026-11-04:baseline-2026-11-04:1:2:score:1":"ce1ea940dbd3da3a69fdc5032e7685f27cefd3d942c0b09dd512bd06f0d5126c","arena-2026-11-04:baseline-2026-11-04:1:3:2":"aba087109f76b28593a9022d4ed8e6f518320bb30bb6f6fe5d56ac18a89fbe8a","arena-2026-11-04:baseline-2026-11-04:1:3:score:1":"614584b531c58f2429cb5f7da7768366c15be556f7806bfd3b3bef89fc8ddd6c","arena-2026-11-04:baseline-2026-11-04:1:4:2":"9f93ce3c277506f98b5e4bedc32e47c8f6b47444e0a5d01a5c0051d9c67d62b3","arena-2026-11-04:baseline-2026-11-04:1:4:score:1":"2a010566dbbb013dcd3a015df577bb12e4cad4ff1cab92f66b304e24deaad449","arena-2026-11-04:baseline-2026-11-04:1:5:2":"afc706575eca7d3bcf4bb09192485705172c4d8bf3da199de16a8dfe52e97d6f","arena-2026-11-04:baseline-2026-11-04:1:5:score:1":"69371065c04d77d7c0f19abecb6336f97d7e18af4600f81071f7718e98f7d585","arena-2026-11-04:baseline-2026-11-04:1:6:2":"daa57bc981ae16345da4b9db77c01374167bc6406ede050fdbfb6a1984cfbc1b","arena-2026-11-04:baseline-2026-11-04:1:6:score:1":"a8ce6baebdcaab8b378ffc4b9b187bd41c6635f877d19fa5b02c8f87c6f721c8","arena-2026-11-04:baseline-2026-11-04:1:7:2":"1549e6234f991d15180372f8a7577445e97663df24106250bc07ca717006e487","arena-2026-11-04:baseline-2026-11-04:1:7:score:2":"e4b8bcf075dbc716abbb6aaadeb7fd4633a3f757fbbaa0218331c92940ccb37d","arena-2026-11-04:baseline-2026-11-04:1:8:2":"baeb336ef7067cbbdf95616927f959bddd65d178ce66ff61a912c8811f64e297","arena-2026-11-04:baseline-2026-11-04:1:8:score:1":"870cdbefce23a668ced9992f7b1db450874aff475cc84063c19baae08637fed5","arena-2026-11-04:baseline-2026-11-04:1:9:2":"e1883c55b1bd6abecef8979eac7d3cf3c2357e81ef9a6ea4041d9a363cc4fbc6","arena-2026-11-04:baseline-2026-11-04:1:9:score:1":"02e39837b49c4845506e62286d9575c84e85356e1f9e8b93fe445ee5da5d79f5","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:1":"e3f7307ac627617ec3b497e993155c71ebfe355258e32c44d0da3a8253f750c8","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:score:1":"38cce388f8ceeb9412e8f484a544107100aff7dc9e38d79cb12f5bbef56612b1","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:1":"5bb1867cf6060bebfcb7757b64736e065e113c1109ccc367b3e71f5dc8805ed7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:score:1":"a140bf594d67ac733822b7485b32ed5404dd0d26d8155e75683d082bdacaa8ba","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:1":"9ea18228fbd25278f994847f61bed6b15c42abc492d478831d2c842b3c6da04c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:score:1":"b5fe0dfc7c5f2bbc92b6860f1659f846cd095371eddb1a74fb3d0943ebdca404","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:1":"c5656aa9d8dbd38b38f822e67c236570791a8606aad06bc752ce43884884ca9a","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:score:2":"4ee9d856dae489b2f6f3833b375f98764bd5cadf4085a63df512ff7f5e419f74","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:1":"c69c06aee22fc00682993308cade3df289b6c86ae369d38cda02615d3f1d40e7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:score:1":"d60234b45e026d6131bf0baf346bc0e235c8a39c424058040f256312d19f16c6","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:1":"5253242358e50d9330eab20994f03a37c57180a9201b9591e8b95977d349092f","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:score:1":"ace362c248e2a0ce75445d035c9f5d9a1a9d7e0ce6ed38e0d4ff0b0e45ed9c31","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:1":"6551d8302eb529ab07de43e403b1ac9aed1b57bf5b37830626d86a9db50a0794","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:score:2":"8a01f3460bb10a059ccaf41760737e12b75f2d34fef3aec1fa3db79cd51c4666","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:1":"5bb3cb1ea5084d33523ad666e11533954eeae2c79886b47d142bdfa6f88f09b5","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:score:1":"7f0a1d6124a62e0ab3690d90e0ffdbbfc93be25b47bc633dab32d362fd801314","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:1":"90d81b88bc22d2ae22233a4e3419590c10bda2642a81d500373c4214e6db8f09","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:score:1":"33c8b45a0fd148a04c3d79c885bb091bd4f59a4317d5f09f5dd98c6183171f0d","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:1":"3b577e1404173fffc38128604b061704543cce8522a36f4393a80b98b40c402c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:score:1":"366f2fd49f3b74b7b02845c3b586d3e96be1a3b98a4b26cb1497ec4f1ae86dfa","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:2":"6dd3ff2b506b49a1414fe5c79b40c9a6159931a23dc955ca7b2cadaca3f9c4c0","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:score:2":"a9abb8f08cbbf93fd514452f51ea42454690222a40b7a407f1d7e03026e53034","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:2":"6ad4309139072aae35ced24ccca74e946e471643460427a85f9ab28582b34d2c","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:score:1":"de02850716ad4960218176053c65b106a3b57443d34c02a80d99feca93f7aa96","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:2":"7c140e38f280ed7da5ae0d4600de60dfcb6bd683b2e4c14d8c5c4eb8ba695869","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:score:1":"855106a7bfe6b4f8e8804db03b0634974ecc797e972cbcf8dc207405ed7ad672","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:2":"a835d4dd2c43409f9da1d17c6755c074490afd439f3430fe11a812b98f1b68d7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:score:2":"8ea0ce827168eda0f1db34ff8b7abbdc935943efbb0c6310e81737f2acdbc48d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:2":"469a394d6b38fa42082cecb60a9b5527a111301846d938efa8f33184b55b64e8","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:score:1":"b2e8138959c60e5a8a03e29b148b81145c08066fc7be601e498e60b95bfecdd2","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:2":"640c30a242ffb245f89a2baf79d7842e3dcd7bd5b9669b0827d0c35a266bd73a","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:score:1":"46c9f883045452af0f41adadd6f833d5e8b9aa5ca8f5dc34b7f25bb327d9863b","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:2":"174704e6c1b8c26c611f86b71a7876eaf1e823f494b8b164c2840099e0a54b79","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:score:1":"59fc3a2306780a10a305e5e3bb839592bbeb9c62d657b821d5e0c77d70d84f72","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:2":"fcb4935ab650f1aeea0df994adaa821f07762867907b61c91c2207330a6c8f69","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:score:1":"4f74e289511944f42a9aa61ca169d4ec127725849f1e41a49af96a9c43891743","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:2":"1af31d6a8bab0e7d5d7e343494726a799c036002dc5fb9b25cc6b746fd21dfd6","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:score:1":"9a8efb80280f1bd4458bf44bafbeab24c40dea391200553e0b3cf6de7b978ee7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:2":"76dfc70854839a94d476feb5d1820ce091dc259be69f367100cb4f450589897d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:score:1":"0fa65c62b262158baf755506e179c0d7bef614a796ccd183a770e6d2d04165a1","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:2":"69bf7032de2e8e199e5fc14f559b3bdfd6fc29303c3130e69f667a5bd07be0e5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:score:1":"82255747ccfbb2cd2dfaaaa72bd8b31193758c74601a20ccaf4c2e25e9680f2a","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:2":"5fcce7e1f302aa1a3743723d222947f73f312fdc3269bee47d71a5dd92382f94","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:score:1":"3b1cbab642a8a2461c5062ba8651c2da4829938ea7e06ee2c4f0a0603a1d5c0c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:2":"dc6f551132330023a00ac99d2688f0ba5916b8d4541b1273403bcae5669bc870","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:score:1":"5358291ddabd288c6fa1b253f22e39980bb0633d29cc644c8f20f6a334f2bfd0","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:2":"d50b4f4896d5ed8a38f246f608d29ded2a38946fb4de88d3e94f279e6d79f70c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:score:1":"e261bb8edac19bc7aac95d9e16c9a711f97bb11cb5083ce6c0895caf0aae28ee","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:2":"99a9a4c1de9b28398d9fde8d149e555052eb33eb0506f7a656a85509e2f6b64f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:score:2":"34bce7bc1ceffd5635459447130aeb45f09391ee8157890c5c3a0a90a94793ce","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:2":"57ac434730560fa91ffda6c1cd46ac6fd3b906a0c6b1dc8ebf8637449a48f9f5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:score:2":"3bb0987fec4adbcc9e6c6104dff21f272919a48f256652980c5a2c6f4bf69155","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:2":"ca2ad62ddaf9df41132a7e2dc0f1f268b307175653822290ef7724f2e67e718f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:score:1":"117860fd998d65710a2db0eb4b2252bf666de71e66d7334fea03e9d7d14057ae","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2":"da81635a5e2686798266f88eab2204ed03f55cf396a0d3b4aab7664eea701715","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:1":"dfa91b91177c4b84fc695bb49c9474be5fb35e7587499454161d129ca985fc93","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:2":"f18b0122e7624f7d44b696a7e81a7bfffafed65a3ad5373f6b36ff969d32b43d","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:score:1":"49e6d18cbeedfd86c356dee7c3c3dcbc92fb9b55249ce6b8e5dab5f294c1f990","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:1":"980d65eef8ab8d393219dd63200a6bb4a5c874af3e8cf57c8233c7a125cd2722","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:score:1":"5444c69ca1572d0bf95c0931c751491bb1da10367bb9f4ed6a2b10d2e3c6092f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:1":"3a5018474c4fd33d31049c26234c930575fff08cadf85ee7798926c9dbd01b5f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:score:1":"2609ffb668e32b68c350fd105109edec1e6659a6d3642082d15f4fa788bc708a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:1":"296198550a6ed1f52eb70f73beab002034e9eacbbac304ad89ed3d6515cf99bc","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:score:1":"2d4c57132379ec0a22e7ee4cae21be92955075130ec9c58bee7cd057733cc1eb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:1":"99eaff1d6b28e09a1f6cfc66078462745c526152a51260bf6cfeb0fd4aca2dba","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:score:1":"e9ffbfc6daa90dfde946a88e4c2c8c96904e7d9eda8dfd8b1dc1083870bf1168","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:1":"5df0685c0d82fbc6a7fe4813d582aa99f9746b9b4f4437d0648da80e005c72bd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:score:1":"568ce58f0b7c9592933223b1f16f7fab49ed60610f5fdcb3a43f87e46c9c9788","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:1":"9d9fd650744103fd520ecf90c46cc523271fb1f55d2a0dc02c86e4f06efc8bbd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:score:2":"7d687835c0b59a63e30190534a838f6dde00438261bed3febac0a008c1cc384f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:1":"56089dd4c7bc7a329f25e1be25241a807ee5aacf32ebade06366a081a5e7020a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:score:1":"89edf98f346ec544be24cfd3666321a522676c25d34e4e12cf2afb6ed72fe416","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:1":"ed6e3cf39eb802623c73e1e01bb78ad884891b7442fef291afaab0d2e0e6a3f8","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:score:1":"4ef43c5ce5392e391518d95aaabde5dc7cfc4f6fc7c642a46da87771d0146fa0","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:1":"886178fdabfaa86ab92e13c3c11c198154faa6c29c49a2087aea58b348869054","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:score:1":"393f54188c21d110a261e109d93cbbb3716b67fee086fcdb506f9c76fd8c6539","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:1":"12cbdb2bba098758d0ba93d5e6db4b21adf02e40fd0380547c4e00fad0870044","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:score:1":"e9f8d4e541002ff5852b5aac668573be6114acfb8d6b978b810d70af82e5e979","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:1":"95338948e5888b0f478b02b1380a5bf983097b11b2f50c230082d5b99749bcdb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:score:1":"4e2bb5f170ef509c059191775b3a9575c831753928f2fa2d861fa55d1485a813","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:2":"beb191f67ad696e2c2330bd68830b3913ec10f35039374bb897d5f1de8919994","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:score:1":"9c99cc9fb39df47879b21ba0222ea51a6f243b694ccc5afa523145f941822e5d","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:2":"677a26b88e9f9471271b34829a2028061fd11ab62b2db8bb41c23a26240f62f3","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:score:1":"163d2343d37864df65e887045d0cd02fe0f1c92e54e2f7b52fa0a5562cda8dac","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:2":"e36d665b2d08e0fa6bd3c5071074d8419d22a50ac4e43a7e20ad02425a451cc1","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:score:1":"7acd6b0b9adc7c1418f23276e9894689db5e25b9a90aa588fd159b721ed4cc42","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:2":"ccc2c59f9fa901edac4dc38adfcc62ce20d29ed8fb8ac2d57d0212aadf8693ce","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:score:1":"b98f2088bdc55e1a70e3a8862afa0af9b42f2cf3aaebfd0bf6daaae5dc502253","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:2":"79362393258fe32c9106fbd66d46a588da1208a4384056d14c454b3a8eed8c03","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:score:2":"d6a960a74e8982bd413231bf8bc3cb82eb975de4b4d75212a20af76e6486fd57","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:2":"f12fa288d6967ce874d47d958f209dd6defc525a51f555f0e7a97077770e0237","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:score:1":"677441e316777cb5e15a0b47b86aebad5c97f366665767f3da08be00e0d5ef27","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:2":"e78453716fde1b5f6e0169f011c449c2d03474c267ea110c2b7aff3d904fe620","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:score:1":"50c1cc1c0aa740086c020e7361a4e5998aea963bfca06e02a7819e9403efd183","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:2":"8939bcd6b6b0dff86d52b8b46aafb14859a88ad4320bb1b7433147013e7b9abc","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:score:1":"fcc675b31f98c2b4f3480b333840437d81221b5a328d7da515106f6039963f3e","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:2":"b72b9f165d00b45379d1595a57f3e7b55f9b9492a368fada154b11029d77bea7","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:score:1":"2707c606c292b38fabc309e3bd5251901071f9a95ec237d85fbb46965de36222","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:2":"531cebe1575831843537a108faea005391e40f0aab765824de65817a37e2987f","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:score:1":"ddd5b23d069830e44da0779bc2bb42ce1fe5598bdc2101344138c330613bdde6","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:1":"0bcb21e59a611550278ebd50218d91439bbe28416ed5c5494c0a0aa78b7cb5f7","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:score:1":"5f6c746905fc5cc4efa0351012138abbceea122f67933493bf13b5792493bad0","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:2":"8fa8c721b68e07e3692e90db871b2090f2a961c3796b2e84b45b0d5d594d7729","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:score:2":"1d2800618e653494e5a2938a7963774e93ec106ae73a02391545161fddb20239","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:1":"8c6530bf91ba2c8e74d4587f4bb92d3b12170ed76b9c9d1449c393b1c7ef54d2","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:score:1":"14be0e4ed29d0f779caeddf5305d3e2e02435371a32a71421ecbb5b43103316a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:1":"f4f77e47bf2cfc3d6da907ff6dd13805ce3a54682df4181ac6e71fdbd7f296be","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:score:2":"c7b7870f73d199967bb78796f2ae01450e3d88cc03644f773010947a6282512b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:2":"1d232b7ae5dafab4534d72943860df6a115d918660d6f46cdd29ee894d02687a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:score:1":"451a0bdb6bc2bdabff91a058b48cc3c6f921e14210bce4bb45062ba5d7a9f47b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:2":"7aaa48d5ad93c6d12fbf83d8c2464338e5e93ed122ba7a17840d8132cfabb914","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:score:1":"b81a5e47e2d6d75aa331a8d086b492b28141c0e54fdbd4f50b3ed99a22b19194","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:2":"d472f1fbbd899f37960d71bee5f634780294f103605be567f9114af236318d97","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:score:1":"d7e83c751477487d151ef8de56815e95d8eb86b520ce8289999f9a9caaab8c0f","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:2":"55d8bad8c69a856bf397d07f560c3250a0ac713ffe3ecc4198f20df74d93a2c8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:score:1":"f9cf79ffcb3afbe56395f0dd28f53c9489413ccf222172cc434102734772b1f8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:2":"ca1c960c80be55dbcb8688fbf76f4ede2c6e2f621fb1fcb8299bbbd5a1571452","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:score:1":"ddc63bc8d4171c03245aca0cf68a0e114319e1923a99dd49e4842864c5cda18c","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:2":"6a95000d52da3f4a83bd9b61a06816309f4b4ffc142c95aefdb79211133b3a2a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:score:1":"c679ac34f92c6aa71f7982672b1ff55d68bd1fea6c9ae20e968d76712c0ef263","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:1":"72345062a5eda8f1002c031a89c80a6d624ffc92571169746a7f4baa64613eb6","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:score:1":"e77bcef5d6375101d92fa50e2ea49d3c13ca867c65b7ba0b1e477b72496145a5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:1":"d80fb6858d35acef4374a9c2c9b50bb6371a3988157009b6d807890de1366a68","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:score:1":"e8c45b0a7463accf37496766eb463004db3b19d90d31e829a1d47095ab4f8573","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:1":"32ebd3a25489b00bae2c2e295d92c9c62f9e6aa4809649b918ba72d42b89bef9","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:score:1":"c4524fe788a6a0b550a93bca1369428a30b42df98ae2d5b35cd15ee85763c160","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:1":"2d6440d80a55a0e824e3808e644b84f1da6fc6b56b6d8713f02d6a943887fa99","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:score:1":"af6d0b2a298b381551e974b07fcb540f686dbc9a49a57446bdd15fe3b2290b44","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:1":"72092fd0c50ec25804fb6653590282186fa828df1806917fb37363cfdaa0882e","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:score:1":"85893364a0c82cb891ff78ffd216dce0165ea55196b3bfdf8df403afab6002f4","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:2":"6457a984cb70552aae9980873f12e9f9894396e09e97e98924ea64d16ec18431","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:score:2":"1aac0a8df06b7111dccad01e17ea7df457d5e7b297978e31b3be057d28f4f408","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:1":"725554ad6e1c330fd08e735935850d729b9f74bf0a4699c5568b132e29fe95de","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:score:1":"05240863dc482143f58e5f1958fc4531d62c53e8f3292107830058acd86a42a7","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:1":"1cbdbbfa8f820c4ed76831a5012db8fb33b5b07e89fc2eaf09b89b37a88b09d5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:score:1":"cf20c855960208fa23ab18aad393699ee3c7d3e10a7e0dc7beb5f1a103421571","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:1":"8a9786ccee46e404633392c39c7aec178a4878b11a67fb6ff35660aa43e35b77","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:score:1":"e8558098e2c5a4facc8283a0a8afa97a636fbe499f510817cdc469b506e58231","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:1":"0a6d9577cbc30ad2be3135cfb55b81bad358919142f76fa4aedafd1a8000d30d","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:score:1":"d30848602d9beb885566befacc038b63adc6a4480ab8ab877139674d01c9f8ee","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:1":"ebbb7072c49a9235e6905ecab0572b4f956f559b21e612409c7bf968a6358a7e","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:score:1":"7f48f29b8d2bc312565fc3147ef1b6fbd3b4d2ba936d361a5ce632fc68cc405f","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:1":"6f2b46eb2a81dcb09d4bbba2b52c76cb85db3aaa89cc56c730a1ec75b67350a0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:4":"e72dcccde665db701ace3f0f9ec16161d95ee58a667a352ddedff05125711bf7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:1":"e9f8906fbf02048a0ae08a9b67c2a00571829b05aff8fda3b7621325fbdfb236","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:score:2":"9f3ee6a0ee77d569551d720062d9f19f11608af8beb5bcde674c20cc0ac13832","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:1":"5c4a462ee6197fc29b0709794eb76d1f81becca4b56483d31206bb3c9cb065f3","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:score:2":"abcc1094d0416a4a691311a0b0fa67bb36399107ed9c2d8572bd29dd7d0ed046","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:1":"1f354b43d71a17932751c5606384a156510fe4a4a1c7857cd360e3db67e148c7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:score:2":"7ce90d0fb3634cc833a1902fca542fc31ec8d565560ff50f06799160d748a3c4","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:1":"5d95bfe5eee06871f87c4ae7fd1d07da102b9a87d57e2caca04d19ee361a3023","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:score:2":"117ba598e8e33f897bc441665920b2d6ff0455464f2707188882435203c50816","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:1":"d2c7002ddaac2fea5336428430e9729fa029d68bd12ae46aa5d8db14fca8b300","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:score:1":"20a150db1984208058d9479bc7d34e4fd7b0d9545ee8e17085cb9d831b52b6da","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:1":"42e49e8d933625a2a39da2f43928e41f2c6da252de4e0dbc1b87894893f380e1","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:score:1":"6e3df1e43453f3916bd5f9be10da8a959a3c3bc40ae229c48a58def416be0e91","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:1":"06d668b9bad8d0d36fd0836de7ae246dfa768ddd7688ec577fb751aa8b030d02","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:score:1":"6624ec3bbff2f7bbf7955a2a79e8bb25390986591fe73a6f3bff0339cf2778e0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:1":"7c925b5e63438385de95c576b19cf50c7fac52d170267a1de700482eaec86d71","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:score:1":"0aa3ade417f05a5f03c9450fd82ef4c4e74df72326d7e1ab0f8d6333fd64cb1b"},"minimum_guard_generation":764,"prior_round_sha256":"bb94085e427b791460649a57b6c728dcdd84ee1a8ba2079b68e82858a897e772","round_sha256":"37724480abaced7468c4cc427c070cce6542821258d12c8c9d99e755bbc69f59","run_hashes":{"arena-2026-11-04:baseline-2026-11-04:1:0:1":"75776b94263c12986d54bbd8f01b89711cef2ce0961ea19abfed3da58ca975a5","arena-2026-11-04:baseline-2026-11-04:1:0:2":"e788ac92ae8baea7db81a38b7ce3699e4b87335fdf8e6ef994672878ce0d3cee","arena-2026-11-04:baseline-2026-11-04:1:0:score:1":"dac7986e203c44ec73598cde933183ca24c61dc95a0414a57b8a077f6aa526e7","arena-2026-11-04:baseline-2026-11-04:1:1:1":"7f0882fbcdbf555d8486533a622b68baafecd51638e7e05c1e006320f71f48c8","arena-2026-11-04:baseline-2026-11-04:1:1:2":"86e20ffe1c1af5ff1430336681f5a05c8ade941ff6374ee2c54383a6faa0c04f","arena-2026-11-04:baseline-2026-11-04:1:1:score:1":"3995cabeab4b46c9cb3d87ac7440030b34564d28abe87b07dec7a2eae7b771d6","arena-2026-11-04:baseline-2026-11-04:1:2:1":"e4f7cb9918ba74b4913f89040c915c6b6b2cd48dd9efd57106603ea042d3a852","arena-2026-11-04:baseline-2026-11-04:1:2:2":"03aac325486254509e5bac264beec6e13935a8cda0f24c1db4ecef5a98ebc441","arena-2026-11-04:baseline-2026-11-04:1:2:score:1":"ce1ea940dbd3da3a69fdc5032e7685f27cefd3d942c0b09dd512bd06f0d5126c","arena-2026-11-04:baseline-2026-11-04:1:3:1":"9d6c596611418270f928edf521f6d1808f6e294678a3c837dc4ea8f64c4845b6","arena-2026-11-04:baseline-2026-11-04:1:3:2":"aba087109f76b28593a9022d4ed8e6f518320bb30bb6f6fe5d56ac18a89fbe8a","arena-2026-11-04:baseline-2026-11-04:1:3:score:1":"614584b531c58f2429cb5f7da7768366c15be556f7806bfd3b3bef89fc8ddd6c","arena-2026-11-04:baseline-2026-11-04:1:4:1":"5d20e1e707ff252a1f5bf1eae5ac5bfcc7b0b1c1b223b7b84e3dc142cd350f5d","arena-2026-11-04:baseline-2026-11-04:1:4:2":"9f93ce3c277506f98b5e4bedc32e47c8f6b47444e0a5d01a5c0051d9c67d62b3","arena-2026-11-04:baseline-2026-11-04:1:4:score:1":"2a010566dbbb013dcd3a015df577bb12e4cad4ff1cab92f66b304e24deaad449","arena-2026-11-04:baseline-2026-11-04:1:5:1":"b81a5394feb68bad974ba93c3418d93f5d3f24b35cede7e854723acccddd10e3","arena-2026-11-04:baseline-2026-11-04:1:5:2":"afc706575eca7d3bcf4bb09192485705172c4d8bf3da199de16a8dfe52e97d6f","arena-2026-11-04:baseline-2026-11-04:1:5:score:1":"69371065c04d77d7c0f19abecb6336f97d7e18af4600f81071f7718e98f7d585","arena-2026-11-04:baseline-2026-11-04:1:6:1":"349a7ddef19f2abcbb38046c2fb6c096f49d3c8b34bccb17df0fc705822c361e","arena-2026-11-04:baseline-2026-11-04:1:6:2":"daa57bc981ae16345da4b9db77c01374167bc6406ede050fdbfb6a1984cfbc1b","arena-2026-11-04:baseline-2026-11-04:1:6:score:1":"a8ce6baebdcaab8b378ffc4b9b187bd41c6635f877d19fa5b02c8f87c6f721c8","arena-2026-11-04:baseline-2026-11-04:1:7:1":"de623fa7976a760dc7b20aaadb1c2ebfb825484ade78c1f46c55efb1237b1430","arena-2026-11-04:baseline-2026-11-04:1:7:2":"1549e6234f991d15180372f8a7577445e97663df24106250bc07ca717006e487","arena-2026-11-04:baseline-2026-11-04:1:7:score:1":"e0d1d97040800fca678befdd41817e8b59f6f9c78d0fc5935dbbf3abbddf7c46","arena-2026-11-04:baseline-2026-11-04:1:7:score:2":"e4b8bcf075dbc716abbb6aaadeb7fd4633a3f757fbbaa0218331c92940ccb37d","arena-2026-11-04:baseline-2026-11-04:1:8:1":"9c4a0d76f891a48e9ced3245b99d9b6187b550606eb227fbde8e9057d68f4b85","arena-2026-11-04:baseline-2026-11-04:1:8:2":"baeb336ef7067cbbdf95616927f959bddd65d178ce66ff61a912c8811f64e297","arena-2026-11-04:baseline-2026-11-04:1:8:score:1":"870cdbefce23a668ced9992f7b1db450874aff475cc84063c19baae08637fed5","arena-2026-11-04:baseline-2026-11-04:1:9:1":"4131dcb08fc2c3653939357ffa19c6c2a2828f3a53c32c4d3a679549723a8519","arena-2026-11-04:baseline-2026-11-04:1:9:2":"e1883c55b1bd6abecef8979eac7d3cf3c2357e81ef9a6ea4041d9a363cc4fbc6","arena-2026-11-04:baseline-2026-11-04:1:9:score:1":"02e39837b49c4845506e62286d9575c84e85356e1f9e8b93fe445ee5da5d79f5","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:1":"e3f7307ac627617ec3b497e993155c71ebfe355258e32c44d0da3a8253f750c8","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:0:score:1":"38cce388f8ceeb9412e8f484a544107100aff7dc9e38d79cb12f5bbef56612b1","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:1":"5bb1867cf6060bebfcb7757b64736e065e113c1109ccc367b3e71f5dc8805ed7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:1:score:1":"a140bf594d67ac733822b7485b32ed5404dd0d26d8155e75683d082bdacaa8ba","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:1":"9ea18228fbd25278f994847f61bed6b15c42abc492d478831d2c842b3c6da04c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:2:score:1":"b5fe0dfc7c5f2bbc92b6860f1659f846cd095371eddb1a74fb3d0943ebdca404","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:1":"c5656aa9d8dbd38b38f822e67c236570791a8606aad06bc752ce43884884ca9a","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:score:1":"8dffb0b5dcadc77969a173e14664a951a15f5aa3213f6548de2087bce3e6244b","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:3:score:2":"4ee9d856dae489b2f6f3833b375f98764bd5cadf4085a63df512ff7f5e419f74","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:1":"c69c06aee22fc00682993308cade3df289b6c86ae369d38cda02615d3f1d40e7","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:4:score:1":"d60234b45e026d6131bf0baf346bc0e235c8a39c424058040f256312d19f16c6","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:1":"5253242358e50d9330eab20994f03a37c57180a9201b9591e8b95977d349092f","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:5:score:1":"ace362c248e2a0ce75445d035c9f5d9a1a9d7e0ce6ed38e0d4ff0b0e45ed9c31","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:1":"6551d8302eb529ab07de43e403b1ac9aed1b57bf5b37830626d86a9db50a0794","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:score:1":"df138c1fc792bf3c8808e3b79ce7629e9dc0dfefd9da5a390f659959e87b9720","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:6:score:2":"8a01f3460bb10a059ccaf41760737e12b75f2d34fef3aec1fa3db79cd51c4666","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:1":"5bb3cb1ea5084d33523ad666e11533954eeae2c79886b47d142bdfa6f88f09b5","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:7:score:1":"7f0a1d6124a62e0ab3690d90e0ffdbbfc93be25b47bc633dab32d362fd801314","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:1":"90d81b88bc22d2ae22233a4e3419590c10bda2642a81d500373c4214e6db8f09","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:8:score:1":"33c8b45a0fd148a04c3d79c885bb091bd4f59a4317d5f09f5dd98c6183171f0d","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:1":"3b577e1404173fffc38128604b061704543cce8522a36f4393a80b98b40c402c","arena-2026-11-04:sub-0510cb617eabd69f2f0aebeb69b37fe7:2:9:score:1":"366f2fd49f3b74b7b02845c3b586d3e96be1a3b98a4b26cb1497ec4f1ae86dfa","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:1":"f892260ae275336fe354d9beb7476cfdf0875a0b7e4a212e12e6f82852917cef","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:2":"6dd3ff2b506b49a1414fe5c79b40c9a6159931a23dc955ca7b2cadaca3f9c4c0","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:score:1":"d81418d8b3afce8709afbaa1dffb98c143da68b4a72922048ca5155d7992e546","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:0:score:2":"a9abb8f08cbbf93fd514452f51ea42454690222a40b7a407f1d7e03026e53034","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:1":"725f459fb0ce4c350c9365bced0f731f2c6094b1b30889bb26af7b7b5d9c2405","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:2":"6ad4309139072aae35ced24ccca74e946e471643460427a85f9ab28582b34d2c","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:1:score:1":"de02850716ad4960218176053c65b106a3b57443d34c02a80d99feca93f7aa96","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:1":"7bf6388c6e4f6d09c6061aba1a2801e79ecb6e914fd3fa3065c8d4be97e895b2","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:2":"7c140e38f280ed7da5ae0d4600de60dfcb6bd683b2e4c14d8c5c4eb8ba695869","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:2:score:1":"855106a7bfe6b4f8e8804db03b0634974ecc797e972cbcf8dc207405ed7ad672","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:1":"8b2fe1a6e2832082a27d01e53a03d1939f84ea508ccb605a3ae7ada6203b9bc7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:2":"a835d4dd2c43409f9da1d17c6755c074490afd439f3430fe11a812b98f1b68d7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:score:1":"b39ef8e17ca989769a2f9eaf80ed19ec096654de0399389fc3276df9801582a9","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:3:score:2":"8ea0ce827168eda0f1db34ff8b7abbdc935943efbb0c6310e81737f2acdbc48d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:1":"87b3d499f34732ecfb07f75e46d5dbc6ea8d434b8045461553660bdb9ead0dfa","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:2":"469a394d6b38fa42082cecb60a9b5527a111301846d938efa8f33184b55b64e8","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:4:score:1":"b2e8138959c60e5a8a03e29b148b81145c08066fc7be601e498e60b95bfecdd2","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:1":"8678c49b7e4edb1f5437d3e0e369a04c050eec7a85fbacd610659d9bd9377bed","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:2":"640c30a242ffb245f89a2baf79d7842e3dcd7bd5b9669b0827d0c35a266bd73a","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:5:score:1":"46c9f883045452af0f41adadd6f833d5e8b9aa5ca8f5dc34b7f25bb327d9863b","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:1":"957f6cce9c3a42d4b74185aa6101cff212e9430aa486fb02e9780764ac17e712","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:2":"174704e6c1b8c26c611f86b71a7876eaf1e823f494b8b164c2840099e0a54b79","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:6:score:1":"59fc3a2306780a10a305e5e3bb839592bbeb9c62d657b821d5e0c77d70d84f72","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:1":"b77f5b1e0e270e7ebb940676537a69a0a339fcc4a61896d24a90fb4945c90a65","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:2":"fcb4935ab650f1aeea0df994adaa821f07762867907b61c91c2207330a6c8f69","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:7:score:1":"4f74e289511944f42a9aa61ca169d4ec127725849f1e41a49af96a9c43891743","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:1":"a8b8cab9362876f4f664d455962de1b101bf5165479c76b75d3dbc9ed8304030","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:2":"1af31d6a8bab0e7d5d7e343494726a799c036002dc5fb9b25cc6b746fd21dfd6","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:8:score:1":"9a8efb80280f1bd4458bf44bafbeab24c40dea391200553e0b3cf6de7b978ee7","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:1":"3abf7ae9d7f6bd781091b0e26430278d13c84f707b0bc70c9e4b2eef3d8dce7b","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:2":"76dfc70854839a94d476feb5d1820ce091dc259be69f367100cb4f450589897d","arena-2026-11-04:sub-0f1ed4111f50a0cd4ab2086394afc457:2:9:score:1":"0fa65c62b262158baf755506e179c0d7bef614a796ccd183a770e6d2d04165a1","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:1":"0fc1bfe0b7def6af446a11dddd2ba76fd2c8117df7b222965ad6c86a243a43ac","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:2":"69bf7032de2e8e199e5fc14f559b3bdfd6fc29303c3130e69f667a5bd07be0e5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:0:score:1":"82255747ccfbb2cd2dfaaaa72bd8b31193758c74601a20ccaf4c2e25e9680f2a","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:1":"496fae5a7c4981c06e5dc62bf8adcc17f5bad7f0a29146cfe80ecab3b09bbd93","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:2":"5fcce7e1f302aa1a3743723d222947f73f312fdc3269bee47d71a5dd92382f94","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:1:score:1":"3b1cbab642a8a2461c5062ba8651c2da4829938ea7e06ee2c4f0a0603a1d5c0c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:1":"11d1656fbcf656566f659d1d247fd361cba3788a813846aedb4a715eaf1171fa","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:2":"dc6f551132330023a00ac99d2688f0ba5916b8d4541b1273403bcae5669bc870","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:2:score:1":"5358291ddabd288c6fa1b253f22e39980bb0633d29cc644c8f20f6a334f2bfd0","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:1":"de3b81ba1040c3593548db685565ca3f327a3edf771e49e44192c7c0631fee61","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:2":"d50b4f4896d5ed8a38f246f608d29ded2a38946fb4de88d3e94f279e6d79f70c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:3:score:1":"e261bb8edac19bc7aac95d9e16c9a711f97bb11cb5083ce6c0895caf0aae28ee","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:1":"60822e928ec2a54d56470289f59d7e56fe4584f82fcb6090c5f5c38cc20d4fb8","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:2":"99a9a4c1de9b28398d9fde8d149e555052eb33eb0506f7a656a85509e2f6b64f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:score:1":"081083cbfb4e2c87d748bab639492d9b67c791f799fca7835e794bcaec1dc189","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:4:score:2":"34bce7bc1ceffd5635459447130aeb45f09391ee8157890c5c3a0a90a94793ce","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:1":"ace7bfcd65abf5c000df293b6e2f2869e9bd06f3274340071d32b302dff42a1c","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:2":"57ac434730560fa91ffda6c1cd46ac6fd3b906a0c6b1dc8ebf8637449a48f9f5","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:score:1":"26b194dfd8e8131599291b244f370c9a63868785a88817f949192ff26752a79f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:5:score:2":"3bb0987fec4adbcc9e6c6104dff21f272919a48f256652980c5a2c6f4bf69155","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:1":"11949cbfdd65fb265e9d1f9495de9316981cd8f0c52c64a5260c192da6949773","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:2":"ca2ad62ddaf9df41132a7e2dc0f1f268b307175653822290ef7724f2e67e718f","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:6:score:1":"117860fd998d65710a2db0eb4b2252bf666de71e66d7334fea03e9d7d14057ae","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:1":"4dbd2216a2b1b6a2017b091ed339c0b8e582712f9e444a3e7b83656b335ed88e","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2":"da81635a5e2686798266f88eab2204ed03f55cf396a0d3b4aab7664eea701715","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:1":"dfa91b91177c4b84fc695bb49c9474be5fb35e7587499454161d129ca985fc93","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:1":"1842a1e9c196c45813d1027ba8f07427702f56b747dc086a86650da8ee7c9992","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:2":"f18b0122e7624f7d44b696a7e81a7bfffafed65a3ad5373f6b36ff969d32b43d","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:8:score:1":"49e6d18cbeedfd86c356dee7c3c3dcbc92fb9b55249ce6b8e5dab5f294c1f990","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:1":"980d65eef8ab8d393219dd63200a6bb4a5c874af3e8cf57c8233c7a125cd2722","arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:9:score:1":"5444c69ca1572d0bf95c0931c751491bb1da10367bb9f4ed6a2b10d2e3c6092f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:1":"3a5018474c4fd33d31049c26234c930575fff08cadf85ee7798926c9dbd01b5f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:0:score:1":"2609ffb668e32b68c350fd105109edec1e6659a6d3642082d15f4fa788bc708a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:1":"296198550a6ed1f52eb70f73beab002034e9eacbbac304ad89ed3d6515cf99bc","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:1:score:1":"2d4c57132379ec0a22e7ee4cae21be92955075130ec9c58bee7cd057733cc1eb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:1":"99eaff1d6b28e09a1f6cfc66078462745c526152a51260bf6cfeb0fd4aca2dba","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:2:score:1":"e9ffbfc6daa90dfde946a88e4c2c8c96904e7d9eda8dfd8b1dc1083870bf1168","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:1":"5df0685c0d82fbc6a7fe4813d582aa99f9746b9b4f4437d0648da80e005c72bd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:3:score:1":"568ce58f0b7c9592933223b1f16f7fab49ed60610f5fdcb3a43f87e46c9c9788","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:1":"9d9fd650744103fd520ecf90c46cc523271fb1f55d2a0dc02c86e4f06efc8bbd","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:score:1":"d3d8bc6d7bf2ae969fb61d0c63369ebcad699240861ed81aba3d30384c4bad56","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:4:score:2":"7d687835c0b59a63e30190534a838f6dde00438261bed3febac0a008c1cc384f","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:1":"56089dd4c7bc7a329f25e1be25241a807ee5aacf32ebade06366a081a5e7020a","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:5:score:1":"89edf98f346ec544be24cfd3666321a522676c25d34e4e12cf2afb6ed72fe416","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:1":"ed6e3cf39eb802623c73e1e01bb78ad884891b7442fef291afaab0d2e0e6a3f8","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:6:score:1":"4ef43c5ce5392e391518d95aaabde5dc7cfc4f6fc7c642a46da87771d0146fa0","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:1":"886178fdabfaa86ab92e13c3c11c198154faa6c29c49a2087aea58b348869054","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:7:score:1":"393f54188c21d110a261e109d93cbbb3716b67fee086fcdb506f9c76fd8c6539","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:1":"12cbdb2bba098758d0ba93d5e6db4b21adf02e40fd0380547c4e00fad0870044","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:8:score:1":"e9f8d4e541002ff5852b5aac668573be6114acfb8d6b978b810d70af82e5e979","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:1":"95338948e5888b0f478b02b1380a5bf983097b11b2f50c230082d5b99749bcdb","arena-2026-11-04:sub-4177fcdcebe8c743a926e41bfad9cdcb:2:9:score:1":"4e2bb5f170ef509c059191775b3a9575c831753928f2fa2d861fa55d1485a813","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:1":"13b0c28bd06ceb077e5523937f266e531649137060b91ce80b870ac14a60bc1c","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:2":"beb191f67ad696e2c2330bd68830b3913ec10f35039374bb897d5f1de8919994","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:0:score:1":"9c99cc9fb39df47879b21ba0222ea51a6f243b694ccc5afa523145f941822e5d","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:1":"9af7c38c4f654ca07b6b06f6fdb3213d1ad6641625ce06b94be2336628c16f3a","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:2":"677a26b88e9f9471271b34829a2028061fd11ab62b2db8bb41c23a26240f62f3","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:1:score:1":"163d2343d37864df65e887045d0cd02fe0f1c92e54e2f7b52fa0a5562cda8dac","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:1":"be29fef4e68df262c125cfd87f97bc9d6144390d7ff6b0493e88ad80d3ec4838","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:2":"e36d665b2d08e0fa6bd3c5071074d8419d22a50ac4e43a7e20ad02425a451cc1","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:2:score:1":"7acd6b0b9adc7c1418f23276e9894689db5e25b9a90aa588fd159b721ed4cc42","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:1":"0c38cdd4627a149944c07afd8846399781155ca80e71def7ed1a78261bd8db42","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:2":"ccc2c59f9fa901edac4dc38adfcc62ce20d29ed8fb8ac2d57d0212aadf8693ce","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:3:score:1":"b98f2088bdc55e1a70e3a8862afa0af9b42f2cf3aaebfd0bf6daaae5dc502253","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:1":"3c8f7373c83fe8c42c166267b22cc0938054ed9a16812703bb78b3f63a60ca0b","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:2":"79362393258fe32c9106fbd66d46a588da1208a4384056d14c454b3a8eed8c03","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:score:1":"f11661fa281bdd7cec60d1bbe25b553b1fc9ea7b90715bec64ed5914ccd02145","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:4:score:2":"d6a960a74e8982bd413231bf8bc3cb82eb975de4b4d75212a20af76e6486fd57","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:1":"82c02e958b503caf2cd2cef67e422cc07d463d975a0fb9bc685d18782afa1a41","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:2":"f12fa288d6967ce874d47d958f209dd6defc525a51f555f0e7a97077770e0237","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:5:score:1":"677441e316777cb5e15a0b47b86aebad5c97f366665767f3da08be00e0d5ef27","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:1":"c0a5d1be6b5ec3aa5f25d38e94184e28c2a0448159221afcf301a7611ead450a","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:2":"e78453716fde1b5f6e0169f011c449c2d03474c267ea110c2b7aff3d904fe620","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:6:score:1":"50c1cc1c0aa740086c020e7361a4e5998aea963bfca06e02a7819e9403efd183","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:1":"b4aebd2a65e99ecca6935d494a3919cdeba0704be61d061ba6043d7aa3522b75","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:2":"8939bcd6b6b0dff86d52b8b46aafb14859a88ad4320bb1b7433147013e7b9abc","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:7:score:1":"fcc675b31f98c2b4f3480b333840437d81221b5a328d7da515106f6039963f3e","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:1":"3c9d87c9ea81c7a11ab83773984013da4d579c58e67816344a52393819aaa84c","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:2":"b72b9f165d00b45379d1595a57f3e7b55f9b9492a368fada154b11029d77bea7","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:8:score:1":"2707c606c292b38fabc309e3bd5251901071f9a95ec237d85fbb46965de36222","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:1":"7a9caf5eb81e01b6d44d6126fac10b029d7187c6de9d318380ac872679c491d4","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:2":"531cebe1575831843537a108faea005391e40f0aab765824de65817a37e2987f","arena-2026-11-04:sub-4dd3b8b9fb7d6ffe2079e17ea74b3ac1:2:9:score:1":"ddd5b23d069830e44da0779bc2bb42ce1fe5598bdc2101344138c330613bdde6","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:1":"0bcb21e59a611550278ebd50218d91439bbe28416ed5c5494c0a0aa78b7cb5f7","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:0:score:1":"5f6c746905fc5cc4efa0351012138abbceea122f67933493bf13b5792493bad0","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:1":"e4d82a860882f637c86bb8b12dfbd8432776524d71d5a6e023cc86dae34c984a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:2":"8fa8c721b68e07e3692e90db871b2090f2a961c3796b2e84b45b0d5d594d7729","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:score:1":"3220cc6882728e0d3e82f63ea6ddb4c6680e76745d2cbb4a32049e48f31c6a9e","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:1:score:2":"1d2800618e653494e5a2938a7963774e93ec106ae73a02391545161fddb20239","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:1":"8c6530bf91ba2c8e74d4587f4bb92d3b12170ed76b9c9d1449c393b1c7ef54d2","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:2:score:1":"14be0e4ed29d0f779caeddf5305d3e2e02435371a32a71421ecbb5b43103316a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:1":"f4f77e47bf2cfc3d6da907ff6dd13805ce3a54682df4181ac6e71fdbd7f296be","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:score:1":"299b2b1210604095eebcdd53261c9d13a88908b16d08c92948c1e360656a494f","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:3:score:2":"c7b7870f73d199967bb78796f2ae01450e3d88cc03644f773010947a6282512b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:1":"d6376c4af19609735830677bf896f4f99a1d24cdb7356d878e7fcfcfe7e62374","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:2":"1d232b7ae5dafab4534d72943860df6a115d918660d6f46cdd29ee894d02687a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:4:score:1":"451a0bdb6bc2bdabff91a058b48cc3c6f921e14210bce4bb45062ba5d7a9f47b","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:1":"709aaf8aff5716cb63ae3e9c8771806a674f865b4c88400e02085c36855f5779","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:2":"7aaa48d5ad93c6d12fbf83d8c2464338e5e93ed122ba7a17840d8132cfabb914","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:5:score:1":"b81a5e47e2d6d75aa331a8d086b492b28141c0e54fdbd4f50b3ed99a22b19194","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:1":"dfcb1ee2cb5750bdc98ef9b6e6db49802c475d2009eab3d378c02fcd6db493ce","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:2":"d472f1fbbd899f37960d71bee5f634780294f103605be567f9114af236318d97","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:6:score:1":"d7e83c751477487d151ef8de56815e95d8eb86b520ce8289999f9a9caaab8c0f","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:1":"cd4e0de1d740fd78fb14af9270652e06809a481963ee14db51cf3d76a41091de","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:2":"55d8bad8c69a856bf397d07f560c3250a0ac713ffe3ecc4198f20df74d93a2c8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:7:score:1":"f9cf79ffcb3afbe56395f0dd28f53c9489413ccf222172cc434102734772b1f8","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:1":"979aae1e83870bafab5afc78a41571410f849d21cd5ef31924ec41dc150a1036","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:2":"ca1c960c80be55dbcb8688fbf76f4ede2c6e2f621fb1fcb8299bbbd5a1571452","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:8:score:1":"ddc63bc8d4171c03245aca0cf68a0e114319e1923a99dd49e4842864c5cda18c","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:1":"d27d2d8e9cc03b8b70e18bf4cd2754b8e0cec400d42c2877faaa1adfbdbe6614","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:2":"6a95000d52da3f4a83bd9b61a06816309f4b4ffc142c95aefdb79211133b3a2a","arena-2026-11-04:sub-5fa210d9d0819769dc97e7111aab05f0:2:9:score:1":"c679ac34f92c6aa71f7982672b1ff55d68bd1fea6c9ae20e968d76712c0ef263","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:1":"72345062a5eda8f1002c031a89c80a6d624ffc92571169746a7f4baa64613eb6","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:0:score:1":"e77bcef5d6375101d92fa50e2ea49d3c13ca867c65b7ba0b1e477b72496145a5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:1":"d80fb6858d35acef4374a9c2c9b50bb6371a3988157009b6d807890de1366a68","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:1:score:1":"e8c45b0a7463accf37496766eb463004db3b19d90d31e829a1d47095ab4f8573","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:1":"32ebd3a25489b00bae2c2e295d92c9c62f9e6aa4809649b918ba72d42b89bef9","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:2:score:1":"c4524fe788a6a0b550a93bca1369428a30b42df98ae2d5b35cd15ee85763c160","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:1":"2d6440d80a55a0e824e3808e644b84f1da6fc6b56b6d8713f02d6a943887fa99","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:3:score:1":"af6d0b2a298b381551e974b07fcb540f686dbc9a49a57446bdd15fe3b2290b44","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:1":"72092fd0c50ec25804fb6653590282186fa828df1806917fb37363cfdaa0882e","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:4:score:1":"85893364a0c82cb891ff78ffd216dce0165ea55196b3bfdf8df403afab6002f4","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:1":"4d06288fd871052d06b403467038bcf46d0d9fb09304c1ccffb4842aa6b60323","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:2":"6457a984cb70552aae9980873f12e9f9894396e09e97e98924ea64d16ec18431","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:score:1":"3b0770c4dc27ec4a878b7dbe5ebd2bfecb50a4e4cdae016feeb750e4761a8a80","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:5:score:2":"1aac0a8df06b7111dccad01e17ea7df457d5e7b297978e31b3be057d28f4f408","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:1":"725554ad6e1c330fd08e735935850d729b9f74bf0a4699c5568b132e29fe95de","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:6:score:1":"05240863dc482143f58e5f1958fc4531d62c53e8f3292107830058acd86a42a7","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:1":"1cbdbbfa8f820c4ed76831a5012db8fb33b5b07e89fc2eaf09b89b37a88b09d5","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:7:score:1":"cf20c855960208fa23ab18aad393699ee3c7d3e10a7e0dc7beb5f1a103421571","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:1":"8a9786ccee46e404633392c39c7aec178a4878b11a67fb6ff35660aa43e35b77","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:8:score:1":"e8558098e2c5a4facc8283a0a8afa97a636fbe499f510817cdc469b506e58231","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:1":"0a6d9577cbc30ad2be3135cfb55b81bad358919142f76fa4aedafd1a8000d30d","arena-2026-11-04:sub-e5819d7d6793b62312f3e5b6de9758eb:2:9:score:1":"d30848602d9beb885566befacc038b63adc6a4480ab8ab877139674d01c9f8ee","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:1":"ebbb7072c49a9235e6905ecab0572b4f956f559b21e612409c7bf968a6358a7e","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:0:score:1":"7f48f29b8d2bc312565fc3147ef1b6fbd3b4d2ba936d361a5ce632fc68cc405f","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:1":"6f2b46eb2a81dcb09d4bbba2b52c76cb85db3aaa89cc56c730a1ec75b67350a0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:1":"4d1dc0d8e939c38d35b4b92c77db54e5a561fcd74e891405fc357183f655a124","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:2":"f756283f3608e3f03d1f0332d8ba173ee3ebbc8e10cf033b283fd8b597da3a2f","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:3":"e1b7d37f9ac1c84f101003bd5c3eff6d7938c8047c7dc9d01a2c8bef157d083a","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:1:score:4":"e72dcccde665db701ace3f0f9ec16161d95ee58a667a352ddedff05125711bf7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:1":"e9f8906fbf02048a0ae08a9b67c2a00571829b05aff8fda3b7621325fbdfb236","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:score:1":"42bf40a75576951c950187b9039063cdbacb4137b38166028b96bee3fc64c2ca","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:2:score:2":"9f3ee6a0ee77d569551d720062d9f19f11608af8beb5bcde674c20cc0ac13832","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:1":"5c4a462ee6197fc29b0709794eb76d1f81becca4b56483d31206bb3c9cb065f3","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:score:1":"cce62acef0f179ca6ad8b7ba0a454d82fc3190b3e316dcdf5ada17a58b8a93e7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:3:score:2":"abcc1094d0416a4a691311a0b0fa67bb36399107ed9c2d8572bd29dd7d0ed046","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:1":"1f354b43d71a17932751c5606384a156510fe4a4a1c7857cd360e3db67e148c7","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:score:1":"212c95a6b5d6ea00dfdfd7eb443cf070a22f71eca211e9c000f6c042de477c31","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:4:score:2":"7ce90d0fb3634cc833a1902fca542fc31ec8d565560ff50f06799160d748a3c4","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:1":"5d95bfe5eee06871f87c4ae7fd1d07da102b9a87d57e2caca04d19ee361a3023","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:score:1":"3d9bb151fc22b98b5c3cff4a94955ef33ba3cee8711b4a4ce65d8f424d766015","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:5:score:2":"117ba598e8e33f897bc441665920b2d6ff0455464f2707188882435203c50816","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:1":"d2c7002ddaac2fea5336428430e9729fa029d68bd12ae46aa5d8db14fca8b300","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:6:score:1":"20a150db1984208058d9479bc7d34e4fd7b0d9545ee8e17085cb9d831b52b6da","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:1":"42e49e8d933625a2a39da2f43928e41f2c6da252de4e0dbc1b87894893f380e1","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:7:score:1":"6e3df1e43453f3916bd5f9be10da8a959a3c3bc40ae229c48a58def416be0e91","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:1":"06d668b9bad8d0d36fd0836de7ae246dfa768ddd7688ec577fb751aa8b030d02","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:8:score:1":"6624ec3bbff2f7bbf7955a2a79e8bb25390986591fe73a6f3bff0339cf2778e0","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:1":"7c925b5e63438385de95c576b19cf50c7fac52d170267a1de700482eaec86d71","arena-2026-11-04:sub-ffbd4d0e7103ce5c79c95fef2035d209:2:9:score:1":"0aa3ade417f05a5f03c9450fd82ef4c4e74df72326d7e1ab0f8d6333fd64cb1b"},"source_archive_sha256":"sha256:9df0c4d451043da9fcec15fe86f9bfeb87fd7c048609aaf43736a6cbf0244cc4","source_normalized_sha256":"sha256:1047b383f4b32dbc38d19e5f052384174d4a9a89a7c48b4620cecb18714ed06d","source_ref":"arena/arena-2026-11-04/sources/baseline-2026-11-04-source-24d9201c3c5fdc86cb1676628774223205c87166.tar.gz","source_size_bytes":468640,"source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/24d9201c3c5fdc86cb1676628774223205c87166.tar.gz","submission_sha256":"a645dc97db82a5e6d2959d7f21f5afbcf49486c800f4effb7d8c593c7c11bd37"}$manifest621$::JSONB;
 r public.lab_arena_rounds; n public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 s public.lab_arena_submissions; replacement public.lab_arena_submissions; future_round public.lab_arena_rounds;
 original public.lab_arena_runs; judge public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; item JSONB; work JSONB:='[]'; originals JSONB:='{}';
 hashes JSONB; accepted_hashes JSONB; retired_hashes JSONB; old_cost JSONB; revision BIGINT; request_hash TEXT; attempt INTEGER; judge_attempt INTEGER;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT future_round FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR UPDATE;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request;
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'baseline_source_supersession_v1'
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR h.new_round_doc->'participants' IS DISTINCT FROM r.participants THEN
   RAISE EXCEPTION 'Source correction621 replay differs'; END IF;
  RETURN;
 END IF;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions
 WHERE round_id=rid AND submission_id='baseline-2026-11-04' FOR UPDATE;
 IF current_setting('session_replication_role')<>'origin' OR c.operator_paused
  OR coalesce(c.guard_commitment,'')<>'' OR coalesce(c.owner_commitment,'')<>''
  OR c.guard_expires_at IS NOT NULL OR coalesce(c.candidate_commit,'')<>''
  OR coalesce(c.restart_scope,'')<>'' OR coalesce(c.restart_phase,'')<>''
  OR c.guard_generation<(manifest->>'minimum_guard_generation')::BIGINT
  OR coalesce(c.captured_leases,'[]')<>'[]'::JSONB
  OR r.status<>'published' OR r.status_generation<>10 OR r.stage_generation<>7
  OR r.scoring_generation_floor<>0 OR r.published_at IS NULL OR r.publication_doc IS NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.effective_reward_epoch IS NOT NULL OR r.signing_key_doc IS NOT NULL
  OR r.champion_fallback_providers<>ARRAY[]::TEXT[] OR NOT r.champion_funding_frozen
  OR clock_timestamp()>='2026-11-04T20:00:00Z'::TIMESTAMPTZ
  OR encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'round_sha256'
  OR encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'submission_sha256'
  OR NOT s.is_king OR s.status<>'frozen'
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND status IN('pending','leased','submitted'))
  OR future_round.status<>'open' OR future_round.champion_funding_frozen
  OR future_round.participants IS NOT NULL OR future_round.benchmark_ref IS NOT NULL
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds prior
    WHERE prior.round_id='arena-2026-11-03' AND prior.status='published'
      AND prior.reward_activated_at IS NOT NULL
      AND encode(extensions.digest(to_jsonb(prior)::TEXT,'sha256'),'hex')=manifest->>'prior_round_sha256'
      AND prior.publication_doc#>>'{king_decision,winner_submission_id}'='sub-28594d902080fb5213f4344c09ffb531'
      AND prior.publication_doc#>>'{king_decision,king_hotkey}'='5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq'
      AND prior.promotion_doc->>'commit'='24d9201c3c5fdc86cb1676628774223205c87166') THEN
  RAISE EXCEPTION 'Source correction621 round, guard, clock or source authority differs'; END IF;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO hashes
 FROM public.lab_arena_runs a WHERE a.round_id=rid;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO accepted_hashes
 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.status='accepted';
 IF hashes IS DISTINCT FROM manifest->'run_hashes'
  OR accepted_hashes IS DISTINCT FROM manifest->'accepted_run_hashes'
  OR (SELECT count(*) FROM jsonb_object_keys(accepted_hashes))<>180 THEN
  RAISE EXCEPTION 'Source correction621 original history differs'; END IF;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO retired_hashes
 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=s.submission_id AND a.stage=1;
 IF (SELECT count(*) FROM jsonb_object_keys(retired_hashes))<>31 THEN
  RAISE EXCEPTION 'Source correction621 original source generation count differs'; END IF;
 FOR attempt IN 0..9 LOOP
  old_cost:=public.lab_arena__successful_icp_cost_state(rid,s.submission_id,attempt);
  IF (old_cost->>'inflight_calls')::BIGINT<>0 OR (old_cost->>'success_unresolved_calls')::BIGINT<>0 THEN
   RAISE EXCEPTION 'Source correction621 original execution liability unresolved'; END IF;
 END LOOP;
 replacement:=s; replacement.source_ref:=manifest->>'source_ref';
 replacement.source_size_bytes:=(manifest->>'source_size_bytes')::BIGINT;
 replacement.source_archive_sha256:=manifest->>'source_archive_sha256';
 replacement.source_normalized_sha256:=manifest->>'source_normalized_sha256';
 n:=r; n.status:='stage2_scoring'; n.status_generation:=r.status_generation+1;
 n.publication_doc:=NULL; n.published_at:=NULL; n.king_outcome:=NULL; n.king_hotkey:=NULL;
 n.promotion_doc:=NULL; n.baseline_promoted_at:=NULL;
 n.champion_submission_id:='sub-28594d902080fb5213f4344c09ffb531';
 n.champion_hotkey:='5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq';
 n.configuration_doc:=jsonb_set(jsonb_set(r.configuration_doc,
  '{baseline_source_url}',manifest->'source_url'),'{reasoning_baseline_source_sha256}',manifest->'source_archive_sha256');
 -- The finite correction window does not reopen miner or admission work.
 n.configuration_doc:=jsonb_set(jsonb_set(n.configuration_doc,
  '{schedule,final_scoring_close}','"2026-11-04T23:55:00Z"'),
  '{schedule,publication_deadline}','"2026-11-04T23:59:00Z"');
 SELECT jsonb_agg(CASE WHEN p->'is_king'='true'::JSONB THEN
   p||jsonb_build_object('source_ref',replacement.source_ref,'source_size_bytes',replacement.source_size_bytes)
   ELSE p END ORDER BY ordinal) INTO n.participants
 FROM jsonb_array_elements(r.participants) WITH ORDINALITY roster(p,ordinal);
 FOR original IN SELECT * FROM public.lab_arena_runs
  WHERE round_id=rid AND submission_id=s.submission_id AND stage=1 AND kind='execute'
   AND status='accepted' ORDER BY icp_position FOR UPDATE LOOP
  SELECT * INTO STRICT judge FROM public.lab_arena_runs
   WHERE round_id=rid AND scored_run_id=original.run_id AND kind='score' AND status='accepted' FOR UPDATE;
  IF original.icp_position NOT BETWEEN 0 AND 9 OR original.terminal_cause<>'accepted'
   OR original.output_ref IS NULL OR original.miner_hotkey<>s.miner_hotkey
   OR judge.terminal_cause<>'accepted' OR judge.output_ref IS NULL OR judge.scoring_generation_floor<>0
   OR (judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position)
    IS DISTINCT FROM (original.submission_id,original.miner_hotkey,original.stage,original.icp_position)
   OR NOT public.lab_arena__judge_calls_clear_v1(judge.assignment_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs later
    WHERE later.assignment_id IN(original.assignment_id,judge.assignment_id)
      AND later.attempt>CASE WHEN later.kind='score' THEN judge.attempt ELSE original.attempt END) THEN
   RAISE EXCEPTION 'Source correction621 original baseline membership differs'; END IF;
  SELECT max(a.attempt)+1 INTO attempt FROM public.lab_arena_runs a WHERE a.assignment_id=original.assignment_id;
  SELECT max(a.attempt)+1 INTO judge_attempt FROM public.lab_arena_runs a WHERE a.assignment_id=judge.assignment_id;
  work:=work||jsonb_build_array(jsonb_build_object('old_run_id',original.run_id,'old_judge_run_id',judge.run_id,
   'new_run_id',original.assignment_id||':'||attempt,'new_attempt',attempt,
   'new_judge_attempt',judge_attempt,'icp_position',original.icp_position));
  originals:=originals||jsonb_build_object(original.run_id,to_jsonb(original),judge.run_id,to_jsonb(judge));
 END LOOP;
 IF jsonb_array_length(work)<>10
  OR (SELECT count(DISTINCT x->>'icp_position') FROM jsonb_array_elements(work)x)<>10 THEN
  RAISE EXCEPTION 'Source correction621 baseline count differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,request,request_hash,'nov04-baseline-source621',txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',jsonb_build_object(
   'operation','baseline_source_supersession_v1','scoring_generation_floor',0,
   'superseded_runs',originals,'work_items',work,'accepted_run_hashes',accepted_hashes,
   'retired_source_run_hashes',retired_hashes,
   'old_submission_doc',to_jsonb(s)-'updated_at','new_submission_doc',to_jsonb(replacement)-'updated_at',
   'claim_control_doc',to_jsonb(c),'original_round_updated_at',r.updated_at,
   'original_submission_updated_at',s.updated_at,'source_manifest',manifest-'run_hashes'-'accepted_run_hashes'));
 UPDATE public.lab_arena_submissions SET source_ref=replacement.source_ref,
  source_size_bytes=replacement.source_size_bytes,source_archive_sha256=replacement.source_archive_sha256,
  source_normalized_sha256=replacement.source_normalized_sha256 WHERE submission_id=s.submission_id;
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,
  configuration_doc=n.configuration_doc,participants=n.participants,publication_doc=NULL,published_at=NULL,
  king_outcome=NULL,king_hotkey=NULL,promotion_doc=NULL,baseline_promoted_at=NULL,
  champion_submission_id=n.champion_submission_id,champion_hotkey=n.champion_hotkey WHERE round_id=rid;
 FOR item IN SELECT value FROM jsonb_array_elements(work) LOOP
  UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',
   terminal_doc=jsonb_build_object('operation','baseline_source_supersession_v1',
    'request_id',request,'new_execution_run_id',item->>'new_run_id')
  WHERE run_id IN(item->>'old_run_id',item->>'old_judge_run_id');
  SELECT * INTO STRICT original FROM public.lab_arena_runs WHERE run_id=item->>'old_run_id';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
   stage,icp_position,attempt,status,stage_generation,kind)
  VALUES(item->>'new_run_id',original.assignment_id,rid,original.submission_id,original.miner_hotkey,
   1,original.icp_position,(item->>'new_attempt')::INTEGER,'pending',r.stage_generation,'execute');
  IF NOT (SELECT public.lab_arena__source_correction621_v1(a) FROM public.lab_arena_runs a
   WHERE a.run_id=item->>'new_run_id') THEN RAISE EXCEPTION 'Source correction621 new root invalid'; END IF;
 END LOOP;
 IF (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton)<>to_jsonb(c)
  OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid)<>
    (SELECT count(*)+10 FROM jsonb_object_keys(manifest->'run_hashes'))
  OR EXISTS(SELECT 1 FROM jsonb_each_text(manifest->'run_hashes') expected
   LEFT JOIN public.lab_arena_runs actual ON actual.run_id=expected.key
   WHERE actual.run_id IS NULL OR CASE WHEN originals ? expected.key THEN
    actual.status<>'failed' OR actual.terminal_cause<>'stage_closed'
    OR (to_jsonb(actual)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])<>
      ((originals->expected.key)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
    ELSE expected.value IS DISTINCT FROM encode(extensions.digest(to_jsonb(actual)::TEXT,'sha256'),'hex') END) THEN
  RAISE EXCEPTION 'Source correction621 preserved work differs'; END IF;
END;
$correct621$;

DO $restore621$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena621_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena621_schema_acl) THEN RAISE EXCEPTION 'Source correction621 schema ACL differs'; END IF;
END;
$restore621$;
NOTIFY pgrst,'reload schema';
COMMIT;
