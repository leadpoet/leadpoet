-- Second phase: the coordinator prepares one durable replacement output with
-- prepare_nov4_miner_disk629.prepare(existing_readonly_service). Supply only
-- that reviewed private artifact as lab_arena.miner_disk629_payload in this
-- operator session. This creates no public RPC or alternate input builder.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $plan629$
DECLARE payload JSONB; r public.lab_arena_rounds; n public.lab_arena_rounds;
 source public.lab_arena_runs; previous public.lab_arena_runs; h public.lab_arena_scoring_supersessions;
 old_history public.lab_arena_scoring_supersessions; item JSONB; plan JSONB; expected JSONB;
 revision BIGINT; request_hash TEXT; slot_count INTEGER;
BEGIN
 IF octet_length(coalesce(current_setting('lab_arena.miner_disk629_payload',TRUE),'')) NOT BETWEEN 2 AND 1048576 THEN
  RAISE EXCEPTION 'Miner629 reviewed preparation missing'; END IF;
 payload:=current_setting('lab_arena.miner_disk629_payload')::JSONB;
 IF payload->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.miner_disk629_preparation.v1'
  OR jsonb_typeof(payload->'expected_round') IS DISTINCT FROM 'object'
  OR jsonb_typeof(payload->'source') IS DISTINCT FROM 'object'
  OR jsonb_typeof(payload->'work_item') IS DISTINCT FROM 'object'
  OR coalesce(payload->>'output_sha256','') !~ '^sha256:[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'Miner629 preparation invalid'; END IF;
 request_hash:='sha256:'||encode(extensions.digest(payload::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-04' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id='nov04-miner-disk629-plan-v1';
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Miner629 replay differs'; END IF;
  RETURN;
 END IF;
 SELECT * INTO STRICT old_history FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov04-miner-disk628-v1';
 SELECT jsonb_object_agg(key,value) INTO expected FROM jsonb_each(to_jsonb(r)) WHERE key=ANY(
  ARRAY['round_id','status','status_generation','stage_generation','scoring_generation_floor','configuration_doc','participants','stage2_scoring_plan_doc']);
 IF expected IS DISTINCT FROM payload->'expected_round'
  OR r.status NOT IN('stage2_scoring','stage2_judged','scored') OR r.scoring_generation_floor<>0
  OR r.configuration_doc IS DISTINCT FROM old_history.new_round_doc->'configuration_doc'
  OR r.participants IS DISTINCT FROM old_history.new_round_doc->'participants'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.baseline_promoted_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR (SELECT count(*) FROM public.lab_arena_restart_claim_control WHERE singleton)<>1
  OR EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE operator_paused OR guard_commitment<>''
   OR owner_commitment<>'' OR guard_expires_at IS NOT NULL OR candidate_commit<>'' OR restart_scope<>'' OR restart_phase<>'') THEN
  RAISE EXCEPTION 'Miner629 round, generation or control differs'; END IF;
 SELECT * INTO STRICT source FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:3' FOR UPDATE;
 SELECT jsonb_object_agg(key,value) INTO expected FROM jsonb_each(to_jsonb(source)) WHERE key=ANY(
  ARRAY['run_id','assignment_id','round_id','submission_id','miner_hotkey','kind','stage','icp_position','attempt','status','terminal_cause','stage_generation','scoring_generation_floor','output_ref']);
 item:=payload->'work_item';
 IF source.status<>'accepted' OR source.terminal_cause<>'accepted' OR source.output_ref IS NULL
  OR NOT public.lab_arena__miner_disk628_v1(source)
  OR expected IS DISTINCT FROM payload->'source'
  OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(item)k) IS DISTINCT FROM
   ARRAY['company_judgment_refs','icp_position','output_ref','scored_run_id','submission_id']::TEXT[]
  OR item->>'scored_run_id' IS DISTINCT FROM source.run_id OR item->>'submission_id' IS DISTINCT FROM source.submission_id
  OR item->>'output_ref' IS DISTINCT FROM source.output_ref OR item->'icp_position' IS DISTINCT FROM to_jsonb(source.icp_position)
  OR NOT public.lab_arena__company_judgment_refs_v1(r,item->'company_judgment_refs')
  OR NOT public.lab_arena__judge_calls_clear_v1(source.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(source.assignment_id||':score')
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id IN(source.assignment_id,source.assignment_id||':score')
   AND a.status IN('pending','leased','submitted')) THEN
  RAISE EXCEPTION 'Miner629 durable source or exact canonical item differs'; END IF;
 SELECT * INTO STRICT previous FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:score:1';
 IF previous.status<>'failed' OR previous.terminal_cause<>'stage_closed'
  OR previous.terminal_doc->>'request_id'<>'nov04-miner-disk628-v1'
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=previous.assignment_id AND a.attempt>1) THEN
  RAISE EXCEPTION 'Miner629 retained judgment lineage differs'; END IF;
 SELECT count(*) INTO slot_count FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items') saved
 WHERE saved->>'submission_id'=source.submission_id AND saved->'icp_position'=to_jsonb(source.icp_position)
  AND saved->>'scored_run_id'='arena-2026-11-04:sub-2246da743ed59dcfe8b248e9d0ddfe50:2:7:2';
 IF slot_count<>1 OR EXISTS(SELECT 1 FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'zero_rows') z
  WHERE z->>'submission_id'=source.submission_id AND z->'icp_position'=to_jsonb(source.icp_position)) THEN
  RAISE EXCEPTION 'Miner629 original slot differs'; END IF;
 SELECT jsonb_set(r.stage2_scoring_plan_doc,'{work_items}',jsonb_agg(CASE WHEN saved->>'submission_id'=source.submission_id
  AND saved->'icp_position'=to_jsonb(source.icp_position) THEN jsonb_build_object('scored_run_id',source.run_id,
   'submission_id',source.submission_id,'icp_position',source.icp_position,'output_ref',source.output_ref)
  ELSE saved END ORDER BY ordinal)) INTO plan
 FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items') WITH ORDINALITY items(saved,ordinal);
 n:=r; n.status:='stage2_scoring';n.status_generation:=r.status_generation+1;n.stage2_scoring_plan_doc:=plan;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov04-miner-disk629-plan-v1',request_hash,'nov04-miner-disk629-plan',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','miner_infrastructure_plan629_v1','scoring_generation_floor',0,
   'work_item',item,'output_sha256',payload->>'output_sha256','preparation',payload));
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,
  stage2_scoring_plan_doc=plan WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
 VALUES(source.assignment_id||':score:2',source.assignment_id||':score',r.round_id,source.submission_id,source.miner_hotkey,
  2,7,2,'pending',r.stage_generation,'score',source.run_id,item->'company_judgment_refs');
END;
$plan629$;
COMMIT;
