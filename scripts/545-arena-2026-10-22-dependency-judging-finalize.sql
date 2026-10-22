-- DRAFT owner-controlled October22 continuation. No production application.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='60s';

-- Fresh model_error is only a label. Before binding any model_zero_scores,
-- root must review the new private setup/runtime diagnostics and record the
-- native evidence for genuine miner fault. Infrastructure receives no zero.
-- The exact future output-dependent payload must be reviewed before application.
-- It is supplied as transaction-local leadpoet.oct22_judging_payload JSON by
-- a separately reviewed binding, never inferred or filled with empty refs.
DO $finalize545$
DECLARE
 rid CONSTANT TEXT:='arena-2026-10-22';sid CONSTANT TEXT:='sub-5f48ab5a81c56a43b1ce1b66048920a2';
 actor CONSTANT TEXT:='oct22-dependency-execution-recovery542';
 payload JSONB;plan JSONB;items JSONB;zeros JSONB;expected_items JSONB;expected_zeros JSONB;expected_incomplete JSONB;
 r public.lab_arena_rounds;n public.lab_arena_rounds;intermediate public.lab_arena_rounds;
 c public.lab_arena_restart_claim_control;h public.lab_arena_scoring_supersessions;p public.lab_arena_scoring_supersessions;
 revision BIGINT;item JSONB;source public.lab_arena_runs;ordinary JSONB;payload_hash TEXT;
BEGIN
 payload:=nullif(current_setting('leadpoet.oct22_judging_payload',TRUE),'')::JSONB;
 IF payload IS NULL OR jsonb_typeof(payload)<>'object'OR octet_length(payload::TEXT)>8388608
  THEN RAISE EXCEPTION 'Oct22 exact reviewed phase2 payload missing or oversized';END IF;
 plan:=payload->'plan';items:=payload->'work_items';zeros:=payload->'model_zero_scores';
 payload_hash:='sha256:'||encode(extensions.digest(payload::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-542-phase1-v1';
 SELECT * INTO STRICT p FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-544-prepare-v1';
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions done WHERE done.round_id=rid
  AND done.request_id='oct22-dependency-545-final-v1'AND done.request_hash=payload_hash)THEN RETURN;END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions done WHERE done.round_id=rid
  AND done.request_id='oct22-dependency-545-final-v1')THEN RAISE EXCEPTION 'Oct22 phase2 replay payload differs';END IF;
 IF NOT c.operator_paused OR c.actor_ref<>actor OR c.pause_reason<>'oct22_dependency_phase2_transition'
  OR c.guard_generation<>(h.recovery_doc#>>'{control_before,guard_generation}')::BIGINT
  OR c.guard_commitment<>''OR c.owner_commitment<>''OR c.restart_phase<>''OR c.guard_expires_at IS NOT NULL
  OR c.captured_leases<>'[]'::JSONB OR r.status NOT IN('stage2','stage2_closed')
  OR(to_jsonb(r)-'updated_at')IS DISTINCT FROM p.new_round_doc
  OR encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex')IS DISTINCT FROM payload->>'round_sha256'
  OR r.configuration_doc#>'{banned_hotkeys}'IS DISTINCT FROM h.old_round_doc#>'{configuration_doc,banned_hotkeys}'
  OR r.publication_doc IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR clock_timestamp()>='2026-10-22T20:30:02Z'::TIMESTAMPTZ
  THEN RAISE EXCEPTION 'Oct22 exact prepared authority differs';END IF;
 IF jsonb_typeof(plan)IS DISTINCT FROM'object'OR jsonb_typeof(items)IS DISTINCT FROM'array'OR jsonb_typeof(zeros)IS DISTINCT FROM'array'
  OR(plan-ARRAY['work_items','zero_rows','incomplete_rows'])IS DISTINCT FROM
     ((h.old_round_doc->'stage2_scoring_plan_doc')-ARRAY['work_items','zero_rows','incomplete_rows'])
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('pending','leased','submitted'))
  THEN RAISE EXCEPTION 'Oct22 phase2 shape or drain differs';END IF;
 IF(SELECT count(*)FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3)<>10
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3
   AND(status NOT IN('accepted','failed')OR result_doc->>'terminal_status'IS DISTINCT FROM terminal_cause))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='score')
  THEN RAISE EXCEPTION 'Oct22 fresh terminal cohort differs';END IF;
 -- Preserve every other original plan row. Only the ten old unmeasured setup
 -- positions are replaced by their actual fresh outcomes; never use their old
 -- model_error labels as proof that a new infrastructure failure deserves zero.
 SELECT coalesce(jsonb_agg(value ORDER BY value->>'scored_run_id'),'[]')INTO expected_items
 FROM(SELECT value FROM jsonb_array_elements(h.old_round_doc#>'{stage2_scoring_plan_doc,work_items}')
  UNION ALL SELECT jsonb_build_object('scored_run_id',run_id,'submission_id',sid,'icp_position',icp_position,'output_ref',output_ref)
  FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='accepted')q;
 IF plan->'work_items'IS DISTINCT FROM expected_items THEN RAISE EXCEPTION 'Oct22 accepted-only canonical plan differs';END IF;
 SELECT coalesce(jsonb_agg(value ORDER BY value->>'submission_id',(value->>'icp_position')::INTEGER),'[]')INTO expected_zeros
 FROM(SELECT value FROM jsonb_array_elements(h.old_round_doc#>'{stage2_scoring_plan_doc,zero_rows}')WHERE value->>'submission_id'<>sid
  UNION ALL SELECT jsonb_build_object('submission_id',sid,'icp_position',a.icp_position,'cause',a.terminal_cause)
  FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid AND a.kind='execute'AND a.attempt=3 AND a.status='failed'
   AND a.terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error'))q;
 SELECT coalesce(jsonb_agg(value ORDER BY value->>'submission_id',(value->>'icp_position')::INTEGER),'[]')INTO expected_incomplete
 FROM(SELECT value FROM jsonb_array_elements(coalesce(h.old_round_doc#>'{stage2_scoring_plan_doc,incomplete_rows}','[]'))WHERE value->>'submission_id'<>sid
  UNION ALL SELECT jsonb_build_object('submission_id',sid,'icp_position',a.icp_position,'cause',a.terminal_cause)
  FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid AND a.kind='execute'AND a.attempt=3 AND a.status='failed'
   AND a.terminal_cause NOT IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error'))q;
 IF plan->'zero_rows'IS DISTINCT FROM expected_zeros OR coalesce(plan->'incomplete_rows','[]')IS DISTINCT FROM expected_incomplete
  THEN RAISE EXCEPTION 'Oct22 fresh failure plan differs';END IF;
 IF jsonb_array_length(items)<>(SELECT count(*)FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='accepted')
  OR(SELECT count(DISTINCT value->>'scored_run_id')FROM jsonb_array_elements(items))<>jsonb_array_length(items)
  THEN RAISE EXCEPTION 'Oct22 new judge coverage differs';END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(items)LOOP
  IF jsonb_typeof(item)IS DISTINCT FROM'object'OR(SELECT array_agg(k ORDER BY k)FROM jsonb_object_keys(item)k)
   IS DISTINCT FROM ARRAY['company_judgment_refs','icp_position','output_ref','scored_run_id','submission_id']::TEXT[]
   THEN RAISE EXCEPTION 'Oct22 exact new judge item fields differ';END IF;
  SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=item->>'scored_run_id'
   AND round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='accepted';
  IF item->>'submission_id'IS DISTINCT FROM sid OR(item->>'icp_position')::INTEGER IS DISTINCT FROM source.icp_position
   OR item->>'output_ref'IS DISTINCT FROM source.output_ref
   OR public.lab_arena__company_judgment_refs_v1(r,item->'company_judgment_refs')IS NOT TRUE
   THEN RAISE EXCEPTION 'Oct22 new judge binding differs';END IF;
 END LOOP;
 IF(SELECT count(DISTINCT value->>'run_id')FROM jsonb_array_elements(zeros))<>jsonb_array_length(zeros)
  OR jsonb_array_length(zeros)<>(SELECT count(*)FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='failed'
  AND terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error'))
  THEN RAISE EXCEPTION 'Oct22 genuine fresh zero coverage differs';END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(zeros)LOOP
  SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=item->>'run_id'
   AND round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='failed'
   AND terminal_cause IN('model_timeout','invalid_output','budget_exhausted','credential_error','model_error');
  IF item->'per_icp_score'IS DISTINCT FROM '0'::JSONB OR NOT public.lab_arena__qualification_doc_valid(item->'qualification_doc')
   OR item#>'{qualification_doc,companies}'IS DISTINCT FROM '[]'::JSONB
   THEN RAISE EXCEPTION 'Oct22 noncanonical fresh failure zero';END IF;
 END LOOP;
 -- All product guards remain enabled. Temporarily clear our hold only within
 -- this advisory-locked transaction so ordinary score/forward-transition guards
 -- run normally; no concurrent claim can observe this intermediate state.
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE WHERE singleton;
 intermediate:=r;intermediate.status:='stage2_closed';intermediate.status_generation:=r.status_generation+1;
 intermediate.stage2_scoring_plan_doc:=plan;
 SELECT greatest(intermediate.stage_generation,coalesce(max(generation_floor),0)+1)INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'oct22-dependency-545-plan-v1',payload_hash,actor,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(intermediate)-'updated_at','{}',jsonb_build_object('operation','oct22_dependency_actual_outcome_plan_v1','payload_sha256',payload_hash));
 UPDATE public.lab_arena_rounds SET status=intermediate.status,status_generation=intermediate.status_generation,stage2_scoring_plan_doc=plan WHERE round_id=rid;
 IF jsonb_array_length(zeros)>0 THEN ordinary:=public.lab_arena_record_run_scores(rid,2::SMALLINT,zeros);
  IF ordinary->>'status'IS DISTINCT FROM'ok'THEN RAISE EXCEPTION 'Oct22 ordinary fresh failure materialization refused';END IF;END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(items)LOOP
  SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=item->>'scored_run_id';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
  VALUES(source.assignment_id||':score:1',source.assignment_id||':score',rid,sid,source.miner_hotkey,2,source.icp_position,1,'pending',intermediate.stage_generation,'score',source.run_id,item->'company_judgment_refs');
 END LOOP;
 SELECT * INTO STRICT intermediate FROM public.lab_arena_rounds WHERE round_id=rid;
 n:=intermediate;n.status:='scored';n.status_generation:=intermediate.status_generation+1;
 SELECT greatest(n.stage_generation,coalesce(max(generation_floor),0)+1)INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'oct22-dependency-545-final-v1',payload_hash,actor,txid_current(),pg_backend_pid(),to_jsonb(intermediate)-'updated_at',to_jsonb(n)-'updated_at','{}',jsonb_build_object('operation','oct22_dependency_judging_continue_v1','payload_sha256',payload_hash,'infrastructure_failures_require_truthful_publication_review',jsonb_array_length(expected_incomplete)>jsonb_array_length(coalesce(h.old_round_doc#>'{stage2_scoring_plan_doc,incomplete_rows}','[]'))));
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation WHERE round_id=rid;
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=(h.recovery_doc#>>'{control_before,operator_paused}')::BOOLEAN,
  actor_ref=h.recovery_doc#>>'{control_before,actor_ref}',pause_reason=h.recovery_doc#>>'{control_before,pause_reason}',updated_at=clock_timestamp()WHERE singleton;
 -- Fresh accepted outputs still require accepted judges and normal score
 -- recovery. Missing judgments cannot be replaced by any historical zero.
END;
$finalize545$;
COMMIT;
