-- Finalize only the ninth exact receipt-bound review; admit only a real settled pass.
-- Preserve all twenty-nine other rows, original eighteen participants, costs and windows.
-- Preserve current operator state and every existing baseline run. Native scheduling
-- may plan this miner only after a settled pass is frozen before miner planning.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
DO $admit638$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds;
 h public.lab_arena_scoring_supersessions; done public.lab_arena_scoring_supersessions;
 s public.lab_arena_submissions; c public.lab_arena_restart_claim_control;
 reservation public.lab_arena_ledger; terminal public.lab_arena_ledger;
 sid TEXT; original JSONB; identity TEXT; proof JSONB:='{}'; admitted JSONB:='[]'; excluded JSONB:='[]';
 generation BIGINT; deadline TIMESTAMPTZ; review_cost BIGINT; runs_before JSONB; control_before JSONB;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-06' FOR UPDATE;
 SELECT * INTO done FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov06-admit-ninth-reviewed638-v1';
 IF FOUND THEN
  IF done.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_admit_late_reviewed636_v1'
   OR done.new_round_doc->'participants' IS DISTINCT FROM r.participants
   OR done.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov06 ninth638 replay differs'; END IF;
  RETURN;
 END IF;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov06-ninth-review637-v1'
  AND recovery_doc->>'operation'='nov06_late_review635_v1';
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 deadline:=LEAST(h.created_at+INTERVAL '30 minutes',(h.old_round_doc#>>'{configuration_doc,schedule,stage_1_close}')::TIMESTAMPTZ);
 control_before:=to_jsonb(c);
 SELECT coalesce(jsonb_object_agg(run_id,to_jsonb(a)),'{}') INTO runs_before FROM public.lab_arena_runs a WHERE round_id=r.round_id;
 IF current_setting('session_replication_role')<>'origin' OR c.operator_paused
  OR c.guard_commitment<>'' OR c.owner_commitment<>'' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit<>'' OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::JSONB
  OR r.status NOT IN('stage1','stage1_closed','stage1_scoring','stage1_judged')
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR r.stage2_scoring_plan_doc IS NOT NULL OR r.finalists IS NOT NULL
  OR NOT (r.stage1_scoring_plan_doc IS NULL OR COALESCE((jsonb_typeof(r.stage1_scoring_plan_doc)='object'
     AND jsonb_typeof(r.stage1_scoring_plan_doc->'work_items')='array' AND jsonb_typeof(r.stage1_scoring_plan_doc->'zero_rows')='array'
     AND NOT EXISTS(SELECT 1 FROM jsonb_array_elements((CASE WHEN jsonb_typeof(r.stage1_scoring_plan_doc->'work_items')='array' THEN r.stage1_scoring_plan_doc->'work_items' ELSE '[]'::JSONB END)||(CASE WHEN jsonb_typeof(r.stage1_scoring_plan_doc->'zero_rows')='array' THEN r.stage1_scoring_plan_doc->'zero_rows' ELSE '[]'::JSONB END)) plan_item(value)
      WHERE plan_item.value->>'submission_id' IS DISTINCT FROM 'baseline-2026-11-06')),FALSE))
  OR (to_jsonb(r)-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[]) IS DISTINCT FROM (h.old_round_doc-ARRAY['status','status_generation','stage_generation','stage1_scoring_plan_doc','updated_at']::TEXT[])
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=r.round_id
   AND (submission_id IS DISTINCT FROM 'baseline-2026-11-06' OR stage IS DISTINCT FROM 1))
  OR EXISTS(SELECT 1 FROM jsonb_each_text(h.recovery_doc->'preserved') keep
   LEFT JOIN public.lab_arena_submissions p ON p.submission_id=keep.key
   WHERE encode(extensions.digest(to_jsonb(p)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value)
  OR (SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'originals'))<>1 THEN
  RAISE EXCEPTION 'Nov06 ninth638 frozen round, preserved entrants, clock or hold differs'; END IF;
 n:=r;
 FOR sid,original IN SELECT key,value FROM jsonb_each(h.recovery_doc->'originals') LOOP
  IF sid IS DISTINCT FROM 'sub-43a929eb0e2e487f68014f48e7d80e6b' THEN RAISE EXCEPTION 'Nov06 ninth638 exact identity differs'; END IF;
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid FOR UPDATE;
  IF public.lab_arena__nov06_late_review635_v1(r,s) IS DISTINCT FROM deadline
   OR s.code_review_status NOT IN('passed','rejected','error') OR s.code_review_expires_at IS NOT NULL
   OR s.code_review_attempts NOT IN(3,4)
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(n.participants) p WHERE p->>'submission_id'=sid) THEN
   RAISE EXCEPTION 'Nov06 ninth638 original or final review differs: %',sid; END IF;
  IF s.code_review_attempts=3 THEN
   IF clock_timestamp()<deadline OR s.code_review_status<>'error' THEN
    RAISE EXCEPTION 'Nov06 ninth638 unspent review window remains: %',sid; END IF;
   proof:=proof||jsonb_build_object(sid,jsonb_build_object('submission',to_jsonb(s),'outcome','window_expired_without_fourth'));
  ELSE
   identity:='sha256:'||encode(extensions.digest(convert_to('lab_arena.code_review:'||sid||':4','UTF8'),'sha256'),'hex');
   terminal:=public.lab_arena__ledger_head(identity);
   SELECT * INTO STRICT reservation FROM public.lab_arena_ledger WHERE call_identity=identity AND entry_kind='reservation';
   IF terminal.entry_kind NOT IN('settlement','uncertain') OR terminal.submission_id IS DISTINCT FROM sid
    OR terminal.round_id IS DISTINCT FROM r.round_id OR terminal.operation_id IS DISTINCT FROM 'openrouter.code_review'
    OR (terminal.entry_doc->>'review_status' IS DISTINCT FROM s.code_review_status
      AND NOT (s.code_review_status='error' AND terminal.entry_kind='uncertain'
       AND terminal.entry_doc->>'reason'='code_review_claim_expired'
       AND terminal.entry_doc->>'error_code'='code_review_transport_error'
       AND terminal.amount_microusd=reservation.amount_microusd))
    OR reservation.submission_id IS DISTINCT FROM sid OR reservation.round_id IS DISTINCT FROM r.round_id
    OR reservation.operation_id IS DISTINCT FROM 'openrouter.code_review'
    OR reservation.entry_doc->'attempt' IS DISTINCT FROM '4'::JSONB
    OR reservation.entry_doc->>'review_model' IS DISTINCT FROM 'openai/gpt-6.1-sol'
    OR reservation.created_at<h.created_at OR reservation.created_at>=deadline
    OR s.code_review_doc->>'model' IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
    RAISE EXCEPTION 'Nov06 ninth638 fourth claim or terminal differs: %',sid; END IF;
   proof:=proof||jsonb_build_object(sid,jsonb_build_object('submission',to_jsonb(s),
    'reservation',to_jsonb(reservation),'terminal',to_jsonb(terminal)));
  END IF;
  IF s.code_review_status='passed' THEN
   SELECT sum((value->>'amount_microusd')::BIGINT) INTO review_cost
    FROM jsonb_array_elements(original->'ledger') WHERE value->>'entry_kind'='settlement';
   IF clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_1_close}')::TIMESTAMPTZ
    OR s.code_review_attempts<>4 OR terminal.entry_kind IS DISTINCT FROM 'settlement'
    OR terminal.created_at>=deadline
    OR s.code_review_doc->'passed' IS DISTINCT FROM 'true'::JSONB
    OR s.code_review_doc->>'verdict' IS DISTINCT FROM 'pass'
    OR s.code_review_doc->'categories' IS DISTINCT FROM '[]'::JSONB
    OR s.code_review_doc->>'cost_status' IS DISTINCT FROM 'settled'
    OR (s.code_review_doc->>'cost_microusd')::BIGINT IS DISTINCT FROM terminal.amount_microusd
    OR (s.code_review_doc->>'review_cost_microusd')::BIGINT IS DISTINCT FROM review_cost+terminal.amount_microusd
    OR s.code_review_doc#>>'{research_control,classification}' IS DISTINCT FROM 'model_directed'
    OR s.code_review_doc#>>'{research_control,policy_id}' IS DISTINCT FROM 'model_directed_research_v1'
    OR s.code_review_doc#>>'{research_control,reviewer_version}' IS DISTINCT FROM 'full_source_reviewer_research_v1'
    OR s.code_review_doc#>>'{research_control,policy_hash}' IS DISTINCT FROM 'sha256:1d30b4c2d838e84692b3cd9aa0fe0da3ff908f37b3836863b3937a3d93c9a73c'
    OR s.code_review_doc#>>'{research_control,source_archive_sha256}' IS DISTINCT FROM s.source_archive_sha256
    OR s.code_review_doc#>>'{research_control,source_manifest_sha256}' IS DISTINCT FROM original->>'manifest_sha256'
    OR s.code_review_doc#>'{research_control,evidence_claims}' IS DISTINCT FROM '["finish_continue","observation_to_action"]'::JSONB
    OR jsonb_typeof(s.code_review_doc#>'{research_control,evidence_count}') IS DISTINCT FROM 'number'
    OR COALESCE(s.code_review_doc#>>'{research_control,evidence_count}','') !~ '^([2-9]|1[0-6])$' THEN
    RAISE EXCEPTION 'Nov06 ninth638 passing full-source proof differs: %',sid; END IF;
   n.participants:=n.participants||jsonb_build_array(jsonb_build_object('submission_id',sid,
    'miner_hotkey',s.miner_hotkey,'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'is_king',FALSE));
   admitted:=admitted||jsonb_build_array(sid);
  ELSE
   excluded:=excluded||jsonb_build_array(sid);
  END IF;
 END LOOP;
 IF jsonb_array_length(n.participants)-1>(r.configuration_doc->>'max_challengers')::INTEGER
  OR (SELECT count(*) FROM jsonb_array_elements(n.participants) p WHERE p->'is_king'='true'::JSONB)<>1 THEN
  RAISE EXCEPTION 'Nov06 ninth638 participant capacity or baseline differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,generation,'nov06-admit-ninth-reviewed638-v1','sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),
  'operator:nov06-admit-ninth638',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',jsonb_build_object(
   'operation','nov06_admit_late_reviewed636_v1','baseline_runs_at_admission',runs_before,'control_at_admission',control_before,'admitted',admitted,'excluded',excluded,'reviews',proof));
 FOR sid IN SELECT jsonb_array_elements_text(admitted) LOOP
  UPDATE public.lab_arena_submissions SET status='frozen',frozen_at=clock_timestamp(),rejection_rule=NULL
   WHERE submission_id=sid AND status='accepted' AND code_review_status='passed';
  IF NOT FOUND THEN RAISE EXCEPTION 'Nov06 ninth638 passing freeze differs'; END IF;
 END LOOP;
 FOR sid IN SELECT jsonb_array_elements_text(excluded) LOOP
  UPDATE public.lab_arena_submissions SET status='rejected',rejection_rule=CASE
   WHEN code_review_status='rejected' THEN 'code_review_rejected' ELSE 'code_review_incomplete' END
   WHERE submission_id=sid AND status='accepted' AND code_review_status IN('rejected','error');
  IF NOT FOUND THEN RAISE EXCEPTION 'Nov06 ninth638 exclusion differs'; END IF;
 END LOOP;
 UPDATE public.lab_arena_rounds SET participants=n.participants WHERE round_id=r.round_id;
 IF (SELECT coalesce(jsonb_object_agg(run_id,to_jsonb(a)),'{}') FROM public.lab_arena_runs a WHERE round_id=r.round_id) IS DISTINCT FROM runs_before
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM control_before THEN
  RAISE EXCEPTION 'Nov06 ninth638 changed existing baseline work or control'; END IF;
END;
$admit638$;
COMMIT;
