-- Exact operator recovery of three unfinished executions. The external
-- validator session replacement corroborates missing outputs; it is not an
-- authenticated host-error credit. Keep both old attempts, all paid calls,
-- the original $6 successful-cost cap and original two-attempt call allowance.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena609_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight609$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_unfinished_execute609_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'e7a403c088e4c0ef3a79e3988c0f261ae113e6ed137468711c9c416c2a752c2d'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 unfinished execution helper differs'; END IF;
END;
$helper_preflight609$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov03_unfinished_execute609_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $scope609$
DECLARE h public.lab_arena_scoring_supersessions; s public.lab_arena_submissions;
 parent public.lab_arena_runs; first_attempt public.lab_arena_runs; proof JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id
  OR p_run.submission_id IS DISTINCT FROM 'sub-f013f522a7c764e18df792e094e7b820'
  OR p_run.kind IS DISTINCT FROM 'execute' OR p_run.stage IS DISTINCT FROM 2
  OR p_run.icp_position NOT IN(4,5,6) OR p_run.attempt IS DISTINCT FROM 3
  OR p_run.assignment_id IS DISTINCT FROM p_run.round_id||':'||p_run.submission_id||':2:'||p_run.icp_position::TEXT
  OR p_run.run_id IS DISTINCT FROM p_run.assignment_id||':3' THEN RETURN FALSE; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-three-unfinished-executions609-v1'
  AND recovery_doc->>'operation'='nov03_three_unfinished_executions609_v1'
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc;
 IF NOT FOUND OR p_run.stage_generation IS DISTINCT FROM (h.new_round_doc->>'stage_generation')::BIGINT
  THEN RETURN FALSE; END IF;
 SELECT * INTO s FROM public.lab_arena_submissions WHERE submission_id=p_run.submission_id;
 IF NOT FOUND OR s.is_king OR s.status IS DISTINCT FROM 'frozen'
  OR p_run.miner_hotkey IS DISTINCT FROM s.miner_hotkey
  OR h.recovery_doc->'source_identity' IS DISTINCT FROM jsonb_build_object(
   'round_id',s.round_id,'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
   'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256)
  THEN RETURN FALSE; END IF;
 SELECT * INTO first_attempt FROM public.lab_arena_runs WHERE run_id=p_run.assignment_id||':1';
 SELECT * INTO parent FROM public.lab_arena_runs WHERE run_id=p_run.assignment_id||':2';
 proof:=h.recovery_doc->'original_run_hashes';
 RETURN coalesce(proof->>first_attempt.run_id=encode(extensions.digest(to_jsonb(first_attempt)::TEXT,'sha256'),'hex')
  AND proof->>parent.run_id=encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
  AND (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
   IS NOT DISTINCT FROM (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position),FALSE);
END;
$scope609$;
ALTER FUNCTION public.lab_arena__nov03_unfinished_execute609_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_unfinished_execute609_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify609$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_unfinished_execute609_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'e7a403c088e4c0ef3a79e3988c0f261ae113e6ed137468711c9c416c2a752c2d'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 unfinished execution helper differs'; END IF;
END;
$helper_verify609$;

-- Preserve every existing cost, per-run quota, stage quota and lease gate.
-- Only these counted operator thirds also receive the existing aggregate
-- assignment quota fence; they receive no extra call allowance.
DO $patch609$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','3ebdca78d95dfb198a0e918dd1bbe65b9356fb8afbc9aaa42814a3e43ee85999','4f78a1c47e9afa29f4162083eec62728aaaa3271bfd2da91e11741b624a5b32c','v','plpgsql',$old609$    AND public.lab_arena_execution_attempt_count(v_run)<v_run.attempt$old609$,$new609$    AND (public.lab_arena_execution_attempt_count(v_run)<v_run.attempt
      OR v_run.run_id IN('arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:3','arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:3','arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:3'))$new609$),
 ('public.lab_arena_scoring_policy_history_v1(text)','5d972b7a1ef6c18fbf814ddc746a3b7fbc114b9f1fd76a1fa0c200fc849e5aeb','454c53ec68adaed3de3c1784f761706593832e489bbf5446a338f613b7c58fff','s','sql',$old609$'nov03_baseline8_corrected_judge607_v1') THEN$old609$,$new609$'nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1') THEN$new609$)
 ) AS patches(signature,preimage,postimage,volatility,language,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],target.language) THEN
   RAISE EXCEPTION 'Nov03 unfinished execution security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>length(target.old_text) THEN
   RAISE EXCEPTION 'Nov03 unfinished execution preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 unfinished execution postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 unfinished execution readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch609$;
-- Exact three-assignment operator append. No old row, source, ledger,
-- configuration, participant, score, or deadline is updated.
DO $append609$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 sid CONSTANT TEXT:='sub-f013f522a7c764e18df792e094e7b820';
 request CONSTANT TEXT:='nov03-three-unfinished-executions609-v1';
 manifest CONSTANT JSONB:=$manifest609${"original_run_hashes":{"arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:1":"d72c986ce4ed06a3d0eaf7a246d397fd637c6f85ad3c9cdca3545c6dcb370df4","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:4:2":"2b65a8e071ea44b5733ea9ab87ba91c65bfa42ac2465a51ead24473b013563b8","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:1":"54745206a409c23b2f1c359e487bc283d299538e2fa7ca10b271b076780257a9","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:5:2":"3dae7cc8b28c7dce5dcd9bf8f7950f255ccf14cf5110c9fc80beecbd6d13738c","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:1":"bb6c5d6f2eb8561a617522a0672c8f3751be7d3023000006f99b62a4e674035f","arena-2026-11-03:sub-f013f522a7c764e18df792e094e7b820:2:6:2":"7726f47dcc0e1cc7e55bf3933e20623ee92d4e20db654c2cd631a910e7dfb2c0"},"source_identity":{"miner_hotkey":"5DZrBs1WQzpopHuMjtP74vbYDCKQhDUVtLWLAuYAyGnM3AFh","round_id":"arena-2026-11-03","source_archive_sha256":"sha256:0e7a88a9f901be0ab14a4d6566c12b2f4c6507a6b2e253ef875af42ad0185c16","source_ref":"arena/arena-2026-11-03/sources/sub-f013f522a7c764e18df792e094e7b820.tar.gz","source_size_bytes":460081,"submission_id":"sub-f013f522a7c764e18df792e094e7b820"}}$manifest609$::JSONB;
 r public.lab_arena_rounds; s public.lab_arena_submissions; old_run public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; third public.lab_arena_runs;
 revision BIGINT; pos INTEGER; n INTEGER; assignment TEXT; old_hashes JSONB; cost JSONB;
 costs JSONB:='{}'; quotas JSONB:='{}'; provider RECORD; actual_calls BIGINT; available BOOLEAN;
 request_hash TEXT; round_before JSONB; control_before JSONB;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT to_jsonb(c) INTO STRICT control_before FROM public.lab_arena_restart_claim_control c WHERE singleton FOR SHARE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 round_before:=to_jsonb(r);
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_three_unfinished_executions609_v1'
   OR h.recovery_doc->'original_run_hashes' IS DISTINCT FROM manifest->'original_run_hashes'
   OR h.recovery_doc->'source_identity' IS DISTINCT FROM manifest->'source_identity'
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov03 unfinished execution replay differs'; END IF;
  FOR pos IN 4..6 LOOP
   assignment:=rid||':'||sid||':2:'||pos::TEXT;
   SELECT * INTO STRICT third FROM public.lab_arena_runs WHERE run_id=assignment||':3';
   IF third.assignment_id IS DISTINCT FROM assignment OR third.attempt IS DISTINCT FROM 3
    OR third.kind IS DISTINCT FROM 'execute' OR third.round_id IS DISTINCT FROM rid
    OR third.submission_id IS DISTINCT FROM sid OR third.icp_position IS DISTINCT FROM pos
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>3)
    OR NOT public.lab_arena__nov03_unfinished_execute609_v1(r,third) THEN
    RAISE EXCEPTION 'Nov03 unfinished execution replay row differs'; END IF;
  END LOOP;
  RETURN; -- Never requeue or modify completed third attempts.
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR r.status IS DISTINCT FROM 'stage2' OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR r.configuration_doc->>'sourcing_cost_eligibility_policy' IS DISTINCT FROM 'successful_calls_flat_per_icp_v1'
  OR r.configuration_doc->>'reasoning_capture_policy' IS DISTINCT FROM 'full_generated_thinking_v1'
  OR r.configuration_doc->'max_attempts_per_assignment' IS DISTINCT FROM '2'::JSONB
  OR r.configuration_doc->'execution_icp_cap_microusd' IS DISTINCT FROM '6000000'::JSONB
  OR r.configuration_doc->'cost_per_company_microusd' IS DISTINCT FROM '6000000'::JSONB
  OR r.configuration_doc#>>'{schedule,stage_2_close}' IS DISTINCT FROM '2026-11-03T14:00:02Z'
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions verifier WHERE verifier.round_id=rid
   AND verifier.request_id='nov03-forward-only-verifier604-v1'
   AND verifier.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
   AND verifier.new_round_doc->'configuration_doc'=r.configuration_doc) THEN
  RAISE EXCEPTION 'Nov03 unfinished execution frozen round or deadline differs'; END IF;
 -- Lock only the six old target runs, in a stable order, then the source.
 PERFORM 1 FROM public.lab_arena_runs WHERE run_id IN(SELECT jsonb_object_keys(manifest->'original_run_hashes')) ORDER BY run_id FOR UPDATE;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid FOR SHARE;
 IF s.round_id IS DISTINCT FROM rid OR s.is_king OR s.status IS DISTINCT FROM 'frozen'
  OR s.code_review_status IS DISTINCT FROM 'passed'
  OR manifest->'source_identity' IS DISTINCT FROM jsonb_build_object('round_id',s.round_id,
   'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,'source_ref',s.source_ref,
   'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256)
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants) p WHERE p->>'submission_id'=sid AND p->>'miner_hotkey'=s.miner_hotkey)
  THEN RAISE EXCEPTION 'Nov03 unfinished execution source identity differs'; END IF;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO old_hashes
 FROM public.lab_arena_runs a WHERE a.run_id IN(SELECT jsonb_object_keys(manifest->'original_run_hashes'));
 IF old_hashes IS DISTINCT FROM manifest->'original_run_hashes'
  OR (SELECT count(*) FROM jsonb_object_keys(manifest->'original_run_hashes'))<>6 THEN
  RAISE EXCEPTION 'Nov03 unfinished execution original six rows differ'; END IF;
 FOR pos IN 4..6 LOOP
  assignment:=rid||':'||sid||':2:'||pos::TEXT;
  IF (SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=assignment)<>2
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid
    AND kind='execute' AND icp_position=pos AND (status IN('pending','leased','submitted','accepted') OR output_ref IS NOT NULL)) THEN
   RAISE EXCEPTION 'Nov03 unfinished execution assignment already has work or output'; END IF;
  FOR n IN 1..2 LOOP
   SELECT * INTO STRICT old_run FROM public.lab_arena_runs WHERE run_id=assignment||':'||n::TEXT;
   IF old_run.kind IS DISTINCT FROM 'execute' OR old_run.stage IS DISTINCT FROM 2
    OR old_run.submission_id IS DISTINCT FROM sid OR old_run.miner_hotkey IS DISTINCT FROM s.miner_hotkey
    OR old_run.icp_position IS DISTINCT FROM pos OR old_run.attempt IS DISTINCT FROM n
    OR old_run.status IS DISTINCT FROM 'failed' OR old_run.terminal_cause IS DISTINCT FROM 'lease_expired'
    OR old_run.result_doc IS NOT NULL OR old_run.output_ref IS NOT NULL
    OR old_run.per_icp_score IS NOT NULL OR old_run.qualification_doc IS NOT NULL
    OR old_run.champion_restart_required OR old_run.stage_generation IS DISTINCT FROM r.stage_generation
    OR public.lab_arena_execution_attempt_count(old_run) IS DISTINCT FROM n THEN
    RAISE EXCEPTION 'Nov03 unfinished execution terminal scope differs'; END IF;
  END LOOP;
  cost:=public.lab_arena__successful_icp_cost_state(rid,sid,pos);
  IF (cost->>'inflight_calls')::BIGINT<>0 OR (cost->>'success_unresolved_calls')::BIGINT<>0
   OR coalesce(cost->>'competition_settled_microusd',cost->>'settled_microusd')::BIGINT>=6000000 THEN
   RAISE EXCEPTION 'Nov03 unfinished execution cost headroom differs'; END IF;
  costs:=costs||jsonb_build_object(pos::TEXT,cost);available:=FALSE;
  FOR provider IN SELECT key,value FROM jsonb_each_text(r.configuration_doc->'call_quotas') LOOP
   actual_calls:=public.lab_arena__execute_assignment_calls_v1(assignment,provider.key);
   quotas:=quotas||jsonb_build_object(assignment||':'||provider.key,actual_calls);
   IF provider.value::BIGINT=0 AND provider.key='deepline'
     AND r.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1' THEN
    available:=TRUE;
   ELSIF provider.value::BIGINT>0 AND actual_calls<provider.value::BIGINT*2
    AND public.lab_arena__submission_stage_consumed(sid,2::SMALLINT,provider.key,'execute')<
     provider.value::BIGINT*2*(r.configuration_doc->>'stage_2_icp_count')::BIGINT THEN
    available:=TRUE;
   END IF;
  END LOOP;
  IF NOT available THEN RAISE EXCEPTION 'Nov03 unfinished execution provider quota exhausted'; END IF;
 END LOOP;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,request,request_hash,'nov03-three-unfinished-executions609',txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',jsonb_build_object(
   'operation','nov03_three_unfinished_executions609_v1','scoring_generation_floor',0,
   'original_run_hashes',manifest->'original_run_hashes','source_identity',manifest->'source_identity',
   'original_icp_costs',costs,'original_assignment_provider_counts',quotas,
   'reason','Three unfinished assignments have expired executions and no retained outputs after external validator session replacement. This is an explicit operator exception; diagnostics are corroborative, not host-credit authority. Keep original costs and call quotas, append only attempt three, and do not create attempt four.'));
 FOR pos IN 4..6 LOOP
  assignment:=rid||':'||sid||':2:'||pos::TEXT;
  SELECT * INTO STRICT old_run FROM public.lab_arena_runs WHERE run_id=assignment||':2';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
   stage,icp_position,attempt,status,lease_generation,stage_generation,kind,previous_runner_hotkey)
  VALUES(assignment||':3',assignment,rid,sid,s.miner_hotkey,2,pos,3,'pending',old_run.lease_generation,
   r.stage_generation,'execute',old_run.runner_hotkey);
  SELECT * INTO STRICT third FROM public.lab_arena_runs WHERE run_id=assignment||':3';
  IF NOT public.lab_arena__nov03_unfinished_execute609_v1(r,third)
   OR public.lab_arena_execution_attempt_count(third) IS DISTINCT FROM 3 THEN
   RAISE EXCEPTION 'Nov03 unfinished execution new counted attempt differs'; END IF;
 END LOOP;
 IF (SELECT to_jsonb(actual) FROM public.lab_arena_rounds actual WHERE round_id=rid) IS DISTINCT FROM round_before
  OR (SELECT to_jsonb(actual) FROM public.lab_arena_restart_claim_control actual WHERE singleton) IS DISTINCT FROM control_before
  OR (SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
   FROM public.lab_arena_runs a WHERE a.run_id IN(SELECT jsonb_object_keys(manifest->'original_run_hashes'))) IS DISTINCT FROM old_hashes THEN
  RAISE EXCEPTION 'Nov03 unfinished execution changed preserved round, control or old attempts'; END IF;
END;
$append609$;

DO $restore609$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena609_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena609_schema_acl) THEN RAISE EXCEPTION 'Nov03 unfinished execution schema ACL differs'; END IF;
END;
$restore609$;
NOTIFY pgrst,'reload schema';
COMMIT;
