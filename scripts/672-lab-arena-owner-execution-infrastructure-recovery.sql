-- One explicit owner-authorized execution successor. This is not host credit.
-- Preserve physical attempts, frozen inputs, accepted work, funding and bills.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena672_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $security672$
DECLARE wanted RECORD; actual RECORD;
BEGIN
 FOR wanted IN SELECT * FROM (VALUES
 ('public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs)','4dee08783888087e296654ded754ad15813eb9db1fe11755dd077fa6ca350f92','s','sql'),
 ('public.lab_arena__operator_execute_recovery_valid_v1(public.lab_arena_rounds,public.lab_arena_runs)','9d0ab4381036326f06060889c2b02573c8c81d5891b621319ebeb4c130097081','s','plpgsql'),
 ('public.lab_arena_grant_execution_infrastructure_recovery_v1(text,text,text,text,text,boolean)','532d14d6356b218a6366a656e5a4a46563d9cb179c4af147d6aaa924ff1d32f2','v','plpgsql')
 ) expected(signature,source_hash,volatility,language) LOOP
  SELECT pg_get_userbyid(p.proowner) AS owner,p.proacl::TEXT AS acl,p.prosecdef,p.provolatile,
   p.proconfig,l.lanname,encode(extensions.digest(p.prosrc,'sha256'),'hex') AS source_hash
  INTO actual FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(wanted.signature);
  IF NOT FOUND THEN CONTINUE; END IF;
  IF actual.owner<>'lab_arena_owner' OR actual.acl IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR actual.prosecdef IS DISTINCT FROM TRUE OR actual.provolatile::TEXT<>wanted.volatility
   OR actual.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR actual.lanname<>wanted.language OR actual.source_hash<>wanted.source_hash THEN
   RAISE EXCEPTION 'Execution recovery672 helper security or source differs: %',wanted.signature; END IF;
 END LOOP;
END;
$security672$;

-- Registration is deliberately separate from validity. A damaged proof must
-- still fence the registered child from fresh liability and automatic retries.
CREATE OR REPLACE FUNCTION public.lab_arena__operator_execute_recovery_child_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $child672$
 SELECT coalesce(p_run.kind='execute' AND EXISTS(
  SELECT 1 FROM public.lab_arena_scoring_supersessions h
  WHERE h.round_id=p_run.round_id AND h.recovery_doc->>'operation'='round_operational_update_v1'
   AND h.recovery_doc->>'recovery_kind'='operator_execution_infrastructure_recovery_v1'
   AND h.recovery_doc#>>'{child_identity,run_id}'=p_run.run_id),FALSE);
$child672$;
ALTER FUNCTION public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__operator_execute_recovery_child_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__operator_execute_recovery_valid_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $valid672$
DECLARE h public.lab_arena_scoring_supersessions; s public.lab_arena_submissions; hashes JSONB;
BEGIN
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_run.round_id AND recovery_doc->>'operation'='round_operational_update_v1'
  AND recovery_doc->>'recovery_kind'='operator_execution_infrastructure_recovery_v1'
  AND recovery_doc#>>'{child_identity,run_id}'=p_run.run_id;
 IF NOT FOUND OR p_run.kind<>'execute' OR p_run.round_id<>p_round.round_id
  OR h.recovery_doc->'child_identity' IS DISTINCT FROM jsonb_build_object(
   'run_id',p_run.run_id,'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,
   'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,'stage',p_run.stage,
   'icp_position',p_run.icp_position,'attempt',p_run.attempt,'stage_generation',p_run.stage_generation,
   'scoring_generation_floor',p_run.scoring_generation_floor)
  OR ((h.new_round_doc->'configuration_doc')-'schedule') IS DISTINCT FROM (p_round.configuration_doc-'schedule')
  OR p_run.scoring_generation_floor<>p_round.scoring_generation_floor THEN RETURN FALSE; END IF;
 SELECT * INTO s FROM public.lab_arena_submissions WHERE submission_id=p_run.submission_id;
 IF NOT FOUND OR h.recovery_doc->'source_identity' IS DISTINCT FROM jsonb_build_object(
  'round_id',s.round_id,'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
  'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256)
  OR s.status<>'frozen' OR s.code_review_status<>'passed' OR s.is_king THEN RETURN FALSE; END IF;
 SELECT coalesce(jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}')
 INTO hashes FROM public.lab_arena_runs a
 WHERE a.assignment_id=p_run.assignment_id AND a.attempt<p_run.attempt;
 RETURN coalesce(hashes=h.recovery_doc->'original_run_hashes',FALSE);
END;
$valid672$;
ALTER FUNCTION public.lab_arena__operator_execute_recovery_valid_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__operator_execute_recovery_valid_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_grant_execution_infrastructure_recovery_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_request_id TEXT,p_actor_ref TEXT,
 p_correction_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $grant672$
DECLARE parent public.lab_arena_runs; child public.lab_arena_runs; r public.lab_arena_rounds;
 s public.lab_arena_submissions; prior public.lab_arena_scoring_supersessions; control public.lab_arena_restart_claim_control;
 parent_hash TEXT; hashes JSONB; source JSONB; proof JSONB; response JSONB; cost JSONB;
 provider RECORD; available BOOLEAN:=FALSE; revision BIGINT; quota_counts JSONB:='{}';
BEGIN
 IF p_parent_run_id IS NULL OR coalesce(p_parent_sha256,'') !~ '^sha256:[0-9a-f]{64}$'
  OR length(coalesce(p_request_id,'')) NOT BETWEEN 1 AND 160
  OR length(coalesce(p_actor_ref,'')) NOT BETWEEN 1 AND 200
  OR length(coalesce(p_correction_ref,'')) NOT BETWEEN 1 AND 500 OR p_dry_run IS NULL THEN
  RAISE EXCEPTION 'execution recovery request invalid'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=parent.round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_execution_infrastructure_recovery_v1'
   OR prior.recovery_doc->>'parent_run_id' IS DISTINCT FROM p_parent_run_id
   OR prior.recovery_doc->>'parent_sha256' IS DISTINCT FROM p_parent_sha256
   OR prior.actor_ref IS DISTINCT FROM p_actor_ref OR prior.recovery_doc->>'correction_ref' IS DISTINCT FROM p_correction_ref THEN
   RAISE EXCEPTION 'execution recovery replay differs'; END IF;
  SELECT * INTO STRICT child FROM public.lab_arena_runs WHERE run_id=prior.recovery_doc#>>'{child_identity,run_id}';
  IF NOT public.lab_arena__operator_execute_recovery_valid_v1(r,child) THEN
   RAISE EXCEPTION 'execution recovery replay proof differs'; END IF;
  RETURN jsonb_build_object('status','granted','idempotent',TRUE,'child_identity',prior.recovery_doc->'child_identity');
 END IF;
 SELECT * INTO STRICT control FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF current_setting('session_replication_role')<>'origin' OR control.operator_paused
  OR coalesce(control.guard_commitment,'')<>'' OR coalesce(control.owner_commitment,'')<>''
  OR control.guard_expires_at IS NOT NULL OR coalesce(control.candidate_commit,'')<>''
  OR coalesce(control.restart_scope,'')<>'' OR coalesce(control.restart_phase,'')<>''
  OR coalesce(control.captured_leases,'[]')<>'[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2' OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.effective_reward_epoch IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR (clock_timestamp()<(r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
   AND clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
   AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ) IS NOT TRUE THEN
  RAISE EXCEPTION 'execution recovery round, control or deadline differs'; END IF;
 -- Lock every historical sibling before binding its immutable row hash.
 PERFORM 1 FROM public.lab_arena_runs WHERE assignment_id=parent.assignment_id ORDER BY attempt FOR UPDATE;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id;
 parent_hash:='sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex');
 IF parent_hash IS DISTINCT FROM p_parent_sha256 OR parent.kind<>'execute' OR parent.stage<>2
  OR parent.status<>'failed' OR parent.terminal_cause<>'lease_expired' OR parent.attempt>=2147483647
  OR parent.result_doc IS NOT NULL OR parent.output_ref IS NOT NULL OR parent.per_icp_score IS NOT NULL
  OR parent.qualification_doc IS NOT NULL OR parent.champion_restart_required
  OR parent.stage_generation<>r.stage_generation OR parent.scoring_generation_floor<>r.scoring_generation_floor
  OR parent.lease_expires_at IS NULL OR parent.lease_expires_at>=clock_timestamp()
  OR parent.lease_generation<1 OR parent.runner_hotkey IS NULL OR parent.lease_token_hash IS NULL
  OR parent.claim_request_id IS NULL OR parent.claim_response->>'status' IS DISTINCT FROM 'leased'
  OR parent.claim_response->'lease_generation' IS DISTINCT FROM to_jsonb(parent.lease_generation)
  OR (public.lab_arena_execution_attempt_count(parent)>=(r.configuration_doc->>'max_attempts_per_assignment')::INTEGER
   AND (r.configuration_doc->>'max_attempts_per_assignment')::INTEGER>0) IS NOT TRUE
  OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id
   AND (a.attempt>parent.attempt OR a.status IN('pending','leased','submitted','accepted') OR a.output_ref IS NOT NULL))
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
   AND h.recovery_doc->>'recovery_kind'='operator_execution_infrastructure_recovery_v1'
   AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id) THEN
  RAISE EXCEPTION 'execution recovery expired latest parent or clear calls differs'; END IF;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=parent.submission_id FOR SHARE;
 IF s.round_id<>r.round_id OR s.miner_hotkey<>parent.miner_hotkey OR s.is_king
  OR s.status<>'frozen' OR s.code_review_status<>'passed' OR s.source_ref IS NULL
  OR s.source_archive_sha256 IS NULL OR s.source_size_bytes IS NULL
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p
   WHERE p->>'submission_id'=s.submission_id AND p->>'miner_hotkey'=s.miner_hotkey) THEN
  RAISE EXCEPTION 'execution recovery frozen source differs'; END IF;
 cost:=public.lab_arena__successful_icp_cost_state(r.round_id,s.submission_id,parent.icp_position);
 IF (cost->>'inflight_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
  OR (cost->>'success_unresolved_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
  OR (coalesce(cost->>'competition_settled_microusd',cost->>'settled_microusd')::BIGINT
   <(r.configuration_doc->>'execution_icp_cap_microusd')::BIGINT) IS NOT TRUE THEN
  RAISE EXCEPTION 'execution recovery unresolved cost or exhausted cap'; END IF;
 FOR provider IN SELECT key,value FROM jsonb_each_text(r.configuration_doc->'call_quotas') LOOP
  quota_counts:=quota_counts||jsonb_build_object(provider.key,
   public.lab_arena__execute_assignment_calls_v1(parent.assignment_id,provider.key));
  IF provider.value::BIGINT=0 AND provider.key='deepline'
   AND r.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1' THEN available:=TRUE;
  ELSIF provider.value::BIGINT>0
   AND public.lab_arena__execute_assignment_calls_v1(parent.assignment_id,provider.key)<
    provider.value::BIGINT*(r.configuration_doc->>'max_attempts_per_assignment')::BIGINT
   AND public.lab_arena__submission_stage_consumed(s.submission_id,parent.stage,provider.key,'execute')<
    provider.value::BIGINT*(r.configuration_doc->>'max_attempts_per_assignment')::BIGINT*
    (r.configuration_doc->>'stage_2_icp_count')::BIGINT THEN available:=TRUE; END IF;
 END LOOP;
 IF NOT available THEN RAISE EXCEPTION 'execution recovery original quota exhausted'; END IF;
 SELECT jsonb_object_agg(a.run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO hashes
 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id;
 source:=jsonb_build_object('round_id',s.round_id,'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
  'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256);
 child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT; child.attempt:=parent.attempt+1;
 proof:=jsonb_build_object('operation','round_operational_update_v1',
  'recovery_kind','operator_execution_infrastructure_recovery_v1','parent_run_id',parent.run_id,
  'parent_sha256',parent_hash,'correction_ref',p_correction_ref,'source_identity',source,
  'original_run_hashes',hashes,'original_cost',cost,'original_assignment_provider_counts',quota_counts,
  'child_identity',jsonb_build_object('run_id',child.run_id,'assignment_id',child.assignment_id,'round_id',child.round_id,
   'submission_id',child.submission_id,'miner_hotkey',child.miner_hotkey,'stage',child.stage,'icp_position',child.icp_position,
   'attempt',child.attempt,'stage_generation',child.stage_generation,'scoring_generation_floor',child.scoring_generation_floor));
 response:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'granted' END,
  'idempotent',FALSE,'child_identity',proof->'child_identity','original_cost',cost);
 IF p_dry_run THEN RETURN response; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),p_actor_ref,
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',proof);
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,lease_generation,stage_generation,scoring_generation_floor,kind,previous_runner_hotkey)
 VALUES(child.run_id,parent.assignment_id,parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,
  parent.icp_position,child.attempt,'pending',parent.lease_generation,r.stage_generation,r.scoring_generation_floor,'execute',parent.runner_hotkey);
 RETURN response;
END;
$grant672$;
ALTER FUNCTION public.lab_arena_grant_execution_infrastructure_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_grant_execution_infrastructure_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Exact shared-function patches are generated below and retain their metadata.
DO $patch672$
DECLARE target RECORD; patch RECORD; function_oid OID; d TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','5e13436bc113afb45e501a7f8e35f24309ee8a0e24af4779f0233a7bcd042529','145f89a0ad3076cfb75da6fd7e733325da242a5d6fb7a60c0de5d7c5c3ec6781'),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)','00562ddcea9dc11ecf70a116174f26adcf031cd301ea9126cc80a0b2123ab0a3','9abcdd8de7b5bd2cf220920297f119ea49a4d804a6dbbbfda81c503e9e84545e'),
 ('public.lab_arena_expire_leases(text)','b752c7a25489d2d438a2203bd8658f8b3c8b0c75e2325ee701dd8c817418129d','373bead617af1a509260cd896b40fec81c0f176ed1275b46e06985b03c42e098'),
 ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)','175b9bf90d9b1aa5436d78dca6f9a724445c9fd689650f76ef92e1d42f3303ed','ccf379077488b3416e0c94c317405f51bcaf99ea02723993eedfcf8a6ed2636c')
 ) expected(signature,preimage,postimage) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(
   pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO d,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid;
  IF d IS NULL OR identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,
   'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
   RAISE EXCEPTION 'Execution recovery672 function security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN
   RAISE EXCEPTION 'Execution recovery672 function source differs: %',target.signature; END IF;
  updated:=d;
  FOR patch IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',$old672$    AND (NOT public.lab_arena__operator_scorer_bug_child_v1(runs)
      OR public.lab_arena__operator_scorer_bug_claim_v1(v_round,runs))$old672$,$new672$    AND (NOT public.lab_arena__operator_scorer_bug_child_v1(runs)
      OR public.lab_arena__operator_scorer_bug_claim_v1(v_round,runs))
    AND (NOT public.lab_arena__operator_execute_recovery_child_v1(runs)
      OR public.lab_arena__operator_execute_recovery_valid_v1(v_round,runs))$new672$),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',$old672$     AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)$old672$,$new672$     AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)
     AND NOT public.lab_arena__operator_execute_recovery_child_v1(v_run)$new672$),
 ('public.lab_arena_expire_leases(text)',$old672$      AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)$old672$,$new672$      AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)
      AND NOT public.lab_arena__operator_execute_recovery_child_v1(v_run)$new672$),
 ('public.lab_arena_expire_leases(text)',$old672$    IF public.lab_arena__execute_host_retry_ready_v1(v_run) THEN$old672$,$new672$    IF NOT public.lab_arena__operator_execute_recovery_child_v1(v_run)
      AND public.lab_arena__execute_host_retry_ready_v1(v_run) THEN$new672$),
 ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',$old672$    AND (public.lab_arena_execution_attempt_count(v_run)<v_run.attempt$old672$,$new672$    AND (public.lab_arena_execution_attempt_count(v_run)<v_run.attempt
      OR public.lab_arena__operator_execute_recovery_child_v1(v_run)$new672$),
 ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)',$old672$  -- Keep existing receipts available for completion and billing above.$old672$,$new672$  IF public.lab_arena__operator_execute_recovery_child_v1(v_run)
    AND NOT public.lab_arena__operator_execute_recovery_valid_v1(v_round,v_run) THEN
    RETURN jsonb_build_object('status','stale');
  END IF;
  -- Keep existing receipts available for completion and billing above.$new672$)
  ) changes(signature,old_text,new_text) WHERE signature=target.signature LOOP
   IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
    RAISE EXCEPTION 'Execution recovery672 function anchor differs: %',target.signature; END IF;
   updated:=replace(updated,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Execution recovery672 function postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Execution recovery672 function readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch672$;
DO $restore672$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena672_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena672_schema_acl) THEN
  RAISE EXCEPTION 'Execution recovery672 schema ACL differs'; END IF;
END;
$restore672$;
NOTIFY pgrst,'reload schema';
COMMIT;
