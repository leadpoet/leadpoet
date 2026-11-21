-- Explicit owner correction of accepted infrastructure-damaged execution.
-- One pinned execution and one pinned judgment, with all original evidence and costs retained.
BEGIN;
SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena703_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $installed703$
DECLARE wanted RECORD; p RECORD;
BEGIN
 FOR wanted IN SELECT * FROM (VALUES
('lab_arena__accepted_execution_receipt_v1','793707d2dbcfeb7462646cc7765f9dac3366aa31d47da2d38b9c89d80a547961','s',0),('lab_arena_correct_accepted_execution_v1','6ae2787ae3000086368ad39c36930712028bcb29b1d23d8d16181cc825e9ce23','v',1),('lab_arena_prepare_corrected_execution_judgment_v1','b7df2faf423f8752fac855b5d3e196bdbb4c80da20b9018cfbe8958a8631ea68','v',1)
 ) functions(name,source_sha,volatility,defaults) LOOP
  FOR p IN SELECT * FROM pg_proc WHERE pronamespace='public'::regnamespace AND proname=wanted.name LOOP
   IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
    OR NOT p.prosecdef OR p.provolatile::TEXT<>wanted.volatility OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
    OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.proisstrict OR p.proleakproof
    OR p.proparallel<>'u' OR p.pronargdefaults<>wanted.defaults
    OR (wanted.defaults=1 AND pg_get_expr(p.proargdefaults,0) IS DISTINCT FROM 'true')
    OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>wanted.source_sha THEN
    RAISE EXCEPTION 'accepted execution correction703 installed helper differs: % %',wanted.name, jsonb_build_object('hash',encode(extensions.digest(p.prosrc,'sha256'),'hex'),'expected',wanted.source_sha,'owner',pg_get_userbyid(p.proowner),'acl',p.proacl::TEXT,'volatility',p.provolatile,'config',p.proconfig,'defaults',p.pronargdefaults,'strict',p.proisstrict,'parallel',p.proparallel); END IF;
  END LOOP;
 END LOOP;
END;
$installed703$;

CREATE OR REPLACE FUNCTION public.lab_arena__accepted_execution_receipt_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $valid703$
DECLARE h public.lab_arena_scoring_supersessions; origin public.lab_arena_scoring_supersessions;
 source public.lab_arena_runs; s public.lab_arena_submissions; item RECORD; retained public.lab_arena_runs; identity JSONB;
BEGIN
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=p_run.round_id
  AND recovery_doc->>'recovery_kind'='operator_execution_infrastructure_recovery_v1'
  AND recovery_doc->>'correction_kind'='accepted_execution_v1'
  AND recovery_doc#>>'{child_identity,run_id}'=p_run.run_id;
 IF NOT FOUND OR p_run.round_id<>p_round.round_id OR p_run.stage<>2
  OR p_run.champion_restart_required OR p_run.scoring_generation_floor<>p_round.scoring_generation_floor
  OR h.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
  OR ((h.new_round_doc->'configuration_doc')-'schedule') IS DISTINCT FROM (p_round.configuration_doc-'schedule')
  OR h.new_round_doc->'participants' IS DISTINCT FROM p_round.participants
  OR (p_run.runner_hotkey IS NOT NULL AND p_run.runner_hotkey IS DISTINCT FROM h.recovery_doc->>'approved_runner_hotkey')
  THEN RETURN FALSE; END IF;
 identity:=jsonb_build_object('run_id',p_run.run_id,'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,
  'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,'stage',p_run.stage,'icp_position',p_run.icp_position,
  'attempt',p_run.attempt,'stage_generation',p_run.stage_generation,'scoring_generation_floor',p_run.scoring_generation_floor);
 IF p_run.kind='score' THEN
  identity:=identity||jsonb_build_object('scored_run_id',p_run.scored_run_id,'company_judgment_refs',p_run.company_judgment_refs);
  SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
  IF NOT FOUND OR source.kind<>'execute' OR source.status<>'accepted' OR source.terminal_cause<>'accepted'
   OR (to_jsonb(source)-ARRAY['per_icp_score','qualification_doc','updated_at']) IS DISTINCT FROM
    ((h.recovery_doc->'accepted_execution')-ARRAY['per_icp_score','qualification_doc','updated_at'])
   OR NOT public.lab_arena__accepted_execution_receipt_v1(p_round,source) THEN RETURN FALSE; END IF;
  SELECT * INTO origin FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round.round_id
   AND request_id=h.recovery_doc->>'execution_request_id';
  IF NOT FOUND OR origin.recovery_doc#>>'{child_identity,run_id}'<>source.run_id
   OR h.recovery_doc->>'approved_runner_hotkey' IS DISTINCT FROM origin.recovery_doc->>'approved_runner_hotkey'
   THEN RETURN FALSE; END IF;
 ELSIF p_run.kind='execute' THEN origin:=h;
 ELSE RETURN FALSE; END IF;
 IF origin.recovery_doc->>'phase' IS DISTINCT FROM 'execution'
  OR origin.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest(origin.recovery_doc::TEXT,'sha256'),'hex')
  OR origin.recovery_doc#>>'{superseded_runs}' IS NULL
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'parent_run_id'))->>'kind' IS DISTINCT FROM 'execute'
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'parent_run_id'))->>'status' IS DISTINCT FROM 'accepted'
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'parent_run_id'))->>'assignment_id'
     IS DISTINCT FROM origin.recovery_doc#>>'{child_identity,assignment_id}'
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'judge_run_id'))->>'kind' IS DISTINCT FROM 'score'
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'judge_run_id'))->>'status' IS DISTINCT FROM 'accepted'
  OR (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'judge_run_id'))->>'scored_run_id'
     IS DISTINCT FROM origin.recovery_doc->>'parent_run_id'
  OR origin.recovery_doc->>'parent_sha256' IS DISTINCT FROM 'sha256:'||encode(extensions.digest(
     (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'parent_run_id'))::TEXT,'sha256'),'hex')
  OR origin.recovery_doc->>'judge_sha256' IS DISTINCT FROM 'sha256:'||encode(extensions.digest(
     (origin.recovery_doc->'superseded_runs'->(origin.recovery_doc->>'judge_run_id'))::TEXT,'sha256'),'hex')
  THEN RETURN FALSE; END IF;
 IF h.recovery_doc->'child_identity' IS DISTINCT FROM identity THEN RETURN FALSE; END IF;
 SELECT * INTO s FROM public.lab_arena_submissions WHERE submission_id=p_run.submission_id;
 IF NOT FOUND OR s.is_king OR s.status<>'frozen' OR s.code_review_status<>'passed'
  OR origin.recovery_doc->'source_identity' IS DISTINCT FROM jsonb_build_object(
   'round_id',s.round_id,'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
   'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256)
  OR jsonb_typeof(origin.recovery_doc->'superseded_runs') IS DISTINCT FROM 'object'
  OR (SELECT count(*) FROM jsonb_object_keys(origin.recovery_doc->'superseded_runs'))<>2 THEN RETURN FALSE; END IF;
 FOR item IN SELECT key,value FROM jsonb_each(origin.recovery_doc->'superseded_runs') LOOP
  SELECT * INTO retained FROM public.lab_arena_runs WHERE run_id=item.key;
  IF NOT FOUND OR retained.status<>'failed' OR retained.terminal_cause<>'stage_closed'
   OR retained.terminal_doc IS DISTINCT FROM jsonb_build_object('operation','operator_accepted_execution_correction_v1',
    'request_id',origin.request_id,'replacement_run_id',origin.recovery_doc#>>'{child_identity,run_id}')
   OR (to_jsonb(retained)-ARRAY['status','terminal_cause','terminal_doc','updated_at']) IS DISTINCT FROM
    (item.value-ARRAY['status','terminal_cause','terminal_doc','updated_at']) THEN RETURN FALSE; END IF;
 END LOOP;
 RETURN TRUE;
END;
$valid703$;
ALTER FUNCTION public.lab_arena__accepted_execution_receipt_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__accepted_execution_receipt_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_correct_accepted_execution_v1(
 p_run_id TEXT,p_run_sha256 TEXT,p_judge_run_id TEXT,p_judge_sha256 TEXT,p_approved_runner TEXT,
 p_evidence_ref TEXT,p_evidence_sha256 TEXT,p_request_id TEXT,p_actor_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $grant703$
DECLARE source public.lab_arena_runs; judge public.lab_arena_runs; child public.lab_arena_runs;
 r public.lab_arena_rounds; s public.lab_arena_submissions; prior public.lab_arena_scoring_supersessions;
 control public.lab_arena_restart_claim_control; proof JSONB; identity JSONB; cost JSONB; result JSONB;
 revision BIGINT; provider RECORD; available BOOLEAN:=FALSE;
BEGIN
 IF p_run_id IS NULL OR p_judge_run_id IS NULL OR coalesce(p_run_sha256,'') !~ '^sha256:[0-9a-f]{64}$'
  OR coalesce(p_judge_sha256,'') !~ '^sha256:[0-9a-f]{64}$' OR coalesce(p_evidence_sha256,'') !~ '^sha256:[0-9a-f]{64}$'
  OR coalesce(p_approved_runner,'') !~ '^[1-9A-HJ-NP-Za-km-z]{47,48}$'
  OR length(btrim(coalesce(p_evidence_ref,''))) NOT BETWEEN 1 AND 500
  OR length(btrim(coalesce(p_request_id,''))) NOT BETWEEN 1 AND 160
  OR length(btrim(coalesce(p_actor_ref,''))) NOT BETWEEN 1 AND 200 OR p_dry_run IS NULL THEN
  RAISE EXCEPTION 'accepted execution correction request invalid'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=p_run_id;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=source.round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'correction_kind' IS DISTINCT FROM 'accepted_execution_v1'
   OR prior.recovery_doc->>'phase' IS DISTINCT FROM 'execution'
   OR prior.recovery_doc->>'parent_run_id' IS DISTINCT FROM p_run_id
   OR prior.recovery_doc->>'parent_sha256' IS DISTINCT FROM p_run_sha256
   OR prior.recovery_doc->>'judge_run_id' IS DISTINCT FROM p_judge_run_id
   OR prior.recovery_doc->>'judge_sha256' IS DISTINCT FROM p_judge_sha256
   OR prior.recovery_doc->>'approved_runner_hotkey' IS DISTINCT FROM p_approved_runner
   OR prior.recovery_doc->>'evidence_ref' IS DISTINCT FROM p_evidence_ref
   OR prior.recovery_doc->>'evidence_sha256' IS DISTINCT FROM p_evidence_sha256
   OR prior.actor_ref IS DISTINCT FROM p_actor_ref THEN RAISE EXCEPTION 'accepted execution correction replay differs'; END IF;
  SELECT * INTO STRICT child FROM public.lab_arena_runs WHERE run_id=prior.recovery_doc#>>'{child_identity,run_id}';
  IF NOT public.lab_arena__accepted_execution_receipt_v1(r,child) THEN RAISE EXCEPTION 'accepted execution correction replay proof differs'; END IF;
  RETURN jsonb_build_object('status','granted','idempotent',TRUE,'child_identity',prior.recovery_doc->'child_identity');
 END IF;
 SELECT * INTO STRICT control FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF current_setting('session_replication_role')<>'origin' OR control.operator_paused
  OR coalesce(control.guard_commitment,'')<>'' OR coalesce(control.owner_commitment,'')<>''
  OR control.guard_expires_at IS NOT NULL OR coalesce(control.candidate_commit,'')<>''
  OR coalesce(control.restart_scope,'')<>'' OR coalesce(control.restart_phase,'')<>''
  OR coalesce(control.captured_leases,'[]')<>'[]'::JSONB OR r.status<>'stage2_scoring'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
  OR (clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
   AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ) IS NOT TRUE THEN
  RAISE EXCEPTION 'accepted execution correction round, control or deadline differs'; END IF;
 PERFORM 1 FROM public.lab_arena_runs WHERE assignment_id IN(source.assignment_id,source.assignment_id||':score') ORDER BY run_id FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=p_run_id;
 SELECT * INTO STRICT judge FROM public.lab_arena_runs WHERE run_id=p_judge_run_id;
 IF 'sha256:'||encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex') IS DISTINCT FROM p_run_sha256
  OR 'sha256:'||encode(extensions.digest(to_jsonb(judge)::TEXT,'sha256'),'hex') IS DISTINCT FROM p_judge_sha256
  OR source.kind<>'execute' OR source.stage<>2 OR source.status<>'accepted' OR source.terminal_cause<>'accepted'
  OR source.output_ref IS NULL OR source.result_doc->>'terminal_status' IS DISTINCT FROM 'accepted'
  OR source.runner_hotkey IS NULL OR source.runner_hotkey=p_approved_runner OR source.attempt>=2147483647
  OR source.champion_restart_required OR source.scoring_generation_floor<>r.scoring_generation_floor
  OR source.stage_generation>r.stage_generation
  OR source.assignment_id IS DISTINCT FROM source.round_id||':'||source.submission_id||':2:'||source.icp_position::TEXT
  OR source.run_id IS DISTINCT FROM source.assignment_id||':'||source.attempt::TEXT
  OR judge.kind<>'score' OR judge.status<>'accepted' OR judge.terminal_cause<>'accepted' OR judge.output_ref IS NULL
  OR judge.result_doc->>'terminal_status' IS DISTINCT FROM 'accepted' OR judge.scored_run_id<>source.run_id
  OR judge.assignment_id IS DISTINCT FROM source.assignment_id||':score'
  OR judge.run_id IS DISTINCT FROM judge.assignment_id||':'||judge.attempt::TEXT
  OR judge.attempt>=2147483647 OR judge.scoring_generation_floor<>r.scoring_generation_floor
  OR (judge.round_id,judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position) IS DISTINCT FROM
   (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
  OR judge.judgment_cache_key IS NOT NULL OR judge.judgment_input_hash IS NOT NULL OR judge.judgment_scope_doc IS NOT NULL
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id IN(source.assignment_id,judge.assignment_id)
   AND (a.status IN('pending','leased','submitted') OR (a.kind='execute' AND a.attempt>source.attempt)
    OR (a.kind='score' AND a.attempt>judge.attempt) OR (a.status='accepted' AND a.run_id NOT IN(source.run_id,judge.run_id))))
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
   AND h.recovery_doc->>'recovery_kind'='operator_execution_infrastructure_recovery_v1'
   AND h.recovery_doc#>>'{child_identity,assignment_id}'=source.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(source.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(judge.assignment_id)
  OR (SELECT count(*) FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items') x
   WHERE x->>'scored_run_id'=source.run_id AND x->>'submission_id'=source.submission_id
    AND x->'icp_position'=to_jsonb(source.icp_position) AND x->>'output_ref'=source.output_ref)<>1 THEN
  RAISE EXCEPTION 'accepted execution correction exact source, judgment or plan differs'; END IF;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=source.submission_id FOR SHARE;
 IF s.is_king OR s.status<>'frozen' OR s.code_review_status<>'passed' OR s.source_ref IS NULL
  OR s.source_archive_sha256 IS NULL OR s.source_size_bytes IS NULL OR s.round_id<>r.round_id OR s.miner_hotkey<>source.miner_hotkey
  OR p_approved_runner=s.miner_hotkey OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p
   WHERE p->>'submission_id'=s.submission_id AND p->>'miner_hotkey'=s.miner_hotkey) THEN
  RAISE EXCEPTION 'accepted execution correction frozen source differs'; END IF;
 cost:=public.lab_arena__successful_icp_cost_state(r.round_id,s.submission_id,source.icp_position);
 IF (cost->>'inflight_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
  OR (cost->>'success_unresolved_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
  OR (coalesce(cost->>'competition_settled_microusd',cost->>'settled_microusd')::BIGINT
   <(r.configuration_doc->>'execution_icp_cap_microusd')::BIGINT) IS NOT TRUE THEN
  RAISE EXCEPTION 'accepted execution correction unresolved cost or exhausted cap'; END IF;
 FOR provider IN SELECT key,value FROM jsonb_each_text(r.configuration_doc->'call_quotas') LOOP
  IF provider.value::BIGINT=0 AND provider.key='deepline'
   AND r.configuration_doc#>>'{deepline_catalog,schema_version}'='leadpoet.lab_arena.deepline_catalog.v1' THEN available:=TRUE;
  ELSIF provider.value::BIGINT>0
   AND public.lab_arena__execute_assignment_calls_v1(source.assignment_id,provider.key)<
    provider.value::BIGINT*(r.configuration_doc->>'max_attempts_per_assignment')::BIGINT
   AND public.lab_arena__submission_stage_consumed(s.submission_id,source.stage,provider.key,'execute')<
    provider.value::BIGINT*(r.configuration_doc->>'max_attempts_per_assignment')::BIGINT*
    (r.configuration_doc->>'stage_2_icp_count')::BIGINT THEN available:=TRUE; END IF;
 END LOOP;
 IF NOT available THEN RAISE EXCEPTION 'accepted execution correction original quota exhausted'; END IF;
 identity:=jsonb_build_object('run_id',source.assignment_id||':'||(source.attempt+1)::TEXT,
  'assignment_id',source.assignment_id,'round_id',source.round_id,'submission_id',source.submission_id,
  'miner_hotkey',source.miner_hotkey,'stage',source.stage,'icp_position',source.icp_position,'attempt',source.attempt+1,
  'stage_generation',r.stage_generation,'scoring_generation_floor',r.scoring_generation_floor);
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_execution_infrastructure_recovery_v1',
  'correction_kind','accepted_execution_v1','phase','execution','parent_run_id',source.run_id,'parent_sha256',p_run_sha256,
  'judge_run_id',judge.run_id,'judge_sha256',p_judge_sha256,'approved_runner_hotkey',p_approved_runner,
  'evidence_ref',p_evidence_ref,'evidence_sha256',p_evidence_sha256,'superseded_runs',jsonb_build_object(source.run_id,to_jsonb(source),judge.run_id,to_jsonb(judge)),
  'source_identity',jsonb_build_object('round_id',s.round_id,'submission_id',s.submission_id,'miner_hotkey',s.miner_hotkey,
   'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'source_archive_sha256',s.source_archive_sha256),
  'original_cost',cost,'child_identity',identity);
 result:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'granted' END,'idempotent',FALSE,
  'child_identity',identity,'original_cost',cost);
 IF p_dry_run THEN RETURN result; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),p_actor_ref,
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',proof);
 UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',
  terminal_doc=jsonb_build_object('operation','operator_accepted_execution_correction_v1','request_id',p_request_id,
   'replacement_run_id',identity->>'run_id') WHERE run_id IN(source.run_id,judge.run_id);
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,lease_generation,stage_generation,scoring_generation_floor,kind,previous_runner_hotkey)
 VALUES(identity->>'run_id',source.assignment_id,source.round_id,source.submission_id,source.miner_hotkey,source.stage,
  source.icp_position,source.attempt+1,'pending',source.lease_generation,r.stage_generation,r.scoring_generation_floor,'execute',source.runner_hotkey);
 RETURN result;
END;
$grant703$;
ALTER FUNCTION public.lab_arena_correct_accepted_execution_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_correct_accepted_execution_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_corrected_execution_judgment_v1(
 p_preparation JSONB,p_request_id TEXT,p_actor_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $plan703$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds; source public.lab_arena_runs; parent public.lab_arena_runs; child public.lab_arena_runs;
 origin public.lab_arena_scoring_supersessions; prior public.lab_arena_scoring_supersessions;
 control public.lab_arena_restart_claim_control; expected JSONB; item JSONB; plan JSONB; identity JSONB; proof JSONB; result JSONB;
 revision BIGINT;
BEGIN
 IF jsonb_typeof(p_preparation) IS DISTINCT FROM 'object' OR octet_length(p_preparation::TEXT)>1048576
  OR p_preparation->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.accepted_execution_preparation.v1'
  OR length(btrim(coalesce(p_request_id,''))) NOT BETWEEN 1 AND 160
  OR length(btrim(coalesce(p_actor_ref,''))) NOT BETWEEN 1 AND 200 OR p_dry_run IS NULL
  OR coalesce(p_preparation->>'output_sha256','') !~ '^sha256:[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'corrected execution preparation invalid'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_preparation#>>'{source,round_id}' FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'phase' IS DISTINCT FROM 'judgment' OR prior.recovery_doc->'preparation' IS DISTINCT FROM p_preparation
   OR prior.actor_ref IS DISTINCT FROM p_actor_ref THEN RAISE EXCEPTION 'corrected execution preparation replay differs'; END IF;
  SELECT * INTO STRICT child FROM public.lab_arena_runs WHERE run_id=prior.recovery_doc#>>'{child_identity,run_id}';
  IF NOT public.lab_arena__accepted_execution_receipt_v1(r,child) THEN RAISE EXCEPTION 'corrected execution preparation replay proof differs'; END IF;
  RETURN jsonb_build_object('status','granted','idempotent',TRUE,'child_identity',prior.recovery_doc->'child_identity');
 END IF;
 SELECT * INTO STRICT control FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF current_setting('session_replication_role')<>'origin' OR control.operator_paused
  OR coalesce(control.guard_commitment,'')<>'' OR coalesce(control.owner_commitment,'')<>''
  OR control.guard_expires_at IS NOT NULL OR coalesce(control.candidate_commit,'')<>''
  OR coalesce(control.restart_scope,'')<>'' OR coalesce(control.restart_phase,'')<>''
  OR coalesce(control.captured_leases,'[]')<>'[]'::JSONB OR r.status<>'stage2_scoring'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
  OR (clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
   AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ) IS NOT TRUE THEN
  RAISE EXCEPTION 'corrected execution preparation round, control or deadline differs'; END IF;
 SELECT jsonb_object_agg(key,value) INTO expected FROM jsonb_each(to_jsonb(r)) WHERE key=ANY(
  ARRAY['round_id','status','status_generation','stage_generation','scoring_generation_floor','configuration_doc','participants','stage2_scoring_plan_doc']);
 IF expected IS DISTINCT FROM p_preparation->'expected_round' THEN RAISE EXCEPTION 'corrected execution preparation round changed'; END IF;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=p_preparation#>>'{source,run_id}' FOR UPDATE;
 SELECT * INTO STRICT origin FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id
  AND recovery_doc->>'correction_kind'='accepted_execution_v1' AND recovery_doc->>'phase'='execution'
  AND recovery_doc#>>'{child_identity,run_id}'=source.run_id;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=origin.recovery_doc->>'judge_run_id' FOR UPDATE;
 SELECT jsonb_object_agg(key,value) INTO expected FROM jsonb_each(to_jsonb(source)) WHERE key=ANY(
  ARRAY['run_id','assignment_id','round_id','submission_id','miner_hotkey','kind','stage','icp_position','attempt','status','terminal_cause','stage_generation','scoring_generation_floor','output_ref']);
 item:=p_preparation->'work_item';
 IF expected IS DISTINCT FROM p_preparation->'source' OR source.kind<>'execute' OR source.status<>'accepted'
  OR source.terminal_cause<>'accepted' OR source.output_ref IS NULL OR NOT public.lab_arena__accepted_execution_receipt_v1(r,source)
  OR jsonb_typeof(item) IS DISTINCT FROM 'object' OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(item)k) IS DISTINCT FROM
   ARRAY['company_judgment_refs','icp_position','output_ref','scored_run_id','submission_id']::TEXT[]
  OR item->>'scored_run_id' IS DISTINCT FROM source.run_id OR item->>'submission_id' IS DISTINCT FROM source.submission_id
  OR item->>'output_ref' IS DISTINCT FROM source.output_ref OR item->'icp_position' IS DISTINCT FROM to_jsonb(source.icp_position)
  OR NOT (CASE WHEN r.configuration_doc ? 'company_quality_policy' OR r.configuration_doc->'scorer_policy' ? 'company_quality_policy'
   THEN public.lab_arena__company_quality_refs_v1(r,item->'company_judgment_refs')
   ELSE public.lab_arena__company_judgment_refs_v1(r,item->'company_judgment_refs') END)
  OR NOT public.lab_arena__judge_calls_clear_v1(source.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id AND a.attempt>parent.attempt)
  OR (SELECT count(*) FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items') x
   WHERE x->>'scored_run_id'=origin.recovery_doc->>'parent_run_id' AND x->>'submission_id'=source.submission_id
    AND x->'icp_position'=to_jsonb(source.icp_position))<>1
  OR EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(r.stage2_scoring_plan_doc->'zero_rows','[]'))x
   WHERE x->>'submission_id'=source.submission_id AND x->'icp_position'=to_jsonb(source.icp_position)) THEN
  RAISE EXCEPTION 'corrected execution preparation source, calls or plan differs'; END IF;
 SELECT jsonb_set(r.stage2_scoring_plan_doc,'{work_items}',jsonb_agg(CASE
  WHEN x->>'scored_run_id'=origin.recovery_doc->>'parent_run_id' THEN item-'company_judgment_refs' ELSE x END ORDER BY ord))
 INTO plan FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items') WITH ORDINALITY a(x,ord);
 n:=r; n.stage2_scoring_plan_doc:=plan;
 identity:=jsonb_build_object('run_id',parent.assignment_id||':'||(parent.attempt+1)::TEXT,
  'assignment_id',parent.assignment_id,'round_id',source.round_id,'submission_id',source.submission_id,
  'miner_hotkey',source.miner_hotkey,'stage',source.stage,'icp_position',source.icp_position,'attempt',parent.attempt+1,
  'stage_generation',r.stage_generation,'scoring_generation_floor',r.scoring_generation_floor,
  'scored_run_id',source.run_id,'company_judgment_refs',item->'company_judgment_refs');
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_execution_infrastructure_recovery_v1',
  'correction_kind','accepted_execution_v1','phase','judgment','execution_request_id',origin.request_id,
  'accepted_execution',to_jsonb(source),'approved_runner_hotkey',origin.recovery_doc->>'approved_runner_hotkey',
  'child_identity',identity,'preparation',p_preparation);
 result:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'granted' END,'idempotent',FALSE,'child_identity',identity);
 IF p_dry_run THEN RETURN result; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),p_actor_ref,
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',proof);
 UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=plan WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
  attempt,status,stage_generation,scoring_generation_floor,kind,scored_run_id,company_judgment_refs)
 VALUES(identity->>'run_id',parent.assignment_id,source.round_id,source.submission_id,source.miner_hotkey,source.stage,
  source.icp_position,parent.attempt+1,'pending',r.stage_generation,r.scoring_generation_floor,'score',source.run_id,
  item->'company_judgment_refs');
 RETURN result;
END;
$plan703$;
ALTER FUNCTION public.lab_arena_prepare_corrected_execution_judgment_v1(JSONB,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_corrected_execution_judgment_v1(JSONB,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $patch703$
DECLARE target RECORD; p RECORD; d TEXT; metadata JSONB;
BEGIN
 FOR target IN SELECT * FROM (VALUES
('lab_arena__operator_execute_recovery_child_v1(lab_arena_runs)','7fb2767da44dddb1b4f6cfd4384d30c3878667cd5ca92fc3d348cd32574a0482','8097654dfa305fa9ad7e9efc379c4610a7f21e8dbbbd3d03382a851119e81054','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"]]'::JSONB,$old703_0_0$p_run.kind='execute'$old703_0_0$,$new703_0_0$p_run.kind IN('execute','score')$new703_0_0$),
('lab_arena__operator_execute_recovery_valid_v1(lab_arena_rounds,lab_arena_runs)','350c4377413f9ac729e606fa97afd477ee8e55b79e9e0345ef877cabd8edb860','f0f195b2d192def1ab21d44aadfda309f890405a15e742a3c58fd0a13ae78a91','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"]]'::JSONB,$old703_1_0$ IF NOT FOUND OR p_run.kind<>'execute'$old703_1_0$,$new703_1_0$ IF FOUND AND h.recovery_doc->>'correction_kind'='accepted_execution_v1' THEN
  RETURN public.lab_arena__accepted_execution_receipt_v1(p_round,p_run);
 END IF;
 IF NOT FOUND OR p_run.kind<>'execute'$new703_1_0$),
('lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','145f89a0ad3076cfb75da6fd7e733325da242a5d6fb7a60c0de5d7c5c3ec6781','f7c1ed03241656ed0111749c04c18ad5ceab019d51dd93ec957a3f52a0f86da0','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"]]'::JSONB,$old703_2_0$    AND (NOT public.lab_arena__operator_execute_recovery_child_v1(runs)
      OR public.lab_arena__operator_execute_recovery_valid_v1(v_round,runs))$old703_2_0$,$new703_2_0$    AND (NOT public.lab_arena__operator_execute_recovery_child_v1(runs)
      OR public.lab_arena__operator_execute_recovery_valid_v1(v_round,runs))
    AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions pinned
      WHERE pinned.round_id=runs.round_id AND pinned.recovery_doc->>'correction_kind'='accepted_execution_v1'
        AND pinned.recovery_doc#>>'{child_identity,run_id}'=runs.run_id
        AND pinned.recovery_doc->>'approved_runner_hotkey' IS DISTINCT FROM p_runner_hotkey)$new703_2_0$),
('lab_arena__baseline_recovery_phase_v1(lab_arena_rounds,lab_arena_runs)','4acbf6eba8b497455270d1f7b1ea4e407e3dbf175d136005e38fe81af90a9428','137bf5274d6f1ccb04c49fa84968e33b72cdf792bac641099cd3fee70a2497a9','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"]]'::JSONB,$old703_3_0$ SELECT COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)$old703_3_0$,$new703_3_0$ SELECT (p_round.status='stage2_scoring'
  AND clock_timestamp()<(p_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND public.lab_arena__accepted_execution_receipt_v1(p_round,p_run))
 OR COALESCE(public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)$new703_3_0$),
('lab_arena_close_scoring(text,smallint)','10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636','17dd46e57155b40d20637bda231cefab050cf98da2eac5180f7223cbc33593c2','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"]]'::JSONB,$old703_4_0$  v_generation := v_round.stage_generation + 1;$old703_4_0$,$new703_4_0$  -- Only explicitly registered unfinished corrections defer this stage.
  IF p_stage=2 AND EXISTS(SELECT 1 FROM public.lab_arena_runs recovery
    JOIN public.lab_arena_scoring_supersessions h ON h.round_id=recovery.round_id
      AND h.recovery_doc#>>'{child_identity,run_id}'=recovery.run_id
    WHERE recovery.round_id=p_round_id AND recovery.kind IN('execute','score')
      AND h.recovery_doc->>'correction_kind'='accepted_execution_v1'
      AND (recovery.status IN('leased','submitted')
        OR (clock_timestamp()<(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
          AND (recovery.status='pending' OR (recovery.kind='execute' AND recovery.status='accepted'
            AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs judge WHERE judge.kind='score'
              AND judge.scored_run_id=recovery.run_id)))))) THEN
    RETURN jsonb_build_object('status','draining','round_status',v_round.status,
      'stage_generation',v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;$new703_4_0$),
('lab_arena_runs_terminal_v1()','745d4b6a5e1a2e247a7054ce8540fad909691a7a18ec36c8bf684fecdd37a712','b20c68699730fbc0acca6d70c8deebbbdd098389e76e515f203963f284b048ab','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"]]'::JSONB,$old703_5_0$BEGIN
$old703_5_0$,$new703_5_0$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind IN('execute','score') AND OLD.status='accepted' AND OLD.terminal_cause='accepted'
    AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
    AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
      (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
        AND h.recovery_doc->>'correction_kind'='accepted_execution_v1'
        AND h.recovery_doc->>'phase'='execution'
        AND h.recovery_doc->'superseded_runs'->OLD.run_id=to_jsonb(OLD)
        AND NEW.terminal_doc=jsonb_build_object('operation','operator_accepted_execution_correction_v1',
          'request_id',h.request_id,'replacement_run_id',h.recovery_doc#>>'{child_identity,run_id}')) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new703_5_0$),
('lab_arena_retired_accepted_judgment_v1(text)','9b75b344ad0dd237fe3825281b1b9b857fa4cc165ec209e89b89c92158c9fa05','f8ece4d9a2505fc6893271a9d5e45b7d9feab236219d80371d1feed3a1495a0f','["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"]]'::JSONB,$old703_6_0$      ((h.recovery_doc->'superseded_judgment')-ARRAY['status','terminal_cause','terminal_doc','updated_at']));$old703_6_0$,$new703_6_0$      ((h.recovery_doc->'superseded_judgment')-ARRAY['status','terminal_cause','terminal_doc','updated_at']))
 OR EXISTS(SELECT 1 FROM public.lab_arena_runs retired
  JOIN public.lab_arena_scoring_supersessions h ON h.round_id=retired.round_id
  JOIN public.lab_arena_runs child ON child.run_id=h.recovery_doc#>>'{child_identity,run_id}'
  JOIN public.lab_arena_rounds r ON r.round_id=retired.round_id
  WHERE retired.run_id=p_run_id AND retired.kind='score'
    AND h.recovery_doc->>'correction_kind'='accepted_execution_v1'
    AND h.recovery_doc->>'phase'='execution' AND h.recovery_doc->>'judge_run_id'=retired.run_id
    AND public.lab_arena__accepted_execution_receipt_v1(r,child));$new703_6_0$)
 ) patches(signature,preimage,postimage,identity,old_text,new_text) LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;
  d:=pg_get_functiondef(p.oid); metadata:=to_jsonb(p)-'prosrc';
  IF jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    IS DISTINCT FROM target.identity OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=
      CASE WHEN p.proname IN('lab_arena__operator_execute_recovery_child_v1','lab_arena__baseline_recovery_phase_v1',
        'lab_arena_retired_accepted_judgment_v1') THEN 'sql' ELSE 'plpgsql' END) THEN
   RAISE EXCEPTION 'accepted execution correction703 security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
    OR (length(d)-length(replace(d,target.old_text,'')))/length(target.old_text)<>1 THEN
   RAISE EXCEPTION 'accepted execution correction703 preimage differs: %',target.signature; END IF;
  d:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'accepted execution correction703 patch differs'; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'accepted execution correction703 metadata changed'; END IF;
 END LOOP;
END;
$patch703$;

DO $restore703$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena703_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena703_acl) THEN
  RAISE EXCEPTION 'accepted execution correction703 schema ACL differs'; END IF;
END;
$restore703$;
NOTIFY pgrst,'reload schema'; COMMIT;
