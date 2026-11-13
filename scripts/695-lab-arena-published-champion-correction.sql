-- Explicit owner-authorized correction of a published no-winner round.
-- Reuse existing finite scorer-bug grants and immutable supersession history.
-- No normal retry limit, score, evidence, cost, or reward rule is changed.
-- Deploy the archived-basis reader before invoking this inactive capability.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena695_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE OR REPLACE FUNCTION public.lab_arena_correct_published_judgments_v1(
 p_round_id TEXT,p_round_sha256 TEXT,p_runtime_image_digest TEXT,p_final_scoring_close TIMESTAMPTZ,
 p_publication_deadline TIMESTAMPTZ,p_requests JSONB,p_request_id TEXT,p_actor_ref TEXT,p_approval_ref TEXT,
 p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $correct695$
DECLARE r public.lab_arena_rounds;n public.lab_arena_rounds;parent public.lab_arena_runs;child public.lab_arena_runs;
 mapping public.lab_arena_reviewer_runtimes;prior public.lab_arena_scoring_supersessions;
 item JSONB;proof JSONB;checkpoints JSONB;signals JSONB;grants JSONB:='[]';revision BIGINT;digest TEXT;round_hash TEXT;
BEGIN
 IF p_round_id IS NULL OR p_round_sha256 IS NULL OR p_round_sha256!~'^sha256:[0-9a-f]{64}$'
  OR p_runtime_image_digest IS NULL OR p_runtime_image_digest!~'^sha256:[0-9a-f]{64}$'
  OR p_request_id IS NULL OR length(p_request_id)NOT BETWEEN 1 AND 160
  OR p_actor_ref IS NULL OR length(btrim(p_actor_ref))NOT BETWEEN 1 AND 200
  OR p_approval_ref IS NULL OR length(btrim(p_approval_ref))NOT BETWEEN 1 AND 500
  OR p_final_scoring_close IS NULL OR p_publication_deadline IS NULL OR p_dry_run IS NULL
  OR jsonb_typeof(p_requests)IS DISTINCT FROM 'array' THEN RAISE EXCEPTION 'postpublication recovery request invalid';END IF;
 IF jsonb_array_length(p_requests)NOT BETWEEN 1 AND 100
  OR (SELECT count(DISTINCT x->>'parent_run_id')FROM jsonb_array_elements(p_requests)x)<>jsonb_array_length(p_requests)
  OR (SELECT count(DISTINCT x->>'request_id')FROM jsonb_array_elements(p_requests)x)<>jsonb_array_length(p_requests) THEN
  RAISE EXCEPTION 'postpublication recovery parent/request list invalid';END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.reward_epoch',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF to_regprocedure('public.lab_arena_correction_reward_history_v1(text,text,integer,integer,integer,boolean)')IS NULL THEN
  RAISE EXCEPTION 'postpublication correction reward history reader unavailable';END IF;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF (prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_postpublication_champion_correction_v1' OR prior.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest(prior.recovery_doc::TEXT,'sha256'),'hex')) OR prior.actor_ref IS DISTINCT FROM p_actor_ref
   OR prior.recovery_doc->>'round_sha256' IS DISTINCT FROM p_round_sha256
   OR prior.recovery_doc->>'runtime_image_digest'IS DISTINCT FROM p_runtime_image_digest
   OR prior.recovery_doc->>'approval_ref'IS DISTINCT FROM p_approval_ref OR prior.recovery_doc->'requests'IS DISTINCT FROM p_requests
   OR (prior.new_round_doc#>>'{configuration_doc,schedule,final_scoring_close}')::TIMESTAMPTZ IS DISTINCT FROM p_final_scoring_close
   OR (prior.new_round_doc#>>'{configuration_doc,schedule,publication_deadline}')::TIMESTAMPTZ IS DISTINCT FROM p_publication_deadline THEN
   RAISE EXCEPTION 'postpublication recovery replay differs';END IF;
  FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
   IF NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions g
    JOIN public.lab_arena_runs a ON a.run_id=g.recovery_doc#>>'{child_identity,run_id}'
    WHERE g.round_id=r.round_id AND g.request_id=item->>'request_id'AND g.actor_ref=p_actor_ref
     AND g.request_hash='sha256:'||encode(extensions.digest(g.recovery_doc::TEXT,'sha256'),'hex')
     AND g.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
     AND g.recovery_doc->>'parent_run_id'=item->>'parent_run_id'AND g.recovery_doc->>'parent_sha256'=item->>'parent_sha256'
     AND g.recovery_doc->>'runtime_image_digest'=p_runtime_image_digest AND g.recovery_doc->>'correction_ref'=p_approval_ref
     AND g.recovery_doc->'checkpoints'=item->'checkpoints'AND g.recovery_doc->'signal_checkpoints'=item->'signal_checkpoints'
     AND public.lab_arena__operator_scorer_bug_child_v1(a))THEN RAISE EXCEPTION 'postpublication recovery replay child differs';END IF;
  END LOOP;
  RETURN jsonb_build_object('status','recovered','idempotent',TRUE,'request_id',p_request_id,'requests',p_requests);
 END IF;
 round_hash:='sha256:'||encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex');
 IF current_setting('session_replication_role')<>'origin' OR round_hash IS DISTINCT FROM p_round_sha256
  OR r.configuration_doc->>'mode'IS DISTINCT FROM 'live'OR r.rewards_enabled IS DISTINCT FROM TRUE
  OR r.status<>'published' OR r.published_at IS NULL OR r.publication_doc IS NULL OR r.reward_activated_at IS NULL
  OR r.reward_basis_doc IS NULL OR r.reward_basis_hash IS NULL OR r.effective_reward_epoch IS NULL OR r.signing_key_doc IS NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
  OR coalesce(r.publication_doc#>>'{king_decision,king_hotkey}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,king_submission_id}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,winner_submission_id}','')<>''
  OR r.king_outcome IS DISTINCT FROM 'no_king' OR r.publication_doc#>>'{king_decision,outcome}'IS DISTINCT FROM 'no_king'
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
    AND h.recovery_doc->>'recovery_kind' IN('operator_postpublication_champion_correction_v1','operator_postpublication_judge_recovery_v1'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_rounds newer WHERE
    (newer.evaluation_date>r.evaluation_date OR(newer.evaluation_date IS NULL AND newer.created_at>r.created_at))
    AND newer.arena_network_name=r.arena_network_name AND newer.arena_netuid=r.arena_netuid
    AND newer.configuration_doc->>'mode'='live' AND(newer.status<>'open' OR newer.benchmark_ref IS NOT NULL
      OR newer.reward_activated_at IS NOT NULL OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=newer.round_id)))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status IN('pending','leased','submitted'))
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE singleton AND NOT operator_paused AND guard_commitment='') THEN
  RAISE EXCEPTION 'postpublication recovery published authority or drained state differs';END IF;
 IF NOT(clock_timestamp()<p_final_scoring_close AND p_final_scoring_close<p_publication_deadline
  AND p_final_scoring_close>(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND p_publication_deadline>(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  AND p_publication_deadline<=clock_timestamp()+interval '24 hours'
  AND date_trunc('second',p_final_scoring_close)=p_final_scoring_close
  AND date_trunc('second',p_publication_deadline)=p_publication_deadline)THEN RAISE EXCEPTION 'postpublication recovery deadline invalid';END IF;
 SELECT * INTO STRICT mapping FROM public.lab_arena_reviewer_runtimes
  WHERE grading_image_digest=r.configuration_doc->>'scorer_image_digest'
   AND grading_image_reference=r.configuration_doc->>'scorer_image_reference'FOR SHARE;
 IF mapping.runtime_image_digest IS DISTINCT FROM p_runtime_image_digest THEN RAISE EXCEPTION 'postpublication recovery selected runtime differs';END IF;
 -- Historical signed authority is retained in this transaction's immutable
 -- old_round_doc. The history reader must be deployed before invoking this.
 n:=r;n.status:='stage2_judged';n.status_generation:=r.status_generation+1;n.published_at:=NULL;n.publication_doc:=NULL;
 n.reward_basis_doc:=NULL;n.reward_basis_hash:=NULL;n.signing_key_doc:=NULL;
 n.reward_activated_at:=NULL;n.effective_reward_epoch:=NULL;n.king_start_epoch:=NULL;
 n.king_outcome:=NULL;n.king_hotkey:=NULL;
 n.promotion_doc:=NULL;n.baseline_promoted_at:=NULL;
 n.configuration_doc:=jsonb_set(jsonb_set(r.configuration_doc,'{schedule,final_scoring_close}',
  to_jsonb(to_char(p_final_scoring_close AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'))),'{schedule,publication_deadline}',
  to_jsonb(to_char(p_publication_deadline AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"')));
 FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
  IF jsonb_typeof(item)IS DISTINCT FROM 'object'
   OR(SELECT array_agg(k ORDER BY k)FROM jsonb_object_keys(item)k)IS DISTINCT FROM
      ARRAY['checkpoints','parent_run_id','parent_sha256','request_id','signal_checkpoints']::TEXT[]
   OR jsonb_typeof(item->'checkpoints')IS DISTINCT FROM 'object'OR jsonb_typeof(item->'signal_checkpoints')IS DISTINCT FROM 'object'
   OR length(coalesce(item->>'request_id',''))NOT BETWEEN 1 AND 160
   OR coalesce(item->>'parent_sha256','')!~'^sha256:[0-9a-f]{64}$'THEN RAISE EXCEPTION 'postpublication recovery parent input invalid';END IF;
  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=item->>'parent_run_id'FOR UPDATE;
  IF parent.round_id<>r.round_id OR parent.kind<>'score'OR parent.status<>'failed'OR parent.terminal_cause NOT IN('judge_error','judge_timeout')
   OR parent.result_doc->>'terminal_status'IS DISTINCT FROM parent.terminal_cause
   OR NOT((parent.result_doc#>>'{failure_diagnostic,stage}'IS NOT DISTINCT FROM 'scorer'AND parent.result_doc#>>'{failure_diagnostic,error_class}'IS NOT DISTINCT FROM 'judge_error')
     OR(parent.result_doc#>>'{failure_diagnostic,stage}'IS NOT DISTINCT FROM 'sandbox'AND parent.result_doc#>>'{failure_diagnostic,error_class}'IS NOT DISTINCT FROM 'judge_timeout'))
   OR parent.scoring_generation_floor<>r.scoring_generation_floor
   OR p_runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
   OR item->>'parent_sha256'IS DISTINCT FROM 'sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id AND(a.attempt>parent.attempt OR a.status='accepted'))
   OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id AND
    (h.request_id=item->>'request_id'OR(h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)))THEN
   RAISE EXCEPTION 'postpublication correction exact parent differs';END IF;
  child:=parent;child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT;child.attempt:=parent.attempt+1;child.stage_generation:=r.stage_generation;
  IF NOT public.lab_arena__judge_continuation_v1(n,child)THEN RAISE EXCEPTION 'postpublication recovery frozen continuation differs';END IF;
  SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
   INTO checkpoints FROM public.lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(parent.company_judgment_refs,'[]'))ref WHERE ref->>'cache_key'=j.cache_key);
  SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
   INTO signals FROM public.lab_arena_signal_judgments j WHERE j.source_scored_run_id=parent.scored_run_id;
  IF checkpoints IS DISTINCT FROM item->'checkpoints'OR signals IS DISTINCT FROM item->'signal_checkpoints'THEN RAISE EXCEPTION 'postpublication recovery approved checkpoints differ';END IF;
 END LOOP;
 IF p_dry_run THEN RETURN jsonb_build_object('status','dry_run','idempotent',FALSE,'request_id',p_request_id,'new_round',to_jsonb(n)-'updated_at','requests',p_requests);END IF;
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_postpublication_champion_correction_v1',
  'round_sha256',round_hash,'runtime_image_digest',p_runtime_image_digest,'approval_ref',p_approval_ref,'requests',p_requests,
  'final_scoring_close',n.configuration_doc#>'{schedule,final_scoring_close}','publication_deadline',n.configuration_doc#>'{schedule,publication_deadline}');
 digest:='sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex');
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
  old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,digest,p_actor_ref,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',proof);
 IF NOT EXISTS(SELECT 1 FROM jsonb_array_elements(public.lab_arena_correction_reward_history_v1(
   'live',r.arena_network_name,r.arena_netuid,200,0,FALSE)) archived
  WHERE archived->>'round_id'=r.round_id AND archived->>'reward_basis_hash'=r.reward_basis_hash
   AND (archived->>'effective_reward_epoch')::BIGINT=r.effective_reward_epoch
   AND archived->'reward_basis_doc'=r.reward_basis_doc AND archived->'signing_key_doc'=r.signing_key_doc)THEN
  RAISE EXCEPTION 'postpublication correction historical reward authority unavailable';END IF;
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,published_at=NULL,publication_doc=NULL,
  configuration_doc=n.configuration_doc,reward_basis_doc=NULL,reward_basis_hash=NULL,signing_key_doc=NULL,
  reward_activated_at=NULL,effective_reward_epoch=NULL,king_start_epoch=NULL,king_outcome=NULL,king_hotkey=NULL,
  promotion_doc=NULL,baseline_promoted_at=NULL WHERE round_id=r.round_id;
 FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
  grants:=grants||jsonb_build_array(public.lab_arena_grant_scorer_bug_recovery_v1(item->>'parent_run_id',item->>'parent_sha256',
   p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,FALSE));
 END LOOP;
 RETURN jsonb_build_object('status','recovered','idempotent',FALSE,'request_id',p_request_id,'requests',p_requests,'grants',grants);
END;
$correct695$;
ALTER FUNCTION public.lab_arena_correct_published_judgments_v1(TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ,JSONB,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_correct_published_judgments_v1(TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ,JSONB,TEXT,TEXT,TEXT,BOOLEAN) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;


DO $restore695$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena695_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena695_acl)THEN RAISE EXCEPTION 'correction695 schema ACL differs';END IF;
END;
$restore695$;
NOTIFY pgrst,'reload schema';COMMIT;
