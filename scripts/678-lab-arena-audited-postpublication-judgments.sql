-- One owner-audited, score-only recovery of an activated no-king publication.
-- Archive the original publication; retain reward/champion authority exactly.
-- Existing finite grants, leases, billing, frozen inputs and finalizers apply.
-- The signed no-king emissions decision remains its original authority. New
-- judgments amend evaluation only; republish cannot create a winner or basis.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena678_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $installed678$
DECLARE target RECORD;p RECORD;t RECORD;
BEGIN
 FOR target IN SELECT * FROM(VALUES
 ('public.lab_arena__postpublication_receipt_v1(public.lab_arena_scoring_supersessions)','ffc3b788f5e60a4c479924078bba13754fa8a95cd7f7dec5a9b18a344db86a7e','s','sql'),
 ('public.lab_arena__postpublication_judge_v1(public.lab_arena_rounds,public.lab_arena_runs)','ad6cf94247f0a4643fd8f676d97c4a6b72e68b1f899bfc239c40608bd42fc26f','s','sql'),
 ('public.lab_arena_recover_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)','31f72c12f71307e2389b6937723254bc6b713db3096a19b281539cb0962f79dd','v','plpgsql'),
 ('public.lab_arena_postpublication_reward_seal_v1()','ef9c9f1eb28eacd2ddf8463e0c0fc1196610de2b35df87f3374136acaff14133','v','plpgsql')
 )f(signature,source_sha,volatility,language)LOOP
  SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure(target.signature);
  IF FOUND AND(pg_get_userbyid(p.proowner)<>'lab_arena_owner'OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile::TEXT<>target.volatility OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=target.language)
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'
   OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>target.source_sha)THEN RAISE EXCEPTION 'postpublication678 installed function differs: %',target.signature;END IF;
 END LOOP;
 SELECT * INTO t FROM pg_trigger WHERE tgrelid='public.lab_arena_rounds'::regclass AND tgname='lab_arena_postpublication_reward_seal';
 IF FOUND AND(t.tgfoid IS DISTINCT FROM to_regprocedure('public.lab_arena_postpublication_reward_seal_v1()')
  OR t.tgenabled<>'O'OR t.tgtype<>19 OR t.tgqual IS NOT NULL OR t.tgnargs<>0 OR t.tgattr::TEXT<>'')THEN
  RAISE EXCEPTION 'postpublication678 installed trigger differs';END IF;
END;
$installed678$;

CREATE OR REPLACE FUNCTION public.lab_arena__postpublication_receipt_v1(h public.lab_arena_scoring_supersessions)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $receipt678$
 SELECT CASE WHEN h.recovery_doc->>'operation'='round_operational_update_v1'
  AND h.recovery_doc->>'recovery_kind'='operator_postpublication_judge_recovery_v1'
  AND jsonb_typeof(h.recovery_doc->'requests')='array' THEN coalesce(
   h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
   AND h.recovery_doc->>'round_sha256'='sha256:'||encode(extensions.digest(h.old_round_doc::TEXT,'sha256'),'hex')
   AND h.old_round_doc->>'round_id'=h.round_id AND h.new_round_doc->>'round_id'=h.round_id
   AND h.old_round_doc->>'status'='published'
   AND h.old_round_doc->'published_at'<>'null'::JSONB AND jsonb_typeof(h.old_round_doc->'publication_doc')='object'
   AND h.old_round_doc->'reward_activated_at'<>'null'::JSONB
   AND h.old_round_doc->'reward_basis_hash'<>'null'::JSONB AND jsonb_typeof(h.old_round_doc->'reward_basis_doc')='object'
   AND h.old_round_doc->>'king_outcome'='no_king'
   AND h.recovery_doc->'final_scoring_close'=h.new_round_doc#>'{configuration_doc,schedule,final_scoring_close}'
   AND h.recovery_doc->'publication_deadline'=h.new_round_doc#>'{configuration_doc,schedule,publication_deadline}'
   AND h.old_round_doc#>>'{publication_doc,king_decision,outcome}'='no_king'
   AND h.new_round_doc->>'status'='stage2_judged'
   AND (h.new_round_doc->>'status_generation')::BIGINT=(h.old_round_doc->>'status_generation')::BIGINT+1
   AND h.new_round_doc->'published_at'='null'::JSONB AND h.new_round_doc->'publication_doc'='null'::JSONB
   AND (h.old_round_doc-ARRAY['status','status_generation','published_at','publication_doc','configuration_doc'])=
       (h.new_round_doc-ARRAY['status','status_generation','published_at','publication_doc','configuration_doc'])
   AND ((h.old_round_doc->'configuration_doc')-'schedule')=((h.new_round_doc->'configuration_doc')-'schedule')
   AND ((h.old_round_doc#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline'])=
       ((h.new_round_doc#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline'])
   AND jsonb_array_length(h.recovery_doc->'requests') BETWEEN 1 AND 100,FALSE) ELSE FALSE END;
$receipt678$;
ALTER FUNCTION public.lab_arena__postpublication_receipt_v1(public.lab_arena_scoring_supersessions) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__postpublication_receipt_v1(public.lab_arena_scoring_supersessions) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__postpublication_judge_v1(r public.lab_arena_rounds,a public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $judge678$
 SELECT coalesce(r.status IN('stage2_judged','scored') AND r.published_at IS NULL AND r.publication_doc IS NULL
  AND r.reward_activated_at IS NOT NULL AND EXISTS(
   SELECT 1 FROM public.lab_arena_scoring_supersessions h
   CROSS JOIN LATERAL jsonb_array_elements(CASE WHEN jsonb_typeof(h.recovery_doc->'requests')='array'
     THEN h.recovery_doc->'requests' ELSE '[]'::JSONB END)item
   JOIN public.lab_arena_runs parent ON parent.run_id=item->>'parent_run_id'
   WHERE h.round_id=r.round_id AND public.lab_arena__postpublication_receipt_v1(h)
    AND (to_jsonb(r)-ARRAY['status','status_generation','updated_at'])=(h.new_round_doc-ARRAY['status','status_generation'])
    AND parent.round_id=r.round_id
    AND item->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
    AND (a.run_id=parent.run_id OR EXISTS(
     SELECT 1 FROM public.lab_arena_scoring_supersessions grant_h
     WHERE grant_h.round_id=r.round_id AND grant_h.request_id=item->>'request_id'
      AND grant_h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
      AND grant_h.request_hash='sha256:'||encode(extensions.digest(grant_h.recovery_doc::TEXT,'sha256'),'hex')
      AND grant_h.recovery_doc->>'parent_run_id'=parent.run_id
      AND grant_h.recovery_doc->>'parent_sha256'=item->>'parent_sha256'
      AND grant_h.recovery_doc->>'approval_ref'=h.recovery_doc->>'approval_ref'
      AND grant_h.recovery_doc->>'runtime_image_digest'=h.recovery_doc->>'runtime_image_digest'
      AND grant_h.recovery_doc->'checkpoints'=item->'checkpoints'
      AND grant_h.recovery_doc->'signal_checkpoints'=item->'signal_checkpoints'
      AND a.run_id=parent.assignment_id||':'||(parent.attempt+1)::TEXT
      AND grant_h.recovery_doc->'child_identity'=jsonb_build_object('run_id',a.run_id,'assignment_id',a.assignment_id,
       'round_id',a.round_id,'submission_id',a.submission_id,'miner_hotkey',a.miner_hotkey,'stage',a.stage,
       'icp_position',a.icp_position,'attempt',a.attempt,'scored_run_id',a.scored_run_id,'stage_generation',a.stage_generation,
       'scoring_generation_floor',a.scoring_generation_floor,'judgment_cache_key',a.judgment_cache_key,
       'judgment_input_hash',a.judgment_input_hash,'judgment_scope_doc',a.judgment_scope_doc,'company_judgment_refs',a.company_judgment_refs)
    ))),FALSE);
$judge678$;
ALTER FUNCTION public.lab_arena__postpublication_judge_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__postpublication_judge_v1(public.lab_arena_rounds,public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_recover_published_judgments_v1(
 p_round_id TEXT,p_round_sha256 TEXT,p_runtime_image_digest TEXT,p_final_scoring_close TIMESTAMPTZ,
 p_publication_deadline TIMESTAMPTZ,p_requests JSONB,p_request_id TEXT,p_actor_ref TEXT,p_approval_ref TEXT,
 p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $recover678$
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
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF NOT public.lab_arena__postpublication_receipt_v1(prior) OR prior.actor_ref IS DISTINCT FROM p_actor_ref
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
     AND g.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
     AND g.recovery_doc->>'parent_run_id'=item->>'parent_run_id'AND g.recovery_doc->>'parent_sha256'=item->>'parent_sha256'
     AND g.recovery_doc->>'runtime_image_digest'=p_runtime_image_digest AND g.recovery_doc->>'approval_ref'=p_approval_ref
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
  OR coalesce(r.publication_doc#>>'{king_decision,king_hotkey}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,king_submission_id}','')<>''
  OR coalesce(r.publication_doc#>>'{king_decision,winner_submission_id}','')<>''
  OR r.king_outcome IS DISTINCT FROM 'no_king' OR r.publication_doc#>>'{king_decision,outcome}'IS DISTINCT FROM 'no_king'
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
    AND h.recovery_doc->>'recovery_kind'='operator_postpublication_judge_recovery_v1')
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
 n:=r;n.status:='stage2_judged';n.status_generation:=r.status_generation+1;n.published_at:=NULL;n.publication_doc:=NULL;
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
  IF parent.round_id<>r.round_id OR parent.kind<>'score'OR parent.status<>'failed'OR parent.terminal_cause<>'judge_error'
   OR parent.result_doc->>'terminal_status'IS DISTINCT FROM 'judge_error'
   OR NOT((parent.result_doc#>>'{failure_diagnostic,stage}'IS NOT DISTINCT FROM 'scorer'AND parent.result_doc#>>'{failure_diagnostic,error_class}'IS NOT DISTINCT FROM 'judge_error')
     OR(parent.result_doc#>>'{failure_diagnostic,stage}'IS NOT DISTINCT FROM 'sandbox'AND parent.result_doc#>>'{failure_diagnostic,error_class}'IS NOT DISTINCT FROM 'judge_timeout'))
   OR parent.scoring_generation_floor<>r.scoring_generation_floor
   OR item->>'parent_sha256'IS DISTINCT FROM 'sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id AND(a.attempt>parent.attempt OR a.status='accepted'))
   OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id AND
    (h.request_id=item->>'request_id'OR(h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)))THEN
   RAISE EXCEPTION 'postpublication recovery exact unspent parent differs';END IF;
  child:=parent;child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT;child.attempt:=parent.attempt+1;child.stage_generation:=r.stage_generation;
  IF NOT public.lab_arena__judge_continuation_v1(n,child)THEN RAISE EXCEPTION 'postpublication recovery frozen continuation differs';END IF;
  SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
   INTO checkpoints FROM public.lab_arena_company_judgments j WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(parent.company_judgment_refs,'[]'))ref WHERE ref->>'cache_key'=j.cache_key);
  SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
   INTO signals FROM public.lab_arena_signal_judgments j WHERE j.source_scored_run_id=parent.scored_run_id;
  IF checkpoints IS DISTINCT FROM item->'checkpoints'OR signals IS DISTINCT FROM item->'signal_checkpoints'THEN RAISE EXCEPTION 'postpublication recovery approved checkpoints differ';END IF;
 END LOOP;
 IF p_dry_run THEN RETURN jsonb_build_object('status','dry_run','idempotent',FALSE,'request_id',p_request_id,'new_round',to_jsonb(n)-'updated_at','requests',p_requests);END IF;
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_postpublication_judge_recovery_v1',
  'round_sha256',round_hash,'runtime_image_digest',p_runtime_image_digest,'approval_ref',p_approval_ref,'requests',p_requests,
  'final_scoring_close',n.configuration_doc#>'{schedule,final_scoring_close}','publication_deadline',n.configuration_doc#>'{schedule,publication_deadline}');
 digest:='sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex');
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
  old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,digest,p_actor_ref,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',proof);
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,published_at=NULL,publication_doc=NULL,
  configuration_doc=n.configuration_doc WHERE round_id=r.round_id;
 FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
  grants:=grants||jsonb_build_array(public.lab_arena_grant_judge_budget_exception_v1(item->>'parent_run_id',item->>'parent_sha256',
   p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,item->'checkpoints',item->'signal_checkpoints',FALSE));
 END LOOP;
 RETURN jsonb_build_object('status','recovered','idempotent',FALSE,'request_id',p_request_id,'requests',p_requests,'grants',grants);
END;
$recover678$;
ALTER FUNCTION public.lab_arena_recover_published_judgments_v1(TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ,JSONB,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_recover_published_judgments_v1(TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ,JSONB,TEXT,TEXT,TEXT,BOOLEAN) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_postpublication_reward_seal_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $seal678$
DECLARE h public.lab_arena_scoring_supersessions;field TEXT;document JSONB:=to_jsonb(NEW);
BEGIN
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=NEW.round_id
  AND recovery_doc->>'recovery_kind'='operator_postpublication_judge_recovery_v1'ORDER BY generation_floor DESC LIMIT 1;
 IF NOT FOUND THEN RETURN NEW;END IF;
 IF NOT public.lab_arena__postpublication_receipt_v1(h)THEN RAISE EXCEPTION 'postpublication recovery receipt damaged';END IF;
 FOREACH field IN ARRAY ARRAY['reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at',
  'king_outcome','king_hotkey','king_start_epoch','promotion_required','promotion_doc','baseline_promoted_at',
  'champion_funding_frozen','champion_submission_id','champion_hotkey','champion_fallback_providers','rewards_enabled']LOOP
  IF document->field IS DISTINCT FROM h.old_round_doc->field THEN RAISE EXCEPTION 'postpublication sealed authority changed: %',field;END IF;
 END LOOP;
 IF ((document->'configuration_doc')-'schedule')IS DISTINCT FROM((h.old_round_doc->'configuration_doc')-'schedule')
  OR ((document#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline'])IS DISTINCT FROM
     ((h.old_round_doc#>'{configuration_doc,schedule}')-ARRAY['final_scoring_close','publication_deadline'])THEN
  RAISE EXCEPTION 'postpublication reward configuration changed';END IF;
 IF NEW.status='published'AND NEW.publication_doc->'king_decision'IS DISTINCT FROM h.old_round_doc#>'{publication_doc,king_decision}'THEN
  RAISE EXCEPTION 'postpublication recovery cannot change king decision';END IF;
 RETURN NEW;
END;
$seal678$;
ALTER FUNCTION public.lab_arena_postpublication_reward_seal_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_postpublication_reward_seal_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_postpublication_reward_seal ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_postpublication_reward_seal BEFORE UPDATE ON public.lab_arena_rounds
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_postpublication_reward_seal_v1();

-- Existing shared definitions must match exact committed pre/postimages.
DO $patch678$
DECLARE target RECORD;p RECORD;d TEXT;metadata JSONB;old_text TEXT;new_text TEXT;
BEGIN
 FOR target IN SELECT * FROM(VALUES
 ('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)','ea1287fd5ca2a7d1fae4da2bf623dfeb3a33b80d1b0a5f9634fd681e58d7b770','6ed2e46a2f323cf3c9ec21f8a4266f542a4cb302b097d66b27084504651f6c04','core'),
 ('public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','f329afb04a5f29afb0253d16fb63da00bc852782b22675f08bd7849c29213c3b','776b7e9e72019ac5c2a060e0d49e629a7bb1507e1617c495c6191fae163e9533','claim')
 )f(signature,preimage,postimage,kind)LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;d:=p.prosrc;metadata:=to_jsonb(p)-'prosrc';
  IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile<>'v'
   OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=CASE WHEN target.kind='core'THEN 'plpgsql'ELSE 'sql'END)OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'THEN RAISE EXCEPTION 'postpublication678 shared function security differs';END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE;END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN RAISE EXCEPTION 'postpublication678 shared function preimage differs';END IF;
  IF target.kind='core'THEN
   old_text:='AND r.reward_activated_at IS NULL';
   new_text:='AND (r.reward_activated_at IS NULL OR (p_recovery_kind=''operator_judge_budget_exception_v1'' AND public.lab_arena__postpublication_judge_v1(r,parent)))';
  ELSE
   old_text:='AND p_round.reward_activated_at IS NULL';
   new_text:='AND (p_round.reward_activated_at IS NULL OR public.lab_arena__postpublication_judge_v1(p_round,p_run))';
  END IF;
  IF(length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'postpublication678 shared anchor differs';END IF;
  d:=replace(d,old_text,new_text);
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'postpublication678 postimage differs';END IF;
  EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
  IF(SELECT to_jsonb(q)-'prosrc'FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
   OR(SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>target.postimage THEN RAISE EXCEPTION 'postpublication678 metadata changed';END IF;
 END LOOP;
END;
$patch678$;
DO $freshness678$
DECLARE target RECORD;p RECORD;d TEXT;metadata JSONB;old_text TEXT:=$oldfresh678$WHERE h.old_round_doc->>'status'='published'$oldfresh678$;
BEGIN
 FOR target IN SELECT * FROM(VALUES
 ('public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)','9f6e76c7f195b2699bf97c176e3a78572b903662c225a120ee3674711fcb68b4','4db417c9709302922407a7a0690093bfb23bf467c61f72b441b9fa7a558e5e58','{lab_arena_owner=X/lab_arena_owner}','s'),
 ('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)','550932f40a73eaf92578b5ac738b796e358dbf4c15fda4d8f5d5600c6f6afca6','fe2c881db64db4ff0999e276ebeaaa49636137276b4859b0c9ba6b37ee8bf2d5','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}','v'),
 ('public.lab_arena_freeze_champion_funding(text)','a415aef8b5f245039791e42f41f4c21826b0edf175bf58dd8f223d07832fbaea','b4ced40bc1ecf206d4caaa179789adcaae29c90fcaca42ac39896edbf8ba25bc','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}','v')
 )f(signature,preimage,postimage,acl,volatility)LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;d:=p.prosrc;metadata:=to_jsonb(p)-'prosrc';
  IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'OR p.proacl::TEXT IS DISTINCT FROM target.acl
   OR NOT p.prosecdef OR p.provolatile::TEXT<>target.volatility OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'THEN RAISE EXCEPTION 'postpublication678 freshness security differs';END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE;END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN RAISE EXCEPTION 'postpublication678 freshness preimage differs: %',target.signature;END IF;
  IF(length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'postpublication678 freshness anchor differs';END IF;
  d:=replace(d,old_text,old_text||E'\n      AND NOT public.lab_arena__postpublication_receipt_v1(h)');
  IF target.signature LIKE '%__baseline_source_current%'THEN
   IF(length(d)-length(replace(d,$pending678$AND pending.status NOT IN ('published','cancelled')$pending678$,'')))/length($pending678$AND pending.status NOT IN ('published','cancelled')$pending678$)<>1 THEN RAISE EXCEPTION 'postpublication678 prior reward anchor differs';END IF;
   d:=replace(d,$pending678$AND pending.status NOT IN ('published','cancelled')$pending678$,$newpending678$AND pending.status NOT IN ('published','cancelled')
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions recovery
        WHERE recovery.round_id=pending.round_id AND public.lab_arena__postpublication_receipt_v1(recovery)
         AND (to_jsonb(pending)-ARRAY['status','status_generation','updated_at'])=
             (recovery.new_round_doc-ARRAY['status','status_generation']))$newpending678$);
  END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'postpublication678 freshness postimage differs';END IF;
  EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
  IF(SELECT to_jsonb(q)-'prosrc'FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
   OR(SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>target.postimage THEN RAISE EXCEPTION 'postpublication678 freshness readback differs';END IF;
 END LOOP;
END;
$freshness678$;
DO $restore678$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena678_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena678_acl)THEN RAISE EXCEPTION 'postpublication678 schema ACL differs';END IF;
END;
$restore678$;
NOTIFY pgrst,'reload schema';COMMIT;
