-- Owner-only correction of an accepted, unpublished ICP after a proved scorer defect.
-- Preserve execution, frozen policy/floor, old judgments/costs, and unaffected caches.
-- One finite normal score child; no round replay, new retry service or publication reopen.
BEGIN;
SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena699_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $installed699$
DECLARE target RECORD; p RECORD;
BEGIN
 FOR target IN SELECT * FROM(VALUES
 ('public.lab_arena_correct_accepted_judgment_v1(text,text,text,text,jsonb,jsonb,jsonb,text,text,text,boolean)','6cd05936880a1549b909e8fffd8aeb1897a8aa6e2f607210f1fffd115f3f50c2','v','plpgsql','{lab_arena_owner=X/lab_arena_owner}'),
 ('public.lab_arena_retired_accepted_judgment_v1(text)','1f2b3468d4d2b50a69553dc379402b71fb0c60b1b03105e387caaa31034187a8','s','sql','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
 )f(signature,source_sha,volatility,language,acl) LOOP
  SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure(target.signature);
  IF FOUND AND(pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM target.acl
   OR NOT p.prosecdef OR p.provolatile::TEXT<>target.volatility
   OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=target.language)
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'
   OR (CASE WHEN target.language='plpgsql' THEN
        p.pronargdefaults<>1 OR pg_get_expr(p.proargdefaults,0) IS DISTINCT FROM 'true'
       ELSE p.pronargdefaults<>0 END)
   OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>target.source_sha) THEN
   RAISE EXCEPTION 'accepted correction699 installed function differs: %',target.signature; END IF;
 END LOOP;
END;
$installed699$;

DO $patch699$
DECLARE target RECORD; p RECORD; d TEXT; metadata JSONB; expected_acl TEXT;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 (
'public.lab_arena_runs_terminal_v1()','5f795a01c8ed4bf2280597d1f42404c4df46291886be9e7b1a83d35b0e09ea49','745d4b6a5e1a2e247a7054ce8540fad909691a7a18ec36c8bf684fecdd37a712',$old699_0$BEGIN
$old699_0$,$new699_0$BEGIN
  -- Accepted corrections retire only an exact owner-audited row in this transaction.
  IF TG_OP='UPDATE' AND OLD.kind='score' AND OLD.status='accepted' AND OLD.terminal_cause='accepted'
    AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
    AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
        AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
        AND h.recovery_doc->'superseded_judgment'=to_jsonb(OLD)
        AND NEW.terminal_doc=jsonb_build_object('operation','operator_accepted_judge_correction_v1',
          'request_id',h.request_id,'replacement_run_id',h.recovery_doc#>>'{child_identity,run_id}')) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new699_0$),
 (
'public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)','04af1799d0d06f5f5090b27daee4f472dc723e9688f92917a05e56b3c310fe26','7c978ff9163176e60815fef4b683c02bf6f458bd3bed275e8f6a53e1e7ad33a3',$old699_1$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1')$old699_1$,$new699_1$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1','operator_accepted_judge_correction_v1')$new699_1$),
 (
'public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','59f7ce07cf8e67b5f331cab59cbce45893cbd71d1ff063db3d38f89c5428f921','ab2d6ae78b02aa9ce843e1705b7596e5da02472c5f497fdf7165dd621f6a2039',$old699_2$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1')$old699_2$,$new699_2$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1','operator_accepted_judge_correction_v1')$new699_2$),
 (
'public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','3f3215228d3e79d3b7d204a9bdb6aba7f116b686efe32eabb21a77a4b1286031','4faf8a4d507630a676410ebfa5e17ad1c93cf99787982664d474a344487c4009',$old699_3$),FALSE);$old699_3$,$new699_3$),FALSE) OR coalesce(public.lab_arena__operator_scorer_bug_child_v1(p_run) AND EXISTS(
  SELECT 1 FROM public.lab_arena_scoring_supersessions h
  JOIN public.lab_arena_runs parent ON parent.run_id=h.recovery_doc->>'parent_run_id'
  WHERE h.round_id=p_run.round_id AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
    AND h.recovery_doc#>>'{child_identity,run_id}'=p_run.run_id
    AND h.recovery_doc->'organizer_openrouter_authorized'='true'::JSONB
    AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
    AND (to_jsonb(parent)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
      ((h.recovery_doc->'superseded_judgment')-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
    AND parent.status='failed' AND parent.terminal_cause='stage_closed'
    AND parent.terminal_doc->>'operation'='operator_accepted_judge_correction_v1'
    AND parent.terminal_doc->>'request_id'=h.request_id
    AND (public.lab_arena__operator_scorer_bug_host_v1(parent)
      OR public.lab_arena__nov06_organizer_openrouter665_v1(parent))),FALSE);$new699_3$),
 (
'public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamptz)','8a55713ca73871eddc438082750411cea633c691440d350c17efe40a402fdd53','d1e06c65df0b185a4e702010855165d2c2f0625f9dbe65053d5c0487c453d1ed',$old699_4$      AND accepted.company_input_hash = v_ref ->> 'company_input_hash'$old699_4$,$new699_4$      AND accepted.company_input_hash = v_ref ->> 'company_input_hash'
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h,
        jsonb_array_elements(h.recovery_doc->'affected_company_authorities') retired
        WHERE h.round_id=v_run.round_id
          AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
          AND retired->>'cache_key'=accepted.cache_key
          AND retired->>'company_input_hash'=accepted.company_input_hash
          AND (retired->>'authority_slot')::INTEGER=accepted.authority_slot
          AND retired->>'evidence_hash'=accepted.evidence_hash)
$new699_4$),
 (
'public.lab_arena_record_run_scores(text,smallint,jsonb)','4b30dc9e41852938c66bb180d2b3a046dbb237ca75c5cececbb648901f37ea08','7bab8b07289311950982265adb6b4b002b93515621e6a1a552a4798bb4a608ce',$old699_5$    IF v_round.scoring_generation_floor>0 AND v_run.status='accepted'$old699_5$,$new699_5$    -- The driver must bind a corrected projection to the accepted child it read,
    -- never an earlier accepted payload obtained before the owner correction.
    IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=p_round_id AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
        AND h.recovery_doc#>>'{child_identity,scored_run_id}'=v_run.run_id)
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs judge
        WHERE judge.run_id=v_score->>'score_run_id' AND judge.scored_run_id=v_run.run_id
          AND judge.round_id=p_round_id AND judge.kind='score' AND judge.status='accepted'
          AND judge.terminal_cause='accepted' AND judge.output_ref IS NOT NULL
          AND judge.scoring_generation_floor=v_round.scoring_generation_floor
          AND public.lab_arena__operator_scorer_bug_child_v1(judge)) THEN
      RAISE EXCEPTION 'lab_arena_corrected_score_authority' USING ERRCODE='42501';
    END IF;
    IF v_round.scoring_generation_floor>0 AND v_run.status='accepted'$new699_5$)
 ) edits(signature,preimage,postimage,old_text,new_text) LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;
  metadata:=to_jsonb(p)-'prosrc'; d:=pg_get_functiondef(p.oid);
  expected_acl:=CASE target.signature
   WHEN 'public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamptz)' THEN
    '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}'
   WHEN 'public.lab_arena_record_run_scores(text,smallint,jsonb)' THEN
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
   ELSE '{lab_arena_owner=X/lab_arena_owner}' END;
  IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM expected_acl THEN
   RAISE EXCEPTION 'accepted correction699 owner/ACL differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
    OR (length(d)-length(replace(d,target.old_text,'')))/length(target.old_text)<>1 THEN
    RAISE EXCEPTION 'accepted correction699 preimage differs: %',target.signature; END IF;
  d:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'accepted correction699 patch differs'; END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>target.postimage THEN
    RAISE EXCEPTION 'accepted correction699 metadata changed'; END IF;
 END LOOP;
END;
$patch699$;

CREATE OR REPLACE FUNCTION public.lab_arena_correct_accepted_judgment_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_source_sha256 TEXT,p_runtime_image_digest TEXT,
 p_affected_company_authorities JSONB,p_checkpoints JSONB,p_signal_checkpoints JSONB,
 p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $grant699$
DECLARE parent public.lab_arena_runs; source public.lab_arena_runs; child public.lab_arena_runs;
 r public.lab_arena_rounds; mapping public.lab_arena_reviewer_runtimes; prior public.lab_arena_scoring_supersessions;
 company_entry public.lab_arena_company_judgments; item JSONB; proof JSONB; response JSONB; checkpoints JSONB;
 signal_checkpoints JSONB; revision BIGINT; organizer BOOLEAN; terminal JSONB;
BEGIN
 IF p_parent_run_id IS NULL OR p_parent_sha256 IS NULL OR p_source_sha256 IS NULL
  OR p_parent_sha256 !~ '^sha256:[0-9a-f]{64}$' OR p_source_sha256 !~ '^sha256:[0-9a-f]{64}$'
  OR p_runtime_image_digest IS NULL OR p_runtime_image_digest !~ '^sha256:[0-9a-f]{64}$'
  OR jsonb_typeof(p_affected_company_authorities) IS DISTINCT FROM 'array'
  OR jsonb_array_length(p_affected_company_authorities) NOT BETWEEN 1 AND 100
  OR jsonb_typeof(p_checkpoints) IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_signal_checkpoints) IS DISTINCT FROM 'object'
  OR p_request_id IS NULL OR length(btrim(p_request_id)) NOT BETWEEN 1 AND 160
  OR p_actor_ref IS NULL OR length(btrim(p_actor_ref)) NOT BETWEEN 1 AND 200
  OR p_correction_ref IS NULL OR length(btrim(p_correction_ref)) NOT BETWEEN 1 AND 500
  OR p_dry_run IS NULL THEN RAISE EXCEPTION 'accepted correction request invalid'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=parent.round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions
  WHERE round_id=r.round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'operator_accepted_judge_correction_v1'
   OR prior.recovery_doc->>'parent_run_id' IS DISTINCT FROM p_parent_run_id
   OR prior.recovery_doc->>'parent_sha256' IS DISTINCT FROM p_parent_sha256
   OR prior.recovery_doc->>'source_sha256' IS DISTINCT FROM p_source_sha256
   OR prior.recovery_doc->>'runtime_image_digest' IS DISTINCT FROM p_runtime_image_digest
   OR prior.recovery_doc->'affected_company_authorities' IS DISTINCT FROM p_affected_company_authorities
   OR prior.recovery_doc->'checkpoints' IS DISTINCT FROM p_checkpoints
   OR prior.recovery_doc->'signal_checkpoints' IS DISTINCT FROM p_signal_checkpoints
   OR prior.actor_ref IS DISTINCT FROM p_actor_ref
   OR prior.recovery_doc->>'correction_ref' IS DISTINCT FROM p_correction_ref THEN
   RAISE EXCEPTION 'accepted correction replay differs'; END IF;
  SELECT * INTO STRICT child FROM public.lab_arena_runs WHERE run_id=prior.recovery_doc#>>'{child_identity,run_id}';
  IF NOT public.lab_arena__operator_scorer_bug_child_v1(child) THEN RAISE EXCEPTION 'accepted correction child differs'; END IF;
  RETURN jsonb_build_object('status','granted','idempotent',TRUE,'request_id',p_request_id,
   'child_identity',prior.recovery_doc->'child_identity','checkpoints',p_checkpoints,
   'signal_checkpoints',p_signal_checkpoints,'affected_company_authorities',p_affected_company_authorities,
   'runtime_image_digest',p_runtime_image_digest,'organizer_openrouter_authorized',prior.recovery_doc->'organizer_openrouter_authorized');
 END IF;
 IF current_setting('session_replication_role')<>'origin' THEN RAISE EXCEPTION 'accepted correction requires protected writes'; END IF;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=parent.scored_run_id FOR UPDATE;
 IF ('sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')) IS DISTINCT FROM p_parent_sha256
  OR ('sha256:'||encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex')) IS DISTINCT FROM p_source_sha256
  OR parent.kind<>'score' OR parent.status<>'accepted' OR parent.terminal_cause<>'accepted'
  OR parent.output_ref IS NULL OR parent.result_doc->>'terminal_status' IS DISTINCT FROM 'accepted'
  OR parent.scoring_generation_floor<>r.scoring_generation_floor OR parent.attempt>=2147483647
  OR source.kind<>'execute' OR source.status<>'accepted' OR source.terminal_cause<>'accepted' OR source.output_ref IS NULL
  OR (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position)
    IS DISTINCT FROM (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
  OR source.assignment_id IS DISTINCT FROM source.round_id||':'||source.submission_id||':'||source.stage::TEXT||':'||source.icp_position::TEXT
  OR source.run_id IS DISTINCT FROM source.assignment_id||':'||source.attempt::TEXT
  OR parent.assignment_id IS DISTINCT FROM source.assignment_id||':score'
  OR parent.run_id IS DISTINCT FROM parent.assignment_id||':'||parent.attempt::TEXT
  OR parent.judgment_cache_key IS NOT NULL OR parent.judgment_input_hash IS NOT NULL OR parent.judgment_scope_doc IS NOT NULL
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants) participant
    WHERE participant->>'submission_id'=parent.submission_id AND participant->>'miner_hotkey'=parent.miner_hotkey)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs other WHERE other.assignment_id=parent.assignment_id
    AND (other.attempt>parent.attempt OR other.status IN('pending','leased','submitted')))
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=r.round_id
    AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
    AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)
  OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id)
  OR NOT (CASE WHEN r.configuration_doc ? 'company_quality_policy'
    OR (r.configuration_doc->'scorer_policy') ? 'company_quality_policy'
    THEN public.lab_arena__company_quality_refs_v1(r,parent.company_judgment_refs)
    ELSE public.lab_arena__company_judgment_refs_v1(r,parent.company_judgment_refs) END)
  OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce((CASE parent.stage WHEN 1 THEN
    r.stage1_scoring_plan_doc ELSE r.stage2_scoring_plan_doc END)->'work_items','[]')) planned
    WHERE planned->>'scored_run_id'=parent.scored_run_id AND planned->>'submission_id'=parent.submission_id
      AND planned->>'icp_position'=parent.icp_position::TEXT AND planned->>'output_ref'=source.output_ref) THEN
  RAISE EXCEPTION 'accepted correction source, latest authority or calls differs'; END IF;
 IF (r.published_at IS NULL AND r.publication_doc IS NULL AND r.reward_activated_at IS NULL
  AND r.reward_basis_doc IS NULL AND r.promotion_doc IS NULL AND r.baseline_promoted_at IS NULL
  AND ((parent.stage=1 AND r.status IN('stage1_scoring','stage1_judged','stage1_scored','stage2','stage2_closed','stage2_scoring','stage2_judged','scored'))
    OR (parent.stage=2 AND r.status IN('stage2_scoring','stage2_judged','scored')))
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  AND public.lab_arena_judge_continuation_enabled_v1(r.round_id)) IS NOT TRUE THEN
  RAISE EXCEPTION 'accepted correction round or deadline differs'; END IF;
 SELECT * INTO STRICT mapping FROM public.lab_arena_reviewer_runtimes
  WHERE grading_image_digest=r.configuration_doc->>'scorer_image_digest'
   AND grading_image_reference=r.configuration_doc->>'scorer_image_reference' FOR SHARE;
 IF mapping.runtime_image_digest IS DISTINCT FROM p_runtime_image_digest
  OR mapping.runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',
    r.configuration_doc->>'scorer_image_digest') THEN RAISE EXCEPTION 'accepted correction runtime differs'; END IF;
 IF (SELECT count(DISTINCT x->>'company_index') FROM jsonb_array_elements(p_affected_company_authorities)x)
    <>jsonb_array_length(p_affected_company_authorities) THEN RAISE EXCEPTION 'accepted correction duplicate company'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(p_affected_company_authorities) LOOP
  IF jsonb_typeof(item) IS DISTINCT FROM 'object' OR jsonb_typeof(item->'company_index') IS DISTINCT FROM 'number'
   OR jsonb_typeof(item->'authority_slot') IS DISTINCT FROM 'number'
   OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(item)k) IS DISTINCT FROM
      ARRAY['authority_slot','cache_key','company_index','company_input_hash','evidence_hash']::TEXT[]
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(parent.company_judgment_refs) ref
    WHERE ref->'company_index'=item->'company_index' AND ref->>'cache_key'=item->>'cache_key'
      AND ref->>'company_input_hash'=item->>'company_input_hash')
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(
     coalesce(parent.claim_response#>'{company_judgment_cache,hits}','[]')||
     coalesce(parent.claim_response#>'{company_judgment_cache,misses}','[]')) claimed
    WHERE claimed->'company_index'=item->'company_index' AND claimed->>'cache_key'=item->>'cache_key'
      AND claimed->'authority_slot'=item->'authority_slot'
      AND claimed->>'company_input_hash'=item->>'company_input_hash') THEN
   RAISE EXCEPTION 'accepted correction company claim differs'; END IF;
  SELECT * INTO STRICT company_entry FROM public.lab_arena_company_judgments
   WHERE cache_key=item->>'cache_key' AND authority_slot=(item->>'authority_slot')::INTEGER FOR SHARE;
  IF company_entry.company_input_hash IS DISTINCT FROM item->>'company_input_hash'
   OR company_entry.evidence_hash IS DISTINCT FROM item->>'evidence_hash'
   OR company_entry.scope_doc->>'round_id' IS DISTINCT FROM r.round_id THEN
   RAISE EXCEPTION 'accepted correction immutable company differs'; END IF;
 END LOOP;
 SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
  INTO checkpoints FROM public.lab_arena_company_judgments j
  WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(parent.company_judgment_refs) ref WHERE ref->>'cache_key'=j.cache_key);
 SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
  INTO signal_checkpoints FROM public.lab_arena_signal_judgments j WHERE j.source_scored_run_id=parent.scored_run_id;
 IF checkpoints IS DISTINCT FROM p_checkpoints OR signal_checkpoints IS DISTINCT FROM p_signal_checkpoints THEN
  RAISE EXCEPTION 'accepted correction checkpoints differ'; END IF;
 child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT;
 child.attempt:=parent.attempt+1; child.stage_generation:=r.stage_generation;
 organizer:=public.lab_arena__operator_scorer_bug_host_v1(parent) OR public.lab_arena__nov06_organizer_openrouter665_v1(parent);
 proof:=jsonb_build_object('operation','round_operational_update_v1','recovery_kind','operator_accepted_judge_correction_v1',
  'parent_run_id',parent.run_id,'parent_sha256',p_parent_sha256,'source_sha256',p_source_sha256,
  'superseded_judgment',to_jsonb(parent),'correction_ref',p_correction_ref,
  'affected_company_authorities',p_affected_company_authorities,'checkpoints',checkpoints,'signal_checkpoints',signal_checkpoints,
  'scoring_generation_floor',r.scoring_generation_floor,'runtime_image_digest',mapping.runtime_image_digest,
  'runtime_image_reference',mapping.runtime_image_reference,'organizer_openrouter_authorized',organizer,
  'child_identity',jsonb_build_object('run_id',child.run_id,'assignment_id',child.assignment_id,'round_id',child.round_id,
   'submission_id',child.submission_id,'miner_hotkey',child.miner_hotkey,'stage',child.stage,'icp_position',child.icp_position,
   'attempt',child.attempt,'scored_run_id',child.scored_run_id,'stage_generation',child.stage_generation,
   'scoring_generation_floor',child.scoring_generation_floor,'judgment_cache_key',child.judgment_cache_key,
   'judgment_input_hash',child.judgment_input_hash,'judgment_scope_doc',child.judgment_scope_doc,'company_judgment_refs',child.company_judgment_refs));
 response:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'granted' END,'idempotent',FALSE,
  'request_id',p_request_id,'child_identity',proof->'child_identity','checkpoints',checkpoints,'signal_checkpoints',signal_checkpoints,
  'affected_company_authorities',p_affected_company_authorities,'runtime_image_digest',mapping.runtime_image_digest,
  'organizer_openrouter_authorized',organizer);
 IF p_dry_run THEN RETURN response; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),p_actor_ref,
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at',jsonb_build_object(source.run_id,
   jsonb_build_object('per_icp_score',source.per_icp_score,'qualification_doc',source.qualification_doc)),proof);
 terminal:=jsonb_build_object('operation','operator_accepted_judge_correction_v1','request_id',p_request_id,'replacement_run_id',child.run_id);
 UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc=terminal WHERE run_id=parent.run_id;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=source.run_id;
 IF NOT public.lab_arena__judge_continuation_v1(r,child) THEN RAISE EXCEPTION 'accepted correction frozen continuation differs'; END IF;
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,
  status,stage_generation,scoring_generation_floor,kind,scored_run_id,previous_runner_hotkey,judgment_cache_key,
  judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,company_judgment_refs)
 VALUES(child.run_id,parent.assignment_id,parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,
  parent.icp_position,child.attempt,'pending',r.stage_generation,r.scoring_generation_floor,'score',parent.scored_run_id,
  parent.runner_hotkey,parent.judgment_cache_key,parent.judgment_input_hash,parent.judgment_scope_doc,
  parent.judgment_group_leader,parent.judgment_group_miner_hotkeys,parent.company_judgment_refs);
 RETURN response;
END;
$grant699$;
ALTER FUNCTION public.lab_arena_correct_accepted_judgment_v1(TEXT,TEXT,TEXT,TEXT,JSONB,JSONB,JSONB,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_correct_accepted_judgment_v1(TEXT,TEXT,TEXT,TEXT,JSONB,JSONB,JSONB,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Immutable originals remain valid evidence for unaffected company checkpoints.
-- The service reads only a boolean; full audit documents stay owner-private.
CREATE OR REPLACE FUNCTION public.lab_arena_retired_accepted_judgment_v1(p_run_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $retired699$
 SELECT EXISTS(SELECT 1 FROM public.lab_arena_runs r
  JOIN public.lab_arena_scoring_supersessions h ON h.round_id=r.round_id
  WHERE r.run_id=p_run_id AND r.kind='score' AND r.status='failed' AND r.terminal_cause='stage_closed'
    AND r.terminal_doc=jsonb_build_object('operation','operator_accepted_judge_correction_v1',
      'request_id',h.request_id,'replacement_run_id',h.recovery_doc#>>'{child_identity,run_id}')
    AND h.recovery_doc->>'operation'='round_operational_update_v1'
    AND h.recovery_doc->>'recovery_kind'='operator_accepted_judge_correction_v1'
    AND h.recovery_doc->>'parent_run_id'=r.run_id
    AND h.recovery_doc#>>'{superseded_judgment,status}'='accepted'
    AND h.recovery_doc#>>'{superseded_judgment,terminal_cause}'='accepted'
    AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
    AND (to_jsonb(r)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
      ((h.recovery_doc->'superseded_judgment')-ARRAY['status','terminal_cause','terminal_doc','updated_at']));
$retired699$;
ALTER FUNCTION public.lab_arena_retired_accepted_judgment_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_retired_accepted_judgment_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_retired_accepted_judgment_v1(TEXT) TO lab_arena_service;

DO $restore699$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena699_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena699_acl) THEN
  RAISE EXCEPTION 'accepted correction699 schema ACL differs'; END IF;
END;
$restore699$;
NOTIFY pgrst,'reload schema'; COMMIT;
