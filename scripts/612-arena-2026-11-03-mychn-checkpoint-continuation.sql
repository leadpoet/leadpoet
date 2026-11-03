-- One exact MyCHN checkpoint continuation after five failed judgments.
-- Keep five attempts, all charges and the completed company/signal judgments.
-- Normal completion/expiry ceilings cannot append attempt seven.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena612_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight612$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_mychn_judge612_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'8405717179732f9ee876d7e4589c520e2dabb652f46f66cb3b23831720f1ef40'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_preflight612$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov03_mychn_judge612_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $judge612$
DECLARE h public.lab_arena_scoring_supersessions; source public.lab_arena_runs;
 parent public.lab_arena_runs; proof JSONB; actual JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03' OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_round.published_at IS NOT NULL OR p_round.publication_doc IS NOT NULL OR p_round.reward_activated_at IS NOT NULL
  OR p_run.run_id IS DISTINCT FROM 'arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:6'
  OR p_run.assignment_id IS DISTINCT FROM 'arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score'
  OR p_run.scored_run_id IS DISTINCT FROM 'arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:2'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id OR p_run.submission_id IS DISTINCT FROM 'sub-0df2d0295c950d907e094e258bf98905'
  OR p_run.stage IS DISTINCT FROM 2 OR p_run.icp_position IS DISTINCT FROM 7
  OR p_run.attempt IS DISTINCT FROM 6 OR p_run.scoring_generation_floor IS DISTINCT FROM 0
  OR p_run.kind IS DISTINCT FROM 'score' OR p_run.status NOT IN('pending','leased','submitted')
  OR clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
  OR p_run.judgment_scope_doc IS NOT NULL
  OR NOT public.lab_arena__company_judgment_refs_v1(p_round,p_run.company_judgment_refs)
  OR public.lab_arena__score_attempt_count_v1(p_run) IS DISTINCT FROM 6 THEN RETURN FALSE; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-mychn-checkpoint612-v1'
  AND recovery_doc->>'operation'='nov03_mychn_checkpoint_continuation612_v1'
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 proof:=h.recovery_doc->'checkpoint_judge';
 IF proof->>'runtime_image_digest' IS DISTINCT FROM 'sha256:1b871e1c80e6d3c4095b14d73166dc5d37dde5f5002aa622583d4350b198fa4b'
  OR proof->>'runtime_image_reference' IS DISTINCT FROM '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:1b871e1c80e6d3c4095b14d73166dc5d37dde5f5002aa622583d4350b198fa4b' THEN RETURN FALSE; END IF;
 IF p_run.status='pending' THEN
  IF NOT public.lab_arena__judge_calls_clear_v1(p_run.assignment_id) THEN RETURN FALSE; END IF;
  IF NOT EXISTS(SELECT 1 FROM public.lab_arena_reviewer_runtimes mapping
   WHERE mapping.grading_image_digest=p_round.configuration_doc->>'scorer_image_digest'
    AND mapping.grading_image_reference=p_round.configuration_doc->>'scorer_image_reference'
    AND mapping.runtime_image_digest=proof->>'runtime_image_digest'
    AND mapping.runtime_image_reference=proof->>'runtime_image_reference') THEN RETURN FALSE; END IF;
 ELSE
  IF p_run.reviewer_runtime_doc IS DISTINCT FROM jsonb_build_object(
   'schema_version','leadpoet.lab_arena.reviewer_runtime.v1','run_id',p_run.run_id,
   'lease_generation',p_run.lease_generation,
   'grading_image_digest',p_round.configuration_doc->>'scorer_image_digest',
   'grading_image_reference',p_round.configuration_doc->>'scorer_image_reference',
   'runtime_image_digest',proof->>'runtime_image_digest',
   'runtime_image_reference',proof->>'runtime_image_reference') THEN RETURN FALSE; END IF;
 END IF;
 SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
 SELECT * INTO parent FROM public.lab_arena_runs WHERE run_id=p_run.assignment_id||':5';
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_runs a WHERE assignment_id=p_run.assignment_id AND attempt<=5;
 IF actual IS DISTINCT FROM proof->'prior_run_hashes' THEN RETURN FALSE; END IF;
 SELECT jsonb_object_agg(j.cache_key,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_company_judgments j
 WHERE (proof->'lab_arena_company_judgments_row_hashes') ? j.cache_key AND j.authority_slot=0;
 IF actual IS DISTINCT FROM proof->'lab_arena_company_judgments_row_hashes' THEN RETURN FALSE; END IF;
 SELECT jsonb_object_agg(j.cache_key,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_signal_judgments j
 WHERE (proof->'lab_arena_signal_judgments_row_hashes') ? j.cache_key AND j.authority_slot=0;
 IF actual IS DISTINCT FROM proof->'lab_arena_signal_judgments_row_hashes' THEN RETURN FALSE; END IF;
 RETURN coalesce(source.kind='execute' AND source.status='accepted' AND source.terminal_cause='accepted'
  AND source.output_ref IS NOT NULL AND source.per_icp_score IS NULL AND source.qualification_doc IS NULL
  AND proof->>'source_sha256'=encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex')
  AND parent.status='failed' AND parent.terminal_cause='judge_error'
  AND parent.result_doc->>'terminal_status'='judge_error'
  AND parent.result_doc#>>'{failure_diagnostic,error_class}'='judge_error'
  AND parent.result_doc#>>'{failure_diagnostic,stage}'='scorer'
  AND (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position,parent.scored_run_id)
   IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position,p_run.scored_run_id)
  AND parent.company_judgment_refs=p_run.company_judgment_refs
  AND proof->'company_judgment_refs'=p_run.company_judgment_refs
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs j WHERE j.assignment_id=p_run.assignment_id AND j.status='accepted'),FALSE);
END;
$judge612$;
ALTER FUNCTION public.lab_arena__nov03_mychn_judge612_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_mychn_judge612_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify612$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_mychn_judge612_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'8405717179732f9ee876d7e4589c520e2dabb652f46f66cb3b23831720f1ef40'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_verify612$;

DO $patch612$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','3b6e313b2c109a0004eb6c3a48a925d5d8ad2da77a89eee7835223e2ad13df9e','db2e3c1c4e4c57edff03a242bcae55739687ae2f018566e3076e8c7554c1b5b5','v','plpgsql',$old612$      OR public.lab_arena__nov03_checkpoint_judge610_v1(v_round,runs))$old612$,$new612$      OR public.lab_arena__nov03_checkpoint_judge610_v1(v_round,runs)
      OR public.lab_arena__nov03_mychn_judge612_v1(v_round,runs))$new612$),
 ('public.lab_arena_scoring_policy_history_v1(text)','a2bbd7410aacc8b1e4f814f2c475b8c8c8b6f2b689e7f39fb79b9eeace92874d','ac48e548e3b47234e4839f9d9d3c1e36bf241693e1e945d78f92f57477873102','s','sql',$old612$'nov03_g3_natural_expiry_executions611_v1') THEN$old612$,$new612$'nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1') THEN$new612$)
 ) AS patches(signature,preimage,postimage,volatility,language,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],target.language) THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>length(target.old_text) THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch612$;
-- Exact one-time append. This transaction does not drain unrelated jobs.
DO $checkpoint_attempt612$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 assignment CONSTANT TEXT:='arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score';
 source_id CONSTANT TEXT:='arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:2';
 manifest CONSTANT JSONB:=$manifest612${"company_judgment_refs":[{"cache_key":"sha256:6e06bd7d1224a35e90bcf1c388c733ef56fb71d336ca84cb55495496037976db","company_index":0,"company_input_hash":"sha256:bb997de5271375deddb532093cb73bcbb4309a7e60bd3c20997a15916c2233e3","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:6e06bd7d1224a35e90bcf1c388c733ef56fb71d336ca84cb55495496037976db","company_input_hash":"sha256:bb997de5271375deddb532093cb73bcbb4309a7e60bd3c20997a15916c2233e3","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:dc7514cda3a446336ab6d31d1c9faba3007e1fcb3e696e7126593cbd3b809ae4","company_index":1,"company_input_hash":"sha256:0c92bb9bc20a49fe796797711044783190803d4a2698d4f4680fea9d125882c9","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:dc7514cda3a446336ab6d31d1c9faba3007e1fcb3e696e7126593cbd3b809ae4","company_input_hash":"sha256:0c92bb9bc20a49fe796797711044783190803d4a2698d4f4680fea9d125882c9","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:b528a2b1d1218199be70c05b4539c9e7ad2ec083f2f1896982b4966d3c357fc6","company_index":2,"company_input_hash":"sha256:3123f0f70e04bf9c143e4455c5051fba8af7f85f0320601322b7ca909bd7c269","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:b528a2b1d1218199be70c05b4539c9e7ad2ec083f2f1896982b4966d3c357fc6","company_input_hash":"sha256:3123f0f70e04bf9c143e4455c5051fba8af7f85f0320601322b7ca909bd7c269","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:5776c85db89cd9c11452263a73b380af9bbb7e29d530cb5ab2b0ead794a62fa1","company_index":3,"company_input_hash":"sha256:4a2f8b6fa486c79f38b223d080c2bba3ace473177a58798c1bb2674d1843cbb5","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:5776c85db89cd9c11452263a73b380af9bbb7e29d530cb5ab2b0ead794a62fa1","company_input_hash":"sha256:4a2f8b6fa486c79f38b223d080c2bba3ace473177a58798c1bb2674d1843cbb5","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:19c768876fb1f1bd239cac4ac839d1ef9327782eeb0c2ba35a64b436abef25e8","company_index":4,"company_input_hash":"sha256:1cd158ec13ae08db267d741fcbf116b88c6307a5fb740d55c3abdd0cbdb70535","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:19c768876fb1f1bd239cac4ac839d1ef9327782eeb0c2ba35a64b436abef25e8","company_input_hash":"sha256:1cd158ec13ae08db267d741fcbf116b88c6307a5fb740d55c3abdd0cbdb70535","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}}],"lab_arena_company_judgments_row_hashes":{"sha256:6e06bd7d1224a35e90bcf1c388c733ef56fb71d336ca84cb55495496037976db":"e8cda6d9342a5a03fc2147598c22103b208e397da6ff77ecab4dbbeee79be033"},"lab_arena_signal_judgments_row_hashes":{"sha256:c1ef2c699cbfa5131fb8a30d881db6cc75b13552cf1a7cc17899910e99049e22":"565fa73fbf184148bd2a63eff07765ac8ef341dbb9d942f6e974fe9466b4f01e"},"prior_run_hashes":{"arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:1":"191b642817e259833ee2bf86f1932c180a9cd41d5d7d4db5eab9d55ac9c5bf63","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:2":"c2ebd197944cc46c704dc83711b8e80bd09f6685d649a59fcf31299ed14322a1","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:3":"36129b6c9ca1ed0a59ef8facebcdf3795986519323177951b4865d44b7c75d1d","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:4":"5e2900607aa4006b5a5fedfc525b4fb4e72f40b30ad98829d8da61a540ff3b35","arena-2026-11-03:sub-0df2d0295c950d907e094e258bf98905:2:7:score:5":"17a1484e6e27cb05c6a31211d6095b03a4522a9cc3fc5f1ae6953bb2e0d326d8"},"runtime_image_digest":"sha256:1b871e1c80e6d3c4095b14d73166dc5d37dde5f5002aa622583d4350b198fa4b","runtime_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:1b871e1c80e6d3c4095b14d73166dc5d37dde5f5002aa622583d4350b198fa4b","source_sha256":"cacbb75b593ecbe40c7eba24b39c69cba7c54ae7cb5ca6854e4dd63aa1ae4795"}$manifest612$::JSONB;
 r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 source public.lab_arena_runs; parent public.lab_arena_runs; successor public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; revision BIGINT; actual JSONB; request_hash TEXT;
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 PERFORM 1 FROM public.lab_arena_runs WHERE assignment_id=assignment ORDER BY attempt FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=source_id FOR UPDATE;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=assignment||':5';
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=rid AND request_id='nov03-mychn-checkpoint612-v1';
 IF FOUND THEN
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_mychn_checkpoint_continuation612_v1'
   OR h.recovery_doc->'checkpoint_judge' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR successor.scored_run_id IS DISTINCT FROM source_id OR successor.attempt IS DISTINCT FROM 6
   OR successor.company_judgment_refs IS DISTINCT FROM parent.company_judgment_refs
   OR successor.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>6) THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge replay differs'; END IF;
  RETURN;
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR c.operator_paused
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2_scoring' OR r.stage_generation IS DISTINCT FROM 6 OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM '9c396e3565a7f9d84d636cc55537ac95968d97770f5de798b1f45b56fbfc8d57'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.configuration_doc#>>'{schedule,final_scoring_close}' IS DISTINCT FROM '2026-11-03T20:30:02Z'
  OR r.configuration_doc#>>'{schedule,publication_deadline}' IS DISTINCT FROM '2026-11-03T20:30:03Z'
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR (r.configuration_doc->>'scoring_cap_microusd')::BIGINT IS DISTINCT FROM 50000000
  OR r.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM 'sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_reviewer_runtimes mapping
   WHERE mapping.grading_image_digest=r.configuration_doc->>'scorer_image_digest'
    AND mapping.grading_image_reference=r.configuration_doc->>'scorer_image_reference'
    AND mapping.runtime_image_digest=manifest->>'runtime_image_digest'
    AND mapping.runtime_image_reference=manifest->>'runtime_image_reference')
  OR source.kind IS DISTINCT FROM 'execute' OR source.status IS DISTINCT FROM 'accepted'
  OR source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
  OR source.per_icp_score IS NOT NULL OR source.qualification_doc IS NOT NULL
  OR NOT public.lab_arena__judge_calls_clear_v1(assignment)
  OR encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'source_sha256'
  OR parent.kind IS DISTINCT FROM 'score' OR parent.status IS DISTINCT FROM 'failed'
  OR parent.terminal_cause IS DISTINCT FROM 'judge_error' OR parent.attempt IS DISTINCT FROM 5
  OR parent.result_doc->>'terminal_status' IS DISTINCT FROM 'judge_error'
  OR parent.result_doc#>>'{failure_diagnostic,error_class}' IS DISTINCT FROM 'judge_error'
  OR parent.result_doc#>>'{failure_diagnostic,stage}' IS DISTINCT FROM 'scorer'
  OR public.lab_arena__score_attempt_count_v1(parent) IS DISTINCT FROM 5
  OR parent.scored_run_id IS DISTINCT FROM source.run_id
  OR (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position)
   IS DISTINCT FROM (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
  OR parent.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
  OR NOT public.lab_arena__company_judgment_refs_v1(r,parent.company_judgment_refs)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND (attempt>5 OR status IN('accepted','leased','submitted'))) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge exact source, runtime, guard or clock differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_runs a WHERE assignment_id=assignment;
 IF actual IS DISTINCT FROM manifest->'prior_run_hashes' THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge five prior attempts differ'; END IF;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'nov03-mychn-checkpoint612-v1',request_hash,'nov03-mychn-checkpoint612',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',
  jsonb_build_object('operation','nov03_mychn_checkpoint_continuation612_v1','scoring_generation_floor',0,
   'checkpoint_judge',manifest,'reason','One explicit checkpoint-backed sixth after the confirmed old-homepage vanity proof omission. Use only the reviewed four-file runtime, reuse the completed company and signal judgments, and preserve five paid failures, the source, grading/cache identity, deadlines and $50 scoring cap. No automatic seventh attempt.'));
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
  stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,previous_runner_hotkey,company_judgment_refs)
 VALUES(assignment||':6',assignment,rid,source.submission_id,source.miner_hotkey,2,7,6,'pending',
  r.stage_generation,'score',source.run_id,parent.runner_hotkey,parent.company_judgment_refs);
 SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
 IF NOT public.lab_arena__nov03_mychn_judge612_v1(r,successor)
  OR NOT public.lab_arena__judge_continuation_v1(r,successor)
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge final authority or preserved control differs'; END IF;
END;
$checkpoint_attempt612$;

DO $restore612$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena612_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena612_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge schema ACL differs'; END IF;
END;
$restore612$;
NOTIFY pgrst,'reload schema';
COMMIT;
