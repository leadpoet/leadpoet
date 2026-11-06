-- One exact Captify checkpoint continuation after five unfinished judgments.
-- Keep five attempts, all charges and the two completed company/signal judgments.
-- Normal completion/expiry ceilings cannot append attempt seven.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena614_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight614$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_captify_judge614_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'ed0c282a20af8913074624efda874145267e279cd500cb08fd136d95caf4eedd'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_preflight614$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov03_captify_judge614_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $judge614$
DECLARE h public.lab_arena_scoring_supersessions; source public.lab_arena_runs;
 parent public.lab_arena_runs; proof JSONB; actual JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03' OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_round.published_at IS NOT NULL OR p_round.publication_doc IS NOT NULL OR p_round.reward_activated_at IS NOT NULL
  OR p_run.run_id IS DISTINCT FROM 'arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:6'
  OR p_run.assignment_id IS DISTINCT FROM 'arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score'
  OR p_run.scored_run_id IS DISTINCT FROM 'arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:1'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id OR p_run.submission_id IS DISTINCT FROM 'sub-6206272caf2b1a19ee2a7e843dc518cf'
  OR p_run.stage IS DISTINCT FROM 2 OR p_run.icp_position IS DISTINCT FROM 3
  OR p_run.attempt IS DISTINCT FROM 6 OR p_run.scoring_generation_floor IS DISTINCT FROM 0
  OR p_run.kind IS DISTINCT FROM 'score' OR p_run.status NOT IN('pending','leased','submitted')
  OR clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
  OR p_run.judgment_scope_doc IS NOT NULL
  OR NOT public.lab_arena__company_judgment_refs_v1(p_round,p_run.company_judgment_refs)
  OR public.lab_arena__score_attempt_count_v1(p_run) IS DISTINCT FROM 6 THEN RETURN FALSE; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-captify-checkpoint614-v1'
  AND recovery_doc->>'operation'='nov03_captify_checkpoint_continuation614_v1'
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 proof:=h.recovery_doc->'checkpoint_judge';
 IF proof->>'runtime_image_digest' IS DISTINCT FROM 'sha256:7d70573f5b8a43e4d4b71e364eadc2bfa582f4881ec3370b46de13770984258e'
  OR proof->>'runtime_image_reference' IS DISTINCT FROM '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:7d70573f5b8a43e4d4b71e364eadc2bfa582f4881ec3370b46de13770984258e' THEN RETURN FALSE; END IF;
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
$judge614$;
ALTER FUNCTION public.lab_arena__nov03_captify_judge614_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_captify_judge614_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify614$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_captify_judge614_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'ed0c282a20af8913074624efda874145267e279cd500cb08fd136d95caf4eedd'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_verify614$;

DO $patch614$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','4a7c37c629b26764a2c0a76b0790d3290a3c87fa361262d07a17a662118d8ba4','1866128216b9f8ebf4e37ebbca9de4aebf025422085a06c1bb2d5b1e95ffde6e','v','plpgsql',$old614$      OR public.lab_arena__nov03_filson_judge613_v1(v_round,runs))$old614$,$new614$      OR public.lab_arena__nov03_filson_judge613_v1(v_round,runs)
      OR public.lab_arena__nov03_captify_judge614_v1(v_round,runs))$new614$),
 ('public.lab_arena_scoring_policy_history_v1(text)','24a55e58e684cdbcec2636999d13f340a003aa9bcb23b83c09272c503a8cb909','c36bbea6177d420460a14f62512351196a34cffe2eb14415751579cbce7fa0ec','s','sql',$old614$'nov03_filson_checkpoint_continuation613_v1') THEN$old614$,$new614$'nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1') THEN$new614$)
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
$patch614$;
-- Exact one-time append. This transaction does not drain unrelated jobs.
DO $checkpoint_attempt614$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 assignment CONSTANT TEXT:='arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score';
 source_id CONSTANT TEXT:='arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:1';
 manifest CONSTANT JSONB:=$manifest614${"company_judgment_refs":[{"cache_key":"sha256:d06e199fc24e216587e4e14ed449ffd0b402f374712c2f7b559dd1d689da9e0b","company_index":0,"company_input_hash":"sha256:568929be85e8cc1dc498b5b9cb6ff6bd03968109491075b72ecb7184a7e29c25","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:d06e199fc24e216587e4e14ed449ffd0b402f374712c2f7b559dd1d689da9e0b","company_input_hash":"sha256:568929be85e8cc1dc498b5b9cb6ff6bd03968109491075b72ecb7184a7e29c25","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:8c13fe150371c7a2c54d33d9fb301348309e64e3dfc00c1132c5da444c63c01b","company_index":1,"company_input_hash":"sha256:f88a83e5bb2201c1f27660096f4600dafcb1364765c8ee18adf3fe11881989f7","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:8c13fe150371c7a2c54d33d9fb301348309e64e3dfc00c1132c5da444c63c01b","company_input_hash":"sha256:f88a83e5bb2201c1f27660096f4600dafcb1364765c8ee18adf3fe11881989f7","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:7f094078401075b333854bda316560f4d52ba00f8df5696f39c9172bb7e1361c","company_index":2,"company_input_hash":"sha256:036a23e34a7065bba5747bd78051b6ae26ad7737ed05ffda127daa074e09d600","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:7f094078401075b333854bda316560f4d52ba00f8df5696f39c9172bb7e1361c","company_input_hash":"sha256:036a23e34a7065bba5747bd78051b6ae26ad7737ed05ffda127daa074e09d600","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:df757e79691bcfc593197c9ba683fb883d2da69bc0798f0798fcef0b36de3de8","company_index":3,"company_input_hash":"sha256:29ed0ae5d08bfd425373a33d786432a262a417d8db5d876b81b63cc9ee988fa8","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:df757e79691bcfc593197c9ba683fb883d2da69bc0798f0798fcef0b36de3de8","company_input_hash":"sha256:29ed0ae5d08bfd425373a33d786432a262a417d8db5d876b81b63cc9ee988fa8","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:4c6f81702d867a1a92f6575a6f3758fc958c943c9868a504a4476247c1b6f323","company_index":4,"company_input_hash":"sha256:b8cd021a1de44ab20dc8d1141e3740ec94ed3f677f6912ddbb15b1e990142ac3","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:4c6f81702d867a1a92f6575a6f3758fc958c943c9868a504a4476247c1b6f323","company_input_hash":"sha256:b8cd021a1de44ab20dc8d1141e3740ec94ed3f677f6912ddbb15b1e990142ac3","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}}],"lab_arena_company_judgments_row_hashes":{"sha256:8c13fe150371c7a2c54d33d9fb301348309e64e3dfc00c1132c5da444c63c01b":"a423030adbb184f1110b13d11c73d4ba2eec8953a1f543dc31d55ff1723ab18d","sha256:d06e199fc24e216587e4e14ed449ffd0b402f374712c2f7b559dd1d689da9e0b":"d742827d18eecc4317434e758f3dbcbb0f858e1c8a0b305e12edba8e79e0437e"},"lab_arena_signal_judgments_row_hashes":{"sha256:b6857450c8aa85b539ae41f543d7270653daec28fe3007bcf1162fcb90c7830b":"558e6eab0c6681d48143eeb739d87b387c6a7bc93136aa14eac7ece87bfdab23","sha256:e0ad92b0a9b7ae2ae0cff6f50628f03f6fc23201d6782e0a9bb55073f1c6e20d":"567bef1235129cb8baa93cc58c4a7eb268c8ecfb02d56713630cd29fcc6a702a"},"prior_run_hashes":{"arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:1":"a54ac5f5c3d759f3dc60af556644d924546c420f98d87093cc110387953fed78","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:2":"6d7c5a2741b914032710c2dca45be609aef9b671c3446934e4cdd635a8bb7c55","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:3":"765234ea14ed9603e9a8670b73ebe5e0f3161bb32e8e78b2cda8c007497e77f0","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:4":"5899a0a2120c692ac3b83daa1efeee2a85d1619b3e3a521f5fbe38f439860593","arena-2026-11-03:sub-6206272caf2b1a19ee2a7e843dc518cf:2:3:score:5":"b2cbbd2591bd87ce794ec4d8e137c5dc043811726cf5b17b50f24f3aed273cd7"},"runtime_image_digest":"sha256:7d70573f5b8a43e4d4b71e364eadc2bfa582f4881ec3370b46de13770984258e","runtime_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:7d70573f5b8a43e4d4b71e364eadc2bfa582f4881ec3370b46de13770984258e","source_sha256":"2b30f712c0584a2f3e4372209763803020ca4a4691f18dcf4357bb1b90fdfc2c"}$manifest614$::JSONB;
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
 WHERE round_id=rid AND request_id='nov03-captify-checkpoint614-v1';
 IF FOUND THEN
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_captify_checkpoint_continuation614_v1'
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
 VALUES(rid,revision,'nov03-captify-checkpoint614-v1',request_hash,'nov03-captify-checkpoint614',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',
  jsonb_build_object('operation','nov03_captify_checkpoint_continuation614_v1','scoring_generation_floor',0,
   'checkpoint_judge',manifest,'reason','One explicit checkpoint-backed sixth after the confirmed intent-name attribution bug. Use only the reviewed attribution runtime, reuse the two completed company and two signal judgments, and preserve five paid failures, the source, grading/cache identity, deadlines and $50 scoring cap. No automatic seventh attempt.'));
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
  stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,previous_runner_hotkey,company_judgment_refs)
 VALUES(assignment||':6',assignment,rid,source.submission_id,source.miner_hotkey,2,3,6,'pending',
  r.stage_generation,'score',source.run_id,parent.runner_hotkey,parent.company_judgment_refs);
 SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
 IF NOT public.lab_arena__nov03_captify_judge614_v1(r,successor)
  OR NOT public.lab_arena__judge_continuation_v1(r,successor)
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge final authority or preserved control differs'; END IF;
END;
$checkpoint_attempt614$;

DO $restore614$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena614_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena614_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge schema ACL differs'; END IF;
END;
$restore614$;
NOTIFY pgrst,'reload schema';
COMMIT;
