-- One exact b66 ICP3 continuation after the confirmed identity-repair bug.
-- Keep five attempts, all charges and the completed company/signal judgments.
-- Normal completion/expiry ceilings cannot append attempt seven.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena639_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight639$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov05_b66_judge639_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'b973d13543d7d3021e2bd9a92223ef3d5116c5ffb6d741d101bb621f0044db8a'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov05 repaired judge helper source or security differs'; END IF;
END;
$helper_preflight639$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov05_b66_judge639_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $judge639$
DECLARE h public.lab_arena_scoring_supersessions; source public.lab_arena_runs;
 parent public.lab_arena_runs; proof JSONB; actual JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-05' OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_round.published_at IS NOT NULL OR p_round.publication_doc IS NOT NULL OR p_round.reward_activated_at IS NOT NULL
  OR p_run.run_id IS DISTINCT FROM 'arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:6'
  OR p_run.assignment_id IS DISTINCT FROM 'arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score'
  OR p_run.scored_run_id IS DISTINCT FROM 'arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:1'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id OR p_run.submission_id IS DISTINCT FROM 'sub-b66ad20113a4402633ba875964ed4b80'
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
 WHERE round_id=p_round.round_id AND request_id='nov05-b66-post-repair639-v1'
  AND recovery_doc->>'operation'='nov05_b66_post_repair639_v1'
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 proof:=h.recovery_doc->'checkpoint_judge';
 IF proof->>'runtime_image_digest' IS DISTINCT FROM 'sha256:0d930b667246e37886a5237ce98ed7838c2a8fefd430d124f2905aa4d19f18db'
  OR proof->>'runtime_image_reference' IS DISTINCT FROM '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:0d930b667246e37886a5237ce98ed7838c2a8fefd430d124f2905aa4d19f18db' THEN RETURN FALSE; END IF;
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
$judge639$;
ALTER FUNCTION public.lab_arena__nov05_b66_judge639_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov05_b66_judge639_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify639$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov05_b66_judge639_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'b973d13543d7d3021e2bd9a92223ef3d5116c5ffb6d741d101bb621f0044db8a'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov05 repaired judge helper source or security differs'; END IF;
END;
$helper_verify639$;

DO $patch639$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','6aabab38cf3474db8e97c0b2aaffd1f071541b694020b0411a023acd780fe637','2acc6ff6c64ed797299cd18f203b3d62459720e2ae1ab9cb1f03a7850e62dd82','v','plpgsql',$old639$      OR public.lab_arena__nov03_captify_judge614_v1(v_round,runs))$old639$,$new639$      OR public.lab_arena__nov03_captify_judge614_v1(v_round,runs)
      OR public.lab_arena__nov05_b66_judge639_v1(v_round,runs))$new639$),
 ('public.lab_arena_scoring_policy_history_v1(text)','5fdfee1a65c755a862b17998607514941a222c5912d60852a00bf21e027777bd','34123c2c2c2e68d4056a94c2f9f5bba4889d158ee8ac38869f6e68da88a2eb84','s','sql',$old639$'nov05_recovery_window641_v1') THEN$old639$,$new639$'nov05_recovery_window641_v1','nov05_b66_post_repair639_v1') THEN$new639$)
 ) AS patches(signature,preimage,postimage,volatility,language,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],target.language) THEN
   RAISE EXCEPTION 'Nov05 repaired judge security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>length(target.old_text) THEN
   RAISE EXCEPTION 'Nov05 repaired judge preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov05 repaired judge postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov05 repaired judge readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch639$;
-- Exact one-time append. This transaction does not drain unrelated jobs.
DO $checkpoint_attempt639$
DECLARE rid CONSTANT TEXT:='arena-2026-11-05';
 assignment CONSTANT TEXT:='arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score';
 source_id CONSTANT TEXT:='arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:1';
 manifest CONSTANT JSONB:=$manifest639${"company_judgment_refs":[{"cache_key":"sha256:1b2850bdefa78ce4bb3c68a83d4fdaf81d45e3a29543960c96cf89bbbd20c0ef","company_index":0,"company_input_hash":"sha256:7e287d49495439c09ecd9d257edfe77796c2b184f6e3546a12e7506c867b9ef4","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:1b2850bdefa78ce4bb3c68a83d4fdaf81d45e3a29543960c96cf89bbbd20c0ef","company_input_hash":"sha256:7e287d49495439c09ecd9d257edfe77796c2b184f6e3546a12e7506c867b9ef4","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-05","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"}},{"cache_key":"sha256:cd46c538a286bb0c7fdde207e0a6fcd32e815fb56769574b4e287e824e651cd8","company_index":1,"company_input_hash":"sha256:62c9ec330119657e2eda2a1406fad9d69304785b8325c7834599c9833d0aeabc","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:cd46c538a286bb0c7fdde207e0a6fcd32e815fb56769574b4e287e824e651cd8","company_input_hash":"sha256:62c9ec330119657e2eda2a1406fad9d69304785b8325c7834599c9833d0aeabc","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-05","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"}},{"cache_key":"sha256:8364b320e6eb4d9eb2f92bbd07ccb98e09f8947d578b3e4a22088e349eb75bd8","company_index":2,"company_input_hash":"sha256:fe38f08d4e57032d879ea7b4cd575bf9fdaf04abfcb36794f3619f9f2e7b9112","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:8364b320e6eb4d9eb2f92bbd07ccb98e09f8947d578b3e4a22088e349eb75bd8","company_input_hash":"sha256:fe38f08d4e57032d879ea7b4cd575bf9fdaf04abfcb36794f3619f9f2e7b9112","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-05","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"}},{"cache_key":"sha256:921270f5f11167f43c86f99befc1b3091d428ff6f0c8dd9793b9e5e8d2c4ed84","company_index":3,"company_input_hash":"sha256:df1f7e804daf19bafc6a974b250b4874fe9f6feca0f72771e3b0199fb5fe60ff","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:921270f5f11167f43c86f99befc1b3091d428ff6f0c8dd9793b9e5e8d2c4ed84","company_input_hash":"sha256:df1f7e804daf19bafc6a974b250b4874fe9f6feca0f72771e3b0199fb5fe60ff","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-05","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"}},{"cache_key":"sha256:705e7e42bb48d22eca35788236a92ef3370068e95d5ac4e6735c3837d6a48a82","company_index":4,"company_input_hash":"sha256:930ff962e008f2c37887523a91c9cdde535a1996fcde962bff7900e56e35cdf8","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:705e7e42bb48d22eca35788236a92ef3370068e95d5ac4e6735c3837d6a48a82","company_input_hash":"sha256:930ff962e008f2c37887523a91c9cdde535a1996fcde962bff7900e56e35cdf8","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-05","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b"}}],"lab_arena_company_judgments_row_hashes":{"sha256:1b2850bdefa78ce4bb3c68a83d4fdaf81d45e3a29543960c96cf89bbbd20c0ef":"73facf4aa94df8e4fc13bb495ae1aab4f65667de5edd36b4c13488433f94e676","sha256:8364b320e6eb4d9eb2f92bbd07ccb98e09f8947d578b3e4a22088e349eb75bd8":"def5da8fd79bd966a8a9a773f5b1ff9cba682710faeb65e18c2642a0b5620cac","sha256:921270f5f11167f43c86f99befc1b3091d428ff6f0c8dd9793b9e5e8d2c4ed84":"7c6e340b638c95db6e2f278c5fa24213634cf2c659186778ad7f3a38a1f0e059","sha256:cd46c538a286bb0c7fdde207e0a6fcd32e815fb56769574b4e287e824e651cd8":"913e84294191e6909a0a847852fb7f99535fd58d0d333dc7eafde14f779cf5c6"},"lab_arena_signal_judgments_row_hashes":{"sha256:4464fa4ea9b95653fcebbe965c0506941b72d4f17bbce42d06dc9691a551da36":"82fcd8a673cf1bf6a7e2083ed192ce3890853942089ae7464be44b885725993c","sha256:630345d0ab71dac2cae1a258ee0a5b4aea55da9673127115993400868edd9d00":"a98b434f42959e1b91f5bb809272bd61b7dfe06b387c786bf5768fdbb0fd4a28","sha256:b5017745b785b1819301eee01b97c25dbec6888109f46e7a88fa78373b565463":"ceb6a7c799bb79f8c36244c34212b65cf54b4b542eb097cef4db7d66de4c6d3d"},"prior_run_hashes":{"arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:1":"20e78edaeec4a3732f8fff14bcdc5611e4b0105ce1a90d27e210dc128c3f40c9","arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:2":"f79a687da7cc3585963615893424905cf750080b70a7df51a6ec269986946805","arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:3":"655a3b60ce61401ad96576c45998bbb6a48143956c7acec39aacf08dfbfe1bc5","arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:4":"bac7b94e4971ea25db6b8c99d37cde3e03b0da86e23bfcf4d944ff42c4c3d184","arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score:5":"d627e75853cd2fbb1eb167cfd5a9d86e9f8e3d1989de2330eab6d0b90620ee2d"},"runtime_image_digest":"sha256:0d930b667246e37886a5237ce98ed7838c2a8fefd430d124f2905aa4d19f18db","runtime_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:0d930b667246e37886a5237ce98ed7838c2a8fefd430d124f2905aa4d19f18db","source_sha256":"904911d19f246cec1d4b63b91d656cc0e64532d9e2731ff373eb6ad6c1f9b689"}$manifest639$::JSONB;
 r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 source public.lab_arena_runs; parent public.lab_arena_runs; successor public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; revision BIGINT; actual JSONB; request_hash TEXT;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 PERFORM 1 FROM public.lab_arena_runs WHERE assignment_id=assignment ORDER BY attempt FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=source_id FOR UPDATE;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=assignment||':5';
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=rid AND request_id='nov05-b66-post-repair639-v1';
 IF FOUND THEN
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov05_b66_post_repair639_v1'
   OR h.recovery_doc->'checkpoint_judge' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR successor.scored_run_id IS DISTINCT FROM source_id OR successor.attempt IS DISTINCT FROM 6
   OR successor.company_judgment_refs IS DISTINCT FROM parent.company_judgment_refs
   OR successor.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>6) THEN
   RAISE EXCEPTION 'Nov05 repaired judge replay differs'; END IF;
  RETURN;
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR c.operator_paused
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR ((r.status,r.status_generation,r.stage_generation) IS DISTINCT FROM ('stage2_scoring'::TEXT,11::BIGINT,9::BIGINT)
    AND (r.status,r.status_generation,r.stage_generation) IS DISTINCT FROM ('stage2_judged'::TEXT,12::BIGINT,10::BIGINT)
    AND (r.status,r.status_generation,r.stage_generation) IS DISTINCT FROM ('scored'::TEXT,13::BIGINT,10::BIGINT))
  OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM '489f605b2dc4f6f68b31c15c21c3b8bedf3b0ff8f38dfd6ba4e7b2fd72f62048'
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.configuration_doc#>>'{schedule,final_scoring_close}' IS DISTINCT FROM '2026-11-06T11:30:00Z'
  OR r.configuration_doc#>>'{schedule,publication_deadline}' IS DISTINCT FROM '2026-11-06T11:35:00Z'
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR (r.configuration_doc->>'scoring_cap_microusd')::BIGINT IS DISTINCT FROM 50000000
  OR r.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM 'sha256:b69931916a8ddfae0836876f20c162799c0b26b1e3b160389885f9d5e656361b'
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
  RAISE EXCEPTION 'Nov05 repaired judge exact source, runtime, guard or clock differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_runs a WHERE assignment_id=assignment;
 IF actual IS DISTINCT FROM manifest->'prior_run_hashes' THEN
  RAISE EXCEPTION 'Nov05 repaired judge five prior attempts differ'; END IF;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'nov05-b66-post-repair639-v1',request_hash,'nov05-b66-post-repair639',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',
  jsonb_build_object('operation','nov05_b66_post_repair639_v1','scoring_generation_floor',0,
   'checkpoint_judge',manifest,'reason','One explicitly approved sixth after the confirmed identity-repair investigation omission. Use only the reviewed frozen-b699 runtime, reuse completed company and signal judgments, and preserve five paid failures, source, grading/cache identity, deadlines and $50 scoring cap. No automatic seventh attempt.'));
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
  stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,previous_runner_hotkey,company_judgment_refs,judgment_group_leader)
 VALUES(assignment||':6',assignment,rid,source.submission_id,source.miner_hotkey,2,3,6,'pending',
  r.stage_generation,'score',source.run_id,parent.runner_hotkey,parent.company_judgment_refs,NULL);
 SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
 IF NOT public.lab_arena__nov05_b66_judge639_v1(r,successor)
  OR NOT public.lab_arena__judge_continuation_v1(r,successor)
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov05 repaired judge final authority or preserved control differs'; END IF;
END;
$checkpoint_attempt639$;

DO $restore639$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena639_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena639_schema_acl) THEN
  RAISE EXCEPTION 'Nov05 repaired judge schema ACL differs'; END IF;
END;
$restore639$;
NOTIFY pgrst,'reload schema';
COMMIT;
