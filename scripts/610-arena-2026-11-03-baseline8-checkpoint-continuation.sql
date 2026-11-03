-- One explicit checkpoint continuation after the sixth judgment timed out.
-- Keep six attempts, all charges and both completed company judgments.
-- Normal completion/expiry ceilings cannot append attempt eight.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena610_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight610$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_checkpoint_judge610_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'3354cffc2bfdb74bc955aa6f7ff9650724b307ff632c1d69cd535f2a08aef052'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_preflight610$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov03_checkpoint_judge610_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $judge610$
DECLARE h public.lab_arena_scoring_supersessions; source public.lab_arena_runs;
 parent public.lab_arena_runs; proof JSONB; actual JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03' OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_round.published_at IS NOT NULL OR p_round.publication_doc IS NOT NULL OR p_round.reward_activated_at IS NOT NULL
  OR p_run.run_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:score:7'
  OR p_run.assignment_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:score'
  OR p_run.scored_run_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:1'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id OR p_run.submission_id IS DISTINCT FROM 'baseline-2026-11-03'
  OR p_run.stage IS DISTINCT FROM 1 OR p_run.icp_position IS DISTINCT FROM 8
  OR p_run.attempt IS DISTINCT FROM 7 OR p_run.scoring_generation_floor IS DISTINCT FROM 0
  OR p_run.kind IS DISTINCT FROM 'score' OR p_run.status NOT IN('pending','leased','submitted')
  OR clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
  OR p_run.judgment_scope_doc IS NOT NULL
  OR NOT public.lab_arena__company_judgment_refs_v1(p_round,p_run.company_judgment_refs)
  OR public.lab_arena__score_attempt_count_v1(p_run) IS DISTINCT FROM 7 THEN RETURN FALSE; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-baseline8-checkpoint610-v1'
  AND recovery_doc->>'operation'='nov03_baseline8_checkpoint_continuation610_v1'
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 proof:=h.recovery_doc->'checkpoint_judge';
 SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
 SELECT * INTO parent FROM public.lab_arena_runs WHERE run_id=p_run.assignment_id||':6';
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_runs a WHERE assignment_id=p_run.assignment_id AND attempt<=6;
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
  AND parent.result_doc#>>'{failure_diagnostic,error_class}'='judge_timeout'
  AND parent.result_doc#>>'{failure_diagnostic,stage}'='sandbox'
  AND parent.output_ref IS NOT NULL
  AND (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position,parent.scored_run_id)
   IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position,p_run.scored_run_id)
  AND parent.company_judgment_refs=p_run.company_judgment_refs
  AND proof->'company_judgment_refs'=p_run.company_judgment_refs
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs j WHERE j.assignment_id=p_run.assignment_id AND j.status='accepted'),FALSE);
END;
$judge610$;
ALTER FUNCTION public.lab_arena__nov03_checkpoint_judge610_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_checkpoint_judge610_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify610$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_checkpoint_judge610_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'3354cffc2bfdb74bc955aa6f7ff9650724b307ff632c1d69cd535f2a08aef052'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge helper source or security differs'; END IF;
END;
$helper_verify610$;

DO $patch610$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','029d538168d859135a2c8a3eb0ed53984f2793e69ed142c9406293ed7db2bba6','3b6e313b2c109a0004eb6c3a48a925d5d8ad2da77a89eee7835223e2ad13df9e','v','plpgsql',$old610$      OR public.lab_arena__nov03_corrected_judge607_v1(v_round,runs))$old610$,$new610$      OR public.lab_arena__nov03_corrected_judge607_v1(v_round,runs)
      OR public.lab_arena__nov03_checkpoint_judge610_v1(v_round,runs))$new610$),
 ('public.lab_arena_scoring_policy_history_v1(text)','454c53ec68adaed3de3c1784f761706593832e489bbf5446a338f613b7c58fff','7b14f6720f04d04018f73eac023b56419aa054445cbb44c9dd159c7d70f2ce9d','s','sql',$old610$'nov03_three_unfinished_executions609_v1') THEN$old610$,$new610$'nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1') THEN$new610$)
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
$patch610$;
-- Exact one-time append. This transaction does not drain unrelated jobs.
DO $checkpoint_attempt610$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 assignment CONSTANT TEXT:='arena-2026-11-03:baseline-2026-11-03:1:8:score';
 source_id CONSTANT TEXT:='arena-2026-11-03:baseline-2026-11-03:1:8:1';
 manifest CONSTANT JSONB:=$manifest610${"company_judgment_refs":[{"cache_key":"sha256:833c9691785d21f34640a8645216c45d51e1d62442bb9caade8811c47be630de","company_index":0,"company_input_hash":"sha256:bdc1cdab7b7de05fb0d335172f5837526dde830ead951b32e9d88328bcfb3759","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:833c9691785d21f34640a8645216c45d51e1d62442bb9caade8811c47be630de","company_input_hash":"sha256:bdc1cdab7b7de05fb0d335172f5837526dde830ead951b32e9d88328bcfb3759","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:6e5263d726bfc2f9d52958ab50866b41ff7bfab3f7af37f1ec91d1cf9ec95b02","company_index":1,"company_input_hash":"sha256:69e9bdb64246c92e1c1152f65b34622d0b36e45c6d23831bdb65cb46b3acafe8","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:6e5263d726bfc2f9d52958ab50866b41ff7bfab3f7af37f1ec91d1cf9ec95b02","company_input_hash":"sha256:69e9bdb64246c92e1c1152f65b34622d0b36e45c6d23831bdb65cb46b3acafe8","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:8414d83c784b38eff7c3c76d0f931bf44a4556e006b7b099a5c0d45222545653","company_index":2,"company_input_hash":"sha256:d09ab471cbbab3c865829cc9026a753800abe965ad8c18c56868a51c3fc70eca","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:8414d83c784b38eff7c3c76d0f931bf44a4556e006b7b099a5c0d45222545653","company_input_hash":"sha256:d09ab471cbbab3c865829cc9026a753800abe965ad8c18c56868a51c3fc70eca","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:d7478fdf910366981016ab75e3165700f4cb85ec75c705e692cbbbd63c28d773","company_index":3,"company_input_hash":"sha256:564c3caaa4b0832eea3b6bd2b2f5ad954923e345aeb035875893e621d34741d4","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:d7478fdf910366981016ab75e3165700f4cb85ec75c705e692cbbbd63c28d773","company_input_hash":"sha256:564c3caaa4b0832eea3b6bd2b2f5ad954923e345aeb035875893e621d34741d4","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:152cc9f81d5870fe5af3b6641ffe6747f21c5564f429a3c3f2b868d0fa4a2370","company_index":4,"company_input_hash":"sha256:4ced26f1888487971af70ee932f17a29af4c3226041fb39819df9bd058f9e883","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:152cc9f81d5870fe5af3b6641ffe6747f21c5564f429a3c3f2b868d0fa4a2370","company_input_hash":"sha256:4ced26f1888487971af70ee932f17a29af4c3226041fb39819df9bd058f9e883","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}}],"lab_arena_company_judgments_row_hashes":{"sha256:6e5263d726bfc2f9d52958ab50866b41ff7bfab3f7af37f1ec91d1cf9ec95b02":"eaec0237f16cda8bc785e71dba84194684424f8cebef7dd5e34b371cdcb778f4","sha256:833c9691785d21f34640a8645216c45d51e1d62442bb9caade8811c47be630de":"2cb5ae6b57375151f92187f7a01985068bb4d7d345c9214b76d56280060869eb"},"lab_arena_signal_judgments_row_hashes":{"sha256:1bf484a09c510188bf25b67366c8e992ea33b0838db4f55326ad540f1d80426d":"815a954b6e7f2ef5175d5f487b3a37e2aa2570645cbe531172394969e00b2262"},"prior_run_hashes":{"arena-2026-11-03:baseline-2026-11-03:1:8:score:1":"ee17bb654544bc05143c6ba9fa1ec070711502edbd683a715876a029c4495cda","arena-2026-11-03:baseline-2026-11-03:1:8:score:2":"63c13e3902a7a3dcb4baf52376ac13b2e2080d1ed853a96f624255105682f376","arena-2026-11-03:baseline-2026-11-03:1:8:score:3":"91661ea7c81e8ad12cda9351faf6f960ae54d46301e8d970217bba2fea29aec0","arena-2026-11-03:baseline-2026-11-03:1:8:score:4":"c44d420a7393c251b23b6cb552f8344632607a2610ea90c1021986aeff4bd9c5","arena-2026-11-03:baseline-2026-11-03:1:8:score:5":"bcc7a37a15147fe70f5b8bf6ed33a216b601de5a63678ca76ae6434d43b3e031","arena-2026-11-03:baseline-2026-11-03:1:8:score:6":"b0789b9d53538067964c634a2d2d5913e8790df6dea6451e30d22717a7f8e329"},"source_sha256":"7fa1f2a24b3276aefec06ea61bdbe782a1a7167ec9ba06f30dedd6379b453f64"}$manifest610$::JSONB;
 r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 source public.lab_arena_runs; parent public.lab_arena_runs; successor public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; revision BIGINT; actual JSONB; request_hash TEXT;
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 PERFORM 1 FROM public.lab_arena_runs WHERE assignment_id=assignment ORDER BY attempt FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=source_id FOR UPDATE;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=assignment||':6';
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=rid AND request_id='nov03-baseline8-checkpoint610-v1';
 IF FOUND THEN
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':7';
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_baseline8_checkpoint_continuation610_v1'
   OR h.recovery_doc->'checkpoint_judge' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR successor.scored_run_id IS DISTINCT FROM source_id OR successor.attempt IS DISTINCT FROM 7
   OR successor.company_judgment_refs IS DISTINCT FROM parent.company_judgment_refs
   OR successor.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>7) THEN
   RAISE EXCEPTION 'Nov03 checkpoint judge replay differs'; END IF;
  RETURN;
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR c.operator_paused
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2' OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR (r.configuration_doc->>'scoring_cap_microusd')::BIGINT IS DISTINCT FROM 50000000
  OR r.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM 'sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=rid AND prior.request_id='nov03-baseline8-corrected607-v1'
    AND prior.recovery_doc->>'operation'='nov03_baseline8_corrected_judge607_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc)
  OR source.kind IS DISTINCT FROM 'execute' OR source.status IS DISTINCT FROM 'accepted'
  OR source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
  OR source.per_icp_score IS NOT NULL OR source.qualification_doc IS NOT NULL
  OR encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'source_sha256'
  OR parent.kind IS DISTINCT FROM 'score' OR parent.status IS DISTINCT FROM 'failed'
  OR parent.terminal_cause IS DISTINCT FROM 'judge_error' OR parent.attempt IS DISTINCT FROM 6
  OR parent.result_doc->>'terminal_status' IS DISTINCT FROM 'judge_error'
  OR parent.result_doc#>>'{failure_diagnostic,error_class}' IS DISTINCT FROM 'judge_timeout'
  OR parent.result_doc#>>'{failure_diagnostic,stage}' IS DISTINCT FROM 'sandbox'
  OR parent.output_ref IS NULL
  OR public.lab_arena__score_attempt_count_v1(parent) IS DISTINCT FROM 6
  OR parent.scored_run_id IS DISTINCT FROM source.run_id
  OR (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position)
   IS DISTINCT FROM (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
  OR parent.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
  OR NOT public.lab_arena__company_judgment_refs_v1(r,parent.company_judgment_refs)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND (attempt>6 OR status IN('accepted','leased','submitted'))) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge exact source, timeout, guard or clock differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO actual FROM public.lab_arena_runs a WHERE assignment_id=assignment;
 IF actual IS DISTINCT FROM manifest->'prior_run_hashes' THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge six prior attempts differ'; END IF;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'nov03-baseline8-checkpoint610-v1',request_hash,'nov03-baseline8-checkpoint610',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',
  jsonb_build_object('operation','nov03_baseline8_checkpoint_continuation610_v1','scoring_generation_floor',0,
   'checkpoint_judge',manifest,'reason','One explicit checkpoint-backed continuation after attempt six timed out. Reuse both completed company judgments; preserve six prior attempts and charges, the source, verifier, deadlines and $50 scoring cap. No automatic eighth attempt.'));
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
  stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,previous_runner_hotkey,company_judgment_refs)
 VALUES(assignment||':7',assignment,rid,source.submission_id,source.miner_hotkey,1,8,7,'pending',
  r.stage_generation,'score',source.run_id,parent.runner_hotkey,parent.company_judgment_refs);
 SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':7';
 IF NOT public.lab_arena__nov03_checkpoint_judge610_v1(r,successor)
  OR NOT public.lab_arena__judge_continuation_v1(r,successor)
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge final authority or preserved control differs'; END IF;
END;
$checkpoint_attempt610$;

DO $restore610$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena610_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena610_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 checkpoint judge schema ACL differs'; END IF;
END;
$restore610$;
NOTIFY pgrst,'reload schema';
COMMIT;
