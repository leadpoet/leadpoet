-- One corrected judge for the confirmed numeric LinkedIn identity failure.
-- Attempts 1–5 and all costs stay unchanged. The counted attempt remains six;
-- existing completion/expiry limits therefore cannot create attempt seven.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena607_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight607$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_corrected_judge607_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'11f1b585b0ec924923c20859d033766210f90e34e7503249c2d4a06ed4657a0d'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 corrected judge helper source or security differs'; END IF;
END;
$helper_preflight607$;

CREATE OR REPLACE FUNCTION public.lab_arena__nov03_corrected_judge607_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $judge607$
DECLARE h public.lab_arena_scoring_supersessions; source public.lab_arena_runs;
 parent public.lab_arena_runs; proof JSONB;
BEGIN
 IF p_round.round_id IS DISTINCT FROM 'arena-2026-11-03' OR p_round.scoring_generation_floor IS DISTINCT FROM 0
  OR p_run.run_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:score:6'
  OR p_run.assignment_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:score'
  OR p_run.scored_run_id IS DISTINCT FROM 'arena-2026-11-03:baseline-2026-11-03:1:8:1'
  OR p_run.round_id IS DISTINCT FROM p_round.round_id OR p_run.submission_id IS DISTINCT FROM 'baseline-2026-11-03'
  OR p_run.stage IS DISTINCT FROM 1 OR p_run.icp_position IS DISTINCT FROM 8
  OR p_run.attempt IS DISTINCT FROM 6 OR p_run.scoring_generation_floor IS DISTINCT FROM 0
  OR p_run.kind IS DISTINCT FROM 'score' OR p_run.status NOT IN('pending','leased','submitted')
  OR (p_run.status='pending' AND clock_timestamp()>=(p_round.configuration_doc#>>'{schedule,stage_1_scoring_close}')::TIMESTAMPTZ)
  OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
  OR p_run.judgment_scope_doc IS NOT NULL
  OR NOT public.lab_arena__company_judgment_refs_v1(p_round,p_run.company_judgment_refs)
  OR public.lab_arena__score_attempt_count_v1(p_run) IS DISTINCT FROM 6 THEN RETURN FALSE; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=p_round.round_id AND request_id='nov03-baseline8-corrected607-v1'
  AND recovery_doc->>'operation'='nov03_baseline8_corrected_judge607_v1'
  AND new_round_doc->'configuration_doc'=p_round.configuration_doc
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 proof:=h.recovery_doc->'corrected_judge';
 SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
 SELECT * INTO parent FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-03:baseline-2026-11-03:1:8:score:5';
 RETURN coalesce(source.run_id=p_run.scored_run_id AND source.kind='execute'
  AND source.status='accepted' AND source.terminal_cause='accepted' AND source.output_ref IS NOT NULL
  AND source.per_icp_score IS NULL AND source.qualification_doc IS NULL
  AND parent.status='failed' AND parent.terminal_cause='judge_error' AND parent.attempt=5
  AND (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position,parent.scored_run_id)
   IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position,p_run.scored_run_id)
  AND (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
   IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
  AND proof->'company_judgment_refs'=p_run.company_judgment_refs
  AND proof->>'source_sha256'=encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex')
  AND proof->>'parent_sha256'=encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs j
   WHERE j.assignment_id=p_run.assignment_id AND j.status='accepted' AND j.scoring_generation_floor=0),FALSE);
END;
$judge607$;
ALTER FUNCTION public.lab_arena__nov03_corrected_judge607_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_corrected_judge607_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $helper_verify607$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_corrected_judge607_v1(public.lab_arena_rounds,public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')<>'11f1b585b0ec924923c20859d033766210f90e34e7503249c2d4a06ed4657a0d'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Nov03 corrected judge helper source or security differs'; END IF;
END;
$helper_verify607$;

DO $patch607$
DECLARE target RECORD; d TEXT; updated TEXT; metadata JSONB; function_oid OID;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','df4e15dc9008ff0774e60f56ac7780b85e6783b7c12add61adac94ba8cbf68aa','029d538168d859135a2c8a3eb0ed53984f2793e69ed142c9406293ed7db2bba6',$old607$    AND (runs.kind<>'score' OR public.lab_arena__score_attempt_count_v1(runs) <= 5)$old607$,$new607$    AND (runs.kind<>'score' OR public.lab_arena__score_attempt_count_v1(runs) <= 5
      OR public.lab_arena__nov03_corrected_judge607_v1(v_round,runs))$new607$),
 ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','dd24da9152f1787e8a556b2ec972feeff6d5c73938c2f0b071cb67db39a78246','ac7f91629dc1a1c7c30cb9b3fcb8eaf8a14132f533758144bea6dbc4257113e4',$old607$AND v_parent.company_judgment_refs IS NOT DISTINCT FROM p_run.company_judgment_refs$old607$,$new607$AND (v_parent.company_judgment_refs IS NOT DISTINCT FROM p_run.company_judgment_refs
      OR public.lab_arena__nov03_corrected_judge607_v1(p_round,p_run))$new607$)
 ) AS patches(signature,preimage,postimage,old_text,new_text) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=function_oid;
  IF d IS NULL OR (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid)
   IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
    CASE WHEN target.signature LIKE 'public.lab_arena_claim_assignment(%'
     THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' ELSE '{lab_arena_owner=X/lab_arena_owner}' END,
    TRUE,CASE WHEN target.signature LIKE 'public.lab_arena_claim_assignment(%' THEN 'v' ELSE 's' END,
    ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
   RAISE EXCEPTION 'Nov03 corrected judge security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage
   OR length(d)-length(replace(d,target.old_text,''))<>length(target.old_text) THEN
   RAISE EXCEPTION 'Nov03 corrected judge preimage differs: %',target.signature; END IF;
  updated:=replace(d,target.old_text,target.new_text);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 corrected judge postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov03 corrected judge readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch607$;
-- Exact one-time corrected attempt607. The source is already accepted;
-- only the numbered judge row is appended. All prior rows remain immutable.
DO $corrected_attempt607$
DECLARE rid CONSTANT TEXT:='arena-2026-11-03';
 assignment CONSTANT TEXT:='arena-2026-11-03:baseline-2026-11-03:1:8:score';
 source_id CONSTANT TEXT:='arena-2026-11-03:baseline-2026-11-03:1:8:1';
 manifest CONSTANT JSONB:=$manifest607${"company_judgment_refs":[{"cache_key":"sha256:833c9691785d21f34640a8645216c45d51e1d62442bb9caade8811c47be630de","company_index":0,"company_input_hash":"sha256:bdc1cdab7b7de05fb0d335172f5837526dde830ead951b32e9d88328bcfb3759","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:833c9691785d21f34640a8645216c45d51e1d62442bb9caade8811c47be630de","company_input_hash":"sha256:bdc1cdab7b7de05fb0d335172f5837526dde830ead951b32e9d88328bcfb3759","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:6e5263d726bfc2f9d52958ab50866b41ff7bfab3f7af37f1ec91d1cf9ec95b02","company_index":1,"company_input_hash":"sha256:69e9bdb64246c92e1c1152f65b34622d0b36e45c6d23831bdb65cb46b3acafe8","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:6e5263d726bfc2f9d52958ab50866b41ff7bfab3f7af37f1ec91d1cf9ec95b02","company_input_hash":"sha256:69e9bdb64246c92e1c1152f65b34622d0b36e45c6d23831bdb65cb46b3acafe8","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:8414d83c784b38eff7c3c76d0f931bf44a4556e006b7b099a5c0d45222545653","company_index":2,"company_input_hash":"sha256:d09ab471cbbab3c865829cc9026a753800abe965ad8c18c56868a51c3fc70eca","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:8414d83c784b38eff7c3c76d0f931bf44a4556e006b7b099a5c0d45222545653","company_input_hash":"sha256:d09ab471cbbab3c865829cc9026a753800abe965ad8c18c56868a51c3fc70eca","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:d7478fdf910366981016ab75e3165700f4cb85ec75c705e692cbbbd63c28d773","company_index":3,"company_input_hash":"sha256:564c3caaa4b0832eea3b6bd2b2f5ad954923e345aeb035875893e621d34741d4","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:d7478fdf910366981016ab75e3165700f4cb85ec75c705e692cbbbd63c28d773","company_input_hash":"sha256:564c3caaa4b0832eea3b6bd2b2f5ad954923e345aeb035875893e621d34741d4","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}},{"cache_key":"sha256:152cc9f81d5870fe5af3b6641ffe6747f21c5564f429a3c3f2b868d0fa4a2370","company_index":4,"company_input_hash":"sha256:4ced26f1888487971af70ee932f17a29af4c3226041fb39819df9bd058f9e883","schema_version":"leadpoet.lab_arena.company_judgment_ref.v1","scope_doc":{"cache_key":"sha256:152cc9f81d5870fe5af3b6641ffe6747f21c5564f429a3c3f2b868d0fa4a2370","company_input_hash":"sha256:4ced26f1888487971af70ee932f17a29af4c3226041fb39819df9bd058f9e883","company_judgment_policy":"company_judgment_v1","evaluation_date":"2026-11-03","integrity_policy":"arena_integrity_v1","netuid":71,"network_name":"finney","round_id":"arena-2026-11-03","schema_version":"leadpoet.lab_arena.company_judgment_scope.v1","scorer_image_digest":"sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b"}}],"parent_sha256":"bcc7a37a15147fe70f5b8bf6ed33a216b601de5a63678ca76ae6434d43b3e031","prior_run_hashes":{"arena-2026-11-03:baseline-2026-11-03:1:8:score:1":"ee17bb654544bc05143c6ba9fa1ec070711502edbd683a715876a029c4495cda","arena-2026-11-03:baseline-2026-11-03:1:8:score:2":"63c13e3902a7a3dcb4baf52376ac13b2e2080d1ed853a96f624255105682f376","arena-2026-11-03:baseline-2026-11-03:1:8:score:3":"91661ea7c81e8ad12cda9351faf6f960ae54d46301e8d970217bba2fea29aec0","arena-2026-11-03:baseline-2026-11-03:1:8:score:4":"c44d420a7393c251b23b6cb552f8344632607a2610ea90c1021986aeff4bd9c5","arena-2026-11-03:baseline-2026-11-03:1:8:score:5":"bcc7a37a15147fe70f5b8bf6ed33a216b601de5a63678ca76ae6434d43b3e031"},"source_sha256":"7fa1f2a24b3276aefec06ea61bdbe782a1a7167ec9ba06f30dedd6379b453f64"}$manifest607$::JSONB;
 r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 source public.lab_arena_runs; parent public.lab_arena_runs; successor public.lab_arena_runs;
 h public.lab_arena_scoring_supersessions; revision BIGINT; old_rows JSONB; request_hash TEXT;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT source FROM public.lab_arena_runs WHERE run_id=source_id FOR UPDATE;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=assignment||':5' FOR UPDATE;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=rid AND request_id='nov03-baseline8-corrected607-v1';
 IF FOUND THEN
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_baseline8_corrected_judge607_v1'
   OR h.recovery_doc->'corrected_judge' IS DISTINCT FROM manifest
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR successor.scored_run_id IS DISTINCT FROM source_id OR successor.attempt IS DISTINCT FROM 6
   OR successor.company_judgment_refs IS DISTINCT FROM manifest->'company_judgment_refs'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>6) THEN
   RAISE EXCEPTION 'Nov03 corrected judge replay differs'; END IF;
  RETURN;
 END IF;
 IF current_setting('session_replication_role')<>'origin'
  OR NOT c.operator_paused OR c.pause_reason IS DISTINCT FROM 'nov03_improved_verifier_required603'
  OR c.guard_commitment IS DISTINCT FROM '' OR c.owner_commitment IS DISTINCT FROM ''
  OR c.guard_expires_at IS NOT NULL OR c.candidate_commit IS DISTINCT FROM ''
  OR c.restart_scope IS DISTINCT FROM '' OR c.restart_phase IS DISTINCT FROM ''
  OR c.captured_leases IS DISTINCT FROM '[]'::JSONB
  OR r.status IS DISTINCT FROM 'stage2' OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=rid AND prior.request_id='nov03-forward-only-verifier604-v1'
    AND prior.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc)
  OR source.kind IS DISTINCT FROM 'execute' OR source.status IS DISTINCT FROM 'accepted'
  OR source.terminal_cause IS DISTINCT FROM 'accepted' OR source.output_ref IS NULL
  OR source.per_icp_score IS NOT NULL OR source.qualification_doc IS NOT NULL
  OR encode(extensions.digest(to_jsonb(source)::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'source_sha256'
  OR parent.kind IS DISTINCT FROM 'score' OR parent.status IS DISTINCT FROM 'failed'
  OR parent.terminal_cause IS DISTINCT FROM 'judge_error' OR parent.attempt IS DISTINCT FROM 5
  OR public.lab_arena__score_attempt_count_v1(parent) IS DISTINCT FROM 5
  OR parent.scored_run_id IS DISTINCT FROM source.run_id
  OR (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position)
   IS DISTINCT FROM (source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position)
  OR NOT public.lab_arena__company_judgment_refs_v1(r,manifest->'company_judgment_refs')
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE assignment_id=assignment AND attempt>5)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('leased','submitted')) THEN
  RAISE EXCEPTION 'Nov03 corrected judge exact source, pause, budget or clock differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
 INTO old_rows FROM public.lab_arena_runs a WHERE assignment_id=assignment;
 IF old_rows IS DISTINCT FROM manifest->'prior_run_hashes' THEN
  RAISE EXCEPTION 'Nov03 corrected judge prior five attempts differ'; END IF;
 SELECT greatest(1,coalesce(max(generation_floor),0)+1) INTO revision
 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'nov03-baseline8-corrected607-v1',request_hash,'nov03-baseline8-corrected607',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',
  jsonb_build_object('operation','nov03_baseline8_corrected_judge607_v1','scoring_generation_floor',0,
   'corrected_judge',manifest,'reason','One corrected verifier attempt for the reproduced numeric LinkedIn identity bug. Keep the five prior paid failures; no attempt seven, budget reset, execution replay or clock extension.'));
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
  stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,previous_runner_hotkey,company_judgment_refs)
 VALUES(assignment||':6',assignment,rid,source.submission_id,source.miner_hotkey,1,8,6,'pending',
  r.stage_generation,'score',source.run_id,parent.runner_hotkey,manifest->'company_judgment_refs');
 SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=assignment||':6';
 IF NOT public.lab_arena__nov03_corrected_judge607_v1(r,successor)
  OR NOT public.lab_arena__judge_continuation_v1(r,successor)
  OR (SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex'))
    FROM public.lab_arena_runs a WHERE assignment_id=assignment AND attempt<=5) IS DISTINCT FROM old_rows
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Nov03 corrected judge final authority or preserved history differs'; END IF;
END;
$corrected_attempt607$;

DO $restore607$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena607_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
  (SELECT nspacl FROM pg_temp.arena607_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 corrected judge schema ACL differs'; END IF;
END;
$restore607$;
NOTIFY pgrst,'reload schema';
COMMIT;
