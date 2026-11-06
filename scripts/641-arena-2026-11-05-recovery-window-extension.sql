-- Exact approved Nov5 infrastructure-recovery schedule extension only.
-- Preserve phase, all work, billing, scoring policy, source and other rounds.
-- Normal early completion remains available; no attempts are created or reset.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
-- Keep this audit receipt at the unchanged scoring floor in existing readers.
DO $history641$
DECLARE target OID:='public.lab_arena_scoring_policy_history_v1(text)'::regprocedure;
 d TEXT; metadata JSONB; expected_post TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((metadata->>'proowner')::OID)<>'lab_arena_owner'
  OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner","lab_arena_service=X/lab_arena_owner"]'::JSONB
  OR metadata->>'prosecdef'<>'true' OR metadata->>'provolatile'<>'s'
  OR metadata->'proconfig' IS DISTINCT FROM '["search_path=pg_catalog, public"]'::JSONB THEN
  RAISE EXCEPTION 'Window641 history security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex') IN('b94409bc742b2a4e51d92679faeeeb0274c40993ca8593e3df0a30ce01b3473e','5fdfee1a65c755a862b17998607514941a222c5912d60852a00bf21e027777bd','34123c2c2c2e68d4056a94c2f9f5bba4889d158ee8ac38869f6e68da88a2eb84') THEN RETURN; END IF;
 expected_post:=CASE encode(extensions.digest(d,'sha256'),'hex')
  WHEN '27da80592b12e32477348f106756d4e5bf9ca8dabe9e604291530519a3a93d43' THEN 'b94409bc742b2a4e51d92679faeeeb0274c40993ca8593e3df0a30ce01b3473e'
  WHEN 'e56c877e3c2ba5440b2fc7b55caa65b2e94935c5117dddd12ed1320c9e3f4a45' THEN '5fdfee1a65c755a862b17998607514941a222c5912d60852a00bf21e027777bd'
  ELSE NULL END;
 IF expected_post IS NULL
  OR length(d)-length(replace(d,$old641$'nov05_infrastructure_recovery633_v1') THEN$old641$,''))<>length($old641$'nov05_infrastructure_recovery633_v1') THEN$old641$) THEN
  RAISE EXCEPTION 'Window641 history preimage differs'; END IF;
 EXECUTE replace(d,$old641$'nov05_infrastructure_recovery633_v1') THEN$old641$,
  $new641$'nov05_infrastructure_recovery633_v1','nov05_recovery_window641_v1') THEN$new641$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>expected_post THEN
  RAISE EXCEPTION 'Window641 history readback differs'; END IF;
END;
$history641$;
DO $window641$
DECLARE
 manifest CONSTANT JSONB:=$manifest641${"configuration_sha256":"51f53895739647654c13e59e4ebeeaba436203342271b302219c5993b027d946","new_configuration_sha256":"69befd3303a62b7ef7ceb28799f2082c83304b281884ddc6c3de693ba19b6e15","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.nov05_window641.v1","schedule":{"stage_2_close":"2026-11-06T06:30:00Z","final_scoring_close":"2026-11-06T08:30:00Z","publication_deadline":"2026-11-06T08:35:00Z"}}$manifest641$;
 r public.lab_arena_rounds; n public.lab_arena_rounds;
 h public.lab_arena_scoring_supersessions; revision BIGINT; request_hash TEXT;
BEGIN
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin' THEN
  RAISE EXCEPTION 'Window641 origin triggers required'; END IF;
 PERFORM 1 FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR UPDATE;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov05-recovery-window641-v1';
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc IS DISTINCT FROM jsonb_build_object('operation','nov05_recovery_window641_v1','manifest',manifest)
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR r.scoring_generation_floor<>0 THEN
   RAISE EXCEPTION 'Window641 replay differs'; END IF;
  RETURN; -- Later claims, accepted work and phase changes are never reset.
 END IF;
 IF r.status IS DISTINCT FROM 'stage2' OR r.status_generation IS DISTINCT FROM 9
  OR r.stage_generation IS DISTINCT FROM 7 OR r.scoring_generation_floor<>0
  OR r.evaluation_date IS DISTINCT FROM '2026-11-05'
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'configuration_sha256'
  OR r.configuration_doc#>>'{schedule,stage_2_close}' IS DISTINCT FROM '2026-11-06T05:30:00Z'
  OR r.configuration_doc#>>'{schedule,final_scoring_close}' IS DISTINCT FROM '2026-11-06T07:00:00Z'
  OR r.configuration_doc#>>'{schedule,publication_deadline}' IS DISTINCT FROM '2026-11-06T07:05:00Z'
  OR r.stage2_scoring_plan_doc IS NOT NULL
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR clock_timestamp()>='2026-11-06T05:30:00Z'::TIMESTAMPTZ
  OR (SELECT count(*) FROM public.lab_arena_restart_claim_control WHERE singleton)<>1
  OR EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE operator_paused OR guard_commitment<>''
   OR owner_commitment<>'' OR guard_expires_at IS NOT NULL OR candidate_commit<>'' OR restart_scope<>'' OR restart_phase<>'')
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=r.round_id AND prior.request_id='nov05-approved-infrastructure633-v1'
    AND prior.recovery_doc->>'operation'='nov05_infrastructure_recovery633_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc) THEN
  RAISE EXCEPTION 'Window641 exact round, clock, scope or control differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_trigger t JOIN pg_proc f ON f.oid=t.tgfoid
  WHERE t.tgrelid='public.lab_arena_rounds'::REGCLASS AND t.tgname='lab_arena_rounds_write_once'
   AND NOT t.tgisinternal AND t.tgenabled='O' AND t.tgtype=27
   AND t.tgfoid='public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE
   AND pg_get_userbyid(f.proowner)='lab_arena_owner' AND NOT f.prosecdef
   AND f.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}' AND f.proconfig=ARRAY['search_path=pg_catalog']::TEXT[]
   AND encode(extensions.digest(pg_get_functiondef(f.oid),'sha256'),'hex')='fc99c04a17205b97c55f522b445cd6533847eaaf48d2807ebd05e55ea9de5d25') THEN
  RAISE EXCEPTION 'Window641 immutable round guard differs'; END IF;
 n:=r;
 n.configuration_doc:=jsonb_set(jsonb_set(jsonb_set(r.configuration_doc,
  '{schedule,stage_2_close}',manifest#>'{schedule,stage_2_close}',FALSE),
  '{schedule,final_scoring_close}',manifest#>'{schedule,final_scoring_close}',FALSE),
  '{schedule,publication_deadline}',manifest#>'{schedule,publication_deadline}',FALSE);
 IF encode(extensions.digest(n.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'new_configuration_sha256' THEN
  RAISE EXCEPTION 'Window641 configuration postimage differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov05-recovery-window641-v1',request_hash,'nov05-user-approved-window641',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','nov05_recovery_window641_v1','manifest',manifest));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc WHERE round_id=r.round_id;
 IF (SELECT to_jsonb(a)-'updated_at' FROM public.lab_arena_rounds a WHERE a.round_id=r.round_id) IS DISTINCT FROM to_jsonb(n)-'updated_at'
  OR clock_timestamp()>='2026-11-06T05:30:00Z'::TIMESTAMPTZ THEN
  RAISE EXCEPTION 'Window641 exact readback differs'; END IF;
END;
$window641$;
COMMIT;
