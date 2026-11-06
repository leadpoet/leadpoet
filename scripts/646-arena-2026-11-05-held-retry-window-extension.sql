-- Third exact approved Nov5 infrastructure-recovery schedule extension only.
-- Preserve phase, all work, billing, scoring policy, source and other rounds.
-- Keep first-attempt host failures eligible for ordinary retries after expiry.
-- Normal early completion remains available; no attempts are created or reset.
-- Preserve the exact execution phase under clear control or our owned hold.
-- Applying after the previous cutoff is safe only while this phase remains intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
-- Reuse the existing typed window operation; its reader already preserves floor0.
DO $window646$
DECLARE
 manifest CONSTANT JSONB:=$manifest646${"configuration_sha256":"55beda7a80b7ac9e9c76e2e2fa9d1f30495dd6e05d6354be2db7dd5eda605996","new_configuration_sha256":"7687d54874aa88d1783c44065e22410c469d6d4cc910549ee967c29b5bfe977e","round_id":"arena-2026-11-05","schema_version":"leadpoet.lab_arena.nov05_window646.v1","schedule":{"stage_2_close":"2026-11-06T08:30:00Z","final_scoring_close":"2026-11-06T09:30:00Z","publication_deadline":"2026-11-06T09:35:00Z"}}$manifest646$;
 r public.lab_arena_rounds; n public.lab_arena_rounds;
 c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions; revision BIGINT; request_hash TEXT;
BEGIN
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin' THEN
  RAISE EXCEPTION 'Window646 origin triggers required'; END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR UPDATE;
 request_hash:='sha256:'||encode(extensions.digest(manifest::TEXT,'sha256'),'hex');
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov05-recovery-window646-v1';
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash
   OR h.recovery_doc IS DISTINCT FROM jsonb_build_object('operation','nov05_recovery_window641_v1','manifest',manifest)
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR r.scoring_generation_floor<>0 THEN
   RAISE EXCEPTION 'Window646 replay differs'; END IF;
  RETURN; -- Later claims, accepted work and phase changes are never reset.
 END IF;
 IF r.status IS DISTINCT FROM 'stage2' OR r.status_generation IS DISTINCT FROM 9
  OR r.stage_generation IS DISTINCT FROM 7 OR r.scoring_generation_floor<>0
  OR r.evaluation_date IS DISTINCT FROM '2026-11-05'
  OR encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'configuration_sha256'
  OR r.configuration_doc#>>'{schedule,stage_2_close}' IS DISTINCT FROM '2026-11-06T07:30:00Z'
  OR r.configuration_doc#>>'{schedule,final_scoring_close}' IS DISTINCT FROM '2026-11-06T09:30:00Z'
  OR r.configuration_doc#>>'{schedule,publication_deadline}' IS DISTINCT FROM '2026-11-06T09:35:00Z'
  OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions applied WHERE applied.round_id=r.round_id AND applied.request_id='nov05-b66-post-repair639-v1')
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs applied WHERE applied.assignment_id='arena-2026-11-05:sub-b66ad20113a4402633ba875964ed4b80:2:3:score' AND applied.attempt>=6)
  OR r.stage2_scoring_plan_doc IS NOT NULL
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR clock_timestamp()>='2026-11-06T08:30:00Z'::TIMESTAMPTZ
  OR (SELECT count(*) FROM public.lab_arena_restart_claim_control WHERE singleton)<>1
  OR EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control WHERE (operator_paused AND NOT(actor_ref='nov05-primary-route-ci-hold' AND pause_reason='Temporary Nov5 primary routing pending protected CI; new claims only.')) OR guard_commitment<>''
   OR owner_commitment<>'' OR guard_expires_at IS NOT NULL OR candidate_commit<>'' OR restart_scope<>'' OR restart_phase<>'' OR captured_leases<>'[]'::JSONB)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=r.round_id AND prior.request_id='nov05-recovery-window644-v1'
    AND prior.recovery_doc->>'operation'='nov05_recovery_window641_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc) THEN
  RAISE EXCEPTION 'Window646 exact round, clock, scope or control differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_trigger t JOIN pg_proc f ON f.oid=t.tgfoid
  WHERE t.tgrelid='public.lab_arena_rounds'::REGCLASS AND t.tgname='lab_arena_rounds_write_once'
   AND NOT t.tgisinternal AND t.tgenabled='O' AND t.tgtype=27
   AND t.tgfoid='public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE
   AND pg_get_userbyid(f.proowner)='lab_arena_owner' AND NOT f.prosecdef
   AND f.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}' AND f.proconfig=ARRAY['search_path=pg_catalog']::TEXT[]
   AND encode(extensions.digest(pg_get_functiondef(f.oid),'sha256'),'hex')='fc99c04a17205b97c55f522b445cd6533847eaaf48d2807ebd05e55ea9de5d25') THEN
  RAISE EXCEPTION 'Window646 immutable round guard differs'; END IF;
 n:=r;
 n.configuration_doc:=jsonb_set(jsonb_set(jsonb_set(r.configuration_doc,
  '{schedule,stage_2_close}',manifest#>'{schedule,stage_2_close}',FALSE),
  '{schedule,final_scoring_close}',manifest#>'{schedule,final_scoring_close}',FALSE),
  '{schedule,publication_deadline}',manifest#>'{schedule,publication_deadline}',FALSE);
 IF encode(extensions.digest(n.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM manifest->>'new_configuration_sha256' THEN
  RAISE EXCEPTION 'Window646 configuration postimage differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov05-recovery-window646-v1',request_hash,'nov05-user-approved-window646',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','nov05_recovery_window641_v1','manifest',manifest));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc WHERE round_id=r.round_id;
 IF (SELECT to_jsonb(a)-'updated_at' FROM public.lab_arena_rounds a WHERE a.round_id=r.round_id) IS DISTINCT FROM to_jsonb(n)-'updated_at'
  OR clock_timestamp()>='2026-11-06T08:30:00Z'::TIMESTAMPTZ
  OR (SELECT to_jsonb(control) FROM public.lab_arena_restart_claim_control control WHERE singleton) IS DISTINCT FROM to_jsonb(c) THEN
  RAISE EXCEPTION 'Window646 exact readback differs'; END IF;
END;
$window646$;
COMMIT;
