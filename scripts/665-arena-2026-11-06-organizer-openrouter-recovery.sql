-- One explicitly approved organizer-OpenRouter judgment per unfinished Nov6 source.
-- Never change old payers, ciphertexts, accepted work, grading, admissions or deadlines.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='90s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena665_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $helper_preflight665$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov06_organizer_openrouter665_v1(public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'1a255735c738f158e1c13b1d0c0a6c37f5b579f81e6af0c0e28d16278f5e9dee'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'sql') THEN
  RAISE EXCEPTION 'Nov6 organizer665 payer helper source or security differs'; END IF;
END;
$helper_preflight665$;

-- This is immutable payer identity, not claim authority. It remains true after
-- completion/publication so reconciliation and exact-call replay keep the payer.
CREATE OR REPLACE FUNCTION public.lab_arena__nov06_organizer_openrouter665_v1(
 p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $payer665$
 SELECT coalesce(p_run.round_id='arena-2026-11-06' AND p_run.kind='score'
  AND p_run.run_id=p_run.assignment_id||':'||p_run.attempt::TEXT
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_run.round_id AND h.request_id='nov06-organizer-openrouter665-v1'
    AND h.recovery_doc->>'operation'='round_operational_update_v1'
    AND h.recovery_doc->>'recovery_kind'='nov06_organizer_openrouter665_v1'
    AND h.recovery_doc->'organizer_openrouter_runs'->p_run.run_id=jsonb_build_object(
      'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,
      'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,
      'stage',p_run.stage,'icp_position',p_run.icp_position,'attempt',p_run.attempt,
      'scored_run_id',p_run.scored_run_id,'scoring_generation_floor',p_run.scoring_generation_floor,
      'judgment_cache_key',p_run.judgment_cache_key,'judgment_input_hash',p_run.judgment_input_hash,
      'judgment_scope_doc',p_run.judgment_scope_doc,'company_judgment_refs',p_run.company_judgment_refs)),FALSE);
$payer665$;
ALTER FUNCTION public.lab_arena__nov06_organizer_openrouter665_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov06_organizer_openrouter665_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $patch665$
DECLARE target RECORD; patch RECORD; function_oid OID; d TEXT; updated TEXT;
 metadata JSONB; identity JSONB;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_provider_funding(text,text)','32fada1e9d9b9f812a7d8a0ff8e0d013b84d8bbaf481eaf055fc84d8e94fb51a','3fc9adbb3808515dd61674e89669821b18798d93543c936db79a7ac0de099d1c','s'),
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','2acc6ff6c64ed797299cd18f203b3d62459720e2ae1ab9cb1f03a7850e62dd82','819ca4df91b5648c055525c34eb40ed0cf488e6c74ab32ab87417661f693c7d4','v'),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)','6c5ee308109b55b3b23ee1f1631d894cd72b8582ad1180c898e25f7ae9106ac0','d052adb5936f51d965aae574f61a567e7779417765aa197eac56efbb76f64f65','v'),
 ('public.lab_arena_expire_leases(text)','b5887c1e2afa6ac5ff342f1dc4ed1f0fc390b92b7302533a3ba31db964dbfaca','13418dc69e3f20877334cbdfc1fbafcd99db7d4697be92ba42dc78ae197edd13','v')
 ) expected(signature,preimage,postimage,volatility) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(
   pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO d,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid;
  IF d IS NULL OR identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,
   target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
   RAISE EXCEPTION 'Nov6 organizer665 function security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN
   RAISE EXCEPTION 'Nov6 organizer665 function source differs: %',target.signature; END IF;
  updated:=d;
  FOR patch IN SELECT * FROM (VALUES
  ('public.lab_arena_provider_funding(text,text)',
   $old665$  v_source := CASE$old665$,
   $new665$  v_source := CASE
    WHEN p_provider='openrouter'
      AND public.lab_arena__nov06_organizer_openrouter665_v1(v_run) THEN 'host'$new665$),
  ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
   $old665$      OR public.lab_arena__nov05_b66_judge639_v1(v_round,runs))$old665$,
   $new665$      OR public.lab_arena__nov05_b66_judge639_v1(v_round,runs)
      OR public.lab_arena__nov06_organizer_openrouter665_v1(runs))
    AND (NOT public.lab_arena__nov06_organizer_openrouter665_v1(runs)
      OR (clock_timestamp()<(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
        AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ))$new665$),
  ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
   $old665$  IF v_status = 'failed'
     AND (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5)$old665$,
   $new665$  IF v_status = 'failed'
     AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)
     AND (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5)$new665$),
  ('public.lab_arena_expire_leases(text)',
   $old665$    IF (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5) AND ($old665$,
   $new665$    IF NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)
      AND (v_run.kind<>'score' OR public.lab_arena__score_attempt_count_v1(v_run) < 5) AND ($new665$),
  ('public.lab_arena_expire_leases(text)',
   $old665$      AND public.lab_arena__score_attempt_count_v1(candidate) < 5$old665$,
   $new665$      AND public.lab_arena__score_attempt_count_v1(candidate) < 5
      AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(candidate)$new665$)
  ) changes(signature,old_text,new_text) WHERE signature=target.signature LOOP
   IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
    RAISE EXCEPTION 'Nov6 organizer665 function anchor differs: %',target.signature; END IF;
   updated:=replace(updated,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov6 organizer665 function postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Nov6 organizer665 function readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch665$;

-- Exact one-time data recovery. No active affected lease or provider call is
-- stopped. The release owner must let those jobs drain under the canonical guard.
DO $data665$
DECLARE rid CONSTANT TEXT:='arena-2026-11-06';
 cohort CONSTANT JSONB:=$cohort665$["sub-01f3260a133e59901869ca1a22ca34a3","sub-2524e20e6ce4b4f3c09b34021e58b789","sub-5ca8f69d36dfb626b8241ea247381a78","sub-653798a4f3525c6de9a0da97d52ad5f8","sub-696973ddfe30ef19396925b21166345e","sub-9e7290663ae9819dc6ef0d55d33fbe0e","sub-a0fff9b8e9550b935a4b7e5989219ad9","sub-a5947fae18bfc9c28ab593821825374e","sub-c8deb2248a68a1946cbbbe343dc07f89","sub-ed89cc69891b0222f523a730a9335063","sub-f5345040943729f422be5c33da8d53fe"]$cohort665$;
 credential_ciphertexts CONSTANT JSONB:=$ciphertexts665${"sub-01f3260a133e59901869ca1a22ca34a3":"c5c32f6ae460c7f724ceae5f77235b4047b874d26485f630e85d36d4f7bec515","sub-2524e20e6ce4b4f3c09b34021e58b789":"98487a21039805c28f21edd573473e0d649b8cf3b7525eb2170ec3cd0d2a9b03","sub-5ca8f69d36dfb626b8241ea247381a78":"38727eaed69c6411f7c9ef4c15f5a8a0bb4f252c696bbb197a287d222c1bfeff","sub-653798a4f3525c6de9a0da97d52ad5f8":"bf8894611d8187502086d008cbd7186e77c87c28f21682dd7c0e2d4609b6846b","sub-696973ddfe30ef19396925b21166345e":"b640406119a8eb05a0f46a0f3c07504eee32ba5ab8ceb483f8acb1b68cc0e8fa","sub-9e7290663ae9819dc6ef0d55d33fbe0e":"c0131ef2163dc12c9f928ba1d9f6c07d632ad85a31ea0978b3d38a24acb29e15","sub-a0fff9b8e9550b935a4b7e5989219ad9":"46b37b51eabb5a76037b87354658f440055e101aebbc5052fe143de83d0b14e6","sub-a5947fae18bfc9c28ab593821825374e":"a54d0ea43fe6bcabb785b3cea2e94b2779e7f57afa18af952ecbdf80ed34051f","sub-c8deb2248a68a1946cbbbe343dc07f89":"6750008d710310b3797d7aacf8f2d9230420ad81d224ce17dabad55156020e38","sub-ed89cc69891b0222f523a730a9335063":"06e978bbcea12dde45cbbf541ab6509b82b21ae60c8e00fc6603750821180247","sub-f5345040943729f422be5c33da8d53fe":"1236b35d32c6e59b879aa0c7a4f5091628980f76f21cf45662d1e7695496b266"}$ciphertexts665$;
 fingerprint CONSTANT TEXT:='sha256:b4cc1221d6885c769ae56cb8768ca82c5d164e1c568fd50f3771cc76bc5c454f';
 c public.lab_arena_restart_claim_control; r public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 source public.lab_arena_runs; parent public.lab_arena_runs; successor public.lab_arena_runs;
 owner public.lab_arena_submissions; proof RECORD; revision BIGINT; next_attempt INTEGER;
 originals JSONB; accepted JSONB; submissions JSONB; ledger_versions JSONB;
 payers JSONB; queued JSONB:='{}'::JSONB; closed JSONB:='{}'::JSONB; auth JSONB:='{}'::JSONB;
 before_round JSONB; request_hash TEXT; created INTEGER:=0; skipped INTEGER:=0; key TEXT; value JSONB;
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=rid AND request_id='nov06-organizer-openrouter665-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'round_operational_update_v1'
   OR h.recovery_doc->>'recovery_kind' IS DISTINCT FROM 'nov06_organizer_openrouter665_v1'
   OR h.recovery_doc->'cohort' IS DISTINCT FROM cohort
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc
   OR h.recovery_doc->'scoring_generation_floor' IS DISTINCT FROM to_jsonb(r.scoring_generation_floor) THEN
   RAISE EXCEPTION 'Nov6 organizer665 replay receipt differs'; END IF;
  FOR key,value IN SELECT * FROM jsonb_each(h.recovery_doc->'organizer_openrouter_runs') LOOP
   SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=key;
   IF NOT public.lab_arena__nov06_organizer_openrouter665_v1(successor)
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs later WHERE later.assignment_id=successor.assignment_id
      AND later.attempt>successor.attempt) THEN
    RAISE EXCEPTION 'Nov6 organizer665 replay identity or finite bound differs'; END IF;
  END LOOP;
  RETURN;
 END IF;
 IF (current_setting('session_replication_role')='origin'
  AND c.guard_commitment ~ '^sha256:[0-9a-f]{64}$'
  AND c.owner_commitment ~ '^sha256:[0-9a-f]{64}$'
  AND c.guard_expires_at>clock_timestamp() AND c.restart_scope IN('gateway','all')
  AND c.restart_phase IN('draining','gateway_destructive','gateway_ready','validator_destructive','validator_ready')
  AND c.guard_commitment=current_setting('leadpoet.arena665.guard_commitment',TRUE)
  AND c.owner_commitment=current_setting('leadpoet.arena665.owner_commitment',TRUE)
  AND c.guard_generation::TEXT=current_setting('leadpoet.arena665.guard_generation',TRUE)
  AND c.candidate_commit=current_setting('leadpoet.arena665.candidate_commit',TRUE)) IS NOT TRUE THEN
  RAISE EXCEPTION 'Nov6 organizer665 exact owned canonical guard differs'; END IF;
 IF (r.status IN('stage2_scoring','stage2_judged','scored')
  AND r.scoring_generation_floor=0 AND r.evaluation_date='2026-11-06'
  AND r.published_at IS NULL AND r.publication_doc IS NULL AND r.reward_activated_at IS NULL
  AND encode(extensions.digest(r.configuration_doc::TEXT,'sha256'),'hex')='5a761eeae62154946cef9af7e18d3a24544713221fb982d4bd6ba8ca91c9be4c'
  AND r.configuration_doc#>>'{schedule,final_scoring_close}'='2026-11-06T23:30:00Z'
  AND r.configuration_doc#>>'{schedule,publication_deadline}'='2026-11-06T23:59:00Z'
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  AND jsonb_array_length(cohort)=11 AND (SELECT count(*) FROM jsonb_object_keys(credential_ciphertexts))=11) IS NOT TRUE THEN
  RAISE EXCEPTION 'Nov6 organizer665 frozen round or deadline differs'; END IF;
 -- NOWAIT loses races without making a live writer a deadlock victim.
 LOCK TABLE public.lab_arena_runs,public.lab_arena_submissions,public.lab_arena_ledger
  IN SHARE ROW EXCLUSIVE MODE NOWAIT;
 IF (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=rid AND kind='execute'
      AND status='accepted' AND terminal_cause='accepted' AND output_ref IS NOT NULL)<>157
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id IN
    (SELECT jsonb_array_elements_text(cohort)) AND a.status IN('leased','submitted')) THEN
  RAISE EXCEPTION 'Nov6 organizer665 accepted executions or affected active jobs differ'; END IF;
 SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) INTO originals FROM public.lab_arena_runs a WHERE a.round_id=rid;
 SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) INTO accepted FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.status='accepted';
 SELECT jsonb_object_agg(s.submission_id,to_jsonb(s)) INTO submissions FROM public.lab_arena_submissions s WHERE s.round_id=rid;
 -- Force keyed run lookups. A round-only full ledger scan is too slow on
 -- production history and would extend this short writer lock unnecessarily.
 -- SAME-TRANSACTION tuple versions prove unchanged rows under the ledger writer
 -- lock. They are deliberately not saved as durable cross-transaction hashes.
 IF NOT EXISTS(SELECT 1 FROM pg_index i JOIN pg_class index_row ON index_row.oid=i.indexrelid
  WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_run_idx')
   AND i.indrelid='public.lab_arena_ledger'::regclass AND i.indisvalid AND i.indisready
   AND pg_get_userbyid(index_row.relowner)='lab_arena_owner'
   AND pg_get_indexdef(i.indexrelid)='CREATE INDEX lab_arena_ledger_run_idx ON public.lab_arena_ledger USING btree (run_id, entry_id) WHERE (run_id IS NOT NULL)') THEN
  RAISE EXCEPTION 'Nov6 organizer665 keyed ledger index differs'; END IF;
 SELECT jsonb_object_agg(l.entry_id::TEXT,jsonb_build_array(l.tableoid,l.ctid,l.xmin)) INTO ledger_versions
 FROM jsonb_object_keys(originals) original_run(run_id)
 CROSS JOIN LATERAL(SELECT found.entry_id,found.tableoid::TEXT AS tableoid,found.ctid::TEXT AS ctid,found.xmin::TEXT AS xmin FROM public.lab_arena_ledger found
  WHERE found.run_id=original_run.run_id AND found.round_id=rid OFFSET 0)l;
 SELECT jsonb_object_agg(a.run_id,jsonb_build_object('openrouter',public.lab_arena_provider_funding(a.run_id,'openrouter'),
  'deepline',public.lab_arena_provider_funding(a.run_id,'deepline'),'scrapingdog',public.lab_arena_provider_funding(a.run_id,'scrapingdog')))
 INTO payers FROM public.lab_arena_runs a WHERE a.round_id=rid;
 before_round:=to_jsonb(r);
 -- Each frozen owner has a bound direct rejection on the exact submitted key.
 FOR owner IN SELECT s.* FROM public.lab_arena_submissions s
  WHERE s.round_id=rid AND s.submission_id IN(SELECT jsonb_array_elements_text(cohort)) ORDER BY s.submission_id
 LOOP
  IF owner.status<>'frozen' OR owner.is_king
   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_submission_credentials credential
     WHERE credential.submission_id=owner.submission_id AND credential.miner_hotkey=owner.miner_hotkey
       AND credential.provider='openrouter' AND encode(extensions.digest(credential.ciphertext,'sha256'),'hex')=credential_ciphertexts->>owner.submission_id)
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p
   WHERE p->>'submission_id'=owner.submission_id AND p->>'miner_hotkey'=owner.miner_hotkey AND p->'is_king'='false'::JSONB) THEN
   RAISE EXCEPTION 'Nov6 organizer665 frozen owner differs'; END IF;
  SELECT reservation.run_id,reservation.call_identity,reservation.entry_id AS reservation_entry_id,
   head.entry_id AS terminal_entry_id INTO proof
  FROM (SELECT a.* FROM public.lab_arena_runs a
   WHERE a.round_id=rid AND a.submission_id=owner.submission_id AND a.miner_hotkey=owner.miner_hotkey
    AND a.kind IN('execute','score') AND a.status='failed' AND a.terminal_cause='credential_error' OFFSET 0)failed
  CROSS JOIN LATERAL(SELECT l.* FROM public.lab_arena_ledger l WHERE l.run_id=failed.run_id
   AND l.entry_kind='reservation' AND l.provider='openrouter' AND l.funding_source='miner_key' OFFSET 0)reservation
  JOIN LATERAL(SELECT l.* FROM public.lab_arena_ledger l WHERE l.call_identity=reservation.call_identity
   ORDER BY l.entry_id DESC LIMIT 1)head ON TRUE
  WHERE failed.round_id=rid AND failed.submission_id=owner.submission_id AND failed.miner_hotkey=owner.miner_hotkey
   AND failed.kind IN('execute','score') AND failed.status='failed' AND failed.terminal_cause='credential_error'
   AND (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage)
     IS NOT DISTINCT FROM (failed.round_id,failed.submission_id,failed.miner_hotkey,failed.stage)
   AND (head.run_id,head.round_id,head.submission_id,head.miner_hotkey,head.stage,head.provider,head.operation_id,head.funding_source)
    IS NOT DISTINCT FROM (reservation.run_id,reservation.round_id,reservation.submission_id,reservation.miner_hotkey,reservation.stage,reservation.provider,reservation.operation_id,reservation.funding_source)
   AND head.entry_kind='settlement' AND head.amount_microusd=0
   AND EXISTS(SELECT 1 FROM public.lab_arena_ledger dispatched WHERE dispatched.call_identity=reservation.call_identity
    AND dispatched.run_id=reservation.run_id AND dispatched.entry_kind='dispatch'
    AND (dispatched.round_id,dispatched.submission_id,dispatched.miner_hotkey,dispatched.stage,
      dispatched.provider,dispatched.operation_id,dispatched.funding_source)
     IS NOT DISTINCT FROM (reservation.round_id,reservation.submission_id,reservation.miner_hotkey,
      reservation.stage,reservation.provider,reservation.operation_id,reservation.funding_source)
    AND dispatched.entry_id>reservation.entry_id AND dispatched.entry_id<head.entry_id)
   AND head.terminal_response->'call_succeeded'='false'::JSONB
   AND head.terminal_response#>>'{account_failure_evidence,error_class}'='account_credential_failure'
   AND head.terminal_response#>'{account_failure_evidence,provider_status}'='401'::JSONB
   AND head.terminal_response#>'{account_failure_evidence,submitted_account_authentication}'='true'::JSONB
   AND public.lab_arena__champion_account_failure_matches(reservation.entry_doc,head.terminal_response->'account_failure_evidence')
  ORDER BY head.entry_id DESC LIMIT 1;
  IF NOT FOUND THEN RAISE EXCEPTION 'Nov6 organizer665 exact rejected-key proof missing: %',owner.submission_id; END IF;
  auth:=auth||jsonb_build_object(owner.submission_id,to_jsonb(proof));
 END LOOP;
 IF (SELECT count(*) FROM jsonb_object_keys(auth))<>11 THEN RAISE EXCEPTION 'Nov6 organizer665 cohort incomplete'; END IF;
 FOR source IN SELECT e.* FROM public.lab_arena_runs e
  WHERE e.round_id=rid AND e.kind='execute' AND e.status='accepted' AND e.terminal_cause='accepted'
   AND e.submission_id IN(SELECT jsonb_array_elements_text(cohort)) ORDER BY e.assignment_id
 LOOP
  IF EXISTS(SELECT 1 FROM public.lab_arena_runs done WHERE done.assignment_id=source.assignment_id||':score'
   AND done.kind='score' AND done.status='accepted') THEN skipped:=skipped+1; CONTINUE; END IF;
  SELECT * INTO parent FROM public.lab_arena_runs a WHERE a.assignment_id=source.assignment_id||':score'
   ORDER BY a.attempt DESC LIMIT 1;
  IF NOT FOUND OR parent.kind<>'score' OR parent.scored_run_id IS DISTINCT FROM source.run_id
   OR parent.scoring_generation_floor<>r.scoring_generation_floor OR parent.status NOT IN('pending','failed')
   OR parent.round_id<>rid OR parent.submission_id<>source.submission_id OR parent.miner_hotkey<>source.miner_hotkey
   OR parent.stage<>source.stage OR parent.icp_position<>source.icp_position
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.stage2_scoring_plan_doc->'work_items')item
     WHERE item->>'scored_run_id'=source.run_id AND item->>'submission_id'=source.submission_id
       AND item->>'icp_position'=source.icp_position::TEXT AND item->>'output_ref'=source.output_ref)
   OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id) THEN
   RAISE EXCEPTION 'Nov6 organizer665 unfinished source, plan or accounting differs'; END IF;
  IF parent.status='pending' THEN
   IF parent.runner_hotkey IS NOT NULL OR parent.claim_request_id IS NOT NULL OR parent.claim_request_hash IS NOT NULL
    OR parent.claim_response IS NOT NULL OR parent.lease_token_hash IS NOT NULL OR parent.lease_expires_at IS NOT NULL
    OR parent.result_doc IS NOT NULL OR parent.output_ref IS NOT NULL
    OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.run_id=parent.run_id)
    OR EXISTS(SELECT 1 FROM public.lab_arena_trajectory_events e WHERE e.run_id=parent.run_id) THEN
    RAISE EXCEPTION 'Nov6 organizer665 pending row was already used'; END IF;
   closed:=closed||jsonb_build_object(parent.run_id,to_jsonb(parent));
  END IF;
  next_attempt:=parent.attempt+1;
  IF next_attempt NOT BETWEEN 2 AND 6 THEN RAISE EXCEPTION 'Nov6 organizer665 attempt outside finite bound'; END IF;
  successor:=parent;
  successor.run_id:=parent.assignment_id||':'||next_attempt::TEXT; successor.attempt:=next_attempt;
  queued:=queued||jsonb_build_object(successor.run_id,jsonb_build_object(
   'assignment_id',successor.assignment_id,'round_id',successor.round_id,'submission_id',successor.submission_id,
   'miner_hotkey',successor.miner_hotkey,'stage',successor.stage,'icp_position',successor.icp_position,
   'attempt',successor.attempt,'scored_run_id',successor.scored_run_id,'scoring_generation_floor',successor.scoring_generation_floor,
   'judgment_cache_key',successor.judgment_cache_key,'judgment_input_hash',successor.judgment_input_hash,
   'judgment_scope_doc',successor.judgment_scope_doc,'company_judgment_refs',successor.company_judgment_refs));
 END LOOP;
 IF queued='{}'::JSONB THEN RAISE EXCEPTION 'Nov6 organizer665 has no unfinished source'; END IF;
 request_hash:='sha256:'||encode(extensions.digest(jsonb_build_object('cohort',cohort,'organizer_openrouter_runs',queued,'auth',auth)::TEXT,'sha256'),'hex');
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'nov06-organizer-openrouter665-v1',request_hash,'nov06-user-approved-organizer-openrouter665',
  txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',jsonb_build_object(
   'operation','round_operational_update_v1','recovery_kind','nov06_organizer_openrouter665_v1',
   'scoring_generation_floor',r.scoring_generation_floor,'cohort',cohort,'rejected_credential_fingerprint',fingerprint,
   'canonical_credential_ciphertext_sha256',credential_ciphertexts,
   'owned_guard_identity',jsonb_build_object('guard_commitment',c.guard_commitment,'owner_commitment',c.owner_commitment,
     'guard_generation',c.guard_generation,'candidate_commit',c.candidate_commit),
   'organizer_openrouter_runs',queued,'authentication_proofs',auth,'closed_pending_originals',closed,
   'preserved_accepted_run_hashes',(SELECT jsonb_object_agg(e.key,encode(extensions.digest(e.value::TEXT,'sha256'),'hex')) FROM jsonb_each(accepted)e),
   'reason','Explicit owner-authorized organizer OpenRouter for one new unfinished judge attempt. Preserve old miner charges, all accepted work, other provider payers, grading, admissions, deadlines and cumulative budgets. No automatic successor.'));
 FOR key,value IN SELECT * FROM jsonb_each(queued) LOOP
  SELECT * INTO STRICT parent FROM public.lab_arena_runs a WHERE a.assignment_id=value->>'assignment_id' ORDER BY a.attempt DESC LIMIT 1;
  IF parent.status='pending' THEN
   UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc=jsonb_build_object(
    'closed_at',clock_timestamp(),'previous_status','pending','recovery_reason','superseded_by_explicit_organizer_judgment665',
    'recovery_request_id','nov06-organizer-openrouter665-v1') WHERE run_id=parent.run_id;
  END IF;
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,
   status,stage_generation,scoring_generation_floor,kind,scored_run_id,previous_runner_hotkey,judgment_cache_key,
   judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,company_judgment_refs)
  VALUES(key,parent.assignment_id,rid,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position,
   (value->>'attempt')::INTEGER,'pending',r.stage_generation,r.scoring_generation_floor,'score',parent.scored_run_id,
   parent.runner_hotkey,parent.judgment_cache_key,parent.judgment_input_hash,parent.judgment_scope_doc,
   parent.judgment_group_leader,parent.judgment_group_miner_hotkeys,parent.company_judgment_refs);
  SELECT * INTO STRICT successor FROM public.lab_arena_runs WHERE run_id=key;
  IF NOT public.lab_arena__nov06_organizer_openrouter665_v1(successor)
   OR public.lab_arena_provider_funding(key,'openrouter')->>'funding_source'<>'host'
   OR public.lab_arena_provider_funding(key,'deepline')->>'funding_source'<>'miner_key'
   OR public.lab_arena_provider_funding(key,'scrapingdog')->>'funding_source'<>'miner_key'
   OR NOT public.lab_arena__judge_continuation_v1(r,successor) THEN
   RAISE EXCEPTION 'Nov6 organizer665 new identity, payer or continuation differs'; END IF;
  created:=created+1;
 END LOOP;
 IF created<>(SELECT count(*) FROM jsonb_object_keys(queued))
  OR (SELECT jsonb_object_agg(a.run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.status='accepted') IS DISTINCT FROM accepted
  OR (SELECT jsonb_object_agg(s.submission_id,to_jsonb(s)) FROM public.lab_arena_submissions s WHERE s.round_id=rid) IS DISTINCT FROM submissions
  OR (SELECT jsonb_object_agg(l.entry_id::TEXT,jsonb_build_array(l.tableoid,l.ctid,l.xmin))
    FROM jsonb_object_keys(originals) original_run(run_id)
    CROSS JOIN LATERAL(SELECT found.entry_id,found.tableoid::TEXT AS tableoid,found.ctid::TEXT AS ctid,found.xmin::TEXT AS xmin FROM public.lab_arena_ledger found
      WHERE found.run_id=original_run.run_id AND found.round_id=rid OFFSET 0)l) IS DISTINCT FROM ledger_versions
  OR EXISTS(SELECT 1 FROM jsonb_object_keys(queued) new_run(run_id)
    CROSS JOIN LATERAL(SELECT 1 FROM public.lab_arena_ledger found WHERE found.run_id=new_run.run_id LIMIT 1 OFFSET 0)l)
  OR (SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE a.round_id=rid) IS DISTINCT FROM before_round
  OR (SELECT to_jsonb(a) FROM public.lab_arena_restart_claim_control a WHERE singleton) IS DISTINCT FROM to_jsonb(c)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND NOT queued ? a.run_id
    AND CASE WHEN closed ? a.run_id THEN
     (to_jsonb(a)-ARRAY['status','terminal_cause','terminal_doc','updated_at']) IS DISTINCT FROM
      ((originals->a.run_id)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
     ELSE to_jsonb(a) IS DISTINCT FROM originals->a.run_id END)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND NOT queued ? a.run_id
    AND payers->a.run_id IS DISTINCT FROM jsonb_build_object('openrouter',public.lab_arena_provider_funding(a.run_id,'openrouter'),
     'deepline',public.lab_arena_provider_funding(a.run_id,'deepline'),'scrapingdog',public.lab_arena_provider_funding(a.run_id,'scrapingdog'))) THEN
  RAISE EXCEPTION 'Nov6 organizer665 preserved state or old payer differs'; END IF;
END;
$data665$;
DO $restore665$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena665_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena665_schema_acl) THEN
  RAISE EXCEPTION 'Nov6 organizer665 schema ACL differs'; END IF;
END;
$restore665$;
NOTIFY pgrst,'reload schema';
COMMIT;
