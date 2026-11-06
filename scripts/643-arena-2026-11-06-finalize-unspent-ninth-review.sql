-- Finalize the exact unused637 grant after native miner planning closed it.
-- Audit exclusion only: no new review, judgment, attempt, admission or deadline.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
DO $unspent643$
DECLARE
 rid CONSTANT TEXT:='arena-2026-11-06';
 sid CONSTANT TEXT:='sub-43a929eb0e2e487f68014f48e7d80e6b';
 r public.lab_arena_rounds; s public.lab_arena_submissions;
 h public.lab_arena_scoring_supersessions; done public.lab_arena_scoring_supersessions;
 c public.lab_arena_restart_claim_control; original JSONB; journal JSONB;
 rows_before JSONB; runs_before JSONB; ledger_before JSONB; receipts_before JSONB;
 control_before JSONB; target_before JSONB; proof JSONB; generation BIGINT;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid FOR UPDATE;
 SELECT * INTO done FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='nov06-admit-ninth-reviewed638-v1';
 IF FOUND THEN
  IF done.actor_ref IS DISTINCT FROM 'operator:nov06-finalize-unspent643'
   OR done.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_admit_late_reviewed636_v1'
   OR done.recovery_doc->'admitted' IS DISTINCT FROM '[]'::JSONB
   OR done.recovery_doc->'excluded' IS DISTINCT FROM jsonb_build_array(sid)
   OR done.recovery_doc->>'source_request_id' IS DISTINCT FROM 'nov06-ninth-review637-v1'
   OR done.recovery_doc->>'source_request_hash' IS DISTINCT FROM 'sha256:dda642385905dab62b935c84aa21ef99fb3c0ef1f2f16e058dc9cd0ebfa72cfe'
   OR done.request_hash IS DISTINCT FROM 'sha256:'||encode(extensions.digest((done.recovery_doc->'reviews')::TEXT,'sha256'),'hex')
   OR done.recovery_doc#>>'{reviews,sub-43a929eb0e2e487f68014f48e7d80e6b,outcome}' IS DISTINCT FROM 'preminer_gate_closed_without_fourth'
   OR s.status IS DISTINCT FROM 'rejected' OR s.rejection_rule IS DISTINCT FROM 'code_review_incomplete'
   OR s.code_review_attempts IS DISTINCT FROM 3 OR s.code_review_status IS DISTINCT FROM 'error'
   OR to_jsonb(s) IS DISTINCT FROM done.recovery_doc->'terminal_submission' THEN
   RAISE EXCEPTION 'Nov06 unspent643 replay differs'; END IF;
  RETURN;
 END IF;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='nov06-ninth-review637-v1';
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 original:=h.recovery_doc->'originals'->sid;
 SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) INTO journal FROM public.lab_arena_ledger l
  WHERE submission_id=sid AND operation_id='openrouter.code_review';
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin'
  OR h.request_hash IS DISTINCT FROM 'sha256:dda642385905dab62b935c84aa21ef99fb3c0ef1f2f16e058dc9cd0ebfa72cfe'
  OR h.created_at IS DISTINCT FROM '2026-11-06T05:26:24.67524Z'::TIMESTAMPTZ
  OR h.recovery_doc->>'operation' IS DISTINCT FROM 'nov06_late_review635_v1'
  OR h.recovery_doc->'max_additional_reservations' IS DISTINCT FROM '1'::JSONB
  OR (SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'originals'))<>1
  OR r.status NOT IN('stage2','stage2_closed','stage2_scoring','stage2_judged','scored','published')
  OR r.configuration_doc IS DISTINCT FROM h.old_round_doc->'configuration_doc'
  OR r.participants IS DISTINCT FROM h.old_round_doc->'participants'
  OR jsonb_array_length(r.participants)<>18
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.stage=2
    AND a.submission_id<>'baseline-2026-11-06' AND EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants) p WHERE p->>'submission_id'=a.submission_id))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE submission_id=sid)
  OR EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants) p WHERE p->>'submission_id'=sid)
  OR c.guard_commitment<>'' OR c.owner_commitment<>'' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit<>'' OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::JSONB
  OR s.round_id IS DISTINCT FROM rid OR s.status IS DISTINCT FROM 'accepted'
  OR s.code_review_status IS DISTINCT FROM 'error' OR s.code_review_attempts IS DISTINCT FROM 3
  OR s.code_review_expires_at IS NOT NULL OR s.is_king OR s.frozen_at IS NOT NULL
  OR (to_jsonb(s)-ARRAY['status','updated_at','code_review_doc']::TEXT[]) IS DISTINCT FROM
    ((original->'submission')-ARRAY['status','updated_at','code_review_doc']::TEXT[])
  OR s.code_review_doc IS DISTINCT FROM (original#>'{submission,code_review_doc}')||jsonb_build_object('retryable',TRUE,
    'nov06_late_review',jsonb_build_object('request_id','nov06-ninth-review637-v1','expires_at',h.created_at+INTERVAL '30 minutes'))
  OR s.source_archive_sha256 IS DISTINCT FROM 'sha256:5ae54960007f2cc87130634b62f5dff3cbe40d5978dd59640fce4574b9855301'
  OR original->>'manifest_sha256' IS DISTINCT FROM 'sha256:30d0d5baddcde4edc7f11ad17a5724108dd9138bf3115a4cbf9eca0d9bee06fd'
  OR journal IS DISTINCT FROM original->'ledger' OR jsonb_array_length(journal)<>9
  OR (SELECT count(*) FROM public.lab_arena_ledger WHERE submission_id=sid AND entry_kind='reservation' AND operation_id='openrouter.code_review')<>3
  OR (SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'preserved'))<>29
  OR (SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=rid AND submission_id<>sid)<>29
  OR EXISTS(SELECT 1 FROM jsonb_each_text(h.recovery_doc->'preserved') keep
    LEFT JOIN public.lab_arena_submissions kept ON kept.submission_id=keep.key
    WHERE encode(extensions.digest(to_jsonb(kept)::TEXT,'sha256'),'hex') IS DISTINCT FROM keep.value) THEN
  RAISE EXCEPTION 'Nov06 unspent643 exact grant, unused source, ledger or miner planning differs'; END IF;
 SELECT jsonb_object_agg(submission_id,to_jsonb(a)) INTO rows_before FROM public.lab_arena_submissions a WHERE round_id=rid AND submission_id<>sid;
 SELECT coalesce(jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}') INTO runs_before FROM public.lab_arena_runs a WHERE round_id=rid;
 SELECT coalesce(jsonb_object_agg(entry_id::TEXT,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}') INTO ledger_before FROM public.lab_arena_ledger a WHERE round_id=rid;
 SELECT jsonb_agg(to_jsonb(a) ORDER BY generation_floor) INTO receipts_before FROM public.lab_arena_scoring_supersessions a WHERE round_id=rid;
 control_before:=to_jsonb(c);target_before:=to_jsonb(s);
 UPDATE public.lab_arena_submissions SET status='rejected',rejection_rule='code_review_incomplete'
  WHERE submission_id=sid AND status='accepted' AND code_review_status='error' AND code_review_attempts=3;
 IF NOT FOUND THEN RAISE EXCEPTION 'Nov06 unspent643 terminal projection differs'; END IF;
 SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=sid;
 IF (to_jsonb(s)-ARRAY['status','rejection_rule','updated_at']::TEXT[]) IS DISTINCT FROM
   (target_before-ARRAY['status','rejection_rule','updated_at']::TEXT[])
  OR (SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=rid) IS DISTINCT FROM to_jsonb(r)
  OR (SELECT jsonb_object_agg(submission_id,to_jsonb(a)) FROM public.lab_arena_submissions a WHERE round_id=rid AND submission_id<>sid) IS DISTINCT FROM rows_before
  OR (SELECT coalesce(jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}') FROM public.lab_arena_runs a WHERE round_id=rid) IS DISTINCT FROM runs_before
  OR (SELECT coalesce(jsonb_object_agg(entry_id::TEXT,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')),'{}') FROM public.lab_arena_ledger a WHERE round_id=rid) IS DISTINCT FROM ledger_before
  OR (SELECT to_jsonb(a) FROM public.lab_arena_restart_claim_control a WHERE singleton) IS DISTINCT FROM control_before
  OR (SELECT jsonb_agg(to_jsonb(a) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions a WHERE round_id=rid) IS DISTINCT FROM receipts_before THEN
  RAISE EXCEPTION 'Nov06 unspent643 changed preserved authority'; END IF;
 proof:=jsonb_build_object(sid,jsonb_build_object('submission',target_before,'ledger',journal,
  'outcome','preminer_gate_closed_without_fourth','additional_reservations',0,'additional_cost_microusd',0));
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,generation,'nov06-admit-ninth-reviewed638-v1','sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),
  'operator:nov06-finalize-unspent643',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',jsonb_build_object(
   'operation','nov06_admit_late_reviewed636_v1','admitted','[]'::JSONB,'excluded',jsonb_build_array(sid),
   'reviews',proof,'terminal_submission',to_jsonb(s),'closure_reason','miner_planning_closed_unused_review',
   'source_request_id',h.request_id,'source_request_hash',h.request_hash));
END;
$unspent643$;
COMMIT;
