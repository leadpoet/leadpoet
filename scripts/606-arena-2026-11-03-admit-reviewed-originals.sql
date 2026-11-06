-- Run only after normal bounded review of both exact originals has stopped.
-- Append only valid source-bound passes. Keep all earlier entrants and work.
-- No review is approved here, no clock is extended and no source is changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $admit606$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds;
 h public.lab_arena_scoring_supersessions;
 s public.lab_arena_submissions; item TEXT; head public.lab_arena_ledger;
 reservation public.lab_arena_ledger; identity TEXT; admitted JSONB:='[]'; excluded JSONB:='[]';
 originals JSONB; proof JSONB:='{}'; participant JSONB; generation BIGINT;
 positions INTEGER; position INTEGER; assignment TEXT;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-03' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id
  AND request_id='nov03-admit-reviewed-originals606-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_admit_reviewed_originals606_v1'
   OR h.new_round_doc->'participants' IS DISTINCT FROM r.participants
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov03 admission606 replay differs'; END IF;
  RETURN; -- Exact completed result is immutable, including all exclusions.
 END IF;
 SELECT recovery_doc->'originals' INTO STRICT originals FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov03-admission-review605-v1'
  AND recovery_doc->>'operation'='nov03_admission_review605_v1'
  AND new_round_doc->'configuration_doc'=r.configuration_doc
  AND new_round_doc->'participants'=r.participants;
 IF current_setting('session_replication_role')<>'origin'
  OR r.status<>'stage2' OR r.scoring_generation_floor<>0
  OR r.publication_doc IS NOT NULL OR r.published_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.configuration_doc->>'execution_sequence_policy'<>'baseline_scored_first_v1'
  OR r.configuration_doc#>>'{schedule,final_scoring_close}'<>'2026-11-03T20:30:02Z'
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE code_review_status='reviewing'
   AND submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')) THEN
  RAISE EXCEPTION 'Nov03 admission606 round, clock or target review differs'; END IF;
 positions:=(r.configuration_doc->>'stage_1_icp_count')::INTEGER
  +(r.configuration_doc->>'stage_2_icp_count')::INTEGER;
 IF positions NOT BETWEEN 2 AND 100 OR jsonb_array_length(r.participants)<1 THEN
  RAISE EXCEPTION 'Nov03 admission606 frozen position contract differs'; END IF;
 n:=r;
 FOREACH item IN ARRAY ARRAY['sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a'] LOOP
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=item FOR UPDATE;
  IF NOT public.lab_arena__nov03_review605_v1(r,s)
   OR s.code_review_attempts NOT BETWEEN 2 AND 4
   OR s.code_review_status NOT IN('passed','rejected','error')
   OR s.code_review_expires_at IS NOT NULL
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE submission_id=item)
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(n.participants)p WHERE p->>'submission_id'=item) THEN
   RAISE EXCEPTION 'Nov03 admission606 original or current review differs: %',item; END IF;
  identity:='sha256:'||encode(extensions.digest('lab_arena.code_review:'||item||':'||s.code_review_attempts::TEXT,'sha256'),'hex');
  head:=public.lab_arena__ledger_head(identity);
  SELECT * INTO STRICT reservation FROM public.lab_arena_ledger WHERE call_identity=identity AND entry_kind='reservation';
  IF head.entry_kind NOT IN('settlement','uncertain') OR head.submission_id<>item
   OR head.round_id<>r.round_id OR head.operation_id<>'openrouter.code_review'
   OR reservation.submission_id<>item OR reservation.round_id<>r.round_id
   OR reservation.operation_id<>'openrouter.code_review'
   OR reservation.entry_doc->'attempt' IS DISTINCT FROM to_jsonb(s.code_review_attempts)
   OR reservation.entry_doc->>'review_model' IS DISTINCT FROM s.code_review_doc->>'model'
   OR (SELECT count(*) FILTER(WHERE entry_doc->>'review_model'='anthropic/claude-sonnet-5.5')>2
     OR count(*) FILTER(WHERE entry_doc->>'review_model'='openai/gpt-6.1-sol')>2
     OR count(*)<>s.code_review_attempts FROM public.lab_arena_ledger
     WHERE submission_id=item AND operation_id='openrouter.code_review' AND entry_kind='reservation') THEN
   RAISE EXCEPTION 'Nov03 admission606 review claim authority differs'; END IF;
  IF s.code_review_status='error' THEN
   -- A provider failure is excluded explicitly; it is never a security reject.
   -- Refusal/transient errors may finish only after their existing budget ends.
   IF NOT (s.code_review_doc->'retryable'='false'::JSONB
    AND (s.code_review_doc->>'error_code' NOT IN('code_review_provider_refused',
      'code_review_preparation_unavailable','code_review_provider_rate_limited',
      'code_review_provider_timeout','code_review_provider_unavailable','code_review_transport_error',
      'review_response_invalid','code_review_research_unverified')
     OR s.code_review_doc->'review_alternate_attempts'='2'::JSONB)) THEN
    RAISE EXCEPTION 'Nov03 admission606 review still retryable'; END IF;
  END IF;
  proof:=proof||jsonb_build_object(item,jsonb_build_object('reviewed_submission_sha256',
   encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex'),'code_review_status',s.code_review_status,
   'code_review_doc',s.code_review_doc,'attempts',s.code_review_attempts,
   'reservation',to_jsonb(reservation),'terminal',to_jsonb(head)));
  IF s.code_review_status='passed' THEN
   IF s.code_review_doc->'passed' IS DISTINCT FROM 'true'::JSONB
    OR s.code_review_doc->>'verdict' IS DISTINCT FROM 'pass'
    OR s.code_review_doc->'categories' IS DISTINCT FROM '[]'::JSONB
    OR s.code_review_doc#>>'{research_control,classification}' IS DISTINCT FROM 'model_directed'
    OR s.code_review_doc#>>'{research_control,policy_id}' IS DISTINCT FROM 'model_directed_research_v1'
    OR s.code_review_doc#>>'{research_control,reviewer_version}' IS DISTINCT FROM 'full_source_reviewer_research_v1'
    OR s.code_review_doc#>>'{research_control,policy_hash}' IS DISTINCT FROM 'sha256:1d30b4c2d838e84692b3cd9aa0fe0da3ff908f37b3836863b3937a3d93c9a73c'
    OR s.code_review_doc#>>'{research_control,source_archive_sha256}' IS DISTINCT FROM s.source_archive_sha256
    OR s.code_review_doc#>'{research_control,evidence_claims}' IS DISTINCT FROM '["finish_continue","observation_to_action"]'::JSONB
    OR head.entry_kind<>'settlement' OR head.entry_doc->>'review_status' IS DISTINCT FROM 'passed' THEN
    RAISE EXCEPTION 'Nov03 admission606 passing source proof differs'; END IF;
   participant:=jsonb_build_object('submission_id',item,'miner_hotkey',s.miner_hotkey,
    'source_ref',s.source_ref,'source_size_bytes',s.source_size_bytes,'is_king',FALSE);
   n.participants:=n.participants||jsonb_build_array(participant);
   admitted:=admitted||jsonb_build_array(item);
  ELSE
   excluded:=excluded||jsonb_build_array(jsonb_build_object('submission_id',item,
    'code_review_status',s.code_review_status,'error_code',s.code_review_doc->'error_code'));
  END IF;
 END LOOP;
 IF jsonb_array_length(n.participants)-1>(r.configuration_doc->>'max_challengers')::INTEGER
  OR (SELECT count(*) FROM jsonb_array_elements(n.participants)p WHERE p->'is_king'='true'::JSONB)<>1 THEN
  RAISE EXCEPTION 'Nov03 admission606 participant contract differs'; END IF;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,generation,'nov03-admit-reviewed-originals606-v1',
  'sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex'),'operator:nov03-admission606',
  txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(n),'{}',jsonb_build_object(
   'operation','nov03_admit_reviewed_originals606_v1','scoring_generation_floor',0,
   'admitted',admitted,'excluded',excluded,'reviews',proof,'stage',2,'icp_positions',positions));
 FOR item IN SELECT jsonb_array_elements_text(admitted) LOOP
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=item;
  UPDATE public.lab_arena_submissions SET status='frozen',frozen_at=clock_timestamp(),rejection_rule=NULL
   WHERE submission_id=item AND status='accepted' AND code_review_status='passed';
  IF NOT FOUND THEN RAISE EXCEPTION 'Nov03 admission606 freeze changed'; END IF;
  FOR position IN 0..positions-1 LOOP
   assignment:=r.round_id||':'||item||':2:'||position::TEXT;
   INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
    stage,icp_position,attempt,status,stage_generation)
   VALUES(assignment||':1',assignment,r.round_id,item,s.miner_hotkey,2,position,1,'pending',r.stage_generation);
  END LOOP;
 END LOOP;
 -- Ordinary terminal lifecycle for rows that did not earn admission. Keep
 -- the current private review error distinct from a genuine security reject.
 FOR item IN SELECT value->>'submission_id' FROM jsonb_array_elements(excluded) LOOP
  UPDATE public.lab_arena_submissions SET status='rejected',rejection_rule=CASE
   WHEN code_review_status='rejected' THEN 'code_review_rejected' ELSE 'code_review_incomplete' END
  WHERE submission_id=item AND status='accepted' AND code_review_status IN('rejected','error');
  IF NOT FOUND THEN RAISE EXCEPTION 'Nov03 admission606 exclusion changed'; END IF;
 END LOOP;
 UPDATE public.lab_arena_rounds SET participants=n.participants WHERE round_id=r.round_id;
 IF (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=r.round_id
   AND submission_id IN(SELECT jsonb_array_elements_text(admitted)))<>positions*jsonb_array_length(admitted)
  OR (SELECT participants FROM public.lab_arena_rounds WHERE round_id=r.round_id) IS DISTINCT FROM n.participants THEN
  RAISE EXCEPTION 'Nov03 admission606 readback differs'; END IF;
END;
$admit606$;
COMMIT;
