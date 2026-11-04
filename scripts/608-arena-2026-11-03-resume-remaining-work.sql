-- Release only the November 3 operator hold after the canonical paired release.
-- The owner must first verify both running hosts and their exact image digests.
-- SQL proves retained state and receipts; it cannot prove a host's runtime.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $resume608$
DECLARE c public.lab_arena_restart_claim_control; r public.lab_arena_rounds;
 h public.lab_arena_scoring_supersessions; judge public.lab_arena_runs;
 actor CONSTANT TEXT:='nov03-forward-recovery608';
BEGIN
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 IF current_setting('session_replication_role') IS DISTINCT FROM 'origin'
  OR c.guard_generation<=754 OR c.guard_commitment IS DISTINCT FROM ''
  OR c.owner_commitment IS DISTINCT FROM '' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit IS DISTINCT FROM '' OR c.restart_scope IS DISTINCT FROM ''
  OR c.restart_phase IS DISTINCT FROM '' OR c.captured_leases IS DISTINCT FROM '[]'::JSONB THEN
  RAISE EXCEPTION 'Nov03 resume608 canonical guard is not released'; END IF;
 IF NOT c.operator_paused THEN
  IF c.actor_ref IS DISTINCT FROM actor OR c.pause_reason IS DISTINCT FROM '' THEN
   RAISE EXCEPTION 'Nov03 resume608 unpaused actor differs'; END IF;
  RETURN; -- Exact completed resume replay does not alter work that has started.
 END IF;
 IF c.pause_reason IS DISTINCT FROM 'nov03_improved_verifier_required603' THEN
  RAISE EXCEPTION 'Nov03 resume608 operator hold differs'; END IF;
 -- Keep the no-active-work check stable through this short release transaction.
 LOCK TABLE public.lab_arena_runs,public.lab_arena_submissions IN SHARE MODE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-03' FOR SHARE;
 IF r.status IS DISTINCT FROM 'stage2' OR r.scoring_generation_floor IS DISTINCT FROM 0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.configuration_doc#>>'{schedule,stage_1_scoring_close}' IS NULL
  OR r.configuration_doc#>>'{schedule,stage_2_close}' IS NULL
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
  OR clock_timestamp()>=(r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('leased','submitted'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE code_review_status='reviewing') THEN
  RAISE EXCEPTION 'Nov03 resume608 round, deadline or active work differs'; END IF;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov03-forward-only-verifier604-v1'
  AND recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
  AND recovery_doc->'scoring_generation_floor'='0'::JSONB
  AND new_round_doc->'configuration_doc'=r.configuration_doc;
 IF (SELECT count(*) FROM jsonb_object_keys(h.recovery_doc->'preserved_judgments'))<>9
  OR (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=r.round_id AND kind='score' AND status='accepted')<>9
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=r.round_id AND prior.request_id='nov03-admission-review605-v1'
    AND prior.recovery_doc->>'operation'='nov03_admission_review605_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions prior
   WHERE prior.round_id=r.round_id AND prior.request_id='nov03-baseline8-corrected607-v1'
    AND prior.recovery_doc->>'operation'='nov03_baseline8_corrected_judge607_v1'
    AND prior.new_round_doc->'configuration_doc'=r.configuration_doc) THEN
  RAISE EXCEPTION 'Nov03 resume608 recovery receipts differ'; END IF;
 FOR judge IN SELECT * FROM public.lab_arena_runs
  WHERE round_id=r.round_id AND kind='score' AND status='accepted' LOOP
  IF public.lab_arena__nov03_preserved_judge_v1(r,judge) IS NULL THEN
   RAISE EXCEPTION 'Nov03 resume608 original judgment proof differs'; END IF;
 END LOOP;
 SELECT * INTO STRICT judge FROM public.lab_arena_runs
 WHERE run_id='arena-2026-11-03:baseline-2026-11-03:1:8:score:6';
 IF judge.status IS DISTINCT FROM 'pending'
  OR public.lab_arena__nov03_corrected_judge607_v1(r,judge) IS DISTINCT FROM TRUE THEN
  RAISE EXCEPTION 'Nov03 resume608 corrected pending judge differs'; END IF;
 UPDATE public.lab_arena_restart_claim_control
 SET operator_paused=FALSE,pause_reason='',actor_ref=actor,updated_at=clock_timestamp()
 WHERE singleton AND to_jsonb(lab_arena_restart_claim_control)=to_jsonb(c);
 IF NOT FOUND THEN RAISE EXCEPTION 'Nov03 resume608 hold changed'; END IF;
END;
$resume608$;
COMMIT;
