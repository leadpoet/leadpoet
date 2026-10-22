-- Owner-only escape from542–545. Preserve every run, lease, output and bill.
-- Does not declare evaluation complete or permit historical setup zeros to win.
-- No deadline predicate: it must remain usable after a missed judging deadline.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='60s';
DO $unwind546$
DECLARE
 rid CONSTANT TEXT:='arena-2026-10-22';sid CONSTANT TEXT:='sub-5f48ab5a81c56a43b1ce1b66048920a2';
 actor CONSTANT TEXT:='oct22-dependency-execution-recovery542';
 r public.lab_arena_rounds;n public.lab_arena_rounds;c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions;revision BIGINT;live_jobs INTEGER;unwind_request_id TEXT;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-542-phase1-v1';
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions WHERE round_id=rid
  AND request_id IN('oct22-dependency-545-final-v1','oct22-dependency-546-terminal-v1'))THEN RETURN;END IF;
 IF c.guard_generation<>(h.recovery_doc#>>'{control_before,guard_generation}')::BIGINT
  OR c.guard_commitment<>''OR c.owner_commitment<>''OR c.restart_phase<>''OR c.guard_expires_at IS NOT NULL
  OR c.captured_leases<>'[]'::JSONB
  OR(c.operator_paused AND(c.actor_ref<>actor OR c.pause_reason<>'oct22_dependency_phase2_transition'))
  THEN RAISE EXCEPTION 'Oct22 unwind refuses another owner or restart';END IF;
 IF r.status NOT IN('stage2','stage2_closed','stage2_scoring','stage2_judged','scored')OR r.scoring_generation_floor<>0
  OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL OR r.reward_basis_doc IS NOT NULL
  OR r.configuration_doc IS DISTINCT FROM h.new_round_doc->'configuration_doc'
  OR(SELECT count(*)FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3)<>10
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3 AND status='pending')

  THEN RAISE EXCEPTION 'Oct22 unwind requires exact already-dispatched recovery authority';END IF;
 SELECT count(*)INTO live_jobs FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid
  AND kind='execute'AND attempt=3 AND status IN('leased','submitted');
 n:=r; -- Worker roster, bans and frozen grading inputs never change.
 -- A live execute retains stage2 lease authority. Release only our hold/access
 -- boundary, let ordinary completion/expiry drain it, and rerun this escape
 -- after all terminal. Never change a live lease or discard a paid call.
 IF live_jobs>0 THEN
  IF r.status<>'stage2'THEN RAISE EXCEPTION 'Oct22 live execute has unexpected phase';END IF;
  unwind_request_id:='oct22-dependency-546-live-v1';
 ELSE
  -- SCORED avoids re-entering the frozen full-cohort stage initializer.
  -- Accepted new outputs without judges remain explicitly unfinished. A new
  -- infrastructure failure retains NULL; root must resolve its audited omission
  -- using existing restoration authority before ordinary publication can finish.
  IF EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid
    AND kind='execute'AND attempt=3 AND status NOT IN('accepted','failed'))THEN
   RAISE EXCEPTION 'Oct22 unwind terminal set differs';END IF;
  IF r.status IN('stage2','stage2_closed')THEN n.status:='scored';END IF;
  -- Any legitimate already-open judging phase and its rows remain untouched.
  unwind_request_id:='oct22-dependency-546-terminal-v1';
 END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND lab_arena_scoring_supersessions.request_id=unwind_request_id)THEN RETURN;END IF;
 n.status_generation:=r.status_generation+1;
 SELECT greatest(n.stage_generation,coalesce(max(generation_floor),0)+1)INTO revision
  FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,unwind_request_id,'sha256:'||encode(extensions.digest((to_jsonb(r)-'updated_at')::TEXT,'sha256'),'hex'),
  actor,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',
  jsonb_build_object('operation','oct22_dependency_owned_unwind_v1','live_execute_count',live_jobs,
   'reason','Release owned operator hold; preserve exact prior worker access; no score or historical cause is changed; unfinished evaluation is not completion.',
   'control_before',to_jsonb(c)));
 -- Forward status/score guards share our advisory lock. The intermediate false
 -- hold cannot be observed by a concurrent allocator before this commit.
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE WHERE singleton;
 UPDATE public.lab_arena_rounds SET status=n.status,
  status_generation=n.status_generation WHERE round_id=rid;
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=(h.recovery_doc#>>'{control_before,operator_paused}')::BOOLEAN,
  actor_ref=h.recovery_doc#>>'{control_before,actor_ref}',pause_reason=h.recovery_doc#>>'{control_before,pause_reason}',updated_at=clock_timestamp()
 WHERE singleton;
END;
$unwind546$;
COMMIT;
