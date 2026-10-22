-- DRAFT owner-controlled October22 continuation. No production application.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='60s';

DO $prepare544$
DECLARE
 rid CONSTANT TEXT:='arena-2026-10-22';sid CONSTANT TEXT:='sub-5f48ab5a81c56a43b1ce1b66048920a2';
 actor CONSTANT TEXT:='oct22-dependency-execution-recovery542';
 r public.lab_arena_rounds;n public.lab_arena_rounds;c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions;already public.lab_arena_scoring_supersessions;
 revision BIGINT;outcomes JSONB;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-542-phase1-v1';
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id IN('oct22-dependency-546-live-v1','oct22-dependency-546-terminal-v1'))THEN
  RAISE NOTICE 'OCT22_RECOVERY_ABORTED: no hold was acquired';RETURN;END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-545-final-v1')THEN RETURN;END IF;
 SELECT * INTO already FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-544-prepare-v1';
 IF FOUND THEN
  IF r.configuration_doc IS DISTINCT FROM already.new_round_doc->'configuration_doc'
   OR NOT c.operator_paused OR c.actor_ref<>actor OR c.pause_reason<>'oct22_dependency_phase2_transition'
   THEN RAISE EXCEPTION 'Oct22 prepared replay differs';END IF;
  RETURN;
 END IF;
 IF c.guard_generation<>(h.recovery_doc#>>'{control_before,guard_generation}')::BIGINT
  OR c.guard_commitment<>''OR c.owner_commitment<>''OR c.restart_phase<>''OR c.guard_expires_at IS NOT NULL
  OR c.captured_leases<>'[]'::JSONB
  OR(c.operator_paused AND(c.actor_ref<>actor OR c.pause_reason<>'oct22_dependency_phase2_transition'))
  THEN RAISE EXCEPTION 'Oct22 preparation control differs';END IF;
 IF r.status NOT IN('stage2','stage2_closed')OR r.scoring_generation_floor<>0
  OR r.configuration_doc IS DISTINCT FROM h.new_round_doc->'configuration_doc'
  OR r.publication_doc IS NOT NULL OR r.promotion_doc IS NOT NULL OR r.reward_basis_doc IS NOT NULL
  OR clock_timestamp()>='2026-10-22T20:30:02Z'::TIMESTAMPTZ
  THEN RAISE EXCEPTION 'Oct22 preparation round differs';END IF;
 IF(SELECT count(*)FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid
   AND a.kind='execute'AND a.attempt=3 AND a.icp_position BETWEEN 0 AND 9)<>10
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid
   AND a.kind='execute'AND a.attempt=3 AND(a.status NOT IN('accepted','failed')
    OR a.run_id<>rid||':'||sid||':2:'||a.icp_position||':3'
    OR a.result_doc->>'terminal_status'IS DISTINCT FROM a.terminal_cause
    OR(a.status='accepted'AND(a.terminal_cause<>'accepted'OR a.output_ref IS NULL))))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid AND a.kind='score')
  THEN RAISE EXCEPTION 'Oct22 actual replacement terminal set differs';END IF;
 IF(SELECT encode(extensions.digest(jsonb_agg(to_jsonb(a)ORDER BY run_id)::TEXT,'sha256'),'hex')
   FROM public.lab_arena_runs a WHERE a.round_id=rid AND a.submission_id=sid AND a.attempt IN(1,2))
   IS DISTINCT FROM h.recovery_doc->>'old_target_rows_sha256'
  THEN RAISE EXCEPTION 'Oct22 historical failures changed';END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('pending','leased','submitted'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_rounds other WHERE other.round_id<>rid
   AND other.configuration_doc->>'mode'='live'AND other.status NOT IN('published','cancelled')
   AND(other.status<>'open'OR(other.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ<=clock_timestamp()))
  OR EXISTS(SELECT 1 FROM(SELECT DISTINCT ON(call_identity)entry_kind,entry_doc,terminal_response
   FROM public.lab_arena_ledger WHERE run_id IN(SELECT run_id FROM public.lab_arena_runs
    WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3)AND call_identity IS NOT NULL
   ORDER BY call_identity,entry_id DESC)head WHERE entry_kind IN('reservation','dispatch')
    OR(entry_kind='uncertain'AND head.entry_doc#>'{call,call_succeeded}'IS DISTINCT FROM 'false'::JSONB)
    OR(entry_kind='settlement'AND jsonb_typeof(head.terminal_response->'call_succeeded')IS DISTINCT FROM 'boolean'))
  THEN RAISE EXCEPTION 'Oct22 preparation is not drained or would hold unrelated work';END IF;
 SELECT jsonb_agg(jsonb_build_object('run_id',run_id,'status',status,'terminal_cause',terminal_cause,
  'output_ref',output_ref,'result_hash',encode(extensions.digest(result_doc::TEXT,'sha256'),'hex'))ORDER BY run_id)
 INTO outcomes FROM public.lab_arena_runs WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3;
 n:=r; -- No round/configuration/worker-access mutation is needed for preparation.
 SELECT greatest(n.stage_generation,coalesce(max(generation_floor),0)+1)INTO revision
  FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(rid,revision,'oct22-dependency-544-prepare-v1','sha256:'||encode(extensions.digest(outcomes::TEXT,'sha256'),'hex'),
  actor,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}'::JSONB,
  jsonb_build_object('operation','oct22_dependency_judging_prepare_v1','actual_terminal_outcomes',outcomes,
   'control_before',to_jsonb(c),'original_control',h.recovery_doc->'control_before',
   'reason','Prepare canonical judging inputs while holding only this drained transition; all worker access remains unchanged.'));
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,
  pause_reason='oct22_dependency_phase2_transition',actor_ref=actor,updated_at=clock_timestamp()WHERE singleton;
END;
$prepare544$;
COMMIT;
