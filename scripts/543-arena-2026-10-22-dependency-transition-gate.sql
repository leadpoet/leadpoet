-- DRAFT: exact owner-controlled October22 data repair; no production application.
-- No function, trigger, permission, scorer, bank, historical run or ledger changes.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='3s';
SET LOCAL statement_timeout='60s';

DO $gate543$
DECLARE
 rid CONSTANT TEXT:='arena-2026-10-22';sid CONSTANT TEXT:='sub-5f48ab5a81c56a43b1ce1b66048920a2';actor CONSTANT TEXT:='oct22-dependency-execution-recovery542';
 r public.lab_arena_rounds;c public.lab_arena_restart_claim_control;h public.lab_arena_scoring_supersessions;
 jobs INTEGER; pending_jobs INTEGER; live_jobs INTEGER; invalid_jobs INTEGER;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO STRICT h FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id='oct22-dependency-542-phase1-v1';
 IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions
  WHERE round_id=rid AND request_id IN('oct22-dependency-546-live-v1','oct22-dependency-546-terminal-v1'))THEN
  RAISE NOTICE 'OCT22_RECOVERY_ABORTED: no hold was acquired';RETURN;END IF;
 IF c.guard_generation<>(h.recovery_doc#>>'{control_before,guard_generation}')::BIGINT
  OR c.guard_commitment<>''OR c.owner_commitment<>''OR c.restart_phase<>''OR c.guard_expires_at IS NOT NULL
  OR c.captured_leases<>'[]'::JSONB THEN RAISE EXCEPTION 'Oct22 gate conflicts with restart'; END IF;
 IF c.operator_paused THEN
  IF c.actor_ref=actor AND c.pause_reason='oct22_dependency_phase2_transition' THEN RETURN; END IF;
  RAISE EXCEPTION 'Oct22 gate belongs to another operator'; END IF;
 SELECT count(*),count(*)FILTER(WHERE status='pending'),count(*)FILTER(WHERE status='leased'AND lease_expires_at>clock_timestamp()),
  count(*)FILTER(WHERE status NOT IN('accepted','failed','leased')OR stage_generation<>r.stage_generation
   OR assignment_id<>rid||':'||sid||':2:'||icp_position OR run_id<>assignment_id||':3'
   OR icp_position NOT BETWEEN 0 AND 9 OR(status='leased'AND(lease_expires_at IS NULL OR lease_expires_at<=clock_timestamp())))
 INTO jobs,pending_jobs,live_jobs,invalid_jobs FROM public.lab_arena_runs
 WHERE round_id=rid AND submission_id=sid AND kind='execute'AND attempt=3;
 IF r.status<>'stage2'OR r.configuration_doc IS DISTINCT FROM h.new_round_doc->'configuration_doc'
  OR r.stage_generation<>(h.new_round_doc->>'stage_generation')::BIGINT
  OR jobs<>10 OR pending_jobs<>0 OR live_jobs<1 OR invalid_jobs<>0 THEN
  RAISE NOTICE 'OCT22_GATE_MISSED: no hold was acquired'; RETURN; END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.status IN('pending','leased','submitted')
   AND NOT(a.round_id=rid AND a.submission_id=sid AND a.kind='execute'AND a.attempt=3))
  OR EXISTS(SELECT 1 FROM public.lab_arena_rounds other WHERE other.round_id<>rid
   AND other.configuration_doc->>'mode'='live'AND other.status NOT IN('published','cancelled')
   AND(other.status<>'open'OR(other.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ<=clock_timestamp()))
  THEN RAISE EXCEPTION 'Oct22 gate would hold unrelated work'; END IF;
 -- Native completion needs roundFORSHARE. The heldFORUPDATE prevents the last
 -- completion from winning after this exact read but before holdcommit.
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE,
  pause_reason='oct22_dependency_phase2_transition',actor_ref=actor,updated_at=clock_timestamp()
 WHERE singleton AND NOT operator_paused AND guard_commitment='';
 IF NOT FOUND THEN RAISE EXCEPTION 'Oct22 gate control changed'; END IF;
END;
$gate543$;
COMMIT;
