-- Apply only after the exact 372 recovery, a completed paired canonical
-- gateway/validator release, and independent provider health proof. The
-- operator must bind those receipts to this exact SQL before application.
-- This step only lifts the October 1 operator claim hold.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0));
LOCK TABLE public.lab_arena_restart_claim_control IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_rounds IN SHARE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
LOCK TABLE public.lab_arena_trajectory_events IN SHARE MODE;
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;

DO $oct01_release$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_old_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-01T23:00:02Z","publication_deadline":"2026-10-01T23:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-01T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-01T17:00:02Z","stage_2_start":"2026-10-01T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
  v_new_schedule CONSTANT JSONB :=
    '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-02T12:00:02Z","publication_deadline":"2026-10-02T12:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-01T20:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-02T08:00:02Z","stage_2_start":"2026-10-01T20:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB;
BEGIN
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF NOT FOUND OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.candidate_commit <> ''
     OR v_control.restart_phase <> ''
     OR v_control.restart_scope <> ''
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct01 release conflicts with restart guard'
      USING ERRCODE='55000';
  END IF;
  IF NOT v_control.operator_paused
     AND v_control.pause_reason = ''
     AND v_control.actor_ref = 'oct01-provider-recovery373' THEN
    RETURN; -- Exact-owner replay changes no timestamp or other row.
  END IF;
  IF NOT v_control.operator_paused
     OR v_control.pause_reason <> 'oct01_deepline_outage'
     OR v_control.actor_ref !~ '^canonical-active-release:[0-9a-f]{40}$'
     OR v_control.actor_ref =
          'canonical-active-release:07016ffd02e174b6deaaa136e0e8e216d498a8d2'
     OR v_control.guard_generation <= 301 THEN
    RAISE EXCEPTION 'Oct01 release does not own completed provider hold'
      USING ERRCODE='55000';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01' FOR SHARE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01-r372archive' FOR SHARE;
  IF v_round.round_id IS NULL OR v_archive.round_id IS NULL
     OR v_round.status <> 'stage1_scoring'
     OR v_round.status_generation <> 10 OR v_round.stage_generation <> 8
     OR v_round.configuration_doc->'schedule' IS DISTINCT FROM v_new_schedule
     OR pg_catalog.encode(extensions.digest(
          pg_catalog.jsonb_set(v_round.configuration_doc,'{schedule}',v_old_schedule)::TEXT,
          'sha256'),'hex') IS DISTINCT FROM
          'e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b'
     OR v_round.benchmark_ref <> 'arena/arena-2026-10-01/benchmark.json'
     OR v_round.evaluation_date <> '2026-10-01'
     OR v_round.icp_set_date <> DATE '2026-09-30'
     OR v_round.stage1_scoring_plan_doc IS NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR pg_catalog.jsonb_array_length(v_round.stage1_scoring_plan_doc->'work_items') <> 10
     OR v_round.finalists IS NOT NULL OR v_round.published_at IS NOT NULL
     OR v_round.publication_doc IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM
          '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19'
     OR (SELECT pg_catalog.encode(extensions.digest(icps::TEXT,'sha256'),'hex')
         FROM public.qualification_private_icp_sets WHERE set_id=20260930) IS DISTINCT FROM
          '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
           'sha256'),'hex') FROM public.lab_arena_submissions s
           WHERE round_id=v_round.round_id) IS DISTINCT FROM
          '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR v_archive.status <> 'cancelled'
     OR v_archive.configuration_doc->>'mode' <> 'shadow'
     OR v_archive.configuration_doc->>'recovery_source_round_id' <> v_round.round_id
     OR v_archive.configuration_doc->>'recovery_preimage_score_rows_sha256' <>
          'dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63'
     OR v_archive.configuration_doc->>'recovery_preimage_stage2_rows_sha256' <>
          '2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a'
     OR v_archive.configuration_doc->'schedule' IS DISTINCT FROM v_old_schedule
     OR v_archive.configuration_doc->>'recovery_archive_runs_sha256' IS DISTINCT FROM
          (SELECT pg_catalog.encode(extensions.digest(
            pg_catalog.jsonb_agg(pg_catalog.to_jsonb(r) ORDER BY run_id)::TEXT,
            'sha256'),'hex') FROM public.lab_arena_runs r
            WHERE round_id=v_archive.round_id)
     OR v_archive.configuration_doc->>'recovery_archive_submissions_sha256' IS DISTINCT FROM
          (SELECT pg_catalog.encode(extensions.digest(
            pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
            'sha256'),'hex') FROM public.lab_arena_submissions s
            WHERE round_id=v_archive.round_id)
     OR (SELECT COUNT(*) FROM public.lab_arena_submissions
         WHERE round_id=v_archive.round_id) <> 14
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_archive.round_id AND stage=1 AND kind='execute'
           AND status='accepted' AND per_icp_score=0) <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_archive.round_id AND stage=2 AND kind='execute'
           AND status='pending') <> 130
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
         WHERE round_id=v_archive.round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events
         WHERE round_id=v_archive.round_id)
  THEN
    RAISE EXCEPTION 'Oct01 release recovered round or archive differs'
      USING ERRCODE='55000';
  END IF;
  IF pg_catalog.clock_timestamp() >=
      (v_new_schedule->>'stage_1_scoring_close')::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'Oct01 release scoring window already closed'
      USING ERRCODE='55000';
  END IF;
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds r
      WHERE r.round_id <> v_round.round_id
        AND r.configuration_doc->>'mode'='live'
        AND r.status IN ('committed','stage1','stage1_closed','stage1_scoring',
                         'stage1_judged','stage1_scored','stage2','stage2_closed',
                         'stage2_scoring','stage2_judged','scored'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      WHERE r.status IN ('leased','submitted'))
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id) <> 31
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id AND kind='execute' AND stage=1
           AND submission_id='baseline-2026-10-01' AND status='accepted'
           AND output_ref IS NOT NULL AND per_icp_score IS NULL) <> 10
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id AND kind='score' AND stage=1
           AND submission_id='baseline-2026-10-01' AND status='failed'
           AND stage_generation=5) <> 11
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(
             run_id,assignment_id,icp_position,attempt,status,terminal_cause,
             stage_generation,scored_run_id) ORDER BY run_id)::TEXT,
           'sha256'),'hex') FROM public.lab_arena_runs
           WHERE round_id=v_round.round_id AND kind='score'
             AND stage_generation=5) IS DISTINCT FROM
          'dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63'
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id AND kind='score' AND stage=1
           AND submission_id='baseline-2026-10-01' AND status='accepted'
           AND icp_position=1 AND stage_generation=5
           AND output_ref IS NOT NULL AND terminal_cause='accepted') <> 1
     OR (SELECT COUNT(*) FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id AND kind='score' AND stage=1
           AND submission_id='baseline-2026-10-01' AND status='pending'
           AND stage_generation=8
           AND output_ref IS NULL AND per_icp_score IS NULL) <> 9
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
         WHERE round_id=v_round.round_id AND stage=2)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions s
         CROSS JOIN (VALUES ('execute'::TEXT),('score'::TEXT)) k(kind)
         CROSS JOIN LATERAL (SELECT public.lab_arena__successful_call_cost_state(
           s.submission_id,k.kind,NULL) state) c
         WHERE s.round_id=v_round.round_id
           AND COALESCE((c.state->>'inflight_calls')::BIGINT,-1) <> 0)
  THEN
    RAISE EXCEPTION 'Oct01 release source, claims, or billing differs'
      USING ERRCODE='55000';
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=FALSE,pause_reason='',
      actor_ref='oct01-provider-recovery373',updated_at=pg_catalog.clock_timestamp()
  WHERE singleton AND operator_paused AND pause_reason='oct01_deepline_outage'
    AND guard_generation=v_control.guard_generation AND actor_ref=v_control.actor_ref;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 release claim control changed'
      USING ERRCODE='55000';
  END IF;
END;
$oct01_release$;
COMMIT;
