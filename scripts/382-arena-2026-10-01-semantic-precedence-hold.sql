-- Hold October 1 new claims and progression for the proved semantic precedence
-- review. Active leases may finish. Frozen inputs, runs, outputs and costs stay.
DO $oct01_semantic_precedence_hold$
DECLARE
  v_round public.lab_arena_rounds;
  v_archive public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_actor CONSTANT TEXT := 'oct01-semantic-precedence-hold382';
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF NOT FOUND OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.restart_phase <> ''
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold conflicts with restart guard';
  END IF;
  IF v_control.operator_paused AND
     (v_control.pause_reason, v_control.actor_ref) IS DISTINCT FROM
     ('oct01_semantic_precedence_review', v_actor) THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold belongs to another operator';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01' FOR UPDATE;
  SELECT * INTO v_archive FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01-r381archive' FOR UPDATE;
  IF NOT FOUND OR v_archive.status IS DISTINCT FROM 'cancelled'
     OR v_archive.configuration_doc->>'recovery_source_round_id' IS DISTINCT FROM 'arena-2026-10-01' THEN
    RAISE EXCEPTION 'Oct01 semantic precedence archive differs';
  END IF;
  IF v_round.round_id IS NULL
     OR (v_round.status,v_round.status_generation,v_round.stage_generation)
       NOT IN (('stage2',23,19),('stage2_closed',24,20),
               ('stage2_scoring',25,21),('stage2_judged',26,22),
               ('scored',27,22))
     OR v_round.benchmark_ref IS DISTINCT FROM
       'arena/arena-2026-10-01/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-10-01'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-30'
     OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
       'sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65'
     OR v_round.configuration_doc->'schedule' IS DISTINCT FROM
       '{"benchmark_deadline":"2026-10-01T00:30:00Z","final_scoring_close":"2026-10-03T08:00:02Z","publication_deadline":"2026-10-03T08:00:03Z","stage_1_close":"2026-10-01T11:00:01Z","stage_1_scoring_close":"2026-10-02T14:00:01Z","stage_1_start":"2026-10-01T00:30:01Z","stage_2_close":"2026-10-03T02:00:02Z","stage_2_start":"2026-10-02T14:00:02Z","submission_cutoff":"2026-10-01T00:00:00Z","submission_open":"2026-09-30T00:00:00Z"}'::JSONB
     OR v_round.configuration_doc->>'scorer_image_reference' IS DISTINCT FROM
       '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:ac4fcea361e079c5ed2dcee87d6cc77eb6bbcfaa2b73a4dd9ea05a7594160f65'
     OR v_round.configuration_doc - ARRAY[
       'round_id','rewards_enabled','schedule','scorer_image_digest',
       'scorer_image_reference'] IS DISTINCT FROM
       v_archive.configuration_doc - ARRAY[
       'round_id','rewards_enabled','schedule','scorer_image_digest',
       'scorer_image_reference','recovery_source_round_id',
       'recovery_original_champion_submission_id',
       'recovery_original_champion_hotkey',
       'recovery_original_champion_fallback_providers',
       'recovery_score_runs_sha256','recovery_score_ledger_sha256',
       'recovery_score_ledger_max_entry_id','recovery_score_events_sha256',
       'recovery_baseline_receipts','recovery_baseline_receipts_sha256']
     OR pg_catalog.encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex') IS DISTINCT FROM
       '872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19'
     OR (SELECT pg_catalog.encode(extensions.digest(icps::TEXT,'sha256'),'hex')
         FROM public.qualification_private_icp_sets WHERE set_id=20260930) IS DISTINCT FROM
       '8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.to_jsonb(s) ORDER BY submission_id)::TEXT,
           'sha256'),'hex') FROM public.lab_arena_submissions s
           WHERE round_id='arena-2026-10-01') IS DISTINCT FROM
       '1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3'
     OR (SELECT pg_catalog.encode(extensions.digest(
           pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
             'run_id',run_id,'submission_id',submission_id,'stage',stage,
             'icp_position',icp_position,'status',status,'terminal_cause',terminal_cause,
             'output_ref',output_ref,'stage_generation',stage_generation)
             ORDER BY icp_position)::TEXT,'sha256'),'hex')
           FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01'
             AND stage=1 AND kind='execute') IS DISTINCT FROM
       '24d7ae6108b0afa39a3108409c8c1cc37d40c62277c4d195e42244d2e777a2aa'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute'
             AND submission_id='baseline-2026-10-01' AND status='accepted'
             AND output_ref IS NOT NULL
             AND per_icp_score IS NOT NULL AND qualification_doc IS NOT NULL) <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='score') <> 10
     OR (SELECT pg_catalog.count(DISTINCT icp_position) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='score'
             AND assignment_id LIKE '%:score:rerun381' AND status='accepted') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs s
           LEFT JOIN public.lab_arena_runs e ON e.run_id=s.scored_run_id
           WHERE s.round_id='arena-2026-10-01' AND s.stage=1 AND s.kind='score'
             AND (s.submission_id<>'baseline-2026-10-01'
               OR s.assignment_id NOT LIKE '%:score:rerun381'
               OR e.round_id IS DISTINCT FROM s.round_id
               OR e.kind IS DISTINCT FROM 'execute'
               OR e.status IS DISTINCT FROM 'accepted'))
     OR (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=2 AND kind='execute') <> 130
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=2 AND kind='score')
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage NOT IN (1,2))
     OR v_round.published_at IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
  THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold frozen round differs';
  END IF;
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds r
      WHERE r.round_id <> v_round.round_id
        AND r.configuration_doc->>'mode'='live'
        AND r.status NOT IN ('open','committed','published','cancelled'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      JOIN public.lab_arena_rounds rr ON rr.round_id=r.round_id
      WHERE r.round_id <> v_round.round_id AND rr.configuration_doc->>'mode'='live'
        AND r.status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold would interrupt another active round';
  END IF;
  IF v_control.operator_paused THEN
    RETURN; -- Same-owner replay leaves active leases and timestamp intact.
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=TRUE,pause_reason='oct01_semantic_precedence_review',
      actor_ref=v_actor,updated_at=pg_catalog.clock_timestamp()
  WHERE singleton AND NOT operator_paused;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 semantic precedence hold claim control changed';
  END IF;
END;
$oct01_semantic_precedence_hold$;
