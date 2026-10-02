-- Hold new claims and progression before October 1 miner scoring begins.
-- Existing miner execution leases may complete; accepted outputs and costs stay.
DO $oct01_corrected_scorer_hold$
DECLARE
  v_round public.lab_arena_rounds;
  v_control public.lab_arena_restart_claim_control;
  v_actor CONSTANT TEXT := 'oct01-corrected-scorer-hold378';
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
    RAISE EXCEPTION 'Oct01 corrected scorer hold conflicts with restart guard';
  END IF;
  IF v_control.operator_paused AND
     (v_control.pause_reason, v_control.actor_ref) IS DISTINCT FROM
     ('oct01_corrected_scorer_rejudge', v_actor) THEN
    RAISE EXCEPTION 'Oct01 corrected scorer hold belongs to another operator';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-10-01' FOR UPDATE;
  IF NOT FOUND OR (v_round.status,v_round.status_generation,v_round.stage_generation)
       NOT IN (('stage1_scoring',13,11),('stage1_judged',14,11),
               ('stage1_scored',15,12),('stage2',16,13))
     OR v_round.benchmark_ref <> 'arena/arena-2026-10-01/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-10-01'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-30'
     OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM
        'sha256:342645a42b52363cb907c7b48627ead707e09547a7197b4b0d803e7e6e57a7ba'
     OR pg_catalog.encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex') IS DISTINCT FROM
        '6b7b62b7f2f4c8a332398f12941305acec1de5cb311c6c7eeb0b574133c2243f'
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
     OR NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds
           WHERE round_id='arena-2026-10-01-r376archive' AND status='cancelled'
             AND configuration_doc->>'recovery_source_round_id'='arena-2026-10-01')
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='execute'
             AND submission_id='baseline-2026-10-01' AND status='accepted'
             AND output_ref IS NOT NULL
             AND ((v_round.status IN ('stage1_scored','stage2') AND per_icp_score IS NOT NULL
                   AND qualification_doc IS NOT NULL)
               OR (v_round.status NOT IN ('stage1_scored','stage2') AND per_icp_score IS NULL
                   AND qualification_doc IS NULL))) <> 10
     OR (SELECT pg_catalog.count(DISTINCT icp_position) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='score'
             AND assignment_id LIKE '%:score:rerun376') <> 10
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='score'
             AND (submission_id<>'baseline-2026-10-01' OR stage_generation<>11
                  OR assignment_id NOT LIKE '%:score:rerun376'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=2 AND kind='score')
     OR (v_round.status='stage2' AND
         ((SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=2) <> 130
          OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
            WHERE round_id='arena-2026-10-01' AND stage=2 AND kind='execute'
              AND stage_generation=13) <> 130))
     OR (v_round.status<>'stage2' AND EXISTS
         (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=2))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage NOT IN (1,2))
     OR (v_round.status IN ('stage1_scored','stage2') AND
         (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id='arena-2026-10-01' AND stage=1 AND kind='score'
             AND status='accepted') <> 10)
     OR v_round.published_at IS NOT NULL OR v_round.reward_activated_at IS NOT NULL
  THEN
    RAISE EXCEPTION 'Oct01 corrected scorer hold frozen round differs';
  END IF;
  IF EXISTS (SELECT 1 FROM public.lab_arena_rounds r
      WHERE r.round_id <> v_round.round_id
        AND r.configuration_doc->>'mode'='live'
        AND r.status NOT IN ('open','committed','published','cancelled'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs r
      JOIN public.lab_arena_rounds rr ON rr.round_id=r.round_id
      WHERE r.round_id <> v_round.round_id AND rr.configuration_doc->>'mode'='live'
        AND r.status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Oct01 corrected scorer hold would interrupt another active round';
  END IF;
  IF v_control.operator_paused THEN
    RETURN; -- Same owner replay does not change the hold or active leases.
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=TRUE,pause_reason='oct01_corrected_scorer_rejudge',
      actor_ref=v_actor,updated_at=pg_catalog.clock_timestamp()
  WHERE singleton AND NOT operator_paused;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Oct01 corrected scorer hold claim control changed';
  END IF;
END;
$oct01_corrected_scorer_hold$;
