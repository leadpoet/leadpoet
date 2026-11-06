-- Exact operator hold requested before the improved PR793 verifier is deployed.
-- The current canonical restart already blocks claims and scoring. Keep that
-- owner, its renewable deadline and all 21 captured leases unchanged; make the
-- pause survive its normal release. No runs, scores, costs or cutoffs change.
-- Release this operator hold only after the improved verifier is verified live.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0));

DO $nov03_verifier_hold603$
DECLARE
  v_control public.lab_arena_restart_claim_control;
  v_round public.lab_arena_rounds;
  v_reason CONSTANT TEXT := 'nov03_improved_verifier_required603';
BEGIN
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR UPDATE;
  IF NOT FOUND OR current_setting('session_replication_role') <> 'origin'
     OR v_control.guard_generation IS DISTINCT FROM 752
     OR v_control.guard_commitment IS DISTINCT FROM
        'sha256:db061410be3979646bc8d8f5d50d97dec6050791a6550a16e3719fce708f14ec'
     OR v_control.owner_commitment IS DISTINCT FROM
        'sha256:7a3b581c660367b270e0ea0f331a9e3d8126fe648572dbde04ee59bd7b6455c5'
     OR v_control.candidate_commit IS DISTINCT FROM
        '161e7443deaa66598772d0fb4a21240db9f277f0'
     OR v_control.restart_scope IS DISTINCT FROM 'all'
     OR v_control.restart_phase IS DISTINCT FROM 'draining'
     OR v_control.guard_expires_at IS NULL
     OR v_control.guard_expires_at <= pg_catalog.clock_timestamp()
     OR jsonb_array_length(v_control.captured_leases) <> 21
     OR encode(extensions.digest(v_control.captured_leases::TEXT,'sha256'),'hex')
        IS DISTINCT FROM '242331b1dc0752323a3d475058ee556000bf506461a73f1b3a6e4ea8709abfd4'
     OR (v_control.operator_paused AND v_control.pause_reason IS DISTINCT FROM v_reason)
     OR (NOT v_control.operator_paused AND
         (v_control.pause_reason,v_control.actor_ref) IS DISTINCT FROM
         ('canonical_restart_guard',
          'canonical-active-release:161e7443deaa66598772d0fb4a21240db9f277f0')) THEN
    RAISE EXCEPTION 'Nov03 verifier hold603 restart owner or preimage differs';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id='arena-2026-11-03' FOR UPDATE;
  IF NOT FOUND OR v_round.status IS DISTINCT FROM 'stage2'
     OR v_round.status_generation IS DISTINCT FROM 5
     OR v_round.stage_generation IS DISTINCT FROM 4
     OR v_round.evaluation_date IS DISTINCT FROM '2026-11-03'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-11-02'
     OR v_round.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-11-03/benchmark.json'
     OR v_round.configuration_doc->>'mode' IS DISTINCT FROM 'live'
     OR encode(extensions.digest(v_round.configuration_doc::TEXT,'sha256'),'hex')
        IS DISTINCT FROM 'e91499fe3037f17d0363f94ffdcfe383f49ceace8a365dc40c4a72277e21337c'
     OR encode(extensions.digest(v_round.participants::TEXT,'sha256'),'hex')
        IS DISTINCT FROM 'a7b48bac1538146d8ec5ceacec4ac6b054a4537621e903ef516dd430ff5d4452'
     OR v_round.published_at IS NOT NULL OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL OR v_round.effective_reward_epoch IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
        WHERE kind='score' AND status IN ('leased','submitted'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_rounds
        WHERE round_id<>v_round.round_id AND configuration_doc->>'mode'='live'
          AND status NOT IN ('open','committed','published','cancelled'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
        WHERE round_id<>v_round.round_id AND status IN ('leased','submitted')) THEN
    RAISE EXCEPTION 'Nov03 verifier hold603 round or active judge preimage differs';
  END IF;
  IF v_control.operator_paused THEN
    RETURN; -- Same live owner replay changes nothing, including renewals.
  END IF;
  UPDATE public.lab_arena_restart_claim_control
  SET operator_paused=TRUE,pause_reason=v_reason
  WHERE singleton AND NOT operator_paused;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Nov03 verifier hold603 claim control changed';
  END IF;
END;
$nov03_verifier_hold603$;
COMMIT;
