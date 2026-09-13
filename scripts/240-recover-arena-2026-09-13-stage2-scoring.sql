-- Recover only the 88 incomplete stage-2 judgments after the local-evidence
-- isolation and provider-cost reconciliation fixes. Historical attempts,
-- accepted work, costs, submissions, and credentials remain immutable.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

DO $lab_arena_240_recover_stage2_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-13';
  v_cost_submission CONSTANT TEXT := 'sub-5dffdbaa2b96e8dc78160aea8f80a7b9';
  v_run_prefix CONSTANT TEXT := 'arena-2026-09-13:score-recovery240:';
  -- Deliberately invalid until the fixed scorer artifact is immutable.
  v_new_digest CONSTANT TEXT := '__FIXED_SCORER_IMAGE_DIGEST__';
  v_new_reference CONSTANT TEXT := '__FIXED_SCORER_IMAGE_REFERENCE__';
  v_round public.lab_arena_rounds%ROWTYPE;
  v_participant_ids TEXT[];
  v_old_runs_hash TEXT;
  v_accepted_hash TEXT;
  v_execute_hash TEXT;
  v_ledger_hash TEXT;
  v_cache_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_unrelated_hash TEXT;
  v_configuration_stable JSONB;
  v_count INTEGER;
  v_costs JSONB;
BEGIN
  IF v_new_digest !~ '^sha256:[0-9a-f]{64}$'
     OR v_new_reference NOT LIKE '%@' || v_new_digest THEN
    RAISE EXCEPTION 'set the exact fixed scorer image before migration 240';
  END IF;
  IF pg_catalog.to_regprocedure('public.lab_arena_submission_costs(text)') IS NULL
     OR pg_catalog.to_regprocedure('extensions.digest(bytea,text)') IS NULL THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 240';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_round_id FOR UPDATE;
  IF NOT FOUND THEN RETURN; END IF;

  -- Exact replay is a no-op after the round has advanced normally.
  SELECT pg_catalog.count(*) INTO v_count FROM public.lab_arena_runs
  WHERE round_id=v_round_id AND run_id LIKE v_run_prefix || '%';
  IF v_count > 0 THEN
    IF v_count <> 88 OR v_round.status = 'cancelled'
       OR v_round.configuration_doc->>'scorer_image_digest' IS DISTINCT FROM v_new_digest
       OR v_round.status NOT IN ('stage2_scoring','stage2_judged','scored',
                                 'stage3','stage3_closed','stage3_scoring',
                                 'stage3_judged','confirmed','published') THEN
      RAISE EXCEPTION 'arena 2026-09-13 stage2 recovery replay differs';
    END IF;
    RETURN;
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 14 OR v_round.stage_generation <> 12
     OR v_round.evaluation_date <> '2026-09-13'
     OR v_round.icp_set_date <> DATE '2026-09-12'
     OR v_round.publication_doc IS NOT NULL OR v_round.published_at IS NOT NULL
     OR v_round.king_outcome IS NOT NULL OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 12
     OR pg_catalog.jsonb_array_length(v_round.finalists) <> 10
     OR pg_catalog.jsonb_array_length(v_round.stage2_scoring_plan_doc->'work_items') <> 110
     OR pg_catalog.jsonb_array_length(v_round.stage2_scoring_plan_doc->'zero_rows') <> 10
     OR v_round.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc->>'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_v1'
     OR v_round.configuration_doc->>'max_attempts_per_assignment' IS DISTINCT FROM '2'
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=v_round_id
                AND status IN ('pending','leased','submitted'))
     OR EXISTS (SELECT 1 FROM public.lab_arena_company_judgment_reservations r
                JOIN public.lab_arena_runs x ON x.run_id=r.run_id
                WHERE x.round_id=v_round_id)
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (l.call_identity) l.entry_kind
         FROM public.lab_arena_ledger l
         WHERE l.round_id=v_round_id AND l.call_identity IS NOT NULL
         ORDER BY l.call_identity,l.entry_id DESC
       ) h WHERE h.entry_kind IN ('reservation','dispatch')
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-13 stage2 recovery state differs';
  END IF;

  SELECT pg_catalog.array_agg(p->>'submission_id' ORDER BY p->>'submission_id')
  INTO v_participant_ids FROM pg_catalog.jsonb_array_elements(v_round.participants) p;
  IF v_participant_ids IS DISTINCT FROM ARRAY[
    'baseline-2026-09-13','sub-00ee4222188195476cdcf63fdb94171f',
    'sub-028c4e5c655e855f1c343fa39274cd2c','sub-1728be05a36000fcdb781a11d35594cd',
    'sub-5c1f20eb379bcd4e11c29912b5251b0d','sub-5d27bd75ca8c5999ae22f28a0a244660',
    'sub-5dffdbaa2b96e8dc78160aea8f80a7b9','sub-5ff557c97e83dcddd7b266226e16239c',
    'sub-6211e8d46819c34df3418ded36f788ef','sub-ca0c9c5da2d2b203f246258213c27263',
    'sub-caf0e1ef30c9712e6385afe24a75375e','sub-cb17cc2f2f351c1e92a59b85e35333c3'
  ]::TEXT[]
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id=v_round_id AND status='frozen') <> 12
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submission_credentials
         WHERE submission_id=ANY(v_participant_ids)) <> 24 THEN
    RAISE EXCEPTION 'arena 2026-09-13 participant state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
      AND stage=2 AND kind='execute' AND status='accepted' AND terminal_cause='accepted') <> 110
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND stage=2 AND kind='execute' AND status='failed' AND terminal_cause='credential_error') <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND stage=2 AND kind='score' AND status='accepted' AND terminal_cause='accepted') <> 22
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND stage=2 AND kind='score' AND status='failed' AND terminal_cause='credential_error') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND stage=2 AND kind='score' AND status='failed' AND terminal_cause='judge_error') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND stage=2 AND kind='score' AND status='failed' AND terminal_cause='stage_closed') <> 85
     OR (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs
         WHERE round_id=v_round_id AND stage=2 AND kind='score') <> 110 THEN
    RAISE EXCEPTION 'arena 2026-09-13 stage2 run counts differ';
  END IF;

  -- Migration 241 must have resolved every target cost head. This checks the
  -- resulting accounting contract rather than trusting a migration marker.
  SELECT public.lab_arena_submission_costs(v_cost_submission) INTO v_costs;
  IF EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(v_costs->'providers') p
               WHERE (p->>'inflight_calls')::BIGINT <> 0
                  OR (p->>'success_unresolved_calls')::BIGINT <> 0) THEN
    RAISE EXCEPTION 'arena 2026-09-13 cost reconciliation incomplete';
  END IF;

  SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
    INTO v_old_runs_hash FROM public.lab_arena_runs x WHERE round_id=v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
    INTO v_accepted_hash FROM public.lab_arena_runs x WHERE round_id=v_round_id AND status='accepted';
  SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
    INTO v_execute_hash FROM public.lab_arena_runs x WHERE round_id=v_round_id AND kind='execute';
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY entry_id),''))
    INTO v_ledger_hash FROM public.lab_arena_ledger x WHERE round_id=v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY cache_key),''))
    INTO v_cache_hash FROM public.lab_arena_judgment_cache x WHERE scope_doc->>'round_id'=v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY submission_id))
    INTO v_submission_hash FROM public.lab_arena_submissions x WHERE round_id=v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY submission_id,provider))
    INTO v_credential_hash FROM public.lab_arena_submission_credentials x WHERE submission_id=ANY(v_participant_ids);
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY round_id),''))
    INTO v_unrelated_hash FROM public.lab_arena_rounds x WHERE round_id<>v_round_id;
  v_configuration_stable := v_round.configuration_doc-'scorer_image_digest'-'scorer_image_reference';

  WITH latest AS (
    SELECT DISTINCT ON (assignment_id) x.*
    FROM public.lab_arena_runs x
    WHERE round_id=v_round_id AND stage=2 AND kind='score'
    ORDER BY assignment_id,(status='accepted') DESC,attempt DESC
  ), targets AS (
    SELECT run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
      icp_position,attempt,kind,scored_run_id,runner_hotkey,
      judgment_input_hash,judgment_group_miner_hotkeys,
      CASE WHEN attempt=1 THEN assignment_id ELSE assignment_id||':recovery240' END AS new_assignment,
      v_run_prefix || pg_catalog.substr(pg_catalog.encode(extensions.digest(
        pg_catalog.convert_to(run_id||':recovery240','UTF8'),'sha256'),'hex'),1,48) AS new_run_id,
      (judgment_scope_doc-'cache_key') || pg_catalog.jsonb_build_object(
        'scorer_image_digest',v_new_digest,'scorer_image_reference',v_new_reference
      ) AS scope_body
    FROM latest WHERE status='failed' AND terminal_cause IN
      ('stage_closed','credential_error','judge_error')
  ), canonical AS (
    SELECT targets.*,'{'||pg_catalog.string_agg(
      pg_catalog.to_jsonb(f.key)::TEXT||':'||f.value::TEXT,',' ORDER BY f.key COLLATE "C")||'}' canonical_json
    FROM targets CROSS JOIN LATERAL pg_catalog.jsonb_each(scope_body) f
    GROUP BY targets.run_id,targets.assignment_id,targets.round_id,targets.submission_id,
      targets.miner_hotkey,targets.stage,targets.icp_position,targets.attempt,targets.kind,
      targets.scored_run_id,targets.runner_hotkey,targets.judgment_input_hash,
      targets.judgment_group_miner_hotkeys,targets.new_assignment,targets.new_run_id,targets.scope_body
  ), prepared AS (
    SELECT canonical.*,'sha256:'||pg_catalog.encode(extensions.digest(
      pg_catalog.convert_to(canonical_json,'UTF8'),'sha256'),'hex') cache_key
    FROM canonical
  )
  INSERT INTO public.lab_arena_runs (
    run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
    attempt,status,lease_generation,stage_generation,kind,scored_run_id,
    previous_runner_hotkey,judgment_cache_key,judgment_input_hash,
    judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys
  ) SELECT new_run_id,new_assignment,round_id,submission_id,miner_hotkey,stage,
    icp_position,CASE WHEN attempt=1 THEN 2 ELSE 1 END,'pending',0,13,kind,
    scored_run_id,runner_hotkey,cache_key,judgment_input_hash,
    scope_body||pg_catalog.jsonb_build_object('cache_key',cache_key),FALSE,
    judgment_group_miner_hotkeys
  FROM prepared;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 88 THEN RAISE EXCEPTION 'arena 2026-09-13 recovery insert count differs'; END IF;

  WITH groups AS (
    SELECT judgment_cache_key,pg_catalog.min(run_id) leader,
      pg_catalog.array_agg(DISTINCT miner_hotkey ORDER BY miner_hotkey) miners
    FROM public.lab_arena_runs WHERE run_id LIKE v_run_prefix||'%'
    GROUP BY judgment_cache_key
  ) UPDATE public.lab_arena_runs x SET judgment_group_leader=(x.run_id=g.leader),
      judgment_group_miner_hotkeys=g.miners FROM groups g
    WHERE x.run_id LIKE v_run_prefix||'%' AND x.judgment_cache_key=g.judgment_cache_key;

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds SET status='stage2_scoring',status_generation=15,
    stage_generation=13,cancel_reason=NULL,
    configuration_doc=pg_catalog.jsonb_set(pg_catalog.jsonb_set(configuration_doc,
      '{scorer_image_digest}',pg_catalog.to_jsonb(v_new_digest),FALSE),
      '{scorer_image_reference}',pg_catalog.to_jsonb(v_new_reference),FALSE)
  WHERE round_id=v_round_id AND status='cancelled' AND status_generation=14 AND stage_generation=12;
  GET DIAGNOSTICS v_count=ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count<>1 THEN RAISE EXCEPTION 'arena 2026-09-13 recovery transition failed'; END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
      AND run_id LIKE v_run_prefix||'%' AND status='pending' AND stage_generation=13)<>88
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND run_id LIKE v_run_prefix||'%' AND attempt=2)<>87
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=v_round_id
         AND run_id LIKE v_run_prefix||'%' AND attempt=1 AND assignment_id LIKE '%:recovery240')<>1
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
         FROM public.lab_arena_runs x WHERE round_id=v_round_id AND run_id NOT LIKE v_run_prefix||'%')
        IS DISTINCT FROM v_old_runs_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
         FROM public.lab_arena_runs x WHERE round_id=v_round_id AND status='accepted') IS DISTINCT FROM v_accepted_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY run_id))
         FROM public.lab_arena_runs x WHERE round_id=v_round_id AND kind='execute') IS DISTINCT FROM v_execute_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY entry_id),''))
         FROM public.lab_arena_ledger x WHERE round_id=v_round_id) IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY cache_key),''))
         FROM public.lab_arena_judgment_cache x WHERE scope_doc->>'round_id'=v_round_id) IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY submission_id))
         FROM public.lab_arena_submissions x WHERE round_id=v_round_id) IS DISTINCT FROM v_submission_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY submission_id,provider))
         FROM public.lab_arena_submission_credentials x WHERE submission_id=ANY(v_participant_ids)) IS DISTINCT FROM v_credential_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(pg_catalog.to_jsonb(x)::TEXT,'|' ORDER BY round_id),''))
         FROM public.lab_arena_rounds x WHERE round_id<>v_round_id) IS DISTINCT FROM v_unrelated_hash
     OR (SELECT configuration_doc-'scorer_image_digest'-'scorer_image_reference'
         FROM public.lab_arena_rounds WHERE round_id=v_round_id) IS DISTINCT FROM v_configuration_stable THEN
    RAISE EXCEPTION 'arena 2026-09-13 stage2 recovery verification failed';
  END IF;
END;
$lab_arena_240_recover_stage2_scoring$;

COMMIT;
