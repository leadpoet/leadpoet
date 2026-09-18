-- Activate the reviewed per-ICP cost policy on the unfinished Sep18 round and
-- on the not-yet-started Sep19 round if it already exists. No work is reset.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_per_icp_cost$
DECLARE v_schema JSONB;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_per_icp_cost_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migration 289 before migration 290';
  END IF;
  SELECT public.lab_arena_per_icp_cost_schema_v1() INTO v_schema;
  IF v_schema IS DISTINCT FROM
       '{"schema_version":"leadpoet.lab_arena.per_icp_cost_schema.v1",'
       '"version":289,"policy":"successful_calls_per_icp_v1"}'::JSONB THEN
    RAISE EXCEPTION 'per-ICP cost schema differs';
  END IF;
END;
$requires_per_icp_cost$;

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $verify_round_trigger$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena rounds write-once trigger is not enabled'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_round_trigger$;

ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;

DO $activate_sep18$
DECLARE
  v_round public.lab_arena_rounds;
  v_before JSONB;
  v_after JSONB;
  v_assignments BIGINT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18' FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep18 round missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       = 'successful_calls_per_icp_v1' THEN
    IF (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
         IS DISTINCT FROM 4000000
       OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
         IS DISTINCT FROM 800000 THEN
      RAISE EXCEPTION 'sep18 per-ICP replay state differs' USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;
  IF v_round.status IS DISTINCT FROM 'stage1'
     OR v_round.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-18'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'integrity_policy'
          IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'contact_policy'
          IS DISTINCT FROM 'contacts_v1'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_v1'
     OR (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT
          IS DISTINCT FROM 80000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR (v_round.configuration_doc ->> 'companies_per_icp')::INTEGER
          IS DISTINCT FROM 5
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 5
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL THEN
    RAISE EXCEPTION 'sep18 activation state differs' USING ERRCODE = '55000';
  END IF;
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id = v_round.round_id AND status = 'frozen') <> 5
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = v_round.round_id AND is_king) <> 1
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE round_id = v_round.round_id
         AND submission_id = 'baseline-2026-09-18' AND is_king
         AND source_size_bytes = 604847
         AND submission_doc ->> 'source_sha256' =
           '7e1bb0747014a57bc50f48f9f822d1a7c936682d63f06564e978d23f54eb7fc1'
         AND submission_doc ->> 'source_commit' =
           'e5341f85829ad196b4a1cb58b38a34155697c8d4'
     )
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
       WHERE NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions AS submission
         WHERE submission.round_id = v_round.round_id
           AND submission.submission_id = participant ->> 'submission_id'
           AND submission.miner_hotkey = participant ->> 'miner_hotkey'
           AND submission.is_king = COALESCE(
             (participant ->> 'is_king')::BOOLEAN, FALSE
           )
       )
     ) THEN
    RAISE EXCEPTION 'sep18 frozen participant snapshot differs'
      USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(DISTINCT (submission_id, icp_position))
  INTO v_assignments FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id AND kind = 'execute'
    AND icp_position BETWEEN 0 AND 19;
  IF v_assignments <> 100
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND (kind = 'score' OR per_icp_score IS NOT NULL
              OR qualification_doc IS NOT NULL)
     )
     OR EXISTS (
       SELECT 1
       FROM (
         SELECT runs.submission_id, runs.icp_position,
           COALESCE(SUM(head.amount_microusd),0)::BIGINT AS actual_microusd
         FROM public.lab_arena_runs AS runs
         LEFT JOIN LATERAL (
           SELECT DISTINCT ON (ledger.run_id, ledger.call_identity)
             ledger.entry_kind, ledger.amount_microusd
           FROM public.lab_arena_ledger AS ledger
           WHERE ledger.run_id = runs.run_id
             AND ledger.call_identity IS NOT NULL
           ORDER BY ledger.run_id, ledger.call_identity, ledger.entry_id DESC
         ) AS head ON head.entry_kind = 'settlement'
         WHERE runs.round_id = v_round.round_id AND runs.kind = 'execute'
         GROUP BY runs.submission_id, runs.icp_position
       ) AS spend WHERE actual_microusd >= 4000000
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE round_id = v_round.round_id AND entry_kind = 'refusal'
         AND entry_doc ->> 'reason' = 'money_cap'
     ) THEN
    RAISE EXCEPTION 'sep18 execution snapshot is not safe for activation'
      USING ERRCODE = '55000';
  END IF;

  v_before := pg_catalog.to_jsonb(v_round);
  UPDATE public.lab_arena_rounds
  SET configuration_doc = pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          configuration_doc, '{sourcing_cost_eligibility_policy}',
          '"successful_calls_per_icp_v1"'::JSONB, FALSE
        ), '{execution_icp_cap_microusd}', '4000000'::JSONB, TRUE
      ),
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round.round_id;
  SELECT pg_catalog.to_jsonb(rounds.*) INTO v_after
  FROM public.lab_arena_rounds AS rounds WHERE round_id = v_round.round_id;
  IF (v_after - 'configuration_doc' - 'updated_at') IS DISTINCT FROM
       (v_before - 'configuration_doc' - 'updated_at')
     OR ((v_after -> 'configuration_doc')
           - 'sourcing_cost_eligibility_policy' - 'execution_icp_cap_microusd')
        IS DISTINCT FROM
       ((v_before -> 'configuration_doc')
           - 'sourcing_cost_eligibility_policy' - 'execution_icp_cap_microusd') THEN
    RAISE EXCEPTION 'sep18 activation changed an unapproved field'
      USING ERRCODE = '55000';
  END IF;
END;
$activate_sep18$;

-- Sep19 can already have been created by the old binary. If present, adopt
-- only before participants or execution exist. If absent, the new binary will
-- create it with the new frozen defaults.
DO $activate_sep19_if_present$
DECLARE v_round public.lab_arena_rounds; v_before JSONB; v_after JSONB;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19' FOR UPDATE;
  IF NOT FOUND THEN RETURN; END IF;
  IF v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       = 'successful_calls_per_icp_v1' THEN
    IF (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
         IS DISTINCT FROM 4000000 THEN
      RAISE EXCEPTION 'sep19 per-ICP replay state differs' USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;
  IF v_round.status IS DISTINCT FROM 'open'
     OR v_round.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-19'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'integrity_policy'
          IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_v1'
     OR (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT
          IS DISTINCT FROM 80000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round.participants IS NOT NULL
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round.round_id) THEN
    RAISE EXCEPTION 'sep19 open activation state differs' USING ERRCODE = '55000';
  END IF;
  v_before := pg_catalog.to_jsonb(v_round);
  UPDATE public.lab_arena_rounds
  SET configuration_doc = pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          configuration_doc, '{sourcing_cost_eligibility_policy}',
          '"successful_calls_per_icp_v1"'::JSONB, FALSE
        ), '{execution_icp_cap_microusd}', '4000000'::JSONB, TRUE
      ), updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round.round_id;
  SELECT pg_catalog.to_jsonb(rounds.*) INTO v_after
  FROM public.lab_arena_rounds AS rounds WHERE round_id = v_round.round_id;
  IF (v_after - 'configuration_doc' - 'updated_at') IS DISTINCT FROM
       (v_before - 'configuration_doc' - 'updated_at')
     OR ((v_after -> 'configuration_doc')
           - 'sourcing_cost_eligibility_policy' - 'execution_icp_cap_microusd')
        IS DISTINCT FROM
       ((v_before -> 'configuration_doc')
           - 'sourcing_cost_eligibility_policy' - 'execution_icp_cap_microusd') THEN
    RAISE EXCEPTION 'sep19 activation changed an unapproved field'
      USING ERRCODE = '55000';
  END IF;
END;
$activate_sep19_if_present$;

ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;

DO $verify_activation$
BEGIN
  IF NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-18'
         AND configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_per_icp_v1'
         AND (configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT = 4000000
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = 'arena-2026-09-19'
         AND (configuration_doc ->> 'sourcing_cost_eligibility_policy'
              IS DISTINCT FROM 'successful_calls_per_icp_v1'
              OR (configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
                 IS DISTINCT FROM 4000000)
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'per-ICP activation verification failed'
      USING ERRCODE = '55000';
  END IF;
END;
$verify_activation$;

COMMIT;
