-- Recover the distinct unfinished stage-1 scoring assignments for the
-- cancelled September 14 Arena round. Historical rows remain immutable.
--
-- Entry 348922 has one authenticated 2,000-microusd provider ledger row whose
-- request identity was created 13 ms after the Arena dispatch. Arena did not
-- retain that identity, so the settlement records the attribution limit. The
-- scorer image and judgment scope remain frozen.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

-- A recovered score uses a fresh assignment id, but it still judges the same
-- planned execution. Close scoring by that logical work-item identity. An
-- accepted attempt wins over every older failed assignment for the item.
DO $lab_arena_244_close_scoring_logical_item$
DECLARE
  v_definition TEXT;
  v_old_incomplete TEXT := $old$
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id) runs.assignment_id, runs.status, runs.terminal_cause
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
  WHERE latest.status <> 'accepted';
$old$;
  v_new_incomplete TEXT := $new$
  -- lab_arena_scoring_logical_item_completion: recovery assignments for one
  -- planned execution do not make the old failed assignment count again.
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT DISTINCT ON (runs.submission_id, runs.icp_position, runs.scored_run_id)
      runs.submission_id, runs.icp_position, runs.scored_run_id,
      runs.status, runs.terminal_cause
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.submission_id, runs.icp_position, runs.scored_run_id,
      (runs.status = 'accepted') DESC, runs.stage_generation DESC,
      runs.attempt DESC, runs.created_at DESC, runs.run_id DESC
  ) AS latest
  WHERE latest.status <> 'accepted';
$new$;
  v_old_baseline TEXT := $old$
  SELECT COUNT(*) INTO v_baseline_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id)
      runs.assignment_id, runs.submission_id, runs.status
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
$old$;
  v_new_baseline TEXT := $new$
  SELECT COUNT(*) INTO v_baseline_incomplete FROM (
    SELECT DISTINCT ON (runs.submission_id, runs.icp_position, runs.scored_run_id)
      runs.submission_id, runs.icp_position, runs.scored_run_id, runs.status
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.submission_id, runs.icp_position, runs.scored_run_id,
      (runs.status = 'accepted') DESC, runs.stage_generation DESC,
      runs.attempt DESC, runs.created_at DESC, runs.run_id DESC
  ) AS latest
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_scoring_logical_item_completion'
     ) > 0 THEN
    RETURN;
  END IF;
  IF (pg_catalog.length(v_definition)
      - pg_catalog.length(pg_catalog.replace(v_definition, v_old_incomplete, '')))
       / pg_catalog.length(v_old_incomplete) <> 1
     OR (pg_catalog.length(v_definition)
      - pg_catalog.length(pg_catalog.replace(v_definition, v_old_baseline, '')))
       / pg_catalog.length(v_old_baseline) <> 1 THEN
    RAISE EXCEPTION 'lab_arena_close_scoring logical-item shape unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_old_incomplete, v_new_incomplete
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_baseline, v_new_baseline
  );
  EXECUTE v_definition;
END;
$lab_arena_244_close_scoring_logical_item$;

DO $lab_arena_244_recover_20260914_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-14';
  v_recovery_suffix CONSTANT TEXT := ':recovery244';
  v_old_digest CONSTANT TEXT :=
    'sha256:412beed7799ecfcceb9c07c44f36644f8214c33513a117d603dede98c0b918e9';
  v_old_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'
    || v_old_digest;
  v_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-14',
    'sub-09f530ce94b6f63221ce4d69962e3988',
    'sub-0bee4ffefcb6a792ca70d16d722fa1ac',
    'sub-287f56f3e3718776a6aa10e4c130e154',
    'sub-2968d4fba9582f9d8881bfd998290cbe',
    'sub-4c5b60cfe00763f2d022031ecbd40039',
    'sub-4e87858d429552dd34271afa3be03a9b',
    'sub-804245a7d75aa38aa4c74a59e07b51d2',
    'sub-8ec81a864da4e10204d803a3fb7c3b58',
    'sub-946ff9115aec053aeb36a02476175b82',
    'sub-a7a50fba0fe795bc6f87a433fb6682f3',
    'sub-dae52a6a6cca8db5914a259116bc1b56',
    'sub-dd76cafd44732cb230a476ab031b24ad'
  ]::TEXT[];

  -- Keep the frozen scorer identity. Recovery does not invalidate its cache
  -- scope or introduce a new scorer image.
  v_new_digest CONSTANT TEXT :=
    'sha256:412beed7799ecfcceb9c07c44f36644f8214c33513a117d603dede98c0b918e9';
  v_new_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'
    || v_new_digest;

  -- The old ledger operation remains scrapingdog.scrape. The provider request
  -- identity was lost, so this is a time-correlated provider charge rather than
  -- a cryptographically exact request-identity join.
  v_cost_operation CONSTANT TEXT := 'scrapingdog.scrape';
  v_cost_billing_operation CONSTANT TEXT := 'firecrawl_scrape';
  v_cost_amount_microusd CONSTANT BIGINT := 2000;
  v_cost_units CONSTANT TEXT := '0.02';
  v_cost_basis CONSTANT TEXT :=
    'deepline_authenticated_billing_ledger_time_correlated_charge';
  v_cost_accounting_kind CONSTANT TEXT := 'provider_ledger_time_correlated';
  v_provider_attribution_exact CONSTANT BOOLEAN := FALSE;
  v_cost_evidence_hash CONSTANT TEXT :=
    'sha256:899a3f2242c6f24d7816c1ff925b9ad0af78c4a91ce70a45b2da4b3a1916ded1';
  v_provider_request_id_hash CONSTANT TEXT :=
    'sha256:7b86e0f91c3f6c5a7d17ff8b59d1e5851c264d6c94fada7888011d9063c073a8';
  v_cost_reservation_entry_id CONSTANT BIGINT := 348871;
  v_cost_dispatch_entry_id CONSTANT BIGINT := 348872;
  v_cost_entry_id CONSTANT BIGINT := 348922;
  v_cost_reserved_microusd CONSTANT BIGINT := 49945650;
  v_cost_request_hash CONSTANT TEXT :=
    'sha256:7b87f4d4738cd42dbe01fc94bb38b5e7e386883b04a2cf26343e6947d143d952';
  v_cost_identity CONSTANT TEXT :=
    'sha256:975bc7fbac081e7f081bdd6bcfc19698cbdb44c5e4b95b05e5c921fda0f7e6a9';
  v_cost_submission CONSTANT TEXT :=
    'sub-09f530ce94b6f63221ce4d69962e3988';
  v_cost_run CONSTANT TEXT :=
    'arena-2026-09-14:sub-09f530ce94b6f63221ce4d69962e3988:1:0:score:1';

  v_round public.lab_arena_rounds%ROWTYPE;
  v_cost_head public.lab_arena_ledger%ROWTYPE;
  v_cost_reservation public.lab_arena_ledger%ROWTYPE;
  v_cost_dispatch public.lab_arena_ledger%ROWTYPE;
  v_ledger_hash TEXT;
  v_accepted_hash TEXT;
  v_execute_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_cache_hash TEXT;
  v_unrelated_hash TEXT;
  v_round_stable JSONB;
  v_configuration_stable JSONB;
  v_cost_settlement_id BIGINT;
  v_count INTEGER;
BEGIN
  IF v_new_digest !~ '^sha256:[0-9a-f]{64}$'
     OR v_new_reference NOT LIKE '%@' || v_new_digest
     OR v_cost_amount_microusd < 0
     OR v_cost_units IS NULL
     OR v_cost_billing_operation <> 'firecrawl_scrape'
     OR v_cost_accounting_kind <> 'provider_ledger_time_correlated'
     OR v_provider_attribution_exact
     OR v_cost_evidence_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_cost_evidence_hash = 'sha256:' || pg_catalog.repeat('0', 64)
     OR v_provider_request_id_hash !~ '^sha256:[0-9a-f]{64}$'
     OR v_cost_request_hash !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION
      'arena 244 requires exact scorer release and authenticated cost proof';
  END IF;
  IF v_cost_operation !~ '^[a-z0-9_.-]{1,64}$'
     OR v_cost_units !~ '^(0|[1-9][0-9]{0,9})([.][0-9]{1,18})?$'
     OR v_cost_basis !~ '^deepline_authenticated_[a-z0-9_.-]{1,96}$'
     OR pg_catalog.ceil(v_cost_units::NUMERIC * 100000)::BIGINT
        IS DISTINCT FROM v_cost_amount_microusd THEN
    RAISE EXCEPTION 'arena 244 authenticated cost binding is invalid';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  -- A committed recovery is idempotent. It must already contain the complete
  -- fresh namespace and the exact cost settlement before becoming a no-op.
  SELECT pg_catalog.count(*) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id
    AND assignment_id LIKE '%' || v_recovery_suffix;
  IF v_count > 0 THEN
    IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
        WHERE round_id = v_round_id
          AND assignment_id LIKE '%' || v_recovery_suffix
          AND attempt = 1) <> 93
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id
             AND assignment_id LIKE '%' || v_recovery_suffix) <> 93
       OR NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_ledger AS settled
         WHERE settled.entry_id = v_cost_entry_id
           AND settled.entry_kind = 'uncertain'
       )
       OR NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_ledger AS settled
         WHERE settled.call_identity = v_cost_identity
           AND settled.entry_kind = 'settlement'
           AND settled.amount_microusd = v_cost_amount_microusd
           AND settled.entry_doc ->> 'arena_244_reconciliation' = 'true'
           AND settled.entry_doc ->> 'accounting_amount_kind'
               = v_cost_accounting_kind
           AND settled.entry_doc ->> 'provider_attribution_exact' = 'false'
           AND settled.entry_doc ->> 'evidence_hash' = v_cost_evidence_hash
           AND settled.entry_doc ->> 'provider_request_id_hash'
               = v_provider_request_id_hash
       ) THEN
      RAISE EXCEPTION 'arena 2026-09-14 recovery replay state differs';
    END IF;
    RETURN;
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 5
     OR v_round.stage_generation <> 4
     OR v_round.evaluation_date <> '2026-09-14'
     OR v_round.icp_set_date <> '2026-09-13'
     OR v_round.published_at IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 13
     OR (
       SELECT pg_catalog.array_agg(item ->> 'submission_id' ORDER BY item ->> 'submission_id')
       FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant(item)
     ) IS DISTINCT FROM v_participant_ids
     OR pg_catalog.jsonb_typeof(v_round.stage1_scoring_plan_doc)
        IS DISTINCT FROM 'object'
     OR v_round.stage1_scoring_plan_doc ->> 'round_id' IS DISTINCT FROM v_round_id
     OR (v_round.stage1_scoring_plan_doc ->> 'stage')::INTEGER IS DISTINCT FROM 1
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) <> 130
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'zero_rows'
        ) <> 0
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment'
        IS DISTINCT FROM '2'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_old_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_old_reference
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND status = 'frozen') <> 13
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)) <> 28
     OR v_round.configuration_doc #>> '{schedule,submission_open}'
        IS DISTINCT FROM '2026-09-13T00:00:00Z'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
        IS DISTINCT FROM '2026-09-14T00:00:00Z'
     OR v_round.configuration_doc #>> '{schedule,stage_1_scoring_close}'
        IS DISTINCT FROM '2026-09-14T11:00:01Z'
     OR v_round.configuration_doc #>> '{schedule,stage_2_start}'
        IS DISTINCT FROM '2026-09-14T11:00:02Z' THEN
    RAISE EXCEPTION 'arena 2026-09-14 frozen round state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
        AND status = 'accepted' AND terminal_cause = 'accepted') <> 130
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 37
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 132
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 130
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed') <> 95
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND attempt = 2) <> 2
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgment_reservations AS reservation
       JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
       WHERE run.round_id = v_round_id
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 run state differs';
  END IF;

  -- Every plan item must have its accepted execute row. The missing set is
  -- derived from the plan and accepted scores, never from failed-row counts.
  IF EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         v_round.stage1_scoring_plan_doc -> 'work_items'
       ) AS plan(item)
       WHERE NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS execution
         WHERE execution.round_id = v_round_id
           AND execution.kind = 'execute'
           AND execution.status = 'accepted'
           AND execution.run_id = plan.item ->> 'scored_run_id'
           AND execution.submission_id = plan.item ->> 'submission_id'
           AND execution.icp_position = (plan.item ->> 'icp_position')::SMALLINT
           AND execution.output_ref = plan.item ->> 'output_ref'
       )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 execution plan binding differs';
  END IF;
  SELECT pg_catalog.count(*) INTO v_count
  FROM pg_catalog.jsonb_array_elements(
    v_round.stage1_scoring_plan_doc -> 'work_items'
  ) AS plan(item)
  WHERE NOT EXISTS (
    SELECT 1
    FROM public.lab_arena_runs AS accepted
    WHERE accepted.round_id = v_round_id
      AND accepted.kind = 'score'
      AND accepted.status = 'accepted'
      AND accepted.submission_id = plan.item ->> 'submission_id'
      AND accepted.icp_position = (plan.item ->> 'icp_position')::SMALLINT
      AND accepted.scored_run_id = plan.item ->> 'scored_run_id'
  );
  IF v_count <> 93 THEN
    RAISE EXCEPTION 'arena 2026-09-14 unfinished assignment count differs';
  END IF;

  -- Preserve exact row identities before the one authorized settlement and the
  -- fresh recovery rows are appended.
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) INTO v_accepted_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND status = 'accepted';
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) INTO v_execute_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND kind = 'execute';
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT), '|' ORDER BY submission_id
         )) INTO v_submission_hash
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT), '|'
           ORDER BY submission_id, provider
         )) INTO v_credential_hash
  FROM public.lab_arena_submission_credentials AS credential
  WHERE submission_id IN (
    SELECT item ->> 'submission_id'
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant(item)
  );
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) INTO v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT), '|' ORDER BY cache_key
         ), '')) INTO v_cache_hash
  FROM public.lab_arena_judgment_cache AS cache_row
  WHERE scope_doc ->> 'round_id' = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT), '|' ORDER BY round_id
         ), '')) INTO v_unrelated_hash
  FROM public.lab_arena_rounds AS other_round
  WHERE round_id <> v_round_id;
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'configuration_doc' - 'updated_at';
  v_configuration_stable := v_round.configuration_doc
    - 'scorer_image_digest' - 'scorer_image_reference';

  -- Bind the old uncertainty to its exact immutable reservation and dispatch.
  SELECT * INTO v_cost_head
  FROM public.lab_arena_ledger
  WHERE entry_id = v_cost_entry_id
  FOR UPDATE;
  IF NOT FOUND
     OR v_cost_head.entry_kind <> 'uncertain'
     OR v_cost_head.round_id IS DISTINCT FROM v_round_id
     OR v_cost_head.submission_id IS DISTINCT FROM v_cost_submission
     OR v_cost_head.run_id IS DISTINCT FROM v_cost_run
     OR v_cost_head.stage IS DISTINCT FROM 1
     OR v_cost_head.call_identity IS DISTINCT FROM v_cost_identity
     OR v_cost_head.provider IS DISTINCT FROM 'deepline'
     OR v_cost_head.operation_id IS DISTINCT FROM v_cost_operation
     OR v_cost_head.funding_source IS DISTINCT FROM 'miner_key'
     OR v_cost_head.entry_doc ->> 'reason' IS DISTINCT FROM 'worker_reported'
     OR v_cost_head.entry_doc #>> '{call,reason}' IS DISTINCT FROM 'transport_failure'
     OR (public.lab_arena__ledger_head(v_cost_identity)).entry_id
        IS DISTINCT FROM v_cost_entry_id THEN
    RAISE EXCEPTION 'arena 2026-09-14 cost head binding differs';
  END IF;
  SELECT * INTO v_cost_reservation
  FROM public.lab_arena_ledger
  WHERE entry_id = v_cost_reservation_entry_id
  FOR UPDATE;
  IF NOT FOUND
     OR v_cost_reservation.entry_kind IS DISTINCT FROM 'reservation'
     OR v_cost_reservation.call_identity IS DISTINCT FROM v_cost_identity
     OR v_cost_reservation.round_id IS DISTINCT FROM v_round_id
     OR v_cost_reservation.run_id IS DISTINCT FROM v_cost_run
     OR v_cost_reservation.submission_id IS DISTINCT FROM v_cost_submission
     OR v_cost_reservation.provider IS DISTINCT FROM 'deepline'
     OR v_cost_reservation.operation_id IS DISTINCT FROM v_cost_operation
     OR v_cost_reservation.funding_source IS DISTINCT FROM 'miner_key'
     OR v_cost_reservation.amount_microusd IS DISTINCT FROM v_cost_reserved_microusd
     OR v_cost_reservation.amount_microusd IS DISTINCT FROM v_cost_head.amount_microusd
     OR v_cost_reservation.entry_doc ->> 'request_hash'
        IS DISTINCT FROM v_cost_request_hash THEN
    RAISE EXCEPTION 'arena 2026-09-14 cost reservation binding differs';
  END IF;
  SELECT * INTO v_cost_dispatch
  FROM public.lab_arena_ledger
  WHERE entry_id = v_cost_dispatch_entry_id
  FOR UPDATE;
  IF NOT FOUND
     OR v_cost_dispatch.entry_kind IS DISTINCT FROM 'dispatch'
     OR v_cost_dispatch.call_identity IS DISTINCT FROM v_cost_identity
     OR v_cost_dispatch.round_id IS DISTINCT FROM v_round_id
     OR v_cost_dispatch.run_id IS DISTINCT FROM v_cost_run
     OR v_cost_dispatch.submission_id IS DISTINCT FROM v_cost_submission
     OR v_cost_dispatch.provider IS DISTINCT FROM 'deepline'
     OR v_cost_dispatch.operation_id IS DISTINCT FROM v_cost_operation
     OR v_cost_dispatch.funding_source IS DISTINCT FROM 'miner_key'
     OR v_cost_dispatch.amount_microusd IS DISTINCT FROM v_cost_reserved_microusd THEN
    RAISE EXCEPTION 'arena 2026-09-14 cost dispatch binding differs';
  END IF;

  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source, amount_microusd,
    entry_doc, terminal_response
  ) VALUES (
    'settlement', v_cost_head.miner_hotkey, v_round_id,
    v_cost_submission, v_cost_run, 1, v_cost_identity, 'deepline',
    v_cost_operation, v_cost_head.funding_source, v_cost_amount_microusd,
    pg_catalog.jsonb_build_object(
      'arena_244_reconciliation', 'true',
      'reconciled_uncertainty_entry_id', v_cost_entry_id,
      'reserved_microusd', v_cost_head.amount_microusd,
      'released_microusd', v_cost_head.amount_microusd - v_cost_amount_microusd,
      'variance_microusd', v_cost_amount_microusd - v_cost_head.amount_microusd,
      'cost_basis', v_cost_basis,
      'accounting_amount_kind', v_cost_accounting_kind,
      'provider_attribution_exact', v_provider_attribution_exact,
      'evidence_hash', v_cost_evidence_hash,
      'provider_request_id_hash', v_provider_request_id_hash
    ),
    pg_catalog.jsonb_build_object(
      'status', 502,
      'headers', pg_catalog.jsonb_build_object(
        'content-type', 'application/json',
        'content-length', '41'
      ),
      'body_b64',
        'eyJlcnJvciI6eyJjb2RlIjoicHJvdmlkZXJfdW5hdmFpbGFibGUifX0=',
      'call_succeeded', FALSE,
      'provider_cost', pg_catalog.jsonb_build_object(
        'basis', v_cost_basis,
        'units', v_cost_units,
        'unit_name', 'credits',
        'operation', v_cost_billing_operation,
        'accounting_amount_kind', v_cost_accounting_kind,
        'provider_attribution_exact', v_provider_attribution_exact,
        'evidence_hash', v_cost_evidence_hash,
        'provider_request_id_hash', v_provider_request_id_hash,
        'original_operation_id', v_cost_operation
      ),
      'accounting_amount_kind', v_cost_accounting_kind,
      'provider_attribution_exact', v_provider_attribution_exact,
      'evidence_hash', v_cost_evidence_hash,
      'provider_request_id_hash', v_provider_request_id_hash
    )
  ) RETURNING entry_id INTO v_cost_settlement_id;

  -- Every unfinished plan item gets a new assignment namespace. This resets
  -- its two-attempt confirmation lifecycle without changing old rows.
  WITH latest AS (
    SELECT DISTINCT ON (assignment_id) run.*
    FROM public.lab_arena_runs AS run
    WHERE run.round_id = v_round_id
      AND run.stage = 1
      AND run.kind = 'score'
    ORDER BY assignment_id, (status = 'accepted') DESC, attempt DESC
  ), targets AS (
    SELECT latest.*,
      latest.assignment_id || v_recovery_suffix AS new_assignment
    FROM latest
    WHERE latest.status <> 'accepted'
  )
  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation, kind,
    scored_run_id, previous_runner_hotkey, judgment_cache_key,
    judgment_input_hash, judgment_scope_doc, judgment_group_leader,
    judgment_group_miner_hotkeys, company_judgment_refs
  )
  SELECT new_assignment || ':1', new_assignment, round_id, submission_id,
    miner_hotkey, stage, icp_position, 1, 'pending', 0, 5, kind,
    scored_run_id, NULL, judgment_cache_key, judgment_input_hash,
    judgment_scope_doc, FALSE, judgment_group_miner_hotkeys,
    company_judgment_refs
  FROM targets;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 93 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery insert count differs';
  END IF;

  WITH groups AS (
    SELECT judgment_cache_key, pg_catalog.min(run_id) AS leader_run_id,
      pg_catalog.array_agg(DISTINCT miner_hotkey ORDER BY miner_hotkey)
        AS miner_hotkeys
    FROM public.lab_arena_runs
    WHERE round_id = v_round_id
      AND assignment_id LIKE '%' || v_recovery_suffix
    GROUP BY judgment_cache_key
  )
  UPDATE public.lab_arena_runs AS run
  SET judgment_group_leader = run.run_id = groups.leader_run_id,
      judgment_group_miner_hotkeys = groups.miner_hotkeys
  FROM groups
  WHERE run.round_id = v_round_id
    AND run.assignment_id LIKE '%' || v_recovery_suffix
    AND run.judgment_cache_key = groups.judgment_cache_key;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1_scoring',
      status_generation = 6,
      stage_generation = 5,
      cancel_reason = NULL,
      configuration_doc = configuration_doc
  WHERE round_id = v_round_id
    AND status = 'cancelled'
    AND status_generation = 5
    AND stage_generation = 4;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery transition failed';
  END IF;

  -- Verify that only the intended append-only namespace and settlement were
  -- added. All accepted, execute, submission, credential, cache, and old
  -- ledger rows must retain their exact serialized hashes.
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id
        AND assignment_id LIKE '%' || v_recovery_suffix
        AND status = 'pending' AND stage_generation = 5) <> 93
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id
           AND assignment_id LIKE '%' || v_recovery_suffix
           AND attempt = 1
           AND run_id = assignment_id || ':1') <> 93
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND status = 'accepted')
          IS DISTINCT FROM v_accepted_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND kind = 'execute')
          IS DISTINCT FROM v_execute_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT), '|' ORDER BY submission_id
         )) FROM public.lab_arena_submissions AS submission
         WHERE round_id = v_round_id) IS DISTINCT FROM v_submission_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT), '|'
           ORDER BY submission_id, provider
         )) FROM public.lab_arena_submission_credentials AS credential
         WHERE submission_id IN (
           SELECT item ->> 'submission_id'
           FROM pg_catalog.jsonb_array_elements(v_round.participants)
             AS participant(item)
         )) IS DISTINCT FROM v_credential_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id AND entry_id <> v_cost_settlement_id)
          IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT), '|' ORDER BY cache_key
         ), '')) FROM public.lab_arena_judgment_cache AS cache_row
         WHERE scope_doc ->> 'round_id' = v_round_id)
          IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT), '|' ORDER BY round_id
         ), '')) FROM public.lab_arena_rounds AS other_round
         WHERE round_id <> v_round_id) IS DISTINCT FROM v_unrelated_hash
     OR (SELECT pg_catalog.to_jsonb(current_round)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'configuration_doc' - 'updated_at'
         FROM public.lab_arena_rounds AS current_round
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR (SELECT configuration_doc
           - 'scorer_image_digest' - 'scorer_image_reference'
         FROM public.lab_arena_rounds
         WHERE round_id = v_round_id) IS DISTINCT FROM v_configuration_stable THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery verification failed';
  END IF;
END;
$lab_arena_244_recover_20260914_scoring$;

COMMIT;
