-- Admit sourcing calls from confirmed settlement spend only. Reservations,
-- dispatches and uncertainties remain immutable lifecycle evidence, but new
-- execute-call reservations carry no monetary hold. Score admission is
-- deliberately unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $confirmed_cost_admission$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$  IF v_reason IS NULL AND v_per_icp_policy AND (v_dynamic OR p_amount_microusd > 0) THEN
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        v_run.round_id || ':' || v_run.submission_id || ':' || v_run.icp_position::TEXT,
        0
      )
    );
    v_icp_cost := public.lab_arena__successful_icp_cost_state(
      v_run.round_id, v_run.submission_id, v_run.icp_position
    );
    -- lab_arena_temporary_hold_admission
    IF (v_icp_cost ->> 'settled_microusd')::BIGINT >=
         (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    ELSIF (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0
       OR (v_icp_cost ->> 'settled_microusd')::BIGINT
          + (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT >=
          (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_expires := pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
      UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
      WHERE run_id = p_run_id;
      RETURN pg_catalog.jsonb_build_object(
        'status', 'budget_busy', 'idempotent', FALSE,
        'reason', CASE WHEN (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0
          THEN 'provider_calls_inflight' ELSE 'provider_cost_uncertain' END,
        'call_identity', p_call_identity, 'lease_expires_at', v_expires
      );
    ELSIF v_dynamic THEN
      p_amount_microusd :=
        (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
        - (v_icp_cost ->> 'settled_microusd')::BIGINT
        - (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT;
    END IF;
  END IF;
$old$;
  v_new TEXT := $new$  IF v_reason IS NULL AND v_per_icp_policy
     AND (v_dynamic OR p_amount_microusd > 0) THEN
    -- lab_arena_confirmed_cost_admission: serialize the confirmed-spend read
    -- with settlement, but do not wait for or price pending provider work.
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        v_run.round_id || ':' || v_run.submission_id || ':' || v_run.icp_position::TEXT,
        0
      )
    );
    v_icp_cost := public.lab_arena__successful_icp_cost_state(
      v_run.round_id, v_run.submission_id, v_run.icp_position
    );
    IF (v_icp_cost ->> 'settled_microusd')::BIGINT >=
         (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    END IF;
  END IF;
$new$;
  v_insert TEXT := $old$  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
    p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
    p_funding_source, p_amount_microusd, p_call_doc
  );$old$;
  v_insert_new TEXT := $new$  -- New execute calls retain their lifecycle row and caller-owned evidence,
  -- but admission has no estimated or pending monetary hold.
  IF v_per_icp_policy THEN
    p_amount_microusd := 0;
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
    p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
    p_funding_source, p_amount_microusd, p_call_doc
  );$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_confirmed_cost_admission') = 0 THEN
    IF pg_catalog.strpos(v_definition, 'lab_arena_per_icp_paid_admission') = 0
       OR pg_catalog.strpos(v_definition, 'lab_arena_temporary_hold_admission') = 0
       OR pg_catalog.strpos(v_definition, 'lab_arena_openrouter_web_search_reservation') = 0
       OR pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_reservation_admission') = 0
       OR (pg_catalog.length(v_definition)
           - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
          / pg_catalog.length(v_old) <> 1
       OR (pg_catalog.length(v_definition)
           - pg_catalog.length(pg_catalog.replace(v_definition, v_insert, '')))
          / pg_catalog.length(v_insert) <> 1 THEN
      RAISE EXCEPTION 'confirmed-cost admission shape unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    EXECUTE pg_catalog.replace(v_definition, v_insert, v_insert_new);
  END IF;
END;
$confirmed_cost_admission$;

DO $confirmed_cost_eligibility$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$  v_spend := (v_state ->> 'successful_microusd')::BIGINT
    + (v_state ->> 'success_unresolved_microusd')::BIGINT;$old$;
  v_new TEXT := $new$  -- lab_arena_confirmed_cost_eligibility: unresolved calls remain a
  -- fail-closed eligibility reason, but only settled successful cost is spend.
  v_spend := (v_state ->> 'successful_microusd')::BIGINT;$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_confirmed_cost_eligibility') = 0 THEN
    IF (pg_catalog.length(v_definition)
        - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
       / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'confirmed-cost eligibility shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$confirmed_cost_eligibility$;

DO $confirmed_cost_markers$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_confirmed_cost_admission') = 0 THEN
    RAISE EXCEPTION 'confirmed-cost admission marker missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_confirmed_cost_eligibility') = 0
     OR pg_catalog.strpos(
       v_definition,
       'v_spend := (v_state ->> ''successful_microusd'')::BIGINT;'
     ) = 0 THEN
    RAISE EXCEPTION 'confirmed-cost eligibility marker missing';
  END IF;
END;
$confirmed_cost_markers$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
