-- Opt-in per-ICP sourcing admission and cost eligibility. Historical rounds
-- retain their frozen submission-wide policy and publication checks.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__successful_icp_cost_state(
  p_round_id TEXT, p_submission_id TEXT, p_icp_position INTEGER
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $successful_icp_cost_state$
  WITH heads AS (
    SELECT DISTINCT ON (ledger.run_id, ledger.call_identity)
      ledger.entry_kind, ledger.amount_microusd,
      CASE
        WHEN ledger.entry_kind = 'settlement'
          AND ledger.entry_doc ->> 'late_reconciliation' = 'true'
          AND ledger.entry_doc ->> 'reconciled_uncertainty_reason' = 'round_cancelled'
        THEN FALSE
        WHEN ledger.entry_kind = 'settlement'
          AND pg_catalog.jsonb_typeof(ledger.terminal_response -> 'call_succeeded') = 'boolean'
        THEN (ledger.terminal_response ->> 'call_succeeded')::BOOLEAN
        WHEN ledger.entry_kind = 'settlement'
          AND ledger.entry_doc ->> 'openrouter_delayed_reconciliation' = 'true'
        THEN (
          SELECT CASE WHEN pg_catalog.jsonb_typeof(
            prior.entry_doc #> '{call,call_succeeded}') = 'boolean'
            THEN (prior.entry_doc #>> '{call,call_succeeded}')::BOOLEAN END
          FROM public.lab_arena_ledger AS prior
          WHERE prior.entry_id =
            (ledger.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
            AND prior.entry_kind = 'uncertain'
        )
        WHEN ledger.entry_kind = 'uncertain'
          AND pg_catalog.jsonb_typeof(
            ledger.entry_doc #> '{call,call_succeeded}') = 'boolean'
        THEN (ledger.entry_doc #>> '{call,call_succeeded}')::BOOLEAN
        WHEN ledger.entry_kind = 'uncertain'
          AND ledger.entry_doc ->> 'reason' = 'round_cancelled'
        THEN FALSE
      END AS call_succeeded
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.round_id = p_round_id
      AND ledger.submission_id = p_submission_id
      AND runs.kind = 'execute'
      AND runs.icp_position = p_icp_position
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.run_id, ledger.call_identity, ledger.entry_id DESC
  )
  SELECT pg_catalog.jsonb_build_object(
    'settled_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind = 'settlement'), 0)::BIGINT,
    'reserved_or_uncertain_microusd', COALESCE(SUM(amount_microusd)
      FILTER (WHERE entry_kind IN ('reservation','dispatch','uncertain')), 0)::BIGINT,
    'inflight_calls', COUNT(*) FILTER (
      WHERE entry_kind IN ('reservation','dispatch'))::BIGINT,
    'uncertain_calls', COUNT(*) FILTER (WHERE entry_kind = 'uncertain')::BIGINT,
    'refused_calls', COUNT(*) FILTER (WHERE entry_kind = 'refusal')::BIGINT,
    'call_count', COUNT(*)::BIGINT,
    'successful_microusd', COALESCE(SUM(amount_microusd) FILTER (
      WHERE entry_kind = 'settlement' AND call_succeeded IS TRUE), 0)::BIGINT,
    'successful_calls', COUNT(*) FILTER (
      WHERE entry_kind = 'settlement' AND call_succeeded IS TRUE)::BIGINT,
    'success_unresolved_microusd', COALESCE(SUM(amount_microusd) FILTER (
      WHERE (entry_kind = 'settlement' AND call_succeeded IS NULL)
         OR (entry_kind = 'uncertain' AND call_succeeded IS DISTINCT FROM FALSE)
         OR entry_kind IN ('reservation','dispatch')), 0)::BIGINT,
    'success_unresolved_calls', COUNT(*) FILTER (
      WHERE (entry_kind = 'settlement' AND call_succeeded IS NULL)
         OR (entry_kind = 'uncertain' AND call_succeeded IS DISTINCT FROM FALSE)
         OR entry_kind IN ('reservation','dispatch'))::BIGINT,
    'paid_inflight_calls', COUNT(*) FILTER (
      WHERE entry_kind IN ('reservation','dispatch')
        AND amount_microusd > 0)::BIGINT,
    'paid_uncertain_calls', COUNT(*) FILTER (
      WHERE entry_kind = 'uncertain'
        AND amount_microusd > 0)::BIGINT
  ) FROM heads;
$successful_icp_cost_state$;
ALTER FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_icp_cost_state(TEXT,TEXT,INTEGER)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_icp_cost_eligibility(
  p_round_id TEXT, p_submission_id TEXT, p_icp_position INTEGER,
  p_qualified_company_count INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $icp_cost_eligibility$
DECLARE
  v_round public.lab_arena_rounds;
  v_state JSONB;
  v_icp_cap BIGINT;
  v_company_cap BIGINT;
  v_spend BIGINT;
  v_reason TEXT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF NOT FOUND
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
        IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
        IS DISTINCT FROM 800000
     OR p_icp_position NOT BETWEEN 0 AND 19
     OR p_qualified_company_count NOT BETWEEN 0 AND 5
     OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions
                    WHERE submission_id = p_submission_id AND round_id = p_round_id) THEN
    RAISE EXCEPTION 'lab_arena_per_icp_cost_policy_invalid' USING ERRCODE = '22023';
  END IF;
  v_icp_cap := (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT;
  v_company_cap := (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;
  IF v_icp_cap IS NULL OR v_icp_cap < 1 OR v_company_cap IS NULL OR v_company_cap < 1 THEN
    RAISE EXCEPTION 'lab_arena_per_icp_cost_policy_invalid' USING ERRCODE = '22023';
  END IF;
  v_state := public.lab_arena__successful_icp_cost_state(
    p_round_id, p_submission_id, p_icp_position
  );
  v_spend := (v_state ->> 'successful_microusd')::BIGINT
    + (v_state ->> 'success_unresolved_microusd')::BIGINT;
  IF (v_state ->> 'inflight_calls')::BIGINT > 0 THEN
    v_reason := 'provider_calls_inflight';
  ELSIF (v_state ->> 'success_unresolved_calls')::BIGINT > 0 THEN
    v_reason := 'provider_cost_uncertain';
  ELSIF v_spend > v_company_cap * p_qualified_company_count THEN
    v_reason := 'cost_per_company_exceeded';
  ELSE
    v_reason := 'eligible';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'icp_position', p_icp_position,
    'eligible', v_reason = 'eligible',
    'eligibility_reason', v_reason,
    'competition_sourcing_microusd', v_spend,
    'execution_icp_cap_microusd', v_icp_cap,
    'cost_per_company_cap_microusd', v_company_cap,
    'eligibility_cap_microusd', v_company_cap * p_qualified_company_count,
    'execution', v_state
  );
END;
$icp_cost_eligibility$;
ALTER FUNCTION public.lab_arena_icp_cost_eligibility(TEXT,TEXT,INTEGER,INTEGER)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_icp_cost_eligibility(TEXT,TEXT,INTEGER,INTEGER)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_icp_cost_eligibility(TEXT,TEXT,INTEGER,INTEGER)
  TO lab_arena_service;

-- The existing reporting RPC exposes successful-call counters only for its
-- original marker. Extend that marker test without changing historical rows.
DO $patch_submission_costs$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
      = 'successful_calls_v1'$old$;
  v_new TEXT := $new$rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
      IN ('successful_calls_v1','successful_calls_per_icp_v1')$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_submission_costs(text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'successful_calls_per_icp_v1') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena per-ICP submission costs shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$patch_submission_costs$;

CREATE OR REPLACE FUNCTION public.lab_arena__cost_kind_summary_v1(
  p_submission_id TEXT, p_kind TEXT
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $cost_kind_summary$
  WITH rows AS (
    SELECT value AS row
    FROM pg_catalog.jsonb_array_elements(
      public.lab_arena_submission_costs(p_submission_id) -> 'providers'
    )
    WHERE value ->> 'kind' = p_kind
  )
  SELECT pg_catalog.jsonb_build_object(
    'settled_microusd', COALESCE(SUM((row ->> 'settled_microusd')::BIGINT),0),
    'reserved_or_uncertain_microusd', COALESCE(SUM((row ->> 'reserved_or_uncertain_microusd')::BIGINT),0),
    'conservative_microusd', COALESCE(SUM((row ->> 'settled_microusd')::BIGINT),0)
      + COALESCE(SUM((row ->> 'reserved_or_uncertain_microusd')::BIGINT),0),
    'inflight_calls', COALESCE(SUM((row ->> 'inflight_calls')::BIGINT),0),
    'uncertain_calls', COALESCE(SUM((row ->> 'uncertain_calls')::BIGINT),0),
    'refused_calls', COALESCE(SUM((row ->> 'refused_calls')::BIGINT),0),
    'call_count', COALESCE(SUM((row ->> 'call_count')::BIGINT),0),
    'successful_microusd', COALESCE(SUM((row ->> 'successful_microusd')::BIGINT),0),
    'successful_calls', COALESCE(SUM((row ->> 'successful_calls')::BIGINT),0),
    'success_unresolved_microusd', COALESCE(SUM((row ->> 'success_unresolved_microusd')::BIGINT),0),
    'success_unresolved_calls', COALESCE(SUM((row ->> 'success_unresolved_calls')::BIGINT),0),
    'providers', COALESCE(pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object(
        'provider', row ->> 'provider',
        'settled_microusd', (row ->> 'settled_microusd')::BIGINT,
        'reserved_or_uncertain_microusd', (row ->> 'reserved_or_uncertain_microusd')::BIGINT,
        'conservative_microusd', (row ->> 'settled_microusd')::BIGINT
          + (row ->> 'reserved_or_uncertain_microusd')::BIGINT,
        'inflight_calls', (row ->> 'inflight_calls')::BIGINT,
        'uncertain_calls', (row ->> 'uncertain_calls')::BIGINT,
        'refused_calls', (row ->> 'refused_calls')::BIGINT,
        'call_count', (row ->> 'call_count')::BIGINT,
        'successful_microusd', (row ->> 'successful_microusd')::BIGINT,
        'successful_calls', (row ->> 'successful_calls')::BIGINT,
        'success_unresolved_microusd', (row ->> 'success_unresolved_microusd')::BIGINT,
        'success_unresolved_calls', (row ->> 'success_unresolved_calls')::BIGINT
      ) ORDER BY row ->> 'provider'
    ), '[]'::JSONB)
  ) FROM rows
  WHERE p_kind IN ('execute','score');
$cost_kind_summary$;
ALTER FUNCTION public.lab_arena__cost_kind_summary_v1(TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__cost_kind_summary_v1(TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

-- Add marker-gated admission to the latest reserve implementation. The
-- per-ICP advisory lock spans the budget read and reservation insert, so one
-- paid call must settle before another paid call for that ICP can be admitted.
DO $patch_reserve$
DECLARE
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_per_icp_paid_admission') = 0 THEN
    v_old := $old$  v_reason TEXT := NULL;$old$;
    v_new := $new$  v_reason TEXT := NULL;
  v_icp_cost JSONB;
  v_per_icp_policy BOOLEAN;$new$;
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena per-ICP reserve declaration shape unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_old := $old$  IF v_reason IS NULL THEN
    -- Lock order stays round, run, submission. This lock serializes the
    -- aggregate check and reservation insert across all providers and runs.$old$;
    v_new := $new$  -- lab_arena_per_icp_paid_admission
  v_per_icp_policy := COALESCE(
    v_run.kind = 'execute' AND
      v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        = 'successful_calls_per_icp_v1',
    FALSE
  );
  IF v_per_icp_policy AND (
       v_round.configuration_doc ->> 'integrity_policy'
         IS DISTINCT FROM 'arena_integrity_v1'
       OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
         IS DISTINCT FROM 4000000
       OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
         IS DISTINCT FROM 800000
     ) THEN
    RAISE EXCEPTION 'lab_arena_per_icp_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_reason IS NULL AND v_per_icp_policy AND (v_dynamic OR p_amount_microusd > 0) THEN
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        v_run.round_id || ':' || v_run.submission_id || ':' || v_run.icp_position::TEXT,
        0
      )
    );
    v_icp_cost := public.lab_arena__successful_icp_cost_state(
      v_run.round_id, v_run.submission_id, v_run.icp_position
    );
    IF (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0 THEN
      v_expires := pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
      UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
      WHERE run_id = p_run_id;
      RETURN pg_catalog.jsonb_build_object(
        'status', 'budget_busy', 'idempotent', FALSE,
        'call_identity', p_call_identity, 'lease_expires_at', v_expires
      );
    ELSIF (v_icp_cost ->> 'settled_microusd')::BIGINT
          + (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT >=
          (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    ELSIF v_dynamic THEN
      p_amount_microusd :=
        (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
        - (v_icp_cost ->> 'settled_microusd')::BIGINT
        - (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT;
    END IF;
  END IF;
  IF v_reason IS NULL THEN
    -- Lock order stays round, run, submission. This lock serializes the
    -- aggregate check and reservation insert across all providers and runs.$new$;
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena per-ICP reserve admission shape unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_old := $old$    IF v_reason IS NULL THEN
      v_money_cap := CASE v_run.kind$old$;
    v_new := $new$    IF v_reason IS NULL AND NOT v_per_icp_policy THEN
      v_money_cap := CASE v_run.kind$new$;
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena per-ICP reserve money shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$patch_reserve$;

-- Validate the marker-specific publication row from immutable run scores and
-- ledger heads. Over-cost ICPs contribute zero; raw per_icp_score is untouched.
CREATE OR REPLACE FUNCTION public.lab_arena__per_icp_publication_valid(
  p_round_id TEXT, p_ranking JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $per_icp_publication_valid$
DECLARE
  v_submission_id TEXT := p_ranking ->> 'submission_id';
  v_cost JSONB := p_ranking -> 'cost_summary';
  v_row JSONB;
  v_summary JSONB;
  v_expected JSONB;
  v_position INTEGER;
  v_expected_score NUMERIC;
  v_hard_reason TEXT;
  v_total_spend BIGINT := 0;
  v_total_qualified BIGINT := 0;
  v_total_returned BIGINT := 0;
  v_eligible_icps BIGINT := 0;
  v_expected_execution JSONB;
  v_expected_judge JSONB;
  v_score_count BIGINT;
BEGIN
  IF v_cost ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR pg_catalog.jsonb_typeof(v_cost -> 'per_icp') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_cost -> 'per_icp') <> 20 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
  END IF;
  FOR v_position IN 0..19 LOOP
    SELECT value INTO v_row FROM pg_catalog.jsonb_array_elements(v_cost -> 'per_icp')
      WHERE value ->> 'icp_position' = v_position::TEXT;
    IF v_row IS NULL THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_summary := public.lab_arena__integrity_submission_summary(
      p_round_id, v_submission_id, ARRAY[v_position]
    );
    IF NOT COALESCE((v_summary ->> 'valid')::BOOLEAN, FALSE) THEN
      RAISE EXCEPTION 'lab_arena_publication_scoring_incomplete'
        USING ERRCODE = '22023';
    END IF;
    v_expected := public.lab_arena_icp_cost_eligibility(
      p_round_id, v_submission_id, v_position,
      (v_summary ->> 'qualified_company_count')::INTEGER
    );
    IF (v_row ->> 'qualified_company_count')::BIGINT IS DISTINCT FROM
         (v_summary ->> 'qualified_company_count')::BIGINT
       OR (v_row ->> 'eligible')::BOOLEAN IS DISTINCT FROM
         (v_expected ->> 'eligible')::BOOLEAN
       OR v_row ->> 'eligibility_reason' IS DISTINCT FROM
         v_expected ->> 'eligibility_reason'
       OR (v_row ->> 'competition_sourcing_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'competition_sourcing_microusd')::BIGINT
       OR (v_row ->> 'eligibility_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'eligibility_cap_microusd')::BIGINT
       OR (v_row ->> 'execution_icp_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'execution_icp_cap_microusd')::BIGINT
       OR (v_row ->> 'cost_per_company_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'cost_per_company_cap_microusd')::BIGINT
       OR v_row -> 'execution' IS DISTINCT FROM v_expected -> 'execution' THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_total_spend := v_total_spend
      + (v_expected ->> 'competition_sourcing_microusd')::BIGINT;
    v_total_qualified := v_total_qualified
      + (v_summary ->> 'qualified_company_count')::BIGINT;
    IF pg_catalog.jsonb_typeof(v_row -> 'returned_company_count') IS DISTINCT FROM 'number'
       OR (v_row ->> 'returned_company_count')::BIGINT NOT BETWEEN 0 AND 50 THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_total_returned := v_total_returned
      + (v_row ->> 'returned_company_count')::BIGINT;
    IF (v_expected ->> 'eligible')::BOOLEAN THEN
      v_eligible_icps := v_eligible_icps + 1;
    END IF;
    IF v_hard_reason IS NULL AND v_expected ->> 'eligibility_reason' IN (
      'provider_calls_inflight','provider_cost_uncertain'
    ) THEN v_hard_reason := v_expected ->> 'eligibility_reason'; END IF;
  END LOOP;
  v_expected_execution := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'execute'
  );
  v_expected_judge := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'score'
  );
  SELECT pg_catalog.count(*), pg_catalog.avg(CASE WHEN
      (public.lab_arena_icp_cost_eligibility(
        p_round_id, v_submission_id, runs.icp_position,
        (public.lab_arena__integrity_submission_summary(
          p_round_id, v_submission_id, ARRAY[runs.icp_position]
        ) ->> 'qualified_company_count')::INTEGER
      ) ->> 'eligible')::BOOLEAN
    THEN runs.per_icp_score ELSE 0 END)
  INTO v_score_count, v_expected_score
  FROM (
    SELECT DISTINCT ON (icp_position) icp_position, per_icp_score
    FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND submission_id = v_submission_id
      AND kind = 'execute' AND per_icp_score IS NOT NULL
    ORDER BY icp_position, attempt DESC
  ) AS runs;
  IF v_score_count <> 20
     OR (p_ranking ->> 'final_score')::NUMERIC IS DISTINCT FROM v_expected_score
     OR (p_ranking ->> 'eligible')::BOOLEAN IS DISTINCT FROM (v_hard_reason IS NULL)
     OR p_ranking ->> 'eligibility_reason' IS DISTINCT FROM COALESCE(v_hard_reason,'eligible')
     OR (v_cost ->> 'competition_sourcing_microusd')::BIGINT IS DISTINCT FROM v_total_spend
     OR (v_cost ->> 'qualified_company_count')::BIGINT IS DISTINCT FROM v_total_qualified
     OR (v_cost ->> 'returned_company_count')::BIGINT IS DISTINCT FROM v_total_returned
     OR (v_cost ->> 'eligible_icp_count')::BIGINT IS DISTINCT FROM v_eligible_icps
     OR (v_cost ->> 'execution_icp_cap_microusd')::BIGINT IS DISTINCT FROM
       (SELECT (configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
        FROM public.lab_arena_rounds WHERE round_id = p_round_id)
     OR (v_cost ->> 'cost_per_company_cap_microusd')::BIGINT IS DISTINCT FROM
       (SELECT (configuration_doc ->> 'cost_per_company_microusd')::BIGINT
        FROM public.lab_arena_rounds WHERE round_id = p_round_id)
     OR v_cost -> 'execution' IS DISTINCT FROM v_expected_execution
     OR v_cost -> 'judge' IS DISTINCT FROM v_expected_judge THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
  END IF;
  RETURN TRUE;
END;
$per_icp_publication_valid$;
ALTER FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $patch_publication_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$    v_main := public.lab_arena__integrity_submission_summary($old$;
  v_new TEXT := $new$    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_per_icp_v1' THEN
      PERFORM public.lab_arena__per_icp_publication_valid(NEW.round_id, v_ranking);
      IF v_submission_id = v_baseline_id THEN
        v_baseline_score := (v_ranking ->> 'final_score')::NUMERIC;
      END IF;
      CONTINUE;
    END IF;
    v_main := public.lab_arena__integrity_submission_summary($new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_integrity_publication_guard_v1()'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena__per_icp_publication_valid') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena per-ICP publication guard shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$patch_publication_guard$;

CREATE OR REPLACE FUNCTION public.lab_arena_per_icp_cost_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version','leadpoet.lab_arena.per_icp_cost_schema.v1',
    'version',289,'policy','successful_calls_per_icp_v1'
  );
$schema$;
ALTER FUNCTION public.lab_arena_per_icp_cost_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_per_icp_cost_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_per_icp_cost_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
