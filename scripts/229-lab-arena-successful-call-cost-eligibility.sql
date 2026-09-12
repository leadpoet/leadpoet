-- New rounds count only trusted successful sourcing calls toward efficiency.
-- Historical rounds keep their exact existing publication and cost rules.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $lab_arena_229_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_submission_costs(text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_publication_baseline_guard_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__integrity_eligibility(text,text,integer[])'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_integrity_publication_guard_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply Lab Arena migrations through 228 first';
  END IF;
END;
$lab_arena_229_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Lease expiry, stage close, and cancellation are database-authored proof that
-- an open dispatched call returned no gateway response. Record that proof on
-- future uncertainty rows. Historical eligibility ignores the added field.
DO $lab_arena_229_termination_outcome$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$
        pg_catalog.jsonb_build_object('reason', p_reason)
$anchor$;
  v_replacement TEXT := $replacement$
        pg_catalog.jsonb_build_object(
          'reason', p_reason,
          'call', pg_catalog.jsonb_build_object('call_succeeded', FALSE)
        )
$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena__terminate_open_calls(text,text)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'call_succeeded') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_terminate_open_calls_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$lab_arena_229_termination_outcome$;

-- Return the latest immutable ledger state for one submission/kind/provider.
-- All existing totals remain actual billing totals. The successful-call totals
-- are separate and are used only by marker-enabled efficiency decisions.
CREATE OR REPLACE FUNCTION public.lab_arena__successful_call_cost_state(
  p_submission_id TEXT,
  p_kind TEXT,
  p_provider TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_successful_call_cost_state$
  WITH heads AS (
    SELECT DISTINCT ON (ledger.call_identity)
      ledger.entry_kind,
      ledger.amount_microusd,
      CASE
        -- A response reconciled only after round cancellation was never
        -- delivered. The trusted database transition overrides its payload.
        WHEN ledger.entry_kind = 'settlement'
             AND ledger.entry_doc ->> 'late_reconciliation' = 'true'
             AND ledger.entry_doc ->> 'reconciled_uncertainty_reason'
                 = 'round_cancelled'
        THEN FALSE
        WHEN ledger.entry_kind = 'settlement'
             AND pg_catalog.jsonb_typeof(
               ledger.terminal_response -> 'call_succeeded'
             ) = 'boolean'
        THEN (ledger.terminal_response ->> 'call_succeeded')::BOOLEAN
        WHEN ledger.entry_kind = 'settlement'
             AND ledger.entry_doc ->> 'openrouter_delayed_reconciliation'
                 = 'true'
        THEN (
          SELECT CASE
            WHEN pg_catalog.jsonb_typeof(
                   prior.entry_doc #> '{call,call_succeeded}'
                 ) = 'boolean'
            THEN (prior.entry_doc #>> '{call,call_succeeded}')::BOOLEAN
          END
          FROM public.lab_arena_ledger AS prior
          WHERE prior.entry_id =
            (ledger.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
            AND prior.entry_kind = 'uncertain'
        )
        WHEN ledger.entry_kind = 'uncertain'
             AND pg_catalog.jsonb_typeof(
               ledger.entry_doc #> '{call,call_succeeded}'
             ) = 'boolean'
        THEN (ledger.entry_doc #>> '{call,call_succeeded}')::BOOLEAN
        -- Cancellation closes the round and call before a response can be
        -- returned. This database-authored reason is trusted negative proof.
        WHEN ledger.entry_kind = 'uncertain'
             AND ledger.entry_doc ->> 'reason' = 'round_cancelled'
        THEN FALSE
      END AS call_succeeded
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = p_kind
      AND (p_provider IS NULL OR ledger.provider = p_provider)
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  )
  SELECT pg_catalog.jsonb_build_object(
    'settled_microusd',
      COALESCE(SUM(amount_microusd)
        FILTER (WHERE entry_kind = 'settlement'), 0)::BIGINT,
    'reserved_or_uncertain_microusd',
      COALESCE(SUM(amount_microusd)
        FILTER (WHERE entry_kind IN ('reservation', 'dispatch', 'uncertain')),
        0)::BIGINT,
    'inflight_calls', COUNT(*) FILTER (
      WHERE entry_kind IN ('reservation', 'dispatch'))::BIGINT,
    'uncertain_calls', COUNT(*) FILTER (
      WHERE entry_kind = 'uncertain')::BIGINT,
    'refused_calls', COUNT(*) FILTER (
      WHERE entry_kind = 'refusal')::BIGINT,
    'call_count', COUNT(*)::BIGINT,
    'successful_microusd',
      COALESCE(SUM(amount_microusd) FILTER (
        WHERE entry_kind = 'settlement' AND call_succeeded IS TRUE
      ), 0)::BIGINT,
    'successful_calls', COUNT(*) FILTER (
      WHERE entry_kind = 'settlement' AND call_succeeded IS TRUE
    )::BIGINT,
    'success_unresolved_microusd',
      COALESCE(SUM(amount_microusd) FILTER (
        WHERE (entry_kind = 'settlement' AND call_succeeded IS NULL)
           OR (entry_kind = 'uncertain' AND call_succeeded IS DISTINCT FROM FALSE)
           OR entry_kind IN ('reservation', 'dispatch')
      ), 0)::BIGINT,
    'success_unresolved_calls', COUNT(*) FILTER (
      WHERE (entry_kind = 'settlement' AND call_succeeded IS NULL)
         OR (entry_kind = 'uncertain' AND call_succeeded IS DISTINCT FROM FALSE)
         OR entry_kind IN ('reservation', 'dispatch')
    )::BIGINT
  )
  FROM heads
  WHERE p_kind IN ('execute', 'score')
    AND (p_provider IS NULL OR p_provider IN (
      'openrouter', 'deepline', 'scrapingdog'
    ));
$lab_arena_successful_call_cost_state$;
ALTER FUNCTION public.lab_arena__successful_call_cost_state(TEXT, TEXT, TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_cost_state(
  TEXT, TEXT, TEXT
) FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.lab_arena_submission_costs(
  p_submission_id TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_submission_costs$
DECLARE
  v_providers JSONB;
  v_successful_policy BOOLEAN;
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$' THEN
    RAISE EXCEPTION 'lab_arena_submission_costs_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT COALESCE(
    rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
      = 'successful_calls_v1',
    FALSE
  ) INTO v_successful_policy
  FROM public.lab_arena_submissions AS submissions
  JOIN public.lab_arena_rounds AS rounds
    ON rounds.round_id = submissions.round_id
  WHERE submissions.submission_id = p_submission_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT COALESCE(
    pg_catalog.jsonb_agg(
      (CASE WHEN v_successful_policy THEN state.costs ELSE state.costs
        - 'successful_microusd' - 'successful_calls'
        - 'success_unresolved_microusd' - 'success_unresolved_calls'
      END) || pg_catalog.jsonb_build_object(
        'kind', grouped.kind, 'provider', grouped.provider
      ) ORDER BY grouped.kind, grouped.provider
    ),
    '[]'::JSONB
  ) INTO v_providers
  FROM (
    SELECT DISTINCT runs.kind, ledger.provider
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND ledger.call_identity IS NOT NULL
  ) AS grouped
  CROSS JOIN LATERAL (
    SELECT public.lab_arena__successful_call_cost_state(
      p_submission_id, grouped.kind, grouped.provider
    ) AS costs
  ) AS state;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.submission_costs.v1',
    'submission_id', p_submission_id,
    'providers', v_providers
  );
END;
$lab_arena_submission_costs$;
ALTER FUNCTION public.lab_arena_submission_costs(TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_costs(TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_costs(TEXT)
  TO lab_arena_service;

-- Compute the marker-enabled cost result once. Both publication guards and
-- confirmation cohort selection consume this same database calculation.
CREATE OR REPLACE FUNCTION public.lab_arena__successful_call_eligibility(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_company_count BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_successful_call_eligibility$
DECLARE
  v_round public.lab_arena_rounds;
  v_execute JSONB;
  v_judge JSONB;
  v_execution_cap BIGINT;
  v_per_company_cap BIGINT;
  v_efficiency BIGINT;
  v_reason TEXT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id;
  IF NOT FOUND
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_v1'
     OR COALESCE(p_company_count, -1) < 0 THEN
    RAISE EXCEPTION 'lab_arena_successful_call_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_execution_cap :=
    (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT;
  v_per_company_cap :=
    (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;
  IF v_execution_cap IS NULL OR v_execution_cap < 1
     OR v_per_company_cap IS NULL OR v_per_company_cap < 1 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_execute := public.lab_arena__successful_call_cost_state(
    p_submission_id, 'execute', NULL
  );
  v_judge := public.lab_arena__successful_call_cost_state(
    p_submission_id, 'score', NULL
  );
  v_efficiency := (v_execute ->> 'successful_microusd')::BIGINT
    + (v_execute ->> 'success_unresolved_microusd')::BIGINT;
  IF (v_execute ->> 'inflight_calls')::BIGINT > 0
     OR (v_judge ->> 'inflight_calls')::BIGINT > 0 THEN
    v_reason := 'provider_calls_inflight';
  ELSIF (v_execute ->> 'success_unresolved_calls')::BIGINT > 0 THEN
    v_reason := 'provider_cost_uncertain';
  ELSIF v_efficiency > v_execution_cap THEN
    v_reason := 'execution_cap_exceeded';
  ELSIF v_efficiency > v_per_company_cap * p_company_count THEN
    v_reason := 'cost_per_company_exceeded';
  ELSE
    v_reason := 'eligible';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'eligible', v_reason = 'eligible',
    'eligibility_reason', v_reason,
    'sourcing_cost_eligibility_policy', 'successful_calls_v1',
    'competition_sourcing_microusd', v_efficiency,
    'execution_cap_microusd', v_execution_cap,
    'cost_per_company_cap_microusd', v_per_company_cap,
    'eligibility_cap_microusd', LEAST(
      v_execution_cap, v_per_company_cap * p_company_count
    ),
    -- Keep the established flattened integrity-helper contract while also
    -- returning the complete nested report used by the new validator.
    'settled_microusd', (v_execute ->> 'settled_microusd')::BIGINT,
    'reserved_or_uncertain_microusd',
      (v_execute ->> 'reserved_or_uncertain_microusd')::BIGINT,
    'conservative_microusd',
      (v_execute ->> 'settled_microusd')::BIGINT
      + (v_execute ->> 'reserved_or_uncertain_microusd')::BIGINT,
    'execution_inflight_calls',
      (v_execute ->> 'inflight_calls')::BIGINT,
    'execution_uncertain_calls',
      (v_execute ->> 'uncertain_calls')::BIGINT,
    'judge_inflight_calls', (v_judge ->> 'inflight_calls')::BIGINT,
    'judge_uncertain_calls', (v_judge ->> 'uncertain_calls')::BIGINT,
    'execution', v_execute,
    'judge', v_judge
  );
END;
$lab_arena_successful_call_eligibility$;
ALTER FUNCTION public.lab_arena__successful_call_eligibility(
  TEXT, TEXT, BIGINT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_eligibility(
  TEXT, TEXT, BIGINT
) FROM PUBLIC;

-- Validate every published total that enters the Python cost summary. Provider
-- detail remains a bounded reporting breakdown and does not affect the total.
CREATE OR REPLACE FUNCTION public.lab_arena__successful_call_publication_valid(
  p_round_id TEXT,
  p_ranking JSONB,
  p_company_count BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_successful_call_publication_valid$
DECLARE
  v_expected JSONB;
  v_cost JSONB;
  v_key TEXT;
BEGIN
  v_expected := public.lab_arena__successful_call_eligibility(
    p_round_id, p_ranking ->> 'submission_id', p_company_count
  );
  v_cost := p_ranking -> 'cost_summary';
  IF v_cost ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM v_expected ->> 'sourcing_cost_eligibility_policy'
     OR (v_cost ->> 'competition_sourcing_microusd')::BIGINT
       IS DISTINCT FROM
         (v_expected ->> 'competition_sourcing_microusd')::BIGINT THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
      USING ERRCODE = '22023';
  END IF;
  FOREACH v_key IN ARRAY ARRAY[
    'execution_cap_microusd', 'cost_per_company_cap_microusd',
    'eligibility_cap_microusd'
  ] LOOP
    IF (v_cost ->> v_key)::BIGINT IS DISTINCT FROM
         (v_expected ->> v_key)::BIGINT THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;
  FOREACH v_key IN ARRAY ARRAY[
    'settled_microusd', 'reserved_or_uncertain_microusd',
    'inflight_calls', 'uncertain_calls', 'refused_calls', 'call_count',
    'successful_microusd', 'successful_calls',
    'success_unresolved_microusd', 'success_unresolved_calls'
  ] LOOP
    IF (v_cost #>> ARRAY['execution', v_key])::BIGINT IS DISTINCT FROM
         (v_expected #>> ARRAY['execution', v_key])::BIGINT
       OR (v_cost #>> ARRAY['judge', v_key])::BIGINT IS DISTINCT FROM
         (v_expected #>> ARRAY['judge', v_key])::BIGINT THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;
  IF (v_cost #>> '{execution,conservative_microusd}')::BIGINT
       IS DISTINCT FROM
       (v_expected #>> '{execution,settled_microusd}')::BIGINT
       + (v_expected #>> '{execution,reserved_or_uncertain_microusd}')::BIGINT
     OR (v_cost #>> '{judge,conservative_microusd}')::BIGINT
       IS DISTINCT FROM
       (v_expected #>> '{judge,settled_microusd}')::BIGINT
       + (v_expected #>> '{judge,reserved_or_uncertain_microusd}')::BIGINT THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
      USING ERRCODE = '22023';
  END IF;
  RETURN TRUE;
END;
$lab_arena_successful_call_publication_valid$;
ALTER FUNCTION public.lab_arena__successful_call_publication_valid(
  TEXT, JSONB, BIGINT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_publication_valid(
  TEXT, JSONB, BIGINT
) FROM PUBLIC;

-- Normal rounds use the raw cost result as their final eligibility. Integrity
-- rounds can apply a separately verified output/account override, so their
-- trigger calls the cost-only validator above.
CREATE OR REPLACE FUNCTION public.lab_arena__successful_call_publication_eligibility_valid(
  p_round_id TEXT,
  p_ranking JSONB,
  p_company_count BIGINT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_successful_call_publication_eligibility_valid$
DECLARE
  v_expected JSONB;
BEGIN
  v_expected := public.lab_arena__successful_call_eligibility(
    p_round_id, p_ranking ->> 'submission_id', p_company_count
  );
  IF (p_ranking ->> 'eligible')::BOOLEAN IS DISTINCT FROM
       (v_expected ->> 'eligible')::BOOLEAN
     OR p_ranking ->> 'eligibility_reason' IS DISTINCT FROM
       v_expected ->> 'eligibility_reason' THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
      USING ERRCODE = '22023';
  END IF;
  RETURN public.lab_arena__successful_call_publication_valid(
    p_round_id, p_ranking, p_company_count
  );
END;
$lab_arena_successful_call_publication_eligibility_valid$;
ALTER FUNCTION public.lab_arena__successful_call_publication_eligibility_valid(
  TEXT, JSONB, BIGINT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__successful_call_publication_eligibility_valid(
  TEXT, JSONB, BIGINT
) FROM PUBLIC;

-- Route only marker-enabled normal rounds through the new calculation. The
-- existing body remains byte-for-byte effective for marker-absent rounds.
DO $lab_arena_229_baseline_guard$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$
      v_returned := (v_summary ->> 'returned_company_count')::BIGINT;
      v_eligibility_cap := LEAST(
$anchor$;
  v_replacement TEXT := $replacement$
      v_returned := (v_summary ->> 'returned_company_count')::BIGINT;
      IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_v1' THEN
        PERFORM public.lab_arena__successful_call_publication_eligibility_valid(
          NEW.round_id, v_ranking, v_returned
        );
        CONTINUE;
      END IF;
      v_eligibility_cap := LEAST(
$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_publication_baseline_guard_v1()'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       'lab_arena__successful_call_publication_eligibility_valid') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_publication_guard_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$lab_arena_229_baseline_guard$;

-- Integrity confirmation and publication already share this helper. Add an
-- early marker-only return and preserve the complete prior branch unchanged.
DO $lab_arena_229_integrity_eligibility$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$
  IF v_execution_cap IS NULL OR v_execution_cap < 1
     OR v_per_company_cap IS NULL OR v_per_company_cap < 1 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT
$anchor$;
  v_replacement TEXT := $replacement$
  IF v_execution_cap IS NULL OR v_execution_cap < 1
     OR v_per_company_cap IS NULL OR v_per_company_cap < 1 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       = 'successful_calls_v1' THEN
    IF NOT COALESCE((v_summary ->> 'valid')::BOOLEAN, FALSE)
       OR v_summary -> 'score' = 'null'::JSONB THEN
      RETURN public.lab_arena__successful_call_eligibility(
        p_round_id, p_submission_id, v_qualified
      ) || pg_catalog.jsonb_build_object(
        'eligible', FALSE,
        'eligibility_reason', 'stored_output_invalid',
        'qualified_company_count', v_qualified
      );
    END IF;
    RETURN public.lab_arena__successful_call_eligibility(
      p_round_id, p_submission_id, v_qualified
    ) || pg_catalog.jsonb_build_object(
      'qualified_company_count', v_qualified
    );
  END IF;
  SELECT
$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena__integrity_eligibility(text,text,integer[])'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       'lab_arena__successful_call_eligibility') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_integrity_eligibility_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$lab_arena_229_integrity_eligibility$;

-- The integrity trigger compares the old totals itself. For a new-policy row,
-- also require the complete successful-call summary to match the same helper.
DO $lab_arena_229_integrity_guard$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$
    IF v_submission_id = v_baseline_id THEN
$anchor$;
  v_replacement TEXT := $replacement$
    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_v1' THEN
      PERFORM public.lab_arena__successful_call_publication_valid(
        NEW.round_id, v_ranking, v_qualified::BIGINT
      );
    END IF;
    IF v_submission_id = v_baseline_id THEN
$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_integrity_publication_guard_v1()'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       'lab_arena__successful_call_publication_valid') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_integrity_publication_guard_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$lab_arena_229_integrity_guard$;

CREATE OR REPLACE FUNCTION public.lab_arena_successful_call_cost_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_successful_call_cost_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.successful_call_cost_schema.v1',
    'version', 229,
    'policy', 'successful_calls_v1'
  );
$lab_arena_successful_call_cost_schema$;
ALTER FUNCTION public.lab_arena_successful_call_cost_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_successful_call_cost_schema_v1()
  FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_successful_call_cost_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
