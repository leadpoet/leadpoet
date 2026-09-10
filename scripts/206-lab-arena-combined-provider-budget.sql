-- Count every provider against the existing submission/kind money caps.
-- This migration changes functions only; existing round configuration is not
-- rewritten. The latest ledger state remains the sole accounting authority.

BEGIN;

CREATE OR REPLACE FUNCTION public.lab_arena__submission_kind_spend(
  p_submission_id TEXT,
  p_kind TEXT
)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena__submission_kind_spend$
  SELECT COALESCE(SUM(head.amount_microusd), 0)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity)
      ledger.entry_kind, ledger.amount_microusd
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = p_kind
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain');
$lab_arena__submission_kind_spend$;
ALTER FUNCTION public.lab_arena__submission_kind_spend(TEXT, TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_reserve_call(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_operation_id TEXT, p_provider TEXT, p_funding_source TEXT,
  p_amount_microusd BIGINT, p_call_doc JSONB, p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_reserve_call$
DECLARE
  v_run public.lab_arena_runs;
  v_round public.lab_arena_rounds;
  v_head public.lab_arena_ledger;
  v_submission public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_quota INTEGER;
  v_stage_quota BIGINT;
  v_consumed BIGINT;
  v_money_cap BIGINT;
  v_spent BIGINT;
  v_locked_spend BIGINT;
  v_dynamic BOOLEAN;
  v_dynamic_inflight BOOLEAN;
  v_reason TEXT := NULL;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_call_identity, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_operation_id, '') !~ '^[a-z0-9_.]{1,64}$'
     OR p_provider NOT IN ('scrapingdog', 'deepline', 'openrouter')
     OR p_funding_source NOT IN ('host', 'miner_key')
     OR COALESCE(p_amount_microusd, -1) < 0
     OR pg_catalog.jsonb_typeof(p_call_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_call_doc::TEXT) > 65536
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_reserve_input_invalid' USING ERRCODE = '22023';
  END IF;
  IF p_call_doc ? 'reserve_remaining_budget' THEN
    IF pg_catalog.jsonb_typeof(p_call_doc -> 'reserve_remaining_budget')
         IS DISTINCT FROM 'boolean'
       OR (p_call_doc ->> 'reserve_remaining_budget')::BOOLEAN IS NOT TRUE
       OR p_provider <> 'deepline'
       OR p_amount_microusd <> 0 THEN
      RAISE EXCEPTION 'lab_arena_dynamic_reserve_input_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_dynamic := TRUE;
  ELSE
    v_dynamic := FALSE;
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = v_run.round_id;
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  v_is_baseline := v_submission.is_king
    AND v_run.submission_id =
      'baseline-' || pg_catalog.regexp_replace(v_run.round_id, '^arena-', '')
    AND v_run.miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey';
  IF (v_is_baseline AND p_funding_source <> 'host')
     OR (NOT v_is_baseline AND p_funding_source <> 'miner_key') THEN
    RAISE EXCEPTION 'lab_arena_funding_source_mismatch' USING ERRCODE = '42501';
  END IF;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NOT NULL THEN
    IF v_head.run_id <> p_run_id THEN
      RAISE EXCEPTION 'lab_arena_call_identity_foreign' USING ERRCODE = '23505';
    END IF;
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  v_quota := CASE v_run.kind
    WHEN 'score' THEN
      ((v_round.configuration_doc -> 'scoring_call_quotas') ->> p_provider)::INTEGER
    ELSE ((v_round.configuration_doc -> 'call_quotas') ->> p_provider)::INTEGER
  END;
  IF v_quota IS NULL OR v_quota < 1 THEN
    RAISE EXCEPTION 'lab_arena_quota_missing' USING ERRCODE = '22023';
  END IF;
  v_stage_quota := v_quota::BIGINT
    * (CASE v_run.stage
        WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT
        ELSE (v_round.configuration_doc ->> 'stage_2_icp_count')::BIGINT
      END)
    * (v_round.configuration_doc ->> 'max_attempts_per_assignment')::BIGINT;
  v_consumed := public.lab_arena__run_consumed(p_run_id, p_provider);
  IF v_consumed >= v_quota THEN
    v_reason := 'per_icp_quota';
  END IF;
  IF v_reason IS NULL THEN
    -- Lock order stays round, run, submission. This lock serializes the
    -- aggregate check and reservation insert across all providers and runs.
    SELECT * INTO v_submission FROM public.lab_arena_submissions
    WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;
    v_consumed := public.lab_arena__submission_stage_consumed(
      v_run.submission_id, v_run.stage, p_provider, v_run.kind
    );
    IF v_consumed >= v_stage_quota THEN
      v_reason := 'stage_quota';
    END IF;
    IF v_reason IS NULL THEN
      v_money_cap := CASE v_run.kind
        WHEN 'score' THEN
          (v_round.configuration_doc ->> 'scoring_cap_microusd')::BIGINT
        ELSE (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT
      END;
      IF v_money_cap IS NULL OR v_money_cap < 1 THEN
        RAISE EXCEPTION 'lab_arena_money_cap_missing' USING ERRCODE = '22023';
      END IF;
      v_spent := public.lab_arena__submission_kind_spend(
        v_run.submission_id, v_run.kind
      );
      IF v_dynamic THEN
        SELECT EXISTS (
          SELECT 1
          FROM (
            SELECT DISTINCT ON (ledger.call_identity)
              ledger.call_identity, ledger.entry_kind
            FROM public.lab_arena_ledger AS ledger
            WHERE ledger.submission_id = v_run.submission_id
              AND ledger.call_identity IS NOT NULL
            ORDER BY ledger.call_identity, ledger.entry_id DESC
          ) AS heads
          JOIN public.lab_arena_ledger AS reservation
            ON reservation.call_identity = heads.call_identity
           AND reservation.entry_kind = 'reservation'
          JOIN public.lab_arena_runs AS reservation_run
            ON reservation_run.run_id = reservation.run_id
          WHERE reservation_run.kind = v_run.kind
            AND heads.entry_kind IN ('reservation', 'dispatch')
            AND reservation.entry_doc -> 'reserve_remaining_budget' = 'true'::JSONB
        ) INTO v_dynamic_inflight;
        IF v_dynamic_inflight THEN
          v_expires := pg_catalog.clock_timestamp()
            + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
          UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
          WHERE run_id = p_run_id;
          RETURN pg_catalog.jsonb_build_object(
            'status', 'budget_busy', 'idempotent', FALSE,
            'call_identity', p_call_identity, 'lease_expires_at', v_expires
          );
        END IF;
        IF v_spent >= v_money_cap THEN
          v_reason := 'money_cap';
        ELSE
          p_amount_microusd := v_money_cap - v_spent;
        END IF;
      ELSIF (p_amount_microusd > 0 AND v_spent >= v_money_cap)
         OR v_spent > v_money_cap - p_amount_microusd THEN
        SELECT COALESCE(SUM(head.amount_microusd), 0)::BIGINT
        INTO v_locked_spend
        FROM (
          SELECT DISTINCT ON (ledger.call_identity)
            ledger.entry_kind, ledger.amount_microusd
          FROM public.lab_arena_ledger AS ledger
          JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
          WHERE ledger.submission_id = v_run.submission_id
            AND runs.kind = v_run.kind
            AND ledger.call_identity IS NOT NULL
          ORDER BY ledger.call_identity, ledger.entry_id DESC
        ) AS head
        WHERE head.entry_kind IN ('settlement', 'uncertain');
        IF v_locked_spend <= v_money_cap - p_amount_microusd THEN
          v_expires := pg_catalog.clock_timestamp()
            + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
          UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
          WHERE run_id = p_run_id;
          RETURN pg_catalog.jsonb_build_object(
            'status', 'budget_busy', 'idempotent', FALSE,
            'call_identity', p_call_identity, 'lease_expires_at', v_expires
          );
        END IF;
        v_reason := 'money_cap';
      END IF;
    END IF;
  END IF;
  v_expires := pg_catalog.clock_timestamp()
    + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  IF v_reason IS NOT NULL THEN
    INSERT INTO public.lab_arena_ledger (
      entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
      call_identity, provider, operation_id, funding_source,
      amount_microusd, entry_doc
    ) VALUES (
      'refusal', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
      p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
      p_funding_source, 0,
      pg_catalog.jsonb_build_object(
        'reason', v_reason, 'requested_microusd', p_amount_microusd,
        'call', p_call_doc
      )
    );
    UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
    WHERE run_id = p_run_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'refused', 'idempotent', FALSE, 'reason', v_reason,
      'call_identity', p_call_identity, 'lease_expires_at', v_expires
    );
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc
  ) VALUES (
    'reservation', v_run.miner_hotkey, v_run.round_id, v_run.submission_id,
    p_run_id, v_run.stage, p_call_identity, p_provider, p_operation_id,
    p_funding_source, p_amount_microusd, p_call_doc
  );
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
  WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'reserved', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', p_amount_microusd,
    'lease_expires_at', v_expires
  );
END;
$lab_arena_reserve_call$;
ALTER FUNCTION public.lab_arena_reserve_call(
  TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_settle_call(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_actual_microusd BIGINT, p_terminal_response JSONB,
  p_lease_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_settle_call$
DECLARE
  v_run public.lab_arena_runs;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_submission public.lab_arena_submissions;
  v_expires TIMESTAMPTZ;
BEGIN
  IF COALESCE(p_actual_microusd, -1) < 0
     OR pg_catalog.jsonb_typeof(p_terminal_response) IS DISTINCT FROM 'object'
     OR pg_catalog.octet_length(p_terminal_response::TEXT) > 4194304
     OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN
    RAISE EXCEPTION 'lab_arena_settle_input_invalid' USING ERRCODE = '22023';
  END IF;
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  -- Match reserve's round, run, submission lock order. A settlement that
  -- exceeds its estimate is visible before another reservation can pass.
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF v_head.entry_kind <> 'dispatch' THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
  WHERE call_identity = p_call_identity AND entry_kind = 'reservation';
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source,
    amount_microusd, entry_doc, terminal_response
  ) VALUES (
    'settlement', v_reservation.miner_hotkey, v_reservation.round_id,
    v_reservation.submission_id, p_run_id, v_reservation.stage,
    p_call_identity, v_reservation.provider, v_reservation.operation_id,
    v_reservation.funding_source, p_actual_microusd,
    pg_catalog.jsonb_build_object(
      'reserved_microusd', v_reservation.amount_microusd,
      'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
      'variance_microusd', p_actual_microusd - v_reservation.amount_microusd
    ),
    p_terminal_response
  );
  v_expires := pg_catalog.clock_timestamp()
    + pg_catalog.make_interval(secs => p_lease_ttl_seconds);
  UPDATE public.lab_arena_runs SET lease_expires_at = v_expires
  WHERE run_id = p_run_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'settled', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'actual_microusd', p_actual_microusd,
    'released_microusd', v_reservation.amount_microusd - p_actual_microusd,
    'variance_microusd', p_actual_microusd - v_reservation.amount_microusd,
    'terminal_response', p_terminal_response,
    'lease_expires_at', v_expires
  );
END;
$lab_arena_settle_call$;
ALTER FUNCTION public.lab_arena_settle_call(
  TEXT, TEXT, TEXT, BIGINT, JSONB, INTEGER
) OWNER TO lab_arena_owner;

-- New cost-policy publications retain every quality score but only an
-- independently verified cost-eligible challenger can become king. Historical
-- publications without cost_per_company_microusd keep their original rules.
CREATE OR REPLACE FUNCTION public.lab_arena_publication_baseline_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_publication_baseline_guard$
DECLARE
  v_decision JSONB;
  v_baseline_id TEXT;
  v_baseline_count BIGINT;
  v_public_baseline_count BIGINT;
  v_baseline_ranking JSONB;
  v_winner_ranking JSONB;
  v_winner_participant JSONB;
  v_baseline_score NUMERIC;
  v_winner_score NUMERIC;
  v_winner_id TEXT;
  v_ranking JSONB;
  v_summary JSONB;
  v_submission_id TEXT;
  v_returned BIGINT;
  v_execution_cap BIGINT;
  v_per_company_cap BIGINT;
  v_eligibility_cap BIGINT;
  v_settled BIGINT;
  v_reserved_or_uncertain BIGINT;
  v_inflight BIGINT;
  v_score_inflight BIGINT;
  v_conservative BIGINT;
  v_expected_eligible BOOLEAN;
  v_expected_reason TEXT;
  v_cost_policy BOOLEAN;
BEGIN
  IF NEW.status <> 'published' OR OLD.status = 'published' THEN
    RETURN NEW;
  END IF;
  SELECT pg_catalog.count(*), pg_catalog.min(participant ->> 'submission_id')
  INTO v_baseline_count, v_baseline_id
  FROM pg_catalog.jsonb_array_elements(
    COALESCE(NEW.participants, '[]'::JSONB)
  ) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_baseline_count <> 1 OR COALESCE(v_baseline_id, '') = '' THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_typeof(NEW.publication_doc -> 'participants')
       IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'final_ranking')
       IS DISTINCT FROM 'array'
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(
         NEW.publication_doc -> 'participants'
       ) AS participant
       WHERE participant ? 'is_king' OR NOT participant ? 'is_baseline'
     )
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(
         NEW.publication_doc -> 'final_ranking'
       ) AS ranking
       WHERE ranking ? 'is_king' OR NOT ranking ? 'is_baseline'
     ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_fields_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.count(*) INTO v_public_baseline_count
  FROM pg_catalog.jsonb_array_elements(
    NEW.publication_doc -> 'participants'
  ) AS participant
  WHERE COALESCE((participant ->> 'is_baseline')::BOOLEAN, FALSE)
    AND participant ->> 'submission_id' = v_baseline_id;
  IF v_public_baseline_count <> 1 OR EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(
      NEW.publication_doc -> 'participants'
    ) AS participant
    WHERE COALESCE((participant ->> 'is_baseline')::BOOLEAN, FALSE)
      AND participant ->> 'submission_id' <> v_baseline_id
  ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT ranking INTO v_baseline_ranking
  FROM pg_catalog.jsonb_array_elements(
    NEW.publication_doc -> 'final_ranking'
  ) AS ranking
  WHERE ranking ->> 'submission_id' = v_baseline_id
    AND COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
  LIMIT 1;
  IF v_baseline_ranking IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_typeof(v_baseline_ranking -> 'final_score') = 'number' THEN
    v_baseline_score := (v_baseline_ranking ->> 'final_score')::NUMERIC;
  END IF;

  v_cost_policy := NEW.configuration_doc ? 'cost_per_company_microusd';
  IF v_cost_policy THEN
    v_execution_cap := (NEW.configuration_doc ->> 'execution_cap_microusd')::BIGINT;
    v_per_company_cap :=
      (NEW.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;
    IF v_execution_cap < 1 OR v_per_company_cap < 1 THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_policy_invalid'
        USING ERRCODE = '22023';
    END IF;
    FOR v_ranking IN SELECT value FROM pg_catalog.jsonb_array_elements(
      NEW.publication_doc -> 'final_ranking'
    ) LOOP
      v_submission_id := v_ranking ->> 'submission_id';
      v_summary := v_ranking -> 'cost_summary';
      IF COALESCE(v_submission_id, '') = ''
         OR pg_catalog.jsonb_typeof(v_ranking -> 'eligible')
           IS DISTINCT FROM 'boolean' THEN
        RAISE EXCEPTION 'lab_arena_publication_cost_report_invalid'
          USING ERRCODE = '22023';
      END IF;
      -- Object storage bytes are outside PostgreSQL. The service may therefore
      -- fail closed on an accepted output it cannot validate, but this row can
      -- never be eligible or become the winner.
      IF v_summary = 'null'::JSONB
         AND (v_ranking ->> 'eligible')::BOOLEAN IS FALSE
         AND v_ranking ->> 'eligibility_reason' = 'stored_output_invalid' THEN
        CONTINUE;
      END IF;
      IF pg_catalog.jsonb_typeof(v_summary) IS DISTINCT FROM 'object'
         OR pg_catalog.jsonb_typeof(v_summary -> 'returned_company_count')
           IS DISTINCT FROM 'number'
         OR (v_summary ->> 'returned_company_count')::NUMERIC
           <> pg_catalog.trunc((v_summary ->> 'returned_company_count')::NUMERIC)
         OR (v_summary ->> 'returned_company_count')::NUMERIC NOT BETWEEN 0 AND 100
         OR pg_catalog.jsonb_typeof(v_summary -> 'execution')
           IS DISTINCT FROM 'object' THEN
        RAISE EXCEPTION 'lab_arena_publication_cost_report_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_returned := (v_summary ->> 'returned_company_count')::BIGINT;
      v_eligibility_cap := LEAST(
        v_execution_cap, v_per_company_cap * v_returned
      );
      SELECT
        COALESCE(SUM(head.amount_microusd)
          FILTER (WHERE head.entry_kind = 'settlement'), 0)::BIGINT,
        COALESCE(SUM(head.amount_microusd)
          FILTER (WHERE head.entry_kind IN
            ('reservation', 'dispatch', 'uncertain')), 0)::BIGINT,
        COUNT(*) FILTER (WHERE head.entry_kind IN
          ('reservation', 'dispatch'))::BIGINT
      INTO v_settled, v_reserved_or_uncertain, v_inflight
      FROM (
        SELECT DISTINCT ON (ledger.call_identity)
          ledger.entry_kind, ledger.amount_microusd
        FROM public.lab_arena_ledger AS ledger
        JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
        WHERE ledger.submission_id = v_submission_id
          AND runs.kind = 'execute'
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head;
      SELECT COUNT(*) FILTER (WHERE head.entry_kind IN
        ('reservation', 'dispatch'))::BIGINT
      INTO v_score_inflight
      FROM (
        SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
        FROM public.lab_arena_ledger AS ledger
        JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
        WHERE ledger.submission_id = v_submission_id
          AND runs.kind = 'score'
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head;
      v_conservative := v_settled + v_reserved_or_uncertain;
      IF v_inflight > 0 OR v_score_inflight > 0 THEN
        v_expected_reason := 'provider_calls_inflight';
      ELSIF v_conservative > v_execution_cap THEN
        v_expected_reason := 'execution_cap_exceeded';
      ELSIF v_conservative > v_per_company_cap * v_returned THEN
        v_expected_reason := 'cost_per_company_exceeded';
      ELSE
        v_expected_reason := 'eligible';
      END IF;
      v_expected_eligible := v_expected_reason = 'eligible';
      IF (v_ranking ->> 'eligible')::BOOLEAN IS DISTINCT FROM v_expected_eligible
         OR v_ranking ->> 'eligibility_reason' IS DISTINCT FROM v_expected_reason
         OR (v_summary ->> 'execution_cap_microusd')::BIGINT
           IS DISTINCT FROM v_execution_cap
         OR (v_summary ->> 'cost_per_company_cap_microusd')::BIGINT
           IS DISTINCT FROM v_per_company_cap
         OR (v_summary ->> 'eligibility_cap_microusd')::BIGINT
           IS DISTINCT FROM v_eligibility_cap
         OR (v_summary #>> '{execution,settled_microusd}')::BIGINT
           IS DISTINCT FROM v_settled
         OR (v_summary #>> '{execution,reserved_or_uncertain_microusd}')::BIGINT
           IS DISTINCT FROM v_reserved_or_uncertain
         OR (v_summary #>> '{execution,conservative_microusd}')::BIGINT
           IS DISTINCT FROM v_conservative
         OR (v_summary #>> '{execution,inflight_calls}')::BIGINT
           IS DISTINCT FROM v_inflight
         OR (v_summary #>> '{judge,inflight_calls}')::BIGINT
           IS DISTINCT FROM v_score_inflight THEN
        RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
          USING ERRCODE = '22023';
      END IF;
    END LOOP;
  END IF;

  v_decision := NEW.publication_doc -> 'king_decision';
  IF pg_catalog.jsonb_typeof(v_decision) IS DISTINCT FROM 'object'
     OR v_decision ->> 'outcome' NOT IN ('crowned', 'no_king') THEN
    RAISE EXCEPTION 'lab_arena_publication_decision_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_decision ->> 'outcome' = 'no_king' THEN
    IF COALESCE(v_decision ->> 'king_hotkey', '') <> ''
       OR COALESCE(v_decision ->> 'king_submission_id', '') <> ''
       OR COALESCE(v_decision ->> 'winner_submission_id', '') <> '' THEN
      RAISE EXCEPTION 'lab_arena_publication_decision_invalid'
        USING ERRCODE = '22023';
    END IF;
    IF v_baseline_score IS NOT NULL AND EXISTS (
      SELECT 1 FROM pg_catalog.jsonb_array_elements(
        NEW.publication_doc -> 'final_ranking'
      ) AS ranking
      WHERE NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
        AND (NOT v_cost_policy OR (ranking ->> 'eligible')::BOOLEAN)
        AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
        AND (ranking ->> 'final_score')::NUMERIC >= v_baseline_score + 1
    ) THEN
      RAISE EXCEPTION 'lab_arena_publication_winner_missing'
        USING ERRCODE = '22023';
    END IF;
    RETURN NEW;
  END IF;
  v_winner_id := v_decision ->> 'winner_submission_id';
  IF COALESCE(v_winner_id, '') = ''
     OR v_decision ->> 'king_submission_id' IS DISTINCT FROM v_winner_id
     OR v_decision ->> 'king_hotkey'
       IS NOT DISTINCT FROM NEW.configuration_doc ->> 'baseline_hotkey' THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT participant INTO v_winner_participant
  FROM pg_catalog.jsonb_array_elements(
    COALESCE(NEW.participants, '[]'::JSONB)
  ) AS participant
  WHERE participant ->> 'submission_id' = v_winner_id LIMIT 1;
  SELECT ranking INTO v_winner_ranking
  FROM pg_catalog.jsonb_array_elements(
    NEW.publication_doc -> 'final_ranking'
  ) AS ranking
  WHERE ranking ->> 'submission_id' = v_winner_id LIMIT 1;
  IF v_winner_participant IS NULL
     OR COALESCE((v_winner_participant ->> 'is_king')::BOOLEAN, FALSE)
     OR v_winner_participant ->> 'miner_hotkey'
       IS DISTINCT FROM v_decision ->> 'king_hotkey'
     OR v_winner_ranking IS NULL
     OR COALESCE((v_winner_ranking ->> 'is_baseline')::BOOLEAN, FALSE)
     OR pg_catalog.jsonb_typeof(v_winner_ranking -> 'final_score')
       IS DISTINCT FROM 'number'
     OR v_baseline_score IS NULL
     OR (v_cost_policy
       AND NOT COALESCE((v_winner_ranking ->> 'eligible')::BOOLEAN, FALSE)) THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_winner_score := (v_winner_ranking ->> 'final_score')::NUMERIC;
  IF v_winner_score < v_baseline_score + 1 OR EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(
      NEW.publication_doc -> 'final_ranking'
    ) AS ranking
    WHERE NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
      AND (NOT v_cost_policy OR (ranking ->> 'eligible')::BOOLEAN)
      AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
      AND (
        (ranking ->> 'final_score')::NUMERIC > v_winner_score
        OR ((ranking ->> 'final_score')::NUMERIC = v_winner_score
          AND ranking ->> 'submission_id' < v_winner_id)
      )
  ) THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid'
      USING ERRCODE = '22023';
  END IF;
  RETURN NEW;
END;
$lab_arena_publication_baseline_guard$;
ALTER FUNCTION public.lab_arena_publication_baseline_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_publication_baseline_guard_v1()
  FROM PUBLIC;

-- A legacy live round may have opened before this policy existed. Patch the
-- existing commit RPC without changing its signature, so the exact new policy
-- freezes in its single locked open-to-committed update. Typed configurations
-- and every shadow round retain their original custom values.
DO $lab_arena_commit_cost_policy$
DECLARE
  v_definition TEXT;
  v_old TEXT := 'configuration_doc = v_round.configuration_doc || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
  v_new TEXT := 'configuration_doc = (
        CASE
          WHEN v_round.configuration_doc ->> ''mode'' = ''live''
               AND NOT v_round.configuration_doc ? ''cost_per_company_microusd''
          THEN v_round.configuration_doc || pg_catalog.jsonb_build_object(
            ''execution_cap_microusd'', 50000000,
            ''cost_per_company_microusd'', 500000
          )
          ELSE v_round.configuration_doc
        END
      ) || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_commit_round_v2'
    AND procedure.pronargs = 7;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_commit_round_v2_missing' USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(
       v_definition,
       'AND NOT v_round.configuration_doc ? ''cost_per_company_microusd'''
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_commit_cost_policy_unknown'
        USING ERRCODE = '55000';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$lab_arena_commit_cost_policy$;

-- Extend the existing write-once exception narrowly. It independently verifies
-- the exact policy values written by the commit RPC and every unchanged field.
DO $lab_arena_cost_policy_write_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := '(NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')';
  v_new TEXT := '(
        (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')
        OR (
          OLD.configuration_doc ->> ''mode'' = ''live''
          AND NOT OLD.configuration_doc ? ''cost_per_company_microusd''
          AND OLD.icp_set_date IS NULL
          AND NEW.icp_set_date IS NOT NULL
          AND (NEW.configuration_doc ->> ''execution_cap_microusd'')::BIGINT = 50000000
          AND (NEW.configuration_doc ->> ''cost_per_company_microusd'')::BIGINT = 500000
          AND (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'' - ''cost_per_company_microusd'') =
              (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'')
        )
      )';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_rounds_write_once_v1'
    AND procedure.pronargs = 0;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_policy_write_guard_missing'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(
       v_definition,
       'AND NOT OLD.configuration_doc ? ''cost_per_company_microusd'''
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_cost_policy_write_guard_unknown'
        USING ERRCODE = '55000';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$lab_arena_cost_policy_write_guard$;

-- Return bounded aggregate accounting only. No request or response payload is
-- exposed, and the six possible kind/provider groups bound the result size.
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
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$' THEN
    RAISE EXCEPTION 'lab_arena_submission_costs_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM public.lab_arena_submissions
    WHERE submission_id = p_submission_id
  ) THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT COALESCE(
    pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object(
        'kind', grouped.kind,
        'provider', grouped.provider,
        'settled_microusd', grouped.settled_microusd,
        'reserved_or_uncertain_microusd',
          grouped.reserved_or_uncertain_microusd,
        'inflight_calls', grouped.inflight_calls,
        'uncertain_calls', grouped.uncertain_calls,
        'refused_calls', grouped.refused_calls,
        'call_count', grouped.call_count
      ) ORDER BY grouped.kind, grouped.provider
    ),
    '[]'::JSONB
  ) INTO v_providers
  FROM (
    SELECT
      heads.kind,
      heads.provider,
      COALESCE(SUM(heads.amount_microusd)
        FILTER (WHERE heads.entry_kind = 'settlement'), 0)::BIGINT
        AS settled_microusd,
      COALESCE(SUM(heads.amount_microusd)
        FILTER (WHERE heads.entry_kind IN
          ('reservation', 'dispatch', 'uncertain')), 0)::BIGINT
        AS reserved_or_uncertain_microusd,
      COUNT(*) FILTER (WHERE heads.entry_kind IN
        ('reservation', 'dispatch'))::BIGINT AS inflight_calls,
      COUNT(*) FILTER (WHERE heads.entry_kind = 'uncertain')::BIGINT
        AS uncertain_calls,
      COUNT(*) FILTER (WHERE heads.entry_kind = 'refusal')::BIGINT
        AS refused_calls,
      COUNT(*)::BIGINT AS call_count
    FROM (
      SELECT DISTINCT ON (ledger.call_identity)
        runs.kind, ledger.provider, ledger.entry_kind, ledger.amount_microusd
      FROM public.lab_arena_ledger AS ledger
      JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
      WHERE ledger.submission_id = p_submission_id
        AND ledger.call_identity IS NOT NULL
      ORDER BY ledger.call_identity, ledger.entry_id DESC
    ) AS heads
    GROUP BY heads.kind, heads.provider
  ) AS grouped;
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
DO $lab_arena_submission_costs_acl$
DECLARE
  role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_submission_costs(TEXT) FROM %I',
        role_name
      );
    END IF;
  END LOOP;
END;
$lab_arena_submission_costs_acl$;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_costs(TEXT)
  TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
COMMIT;
