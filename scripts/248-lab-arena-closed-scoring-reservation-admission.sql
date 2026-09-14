-- A failed, closed scoring request with a strictly bound dynamic Deepline
-- reservation keeps its full uncertain amount in the audit ledger, but no
-- longer reserves the whole remaining scoring allowance for admission. A
-- later exact settlement becomes ordinary admitted spend automatically.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__closed_score_dynamic_uncertainty_v1(
  p_uncertain_entry_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $closed_score_dynamic_uncertainty$
  SELECT COALESCE((
    SELECT
      uncertainty.entry_kind = 'uncertain'
      AND uncertainty.provider = 'deepline'
      AND uncertainty.funding_source = 'miner_key'
      AND runs.kind = 'score'
      AND runs.status = 'failed'
      AND runs.terminal_cause IN (
        'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
        'stage_closed', 'judge_error', 'judge_timeout'
      )
      AND reservation.entry_doc -> 'reserve_remaining_budget' = 'true'::JSONB
      AND pg_catalog.jsonb_typeof(
            uncertainty.entry_doc #> '{call,call_succeeded}'
          ) = 'boolean'
      AND (uncertainty.entry_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
      AND public.lab_arena__deepline_cost_binding_v1(
            uncertainty.entry_doc,
            reservation.entry_doc,
            uncertainty.call_identity
          )
      AND EXISTS (
        SELECT 1
        FROM public.lab_arena_ledger AS dispatch
        WHERE dispatch.call_identity = uncertainty.call_identity
          AND dispatch.entry_kind = 'dispatch'
          AND dispatch.run_id = uncertainty.run_id
          AND dispatch.round_id = uncertainty.round_id
          AND dispatch.submission_id = uncertainty.submission_id
          AND dispatch.miner_hotkey = uncertainty.miner_hotkey
          AND dispatch.stage = uncertainty.stage
          AND dispatch.provider = uncertainty.provider
          AND dispatch.operation_id = uncertainty.operation_id
          AND dispatch.funding_source = uncertainty.funding_source
      )
    FROM public.lab_arena_ledger AS uncertainty
    JOIN public.lab_arena_ledger AS reservation
      ON reservation.call_identity = uncertainty.call_identity
     AND reservation.entry_kind = 'reservation'
     AND reservation.run_id = uncertainty.run_id
     AND reservation.round_id = uncertainty.round_id
     AND reservation.submission_id = uncertainty.submission_id
     AND reservation.miner_hotkey = uncertainty.miner_hotkey
     AND reservation.stage = uncertainty.stage
     AND reservation.provider = uncertainty.provider
     AND reservation.operation_id = uncertainty.operation_id
     AND reservation.funding_source = uncertainty.funding_source
    JOIN public.lab_arena_runs AS runs
      ON runs.run_id = uncertainty.run_id
     AND runs.round_id = uncertainty.round_id
     AND runs.submission_id = uncertainty.submission_id
     AND runs.miner_hotkey = uncertainty.miner_hotkey
     AND runs.stage = uncertainty.stage
    WHERE uncertainty.entry_id = p_uncertain_entry_id
  ), FALSE);
$closed_score_dynamic_uncertainty$;
ALTER FUNCTION public.lab_arena__closed_score_dynamic_uncertainty_v1(BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__closed_score_dynamic_uncertainty_v1(BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__submission_kind_admission_spend_v1(
  p_submission_id TEXT,
  p_kind TEXT,
  p_locked_only BOOLEAN DEFAULT FALSE
)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $submission_kind_admission_spend$
  SELECT COALESCE(SUM(head.amount_microusd), 0)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity)
      ledger.entry_id, ledger.entry_kind, ledger.amount_microusd
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = p_kind
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN (
          'settlement', 'uncertain',
          CASE WHEN p_locked_only THEN NULL ELSE 'reservation' END,
          CASE WHEN p_locked_only THEN NULL ELSE 'dispatch' END
        )
    AND NOT (
      head.entry_kind = 'uncertain'
      AND public.lab_arena__closed_score_dynamic_uncertainty_v1(head.entry_id)
    );
$submission_kind_admission_spend$;
ALTER FUNCTION public.lab_arena__submission_kind_admission_spend_v1(TEXT,TEXT,BOOLEAN)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__submission_kind_admission_spend_v1(TEXT,TEXT,BOOLEAN)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__submission_kind_has_admission_uncertainty_v1(
  p_submission_id TEXT,
  p_kind TEXT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $submission_kind_has_admission_uncertainty$
  SELECT EXISTS (
    SELECT 1
    FROM (
      SELECT DISTINCT ON (ledger.call_identity)
        ledger.entry_id, ledger.entry_kind
      FROM public.lab_arena_ledger AS ledger
      JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
      WHERE ledger.submission_id = p_submission_id
        AND runs.kind = p_kind
        AND ledger.call_identity IS NOT NULL
      ORDER BY ledger.call_identity, ledger.entry_id DESC
    ) AS head
    WHERE head.entry_kind = 'uncertain'
      AND NOT public.lab_arena__closed_score_dynamic_uncertainty_v1(head.entry_id)
  );
$submission_kind_has_admission_uncertainty$;
ALTER FUNCTION public.lab_arena__submission_kind_has_admission_uncertainty_v1(TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__submission_kind_has_admission_uncertainty_v1(TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $closed_scoring_budget_admission$
DECLARE
  v_definition TEXT;
  v_old TEXT := 'v_spent := public.lab_arena__submission_kind_spend(';
  v_new TEXT := 'v_spent := public.lab_arena__submission_kind_admission_spend_v1(';
  v_locked TEXT := $marker$        IF v_locked_spend <= v_money_cap - p_amount_microusd THEN$marker$;
  v_locked_new TEXT := $replacement$        -- lab_arena_closed_scoring_reservation_admission
        v_locked_spend := public.lab_arena__submission_kind_admission_spend_v1(
          v_run.submission_id, v_run.kind, TRUE
        );
        IF v_locked_spend <= v_money_cap - p_amount_microusd THEN$replacement$;
  v_insert TEXT := $marker$    INSERT INTO public.lab_arena_ledger (
$marker$;
  v_insert_new TEXT := $replacement$    -- A retired uncertainty is still reported at its original amount. It no
    -- longer supplies evidence that this refusal came from unknown spend.
    IF v_reason = 'provider_cost_uncertain'
       AND NOT public.lab_arena__submission_kind_has_admission_uncertainty_v1(
             v_run.submission_id, v_run.kind
           ) THEN
      v_reason := 'money_cap';
    END IF;
    INSERT INTO public.lab_arena_ledger (
$replacement$;
  v_insert_at INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_reservation_admission') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'v_prior_miner_credential_refusal') = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena_provider_funding') = 0
     OR (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
        / pg_catalog.length(v_old) <> 1
     OR pg_catalog.strpos(v_definition, v_locked) = 0 THEN
    RAISE EXCEPTION 'lab_arena closed scoring budget admission shape unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  v_definition := pg_catalog.replace(v_definition, v_locked, v_locked_new);
  v_insert_at := pg_catalog.strpos(v_definition, v_insert);
  IF v_insert_at = 0 THEN
    RAISE EXCEPTION 'lab_arena closed scoring refusal shape unexpected';
  END IF;
  v_definition := pg_catalog.substr(v_definition, 1, v_insert_at - 1)
    || v_insert_new
    || pg_catalog.substr(v_definition, v_insert_at + pg_catalog.length(v_insert));
  EXECUTE v_definition;
END;
$closed_scoring_budget_admission$;

DO $closed_scoring_claim_admission$
DECLARE
  v_definition TEXT;
  v_start TEXT := $marker$    -- lab_arena_deepline_reconciliation_retry_deferral:$marker$;
  v_end TEXT := $marker$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey$marker$;
  v_start_at INTEGER;
  v_end_at INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_reservation_claim') = 0 THEN
    v_start_at := pg_catalog.strpos(v_definition, v_start);
    v_end_at := pg_catalog.strpos(v_definition, v_end);
    IF v_start_at = 0 OR v_end_at <= v_start_at THEN
      RAISE EXCEPTION 'lab_arena closed scoring claim shape unexpected';
    END IF;
    -- lab_arena_closed_scoring_reservation_claim: admission now decides
    -- whether an unknown charge can proceed. A ledger row cannot leave the
    -- scoring queue permanently unclaimable.
    EXECUTE pg_catalog.substr(v_definition, 1, v_start_at - 1)
      || E'    -- lab_arena_closed_scoring_reservation_claim\n'
      || pg_catalog.substr(v_definition, v_end_at);
  END IF;
END;
$closed_scoring_claim_admission$;

CREATE OR REPLACE FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.deepline_cost_reconciliation_schema.v1',
    'version', 248
  );
$schema$;
ALTER FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
