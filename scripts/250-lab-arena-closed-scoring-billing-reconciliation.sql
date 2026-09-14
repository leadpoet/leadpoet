-- Keep exact billing reconciliation alive after scoring has finished. Only
-- closed, interrupted judge calls qualify. Published scores, signed cost
-- snapshots, sourcing eligibility and reward decisions remain immutable.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $closed_billing_list$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$            AND rounds.status NOT IN ('open', 'published')$old$;
  v_new TEXT := $new$            -- lab_arena_closed_scoring_billing_reconciliation
            AND (
              rounds.status NOT IN ('open', 'published')
              OR (
                rounds.status = 'published'
                AND rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
                    = 'successful_calls_v1'
                AND public.lab_arena__closed_score_dynamic_uncertainty_v1(
                      uncertainty.entry_id
                    )
              )
            )$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_billing_reconciliation') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'closed scoring billing list shape unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$closed_billing_list$;

DO $closed_billing_settle$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$  IF v_round.status IN ('open', 'published') THEN$old$;
  v_new TEXT := $new$  IF v_round.status = 'open' THEN$new$;
  v_head_old TEXT := $old$  v_head := public.lab_arena__ledger_head(p_call_identity);$old$;
  v_head_new TEXT := $new$  v_head := public.lab_arena__ledger_head(p_call_identity);
  -- lab_arena_closed_scoring_billing_reconciliation: a published result is an
  -- as-of snapshot. An exact later judge bill changes only the audit ledger.
  IF v_round.status = 'published' AND NOT COALESCE((
       v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_v1'
       AND public.lab_arena__closed_score_dynamic_uncertainty_v1(
         CASE
           WHEN v_head.entry_kind = 'uncertain' THEN v_head.entry_id
           WHEN v_head.entry_kind = 'settlement'
             AND v_head.entry_doc ->> 'deepline_delayed_reconciliation' = 'true'
             AND v_head.entry_doc ->> 'reconciled_uncertainty_entry_id' ~ '^[0-9]{1,18}$'
           THEN (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
           ELSE NULL
         END
       )
     ), FALSE) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_billing_reconciliation') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(v_definition, v_head_old) = 0 THEN
      RAISE EXCEPTION 'closed scoring billing settlement shape unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    EXECUTE pg_catalog.replace(v_definition, v_head_old, v_head_new);
  END IF;
END;
$closed_billing_settle$;

-- One round-scoped billing candidate per driver tick. The cursor wraps so a
-- permanently missing receipt cannot starve a newer or older exact receipt.
CREATE OR REPLACE FUNCTION public.lab_arena_next_closed_deepline_reconciliation_v1(
  p_mode TEXT, p_network_name TEXT, p_netuid INTEGER,
  p_round_id TEXT, p_after_entry_id BIGINT
)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $next_closed$
  SELECT COALESCE((
    SELECT pg_catalog.jsonb_build_object(
      'status', 'ok', 'uncertain_entry_id', uncertainty.entry_id,
      'round_id', uncertainty.round_id, 'run_id', uncertainty.run_id
    )
    FROM public.lab_arena_ledger AS uncertainty
    JOIN public.lab_arena_rounds AS rounds
      ON rounds.round_id = uncertainty.round_id
    WHERE uncertainty.entry_kind = 'uncertain'
      AND uncertainty.provider = 'deepline'
      AND rounds.status IN ('cancelled', 'published')
      AND rounds.configuration_doc ->> 'mode' = p_mode
      AND rounds.configuration_doc ->> 'network_name' = p_network_name
      AND rounds.configuration_doc ->> 'netuid' = p_netuid::TEXT
      AND (COALESCE(p_round_id, '') = '' OR uncertainty.round_id = p_round_id)
      AND (
        rounds.status = 'cancelled'
        OR rounds.configuration_doc ->> 'sourcing_cost_eligibility_policy'
             = 'successful_calls_v1'
      )
      AND public.lab_arena__closed_score_dynamic_uncertainty_v1(uncertainty.entry_id)
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_ledger AS later
        WHERE later.call_identity = uncertainty.call_identity
          AND later.entry_id > uncertainty.entry_id
      )
    ORDER BY (uncertainty.entry_id <= COALESCE(p_after_entry_id, 0)),
             uncertainty.entry_id
    LIMIT 1
  ), pg_catalog.jsonb_build_object('status', 'none'));
$next_closed$;
ALTER FUNCTION public.lab_arena_next_closed_deepline_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_next_closed_deepline_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_next_closed_deepline_reconciliation_v1(TEXT,TEXT,INTEGER,TEXT,BIGINT)
  TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
