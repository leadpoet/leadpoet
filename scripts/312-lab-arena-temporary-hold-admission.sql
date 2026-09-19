-- A temporary reservation is not confirmed exhaustion. Keep its liability,
-- but use the existing passive admission wait until exact billing arrives.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $temporary_hold_admission$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$    IF (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0 THEN$old$;
  v_new TEXT := $new$    -- lab_arena_temporary_hold_admission
    IF (v_icp_cost ->> 'settled_microusd')::BIGINT >=
         (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
    ELSIF (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0
       OR (v_icp_cost ->> 'settled_microusd')::BIGINT
          + (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT >=
          (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN$new$;
  v_reply_old TEXT := $old$        'status', 'budget_busy', 'idempotent', FALSE,
        'call_identity', p_call_identity, 'lease_expires_at', v_expires$old$;
  v_reply_new TEXT := $new$        'status', 'budget_busy', 'idempotent', FALSE,
        'reason', CASE WHEN (v_icp_cost ->> 'paid_inflight_calls')::BIGINT > 0
          THEN 'provider_calls_inflight' ELSE 'provider_cost_uncertain' END,
        'call_identity', p_call_identity, 'lease_expires_at', v_expires$new$;
  v_exhausted TEXT := $old$    ELSIF (v_icp_cost ->> 'settled_microusd')::BIGINT
          + (v_icp_cost ->> 'reserved_or_uncertain_microusd')::BIGINT >=
          (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT THEN
      v_reason := 'money_cap';
$old$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_temporary_hold_admission') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(v_definition, v_reply_old) = 0
       OR pg_catalog.strpos(v_definition, v_exhausted) = 0 THEN
      RAISE EXCEPTION 'temporary hold admission shape unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_definition := pg_catalog.replace(v_definition, v_exhausted, '');
    EXECUTE pg_catalog.replace(v_definition, v_reply_old, v_reply_new);
  END IF;
END;
$temporary_hold_admission$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
