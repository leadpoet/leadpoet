-- Decode the trusted settlement success JSON once per ledger head.
-- Keep cancellation precedence, delayed reconciliation, and all cost filters.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '30s';

DO $settlement_410_call$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
    INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)');
  IF v_definition IS NULL
     OR v_identity ->> 0 IS DISTINCT FROM 'lab_arena_owner'
     OR v_identity ->> 2 IS DISTINCT FROM 'true'
     OR v_identity ->> 3 IS DISTINCT FROM 's' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_call_security_shape_changed';
  END IF;
  IF pg_catalog.md5(v_definition) = '2675cc09a685635c6a910df417312c1b' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> '82e3bc799dc185d7034771ff46542985' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_call_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    $settlement_410_call_old$        WHEN ledger.entry_kind = 'settlement'
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
$settlement_410_call_old$,
    $settlement_410_call_new$        WHEN ledger.entry_kind = 'settlement' THEN COALESCE(
          CASE ledger.terminal_response -> 'call_succeeded'
            WHEN 'true'::JSONB THEN TRUE
            WHEN 'false'::JSONB THEN FALSE
          END,
          CASE WHEN ledger.entry_doc ->> 'openrouter_delayed_reconciliation'
                    = 'true' THEN (
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
        ) END
        )
$settlement_410_call_new$);
  IF pg_catalog.md5(v_updated) <> '2675cc09a685635c6a910df417312c1b' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_call_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))) <> '2675cc09a685635c6a910df417312c1b'
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
         FROM pg_catalog.pg_proc AS p
         JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
         WHERE p.oid = pg_catalog.to_regprocedure('public.lab_arena__successful_call_cost_state(text,text,text)'))
       IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'lab_arena_cost_410_call_readback_changed';
  END IF;
END;
$settlement_410_call$;

DO $settlement_410_icp$
DECLARE
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
    INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)');
  IF v_definition IS NULL
     OR v_identity ->> 0 IS DISTINCT FROM 'lab_arena_owner'
     OR v_identity ->> 2 IS DISTINCT FROM 'true'
     OR v_identity ->> 3 IS DISTINCT FROM 's' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_icp_security_shape_changed';
  END IF;
  IF pg_catalog.md5(v_definition) = '0ea743f538c1f3f704c903ba7f986a7d' THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_definition) <> 'f220f32be89411ddb8af6deda805c21e' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_icp_preimage_changed';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    $settlement_410_icp_old$        WHEN ledger.entry_kind = 'settlement'
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
$settlement_410_icp_old$,
    $settlement_410_icp_new$        WHEN ledger.entry_kind = 'settlement' THEN COALESCE(
          CASE ledger.terminal_response -> 'call_succeeded'
            WHEN 'true'::JSONB THEN TRUE
            WHEN 'false'::JSONB THEN FALSE
          END,
          CASE WHEN ledger.entry_doc ->> 'openrouter_delayed_reconciliation'
                    = 'true' THEN (
          SELECT CASE WHEN pg_catalog.jsonb_typeof(
            prior.entry_doc #> '{call,call_succeeded}') = 'boolean'
            THEN (prior.entry_doc #>> '{call,call_succeeded}')::BOOLEAN END
          FROM public.lab_arena_ledger AS prior
          WHERE prior.entry_id =
            (ledger.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
            AND prior.entry_kind = 'uncertain'
        ) END
        )
$settlement_410_icp_new$);
  IF pg_catalog.md5(v_updated) <> '0ea743f538c1f3f704c903ba7f986a7d' THEN
    RAISE EXCEPTION 'lab_arena_cost_410_icp_postimage_invalid';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))) <> '0ea743f538c1f3f704c903ba7f986a7d'
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
         FROM pg_catalog.pg_proc AS p
         JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
         WHERE p.oid = pg_catalog.to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)'))
       IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'lab_arena_cost_410_icp_readback_changed';
  END IF;
END;
$settlement_410_icp$;

NOTIFY pgrst, 'reload schema';
COMMIT;
