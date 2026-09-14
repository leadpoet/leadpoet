-- Public Deepline API keys receive a server-generated job identity. Preserve
-- the broker's pre-dispatch binding, but reconcile the retained native receipt
-- when one was received. No existing ledger row or round state is rewritten.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $native_list$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$reservation.entry_doc ->> 'deepline_request_id' AS request_id,$old$;
  v_new TEXT := $new$-- lab_arena_deepline_native_billing_identity
            COALESCE(uncertainty.entry_doc #>> '{call,deepline_job_id}',
                     reservation.entry_doc ->> 'deepline_request_id') AS request_id,$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_native_billing_identity') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_native_billing_list_shape_unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$native_list$;

DO $native_settle$
DECLARE
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_native_billing_identity') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_interrupted_cost_reconciliation') = 0 THEN
    RAISE EXCEPTION 'apply migration 246 before migration 247';
  END IF;
  v_old := $old$     OR COALESCE(p_request_id, '') !~
        '^ctx-tool-[0-9a-f]{32}$'
     OR p_request_id IS DISTINCT FROM
        'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32)
$old$;
  v_new := $new$     OR COALESCE(p_request_id, '') !~
        '^(ctx-tool-[0-9a-f]{32}|[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64})$'
     OR (p_request_id LIKE 'ctx-tool-%' AND p_request_id IS DISTINCT FROM
         'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32))
$new$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_native_billing_input_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);

  v_old := $old$  IF v_head.entry_id IS DISTINCT FROM p_uncertain_entry_id
$old$;
  v_new := $new$  -- lab_arena_deepline_native_billing_identity: the caller cannot choose
  -- a different provider charge. Its id must be the immutable worker receipt.
  IF p_request_id IS DISTINCT FROM COALESCE(
       v_head.entry_doc #>> '{call,deepline_job_id}',
       v_head.entry_doc #>> '{call,deepline_request_id}',
       'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32)
     ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  IF v_head.entry_id IS DISTINCT FROM p_uncertain_entry_id
$new$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_native_billing_head_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);

  v_old := $old$            'deepline_request_id', p_request_id,$old$;
  v_new := $new$            'deepline_request_id',
              'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32),$new$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_native_billing_binding_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);

  v_old := $old$     OR v_reservation.entry_doc ->> 'deepline_request_id'
        IS DISTINCT FROM p_request_id
$old$;
  v_new := $new$     OR v_reservation.entry_doc ->> 'deepline_request_id'
        IS DISTINCT FROM 'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32)
$new$;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_native_billing_reservation_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$native_settle$;

CREATE OR REPLACE FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.deepline_cost_reconciliation_schema.v1',
    'version', 247
  );
$schema$;
ALTER FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
