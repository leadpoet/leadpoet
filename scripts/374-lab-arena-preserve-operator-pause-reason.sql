-- A canonical restart must not replace an existing operator hold reason.
-- The release/abort RPCs already preserve the reason when operator_paused;
-- repair only the acquire assignment, keeping all guard locks and claims.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $preserve_operator_pause_reason$
DECLARE
  v_definition TEXT;
  v_old CONSTANT TEXT := 'pause_reason = ''canonical_restart_guard'',';
  v_new CONSTANT TEXT :=
    'pause_reason = CASE WHEN operator_paused THEN pause_reason ELSE ''canonical_restart_guard'' END,';
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  PERFORM 1 FROM public.lab_arena_restart_claim_control
    WHERE singleton FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_restart_claim_control_missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_acquire_restart_guard_v1(text,text,bigint,integer,text,text,text)'
  )) INTO v_definition;
  IF v_definition IS NULL
     OR pg_catalog.strpos(v_definition, 'lab-arena-claim-control') = 0
     OR pg_catalog.strpos(v_definition, 'FOR UPDATE') = 0
     OR pg_catalog.strpos(v_definition, 'captured_leases = v_snapshot') = 0 THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_acquire_shape_unexpected';
  END IF;
  IF pg_catalog.strpos(v_definition, v_new) > 0
     AND pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RETURN; -- Exact, harmless reapply.
  END IF;
  IF pg_catalog.strpos(v_definition, v_old) = 0
     OR pg_catalog.strpos(v_definition, v_new) > 0 THEN
    RAISE EXCEPTION 'lab_arena_restart_guard_acquire_reason_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$preserve_operator_pause_reason$;

COMMIT;
