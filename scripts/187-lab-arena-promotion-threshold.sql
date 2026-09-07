-- 187-lab-arena-promotion-threshold.sql
-- Require a finalist to beat the daily organizer baseline by at least exactly
-- 1.0 point. NUMERIC arithmetic keeps the decimal boundary deterministic.

BEGIN;

DO $lab_arena_promotion_threshold$
DECLARE
  v_definition TEXT;
  v_missing_old TEXT := '(ranking ->> ''final_score'')::NUMERIC > v_baseline_score';
  v_missing_new TEXT := '(ranking ->> ''final_score'')::NUMERIC >= v_baseline_score + 1';
  v_winner_old TEXT := 'v_winner_score <= v_baseline_score OR EXISTS (';
  v_winner_new TEXT := 'v_winner_score < v_baseline_score + 1 OR EXISTS (';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_publication_baseline_guard_v1'
    AND procedure.pronargs = 0;

  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_guard_missing'
      USING ERRCODE = '55000';
  END IF;

  -- Idempotency: an already-updated definition needs no text replacement.
  IF pg_catalog.strpos(v_definition, v_missing_new) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_missing_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_publication_missing_winner_gate_unknown'
        USING ERRCODE = '55000';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_missing_old, v_missing_new);
  END IF;
  IF pg_catalog.strpos(v_definition, v_winner_new) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_winner_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_publication_winner_gate_unknown'
        USING ERRCODE = '55000';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_winner_old, v_winner_new);
  END IF;

  EXECUTE v_definition;
END;
$lab_arena_promotion_threshold$;

-- The existing trigger only checks transitions into published. Existing
-- published rows therefore remain valid historical records.

COMMIT;
