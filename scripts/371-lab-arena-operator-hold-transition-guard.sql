-- Operator claim holds must also stop deadline-driven round progression.
-- An old driver can still call a transition while claims are paused, so the
-- database is the final guard. Accepted completions and intake stay available.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $prerequisites$
BEGIN
  IF pg_catalog.to_regclass('public.lab_arena_restart_claim_control') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_rounds') IS NULL
     OR NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='lab_arena_owner')
     OR NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='lab_arena_service') THEN
    RAISE EXCEPTION 'apply Arena restart control before migration 371';
  END IF;
END;
$prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_operator_hold_active_v1()
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $operator_hold_active$
DECLARE
  v_paused BOOLEAN;
BEGIN
  SELECT operator_paused OR guard_commitment <> '' INTO v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_claim_control_missing' USING ERRCODE='55000';
  END IF;
  RETURN v_paused;
END;
$operator_hold_active$;
ALTER FUNCTION public.lab_arena_operator_hold_active_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_operator_hold_active_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_operator_hold_active_v1()
  TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_operator_hold_transition_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $operator_hold_transition_guard$
DECLARE
  v_old_rank INTEGER;
  v_new_rank INTEGER;
  v_paused BOOLEAN;
BEGIN
  IF OLD.status IS NOT DISTINCT FROM NEW.status THEN
    RETURN NEW;
  END IF;
  -- Cutoff must still freeze the prior-day bank, scorer, and participants.
  -- This transition creates no execution or scoring assignment.
  IF OLD.status = 'open' AND NEW.status = 'committed' THEN
    RETURN NEW;
  END IF;
  v_old_rank := CASE OLD.status
    WHEN 'open' THEN 0 WHEN 'committed' THEN 1
    WHEN 'stage1' THEN 2 WHEN 'stage1_closed' THEN 3
    WHEN 'stage1_scoring' THEN 4 WHEN 'stage1_judged' THEN 5
    WHEN 'stage1_scored' THEN 6 WHEN 'stage2' THEN 7
    WHEN 'stage2_closed' THEN 8 WHEN 'stage2_scoring' THEN 9
    WHEN 'stage2_judged' THEN 10 WHEN 'scored' THEN 11
    WHEN 'published' THEN 12 WHEN 'cancelled' THEN 12
  END;
  v_new_rank := CASE NEW.status
    WHEN 'open' THEN 0 WHEN 'committed' THEN 1
    WHEN 'stage1' THEN 2 WHEN 'stage1_closed' THEN 3
    WHEN 'stage1_scoring' THEN 4 WHEN 'stage1_judged' THEN 5
    WHEN 'stage1_scored' THEN 6 WHEN 'stage2' THEN 7
    WHEN 'stage2_closed' THEN 8 WHEN 'stage2_scoring' THEN 9
    WHEN 'stage2_judged' THEN 10 WHEN 'scored' THEN 11
    WHEN 'published' THEN 12 WHEN 'cancelled' THEN 12
  END;
  IF v_old_rank IS NULL OR v_new_rank IS NULL THEN
    RAISE EXCEPTION 'lab_arena_round_status_unknown' USING ERRCODE='22023';
  END IF;
  IF v_new_rank <= v_old_rank THEN
    RETURN NEW; -- An exact guarded recovery may rewind a damaged round.
  END IF;
  -- The same transaction advisory lock serializes with the operator's hold
  -- writer. If an old transition already holds the round row, a deadlock
  -- aborts one transaction rather than allowing a partial transition.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT operator_paused OR guard_commitment <> '' INTO v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF COALESCE(v_paused, TRUE) THEN
    RAISE EXCEPTION 'lab_arena_round_progression_paused' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$operator_hold_transition_guard$;
ALTER FUNCTION public.lab_arena_operator_hold_transition_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_operator_hold_transition_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_rounds_operator_hold_transition_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_rounds_operator_hold_transition_guard
BEFORE UPDATE OF status ON public.lab_arena_rounds
FOR EACH ROW EXECUTE FUNCTION public.lab_arena_operator_hold_transition_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_operator_hold_score_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $operator_hold_score_guard$
DECLARE
  v_paused BOOLEAN;
BEGIN
  IF NEW.per_icp_score IS NULL
     OR NEW.per_icp_score IS NOT DISTINCT FROM OLD.per_icp_score THEN
    RETURN NEW;
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT operator_paused OR guard_commitment <> '' INTO v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF COALESCE(v_paused, TRUE) THEN
    RAISE EXCEPTION 'lab_arena_round_progression_paused' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$operator_hold_score_guard$;
ALTER FUNCTION public.lab_arena_operator_hold_score_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_operator_hold_score_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DROP TRIGGER IF EXISTS lab_arena_runs_operator_hold_score_guard
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_runs_operator_hold_score_guard
BEFORE UPDATE OF per_icp_score ON public.lab_arena_runs
FOR EACH ROW EXECUTE FUNCTION public.lab_arena_operator_hold_score_guard_v1();

NOTIFY pgrst, 'reload schema';
COMMIT;
