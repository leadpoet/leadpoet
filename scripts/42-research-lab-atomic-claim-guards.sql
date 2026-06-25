-- Research Lab: DB-side atomic claim guards for fleet workers.
--
-- Deployment policy:
--   * Apply after scripts 29 and 33.
--   * This does not change schema shape or enable any live writes.
--   * It prevents double-claim races caused by client-side event seq allocation.

BEGIN;

CREATE OR REPLACE FUNCTION public.guard_research_lab_run_claim()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
BEGIN
    IF NEW.event_type <> 'started' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_run_claim'),
        pg_catalog.hashtext(NEW.run_id::TEXT)
    );

    SELECT e.event_type
      INTO latest_status
      FROM public.research_loop_run_queue_events e
     WHERE e.run_id = NEW.run_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status IS DISTINCT FROM 'queued' THEN
        RAISE EXCEPTION
            'research_lab_run_claim_conflict: run % latest status %',
            NEW.run_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.guard_research_lab_candidate_claim()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
BEGIN
    IF NEW.candidate_status <> 'assigned' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_candidate_claim'),
        pg_catalog.hashtext(NEW.candidate_id::TEXT)
    );

    SELECT e.candidate_status
      INTO latest_status
      FROM public.research_lab_candidate_evaluation_events e
     WHERE e.candidate_id = NEW.candidate_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status IS DISTINCT FROM 'queued' THEN
        RAISE EXCEPTION
            'research_lab_candidate_claim_conflict: candidate % latest status %',
            NEW.candidate_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.guard_research_lab_credit_consume()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
BEGIN
    IF NEW.credit_status <> 'consumed' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_credit_consume'),
        pg_catalog.hashtext(NEW.credit_id::TEXT)
    );

    SELECT e.credit_status
      INTO latest_status
      FROM public.research_loop_start_credit_events e
     WHERE e.credit_id = NEW.credit_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status IS DISTINCT FROM 'available' THEN
        RAISE EXCEPTION
            'research_lab_credit_consume_conflict: credit % latest status %',
            NEW.credit_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_run_claim()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.guard_research_lab_candidate_claim()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.guard_research_lab_credit_consume()
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.guard_research_lab_run_claim()
    TO service_role;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_candidate_claim()
    TO service_role;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_credit_consume()
    TO service_role;

DROP TRIGGER IF EXISTS guard_research_loop_run_claim_insert
    ON public.research_loop_run_queue_events;
CREATE TRIGGER guard_research_loop_run_claim_insert
    BEFORE INSERT ON public.research_loop_run_queue_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_run_claim();

DROP TRIGGER IF EXISTS guard_research_lab_candidate_claim_insert
    ON public.research_lab_candidate_evaluation_events;
CREATE TRIGGER guard_research_lab_candidate_claim_insert
    BEFORE INSERT ON public.research_lab_candidate_evaluation_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_candidate_claim();

DROP TRIGGER IF EXISTS guard_research_loop_start_credit_consume_insert
    ON public.research_loop_start_credit_events;
CREATE TRIGGER guard_research_loop_start_credit_consume_insert
    BEFORE INSERT ON public.research_loop_start_credit_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_credit_consume();

COMMENT ON FUNCTION public.guard_research_lab_run_claim() IS
    'Serializes hosted Research Lab run claims and permits started only when queued is latest.';
COMMENT ON FUNCTION public.guard_research_lab_candidate_claim() IS
    'Serializes Research Lab candidate scoring claims and permits assigned only when queued is latest.';
COMMENT ON FUNCTION public.guard_research_lab_credit_consume() IS
    'Serializes Research Lab loop-start credit consumption and permits consumed only when available is latest.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT tgname
--   FROM pg_trigger
--   WHERE tgname IN (
--     'guard_research_loop_run_claim_insert',
--     'guard_research_lab_candidate_claim_insert',
--     'guard_research_loop_start_credit_consume_insert'
--   )
--   ORDER BY tgname;
