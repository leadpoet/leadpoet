-- Research Lab egress reduction: atomic single-owner hosted-run claim.
--
-- _next_queued_run had every hosted worker download the same queued-run window
-- and pick locally by a client-side hash partition, then read-write-read to
-- append the started event. This RPC selects the next queued run by priority
-- (oldest-first within a priority) INSIDE Postgres, reserves it with a
-- short-TTL claim under an advisory lock, and returns only the columns the
-- claim path consumes. Concurrent hosted workers therefore each get a DIFFERENT
-- run in one call -- the client hash partition is no longer needed. The
-- started-event append stays in Python (anchored_hash parity), backstopped by
-- the existing guard_research_lab_run_claim trigger (script 42). A dead worker's
-- claim expires and the still-queued run is re-claimable; a claim is ignored
-- once the run's status advances past it (claimed_at < current_status_at).

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_loop_run_claim (
    run_id     UUID        PRIMARY KEY,
    holder_ref TEXT        NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE public.research_loop_run_claim ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_loop_run_claim FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.research_loop_run_claim TO service_role;
DROP POLICY IF EXISTS service_role_all ON public.research_loop_run_claim;
CREATE POLICY service_role_all ON public.research_loop_run_claim
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE OR REPLACE FUNCTION public.claim_next_research_loop_run(
    p_holder_ref      TEXT,
    p_ttl_seconds     INTEGER,
    p_allowed_run_ids UUID[]
)
RETURNS TABLE(
    run_id            UUID,
    ticket_id         UUID,
    queue_priority    INTEGER,
    current_event_hash TEXT,
    current_status_at TIMESTAMPTZ
)
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = ''
AS $$
-- OUT column names (run_id, ticket_id, ...) match table columns; resolve
-- ambiguous references in the body to the columns.
#variable_conflict use_column
DECLARE
    v_now     TIMESTAMPTZ := pg_catalog.now();
    v_expires TIMESTAMPTZ;
    v_id      UUID;
BEGIN
    IF p_holder_ref IS NULL OR p_ttl_seconds IS NULL
       OR p_ttl_seconds <= 0 OR p_ttl_seconds > 3600 THEN
        RAISE EXCEPTION 'run claim arguments are invalid' USING ERRCODE = '22023';
    END IF;
    v_expires := v_now + pg_catalog.make_interval(secs => p_ttl_seconds);

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_loop_run_claim'));

    SELECT q.run_id INTO v_id
    FROM public.research_loop_run_queue_current q
    WHERE q.current_queue_status = 'queued'
      AND (
          COALESCE(pg_catalog.cardinality(p_allowed_run_ids), 0) = 0
          OR q.run_id = ANY (p_allowed_run_ids)
      )
      AND NOT EXISTS (
          SELECT 1 FROM public.research_loop_run_claim cl
          WHERE cl.run_id = q.run_id
            AND cl.expires_at > v_now
            AND cl.claimed_at >= q.current_status_at
      )
    ORDER BY q.queue_priority ASC, q.current_status_at ASC
    LIMIT 1;

    IF v_id IS NULL THEN
        RETURN;
    END IF;

    INSERT INTO public.research_loop_run_claim AS cl
        (run_id, holder_ref, claimed_at, expires_at)
    VALUES (v_id, p_holder_ref, v_now, v_expires)
    ON CONFLICT (run_id) DO UPDATE
        SET holder_ref = EXCLUDED.holder_ref,
            claimed_at = EXCLUDED.claimed_at,
            expires_at = EXCLUDED.expires_at;

    RETURN QUERY
    SELECT q.run_id, q.ticket_id, q.queue_priority, q.current_event_hash,
           q.current_status_at
    FROM public.research_loop_run_queue_current q
    WHERE q.run_id = v_id;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_next_research_loop_run(TEXT, INTEGER, UUID[])
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.claim_next_research_loop_run(TEXT, INTEGER, UUID[])
    TO service_role;

COMMIT;
