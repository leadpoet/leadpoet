-- Atomic, idempotent resume for later Research Lab runs parked for credit.
--
-- The signed miner resume route can be invoked concurrently or retried after
-- a committed response is lost.  Serialize on the exact run and append one
-- deterministic queued event only while the caller's paused head is current.

BEGIN;

CREATE OR REPLACE FUNCTION public.resume_research_lab_credit_blocked_run_v1(
    p_run_id              UUID,
    p_ticket_id           UUID,
    p_expected_event_seq  INTEGER,
    p_expected_event_hash TEXT,
    p_event_id            UUID,
    p_anchored_hash       TEXT,
    p_queue_priority      INTEGER,
    p_worker_ref          TEXT,
    p_reason              TEXT,
    p_event_doc           JSONB
)
RETURNS SETOF public.research_loop_run_queue_events
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    existing public.research_loop_run_queue_events%ROWTYPE;
    head     public.research_loop_run_queue_events%ROWTYPE;
    inserted public.research_loop_run_queue_events%ROWTYPE;
BEGIN
    IF p_run_id IS NULL
       OR p_ticket_id IS NULL
       OR p_event_id IS NULL
       OR p_expected_event_seq < 0
       OR p_expected_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_anchored_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_reason IS DISTINCT FROM 'credit_topup_resume'
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_event_doc->>'schema_version' IS DISTINCT FROM '1.0'
       OR p_event_doc->>'resume_source' IS DISTINCT FROM 'miner_credit_topup_resume'
       OR p_event_doc->>'previous_event_hash' IS DISTINCT FROM p_expected_event_hash
    THEN
        RAISE EXCEPTION 'research_lab_credit_resume_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_credit_resume'),
        pg_catalog.hashtext(p_run_id::TEXT)
    );

    SELECT e.*
      INTO existing
      FROM public.research_loop_run_queue_events AS e
     WHERE e.event_id = p_event_id;

    IF FOUND THEN
        IF existing.run_id IS DISTINCT FROM p_run_id
           OR existing.ticket_id IS DISTINCT FROM p_ticket_id
           OR existing.seq IS DISTINCT FROM p_expected_event_seq + 1
           OR existing.event_type IS DISTINCT FROM 'queued'
           OR existing.queue_priority IS DISTINCT FROM p_queue_priority
           OR existing.worker_ref IS DISTINCT FROM p_worker_ref
           OR existing.reason IS DISTINCT FROM p_reason
           OR existing.anchored_hash IS DISTINCT FROM p_anchored_hash
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_credit_resume_replay_differs'
                USING ERRCODE = '23505';
        END IF;
        RETURN NEXT existing;
        RETURN;
    END IF;

    SELECT e.*
      INTO head
      FROM public.research_loop_run_queue_events AS e
     WHERE e.run_id = p_run_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF NOT FOUND
       OR head.ticket_id IS DISTINCT FROM p_ticket_id
       OR head.seq IS DISTINCT FROM p_expected_event_seq
       OR head.anchored_hash IS DISTINCT FROM p_expected_event_hash
       OR head.event_type IS DISTINCT FROM 'paused'
       OR head.reason IS DISTINCT FROM 'blocked_for_credit'
    THEN
        RAISE EXCEPTION 'research_lab_credit_resume_head_conflict'
            USING ERRCODE = '23505';
    END IF;

    INSERT INTO public.research_loop_run_queue_events (
        event_id,
        schema_version,
        run_id,
        ticket_id,
        seq,
        event_type,
        queue_priority,
        worker_ref,
        reason,
        anchored_hash,
        event_doc
    ) VALUES (
        p_event_id,
        '1.0',
        p_run_id,
        p_ticket_id,
        p_expected_event_seq + 1,
        'queued',
        p_queue_priority,
        p_worker_ref,
        p_reason,
        p_anchored_hash,
        p_event_doc
    )
    RETURNING * INTO inserted;

    RETURN NEXT inserted;
END;
$$;

REVOKE ALL ON FUNCTION public.resume_research_lab_credit_blocked_run_v1(
    UUID, UUID, INTEGER, TEXT, UUID, TEXT, INTEGER, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.resume_research_lab_credit_blocked_run_v1(
    UUID, UUID, INTEGER, TEXT, UUID, TEXT, INTEGER, TEXT, TEXT, JSONB
) TO service_role;

COMMENT ON FUNCTION public.resume_research_lab_credit_blocked_run_v1(
    UUID, UUID, INTEGER, TEXT, UUID, TEXT, INTEGER, TEXT, TEXT, JSONB
) IS
    'Atomically appends or replays one expected-head credit-top-up resume event for a paused Research Lab run.';

NOTIFY pgrst, 'reload schema';

COMMIT;
