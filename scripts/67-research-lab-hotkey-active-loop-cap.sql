-- Research Lab: allow a configurable active auto-research loop cap per hotkey.
--
-- Deployment policy:
--   * Apply after scripts 43, 54, and 58.
--   * Safe to apply repeatedly.
--   * Keeps the queue admission guard atomic while raising the default
--     same-hotkey active-loop cap from 1 to 2.

BEGIN;

CREATE OR REPLACE FUNCTION public.guard_research_lab_queue_capacity()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    capacity_text TEXT;
    hotkey_capacity_text TEXT;
    stale_text TEXT;
    capacity INTEGER;
    hotkey_capacity INTEGER;
    stale_after_seconds INTEGER;
    cutoff TIMESTAMPTZ;
    v_miner_hotkey TEXT;
    active_count INTEGER;
    same_hotkey_count INTEGER;
    is_existing_run_requeue BOOLEAN;
BEGIN
    IF NEW.event_type <> 'queued' THEN
        RETURN NEW;
    END IF;

    capacity_text := NEW.event_doc->>'autoresearch_capacity';
    IF capacity_text IS NULL THEN
        RETURN NEW;
    END IF;
    IF capacity_text !~ '^[0-9]+$' THEN
        RAISE EXCEPTION
            'research_lab_queue_capacity_conflict: invalid capacity for run %',
            NEW.run_id
            USING ERRCODE = '23505';
    END IF;

    hotkey_capacity_text := COALESCE(NEW.event_doc->>'autoresearch_hotkey_capacity', '2');
    IF hotkey_capacity_text !~ '^[0-9]+$' THEN
        hotkey_capacity_text := '2';
    END IF;

    stale_text := COALESCE(NEW.event_doc->>'active_loop_stale_after_seconds', '7200');
    IF stale_text !~ '^[0-9]+$' THEN
        stale_text := '7200';
    END IF;

    capacity := capacity_text::INTEGER;
    hotkey_capacity := GREATEST(1, hotkey_capacity_text::INTEGER);
    stale_after_seconds := GREATEST(60, stale_text::INTEGER);
    cutoff := NOW() - (stale_after_seconds::TEXT || ' seconds')::INTERVAL;

    IF capacity <= 0 THEN
        RAISE EXCEPTION
            'research_lab_queue_capacity_conflict: capacity closed for run %',
            NEW.run_id
            USING ERRCODE = '23505';
    END IF;

    SELECT t.miner_hotkey
      INTO v_miner_hotkey
      FROM public.research_loop_tickets t
     WHERE t.ticket_id = NEW.ticket_id
     LIMIT 1;

    IF v_miner_hotkey IS NULL OR btrim(v_miner_hotkey) = '' THEN
        RAISE EXCEPTION
            'research_lab_queue_capacity_conflict: missing miner hotkey for run %',
            NEW.run_id
            USING ERRCODE = '23505';
    END IF;

    is_existing_run_requeue :=
        (NEW.event_doc ? 'resume_source')
        OR (NEW.event_doc ? 'recovering_worker_ref');

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_queue_capacity'),
        pg_catalog.hashtext(COALESCE(NEW.event_doc->>'autoresearch_capacity_policy', 'proxy_worker_capacity:v1'))
    );

    SELECT COUNT(*)
      INTO same_hotkey_count
      FROM public.research_loop_run_queue_current q
      JOIN public.research_loop_tickets t ON t.ticket_id = q.ticket_id
     WHERE q.run_id <> NEW.run_id
       AND (
           (
             is_existing_run_requeue
             AND q.current_queue_status IN ('queued', 'started')
             AND q.current_status_at >= cutoff
           )
           OR (
             NOT is_existing_run_requeue
             AND q.current_queue_status IN ('queued', 'started', 'paused')
             AND q.current_status_at >= cutoff
           )
       )
       AND btrim(t.miner_hotkey) = btrim(v_miner_hotkey);

    IF same_hotkey_count >= hotkey_capacity THEN
        RAISE EXCEPTION
            'research_lab_queue_hotkey_conflict: miner % active % cap %',
            v_miner_hotkey,
            same_hotkey_count,
            hotkey_capacity
            USING ERRCODE = '23505';
    END IF;

    SELECT COUNT(*)
     INTO active_count
      FROM public.research_loop_run_queue_current q
     WHERE q.run_id <> NEW.run_id
       AND (
           (
             is_existing_run_requeue
             AND q.current_queue_status IN ('queued', 'started')
             AND q.current_status_at >= cutoff
           )
           OR (
             NOT is_existing_run_requeue
             AND q.current_queue_status IN ('queued', 'started', 'paused')
             AND q.current_status_at >= cutoff
           )
       );

    IF active_count >= capacity THEN
        RAISE EXCEPTION
            'research_lab_queue_capacity_conflict: active % capacity %',
            active_count,
            capacity
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_queue_capacity()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_queue_capacity()
    TO service_role;

COMMENT ON FUNCTION public.guard_research_lab_queue_capacity() IS
    'Serializes Research Lab queued event admission; maintenance requeues ignore paused rows while enforcing a configurable same-hotkey active-loop cap.';

COMMIT;

-- Verification:
-- SELECT pg_get_functiondef('public.guard_research_lab_queue_capacity()'::regprocedure);
