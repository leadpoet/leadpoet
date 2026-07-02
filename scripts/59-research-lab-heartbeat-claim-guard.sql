-- Research Lab: allow same-worker queue heartbeats through the run claim guard.
--
-- Deployment policy:
--   * Apply after scripts 42 and 54.
--   * Safe to apply repeatedly.
--   * Ship this trigger change BEFORE the worker-side heartbeat claim-lost
--     abort. For unchanged workers this is a strict no-op: the only two
--     writers that ever insert `started` are the claim (which only fires when
--     the latest event is `queued`) and the heartbeat (which the previous
--     guard rejected unconditionally), and both always set `worker_ref`.
--   * Fixes: the scripts/42 guard rejected any `started` insert whose latest
--     event was not `queued`, so every hosted-worker queue heartbeat failed
--     silently. Runs in phases longer than the stale window with no loop
--     events (docker builds, threaded work) were requeued while still
--     running, and the frozen queue `current_status_at` broke the scripts/54
--     capacity/hotkey guard's active-run counting.
--   * Cross-worker fencing is preserved: a `started` insert is additionally
--     allowed ONLY when the run's latest event is `started` AND carries the
--     same non-null `worker_ref` (the claim owner refreshing its own claim).
--     After another worker re-claims, the superseded worker's heartbeat
--     still conflicts.

BEGIN;

CREATE OR REPLACE FUNCTION public.guard_research_lab_run_claim()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
    latest_worker_ref TEXT;
BEGIN
    IF NEW.event_type <> 'started' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_run_claim'),
        pg_catalog.hashtext(NEW.run_id::TEXT)
    );

    SELECT e.event_type, e.worker_ref
      INTO latest_status, latest_worker_ref
      FROM public.research_loop_run_queue_events e
     WHERE e.run_id = NEW.run_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status IS DISTINCT FROM 'queued' THEN
        -- Same-worker heartbeat: the current claim owner may refresh its own
        -- `started` claim. Requires the same non-null worker_ref on both
        -- sides so a superseded worker's heartbeat still conflicts after a
        -- re-claim and a worker_ref-less event never grants heartbeat rights.
        IF latest_status = 'started'
           AND NEW.worker_ref IS NOT NULL
           AND latest_worker_ref IS NOT NULL
           AND NEW.worker_ref = latest_worker_ref THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION
            'research_lab_run_claim_conflict: run % latest status %',
            NEW.run_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_run_claim()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_run_claim()
    TO service_role;

COMMENT ON FUNCTION public.guard_research_lab_run_claim() IS
    'Serializes hosted Research Lab run claims; permits started when queued is latest, plus same-worker_ref heartbeat refresh while started is latest.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT pg_get_functiondef('public.guard_research_lab_run_claim()'::regprocedure);
--
--   SELECT tgname
--   FROM pg_trigger
--   WHERE tgname = 'guard_research_loop_run_claim_insert';
--
-- Verify claims still work (a fresh queued run must still be claimable, a
-- second claim by a different worker must still conflict) before shipping the
-- worker-side heartbeat claim-lost abort.
--
-- Instant rollback (restores the scripts/42 guard verbatim):
--
--   CREATE OR REPLACE FUNCTION public.guard_research_lab_run_claim()
--   RETURNS trigger
--   LANGUAGE plpgsql
--   SET search_path = ''
--   AS $$
--   DECLARE
--       latest_status TEXT;
--   BEGIN
--       IF NEW.event_type <> 'started' THEN
--           RETURN NEW;
--       END IF;
--
--       PERFORM pg_catalog.pg_advisory_xact_lock(
--           pg_catalog.hashtext('research_lab_run_claim'),
--           pg_catalog.hashtext(NEW.run_id::TEXT)
--       );
--
--       SELECT e.event_type
--         INTO latest_status
--         FROM public.research_loop_run_queue_events e
--        WHERE e.run_id = NEW.run_id
--        ORDER BY e.seq DESC, e.created_at DESC
--        LIMIT 1;
--
--       IF latest_status IS DISTINCT FROM 'queued' THEN
--           RAISE EXCEPTION
--               'research_lab_run_claim_conflict: run % latest status %',
--               NEW.run_id,
--               COALESCE(latest_status, '<none>')
--               USING ERRCODE = '23505';
--       END IF;
--
--       RETURN NEW;
--   END;
--   $$;
