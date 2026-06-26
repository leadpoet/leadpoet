-- Research Lab: operator maintenance pause/resume controls.
--
-- Deployment policy:
--   * Apply after scripts 29, 34, 42, and 43.
--   * Adds append-only gateway control events plus paused/checkpoint states.
--   * Lets operators stop new paid loops, checkpoint active loops, deploy, and
--     resume without losing miner-funded work.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

ALTER TABLE public.research_loop_run_queue_events
    DROP CONSTRAINT IF EXISTS research_loop_run_queue_events_event_type_check;
ALTER TABLE public.research_loop_run_queue_events
    ADD CONSTRAINT research_loop_run_queue_events_event_type_check
    CHECK (
        event_type IN (
            'queued',
            'started',
            'paused',
            'completed',
            'failed',
            'cancelled',
            'tombstoned'
        )
    );

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;
ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check
    CHECK (
        event_type IN (
            'loop_started',
            'loop_resumed',
            'hypothesis_drafted',
            'patch_drafted',
            'patch_validation_passed',
            'patch_validation_failed',
            'dev_check_passed',
            'dev_check_failed',
            'reflection_recorded',
            'checkpoint_saved',
            'loop_paused',
            'candidate_selected',
            'loop_completed',
            'loop_failed'
        )
    );

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_loop_status_check;
ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_loop_status_check
    CHECK (
        loop_status IN (
            'running',
            'paused',
            'completed',
            'failed'
        )
    );

CREATE TABLE IF NOT EXISTS public.research_lab_gateway_control_events (
    event_id       UUID        PRIMARY KEY,
    schema_version TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    control_key    TEXT        NOT NULL CHECK (control_key <> ''),
    seq            INTEGER     NOT NULL CHECK (seq >= 0),
    event_type     TEXT        NOT NULL CHECK (
                               event_type IN (
                                   'pause_requested',
                                   'resume_requested',
                                   'tombstoned'
                               )),
    control_status TEXT        NOT NULL CHECK (
                               control_status IN (
                                   'active',
                                   'inactive',
                                   'tombstoned'
                               )),
    actor_ref      TEXT,
    reason         TEXT        NOT NULL,
    anchored_hash  TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc      JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                               jsonb_typeof(event_doc) = 'object'
                               AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                               ),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_gateway_control_events_key_seq_key UNIQUE (control_key, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_gateway_control_events_latest
    ON public.research_lab_gateway_control_events(control_key, seq DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_gateway_control_events_mutation
    ON public.research_lab_gateway_control_events;
CREATE TRIGGER prevent_research_lab_gateway_control_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_gateway_control_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_gateway_control_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_gateway_control_events TO service_role;

ALTER TABLE public.research_lab_gateway_control_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_gateway_control_events;
CREATE POLICY service_role_read ON public.research_lab_gateway_control_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_gateway_control_events;
CREATE POLICY service_role_insert ON public.research_lab_gateway_control_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_gateway_control_current
WITH (security_invoker = true) AS
SELECT
    e.control_key,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.control_status AS current_control_status,
    e.actor_ref,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.event_doc,
    e.created_at AS current_status_at
FROM (
    SELECT DISTINCT ON (control_key) *
    FROM public.research_lab_gateway_control_events
    ORDER BY control_key, seq DESC, created_at DESC
) e;

REVOKE ALL ON TABLE public.research_lab_gateway_control_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_gateway_control_current TO service_role;

CREATE OR REPLACE FUNCTION public.guard_research_lab_queue_capacity()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    capacity_text TEXT;
    stale_text TEXT;
    capacity INTEGER;
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

    stale_text := COALESCE(NEW.event_doc->>'active_loop_stale_after_seconds', '300');
    IF stale_text !~ '^[0-9]+$' THEN
        stale_text := '300';
    END IF;

    capacity := capacity_text::INTEGER;
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
       AND q.current_queue_status IN ('queued', 'started', 'paused')
       AND (q.current_queue_status = 'paused' OR q.current_status_at >= cutoff)
       AND btrim(t.miner_hotkey) = btrim(v_miner_hotkey);

    IF same_hotkey_count > 0 THEN
        RAISE EXCEPTION
            'research_lab_queue_hotkey_conflict: miner % already has active run',
            v_miner_hotkey
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
             AND (q.current_queue_status = 'paused' OR q.current_status_at >= cutoff)
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

COMMENT ON TABLE public.research_lab_gateway_control_events IS
    'Append-only gateway operator controls such as Research Lab autoresearch maintenance pause/resume.';
COMMENT ON VIEW public.research_lab_gateway_control_current IS
    'Current gateway operator control projection.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT control_key, current_control_status
--   FROM public.research_lab_gateway_control_current
--   WHERE control_key = 'autoresearch_maintenance';
--
--   SELECT tgname
--   FROM pg_trigger
--   WHERE tgname = 'prevent_research_lab_gateway_control_events_mutation';
