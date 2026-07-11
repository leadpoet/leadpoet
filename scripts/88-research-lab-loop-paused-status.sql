-- Research Lab loop projection: allow paused state (apply after migration 87).
--
-- Additive projection hardening only. No scoring, benchmark, promotion,
-- reward, allocation, SOURCE_ADD, fulfillment, emission, or validator-weight
-- behavior is changed by this migration.
--
-- Queue rows pause cooperatively (operator pause, blocked_for_credit), but the
-- loop event schema only allowed running/completed/failed, so the loop
-- projection kept displaying paused loops as running. Allow the paused state
-- and its transition events so the reconciler can align the projection.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;

ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check CHECK (
        event_type IN (
            'loop_started',
            'hypothesis_drafted',
            'patch_drafted',
            'patch_validation_passed',
            'patch_validation_failed',
            'dev_check_passed',
            'dev_check_failed',
            'reflection_recorded',
            'candidate_selected',
            'loop_completed',
            'loop_failed',
            'loop_paused',
            'loop_resumed'
        ));

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_loop_status_check;

ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_loop_status_check CHECK (
        loop_status IN (
            'running',
            'paused',
            'completed',
            'failed'
        ));

COMMIT;
