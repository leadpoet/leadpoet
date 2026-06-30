-- Research Lab: widen public loop-card event_type / outcome_label CHECK constraints
-- to the canonical status vocabulary emitted by derive_public_loop_outcome().
--
-- New canonical statuses (in addition to the existing set):
--   waiting_for_credits   - run paused on OpenRouter insufficient-credit (resumable)
--   waiting_for_baseline  - candidate queued, blocked on the private baseline not ready
--   needs_rescore         - candidate rejected stale-parent, needs rebase/rescore
--   completed_no_candidate- queue completed but produced no candidate (ops/investigate)
--
-- Operator-run migration (Supabase). Idempotent: drops then re-adds the constraints.
-- COORDINATION: the dashboard frontend consumes these label strings — deploy this in
-- lockstep with the frontend update that recognizes the new labels.

BEGIN;

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_event_type_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_event_type_check CHECK (
        event_type IN (
            'submitted',
            'queued',
            'running',
            'candidate_generation_complete',
            'scoring',
            'scored',
            'promotion_passed',
            'promoted',
            'failed',
            'tombstoned',
            -- canonical additions
            'waiting_for_credits',
            'waiting_for_baseline',
            'needs_rescore',
            'completed_no_candidate'
        )
    );

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_outcome_label_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_outcome_label_check CHECK (
        outcome_label IN (
            'submitted',
            'queued',
            'running',
            'candidate_generation_complete',
            'scoring',
            'scored_no_gain',
            'scored_promising',
            'promotion_passed',
            'promoted',
            'failed',
            -- canonical additions
            'waiting_for_credits',
            'waiting_for_baseline',
            'needs_rescore',
            'completed_no_candidate'
        )
    );

COMMIT;
