-- Research Lab: add completed-without-candidate public loop status.
--
-- Deployment policy:
--   * Apply after scripts 55 and 56.
--   * Safe to apply repeatedly.
--   * Preserves existing truthful states from script 55 and adds
--     completed_no_candidate for loops whose queue completed without a
--     scoreable candidate artifact.

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
            'waiting_for_baseline',
            'needs_rescore',
            'blocked_for_credit',
            'not_started',
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
            'waiting_for_baseline',
            'needs_rescore',
            'blocked_for_credit',
            'not_started',
            'completed_no_candidate'
        )
    );

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_outcome_band_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_outcome_band_check CHECK (
        outcome_band IN (
            'pending',
            'no_gain',
            'small_gain',
            'passed_threshold',
            'promoted',
            'failed',
            'blocked'
        )
    );

COMMENT ON TABLE public.research_lab_public_loop_card_events IS
    'Append-only sanitized Research Lab public activity lifecycle and outcome events. Status labels include blocked/waiting/rescore/not-started/completed-without-candidate states.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_public_loop_card_events'::regclass
--   AND conname IN (
--       'research_lab_public_loop_card_events_event_type_check',
--       'research_lab_public_loop_card_events_outcome_label_check',
--       'research_lab_public_loop_card_events_outcome_band_check'
--   )
-- ORDER BY conname;
