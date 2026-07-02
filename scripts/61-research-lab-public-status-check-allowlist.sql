-- Research Lab: allowlist the awaiting-payment public loop status.
--
-- Deployment policy:
--   * Apply after scripts 55 and 57.
--   * Safe to apply repeatedly.
--   * STRICT SUPERSET of the script-57 allowlists: every label a deployed writer
--     can already insert stays valid, so old and new gateway code keep working
--     during rollout (DB first, code second).
--   * Adds `awaiting_payment` to event_type and outcome_label. The projection has
--     emitted it since the canonical-lifecycle change (public_activity.py), but
--     script 57 never allowlisted it, so the insert fails the CHECK constraint and
--     `safe_project_public_loop_activity` swallows the error — leaving the
--     canonical public_status/payment_state fields empty on every live card.
--   * `ALTER TABLE ... DROP/ADD CONSTRAINT` takes a brief ACCESS EXCLUSIVE
--     metadata lock on research_lab_public_loop_card_events. Run off-peak; the
--     transaction is short (constraint validation scans the table once).
--   * Verify with a staging insert of an `awaiting_payment` event before prod.

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
            'completed_no_candidate',
            'awaiting_payment'
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
            'completed_no_candidate',
            'awaiting_payment'
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
    'Append-only sanitized Research Lab public activity lifecycle and outcome events. Status labels include blocked/waiting/rescore/not-started/completed-without-candidate/awaiting-payment states.';

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
