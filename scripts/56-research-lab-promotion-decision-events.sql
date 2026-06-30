-- Research Lab: complete promotion decision event/status labels.
--
-- Deployment policy:
--   * Apply after scripts 37 and 52.
--   * Safe to apply repeatedly.
--   * Allows every scored candidate to leave an auditable terminal promotion
--     decision, including public-holdout rejection, disabled promotion, and
--     optional scoring-health quarantine.

BEGIN;

ALTER TABLE public.research_lab_candidate_promotion_events
    DROP CONSTRAINT IF EXISTS research_lab_candidate_promotion_events_event_type_check;

ALTER TABLE public.research_lab_candidate_promotion_events
    ADD CONSTRAINT research_lab_candidate_promotion_events_event_type_check
    CHECK (
        event_type IN (
            'promotion_checked',
            'below_threshold',
            'stale_parent_detected',
            'rebase_queued',
            'rebase_scored',
            'unsupported_candidate_kind',
            'public_holdout_rejected',
            'promotion_disabled',
            'scoring_health_quarantined',
            'promotion_failed',
            'promotion_passed',
            'active_version_created',
            'champion_reward_pending_uid',
            'champion_reward_created',
            'tombstoned'
        )
    );

ALTER TABLE public.research_lab_candidate_promotion_events
    DROP CONSTRAINT IF EXISTS research_lab_candidate_promotion_events_promotion_status_check;

ALTER TABLE public.research_lab_candidate_promotion_events
    ADD CONSTRAINT research_lab_candidate_promotion_events_promotion_status_check
    CHECK (
        promotion_status IN (
            'checked',
            'rejected',
            'rebase_required',
            'rebenchmarking',
            'stale_parent_needs_rescore',
            'disabled',
            'failed',
            'passed',
            'merged',
            'reward_pending_uid',
            'reward_created',
            'tombstoned'
        )
    );

COMMENT ON TABLE public.research_lab_candidate_promotion_events IS
    'Append-only Research Lab candidate promotion decisions and active-model lineage events. Includes terminal disabled/holdout/health-quarantine decisions.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_candidate_promotion_events'::regclass
--   AND conname IN (
--       'research_lab_candidate_promotion_events_event_type_check',
--       'research_lab_candidate_promotion_events_promotion_status_check'
--   )
-- ORDER BY conname;
