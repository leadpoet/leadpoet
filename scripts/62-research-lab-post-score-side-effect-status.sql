-- Research Lab: allowlist the post-score side-effect status (bug #9 / Chain B).
--
-- The scoring worker no longer mislabels a post-score side-effect failure as
-- promotion_failed (which outranked scored history on public cards); it now
-- records event_type/promotion_status 'post_score_side_effect_failed', which
-- the public-status derivation explicitly ignores. Without this allowlist the
-- rename's event writes fail the CHECK and are silently swallowed.
--
-- Deployment policy:
--   * Apply after script 56, in the SAME deploy as the scoring-worker rename.
--   * Strict superset of script 56 — old writers keep working.
--   * Safe to apply repeatedly.

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
            'post_score_side_effect_failed',
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
            'post_score_side_effect_failed',
            'failed',
            'passed',
            'merged',
            'reward_pending_uid',
            'reward_created',
            'tombstoned'
        )
    );

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
