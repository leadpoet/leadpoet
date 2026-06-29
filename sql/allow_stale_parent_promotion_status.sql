-- Allow promotion_status = 'stale_parent_needs_rescore'.
--
-- When a candidate's parent model has gone stale, the research-lab scoring worker
-- records a candidate promotion event with promotion_status='stale_parent_needs_rescore'
-- and then marks the candidate rejected (graceful "parent moved on, needs rescore").
-- The CHECK constraint did not include that value, so the promotion-event insert hit a
-- 23514 check-constraint violation and the whole candidate failed with a platform error
-- instead of the intended graceful rejection. This widens the constraint to allow the
-- value; it only ADDS an allowed value, so all existing rows remain valid and the change
-- is reversible by re-adding the prior list.
--
-- Run against the Supabase session pooler (port 5432). DDL is transactional.

BEGIN;

ALTER TABLE research_lab_candidate_promotion_events
  DROP CONSTRAINT IF EXISTS research_lab_candidate_promotion_events_promotion_status_check;

ALTER TABLE research_lab_candidate_promotion_events
  ADD CONSTRAINT research_lab_candidate_promotion_events_promotion_status_check
  CHECK (promotion_status = ANY (ARRAY[
    'checked',
    'rejected',
    'rebase_required',
    'rebenchmarking',
    'failed',
    'passed',
    'merged',
    'reward_pending_uid',
    'reward_created',
    'tombstoned',
    'stale_parent_needs_rescore'
  ]::text[]));

COMMIT;
