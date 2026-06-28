-- Research Lab: expose image-build candidate fields to the scoring queue view.
--
-- Deployment policy:
--   * Apply after scripts 46 and 47.
--   * Safe to apply repeatedly.
--   * Fixes stale view shape where research_lab_candidate_evaluation_current
--     was created before image-build columns existed on candidate_artifacts.

BEGIN;

DROP VIEW IF EXISTS public.research_lab_candidate_evaluation_current;

CREATE VIEW public.research_lab_candidate_evaluation_current
WITH (security_invoker = true) AS
SELECT
    c.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.candidate_status AS current_candidate_status,
    e.evaluator_ref AS current_evaluator_ref,
    e.reason AS current_reason,
    e.score_bundle_id AS current_score_bundle_id,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_candidate_artifacts c
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_candidate_evaluation_events e
    WHERE e.candidate_id = c.candidate_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_candidate_evaluation_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_candidate_evaluation_current TO service_role;

COMMENT ON VIEW public.research_lab_candidate_evaluation_current IS
    'Latest candidate evaluation status joined to immutable candidate artifact rows, including image-build manifest columns.';

ALTER TABLE public.research_lab_candidate_promotion_events
    DROP CONSTRAINT IF EXISTS research_lab_candidate_promotion_events_event_type_check;

ALTER TABLE public.research_lab_candidate_promotion_events
    ADD CONSTRAINT research_lab_candidate_promotion_events_event_type_check
    CHECK (
        event_type IN (
            'promotion_checked',
            'unsupported_candidate_kind',
            'below_threshold',
            'stale_parent_detected',
            'rebase_queued',
            'rebase_scored',
            'promotion_failed',
            'promotion_passed',
            'active_version_created',
            'champion_reward_pending_uid',
            'champion_reward_created',
            'tombstoned'
        )
    );

COMMIT;

-- Verification:
-- SELECT column_name
-- FROM information_schema.columns
-- WHERE table_schema = 'public'
--   AND table_name = 'research_lab_candidate_evaluation_current'
--   AND column_name IN (
--       'candidate_kind',
--       'candidate_model_manifest_hash',
--       'candidate_model_manifest_doc',
--       'candidate_source_diff_hash',
--       'candidate_build_doc'
--   )
-- ORDER BY column_name;
--
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_candidate_promotion_events'::regclass
--   AND conname = 'research_lab_candidate_promotion_events_event_type_check';
