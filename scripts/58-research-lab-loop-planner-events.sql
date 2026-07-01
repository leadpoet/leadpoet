-- Research Lab: add loop planner, alignment judge, and novelty event labels.
--
-- Deployment policy:
--   * Apply after scripts 51 and 57.
--   * Safe to apply repeatedly.
--   * Allows hosted auto-research loops to audit miner-focus planning,
--     plan-alignment judging, duplicate candidate reuse, and no-viable-patch
--     outcomes without changing scoring, promotion, or reimbursement tables.

BEGIN;

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
            'loop_failed',
            'code_edit_drafted',
            'code_edit_validation_passed',
            'code_edit_validation_failed',
            'candidate_build_started',
            'candidate_build_passed',
            'candidate_build_failed',
            'source_inspection_requested',
            'source_inspection_resolved',
            'source_inspection_failed',
            'code_edit_repair_requested',
            'code_edit_repair_drafted',
            'code_edit_repair_failed',
            'candidate_patch_apply_failed',
            'candidate_test_failed',
            'candidate_image_build_failed',
            'loop_direction_planned',
            'plan_alignment_judged',
            'code_edit_alignment_rejected',
            'duplicate_candidate_reused',
            'no_viable_patch'
        )
    );

COMMENT ON TABLE public.research_lab_auto_research_loop_events IS
    'Append-only hosted Research Lab auto-research loop lifecycle, code-edit, planner, alignment, and candidate-generation events.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid = 'public.research_lab_auto_research_loop_events'::regclass
--   AND conname = 'research_lab_auto_research_loop_events_event_type_check';
