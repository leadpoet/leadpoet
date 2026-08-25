BEGIN;

-- Candidate artifact derivation failures are emitted by the production code
-- loop and must remain durable instead of aborting the enclosing V2 decision.
ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;
ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check CHECK (
        event_type IN (
            'loop_started', 'loop_resumed', 'hypothesis_drafted', 'patch_drafted',
            'patch_validation_passed', 'patch_validation_failed', 'dev_check_passed',
            'dev_check_failed', 'reflection_recorded', 'checkpoint_saved', 'loop_paused',
            'candidate_selected', 'loop_completed', 'loop_failed', 'code_edit_drafted',
            'code_edit_validation_passed', 'code_edit_validation_failed',
            'candidate_build_started', 'candidate_build_passed', 'candidate_build_failed',
            'source_inspection_requested', 'source_inspection_seeded',
            'source_inspection_resolved',
            'source_inspection_failed', 'code_edit_repair_requested',
            'code_edit_repair_drafted', 'code_edit_repair_failed',
            'candidate_patch_apply_failed', 'candidate_patch_parse_failed',
            'candidate_patch_empty_or_noop', 'candidate_test_failed',
            'candidate_patch_test_failed', 'candidate_image_build_failed',
            'candidate_artifact_missing', 'candidate_repair_exhausted',
            'candidate_generation_fallback_requested',
            'candidate_generation_fallback_drafted',
            'candidate_generation_fallback_failed', 'loop_direction_planned',
            'plan_alignment_judged', 'code_edit_alignment_rejected',
            'duplicate_candidate_reused', 'no_viable_patch', 'allocator_decision',
            'probe_requested', 'probe_resolved', 'probe_blocked',
            'candidate_derived_artifact_failed'
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_auto_research_loop_events
    VALIDATE CONSTRAINT research_lab_auto_research_loop_events_event_type_check;

COMMIT;
