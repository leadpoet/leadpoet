-- Research Lab loop events: allow the source_inspection_seeded event type.
--
-- Additive constraint widening only. No scoring, benchmark, promotion,
-- reward, allocation, SOURCE_ADD, fulfillment, emission, or validator-weight
-- behavior is changed by this migration.
--
-- The symbol-slice seeding stage (shadow mode) emits a
-- 'source_inspection_seeded' telemetry event per code-edit iteration, but the
-- event_type check constraint predates that value, so every emit fails with
-- 23514 and the seeding stage aborts fail-open. The seeding telemetry cannot
-- accumulate until the constraint admits the new value.
--
-- The value list below is the live list (matching scripts/72, verified against
-- the deployed database's accepted inserts on 2026-07-14) plus
-- 'source_inspection_seeded'. The loop_status constraint is not touched.

BEGIN;

SET LOCAL lock_timeout = '5s';

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
            'source_inspection_seeded',
            'code_edit_repair_requested',
            'code_edit_repair_drafted',
            'code_edit_repair_failed',
            'candidate_patch_apply_failed',
            'candidate_patch_parse_failed',
            'candidate_patch_empty_or_noop',
            'candidate_test_failed',
            'candidate_patch_test_failed',
            'candidate_image_build_failed',
            'candidate_artifact_missing',
            'candidate_repair_exhausted',
            'candidate_generation_fallback_requested',
            'candidate_generation_fallback_drafted',
            'candidate_generation_fallback_failed',
            'loop_direction_planned',
            'plan_alignment_judged',
            'code_edit_alignment_rejected',
            'duplicate_candidate_reused',
            'no_viable_patch',
            'allocator_decision',
            'probe_requested',
            'probe_resolved',
            'probe_blocked'
        )
    );

COMMIT;

-- Verify after applying:
--   SELECT pg_get_constraintdef(oid)
--   FROM pg_constraint
--   WHERE conname = 'research_lab_auto_research_loop_events_event_type_check';
