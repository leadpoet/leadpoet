-- Research Lab: candidate-generation diagnostics and no-buildable public label.
--
-- Deployment policy:
--   * Apply after scripts 61, 64, and 68.
--   * Safe to apply repeatedly.
--   * Strict supersets of the deployed CHECK allowlists so old gateway/scoring
--     code and historical rows remain valid during rollout.
--   * Apply before deploying gateway code that emits the new event types or
--     public `no_buildable_candidate` card events.

BEGIN;

-- Public loop cards: replace generic failed zero-candidate cards with a
-- truthful terminal "No buildable candidate" status.

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
            'awaiting_payment',
            'no_buildable_candidate'
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
            'awaiting_payment',
            'no_buildable_candidate'
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
    'Append-only sanitized Research Lab public activity lifecycle and outcome events. Status labels include blocked/waiting/rescore/not-started/completed-without-candidate/awaiting-payment/no-buildable-candidate states.';

-- Auto-research loop events: split candidate-generation/build failures without
-- overloading no_valid_image_build_finalists.

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
            'allocator_decision'
        )
    );

COMMENT ON TABLE public.research_lab_auto_research_loop_events IS
    'Append-only hosted Research Lab auto-research lifecycle, source-inspection, planner, repair, fallback, and image-build candidate diagnostics.';

COMMIT;

-- Verification:
-- SELECT conname, pg_get_constraintdef(oid)
-- FROM pg_constraint
-- WHERE conrelid IN (
--     'public.research_lab_public_loop_card_events'::regclass,
--     'public.research_lab_auto_research_loop_events'::regclass
-- )
-- AND conname IN (
--     'research_lab_public_loop_card_events_event_type_check',
--     'research_lab_public_loop_card_events_outcome_label_check',
--     'research_lab_public_loop_card_events_outcome_band_check',
--     'research_lab_auto_research_loop_events_event_type_check'
-- )
-- ORDER BY conrelid::regclass::text, conname;
