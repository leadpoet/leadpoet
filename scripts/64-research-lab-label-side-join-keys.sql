-- 64-research-lab-label-side-join-keys.sql
--
-- trajectoryimprovements.md P18: the label side of the corpus joins through
-- loose/nullable keys or not at all. This migration adds the explicit keys:
--
--   1. execution_traces.trajectory_id — reward → score bundle → execution
--      trace → trajectory must resolve with explicit keys, not uuid5
--      re-derivation. Populated by the projector when
--      RESEARCH_LAB_EXECUTION_TRACE_TRAJECTORY_ID_ENABLED=true (set the flag
--      only after this script is applied).
--
--   2. 'allocator_decision' loop-event type — one pointer-scale record per
--      run start of which cell priors aimed the run (the meta-allocator's own
--      future training data). Emitted when
--      RESEARCH_LAB_ALLOCATOR_DECISION_EVENTS_ENABLED=true.
--
--   3. research_lab_shadow_monitor_windows — shadow windows become DB rows
--      keyed to (active_version_id, promotion_event_ref); an explicit
--      not_monitored row is recordable when the monitor was off during a
--      merge window. Populated by the gateway-side importer
--      (gateway/research_lab/trace_reconciler.py::import_shadow_windows);
--      the shadow process itself stays read-only by design.
--
-- Deployment policy: apply after scripts/27 and scripts/58; safe to apply
-- repeatedly.

BEGIN;

-- 1. explicit trajectory join key on execution traces -----------------------

ALTER TABLE public.execution_traces
    ADD COLUMN IF NOT EXISTS trajectory_id uuid;

COMMENT ON COLUMN public.execution_traces.trajectory_id IS
    'Deterministic research_trajectories.trajectory_id this trace belongs to (P18 explicit join key; NULL for rows projected before scripts/64).';

CREATE INDEX IF NOT EXISTS idx_execution_traces_trajectory_id
    ON public.execution_traces (trajectory_id)
    WHERE trajectory_id IS NOT NULL;

-- Backfill note: trajectory_id for historical rows is re-derivable via the
-- deterministic uuid5 scheme (trajectory_projector.execution_trace_id_for_*),
-- or simply by rerunning `--traces-backfill` after enabling the flag.

-- 2. allocator_decision loop events -----------------------------------------

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;

-- The list below is scripts/58's exact allowlist plus 'allocator_decision'.
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
            'no_viable_patch',
            'allocator_decision'
        )
    );

-- 3. shadow monitor windows as first-class rows ------------------------------

CREATE TABLE IF NOT EXISTS public.research_lab_shadow_monitor_windows (
    window_id uuid PRIMARY KEY,
    active_version_id text NOT NULL,
    promotion_event_ref text,
    window_status text NOT NULL CHECK (
        window_status IN ('open', 'completed', 'alerted', 'not_monitored')
    ),
    comparable_day_count integer NOT NULL DEFAULT 0,
    cumulative_shadow_live_diff_points double precision,
    mean_shadow_live_diff_points double precision,
    alert_count integer NOT NULL DEFAULT 0,
    window_report_uri text,
    window_doc jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (active_version_id)
);

COMMENT ON TABLE public.research_lab_shadow_monitor_windows IS
    'P18: one row per post-merge shadow window (or an explicit not_monitored row) keyed to the promotion it adjudicates; imported from the read-only shadow process''s S3 report docs.';

COMMIT;
