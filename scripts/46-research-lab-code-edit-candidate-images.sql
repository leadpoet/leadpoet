-- Research Lab: code-edit candidate private model image pipeline.
--
-- Deployment policy:
--   * Apply after scripts 33 and 44.
--   * Adds explicit image-build candidate fields while keeping patch candidates.
--   * Expands auto-research loop events for code edit/build lifecycle evidence.

BEGIN;

ALTER TABLE public.research_lab_candidate_artifacts
    ADD COLUMN IF NOT EXISTS candidate_kind TEXT NOT NULL DEFAULT 'patch'
        CHECK (candidate_kind IN ('patch', 'image_build')),
    ADD COLUMN IF NOT EXISTS candidate_model_manifest_hash TEXT
        CHECK (
            candidate_model_manifest_hash IS NULL
            OR candidate_model_manifest_hash ~ '^sha256:[0-9a-f]{64}$'
        ),
    ADD COLUMN IF NOT EXISTS candidate_model_manifest_doc JSONB
        CHECK (
            candidate_model_manifest_doc IS NULL
            OR (
                jsonb_typeof(candidate_model_manifest_doc) = 'object'
                AND candidate_model_manifest_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|proxy[_-]?url|://[^/]+:[^/@]+@)'
            )
        ),
    ADD COLUMN IF NOT EXISTS candidate_source_diff_hash TEXT
        CHECK (
            candidate_source_diff_hash IS NULL
            OR candidate_source_diff_hash ~ '^sha256:[0-9a-f]{64}$'
        ),
    ADD COLUMN IF NOT EXISTS candidate_build_doc JSONB NOT NULL DEFAULT '{}'::JSONB
        CHECK (
            jsonb_typeof(candidate_build_doc) = 'object'
            AND candidate_build_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|proxy[_-]?url|://[^/]+:[^/@]+@)'
        );

ALTER TABLE public.research_lab_candidate_artifacts
    DROP CONSTRAINT IF EXISTS research_lab_candidate_artifacts_image_build_check;
ALTER TABLE public.research_lab_candidate_artifacts
    ADD CONSTRAINT research_lab_candidate_artifacts_image_build_check
    CHECK (
        candidate_kind <> 'image_build'
        OR (
            candidate_model_manifest_hash IS NOT NULL
            AND candidate_model_manifest_doc IS NOT NULL
            AND candidate_source_diff_hash IS NOT NULL
        )
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_candidate_artifacts_kind
    ON public.research_lab_candidate_artifacts(candidate_kind, created_at DESC);

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
            'candidate_build_failed'
        )
    );

COMMENT ON COLUMN public.research_lab_candidate_artifacts.candidate_kind IS
    'Candidate artifact type: patch compatibility candidate or built private model image candidate.';
COMMENT ON COLUMN public.research_lab_candidate_artifacts.candidate_model_manifest_doc IS
    'For image_build candidates, the signed immutable private model manifest produced by the gateway builder.';
COMMENT ON COLUMN public.research_lab_candidate_artifacts.candidate_build_doc IS
    'Redacted build evidence for code-edit image candidates.';

COMMIT;

-- Smoke checks after applying:
--
--   SELECT column_name
--   FROM information_schema.columns
--   WHERE table_schema = 'public'
--     AND table_name = 'research_lab_candidate_artifacts'
--     AND column_name IN (
--       'candidate_kind',
--       'candidate_model_manifest_hash',
--       'candidate_model_manifest_doc',
--       'candidate_source_diff_hash',
--       'candidate_build_doc'
--     )
--   ORDER BY column_name;
