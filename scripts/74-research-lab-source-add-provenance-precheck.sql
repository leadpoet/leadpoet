-- SOURCE_ADD provenance precheck metadata.
--
-- Additive migration for the existing append-only SOURCE_ADD submissions table.
-- This does not modify the accepted catalog or SOURCE_ADD reward tables.

BEGIN;

ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS research_lab_source_add_submissions_stage_check;

ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submissions_stage_check
    CHECK (
        stage IN (
            'submitted',
            'manifest_validated',
            'provenance_precheck_passed',
            'needs_manual_review',
            'rejected_precheck',
            'static_scan_passed',
            'llm_review_passed',
            'trial_completed',
            'accepted',
            'rejected'
        )
    );

ALTER TABLE public.research_lab_source_add_submissions
    ADD COLUMN IF NOT EXISTS precheck_status TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_source_add_submissions
    ADD COLUMN IF NOT EXISTS precheck_doc JSONB NOT NULL DEFAULT '{}'::JSONB;

ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS research_lab_source_add_submissions_precheck_status_check;

ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submissions_precheck_status_check
    CHECK (
        precheck_status IN (
            '',
            'provenance_precheck_passed',
            'needs_manual_review',
            'rejected_precheck'
        )
    );

ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS research_lab_source_add_submissions_precheck_doc_safe_check;

ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submissions_precheck_doc_safe_check
    CHECK (
        jsonb_typeof(precheck_doc) = 'object'
        AND precheck_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|api_key"\s*:\s*"[a-z0-9]|password"\s*:)'
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_source_add_submissions_precheck
    ON public.research_lab_source_add_submissions (precheck_status, created_at DESC);

COMMENT ON COLUMN public.research_lab_source_add_submissions.precheck_status IS
    'SOURCE_ADD provenance precheck result before manual review; never implies accepted catalog status.';

COMMENT ON COLUMN public.research_lab_source_add_submissions.precheck_doc IS
    'Sanitized SOURCE_ADD provenance precheck audit doc: provider statuses, reference domains, reasons, and bounded excerpts only.';

COMMIT;
