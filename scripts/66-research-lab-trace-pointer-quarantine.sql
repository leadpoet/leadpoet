-- Research Lab trace pointer quarantine.
--
-- Historical beta raw trace uploads were optimistic: the corpus row could
-- receive a pointer before the async S3 upload completed. If that upload later
-- failed, the pointer is dangling. This table records operator decisions that
-- a missing pointer is intentionally unrecoverable, without mutating the source
-- corpus rows or pretending an object exists.
--
-- Access model:
--   * private service_role-only table
--   * no anon/authenticated grants
--   * RLS enabled for exposed-schema defense in depth

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_trace_pointer_quarantine (
    s3_ref          TEXT        NOT NULL,
    sha256          TEXT        NOT NULL DEFAULT '',
    source          TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('unrecoverable')),
    reason          TEXT        NOT NULL,
    operator_note   TEXT        NOT NULL DEFAULT '',
    marked_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quarantine_doc  JSONB       NOT NULL DEFAULT '{}'::JSONB
                                CHECK (jsonb_typeof(quarantine_doc) = 'object'),
    PRIMARY KEY (s3_ref, sha256),
    CHECK (s3_ref LIKE 's3://%'),
    CHECK (sha256 = '' OR sha256 LIKE 'sha256:%')
);

CREATE INDEX IF NOT EXISTS idx_trace_pointer_quarantine_status_marked
    ON public.research_lab_trace_pointer_quarantine(status, marked_at DESC);

CREATE INDEX IF NOT EXISTS idx_trace_pointer_quarantine_source
    ON public.research_lab_trace_pointer_quarantine(source, marked_at DESC);

REVOKE ALL ON TABLE public.research_lab_trace_pointer_quarantine
    FROM anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE
    ON TABLE public.research_lab_trace_pointer_quarantine
    TO service_role;

ALTER TABLE public.research_lab_trace_pointer_quarantine
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all
    ON public.research_lab_trace_pointer_quarantine;

CREATE POLICY service_role_all
    ON public.research_lab_trace_pointer_quarantine
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

COMMENT ON TABLE public.research_lab_trace_pointer_quarantine IS
    'Operator audit table for missing trace S3 pointers that are intentionally marked unrecoverable; reconciliation classifies these separately from active capture failures.';

COMMENT ON COLUMN public.research_lab_trace_pointer_quarantine.reason IS
    'Machine-readable reason, e.g. historical_beta_optimistic_upload_missing.';

COMMIT;
