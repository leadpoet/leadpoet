-- Research Lab sales-LLM capture metadata.
--
-- Additive only. Stores sanitized, queryable company-level labels and a
-- corpus-readiness view for future offline dataset construction. This does not
-- mark any row training-approved and does not alter scoring, promotion,
-- rewards, reimbursements, or validator weight paths.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_company_label_examples (
    label_id             UUID        PRIMARY KEY,
    schema_version       TEXT        NOT NULL DEFAULT '1.0'
                                      CHECK (schema_version = '1.0'),
    context_ref          TEXT        NOT NULL,
    run_id               UUID,
    ticket_id            UUID,
    candidate_id         TEXT,
    score_bundle_id      TEXT,
    model_manifest_hash  TEXT,
    model_side           TEXT        NOT NULL CHECK (
                                      model_side IN (
                                          'candidate',
                                          'champion',
                                          'baseline_arm'
                                      )
                                    ),
    is_reference_model   BOOLEAN     NOT NULL DEFAULT FALSE,
    icp_ref              TEXT        NOT NULL,
    icp_hash             TEXT        NOT NULL DEFAULT '',
    company_name         TEXT,
    company_website      TEXT,
    company_linkedin     TEXT,
    industry             TEXT,
    sub_industry         TEXT,
    employee_count       TEXT,
    company_stage        TEXT,
    city                 TEXT,
    state                TEXT,
    country              TEXT,
    model_claimed_score  DOUBLE PRECISION,
    intent_source        TEXT,
    intent_claimed_signal TEXT,
    intent_evidence_url  TEXT,
    intent_evidence_date TEXT,
    attribute_evidence_url TEXT,
    final_score          DOUBLE PRECISION NOT NULL DEFAULT 0,
    failure_reason       TEXT,
    failure_stage        TEXT,
    fit_passed           BOOLEAN,
    attribute_passed     BOOLEAN,
    intent_passed        BOOLEAN,
    icp_fit              DOUBLE PRECISION,
    intent_signal_raw    DOUBLE PRECISION,
    time_decay_multiplier DOUBLE PRECISION,
    intent_signal        DOUBLE PRECISION,
    scorer_trace_ref     TEXT,
    scorer_trace_sha256  TEXT,
    raw_trace_refs       JSONB       NOT NULL DEFAULT '[]'::JSONB CHECK (
                                      jsonb_typeof(raw_trace_refs) = 'array'
                                    ),
    capture_doc          JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                      jsonb_typeof(capture_doc) = 'object'
                                      AND capture_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|raw_secret|service_role|hidden_prompt|provider_output|request_body|response_body|page_content|raw_content|judge_prompt|private_repo|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                    ),
    capture_state        TEXT        NOT NULL DEFAULT 'captured_unreviewed'
                                      CHECK (capture_state IN (
                                          'captured_unreviewed',
                                          'tombstoned'
                                      )),
    rights_state         TEXT        NOT NULL DEFAULT 'rights_unreviewed'
                                      CHECK (rights_state IN (
                                          'rights_unreviewed',
                                          'rights_approved',
                                          'rights_rejected'
                                      )),
    pii_state            TEXT        NOT NULL DEFAULT 'pii_unreviewed'
                                      CHECK (pii_state IN (
                                          'pii_unreviewed',
                                          'pii_approved',
                                          'pii_rejected'
                                      )),
    legal_state          TEXT        NOT NULL DEFAULT 'legal_unreviewed'
                                      CHECK (legal_state IN (
                                          'legal_unreviewed',
                                          'legal_approved',
                                          'legal_rejected'
                                      )),
    eligible_for_training BOOLEAN    NOT NULL DEFAULT FALSE
                                      CHECK (eligible_for_training IS FALSE),
    captured_at          TIMESTAMPTZ NOT NULL,
    dedup_key            TEXT        NOT NULL UNIQUE,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (raw_trace_refs::TEXT !~* '(sk-or-|openrouter_api_key|raw_secret|service_role|hidden_prompt|provider_output|request_body|response_body|page_content|raw_content|judge_prompt|private_repo|proxy[_-]?url|://[^/]+:[^/@]+@)'),
    CHECK (COALESCE(company_name, '') !~* '(sk-or-|sb_secret|service_role|api[_-]?key|password|://[^/]+:[^/@]+@)'),
    CHECK (COALESCE(intent_claimed_signal, '') !~* '(sk-or-|sb_secret|service_role|api[_-]?key|password|page content|raw content|judge prompt|://[^/]+:[^/@]+@)'),
    CHECK (COALESCE(failure_reason, '') !~* '(sk-or-|sb_secret|service_role|api[_-]?key|password|://[^/]+:[^/@]+@)')
);

CREATE INDEX IF NOT EXISTS idx_research_lab_company_labels_context
    ON public.research_lab_company_label_examples(context_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_company_labels_candidate
    ON public.research_lab_company_label_examples(candidate_id, created_at DESC)
    WHERE candidate_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_research_lab_company_labels_icp
    ON public.research_lab_company_label_examples(icp_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_company_labels_side_score
    ON public.research_lab_company_label_examples(model_side, final_score DESC, created_at DESC);

DROP TRIGGER IF EXISTS prevent_research_lab_company_label_examples_mutation
    ON public.research_lab_company_label_examples;
CREATE TRIGGER prevent_research_lab_company_label_examples_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_company_label_examples
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_company_label_examples FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_company_label_examples TO service_role;

ALTER TABLE public.research_lab_company_label_examples ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_company_label_examples;
CREATE POLICY service_role_read ON public.research_lab_company_label_examples
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_company_label_examples;
CREATE POLICY service_role_insert ON public.research_lab_company_label_examples
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_sales_llm_corpus_metadata_current
WITH (security_invoker = true) AS
SELECT
    t.trajectory_id,
    t.brief_id,
    t.island,
    t.funder_hotkey,
    t.brief_sanitized_ref,
    t.champion_base,
    t.created_at,
    'captured_unreviewed'::TEXT AS capture_state,
    'rights_unreviewed'::TEXT AS rights_state,
    'pii_unreviewed'::TEXT AS pii_state,
    'legal_unreviewed'::TEXT AS legal_state,
    FALSE AS eligible_for_training,
    COALESCE(trace_counts.execution_trace_count, 0) AS execution_trace_count,
    COALESCE(trace_counts.champion_trace_count, 0) AS champion_trace_count,
    COALESCE(trace_counts.baseline_arm_trace_count, 0) AS baseline_arm_trace_count,
    COALESCE(evidence_counts.evidence_bundle_count, 0) AS evidence_bundle_count,
    COALESCE(score_counts.score_bundle_count, 0) AS score_bundle_count,
    COALESCE(label_counts.company_label_count, 0) AS company_label_count,
    COALESCE(label_counts.positive_label_count, 0) AS positive_label_count,
    COALESCE(label_counts.negative_label_count, 0) AS negative_label_count,
    COALESCE(source_diff_counts.candidate_count, 0) AS candidate_count,
    COALESCE(source_diff_counts.candidate_source_diff_uri_count, 0) AS candidate_source_diff_uri_count,
    COALESCE(trace_counts.s3_pointer_count, 0) AS s3_pointer_count
FROM public.research_trajectories t
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS execution_trace_count,
        COUNT(*) FILTER (WHERE role = 'champion') AS champion_trace_count,
        COUNT(*) FILTER (WHERE role = 'baseline_arm') AS baseline_arm_trace_count,
        COALESCE(SUM((
            SELECT COUNT(*)
            FROM jsonb_array_elements(et.calls) call
            WHERE call ? 's3_ref' AND COALESCE(call->>'s3_ref', '') <> ''
        )), 0) AS s3_pointer_count
    FROM public.execution_traces et
    WHERE et.trajectory_id = t.trajectory_id
) trace_counts ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS evidence_bundle_count
    FROM public.evidence_bundles eb
    WHERE eb.bundle_doc->>'trajectory_ref' = 'trajectory:' || t.trajectory_id::TEXT
) evidence_counts ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(DISTINCT CASE
        WHEN et.score_bundle_ref LIKE 'score_bundle:%'
        THEN NULLIF(REPLACE(et.score_bundle_ref, 'score_bundle:', ''), 'unavailable')
        ELSE NULL
    END) AS score_bundle_count
    FROM public.execution_traces et
    WHERE et.trajectory_id = t.trajectory_id
) score_counts ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS company_label_count,
        COUNT(*) FILTER (WHERE final_score > 0 AND COALESCE(failure_reason, '') = '') AS positive_label_count,
        COUNT(*) FILTER (WHERE final_score <= 0 OR COALESCE(failure_reason, '') <> '') AS negative_label_count
    FROM public.research_lab_company_label_examples l
    WHERE l.context_ref IN (
        SELECT DISTINCT
            COALESCE(
                NULLIF(REPLACE(et.trace_doc->>'candidate_ref', 'candidate:candidate:', 'candidate:'), ''),
                NULLIF(et.trace_doc->>'lab_run_ref', ''),
                ''
            )
        FROM public.execution_traces et
        WHERE et.trajectory_id = t.trajectory_id
    )
    OR l.candidate_id IN (
        SELECT DISTINCT
            REPLACE(
                REPLACE(et.trace_doc->>'candidate_ref', 'candidate:candidate:', 'candidate:'),
                'candidate:',
                ''
            )
        FROM public.execution_traces et
        WHERE et.trajectory_id = t.trajectory_id
    )
    OR l.run_id::TEXT IN (
        SELECT DISTINCT REPLACE(et.trace_doc->>'lab_run_ref', 'research_loop_run:', '')
        FROM public.execution_traces et
        WHERE et.trajectory_id = t.trajectory_id
    )
) label_counts ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS candidate_count,
        COUNT(*) FILTER (
            WHERE COALESCE(c.candidate_build_doc->>'source_diff_artifact_uri', '') <> ''
        ) AS candidate_source_diff_uri_count
    FROM public.research_lab_candidate_artifacts c
    WHERE c.run_id::TEXT IN (
        SELECT REPLACE(et.trace_doc->>'lab_run_ref', 'research_loop_run:', '')
        FROM public.execution_traces et
        WHERE et.trajectory_id = t.trajectory_id
    )
) source_diff_counts ON TRUE;

REVOKE ALL ON TABLE public.research_lab_sales_llm_corpus_metadata_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_sales_llm_corpus_metadata_current TO service_role;

COMMENT ON TABLE public.research_lab_company_label_examples IS
    'Append-only sanitized company-level scorer labels for future offline sales-LLM dataset construction. Full traces remain in SSE-KMS S3.';
COMMENT ON COLUMN public.research_lab_company_label_examples.eligible_for_training IS
    'Always false in this capture layer; offline training approval is intentionally outside this runtime.';
COMMENT ON VIEW public.research_lab_sales_llm_corpus_metadata_current IS
    'Read-only Research Lab sales-LLM corpus coverage view. It reports capture metadata only and never claims training approval.';

COMMIT;
