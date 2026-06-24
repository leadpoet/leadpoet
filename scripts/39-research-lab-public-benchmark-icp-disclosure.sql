-- Research Lab public benchmark ICP disclosure.
--
-- Deployment policy:
--   * Apply after scripts 36, 37, and 38.
--   * Allows miner-facing public benchmark reports to reveal exact public ICP
--     intent_signals for the configured public split only.
--   * Keeps public benchmark rows append-only and service-role only.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext markers, judge prompts, private image refs, candidate patch
--     manifests, proxy credentials, credential URLs, or raw provider secrets
--     may be stored here.

BEGIN;

DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.research_lab_public_benchmark_reports'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) ILIKE '%intent_signals%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_public_benchmark_reports DROP CONSTRAINT %I',
            constraint_name
        );
    END LOOP;
END;
$$;

DO $$
DECLARE
    constraint_name TEXT;
BEGIN
    FOR constraint_name IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.research_lab_public_benchmark_report_events'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) ILIKE '%intent_signals%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_public_benchmark_report_events DROP CONSTRAINT %I',
            constraint_name
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_public_benchmark_reports
    DROP CONSTRAINT IF EXISTS research_lab_public_benchmark_reports_report_doc_public_icp_safe;
ALTER TABLE public.research_lab_public_benchmark_reports
    ADD CONSTRAINT research_lab_public_benchmark_reports_report_doc_public_icp_safe
    CHECK (
        jsonb_typeof(report_doc) = 'object'
        AND report_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@|https?://)'
    );

ALTER TABLE public.research_lab_public_benchmark_report_events
    DROP CONSTRAINT IF EXISTS research_lab_public_benchmark_report_events_event_doc_public_icp_safe;
ALTER TABLE public.research_lab_public_benchmark_report_events
    ADD CONSTRAINT research_lab_public_benchmark_report_events_event_doc_public_icp_safe
    CHECK (
        jsonb_typeof(event_doc) = 'object'
        AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@|https?://)'
    );

COMMENT ON TABLE public.research_lab_public_benchmark_reports IS
    'Append-only Research Lab miner-facing public benchmark reports. Reports may include exact public-split ICP intent_signals but never private holdout ICP plaintext.';

COMMIT;
