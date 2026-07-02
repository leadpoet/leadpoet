-- Research Lab Improvement Engine private issue/eval/fix state.
--
-- Deployment policy:
--   * Apply after scripts/62.
--   * Service-role only; no anon/authenticated access.
--   * Engine tables are advisory. Score bundles/verifier artifacts remain
--     canonical for rewards.

BEGIN;

CREATE TABLE IF NOT EXISTS public.engine_issues (
    id uuid primary key default gen_random_uuid(),
    issue_key text not null unique,
    title text not null,
    status text not null check (status in ('open','in_review','fixed','ignored','reopened')),
    priority text not null check (priority in ('low','medium','high','critical')),
    category text not null,
    fingerprint text not null,
    first_seen_at timestamptz not null,
    last_seen_at timestamptz not null,
    occurrence_count integer not null default 0 check (occurrence_count >= 0),
    severity_score numeric not null default 0 check (severity_score >= 0),
    confidence numeric not null default 0 check (confidence >= 0 and confidence <= 1),
    root_cause_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(root_cause_doc) = 'object'
        and root_cause_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    suggested_fix_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(suggested_fix_doc) = 'object'
        and suggested_fix_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    evaluator_spec_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(evaluator_spec_doc) = 'object'
        and evaluator_spec_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    dataset_spec_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(dataset_spec_doc) = 'object'
        and dataset_spec_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    linked_trace_ids text[] not null default '{}',
    linked_score_bundle_hashes text[] not null default '{}',
    linked_run_ids uuid[] not null default '{}',
    linked_ticket_ids uuid[] not null default '{}',
    created_pr_url text,
    created_candidate_id text,
    created_by_engine_version text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS public.engine_issue_events (
    id uuid primary key default gen_random_uuid(),
    issue_id uuid not null references public.engine_issues(id) on delete restrict,
    event_type text not null check (
        event_type in (
            'opened',
            'recurrence_detected',
            'status_changed',
            'diagnosis_generated',
            'evaluator_generated',
            'dataset_generated',
            'fix_generated',
            'miner_opportunity_generated',
            'notification_sent'
        )
    ),
    event_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(event_doc) = 'object'
        and event_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS public.engine_generated_evaluators (
    id uuid primary key default gen_random_uuid(),
    issue_id uuid not null references public.engine_issues(id) on delete restrict,
    name text not null,
    evaluator_type text not null check (evaluator_type in ('deterministic','llm_judge','hybrid')),
    applies_to text not null,
    code_ref text,
    rubric_doc jsonb not null default '{}'::jsonb check (jsonb_typeof(rubric_doc) = 'object'),
    test_status text not null default 'draft' check (test_status in ('draft','shadow','accepted','rejected')),
    created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS public.engine_generated_datasets (
    id uuid primary key default gen_random_uuid(),
    issue_id uuid not null references public.engine_issues(id) on delete restrict,
    dataset_kind text not null check (
        dataset_kind in (
            'redacted_trace_regression',
            'source_evidence_regression',
            'schema_failure_regression',
            'runtime_patch_regression',
            'cost_regression',
            'private_ref_regression'
        )
    ),
    dataset_doc jsonb not null default '{}'::jsonb check (
        jsonb_typeof(dataset_doc) = 'object'
        and dataset_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    langfuse_dataset_id text,
    created_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS public.engine_generated_candidates (
    id uuid primary key default gen_random_uuid(),
    issue_id uuid not null references public.engine_issues(id) on delete restrict,
    candidate_kind text not null check (
        candidate_kind in ('prompt','runtime_patch','repo_pr','evaluator_only','research_direction','miner_opportunity')
    ),
    candidate_doc jsonb not null check (
        jsonb_typeof(candidate_doc) = 'object'
        and candidate_doc::text !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|hidden_benchmark|page_content|llm_response)'
    ),
    patch_hash text,
    pr_url text,
    evaluation_status text not null default 'not_run' check (
        evaluation_status in ('not_run','queued','running','scored','failed','rejected')
    ),
    score_bundle_hash text,
    accepted boolean,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

CREATE TABLE IF NOT EXISTS public.engine_trace_mappings (
    id uuid primary key default gen_random_uuid(),
    execution_trace_ref text not null,
    langfuse_trace_id text not null,
    langfuse_project text not null,
    score_bundle_hash text,
    run_id uuid,
    ticket_id uuid,
    created_at timestamptz not null default now(),
    unique (execution_trace_ref, langfuse_project)
);

CREATE INDEX IF NOT EXISTS idx_engine_issues_status_priority
    ON public.engine_issues(status, priority, last_seen_at desc);
CREATE INDEX IF NOT EXISTS idx_engine_issues_category
    ON public.engine_issues(category, last_seen_at desc);
CREATE INDEX IF NOT EXISTS idx_engine_issue_events_issue
    ON public.engine_issue_events(issue_id, created_at desc);
CREATE INDEX IF NOT EXISTS idx_engine_trace_mappings_execution_ref
    ON public.engine_trace_mappings(execution_trace_ref);
CREATE INDEX IF NOT EXISTS idx_engine_trace_mappings_langfuse
    ON public.engine_trace_mappings(langfuse_trace_id);

CREATE OR REPLACE FUNCTION public.prevent_engine_issue_event_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION 'engine_issue_events is append-only; write a new event instead';
END;
$$;

DROP TRIGGER IF EXISTS prevent_engine_issue_events_mutation ON public.engine_issue_events;
CREATE TRIGGER prevent_engine_issue_events_mutation
    BEFORE UPDATE OR DELETE ON public.engine_issue_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_engine_issue_event_mutation();

REVOKE ALL ON TABLE public.engine_issues FROM anon, authenticated;
REVOKE ALL ON TABLE public.engine_issue_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.engine_generated_evaluators FROM anon, authenticated;
REVOKE ALL ON TABLE public.engine_generated_datasets FROM anon, authenticated;
REVOKE ALL ON TABLE public.engine_generated_candidates FROM anon, authenticated;
REVOKE ALL ON TABLE public.engine_trace_mappings FROM anon, authenticated;

GRANT SELECT, INSERT, UPDATE ON TABLE public.engine_issues TO service_role;
GRANT SELECT, INSERT ON TABLE public.engine_issue_events TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.engine_generated_evaluators TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.engine_generated_datasets TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.engine_generated_candidates TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.engine_trace_mappings TO service_role;

ALTER TABLE public.engine_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.engine_issue_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.engine_generated_evaluators ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.engine_generated_datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.engine_generated_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.engine_trace_mappings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all ON public.engine_issues;
CREATE POLICY service_role_all ON public.engine_issues
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_all ON public.engine_issue_events;
CREATE POLICY service_role_all ON public.engine_issue_events
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_all ON public.engine_generated_evaluators;
CREATE POLICY service_role_all ON public.engine_generated_evaluators
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_all ON public.engine_generated_datasets;
CREATE POLICY service_role_all ON public.engine_generated_datasets
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_all ON public.engine_generated_candidates;
CREATE POLICY service_role_all ON public.engine_generated_candidates
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_all ON public.engine_trace_mappings;
CREATE POLICY service_role_all ON public.engine_trace_mappings
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE public.engine_issues IS
    'Private advisory Improvement Engine issues. Not reward canonical.';
COMMENT ON TABLE public.engine_trace_mappings IS
    'Mapping between canonical execution_trace_ref and redacted Langfuse traces.';

COMMIT;
