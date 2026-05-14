-- Migration 18: add deep research analysis columns to fulfillment_requests
--
-- WHY:
--   When a fulfillment chain reaches status='fulfilled' (all winning leads
--   delivered for the chain's quota), the admin dashboard runs a final
--   QA pass that reviews every delivered lead against the original ICP.
--   That pass is invoked entirely from the admin panel (subnet71.com/admin)
--   using Perplexity Sonar Deep Research via OpenRouter — the gateway
--   does NOT call OpenRouter for this feature. The gateway's only role
--   is to flip status to 'fulfilled' (which it already does today); these
--   columns give the admin panel somewhere to persist its analysis.
--
-- STORED ON:
--   The chain LEAF (the request_id row that carries status='fulfilled').
--   Chain consumers walking from any row to the leaf already follow the
--   existing successor_request_id pointers, so the dashboard's request-
--   detail loader can read from the leaf without schema changes.
--
-- COLUMNS:
--   deep_research_analysis     JSONB   -- structured output (see SHAPE)
--   deep_research_status       TEXT    -- NULL | pending | in_progress | completed | failed
--   deep_research_attempts     INT     -- retry counter (3-strikes then 'failed')
--   deep_research_error        TEXT    -- last error message (for the UI's failure state)
--   deep_research_started_at   TIMESTAMPTZ  -- when in_progress claim was taken
--                                          -- (used to recover stranded claims after a restart)
--   deep_research_generated_at TIMESTAMPTZ  -- when 'completed' was last written
--
-- STATE MACHINE (driven by the admin panel, not the gateway):
--   NULL -> pending          (admin panel queues when first viewing a fulfilled chain)
--   pending -> in_progress   (admin panel claims a row before calling OpenRouter)
--   in_progress -> completed (LLM call succeeded; analysis JSON persisted)
--   in_progress -> pending   (LLM call failed AND attempts < 3; retry next view)
--   in_progress -> failed    (LLM call failed AND attempts >= 3; UI shows retry button)
--
--   Manual re-run from the dashboard resets: status='pending', attempts=0,
--   error=NULL, analysis=NULL.
--
-- SHAPE:
--   {
--     "summary": {
--       "total_reviewed": 10,
--       "client_ready": 7,
--       "needs_edit": 2,
--       "needs_re_research": 1,
--       "remove": 0,
--       "top_issues": ["...", "..."],
--       "recommended_delivery_decision": "Deliver with edits"
--     },
--     "leads": [
--       {
--         "company": "Acme Corp",
--         "contact": "Jane Doe",
--         "icp_fit": "Strong",        -- Strong | Borderline | Poor
--         "intent_fit": "Strong",
--         "data_confidence": "High",  -- High | Medium | Low
--         "final_status": "Client Ready",  -- Client Ready | Needs Edit | Needs Re-Research | Remove
--         "reasoning": "...",
--         "data_issues_found": "...",
--         "recommended_fix": "..."
--       }
--     ],
--     "model": "perplexity/sonar-deep-research",
--     "raw_response": "...",          -- LLM output verbatim (for audit)
--     "icp_snapshot": {...}           -- ICP at analysis time, for staleness detection
--   }

ALTER TABLE fulfillment_requests
    ADD COLUMN IF NOT EXISTS deep_research_analysis JSONB,
    ADD COLUMN IF NOT EXISTS deep_research_status TEXT,
    ADD COLUMN IF NOT EXISTS deep_research_attempts INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS deep_research_error TEXT,
    ADD COLUMN IF NOT EXISTS deep_research_started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS deep_research_generated_at TIMESTAMPTZ;

-- Partial index speeds up admin panel polling for in-flight runs.
-- WHERE clause keeps the index tiny (most rows have status=NULL).
CREATE INDEX IF NOT EXISTS idx_fulfillment_requests_deep_research_pending
    ON fulfillment_requests (deep_research_status, deep_research_attempts)
    WHERE deep_research_status = 'pending'
       OR deep_research_status = 'in_progress';

COMMENT ON COLUMN fulfillment_requests.deep_research_analysis IS
    'Structured QA analysis from Perplexity Sonar Deep Research (via OpenRouter), '
    'generated on demand by the admin panel when the chain is fulfilled. The '
    'gateway never reads or writes this column. Shape: {summary, leads[], '
    'model, raw_response, icp_snapshot}.';

COMMENT ON COLUMN fulfillment_requests.deep_research_status IS
    'State machine for the admin panel''s deep research worker. NULL=not '
    'yet eligible (chain not fulfilled, or never viewed in admin); '
    'pending=admin panel queued it; in_progress=admin panel is actively '
    'calling OpenRouter; completed=analysis JSON present; failed=3 '
    'attempts exhausted, UI shows retry button.';
