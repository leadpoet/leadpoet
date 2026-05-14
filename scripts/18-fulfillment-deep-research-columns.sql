-- Migration 18: add deep research analysis columns to fulfillment_requests
--
-- WHY:
--   When a fulfillment chain reaches status='fulfilled' (all winning leads
--   delivered for the chain's quota), the admin dashboard needs a final QA
--   pass that reviews every delivered lead against the original ICP. This
--   pass uses Perplexity Sonar Deep Research via OpenRouter, which does
--   multi-step web verification — far more thorough than the per-signal LLM
--   we already run on submit, since this run is allowed to confirm
--   information against the live web. The output is a structured JSON
--   blob (summary + per-lead findings) that the dashboard renders as a
--   "Deep Research Analysis" tab so the operator can decide whether the
--   chain is safe to deliver as-is.
--
--   We store the analysis on the chain LEAF (the request_id row that
--   carries status='fulfilled'). Chain consumers walking from any row to
--   the leaf already follow the existing successor_request_id pointers,
--   so the dashboard's request-detail loader can read from the leaf
--   without schema changes.
--
-- COLUMNS:
--   deep_research_analysis     JSONB   -- structured output (see SHAPE)
--   deep_research_status       TEXT    -- NULL | pending | in_progress | completed | failed
--   deep_research_attempts     INT     -- retry counter (3-strikes then 'failed')
--   deep_research_error        TEXT    -- last error message (for the UI's failure state)
--   deep_research_started_at   TIMESTAMPTZ  -- when in_progress claim was taken
--                                          -- (used to recover stranded claims after gateway restart)
--   deep_research_generated_at TIMESTAMPTZ  -- when 'completed' was last written
--
-- STATE MACHINE:
--   NULL -> pending          (set when status flips to 'fulfilled')
--   pending -> in_progress   (sweep claims a row; deep_research_attempts += 1)
--   in_progress -> completed (LLM call succeeded; analysis JSON persisted)
--   in_progress -> pending   (LLM call failed AND attempts < 3; retry next sweep)
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

-- Partial index speeds up the sweep query (pending + retryable rows).
-- WHERE clause keeps the index tiny (most rows have status=NULL).
CREATE INDEX IF NOT EXISTS idx_fulfillment_requests_deep_research_pending
    ON fulfillment_requests (deep_research_status, deep_research_attempts)
    WHERE deep_research_status = 'pending'
       OR deep_research_status = 'in_progress';

COMMENT ON COLUMN fulfillment_requests.deep_research_analysis IS
    'Structured QA analysis from Perplexity Sonar Deep Research (via OpenRouter), '
    'generated once per chain when status flips to fulfilled. Shape: {summary, '
    'leads[], model, raw_response, icp_snapshot}. Rendered by the dashboards '
    '"Deep Research Analysis" tab. Manual re-run from the UI resets this '
    'column to NULL so the next sweep regenerates it.';

COMMENT ON COLUMN fulfillment_requests.deep_research_status IS
    'State machine for the deep research analysis worker. NULL=not yet '
    'eligible (chain not fulfilled); pending=queued for the next sweep; '
    'in_progress=actively running an LLM call; completed=analysis JSON '
    'present; failed=3 attempts exhausted, UI shows retry button.';
