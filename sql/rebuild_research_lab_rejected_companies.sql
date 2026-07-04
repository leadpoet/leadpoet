-- Rebuild research_lab_rejected_companies with (a) model-claim + decay columns
-- for false-rejection analysis and (b) a readable column order:
--   run context -> company identity -> MODEL claims -> HARNESS outcome -> dedup.
--
-- Prod-safe: single transaction (create shadow -> copy -> drop old -> rename),
-- writers insert by column NAME via PostgREST so the reordered superset stays
-- compatible; RLS + unique dedup constraint + indexes are recreated.

BEGIN;

CREATE TABLE research_lab_rejected_companies_rebuild (
    -- ── run context ─────────────────────────────────────────────
    id                    BIGSERIAL PRIMARY KEY,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    captured_at           TIMESTAMPTZ,
    context_ref           TEXT NOT NULL,          -- benchmark / candidate run context
    is_reference_model    BOOLEAN,                -- TRUE = daily baseline, FALSE = candidate
    candidate_id          TEXT,                   -- which candidate (NULL for baseline)
    model_manifest_hash   TEXT,                   -- model version that produced it
    icp_ref               TEXT NOT NULL,          -- ICP name (reused across runs)
    icp_hash              TEXT,                   -- stable ICP content hash (dedup input)

    -- ── company identity (what the model sourced) ───────────────
    company_name          TEXT,
    company_website       TEXT,
    company_linkedin      TEXT,
    industry              TEXT,
    sub_industry          TEXT,
    employee_count        TEXT,
    company_stage         TEXT,
    city                  TEXT,
    state                 TEXT,
    country               TEXT,

    -- ── MODEL claims (what the model asserted; audit vs harness) ─
    model_claimed_score   DOUBLE PRECISION,       -- the model's own score for this company
    intent_source         TEXT,                   -- fetch source (news / linkedin / ...)
    intent_claimed_signal TEXT,                   -- which ICP signal the model claims it matched
    intent_evidence_url   TEXT,                   -- evidence URL the model cited
    intent_evidence_date  TEXT,                   -- signal date the model reported (raw string)
    attribute_evidence_url TEXT,                  -- required-attribute evidence URL

    -- ── HARNESS outcome (why it was rejected) ────────────────────
    final_score           DOUBLE PRECISION,
    failure_reason        TEXT,
    failure_stage         TEXT,                   -- gate that failed: pre_checks / attribute / intent
    fit_passed            BOOLEAN,
    attribute_passed      BOOLEAN,
    intent_passed         BOOLEAN,
    icp_fit               DOUBLE PRECISION,       -- 0-40 fit component
    intent_signal_raw     DOUBLE PRECISION,       -- intent BEFORE time decay
    time_decay_multiplier DOUBLE PRECISION,       -- 1.0 / 0.5 / 0.25 (fresh / aging / stale)
    intent_signal         DOUBLE PRECISION,       -- intent AFTER decay (final)

    -- ── dedup: sha256(icp_hash | normalized company_name | failure_reason)
    --    SAME company + SAME error + SAME icp => one row (across re-runs AND
    --    across baseline/candidate).
    dedup_key             TEXT NOT NULL,
    CONSTRAINT uq_rlrc_dedup_rebuild UNIQUE (dedup_key)
);

INSERT INTO research_lab_rejected_companies_rebuild (
    created_at, captured_at, context_ref, is_reference_model, candidate_id,
    model_manifest_hash, icp_ref, icp_hash,
    company_name, company_website, company_linkedin, industry, sub_industry,
    employee_count, company_stage, city, state, country,
    final_score, failure_reason, failure_stage, fit_passed, attribute_passed,
    intent_passed, icp_fit, intent_signal, dedup_key
)
SELECT
    created_at, captured_at, context_ref, is_reference_model, candidate_id,
    model_manifest_hash, icp_ref, icp_hash,
    company_name, company_website, company_linkedin, industry, sub_industry,
    employee_count, company_stage, city, state, country,
    final_score, failure_reason, failure_stage, fit_passed, attribute_passed,
    intent_passed, icp_fit, intent_signal, dedup_key
FROM research_lab_rejected_companies;

DROP TABLE research_lab_rejected_companies;
ALTER TABLE research_lab_rejected_companies_rebuild RENAME TO research_lab_rejected_companies;
ALTER TABLE research_lab_rejected_companies RENAME CONSTRAINT uq_rlrc_dedup_rebuild TO uq_rlrc_dedup;

CREATE INDEX idx_rlrc_icp        ON research_lab_rejected_companies (icp_ref);
CREATE INDEX idx_rlrc_reason     ON research_lab_rejected_companies (failure_reason);
CREATE INDEX idx_rlrc_created    ON research_lab_rejected_companies (created_at DESC);
CREATE INDEX idx_rlrc_refmodel   ON research_lab_rejected_companies (is_reference_model);
CREATE INDEX idx_rlrc_candidate  ON research_lab_rejected_companies (candidate_id);

ALTER TABLE research_lab_rejected_companies ENABLE ROW LEVEL SECURITY;

COMMIT;

-- Example false-rejection audit — model was confident, harness zeroed it:
--   SELECT company_name, model_claimed_score, final_score, failure_reason,
--          intent_evidence_url, intent_evidence_date, time_decay_multiplier
--   FROM research_lab_rejected_companies
--   WHERE model_claimed_score >= 50 AND final_score = 0
--   ORDER BY model_claimed_score DESC;
