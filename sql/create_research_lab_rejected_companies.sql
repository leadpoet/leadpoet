-- Sourced-but-rejected companies: companies the private model SOURCED but the
-- harness scorer REJECTED (final_score 0 / failed a gate). Persisted for later
-- false-rejection ("fake rejection" / false-negative) analysis — e.g. companies
-- zeroed on a soft, non-deterministic gate (stage / intent) that the model was
-- confident about, vs genuine hard structural mismatches.
--
-- Written best-effort by the research-lab scorer at the score_with_breakdowns
-- boundary (baseline AND candidate paths); never blocks scoring. Read-only
-- analysis target.
--
-- Dedup: each unique rejection stored ONCE via dedup_key = sha256(icp_hash,
-- normalized company_name, failure_reason). SAME company + SAME error + SAME ICP
-- collapses to one row across daily re-runs AND across baseline/candidate. A
-- different failure_reason on the same company is a distinct row.

CREATE TABLE IF NOT EXISTS research_lab_rejected_companies (
    id                   BIGSERIAL PRIMARY KEY,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- run / model context
    context_ref          TEXT NOT NULL,          -- benchmark / candidate run context
    is_reference_model   BOOLEAN,                -- TRUE = daily baseline, FALSE = candidate
    candidate_id         TEXT,                   -- which candidate (NULL for baseline)
    model_manifest_hash  TEXT,                   -- model version that produced it
    icp_ref              TEXT NOT NULL,          -- ICP name (NOT unique per row — reused across runs)
    icp_hash             TEXT,                   -- stable ICP content hash (dedup key input)

    -- company identity (what the model sourced)
    company_name         TEXT,
    company_website      TEXT,
    company_linkedin     TEXT,
    industry             TEXT,
    sub_industry         TEXT,
    employee_count       TEXT,
    city                 TEXT,
    state                TEXT,
    country              TEXT,

    -- harness scoring outcome (why it was rejected)
    final_score          DOUBLE PRECISION,
    failure_reason       TEXT,                   -- e.g. company_stage_mismatch, intent_fabricated
    failure_stage        TEXT,                   -- gate that failed: pre_checks / attribute / intent
    fit_passed           BOOLEAN,
    attribute_passed     BOOLEAN,
    intent_passed        BOOLEAN,
    icp_fit              DOUBLE PRECISION,
    intent_signal        DOUBLE PRECISION,
    captured_at          TIMESTAMPTZ,

    -- dedup: sha256 of icp_hash | normalized company_name | failure_reason,
    -- computed by the gateway. SAME company + SAME error + SAME icp => one row
    -- (baseline and candidate collapse together; re-runs never duplicate).
    dedup_key            TEXT NOT NULL,
    CONSTRAINT uq_rlrc_dedup UNIQUE (dedup_key)
);

CREATE INDEX IF NOT EXISTS idx_rlrc_icp        ON research_lab_rejected_companies (icp_ref);
CREATE INDEX IF NOT EXISTS idx_rlrc_reason     ON research_lab_rejected_companies (failure_reason);
CREATE INDEX IF NOT EXISTS idx_rlrc_created    ON research_lab_rejected_companies (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rlrc_refmodel   ON research_lab_rejected_companies (is_reference_model);
CREATE INDEX IF NOT EXISTS idx_rlrc_candidate  ON research_lab_rejected_companies (candidate_id);

-- Example false-rejection audit (soft-gate rejections where fit looked OK):
--   SELECT icp_ref, company_name, company_linkedin, failure_reason, icp_fit, intent_signal
--   FROM research_lab_rejected_companies
--   WHERE failure_reason IN ('company_stage_mismatch','intent_fabricated','hallucinated_or_generic_intent')
--     AND icp_fit >= 0.5
--   ORDER BY created_at DESC;
