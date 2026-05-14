-- Migration 17: role_translations cache table
--
-- WHY:
--   The Stage-4 LinkedIn role compare in
--   validator_models/fulfillment_person_verification.py compares an
--   Apify-returned LinkedIn `position` to the miner's submitted role.
--   When the LinkedIn profile is in a non-English locale Apify returns
--   the role in that locale ("Diretora de Tecnologia", "首席技术官",
--   "Director Comercial"), so the English-only exact + LLM compare
--   falsely rejects an otherwise-valid lead.
--
--   The pipeline now calls the gateway endpoint POST /fulfillment/
--   translate-role on miss, gets the English equivalent, and re-runs
--   the compare on the translated string.  This table is the L2 cache
--   behind that endpoint, so (a) DeepL is paid once per distinct
--   normalized role globally and (b) every validator in the fleet
--   learns from a single first call.
--
-- ACCESS PATTERN:
--   Hot read (every gateway translate-role L1 miss):
--     SELECT translated_en FROM role_translations
--      WHERE role_normalized = $1 AND verified = TRUE LIMIT 1;
--   Served by the implicit PK B-tree index — O(log n) lookup.
--
--   Write (only on L2 miss + successful DeepL call):
--     INSERT ... ON CONFLICT (role_normalized) DO UPDATE
--     -- via PostgREST: Prefer: resolution=merge-duplicates
--
-- COLUMNS:
--   role_normalized   primary key.  Lowercased, NFKD-stripped, whitespace-
--                     collapsed form of the role string.  Distinct casings
--                     and diacritic variants collapse into one row.
--   translated_en     DeepL's English equivalent.  For inputs DeepL already
--                     sees as English, this is the input echoed back.
--   src_lang          DeepL's detected_source_language (ISO 2-letter code).
--                     Nullable; analytics only.
--   verified          Operator-controlled override.  Flip to FALSE to
--                     invalidate a bad translation; next request retranslates.
--   translated_at     When the row was first written.  Never updated.
--   last_used_at      Reserved for future analytics.  NOT updated on read
--                     today — would be a write-amplification cost.
--   hit_count         Reserved for future analytics.  NOT incremented today.

CREATE TABLE IF NOT EXISTS role_translations (
    role_normalized TEXT PRIMARY KEY,
    translated_en   TEXT NOT NULL,
    src_lang        TEXT,
    verified        BOOLEAN DEFAULT TRUE,
    translated_at   TIMESTAMPTZ DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ DEFAULT NOW(),
    hit_count       INTEGER DEFAULT 0
);

COMMENT ON TABLE role_translations IS
    'DeepL-backed role translation cache.  L2 cache for the gateway '
    'endpoint POST /fulfillment/translate-role.  Read on every L1 miss; '
    'written only on DeepL success.';
