-- Source experiments & SOURCE_ADD rollout bundle (sourceexperiments.md §2 SQL).
--
-- One additive bundle:
--   1. provider registry audit table (W3) — hash-audited registry snapshots;
--   2. proxy usage ledger (W3) — per-call provider usage with caller attribution;
--   3. SOURCE_ADD submissions funnel + source catalog (W5) — append-only;
--   4. SOURCE_ADD reward obligations + events + current view (W6) — mirrors
--      the champion-reward append-only constraints; rewards ride the existing
--      champion allocation rails with a reward_kind label;
--   5. loop-events event_type CHECK extended with the W4 probe events.
--
-- Everything is additive/append-only; deploy is behavior-inert until the
-- activation flags flip (sourceexperiments.md §5.3).

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Provider registry audit (W3). The proxy loads its registry from file/env;
--    each distinct registry content is snapshotted here for audit.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.research_lab_provider_registry (
    registry_snapshot_id UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version       TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    registry_hash        TEXT        NOT NULL UNIQUE CHECK (registry_hash ~ '^sha256:[0-9a-f]{64}$'),
    provider_count       INTEGER     NOT NULL CHECK (provider_count > 0),
    registry_doc         JSONB       NOT NULL CHECK (
                                       jsonb_typeof(registry_doc) = 'object'
                                       AND registry_doc::TEXT !~* '(sk-or-|api_key"\s*:\s*"[a-z0-9]|raw_secret|password)'
                                     ),
    created_by           TEXT        NOT NULL DEFAULT '',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_provider_registry_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_provider_registry is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_research_lab_provider_registry_no_mutation
    ON public.research_lab_provider_registry;
CREATE TRIGGER trg_research_lab_provider_registry_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_provider_registry
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_provider_registry_mutation();

-- ---------------------------------------------------------------------------
-- 2. Proxy usage ledger (W3): per-call provider usage with caller attribution.
--    The proxy appends JSONL locally; this table is the durable projection.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.research_lab_provider_usage_ledger (
    usage_row_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version      TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    utc_day             TEXT        NOT NULL CHECK (utc_day ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'),
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider_id         TEXT        NOT NULL,
    endpoint_class      TEXT        NOT NULL DEFAULT '' CHECK (endpoint_class !~* '[?&=]'),
    request_fingerprint TEXT        NOT NULL DEFAULT '',
    evidence            TEXT        NOT NULL CHECK (
                                      evidence IN (
                                        'hit', 'recorded', 'error', 'blocked',
                                        'quota_exhausted', 'credential_missing', 'replay_miss'
                                      )
                                    ),
    status              INTEGER     NOT NULL DEFAULT 0,
    est_cost_microusd   BIGINT      NOT NULL DEFAULT 0 CHECK (est_cost_microusd >= 0),
    caller_doc          JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                      jsonb_typeof(caller_doc) = 'object'
                                      AND caller_doc::TEXT !~* '(sk-or-|raw_secret|password|api_key"\s*:\s*"[a-z0-9])'
                                    )
);

CREATE INDEX IF NOT EXISTS idx_research_lab_provider_usage_day_provider
    ON public.research_lab_provider_usage_ledger (utc_day, provider_id);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_provider_usage_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_provider_usage_ledger is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_research_lab_provider_usage_no_mutation
    ON public.research_lab_provider_usage_ledger;
CREATE TRIGGER trg_research_lab_provider_usage_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_provider_usage_ledger
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_provider_usage_mutation();

-- ---------------------------------------------------------------------------
-- 3. SOURCE_ADD submissions funnel + source catalog (W5).
--    Submissions are append-only per funnel transition; the current view takes
--    the newest stage row per submission.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_submissions (
    submission_row_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version      TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    submission_id       TEXT        NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id          TEXT        NOT NULL,
    miner_hotkey        TEXT        NOT NULL,
    stage               TEXT        NOT NULL CHECK (
                                      stage IN (
                                        'submitted', 'manifest_validated', 'static_scan_passed',
                                        'llm_review_passed', 'trial_completed', 'accepted', 'rejected'
                                      )
                                    ),
    seq                 INTEGER     NOT NULL CHECK (seq >= 0),
    measured_trial_yield NUMERIC(10, 6),
    submission_doc      JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                      jsonb_typeof(submission_doc) = 'object'
                                      -- Credential envelopes are KMS ciphertext refs only;
                                      -- raw key material must never land here.
                                      AND submission_doc::TEXT !~* '(raw_credential|"password"|sk-or-)'
                                    ),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (submission_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_source_add_submissions_hotkey
    ON public.research_lab_source_add_submissions (miner_hotkey, created_at DESC);

CREATE OR REPLACE VIEW public.research_lab_source_add_submission_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    *
FROM public.research_lab_source_add_submissions
ORDER BY submission_id, seq DESC, created_at DESC;

CREATE TABLE IF NOT EXISTS public.research_lab_source_catalog (
    catalog_id            TEXT        PRIMARY KEY CHECK (catalog_id ~ '^source_catalog:[0-9a-f]{16}$'),
    schema_version        TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    adapter_id            TEXT        NOT NULL UNIQUE,
    miner_ref             TEXT        NOT NULL,
    source_name           TEXT        NOT NULL,
    source_kind           TEXT        NOT NULL CHECK (
                                        source_kind IN ('web', 'filing', 'news', 'registry', 'procurement', 'social')
                                      ),
    declared_base_domains JSONB       NOT NULL CHECK (jsonb_typeof(declared_base_domains) = 'array'),
    registry_provider_id  TEXT        NOT NULL DEFAULT '',
    measured_trial_yield  NUMERIC(10, 6) NOT NULL CHECK (measured_trial_yield >= 0),
    accepted_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    catalog_doc           JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (jsonb_typeof(catalog_doc) = 'object')
);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_source_catalog_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_source_catalog is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_research_lab_source_catalog_no_mutation
    ON public.research_lab_source_catalog;
CREATE TRIGGER trg_research_lab_source_catalog_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_catalog
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_catalog_mutation();

-- ---------------------------------------------------------------------------
-- 4. SOURCE_ADD reward obligations + events (W6), mirroring the
--    champion-reward append-only pattern. Rows surface into the allocation
--    build as champion-type obligations with a reward_kind label, so the
--    validator pays them with zero code change.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_reward_obligations (
    reward_ref            TEXT        PRIMARY KEY CHECK (reward_ref ~ '^source_add_reward:[0-9a-f]{16}$'),
    schema_version        TEXT        NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    adapter_id            TEXT        NOT NULL,
    catalog_id            TEXT        REFERENCES public.research_lab_source_catalog(catalog_id)
                                      ON DELETE RESTRICT,
    miner_hotkey          TEXT        NOT NULL,
    leg                   INTEGER     NOT NULL CHECK (leg IN (1, 2)),
    reward_kind           TEXT        NOT NULL CHECK (
                                        reward_kind IN ('source_acceptance', 'source_implementation')
                                      ),
    alpha_percent         NUMERIC(10, 6) NOT NULL CHECK (alpha_percent > 0 AND alpha_percent <= 100),
    reward_epochs         INTEGER     NOT NULL CHECK (reward_epochs > 0),
    start_epoch           INTEGER     NOT NULL CHECK (start_epoch >= 0),
    trigger_evidence_doc  JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                        jsonb_typeof(trigger_evidence_doc) = 'object'
                                      ),
    public_label          TEXT        NOT NULL DEFAULT '',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Leg 2 is never created without the shadow-window + ablation evidence.
    CHECK (
        leg = 1
        OR (
            (trigger_evidence_doc ->> 'shadow_window_passed')::BOOLEAN IS TRUE
            AND (trigger_evidence_doc ->> 'ablation_passed')::BOOLEAN IS TRUE
        )
    ),
    -- One leg-1 and one leg-2 per adapter, ever.
    UNIQUE (adapter_id, leg)
);

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_reward_events (
    event_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    reward_ref    TEXT        NOT NULL
                              REFERENCES public.research_lab_source_add_reward_obligations(reward_ref)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    reward_status TEXT        NOT NULL CHECK (
                                reward_status IN ('active', 'queued', 'partially_paid', 'stopped_forward')
                              ),
    reason        TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (reward_ref, seq)
);

CREATE OR REPLACE VIEW public.research_lab_source_add_reward_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    e.seq AS current_event_seq,
    e.reward_status AS current_reward_status,
    e.reason AS current_reason,
    e.created_at AS current_status_at,
    -- Column-compat with the champion obligation reader.
    r.alpha_percent AS desired_alpha_percent,
    r.reward_epochs AS epoch_count
FROM public.research_lab_source_add_reward_obligations r
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_source_add_reward_events e
    WHERE e.reward_ref = r.reward_ref
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_source_add_reward_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;

DROP TRIGGER IF EXISTS trg_research_lab_source_add_reward_obligations_no_mutation
    ON public.research_lab_source_add_reward_obligations;
CREATE TRIGGER trg_research_lab_source_add_reward_obligations_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_reward_obligations
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_reward_mutation();

DROP TRIGGER IF EXISTS trg_research_lab_source_add_reward_events_no_mutation
    ON public.research_lab_source_add_reward_events;
CREATE TRIGGER trg_research_lab_source_add_reward_events_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_reward_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_reward_mutation();

-- ---------------------------------------------------------------------------
-- 5. Loop events: extend the event_type CHECK (scripts/69 list) with the W4
--    probe events.
-- ---------------------------------------------------------------------------

ALTER TABLE public.research_lab_auto_research_loop_events
    DROP CONSTRAINT IF EXISTS research_lab_auto_research_loop_events_event_type_check;

ALTER TABLE public.research_lab_auto_research_loop_events
    ADD CONSTRAINT research_lab_auto_research_loop_events_event_type_check
    CHECK (
        event_type IN (
            'loop_started',
            'loop_resumed',
            'hypothesis_drafted',
            'patch_drafted',
            'patch_validation_passed',
            'patch_validation_failed',
            'dev_check_passed',
            'dev_check_failed',
            'reflection_recorded',
            'checkpoint_saved',
            'loop_paused',
            'candidate_selected',
            'loop_completed',
            'loop_failed',
            'code_edit_drafted',
            'code_edit_validation_passed',
            'code_edit_validation_failed',
            'candidate_build_started',
            'candidate_build_passed',
            'candidate_build_failed',
            'source_inspection_requested',
            'source_inspection_resolved',
            'source_inspection_failed',
            'code_edit_repair_requested',
            'code_edit_repair_drafted',
            'code_edit_repair_failed',
            'candidate_patch_apply_failed',
            'candidate_patch_parse_failed',
            'candidate_patch_empty_or_noop',
            'candidate_test_failed',
            'candidate_patch_test_failed',
            'candidate_image_build_failed',
            'candidate_artifact_missing',
            'candidate_repair_exhausted',
            'candidate_generation_fallback_requested',
            'candidate_generation_fallback_drafted',
            'candidate_generation_fallback_failed',
            'loop_direction_planned',
            'plan_alignment_judged',
            'code_edit_alignment_rejected',
            'duplicate_candidate_reused',
            'no_viable_patch',
            'allocator_decision',
            -- W4 provider probes (sourceexperiments.md).
            'probe_requested',
            'probe_resolved',
            'probe_blocked'
        )
    );

COMMENT ON TABLE public.research_lab_provider_registry IS
    'Append-only hash-audited snapshots of the evidence-proxy provider routing registry (W3).';
COMMENT ON TABLE public.research_lab_provider_usage_ledger IS
    'Append-only per-call provider usage rows with spawn-bound caller attribution (W3). No bodies or query strings.';
COMMENT ON TABLE public.research_lab_source_add_submissions IS
    'Append-only SOURCE_ADD funnel transitions (submitted → … → accepted/rejected) with sanitized submission docs (W5).';
COMMENT ON TABLE public.research_lab_source_catalog IS
    'Accepted SOURCE_ADD adapters: the graduated source catalog (W5).';
COMMENT ON TABLE public.research_lab_source_add_reward_obligations IS
    'Append-only SOURCE_ADD reward legs (1% credible-source submission / +5% implementation rider × 20 epochs) riding the champion allocation rails with reward_kind labels (W6).';

COMMIT;

-- Verification:
-- SELECT COUNT(*) FROM public.research_lab_provider_registry;
-- SELECT COUNT(*) FROM public.research_lab_provider_usage_ledger;
-- SELECT COUNT(*) FROM public.research_lab_source_add_submission_current;
-- SELECT COUNT(*) FROM public.research_lab_source_add_reward_current;
-- SELECT pg_get_constraintdef(oid) FROM pg_constraint
--   WHERE conname = 'research_lab_auto_research_loop_events_event_type_check';
