-- Research Lab Phase 1: deterministic Lab emission allocator state.
--
-- Deployment policy:
--   * Apply after scripts 27, 28, 29, 30, 31, 32, 33, and 34.
--   * Stores champion reward obligations and per-epoch Lab allocation snapshots.
--   * Does not activate paid loops, reimbursements, crowning, fulfillment writes,
--     or weight mutation.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, or raw provider secrets may be stored here.

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

CREATE TABLE IF NOT EXISTS public.research_lab_champion_reward_obligations (
    champion_reward_id          TEXT        PRIMARY KEY
                                            CHECK (champion_reward_id ~ '^champion_reward:sha256:[0-9a-f]{64}$'),
    schema_version              TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    score_bundle_id             TEXT        REFERENCES public.research_evaluation_score_bundles(score_bundle_id)
                                            ON DELETE RESTRICT,
    candidate_id                TEXT        REFERENCES public.research_lab_candidate_artifacts(candidate_id)
                                            ON DELETE RESTRICT,
    run_id                      UUID        NOT NULL,
    ticket_id                   UUID        REFERENCES public.research_loop_tickets(ticket_id)
                                            ON DELETE RESTRICT,
    miner_hotkey                TEXT        NOT NULL,
    miner_uid                   INTEGER     NOT NULL CHECK (miner_uid >= 0),
    island                      TEXT        NOT NULL,
    policy_id                   TEXT        NOT NULL,
    evaluation_epoch            INTEGER     NOT NULL CHECK (evaluation_epoch >= 0),
    start_epoch                 INTEGER     NOT NULL CHECK (start_epoch >= 0),
    epoch_count                 INTEGER     NOT NULL CHECK (epoch_count > 0),
    improvement_points          NUMERIC(14, 6) NOT NULL,
    threshold_points            NUMERIC(14, 6) NOT NULL,
    desired_alpha_percent       NUMERIC(10, 6) NOT NULL CHECK (
                                            desired_alpha_percent >= 0
                                            AND desired_alpha_percent <= 100
                                            ),
    source_score_bundle_hash    TEXT        CHECK (
                                            source_score_bundle_hash IS NULL
                                            OR source_score_bundle_hash ~ '^sha256:[0-9a-f]{64}$'
                                            ),
    input_hash                  TEXT        NOT NULL UNIQUE CHECK (input_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash               TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    obligation_doc              JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(obligation_doc) = 'object'
                                            AND obligation_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                            ),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (improvement_points >= threshold_points)
);

CREATE TABLE IF NOT EXISTS public.research_lab_champion_reward_events (
    event_id              UUID        PRIMARY KEY,
    schema_version        TEXT        NOT NULL DEFAULT '1.0'
                                      CHECK (schema_version = '1.0'),
    champion_reward_id    TEXT        NOT NULL
                                      REFERENCES public.research_lab_champion_reward_obligations(champion_reward_id)
                                      ON DELETE RESTRICT,
    seq                   INTEGER     NOT NULL CHECK (seq >= 0),
    event_type            TEXT        NOT NULL CHECK (
                                      event_type IN (
                                          'active',
                                          'queued',
                                          'partially_paid',
                                          'paid',
                                          'voided',
                                          'tombstoned'
                                      )),
    reward_status         TEXT        NOT NULL CHECK (
                                      reward_status IN (
                                          'active',
                                          'queued',
                                          'partially_paid',
                                          'paid',
                                          'voided',
                                          'tombstoned'
                                      )),
    reason                TEXT,
    anchored_hash         TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc             JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                      jsonb_typeof(event_doc) = 'object'
                                      AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                      ),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_champion_reward_events_seq_key UNIQUE (champion_reward_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_lab_emission_allocation_snapshots (
    allocation_id                  TEXT        PRIMARY KEY
                                                CHECK (allocation_id ~ '^lab_allocation:sha256:[0-9a-f]{64}$'),
    schema_version                 TEXT        NOT NULL DEFAULT '1.0'
                                                CHECK (schema_version = '1.0'),
    epoch                          INTEGER     NOT NULL CHECK (epoch >= 0),
    netuid                         INTEGER     NOT NULL CHECK (netuid > 0),
    policy_id                      TEXT        NOT NULL,
    snapshot_status                TEXT        NOT NULL CHECK (
                                                snapshot_status IN (
                                                    'shadow',
                                                    'candidate',
                                                    'active',
                                                    'tombstoned'
                                                )),
    lab_cap_alpha_percent          NUMERIC(10, 6) NOT NULL CHECK (
                                                lab_cap_alpha_percent >= 0
                                                AND lab_cap_alpha_percent <= 100
                                                ),
    reimbursement_alpha_percent    NUMERIC(10, 6) NOT NULL CHECK (
                                                reimbursement_alpha_percent >= 0
                                                AND reimbursement_alpha_percent <= 100
                                                ),
    champion_alpha_percent         NUMERIC(10, 6) NOT NULL CHECK (
                                                champion_alpha_percent >= 0
                                                AND champion_alpha_percent <= 100
                                                ),
    queued_champion_alpha_percent  NUMERIC(10, 6) NOT NULL CHECK (
                                                queued_champion_alpha_percent >= 0
                                                AND queued_champion_alpha_percent <= 100
                                                ),
    unallocated_alpha_percent      NUMERIC(10, 6) NOT NULL CHECK (
                                                unallocated_alpha_percent >= 0
                                                AND unallocated_alpha_percent <= 100
                                                ),
    input_hash                     TEXT        NOT NULL CHECK (input_hash ~ '^sha256:[0-9a-f]{64}$'),
    allocation_hash                TEXT        NOT NULL UNIQUE CHECK (allocation_hash ~ '^sha256:[0-9a-f]{64}$'),
    allocation_doc                 JSONB       NOT NULL CHECK (
                                                jsonb_typeof(allocation_doc) = 'object'
                                                AND allocation_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext)'
                                                ),
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        reimbursement_alpha_percent
        + champion_alpha_percent
        + queued_champion_alpha_percent
        + unallocated_alpha_percent
        <= lab_cap_alpha_percent + 0.000001
    )
);

CREATE INDEX IF NOT EXISTS idx_research_lab_champion_rewards_epoch
    ON public.research_lab_champion_reward_obligations(start_epoch, epoch_count);
CREATE INDEX IF NOT EXISTS idx_research_lab_champion_rewards_miner
    ON public.research_lab_champion_reward_obligations(miner_hotkey, start_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_champion_reward_events_latest
    ON public.research_lab_champion_reward_events(champion_reward_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_emission_alloc_epoch
    ON public.research_lab_emission_allocation_snapshots(epoch, netuid, snapshot_status);

DROP TRIGGER IF EXISTS prevent_research_lab_champion_rewards_mutation
    ON public.research_lab_champion_reward_obligations;
CREATE TRIGGER prevent_research_lab_champion_rewards_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_champion_reward_obligations
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_champion_reward_events_mutation
    ON public.research_lab_champion_reward_events;
CREATE TRIGGER prevent_research_lab_champion_reward_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_champion_reward_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_emission_allocations_mutation
    ON public.research_lab_emission_allocation_snapshots;
CREATE TRIGGER prevent_research_lab_emission_allocations_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_emission_allocation_snapshots
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_champion_reward_obligations FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_champion_reward_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_emission_allocation_snapshots FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_champion_reward_obligations TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_champion_reward_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_emission_allocation_snapshots TO service_role;

ALTER TABLE public.research_lab_champion_reward_obligations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_champion_reward_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_emission_allocation_snapshots ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_champion_reward_obligations;
CREATE POLICY service_role_read ON public.research_lab_champion_reward_obligations
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_champion_reward_obligations;
CREATE POLICY service_role_insert ON public.research_lab_champion_reward_obligations
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_champion_reward_events;
CREATE POLICY service_role_read ON public.research_lab_champion_reward_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_champion_reward_events;
CREATE POLICY service_role_insert ON public.research_lab_champion_reward_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_emission_allocation_snapshots;
CREATE POLICY service_role_read ON public.research_lab_emission_allocation_snapshots
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_emission_allocation_snapshots;
CREATE POLICY service_role_insert ON public.research_lab_emission_allocation_snapshots
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_champion_reward_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.reward_status AS current_reward_status,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_champion_reward_obligations r
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_champion_reward_events e
    WHERE e.champion_reward_id = r.champion_reward_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_lab_emission_allocation_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (epoch, netuid, policy_id)
    *
FROM public.research_lab_emission_allocation_snapshots
WHERE snapshot_status IN ('shadow', 'candidate', 'active')
ORDER BY epoch, netuid, policy_id, created_at DESC;

REVOKE ALL ON TABLE public.research_lab_champion_reward_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_emission_allocation_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_champion_reward_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_emission_allocation_current TO service_role;

COMMENT ON TABLE public.research_lab_champion_reward_obligations IS
    'Append-only Research Lab champion reward obligations derived from validator-scored private-model improvements.';
COMMENT ON TABLE public.research_lab_champion_reward_events IS
    'Append-only champion reward lifecycle events.';
COMMENT ON TABLE public.research_lab_emission_allocation_snapshots IS
    'Append-only per-epoch Research Lab emission allocation snapshots.';
COMMENT ON VIEW public.research_lab_emission_allocation_current IS
    'Latest Research Lab emission allocation projection by epoch, netuid, and policy.';

COMMIT;

-- Smoke checks after applying this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_lab_champion_reward_obligations',
--       'research_lab_champion_reward_events',
--       'research_lab_emission_allocation_snapshots'
--     )
--   ORDER BY table_name;
--
--   SELECT table_name
--   FROM information_schema.views
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_lab_champion_reward_current',
--       'research_lab_emission_allocation_current'
--     )
--   ORDER BY table_name;
