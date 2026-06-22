-- Research Lab Phase 0/1: loop-state, payment, reimbursement,
-- participation, weight-input, and evidence-link storage.
--
-- Deployment policy:
--   * Draft for local/staging review only. Do not apply to production until
--     Work Packages 2A and 3 are reviewed together.
--   * Apply after scripts 25, 26, and 27 in any staging database.
--   * This script does not touch fulfillment tables.
--   * Structured Research Lab state is private, service_role only.
--   * No anon/authenticated grants are created.
--   * All tables in this script are append-only by grant and trigger. Write a
--     new correction/tombstone row instead of updating or deleting.
--   * Miner OpenRouter keys are represented only by encrypted/ephemeral refs.
--     Raw key material must never be inserted into any JSONB document here.
--   * If an earlier draft was applied to staging, drop these Research Lab
--     tables before retesting; CREATE TABLE IF NOT EXISTS will not retrofit
--     changed constraints onto older draft tables.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_loop_tickets (
    ticket_id                     UUID        PRIMARY KEY,
    schema_version                TEXT        NOT NULL DEFAULT '1.0'
                                                CHECK (schema_version = '1.0'),
    miner_hotkey                  TEXT        NOT NULL,
    island                        TEXT        NOT NULL,
    brief_id                      UUID,
    brief_sanitized_ref           TEXT        NOT NULL,
    requested_loop_count          INTEGER     NOT NULL DEFAULT 1
                                                CHECK (requested_loop_count > 0),
    ticket_status                 TEXT        NOT NULL CHECK (
                                                ticket_status IN (
                                                    'opened',
                                                    'probe_created',
                                                    'funding_pending',
                                                    'funded',
                                                    'queued',
                                                    'running',
                                                    'completed',
                                                    'cancelled',
                                                    'tombstoned'
                                                )),
    loop_start_fee_required_usd   NUMERIC(12, 6) NOT NULL DEFAULT 5.000000
                                                CHECK (loop_start_fee_required_usd >= 0),
    loop_start_fee_payment_ref    TEXT,
    miner_openrouter_key_ref      TEXT,
    miner_openrouter_key_handling TEXT CHECK (
                                                miner_openrouter_key_handling IS NULL
                                                OR miner_openrouter_key_handling IN (
                                                    'encrypted_ref',
                                                    'ephemeral_ref'
                                                )),
    miner_openrouter_preflight_status TEXT CHECK (
                                                miner_openrouter_preflight_status IS NULL
                                                OR miner_openrouter_preflight_status IN (
                                                    'passed',
                                                    'failed',
                                                    'not_run'
                                                )),
    ticket_hash                   TEXT        NOT NULL UNIQUE,
    ticket_doc                    JSONB       NOT NULL DEFAULT '{}'::JSONB
                                                CHECK (
                                                    jsonb_typeof(ticket_doc) = 'object'
                                                    AND ticket_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                                                ),
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        miner_openrouter_key_ref IS NULL
        OR miner_openrouter_key_ref !~* '(sk-or-|api[_-]?key|raw[_-]?openrouter|secret)'
    )
);

CREATE TABLE IF NOT EXISTS public.research_loop_balance_ledger (
    ledger_entry_id   UUID        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                CHECK (schema_version = '1.0'),
    miner_hotkey      TEXT        NOT NULL,
    ticket_id         UUID        REFERENCES public.research_loop_tickets(ticket_id)
                                ON DELETE RESTRICT,
    balance_event_type TEXT       NOT NULL CHECK (
                                balance_event_type IN (
                                    'deposit',
                                    'loop_debit',
                                    'refund',
                                    'retry_credit_grant',
                                    'retry_credit_consume',
                                    'operator_adjustment',
                                    'tombstone'
                                )),
    amount_microusd   BIGINT      NOT NULL,
    amount_usd        NUMERIC(14, 6) NOT NULL,
    balance_after_microusd BIGINT,
    idempotency_key   TEXT        NOT NULL UNIQUE,
    anchored_hash     TEXT        NOT NULL UNIQUE,
    ledger_doc        JSONB       NOT NULL DEFAULT '{}'::JSONB
                                CHECK (jsonb_typeof(ledger_doc) = 'object'),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK ((amount_microusd >= 0 AND amount_usd >= 0) OR balance_event_type IN ('loop_debit', 'refund'))
);

CREATE TABLE IF NOT EXISTS public.research_loop_start_payments (
    payment_id             UUID        PRIMARY KEY,
    schema_version         TEXT        NOT NULL DEFAULT '1.0'
                                        CHECK (schema_version = '1.0'),
    ticket_id              UUID        REFERENCES public.research_loop_tickets(ticket_id)
                                        ON DELETE RESTRICT,
    payment_ref            TEXT        NOT NULL UNIQUE,
    block_hash             TEXT        NOT NULL,
    extrinsic_index        INTEGER     NOT NULL CHECK (extrinsic_index >= 0),
    network                TEXT        NOT NULL,
    netuid                 INTEGER     NOT NULL CHECK (netuid > 0),
    miner_hotkey           TEXT        NOT NULL,
    miner_coldkey          TEXT,
    destination_wallet     TEXT        NOT NULL,
    required_usd           NUMERIC(12, 6) NOT NULL CHECK (required_usd >= 0),
    amount_tao             NUMERIC(20, 9) NOT NULL DEFAULT 0 CHECK (amount_tao >= 0),
    amount_usd             NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (amount_usd >= 0),
    tao_price_usd          NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (tao_price_usd >= 0),
    payment_status         TEXT        NOT NULL CHECK (
                                        payment_status IN (
                                            'verified',
                                            'rejected',
                                            'voided',
                                            'tombstoned'
                                        )),
    verification_error     TEXT,
    verification_doc       JSONB       NOT NULL DEFAULT '{}'::JSONB
                                        CHECK (
                                            jsonb_typeof(verification_doc) = 'object'
                                            AND verification_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                                        ),
    verified_at            TIMESTAMPTZ,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_loop_start_payments_block_extrinsic_key
        UNIQUE (block_hash, extrinsic_index),
    CHECK (payment_ref = block_hash || ':' || extrinsic_index::TEXT)
);

CREATE TABLE IF NOT EXISTS public.research_loop_start_credits (
    credit_event_id          UUID        PRIMARY KEY,
    schema_version           TEXT        NOT NULL DEFAULT '1.0'
                                          CHECK (schema_version = '1.0'),
    credit_id                TEXT        NOT NULL,
    ticket_id                UUID        REFERENCES public.research_loop_tickets(ticket_id)
                                          ON DELETE RESTRICT,
    payment_id               UUID        REFERENCES public.research_loop_start_payments(payment_id)
                                          ON DELETE RESTRICT,
    payment_ref              TEXT        NOT NULL,
    miner_hotkey             TEXT        NOT NULL,
    credit_status            TEXT        NOT NULL CHECK (
                                          credit_status IN (
                                              'available',
                                              'consumed',
                                              'voided',
                                              'tombstoned'
                                          )),
    reason                   TEXT        NOT NULL,
    created_from_decision_id TEXT        NOT NULL,
    consumed_by_loop_id      TEXT,
    credit_hash              TEXT        NOT NULL UNIQUE,
    credit_doc               JSONB       NOT NULL DEFAULT '{}'::JSONB
                                          CHECK (jsonb_typeof(credit_doc) = 'object'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_loop_receipts (
    receipt_id               UUID        PRIMARY KEY,
    schema_version           TEXT        NOT NULL DEFAULT '1.0'
                                          CHECK (schema_version = '1.0'),
    ticket_id                UUID        NOT NULL
                                          REFERENCES public.research_loop_tickets(ticket_id)
                                          ON DELETE RESTRICT,
    trajectory_id            UUID        REFERENCES public.research_trajectories(trajectory_id)
                                          ON DELETE RESTRICT,
    -- Hosted Research Lab run UUID from the event-sourced run queue.
    -- This intentionally does not reference execution_traces; private hosted
    -- runs are queued in research_loop_run_queue_events.
    run_id                   UUID,
    loop_start_payment_id    UUID        REFERENCES public.research_loop_start_payments(payment_id)
                                          ON DELETE RESTRICT,
    loop_start_credit_id     TEXT,
    miner_hotkey             TEXT        NOT NULL,
    island                   TEXT        NOT NULL,
    receipt_status           TEXT        NOT NULL CHECK (
                                          receipt_status IN (
                                              'queued',
                                              'completed',
                                              'failed',
                                              'cancelled',
                                              'tombstoned'
                                          )),
    loop_count               INTEGER     NOT NULL DEFAULT 1 CHECK (loop_count > 0),
    miner_openrouter_key_ref TEXT,
    provider_usage           JSONB       NOT NULL DEFAULT '[]'::JSONB
                                          CHECK (
                                              jsonb_typeof(provider_usage) = 'array'
                                              AND provider_usage::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                                          ),
    cost_ledger              JSONB       NOT NULL DEFAULT '{}'::JSONB
                                          CHECK (jsonb_typeof(cost_ledger) = 'object'),
    receipt_hash             TEXT        NOT NULL UNIQUE,
    public_receipt_ref       TEXT,
    receipt_doc              JSONB       NOT NULL DEFAULT '{}'::JSONB
                                          CHECK (
                                              jsonb_typeof(receipt_doc) = 'object'
                                              AND receipt_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                                          ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        miner_openrouter_key_ref IS NULL
        OR miner_openrouter_key_ref !~* '(sk-or-|api[_-]?key|raw[_-]?openrouter|secret)'
    )
);

CREATE TABLE IF NOT EXISTS public.research_island_participation_snapshots (
    participation_snapshot_id UUID        PRIMARY KEY,
    schema_version            TEXT        NOT NULL DEFAULT '1.0'
                                           CHECK (schema_version = '1.0'),
    snapshot_ref              TEXT        NOT NULL UNIQUE,
    island                    TEXT        NOT NULL,
    lookback_start            TIMESTAMPTZ NOT NULL,
    lookback_end              TIMESTAMPTZ NOT NULL,
    distinct_funded_hotkeys   INTEGER     NOT NULL CHECK (distinct_funded_hotkeys >= 0),
    paid_loop_count           INTEGER     NOT NULL CHECK (paid_loop_count >= 0),
    unique_brief_count        INTEGER     NOT NULL CHECK (unique_brief_count >= 0),
    source_add_count          INTEGER     NOT NULL DEFAULT 0 CHECK (source_add_count >= 0),
    red_team_count            INTEGER     NOT NULL DEFAULT 0 CHECK (red_team_count >= 0),
    participation_score       NUMERIC(14, 6) NOT NULL CHECK (participation_score >= 0),
    policy_id                 TEXT        NOT NULL,
    input_hash                TEXT        NOT NULL UNIQUE,
    snapshot_doc              JSONB       NOT NULL DEFAULT '{}'::JSONB
                                           CHECK (jsonb_typeof(snapshot_doc) = 'object'),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (lookback_end > lookback_start)
);

CREATE TABLE IF NOT EXISTS public.research_reimbursement_awards (
    award_id                    TEXT        PRIMARY KEY,
    schema_version              TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    receipt_id                  UUID        REFERENCES public.research_loop_receipts(receipt_id)
                                            ON DELETE RESTRICT,
    participation_snapshot_id   UUID        REFERENCES public.research_island_participation_snapshots(participation_snapshot_id)
                                            ON DELETE RESTRICT,
    run_id                      TEXT        NOT NULL,
    miner_hotkey                TEXT        NOT NULL,
    island                      TEXT        NOT NULL,
    run_day                     DATE        NOT NULL,
    policy_id                   TEXT        NOT NULL,
    award_status                TEXT        NOT NULL CHECK (
                                            award_status IN (
                                                'awarded',
                                                'disabled',
                                                'ineligible',
                                                'capped_to_zero',
                                                'tombstoned'
                                            )),
    participation_score         NUMERIC(14, 6) NOT NULL CHECK (participation_score >= 0),
    participation_fraction      NUMERIC(8, 6)  NOT NULL CHECK (
                                            participation_fraction >= 0
                                            AND participation_fraction <= 1
                                            ),
    rebate_rate                 NUMERIC(8, 6)  NOT NULL CHECK (
                                            rebate_rate >= 0
                                            AND rebate_rate <= 1
                                            ),
    eligible_cost_microusd      BIGINT      NOT NULL CHECK (eligible_cost_microusd >= 0),
    target_reimbursement_microusd BIGINT    NOT NULL CHECK (target_reimbursement_microusd >= 0),
    reimbursement_epochs        INTEGER     NOT NULL CHECK (reimbursement_epochs >= 0),
    loop_start_fee_included     BOOLEAN     NOT NULL DEFAULT FALSE,
    input_hash                  TEXT        NOT NULL UNIQUE,
    award_doc                   JSONB       NOT NULL CHECK (jsonb_typeof(award_doc) = 'object'),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_reimbursement_schedules (
    schedule_id       TEXT        PRIMARY KEY,
    schema_version    TEXT        NOT NULL DEFAULT '1.0'
                                CHECK (schema_version = '1.0'),
    award_id          TEXT        NOT NULL
                                REFERENCES public.research_reimbursement_awards(award_id)
                                ON DELETE RESTRICT,
    schedule_status   TEXT        NOT NULL CHECK (
                                schedule_status IN (
                                    'scheduled',
                                    'empty',
                                    'voided',
                                    'tombstoned'
                                )),
    start_epoch       INTEGER     NOT NULL CHECK (start_epoch >= 0),
    epoch_count       INTEGER     NOT NULL CHECK (epoch_count >= 0),
    total_microusd    BIGINT      NOT NULL CHECK (total_microusd >= 0),
    entries           JSONB       NOT NULL DEFAULT '[]'::JSONB
                                CHECK (jsonb_typeof(entries) = 'array'),
    schedule_hash     TEXT        NOT NULL UNIQUE,
    schedule_doc      JSONB       NOT NULL DEFAULT '{}'::JSONB
                                CHECK (jsonb_typeof(schedule_doc) = 'object'),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_weight_input_snapshots (
    weight_input_snapshot_id UUID        PRIMARY KEY,
    schema_version           TEXT        NOT NULL DEFAULT '1.0'
                                         CHECK (schema_version = '1.0'),
    epoch                    INTEGER     NOT NULL CHECK (epoch >= 0),
    netuid                   INTEGER     NOT NULL CHECK (netuid > 0),
    snapshot_status          TEXT        NOT NULL CHECK (
                                         snapshot_status IN (
                                             'shadow',
                                             'candidate',
                                             'active',
                                             'tombstoned'
                                         )),
    fulfillment_weight_ref   TEXT,
    leaderboard_weight_ref   TEXT,
    improvement_grant_ref    TEXT,
    reimbursement_weight_ref TEXT,
    active_researcher_floor_ref TEXT,
    source_bundle_ref        TEXT        NOT NULL,
    input_state_hash         TEXT        NOT NULL UNIQUE,
    weight_vector_hash       TEXT,
    snapshot_doc             JSONB       NOT NULL DEFAULT '{}'::JSONB
                                         CHECK (jsonb_typeof(snapshot_doc) = 'object'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_evidence_trajectory_links (
    link_id        UUID        PRIMARY KEY,
    schema_version TEXT        NOT NULL DEFAULT '1.0'
                                CHECK (schema_version = '1.0'),
    trajectory_id  UUID        NOT NULL,
    event_seq      INTEGER     NOT NULL CHECK (event_seq >= 0),
    bundle_id      UUID        REFERENCES public.evidence_bundles(bundle_id)
                                ON DELETE RESTRICT,
    run_id         UUID        REFERENCES public.execution_traces(run_id)
                                ON DELETE SET NULL,
    link_type      TEXT        NOT NULL CHECK (
                                link_type IN (
                                    'event_evidence',
                                    'run_evidence',
                                    'receipt_evidence',
                                    'reimbursement_evidence',
                                    'weight_input_evidence',
                                    'tombstone'
                                )),
    link_hash      TEXT        NOT NULL UNIQUE,
    link_doc       JSONB       NOT NULL DEFAULT '{}'::JSONB
                                CHECK (jsonb_typeof(link_doc) = 'object'),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (trajectory_id, event_seq)
        REFERENCES public.research_trajectory_events(trajectory_id, seq)
        ON DELETE RESTRICT,
    CHECK (bundle_id IS NOT NULL OR run_id IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_loop_start_credits_available
    ON public.research_loop_start_credits(credit_id)
    WHERE credit_status = 'available';

CREATE INDEX IF NOT EXISTS idx_research_loop_tickets_miner_created
    ON public.research_loop_tickets(miner_hotkey, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_tickets_island_status
    ON public.research_loop_tickets(island, ticket_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_balance_ledger_miner_created
    ON public.research_loop_balance_ledger(miner_hotkey, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_payments_miner_created
    ON public.research_loop_start_payments(miner_hotkey, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_payments_status
    ON public.research_loop_start_payments(payment_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_credits_miner_status
    ON public.research_loop_start_credits(miner_hotkey, credit_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_receipts_ticket_created
    ON public.research_loop_receipts(ticket_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_receipts_trajectory
    ON public.research_loop_receipts(trajectory_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_island_participation_snapshots_island
    ON public.research_island_participation_snapshots(island, lookback_end DESC);

CREATE INDEX IF NOT EXISTS idx_research_reimbursement_awards_miner_day
    ON public.research_reimbursement_awards(miner_hotkey, run_day DESC);

CREATE INDEX IF NOT EXISTS idx_research_reimbursement_awards_island_status
    ON public.research_reimbursement_awards(island, award_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_reimbursement_schedules_award
    ON public.research_reimbursement_schedules(award_id);

CREATE INDEX IF NOT EXISTS idx_research_weight_input_snapshots_epoch_status
    ON public.research_weight_input_snapshots(epoch, snapshot_status);

CREATE INDEX IF NOT EXISTS idx_research_evidence_trajectory_links_trajectory
    ON public.research_evidence_trajectory_links(trajectory_id, event_seq);

CREATE INDEX IF NOT EXISTS idx_research_evidence_trajectory_links_bundle
    ON public.research_evidence_trajectory_links(bundle_id);

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

DROP TRIGGER IF EXISTS prevent_research_loop_tickets_mutation
    ON public.research_loop_tickets;
CREATE TRIGGER prevent_research_loop_tickets_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_tickets
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_balance_ledger_mutation
    ON public.research_loop_balance_ledger;
CREATE TRIGGER prevent_research_loop_balance_ledger_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_balance_ledger
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_start_payments_mutation
    ON public.research_loop_start_payments;
CREATE TRIGGER prevent_research_loop_start_payments_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_start_payments
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_start_credits_mutation
    ON public.research_loop_start_credits;
CREATE TRIGGER prevent_research_loop_start_credits_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_start_credits
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_receipts_mutation
    ON public.research_loop_receipts;
CREATE TRIGGER prevent_research_loop_receipts_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_receipts
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_island_participation_snapshots_mutation
    ON public.research_island_participation_snapshots;
CREATE TRIGGER prevent_research_island_participation_snapshots_mutation
    BEFORE UPDATE OR DELETE ON public.research_island_participation_snapshots
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_reimbursement_awards_mutation
    ON public.research_reimbursement_awards;
CREATE TRIGGER prevent_research_reimbursement_awards_mutation
    BEFORE UPDATE OR DELETE ON public.research_reimbursement_awards
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_reimbursement_schedules_mutation
    ON public.research_reimbursement_schedules;
CREATE TRIGGER prevent_research_reimbursement_schedules_mutation
    BEFORE UPDATE OR DELETE ON public.research_reimbursement_schedules
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_weight_input_snapshots_mutation
    ON public.research_weight_input_snapshots;
CREATE TRIGGER prevent_research_weight_input_snapshots_mutation
    BEFORE UPDATE OR DELETE ON public.research_weight_input_snapshots
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_evidence_trajectory_links_mutation
    ON public.research_evidence_trajectory_links;
CREATE TRIGGER prevent_research_evidence_trajectory_links_mutation
    BEFORE UPDATE OR DELETE ON public.research_evidence_trajectory_links
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_loop_tickets FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_balance_ledger FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_start_payments FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_start_credits FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_receipts FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_island_participation_snapshots FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_reimbursement_awards FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_reimbursement_schedules FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_weight_input_snapshots FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_evidence_trajectory_links FROM anon, authenticated;

GRANT SELECT, INSERT ON TABLE public.research_loop_tickets TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_balance_ledger TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_start_payments TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_start_credits TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_receipts TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_island_participation_snapshots TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_reimbursement_awards TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_reimbursement_schedules TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_weight_input_snapshots TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_evidence_trajectory_links TO service_role;

ALTER TABLE public.research_loop_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_balance_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_start_payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_start_credits ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_island_participation_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_reimbursement_awards ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_reimbursement_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_weight_input_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_evidence_trajectory_links ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_loop_tickets;
CREATE POLICY service_role_read ON public.research_loop_tickets
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_tickets;
CREATE POLICY service_role_insert ON public.research_loop_tickets
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_balance_ledger;
CREATE POLICY service_role_read ON public.research_loop_balance_ledger
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_balance_ledger;
CREATE POLICY service_role_insert ON public.research_loop_balance_ledger
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_start_payments;
CREATE POLICY service_role_read ON public.research_loop_start_payments
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_start_payments;
CREATE POLICY service_role_insert ON public.research_loop_start_payments
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_start_credits;
CREATE POLICY service_role_read ON public.research_loop_start_credits
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_start_credits;
CREATE POLICY service_role_insert ON public.research_loop_start_credits
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_receipts;
CREATE POLICY service_role_read ON public.research_loop_receipts
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_receipts;
CREATE POLICY service_role_insert ON public.research_loop_receipts
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_island_participation_snapshots;
CREATE POLICY service_role_read ON public.research_island_participation_snapshots
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_island_participation_snapshots;
CREATE POLICY service_role_insert ON public.research_island_participation_snapshots
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_reimbursement_awards;
CREATE POLICY service_role_read ON public.research_reimbursement_awards
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_reimbursement_awards;
CREATE POLICY service_role_insert ON public.research_reimbursement_awards
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_reimbursement_schedules;
CREATE POLICY service_role_read ON public.research_reimbursement_schedules
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_reimbursement_schedules;
CREATE POLICY service_role_insert ON public.research_reimbursement_schedules
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_weight_input_snapshots;
CREATE POLICY service_role_read ON public.research_weight_input_snapshots
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_weight_input_snapshots;
CREATE POLICY service_role_insert ON public.research_weight_input_snapshots
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_evidence_trajectory_links;
CREATE POLICY service_role_read ON public.research_evidence_trajectory_links
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_evidence_trajectory_links;
CREATE POLICY service_role_insert ON public.research_evidence_trajectory_links
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

COMMENT ON TABLE public.research_loop_tickets IS
    'Append-only Research Lab loop-start ticket envelope. State corrections use new rows/events, never mutation.';

COMMENT ON TABLE public.research_loop_balance_ledger IS
    'Append-only miner loop-balance ledger for deposits, loop debits, refunds, retry credits, and operator tombstones.';

COMMENT ON TABLE public.research_loop_start_payments IS
    'Append-only Research Lab loop-start TAO payment records. Unique block_hash/extrinsic_index prevents duplicate payment reuse.';

COMMENT ON TABLE public.research_loop_start_credits IS
    'Append-only loop-start retry-credit events for verified payments that fail before queueing/useful work begins.';

COMMENT ON TABLE public.research_loop_receipts IS
    'Append-only paid-loop receipts. Provider usage records key refs/sources and costs without raw secret values.';

COMMENT ON TABLE public.research_island_participation_snapshots IS
    'Append-only island participation snapshots used to recompute participation-based alpha reimbursement rates.';

COMMENT ON TABLE public.research_reimbursement_awards IS
    'Append-only alpha reimbursement awards derived from verified loop receipts, cost ledgers, participation snapshots, and policy.';

COMMENT ON TABLE public.research_reimbursement_schedules IS
    'Append-only fixed-window alpha reimbursement schedules. Entries must be verifier-recomputable from award state.';

COMMENT ON TABLE public.research_weight_input_snapshots IS
    'Append-only deterministic inputs to Research Lab weight-vector composition, including shadow snapshots.';

COMMENT ON TABLE public.research_evidence_trajectory_links IS
    'Append-only explicit link table from trajectory events to evidence bundles and execution traces.';

COMMENT ON COLUMN public.research_loop_tickets.miner_openrouter_key_ref IS
    'Encrypted or ephemeral reference only. Never store raw OpenRouter key material.';

COMMENT ON COLUMN public.research_loop_start_payments.payment_ref IS
    'Canonical duplicate-prevention key: block_hash || '':'' || extrinsic_index.';

COMMENT ON COLUMN public.research_loop_receipts.provider_usage IS
    'Array of provider usage entries. OpenRouter should reference miner key source; Exa/ScrapingDog should reference Leadpoet server-side key source (leadpoet_server_side).';

COMMIT;

-- Local/staging smoke checks after applying to a non-production database:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_loop_tickets',
--       'research_loop_balance_ledger',
--       'research_loop_start_payments',
--       'research_loop_start_credits',
--       'research_loop_receipts',
--       'research_island_participation_snapshots',
--       'research_reimbursement_awards',
--       'research_reimbursement_schedules',
--       'research_weight_input_snapshots',
--       'research_evidence_trajectory_links'
--     )
--   ORDER BY table_name;
--
--   SELECT relname, relrowsecurity
--   FROM pg_class
--   WHERE relname IN (
--       'research_loop_tickets',
--       'research_loop_balance_ledger',
--       'research_loop_start_payments',
--       'research_loop_start_credits',
--       'research_loop_receipts',
--       'research_island_participation_snapshots',
--       'research_reimbursement_awards',
--       'research_reimbursement_schedules',
--       'research_weight_input_snapshots',
--       'research_evidence_trajectory_links'
--   );
--
--   SELECT conname
--   FROM pg_constraint
--   WHERE conrelid = 'public.research_loop_start_payments'::regclass
--     AND conname = 'research_loop_start_payments_block_extrinsic_key';
--
-- Rollback/tombstone policy:
--   * Before production activation, staging rollback may DROP these tables.
--   * After any live data exists, do not delete rows. Insert tombstone rows or
--     freeze new writes with feature flags, preserving verifier history.
