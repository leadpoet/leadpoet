-- Research Lab Phase 1: append-only state events and current-state projections.
--
-- Deployment policy:
--   * Apply after scripts 27 and 28.
--   * This script is production-intended, but it does not activate paid loops,
--     reimbursements, crowning, fulfillment writes, or weight mutation.
--   * It fixes the status/event ambiguity in script 28 by making every mutable
--     workflow state an append-only event stream. Current status is read from
--     views only; writers must append events, never update prior rows.
--   * No anon/authenticated grants are created.

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

CREATE TABLE IF NOT EXISTS public.research_loop_ticket_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    ticket_id     UUID        NOT NULL REFERENCES public.research_loop_tickets(ticket_id)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
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
    actor_hotkey  TEXT,
    reason        TEXT,
    anchored_hash TEXT        NOT NULL UNIQUE,
    event_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(event_doc) = 'object'
                              AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_loop_ticket_events_ticket_seq_key UNIQUE (ticket_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_loop_start_credit_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    credit_id     TEXT        NOT NULL,
    ticket_id     UUID        REFERENCES public.research_loop_tickets(ticket_id)
                              ON DELETE RESTRICT,
    payment_id    UUID        REFERENCES public.research_loop_start_payments(payment_id)
                              ON DELETE RESTRICT,
    payment_ref   TEXT        NOT NULL,
    miner_hotkey  TEXT        NOT NULL,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
                                  'granted',
                                  'consumed',
                                  'voided',
                                  'tombstoned'
                              )),
    credit_status TEXT        NOT NULL CHECK (
                              credit_status IN (
                                  'available',
                                  'consumed',
                                  'voided',
                                  'tombstoned'
                              )),
    reason        TEXT        NOT NULL,
    consumed_by_loop_id TEXT,
    anchored_hash TEXT        NOT NULL UNIQUE,
    event_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(event_doc) = 'object'
                              AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_loop_start_credit_events_credit_seq_key UNIQUE (credit_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_loop_run_queue_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    run_id        UUID        NOT NULL,
    ticket_id     UUID        NOT NULL REFERENCES public.research_loop_tickets(ticket_id)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
                                  'queued',
                                  'started',
                                  'completed',
                                  'failed',
                                  'cancelled',
                                  'tombstoned'
                              )),
    queue_priority INTEGER    NOT NULL DEFAULT 0,
    worker_ref    TEXT,
    reason        TEXT,
    anchored_hash TEXT        NOT NULL UNIQUE,
    event_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(event_doc) = 'object'
                              AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_loop_run_queue_events_run_seq_key UNIQUE (run_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_loop_receipt_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    receipt_id    UUID        NOT NULL REFERENCES public.research_loop_receipts(receipt_id)
                              ON DELETE RESTRICT,
    ticket_id     UUID        NOT NULL REFERENCES public.research_loop_tickets(ticket_id)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
                                  'queued',
                                  'completed',
                                  'failed',
                                  'cancelled',
                                  'tombstoned'
                              )),
    receipt_status TEXT       NOT NULL CHECK (
                              receipt_status IN (
                                  'queued',
                                  'completed',
                                  'failed',
                                  'cancelled',
                                  'tombstoned'
                              )),
    anchored_hash TEXT        NOT NULL UNIQUE,
    event_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(event_doc) = 'object'
                              AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_loop_receipt_events_receipt_seq_key UNIQUE (receipt_id, seq)
);

CREATE TABLE IF NOT EXISTS public.research_reimbursement_award_events (
    event_id      UUID        PRIMARY KEY,
    schema_version TEXT       NOT NULL DEFAULT '1.0' CHECK (schema_version = '1.0'),
    award_id      TEXT        NOT NULL REFERENCES public.research_reimbursement_awards(award_id)
                              ON DELETE RESTRICT,
    seq           INTEGER     NOT NULL CHECK (seq >= 0),
    event_type    TEXT        NOT NULL CHECK (
                              event_type IN (
                                  'awarded',
                                  'disabled',
                                  'ineligible',
                                  'capped_to_zero',
                                  'voided',
                                  'tombstoned'
                              )),
    award_status  TEXT        NOT NULL CHECK (
                              award_status IN (
                                  'awarded',
                                  'disabled',
                                  'ineligible',
                                  'capped_to_zero',
                                  'voided',
                                  'tombstoned'
                              )),
    anchored_hash TEXT        NOT NULL UNIQUE,
    event_doc     JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                              jsonb_typeof(event_doc) = 'object'
                              AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret)'
                              ),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_reimbursement_award_events_award_seq_key UNIQUE (award_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_loop_ticket_events_latest
    ON public.research_loop_ticket_events(ticket_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_loop_ticket_events_type_created
    ON public.research_loop_ticket_events(event_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_credit_events_latest
    ON public.research_loop_start_credit_events(credit_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_loop_start_credit_events_miner_status
    ON public.research_loop_start_credit_events(miner_hotkey, credit_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_run_queue_events_latest
    ON public.research_loop_run_queue_events(run_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_loop_run_queue_events_ticket
    ON public.research_loop_run_queue_events(ticket_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_receipt_events_latest
    ON public.research_loop_receipt_events(receipt_id, seq DESC);

CREATE INDEX IF NOT EXISTS idx_research_reimbursement_award_events_latest
    ON public.research_reimbursement_award_events(award_id, seq DESC);

DROP TRIGGER IF EXISTS prevent_research_loop_ticket_events_mutation
    ON public.research_loop_ticket_events;
CREATE TRIGGER prevent_research_loop_ticket_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_ticket_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_start_credit_events_mutation
    ON public.research_loop_start_credit_events;
CREATE TRIGGER prevent_research_loop_start_credit_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_start_credit_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_run_queue_events_mutation
    ON public.research_loop_run_queue_events;
CREATE TRIGGER prevent_research_loop_run_queue_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_run_queue_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_loop_receipt_events_mutation
    ON public.research_loop_receipt_events;
CREATE TRIGGER prevent_research_loop_receipt_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_loop_receipt_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_reimbursement_award_events_mutation
    ON public.research_reimbursement_award_events;
CREATE TRIGGER prevent_research_reimbursement_award_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_reimbursement_award_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_loop_ticket_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_start_credit_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_run_queue_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_receipt_events FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_reimbursement_award_events FROM anon, authenticated;

GRANT SELECT, INSERT ON TABLE public.research_loop_ticket_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_start_credit_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_run_queue_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_loop_receipt_events TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_reimbursement_award_events TO service_role;

ALTER TABLE public.research_loop_ticket_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_start_credit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_run_queue_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_loop_receipt_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_reimbursement_award_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_loop_ticket_events;
CREATE POLICY service_role_read ON public.research_loop_ticket_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_ticket_events;
CREATE POLICY service_role_insert ON public.research_loop_ticket_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_start_credit_events;
CREATE POLICY service_role_read ON public.research_loop_start_credit_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_start_credit_events;
CREATE POLICY service_role_insert ON public.research_loop_start_credit_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_run_queue_events;
CREATE POLICY service_role_read ON public.research_loop_run_queue_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_run_queue_events;
CREATE POLICY service_role_insert ON public.research_loop_run_queue_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_loop_receipt_events;
CREATE POLICY service_role_read ON public.research_loop_receipt_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_loop_receipt_events;
CREATE POLICY service_role_insert ON public.research_loop_receipt_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_reimbursement_award_events;
CREATE POLICY service_role_read ON public.research_reimbursement_award_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_reimbursement_award_events;
CREATE POLICY service_role_insert ON public.research_reimbursement_award_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_loop_ticket_current
WITH (security_invoker = true) AS
SELECT
    t.*,
    e.seq AS current_event_seq,
    e.event_type AS current_ticket_status,
    e.actor_hotkey AS current_actor_hotkey,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_loop_tickets t
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_loop_ticket_events e
    WHERE e.ticket_id = t.ticket_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_loop_start_credit_current
WITH (security_invoker = true) AS
SELECT
    e.credit_id,
    e.ticket_id,
    e.payment_id,
    e.payment_ref,
    e.miner_hotkey,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.credit_status AS current_credit_status,
    e.reason AS current_reason,
    e.consumed_by_loop_id,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM (
    SELECT DISTINCT ON (credit_id) *
    FROM public.research_loop_start_credit_events
    ORDER BY credit_id, seq DESC, created_at DESC
) e;

CREATE OR REPLACE VIEW public.research_loop_available_credits
WITH (security_invoker = true) AS
SELECT *
FROM public.research_loop_start_credit_current
WHERE current_credit_status = 'available';

CREATE OR REPLACE VIEW public.research_loop_run_queue_current
WITH (security_invoker = true) AS
SELECT
    e.run_id,
    e.ticket_id,
    e.seq AS current_event_seq,
    e.event_type AS current_queue_status,
    e.queue_priority,
    e.worker_ref,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM (
    SELECT DISTINCT ON (run_id) *
    FROM public.research_loop_run_queue_events
    ORDER BY run_id, seq DESC, created_at DESC
) e;

CREATE OR REPLACE VIEW public.research_loop_receipt_current
WITH (security_invoker = true) AS
SELECT
    r.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.receipt_status AS current_receipt_status,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_loop_receipts r
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_loop_receipt_events e
    WHERE e.receipt_id = r.receipt_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_reimbursement_award_current
WITH (security_invoker = true) AS
SELECT
    a.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.award_status AS current_award_status,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_reimbursement_awards a
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_reimbursement_award_events e
    WHERE e.award_id = a.award_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

CREATE OR REPLACE VIEW public.research_loop_balance_current
WITH (security_invoker = true) AS
SELECT
    miner_hotkey,
    ticket_id,
    SUM(amount_microusd) AS balance_microusd,
    MAX(created_at) AS last_balance_event_at,
    COUNT(*) AS balance_event_count
FROM public.research_loop_balance_ledger
GROUP BY miner_hotkey, ticket_id;

CREATE OR REPLACE VIEW public.research_loop_shadow_weight_inputs
WITH (security_invoker = true) AS
SELECT *
FROM public.research_weight_input_snapshots
WHERE snapshot_status = 'shadow';

REVOKE ALL ON TABLE public.research_loop_ticket_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_start_credit_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_available_credits FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_run_queue_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_receipt_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_reimbursement_award_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_balance_current FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_loop_shadow_weight_inputs FROM anon, authenticated;

GRANT SELECT ON TABLE public.research_loop_ticket_current TO service_role;
GRANT SELECT ON TABLE public.research_loop_start_credit_current TO service_role;
GRANT SELECT ON TABLE public.research_loop_available_credits TO service_role;
GRANT SELECT ON TABLE public.research_loop_run_queue_current TO service_role;
GRANT SELECT ON TABLE public.research_loop_receipt_current TO service_role;
GRANT SELECT ON TABLE public.research_reimbursement_award_current TO service_role;
GRANT SELECT ON TABLE public.research_loop_balance_current TO service_role;
GRANT SELECT ON TABLE public.research_loop_shadow_weight_inputs TO service_role;

COMMENT ON TABLE public.research_loop_ticket_events IS
    'Append-only state events for Research Lab loop tickets. Current ticket status is derived from research_loop_ticket_current.';
COMMENT ON TABLE public.research_loop_start_credit_events IS
    'Append-only state events for loop-start retry credits. Availability is derived from research_loop_available_credits.';
COMMENT ON TABLE public.research_loop_run_queue_events IS
    'Append-only queue state events for hosted Research Lab runs.';
COMMENT ON TABLE public.research_loop_receipt_events IS
    'Append-only receipt lifecycle events. Receipts remain immutable, status is projected.';
COMMENT ON TABLE public.research_reimbursement_award_events IS
    'Append-only alpha reimbursement award status events. Reimbursement current status is projected.';
COMMENT ON VIEW public.research_loop_ticket_current IS
    'Current Research Lab ticket state projection. Derived, not a source of truth.';
COMMENT ON VIEW public.research_loop_available_credits IS
    'Current available loop-start retry credits derived from credit events.';
COMMENT ON VIEW public.research_loop_balance_current IS
    'Current loop-balance projection from the append-only balance ledger.';

COMMIT;

-- Smoke checks after user applies this migration:
--
--   SELECT table_name
--   FROM information_schema.tables
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_loop_ticket_events',
--       'research_loop_start_credit_events',
--       'research_loop_run_queue_events',
--       'research_loop_receipt_events',
--       'research_reimbursement_award_events'
--     )
--   ORDER BY table_name;
--
--   SELECT table_name
--   FROM information_schema.views
--   WHERE table_schema = 'public'
--     AND table_name IN (
--       'research_loop_ticket_current',
--       'research_loop_available_credits',
--       'research_loop_run_queue_current',
--       'research_loop_receipt_current',
--       'research_reimbursement_award_current',
--       'research_loop_balance_current',
--       'research_loop_shadow_weight_inputs'
--     )
--   ORDER BY table_name;
