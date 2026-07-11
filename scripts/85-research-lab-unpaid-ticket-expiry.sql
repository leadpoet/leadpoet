-- Research Lab unpaid ticket expiration (apply after migration 84).
--
-- Additive lifecycle hardening only. No scoring, benchmark, promotion, reward,
-- allocation, SOURCE_ADD, fulfillment, emission, or validator-weight behavior
-- is changed by this migration.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_loop_ticket_events
    DROP CONSTRAINT IF EXISTS research_loop_ticket_events_event_type_check;

ALTER TABLE public.research_loop_ticket_events
    ADD CONSTRAINT research_loop_ticket_events_event_type_check CHECK (
        event_type IN (
            'opened',
            'probe_created',
            'funding_pending',
            'funded',
            'queued',
            'running',
            'completed',
            'cancelled',
            'tombstoned',
            'expired'
        )
    ) NOT VALID;

ALTER TABLE public.research_loop_ticket_events
    VALIDATE CONSTRAINT research_loop_ticket_events_event_type_check;

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_event_type_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_event_type_check CHECK (
        event_type IN (
            'submitted',
            'queued',
            'running',
            'candidate_generation_complete',
            'scoring',
            'scored',
            'promotion_passed',
            'promoted',
            'failed',
            'tombstoned',
            'waiting_for_baseline',
            'needs_rescore',
            'blocked_for_credit',
            'not_started',
            'completed_no_candidate',
            'awaiting_payment',
            'no_buildable_candidate',
            'expired'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_public_loop_card_events
    VALIDATE CONSTRAINT research_lab_public_loop_card_events_event_type_check;

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_outcome_label_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_outcome_label_check CHECK (
        outcome_label IN (
            'submitted',
            'queued',
            'running',
            'candidate_generation_complete',
            'scoring',
            'scored_no_gain',
            'scored_promising',
            'promotion_passed',
            'promoted',
            'failed',
            'waiting_for_baseline',
            'needs_rescore',
            'blocked_for_credit',
            'not_started',
            'completed_no_candidate',
            'awaiting_payment',
            'no_buildable_candidate',
            'expired'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_public_loop_card_events
    VALIDATE CONSTRAINT research_lab_public_loop_card_events_outcome_label_check;

ALTER TABLE public.research_lab_public_loop_card_events
    DROP CONSTRAINT IF EXISTS research_lab_public_loop_card_events_outcome_band_check;

ALTER TABLE public.research_lab_public_loop_card_events
    ADD CONSTRAINT research_lab_public_loop_card_events_outcome_band_check CHECK (
        outcome_band IN (
            'pending',
            'no_gain',
            'small_gain',
            'passed_threshold',
            'promoted',
            'failed',
            'blocked',
            'expired'
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_public_loop_card_events
    VALIDATE CONSTRAINT research_lab_public_loop_card_events_outcome_band_check;

CREATE OR REPLACE FUNCTION public.research_lab_unpaid_ticket_expires_at(
    ticket_created_at TIMESTAMPTZ
)
RETURNS TIMESTAMPTZ
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
SET search_path = ''
AS $$
    SELECT ticket_created_at + INTERVAL '24 hours';
$$;

REVOKE ALL ON FUNCTION public.research_lab_unpaid_ticket_expires_at(TIMESTAMPTZ)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_unpaid_ticket_expires_at(TIMESTAMPTZ)
    TO service_role;

CREATE OR REPLACE VIEW public.research_loop_ticket_current
WITH (security_invoker = true) AS
SELECT
    t.*,
    e.seq AS current_event_seq,
    e.event_type AS current_ticket_status,
    e.actor_hotkey AS current_actor_hotkey,
    e.reason AS current_reason,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at,
    public.research_lab_unpaid_ticket_expires_at(t.created_at) AS unpaid_expires_at
FROM public.research_loop_tickets t
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_loop_ticket_events e
    WHERE e.ticket_id = t.ticket_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_loop_ticket_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_loop_ticket_current TO service_role;

CREATE INDEX IF NOT EXISTS idx_research_loop_tickets_created
    ON public.research_loop_tickets(created_at, ticket_id);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_payments_ticket
    ON public.research_loop_start_payments(ticket_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_credits_ticket
    ON public.research_loop_start_credits(ticket_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_start_credit_events_ticket
    ON public.research_loop_start_credit_events(ticket_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_loop_balance_ledger_ticket
    ON public.research_loop_balance_ledger(ticket_id, created_at DESC);

CREATE OR REPLACE VIEW public.research_lab_unpaid_ticket_expiry_candidates
WITH (security_invoker = true) AS
SELECT
    t.ticket_id,
    t.miner_hotkey,
    t.current_ticket_status,
    t.current_event_seq,
    t.current_event_hash,
    t.current_status_at,
    t.created_at,
    t.unpaid_expires_at
FROM public.research_loop_ticket_current t
WHERE t.current_ticket_status IN ('opened', 'probe_created', 'funding_pending')
  AND t.unpaid_expires_at <= NOW()
  AND COALESCE(t.ticket_doc->>'arm', '') <> 'house'
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_start_payments p WHERE p.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_start_credits c WHERE c.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_start_credit_events ce WHERE ce.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_balance_ledger bl WHERE bl.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_run_queue_events q WHERE q.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_loop_receipts r WHERE r.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_lab_auto_research_loop_events l WHERE l.ticket_id = t.ticket_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM public.research_lab_candidate_artifacts c WHERE c.ticket_id = t.ticket_id
  );

REVOKE ALL ON TABLE public.research_lab_unpaid_ticket_expiry_candidates FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_unpaid_ticket_expiry_candidates TO service_role;

CREATE OR REPLACE FUNCTION public.guard_research_lab_ticket_lifecycle()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
    ticket_created_at TIMESTAMPTZ;
    ticket_doc JSONB;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_ticket_lifecycle'),
        pg_catalog.hashtext(NEW.ticket_id::TEXT)
    );

    SELECT t.created_at, t.ticket_doc
      INTO ticket_created_at, ticket_doc
      FROM public.research_loop_tickets t
     WHERE t.ticket_id = NEW.ticket_id;

    IF ticket_created_at IS NULL THEN
        RAISE EXCEPTION 'research_lab_ticket_expiry_conflict: missing ticket %', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    SELECT e.event_type
      INTO latest_status
      FROM public.research_loop_ticket_events e
     WHERE e.ticket_id = NEW.ticket_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status = 'expired' THEN
        RAISE EXCEPTION 'research_lab_ticket_expired: ticket % is expired', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    IF NEW.event_type <> 'expired' THEN
        RETURN NEW;
    END IF;

    IF latest_status NOT IN ('opened', 'probe_created', 'funding_pending') THEN
        RAISE EXCEPTION
            'research_lab_ticket_expiry_conflict: ticket % latest status % is not unpaid',
            NEW.ticket_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23514';
    END IF;

    IF COALESCE(ticket_doc->>'arm', '') = 'house' THEN
        RAISE EXCEPTION 'research_lab_ticket_expiry_conflict: house ticket % is exempt', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    IF pg_catalog.clock_timestamp() < public.research_lab_unpaid_ticket_expires_at(ticket_created_at) THEN
        RAISE EXCEPTION 'research_lab_ticket_expiry_conflict: ticket % has not reached its deadline', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    IF EXISTS (SELECT 1 FROM public.research_loop_start_payments p WHERE p.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_loop_start_credits c WHERE c.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_loop_start_credit_events ce WHERE ce.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_loop_balance_ledger bl WHERE bl.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_loop_run_queue_events q WHERE q.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_loop_receipts r WHERE r.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_lab_auto_research_loop_events l WHERE l.ticket_id = NEW.ticket_id)
       OR EXISTS (SELECT 1 FROM public.research_lab_candidate_artifacts c WHERE c.ticket_id = NEW.ticket_id)
    THEN
        RAISE EXCEPTION 'research_lab_ticket_expiry_conflict: ticket % has lifecycle evidence', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.guard_research_lab_loop_start_payment_expiry()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
    ticket_created_at TIMESTAMPTZ;
    ticket_doc JSONB;
    payment_kind TEXT;
BEGIN
    IF NEW.ticket_id IS NULL THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_ticket_lifecycle'),
        pg_catalog.hashtext(NEW.ticket_id::TEXT)
    );

    SELECT t.created_at, t.ticket_doc
      INTO ticket_created_at, ticket_doc
      FROM public.research_loop_tickets t
     WHERE t.ticket_id = NEW.ticket_id;

    SELECT e.event_type
      INTO latest_status
      FROM public.research_loop_ticket_events e
     WHERE e.ticket_id = NEW.ticket_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status = 'expired' THEN
        RAISE EXCEPTION 'research_lab_ticket_expired: ticket % is expired', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    payment_kind := COALESCE(NULLIF(NEW.verification_doc->>'payment_kind', ''), 'loop_start');
    IF payment_kind = 'loop_start'
       AND COALESCE(ticket_doc->>'arm', '') <> 'house'
       AND pg_catalog.clock_timestamp() >= public.research_lab_unpaid_ticket_expires_at(ticket_created_at)
    THEN
        RAISE EXCEPTION 'research_lab_ticket_expired: ticket % passed its unpaid deadline', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.guard_research_lab_credit_consume()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
    latest_status TEXT;
    latest_ticket_status TEXT;
BEGIN
    IF NEW.credit_status <> 'consumed' THEN
        RETURN NEW;
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_credit_consume'),
        pg_catalog.hashtext(NEW.credit_id::TEXT)
    );

    SELECT e.credit_status
      INTO latest_status
      FROM public.research_loop_start_credit_events e
     WHERE e.credit_id = NEW.credit_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_status IS DISTINCT FROM 'available' THEN
        RAISE EXCEPTION
            'research_lab_credit_consume_conflict: credit % latest status %',
            NEW.credit_id,
            COALESCE(latest_status, '<none>')
            USING ERRCODE = '23505';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_ticket_lifecycle'),
        pg_catalog.hashtext(NEW.ticket_id::TEXT)
    );

    SELECT e.event_type
      INTO latest_ticket_status
      FROM public.research_loop_ticket_events e
     WHERE e.ticket_id = NEW.ticket_id
     ORDER BY e.seq DESC, e.created_at DESC
     LIMIT 1;

    IF latest_ticket_status = 'expired' THEN
        RAISE EXCEPTION 'research_lab_ticket_expired: ticket % is expired', NEW.ticket_id
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.guard_research_lab_ticket_lifecycle()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.guard_research_lab_loop_start_payment_expiry()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.guard_research_lab_credit_consume()
    FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.guard_research_lab_ticket_lifecycle() TO service_role;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_loop_start_payment_expiry() TO service_role;
GRANT EXECUTE ON FUNCTION public.guard_research_lab_credit_consume() TO service_role;

DROP TRIGGER IF EXISTS guard_research_lab_ticket_lifecycle_insert
    ON public.research_loop_ticket_events;
CREATE TRIGGER guard_research_lab_ticket_lifecycle_insert
    BEFORE INSERT ON public.research_loop_ticket_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_ticket_lifecycle();

DROP TRIGGER IF EXISTS guard_research_lab_loop_start_payment_expiry_insert
    ON public.research_loop_start_payments;
CREATE TRIGGER guard_research_lab_loop_start_payment_expiry_insert
    BEFORE INSERT ON public.research_loop_start_payments
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_loop_start_payment_expiry();

DROP TRIGGER IF EXISTS guard_research_loop_start_credit_consume_insert
    ON public.research_loop_start_credit_events;
CREATE TRIGGER guard_research_loop_start_credit_consume_insert
    BEFORE INSERT ON public.research_loop_start_credit_events
    FOR EACH ROW EXECUTE FUNCTION public.guard_research_lab_credit_consume();

COMMENT ON VIEW public.research_lab_unpaid_ticket_expiry_candidates IS
    'Service-role-only miner tickets eligible for append-only expiration after 24 unpaid hours.';
COMMENT ON FUNCTION public.guard_research_lab_ticket_lifecycle() IS
    'Serializes ticket expiration and rejects any ticket transition after expiration.';
COMMENT ON FUNCTION public.guard_research_lab_loop_start_payment_expiry() IS
    'Serializes initial loop-start payment insertion against the fixed unpaid ticket deadline.';

NOTIFY pgrst, 'reload schema';

COMMIT;
