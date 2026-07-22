-- Research Lab egress reduction: single-owner maintenance lease.
--
-- Every hosted/scoring worker ran the global maintenance sweeps (stale-run and
-- stale-candidate recovery, projection/card reconciles, champion-reward
-- reconciliation, corpus projection/backfill) on every pass. With N workers
-- that is N x the same set of scans and writes.
--
-- This DB-backed lease lets exactly one worker hold "ownership" of a named
-- maintenance scope at a time; only the holder runs the global sweeps. If the
-- holder dies, the lease expires and another worker takes over (no dedicated
-- always-on process required, and correct across future gateway replicas). The
-- individual sweep operations are unchanged -- only WHICH worker runs them.
--
-- Reward safety: the reconciled operations (champion rewards, reimbursements)
-- are not modified. If the lease cannot be acquired (contention or a Supabase
-- error) a worker simply skips the sweep this pass; the holder (or the next
-- taker after expiry) performs it. Fail-closed = no duplication, at worst a
-- brief delay bounded by the lease TTL.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_maintenance_lease (
    lease_name   TEXT        PRIMARY KEY,
    holder_ref   TEXT        NOT NULL,
    acquired_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.research_lab_maintenance_lease ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_lab_maintenance_lease
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_maintenance_lease
    TO service_role;
DROP POLICY IF EXISTS service_role_all ON public.research_lab_maintenance_lease;
CREATE POLICY service_role_all ON public.research_lab_maintenance_lease
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Atomically acquire or renew a lease. Grants to p_holder_ref iff the lease is
-- unheld, already held by p_holder_ref (renewal), or expired. Returns whether
-- this caller now holds it plus the current holder/expiry for observability.
CREATE OR REPLACE FUNCTION public.research_lab_acquire_maintenance_lease(
    p_lease_name  TEXT,
    p_holder_ref  TEXT,
    p_ttl_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    v_now       TIMESTAMPTZ := pg_catalog.now();
    v_expires   TIMESTAMPTZ;
    v_holder    TEXT;
    v_expires_at TIMESTAMPTZ;
BEGIN
    IF p_lease_name IS NULL OR p_holder_ref IS NULL
       OR p_ttl_seconds IS NULL OR p_ttl_seconds <= 0
       OR p_ttl_seconds > 86400 THEN
        RAISE EXCEPTION 'maintenance lease arguments are invalid'
            USING ERRCODE = '22023';
    END IF;
    v_expires := v_now + pg_catalog.make_interval(secs => p_ttl_seconds);

    -- Serialize concurrent acquirers of the same lease name deterministically.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_maintenance_lease'),
        pg_catalog.hashtext(p_lease_name)
    );

    INSERT INTO public.research_lab_maintenance_lease AS l
        (lease_name, holder_ref, acquired_at, expires_at, updated_at)
    VALUES (p_lease_name, p_holder_ref, v_now, v_expires, v_now)
    ON CONFLICT (lease_name) DO UPDATE
        SET holder_ref  = EXCLUDED.holder_ref,
            acquired_at  = CASE
                WHEN l.holder_ref = EXCLUDED.holder_ref THEN l.acquired_at
                ELSE EXCLUDED.acquired_at END,
            expires_at   = EXCLUDED.expires_at,
            updated_at   = EXCLUDED.updated_at
        WHERE l.expires_at < v_now
           OR l.holder_ref = EXCLUDED.holder_ref
    RETURNING l.holder_ref, l.expires_at INTO v_holder, v_expires_at;

    IF v_holder IS NULL THEN
        -- The lease is held by another live holder; report the current state.
        SELECT l.holder_ref, l.expires_at INTO v_holder, v_expires_at
        FROM public.research_lab_maintenance_lease l
        WHERE l.lease_name = p_lease_name;
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'acquired', (v_holder = p_holder_ref),
        'holder_ref', v_holder,
        'expires_at', v_expires_at
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_acquire_maintenance_lease(TEXT, TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_acquire_maintenance_lease(TEXT, TEXT, INTEGER)
    TO service_role;

COMMIT;
