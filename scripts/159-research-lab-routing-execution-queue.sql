-- Durable, bounded queue leasing for Research Lab routing execution requests.
--
-- This migration is additive to 157.  The immutable request table remains
-- append-only.  Queue ownership is kept in a separate, service-role-only
-- lease table so workers can use PostgreSQL row locks and SKIP LOCKED.
-- No claim nonce or provider credential is accepted or stored here.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS public.research_lab_routing_execution_request_leases_v2 (
    request_hash TEXT PRIMARY KEY
        REFERENCES public.research_lab_routing_execution_requests_v2(request_hash),
    experiment_hash TEXT NOT NULL UNIQUE
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    lease_hash TEXT NOT NULL CHECK (lease_hash ~ '^sha256:[0-9a-f]{64}$'),
    worker_ref TEXT NOT NULL
        CHECK (worker_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    lease_generation BIGINT NOT NULL CHECK (lease_generation > 0),
    lease_state TEXT NOT NULL CHECK (lease_state IN ('claimed', 'completed', 'failed', 'recovered')),
    lease_expires_at TIMESTAMPTZ,
    close_reason TEXT
        CHECK (close_reason IS NULL OR close_reason IN ('completed', 'failed', 'recovered')),
    execution_claim_key TEXT
        CHECK (execution_claim_key IS NULL OR execution_claim_key ~ '^sha256:[0-9a-f]{64}$'),
    execution_claim_generation BIGINT
        CHECK (execution_claim_generation IS NULL OR execution_claim_generation > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK ((lease_state = 'claimed' AND lease_expires_at IS NOT NULL AND close_reason IS NULL)
        OR (lease_state IN ('completed', 'failed', 'recovered') AND close_reason = lease_state))
    ,CHECK ((execution_claim_key IS NULL AND execution_claim_generation IS NULL)
        OR (execution_claim_key IS NOT NULL AND execution_claim_generation IS NOT NULL))
);
ALTER TABLE public.research_lab_routing_execution_request_leases_v2
    ADD COLUMN IF NOT EXISTS execution_claim_key TEXT
        CHECK (execution_claim_key IS NULL OR execution_claim_key ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS execution_claim_generation BIGINT
        CHECK (execution_claim_generation IS NULL OR execution_claim_generation > 0);
ALTER TABLE public.research_lab_routing_execution_request_leases_v2
    DROP CONSTRAINT IF EXISTS rl_route_request_lease_claim_pair_v3;
ALTER TABLE public.research_lab_routing_execution_request_leases_v2
    ADD CONSTRAINT rl_route_request_lease_claim_pair_v3
    CHECK ((execution_claim_key IS NULL AND execution_claim_generation IS NULL)
        OR (execution_claim_key IS NOT NULL AND execution_claim_generation IS NOT NULL));

CREATE INDEX IF NOT EXISTS rl_route_request_lease_pending_idx
    ON public.research_lab_routing_execution_request_leases_v2(
        lease_state, lease_expires_at, updated_at, request_hash
    );

CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_execution_requests_v2(
    p_worker_ref TEXT,
    p_batch_size INTEGER,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $claim_execution_requests$
DECLARE
    request_row RECORD;
    current_lease public.research_lab_routing_execution_request_leases_v2%ROWTYPE;
    next_generation BIGINT;
    next_lease_hash TEXT;
    lease_expires TIMESTAMPTZ;
    claimed JSONB := '[]'::JSONB;
    recovery_key TEXT;
    recovery_event_hash TEXT;
    recovery_doc JSONB;
    recovery_event_doc JSONB;
BEGIN
    IF p_worker_ref IS NULL OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$' THEN
        RAISE EXCEPTION 'research_lab_routing_execution_queue_worker_invalid';
    END IF;
    IF p_batch_size IS NULL OR p_batch_size < 1 OR p_batch_size > 100 THEN
        RAISE EXCEPTION 'research_lab_routing_execution_queue_batch_invalid';
    END IF;
    IF p_lease_seconds IS NULL OR p_lease_seconds < 30 OR p_lease_seconds > 3600 THEN
        RAISE EXCEPTION 'research_lab_routing_execution_queue_lease_invalid';
    END IF;

    -- A request whose expired queue lease was already bound to a product
    -- claim is never reclaimable.  The provider boundary may have been
    -- crossed, so terminally recover the experiment and retain every open
    -- reservation at its full uncertain ceiling.  A new immutable experiment
    -- is required for later work.
    FOR current_lease IN
        SELECT lease.*
          FROM public.research_lab_routing_execution_request_leases_v2 lease
         WHERE lease.lease_state = 'claimed'
           AND lease.lease_expires_at <= pg_catalog.clock_timestamp()
           AND lease.execution_claim_key IS NOT NULL
           AND lease.execution_claim_generation IS NOT NULL
         ORDER BY lease.updated_at, lease.request_hash
         FOR UPDATE SKIP LOCKED
         LIMIT p_batch_size
    LOOP
        recovery_key := public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.research_lab.routing_queue_recovery.v3',
                'experiment_hash', current_lease.experiment_hash,
                'request_hash', current_lease.request_hash,
                'lease_hash', current_lease.lease_hash,
                'lease_generation', current_lease.lease_generation,
                'stale_claim_key', current_lease.execution_claim_key,
                'stale_claim_generation', current_lease.execution_claim_generation
            )
        );
        recovery_doc := pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim_recovery.v3',
            'worker_ref', p_worker_ref,
            'stale_claim_key', current_lease.execution_claim_key,
            'stale_claim_generation', current_lease.execution_claim_generation,
            'request_hash', current_lease.request_hash,
            'lease_hash', current_lease.lease_hash,
            'lease_generation', current_lease.lease_generation,
            'reason_code', 'queue_lease_expired'
        );
        recovery_event_doc := pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_event.v2',
            'experiment_hash', current_lease.experiment_hash,
            'recovery_key', recovery_key,
            'worker_ref', p_worker_ref,
            'event_type', 'claim_recovered',
            'reason_code', 'queue_lease_expired'
        );
        recovery_event_hash := public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'event_type', 'claim_recovered',
                'event_doc', recovery_event_doc
            )
        );
        PERFORM public.research_lab_routing_recover_claim_v3(
            current_lease.experiment_hash,
            recovery_key,
            p_worker_ref,
            recovery_doc,
            recovery_event_hash,
            recovery_event_doc
        );
    END LOOP;

    -- The request row is the lock coordinate.  Competing consumers skip it,
    -- so two workers cannot receive the same pending request in one lease.
    FOR request_row IN
        SELECT request.request_hash, request.experiment_hash
        FROM public.research_lab_routing_execution_requests_v2 AS request
        LEFT JOIN public.research_lab_routing_execution_request_leases_v2 AS lease
            ON lease.request_hash = request.request_hash
        WHERE lease.request_hash IS NULL
           OR (lease.lease_state = 'claimed'
               AND lease.execution_claim_key IS NULL
               AND lease.lease_expires_at <= pg_catalog.clock_timestamp())
        ORDER BY request.created_at, request.request_hash
        FOR UPDATE OF request SKIP LOCKED
        LIMIT p_batch_size
    LOOP
        SELECT * INTO current_lease
        FROM public.research_lab_routing_execution_request_leases_v2
        WHERE request_hash = request_row.request_hash
        FOR UPDATE;

        next_generation := COALESCE(current_lease.lease_generation, 0) + 1;
        next_lease_hash := 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    request_row.request_hash || ':' || p_worker_ref || ':'
                        || next_generation::TEXT || ':'
                        || pg_catalog.clock_timestamp()::TEXT,
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        );
        lease_expires := pg_catalog.clock_timestamp()
            + pg_catalog.make_interval(secs => p_lease_seconds);

        INSERT INTO public.research_lab_routing_execution_request_leases_v2 (
            request_hash, experiment_hash, lease_hash, worker_ref,
            lease_generation, lease_state, lease_expires_at,
            close_reason, created_at, updated_at
        ) VALUES (
            request_row.request_hash, request_row.experiment_hash,
            next_lease_hash, p_worker_ref, next_generation, 'claimed',
            lease_expires, NULL, pg_catalog.clock_timestamp(), pg_catalog.clock_timestamp()
        )
        ON CONFLICT (request_hash) DO UPDATE SET
            experiment_hash = EXCLUDED.experiment_hash,
            lease_hash = EXCLUDED.lease_hash,
            worker_ref = EXCLUDED.worker_ref,
            lease_generation = EXCLUDED.lease_generation,
            lease_state = EXCLUDED.lease_state,
            lease_expires_at = EXCLUDED.lease_expires_at,
            close_reason = NULL,
            updated_at = EXCLUDED.updated_at;

        claimed := claimed || pg_catalog.jsonb_build_array(
            pg_catalog.jsonb_build_object(
                'request_hash', request_row.request_hash,
                'experiment_hash', request_row.experiment_hash,
                'lease_hash', next_lease_hash,
                'worker_ref', p_worker_ref,
                'lease_generation', next_generation,
                'lease_expires_at', lease_expires
            )
        );
    END LOOP;
    RETURN pg_catalog.jsonb_build_object('requests', claimed);
END;
$claim_execution_requests$;

-- Bind a product execution claim to the queue lease exactly once.  The queue
-- row and the experiment claim share an advisory transaction lock, so an
-- expired worker cannot race a reclaiming worker into a second binding.
CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_execution_v3(
    p_request_hash TEXT,
    p_lease_hash TEXT,
    p_lease_generation BIGINT,
    p_worker_ref TEXT,
    p_claim_key TEXT,
    p_claim_lease_seconds INTEGER,
    p_claim_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $claim_execution_v3$
DECLARE
    lease_row public.research_lab_routing_execution_request_leases_v2%ROWTYPE;
    claim_result JSONB;
    updated_count INTEGER;
BEGIN
    IF p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_generation IS NULL OR p_lease_generation < 1
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref IS NULL
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_lease_seconds IS NULL
       OR p_claim_lease_seconds < 1 OR p_claim_lease_seconds > 3600
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_execution_claim_v3_invalid' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_claim_doc, 'routing execution claim v3');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing execution claim event v3');
    SELECT * INTO lease_row
      FROM public.research_lab_routing_execution_request_leases_v2
     WHERE request_hash = p_request_hash
     FOR UPDATE;
    IF NOT FOUND
       OR lease_row.lease_hash IS DISTINCT FROM p_lease_hash
       OR lease_row.lease_generation IS DISTINCT FROM p_lease_generation
       OR lease_row.worker_ref IS DISTINCT FROM p_worker_ref
       OR lease_row.lease_state IS DISTINCT FROM 'claimed'
       OR lease_row.lease_expires_at <= pg_catalog.clock_timestamp()
    THEN
        RAISE EXCEPTION 'research_lab_routing_execution_claim_v3_lease_stale'
            USING ERRCODE = '42501';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(lease_row.experiment_hash, 0)
    );
    IF lease_row.execution_claim_key IS NOT NULL
       AND lease_row.execution_claim_key IS DISTINCT FROM p_claim_key
    THEN
        RAISE EXCEPTION 'research_lab_routing_execution_claim_v3_already_bound'
            USING ERRCODE = '23505';
    END IF;
    claim_result := public.research_lab_routing_claim_experiment_v3(
        lease_row.experiment_hash,
        p_request_hash,
        p_lease_hash,
        p_lease_generation,
        p_claim_key,
        p_worker_ref,
        p_claim_lease_seconds,
        p_claim_doc,
        p_event_hash,
        p_event_doc
    );
    IF (claim_result->>'claimed')::BOOLEAN IS TRUE
       AND lease_row.execution_claim_key IS NULL
    THEN
        UPDATE public.research_lab_routing_execution_request_leases_v2
           SET execution_claim_key = p_claim_key,
               execution_claim_generation = (claim_result->>'claim_generation')::BIGINT,
               updated_at = pg_catalog.clock_timestamp()
         WHERE request_hash = p_request_hash
           AND lease_hash = p_lease_hash
           AND lease_generation = p_lease_generation
           AND worker_ref = p_worker_ref
           AND lease_state = 'claimed'
           AND lease_expires_at > pg_catalog.clock_timestamp()
           AND execution_claim_key IS NULL;
        GET DIAGNOSTICS updated_count = ROW_COUNT;
        IF updated_count <> 1 THEN
            RAISE EXCEPTION 'research_lab_routing_execution_claim_v3_bind_failed'
                USING ERRCODE = '40001';
        END IF;
    ELSIF lease_row.execution_claim_key IS NOT NULL
       AND lease_row.execution_claim_generation IS DISTINCT FROM
            (claim_result->>'claim_generation')::BIGINT
    THEN
        RAISE EXCEPTION 'research_lab_routing_execution_claim_v3_generation_conflict'
            USING ERRCODE = '23505';
    END IF;
    RETURN claim_result || pg_catalog.jsonb_build_object(
        'request_hash', p_request_hash,
        'lease_hash', p_lease_hash,
        'lease_generation', p_lease_generation,
        'worker_ref', p_worker_ref
    );
END;
$claim_execution_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_renew_execution_request_lease_v2(
    p_request_hash TEXT,
    p_worker_ref TEXT,
    p_lease_hash TEXT,
    p_lease_generation BIGINT,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $renew_execution_request$
DECLARE
    updated_count INTEGER;
    lease_expires TIMESTAMPTZ;
BEGIN
    IF p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref IS NULL
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_lease_generation IS NULL OR p_lease_generation < 1
       OR p_lease_seconds IS NULL OR p_lease_seconds < 30 OR p_lease_seconds > 3600 THEN
        RAISE EXCEPTION 'research_lab_routing_execution_queue_renew_invalid';
    END IF;
    lease_expires := pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_seconds);
    UPDATE public.research_lab_routing_execution_request_leases_v2
    SET lease_expires_at = lease_expires,
        updated_at = pg_catalog.clock_timestamp()
    WHERE request_hash = p_request_hash
      AND worker_ref = p_worker_ref
      AND lease_hash = p_lease_hash
      AND lease_generation = p_lease_generation
      AND lease_state = 'claimed'
      AND lease_expires_at > pg_catalog.clock_timestamp();
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN pg_catalog.jsonb_build_object(
        'renewed', updated_count = 1,
        'request_hash', p_request_hash,
        'lease_generation', p_lease_generation,
        'lease_expires_at', lease_expires
    );
END;
$renew_execution_request$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_close_execution_request_lease_v2(
    p_request_hash TEXT,
    p_worker_ref TEXT,
    p_lease_hash TEXT,
    p_lease_generation BIGINT,
    p_close_reason TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $close_execution_request$
DECLARE
    updated_count INTEGER;
BEGIN
    IF p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref IS NULL
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_lease_generation IS NULL OR p_lease_generation < 1
       OR p_close_reason NOT IN ('completed', 'failed', 'recovered') THEN
        RAISE EXCEPTION 'research_lab_routing_execution_queue_close_invalid';
    END IF;

    UPDATE public.research_lab_routing_execution_request_leases_v2
    SET lease_state = p_close_reason,
        lease_expires_at = NULL,
        close_reason = p_close_reason,
        updated_at = pg_catalog.clock_timestamp()
    WHERE request_hash = p_request_hash
      AND worker_ref = p_worker_ref
      AND lease_hash = p_lease_hash
      AND lease_generation = p_lease_generation
      AND lease_state = 'claimed'
      AND lease_expires_at > pg_catalog.clock_timestamp();
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    IF updated_count = 0 THEN
        RETURN pg_catalog.jsonb_build_object(
            'closed', FALSE,
            'stale', TRUE,
            'request_hash', p_request_hash,
            'lease_generation', p_lease_generation
        );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'closed', TRUE,
        'stale', FALSE,
        'request_hash', p_request_hash,
        'lease_generation', p_lease_generation,
        'close_reason', p_close_reason
    );
END;
$close_execution_request$;

ALTER TABLE public.research_lab_routing_execution_request_leases_v2 ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS rl_route_request_lease_service_read
    ON public.research_lab_routing_execution_request_leases_v2;
CREATE POLICY rl_route_request_lease_service_read
    ON public.research_lab_routing_execution_request_leases_v2
    FOR SELECT TO service_role USING (TRUE);

REVOKE ALL ON TABLE public.research_lab_routing_execution_request_leases_v2
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE TRUNCATE ON TABLE public.research_lab_routing_execution_request_leases_v2
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.research_lab_routing_execution_request_leases_v2
    TO service_role;

REVOKE ALL ON FUNCTION public.research_lab_routing_claim_execution_requests_v2(TEXT, INTEGER, INTEGER)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_renew_execution_request_lease_v2(TEXT, TEXT, TEXT, BIGINT, INTEGER)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_close_execution_request_lease_v2(TEXT, TEXT, TEXT, BIGINT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_execution_v3(TEXT, TEXT, BIGINT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_execution_requests_v2(TEXT, INTEGER, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_renew_execution_request_lease_v2(TEXT, TEXT, TEXT, BIGINT, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_close_execution_request_lease_v2(TEXT, TEXT, TEXT, BIGINT, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_execution_v3(TEXT, TEXT, BIGINT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB)
    TO service_role;

COMMENT ON TABLE public.research_lab_routing_execution_request_leases_v2 IS
    'Mutable queue lease state for immutable Research Lab execution requests; service-role RPC only, no raw claim nonce.';

NOTIFY pgrst, 'reload schema';

COMMIT;
